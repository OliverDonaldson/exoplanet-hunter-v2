"""Cross-validated training for the per-diagnostic branch model.

Same evaluation contract as `train.py`: StratifiedGroupKFold with group =
tic_id, a `stratified_inner_split` for early stopping and the Platt fit, and a
`cv_summary.json` in the schema the promotion gate reads. Reusing that schema
matters — two implementations of it would drift, and the gate is what decides
whether a run is better than the incumbent.

Also the same *artefact* contract, which it did not have until 2026-08-06: a
per-fold checkpoint and calibration bundle, and every metric measured on the
reloaded checkpoint rather than on whatever `fit()` left in memory. Run 1 of
stage 4 (old 2(a)) wrote no checkpoint at all, so the model behind its numbers
no longer exists.

Nothing here promotes anything. It writes a summary; comparing it to the
incumbent is `promotion_gate.py`'s job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet_hunter.datasets.viewset_pipeline import (
    AugmentConfig,
    Split,
    fit_scalar_constants,
    make_split_table,
    make_viewset_dataset,
    make_weight_table,
    parse_viewset_shards,
)
from exoplanet_hunter.datasets.viewset_tfrecords import list_shards, load_index, load_metadata
from exoplanet_hunter.eval.comparison import MISSION_COLUMN, SliceMetrics, recall_at_fpr
from exoplanet_hunter.eval.metrics import classification_metrics
from exoplanet_hunter.eval.observation_bias import measure_observation_bias
from exoplanet_hunter.models.cnn_branches import build_cnn_branches
from exoplanet_hunter.training.calibration import PlattScaler, expected_calibration_error
from exoplanet_hunter.training.splits import (
    assigned_group_kfold,
    assignment_mask,
    load_fold_assignment,
    stratified_inner_split,
)
from exoplanet_hunter.utils.logging import get_logger
from exoplanet_hunter.utils.provenance import git_provenance
from exoplanet_hunter.validation.leakage import drop_quarantined, load_quarantine
from exoplanet_hunter.validation.promotion import GATE_MISSION

log = get_logger(__name__)

SUMMARY_KEYS = ("test_roc_auc", "test_pr_auc", "test_f1", "test_brier", "test_ece")
CHECKPOINT_NAME = "cnn_branches.keras"
BUNDLE_NAME = "cnn_calibrator.joblib"

#: The false-positive rate the shortlist operates at. `recall_at_fpr` is called
#: with it here and in `SliceMetrics`; the two must not drift, because one is the
#: gate's number and the other is the noise floor that number is read against.
GATE_FPR = 0.01

#: Per-member statistics whose spread `summary.variance` decomposes, as
#: `(per-member key in each fold row, prefix on the reported sd)`.
#:
#: AUC was the only one until 2026-08-08, and the omission mattered: **recall
#: @1% FPR is the criterion that rejected all four arms of stage 4** — run 3 on
#: 0.145 against the incumbent's 0.307 — while having no variance estimate at
#: all. AUC's floor was measured (`seed_sd 0.0081`, `fold_sd 0.0094`) and "a
#: margin under ~0.009 is not a decision" adopted from it; the statistic doing
#: the actual rejecting never got the same treatment, so the capacity arm's
#: 0.145 -> 0.236 could be neither believed nor dismissed.
#:
#: Two recall entries rather than one. `model_recall_at_1pct_fpr` mirrors the
#: AUC exactly — the whole fold, every mission — and `model_gate_recall_...` is
#: the same statistic over that fold's `GATE_MISSION` rows alone. They are not
#: interchangeable: the gate reads TESS, which is ~44% of the rows, so the
#: all-mission floor is measured on a population no decision is taken over.
VARIANCE_COMPONENTS = (
    ("model_roc_auc", ""),
    ("model_recall_at_1pct_fpr", "recall_"),
    ("model_gate_recall_at_1pct_fpr", "gate_recall_"),
)

#: Column prefix for each member's own uncalibrated score in
#: `predictions.parquet`. They exist so the pooled out-of-fold statistic can be
#: re-formed one member at a time — see `_pooled_member_draws`.
MEMBER_SCORE_PREFIX = "member_score_"


def _checkpoint_name(index: int, n_models: int) -> str:
    """One model per fold keeps the historical filename; an ensemble numbers them."""
    return CHECKPOINT_NAME if n_models <= 1 else f"model_{index}_{CHECKPOINT_NAME}"


class _MemberRun(NamedTuple):
    """One trained member's scores, before averaging and calibration."""

    val_scores: np.ndarray
    val_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    test_tics: np.ndarray
    #: Per-epoch series from `fit`, as Keras returns them. Purely observational
    #: — recording what training already did, never changing it.
    history: dict[str, list[float]]


def _format_sd(value: float | None) -> str:
    """`None` is "nobody measured this", which must not print as a number."""
    return "  n/a " if value is None else f"{value:.4f}"


def _component_sds(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    """`(fold_sd, seed_sd)` for one per-member statistic.

    A fold that recorded nothing for `key` — an empty list, which is how a fold
    with no rows in the population reports itself — contributes to neither, so a
    statistic nobody could measure comes back None rather than as a number
    computed over the folds that happened to have data.

    A fold that recorded a *non-finite* value raises instead. NaN loses every
    inequality, so a `recall_seed_sd` of NaN would read as "this margin is not
    inside the noise" in exactly the comparison the number exists to arbitrate —
    the same shape as the NaN that once promoted a degenerate run.
    """
    per_fold = [list(r.get(key) or []) for r in rows]
    for fold, members in enumerate(per_fold):
        if members and not np.all(np.isfinite(members)):
            raise ValueError(
                f"fold {fold} recorded a non-finite {key}: {members}. A single-class "
                f"slice cannot yield this statistic; the fold's population is degenerate."
            )
    fold_means = [float(np.mean(m)) for m in per_fold if m]
    within = [float(np.std(m, ddof=1)) for m in per_fold if len(m) > 1]
    return (
        float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else None,
        float(np.mean(within)) if within else None,
    )


def _pooled_member_draws(predictions: pd.DataFrame) -> dict[str, Any]:
    """Independent draws of the *pooled* gate statistic, one per ensemble member.

    `gate_recall_seed_sd` is measured on a fold's TESS slice — ~215 negatives, so
    a 1% FPR cut of two rows, and a statistic set by where the third-highest
    negative lands. The gate reads the pooled out-of-fold set instead: ~1,074
    negatives, a cut of ten rows, and a materially better-conditioned number.
    Bounding the second with the first only ever overstates the noise.

    Member `i`'s score column is filled by whichever fold held each row, so
    stacking them re-forms a complete out-of-fold prediction set for member `i`
    alone — the same protocol as the ensemble, one seed instead of three. Their
    spread is the run-level reseeding sd directly, with no `sqrt(n)` argument in
    the way. Three draws is a thin sd and it is reported with its `n`.
    """
    # Sorted on the integer, not the string: `member_score_10` sorts before
    # `member_score_2` lexicographically, which would reorder the draws. The
    # order does not change their sd, but it changes which draw is which in the
    # recorded list, and that list is read against the folds' own members.
    columns = sorted(
        (c for c in predictions.columns if c.startswith(MEMBER_SCORE_PREFIX)),
        key=lambda c: int(c.removeprefix(MEMBER_SCORE_PREFIX)),
    )
    gate = (
        predictions[predictions[MISSION_COLUMN] == GATE_MISSION]
        if (MISSION_COLUMN in predictions.columns)
        else predictions.iloc[:0]
    )
    if not columns or gate.empty:
        # The same keys either way. A summary whose schema depends on whether a
        # measurement succeeded is one where a missing key and a null mean the
        # same thing to a reader and different things to a program.
        return {
            "pooled_gate_recall": [],
            "pooled_gate_recall_seed_sd": None,
            "pooled_gate_recall_n_draws": 0,
            "pooled_gate_n": len(gate),
        }
    labels = gate["label"].to_numpy()
    draws = []
    for column in columns:
        scores = gate[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(scores)):
            # A fold that trained fewer members leaves NaN here rather than a
            # short column, and a NaN score silently sinks those rows to the
            # bottom of the ranking — a plausible number over a population that
            # is not the one named.
            raise ValueError(
                f"{column} is not finite on every gate-mission row; the folds did "
                f"not all train the same number of members"
            )
        draws.append(float(recall_at_fpr(labels, scores, GATE_FPR)))
    if not np.all(np.isfinite(draws)):
        # `recall_at_fpr` returns NaN on a single-class slice rather than
        # raising, and an sd over NaN is NaN — which loses every inequality it
        # would later be compared with. Empty was handled above; this is the
        # other way the statistic can fail to exist.
        raise ValueError(
            f"the pooled {GATE_MISSION} slice yielded non-finite recall {draws} over "
            f"{len(labels)} rows with {int(labels.sum())} positives; it is single-class"
        )
    return {
        "pooled_gate_recall": draws,
        "pooled_gate_recall_seed_sd": float(np.std(draws, ddof=1)) if len(draws) > 1 else None,
        "pooled_gate_recall_n_draws": len(draws),
        "pooled_gate_n": len(labels),
    }


def _variance_decomposition(rows: list[dict]) -> dict[str, float | None]:
    """Split the run's spread into seed variance and fold difficulty.

    The `±` this project has been quoting is the spread of fold means within one
    run, and it has been read as the run's own repeatability. They are different
    quantities: it mixes how hard each fold is with how much a single training
    draw wanders, and only the second says anything about whether a rerun would
    land in the same place. `seed` is None until a fold trains more than one
    model, because with one draw per fold there is nothing to measure it from.

    Reported for every entry in `VARIANCE_COMPONENTS`, so recall @1% FPR — the
    criterion that has done all the rejecting — carries the error bar AUC has
    had since 2026-08-08. Purely additive: the promotion gate reads named keys
    and the AUC pair keeps its unprefixed names.
    """
    decomposition: dict[str, float | None] = {}
    for key, prefix in VARIANCE_COMPONENTS:
        fold_sd, seed_sd = _component_sds(rows, key)
        decomposition[f"{prefix}fold_sd"] = fold_sd
        decomposition[f"{prefix}seed_sd"] = seed_sd
    members = [r.get("model_roc_auc") or [] for r in rows]
    decomposition["n_models_per_fold"] = len(members[0]) if members and members[0] else 1
    return decomposition


@dataclass
class CVConfig:
    n_splits: int = 5
    val_frac: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    patience: int = 8
    learning_rate: float = 1e-3
    #: Fixes the split and the initialisation, **not the result**. Augmentation
    #: draws from stateful `tf.random` inside a parallel `map`, nothing sets
    #: `enable_op_determinism`, and Metal reduces nondeterministically — so a
    #: rerun of this exact config lands somewhere else. Measured 2026-08-08 at
    #: `seed_sd = 0.0081` per model. Recording the seed without that number
    #: implies a reproducibility this does not have, which is why
    #: `summary.variance` reports it beside every run.
    seed: int = 42
    #: Models trained per fold, averaged before calibration. At 1 a fold's score
    #: is a single seed draw, and the measured spread of that draw is sd 0.0106
    #: — larger than most differences this project decides on. Averaging n draws
    #: shrinks it by ~sqrt(n), and each model's own metrics are recorded so
    #: seed variance and fold difficulty can finally be told apart.
    n_models_per_fold: int = 1
    #: None disables augmentation. Run 1 of stage 4 trained without it while
    #: the incumbent it was compared against had it — one of the ways that
    #: comparison was not like-for-like.
    augment: AugmentConfig | None = field(default_factory=AugmentConfig)
    #: Stage 8's baseline-bias arm: None (control), "propensity" or "stratified".
    #: Recorded here rather than passed loose so it lands in `run_config` — two
    #: runs differing only in their intervention would otherwise be
    #: indistinguishable from their summaries, which is exactly the defect that
    #: made run 1's comparison against the incumbent unreadable.
    baseline_intervention: str | None = None
    #: Strata for that arm. 16 removes the training-set correlation to +0.0025
    #: for propensity weighting and 8 reaches +0.0060 for stratified sampling,
    #: measured on the real TESS slice 2026-08-12.
    baseline_strata: int = 16
    #: Path to a group→fold map that pins the outer split, or None for this
    #: run's own `StratifiedGroupKFold`. Stage 10.5 needs two trainers over two
    #: different shard sets to partition identically, which no shared seed can
    #: deliver. Recorded here rather than passed loose so it lands in
    #: `run_config`: stage 8 shipped two arms distinguishable only by
    #: `n_examples` because the thing that differed between them was a path the
    #: summary never recorded, and this is that same defect waiting to happen.
    fold_assignment: str | None = None


def _resolved_model_config(model_cfg: object) -> dict[str, Any] | None:
    """The architecture this run actually built, as JSON.

    `CVConfig` records how training ran; nothing recorded *what was trained*, so
    two runs with the same CVConfig and different architectures were
    indistinguishable from their summaries alone — and the promotion gate reads
    nothing else. None means the caller passed no config, which is itself worth
    recording: `run_cv` then takes every architecture default via `getattr`.
    """
    data = getattr(model_cfg, "__dict__", None)
    if not data:
        return None
    return {key: value for key, value in data.items() if not key.startswith("_")}


def per_mission_summary(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pooled out-of-fold metrics per mission, plus the pooled aggregate.

    The gate reads TESS from here; Kepler and K2 are reported diagnostics with
    an alarm threshold, and `all` never gates. See `evaluate_promotion` and the
    roadmap's gate-population pre-commitment for why the aggregate is unfit to
    decide anything: its mission weights are a sampling decision.
    """
    slices = {
        str(mission): asdict(
            SliceMetrics.measure(group["label"].to_numpy(), group["score"].to_numpy())
        )
        for mission, group in predictions.groupby("mission")
    }
    slices["all"] = asdict(
        SliceMetrics.measure(predictions["label"].to_numpy(), predictions["score"].to_numpy())
    )
    return slices


def _split_codes(n: int, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> np.ndarray:
    codes = np.full(n, -1, dtype=np.int64)
    codes[train] = int(Split.TRAIN)
    codes[val] = int(Split.VAL)
    codes[test] = int(Split.TEST)
    return codes


def _predict(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, labels and TIC IDs in stream order — unshuffled, so aligned.

    The TIC ID rides along so the caller can assert alignment against the row's
    identity. Checking labels instead only catches a misalignment that also
    permutes them, and labels are binary over ~1,085 test rows.
    """
    scores, labels, tic_ids = [], [], []
    for inputs, y, tic in dataset:
        scores.append(model(inputs, training=False).numpy().ravel())
        labels.append(y.numpy().ravel())
        tic_ids.append(tic.numpy().ravel())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(tic_ids)


def run_fold(
    shard_dir: Path,
    index: pd.DataFrame,
    metadata: dict,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: CVConfig,
    model_cfg: object,
    fold_dir: Path | None = None,
    base: tf.data.Dataset | None = None,
    sample_weights: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train one fold; return its metrics and its test-row predictions.

    With `fold_dir`, the best epoch is checkpointed there and reloaded before
    scoring — `train.py`'s "score what ships" rule, which this trainer did not
    have. Without it there is no artefact: run 1 of stage 4 scored weights
    that existed only in memory, so its model cannot be rescored, recalibrated,
    promoted or served.
    """
    shards = list_shards(shard_dir)
    tic_ids = index["tic_id"].to_numpy()
    table = make_split_table(tic_ids, _split_codes(len(index), train_idx, val_idx, test_idx))
    # Fitted on the training rows only: a validation row must never influence
    # the scale a training row is measured against.
    constants = fit_scalar_constants(index.iloc[train_idx], list(metadata["scalar_columns"]))
    weights = (
        make_weight_table(tic_ids, np.asarray(sample_weights, dtype=float))
        if sample_weights is not None
        else None
    )

    def stream(split: Split, *, shuffle: bool, identify: bool = False) -> tf.data.Dataset:
        return make_viewset_dataset(
            shards,
            metadata,
            base=base,
            split_table=table,
            split=split,
            scalar_constants=constants,
            batch_size=config.batch_size,
            shuffle=shuffle,
            # Training only: a validation or test row must be scored as it is.
            augment=config.augment if split is Split.TRAIN else None,
            seed=config.seed,
            with_tic_id=identify,
            # Training only, for the same reason. A weighted validation stream
            # would make early stopping optimise the reweighted population
            # rather than the one the run is measured on, and a weighted test
            # stream would weight the metrics themselves.
            weight_table=weights if split is Split.TRAIN else None,
        )

    def train_one(index: int) -> _MemberRun:
        """Fit one model and return its val and test scores, plus their labels."""
        # A distinct init per model, so the n draws are actually independent —
        # otherwise averaging them buys nothing. The stream's own shuffle and
        # augmentation seeds stay pinned to config.seed, so the data order is
        # identical across models and only the weights differ.
        tf.keras.utils.set_random_seed(config.seed * 1_000 + index)
        model = build_cnn_branches(
            model_cfg,
            scalar_columns=list(metadata["scalar_columns"]),
            mask_columns=list(metadata["mask_columns"]),
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
        )
        callbacks: list[tf.keras.callbacks.Callback] = [
            # AUC-PR rather than loss: the ExoMiner papers stop on it, and it
            # is the metric that tracks the minority class we care about.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=config.patience,
                restore_best_weights=True,
            )
        ]
        ckpt_path = None
        if fold_dir is not None:
            fold_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = fold_dir / _checkpoint_name(index, config.n_models_per_fold)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(ckpt_path), monitor="val_pr_auc", mode="max", save_best_only=True
                )
            )

        fitted = model.fit(
            stream(Split.TRAIN, shuffle=True),
            validation_data=stream(Split.VAL, shuffle=False),
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=0,
        )
        if ckpt_path is not None:
            # In-memory weights after fit() are not guaranteed to match the file
            # (measured drift up to 0.31 per example on cebb0fe6). Every metric
            # and calibrator below describes the checkpoint.
            model = tf.keras.models.load_model(str(ckpt_path), compile=False)

        val_scores, val_labels, _ = _predict(model, stream(Split.VAL, shuffle=False, identify=True))
        test_scores, test_labels, test_tics = _predict(
            model, stream(Split.TEST, shuffle=False, identify=True)
        )
        # Floats, not numpy scalars: this lands in JSON, and np.float32 is
        # not serialisable. Rounded to 6 places — a loss curve does not need
        # seventeen significant figures and the file is read by humans.
        history = {
            name: [round(float(v), 6) for v in series] for name, series in fitted.history.items()
        }
        return _MemberRun(val_scores, val_labels, test_scores, test_labels, test_tics, history)

    runs = [train_one(i) for i in range(max(1, config.n_models_per_fold))]
    val_labels, test_labels, test_tics = runs[0].val_labels, runs[0].test_labels, runs[0].test_tics
    # Averaged before calibration, so the Platt fit describes the ensemble that
    # actually ships rather than one of its members.
    val_scores = np.mean([r.val_scores for r in runs], axis=0)
    test_scores = np.mean([r.test_scores for r in runs], axis=0)

    calibrator = PlattScaler.from_validation(val_scores, val_labels)
    calibrated = calibrator.predict(test_scores)
    if fold_dir is not None:
        # The scoring path's contract: the checkpoint alone is not servable,
        # because the scalar constants and the Platt fit are per-fold too.
        joblib.dump(
            {
                "calibrator": calibrator,
                "platt_a": calibrator.a,
                "platt_b": calibrator.b,
                "scalar_constants": constants,
            },
            fold_dir / BUNDLE_NAME,
        )

    metrics = classification_metrics(test_labels, calibrated)
    # Test rows in stream order, which is index order — the observation-bias
    # measurement needs a score per row, and without it stage 7's success
    # criterion cannot be evaluated at all.
    predictions = index.iloc[np.sort(test_idx)].copy()
    predictions["score"] = calibrated
    # Each member's own uncalibrated score per row, so the *pooled* out-of-fold
    # statistic can be re-formed one member at a time after the run. That is the
    # only way to get the reseeding spread of the number the gate actually reads:
    # a fold's TESS slice holds ~215 negatives, so its 1% FPR cut is two rows and
    # the fold-level statistic is far coarser than the pooled one it is used to
    # bound. Costs three float columns; no extra training and no extra inference.
    # Uncalibrated on purpose — the Platt fit was fitted on the ensemble mean, so
    # applying it to a single member would describe a calibrator that never
    # existed, and every statistic taken from these is rank-based anyway.
    for i, member in enumerate(runs):
        predictions[f"{MEMBER_SCORE_PREFIX}{i}"] = member.test_scores
    # The stream yields test rows in ascending index position, so they line up
    # with sorted(test_idx). Asserted rather than assumed: a silent
    # misalignment would attach every score to the wrong target and still
    # produce a plausible AUC. Checked on tic_id, not label — labels are binary
    # over ~1,085 rows, so any permutation within a label block passes.
    if not np.array_equal(predictions["tic_id"].to_numpy(), test_tics.astype(np.int64)):
        raise RuntimeError("test predictions are not aligned with the index rows")

    # Per member, and only past the alignment assertion above — the gate mask is
    # taken from the index rows, so it is only a valid mask over the score
    # arrays once those rows are known to be the ones that were scored.
    #
    # Uncalibrated. The spread across these *within* a fold is seed variance;
    # the spread of fold means across folds is fold difficulty. The reported ±
    # has always conflated the two. Platt is monotone, so both recall figures
    # are identical calibrated or not; the AUC is too, and it always was.
    gate_rows = (
        (predictions[MISSION_COLUMN].to_numpy() == GATE_MISSION)
        if (MISSION_COLUMN in predictions.columns)
        else np.zeros(len(test_labels), dtype=bool)
    )
    per_model_auc = [
        float(classification_metrics(test_labels, r.test_scores).roc_auc) for r in runs
    ]
    per_model_recall = [float(recall_at_fpr(test_labels, r.test_scores, GATE_FPR)) for r in runs]
    # `[]` rather than NaN when the fold holds no gate-mission rows: unmeasured
    # and degenerate are different claims, and `_component_sds` treats them so.
    per_model_gate_recall = (
        [
            float(recall_at_fpr(test_labels[gate_rows], r.test_scores[gate_rows], GATE_FPR))
            for r in runs
        ]
        if gate_rows.any()
        else []
    )
    return {
        "test_roc_auc": metrics.roc_auc,
        "test_pr_auc": metrics.pr_auc,
        "test_f1": metrics.f1,
        "test_brier": metrics.brier,
        "test_ece": float(expected_calibration_error(test_labels, calibrated)),
        "n_test": len(test_labels),
        "n_test_gate_mission": int(gate_rows.sum()),
        # Per-epoch loss and AUC-PR for each member, train and validation.
        # Recorded because `patience` is otherwise a number nobody can check:
        # a summary that reports only the final metrics cannot say whether
        # early stopping fired at epoch 9 or ran to the 40-epoch ceiling, so
        # neither over- nor under-training is diagnosable after the fact. Pure
        # observation — `fit` has already happened when this is read.
        "epoch_history": [r.history for r in runs],
        "model_roc_auc": per_model_auc,
        # The gate's own statistic, which until 2026-08-08 had no error bar at
        # all while rejecting every arm of stage 4. Note `classification_metrics`
        # cannot supply it: its `.recall` is recall at threshold 0.5.
        "model_recall_at_1pct_fpr": per_model_recall,
        "model_gate_recall_at_1pct_fpr": per_model_gate_recall,
    }, predictions


def _apply_baseline_intervention(
    index: pd.DataFrame, config: CVConfig
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any] | None]:
    """Stage 8's arm, applied to the run's index before any fold is cut.

    Returns the index to train on, per-example weights or None, and a report to
    carry into `run_config`.

    **Both arms act before the split, and that is deliberate.** Applied per fold
    they would resample or reweight against each fold's own baseline
    distribution, so the five folds would train on five different
    interventions and the run-level number would describe none of them.

    The stratified arm returns a *smaller index*. Rows it drops never reach a
    split, so they are absent from training, validation **and test** — which is
    the honest reading: a model trained on a resampled population has not been
    evaluated on the rows that population excluded, and quietly testing on them
    would report a number for a population the model never saw.
    """
    arm = config.baseline_intervention
    if arm is None:
        return index, None, None

    # Imported here rather than at module scope: this trainer is imported by the
    # serving path, and the intervention modules pull in scipy.
    from exoplanet_hunter.datasets.baseline_bias import (
        propensity_weights,
        stratified_negative_sample,
    )

    if index["tic_id"].duplicated().any():
        raise ValueError(
            "the weight and split tables key on tic_id, and this index carries duplicates — "
            "every planet of a multi-planet host would share one weight. Key on the row "
            "instead before running an intervention arm"
        )

    if arm == "propensity":
        weighted = propensity_weights(index, n_strata=config.baseline_strata)
        log.info("[train-branches] arm 'propensity': %s", weighted.report())
        return (
            index,
            weighted.weights,
            {
                "arm": arm,
                "n_strata": config.baseline_strata,
                "correlation_before": weighted.before,
                "correlation_after": weighted.after,
                "n_clipped": weighted.n_clipped,
                "n_strata_used": weighted.n_strata_used,
            },
        )

    if arm == "stratified":
        sampled = stratified_negative_sample(
            index, n_strata=config.baseline_strata, seed=config.seed
        )
        log.info("[train-branches] arm 'stratified': %s", sampled.report())
        kept = pd.DataFrame(index.iloc[sampled.index]).reset_index(drop=True)
        return (
            kept,
            None,
            {
                "arm": arm,
                "n_strata": config.baseline_strata,
                "correlation_before": sampled.before,
                "correlation_after": sampled.after,
                "n_dropped": sampled.n_dropped,
                "n_kept": len(kept),
            },
        )

    raise ValueError(
        f"unknown baseline_intervention {arm!r}; expected None, 'propensity' or 'stratified'"
    )


def run_cv(
    shard_dir: Path,
    out_dir: Path,
    *,
    config: CVConfig | None = None,
    model_cfg: object | None = None,
    labels_dir: Path | None = None,
) -> dict:
    """Full CV over a view-set shard set; writes `cv_summary.json`."""
    config = config or CVConfig()
    model_cfg = model_cfg if model_cfg is not None else object()
    metadata = load_metadata(shard_dir)
    index = load_index(shard_dir)

    # A row whose label flipped belongs to the since-confirmed temporal holdout,
    # which `eval_since_confirmed.py` scores separately. Letting it into any fold
    # destroys that holdout and leaks a future label into training. The gate
    # recorded these; until 2026-08-07 nothing read them back.
    quarantined = load_quarantine(labels_dir or Path("data/labels"))
    if quarantined:
        before = len(index)
        index = drop_quarantined(index, quarantined)
        log.info(
            "[train-branches] %d quarantined rows held out of CV (%d -> %d)",
            before - len(index),
            before,
            len(index),
        )
    # Restriction runs *before* the intervention, not after. Propensity weights
    # and the stratified resample are both fitted to whatever population they
    # are handed; computing them over rows the fold assignment then discards
    # would leave the surviving rows carrying weights calibrated for a
    # population that never trained.
    fold_map: dict[int, int] | None = None
    if config.fold_assignment is not None:
        fold_map, fold_provenance = load_fold_assignment(Path(config.fold_assignment))
        keep_column = "group_tic" if "group_tic" in index.columns else "tic_id"
        keep = assignment_mask(index[keep_column].to_numpy(), fold_map)
        log.info(
            "[train-branches] fold assignment %s: %d of %d rows covered (%d dropped), "
            "%d groups over %d folds",
            config.fold_assignment,
            int(keep.sum()),
            len(index),
            int((~keep).sum()),
            fold_provenance.get("n_groups", len(fold_map)),
            fold_provenance.get("n_folds", config.n_splits),
        )
        if not keep.any():
            raise ValueError(
                f"the fold assignment at {config.fold_assignment} covers none of this "
                "shard set's rows — it was almost certainly built over a different "
                "population, and training on zero rows would be reported as a run"
            )
        index = index.loc[keep].reset_index(drop=True)

    index, sample_weights, intervention = _apply_baseline_intervention(index, config)
    y = index["label"].to_numpy().astype(int)
    # `group_tic` when the shard set carries it, `tic_id` otherwise.
    #
    # Stage 8's synthetic negatives are built *from* a real star's light curve,
    # so a scrambled row and the row it came from are the same star seen twice.
    # Grouping on `tic_id` alone would let them fall in different folds, and the
    # model would then be tested on a star whose own light curve — noise, gaps,
    # systematics and all — it had already trained on. That is the leakage the
    # grouped split exists to prevent, arriving through a door the split cannot
    # see, because the synthetic row carries a `tic_id` of its own so the split
    # and weight tables stay one-to-one.
    group_column = "group_tic" if "group_tic" in index.columns else "tic_id"
    groups = index[group_column].to_numpy()
    if group_column == "group_tic":
        log.info(
            "[train-branches] grouping folds on 'group_tic' (%d rows over %d stars)",
            len(index),
            len(np.unique(groups)),
        )

    # Decoded once for the whole run. Only normalisation is per-fold, and that
    # runs downstream of this — five folds x four streams was 20 full decodes of
    # all 11 shards, and four live caches per fold.
    base = parse_viewset_shards(list_shards(shard_dir), metadata)

    # A pinned assignment replays one partition in both trainers; without one
    # each builds its own over its own shard set, and two different populations
    # never partition alike whatever the seed.
    if fold_map is not None:
        folds = assigned_group_kfold(groups, fold_map, n_splits=config.n_splits)
    else:
        folds = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        ).split(np.arange(len(y)), y, groups)
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for fold, (trainval, test_idx) in enumerate(folds):
        train_idx, val_idx = stratified_inner_split(
            trainval, y, groups, val_frac=config.val_frac, seed=config.seed * 1000 + fold
        )
        metrics, fold_predictions = run_fold(
            shard_dir,
            index,
            metadata,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            config=config,
            model_cfg=model_cfg,
            fold_dir=out_dir / f"fold_{fold}",
            base=base,
            sample_weights=sample_weights,
        )
        metrics["fold"] = fold
        rows.append(metrics)
        fold_predictions["fold"] = fold
        predictions.append(fold_predictions)
        log.info(
            "[train-branches] fold %d  AUC %.4f  Brier %.4f  ECE %.4f  (n=%d)",
            fold,
            metrics["test_roc_auc"],
            metrics["test_brier"],
            metrics["test_ece"],
            metrics["n_test"],
        )

    # Written before anything is summarised. Every guard downstream of here can
    # raise, and a run whose summary refuses to compute should not also lose the
    # hour of training behind it — the predictions are what the summary is
    # derived from, so `evaluate.py summarise` can rebuild it once the reason is
    # understood. Every row is tested exactly once across folds, so this is a
    # full out-of-fold prediction set — what the observation-bias metric reads.
    all_predictions = pd.concat(predictions, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_parquet(out_dir / "predictions.parquet", index=False)

    summary: dict[str, Any] = {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "std": float(np.std([r[key] for r in rows])),
        }
        for key in SUMMARY_KEYS
    }
    summary["variance"] = {
        **_variance_decomposition(rows),
        **_pooled_member_draws(all_predictions),
    }

    per_mission = per_mission_summary(all_predictions)
    payload = {
        "folds": rows,
        "summary": summary,
        "per_mission": per_mission,
        # What produced these numbers, in the artefact that carries them. The
        # view resolution and whether augmentation ran are exactly the two things
        # that made run 1's comparison against the incumbent unlike-for-like, and
        # neither was recoverable from its summary.
        "run_config": {
            **asdict(config),
            "view_shapes": {k: list(v) for k, v in metadata["view_shapes"].items()},
            "n_examples": len(index),
            **git_provenance().as_dict(),
            "model_config": _resolved_model_config(model_cfg),
            # The arm that produced these numbers, in the artefact that carries
            # them. Two runs differing only in their intervention are otherwise
            # indistinguishable from their summaries alone.
            "baseline_intervention": intervention,
        },
    }
    # `default=str` so a value that is not JSON-native is still recorded rather
    # than raising here and discarding an hour of training along with every
    # number in this payload. Everything that comes out of the model YAML is
    # JSON-native already, so nothing that matters is coerced.
    (out_dir / "cv_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    for mission, slice_metrics in per_mission.items():
        log.info(
            "[train-branches] %-7s n=%-5d AUC %.4f  Brier %.4f  ECE %.4f  R@1%%FPR %.3f%s",
            mission,
            slice_metrics["n"],
            slice_metrics["roc_auc"],
            slice_metrics["brier"],
            slice_metrics["ece"],
            slice_metrics["recall_at_1pct_fpr"],
            "  <- gates" if mission == GATE_MISSION else "",
        )
    bias = measure_observation_bias(all_predictions["score"].to_numpy(), all_predictions)
    (out_dir / "observation_bias.json").write_text(json.dumps(asdict(bias), indent=2))
    log.info(
        "[train-branches] observation bias: transit %+.3f  baseline %+.3f  "
        "completeness %+.3f  (labelled set: incumbent -0.087 / +0.238, "
        "label itself -0.073 / +0.278)",
        bias.transit_sensitivity,
        bias.baseline_sensitivity,
        bias.completeness_sensitivity,
    )
    log.info(
        "[train-branches] ROC-AUC %.4f ± %.4f  Brier %.4f ± %.4f  ECE %.4f ± %.4f -> %s",
        summary["test_roc_auc"]["mean"],
        summary["test_roc_auc"]["std"],
        summary["test_brier"]["mean"],
        summary["test_brier"]["std"],
        summary["test_ece"]["mean"],
        summary["test_ece"]["std"],
        out_dir / "cv_summary.json",
    )
    # The noise floors, in the log rather than only in the JSON: they are what
    # every margin in this run has to clear, and reading a margin without them
    # is how four arms were decided on unquantified deltas.
    variance = summary["variance"]
    for _, prefix in VARIANCE_COMPONENTS:
        log.info(
            "[train-branches] %-18s seed_sd %s  fold_sd %s  (n_models_per_fold=%s)",
            f"{prefix or 'auc_'}floor",
            _format_sd(variance[f"{prefix}seed_sd"]),
            _format_sd(variance[f"{prefix}fold_sd"]),
            variance["n_models_per_fold"],
        )
    log.info(
        "[train-branches] %-18s seed_sd %s  over %s pooled draws on n=%s %s rows: %s",
        "pooled gate floor",
        _format_sd(variance["pooled_gate_recall_seed_sd"]),
        variance["pooled_gate_recall_n_draws"],
        variance["pooled_gate_n"],
        GATE_MISSION,
        [round(v, 4) for v in variance["pooled_gate_recall"]],
    )
    return payload
