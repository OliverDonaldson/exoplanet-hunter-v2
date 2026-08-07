"""Cross-validated training for the per-diagnostic branch model.

Same evaluation contract as `train.py`: StratifiedGroupKFold with group =
tic_id, an inner GroupShuffleSplit for early stopping and the Platt fit, and a
`cv_summary.json` in the schema the promotion gate reads. Reusing that schema
matters — two implementations of it would drift, and the gate is what decides
whether a run is better than the incumbent.

Also the same *artefact* contract, which it did not have until 2026-08-06: a
per-fold checkpoint and calibration bundle, and every metric measured on the
reloaded checkpoint rather than on whatever `fit()` left in memory. Run 1 of
stage 2(a) wrote no checkpoint at all, so the model behind its numbers no
longer exists.

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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from exoplanet_hunter.datasets.viewset_pipeline import (
    AugmentConfig,
    Split,
    fit_scalar_constants,
    make_split_table,
    make_viewset_dataset,
    parse_viewset_shards,
)
from exoplanet_hunter.datasets.viewset_tfrecords import list_shards, load_index, load_metadata
from exoplanet_hunter.eval.comparison import SliceMetrics
from exoplanet_hunter.eval.metrics import classification_metrics
from exoplanet_hunter.eval.observation_bias import measure_observation_bias
from exoplanet_hunter.models.cnn_branches import build_cnn_branches
from exoplanet_hunter.training.calibration import PlattScaler, expected_calibration_error
from exoplanet_hunter.utils.logging import get_logger
from exoplanet_hunter.utils.provenance import git_provenance
from exoplanet_hunter.validation.leakage import drop_quarantined, load_quarantine
from exoplanet_hunter.validation.promotion import GATE_MISSION

log = get_logger(__name__)

SUMMARY_KEYS = ("test_roc_auc", "test_pr_auc", "test_f1", "test_brier", "test_ece")
CHECKPOINT_NAME = "cnn_branches.keras"
BUNDLE_NAME = "cnn_calibrator.joblib"


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


def _variance_decomposition(rows: list[dict]) -> dict[str, float | None]:
    """Split the run's spread into seed variance and fold difficulty.

    The `±` this project has been quoting is the spread of fold means within one
    run, and it has been read as the run's own repeatability. They are different
    quantities: it mixes how hard each fold is with how much a single training
    draw wanders, and only the second says anything about whether a rerun would
    land in the same place. `seed` is None until a fold trains more than one
    model, because with one draw per fold there is nothing to measure it from.
    """
    per_fold = [r.get("model_roc_auc") or [] for r in rows]
    fold_means = [float(np.mean(m)) for m in per_fold if m]
    within = [float(np.std(m, ddof=1)) for m in per_fold if len(m) > 1]
    return {
        "fold_sd": float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else None,
        "seed_sd": float(np.mean(within)) if within else None,
        "n_models_per_fold": len(per_fold[0]) if per_fold and per_fold[0] else 1,
    }


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
    #: None disables augmentation. Run 1 of stage 2(a) trained without it while
    #: the incumbent it was compared against had it — one of the ways that
    #: comparison was not like-for-like.
    augment: AugmentConfig | None = field(default_factory=AugmentConfig)


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
) -> tuple[dict, pd.DataFrame]:
    """Train one fold; return its metrics and its test-row predictions.

    With `fold_dir`, the best epoch is checkpointed there and reloaded before
    scoring — `train.py`'s "score what ships" rule, which this trainer did not
    have. Without it there is no artefact: run 1 of stage 2(a) scored weights
    that existed only in memory, so its model cannot be rescored, recalibrated,
    promoted or served.
    """
    shards = list_shards(shard_dir)
    tic_ids = index["tic_id"].to_numpy()
    table = make_split_table(tic_ids, _split_codes(len(index), train_idx, val_idx, test_idx))
    # Fitted on the training rows only: a validation row must never influence
    # the scale a training row is measured against.
    constants = fit_scalar_constants(index.iloc[train_idx], list(metadata["scalar_columns"]))

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

        model.fit(
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
        return _MemberRun(val_scores, val_labels, test_scores, test_labels, test_tics)

    runs = [train_one(i) for i in range(max(1, config.n_models_per_fold))]
    val_labels, test_labels, test_tics = runs[0].val_labels, runs[0].test_labels, runs[0].test_tics
    # Averaged before calibration, so the Platt fit describes the ensemble that
    # actually ships rather than one of its members.
    val_scores = np.mean([r.val_scores for r in runs], axis=0)
    test_scores = np.mean([r.test_scores for r in runs], axis=0)
    # Each member's own test AUC, which is what makes seed variance measurable
    # separately from fold difficulty.
    per_model = [float(classification_metrics(test_labels, r.test_scores).roc_auc) for r in runs]

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
    # measurement needs a score per row, and without it stage 2(b)'s success
    # criterion cannot be evaluated at all.
    predictions = index.iloc[np.sort(test_idx)].copy()
    predictions["score"] = calibrated
    # The stream yields test rows in ascending index position, so they line up
    # with sorted(test_idx). Asserted rather than assumed: a silent
    # misalignment would attach every score to the wrong target and still
    # produce a plausible AUC. Checked on tic_id, not label — labels are binary
    # over ~1,085 rows, so any permutation within a label block passes.
    if not np.array_equal(predictions["tic_id"].to_numpy(), test_tics.astype(np.int64)):
        raise RuntimeError("test predictions are not aligned with the index rows")
    return {
        "test_roc_auc": metrics.roc_auc,
        "test_pr_auc": metrics.pr_auc,
        "test_f1": metrics.f1,
        "test_brier": metrics.brier,
        "test_ece": float(expected_calibration_error(test_labels, calibrated)),
        "n_test": len(test_labels),
        # Uncalibrated, per member. The spread across these *within* a fold is
        # seed variance; the spread of fold means across folds is fold
        # difficulty. The reported ± has always conflated the two.
        "model_roc_auc": per_model,
    }, predictions


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
    y = index["label"].to_numpy().astype(int)
    groups = index["tic_id"].to_numpy()

    # Decoded once for the whole run. Only normalisation is per-fold, and that
    # runs downstream of this — five folds x four streams was 20 full decodes of
    # all 11 shards, and four live caches per fold.
    base = parse_viewset_shards(list_shards(shard_dir), metadata)

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    positions = np.arange(len(y))
    for fold, (trainval, test_idx) in enumerate(splitter.split(positions, y, groups)):
        inner = GroupShuffleSplit(
            n_splits=1, test_size=config.val_frac, random_state=config.seed * 1000 + fold
        )
        tr_rel, va_rel = next(inner.split(trainval, y[trainval], groups[trainval]))
        metrics, fold_predictions = run_fold(
            shard_dir,
            index,
            metadata,
            train_idx=trainval[tr_rel],
            val_idx=trainval[va_rel],
            test_idx=test_idx,
            config=config,
            model_cfg=model_cfg,
            fold_dir=out_dir / f"fold_{fold}",
            base=base,
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

    summary: dict[str, Any] = {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "std": float(np.std([r[key] for r in rows])),
        }
        for key in SUMMARY_KEYS
    }
    summary["variance"] = _variance_decomposition(rows)

    # Every row is tested exactly once across folds, so this is a full
    # out-of-fold prediction set — what the observation-bias metric reads.
    all_predictions = pd.concat(predictions, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_parquet(out_dir / "predictions.parquet", index=False)

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
    return payload
