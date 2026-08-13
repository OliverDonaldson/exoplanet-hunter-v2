"""CV runner for the branch model: split integrity and summary schema.

**Every `run_cv` here is shared where it can be, and that is a correctness
property of the suite rather than tidiness.** Repeated `run_cv` calls in one
process get monotonically slower — measured 2026-08-08 on the fixture below:
86s, 108s, 161s, 190s for four identical runs, and the fourteenth consecutive
call in this file took 72 minutes. `tf.keras.backend.clear_session()` between
them does **not** help (84s, 102s, 153s, 192s), so the accumulation is not
Keras graph state and the cause is still unknown — see the audit's "still open".
Until it is understood, the only lever is calling `run_cv` fewer times, so the
module-scoped fixtures below do it four times instead of eighteen. A test that
needs its own run must genuinely need different inputs, and say why.
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_pipeline import Split
from exoplanet_hunter.datasets.viewset_tfrecords import write_viewset_shards
from exoplanet_hunter.eval.comparison import recall_at_fpr
from exoplanet_hunter.eval.metrics import classification_metrics
from exoplanet_hunter.training import train_branches
from exoplanet_hunter.training.train_branches import (
    BUNDLE_NAME,
    CHECKPOINT_NAME,
    GATE_FPR,
    SUMMARY_KEYS,
    VARIANCE_COMPONENTS,
    CVConfig,
    _variance_decomposition,
    run_cv,
)
from exoplanet_hunter.validation.promotion import GATE_MISSION, evaluate_promotion

#: 40 rows over 2 folds leaves ~20 for training, so the default 0.2 inner
#: split is 4 validation rows — too few for a Platt fit to reliably see both
#: classes, which it requires. Production splits ~868. This is a property of
#: the fixture, not of the trainer.
FAST_CV = {"n_splits": 2, "epochs": 1, "batch_size": 8, "patience": 1, "val_frac": 0.5}


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    from conftest import _make_view_set

    path = tmp_path_factory.mktemp("shards")
    write_viewset_shards(_make_view_set(n=40, seed=3), path, examples_per_shard=16)
    return path


@pytest.fixture(scope="module")
def single_run(shard_dir, tmp_path_factory):
    """One model per fold — the historical checkpoint layout, and `seed_sd`
    unmeasurable because one draw cannot give a spread."""
    out = tmp_path_factory.mktemp("single") / "cv"
    return run_cv(shard_dir, out, config=CVConfig(**FAST_CV)), out


@pytest.fixture(scope="module")
def ensemble_run(shard_dir, tmp_path_factory):
    """Two models per fold — every variance component measurable, every member
    checkpointed, and a member score column per member in the predictions."""
    out = tmp_path_factory.mktemp("ensemble") / "cv"
    return run_cv(shard_dir, out, config=CVConfig(**FAST_CV, n_models_per_fold=2)), out


def test_cv_writes_a_summary_the_promotion_gate_can_read(single_run):
    payload, out = single_run
    assert len(payload["folds"]) == 2
    for key in SUMMARY_KEYS:
        assert {"mean", "std"} == set(payload["summary"][key])

    on_disk = json.loads((out / "cv_summary.json").read_text())
    assert on_disk == payload
    # The gate reads this shape; if it drifts, promotion silently stops working.
    decision = evaluate_promotion(on_disk, on_disk)
    assert decision is not None


def test_every_example_is_tested_exactly_once(single_run):
    # Group = tic_id, so a star appears in exactly one fold's test split. A
    # star split across folds is the leakage the whole CV design prevents.
    payload, _ = single_run
    assert sum(fold["n_test"] for fold in payload["folds"]) == 40


def test_a_multi_planet_host_never_spans_two_folds(tmp_path, make_view_set):
    """The count above holds whether or not grouping works, so it cannot catch
    the leakage it names. With every row on its own star — the live catalogue's
    state, 5,703 rows over 5,703 TICs — no test can. This spreads 40 rows over
    8 stars so the group constraint is load-bearing, and asserts what the CV
    design actually promises.

    Its own run: the shared fixtures put every row on its own star, which is the
    case this test exists to be unlike."""
    shards = tmp_path / "shards"
    write_viewset_shards(make_view_set(n=40, seed=3, hosts=8), shards, examples_per_shard=16)
    run_cv(shards, tmp_path / "cv", config=CVConfig(**FAST_CV))

    predictions = pd.read_parquet(tmp_path / "cv" / "predictions.parquet")
    assert predictions["tic_id"].nunique() == 8
    spread = predictions.groupby("tic_id")["fold"].nunique()
    assert (spread == 1).all(), f"stars held out by >1 fold: {spread[spread > 1].to_dict()}"


def test_metrics_are_finite_and_in_range(single_run):
    payload, _ = single_run
    for fold in payload["folds"]:
        assert np.isfinite(fold["test_roc_auc"]) and 0.0 <= fold["test_roc_auc"] <= 1.0
        assert np.isfinite(fold["test_brier"]) and 0.0 <= fold["test_brier"] <= 1.0
        assert np.isfinite(fold["test_ece"]) and 0.0 <= fold["test_ece"] <= 1.0


def test_every_fold_leaves_a_servable_artefact(single_run):
    """`train.py` reloads its checkpoint before scoring ("score what ships");
    this trainer wrote no checkpoint at all, so run 1 of stage 4 scored
    weights that existed only in memory and left nothing to promote or serve."""
    _, out = single_run
    for fold in range(2):
        assert (out / f"fold_{fold}" / CHECKPOINT_NAME).is_file()
        bundle = joblib.load(out / f"fold_{fold}" / BUNDLE_NAME)
        assert {"calibrator", "platt_a", "platt_b", "scalar_constants"} <= set(bundle)


def test_the_checkpoint_reloads_without_disabling_keras_safe_mode(single_run):
    """A `Lambda` over a Python lambda needs `safe_mode=False` to deserialise,
    which would make every promoted checkpoint unloadable without waiving a
    safety check. The gating and column-picking layers are registered instead."""
    _, out = single_run
    model = tf.keras.models.load_model(out / "fold_0" / CHECKPOINT_NAME, compile=False)
    assert model.count_params() > 0


def test_an_ensemble_fold_writes_every_member_and_averages_them(ensemble_run):
    """At one model per fold the fold's score is a single seed draw, and the
    measured spread of that draw is sd 0.0106 — larger than most differences
    this project decides on."""
    payload, out = ensemble_run
    for fold in range(2):
        members = sorted((out / f"fold_{fold}").glob(f"model_*_{CHECKPOINT_NAME}"))
        assert len(members) == 2, "each member needs its own checkpoint"
    assert all(len(row["model_roc_auc"]) == 2 for row in payload["folds"])


def test_the_summary_separates_seed_variance_from_fold_difficulty(ensemble_run):
    """The reported ± has always been the spread of fold means within one run,
    read as the run's repeatability. They are different quantities."""
    payload, _ = ensemble_run
    variance = payload["summary"]["variance"]
    assert variance["n_models_per_fold"] == 2
    assert variance["seed_sd"] is not None and variance["seed_sd"] >= 0.0
    assert variance["fold_sd"] is not None


def test_seed_variance_is_unmeasurable_from_a_single_draw_per_fold(single_run):
    payload, _ = single_run
    variance = payload["summary"]["variance"]
    assert variance["seed_sd"] is None
    # One draw cannot measure spread, whichever statistic it is drawn from.
    assert variance["recall_seed_sd"] is None
    assert variance["gate_recall_seed_sd"] is None


def test_recall_at_1pct_fpr_gets_the_same_error_bar_as_auc(ensemble_run):
    """`recall @1% FPR` is the criterion that rejected all four arms of stage 4
    — run 3 on 0.145 against 0.307 — while having no variance estimate at all,
    so no margin in it could be told from noise. It now carries one."""
    payload, _ = ensemble_run
    variance = payload["summary"]["variance"]
    for _, prefix in VARIANCE_COMPONENTS:
        assert variance[f"{prefix}seed_sd"] is not None and variance[f"{prefix}seed_sd"] >= 0.0
        assert variance[f"{prefix}fold_sd"] is not None and variance[f"{prefix}fold_sd"] >= 0.0
    for row in payload["folds"]:
        assert len(row["model_recall_at_1pct_fpr"]) == 2
        assert len(row["model_gate_recall_at_1pct_fpr"]) == 2


def test_recall_at_1pct_fpr_is_not_recall_at_threshold_half():
    """The two are different numbers and the trainer recorded the wrong one for
    every run before 2026-08-08. Pinned on a hand-built ranking where they
    disagree, so the distinction does not rest on a trained model happening to
    separate them."""
    labels = np.array([1] * 4 + [0] * 100)
    # Every positive scores above every negative but below 0.5: a perfect
    # ranking, and recall at threshold 0.5 of exactly zero.
    scores = np.concatenate([np.full(4, 0.4), np.linspace(0.0, 0.3, 100)])
    assert classification_metrics(labels, scores).recall == 0.0
    assert recall_at_fpr(labels, scores, GATE_FPR) == 1.0


def test_the_recorded_recall_is_recomputable_from_the_scores_on_disk(single_run):
    """Each member's own score column is what the summary's number was measured
    from, so recomputing it there is exact — no assumption about the sign of the
    Platt fit, which on a 40-row fixture is not guaranteed to be positive."""
    payload, out = single_run
    predictions = pd.read_parquet(out / "predictions.parquet")
    for row in payload["folds"]:
        fold = predictions[predictions["fold"] == row["fold"]]
        gate = fold[fold["mission"] == GATE_MISSION]
        expected = recall_at_fpr(
            gate["label"].to_numpy(), gate["member_score_0"].to_numpy(dtype=float), GATE_FPR
        )
        assert row["model_gate_recall_at_1pct_fpr"] == [pytest.approx(expected, abs=1e-9)]


def test_the_gate_slice_is_measured_apart_from_the_whole_fold(single_run):
    """The gate reads TESS, ~44% of the rows. A noise floor measured over every
    mission is a floor for a population no decision is taken over, so the two
    are recorded separately rather than one standing in for the other."""
    payload, _ = single_run
    for row in payload["folds"]:
        assert row["n_test_gate_mission"] > 0
        assert "model_recall_at_1pct_fpr" in row and "model_gate_recall_at_1pct_fpr" in row


def test_a_non_finite_member_statistic_raises_instead_of_becoming_a_nan_sd():
    """NaN loses every inequality, so a NaN `recall_seed_sd` reads as "this
    margin is not inside the noise" — the same shape as the NaN that once
    promoted a degenerate run. Made to fire, because a guard that cannot be
    observed to fire is not a guard."""
    healthy = [
        {"model_roc_auc": [0.9, 0.91], "model_recall_at_1pct_fpr": [0.3, 0.31]},
        {"model_roc_auc": [0.8, 0.82], "model_recall_at_1pct_fpr": [0.2, 0.25]},
    ]
    assert _variance_decomposition(healthy)["recall_seed_sd"] is not None

    degenerate = [dict(row) for row in healthy]
    degenerate[1]["model_recall_at_1pct_fpr"] = [0.2, float("nan")]
    with pytest.raises(ValueError, match="fold 1 recorded a non-finite"):
        _variance_decomposition(degenerate)


def test_a_population_with_no_rows_is_unmeasured_rather_than_degenerate():
    """An empty list is "this fold held none of that population" and a NaN is
    "the population was there and came out single-class". Collapsing the two
    would either raise on a legitimate run or return a number for one that
    measured nothing."""
    rows = [
        {"model_roc_auc": [0.9], "model_gate_recall_at_1pct_fpr": []},
        {"model_roc_auc": [0.8], "model_gate_recall_at_1pct_fpr": []},
    ]
    variance = _variance_decomposition(rows)
    assert variance["gate_recall_fold_sd"] is None
    assert variance["fold_sd"] is not None


def test_the_recall_variance_keys_are_additive_and_leave_the_gate_untouched(ensemble_run):
    """The promotion gate reads named keys. The AUC pair keeps its unprefixed
    names so nothing downstream has to know this change happened."""
    payload, _ = ensemble_run
    variance = payload["summary"]["variance"]
    assert {
        "seed_sd",
        "fold_sd",
        "recall_seed_sd",
        "recall_fold_sd",
        "gate_recall_seed_sd",
        "gate_recall_fold_sd",
        "n_models_per_fold",
        "pooled_gate_recall",
        "pooled_gate_recall_seed_sd",
        "pooled_gate_recall_n_draws",
        "pooled_gate_n",
    } == set(variance)
    assert evaluate_promotion(payload, payload) is not None


def test_the_pooled_gate_statistic_is_drawn_once_per_member(ensemble_run):
    """A fold's TESS slice holds ~215 negatives in production, so its 1% FPR cut
    is two rows; the gate reads the pooled set, where it is ten. Re-forming the
    pooled set one member at a time measures the reseeding spread of the number
    the gate actually reads, with no sqrt(n) argument standing in for it."""
    payload, out = ensemble_run
    variance = payload["summary"]["variance"]
    assert variance["pooled_gate_recall_n_draws"] == 2
    assert variance["pooled_gate_recall_seed_sd"] is not None

    predictions = pd.read_parquet(out / "predictions.parquet")
    gate = predictions[predictions["mission"] == GATE_MISSION]
    assert variance["pooled_gate_n"] == len(gate)
    # Every row carries every member's score, whichever fold held it — that is
    # what makes each column a complete out-of-fold set rather than one fold's.
    for member in range(2):
        column = gate[f"member_score_{member}"].to_numpy(dtype=float)
        assert np.all(np.isfinite(column))
        expected = recall_at_fpr(gate["label"].to_numpy(), column, GATE_FPR)
        assert variance["pooled_gate_recall"][member] == pytest.approx(expected, abs=1e-9)


def test_a_member_column_with_a_hole_in_it_raises(ensemble_run):
    """A NaN score does not error — it sinks those rows to the bottom of the
    ranking and returns a plausible recall over a population that is not the one
    named. Made to fire, because a guard that cannot be observed to is not one."""
    _, out = ensemble_run
    predictions = pd.read_parquet(out / "predictions.parquet")
    predictions.loc[predictions.index[0], "member_score_1"] = np.nan
    with pytest.raises(ValueError, match="member_score_1 is not finite"):
        train_branches._pooled_member_draws(predictions)


def test_a_single_class_gate_slice_raises_rather_than_returning_a_nan_sd(ensemble_run):
    """`recall_at_fpr` returns NaN on a single-class slice rather than raising —
    correct for a metric, wrong for a noise floor, because an sd over NaN loses
    every later inequality. Empty and single-class are different failures and
    only the second is a degenerate population."""
    _, out = ensemble_run
    predictions = pd.read_parquet(out / "predictions.parquet")

    with pytest.raises(ValueError, match="single-class"):
        train_branches._pooled_member_draws(predictions.assign(label=1))

    # A run holding no gate-mission rows at all measured nothing; that is a null,
    # not a failure, and the schema is the same either way.
    drawn = train_branches._pooled_member_draws(predictions.assign(mission="Kepler"))
    assert drawn["pooled_gate_recall_seed_sd"] is None
    assert drawn["pooled_gate_recall_n_draws"] == 0


def test_the_summary_carries_the_per_mission_block_the_gate_reads(single_run):
    payload, _ = single_run
    assert "all" in payload["per_mission"]
    assert {"roc_auc", "brier", "ece", "recall_at_1pct_fpr", "n"} <= set(
        payload["per_mission"]["all"]
    )


def test_augmentation_is_training_only(shard_dir, tmp_path, monkeypatch):
    """A validation or test row must be scored as it is; augmenting it would
    make the early-stopping signal and every reported metric noise.

    Its own run: it spies on `make_viewset_dataset`, so it has to observe the
    calls rather than read an artefact a shared fixture already produced."""
    splits_augmented = []
    original = train_branches.make_viewset_dataset

    def spy(*args, augment=None, split=None, **kwargs):
        splits_augmented.append((split, augment is not None))
        return original(*args, augment=augment, split=split, **kwargs)

    monkeypatch.setattr(train_branches, "make_viewset_dataset", spy)
    run_cv(shard_dir, tmp_path / "cv", config=CVConfig(**FAST_CV))

    assert {augmented for split, augmented in splits_augmented if split is Split.TRAIN} == {True}
    assert {augmented for split, augmented in splits_augmented if split is not Split.TRAIN} == {
        False
    }


# ------------------------------- stage 8: the baseline-bias intervention arms --


def _biased_index(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """An index carrying the real confound: long baseline -> likely positive.

    Sized so every quantile stratum holds both labels. At n=240 with p=0.9/0.1 a
    stratum comes out single-class, the propensity has no inverse there, and the
    residual guard fires on the fixture rather than on the code under test — the
    interventions are covered against their own extremes in
    `test_baseline_bias.py`.
    """
    rng = np.random.default_rng(seed)
    period = rng.uniform(1.0, 20.0, n)
    expected = rng.integers(2, 300, n)
    span = (expected - 1) * period
    p = np.where(span > np.median(span), 0.8, 0.2)
    return pd.DataFrame(
        {
            "tic_id": np.arange(1, n + 1),
            "label": rng.binomial(1, p),
            "period": period,
            "expected_transit_count": expected,
        }
    )


def test_the_control_arm_changes_nothing_at_all():
    """None must be a true no-op: an arm that quietly reweighted the control
    would make every comparison against it meaningless."""
    index = _biased_index()
    kept, weights, report = train_branches._apply_baseline_intervention(index, CVConfig())
    assert kept is index
    assert weights is None
    assert report is None


def test_the_propensity_arm_returns_weights_and_keeps_every_row():
    index = _biased_index()
    kept, weights, report = train_branches._apply_baseline_intervention(
        index, CVConfig(baseline_intervention="propensity", baseline_strata=8)
    )
    assert len(kept) == len(index)
    assert weights is not None and len(weights) == len(index)
    assert report is not None
    assert abs(report["correlation_after"]) < abs(report["correlation_before"])


def test_the_stratified_arm_shrinks_the_index_and_says_by_how_much():
    index = _biased_index()
    kept, weights, report = train_branches._apply_baseline_intervention(
        index, CVConfig(baseline_intervention="stratified", baseline_strata=4)
    )
    assert weights is None
    assert len(kept) < len(index)
    assert report is not None
    assert report["n_dropped"] == len(index) - len(kept)


def test_the_arm_is_recorded_so_two_runs_are_distinguishable():
    """Without this in run_config, a control and an intervention run differ only
    in numbers nobody can attribute — the defect that made run 1's comparison
    against the incumbent unreadable."""
    _, _, report = train_branches._apply_baseline_intervention(
        _biased_index(), CVConfig(baseline_intervention="propensity", baseline_strata=8)
    )
    assert report is not None and report["arm"] == "propensity"
    assert report["n_strata"] == 8


def test_an_unknown_arm_raises_rather_than_running_the_control():
    """A typo that silently ran the control would record an intervention that
    never happened."""
    with pytest.raises(ValueError, match="unknown baseline_intervention"):
        train_branches._apply_baseline_intervention(
            _biased_index(), CVConfig(baseline_intervention="propnesity")
        )


def test_a_duplicated_tic_id_refuses_because_the_tables_key_on_it():
    """The weight and split tables look up by tic_id, so a multi-planet host
    would silently share one weight across all of its planets."""
    index = pd.concat([_biased_index(n=200), _biased_index(n=200)], ignore_index=True)
    with pytest.raises(ValueError, match="carries duplicates"):
        train_branches._apply_baseline_intervention(
            index, CVConfig(baseline_intervention="propensity")
        )


# ------------------------------------------ per-epoch training curves (2026-08-13) --


def test_every_fold_records_a_curve_per_member(ensemble_run):
    """`patience` is otherwise a number nobody can check: a summary reporting
    only final metrics cannot say whether early stopping fired at epoch 9 or ran
    to the ceiling, so neither over- nor under-training is diagnosable after the
    fact."""
    summary, _ = ensemble_run
    for fold in summary["folds"]:
        assert "epoch_history" in fold
        assert len(fold["epoch_history"]) == fold_member_count(summary)


def fold_member_count(summary: dict) -> int:
    return int(summary["run_config"]["n_models_per_fold"])


def test_the_curve_carries_train_and_validation_series(ensemble_run):
    summary, _ = ensemble_run
    history = summary["folds"][0]["epoch_history"][0]
    assert "loss" in history and "val_loss" in history
    # Stopping is on val_pr_auc, so it has to be recorded or the stop point
    # cannot be located on the curve that decided it.
    assert any("pr_auc" in k for k in history)


def test_the_curve_is_json_serialisable_and_finite(ensemble_run):
    """np.float32 is not JSON-serialisable, and a NaN would sail through
    json.dump only to become `NaN`, which is not valid JSON for strict readers."""
    summary, _ = ensemble_run
    history = summary["folds"][0]["epoch_history"][0]
    for name, series in history.items():
        assert all(isinstance(v, float) for v in series), name
        assert all(math.isfinite(v) for v in series), name
    json.dumps(history)


def test_the_curve_length_shows_where_early_stopping_landed(ensemble_run):
    """The whole point: epochs run is recoverable, so `patience` is auditable."""
    summary, _ = ensemble_run
    ran = len(summary["folds"][0]["epoch_history"][0]["loss"])
    assert 1 <= ran <= int(summary["run_config"]["epochs"])
