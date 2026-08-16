"""The dual-view trainer's multi-member path and the variance it reports.

Added 2026-08-14 with `n_models_per_fold`. The training loop itself is covered
by the smoke run rather than here — what these test is the arithmetic and the
naming that decide whether a run's numbers can be read at all: the checkpoint
name every earlier run is loaded by, and the floor stage 10.5's outcome table is
read against.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.training.mlflow_utils import MEMBER_SCORE_PREFIX, member_checkpoint_name
from exoplanet_hunter.training.train import _aggregate_cv

BASE = "cnn_dualview.keras"

#: Enough rows per mission that a 1% FPR cut lands on more than one negative and
#: every slice carries both classes; a smaller set makes the recall statistics
#: degenerate rather than merely noisy.
MISSIONS = ["TESS"] * 60 + ["Kepler"] * 20 + ["K2"] * 20


def test_a_single_member_keeps_the_historical_checkpoint_name():
    """Every run before 2026-08-14 is found by this exact filename — the
    registry, the control-arm harness and the serving path all load it by path."""
    assert member_checkpoint_name(0, 1, BASE) == BASE
    assert member_checkpoint_name(0, 0, BASE) == BASE


def test_members_are_numbered_without_colliding():
    names = {member_checkpoint_name(i, 3, BASE) for i in range(3)}
    assert len(names) == 3
    assert BASE not in names, "a member must not overwrite the single-model artefact"


def test_both_trainers_share_one_member_score_contract():
    """One reader re-forms the pooled out-of-fold set for either trainer. Two
    definitions that drifted would change how many draws a bar is computed from."""
    from exoplanet_hunter.training.train_branches import (
        MEMBER_SCORE_PREFIX as branch_prefix,
    )

    assert branch_prefix is MEMBER_SCORE_PREFIX


def _rows(per_fold_members: list[list[float]]) -> list[dict]:
    """Fold rows carrying only what `_aggregate_cv` reads."""
    keys = (
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_brier",
        "test_ece",
        "best_threshold",
        "platt_a",
        "platt_b",
    )
    return [
        {
            **{k: float(np.mean(members)) for k in keys},
            "model_roc_auc": members,
            "n_models_per_fold": len(members),
        }
        for members in per_fold_members
    ]


def write_predictions(cv_root: Path, n_members: int, n_folds: int, *, mission: bool = True) -> None:
    """The pooled out-of-fold set the cross-validation loop writes before it
    aggregates.

    The two classes overlap. Perfectly separated scores put recall at 1.0 for
    every member, and a floor of exactly zero cannot be told apart from one that
    was never measured — which is the failure these fixtures exist to catch.
    Each member is jittered independently so the pooled draws differ.
    """
    rng = np.random.default_rng(0)
    n = len(MISSIONS)
    label = np.tile([1, 0], n // 2)
    score = np.where(label == 1, rng.uniform(0.35, 0.95, n), rng.uniform(0.05, 0.65, n))
    frame = pd.DataFrame(
        {
            "row": np.arange(n),
            "tic_id": np.arange(n) + 1,
            "fold": np.arange(n) % n_folds,
            "y_true": label,
            "prob_raw": score,
            "prob_calibrated": score,
        }
    )
    if mission:
        frame["mission"] = MISSIONS
    # Only above one member, because that is what the trainer does: a single
    # model has no second draw to disagree with, and a lone member column would
    # let a fixture report a draw count the real run never writes.
    if n_members > 1:
        for member in range(n_members):
            frame[f"{MEMBER_SCORE_PREFIX}{member}"] = np.clip(score + rng.normal(0, 0.06, n), 0, 1)
    cv_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cv_root / "predictions.parquet", index=False)


def write_label_catalogue(repo_root: Path) -> None:
    """The catalogue the summary joins missions from when predictions omit them,
    at the path it is resolved to positionally from the run directory."""
    path = repo_root / "data" / "tables" / "labels" / "labels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tic_id": np.arange(len(MISSIONS)) + 1, "mission": MISSIONS}).to_parquet(
        path, index=False
    )


@pytest.fixture
def aggregate(tmp_path: Path):
    """`_aggregate_cv` under an MLflow run scoped to tmp, returning the payload.

    The run directory is laid out as `models/cv/<run>` because the summary
    resolves the label catalogue positionally from it, and the pooled
    predictions are written because the trainer writes them immediately before
    it aggregates. A fixture without them exercises a state the trainer cannot
    produce, and every field the promotion gate reads comes from them.
    """

    def run(per_fold_members: list[list[float]], *, mission: bool = True) -> dict:
        # sqlite, not a file store: MLflow's filesystem backend now refuses to
        # start, and the repo's own tracking DB must not collect test runs.
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        cv_root = tmp_path / "models" / "cv" / "run"
        write_predictions(cv_root, len(per_fold_members[0]), len(per_fold_members), mission=mission)
        if not mission:
            write_label_catalogue(tmp_path)
        with mlflow.start_run():
            _aggregate_cv(_rows(per_fold_members), cv_root)
        return json.loads((cv_root / "cv_summary.json").read_text())

    return run


def test_the_spread_it_averaged_over_is_reported(aggregate):
    variance = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])["summary"]["variance"]
    assert variance["n_models_per_fold"] == 3
    assert variance["n_folds"] == 2
    # Mean of the within-fold sds; both folds have the same 0.02 spacing.
    assert variance["seed_sd"] == pytest.approx(np.std([0.90, 0.92, 0.94], ddof=1))
    assert variance["fold_sd"] == pytest.approx(np.std([0.92, 0.82], ddof=1))


def test_one_member_reports_no_seed_noise_rather_than_none_of_it(aggregate):
    """`None` is "nobody measured this". Zero would read as "the noise is zero"
    in exactly the comparison the number exists to arbitrate."""
    variance = aggregate([[0.91], [0.89]])["summary"]["variance"]
    assert variance["seed_sd"] is None
    assert variance["n_models_per_fold"] == 1
    assert variance["fold_sd"] == pytest.approx(np.std([0.91, 0.89], ddof=1))


def test_a_single_fold_reports_no_fold_noise(aggregate):
    variance = aggregate([[0.90, 0.92]])["summary"]["variance"]
    assert variance["fold_sd"] is None
    assert variance["seed_sd"] == pytest.approx(np.std([0.90, 0.92], ddof=1))


def test_the_variance_block_does_not_disturb_the_fold_means(aggregate):
    """The gate reads these; adding a block beside them must not move them."""
    summary = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])["summary"]
    assert summary["test_roc_auc"]["mean"] == pytest.approx(np.mean([0.92, 0.82]))
    assert summary["test_roc_auc"]["std"] == pytest.approx(np.std([0.92, 0.82]))


def test_a_run_with_no_pooled_predictions_raises_rather_than_writing_a_thin_summary(tmp_path):
    """Without them the summary carries no mission block and no recall floor,
    and the gate refuses it on provenance — a refusal a reader cannot tell apart
    from the candidate having lost. The trainer writes this file immediately
    before aggregating, so its absence means the run did not finish."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    cv_root = tmp_path / "models" / "cv" / "run"
    cv_root.mkdir(parents=True)
    with mlflow.start_run(), pytest.raises(FileNotFoundError, match="pooled out-of-fold"):
        _aggregate_cv(_rows([[0.90, 0.92, 0.94]]), cv_root)
    assert not (cv_root / "cv_summary.json").exists()


# --------------------------------------------------------------------------
# What the promotion gate actually reads. The trainer wrote fold means and a
# variance block and neither of the two fields the gate decides on, so every
# candidate it produced was refused on provenance before a metric was compared.
# These assertions live here, on the trainer that had the hole, not only on the
# one that never did.
# --------------------------------------------------------------------------


def test_the_summary_carries_the_slice_the_gate_decides_on(aggregate):
    payload = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])
    assert "per_mission" in payload
    tess = payload["per_mission"]["TESS"]
    assert tess["n"] == MISSIONS.count("TESS")
    assert 0.0 < tess["recall_at_1pct_fpr"] < 1.0
    # The diagnostics and the pooled slice are reported beside it, never instead.
    assert {"Kepler", "K2", "all"} <= set(payload["per_mission"])


def test_the_summary_carries_a_recall_floor_the_gate_can_size_its_tolerance_from(aggregate):
    """Without this the gate falls back to a constant nobody measured against
    this run, on the one criterion that has rejected every arm."""
    variance = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])["summary"]["variance"]
    assert variance["pooled_gate_recall_seed_sd"] is not None
    assert variance["pooled_gate_recall_n_draws"] == 3
    assert len(variance["pooled_gate_recall"]) == 3
    assert variance["pooled_gate_n"] == MISSIONS.count("TESS")


def test_one_member_reports_no_recall_floor_rather_than_a_zero_one(aggregate):
    """A single member has nothing to disagree with. Zero would read as "this run
    is noiseless" in the comparison the number exists to arbitrate."""
    variance = aggregate([[0.91], [0.89]])["summary"]["variance"]
    assert variance["pooled_gate_recall_seed_sd"] is None
    assert variance["pooled_gate_recall_n_draws"] == 0


def test_the_gate_reaches_its_criteria_on_this_summary_instead_of_refusing(aggregate):
    """The end of the path: a summary this trainer wrote, read by the real gate.
    Compared against itself it is a tie and does not promote — what is pinned is
    that it gets as far as the recall criterion rather than being refused on
    paperwork, which is what happened to every candidate before."""
    from exoplanet_hunter.validation import decision_floor, evaluate_promotion

    payload = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])
    assert decision_floor(payload).recall is not None

    decision = evaluate_promotion(payload, payload)
    assert not any("predates the per_mission block" in r for r in decision.reasons)
    assert not any("populations differ" in r for r in decision.reasons)
    assert any("gated on TESS" in r for r in decision.reasons)


def test_the_mission_is_joined_from_the_catalogue_when_predictions_omit_it(aggregate):
    """The trainer's own prediction set carries no mission column, so the join is
    the live path — not the shortcut of a fixture that writes one."""
    payload = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]], mission=False)
    assert payload["per_mission"]["TESS"]["n"] == MISSIONS.count("TESS")


def test_the_repository_root_is_asserted_rather_than_assumed(tmp_path):
    """The label catalogue is resolved by counting parents up from the run
    directory. That derivation is an assumption about layout, and layout has
    moved before; a silently wrong root writes a summary with no mission block."""
    from exoplanet_hunter.training.train import LABELS_RELATIVE, _labels_path

    assert _labels_path(tmp_path / "models" / "cv" / "run") == tmp_path / LABELS_RELATIVE
    with pytest.raises(ValueError, match="models/cv"):
        _labels_path(tmp_path / "somewhere" / "else" / "run")
