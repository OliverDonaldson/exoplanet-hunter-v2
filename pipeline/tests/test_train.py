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
import pytest

from exoplanet_hunter.training.mlflow_utils import MEMBER_SCORE_PREFIX, member_checkpoint_name
from exoplanet_hunter.training.train import _aggregate_cv

BASE = "cnn_dualview.keras"


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


@pytest.fixture
def aggregate(tmp_path: Path):
    """`_aggregate_cv` under an MLflow run scoped to tmp, returning the summary."""

    def run(per_fold_members: list[list[float]]) -> dict:
        # sqlite, not a file store: MLflow's filesystem backend now refuses to
        # start, and the repo's own tracking DB must not collect test runs.
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        cv_root = tmp_path / "cv"
        with mlflow.start_run():
            _aggregate_cv(_rows(per_fold_members), cv_root)
        return json.loads((cv_root / "cv_summary.json").read_text())["summary"]

    return run


def test_the_spread_it_averaged_over_is_reported(aggregate):
    summary = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])
    variance = summary["variance"]
    assert variance["n_models_per_fold"] == 3
    assert variance["n_folds"] == 2
    # Mean of the within-fold sds; both folds have the same 0.02 spacing.
    assert variance["seed_sd"] == pytest.approx(np.std([0.90, 0.92, 0.94], ddof=1))
    assert variance["fold_sd"] == pytest.approx(np.std([0.92, 0.82], ddof=1))


def test_one_member_reports_no_seed_noise_rather_than_none_of_it(aggregate):
    """`None` is "nobody measured this". Zero would read as "the noise is zero"
    in exactly the comparison the number exists to arbitrate."""
    variance = aggregate([[0.91], [0.89]])["variance"]
    assert variance["seed_sd"] is None
    assert variance["n_models_per_fold"] == 1
    assert variance["fold_sd"] == pytest.approx(np.std([0.91, 0.89], ddof=1))


def test_a_single_fold_reports_no_fold_noise(aggregate):
    variance = aggregate([[0.90, 0.92]])["variance"]
    assert variance["fold_sd"] is None
    assert variance["seed_sd"] == pytest.approx(np.std([0.90, 0.92], ddof=1))


def test_the_variance_block_does_not_disturb_the_fold_means(aggregate):
    """The gate reads these; adding a block beside them must not move them."""
    summary = aggregate([[0.90, 0.92, 0.94], [0.80, 0.82, 0.84]])
    assert summary["test_roc_auc"]["mean"] == pytest.approx(np.mean([0.92, 0.82]))
    assert summary["test_roc_auc"]["std"] == pytest.approx(np.std([0.92, 0.82]))
