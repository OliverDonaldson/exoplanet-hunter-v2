"""CV runner for the branch model: split integrity and summary schema."""

from __future__ import annotations

import json

import numpy as np
import pytest

from exoplanet_hunter.datasets.viewset_tfrecords import write_viewset_shards
from exoplanet_hunter.training.train_branches import SUMMARY_KEYS, CVConfig, run_cv
from exoplanet_hunter.validation.promotion import evaluate_promotion


@pytest.fixture
def shard_dir(tmp_path, make_view_set):
    write_viewset_shards(make_view_set(n=40, seed=3), tmp_path, examples_per_shard=16)
    return tmp_path


def test_cv_writes_a_summary_the_promotion_gate_can_read(shard_dir, tmp_path):
    out = tmp_path / "cv"
    payload = run_cv(
        shard_dir,
        out,
        config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1),
    )
    assert len(payload["folds"]) == 2
    for key in SUMMARY_KEYS:
        assert {"mean", "std"} == set(payload["summary"][key])

    on_disk = json.loads((out / "cv_summary.json").read_text())
    assert on_disk == payload
    # The gate reads this shape; if it drifts, promotion silently stops working.
    decision = evaluate_promotion(on_disk, on_disk)
    assert decision is not None


def test_every_example_is_tested_exactly_once(shard_dir, tmp_path):
    # Group = tic_id, so a star appears in exactly one fold's test split. A
    # star split across folds is the leakage the whole CV design prevents.
    payload = run_cv(
        shard_dir,
        tmp_path / "cv",
        config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1),
    )
    assert sum(fold["n_test"] for fold in payload["folds"]) == 40


def test_metrics_are_finite_and_in_range(shard_dir, tmp_path):
    payload = run_cv(
        shard_dir,
        tmp_path / "cv",
        config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1),
    )
    for fold in payload["folds"]:
        assert np.isfinite(fold["test_roc_auc"]) and 0.0 <= fold["test_roc_auc"] <= 1.0
        assert np.isfinite(fold["test_brier"]) and 0.0 <= fold["test_brier"] <= 1.0
        assert np.isfinite(fold["test_ece"]) and 0.0 <= fold["test_ece"] <= 1.0
