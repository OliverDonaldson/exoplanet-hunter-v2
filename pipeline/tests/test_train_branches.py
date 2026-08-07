"""CV runner for the branch model: split integrity and summary schema."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_pipeline import Split
from exoplanet_hunter.datasets.viewset_tfrecords import write_viewset_shards
from exoplanet_hunter.training import train_branches
from exoplanet_hunter.training.train_branches import (
    BUNDLE_NAME,
    CHECKPOINT_NAME,
    SUMMARY_KEYS,
    CVConfig,
    run_cv,
)
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


def test_every_fold_leaves_a_servable_artefact(shard_dir, tmp_path):
    """`train.py` reloads its checkpoint before scoring ("score what ships");
    this trainer wrote no checkpoint at all, so run 1 of stage 2(a) scored
    weights that existed only in memory and left nothing to promote or serve."""
    out = tmp_path / "cv"
    run_cv(shard_dir, out, config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1))

    for fold in range(2):
        assert (out / f"fold_{fold}" / CHECKPOINT_NAME).is_file()
        bundle = joblib.load(out / f"fold_{fold}" / BUNDLE_NAME)
        assert {"calibrator", "platt_a", "platt_b", "scalar_constants"} <= set(bundle)


def test_the_checkpoint_reloads_without_disabling_keras_safe_mode(shard_dir, tmp_path):
    """A `Lambda` over a Python lambda needs `safe_mode=False` to deserialise,
    which would make every promoted checkpoint unloadable without waiving a
    safety check. The gating and column-picking layers are registered instead."""
    out = tmp_path / "cv"
    run_cv(shard_dir, out, config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1))

    model = tf.keras.models.load_model(out / "fold_0" / CHECKPOINT_NAME, compile=False)
    assert model.count_params() > 0


def test_an_ensemble_fold_writes_every_member_and_averages_them(shard_dir, tmp_path):
    """At one model per fold the fold's score is a single seed draw, and the
    measured spread of that draw is sd 0.0106 — larger than most differences
    this project decides on."""
    out = tmp_path / "cv"
    payload = run_cv(
        shard_dir,
        out,
        config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1, n_models_per_fold=2),
    )
    for fold in range(2):
        members = sorted((out / f"fold_{fold}").glob(f"model_*_{CHECKPOINT_NAME}"))
        assert len(members) == 2, "each member needs its own checkpoint"
    assert all(len(row["model_roc_auc"]) == 2 for row in payload["folds"])


def test_the_summary_separates_seed_variance_from_fold_difficulty(shard_dir, tmp_path):
    """The reported ± has always been the spread of fold means within one run,
    read as the run's repeatability. They are different quantities."""
    payload = run_cv(
        shard_dir,
        tmp_path / "cv",
        config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1, n_models_per_fold=2),
    )
    variance = payload["summary"]["variance"]
    assert variance["n_models_per_fold"] == 2
    assert variance["seed_sd"] is not None and variance["seed_sd"] >= 0.0
    assert variance["fold_sd"] is not None


def test_seed_variance_is_unmeasurable_from_a_single_draw_per_fold(shard_dir, tmp_path):
    payload = run_cv(
        shard_dir, tmp_path / "cv", config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1)
    )
    assert payload["summary"]["variance"]["seed_sd"] is None


def test_the_summary_carries_the_per_mission_block_the_gate_reads(shard_dir, tmp_path):
    payload = run_cv(
        shard_dir, tmp_path / "cv", config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1)
    )
    assert "all" in payload["per_mission"]
    assert {"roc_auc", "brier", "ece", "recall_at_1pct_fpr", "n"} <= set(
        payload["per_mission"]["all"]
    )


def test_augmentation_is_training_only(shard_dir, tmp_path, monkeypatch):
    """A validation or test row must be scored as it is; augmenting it would
    make the early-stopping signal and every reported metric noise."""
    splits_augmented = []
    original = train_branches.make_viewset_dataset

    def spy(*args, augment=None, split=None, **kwargs):
        splits_augmented.append((split, augment is not None))
        return original(*args, augment=augment, split=split, **kwargs)

    monkeypatch.setattr(train_branches, "make_viewset_dataset", spy)
    run_cv(
        shard_dir, tmp_path / "cv", config=CVConfig(n_splits=2, epochs=1, batch_size=8, patience=1)
    )

    assert {augmented for split, augmented in splits_augmented if split is Split.TRAIN} == {True}
    assert {augmented for split, augmented in splits_augmented if split is not Split.TRAIN} == {
        False
    }
