"""The offline control-arm driver — the guards that stop a wrong number.

`test_control_arm.py` covers `eval/control_arm.py`, the library. This file
covers `scripts/control_arm.py`, the driver, which had no test of its own until
2026-08-12 — including the alignment guard added the same week to fix a defect
that had already silently overwritten two thirds of a measurement.

Every guard here is made to fire, because each failure it prevents returns a
plausible pass rate: a shard stream read back in the wrong order, a run
directory scored down the wrong lane, and a scalar frame written a column short.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.viewset_tfrecords import FEATURE_COLUMNS, MASK_COLUMNS


def _driver():
    """The script as a module. It is not importable as a package path."""
    spec = importlib.util.spec_from_file_location(
        "_control_arm_driver",
        Path(__file__).resolve().parents[1] / "scripts" / "control_arm.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _driver()


# --------------------------------------------------------------------------
# The alignment guard. This is the one that matters.
# --------------------------------------------------------------------------


def test_an_aligned_stream_passes():
    index = pd.DataFrame({"tic_id": [11, 22, 33]})
    driver.assert_stream_aligned(np.array([11, 22, 33]), index)


def test_a_permuted_stream_raises_even_though_every_tic_id_is_present():
    """The exact defect the guard was built for.

    The earlier implementation matched scores onto rows by `tic_id`. Under a
    permutation every id still matches, so it wrote a score for every row and
    reported a believable rate — while the three periods of one host collapsed
    onto a single score and the last write won. A guard that only checked
    membership would pass this case, so the test permutes rather than substitutes.
    """
    index = pd.DataFrame({"tic_id": [11, 22, 33]})
    with pytest.raises(RuntimeError, match="not aligned"):
        driver.assert_stream_aligned(np.array([33, 11, 22]), index)


def test_a_host_repeated_per_period_is_still_ordered_not_just_counted():
    """Three rows per host is the real shape — 580 hosts x 3 periods.

    The multiset is identical under this permutation, so anything comparing
    sorted ids or value counts passes it.
    """
    index = pd.DataFrame({"tic_id": [11, 11, 11, 22, 22, 22]})
    driver.assert_stream_aligned(np.array([11, 11, 11, 22, 22, 22]), index)
    with pytest.raises(RuntimeError, match="not aligned"):
        driver.assert_stream_aligned(np.array([11, 11, 22, 11, 22, 22]), index)


def test_a_short_stream_raises_rather_than_broadcasting():
    index = pd.DataFrame({"tic_id": [11, 22, 33]})
    with pytest.raises(RuntimeError, match="row"):
        driver.assert_stream_aligned(np.array([11, 22]), index)


def test_the_guard_reports_how_many_rows_moved():
    """The message has to say how bad it is, not just that it happened."""
    index = pd.DataFrame({"tic_id": [11, 22, 33, 44]})
    with pytest.raises(RuntimeError, match="2 of 4"):
        driver.assert_stream_aligned(np.array([11, 22, 44, 33]), index)


# --------------------------------------------------------------------------
# Lane selection. Scoring a dual-view run down the eleven-view path fails with
# a shape error at best and a wrong number at worst.
# --------------------------------------------------------------------------


def _fold_0(tmp_path: Path, *names: str) -> Path:
    fold = tmp_path / "fold_0"
    fold.mkdir(parents=True)
    for name in names:
        (fold / name).write_bytes(b"")
    return tmp_path


def test_a_branch_run_is_recognised_from_its_checkpoints(tmp_path):
    run = _fold_0(tmp_path, "model_0_cnn_branches.keras", "model_1_cnn_branches.keras")
    assert driver.run_kind(run) == "branch"


def test_a_dualview_run_is_recognised_from_its_checkpoint(tmp_path):
    assert driver.run_kind(_fold_0(tmp_path, "cnn_dualview.keras")) == "dualview"


def test_a_run_carrying_both_kinds_refuses_to_choose(tmp_path):
    run = _fold_0(tmp_path, "model_0_cnn_branches.keras", "cnn_dualview.keras")
    with pytest.raises(ValueError, match="both checkpoint kinds"):
        driver.run_kind(run)


def test_a_run_carrying_neither_kind_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        driver.run_kind(_fold_0(tmp_path, "cv_summary.json"))


# A multi-member dual-view run numbers its checkpoints, so the bare name stops
# existing. Until 2026-08-15 this lane matched the exact name only, and stage
# 10.5's own dual-view run — the one its ensemble is measured against — could
# not be scored at all. The branch lane globbed from the start and was fine,
# which is why the asymmetry survived a whole stage.


def test_a_multi_member_dualview_run_is_recognised(tmp_path):
    run = _fold_0(
        tmp_path,
        "model_0_cnn_dualview.keras",
        "model_1_cnn_dualview.keras",
        "model_2_cnn_dualview.keras",
    )
    assert driver.run_kind(run) == "dualview"


def test_every_member_is_returned_so_none_is_silently_dropped(tmp_path):
    run = _fold_0(
        tmp_path,
        "model_0_cnn_dualview.keras",
        "model_1_cnn_dualview.keras",
        "model_2_cnn_dualview.keras",
    )
    members = driver.dualview_members(run / "fold_0")
    assert [p.name for p in members] == [
        "model_0_cnn_dualview.keras",
        "model_1_cnn_dualview.keras",
        "model_2_cnn_dualview.keras",
    ]


def test_a_single_member_run_still_resolves_by_its_historical_name(tmp_path):
    members = driver.dualview_members(_fold_0(tmp_path, "cnn_dualview.keras") / "fold_0")
    assert [p.name for p in members] == ["cnn_dualview.keras"]


def test_the_numbered_layout_wins_when_both_are_present(tmp_path):
    """Not a layout any trainer writes, but a stale bare checkpoint beside a
    numbered set would otherwise score one member and call it the ensemble."""
    run = _fold_0(tmp_path, "cnn_dualview.keras", "model_0_cnn_dualview.keras")
    assert [p.name for p in driver.dualview_members(run / "fold_0")] == [
        "model_0_cnn_dualview.keras"
    ]


def test_a_fold_with_no_dualview_checkpoint_returns_nothing(tmp_path):
    assert driver.dualview_members(_fold_0(tmp_path, "cv_summary.json") / "fold_0") == []


# The aux width is a property of the run, not a constant. The incumbent serves
# the 9-dim legacy layout; every dual-view run trained since the vetting
# features landed expects 13. Assuming 9 raised inside sklearn's imputer *after*
# the entire host view build had been paid for.


class _FakePipeline:
    def __init__(self, n):
        self.n_features_in_ = n


class _UndeclaredPipeline:
    """An aux pipeline that does not say how wide it is. Module-level so joblib
    can pickle it."""


def _bundle(tmp_path: Path, pipeline) -> Path:
    import joblib

    fold = tmp_path / "fold_0"
    fold.mkdir(parents=True, exist_ok=True)
    joblib.dump({"aux_pipeline": pipeline, "calibrator": None}, fold / "cnn_calibrator.joblib")
    return tmp_path


def test_the_aux_width_comes_from_the_runs_own_pipeline(tmp_path):
    assert driver.dualview_aux_dim(_bundle(tmp_path, _FakePipeline(13))) == 13


def test_a_legacy_nine_dim_run_still_reads_as_nine(tmp_path):
    assert driver.dualview_aux_dim(_bundle(tmp_path, _FakePipeline(9))) == 9


def test_a_bundle_with_no_pipeline_falls_back_to_the_legacy_width(tmp_path):
    assert driver.dualview_aux_dim(_bundle(tmp_path, None)) == 9


def test_a_pipeline_that_declares_no_width_raises_rather_than_guessing(tmp_path):
    """Guessing here is what produced the original failure. An undeclared width
    is unknowable, and a wrong guess costs a full build before it surfaces."""
    with pytest.raises(ValueError, match="n_features_in_"):
        driver.dualview_aux_dim(_bundle(tmp_path, _UndeclaredPipeline()))


# --------------------------------------------------------------------------
# The scalar frame. A column short here still passes every downstream gate.
# --------------------------------------------------------------------------


def test_every_declared_scalar_is_present_so_the_vector_is_never_short():
    """Written through the schema, not a literal list.

    `write_viewset_shards` raises on an absent declared column, but only if this
    function actually produced the frame it is given — the point of the test is
    that a branch added to `FEATURE_COLUMNS` cannot leave this path writing a
    shorter vector that normalisation then happily accepts.
    """
    frame = driver.control_scalars([{"tic_id": 1, "mission": "TESS", "period": 3.0}])
    absent = [c for c in (*FEATURE_COLUMNS, *MASK_COLUMNS) if c not in frame.columns]
    assert absent == []


def test_unmeasured_scalars_are_nan_and_masks_are_false():
    """NaN, not 0.0 — the reader imputes NaN to the fold's own fitted centre,
    where it carries no information. A zero lands at a real percentile of the
    column and reads as a weak measurement."""
    frame = driver.control_scalars([{"tic_id": 1, "mission": "TESS", "period": 3.0}])
    for column in FEATURE_COLUMNS:
        assert frame[column].isna().all(), f"{column} should be unmeasured, not filled"
    for column in MASK_COLUMNS:
        assert not frame[column].any(), f"{column} should be False"


def test_supplied_scalars_survive_the_schema_fill():
    """The fill must not overwrite what the caller measured."""
    rows = [{"tic_id": 1, "mission": "TESS", "observed_transit_count": 7, "dv_usable": False}]
    frame = driver.control_scalars(rows)
    assert int(frame["observed_transit_count"].iloc[0]) == 7


# --------------------------------------------------------------------------
# The synthetic ephemeris.
# --------------------------------------------------------------------------


def test_transit_duration_scales_as_the_cube_root_of_period():
    """A central transit of a Sun-like star, scaled from Earth's 13 h."""
    assert driver.transit_duration_hours(365.25) == pytest.approx(13.0)
    # Eight times the period is twice the duration.
    assert driver.transit_duration_hours(8.0) == pytest.approx(
        2.0 * driver.transit_duration_hours(1.0)
    )
    for period in (3.0, 7.0, 12.0):
        assert 0.0 < driver.transit_duration_hours(period) < 24.0
