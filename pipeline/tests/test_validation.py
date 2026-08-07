"""Tests for the validation gates: schemas, leakage guard, promotion."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from exoplanet_hunter.datasets import ViewArrays
from exoplanet_hunter.validation import (
    assert_refresh_safe,
    candidate_catalogue_schema,
    check_catalogue_shrink,
    check_dv_archive,
    check_views,
    diff_label_catalogues,
    drop_quarantined,
    evaluate_promotion,
    label_catalogue_schema,
    load_incumbent_summary,
    load_quarantine,
    paired_folds,
    promote,
    publishable_cv_dirs,
    quarantine_tics,
    record_quarantine,
)

# ------------------------------------------------------------------ schemas --


def good_labels(n: int = 7) -> pd.DataFrame:
    """A well-formed slice of the real catalogue: TESS, Kepler and K2 rows."""
    return pd.DataFrame(
        {
            "tic_id": np.arange(1, n + 1),
            "period": np.linspace(1.0, 10.0, n),
            "t0": np.full(n, 2458326.0),
            "duration": np.full(n, 0.1),
            "depth": np.full(n, 500.0),
            "disposition": ["CP", "KP", "FP", "FA", "PC", "CONFIRMED", "REFUTED"][:n],
            "label": [1, 1, 0, 0, -1, 1, 0][:n],
            "mission": ["TESS", "TESS", "TESS", "TESS", "TESS", "Kepler", "K2"][:n],
        }
    )


def test_label_schema_accepts_good_catalogue():
    label_catalogue_schema.validate(good_labels(), lazy=True)


def test_label_schema_rejects_bad_domain_and_duplicates():
    bad = good_labels()
    bad.loc[0, "disposition"] = "MAYBE"
    with pytest.raises(pandera.errors.SchemaErrors):
        label_catalogue_schema.validate(bad, lazy=True)

    dup = pd.concat([good_labels(), good_labels().iloc[[0]]], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        label_catalogue_schema.validate(dup, lazy=True)


def test_label_schema_rejects_single_class_catalogue():
    one_class = good_labels()
    one_class["label"] = 1
    with pytest.raises(pandera.errors.SchemaErrors):
        label_catalogue_schema.validate(one_class, lazy=True)


def test_label_schema_accepts_koi_vocabulary():
    """Regression (2026-07-12): the first multi-mission build failed the gate
    because Kepler rows carry koi_disposition strings, not TFOPWG codes."""
    df = good_labels()
    df.loc[5, ["disposition", "label", "mission"]] = ["CONFIRMED", 1, "Kepler"]
    df.loc[4, ["disposition", "label", "mission"]] = ["FALSE POSITIVE", 0, "Kepler"]
    label_catalogue_schema.validate(df, lazy=True)


def test_label_schema_accepts_k2_vocabulary():
    """Regression (2026-07-25): the Step-2c K2 build (7ed5603) failed the gate
    because "K2" was never added to the mission domain, and k2pandc's REFUTED
    was missing from the disposition domain."""
    df = good_labels()
    for disposition, label in (
        ("CONFIRMED", 1),
        ("FALSE POSITIVE", 0),
        ("REFUTED", 0),
        ("CANDIDATE", -1),
    ):
        df.loc[6, ["disposition", "label", "mission"]] = [disposition, label, "K2"]
        label_catalogue_schema.validate(df, lazy=True)


def test_label_schema_rejects_unknown_mission():
    df = good_labels()
    df.loc[6, "mission"] = "CHEOPS"
    with pytest.raises(pandera.errors.SchemaErrors):
        label_catalogue_schema.validate(df, lazy=True)


def test_label_schema_rejects_zero_duration_but_allows_unknown():
    """Regression (2026-07-25): two K2 rows carried k2pandc's `pl_trandur = 0`
    placeholder. A zero-length transit is unphysical, so the ingest maps it to
    NaN ("unknown") and the gate keeps rejecting a literal 0."""
    df = good_labels()
    df.loc[6, "duration"] = 0.0
    with pytest.raises(pandera.errors.SchemaErrors):
        label_catalogue_schema.validate(df, lazy=True)

    df.loc[6, "duration"] = np.nan
    label_catalogue_schema.validate(df, lazy=True)


def test_label_schema_allows_same_tic_across_missions():
    df = good_labels()
    df.loc[1, "tic_id"] = df.loc[0, "tic_id"]
    df.loc[1, "mission"] = "Kepler"
    label_catalogue_schema.validate(df, lazy=True)


def test_candidate_schema_accepts_real_shape():
    df = pd.DataFrame(
        {
            "source": ["TOI", "CTOI"],
            "name": ["TOI-101.01", "TIC 160363.01"],
            "tic_id": [231663901, 160363],
            "disposition": ["KP", None],
            "ra_deg": [318.7, 12.0],
            "dec_deg": [-55.9, 0.0],
            "period_days": [1.43, 0.0],  # ExoFOP uses 0.0 for unknown
            "duration_hours": [1.6, None],
            "depth_ppm": [18960.7, 890.0],
            "tess_mag": [12.4, 9.1],
        }
    )
    candidate_catalogue_schema.validate(df, lazy=True)


def test_candidate_schema_rejects_null_name_and_bad_coords():
    df = pd.DataFrame(
        {
            "source": ["TOI"],
            "name": [None],
            "tic_id": [1],
            "disposition": ["PC"],
            "ra_deg": [400.0],
            "dec_deg": [-95.0],
            "period_days": [1.0],
            "duration_hours": [1.0],
            "depth_ppm": [10.0],
            "tess_mag": [10.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        candidate_catalogue_schema.validate(df, lazy=True)


# -------------------------------------------------------------------- views --


def make_views(n: int = 8) -> ViewArrays:
    rng = np.random.default_rng(0)
    return ViewArrays(
        global_views=rng.normal(size=(n, 32)).astype(np.float32),
        local_views=rng.normal(size=(n, 8)).astype(np.float32),
        labels=np.array([0, 1] * (n // 2), dtype=np.int8),
        tic_ids=np.arange(1, n + 1, dtype=np.int64),
        aux_features=rng.normal(size=(n, 3)).astype(np.float32),
    )


def test_check_views_passes_clean_set():
    assert check_views(make_views()) == []


def test_check_views_flags_all_nan_fold_and_label_domain():
    views = make_views()
    views.global_views[2, :] = np.nan
    views.labels = views.labels.astype(np.int8)
    views.labels[0] = -1
    problems = check_views(views)
    assert any("all-NaN" in p for p in problems)
    assert any("labels" in p for p in problems)


def test_check_views_flags_dead_aux_column():
    views = make_views()
    views.aux_features[:, 1] = np.nan
    assert any("aux_features" in p for p in check_views(views))


# ----------------------------------------------------------------- DV archive --


def make_dv_archive(tmp_path: Path, tics: list[int], *, without_dv: tuple[int, ...] = ()) -> Path:
    cache = tmp_path / "raw_dv"
    cache.mkdir()
    manifest: dict[str, dict] = {}
    for tic in tics:
        if tic in without_dv:
            manifest[str(tic)] = {"success": False, "n_available": 0, "reason": "no DV products"}
            continue
        path = cache / f"tic_{tic}" / f"tess-s0001-s0009-{tic:016d}-00001_dvr.xml"
        path.parent.mkdir()
        path.write_text("<xml/>")
        manifest[str(tic)] = {"success": True, "n_available": 1, "paths": [str(path)]}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return cache


def test_dv_gate_accepts_an_archive_where_some_targets_have_none(tmp_path):
    # ~19% genuinely have no DV products; that is the archive being honest.
    tics = list(range(1, 11))
    cache = make_dv_archive(tmp_path, tics, without_dv=(9, 10))
    assert check_dv_archive(cache, tics) == []


def test_dv_gate_catches_an_interrupted_fetch(tmp_path):
    # The headline check: targets never queried are indistinguishable from
    # targets with no DV products unless the manifest is complete, so an
    # interrupted fetch would silently mask out real data for everything after
    # the interruption.
    tics = list(range(1, 11))
    cache = make_dv_archive(tmp_path, tics[:6])
    problems = check_dv_archive(cache, tics)
    assert any("never queried" in p for p in problems)


def test_dv_gate_catches_a_broken_query_via_the_coverage_floor(tmp_path):
    tics = list(range(1, 11))
    cache = make_dv_archive(tmp_path, tics, without_dv=tuple(tics[2:]))
    assert any("suspect the query" in p for p in check_dv_archive(cache, tics))


def test_dv_gate_catches_truncated_and_vanished_files(tmp_path):
    tics = [1, 2]
    cache = make_dv_archive(tmp_path, tics)
    next(cache.glob("tic_1/*.xml")).write_text("")
    next(cache.glob("tic_2/*.xml")).unlink()
    problems = check_dv_archive(cache, tics)
    assert any("is empty" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_dv_gate_reports_an_absent_or_unreadable_manifest(tmp_path):
    absent = tmp_path / "nothing"
    assert check_dv_archive(absent) == [f"no manifest at {absent / 'manifest.json'}"]
    cache = tmp_path / "raw_dv"
    cache.mkdir()
    (cache / "manifest.json").write_text("{oops")
    assert any("not readable JSON" in p for p in check_dv_archive(cache))


# ------------------------------------------------------------------ leakage --


def test_diff_and_quarantine_catch_since_confirmed_flip():
    old = good_labels()
    new = good_labels()
    new.loc[4, ["label", "disposition"]] = [1, "CP"]  # PC -> confirmed after close

    flips = diff_label_catalogues(old, new)
    assert len(flips) == 1
    assert flips.iloc[0]["label_old"] == -1 and flips.iloc[0]["label_new"] == 1
    assert quarantine_tics(flips) == {("TESS", 5)}


def test_a_quarantined_target_is_removed_from_the_training_index(tmp_path):
    """The guard's own docstring said `quarantine_tics` "is what the training
    path applies" — and until 2026-08-07 nothing outside its test called it, so
    under the 2% flip threshold a since-confirmed row trained happily."""
    old, new = good_labels(), good_labels()
    new.loc[4, ["label", "disposition"]] = [1, "CP"]
    record_quarantine(diff_label_catalogues(old, new), tmp_path)

    index = pd.DataFrame({"mission": ["TESS"] * 6, "tic_id": [1, 2, 3, 4, 5, 6]})
    kept = drop_quarantined(index, load_quarantine(tmp_path))

    assert 5 not in set(kept["tic_id"])
    assert len(kept) == 5


def test_the_quarantine_accumulates_across_refreshes(tmp_path):
    """Deriving it from the current pair of catalogues alone would readmit a row
    that flipped three refreshes ago."""
    first, second = good_labels(), good_labels()
    first.loc[4, ["label", "disposition"]] = [1, "CP"]
    second.loc[3, ["label", "disposition"]] = [1, "CP"]

    record_quarantine(diff_label_catalogues(good_labels(), first), tmp_path)
    record_quarantine(diff_label_catalogues(good_labels(), second), tmp_path)

    assert load_quarantine(tmp_path) == {("TESS", 5), ("TESS", 4)}


def test_assert_refresh_safe_rejects_mass_flip():
    old = good_labels()
    new = good_labels()
    new["label"] = 1 - new["label"].clip(0, 1)  # nearly everything flips
    with pytest.raises(ValueError, match="refusing the refresh"):
        assert_refresh_safe(old, new)


def test_assert_refresh_safe_rejects_disjoint_catalogues():
    old = good_labels()
    new = good_labels()
    new["tic_id"] += 1000
    with pytest.raises(ValueError, match="no targets"):
        assert_refresh_safe(old, new)


# ------------------------------------------------------------------- shrink --


def labels_by_mission(**counts: int) -> pd.DataFrame:
    missions = [m for mission, k in counts.items() for m in [mission] * k]
    n = len(missions)
    return pd.DataFrame(
        {
            "tic_id": np.arange(1, n + 1),
            "period": np.full(n, 3.0),
            "t0": np.full(n, 2458326.0),
            "duration": np.full(n, 0.1),
            "depth": np.full(n, 500.0),
            "disposition": ["CP", "FP"] * (n // 2) + ["CP"] * (n % 2),
            "label": [1, 0] * (n // 2) + [1] * (n % 2),
            "mission": missions,
        }
    )


def test_shrink_guard_passes_on_growth():
    old = labels_by_mission(TESS=100, Kepler=100)
    new = labels_by_mission(TESS=120, Kepler=100, K2=30)
    assert check_catalogue_shrink(old, new) == []


def test_shrink_guard_tolerates_reduction_inside_threshold():
    old = labels_by_mission(TESS=100, Kepler=100)
    new = labels_by_mission(TESS=95, Kepler=95)
    assert check_catalogue_shrink(old, new) == []


def test_shrink_guard_catches_the_weekly_refresh_regression():
    """Regression (2026-07-25): the refresh rewrote the data-of-record from
    5,686 rows across three missions down to 1,000 TESS-only rows and all four
    existing gates reported PASS."""
    old = labels_by_mission(TESS=2656, Kepler=2500, K2=530)
    new = labels_by_mission(TESS=1000)

    problems = check_catalogue_shrink(old, new)
    assert any("5686 -> 1000" in p for p in problems)
    assert any("Kepler" in p and "K2" in p for p in problems)


def test_shrink_guard_flags_lost_mission_at_stable_row_count():
    old = labels_by_mission(TESS=100, K2=50)
    new = labels_by_mission(TESS=150)

    problems = check_catalogue_shrink(old, new)
    assert len(problems) == 1
    assert "K2 (50 rows)" in problems[0]


def test_shrink_guard_threshold_is_configurable():
    """The Step 2b DR25 certification retired ~21% of bare Kepler FPs."""
    old = labels_by_mission(TESS=100, Kepler=100)
    new = labels_by_mission(TESS=100, Kepler=58)

    assert check_catalogue_shrink(old, new) != []
    assert check_catalogue_shrink(old, new, max_shrink_frac=0.30) == []


def test_shrink_guard_skips_empty_previous_catalogue():
    assert check_catalogue_shrink(labels_by_mission(), labels_by_mission(TESS=10)) == []


# ------------------------------------------------------------- shrink gate --

VALIDATE_DATA = Path(__file__).resolve().parents[1] / "scripts" / "validate_data.py"


def run_validate_data(tmp_path, *extra: str) -> tuple[int, str]:
    """Run validate_data.py with the other artefacts pointed at absent paths.

    Every non-label path must be named explicitly. A gate left on its default
    resolves against the subprocess's cwd — the repo root under pytest — so it
    would silently start validating the real artefact the moment one exists,
    which is how the DV gate broke four shrink tests the day `data/raw_dv` was
    first fetched.

    Returns the exit code and the whitespace-normalised gate report (rich wraps
    the handler output at the console width).
    """
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATE_DATA),
            "--labels",
            str(tmp_path / "labels.parquet"),
            "--candidates",
            str(tmp_path / "absent.parquet"),
            "--views",
            str(tmp_path / "absent.npz"),
            "--dv",
            str(tmp_path / "absent_dv"),
            "--viewset",
            str(tmp_path / "absent_viewset"),
            *extra,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )
    return result.returncode, " ".join(result.stdout.split())


def shrink_gate(tmp_path, old: pd.DataFrame, new: pd.DataFrame, *extra: str) -> tuple[int, str]:
    previous = tmp_path / "labels.previous.parquet"
    old.to_parquet(previous)
    new.to_parquet(tmp_path / "labels.parquet")
    return run_validate_data(tmp_path, "--previous-labels", str(previous), *extra)


def test_shrink_gate_fails_the_run(tmp_path):
    code, report = shrink_gate(
        tmp_path,
        labels_by_mission(TESS=2656, Kepler=2500, K2=530),
        labels_by_mission(TESS=1000),
    )
    assert code == 1
    assert "label-shrink FAIL" in report
    assert "--allow-shrink" in report


def test_shrink_gate_passes_a_healthy_refresh(tmp_path):
    code, report = shrink_gate(
        tmp_path, labels_by_mission(TESS=100, Kepler=100), labels_by_mission(TESS=110, Kepler=100)
    )
    assert code == 0
    assert "label-shrink PASS" in report


def test_shrink_gate_override_allows_intentional_reduction(tmp_path):
    code, report = shrink_gate(
        tmp_path,
        labels_by_mission(TESS=100, Kepler=100),
        labels_by_mission(TESS=100, Kepler=58),
        "--allow-shrink",
    )
    assert code == 0
    assert "shrink allowed" in report
    assert "label-shrink PASS" in report


def test_shrink_gate_honours_the_threshold_flag(tmp_path):
    args = (
        tmp_path,
        labels_by_mission(TESS=100, Kepler=100),
        labels_by_mission(TESS=100, Kepler=58),
    )
    assert shrink_gate(*args)[0] == 1
    assert shrink_gate(*args, "--max-shrink-frac", "0.30")[0] == 0


def test_shrink_gate_absent_without_previous_labels(tmp_path):
    labels_by_mission(TESS=10, Kepler=10).to_parquet(tmp_path / "labels.parquet")
    code, report = run_validate_data(tmp_path)
    assert code == 0
    assert "label-shrink" not in report


# ---------------------------------------------------------------- promotion --


def summary(auc: float, brier: float, ece: float | None = None) -> dict:
    result = {
        "folds": [],
        "summary": {
            "test_roc_auc": {"mean": auc, "std": 0.01},
            "test_brier": {"mean": brier, "std": 0.005},
        },
    }
    if ece is not None:
        result["summary"]["test_ece"] = {"mean": ece, "std": 0.005}
    return result


def pooled(candidate: dict, incumbent: dict):
    """A deliberately pooled comparison.

    Summaries built by `summary()` carry no per_mission block, so the gate now
    refuses them by default — the rows behind each mean are unknowable. These
    tests exercise the calibration, reliability and shortlist guards that sit
    behind that check, so they opt in explicitly.
    """
    return evaluate_promotion(candidate, incumbent, allow_unmatched_populations=True)


def test_first_model_promotes():
    decision = evaluate_promotion(summary(0.90, 0.10), None)
    assert decision.promoted


def test_better_auc_with_stable_calibration_promotes():
    decision = pooled(summary(0.93, 0.101), summary(0.92, 0.100))
    assert decision.promoted


def test_worse_auc_rejected():
    decision = pooled(summary(0.91, 0.05), summary(0.92, 0.10))
    assert not decision.promoted


def test_better_auc_but_degraded_calibration_rejected():
    decision = pooled(summary(0.93, 0.12), summary(0.92, 0.10))
    assert not decision.promoted
    assert any("calibration" in r for r in decision.reasons)


def test_better_brier_but_degraded_ece_rejected():
    decision = pooled(summary(0.95, 0.09, ece=0.13), summary(0.87, 0.10, ece=0.03))
    assert not decision.promoted
    assert any("reliability" in r for r in decision.reasons)


def test_ece_within_tolerance_promotes():
    decision = pooled(summary(0.93, 0.10, ece=0.035), summary(0.92, 0.10, ece=0.030))
    assert decision.promoted


def test_missing_ece_skips_the_guard():
    # Summaries written before the test_ece field must still be comparable.
    decision = pooled(summary(0.93, 0.10, ece=0.13), summary(0.92, 0.10))
    assert decision.promoted
    assert any("skipped" in r for r in decision.reasons)


# ------------------------------------------- promotion: the gate population --


def slices(**missions: dict) -> dict:
    """A per_mission block; each mission takes auc/brier/ece/recall overrides."""
    defaults = {"n": 2000, "roc_auc": 0.90, "brier": 0.10, "ece": 0.03, "recall_at_1pct_fpr": 0.30}
    return {"per_mission": {m: {**defaults, **v} for m, v in missions.items()}}


def gated(auc: float, brier: float, **rest) -> dict:
    """A summary whose pooled means are deliberately hostile to the verdict, so
    a test that passes can only have read the TESS slice."""
    return summary(0.10, 0.90, ece=0.90) | slices(TESS={"roc_auc": auc, "brier": brier, **rest})


def test_the_gate_reads_tess_not_the_aggregate():
    decision = evaluate_promotion(gated(0.92, 0.10), gated(0.91, 0.10))
    assert decision.promoted
    assert any("gated on TESS" in r for r in decision.reasons)


def test_a_tess_regression_rejects_however_good_the_aggregate_is():
    candidate = summary(0.99, 0.01, ece=0.01) | slices(
        TESS={"roc_auc": 0.90}, Kepler={"roc_auc": 0.999}
    )
    incumbent = summary(0.80, 0.20, ece=0.20) | slices(TESS={"roc_auc": 0.91})
    decision = evaluate_promotion(candidate, incumbent)
    assert not decision.promoted
    assert any("does not beat" in r for r in decision.reasons)


def test_the_aggregate_is_reported_and_never_gates():
    """Kepler is drawn at exactly 1,250/1,250, so the pooled slice is weighted
    by a sampling decision in a comparison whose consequences are all TESS."""
    candidate = gated(0.92, 0.10) | slices(
        TESS={"roc_auc": 0.92, "brier": 0.10}, all={"roc_auc": 0.80}
    )
    incumbent = gated(0.91, 0.10) | slices(TESS={"roc_auc": 0.91}, all={"roc_auc": 0.95})
    decision = evaluate_promotion(candidate, incumbent)
    assert decision.promoted
    assert any("never gates" in r for r in decision.reasons)


def test_a_kepler_collapse_alarms_without_blocking():
    candidate = gated(0.92, 0.10) | slices(
        TESS={"roc_auc": 0.92, "brier": 0.10}, Kepler={"roc_auc": 0.90}
    )
    incumbent = gated(0.91, 0.10) | slices(TESS={"roc_auc": 0.91}, Kepler={"roc_auc": 0.99})
    decision = evaluate_promotion(candidate, incumbent)
    assert decision.promoted
    assert decision.alarms and "Kepler" in decision.alarms[0]
    assert "written explanation" in decision.alarms[0]


def test_a_k2_drop_inside_the_alarm_threshold_is_quiet():
    candidate = gated(0.92, 0.10) | slices(
        TESS={"roc_auc": 0.92, "brier": 0.10}, K2={"roc_auc": 0.90}
    )
    incumbent = gated(0.91, 0.10) | slices(TESS={"roc_auc": 0.91}, K2={"roc_auc": 0.915})
    decision = evaluate_promotion(candidate, incumbent)
    assert decision.promoted and not decision.alarms


# ---------------------------------------------- promotion: shortlist recall --


def test_recall_at_1pct_fpr_can_reject_a_model_that_wins_on_auc():
    """The stage 2(a) case: TESS AUC within noise, recall @1% FPR 0.307 -> 0.238.
    AUC scores ranking everywhere; the shortlist lives at one threshold."""
    candidate = gated(0.9100, 0.1194, recall_at_1pct_fpr=0.238)
    incumbent = gated(0.9079, 0.1211, recall_at_1pct_fpr=0.307)
    decision = evaluate_promotion(candidate, incumbent)
    assert not decision.promoted
    assert any("shortlist recall" in r for r in decision.reasons)


def test_recall_within_tolerance_still_promotes():
    decision = evaluate_promotion(
        gated(0.92, 0.10, recall_at_1pct_fpr=0.295), gated(0.91, 0.10, recall_at_1pct_fpr=0.300)
    )
    assert decision.promoted


def test_a_nan_metric_rejects_instead_of_sailing_through_every_guard():
    """Every guard is an inequality and NaN loses all of them, so a degenerate
    run — a single-class fold, an empty slice, a blown-up loss — would promote
    itself with `ROC-AUC nan vs incumbent 0.9581`."""
    decision = evaluate_promotion(summary(float("nan"), 0.0, ece=0.0), summary(0.9581, 0.079))
    assert not decision.promoted
    assert any("not measurable" in r for r in decision.reasons)


def test_a_nan_on_the_incumbent_side_also_rejects():
    """A corrupt registry entry must not be something a candidate slips past."""
    decision = evaluate_promotion(summary(0.95, 0.08), summary(float("nan"), 0.079))
    assert not decision.promoted
    assert any("incumbent" in r for r in decision.reasons)


@pytest.mark.parametrize("metric", ["roc_auc", "brier", "ece", "recall_at_1pct_fpr"])
def test_a_nan_in_any_gating_metric_rejects(metric):
    decision = evaluate_promotion(
        gated(0.92, 0.10) | slices(TESS={"roc_auc": 0.92, "brier": 0.10, metric: float("nan")}),
        gated(0.91, 0.10),
    )
    assert not decision.promoted
    assert any("not measurable" in r for r in decision.reasons)


def test_a_pooled_comparison_over_unmatched_rows_refuses_rather_than_guessing():
    """The live incumbent `ca906040` carries no per_mission block, so the TESS
    gate silently degraded to a pooled comparison — its 4,818 rows with zero K2
    against a current run's 5,426 including 527. That reads as a model
    difference and is partly a population difference."""
    decision = evaluate_promotion(summary(0.93, 0.10, ece=0.03), summary(0.92, 0.10, ece=0.03))
    assert not decision.promoted
    assert any("predates the per_mission block" in r for r in decision.reasons)
    assert any("rows behind each mean are unknown" in r for r in decision.reasons)
    assert any("re-baseline the incumbent" in r for r in decision.reasons)


def test_the_unmatched_population_refusal_can_be_overridden_deliberately():
    decision = evaluate_promotion(
        summary(0.93, 0.10, ece=0.03),
        summary(0.92, 0.10, ece=0.03),
        allow_unmatched_populations=True,
    )
    assert decision.promoted
    assert any("recall guard skipped" in r for r in decision.reasons)


def folds(*values: float, seed: int = 42, n_test: int = 100) -> dict:
    """A summary carrying per-fold scores, so folds can be paired."""
    return {
        "folds": [{"test_roc_auc": v, "n_test": n_test} for v in values],
        "run_config": {"seed": seed, "n_splits": len(values)},
    }


def test_folds_of_different_sizes_pair_but_are_flagged_inexact():
    """Run 1 and run 2 differ here — three FFI rows arrived between the builds,
    so fold k is not quite the same row set. Equal sizes cannot prove identical
    membership, but unequal sizes disprove it."""
    inexact = paired_folds(folds(0.94, 0.95), folds(0.90, 0.91, n_test=97))
    assert inexact is not None and not inexact.exact
    assert "approximately matched" in str(inexact)
    assert paired_folds(folds(0.94, 0.95), folds(0.90, 0.91)).exact


def test_a_candidate_that_wins_on_average_but_loses_most_folds_is_alarmed():
    """Winning the mean while losing fold by fold is what winning on training
    noise looks like — one lucky fold carries it."""
    candidate = gated(0.92, 0.10) | folds(0.99, 0.88, 0.88, 0.88, 0.88)
    incumbent = gated(0.91, 0.10) | folds(0.90, 0.90, 0.90, 0.90, 0.90)
    decision = evaluate_promotion(candidate, incumbent)
    assert decision.promoted
    assert any("won only 1/5 folds" in a for a in decision.alarms)


def test_a_consistent_fold_by_fold_win_is_reported_without_alarm():
    candidate = gated(0.92, 0.10) | folds(0.94, 0.96, 0.93, 0.97, 0.95)
    incumbent = gated(0.91, 0.10) | folds(0.90, 0.91, 0.89, 0.92, 0.90)
    decision = evaluate_promotion(candidate, incumbent)
    assert decision.promoted
    assert any("won 5/5" in r for r in decision.reasons)
    assert not decision.alarms


def test_folds_from_a_different_split_are_not_paired():
    """Pairing assumes fold k held out the same rows in both runs."""
    assert paired_folds(folds(0.9, 0.9, 0.9), folds(0.8, 0.8, 0.8, 0.8)) is None
    assert paired_folds(folds(0.9, 0.9, seed=1), folds(0.8, 0.8, seed=2)) is None


def test_no_p_value_is_reported_where_it_could_never_reach_significance():
    """Five pairs floor the two-sided Wilcoxon at p=0.0625; printing it invites
    reading "not significant" as evidence of no effect."""
    assert paired_folds(folds(*[0.99] * 5), folds(*[0.80] * 5)).p_value is None
    assert paired_folds(folds(*[0.99] * 6), folds(*[0.80] * 6)).p_value is not None


def test_a_mission_only_one_run_scored_alarms_but_does_not_block_the_tess_gate():
    """Gating on TESS compares a mission both runs scored, so K2 appearing on
    one side only is worth saying and not worth blocking on."""
    candidate = gated(0.92, 0.10) | slices(TESS={"roc_auc": 0.92, "brier": 0.10}, K2={})
    decision = evaluate_promotion(candidate, gated(0.91, 0.10))
    assert decision.promoted
    assert any("only the candidate scored K2" in a for a in decision.alarms)


def write_folds(run_dir, n: int = 2, *, weights: bool = True, calibrator: bool = True):
    """The artefacts a run needs to be servable, which is what promotion means."""
    for i in range(n):
        fold = run_dir / f"fold_{i}"
        fold.mkdir(parents=True, exist_ok=True)
        if weights:
            (fold / "cnn_branches.keras").write_bytes(b"")
        if calibrator:
            (fold / "cnn_calibrator.joblib").write_bytes(b"")
    return run_dir


def test_registry_roundtrip(tmp_path):
    cv_dir = tmp_path / "cv" / "run123"
    cv_dir.mkdir(parents=True)
    write_folds(cv_dir)
    summary_path = cv_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary(0.92, 0.10)))

    assert load_incumbent_summary(tmp_path) is None
    promote(tmp_path, "run123", summary_path)
    incumbent = load_incumbent_summary(tmp_path)
    assert incumbent is not None
    assert incumbent["summary"]["test_roc_auc"]["mean"] == 0.92


@pytest.mark.parametrize("missing", ["weights", "calibrator", "folds"])
def test_a_run_with_nothing_to_serve_cannot_be_promoted(tmp_path, missing):
    """Stage 2(a) run 1 wrote no checkpoints at all — a metrics-only run that
    would promote cleanly and fail at serve time, long after the decision."""
    cv_dir = tmp_path / "cv" / "run123"
    cv_dir.mkdir(parents=True)
    if missing != "folds":
        write_folds(cv_dir, weights=missing != "weights", calibrator=missing != "calibrator")
    summary_path = cv_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary(0.92, 0.10)))

    with pytest.raises(ValueError, match="serve"):
        promote(tmp_path, "run123", summary_path)
    assert load_incumbent_summary(tmp_path) is None


# ------------------------------------------------------------------ publish --


def cv_run(models_dir, run_id: str, tracked: bool = False, with_summary: bool = True):
    run_dir = models_dir / "cv" / run_id
    run_dir.mkdir(parents=True)
    write_folds(run_dir)
    if with_summary:
        (run_dir / "cv_summary.json").write_text(json.dumps(summary(0.90, 0.10)))
    if tracked:
        (models_dir / "cv" / f"{run_id}.dvc").write_text("outs: []\n")
    return run_dir


def test_publishable_cv_dirs_allowlists_promoted_and_tracked(tmp_path):
    promoted = cv_run(tmp_path, "aaa_promoted")
    tracked = cv_run(tmp_path, "bbb_tracked", tracked=True)
    cv_run(tmp_path, "ccc_trial")  # tuning-trial checkpoint: has a summary, no pointer
    promote(tmp_path, "aaa_promoted", promoted / "cv_summary.json")

    assert publishable_cv_dirs(tmp_path) == [promoted, tracked]


def test_publishable_cv_dirs_without_registry_keeps_tracked_only(tmp_path):
    tracked = cv_run(tmp_path, "aaa_tracked", tracked=True)
    cv_run(tmp_path, "bbb_trial")

    assert publishable_cv_dirs(tmp_path) == [tracked]


def test_publishable_cv_dirs_skips_partial_dirs(tmp_path):
    # pointer present but contents half-restored: re-adding would clobber the pointer
    cv_run(tmp_path, "aaa_partial", tracked=True, with_summary=False)

    assert publishable_cv_dirs(tmp_path) == []


def test_publishable_cv_dirs_empty_without_cv_root(tmp_path):
    assert publishable_cv_dirs(tmp_path) == []


def test_incumbent_summary_resolves_registry_path_from_any_cwd(tmp_path, monkeypatch):
    """Registry paths are repo-root-relative; the gate must not resolve them
    against the caller's cwd (`promotion_gate.py --models-dir` from elsewhere
    read the wrong file or crashed)."""
    root = tmp_path / "repo"
    models_dir = root / "models"
    cv_dir = models_dir / "cv" / "run123"
    cv_dir.mkdir(parents=True)
    (cv_dir / "cv_summary.json").write_text(json.dumps(summary(0.93, 0.09)))
    (models_dir / "registry.json").write_text(
        json.dumps({"run_id": "run123", "cv_summary": "models/cv/run123/cv_summary.json"})
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    incumbent = load_incumbent_summary(models_dir)
    assert incumbent is not None
    assert incumbent["summary"]["test_roc_auc"]["mean"] == 0.93
