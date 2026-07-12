"""Tests for the validation gates: schemas, leakage guard, promotion."""

import json

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from exoplanet_hunter.datasets import ViewArrays
from exoplanet_hunter.validation import (
    assert_refresh_safe,
    candidate_catalogue_schema,
    check_views,
    diff_label_catalogues,
    evaluate_promotion,
    label_catalogue_schema,
    load_incumbent_summary,
    promote,
    quarantine_tics,
)

# ------------------------------------------------------------------ schemas --


def good_labels(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tic_id": np.arange(1, n + 1),
            "period": np.linspace(1.0, 10.0, n),
            "t0": np.full(n, 2458326.0),
            "duration": np.full(n, 0.1),
            "depth": np.full(n, 500.0),
            "disposition": ["CP", "KP", "FP", "FA", "PC", "CP"][:n],
            "label": [1, 1, 0, 0, -1, 1][:n],
            "mission": ["TESS"] * (n - 1) + ["Kepler"],
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


# ------------------------------------------------------------------ leakage --


def test_diff_and_quarantine_catch_since_confirmed_flip():
    old = good_labels()
    new = good_labels()
    new.loc[4, ["label", "disposition"]] = [1, "CP"]  # PC -> confirmed after close

    flips = diff_label_catalogues(old, new)
    assert len(flips) == 1
    assert flips.iloc[0]["label_old"] == -1 and flips.iloc[0]["label_new"] == 1
    assert quarantine_tics(flips) == {("TESS", 5)}


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


def test_first_model_promotes():
    decision = evaluate_promotion(summary(0.90, 0.10), None)
    assert decision.promoted


def test_better_auc_with_stable_calibration_promotes():
    decision = evaluate_promotion(summary(0.93, 0.101), summary(0.92, 0.100))
    assert decision.promoted


def test_worse_auc_rejected():
    decision = evaluate_promotion(summary(0.91, 0.05), summary(0.92, 0.10))
    assert not decision.promoted


def test_better_auc_but_degraded_calibration_rejected():
    decision = evaluate_promotion(summary(0.93, 0.12), summary(0.92, 0.10))
    assert not decision.promoted
    assert any("calibration" in r for r in decision.reasons)


def test_better_brier_but_degraded_ece_rejected():
    # Brier alone is blind to this: a discrimination gain can pay for
    # arbitrary miscalibration — exactly how the full-scale run promoted
    # with ECE 0.136 vs the incumbent's 0.031.
    decision = evaluate_promotion(summary(0.95, 0.09, ece=0.13), summary(0.87, 0.10, ece=0.03))
    assert not decision.promoted
    assert any("reliability" in r for r in decision.reasons)


def test_ece_within_tolerance_promotes():
    decision = evaluate_promotion(summary(0.93, 0.10, ece=0.035), summary(0.92, 0.10, ece=0.030))
    assert decision.promoted


def test_missing_ece_skips_the_guard():
    # Summaries written before the test_ece field must still be comparable.
    decision = evaluate_promotion(summary(0.93, 0.10, ece=0.13), summary(0.92, 0.10))
    assert decision.promoted
    assert any("skipped" in r for r in decision.reasons)


def test_registry_roundtrip(tmp_path):
    cv_dir = tmp_path / "cv" / "run123"
    cv_dir.mkdir(parents=True)
    summary_path = cv_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary(0.92, 0.10)))

    assert load_incumbent_summary(tmp_path) is None
    promote(tmp_path, "run123", summary_path)
    incumbent = load_incumbent_summary(tmp_path)
    assert incumbent is not None
    assert incumbent["summary"]["test_roc_auc"]["mean"] == 0.92
