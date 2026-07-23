"""Catalogue subsampling must be stable across refreshes (no positional churn)."""

from __future__ import annotations

import pandas as pd

from exoplanet_hunter.data import catalog as catalog_mod
from exoplanet_hunter.data.catalog import (
    CatalogRequest,
    _query_certified_fp,
    _query_k2,
    _stable_sample,
    build_label_catalog,
)


def pool(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"tic_id": ids, "label": 1})


def test_selection_survives_reordering():
    ids = list(range(1000, 1100))
    a = _stable_sample(pool(ids), 30, seed=42)
    b = _stable_sample(pool(list(reversed(ids))), 30, seed=42)
    assert set(a.tic_id) == set(b.tic_id)


def test_selection_stable_under_realistic_pool_growth():
    # Churn is proportional to the pool delta — never a reshuffle.
    ids = list(range(1000, 1100))
    before = set(_stable_sample(pool(ids), 30, seed=42).tic_id)
    after = set(_stable_sample(pool([*ids, 5000, 5001]), 30, seed=42).tic_id)
    assert len(before & after) >= 28


def test_seed_changes_selection():
    ids = list(range(1000, 1100))
    a = set(_stable_sample(pool(ids), 30, seed=42).tic_id)
    b = set(_stable_sample(pool(ids), 30, seed=7).tic_id)
    assert a != b


def test_request_larger_than_pool_returns_everything():
    df = pool([1, 2, 3])
    assert _stable_sample(df, 10, seed=42) is df


# --- Kepler certified-false-positive negatives (Step 2b) --------------------


def test_query_certified_fp_targets_dr25_and_parses_names(monkeypatch):
    seen = {}

    def fake_tap(adql, *a, **k):
        seen["adql"] = adql
        return pd.DataFrame({"kepoi_name": ["K00001.01", "K00002.01", None]})

    monkeypatch.setattr(catalog_mod, "_tap_query", fake_tap)
    names = _query_certified_fp()
    assert names == {"K00001.01", "K00002.01"}  # NaN dropped
    assert "q1_q17_dr25_koi" in seen["adql"]
    assert "koi_disposition = 'FALSE POSITIVE'" in seen["adql"]
    assert "koi_score < 0.5" in seen["adql"]


def _wire_sources(monkeypatch, certified: set[str]) -> None:
    """Stub every TAP-backed source so build_label_catalog runs offline. One
    Kepler FP is DR25-certified (K2.01), one is not (K3.01)."""
    monkeypatch.setattr(
        catalog_mod,
        "_query_confirmed_planets",
        lambda: pd.DataFrame({"tic_id": [1], "label": [1], "mission": ["TESS"]}),
    )
    monkeypatch.setattr(
        catalog_mod,
        "_query_toi",
        lambda: pd.DataFrame({"tic_id": [3, 4, 5], "label": [1, 0, -1], "mission": ["TESS"] * 3}),
    )
    monkeypatch.setattr(
        catalog_mod,
        "_query_koi",
        lambda: pd.DataFrame(
            {
                "tic_id": [10, 11, 12, 13],
                "name": ["K1.01", "K2.01", "K3.01", "K4.01"],
                "label": [1, 0, 0, -1],
                "mission": ["Kepler"] * 4,
            }
        ),
    )
    monkeypatch.setattr(catalog_mod, "_query_certified_fp", lambda: certified)


def test_build_restricts_kepler_negatives_to_certified(monkeypatch, tmp_path):
    _wire_sources(monkeypatch, certified={"K2.01"})
    req = CatalogRequest(
        n_confirmed=100, n_false_pos=100, n_confirmed_kepler=100, n_false_pos_kepler=100
    )
    cat = build_label_catalog(req, tmp_path)

    kep_neg = cat[(cat["mission"] == "Kepler") & (cat["label"] == 0)]
    assert set(kep_neg["tic_id"]) == {11}  # K2.01 certified; K3.01 (12) dropped
    assert 12 not in set(cat["tic_id"])
    assert 10 in set(cat["tic_id"])  # Kepler positive untouched
    assert 4 in set(cat[cat["label"] == 0]["tic_id"])  # TESS negative untouched


def test_build_fails_open_when_no_certified_fps(monkeypatch, tmp_path):
    """An empty certified set must not zero the negatives — keep the raw FPs."""
    _wire_sources(monkeypatch, certified=set())
    req = CatalogRequest(
        n_confirmed=100, n_false_pos=100, n_confirmed_kepler=100, n_false_pos_kepler=100
    )
    cat = build_label_catalog(req, tmp_path)
    kep_neg = cat[(cat["mission"] == "Kepler") & (cat["label"] == 0)]
    assert set(kep_neg["tic_id"]) == {11, 12}  # both retained


# --- K2 integration (Step 2c) -----------------------------------------------


def test_query_k2_parses_epic_prefers_default_and_maps_labels(monkeypatch):
    seen = {}

    def fake_tap(adql, *a, **k):
        seen["adql"] = adql
        return pd.DataFrame(
            {
                "epic_hostname": ["EPIC 100", "EPIC 100", "EPIC 200", "EPIC 300"],
                "name": ["EPIC 100.01", "EPIC 100.01", "EPIC 200.01", "EPIC 300.01"],
                "disposition": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE", "REFUTED"],
                "default_flag": [0, 1, 1, 1],  # EPIC 100 has a non-default + default row
                "period": [5.5, 5.0, 3.0, 2.0],  # default row's period is 5.0
                "t0": [10.0, 10.0, 11.0, 12.0],
                "depth": [0.01, 0.01, 0.02, 0.03],
                "duration": [0.1, 0.1, 0.1, 0.1],
                "teff": [5000, 5000, 4000, 6000],
                "radius": [1.0, 1.0, 0.8, 1.2],
                "logg": [4.5, 4.5, 4.6, 4.3],
                "tmag": [12, 12, 13, 11],
            }
        )

    monkeypatch.setattr(catalog_mod, "_tap_query", fake_tap)
    df = _query_k2()
    assert "k2pandc" in seen["adql"] and "pl_trandur is not null" in seen["adql"]
    assert set(df["tic_id"]) == {100, 200, 300}  # EPIC parsed to int + deduped per star
    assert set(df["mission"]) == {"K2"}
    labels_by_id = dict(zip(df["tic_id"], df["label"], strict=True))
    assert labels_by_id == {100: 1, 200: 0, 300: 0}  # REFUTED -> 0
    assert df.loc[df["tic_id"] == 100, "period"].iloc[0] == 5.0  # default row preferred
    assert "default_flag" not in df.columns


def test_build_includes_k2_when_requested(monkeypatch, tmp_path):
    _wire_sources(monkeypatch, certified={"K2.01"})
    monkeypatch.setattr(
        catalog_mod,
        "_query_k2",
        lambda: pd.DataFrame(
            {
                "tic_id": [500, 501, 502],
                "name": ["EPIC 500.01", "EPIC 501.01", "EPIC 502.01"],
                "label": [1, 0, -1],
                "mission": ["K2"] * 3,
            }
        ),
    )
    req = CatalogRequest(n_confirmed=100, n_false_pos=100, n_confirmed_k2=100, n_false_pos_k2=100)
    cat = build_label_catalog(req, tmp_path)
    k2 = cat[cat["mission"] == "K2"]
    assert set(k2["tic_id"]) == {500, 501}  # CP + FP trained; PC (502) held out
    assert 502 not in set(cat["tic_id"])
