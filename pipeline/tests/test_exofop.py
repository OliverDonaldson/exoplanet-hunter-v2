"""Tests for ExoFOP-sourced catalogue enrichment."""

import numpy as np
import pandas as pd

from exoplanet_hunter.data.exofop import enrich_catalog_snr, toi_snr_by_tic


def candidates_fixture(tmp_path):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "source": ["TOI", "TOI", "TOI", "CTOI"],
            "name": ["TOI-1.01", "TOI-1.02", "TOI-2.01", "TIC 9.01"],
            "tic_id": [100, 100, 200, 900],
            "planet_snr": [50.0, 12.0, np.nan, 99.0],  # CTOI snr must be ignored
        }
    ).to_parquet(path, index=False)
    return path


def test_toi_snr_takes_strongest_per_tic(tmp_path):
    snr = toi_snr_by_tic(candidates_fixture(tmp_path))
    assert snr[100] == 50.0  # max over the system's TOIs
    assert 200 not in snr  # NaN-only TICs excluded
    assert 900 not in snr  # CTOI source excluded


def test_enrich_fills_only_missing_tess_rows(tmp_path):
    catalog = pd.DataFrame(
        {
            "tic_id": [100, 200, 300, 400],
            "mission": ["TESS", "TESS", "Kepler", "TESS"],
            "snr": [np.nan, np.nan, 77.0, 5.0],
        }
    )
    out = enrich_catalog_snr(catalog, candidates_fixture(tmp_path))
    assert out.loc[0, "snr"] == 50.0  # filled from TOI export
    assert np.isnan(out.loc[1, "snr"])  # no ExoFOP SNR for this TIC
    assert out.loc[2, "snr"] == 77.0  # Kepler koi_model_snr untouched
    assert out.loc[3, "snr"] == 5.0  # existing values never overwritten


def test_enrich_without_snr_column_and_missing_file(tmp_path):
    catalog = pd.DataFrame({"tic_id": [100], "mission": ["TESS"]})
    out = enrich_catalog_snr(catalog, candidates_fixture(tmp_path))
    assert out.loc[0, "snr"] == 50.0

    untouched = enrich_catalog_snr(catalog, tmp_path / "absent.parquet")
    assert "snr" not in untouched.columns  # best-effort no-op
