"""Tests for ExoFOP-sourced catalogue enrichment and format dialects."""

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.data.exofop import (
    enrich_catalog_snr,
    load_ctoi_table,
    load_toi_table,
    toi_snr_by_tic,
)


def test_endpoint_dialect_toi_sexagesimal_coords(tmp_path):
    """download_toi.php serves sexagesimal RA/Dec and abbreviated columns
    (2026-07-10 orchestrator regression)."""
    path = tmp_path / "tois.csv"
    path.write_text(
        "TIC ID,TOI,TFOPWG Disposition,TESS Mag,RA,Dec,Epoch (BJD),Period (days),"
        "Duration (hours),Depth (ppm),Planet Radius (R_Earth),Planet SNR,"
        "Predicted RV Semi-amplitude (m/s),Stellar Eff Temp (K)\n"
        "231663901,101.01,KP,12.4,21:14:56.88,-55:52:18.71,2458326.0,1.43,"
        "1.6,18960.7,13.19,151.7,63.3,5600\n"
    )
    out = load_toi_table(path)
    assert len(out) == 1
    assert out.iloc[0]["name"] == "TOI-101.01"
    assert out.iloc[0]["ra_deg"] == pytest.approx(318.737, abs=0.01)
    assert out.iloc[0]["dec_deg"] == pytest.approx(-55.872, abs=0.01)
    assert out.iloc[0]["predicted_k_ms"] == 63.3


def test_endpoint_dialect_ctoi_columns(tmp_path):
    """download_ctoi.php uses 'CTOI' for the candidate id, 'Depth ppm',
    'Duration (hrs)', and publishes a fitted Teq."""
    path = tmp_path / "ctois.csv"
    path.write_text(
        "TIC ID,CTOI,Promoted to TOI,Candidate Name,TFOPWG Disposition,TESS Mag,"
        "RA,Dec,Transit Epoch (BJD),Period (days),Duration (hrs),Depth ppm,"
        "Planet Radius (R_Earth),Equilibrium Temp (K),Stellar Eff Temp (K),"
        "Stellar log(g) (cm/s^2),Stellar Radius (R_Sun),Notes,CTOI lastmod\n"
        "17361,17361.01,,,,11.34,219.336,-24.959,2458600.0,10.5,3.2,500.0,"
        "2.1,890.0,5775.3,4.25,1.26,note,2026-07-01\n"
    )
    out = load_ctoi_table(path)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["name"] == "TIC 17361.01"
    assert row["duration_hours"] == 3.2
    assert row["depth_ppm"] == 500.0
    assert row["teq_k"] == 890.0  # fitted value kept, not overwritten
    assert row["date_modified"] == "2026-07-01"


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
