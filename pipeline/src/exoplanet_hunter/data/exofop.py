"""Load and normalise ExoFOP candidate-table exports (TOI + CTOI).

The inputs are bulk CSV exports from the Exoplanet Follow-up Observing
Program (https://exofop.ipac.caltech.edu):

  * TOI table:  https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv
  * CTOI table: https://exofop.ipac.caltech.edu/tess/download_ctoi.php?output=csv

Both are normalised to one schema (`CATALOGUE_COLUMNS`) and concatenated
into the candidate catalogue that the API serves and the vetting console
browses. CTOIs already promoted to TOIs are dropped — they appear in the
TOI table with follow-up-vetted parameters, so keeping both would double-
count the candidate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.features import followup
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: Unified candidate-catalogue schema, in output column order.
CATALOGUE_COLUMNS: list[str] = [
    "source",
    "name",
    "tic_id",
    "disposition",
    "tess_mag",
    "ra_deg",
    "dec_deg",
    "epoch_bjd",
    "period_days",
    "duration_hours",
    "depth_ppm",
    "planet_radius_re",
    "planet_snr",
    "teq_k",
    "tsm",
    "esm",
    "insolation_earth",
    "hz_inner_au",
    "hz_outer_au",
    "predicted_mass_me",
    "predicted_k_ms",
    "stellar_teff_k",
    "stellar_logg",
    "stellar_radius_rsun",
    "stellar_distance_pc",
    "sectors",
    "promoted_to_toi",
    "comments",
    "date_modified",
]

_NUMERIC_COLUMNS = [
    "tess_mag",
    "ra_deg",
    "dec_deg",
    "epoch_bjd",
    "period_days",
    "duration_hours",
    "depth_ppm",
    "planet_radius_re",
    "planet_snr",
    "teq_k",
    "tsm",
    "esm",
    "insolation_earth",
    "hz_inner_au",
    "hz_outer_au",
    "predicted_mass_me",
    "predicted_k_ms",
    "stellar_teff_k",
    "stellar_logg",
    "stellar_radius_rsun",
    "stellar_distance_pc",
]

# Rename maps carry both ExoFOP dialects: the dashboard bulk-export names
# and the download_{toi,ctoi}.php endpoint names (which differ in casing,
# abbreviations, and units punctuation). Each file contains one dialect, so
# the unused keys are simply absent.
_TOI_RENAMES = {
    "TIC ID": "tic_id",
    "TFOPWG Disposition": "disposition",
    "TESS Mag": "tess_mag",
    "RA": "ra_deg",
    "Dec": "dec_deg",
    "Epoch (BJD)": "epoch_bjd",
    "Period (days)": "period_days",
    "Duration (hours)": "duration_hours",
    "Depth (ppm)": "depth_ppm",
    "Planet Radius (R_Earth)": "planet_radius_re",
    "Planet SNR": "planet_snr",
    "Planet Equil Temp (K)": "teq_k",
    "TSM": "tsm",
    "ESM": "esm",
    "Predicted Mass (M_Earth)": "predicted_mass_me",
    "Predicted Radial Velocity Semi-amplitude (m/s)": "predicted_k_ms",
    "Predicted RV Semi-amplitude (m/s)": "predicted_k_ms",  # endpoint dialect
    "Stellar Eff Temp (K)": "stellar_teff_k",
    "Stellar log(g) (cm/s^2)": "stellar_logg",
    "Stellar Radius (R_Sun)": "stellar_radius_rsun",
    "Stellar Distance (pc)": "stellar_distance_pc",
    "Sectors": "sectors",
    "Comments": "comments",
    "Date Modified": "date_modified",
}

_CTOI_RENAMES = {
    "TIC ID": "tic_id",
    "Promoted to TOI": "promoted_to_toi",
    "TFOPWG Disposition": "disposition",
    "TESS mag": "tess_mag",
    "TESS Mag": "tess_mag",  # endpoint dialect
    "RA (deg)": "ra_deg",
    "RA": "ra_deg",  # endpoint dialect
    "Dec (deg)": "dec_deg",
    "Dec": "dec_deg",  # endpoint dialect
    "Transit Epoch (BJD)": "epoch_bjd",
    "Period (days)": "period_days",
    "Duration (hours)": "duration_hours",
    "Duration (hrs)": "duration_hours",  # endpoint dialect
    "Depth (ppm)": "depth_ppm",
    "Depth ppm": "depth_ppm",  # endpoint dialect
    "Planet Radius (R_Earth)": "planet_radius_re",
    "Equilibrium Temp (K)": "teq_k",  # endpoint dialect (dashboard export lacks it)
    "Stellar Teff (K)": "stellar_teff_k",
    "Stellar Eff Temp (K)": "stellar_teff_k",  # endpoint dialect
    "Stellar log(g) (cm/s2)": "stellar_logg",
    "Stellar log(g) (cm/s^2)": "stellar_logg",  # endpoint dialect
    "Stellar Radius (R_Sun)": "stellar_radius_rsun",
    "Stellar Distance (pc)": "stellar_distance_pc",
    "Notes": "comments",
    "Date Modified": "date_modified",
    "CTOI lastmod": "date_modified",  # endpoint dialect
}


def _read_exofop_csv(path: Path) -> pd.DataFrame:
    """Read an ExoFOP export, tolerating the optional provenance banner line.

    CTOI exports open with a "This file was produced by ..." line before the
    header; TOI exports start at the header. Sniff rather than hard-code so
    either file style works for both tables.
    """
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    skiprows = 0 if "TIC ID" in first else 1
    return pd.read_csv(path, encoding="utf-8-sig", skiprows=skiprows, low_memory=False)


def _coerce_sexagesimal_coords(out: pd.DataFrame) -> None:
    """Convert "21:14:56.88" / "-55:52:18.71" coordinates to degrees in place.

    The download_toi.php endpoint serves sexagesimal RA/Dec (hourangle,
    degrees); the dashboard export serves decimal degrees. Without this,
    to_numeric would silently NaN every coordinate of an endpoint file.
    """
    ra = out["ra_deg"]
    if ra.dtype != object or not ra.astype(str).str.contains(":").any():
        return
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    mask = ra.notna() & out["dec_deg"].notna()
    coords = SkyCoord(
        out.loc[mask, "ra_deg"].astype(str).tolist(),
        out.loc[mask, "dec_deg"].astype(str).tolist(),
        unit=(u.hourangle, u.deg),
    )
    out.loc[mask, "ra_deg"] = coords.ra.deg
    out.loc[mask, "dec_deg"] = coords.dec.deg


def _normalise(raw: pd.DataFrame, renames: dict[str, str], source: str) -> pd.DataFrame:
    out = raw.rename(columns=renames).reindex(columns=CATALOGUE_COLUMNS)
    out["source"] = source
    _coerce_sexagesimal_coords(out)
    for col in _NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["tic_id"] = pd.to_numeric(out["tic_id"], errors="coerce").astype("Int64")
    return out.dropna(subset=["tic_id"])


def load_toi_table(path: Path) -> pd.DataFrame:
    raw = _read_exofop_csv(path)
    out = _normalise(raw, _TOI_RENAMES, source="TOI")
    out["name"] = "TOI-" + raw["TOI"].astype(str)
    log.info("[exofop] TOI table: %d candidates from %s", len(out), path)
    return out


def load_ctoi_table(path: Path) -> pd.DataFrame:
    raw = _read_exofop_csv(path)
    out = _normalise(raw, _CTOI_RENAMES, source="CTOI")
    # ExoFOP leaves "Candidate Name" blank for nearly every CTOI; the stable
    # identifier is the candidate id column — "Candidate TIC ID" in dashboard
    # exports, "CTOI" from the download endpoint (both e.g. 160363.01). Build
    # the conventional "TIC 160363.01" name from it and keep a custom name
    # only where a submitter actually provided one.
    id_col = "Candidate TIC ID" if "Candidate TIC ID" in raw.columns else "CTOI"
    tic_name = "TIC " + raw[id_col].map(lambda v: f"{float(v):.2f}")
    out["name"] = raw["Candidate Name"].astype("string").fillna(tic_name)

    # NExScI computes Teq/TSM/ESM/predicted mass/K for TOIs only. For CTOIs
    # we apply the same recipes (features.followup, pinned to the NExScI
    # worked example) where the export carries the inputs. TSM/ESM stay null
    # here: they need 2MASS J/K magnitudes, which the CTOI export lacks —
    # the serving path can fill them per-target from a TIC query later.
    m_star = followup.stellar_mass_from_logg(
        out["stellar_logg"].to_numpy(), out["stellar_radius_rsun"].to_numpy()
    )
    # The endpoint dialect publishes a fitted Teq for some CTOIs — keep those
    # and compute only the gaps.
    computed_teq = followup.equilibrium_temperature_k(
        m_star,
        out["period_days"].to_numpy(),
        out["stellar_radius_rsun"].to_numpy(),
        out["stellar_teff_k"].to_numpy(),
    )
    out["teq_k"] = out["teq_k"].fillna(pd.Series(computed_teq, index=out.index))
    out["predicted_mass_me"] = followup.predict_planet_mass_me(out["planet_radius_re"].to_numpy())
    out["predicted_k_ms"] = followup.rv_semi_amplitude_ms(
        out["period_days"].to_numpy(), m_star, out["predicted_mass_me"].to_numpy()
    )
    promoted = out["promoted_to_toi"].astype("string").str.strip().fillna("") != ""
    log.info(
        "[exofop] CTOI table: %d candidates from %s (dropping %d promoted to TOI)",
        len(out),
        path,
        int(promoted.sum()),
    )
    return out[~promoted]


def toi_snr_by_tic(candidates_path: Path) -> pd.Series:
    """tic_id -> ExoFOP TOI transit SNR (strongest signal per TIC).

    Multi-planet systems have several TOIs per TIC; the max SNR corresponds
    to the dominant transit signal (conventionally the .01 entry), which is
    the signal the label catalogue's ephemeris row describes.
    """
    cand = pd.read_parquet(candidates_path)
    toi = cand[(cand["source"] == "TOI") & cand["planet_snr"].notna()]
    return toi.groupby("tic_id")["planet_snr"].max()


def enrich_catalog_snr(catalog: pd.DataFrame, candidates_path: Path) -> pd.DataFrame:
    """Fill missing `snr` on TESS rows from the ExoFOP TOI export.

    The TAP label catalogue only carries SNR for Kepler rows
    (`koi_model_snr`); the NEA TOI table exposes none for TESS, so a
    TESS-only build otherwise ships an all-NaN snr column — which the
    views validation gate rejects. Best-effort: a missing candidate
    catalogue logs a warning and leaves the input unchanged.
    """
    if not candidates_path.exists():
        log.warning(
            "[exofop] %s missing — TESS snr stays NaN (run ingest_exofop.py first)",
            candidates_path,
        )
        return catalog
    snr = toi_snr_by_tic(candidates_path)
    out = catalog.copy()
    if "snr" not in out.columns:
        out["snr"] = np.nan
    fill = (out["mission"] == "TESS") & out["snr"].isna()
    out.loc[fill, "snr"] = out.loc[fill, "tic_id"].map(snr)
    log.info(
        "[exofop] snr enriched from TOI export: %d/%d TESS rows filled",
        int(out.loc[fill, "snr"].notna().sum()),
        int(fill.sum()),
    )
    return out


def _add_poe_observables(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Fill insolation + habitable-zone columns from the archive's POE formulae.

    Computed uniformly for TOIs and CTOIs from the stellar radius/Teff/logg and
    the orbital period — neither ExoFOP export publishes them. Insolation needs
    the semi-major axis (Kepler-3 from the logg-derived mass), the HZ edges only
    the Stefan-Boltzmann luminosity. NaN stellar inputs propagate to NaN, as for
    the other `followup` columns.
    """
    out = catalogue
    r_star = out["stellar_radius_rsun"].to_numpy()
    teff = out["stellar_teff_k"].to_numpy()
    lum = followup.stellar_luminosity_lsun(r_star, teff)
    m_star = followup.stellar_mass_from_logg(out["stellar_logg"].to_numpy(), r_star)
    a_au = followup.semi_major_axis_au(m_star, out["period_days"].to_numpy())
    out["insolation_earth"] = followup.insolation_flux_earth(lum, a_au)
    out["hz_inner_au"], out["hz_outer_au"] = followup.habitable_zone_au(lum)
    return out


def build_candidate_catalogue(toi_path: Path, ctoi_path: Path) -> pd.DataFrame:
    """One row per candidate: all TOIs plus all not-yet-promoted CTOIs."""
    catalogue = pd.concat(
        [load_toi_table(toi_path), load_ctoi_table(ctoi_path)],
        ignore_index=True,
    ).sort_values(["source", "tic_id"], ignore_index=True)
    if catalogue["name"].isna().any():
        raise ValueError("candidate catalogue has rows without a name — ingest bug")
    catalogue = _add_poe_observables(catalogue)
    log.info("[exofop] combined catalogue: %d candidates", len(catalogue))
    return catalogue
