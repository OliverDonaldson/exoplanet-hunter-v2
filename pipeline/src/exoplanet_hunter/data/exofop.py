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

import pandas as pd

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
    "stellar_teff_k",
    "stellar_logg",
    "stellar_radius_rsun",
    "stellar_distance_pc",
]

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
    "RA (deg)": "ra_deg",
    "Dec (deg)": "dec_deg",
    "Transit Epoch (BJD)": "epoch_bjd",
    "Period (days)": "period_days",
    "Duration (hours)": "duration_hours",
    "Depth (ppm)": "depth_ppm",
    "Planet Radius (R_Earth)": "planet_radius_re",
    "Stellar Teff (K)": "stellar_teff_k",
    "Stellar log(g) (cm/s2)": "stellar_logg",
    "Stellar Radius (R_Sun)": "stellar_radius_rsun",
    "Stellar Distance (pc)": "stellar_distance_pc",
    "Notes": "comments",
    "Date Modified": "date_modified",
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


def _normalise(raw: pd.DataFrame, renames: dict[str, str], source: str) -> pd.DataFrame:
    out = raw.rename(columns=renames).reindex(columns=CATALOGUE_COLUMNS)
    out["source"] = source
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
    # identifier is "Candidate TIC ID" (e.g. 160363.01), so build the
    # conventional "TIC 160363.01" name from it and keep a custom name only
    # where a submitter actually provided one.
    tic_name = "TIC " + raw["Candidate TIC ID"].map(lambda v: f"{float(v):.2f}")
    out["name"] = raw["Candidate Name"].astype("string").fillna(tic_name)
    promoted = out["promoted_to_toi"].astype("string").str.strip().fillna("") != ""
    log.info(
        "[exofop] CTOI table: %d candidates from %s (dropping %d promoted to TOI)",
        len(out),
        path,
        int(promoted.sum()),
    )
    return out[~promoted]


def build_candidate_catalogue(toi_path: Path, ctoi_path: Path) -> pd.DataFrame:
    """One row per candidate: all TOIs plus all not-yet-promoted CTOIs."""
    catalogue = pd.concat(
        [load_toi_table(toi_path), load_ctoi_table(ctoi_path)],
        ignore_index=True,
    ).sort_values(["source", "tic_id"], ignore_index=True)
    if catalogue["name"].isna().any():
        raise ValueError("candidate catalogue has rows without a name — ingest bug")
    log.info("[exofop] combined catalogue: %d candidates", len(catalogue))
    return catalogue
