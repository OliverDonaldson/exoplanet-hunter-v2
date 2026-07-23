"""Statistically validate top shortlist candidates with TRICERATOPS.

A slow, offline follow-on to ``score_candidates.py``: for the highest-scoring
candidates it computes the false-positive probability (FPP) and nearby-FPP
(NFPP) from the TESS pixel data + surrounding stars — the background/nearby
eclipsing-binary discrimination the light-curve-only CNN cannot do. See
``exoplanet_hunter.validation.statistical`` for the method and thresholds.

Needs the optional dependency and network access (MAST + TIC):
    pip install -e 'pipeline[validation]'

Usage (terminal-first — this is minutes per target):
    python scripts/validate_candidates.py \
        --candidates data/labels/candidates.parquet \
        --shortlist results/candidates_scored.parquet \
        --top 20 --out results/candidates_validated.csv

``--candidates`` supplies the ephemeris (tic_id, period, t0, duration, depth) —
the held-out candidate table `score_candidates.py` scores, NOT the ExoFOP
`data/catalogue` table (different column names/units);
``--shortlist`` (optional) ranks by ``prob_calibrated`` to pick ``--top`` TESS
targets. Rows that fail (no SAP light curve, TIC gaps) are logged and skipped;
output is written incrementally so a long run is resumable-by-rerun.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation.statistical import (
    estimate_snr,
    prepare_lightcurve,
    validate_target,
)

log = get_logger(__name__)


def _fetch_sap_lightcurve(
    tic_id: int, mission: str = "TESS", author: str = "SPOC"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SAP flux (NOT PDCSAP — PDC removes nearby-star contamination), stitched
    across sectors and per-sector median-normalised, plus the observed sectors
    and the median CDPP [ppm]."""
    import lightkurve as lk

    search = lk.search_lightcurve(f"TIC {tic_id}", mission=mission, author=author)
    if len(search) == 0:
        raise RuntimeError(f"no {author} light curve for TIC {tic_id}")
    times, fluxes, sectors, cdpps = [], [], [], []
    for lc in search.download_all():
        sap = np.asarray(lc["sap_flux"].value, dtype=float)
        t = np.asarray(lc.time.value, dtype=float)
        ok = np.isfinite(t) & np.isfinite(sap) & (sap > 0)
        if ok.sum() == 0:
            continue
        med = float(np.nanmedian(sap[ok]))
        times.append(t[ok])
        fluxes.append(sap[ok] / med)
        sector = lc.meta.get("SECTOR")
        if sector is not None:
            sectors.append(int(sector))
        with contextlib.suppress(Exception):  # CDPP is best-effort
            cdpps.append(float(lc.estimate_cdpp().value))
    if not times:
        raise RuntimeError(f"no usable SAP cadences for TIC {tic_id}")
    cdpp = float(np.median(cdpps)) if cdpps else float("nan")
    return np.concatenate(times), np.concatenate(fluxes), np.array(sorted(set(sectors))), cdpp


def _select(
    candidates: pd.DataFrame, shortlist: Path | None, top: int, mission: str
) -> pd.DataFrame:
    df = candidates.copy()
    if "mission" in df.columns:
        df = df[df["mission"] == mission]
    if shortlist is not None:
        scored = (
            pd.read_parquet(shortlist) if shortlist.suffix == ".parquet" else pd.read_csv(shortlist)
        )
        rank = scored[["tic_id", "prob_calibrated"]].dropna()
        df = df.merge(rank, on="tic_id", how="inner").sort_values(
            "prob_calibrated", ascending=False
        )
    return df.head(top).reset_index(drop=True)


def _validate_row(row: pd.Series, mission: str, n_draws: int, search_radius: int) -> dict:
    tic_id = int(row["tic_id"])
    period, t0, duration = float(row["period"]), float(row["t0"]), float(row["duration"])
    depth_ppm = float(row["depth"]) * 1e6  # catalogue depth is fractional
    time, flux, sectors, cdpp = _fetch_sap_lightcurve(tic_id, mission=mission)
    phase_time, norm_flux, sigma = prepare_lightcurve(time, flux, period, t0, duration)
    baseline_days = float(time.max() - time.min())
    snr = estimate_snr(depth_ppm, cdpp, int(baseline_days // period))
    result = validate_target(
        tic_id=tic_id,
        sectors=sectors,
        period_days=period,
        depth_ppm=depth_ppm,
        phase_time=phase_time,
        flux=norm_flux,
        flux_err=sigma,
        mission=mission,
        n_draws=n_draws,
        search_radius=search_radius,
        snr=snr,
    )
    return {
        "tic_id": tic_id,
        "fpp": result.fpp,
        "nfpp": result.nfpp,
        "classification": result.classification,
        "best_scenario": result.best_scenario,
        "n_nearby_stars": result.n_nearby_stars,
        "snr": result.snr,
        "snr_reliable": result.snr_reliable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/labels/candidates.parquet"))
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--mission", default="TESS")
    parser.add_argument("--out", type=Path, default=Path("results/candidates_validated.csv"))
    parser.add_argument("--n-draws", type=int, default=1_000_000)
    parser.add_argument("--search-radius", type=int, default=10)
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates)
    targets = _select(candidates, args.shortlist, args.top, args.mission)
    log.info("[validate] %d %s targets to validate", len(targets), args.mission)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, (_, row) in enumerate(targets.iterrows(), 1):
        tic_id = int(row["tic_id"])
        try:
            out = _validate_row(row, args.mission, args.n_draws, args.search_radius)
            log.info(
                "[validate] %d/%d TIC %d: FPP=%.3g NFPP=%.3g -> %s",
                i,
                len(targets),
                tic_id,
                out["fpp"],
                out["nfpp"],
                out["classification"],
            )
        except Exception as exc:
            log.warning("[validate] TIC %d failed: %s", tic_id, exc)
            out = {"tic_id": tic_id, "classification": "error", "error": str(exc)}
        rows.append(out)
        pd.DataFrame(rows).to_csv(args.out, index=False)  # incremental, resumable-by-rerun

    log.info("[validate] wrote %d results -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
