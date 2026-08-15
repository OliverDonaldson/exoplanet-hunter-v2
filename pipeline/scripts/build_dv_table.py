"""Reduce the fetched DV archive to one parquet row per target.

    python pipeline/scripts/build_dv_table.py --out data/processed/dv_scalars.parquet

No network. Difference-image pixels stay in the XML until stage 9 fixes a stamp
size; this is the scalar half.

`dv_usable` is the column that matters: a report holds one `planetResults` per
TCE, and rows whose best-matching TCE is further than `--max-mismatch` from the
catalogue period describe a different signal and must be masked out.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.data.dv_xml import parse_dv_xml
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

SCALARS = (
    "observed_transit_count",
    "expected_transit_count",
    "max_multiple_event_sigma",
    "max_single_event_sigma",
    "max_ses_in_mes",
    "robust_statistic",
    "bootstrap_significance",
    "bootstrap_threshold_pfa",
    "ghost_core_statistic",
    "ghost_core_significance",
    "ghost_halo_statistic",
    "ghost_halo_significance",
    "chi_square_gof",
    "chi_square_gof_dof",
    "model_fit_snr",
    "odd_even_statistic",
    "odd_even_significance",
    "longer_period_statistic",
    "shorter_period_statistic",
    "weak_secondary_max_mes",
    "weak_secondary_depth_ppm",
    "weak_secondary_robust_statistic",
    "albedo_comparison_statistic",
    "mean_sky_offset",
    "mean_sky_offset_uncertainty",
    "control_sky_offset",
    "control_sky_offset_uncertainty",
    "effective_temp",
    "log_g",
    "log_metallicity",
    "stellar_density",
    "stellar_radius",
    "tess_mag",
    "summary_quality_fraction",
)


def _periods(root: Path) -> dict[int, float]:
    """Catalogue period per TIC, for picking the right TCE."""
    periods: dict[int, float] = {}
    for name in ("data/tables/labels/labels.parquet", "data/tables/labels/candidates.parquet"):
        path = root / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for tic, period in zip(df["tic_id"].astype(int), df["period"], strict=False):
            if tic not in periods and pd.notna(period) and period > 0:
                periods[int(tic)] = float(period)
    return periods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--max-mismatch",
        type=float,
        default=0.01,
        help="Fractional period difference above which a TCE is the wrong signal",
    )
    args = parser.parse_args()

    dv_dir = args.dv or (args.root / "data" / "raw" / "tess" / "dv")
    out = args.out or (args.root / "data" / "processed" / "dv_scalars.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dv_dir / "manifest.json").read_text())
    periods = _periods(args.root)
    # Expected for ~9% of targets and recorded per row; do not flood the log.
    logging.getLogger("exoplanet_hunter.data.dv_xml").setLevel(logging.ERROR)

    rows: list[dict] = []
    failures: list[tuple[int, str]] = []
    fetched = {int(k): v for k, v in manifest.items() if v.get("success")}
    for n, (tic, entry) in enumerate(sorted(fetched.items()), start=1):
        for raw in entry.get("paths", []):
            try:
                r = parse_dv_xml(Path(raw), period_days=periods.get(tic))
            except Exception as exc:
                failures.append((tic, f"{type(exc).__name__}: {exc}"))
                continue
            quality = [
                im.quality_metric for im in r.difference_images if im.quality_metric is not None
            ]
            row = {
                "tic_id": tic,
                "dv_file": Path(raw).name,
                "n_planet_candidates": r.n_planet_candidates,
                "n_difference_images": r.n_difference_images,
                "n_sectors_observed": len(r.sectors_observed),
                "diff_image_min_px": min((im.shape[0] for im in r.difference_images), default=0),
                "diff_image_max_px": max((im.shape[0] for im in r.difference_images), default=0),
                "diff_quality_median": float(np.median(quality)) if quality else np.nan,
                "matched_period_days": r.matched_period_days,
                "period_mismatch_frac": r.period_mismatch_frac,
                "stellar_mass": r.stellar_mass,
            }
            row.update({name: getattr(r, name) for name in SCALARS})
            rows.append(row)
        if n % 1000 == 0:
            log.info("[dv-table] %d/%d targets parsed", n, len(fetched))

    df = pd.DataFrame(rows)
    mismatch = df["period_mismatch_frac"]
    # No catalogue period means we could not check, not that it matched.
    df["dv_usable"] = mismatch.notna() & (mismatch <= args.max_mismatch)
    df.to_parquet(out, index=False)

    log.info(
        "[dv-table] wrote %d rows for %d targets -> %s  (%d parse failures)",
        len(df),
        df["tic_id"].nunique(),
        out,
        len(failures),
    )
    log.info(
        "[dv-table] usable %d (%.1f%%)  wrong-signal %d  unverifiable %d",
        int(df["dv_usable"].sum()),
        100.0 * df["dv_usable"].mean(),
        int((mismatch.notna() & (mismatch > args.max_mismatch)).sum()),
        int(mismatch.isna().sum()),
    )
    for tic, why in failures[:5]:
        log.warning("[dv-table] TIC %d: %s", tic, why)


if __name__ == "__main__":
    main()
