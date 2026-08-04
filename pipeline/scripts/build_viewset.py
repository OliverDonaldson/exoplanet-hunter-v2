"""Build the 301/31 view set for every target in a catalogue.

    python pipeline/scripts/build_viewset.py --root /Users/ollie/Project/v2

No network: light curves come from the existing caches, stellar parameters from
`dv_scalars.parquet` rather than a per-target TIC query. Writes one small npz
per target under `--cache` first, so an interrupted run resumes and shard size
can be measured after a few hundred targets rather than at the end.

Leaves the legacy `views.npz` alone — that artefact feeds the live model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.preprocess.viewset import build_view_set
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

SIGMA_CLIP = 5.0
WINDOW_LENGTH = 401
POLYORDER = 2


def _lightcurve_path(root: Path, tic: int, mission: str) -> tuple[Path | None, str]:
    """Where this target's FITS lives, and which source it came from."""
    if mission == "Kepler":
        for base in (root / "data" / "raw_kepler", root / "data" / "raw"):
            path = base / f"kic_{tic}.fits"
            if path.exists():
                return path, "kepler"
        return None, "missing"
    if mission == "K2":
        path = root / "data" / "raw" / f"epic_{tic}.fits"
        return (path, "k2") if path.exists() else (None, "missing")
    spoc = root / "data" / "raw" / f"tic_{tic}.fits"
    if spoc.exists():
        return spoc, "spoc-2min"
    ffi = root / "data" / "raw_ffi" / f"tic_{tic}.fits"
    if ffi.exists():
        return ffi, "ffi"
    return None, "missing"


def _side_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dv_path = root / "data" / "processed" / "dv_scalars.parquet"
    ruwe_path = root / "data" / "gaia" / "ruwe.parquet"
    dv = pd.read_parquet(dv_path) if dv_path.exists() else pd.DataFrame(columns=["tic_id"])
    if len(dv):
        # One DV row per target: the widest run, which is the usable one.
        dv = dv.sort_values(["tic_id", "dv_usable", "n_sectors_observed"]).drop_duplicates(
            "tic_id", keep="last"
        )
    ruwe = pd.read_parquet(ruwe_path) if ruwe_path.exists() else pd.DataFrame(columns=["tic_id"])
    return dv, ruwe


def _build_one(path: Path, row: pd.Series) -> tuple[object, dict] | None:
    import lightkurve as lk

    period, t0, duration = float(row["period"]), float(row["t0"]), float(row["duration"])
    raw = lk.read(str(path))
    cleaned = clean_lightcurve(raw, sigma_clip=SIGMA_CLIP)
    flat = flatten_lightcurve(
        cleaned,
        window_length=WINDOW_LENGTH,
        polyorder=POLYORDER,
        period=period,
        t0=t0,
        duration=duration,
    )
    views = build_view_set(
        flat, period=period, t0=t0, duration=duration, trend_lc=cleaned, raw_lc=raw
    )
    scalars = {
        "observed_transit_count": views.observed_transit_count,
        "expected_transit_count": views.expected_transit_count,
        "secondary_phase": views.secondary_phase,
    }
    return views, scalars


def _cache_path(cache: Path, mission: str, tic: int) -> Path:
    return cache / f"{mission.lower()}_{tic}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalogue", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Skip building; stack whatever the cache already holds",
    )
    args = parser.parse_args()

    catalogue = args.catalogue or (args.root / "data" / "labels" / "labels.parquet")
    out = args.out or (args.root / "data" / "processed")
    cache = args.cache or (args.root / "data" / "interim" / "viewset")
    cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(catalogue)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("[viewset] %d labelled targets from %s", len(df), catalogue)

    skips = {"missing_fits": 0, "missing_ephemeris": 0, "preprocess_error": 0}
    built = 0
    started = time.time()

    if not args.assemble_only:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            row = pd.Series(row._asdict())
            tic, mission = int(row["tic_id"]), str(row.get("mission", "TESS"))
            target_cache = _cache_path(cache, mission, tic)
            if target_cache.exists():
                built += 1
                continue
            if not np.isfinite([row.get("period"), row.get("t0"), row.get("duration")]).all():
                skips["missing_ephemeris"] += 1
                continue
            path, source = _lightcurve_path(args.root, tic, mission)
            if path is None:
                skips["missing_fits"] += 1
                continue
            try:
                result = _build_one(path, row)
            except Exception as exc:
                log.debug("[viewset] %s %d failed: %s", mission, tic, exc)
                skips["preprocess_error"] += 1
                continue
            views, scalars = result
            np.savez_compressed(
                target_cache,
                **{name: getattr(views, name) for name in VIEW_SHAPES},
                **{k: np.asarray(v) for k, v in scalars.items()},
                lc_source=np.asarray(source),
            )
            built += 1
            if i % 200 == 0:
                elapsed = time.time() - started
                rate = i / elapsed
                size_mb = sum(p.stat().st_size for p in cache.glob("*.npz")) / 1e6
                log.info(
                    "[viewset] %d/%d built=%d skips=%s  %.1f/s  cache %.0f MB  eta %.0f min",
                    i,
                    len(df),
                    built,
                    skips,
                    rate,
                    size_mb,
                    (len(df) - i) / rate / 60,
                )

    # --- assemble -------------------------------------------------------
    dv, ruwe = _side_tables(args.root)
    view_sets: list[dict] = []
    rows: list[dict] = []
    for row in df.itertuples(index=False):
        row = pd.Series(row._asdict())
        tic, mission = int(row["tic_id"]), str(row.get("mission", "TESS"))
        target_cache = _cache_path(cache, mission, tic)
        if not target_cache.exists():
            continue
        with np.load(target_cache, allow_pickle=False) as f:
            view_sets.append({name: f[name] for name in VIEW_SHAPES})
            rows.append(
                {
                    "tic_id": tic,
                    "mission": mission,
                    "label": int(row["label"]),
                    "period": float(row["period"]),
                    "duration": float(row["duration"]),
                    "depth": float(row["depth"]) if pd.notna(row.get("depth")) else np.nan,
                    "observed_transit_count": int(f["observed_transit_count"]),
                    "expected_transit_count": int(f["expected_transit_count"]),
                    "secondary_phase": float(f["secondary_phase"]),
                    "lc_source": str(f["lc_source"]),
                }
            )

    if not rows:
        raise SystemExit("no targets built — check the light-curve caches")

    scalars = pd.DataFrame(rows)
    scalars["transit_completeness"] = scalars["observed_transit_count"] / scalars[
        "expected_transit_count"
    ].replace(0, np.nan)
    # DV and RUWE are TESS-only by construction; Kepler and K2 rows keep NaN and
    # a False mask rather than a silently imputed value.
    scalars = scalars.merge(dv.drop(columns=["dv_file"], errors="ignore"), on="tic_id", how="left")
    scalars = scalars.merge(ruwe[["tic_id", "ruwe", "n_dr3_candidates"]], on="tic_id", how="left")
    scalars["dv_usable"] = scalars["dv_usable"].fillna(False).astype(bool)
    scalars["has_ruwe"] = scalars["ruwe"].notna()

    views = {
        name: np.stack([vs[name] for vs in view_sets]).astype(np.float32) for name in VIEW_SHAPES
    }
    arrays = ViewSetArrays(views=views, scalars=scalars)
    problems = arrays.validate()
    if problems:
        raise SystemExit(f"assembled view set is malformed: {problems}")
    arrays.save(out)

    total_mb = (
        sum((out / n).stat().st_size for n in ("viewset.npz", "viewset_scalars.parquet")) / 1e6
    )
    log.info(
        "[viewset] wrote %d examples -> %s  (%.0f MB;  pos=%d neg=%d;  skips=%s)",
        len(scalars),
        out,
        total_mb,
        int((scalars["label"] == 1).sum()),
        int((scalars["label"] == 0).sum()),
        skips,
    )
    log.info(
        "[viewset] DV usable %d (%.0f%%)  RUWE %d (%.0f%%)  sources=%s",
        int(scalars["dv_usable"].sum()),
        100.0 * scalars["dv_usable"].mean(),
        int(scalars["has_ruwe"].sum()),
        100.0 * scalars["has_ruwe"].mean(),
        scalars["lc_source"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
