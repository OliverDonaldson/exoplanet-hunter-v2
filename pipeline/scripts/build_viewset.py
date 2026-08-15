"""Build the branch-model view set for every target in a catalogue.

Resolution comes from `preprocess.viewset`'s own constants and is not restated
here — this line read "301/31" while the builder emitted 2001/201.

    python pipeline/scripts/build_viewset.py --root /Users/ollie/Project/v2

No network: light curves come from the existing caches, stellar parameters from
`dv_scalars.parquet` rather than a per-target TIC query. Writes one small npz
per target under `--cache` first, so an interrupted run resumes and shard size
can be measured after a few hundred targets rather than at the end.

Leaves the legacy `views.npz` alone — that artefact feeds the live model.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.preprocess.viewset import GLOBAL_BINS, LOCAL_BINS, build_view_set
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation.schemas import MISSIONS

log = get_logger(__name__)

SIGMA_CLIP = 5.0
WINDOW_LENGTH = 401
POLYORDER = 2


def _assert_missions_resolve(df: pd.DataFrame, catalogue: Path) -> None:
    """Every row names a mission this builder knows how to find files for.

    `row.get("mission", "TESS")` on a Series returns the *stored* NaN rather than
    the default, so a null mission becomes the string "nan": it keys its own
    cache directory, and `_lightcurve_path` falls through to the TESS branch and
    reads a TESS path for a Kepler target. Both produce a view set that builds
    cleanly and is wrong, so the catalogue is checked once here instead.
    """
    if "mission" not in df.columns:
        log.info("[viewset] no mission column in %s — every row treated as TESS", catalogue)
        return
    unknown = sorted(set(df["mission"].dropna().unique()) - set(MISSIONS))
    if null_count := int(df["mission"].isna().sum()):
        unknown.append(f"{null_count} null")
    if unknown:
        raise SystemExit(
            f"{catalogue} carries mission value(s) {unknown} outside {MISSIONS} — these would "
            "key their own view cache and resolve to a TESS light-curve path"
        )


def _lightcurve_path(root: Path, tic: int, mission: str) -> tuple[Path | None, str]:
    """Where this target's FITS lives, and which source it came from."""
    if mission == "Kepler":
        for base in (
            root / "data" / "raw" / "kepler" / "lightcurves",
            root / "data" / "raw" / "tess" / "lightcurves",
        ):
            path = base / f"kic_{tic}.fits"
            if path.exists():
                return path, "kepler"
        return None, "missing"
    if mission == "K2":
        path = root / "data" / "raw" / "k2" / "lightcurves" / f"epic_{tic}.fits"
        return (path, "k2") if path.exists() else (None, "missing")
    spoc = root / "data" / "raw" / "tess" / "lightcurves" / f"tic_{tic}.fits"
    if spoc.exists():
        return spoc, "spoc-2min"
    ffi = root / "data" / "raw" / "tess" / "ffi" / f"tic_{tic}.fits"
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


def _ephemeris_key(row: pd.Series) -> str:
    """Digest of the ephemeris a target's views are folded on.

    Two planets on one host share `(mission, tic)` and differ only here. Keyed
    on the target alone, the second would silently read back the first's views —
    built from the first's period, epoch and duration — and train on them as its
    own. `labels.parquet` currently holds 5,703 rows across 5,703 unique TICs so
    nothing collides today, but a catalogue refresh is precisely what introduces
    a multi-planet host, and `StratifiedGroupKFold(groups=tic_id)` protects the
    split while this protects what the split is made of.

    Formatted to 10 significant figures before hashing so a float that
    round-trips through parquet at a different last bit does not invalidate a
    cache entry.
    """
    raw = "|".join(f"{float(row[key]):.10g}" for key in ("period", "t0", "duration"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cache_path(cache: Path, mission: str, tic: int, ephemeris: str) -> Path:
    """Per-target cache, keyed by resolution and ephemeris.

    The bin counts are baked into every cached array, so a rebuild at a new
    resolution against an unkeyed cache reads back the old shapes. `validate()`
    would catch it — but only after re-reading all 5,423 files, and only if the
    whole cache is stale rather than half of it.

    The ephemeris entered the key on 2026-08-08 and **invalidates every existing
    entry**: the next build re-derives the interim cache from the light curves
    rather than reusing it. That is the one-time cost of making a multi-planet
    host impossible to get wrong, and the interim cache is derived data.
    """
    return _cache_dir(cache) / f"{mission.lower()}_{tic}_{ephemeris}.npz"


def _cache_dir(cache: Path) -> Path:
    """The resolution-scoped directory every cached target lives under."""
    return cache / f"g{GLOBAL_BINS}l{LOCAL_BINS}"


def _has_ephemeris(row: pd.Series) -> bool:
    """Whether this row can be folded at all. No ephemeris, no views."""
    return bool(np.isfinite([row.get("period"), row.get("t0"), row.get("duration")]).all())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalogue", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also build label -1 rows (candidates) — for scoring, not training",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Skip building; stack whatever the cache already holds",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Accept a view set smaller than the catalogue. Without this, targets that have "
            "no cached views and no recorded skip reason are a hard error."
        ),
    )
    args = parser.parse_args()

    catalogue = args.catalogue or (args.root / "data" / "labels" / "labels.parquet")
    out = args.out or (args.root / "data" / "processed")
    cache = args.cache or (args.root / "data" / "interim" / "viewset")
    _cache_dir(cache).mkdir(parents=True, exist_ok=True)
    log.info("[viewset] %d/%d bins, cache %s", GLOBAL_BINS, LOCAL_BINS, cache)

    df = pd.read_parquet(catalogue)
    # Candidates carry label -1. They cannot train, but scoring them is the only
    # way to measure model behaviour on the population the observation-bias
    # finding was made on.
    keep = [-1, 0, 1] if args.include_candidates else [0, 1]
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    _assert_missions_resolve(df, catalogue)
    log.info("[viewset] %d targets from %s", len(df), catalogue)

    skips = {"missing_fits": 0, "missing_ephemeris": 0, "preprocess_error": 0}
    built = 0
    started = time.time()

    if not args.assemble_only:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            row = pd.Series(row._asdict())
            tic, mission = int(row["tic_id"]), str(row.get("mission", "TESS"))
            # Ordered before the cache lookup: a row with no ephemeris cannot be
            # folded, so there is nothing to key it on and nothing to find.
            if not _has_ephemeris(row):
                skips["missing_ephemeris"] += 1
                continue
            target_cache = _cache_path(cache, mission, tic, _ephemeris_key(row))
            if target_cache.exists():
                built += 1
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
    uncached = 0
    for row in df.itertuples(index=False):
        row = pd.Series(row._asdict())
        tic, mission = int(row["tic_id"]), str(row.get("mission", "TESS"))
        if not _has_ephemeris(row):
            # Never buildable, so its absence is accounted for and is not the
            # silent shortfall `uncached` exists to catch.
            continue
        target_cache = _cache_path(cache, mission, tic, _ephemeris_key(row))
        if not target_cache.exists():
            uncached += 1
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

    # A target missing from the cache is expected exactly when the build loop
    # skipped it and said why. Anything beyond that is a target that should be
    # here and is not, and `--assemble-only` records no skips at all — so a run
    # over a half-populated cache would otherwise write a smaller view set and
    # log a clean build. Everything downstream would then be measured on a
    # population nobody chose.
    # Only the skips that *could* have been built are subtracted: a row with no
    # ephemeris never reaches `uncached` at all, so counting it on both sides
    # would cancel a real shortfall out of existence.
    explained = skips["missing_fits"] + skips["preprocess_error"]
    if (unaccounted := uncached - explained) > 0 and not args.allow_partial:
        raise SystemExit(
            f"{unaccounted} of {len(df)} targets have no cached views and no recorded reason "
            f"({uncached} uncached, {explained} skipped with a reason: {skips}). Re-run without "
            "--assemble-only to build them, or pass --allow-partial to accept a smaller set."
        )

    scalars = pd.DataFrame(rows)
    scalars["transit_completeness"] = scalars["observed_transit_count"] / scalars[
        "expected_transit_count"
    ].replace(0, np.nan)
    # DV publishes its own transit counts. Merging them unrenamed collides with
    # ours and pandas silently suffixes *both* to _x/_y — which dropped the two
    # scalars the whole unfolded-branch thesis rests on, past every gate. Ours
    # are measured from the light curve we actually built views from, so they
    # keep the plain names and DV's are prefixed.
    dv = dv.drop(columns=["dv_file"], errors="ignore").rename(
        columns={
            "observed_transit_count": "dv_observed_transit_count",
            "expected_transit_count": "dv_expected_transit_count",
        }
    )
    collisions = (set(dv.columns) | set(ruwe.columns)) & set(scalars.columns) - {"tic_id"}
    if collisions:
        raise SystemExit(f"side tables would collide with view scalars: {sorted(collisions)}")

    # DV and RUWE are TESS-only by construction; Kepler and K2 rows keep NaN and
    # a False mask rather than a silently imputed value.
    scalars = scalars.merge(dv, on="tic_id", how="left")
    scalars = scalars.merge(ruwe[["tic_id", "ruwe", "n_dr3_candidates"]], on="tic_id", how="left")
    scalars["dv_usable"] = scalars["dv_usable"].astype("boolean").fillna(False).astype(bool)
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
