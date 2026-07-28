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
``--shortlist`` (optional) ranks by ``prob_mean`` to pick ``--top`` TESS
targets. Rows that fail (no SAP light curve, TIC gaps) are logged and skipped;
output is written incrementally so a long run is resumable-by-rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation.statistical import (
    estimate_snr,
    prepare_lightcurve,
    validate_target,
)


class TargetTimeout(Exception):
    """One target exceeded its wall-clock budget."""


@contextlib.contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """Abandon a target that overruns, instead of stalling the whole run.

    Runtime is wildly uneven: most targets finish in minutes, but one sat at
    99% CPU for ten hours and produced nothing, which from the terminal is
    indistinguishable from having finished. SIGALRM fires between operations,
    so a single long NumPy call still has to return first — good enough to
    escape a scenario loop, not a hard kill.
    """
    if seconds <= 0:
        yield
        return

    def _raise(signum, frame):
        raise TargetTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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
        rank = scored[["tic_id", "prob_mean"]].dropna()
        df = df.merge(rank, on="tic_id", how="inner").sort_values("prob_mean", ascending=False)
    return df.head(top).reset_index(drop=True)


def _trilegal_cached(cache_dir: Path | None, tic_id: int) -> str | None:
    """Path to this target's saved TRILEGAL population, if we have one."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{tic_id}_TRILEGAL.csv"
    return str(path) if path.exists() else None


def _stash_trilegal(cache_dir: Path | None, tic_id: int) -> None:
    """Move the population TRICERATOPS just wrote into the cache.

    ``save_trilegal`` writes ``<TIC>_TRILEGAL.csv`` into the current directory
    with no way to redirect it, so the files pile up in the repo root (one got
    committed by accident). Moving them into the cache both tidies that up and
    makes the next run reuse the population instead of re-querying.
    """
    if cache_dir is None:
        return
    dropped = Path(f"{tic_id}_TRILEGAL.csv")
    if dropped.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        dropped.replace(cache_dir / dropped.name)


def _validate_row(
    row: pd.Series,
    mission: str,
    n_draws: int,
    search_radius: int,
    verify_ssl: bool,
    use_pipeline_aperture: bool,
    trilegal_cache: Path | None = None,
) -> dict:
    tic_id = int(row["tic_id"])
    period, t0, duration = float(row["period"]), float(row["t0"]), float(row["duration"])
    depth_ppm = float(row["depth"]) * 1e6  # catalogue depth is fractional
    time, flux, sectors, cdpp = _fetch_sap_lightcurve(tic_id, mission=mission)
    phase_time, norm_flux, sigma = prepare_lightcurve(time, flux, period, t0, duration)
    baseline_days = float(time.max() - time.min())
    snr = estimate_snr(depth_ppm, cdpp, int(baseline_days // period))
    # TRILEGAL is a Monte Carlo galaxy simulation: each query returns a
    # different background population, and its star count feeds the background
    # prior directly. Two runs of the same target can therefore disagree on the
    # BEB-vs-NEB balance — one flipped between likely_fp and likely_nearby_fp.
    # Reusing a saved population makes the comparison honest and skips a slow,
    # flaky query.
    cached = _trilegal_cached(trilegal_cache, tic_id)
    if cached:
        log.info("[validate] TIC %d: reusing cached TRILEGAL population", tic_id)
    try:
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
            trilegal_fname=cached,
            verify_ssl=verify_ssl,
            use_pipeline_aperture=use_pipeline_aperture,
        )
    finally:
        _stash_trilegal(trilegal_cache, tic_id)
    return {
        "tic_id": tic_id,
        "fpp": result.fpp,
        "nfpp": result.nfpp,
        "classification": result.classification,
        "best_scenario": result.best_scenario,
        "n_nearby_stars": result.n_nearby_stars,
        "snr": result.snr,
        "snr_reliable": result.snr_reliable,
        "degenerate": result.degenerate,
        "degenerate_reason": result.degenerate_reason,
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
    parser.add_argument(
        "--per-target-timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Abandon a target after this long and move on (0 disables). "
        "Most finish in minutes; one has run for 10 hours at full CPU without "
        "producing anything, which stalls every target behind it.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Resume: keep rows already in --out and re-run only what is missing. "
        "Rows that ended in error or timeout are retried, not kept.",
    )
    parser.add_argument(
        "--trilegal-cache",
        type=Path,
        default=Path("data/trilegal"),
        help="Directory of saved TRILEGAL background populations, one per target. "
        "TRILEGAL is a Monte Carlo galaxy simulation, so re-querying gives a "
        "different population and a different background prior — caching makes "
        "runs reproducible and comparable. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per target for transient failures (MAST/TRILEGAL timeouts), "
        "with 60s-per-attempt backoff. Timeouts are not retried.",
    )
    parser.add_argument(
        "--insecure-trilegal",
        action="store_true",
        help="Skip SSL verification for the public TRILEGAL star-count query "
        "(its server ships a broken cert chain). Only RA/Dec is sent.",
    )
    parser.add_argument(
        "--no-pipeline-aperture",
        action="store_true",
        help="Use TRICERATOPS' 5x5 default aperture instead of the SPOC pipeline "
        "aperture (FPP will read looser/higher).",
    )
    args = parser.parse_args()
    if args.trilegal_cache and str(args.trilegal_cache):
        args.trilegal_cache.mkdir(parents=True, exist_ok=True)
    else:
        args.trilegal_cache = None

    candidates = pd.read_parquet(args.candidates)
    targets = _select(candidates, args.shortlist, args.top, args.mission)
    log.info("[validate] %d %s targets to validate", len(targets), args.mission)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume. The docstring claimed "resumable-by-rerun" but nothing skipped
    # completed work, so an interrupted run redid everything: one restart cost
    # ~3 h re-deriving ten targets that were already on disk. Rows that ended
    # in `error` or `timeout` are NOT treated as done — those are usually a
    # dropped MAST/TRILEGAL connection and deserve another attempt.
    rows: list[dict] = []
    done: set[int] = set()
    if args.skip_completed and args.out.exists():
        prior = pd.read_csv(args.out)
        rows = prior.to_dict("records")
        unfinished = {"error", "timeout"}
        done = {
            int(r["tic_id"])
            for r in rows
            if str(r.get("classification")) not in unfinished and pd.notna(r.get("tic_id"))
        }
        log.info(
            "[validate] resuming from %s: %d already done, %d earlier failure(s) to retry",
            args.out,
            len(done),
            len(rows) - len(done),
        )

    for i, (_, row) in enumerate(targets.iterrows(), 1):
        tic_id = int(row["tic_id"])
        if tic_id in done:
            log.info("[validate] %d/%d TIC %d: already done, skipping", i, len(targets), tic_id)
            continue
        # A retry supersedes its earlier failed row rather than duplicating it.
        rows = [r for r in rows if int(r["tic_id"]) != tic_id]
        # Announce before starting: the only prior output was on completion, so
        # a target that never finished looked identical to a finished run.
        log.info("[validate] %d/%d TIC %d: starting", i, len(targets), tic_id)
        started = time.monotonic()
        try:
            # MAST goes slow in patches: a whole back half of one run was lost
            # to astroquery's own 600 s limit. Those are worth another go —
            # a target that failed at 23:00 succeeded on the next attempt.
            for attempt in range(1, args.retries + 2):
                try:
                    with _time_limit(args.per_target_timeout):
                        out = _validate_row(
                            row,
                            args.mission,
                            args.n_draws,
                            args.search_radius,
                            not args.insecure_trilegal,
                            not args.no_pipeline_aperture,
                            args.trilegal_cache,
                        )
                    break
                except TargetTimeout:
                    raise  # a compute blowup will just blow up again
                except Exception as exc:
                    if attempt > args.retries:
                        raise
                    backoff = 60 * attempt
                    log.warning(
                        "[validate] TIC %d attempt %d/%d failed (%s) — retrying in %ds",
                        tic_id,
                        attempt,
                        args.retries + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
            log.info(
                "[validate] %d/%d TIC %d: FPP=%.3g NFPP=%.3g -> %s  (%.1f min)%s",
                i,
                len(targets),
                tic_id,
                out["fpp"],
                out["nfpp"],
                out["classification"],
                (time.monotonic() - started) / 60,
                "  [DEGENERATE — not a verdict]" if out.get("degenerate") else "",
            )
        except TargetTimeout as exc:
            log.warning(
                "[validate] %d/%d TIC %d ABANDONED after %.1f min (%s) — moving on",
                i,
                len(targets),
                tic_id,
                (time.monotonic() - started) / 60,
                exc,
            )
            out = {"tic_id": tic_id, "classification": "timeout", "error": str(exc)}
        except Exception as exc:
            log.warning("[validate] TIC %d failed: %s", tic_id, exc)
            out = {"tic_id": tic_id, "classification": "error", "error": str(exc)}
        rows.append(out)
        pd.DataFrame(rows).to_csv(args.out, index=False)  # incremental, resumable-by-rerun

    usable = sum(
        1
        for r in rows
        if r.get("classification") not in {"error", "timeout"} and not r.get("degenerate")
    )
    log.info(
        "[validate] wrote %d results (%d usable) -> %s",
        len(rows),
        usable,
        args.out,
    )


if __name__ == "__main__":
    main()
