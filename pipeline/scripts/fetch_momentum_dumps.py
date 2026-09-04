"""Fetch TESS reaction-wheel desaturation ("momentum dump") times, per sector.

    python pipeline/scripts/fetch_momentum_dumps.py --out data/tables/momentum_dumps.parquet

Needs network (MAST). Re-running skips sectors already in `--out`.

**Why this is one light curve per sector rather than one per target.** A
momentum dump is a spacecraft attitude event: the reaction wheels are spun down
and every target on the focal plane is flagged in the same cadences. Measured
2026-08-27 on sector 1, four independent TICs (261136679, 355094959, 389729346,
117516398) each carry **the same 70 flagged timestamps**, equal to 1e-6 d — not
merely a similar count. So one representative target fixes the dump times for
every target in that sector, and this table is 10 to 20 rows per sector rather
than a per-target column.

**Why it has to be fetched at all, when 29 GB of TESS light curves are already
on disk.** The cached curves were downloaded through lightkurve's *default*
quality bitmask, which drops desaturation cadences before the file is written.
Bit 32 is therefore zero on every cadence of all 6,192 cached targets — the
flag survives nowhere in the cache, and `preprocess/viewset.py::_gap_view`
already records that finding. These downloads use `quality_bitmask="none"`, so
the cadences are present and the flag can be read.

The table is `[sector, time]`, one row per flagged **cadence** (not per event),
in BTJD — the time system the cached light curves use. Cadences are kept rather
than the events they group into, because clustering is the consumer's business
and a stored interval cannot be un-clustered. Every row comes from a 120-s
product; see `_dump_times`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

#: The DQ bit TESS sets on a reaction-wheel desaturation cadence.
MOMENTUM_DUMP_BIT = 32

#: Consecutive empty sectors before the scan stops. Two, not one: a sector with
#: no 2-min product at all would otherwise end the scan early and leave the
#: table quietly short, which reads downstream as "this target saw no dumps".
_EMPTY_SECTORS_TO_STOP = 2


#: Substrings of a MAST error that mean "try again", not "this sector is empty".
#: Same list `data/download.py::_TRANSIENT_ERROR_MARKERS` keeps, for the same
#: reason: caching one of these as a permanent answer writes a sector out of the
#: table, and a target in it then reads "measured, no dumps" instead of "never
#: fetched". Seen 2026-08-27 mid-run — `SHUTDOWN is in progress. Login failed for
#: user 'STSCI\\mastiisdist'`.
_TRANSIENT = (
    "SHUTDOWN",
    "kill state",
    "Login failed",
    "CAOMv240",
    "Timeout",
    "timed out",
    "SQL Server",
    "Connection aborted",
    "RemoteDisconnected",
    "500 Server",
    "503",
    "network-related",
)


def _is_transient(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _TRANSIENT)


def _sector_targets(sector: int, limit: int = 4, *, attempts: int = 6) -> list[str]:
    """A few TIC IDs with 2-min SPOC data in this sector.

    Retried with backoff on a transient archive error. An empty list means the
    archive genuinely has no 2-min product for the sector, which is what ends the
    scan — so a MAST outage must never be allowed to look like one.
    """
    import time

    from astroquery.mast import Observations

    for attempt in range(attempts):
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                sequence_number=sector,
                dataproduct_type="timeseries",
                t_exptime=120,
            )
        except Exception as exc:
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            wait = min(60 * 2**attempt, 900)
            log.warning(
                "[momentum] sector %d query failed (%s); retry %d/%d in %ds",
                sector,
                str(exc).splitlines()[0][:80],
                attempt + 1,
                attempts - 1,
                wait,
            )
            time.sleep(wait)
            continue
        if obs is None or not len(obs):
            return []
        return [str(name) for name in obs["target_name"][:limit]]
    return []


def _dump_times(tic: str, sector: int, download_dir: Path) -> np.ndarray | None:
    """Flagged cadence times for one target-sector, or None if it will not load."""
    import lightkurve as lk

    # `exptime=120` is pinned, not incidental: the 20-s products carry the same
    # dump *intervals* sampled six times as finely, and a table mixing the two
    # would make a sector's cadence count depend on which target represented it.
    search = lk.search_lightcurve(
        f"TIC {tic}", mission="TESS", author="SPOC", sector=sector, exptime=120
    )
    if not len(search):
        return None
    # The whole point of the fetch: the cached copies were masked at download
    # time and carry the flag on no cadence at all.
    curve = search.download(quality_bitmask="none", download_dir=str(download_dir))
    if curve is None:
        return None
    quality = np.asarray(curve.quality, dtype=np.int64)
    time = np.asarray(curve.time.value, dtype=float)
    flagged = (quality & MOMENTUM_DUMP_BIT) != 0
    return np.sort(time[flagged & np.isfinite(time)])


def _verify_common_mode(sector: int, reference: np.ndarray, download_dir: Path) -> bool:
    """Does a second target in this sector carry the identical flagged cadences?

    The table's whole construction rests on the flag being a property of the
    spacecraft rather than of the target. That was measured on sector 1 before
    this script existed; checking it again on the sector being written costs one
    download and turns the assumption into a per-sector assertion.
    """
    for tic in _sector_targets(sector, limit=3):
        other = _dump_times(tic, sector, download_dir)
        if other is None:
            continue
        if other.size != reference.size:
            log.warning(
                "[momentum] sector %d: TIC %s has %d flagged cadences, reference has %d",
                sector,
                tic,
                other.size,
                reference.size,
            )
            return False
        return bool(np.allclose(other, reference, atol=1e-6))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--first-sector", type=int, default=1)
    parser.add_argument("--last-sector", type=int, default=120)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="lightkurve cache for the representative curves. Kept out of "
        "data/raw so this fetch cannot be mistaken for the target cache",
    )
    parser.add_argument(
        "--verify-every",
        type=int,
        default=10,
        help="re-check the common-mode property on every Nth sector; 0 disables",
    )
    args = parser.parse_args()

    out = args.out or (args.root / "data" / "tables" / "momentum_dumps.parquet")
    download_dir = args.download_dir or (args.root / ".phase1-scratch" / "mdcache")
    download_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["sector", "time"])
    done = {int(s) for s in existing["sector"].unique()}
    rows: list[pd.DataFrame] = [existing] if len(existing) else []
    log.info("[momentum] %d sector(s) already in %s", len(done), out)

    empty_run = 0
    for sector in range(args.first_sector, args.last_sector + 1):
        if sector in done:
            continue
        targets = _sector_targets(sector)
        if not targets:
            empty_run += 1
            log.info("[momentum] sector %d: no 2-min product", sector)
            if empty_run >= _EMPTY_SECTORS_TO_STOP:
                log.info("[momentum] stopping at sector %d — archive has no more", sector)
                break
            continue
        empty_run = 0

        times = None
        for tic in targets:
            try:
                times = _dump_times(tic, sector, download_dir)
            except Exception as exc:  # one bad product is not a failed sector
                log.warning(
                    "[momentum] sector %d TIC %s failed%s: %s",
                    sector,
                    tic,
                    " (transient)" if _is_transient(exc) else "",
                    str(exc).splitlines()[0][:120],
                )
                if _is_transient(exc):
                    # Every target in the sector would hit the same outage, so
                    # trying the next one just burns the candidate list and
                    # leaves the sector SKIPPED — which reads downstream as a
                    # sector with no dumps. Back off and retry this one instead.
                    import time

                    time.sleep(120)
                continue
            if times is not None:
                break
        if times is None:
            log.warning("[momentum] sector %d: no target loaded, sector SKIPPED", sector)
            continue
        if not times.size:
            # Real for the sectors flown after the wheels were biased off-null.
            # Written as an empty sector rather than omitted, so a target there
            # reads "measured, no dumps" instead of "sector never fetched".
            log.info("[momentum] sector %d: 0 flagged cadences", sector)

        if args.verify_every and sector % args.verify_every == 0 and times.size:
            agreed = _verify_common_mode(sector, times, download_dir)
            log.info(
                "[momentum] sector %d common-mode check: %s",
                sector,
                "identical on a second target" if agreed else "DISAGREED — see warning above",
            )

        rows.append(pd.DataFrame({"sector": sector, "time": times}))
        table = pd.concat(rows, ignore_index=True).sort_values(["sector", "time"])
        table.to_parquet(out, index=False)
        log.info(
            "[momentum] sector %2d: %4d flagged cadences  (table now %d rows over %d sectors)",
            sector,
            times.size,
            len(table),
            table["sector"].nunique(),
        )

    if rows:
        table = pd.concat(rows, ignore_index=True).sort_values(["sector", "time"])
        table.to_parquet(out, index=False)
        log.info(
            "[momentum] wrote %d cadences over %d sectors -> %s  (BTJD %.2f .. %.2f)",
            len(table),
            table["sector"].nunique(),
            out,
            table["time"].min(),
            table["time"].max(),
        )


if __name__ == "__main__":
    main()
