"""Fetch FFI light curves for candidates with no 2-minute SPOC product.

    python pipeline/scripts/fetch_ffi.py --root /Users/ollie/Project/v2

Targets default to the `no_fits` rows of `results/candidates_scored.parquet`.
Needs network; resumable. **Do not run alongside another MAST job.**
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from exoplanet_hunter.data.ffi import FFIDownloader
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def _default_tics(root: Path) -> list[int]:
    scored = root / "results" / "candidates_scored.parquet"
    if not scored.exists():
        return []
    df = pd.read_parquet(scored)
    return sorted({int(t) for t in df[df["status"] == "no_fits"]["tic_id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out or (args.root / "data" / "raw" / "tess" / "ffi")
    tics = _default_tics(args.root)
    if args.limit:
        tics = tics[: args.limit]

    downloader = FFIDownloader(out)
    todo = [t for t in tics if args.force or not downloader.is_done(t)]
    log.info(
        "[ffi] %d targets, %d already done, %d to fetch -> %s",
        len(tics),
        len(tics) - len(todo),
        len(todo),
        out,
    )
    if not todo:
        return

    authors: Counter[str] = Counter()
    n_ok = n_fail = 0
    started = time.time()
    for i, tic in enumerate(todo, start=1):
        result = downloader.download_one(tic, force=args.force)
        if result.success:
            n_ok += 1
            authors[result.author or "?"] += 1
        else:
            n_fail += 1
        elapsed = time.time() - started
        log.info(
            "[ffi] %d/%d TIC %d %s  ok=%d fail=%d  eta %.0f min",
            i,
            len(todo),
            tic,
            f"{result.author} {result.n_sectors}sec {result.cadence_seconds:.0f}s"
            if result.success and result.cadence_seconds
            else (result.reason or "failed")[:60],
            n_ok,
            n_fail,
            (len(todo) - i) * elapsed / i / 60,
        )

    log.info(
        "[ffi] finished: %d recovered, %d failed, in %.0f min  authors=%s",
        n_ok,
        n_fail,
        (time.time() - started) / 60,
        dict(authors),
    )


if __name__ == "__main__":
    main()
