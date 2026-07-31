"""Fetch TESS DV report XML for every target the pipeline knows about.

Resumable and manifest-tracked from the first run — every long job here has
been interrupted at least once. Re-running skips finished targets and targets
known to have no DV products, and retries only transient failures.

    python pipeline/scripts/fetch_dv.py                    # labelled TESS + scored candidates
    python pipeline/scripts/fetch_dv.py --limit 400        # measure shard growth first
    python pipeline/scripts/fetch_dv.py --tics my_tics.csv # an explicit list

**Do not run alongside another MAST job** — contention with the validation runs
is what tripped astroquery's 600 s limit and cost the back half of a run.

Sized 2026-08-01 from 200 sampled targets: 81.5% have DV products at ~0.53 MB
selected each, so 7,199 targets is roughly 3.8 GB and ~35 min of queries.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from exoplanet_hunter.data.dv import DVArchive
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def _default_tics(root: Path) -> list[int]:
    """Labelled TESS targets ∪ scored candidates — the 7,199 that need DV."""
    tics: set[int] = set()
    labels = root / "data" / "labels" / "labels.parquet"
    if labels.exists():
        df = pd.read_parquet(labels)
        tics |= {int(t) for t in df[df["mission"] == "TESS"]["tic_id"]}
    scored = root / "results" / "candidates_scored.parquet"
    if scored.exists():
        tics |= {int(t) for t in pd.read_parquet(scored)["tic_id"]}
    return sorted(tics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository root (for default TIC sources)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Cache dir (default <root>/data/raw_dv)"
    )
    parser.add_argument(
        "--tics", type=Path, default=None, help="CSV or parquet with a tic_id column"
    )
    parser.add_argument("--batch-size", type=int, default=40, help="TICs per MAST round trip")
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after this many targets (0 = all)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-query and re-download even for finished targets"
    )
    args = parser.parse_args()

    out = args.out or (args.root / "data" / "raw_dv")
    if args.tics is not None:
        df = (
            pd.read_parquet(args.tics) if args.tics.suffix == ".parquet" else pd.read_csv(args.tics)
        )
        tics = sorted({int(t) for t in df["tic_id"]})
    else:
        tics = _default_tics(args.root)
    if args.limit:
        tics = tics[: args.limit]

    archive = DVArchive(out, batch_size=args.batch_size)
    todo = [t for t in tics if args.force or not archive.is_done(t)]
    log.info(
        "[dv] %d targets, %d already done, %d to fetch -> %s",
        len(tics),
        len(tics) - len(todo),
        len(todo),
        out,
    )
    if not todo:
        return

    n_ok = n_none = n_fail = 0
    n_bytes = 0
    started = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        results = archive.fetch_batch(batch, force=args.force)
        for r in results:
            if r.success:
                n_ok += 1
                n_bytes += sum(p.stat().st_size for p in r.paths if p.exists())
            elif r.reason == "no DV products":
                n_none += 1
            else:
                n_fail += 1
        done = i + len(batch)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0.0
        # One line per batch, not per target: 7,199 targets at 40 per line is
        # 180 lines, which stays readable in a log tailed over hours.
        log.info(
            "[dv] %d/%d  ok=%d none=%d fail=%d  %.2f GB  %.1f targets/s  eta %.0f min",
            done,
            len(todo),
            n_ok,
            n_none,
            n_fail,
            n_bytes / 1e9,
            rate,
            (len(todo) - done) / rate / 60 if rate else 0.0,
        )

    log.info(
        "[dv] finished: %d fetched, %d with no DV products, %d failed, %.2f GB in %.0f min",
        n_ok,
        n_none,
        n_fail,
        n_bytes / 1e9,
        (time.time() - started) / 60,
    )


if __name__ == "__main__":
    main()
