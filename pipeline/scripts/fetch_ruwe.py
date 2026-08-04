"""Fetch Gaia DR3 RUWE for the pipeline's TESS targets.

    python pipeline/scripts/fetch_ruwe.py --out data/gaia/ruwe.parquet

Needs network (MAST for TIC->Gaia DR2, Gaia TAP for DR2->DR3). Re-running skips
TICs already in `--out`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from exoplanet_hunter.data.gaia import fetch_ruwe
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def _default_tics(root: Path) -> list[int]:
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
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = args.out or (args.root / "data" / "gaia" / "ruwe.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    tics = _default_tics(args.root)
    if args.limit:
        tics = tics[: args.limit]

    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["tic_id"])
    todo = sorted(set(tics) - set(existing["tic_id"].astype(int)))
    log.info(
        "[gaia] %d targets, %d already done, %d to fetch",
        len(tics),
        len(tics) - len(todo),
        len(todo),
    )
    if not todo:
        return

    fetched = fetch_ruwe(todo)
    combined = pd.concat([existing, fetched], ignore_index=True).drop_duplicates("tic_id")
    combined.to_parquet(out, index=False)

    have = combined["ruwe"].notna()
    log.info(
        "[gaia] wrote %d rows -> %s  (%d with RUWE; median %.3f, %.1f%% above 1.4)",
        len(combined),
        out,
        int(have.sum()),
        float(combined.loc[have, "ruwe"].median()) if have.any() else float("nan"),
        100.0 * float((combined.loc[have, "ruwe"] > 1.4).mean()) if have.any() else float("nan"),
    )


if __name__ == "__main__":
    main()
