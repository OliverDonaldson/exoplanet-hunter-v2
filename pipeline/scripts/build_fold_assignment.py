"""One outer-CV partition, shared by both trainers.

**Why this exists.** `train.py` streams `data/processed/tfrecords` and
`train_branches.py` streams `data/processed/viewset_tfrecords`. They are
overlapping but different populations — 5,380 and 5,426 rows, sharing 5,375 —
so each building its own `StratifiedGroupKFold` puts the same host in different
folds whatever seed they share. An ensemble scored across two such runs is
scored on two different out-of-fold populations while reporting one number.

This resolves the partition **once**, over the intersection, and writes it where
both trainers can replay it:

    python pipeline/scripts/build_fold_assignment.py \\
        --out models/fold_assignments/stage10_5.json

Then `--fold-assignment` on the branch trainer, `train.fold_assignment=...` on
the dual-view one. Rows outside the intersection are dropped from CV by both,
which each logs with a count.

The logic lives in `training.splits` and is tested there; this is the driver
that points it at the two real shard indices and records what it did.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from exoplanet_hunter.training.splits import build_fold_assignment, write_fold_assignment
from exoplanet_hunter.utils.logging import get_logger
from exoplanet_hunter.utils.provenance import git_provenance

log = get_logger(__name__)

DEFAULT_INDICES = (
    Path("data/processed/tfrecords/index.parquet"),
    Path("data/processed/viewset_tfrecords/index.parquet"),
)


def shared_population(indices: list[Path]) -> pd.DataFrame:
    """One row per group present in **every** index, carrying its label.

    An intersection rather than a union: a group only one trainer can see has no
    out-of-fold score from the other, so an ensemble cannot be formed for it, and
    including it would quietly make the two arms' populations differ again.
    """
    frames = []
    for path in indices:
        if not path.exists():
            raise SystemExit(f"no shard index at {path} — build the shard set first")
        frame = pd.read_parquet(path)
        missing = {"tic_id", "label"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} carries no {sorted(missing)} column")
        frames.append(frame[["tic_id", "label"]].drop_duplicates("tic_id"))
        log.info("[folds] %s: %d rows, %d groups", path, len(frame), frame["tic_id"].nunique())

    shared = set.intersection(*(set(f["tic_id"]) for f in frames))
    if not shared:
        raise SystemExit("the shard indices share no groups at all")

    base = frames[0][frames[0]["tic_id"].isin(shared)].sort_values("tic_id").reset_index(drop=True)
    # A group whose label differs between indices would silently stratify one way
    # and be scored another. Cheap to check, impossible to notice later.
    #
    # Compared as values, not with `Series.equals`: the two indices store `label`
    # at different integer widths, and `equals` is dtype-strict, so it reports a
    # mismatch on identical labels — a guard that fires on nothing is worse than
    # no guard, because the next person removes it.
    for frame in frames[1:]:
        other = frame[frame["tic_id"].isin(shared)].sort_values("tic_id").reset_index(drop=True)
        differ = base["label"].to_numpy().astype(int) != other["label"].to_numpy().astype(int)
        if differ.any():
            raise SystemExit(
                f"{int(differ.sum())} group(s) carry different labels between shard indices"
            )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = shared_population(list(args.index))
    log.info("[folds] %d groups shared by all %d indices", len(base), len(args.index))

    assignment = build_fold_assignment(
        base["tic_id"].to_numpy(),
        base["label"].to_numpy().astype(int),
        n_splits=args.n_splits,
        seed=args.seed,
    )
    sizes = pd.Series(list(assignment.values())).value_counts().sort_index()
    log.info("[folds] fold sizes: %s", sizes.to_dict())

    write_fold_assignment(
        args.out,
        assignment,
        seed=args.seed,
        n_splits=args.n_splits,
        sources=[str(p) for p in args.index],
        **git_provenance().as_dict(),
    )
    log.info("[folds] wrote %s", args.out)


if __name__ == "__main__":
    main()
