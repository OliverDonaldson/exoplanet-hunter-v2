"""Convert a built view set into a TFRecord shard set for training.

    python pipeline/scripts/shard_viewset.py
    python pipeline/scripts/shard_viewset.py --viewset data/processed \
        --out-dir data/processed/viewset_tfrecords --examples-per-shard 512

    # stage 8 arm N: the catalogue plus synthetic negatives, in its own shard set
    python pipeline/scripts/shard_viewset.py \
        --extra data/processed/synthetic_negatives \
        --out-dir data/processed/viewset_tfrecords_synneg

Runs the view-set gate first: a malformed set must not reach shards, because
once sharded the shape errors look like model bugs.

`--extra` appends another `ViewSetArrays` directory. It writes to a **separate**
`--out-dir` by design — the control arm and arms P and S read the unaugmented
shard set, and overwriting it in place would silently change what every other
arm trains on and make the comparison meaningless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.datasets.viewset_tfrecords import write_viewset_shards
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import check_view_set

log = get_logger(__name__)

#: Fold-grouping key. Present only once a view set carries rows derived from
#: another row's star — see `build_synthetic_negatives.py`.
GROUP_COLUMN = "group_tic"


def append(base: ViewSetArrays, extra: ViewSetArrays) -> ViewSetArrays:
    """Concatenate two view sets, keeping the fold-grouping key complete.

    Every row ends up with a `group_tic`: a real row is its own group, and a
    synthetic row carries the star it was derived from. Left partly missing, the
    trainer's grouped split would put a synthetic negative and the star it came
    from in different folds — testing the model on a light curve whose noise,
    gaps and systematics it trained on.

    Raises on a `tic_id` collision rather than merging: the split and weight
    tables key on it, so two rows sharing one would silently share a split code.
    """
    for name in VIEW_SHAPES:
        if name not in base.views or name not in extra.views:
            raise KeyError(f"both view sets need every declared view; {name} is missing from one")

    overlap = set(base.scalars["tic_id"]) & set(extra.scalars["tic_id"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} tic_id(s) appear in both view sets, e.g. {sorted(overlap)[:5]}. "
            "The split and weight tables key on tic_id, so these rows would share a split "
            "code — give the appended rows identifiers of their own"
        )

    scalars = []
    for part in (base.scalars, extra.scalars):
        frame = part.copy()
        if GROUP_COLUMN not in frame.columns:
            frame[GROUP_COLUMN] = frame["tic_id"]
        else:
            frame[GROUP_COLUMN] = frame[GROUP_COLUMN].fillna(frame["tic_id"])
        scalars.append(frame)

    merged = pd.concat(scalars, ignore_index=True)
    views = {
        name: np.concatenate([base.views[name], extra.views[name]]).astype(np.float32)
        for name in VIEW_SHAPES
    }
    log.info(
        "[shard-viewset] appended %d row(s) to %d (%d group(s), %d positive)",
        len(extra.scalars),
        len(base.scalars),
        merged[GROUP_COLUMN].nunique(),
        int(merged["label"].sum()),
    )
    return ViewSetArrays(views=views, scalars=merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewset", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/viewset_tfrecords"))
    parser.add_argument("--examples-per-shard", type=int, default=512)
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=None,
        help="another ViewSetArrays directory to append; repeatable. Write these to a "
        "separate --out-dir, or every other arm silently trains on them too",
    )
    args = parser.parse_args()

    arrays = ViewSetArrays.load(args.viewset)
    for extra in args.extra or []:
        arrays = append(arrays, ViewSetArrays.load(extra))

    problems = check_view_set(arrays)
    if problems:
        for problem in problems:
            log.error("[shard-viewset] %s", problem)
        sys.exit(1)

    write_viewset_shards(arrays, args.out_dir, examples_per_shard=args.examples_per_shard)


if __name__ == "__main__":
    main()
