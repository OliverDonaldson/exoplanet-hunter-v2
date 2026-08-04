"""Convert a built view set into a TFRecord shard set for training.

    python pipeline/scripts/shard_viewset.py
    python pipeline/scripts/shard_viewset.py --viewset data/processed \
        --out-dir data/processed/viewset_tfrecords --examples-per-shard 512

Runs the view-set gate first: a malformed set must not reach shards, because
once sharded the shape errors look like model bugs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exoplanet_hunter.datasets.viewset_io import ViewSetArrays
from exoplanet_hunter.datasets.viewset_tfrecords import write_viewset_shards
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import check_view_set

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewset", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/viewset_tfrecords"))
    parser.add_argument("--examples-per-shard", type=int, default=512)
    args = parser.parse_args()

    arrays = ViewSetArrays.load(args.viewset)
    problems = check_view_set(arrays)
    if problems:
        for problem in problems:
            log.error("[shard-viewset] %s", problem)
        sys.exit(1)

    write_viewset_shards(arrays, args.out_dir, examples_per_shard=args.examples_per_shard)


if __name__ == "__main__":
    main()
