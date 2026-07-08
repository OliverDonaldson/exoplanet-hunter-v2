"""Convert a processed views.npz into a TFRecord shard set for training.

Usage (from the repository root):

    python pipeline/scripts/shard_views.py
    python pipeline/scripts/shard_views.py --views data/processed/views.npz \
        --out-dir data/processed/tfrecords --examples-per-shard 1024

The shard set (shards + metadata.json + index.parquet) is what
`exoplanet_hunter.training.train` streams from, and what eventually syncs
to R2 for the GPU burst.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_hunter.datasets import load_views, write_tfrecord_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, default=Path("data/processed/views.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tfrecords"))
    parser.add_argument("--examples-per-shard", type=int, default=1024)
    args = parser.parse_args()

    views = load_views(args.views)
    write_tfrecord_shards(views, args.out_dir, examples_per_shard=args.examples_per_shard)


if __name__ == "__main__":
    main()
