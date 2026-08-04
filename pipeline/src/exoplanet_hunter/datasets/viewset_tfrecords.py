"""TFRecord shards for the 301/31 view set.

Parallel to `tfrecords.py`, which stays pinned to the legacy 2001/201 schema
that feeds the live model. This one is generic over `VIEW_SHAPES`, so adding a
branch means adding it there and nowhere else.

    viewset-00000-of-00006.tfrecord
    metadata.json   view shapes, scalar columns, n_examples, n_shards
    index.parquet   the scalars table, in shard order

Measured 2026-08-01: ~26 kB per example, so the full labelled set is ~150 MB —
about 3x the legacy 47 MB, not the 20-50x the roadmap allowed for. That keeps
`tf.data.cache()` viable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

METADATA_NAME = "metadata.json"
INDEX_NAME = "index.parquet"
#: Scalars the model may condition on. Anything else in the parquet is metadata.
FEATURE_COLUMNS = (
    "observed_transit_count",
    "expected_transit_count",
    "transit_completeness",
    "secondary_phase",
    "ruwe",
    "max_multiple_event_sigma",
    "robust_statistic",
    "bootstrap_significance",
    "ghost_core_statistic",
    "ghost_halo_statistic",
    "odd_even_statistic",
    "weak_secondary_max_mes",
    "mean_sky_offset",
    "control_sky_offset",
    "summary_quality_fraction",
)
#: Presence masks, so a missing branch is never read as a measured zero.
MASK_COLUMNS = ("dv_usable", "has_ruwe")


def _float_feature(values: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values.ravel().tolist()))


def _int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_viewset_shards(
    arrays: ViewSetArrays, out_dir: Path, *, examples_per_shard: int = 512
) -> dict:
    """Serialise a `ViewSetArrays` into shards + metadata + index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # A rebuild with a different example count names its shards differently, so
    # files from the previous set would survive and poison readers with a mixed
    # schema. A shard set is all-or-nothing.
    for stale in out_dir.glob("viewset-*.tfrecord"):
        stale.unlink()

    scalars = arrays.scalars.reset_index(drop=True)
    n = len(scalars)
    n_shards = max(1, int(np.ceil(n / examples_per_shard)))
    features = [c for c in FEATURE_COLUMNS if c in scalars.columns]
    masks = [c for c in MASK_COLUMNS if c in scalars.columns]
    # Hoisted out of the per-example loop: pandas scalar access per row is the
    # slowest part of writing 5,700 examples.
    labels = scalars["label"].to_numpy(dtype=np.int64)
    tic_ids = scalars["tic_id"].to_numpy(dtype=np.int64)
    # NaN is not a value a dense layer can consume; the mask columns are what
    # say "this was never measured".
    feature_values = (
        np.nan_to_num(scalars[features].to_numpy(dtype=np.float32), nan=0.0)
        if features
        else np.empty((n, 0), dtype=np.float32)
    )
    mask_values = (
        scalars[masks].to_numpy(dtype=np.float32) if masks else np.empty((n, 0), dtype=np.float32)
    )

    for shard_idx in range(n_shards):
        lo = shard_idx * examples_per_shard
        hi = min(lo + examples_per_shard, n)
        path = out_dir / f"viewset-{shard_idx:05d}-of-{n_shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for i in range(lo, hi):
                feature = {name: _float_feature(arrays.views[name][i]) for name in VIEW_SHAPES}
                feature["label"] = _int_feature(int(labels[i]))
                feature["tic_id"] = _int_feature(int(tic_ids[i]))
                if features:
                    feature["scalars"] = _float_feature(feature_values[i])
                if masks:
                    feature["masks"] = _float_feature(mask_values[i])
                writer.write(
                    tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    ).SerializeToString()
                )

    metadata = {
        "n_examples": n,
        "n_shards": n_shards,
        "view_shapes": {k: list(v) for k, v in VIEW_SHAPES.items()},
        "scalar_columns": features,
        "mask_columns": masks,
    }
    (out_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    scalars.to_parquet(out_dir / INDEX_NAME, index=False)

    log.info(
        "[viewset-tfrecords] wrote %d examples (%d shards, %d scalars, %d masks) to %s",
        n,
        n_shards,
        len(features),
        len(masks),
        out_dir,
    )
    return metadata


def load_metadata(shard_dir: Path) -> dict:
    return json.loads((shard_dir / METADATA_NAME).read_text())


def list_shards(shard_dir: Path) -> list[str]:
    return sorted(str(p) for p in shard_dir.glob("viewset-*.tfrecord"))


def load_index(shard_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(shard_dir / INDEX_NAME)


def make_parse_fn(metadata: dict) -> Callable[[tf.Tensor], tuple[dict, tf.Tensor]]:
    """Build the parser for one serialised example → (features dict, label)."""
    shapes = {k: tuple(v) for k, v in metadata["view_shapes"].items()}
    spec: dict[str, tf.io.FixedLenFeature] = {
        name: tf.io.FixedLenFeature([int(np.prod(shape))], tf.float32)
        for name, shape in shapes.items()
    }
    spec["label"] = tf.io.FixedLenFeature([], tf.int64)
    spec["tic_id"] = tf.io.FixedLenFeature([], tf.int64)
    if metadata["scalar_columns"]:
        spec["scalars"] = tf.io.FixedLenFeature([len(metadata["scalar_columns"])], tf.float32)
    if metadata["mask_columns"]:
        spec["masks"] = tf.io.FixedLenFeature([len(metadata["mask_columns"])], tf.float32)

    def parse(serialized: tf.Tensor) -> tuple[dict, tf.Tensor]:
        ex = tf.io.parse_single_example(serialized, spec)
        features: dict = {name: tf.reshape(ex[name], shape) for name, shape in shapes.items()}
        features["tic_id"] = ex["tic_id"]
        if "scalars" in ex:
            features["scalars"] = ex["scalars"]
        if "masks" in ex:
            features["masks"] = ex["masks"]
        return features, tf.cast(ex["label"], tf.float32)

    return parse
