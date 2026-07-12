"""TFRecord serialization of processed views (L6: the large-dataset story).

A shard set is a directory of

    views-00000-of-00012.tfrecord
    metadata.json          n_examples, view lengths, aux_dim, n_shards
    index.parquet          per-example (row, tic_id, label [, aux_0..aux_k])

The parquet index is the small, random-access companion to the sequential
shards: CV splits are computed from it (StratifiedGroupKFold needs labels and
groups in memory — a few KB), aux normalisation is fitted from its aux
columns, and evaluation reads y_true from it in shard order. Examples are
written in index-row order and `make_dataset` preserves that order for
unshuffled reads, so predictions line up with index rows by position.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from exoplanet_hunter.datasets.views_io import ViewArrays
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

METADATA_NAME = "metadata.json"
INDEX_NAME = "index.parquet"


@dataclass
class ShardMetadata:
    n_examples: int
    global_bins: int
    local_bins: int
    aux_dim: int  # 0 = no aux features
    n_shards: int

    @classmethod
    def load(cls, shard_dir: Path) -> ShardMetadata:
        return cls(**json.loads((shard_dir / METADATA_NAME).read_text()))


def _float_feature(values: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values.tolist()))


def _int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord_shards(
    views: ViewArrays,
    out_dir: Path,
    *,
    examples_per_shard: int = 1024,
) -> ShardMetadata:
    """Serialise a `ViewArrays` into TFRecord shards + metadata + index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # A rebuild with a different example count names its shards differently
    # (…-of-0000N), so files from the previous set would survive and poison
    # readers with a mixed schema — the 2026-07-12 expansion-run crash.
    # A shard set is all-or-nothing: clear before writing.
    for stale in out_dir.glob("views-*.tfrecord"):
        stale.unlink()
    n = len(views.labels)
    aux_dim = 0 if views.aux_features is None else int(views.aux_features.shape[1])
    n_shards = max(1, int(np.ceil(n / examples_per_shard)))

    for shard_idx in range(n_shards):
        lo = shard_idx * examples_per_shard
        hi = min(lo + examples_per_shard, n)
        shard_path = out_dir / f"views-{shard_idx:05d}-of-{n_shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for i in range(lo, hi):
                feature = {
                    "global_view": _float_feature(views.global_views[i]),
                    "local_view": _float_feature(views.local_views[i]),
                    "label": _int_feature(int(views.labels[i])),
                    "tic_id": _int_feature(int(views.tic_ids[i])),
                }
                if aux_dim:
                    assert views.aux_features is not None
                    feature["aux_features"] = _float_feature(views.aux_features[i])
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    metadata = ShardMetadata(
        n_examples=n,
        global_bins=int(views.global_views.shape[1]),
        local_bins=int(views.local_views.shape[1]),
        aux_dim=aux_dim,
        n_shards=n_shards,
    )
    (out_dir / METADATA_NAME).write_text(json.dumps(asdict(metadata), indent=2))

    index = pd.DataFrame(
        {
            "row": np.arange(n),
            "tic_id": views.tic_ids.astype(np.int64),
            "label": views.labels.astype(np.int8),
        }
    )
    if aux_dim:
        assert views.aux_features is not None
        for k in range(aux_dim):
            index[f"aux_{k}"] = views.aux_features[:, k]
    index.to_parquet(out_dir / INDEX_NAME, index=False)

    log.info(
        "[tfrecords] wrote %d examples (%d shards, aux_dim=%d) to %s",
        n,
        n_shards,
        aux_dim,
        out_dir,
    )
    return metadata


def list_shards(shard_dir: Path) -> list[str]:
    """Shard file paths in written (index-row) order."""
    return sorted(str(p) for p in shard_dir.glob("views-*.tfrecord"))


def load_index(shard_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(shard_dir / INDEX_NAME)


def make_parse_fn(
    metadata: ShardMetadata,
) -> Callable[[tf.Tensor], tuple[dict[str, tf.Tensor], tf.Tensor]]:
    """Build the parser for one serialised example → (features dict, label).

    tic_id rides along inside the features dict (as scalar "tic_id") so the
    split filter can route examples; `pipeline.make_dataset` strips it before
    batches reach the model.
    """
    spec = {
        "global_view": tf.io.FixedLenFeature([metadata.global_bins], tf.float32),
        "local_view": tf.io.FixedLenFeature([metadata.local_bins], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "tic_id": tf.io.FixedLenFeature([], tf.int64),
    }
    if metadata.aux_dim:
        spec["aux_features"] = tf.io.FixedLenFeature([metadata.aux_dim], tf.float32)

    def parse(serialized: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        ex = tf.io.parse_single_example(serialized, spec)
        features = {
            "global_view": ex["global_view"][:, None],  # add channel axis
            "local_view": ex["local_view"][:, None],
            "tic_id": ex["tic_id"],
        }
        if metadata.aux_dim:
            features["aux_features"] = ex["aux_features"]
        return features, tf.cast(ex["label"], tf.float32)

    return parse
