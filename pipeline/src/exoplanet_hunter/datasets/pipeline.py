"""tf.data input pipeline over TFRecord shards (L6 backbone).

Stage order, and why:

    TFRecordDataset(shards, parallel reads)   sequential I/O, interleaved
      -> map(parse)                           deterministic
      -> filter(tic_id in split)              deterministic, via StaticHashTable
      -> map(aux normalise)                   deterministic (fitted constants)
      -> cache()                              everything above runs ONCE
      -> shuffle                              train only
      -> map(augment)                         stochastic — fresh every epoch
      -> batch -> prefetch(AUTOTUNE)          keep the GPU fed

Split membership is by TIC ID, not row index: a `StaticHashTable` maps
tic_id -> split code, so one pass over the shards yields any fold's
train/val/test streams while preserving the leakage guarantee (a star is in
exactly one split). Unshuffled datasets preserve shard order == index-row
order, so `predict()` output aligns positionally with index-derived labels.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import tensorflow as tf

from exoplanet_hunter.datasets.augment import AugmentConfig, augment_views
from exoplanet_hunter.datasets.aux_transform import AuxConstants, tf_aux_transform
from exoplanet_hunter.datasets.tfrecords import ShardMetadata, make_parse_fn


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2


def make_split_table(
    tic_ids: np.ndarray,
    split_codes: np.ndarray,
) -> tf.lookup.StaticHashTable:
    """tic_id -> Split code lookup; unknown TICs map to -1 (dropped)."""
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tic_ids.astype(np.int64)),
            values=tf.constant(split_codes.astype(np.int64)),
        ),
        default_value=tf.constant(-1, tf.int64),
    )


def make_dataset(
    shard_files: list[str],
    metadata: ShardMetadata,
    *,
    split_table: tf.lookup.StaticHashTable | None = None,
    split: Split | None = None,
    aux_constants: AuxConstants | None = None,
    use_aux: bool = False,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1024,
    augment: AugmentConfig | None = None,
    cache: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build one split's (inputs_dict, label) stream from a shard set."""
    if use_aux and metadata.aux_dim == 0:
        raise ValueError("use_aux=True but the shard set has no aux features")
    if use_aux and aux_constants is None:
        raise ValueError("use_aux=True requires fitted aux_constants")
    if (split_table is None) != (split is None):
        raise ValueError("split_table and split must be passed together")

    # Parallel shard reads interleave blocks across files — fine (and faster)
    # when the stream is about to be shuffled, but it breaks the positional
    # alignment eval relies on. Unshuffled streams read sequentially.
    ds = tf.data.TFRecordDataset(
        shard_files,
        num_parallel_reads=tf.data.AUTOTUNE if shuffle else None,
    )
    ds = ds.map(make_parse_fn(metadata), num_parallel_calls=tf.data.AUTOTUNE)

    if split_table is not None:
        assert split is not None  # enforced by the pairing check above
        want = tf.constant(int(split), tf.int64)
        ds = ds.filter(lambda feats, label: split_table.lookup(feats["tic_id"]) == want)

    def finalize(feats: dict, label: tf.Tensor) -> tuple[dict, tf.Tensor]:
        inputs = {"global_view": feats["global_view"], "local_view": feats["local_view"]}
        if use_aux:
            assert aux_constants is not None
            inputs["aux_features"] = tf_aux_transform(feats["aux_features"], aux_constants)
        return inputs, label

    ds = ds.map(finalize, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    if augment is not None:

        def _augment(inputs: dict, label: tf.Tensor) -> tuple[dict, tf.Tensor]:
            g, l = augment_views(inputs["global_view"], inputs["local_view"], augment)
            out = dict(inputs, global_view=g, local_view=l)
            return out, label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
