"""tf.data input pipeline over the view-set shards.

Same stage order and split semantics as `pipeline.py` — membership by TIC ID
through a `StaticHashTable`, so a star lands in exactly one split, and
unshuffled streams keep shard order so predictions align with the index.

Normalisation constants are fitted on the **training split only** and applied
as fixed numbers, so a validation row can never influence the scale a training
row is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from exoplanet_hunter.datasets.pipeline import Split, make_split_table
from exoplanet_hunter.datasets.viewset_tfrecords import make_parse_fn

__all__ = [
    "ScalarConstants",
    "Split",
    "fit_scalar_constants",
    "make_split_table",
    "make_viewset_dataset",
]


@dataclass(frozen=True)
class ScalarConstants:
    """Per-column median and scale for the scalar feature vector."""

    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_arrays(cls, median: np.ndarray, scale: np.ndarray) -> ScalarConstants:
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(median=np.nan_to_num(median).astype(np.float32), scale=scale.astype(np.float32))


def fit_scalar_constants(index: pd.DataFrame, columns: list[str]) -> ScalarConstants:
    """Fit robust centre and scale from the training rows given.

    Median and MAD rather than mean and standard deviation: the DV scalars have
    heavy tails (bootstrap significance spans 1e-90 to 1) and one outlier would
    otherwise set the scale for the whole column.
    """
    if not columns:
        return ScalarConstants.from_arrays(np.zeros(0), np.ones(0))
    values = index[columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    return ScalarConstants.from_arrays(median, 1.4826 * mad)


def make_viewset_dataset(
    shard_files: list[str],
    metadata: dict,
    *,
    split_table: tf.lookup.StaticHashTable | None = None,
    split: Split | None = None,
    scalar_constants: ScalarConstants | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1024,
    cache: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build one split's (inputs_dict, label) stream from a view-set shard set."""
    if (split_table is None) != (split is None):
        raise ValueError("split_table and split must be passed together")

    ds = tf.data.TFRecordDataset(
        shard_files, num_parallel_reads=tf.data.AUTOTUNE if shuffle else None
    )
    ds = ds.map(make_parse_fn(metadata), num_parallel_calls=tf.data.AUTOTUNE)

    if split_table is not None:
        assert split is not None
        want = tf.constant(int(split), tf.int64)
        ds = ds.filter(lambda feats, label: split_table.lookup(feats["tic_id"]) == want)

    view_names = list(metadata["view_shapes"])
    has_scalars = bool(metadata["scalar_columns"])
    has_masks = bool(metadata["mask_columns"])
    if has_scalars and scalar_constants is not None:
        centre = tf.constant(scalar_constants.median, tf.float32)
        spread = tf.constant(scalar_constants.scale, tf.float32)
    else:
        centre = spread = None

    def finalize(feats: dict, label: tf.Tensor) -> tuple[dict, tf.Tensor]:
        inputs = {name: feats[name] for name in view_names}
        if has_scalars:
            scalars = feats["scalars"]
            if centre is not None:
                # Clipped after scaling: a 40-sigma DV statistic is real but
                # saturates the first dense layer if it arrives unbounded.
                scalars = tf.clip_by_value((scalars - centre) / spread, -10.0, 10.0)
            inputs["scalars"] = scalars
        if has_masks:
            inputs["masks"] = feats["masks"]
        return inputs, label

    ds = ds.map(finalize, num_parallel_calls=tf.data.AUTOTUNE)

    # ~100 MB for the full labelled set, measured 2026-08-01, so caching the
    # decoded stream is affordable and everything above runs once.
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
