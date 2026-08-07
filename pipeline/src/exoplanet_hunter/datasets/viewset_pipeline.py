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
from exoplanet_hunter.datasets.viewset_augment import AugmentConfig, augment_viewset
from exoplanet_hunter.datasets.viewset_tfrecords import SCALAR_ENCODING, make_parse_fn
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "AugmentConfig",
    "ScalarConstants",
    "Split",
    "fit_scalar_constants",
    "make_split_table",
    "make_viewset_dataset",
    "parse_viewset_shards",
]


@dataclass(frozen=True)
class ScalarConstants:
    """Per-column median and scale for the scalar feature vector."""

    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_arrays(cls, median: np.ndarray, scale: np.ndarray) -> ScalarConstants:
        degenerate = ~(np.isfinite(scale) & (scale > 1e-12))
        if degenerate.any():
            # Substituting 1.0 turns a near-constant column into a near-constant
            # input, which is a dead lane rather than an error. Say so.
            log.warning(
                "[viewset-pipeline] %d column(s) had a degenerate spread and were scaled by 1.0 "
                "— indices %s. A column that reaches here contributes almost nothing.",
                int(degenerate.sum()),
                np.flatnonzero(degenerate).tolist(),
            )
        return cls(
            median=np.nan_to_num(median).astype(np.float32),
            scale=np.where(degenerate, 1.0, scale).astype(np.float32),
        )


def fit_scalar_constants(index: pd.DataFrame, columns: list[str]) -> ScalarConstants:
    """Fit robust centre and scale from the training rows given.

    Median and MAD rather than mean and standard deviation: the DV scalars have
    heavy tails and one outlier would otherwise set the scale for the whole
    column. Columns needing a log scale are already stored that way — the shard
    writer applies it, because a float32 record cannot hold the raw values (see
    `LOG_SCALED_COLUMNS`), so the index this reads and the shards the model
    reads carry the same numbers.
    """
    if not columns:
        return ScalarConstants.from_arrays(np.zeros(0), np.ones(0))
    values = index[columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    return ScalarConstants.from_arrays(median, 1.4826 * mad)


def parse_viewset_shards(shard_files: list[str], metadata: dict) -> tf.data.Dataset:
    """Decode every shard once, cached, for reuse across folds and splits.

    Only normalisation is per-fold, and that happens downstream — so the parsed
    stream is shareable. It was being rebuilt per call instead: `run_fold` opens
    four streams (train, the validation stream inside `fit`, validation again to
    calibrate, then test) and five folds made 20 full decodes of all 11 shards,
    ~13 GB of redundant parse work, plus one live cache per stream.

    Reads are deterministic, not AUTOTUNE-parallel: the cache fixes whatever
    order it first materialises, and unshuffled order matching the index is what
    prediction alignment rests on.
    """
    return (
        tf.data.TFRecordDataset(shard_files)
        .map(make_parse_fn(metadata), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
    )


def make_viewset_dataset(
    shard_files: list[str],
    metadata: dict,
    *,
    base: tf.data.Dataset | None = None,
    split_table: tf.lookup.StaticHashTable | None = None,
    split: Split | None = None,
    scalar_constants: ScalarConstants | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 4096,
    augment: AugmentConfig | None = None,
    seed: int = 42,
    with_tic_id: bool = False,
) -> tf.data.Dataset:
    """Build one split's (inputs_dict, label) stream from a view-set shard set.

    Pass `base` — a `parse_viewset_shards` result — to share one decode across
    every fold and split; without it each call decodes the shard set itself.

    `augment` applies only to the training split and runs after the cache, so
    every epoch draws fresh.

    `with_tic_id` yields a third element for prediction streams, so alignment
    can be asserted against the identity of the row rather than its label. It
    is not for `fit`, which takes the two-tuple.
    """
    if with_tic_id and augment is not None:
        raise ValueError("with_tic_id is for prediction streams; augmentation is training-only")
    if (split_table is None) != (split is None):
        raise ValueError("split_table and split must be passed together")
    if metadata["scalar_columns"] and metadata.get("scalar_encoding") != SCALAR_ENCODING:
        raise ValueError(
            f"this shard set encodes unmeasured scalars as "
            f"{metadata.get('scalar_encoding', '0.0 (pre-2026-08-07)')!r}, which normalisation "
            "maps onto a real percentile and turns into a mission indicator — rebuild it with "
            "`python pipeline/scripts/build_viewset.py` before training"
        )

    ds = base if base is not None else parse_viewset_shards(shard_files, metadata)

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

    def finalize(
        feats: dict, label: tf.Tensor
    ) -> tuple[dict, tf.Tensor] | tuple[dict, tf.Tensor, tf.Tensor]:
        inputs = {name: feats[name] for name in view_names}
        if has_scalars:
            scalars = feats["scalars"]
            if centre is not None:
                # Impute before scaling, so an unmeasured scalar lands at exactly
                # z=0 and says nothing. Filling with any constant at write time
                # instead puts it at a real percentile of the column, which the
                # mask is supposed to be the only signal for.
                scalars = tf.where(tf.math.is_nan(scalars), centre, scalars)
                # Clipped after scaling: a 40-sigma DV statistic is real but
                # saturates the first dense layer if it arrives unbounded.
                scalars = tf.clip_by_value((scalars - centre) / spread, -10.0, 10.0)
            inputs["scalars"] = scalars
        if has_masks:
            inputs["masks"] = feats["masks"]
        if with_tic_id:
            return inputs, label, feats["tic_id"]
        return inputs, label

    # The cache sits on the parsed stream in `parse_viewset_shards`, upstream of
    # this, so the decode is shared and only the per-fold normalisation reruns
    # each epoch — a handful of tensor ops against ~670 MB of parsing.
    ds = ds.map(finalize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        ds = ds.map(
            lambda inputs, label: (augment_viewset(inputs, augment), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
