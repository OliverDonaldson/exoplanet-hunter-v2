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
#: How an unmeasured scalar is written. Shards built before 2026-08-07 stored
#: `0.0`, which normalisation then mapped to a z-value sitting inside the real
#: distribution — a missing `odd_even_statistic` landed at −0.679 against a real
#: 5th percentile of −0.68, so "never measured" was indistinguishable from "weak
#: detection". Since every Kepler and K2 row is missing and only 12.8% of TESS
#: is, that made the fill value a near-perfect mission indicator, read through
#: the lane `MASK_COLUMNS` exists to own. NaN is now written through and imputed
#: to the fitted centre at normalisation time, where it means exactly nothing.
SCALAR_ENCODING = "nan"
#: Columns stored as log10, because they are not representable otherwise.
#: `bootstrap_significance` is a false-alarm probability reaching 1e-146, and a
#: TFRecord float list is float32 — 1,639 of its 2,235 measured values underflow
#: to exactly 0.0 below float32's smallest normal (1.18e-38). Transforming at
#: read time cannot recover them: by then the information is already gone, and
#: the constants fitted on the float64 index describe values the model never
#: sees. A non-positive entry is not a measurement on this scale — a negative is
#: DV's -1.0 sentinel — so it is written as NaN and imputed like any other gap.
LOG_SCALED_COLUMNS = frozenset({"bootstrap_significance"})


def _float_feature(values: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values.ravel().tolist()))


def _int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_viewset_shards(
    arrays: ViewSetArrays, out_dir: Path, *, examples_per_shard: int = 512, shuffle_seed: int = 42
) -> dict:
    """Serialise a `ViewSetArrays` into shards + metadata + index.

    Rows are permuted once, deterministically, before sharding. The catalogue
    arrives mission-blocked — TESS, then Kepler, then K2 — and a `tf.data`
    shuffle buffer of 1,024 over a ~3,470-row split never spans that, so every
    batch held one mission. With a `BatchNormalization` in all 11 conv towers
    that means batch statistics computed on a single mission, and an epoch that
    walks the missions in order as an unintended curriculum. The index is
    permuted with the views, so shard order still matches it row for row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # A rebuild with a different example count names its shards differently, so
    # files from the previous set would survive and poison readers with a mixed
    # schema. A shard set is all-or-nothing.
    for stale in out_dir.glob("viewset-*.tfrecord"):
        stale.unlink()

    order = np.random.default_rng(shuffle_seed).permutation(len(arrays.scalars))
    scalars = arrays.scalars.iloc[order].reset_index(drop=True)
    views = {name: array[order] for name, array in arrays.views.items()}
    n = len(scalars)
    n_shards = max(1, int(np.ceil(n / examples_per_shard)))
    features = [c for c in FEATURE_COLUMNS if c in scalars.columns]
    masks = [c for c in MASK_COLUMNS if c in scalars.columns]
    # Loud, not silent. A declared feature column that is absent — because a
    # merge suffixed it to _x/_y, say — otherwise writes a shorter scalar vector
    # and every gate still passes. That is how the transit counts went missing.
    absent = [c for c in (*FEATURE_COLUMNS, *MASK_COLUMNS) if c not in scalars.columns]
    if absent:
        raise ValueError(
            f"{len(absent)} declared scalars absent from the index and so not written: {absent}. "
            "A shorter scalar vector still passes every downstream gate — this warned rather "
            "than raised until 2026-08-07, which is how the transit counts went missing."
        )
    # Hoisted out of the per-example loop: pandas scalar access per row is the
    # slowest part of writing 5,700 examples.
    labels = scalars["label"].to_numpy(dtype=np.int64)
    tic_ids = scalars["tic_id"].to_numpy(dtype=np.int64)
    # NaN is written through rather than filled here — see SCALAR_ENCODING. The
    # reader imputes it to the fold's own fitted centre, which is the only value
    # that carries no information; any constant chosen at this layer is a value
    # inside the real distribution once normalisation runs.
    if features:
        wide = scalars[features].astype(np.float64)
        for column in features:
            if column in LOG_SCALED_COLUMNS:
                values = wide[column].to_numpy()
                measured = values > 0.0
                scaled = np.full(len(values), np.nan)
                scaled[measured] = np.log10(values[measured])
                wide[column] = scaled
        # The index is written from the same frame, so whatever the model reads
        # is what the normalisation constants are fitted on.
        scalars = scalars.assign(**{c: wide[c] for c in features if c in LOG_SCALED_COLUMNS})
        feature_values = wide.to_numpy(dtype=np.float32)
    else:
        feature_values = np.empty((n, 0), dtype=np.float32)
    mask_values = (
        scalars[masks].to_numpy(dtype=np.float32) if masks else np.empty((n, 0), dtype=np.float32)
    )

    for shard_idx in range(n_shards):
        lo = shard_idx * examples_per_shard
        hi = min(lo + examples_per_shard, n)
        path = out_dir / f"viewset-{shard_idx:05d}-of-{n_shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for i in range(lo, hi):
                feature = {name: _float_feature(views[name][i]) for name in VIEW_SHAPES}
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
        "scalar_encoding": SCALAR_ENCODING,
        "log_scaled_columns": [c for c in features if c in LOG_SCALED_COLUMNS],
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
