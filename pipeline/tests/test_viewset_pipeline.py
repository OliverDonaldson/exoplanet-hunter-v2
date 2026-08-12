"""View-set tf.data pipeline: splits, scalar normalisation, ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES
from exoplanet_hunter.datasets.viewset_pipeline import (
    AugmentConfig,
    ScalarConstants,
    Split,
    fit_scalar_constants,
    make_split_table,
    make_viewset_dataset,
    make_weight_table,
)
from exoplanet_hunter.datasets.viewset_tfrecords import (
    list_shards,
    load_index,
    write_viewset_shards,
)


@pytest.fixture
def shards(tmp_path, make_view_set):
    arrays = make_view_set(n=12)
    metadata = write_viewset_shards(arrays, tmp_path, examples_per_shard=5)
    return tmp_path, metadata


def test_stream_yields_every_view_and_the_label(shards):
    path, metadata = shards
    ds = make_viewset_dataset(list_shards(path), metadata, batch_size=4)
    inputs, labels = next(iter(ds))
    for name, shape in VIEW_SHAPES.items():
        assert tuple(inputs[name].shape) == (4, *shape)
    assert tuple(labels.shape) == (4,)


def test_split_filter_keeps_a_star_in_exactly_one_split(shards):
    path, metadata = shards
    index = load_index(path)
    codes = np.array([Split.TRAIN, Split.VAL, Split.TEST] * 4, dtype=np.int64)
    table = make_split_table(index["tic_id"].to_numpy(), codes)

    seen: dict[int, list[int]] = {}
    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        ds = make_viewset_dataset(
            list_shards(path), metadata, split_table=table, split=split, batch_size=16
        )
        for _, labels in ds:
            seen.setdefault(int(split), []).append(len(labels))
    assert sum(sum(v) for v in seen.values()) == len(index)


def test_unknown_tics_are_dropped(shards):
    path, metadata = shards
    table = make_split_table(np.array([99999], dtype=np.int64), np.array([Split.TRAIN], np.int64))
    ds = make_viewset_dataset(
        list_shards(path), metadata, split_table=table, split=Split.TRAIN, batch_size=4
    )
    assert sum(len(labels) for _, labels in ds) == 0


def test_scalar_constants_are_robust_to_an_outlier():
    # DV statistics have heavy tails; a mean/std fit would let one row set the
    # scale for the whole column.
    index = pd.DataFrame({"ruwe": [1.0, 1.1, 0.9, 1.05, 1e6]})
    constants = fit_scalar_constants(index, ["ruwe"])
    assert constants.median[0] == pytest.approx(1.05, abs=0.05)
    assert constants.scale[0] < 1.0


def test_zero_spread_column_does_not_divide_by_zero():
    index = pd.DataFrame({"ruwe": [1.0, 1.0, 1.0]})
    constants = fit_scalar_constants(index, ["ruwe"])
    assert constants.scale[0] == pytest.approx(1.0)


def test_normalisation_centres_and_clips(shards):
    path, metadata = shards
    columns = metadata["scalar_columns"]
    constants = ScalarConstants.from_arrays(np.zeros(len(columns)), np.full(len(columns), 1e-6))
    ds = make_viewset_dataset(
        list_shards(path), metadata, scalar_constants=constants, batch_size=12
    )
    inputs, _ = next(iter(ds))
    # A tiny scale would blow the values up without the clip.
    assert float(tf.reduce_max(tf.abs(inputs["scalars"]))) <= 10.0


def test_unshuffled_order_matches_the_index(shards):
    # Predictions are aligned to the index positionally, so order is a contract.
    path, metadata = shards
    ds = make_viewset_dataset(list_shards(path), metadata, batch_size=12, shuffle=False)
    _, labels = next(iter(ds))
    np.testing.assert_array_equal(
        labels.numpy().astype(int), load_index(path)["label"].to_numpy()[:12]
    )


def test_split_table_and_split_must_be_passed_together(shards):
    path, metadata = shards
    with pytest.raises(ValueError):
        make_viewset_dataset(list_shards(path), metadata, split=Split.TRAIN)


# ------------------------------------- stage 8: per-example training weights --


def test_a_weight_table_puts_a_sample_weight_in_the_third_slot(shards):
    """`fit` reads a three-tuple as (inputs, label, sample_weight)."""
    path, metadata = shards
    tics = load_index(path)["tic_id"].to_numpy()
    table = make_weight_table(tics, np.full(len(tics), 2.5))
    ds = make_viewset_dataset(list_shards(path), metadata, batch_size=4, weight_table=table)
    batch = next(iter(ds))
    assert len(batch) == 3
    assert np.allclose(batch[2].numpy(), 2.5)


def test_a_tic_the_caller_forgot_to_weight_trains_at_one_not_zero():
    """A default of 0.0 removes the example from the loss while every batch
    still looks the right size — invisible, and the population is no longer the
    one the arm describes."""
    table = make_weight_table(np.array([1, 2]), np.array([3.0, 4.0]))
    assert float(table.lookup(tf.constant(999, tf.int64)).numpy()) == 1.0


def test_a_weight_table_refuses_a_nan(shards):
    with pytest.raises(ValueError, match="non-finite weight"):
        make_weight_table(np.array([1, 2]), np.array([1.0, np.nan]))


def test_a_weight_table_refuses_a_negative_weight():
    """It does not down-weight the example, it inverts its gradient."""
    with pytest.raises(ValueError, match="inverts the gradient"):
        make_weight_table(np.array([1, 2]), np.array([1.0, -1.0]))


def test_a_weight_table_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="tic_ids but"):
        make_weight_table(np.array([1, 2, 3]), np.array([1.0, 1.0]))


def test_weights_and_tic_ids_cannot_both_claim_the_third_slot(shards):
    """Passing both would hand fit() a tensor of TIC IDs as sample weights,
    which trains happily and is nonsense."""
    path, metadata = shards
    tics = load_index(path)["tic_id"].to_numpy()
    with pytest.raises(ValueError, match="third element"):
        make_viewset_dataset(
            list_shards(path),
            metadata,
            with_tic_id=True,
            weight_table=make_weight_table(tics, np.ones(len(tics))),
        )


def test_augmentation_does_not_drop_the_weight(shards):
    """An augmented weighted stream that silently reverted to a two-tuple would
    train unweighted while the arm was recorded as weighted — the exact failure
    the intervention exists to avoid."""
    path, metadata = shards
    tics = load_index(path)["tic_id"].to_numpy()
    ds = make_viewset_dataset(
        list_shards(path),
        metadata,
        batch_size=4,
        augment=AugmentConfig(),
        weight_table=make_weight_table(tics, np.full(len(tics), 3.0)),
    )
    batch = next(iter(ds))
    assert len(batch) == 3
    assert np.allclose(batch[2].numpy(), 3.0)


def test_weights_are_looked_up_per_row_not_broadcast(shards):
    """Every row weighted the same would pass the shape checks above while
    carrying none of the intervention."""
    path, metadata = shards
    index = load_index(path)
    tics = index["tic_id"].to_numpy()
    per_tic = {int(t): 1.0 + i for i, t in enumerate(np.unique(tics))}
    table = make_weight_table(
        np.array(list(per_tic)), np.array(list(per_tic.values()), dtype=float)
    )
    ds = make_viewset_dataset(
        list_shards(path), metadata, batch_size=len(index), weight_table=table
    )
    _, _, weights = next(iter(ds))
    assert len(np.unique(weights.numpy())) > 1
