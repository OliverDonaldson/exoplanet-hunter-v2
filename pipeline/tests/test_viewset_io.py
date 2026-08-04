"""View-set interchange, shards and the view-set gate."""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.datasets.viewset_tfrecords import (
    load_index,
    load_metadata,
    make_parse_fn,
    write_viewset_shards,
)
from exoplanet_hunter.validation import check_view_set


def test_round_trips_through_disk(make_view_set, tmp_path):
    arrays = make_view_set()
    arrays.save(tmp_path)
    back = ViewSetArrays.load(tmp_path)
    assert back.validate() == []
    assert set(back.views) == set(VIEW_SHAPES)
    for name in VIEW_SHAPES:
        np.testing.assert_allclose(back.views[name], arrays.views[name])
    assert len(back.scalars) == len(arrays.scalars)


def test_validate_catches_a_view_of_the_wrong_shape(make_view_set):
    arrays = make_view_set()
    arrays.views["local_view"] = arrays.views["local_view"][:, :10]
    assert any("local_view" in p for p in arrays.validate())


def test_gate_passes_a_healthy_set(make_view_set):
    assert check_view_set(make_view_set()) == []


def test_gate_catches_a_dead_branch(make_view_set):
    # A branch all-zero for every row is the momentum-dump QUALITY bit, and the
    # 13-dim aux vector: present in the schema, carrying nothing.
    arrays = make_view_set()
    arrays.views["gap_view"][:] = 0.0
    assert any("dead branch" in p for p in check_view_set(arrays))


def test_gate_catches_an_unpopulated_presence_mask(make_view_set):
    arrays = make_view_set()
    arrays.views["centroid_view"][..., -1] = 1.0
    assert any("mask not populated" in p for p in check_view_set(arrays))


def test_gate_catches_nan_and_label_problems(make_view_set):
    arrays = make_view_set()
    arrays.views["global_view"][2, 5, 0] = np.nan
    arrays.scalars.loc[0, "label"] = 7
    problems = check_view_set(arrays)
    assert any("NaN" in p for p in problems)
    assert any("labels" in p for p in problems)


def test_shards_round_trip_through_tensorflow(make_view_set, tmp_path):
    import tensorflow as tf

    arrays = make_view_set(n=8)
    metadata = write_viewset_shards(arrays, tmp_path, examples_per_shard=3)
    assert metadata["n_examples"] == 8 and metadata["n_shards"] == 3
    assert load_metadata(tmp_path) == metadata
    assert len(load_index(tmp_path)) == 8

    parse = make_parse_fn(metadata)
    dataset = tf.data.TFRecordDataset(
        sorted(str(p) for p in tmp_path.glob("viewset-*.tfrecord"))
    ).map(parse)
    features, label = next(iter(dataset))
    for name, shape in VIEW_SHAPES.items():
        assert tuple(features[name].shape) == shape
    np.testing.assert_allclose(features["global_view"].numpy(), arrays.views["global_view"][0])
    assert float(label) == float(arrays.scalars["label"].iloc[0])
    assert features["scalars"].shape[0] == len(metadata["scalar_columns"])
    assert features["masks"].shape[0] == len(metadata["mask_columns"])


def test_rewriting_a_smaller_set_leaves_no_stale_shards(make_view_set, tmp_path):
    # Shards from a previous, larger build survive a rename and poison readers
    # with a mixed schema — the 2026-07-12 expansion-run crash.
    write_viewset_shards(make_view_set(n=8), tmp_path, examples_per_shard=3)
    write_viewset_shards(make_view_set(n=4), tmp_path, examples_per_shard=3)
    assert len(list(tmp_path.glob("viewset-*.tfrecord"))) == 2


def test_nan_scalars_are_zeroed_and_flagged_by_the_mask(make_view_set, tmp_path):
    arrays = make_view_set(n=4)
    arrays.scalars.loc[0, "ruwe"] = np.nan
    arrays.scalars.loc[0, "has_ruwe"] = False
    metadata = write_viewset_shards(arrays, tmp_path, examples_per_shard=4)

    import tensorflow as tf

    parse = make_parse_fn(metadata)
    features, _ = next(
        iter(tf.data.TFRecordDataset(str(next(tmp_path.glob("*.tfrecord")))).map(parse))
    )
    ruwe_idx = metadata["scalar_columns"].index("ruwe")
    mask_idx = metadata["mask_columns"].index("has_ruwe")
    # 0.0 is a value a dense layer can consume; the mask is what says it was
    # never measured.
    assert float(features["scalars"][ruwe_idx]) == pytest.approx(0.0)
    assert float(features["masks"][mask_idx]) == pytest.approx(0.0)


def test_absent_declared_scalars_are_reported_not_silently_dropped(tmp_path, make_view_set, caplog):
    # A merge that suffixes a column to _x/_y leaves the declared name absent.
    # Writing a shorter scalar vector and passing every gate is how the transit
    # counts — the whole point of the unfolded branch — went missing.
    arrays = make_view_set(n=4)
    arrays.scalars = arrays.scalars.rename(
        columns={"observed_transit_count": "observed_transit_count_x"}
    )
    with caplog.at_level("WARNING"):
        metadata = write_viewset_shards(arrays, tmp_path, examples_per_shard=4)
    assert "observed_transit_count" not in metadata["scalar_columns"]
    assert "observed_transit_count" in caplog.text
