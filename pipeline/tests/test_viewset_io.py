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
    assert features["scalars"].shape[0] == len(metadata["scalar_columns"])
    assert features["masks"].shape[0] == len(metadata["mask_columns"])

    # Rows are permuted at write time to break the mission blocking, and the
    # index is permuted with them. That the stream's row k is the index's row k
    # is the invariant every prediction alignment depends on.
    source = int(
        arrays.scalars.index[arrays.scalars["tic_id"] == load_index(tmp_path)["tic_id"].iloc[0]][0]
    )
    np.testing.assert_allclose(features["global_view"].numpy(), arrays.views["global_view"][source])
    assert float(label) == float(arrays.scalars["label"].iloc[source])


def test_rewriting_a_smaller_set_leaves_no_stale_shards(make_view_set, tmp_path):
    # Shards from a previous, larger build survive a rename and poison readers
    # with a mixed schema — the 2026-07-12 expansion-run crash.
    write_viewset_shards(make_view_set(n=8), tmp_path, examples_per_shard=3)
    write_viewset_shards(make_view_set(n=4), tmp_path, examples_per_shard=3)
    assert len(list(tmp_path.glob("viewset-*.tfrecord"))) == 2


def test_nan_scalars_are_written_through_rather_than_filled(make_view_set, tmp_path):
    """Filling at write time puts "never measured" at a real percentile of the
    column once normalisation runs — a missing `odd_even_statistic` landed at
    z=-0.679 against a real 5th percentile of -0.68, making the fill value a
    near-perfect mission indicator. The reader imputes to the fitted centre."""
    arrays = make_view_set(n=4)
    arrays.scalars.loc[0, "ruwe"] = np.nan
    arrays.scalars.loc[0, "has_ruwe"] = False
    unmeasured = int(arrays.scalars["tic_id"].iloc[0])
    metadata = write_viewset_shards(arrays, tmp_path, examples_per_shard=4)
    assert metadata["scalar_encoding"] == "nan"

    import tensorflow as tf

    parse = make_parse_fn(metadata)
    rows = list(tf.data.TFRecordDataset(str(next(tmp_path.glob("*.tfrecord")))).map(parse))
    position = int(load_index(tmp_path).index[load_index(tmp_path)["tic_id"] == unmeasured][0])
    features, _ = rows[position]

    ruwe_idx = metadata["scalar_columns"].index("ruwe")
    mask_idx = metadata["mask_columns"].index("has_ruwe")
    assert np.isnan(float(features["scalars"][ruwe_idx]))
    assert float(features["masks"][mask_idx]) == pytest.approx(0.0)


def test_absent_declared_scalars_raise_rather_than_writing_a_shorter_vector(
    tmp_path, make_view_set
):
    # A merge that suffixes a column to _x/_y leaves the declared name absent.
    # Writing a shorter scalar vector and passing every gate is how the transit
    # counts — the whole point of the unfolded branch — went missing. This
    # warned rather than raised until 2026-08-07.
    arrays = make_view_set(n=4)
    arrays.scalars = arrays.scalars.rename(
        columns={"observed_transit_count": "observed_transit_count_x"}
    )
    with pytest.raises(ValueError, match="observed_transit_count"):
        write_viewset_shards(arrays, tmp_path, examples_per_shard=4)
