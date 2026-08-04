"""View-set interchange, shards and the view-set gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.datasets.viewset_tfrecords import (
    load_index,
    load_metadata,
    make_parse_fn,
    write_viewset_shards,
)
from exoplanet_hunter.validation import check_view_set


def make_arrays(n: int = 6, *, seed: int = 0) -> ViewSetArrays:
    rng = np.random.default_rng(seed)
    views = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(0.0, 1.0, size=(n, *shape)).astype(np.float32)
        # Presence channel is 0/1, and not all 1 — a mask stuck at 1 is the
        # failure the gate looks for.
        arr[..., -1] = (rng.random((n, *shape[:-1])) > 0.2).astype(np.float32)
        views[name] = arr
    scalars = pd.DataFrame(
        {
            "tic_id": np.arange(1, n + 1),
            "mission": ["TESS"] * n,
            "label": [1, 0] * (n // 2),
            "observed_transit_count": rng.integers(1, 20, n),
            "expected_transit_count": rng.integers(20, 40, n),
            "transit_completeness": rng.random(n),
            "secondary_phase": rng.random(n) - 0.5,
            "ruwe": rng.normal(1.0, 0.2, n),
            "dv_usable": [True] * (n - 1) + [False],
            "has_ruwe": [True] * n,
        }
    )
    return ViewSetArrays(views=views, scalars=scalars)


def test_round_trips_through_disk(tmp_path):
    arrays = make_arrays()
    arrays.save(tmp_path)
    back = ViewSetArrays.load(tmp_path)
    assert back.validate() == []
    assert set(back.views) == set(VIEW_SHAPES)
    for name in VIEW_SHAPES:
        np.testing.assert_allclose(back.views[name], arrays.views[name])
    assert len(back.scalars) == len(arrays.scalars)


def test_validate_catches_a_view_of_the_wrong_shape():
    arrays = make_arrays()
    arrays.views["local_view"] = arrays.views["local_view"][:, :10]
    assert any("local_view" in p for p in arrays.validate())


def test_gate_passes_a_healthy_set():
    assert check_view_set(make_arrays()) == []


def test_gate_catches_a_dead_branch():
    # A branch all-zero for every row is the momentum-dump QUALITY bit, and the
    # 13-dim aux vector: present in the schema, carrying nothing.
    arrays = make_arrays()
    arrays.views["gap_view"][:] = 0.0
    assert any("dead branch" in p for p in check_view_set(arrays))


def test_gate_catches_an_unpopulated_presence_mask():
    arrays = make_arrays()
    arrays.views["centroid_view"][..., -1] = 1.0
    assert any("mask not populated" in p for p in check_view_set(arrays))


def test_gate_catches_nan_and_label_problems():
    arrays = make_arrays()
    arrays.views["global_view"][2, 5, 0] = np.nan
    arrays.scalars.loc[0, "label"] = 7
    problems = check_view_set(arrays)
    assert any("NaN" in p for p in problems)
    assert any("labels" in p for p in problems)


def test_shards_round_trip_through_tensorflow(tmp_path):
    import tensorflow as tf

    arrays = make_arrays(n=8)
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


def test_rewriting_a_smaller_set_leaves_no_stale_shards(tmp_path):
    # Shards from a previous, larger build survive a rename and poison readers
    # with a mixed schema — the 2026-07-12 expansion-run crash.
    write_viewset_shards(make_arrays(n=8), tmp_path, examples_per_shard=3)
    write_viewset_shards(make_arrays(n=4), tmp_path, examples_per_shard=3)
    assert len(list(tmp_path.glob("viewset-*.tfrecord"))) == 2


def test_nan_scalars_are_zeroed_and_flagged_by_the_mask(tmp_path):
    arrays = make_arrays(n=4)
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
