"""Appending a view set — the guards that keep a merged set trainable.

Two failures matter here and both are silent. A `tic_id` collision makes two
rows share a split code, because the split and weight tables key on it. A
missing `group_tic` puts a synthetic negative and the star it was derived from
in different folds, so the model is tested on a light curve it trained on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _script():
    spec = importlib.util.spec_from_file_location(
        "_shard_viewset",
        Path(__file__).resolve().parents[1] / "scripts" / "shard_viewset.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shard_viewset = _script()


def test_appending_concatenates_rows_and_views(make_view_set):
    base = make_view_set(n=6)
    extra = make_view_set(n=4)
    extra.scalars["tic_id"] = -np.arange(1, 5)
    merged = shard_viewset.append(base, extra)
    assert len(merged.scalars) == 10
    for name, array in merged.views.items():
        assert array.shape[0] == 10, name
    assert merged.validate() == []


def test_every_row_ends_up_with_a_group_key(make_view_set):
    """A real row is its own group; a synthetic row keeps its parent's. Left
    partly missing, the grouped split silently stops grouping."""
    base = make_view_set(n=6)
    extra = make_view_set(n=4)
    extra.scalars["tic_id"] = -np.arange(1, 5)
    extra.scalars["group_tic"] = base.scalars["tic_id"].to_numpy()[:4]
    merged = shard_viewset.append(base, extra)
    assert merged.scalars["group_tic"].notna().all()
    # The four appended rows share groups with real rows, so the group count is
    # below the row count — which is the whole point.
    assert merged.scalars["group_tic"].nunique() < len(merged.scalars)


def test_a_real_row_without_a_group_key_becomes_its_own_group(make_view_set):
    base = make_view_set(n=6)
    extra = make_view_set(n=2)
    extra.scalars["tic_id"] = [-1, -2]
    merged = shard_viewset.append(base, extra)
    real = merged.scalars[merged.scalars["tic_id"] > 0]
    assert (real["group_tic"] == real["tic_id"]).all()


def test_a_tic_id_collision_raises_rather_than_merging(make_view_set):
    """Two rows sharing a tic_id share a split code, because that is what the
    table keys on — one would follow the other into training or test."""
    base = make_view_set(n=6)
    extra = make_view_set(n=4)  # same synthetic tic_ids as base
    with pytest.raises(ValueError, match="appear in both view sets"):
        shard_viewset.append(base, extra)


def test_a_view_set_missing_a_declared_view_raises(make_view_set):
    base = make_view_set(n=4)
    extra = make_view_set(n=2)
    extra.scalars["tic_id"] = [-1, -2]
    extra.views.pop(next(iter(extra.views)))
    with pytest.raises(KeyError, match="every declared view"):
        shard_viewset.append(base, extra)
