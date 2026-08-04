"""Per-diagnostic branch model: wiring, presence gating, scoped scalars."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES
from exoplanet_hunter.datasets.viewset_tfrecords import FEATURE_COLUMNS, MASK_COLUMNS
from exoplanet_hunter.models.cnn_branches import BRANCH_SCALARS, build_cnn_branches

SCALARS = list(FEATURE_COLUMNS)
MASKS = list(MASK_COLUMNS)


@pytest.fixture(scope="module")
def model():
    return build_cnn_branches(SimpleNamespace(), scalar_columns=SCALARS, mask_columns=MASKS)


def make_batch(n: int = 4, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    batch = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(size=(n, *shape)).astype(np.float32)
        arr[..., -1] = (rng.random((n, *shape[:-1])) > 0.3).astype(np.float32)
        batch[name] = arr
    batch["scalars"] = rng.normal(size=(n, len(SCALARS))).astype(np.float32)
    batch["masks"] = np.ones((n, len(MASKS)), dtype=np.float32)
    return batch


def test_model_takes_every_view_plus_scalars_and_masks(model):
    names = {t.name.split(":")[0] for t in model.inputs}
    assert set(VIEW_SHAPES) <= names
    assert {"scalars", "masks"} <= names


def test_output_is_a_probability(model):
    out = model(make_batch(), training=False).numpy()
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


@pytest.mark.parametrize("view", ["centroid_view", "trend_view", "unfolded_view"])
def test_an_absent_branch_contributes_exactly_zero(model, view):
    # The whole point of the presence channel: a branch with nothing measured
    # must contribute nothing, not a learned bias on a zero tensor. Without
    # this, a mission that lacks a branch has every row poisoned.
    gate = model.get_layer(f"{view}_gate")
    probe = tf.keras.Model(inputs=model.inputs, outputs=gate.output)

    batch = make_batch()
    assert np.abs(probe(batch, training=False).numpy()).sum() > 0

    batch[view] = batch[view].copy()
    batch[view][..., -1] = 0.0
    assert np.abs(probe(batch, training=False).numpy()).sum() == pytest.approx(0.0)


def test_a_present_branch_is_not_gated_off(model):
    gate = model.get_layer("global_view_gate")
    probe = tf.keras.Model(inputs=model.inputs, outputs=gate.output)
    batch = make_batch()
    batch["global_view"][..., -1] = 1.0
    assert np.abs(probe(batch, training=False).numpy()).sum() > 0


def test_scoped_scalars_reach_their_own_branch(model):
    # A scalar concatenated into one global vector is the 13-dim aux null. It
    # has to move the branch it qualifies.
    name = "centroid_view"
    picked = BRANCH_SCALARS[name][0]
    probe = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(f"{name}_fc").output)

    batch = make_batch()
    before = probe(batch, training=False).numpy()
    batch["scalars"] = batch["scalars"].copy()
    batch["scalars"][:, SCALARS.index(picked)] += 25.0
    after = probe(batch, training=False).numpy()
    assert not np.allclose(before, after)


def test_an_unscoped_scalar_leaves_an_unrelated_branch_alone(model):
    # gap_view has no scoped scalars, so nothing in the scalar vector should
    # move it — that is what "scoped" means.
    probe = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("gap_view_fc").output)
    batch = make_batch()
    before = probe(batch, training=False).numpy()
    batch["scalars"] = batch["scalars"] + 25.0
    assert np.allclose(before, probe(batch, training=False).numpy())


def test_masks_change_the_prediction(model):
    batch = make_batch()
    with_data = model(batch, training=False).numpy()
    batch["masks"] = np.zeros_like(batch["masks"])
    assert not np.allclose(with_data, model(batch, training=False).numpy())


def test_every_branch_scalar_name_exists_in_the_feature_vector():
    # A typo here silently drops a scalar from its branch rather than failing.
    for names in BRANCH_SCALARS.values():
        for name in names:
            assert name in SCALARS, f"{name} is not a shard feature column"


def test_model_trains_one_step(model):
    model.compile(optimizer="adam", loss="binary_crossentropy")
    batch = make_batch(n=8)
    labels = np.array([0, 1] * 4, dtype=np.float32)
    before = model.evaluate(batch, labels, verbose=0)
    model.fit(batch, labels, epochs=3, verbose=0)
    assert model.evaluate(batch, labels, verbose=0) < before
