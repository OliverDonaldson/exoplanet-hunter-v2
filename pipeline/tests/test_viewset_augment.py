"""Augmentation must never turn an absent branch into a present one.

`_gated` in `cnn_branches.py` reads `reduce_max(present) > 0`. Noise on a
genuinely-absent branch's zeros flips that predicate and disables the gating the
whole design rests on — for Kepler and K2, where `dv_usable` is 0%, on 3,027 of
5,423 rows.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_augment import VIEW_KINDS, AugmentConfig, augment_viewset
from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES

LOUD = AugmentConfig(time_shift_frac=0.2, noise_std=0.5, scale_range=0.5, mask_prob=0.5)


def absent(shape):
    """A branch with nothing measured: zeros everywhere, presence included."""
    return tf.zeros(shape, tf.float32)


def present_view(shape, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(*shape[:-1], shape[-1] - 1)).astype(np.float32)
    return tf.constant(np.concatenate([data, np.ones((*shape[:-1], 1), np.float32)], axis=-1))


def gate_is_open(view: tf.Tensor) -> bool:
    """The predicate `_gated` computes, on one unbatched view."""
    return bool(tf.reduce_max(view[..., -1]) > 0.0)


# ------------------------------------------------- the gating invariant --


@pytest.mark.parametrize("name", sorted(VIEW_SHAPES))
def test_an_all_absent_branch_stays_gated_off(name):
    shape = VIEW_SHAPES[name]
    tf.random.set_seed(0)
    for _ in range(20):
        out = augment_viewset({name: absent(shape)}, LOUD)[name]
        assert not gate_is_open(out), f"{name} was flipped to present"
        assert np.all(out.numpy() == 0.0), f"{name} invented data in an unmeasured bin"


def test_a_present_branch_stays_gated_on():
    tf.random.set_seed(0)
    out = augment_viewset({"local_view": present_view((31, 3))}, LOUD)["local_view"]
    assert gate_is_open(out)


def test_presence_survives_masking_that_zeros_every_data_bin():
    """mask_prob=1.0 zeros all data; the bins were still measured."""
    view = present_view((31, 3))
    out = augment_viewset({"local_view": view}, AugmentConfig(0.0, 0.0, 0.0, 1.0))["local_view"]
    assert np.all(out.numpy()[..., :-1] == 0.0)
    assert np.all(out.numpy()[..., -1] == 1.0)
    assert gate_is_open(out)


def test_a_partially_measured_view_keeps_its_gaps_empty():
    """Only bins that held a cadence may carry a value after augmentation."""
    data = np.ones((31, 2), np.float32)
    present = np.zeros((31, 1), np.float32)
    present[:10] = 1.0
    data[10:] = 0.0
    view = tf.constant(np.concatenate([data, present], axis=-1))

    tf.random.set_seed(0)
    out = augment_viewset({"local_view": view}, AugmentConfig(0.0, 0.5, 0.0, 0.0))[
        "local_view"
    ].numpy()

    assert np.all(out[10:, :-1] == 0.0)
    assert np.array_equal(out[:, -1], present.ravel())


# ------------------------------------------------------- per-view semantics --


def test_every_declared_view_has_a_kind():
    """A view added to `VIEW_SHAPES` must say what its bin axis means rather
    than being augmented as if it were folded flux."""
    assert set(VIEW_SHAPES) == set(VIEW_KINDS)


def test_an_unknown_view_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="VIEW_KINDS"):
        augment_viewset({"mystery_view": present_view((31, 3))}, LOUD)


def test_the_phase_shift_is_coherent_across_views():
    """One star, one moment: the same shift moves every phase-indexed view."""
    marker = np.zeros((31, 3), np.float32)
    marker[5] = [1.0, 1.0, 1.0]
    views = {n: tf.constant(marker) for n in ("local_view", "odd_view", "secondary_view")}
    shifted = augment_viewset(views, AugmentConfig(0.4, 0.0, 0.0, 0.0))
    peaks = {int(np.argmax(v.numpy()[:, 0])) for v in shifted.values()}
    assert len(peaks) == 1


def test_the_periodogram_is_not_phase_shifted():
    """Its axis is period, so rolling moves the peak onto a different period."""
    marker = np.zeros((256, 2), np.float32)
    marker[100] = [1.0, 1.0]
    out = augment_viewset(
        {"periodogram_view": tf.constant(marker)}, AugmentConfig(0.4, 0.0, 0.0, 0.0)
    )["periodogram_view"]
    assert int(np.argmax(out.numpy()[:, 0])) == 100


def test_the_unfolded_stack_shifts_phase_not_transit_order():
    stack = np.zeros((20, 31, 3), np.float32)
    stack[3, 5] = [1.0, 1.0, 1.0]
    out = augment_viewset({"unfolded_view": tf.constant(stack)}, AugmentConfig(0.4, 0.0, 0.0, 0.0))[
        "unfolded_view"
    ].numpy()
    transit, phase = np.unravel_index(np.argmax(out[..., 0]), out[..., 0].shape)
    assert transit == 3
    assert phase != 5


def test_depth_scaling_reaches_flux_but_not_a_gap_fraction():
    """A gap fraction is a fraction, not a depth: it does not move when the
    transit gets deeper."""
    flux = np.concatenate([np.full((31, 2), 1.0), np.ones((31, 1))], axis=-1).astype(np.float32)
    gap = np.concatenate([np.full((301, 1), 0.5), np.ones((301, 1))], axis=-1).astype(np.float32)
    cfg = AugmentConfig(0.0, 0.0, 0.5, 0.0)

    tf.random.set_seed(3)
    out = augment_viewset({"local_view": tf.constant(flux), "gap_view": tf.constant(gap)}, cfg)

    assert not np.allclose(out["local_view"].numpy()[:, 0], 1.0)
    assert np.allclose(out["gap_view"].numpy()[:, 0], 0.5)


def test_scalars_and_masks_pass_through_untouched():
    scalars = tf.constant(np.arange(15, dtype=np.float32))
    masks = tf.constant([1.0, 0.0])
    out = augment_viewset(
        {"local_view": present_view((31, 3)), "scalars": scalars, "masks": masks}, LOUD
    )
    assert np.array_equal(out["scalars"].numpy(), scalars.numpy())
    assert np.array_equal(out["masks"].numpy(), masks.numpy())


def test_a_disabled_config_is_close_to_a_no_op():
    view = present_view((31, 3))
    out = augment_viewset({"local_view": view}, AugmentConfig(0.0, 0.0, 0.0, 0.0))["local_view"]
    assert np.allclose(out.numpy(), view.numpy())


def test_draws_are_fresh_on_every_call():
    view = present_view((31, 3))
    cfg = AugmentConfig(0.0, 0.1, 0.0, 0.0)
    tf.random.set_seed(0)
    a = augment_viewset({"local_view": view}, cfg)["local_view"].numpy()
    b = augment_viewset({"local_view": view}, cfg)["local_view"].numpy()
    assert not np.allclose(a, b)
