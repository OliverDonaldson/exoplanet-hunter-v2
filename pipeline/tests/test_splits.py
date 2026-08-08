"""The inner train/validation split: grouping, class balance, and index frame."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit

from exoplanet_hunter.training.splits import stratified_inner_split

VAL_FRAC = 0.2
SEED = 42


def population(n_hosts: int = 60, per_host: int = 2, *, seed: int = 0):
    """One label per host, so a group never straddles the class boundary."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_hosts), per_host)
    host_label = (rng.random(n_hosts) > 0.5).astype(int)
    y = np.repeat(host_label, per_host)
    return np.arange(len(y)), y, groups


def test_it_returns_positions_into_y_not_offsets_into_trainval():
    """The call sites used to index `trainval[tr_rel]` themselves. Returning
    absolute positions removes a frame of reference that can be got wrong."""
    positions, y, groups = population()
    trainval = positions[10:]  # deliberately not starting at zero

    train_idx, val_idx = stratified_inner_split(trainval, y, groups, val_frac=VAL_FRAC, seed=SEED)
    both = np.concatenate([train_idx, val_idx])
    assert set(both.tolist()) <= set(trainval.tolist())
    assert both.min() >= 10, "returned an offset into trainval rather than a position into y"
    assert len(both) == len(trainval)
    assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_no_host_spans_the_split():
    """The leakage rule the old splitter did get right, kept."""
    positions, y, groups = population()
    train_idx, val_idx = stratified_inner_split(positions, y, groups, val_frac=VAL_FRAC, seed=SEED)
    assert not set(groups[train_idx]) & set(groups[val_idx])


@pytest.mark.parametrize("seed", range(12))
def test_both_classes_reach_both_sides(seed):
    positions, y, groups = population(seed=seed)
    train_idx, val_idx = stratified_inner_split(positions, y, groups, val_frac=VAL_FRAC, seed=seed)
    assert len(np.unique(y[val_idx])) == 2
    assert len(np.unique(y[train_idx])) == 2


def test_it_fixes_a_split_the_old_one_gets_wrong():
    """The regression this module exists for.

    Few hosts and a lopsided base rate is exactly the shape where
    `GroupShuffleSplit` can hand the Platt fit a single-class validation split —
    it balances nothing, it only keeps hosts intact. On 20 hosts, 3 of them
    positive, at seed 1 it does exactly that and the stratified split does not.
    """
    n_hosts, n_positive, seed = 20, 3, 1
    groups = np.repeat(np.arange(n_hosts), 2)
    y = np.repeat(np.array([1] * n_positive + [0] * (n_hosts - n_positive)), 2)
    positions = np.arange(len(y))

    old = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, va_rel = next(old.split(positions, y, groups))
    assert len(np.unique(y[positions[va_rel]])) == 1, "fixture no longer reproduces the defect"

    _, val_idx = stratified_inner_split(positions, y, groups, val_frac=0.2, seed=seed)
    assert len(np.unique(y[val_idx])) == 2


def test_val_frac_sets_the_held_out_share():
    positions, y, groups = population(n_hosts=100)
    for val_frac, expected in ((0.5, 0.5), (0.25, 0.25), (0.2, 0.2)):
        _, val_idx = stratified_inner_split(positions, y, groups, val_frac=val_frac, seed=SEED)
        assert len(val_idx) / len(positions) == pytest.approx(expected, abs=0.06)


def test_a_single_class_side_raises_rather_than_calibrating_on_it():
    """Stratification cannot rescue a population that has one class, and a
    plausible-looking calibrator fitted on it is the artefact that ships."""
    groups = np.arange(20)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="holds only class"):
        stratified_inner_split(np.arange(20), y, groups, val_frac=0.2, seed=SEED)


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_val_frac_raises(val_frac):
    positions, y, groups = population()
    with pytest.raises(ValueError, match="val_frac must be in"):
        stratified_inner_split(positions, y, groups, val_frac=val_frac, seed=SEED)


def test_the_split_is_deterministic_for_a_seed():
    positions, y, groups = population()
    first = stratified_inner_split(positions, y, groups, val_frac=VAL_FRAC, seed=SEED)
    second = stratified_inner_split(positions, y, groups, val_frac=VAL_FRAC, seed=SEED)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)

    other = stratified_inner_split(positions, y, groups, val_frac=VAL_FRAC, seed=SEED + 1)
    assert not np.array_equal(first[1], other[1])
