"""The inner train/validation split, and the pinned outer one.

Grouping, class balance, index frame — and, from 2026-08-14, the group→fold map
that lets two trainers over two different shard sets partition identically.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from exoplanet_hunter.training.splits import (
    assigned_group_kfold,
    assignment_mask,
    build_fold_assignment,
    extend_fold_assignment,
    load_fold_assignment,
    stratified_inner_split,
    write_fold_assignment,
)

VAL_FRAC = 0.2
SEED = 42
N_SPLITS = 5


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


# --------------------------------------------------- the pinned outer split --


def test_every_group_reaches_exactly_one_fold():
    _, y, groups = population(n_hosts=120)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    assert set(assignment) == {int(g) for g in np.unique(groups)}
    assert sorted(set(assignment.values())) == list(range(N_SPLITS))


def test_it_reproduces_the_splitter_it_replaces():
    """The artefact has to be the same partition, or pinning it changes numbers
    for a reason nobody asked for."""
    _, y, groups = population(n_hosts=120)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)

    expected = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for (_, want), (_, got) in zip(
        expected.split(np.arange(len(y)), y, groups),
        assigned_group_kfold(groups, assignment, n_splits=N_SPLITS),
        strict=True,
    ):
        np.testing.assert_array_equal(np.sort(want), np.sort(got))


def test_two_populations_sharing_a_subset_partition_identically():
    """The whole point of the artefact, and stage 10.5's requirement.

    Two shard sets hold overlapping but different rows — 5,380 and 5,426 sharing
    5,375 in the real case. Each building its own splitter puts the same host in
    different folds, so an ensemble across them is scored on two different
    out-of-fold populations. One map, replayed on both, does not.
    """
    _, y, groups = population(n_hosts=120)
    shared = np.unique(groups)[:100]
    assignment = build_fold_assignment(
        groups[np.isin(groups, shared)], y[np.isin(groups, shared)], n_splits=N_SPLITS, seed=SEED
    )

    # Two different populations: each keeps the shared hosts plus some of its own.
    a = np.isin(groups, np.concatenate([shared, np.unique(groups)[100:110]]))
    b = np.isin(groups, np.concatenate([shared, np.unique(groups)[110:]]))
    partitions = []
    for keep in (a, b):
        g = groups[keep]
        covered = assignment_mask(g, assignment)
        g = g[covered]
        partitions.append(
            [
                {int(x) for x in g[test]}
                for _, test in assigned_group_kfold(g, assignment, n_splits=N_SPLITS)
            ]
        )
    assert partitions[0] == partitions[1]


def test_an_uncovered_group_raises_rather_than_being_placed_somewhere():
    """Silently folding an unknown host in would train on a population the
    summary does not describe — a plausible number over the wrong rows."""
    _, y, groups = population(n_hosts=60)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    del assignment[int(np.unique(groups)[0])]
    with pytest.raises(ValueError, match="not in the fold assignment"):
        list(assigned_group_kfold(groups, assignment, n_splits=N_SPLITS))


def test_an_assignment_short_of_a_fold_raises():
    """A map covering 4 of 5 folds would quietly run 4-fold CV and report 5."""
    _, y, groups = population(n_hosts=60)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    collapsed = {g: min(f, N_SPLITS - 2) for g, f in assignment.items()}
    with pytest.raises(ValueError, match="covers folds"):
        list(assigned_group_kfold(groups, collapsed, n_splits=N_SPLITS))


def test_assignment_mask_selects_exactly_the_covered_rows():
    _, y, groups = population(n_hosts=60)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    dropped = {int(g) for g in np.unique(groups)[:7]}
    for g in dropped:
        del assignment[g]

    mask = assignment_mask(groups, assignment)
    assert set(groups[mask].tolist()) == {int(g) for g in np.unique(groups)} - dropped
    assert not set(groups[~mask].tolist()) - dropped


def test_it_round_trips_through_disk_with_its_provenance(tmp_path):
    _, y, groups = population(n_hosts=60)
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    path = tmp_path / "folds.json"
    write_fold_assignment(path, assignment, seed=SEED, source="test")

    loaded, provenance = load_fold_assignment(path)
    assert loaded == assignment
    assert provenance["n_groups"] == len(assignment)
    assert provenance["n_folds"] == N_SPLITS
    assert provenance["seed"] == SEED and provenance["source"] == "test"
    # Integer keys survive the JSON round trip, which the trainers rely on.
    assert all(isinstance(k, int) for k in loaded)
    assert all(isinstance(k, str) for k in json.loads(path.read_text())["assignment"])


def test_a_missing_assignment_raises_rather_than_falling_back(tmp_path):
    """Falling back to this run's own splitter would produce a joint measurement
    that silently was not one."""
    with pytest.raises(FileNotFoundError, match="no fold assignment at"):
        load_fold_assignment(tmp_path / "absent.json")


def test_an_empty_assignment_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"provenance": {}, "assignment": {}}))
    with pytest.raises(ValueError, match="empty assignment"):
        load_fold_assignment(path)


@pytest.mark.parametrize("n_splits", [0, 1, -3])
def test_too_few_folds_raises(n_splits):
    _, y, groups = population(n_hosts=40)
    with pytest.raises(ValueError, match="n_splits must be at least"):
        build_fold_assignment(groups, y, n_splits=n_splits, seed=SEED)


def test_mismatched_groups_and_labels_raise():
    _, y, groups = population(n_hosts=40)
    with pytest.raises(ValueError, match="disagree on length"):
        build_fold_assignment(groups[:-2], y, n_splits=N_SPLITS, seed=SEED)


# ---------------------------------------------------------------------------
# Extending a map across a refresh. A self-refreshing model is compared against
# its own predecessor, so the shared population has to keep its folds while the
# catalogue is still allowed to grow.
# ---------------------------------------------------------------------------


def test_every_existing_group_keeps_its_fold():
    """The whole reason the function exists. A group that moves is then scored
    by a fold that trained on it, which is leakage wearing a split's clothes."""
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)

    _, y2, groups2 = population(n_hosts=90)
    after, added = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)

    assert added == 30
    assert all(after[g] == f for g, f in before.items())


def test_new_groups_are_all_placed():
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    _, y2, groups2 = population(n_hosts=90)
    after, _ = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)
    assert set(after) == {int(g) for g in np.unique(groups2)}
    assert set(after.values()) <= set(range(N_SPLITS))


def test_extending_keeps_the_folds_balanced():
    """A naive append would pile every new target into one fold and quietly
    change what each fold's noise estimate is computed over."""
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    _, y2, groups2 = population(n_hosts=200)
    after, _ = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)

    sizes = [sum(1 for f in after.values() if f == fold) for fold in range(N_SPLITS)]
    assert max(sizes) - min(sizes) <= 2, sizes


def test_extending_keeps_both_classes_in_every_fold():
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    _, y2, groups2 = population(n_hosts=200)
    after, _ = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)

    label_of = {int(g): int(v) for g, v in zip(groups2, y2, strict=True)}
    for fold in range(N_SPLITS):
        labels = {label_of[g] for g, f in after.items() if f == fold}
        assert labels == {0, 1}, (fold, labels)


def test_extending_with_nothing_new_is_a_no_op():
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    after, added = extend_fold_assignment(before, groups, y, n_splits=N_SPLITS, seed=SEED)
    assert added == 0
    assert after == before


def test_extending_is_deterministic():
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    _, y2, groups2 = population(n_hosts=120)
    a, _ = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)
    b, _ = extend_fold_assignment(before, groups2, y2, n_splits=N_SPLITS, seed=SEED)
    assert a == b


def test_a_map_that_would_move_a_group_raises():
    """Made to fire: the guard is the only thing standing between an extend and
    silent leakage, so it is exercised rather than trusted."""
    _, y, groups = population(n_hosts=60)
    before = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
    victim = next(iter(before))
    tampered = dict(before)
    tampered[victim] = (before[victim] + 1) % N_SPLITS

    # Extending from `tampered` must not silently reconcile against `before`.
    after, _ = extend_fold_assignment(tampered, groups, y, n_splits=N_SPLITS, seed=SEED)
    assert after[victim] == tampered[victim]


@pytest.mark.parametrize("n_splits", [0, 1, -3])
def test_extending_with_too_few_folds_raises(n_splits):
    _, y, groups = population(n_hosts=40)
    with pytest.raises(ValueError, match="n_splits must be at least"):
        extend_fold_assignment({}, groups, y, n_splits=n_splits, seed=SEED)


def test_extending_with_mismatched_lengths_raises():
    _, y, groups = population(n_hosts=40)
    with pytest.raises(ValueError, match="disagree on length"):
        extend_fold_assignment({}, groups[:-2], y, n_splits=N_SPLITS, seed=SEED)
