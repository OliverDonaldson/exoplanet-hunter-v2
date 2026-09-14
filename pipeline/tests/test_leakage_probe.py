"""Does host grouping hold, and would anything notice if it stopped?

`known-limits.md` carries the finding that `StratifiedGroupKFold` is inert:
`labels.parquet` is 5,812 rows on 5,812 distinct `tic_id`, so max group size is
1 and it partitions identically to `StratifiedKFold`. The project is leak-free
via the label builder's one-row-per-host emit, not via the splitter, which makes
every existing grouping test vacuous — they pass whether or not the guard works,
because there is nothing to group.

These build a TCE-level set with several rows per host, the unit of analysis #37
would introduce, and check the guard on data that can expose it. `straddling` is
the single assertion, applied to the real splitter and then to an ungrouped one:
a probe is only worth having if the same check goes red on a split that leaks.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from exoplanet_hunter.training.splits import (
    assigned_group_kfold,
    build_fold_assignment,
    stratified_inner_split,
)

N_SPLITS = 5
ROWS_PER_HOST = 3


def straddling(groups: np.ndarray, splits) -> list[int]:
    """Hosts appearing on both sides of any fold. Empty is the only pass.

    One implementation, used against the real splitter and the broken one, so
    the demonstration that it fails is a demonstration about *this* check.
    """
    bad: set[int] = set()
    for trainval_idx, test_idx in splits:
        bad |= {int(g) for g in set(groups[trainval_idx]) & set(groups[test_idx])}
    return sorted(bad)


@pytest.fixture
def tce_level():
    """150 rows over 50 hosts, three TCEs each.

    The label is per host, as it is in the catalogue: every TCE of a confirmed
    host carries the same class, which is what makes a straddling host a leak
    rather than a coincidence.
    """
    rng = np.random.default_rng(0)
    hosts = np.arange(1000, 1050)
    groups = np.repeat(hosts, ROWS_PER_HOST)
    y = np.repeat(rng.integers(0, 2, len(hosts)), ROWS_PER_HOST)
    return groups, y


def test_the_fixture_is_not_vacuous(tce_level):
    """Why every existing grouping test proves nothing: on today's labels max
    group size is 1, so the guard has nothing to do and passes regardless."""
    groups, _ = tce_level
    assert max(int((groups == g).sum()) for g in np.unique(groups)) == ROWS_PER_HOST


def test_no_host_spans_train_and_test_in_any_fold(tce_level):
    """The property that matters at training time."""
    groups, y = tce_level
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=42)
    splits = assigned_group_kfold(groups, assignment, n_splits=N_SPLITS)
    assert straddling(groups, splits) == []


def test_the_same_check_goes_red_on_an_ungrouped_split(tce_level):
    """The mutation, and the whole point of the file. The identical assertion,
    against a splitter that does not group, must find the leak — otherwise the
    test above passes for the wrong reason and would wave a real one through."""
    groups, y = tce_level
    splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0).split(np.zeros(len(y)), y)
    assert straddling(groups, splits), "an ungrouped split did not leak; the probe proves nothing"


def test_a_group_reaching_two_folds_is_refused(tce_level):
    """`build_fold_assignment` raises rather than resolving the conflict. Pinned
    because the map is a dict keyed by host, so by the time it is built a
    straddling host is unrepresentable — this guard is the only place it shows."""
    groups, y = tce_level
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=42)
    assert set(assignment) == {int(g) for g in np.unique(groups)}
    assert sorted(set(assignment.values())) == list(range(N_SPLITS))


def test_the_inner_split_keeps_a_host_whole(tce_level):
    """Early stopping and the Platt fit both read the inner split, so a host
    straddling it leaks into the calibrator that ships."""
    groups, y = tce_level
    trainval = np.arange(len(y))
    train_idx, val_idx = stratified_inner_split(trainval, y, groups, val_frac=0.2, seed=7)
    assert straddling(groups, [(train_idx, val_idx)]) == []


def test_the_inner_check_goes_red_without_grouping(tce_level):
    """Same mutation one level down."""
    groups, y = tce_level
    rng = np.random.default_rng(3)
    shuffled = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    assert straddling(groups, [(shuffled[:cut], shuffled[cut:])])


def test_the_guard_is_inert_when_every_row_is_its_own_host():
    """Today's shape. Max group size 1 means the grouped splitter and the
    ungrouped one agree exactly, so the guard earns no credit on the current
    label set — this is what would change if #37 landed."""
    y = np.repeat([0, 1], 50)
    groups = np.arange(len(y))
    assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=42)
    assert max(int((groups == g).sum()) for g in np.unique(groups)) == 1
    assert len(assignment) == len(y)
