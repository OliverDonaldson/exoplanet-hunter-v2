"""Catalogue subsampling: selection must be stable across refreshes.

Positional `.sample(random_state=...)` reshuffled its picks whenever the
upstream TAP catalogue reordered or grew, and the churn inflated the
refresh-trigger's new-target count. `_stable_sample` ranks by a content
hash instead, so membership only changes when the pool itself does.
"""

from __future__ import annotations

import pandas as pd

from exoplanet_hunter.data.catalog import _stable_sample


def pool(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"tic_id": ids, "label": 1})


def test_selection_survives_reordering():
    ids = list(range(1000, 1100))
    a = _stable_sample(pool(ids), 30, seed=42)
    b = _stable_sample(pool(list(reversed(ids))), 30, seed=42)
    assert set(a.tic_id) == set(b.tic_id)


def test_selection_stable_under_realistic_pool_growth():
    # Churn is proportional to the pool delta: a new row displaces a pick
    # only if its hash rank lands in the top n. KOI moves ~0-2 rows per
    # refresh, so at most that many picks may swap — never a reshuffle.
    ids = list(range(1000, 1100))
    before = set(_stable_sample(pool(ids), 30, seed=42).tic_id)
    after = set(_stable_sample(pool([*ids, 5000, 5001]), 30, seed=42).tic_id)
    assert len(before & after) >= 28


def test_seed_changes_selection():
    ids = list(range(1000, 1100))
    a = set(_stable_sample(pool(ids), 30, seed=42).tic_id)
    b = set(_stable_sample(pool(ids), 30, seed=7).tic_id)
    assert a != b


def test_request_larger_than_pool_returns_everything():
    df = pool([1, 2, 3])
    assert _stable_sample(df, 10, seed=42) is df
