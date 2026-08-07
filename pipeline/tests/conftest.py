"""Shared fixtures for the view-set tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays


def _make_view_set(n: int = 6, *, seed: int = 0, hosts: int | None = None) -> ViewSetArrays:
    """A well-formed view set: presence channels vary, both classes present.

    `hosts` spreads the `n` rows over that many stars, so more than one row
    shares a `tic_id`. The default gives every row its own star, which makes
    `StratifiedGroupKFold(groups=tic_id)` degenerate — the guard holds trivially
    and a test cannot tell grouped splitting from ungrouped. The live catalogue
    is in exactly that state (5,703 rows, 5,703 unique TICs), so a multi-planet
    host has to be constructed to be tested at all.
    """
    rng = np.random.default_rng(seed)
    views = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(0.0, 1.0, size=(n, *shape)).astype(np.float32)
        # A presence channel stuck at 1 is a gate failure the tests look for.
        arr[..., -1] = (rng.random((n, *shape[:-1])) > 0.2).astype(np.float32)
        views[name] = arr
    dv_usable = np.array([True] * (n - 1) + [False])
    scalars = pd.DataFrame(
        {
            # Contiguous blocks, not `% hosts`: labels alternate, so a modulo
            # assignment with an even `hosts` puts every same-parity row on the
            # same star and each host comes out single-class — which is both
            # unlike a real multi-planet system and enough to starve the Platt
            # fit of one class.
            "tic_id": np.arange(n) // max(1, n // (hosts or n)) + 1,
            "mission": ["TESS"] * n,
            "label": [1, 0] * (n // 2),
            "observed_transit_count": rng.integers(1, 20, n),
            "expected_transit_count": rng.integers(20, 40, n),
            # The real scalars carry it and the observation-bias metric derives
            # baseline_days from it, so a fixture without it is not the schema.
            "period": rng.uniform(0.5, 50.0, n),
            "transit_completeness": rng.random(n),
            "secondary_phase": rng.random(n) - 0.5,
            "ruwe": rng.normal(1.0, 0.2, n),
            "dv_usable": dv_usable,
            "has_ruwe": [True] * n,
        }
    )
    # Every remaining declared scalar comes from the DV report, so it is NaN on
    # exactly the rows where `dv_usable` is False. A fixture carrying four of
    # the fifteen let a build that silently drops a column pass its own tests.
    for column, values in {
        "max_multiple_event_sigma": rng.uniform(7.0, 60.0, n),
        "robust_statistic": rng.uniform(0.0, 40.0, n),
        "bootstrap_significance": 10.0 ** rng.uniform(-140.0, 0.0, n),
        "ghost_core_statistic": rng.normal(0.0, 3.0, n),
        "ghost_halo_statistic": rng.normal(0.0, 3.0, n),
        "odd_even_statistic": rng.uniform(0.0, 5.0, n),
        "weak_secondary_max_mes": rng.uniform(0.0, 20.0, n),
        "mean_sky_offset": rng.uniform(0.0, 10.0, n),
        "control_sky_offset": rng.uniform(0.0, 10.0, n),
        "summary_quality_fraction": rng.uniform(0.5, 1.0, n),
    }.items():
        scalars[column] = np.where(dv_usable, values, np.nan)
    return ViewSetArrays(views=views, scalars=scalars)


@pytest.fixture
def make_view_set():
    """The factory, so each test picks its own n and seed."""
    return _make_view_set
