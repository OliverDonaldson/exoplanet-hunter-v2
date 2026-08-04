"""Shared fixtures for the view-set tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays


def _make_view_set(n: int = 6, *, seed: int = 0) -> ViewSetArrays:
    """A well-formed view set: presence channels vary, both classes present."""
    rng = np.random.default_rng(seed)
    views = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(0.0, 1.0, size=(n, *shape)).astype(np.float32)
        # A presence channel stuck at 1 is a gate failure the tests look for.
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


@pytest.fixture
def make_view_set():
    """The factory, so each test picks its own n and seed."""
    return _make_view_set
