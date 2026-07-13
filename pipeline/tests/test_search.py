"""BLS trial-period grid: astropy-standard spacing, bounded count."""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.search.bls import bls_period_search, period_grid

GRID_KW = {"period_min": 0.5, "period_max": 15.0, "min_duration": 0.05, "max_periods": 5_000}


def test_short_baseline_uses_standard_spacing():
    # 5-day baseline: natural grid (~967 periods) fits under the cap.
    grid = period_grid(5.0, **GRID_KW)
    df_expected = 0.05 / 5.0**2
    n_expected = int(np.ceil((1 / 0.5 - 1 / 15.0) / df_expected))
    assert len(grid) == n_expected
    assert grid.min() == pytest.approx(0.5)
    assert grid.max() == pytest.approx(15.0)


def test_single_sector_grid_saturates_the_cap():
    # 27-day sector: natural grid is ~28k periods, so the cap must bind —
    # a regression here starves the search (261 periods at one point).
    assert len(period_grid(27.0, **GRID_KW)) == 5_000


def test_long_baseline_is_capped():
    assert len(period_grid(700.0, **GRID_KW)) == 5_000


def test_search_recovers_injected_transit():
    rng = np.random.default_rng(0)
    import lightkurve as lk

    period, t0, duration, depth = 3.3, 1.1, 0.1, 0.01
    time = np.arange(0.0, 27.0, 2.0 / 60 / 24)  # one sector at 2-min cadence
    flux = 1.0 + rng.normal(0, 0.001, time.size)
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    flux[np.abs(phase) * period < duration / 2] -= depth

    found = bls_period_search(lk.LightCurve(time=time, flux=flux))
    assert found.period == pytest.approx(period, rel=0.01)
