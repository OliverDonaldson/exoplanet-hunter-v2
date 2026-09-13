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


# --------------------------------------------------------------------------
# The period ceiling. A fixed 15 d cap excluded 20% of the catalogue (#20).
# --------------------------------------------------------------------------


def test_a_single_sector_keeps_the_old_ceiling():
    """27 d of data supports 13.5 d on two transits, which is *narrower* than
    the cap it replaces — so short baselines must not lose range."""
    from exoplanet_hunter.search.bls import MIN_PERIOD_CEILING, ceiling_for

    assert ceiling_for(27.0) == MIN_PERIOD_CEILING == 15.0


def test_a_long_baseline_lifts_the_ceiling():
    """The catalogue runs to 1,825 d; a 700 d baseline can carry two transits
    of a 350 d planet and the search should be allowed to look."""
    from exoplanet_hunter.search.bls import ceiling_for

    assert ceiling_for(700.0) == 350.0


def test_the_ceiling_never_promises_a_single_transit():
    """One transit is not a period. Half the baseline is the physical bound."""
    from exoplanet_hunter.search.bls import ceiling_for

    for baseline in (30.0, 100.0, 350.0, 700.0, 1400.0):
        assert ceiling_for(baseline) <= baseline / 2 or ceiling_for(baseline) == 15.0


def test_widening_costs_no_extra_trial_periods():
    """Why this is affordable: the grid is uniform in frequency and capped in
    count, so it is dominated by the short-period end. Both saturate the cap."""
    narrow = period_grid(700.0, **GRID_KW)
    wide = period_grid(700.0, **{**GRID_KW, "period_max": 350.0})
    assert len(narrow) == len(wide) == 5_000


def test_widening_barely_coarsens_the_spacing():
    """3% measured. If this ever regresses badly the trade has changed."""
    wide = period_grid(700.0, **{**GRID_KW, "period_max": 350.0})
    narrow = period_grid(700.0, **GRID_KW)
    df = lambda g: (1 / g.min() - 1 / g.max()) / (len(g) - 1)  # noqa: E731
    assert df(wide) / df(narrow) < 1.05


def test_an_explicit_ceiling_still_wins():
    """`build_viewset` passes PERIODOGRAM_RANGE explicitly to build a *model
    input*. Deriving one there would silently change the shard set."""
    import lightkurve as lk

    rng = np.random.default_rng(0)
    time = np.arange(0.0, 400.0, 0.02)
    flux = 1.0 + rng.normal(0, 1e-4, time.size)
    found = bls_period_search(lk.LightCurve(time=time, flux=flux), period_max=15.0)
    assert found.period <= 15.0
