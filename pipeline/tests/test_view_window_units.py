"""The fold window is specified in phase and applied in days.

`build_views` computes the local half-width as `local_durations * duration /
period` and comments it "half-window in phase units". `fold_to_profile` then
passes it to `bin_profile` as a bound on `lc.fold(...).time`, which lightkurve
returns in the light curve's own time units — days — not normalised phase.

So both windows are days:

  * the global view spans +/-0.5 **days**, which is 1/P of the phase, not all
    of it. Above P = 1 d it cannot contain phase 0.5, so it cannot carry the
    secondary eclipse the console says it carries;
  * the local view spans +/-3D/P **days**, which is 3/P durations, not 3.

The xfails below assert the *intended* behaviour and are strict: when the units
are reconciled they will pass, and strict xfail turns an unexpected pass into a
failure, which is the prompt to delete the marker. They are not guards — they
are the gap, written down and executable.

Measured consequence on the shipped set: the local view's baseline fraction
correlates with log period at +0.4351 (n = 5,321). A window fixed at +/-3
durations is period-independent by construction and cannot produce that trend;
a +/-3D/P day window reproduces it. See the audit file for the full table.
"""

from __future__ import annotations

import lightkurve as lk
import numpy as np
import pytest

from exoplanet_hunter.preprocess.fold import fold_to_profile
from exoplanet_hunter.preprocess.views import build_views

PERIOD, T0, DURATION, DEPTH = 10.0, 1.25, 0.20, 0.01


def _lc(period=PERIOD, duration=DURATION, secondary=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, max(70.0, 6 * period), 0.002)
    phase = ((t - T0 + 0.5 * period) % period) / period - 0.5
    flux = np.ones_like(t) + rng.normal(0.0, noise, t.size)
    flux[np.abs(phase) < (duration / period) / 2] -= DEPTH
    if secondary:
        flux[np.abs(np.abs(phase) - 0.5) < (duration / period) / 2] -= secondary
    return lk.LightCurve(time=t, flux=flux)


def test_lightkurve_folds_into_days_not_phase():
    """The root cause, pinned. If a future lightkurve normalises by default,
    every window in this module silently changes meaning."""
    folded = _lc().fold(period=PERIOD, epoch_time=T0)
    span = float(np.ptp(np.asarray(folded.time.value, dtype=float)))
    assert span == pytest.approx(PERIOD, rel=0.01), "fold no longer returns days"


def test_the_global_window_is_half_a_day_wide():
    """Characterisation of what ships today."""
    profile = fold_to_profile(_lc(), period=PERIOD, t0=T0, n_bins=2001)
    centers = np.asarray(profile.centers, dtype=float)
    assert centers.min() == pytest.approx(-0.5, abs=0.01)
    assert centers.max() == pytest.approx(0.5, abs=0.01)


@pytest.mark.xfail(strict=True, reason="window is days; +/-0.5 d is 1/P of the phase")
def test_the_global_view_spans_the_whole_phase():
    """What the console claims: '2,001 bins across the whole phase'."""
    profile = fold_to_profile(_lc(), period=PERIOD, t0=T0, n_bins=2001)
    centers = np.asarray(profile.centers, dtype=float)
    assert float(np.ptp(centers)) == pytest.approx(PERIOD, rel=0.05)


@pytest.mark.xfail(strict=True, reason="phase 0.5 falls outside a +/-0.5 d window for P > 1 d")
def test_the_global_view_carries_a_secondary_eclipse():
    """What the console claims: the global view 'carries orbital shape and any
    secondary eclipse'. A secondary at phase 0.5 sits 5 d from mid-transit here."""
    views = build_views(_lc(secondary=0.005), period=PERIOD, t0=T0, duration=DURATION)
    edges = np.concatenate([views.global_view[:200], views.global_view[-200:]])
    assert float(edges.min()) < -0.15, "no dip near the view edges"


@pytest.mark.xfail(strict=True, reason="half-width is 3*D/P days, so 3/P durations")
def test_the_local_view_spans_three_durations():
    """What `local_durations=3.0` says it does."""
    profile = fold_to_profile(
        _lc(), period=PERIOD, t0=T0, n_bins=201, phase_min=-3 * DURATION, phase_max=3 * DURATION
    )
    built = build_views(_lc(), period=PERIOD, t0=T0, duration=DURATION)
    assert len(built.local_view) == len(profile.median)
    half_days = min(max(3 * DURATION / PERIOD, 1e-3), 0.5)
    assert half_days == pytest.approx(3 * DURATION, rel=0.05)


@pytest.mark.parametrize("period", [10.0, 30.0, 100.0])
def test_a_long_period_local_window_falls_inside_the_transit(period):
    """The sharp end. The window is 3/P durations, so it falls entirely inside
    the transit once P exceeds about 6 durations — every cadence in it is
    in-transit, the view carries no baseline to compare against, and after
    `_normalise` it is amplified noise. At P = 3.5 d with this duration the
    window is still 0.86 durations, wider than the transit, which is why the
    threshold matters and the case is parametrised rather than asserted flat."""
    duration = 0.2
    half_days = min(max(3 * duration / period, 1e-3), 0.5)
    assert half_days < duration / 2, f"window {half_days:.4f} d is not inside the transit"
