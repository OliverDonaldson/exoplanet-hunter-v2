"""The 301/31 view set.

The normalisation tests are the important ones: scaling each comparison view by
its own depth sends odd, even and secondary all to -1, which looks reasonable
and destroys the three diagnostics they exist to provide.
"""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.preprocess.viewset import build_view_set

PERIOD = 4.0
T0 = 2.0
DURATION = 0.2


class FakeTime:
    def __init__(self, value):
        self.value = value


class FakeLightCurve:
    """Minimal stand-in for the parts of lk.LightCurve build_view_set touches.

    Needs boolean indexing as well as .time/.flux, because the masked
    periodogram drops the in-transit cadences before re-running BLS.
    """

    def __init__(self, time, flux, columns=()):
        self.time = FakeTime(np.asarray(time, dtype=float))
        self.flux = FakeTime(np.asarray(flux, dtype=float))
        self.columns = list(columns)

    def __getitem__(self, mask):
        return FakeLightCurve(self.time.value[mask], self.flux.value[mask], self.columns)


def make_lc(
    *,
    n_transits: int = 10,
    depth: float = 0.01,
    odd_depth: float | None = None,
    secondary_depth: float = 0.0,
    cadence: float = 0.002,
    noise: float = 0.0,
    seed: int = 0,
):
    """A box-transit light curve with optional odd/even and secondary structure."""
    rng = np.random.default_rng(seed)
    time = np.arange(T0 - PERIOD / 2, T0 + (n_transits - 0.5) * PERIOD, cadence)
    flux = np.ones_like(time)
    if noise:
        flux += rng.normal(0.0, noise, size=time.size)
    epochs = np.round((time - T0) / PERIOD).astype(int)
    phase_days = time - (T0 + epochs * PERIOD)
    in_transit = np.abs(phase_days) < DURATION / 2
    this_depth = np.where((epochs % 2 == 1) & (odd_depth is not None), odd_depth or depth, depth)
    flux[in_transit] -= this_depth[in_transit]
    if secondary_depth:
        sec = np.abs(phase_days - PERIOD / 2) < DURATION / 2
        flux[sec] -= secondary_depth
    return FakeLightCurve(time, flux)


def build(lc, **kwargs):
    return build_view_set(lc, period=PERIOD, t0=T0, duration=DURATION, **kwargs)


def test_view_shapes_and_channels():
    vs = build(make_lc())
    assert vs.global_view.shape == (301, 3)
    for name in ("local_view", "odd_view", "even_view", "secondary_view"):
        assert getattr(vs, name).shape == (31, 3)
    assert vs.trend_view.shape == (301, 3)
    assert vs.unfolded_view.shape == (20, 31, 3)


def test_every_channel_is_finite():
    # NaN reaching a view poisons gradients for the whole batch.
    vs = build(make_lc(noise=0.001))
    for name in ("global_view", "local_view", "odd_view", "even_view", "secondary_view"):
        assert np.isfinite(getattr(vs, name)).all()
    assert np.isfinite(vs.unfolded_view).all()


def test_primary_is_normalised_to_unit_depth():
    vs = build(make_lc())
    assert vs.local_view[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
    assert vs.global_view[:, 0].min() == pytest.approx(-1.0, abs=1e-6)


def test_odd_even_depth_difference_survives_normalisation():
    # An eclipsing binary at twice the catalogue period: alternating depths.
    # Per-view normalisation would send both to -1 and hide it completely.
    # The shared scale is the folded primary depth — here the mean of the two —
    # so what must be preserved is their ratio, not either absolute value.
    vs = build(make_lc(depth=0.01, odd_depth=0.005))
    odd_depth = vs.odd_view[:, 0].min()
    even_depth = vs.even_view[:, 0].min()
    assert even_depth / odd_depth == pytest.approx(0.01 / 0.005, rel=0.05)
    assert abs(odd_depth - even_depth) > 0.3
    assert odd_depth != pytest.approx(even_depth, abs=0.1)


def test_equal_odd_and_even_depths_look_equal():
    vs = build(make_lc(depth=0.01))
    assert vs.odd_view[:, 0].min() == pytest.approx(vs.even_view[:, 0].min(), abs=0.05)


def test_secondary_depth_is_relative_to_the_primary():
    vs = build(make_lc(depth=0.01, secondary_depth=0.002))
    # A fifth as deep as the primary, and it must read that way.
    assert vs.secondary_view[:, 0].min() == pytest.approx(-0.2, abs=0.08)
    # Found at the opposite phase. The box is flat-bottomed so argmin lands
    # anywhere inside it — what matters is that it is half an orbit from the
    # primary, not which edge of the eclipse it picked.
    assert abs(abs(vs.secondary_phase) - 0.5) < DURATION / PERIOD


def test_a_target_with_no_secondary_shows_a_shallow_one():
    vs = build(make_lc(depth=0.01, noise=0.0005, seed=3))
    assert vs.secondary_view[:, 0].min() > -0.3  # noise, not an eclipse


def test_transit_counts_expose_completeness_the_fold_hides():
    # The measured failure: score tracks baseline (+0.211) not transit count
    # (-0.003). Folded views cannot tell these two apart; the counts can.
    many = build(make_lc(n_transits=10))
    assert (many.observed_transit_count, many.expected_transit_count) == (10, 10)

    lc = make_lc(n_transits=10)
    gap = (lc.time.value > T0 + 1.5 * PERIOD) & (lc.time.value < T0 + 7.5 * PERIOD)
    sparse = build(FakeLightCurve(lc.time.value[~gap], lc.flux.value[~gap]))
    assert sparse.expected_transit_count == 10
    assert sparse.observed_transit_count == 4  # six fell in the gap


def test_unfolded_rows_are_padded_and_flagged_not_silently_zero():
    vs = build(make_lc(n_transits=3))
    covered = vs.unfolded_view[:, :, 2].sum(axis=1) > 0
    assert covered.sum() == 3
    # Padding rows are zero in every channel including `present`, so a consumer
    # can tell "no transit here" from "a transit measured as flat".
    assert vs.unfolded_view[3:].sum() == pytest.approx(0.0)


def test_unfolded_transits_keep_their_relative_depths():
    lc = make_lc(depth=0.01, odd_depth=0.005, n_transits=6)
    vs = build(lc)
    depths = vs.unfolded_view[:6, :, 0].min(axis=1)
    # Alternating deep/shallow, preserved because every transit shares the
    # primary's scale rather than being stretched to -1 individually. Scaled by
    # the folded depth (the mean of the two), so the ratio is what to assert.
    assert depths.min() / depths.max() == pytest.approx(0.01 / 0.005, rel=0.1)
    assert np.ptp(depths) > 0.3


def test_more_transits_than_the_cap_still_counts_them_all():
    vs = build(make_lc(n_transits=25))
    assert vs.observed_transit_count == 25  # counted
    assert vs.unfolded_view.shape[0] == 20  # but only 20 stored


def test_presence_channel_marks_empty_bins():
    # Two narrow clumps of cadences leave most global bins empty; those must be
    # flagged, not read as a flat measurement.
    time = np.concatenate(
        [np.linspace(T0 - 0.05, T0 + 0.05, 200), np.linspace(T0 + 2.0, T0 + 2.1, 200)]
    )
    vs = build(FakeLightCurve(time, np.ones_like(time)))
    assert vs.global_view[:, 2].mean() < 0.2
    assert set(np.unique(vs.global_view[:, 2])) <= {0.0, 1.0}


def test_missing_trend_branch_is_absent_not_flat():
    vs = build(make_lc())
    assert vs.trend_view[:, 2].sum() == 0  # present == 0 everywhere
    assert vs.trend_view.sum() == pytest.approx(0.0)

    with_trend = build(make_lc(), trend_lc=make_lc(depth=0.02))
    assert with_trend.trend_view[:, 2].sum() > 0


def test_gap_view_measures_within_segment_holes_not_the_baseline():
    # Punch a hole at one phase inside an otherwise continuous run. Measuring
    # against the whole baseline instead of the observed segments pins every
    # bin near the same large number — a branch with no discriminating power
    # dressed up as a measurement.
    lc = make_lc(n_transits=6)
    t = lc.time.value
    phase = ((t - T0) / PERIOD + 0.5) % 1.0 - 0.5
    keep = ~((phase > 0.20) & (phase < 0.25))
    vs = build(FakeLightCurve(t[keep], lc.flux.value[keep]))
    gap = vs.gap_view[:, 0]
    assert vs.gap_view.shape == (301, 2)
    assert gap.max() > 0.5  # the punched phase
    assert np.median(gap) < 0.1  # everywhere else is intact
    assert (gap >= 0).all() and (gap <= 1).all()


def test_gap_view_is_flat_when_nothing_is_missing():
    vs = build(make_lc(n_transits=4))
    assert vs.gap_view[:, 0].max() < 0.35


def test_periodogram_lands_on_a_fixed_grid():
    # Bin k must be the same period for every target, or a conv filter learns
    # nothing transferable. Two targets with different baselines, same grid.
    short = build(make_lc(n_transits=4))
    long = build(make_lc(n_transits=12))
    assert short.periodogram_view.shape == (256, 2)
    assert long.periodogram_view.shape == (256, 2)
    assert np.isfinite(short.periodogram_view).all()
    # Power is normalised to its own peak, so the scale is comparable too.
    assert short.periodogram_view[:, 0].max() == pytest.approx(1.0)


def test_masking_the_transit_changes_the_periodogram():
    vs = build(make_lc(n_transits=10, depth=0.02))
    assert not np.allclose(vs.periodogram_view[:, 0], vs.periodogram_masked_view[:, 0])


def test_centroid_branch_absent_without_a_raw_curve():
    vs = build(make_lc())
    assert vs.centroid_view.shape == (31, 3)
    assert vs.centroid_view[:, 2].sum() == 0  # present == 0: absent, not flat
    assert vs.centroid_view.sum() == pytest.approx(0.0)


def test_rejects_an_unusable_ephemeris():
    lc = make_lc()
    for bad in ({"period": 0.0}, {"period": np.nan}, {"duration": -1.0}):
        kwargs = {"period": PERIOD, "t0": T0, "duration": DURATION, **bad}
        with pytest.raises(ValueError):
            build_view_set(lc, **kwargs)
