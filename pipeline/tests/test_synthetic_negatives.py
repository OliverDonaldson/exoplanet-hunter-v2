"""Synthetic negatives — every guard made to fire.

The failure this module exists to prevent is a single one wearing several
disguises: a "synthetic negative" that still contains its transit. That is a
mislabelled positive, training accepts it in silence, the loss barely moves, and
stage 8's intervention gets recorded as having been tried. So the tests below
mostly do not check that the constructions work — they check that the module
*notices when they have not*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.synthetic_negatives import (
    INVERT,
    MAX_SURVIVING_SIGMA,
    MIN_SEGMENTS,
    SCRAMBLE,
    assert_transit_destroyed,
    draw_negative_hosts,
    folded_depth,
    invert_flux,
    make_synthetic_negative,
    scramble_flux,
    transit_significance,
)
from exoplanet_hunter.eval.injection_recovery import inject_box_transit

PERIOD, T0, DURATION, DEPTH = 3.0, 1.0, 0.12, 0.02


def curve(n: int = 4000, span: float = 40.0, seed: int = 0, depth: float = DEPTH):
    """A real-ish light curve with a box transit injected at a known ephemeris."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, span, n)
    flux = 1.0 + rng.normal(0.0, 1e-4, n)
    return time, inject_box_transit(time, flux, PERIOD, T0, DURATION, depth)


# ------------------------------------------------------------- the fold itself --


def test_the_fold_recovers_an_injected_depth():
    """Pins the measuring stick before anything is measured with it."""
    time, flux = curve()
    assert folded_depth(time, flux, PERIOD, T0, DURATION) == pytest.approx(DEPTH, abs=2e-3)


def test_a_fold_with_no_cadence_in_transit_raises_rather_than_returning_zero():
    """Zero depth and no measurement are different statements. Returning 0.0
    here would read as 'the transit is gone' — the exact false pass.

    The window is chosen to provably miss every transit rather than by shrinking
    the duration: transits sit at t=1, 4, 7…, and [2.0, 2.5] contains none.
    """
    time = np.linspace(2.0, 2.5, 200)
    with pytest.raises(ValueError, match="no finite cadence"):
        folded_depth(time, np.ones_like(time), PERIOD, T0, DURATION)


def test_a_nonpositive_period_raises():
    time, flux = curve()
    with pytest.raises(ValueError, match="positive period"):
        folded_depth(time, flux, 0.0, T0, DURATION)


# ------------------------------------------------------------------ inversion --


def test_inversion_turns_a_transit_into_a_brightening():
    time, flux = curve()
    inverted = invert_flux(flux)
    # Depth flips sign: the dip now points up.
    assert folded_depth(time, inverted, PERIOD, T0, DURATION) < -0.5 * DEPTH


def test_inversion_preserves_the_noise_it_was_supposed_to_preserve():
    """A generated curve would be easy. The point is that the star's own scatter
    and systematics survive, so the negative is hard in the same way real ones are."""
    _, flux = curve()
    assert np.std(invert_flux(flux)) == pytest.approx(np.std(flux), rel=1e-9)


def test_inverting_an_all_nan_curve_raises():
    with pytest.raises(ValueError, match="no finite flux"):
        invert_flux(np.full(100, np.nan))


# ------------------------------------------------------------------ scrambling --


def test_scrambling_destroys_the_transit_at_its_original_ephemeris():
    time, flux = curve()
    scrambled = scramble_flux(time, flux, n_segments=8, seed=1)
    assert abs(folded_depth(time, scrambled, PERIOD, T0, DURATION)) < 0.25 * DEPTH


def test_scrambling_keeps_every_cadence_it_was_given():
    """A permutation, not a filter — the flux multiset must be identical or the
    noise properties have been altered along with the signal."""
    time, flux = curve()
    scrambled = scramble_flux(time, flux, n_segments=8, seed=1)
    assert np.allclose(np.sort(scrambled), np.sort(flux))


def test_a_single_segment_scramble_raises_because_it_is_the_identity():
    """The worst case in the module: it returns the curve untouched and the
    caller labels it negative."""
    time, flux = curve()
    with pytest.raises(ValueError, match=f"at least {MIN_SEGMENTS} segments"):
        scramble_flux(time, flux, n_segments=1)


def test_a_scramble_never_returns_the_identity_permutation():
    """`permutation` may legally return the identity, and at n_segments=2 that
    is a coin flip. Swept over seeds rather than asserted once."""
    time, flux = curve(n=400, span=8.0)
    for seed in range(40):
        assert not np.array_equal(scramble_flux(time, flux, n_segments=2, seed=seed), flux)


def test_more_segments_than_cadences_raises():
    time, flux = curve(n=10, span=1.0)
    with pytest.raises(ValueError, match="cannot be cut into"):
        scramble_flux(time, flux, n_segments=50)


def test_mismatched_time_and_flux_lengths_raise():
    with pytest.raises(ValueError, match="timestamps but"):
        scramble_flux(np.arange(10.0), np.arange(9.0), n_segments=3)


def test_scrambling_is_reproducible_from_its_seed():
    time, flux = curve()
    assert np.array_equal(scramble_flux(time, flux, seed=7), scramble_flux(time, flux, seed=7))


# --------------------------------------------- the guard that carries the module --


def test_a_destroyed_transit_passes_and_reports_what_survived():
    time, flux = curve()
    scrambled = scramble_flux(time, flux, n_segments=16, seed=2)
    sigma = assert_transit_destroyed(time, flux, scrambled, PERIOD, T0, DURATION)
    assert abs(sigma) <= MAX_SURVIVING_SIGMA


def test_a_surviving_transit_raises_and_names_it_a_mislabelled_positive():
    """The construction is skipped entirely — the 'negative' is the original
    curve. Nothing downstream would ever notice."""
    time, flux = curve()
    with pytest.raises(ValueError, match="mislabelled positive"):
        assert_transit_destroyed(time, flux, flux, PERIOD, T0, DURATION)


def test_a_scramble_that_preserved_phase_is_caught():
    """The real edge case. Segments an exact multiple of the period long put
    every cadence back at the phase it started from, so the fold returns the
    transit unchanged — and the curve *has* been permuted, so every
    ordering-based check passes. Only measuring the depth catches it.
    """
    period = 2.0
    time = np.linspace(0.0, 16.0, 1600, endpoint=False)
    rng = np.random.default_rng(0)
    flux = inject_box_transit(time, 1.0 + rng.normal(0.0, 1e-4, time.size), period, 0.5, 0.2, DEPTH)
    # Eight whole-period blocks, rotated by one. Timestamps stay put, so phase
    # is preserved exactly and the transit survives the permutation intact. The
    # noise makes the rotated array differ elementwise, which is what makes this
    # nasty: it is a genuine permutation and every ordering check passes.
    blocks = np.split(flux, 8)
    rotated = np.concatenate(blocks[1:] + blocks[:1])
    assert not np.array_equal(rotated, flux)
    assert np.allclose(np.sort(rotated), np.sort(flux))
    with pytest.raises(ValueError, match="mislabelled positive"):
        assert_transit_destroyed(time, flux, rotated, period, 0.5, 0.2)


def test_an_undetectable_original_raises_because_the_check_cannot_pass_or_fail():
    """Nothing to destroy means the guard is vacuous. Saying so beats returning
    a number that reads as a clean pass."""
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 40.0, 4000)
    noise = 1.0 + rng.normal(0.0, 1e-4, time.size)
    with pytest.raises(ValueError, match="no detectable transit to destroy"):
        assert_transit_destroyed(time, noise, noise, PERIOD, T0, DURATION)


def test_the_guard_reads_significance_not_a_fraction_of_the_original_depth():
    """The regression the first real run exposed.

    A catalogue transit is often only a few sigma, so `after / before` divides
    two noisy small numbers: on the first real build three of four scrambles
    were rejected, one reporting 620% of the original depth because the scramble
    had moved a low-flux chunk into the transit window. That is noise, not a
    surviving transit.

    What this pins is the new statistic behaving correctly on a shallow but
    genuinely detectable transit — the regime the old one mishandled. It does
    *not* reproduce the 620% case: that needed real light curves with gaps and
    correlated noise, and tuning a Gaussian fixture until the old statistic
    misfired would be staging the evidence rather than testing anything.
    """
    rng = np.random.default_rng(3)
    time = np.linspace(0.0, 60.0, 12000)
    # A shallow transit — detectable, but only just, like most of the catalogue.
    flux = inject_box_transit(
        time, 1.0 + rng.normal(0.0, 3e-4, time.size), PERIOD, T0, DURATION, 8e-4
    )
    before = transit_significance(time, flux, PERIOD, T0, DURATION)
    assert before > MAX_SURVIVING_SIGMA, "the fixture must carry a detectable transit"

    scrambled = scramble_flux(time, flux, n_segments=24, seed=11)
    after = assert_transit_destroyed(time, flux, scrambled, PERIOD, T0, DURATION)
    assert abs(after) <= MAX_SURVIVING_SIGMA


def test_significance_is_signed_so_a_brightening_is_distinguishable():
    """Inversion produces a negative significance, which is proof it is not a
    transit rather than evidence of one."""
    time, flux = curve()
    assert transit_significance(time, flux, PERIOD, T0, DURATION) > 0
    assert transit_significance(time, invert_flux(flux), PERIOD, T0, DURATION) < 0


def test_significance_refuses_a_fold_with_too_few_cadences_to_measure():
    time = np.linspace(2.0, 2.5, 200)
    with pytest.raises(ValueError, match="a significance needs at least two"):
        transit_significance(time, np.ones_like(time), PERIOD, T0, DURATION)


def test_both_kinds_survive_the_guard_end_to_end():
    time, flux = curve()
    for kind in (INVERT, SCRAMBLE):
        built = make_synthetic_negative(time, flux, kind, seed=3)
        # Inversion flips the sign; the guard reads magnitude, which is what
        # "contains no transit at this ephemeris" actually means.
        if kind == SCRAMBLE:
            assert_transit_destroyed(time, flux, built, PERIOD, T0, DURATION)


def test_an_unknown_kind_raises_rather_than_defaulting_to_one():
    time, flux = curve()
    with pytest.raises(ValueError, match="unknown synthetic-negative kind"):
        make_synthetic_negative(time, flux, "shuffle")


# --------------------------------------------------------------- the host draw --


def hosts(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A pool with the catalogue's own shape: positives observed far longer."""
    rng = np.random.default_rng(seed)
    label = np.array([1, 0] * (n // 2))
    period = rng.uniform(1.0, 20.0, n)
    # Positives get many more expected transits, which is the +0.387 confound.
    expected = np.where(label == 1, rng.integers(60, 300, n), rng.integers(2, 40, n))
    return pd.DataFrame(
        {
            "tic_id": np.arange(n),
            "label": label,
            "period": period,
            "expected_transit_count": expected,
        }
    )


def test_the_draw_matches_the_positives_baselines_not_the_pools():
    """The whole intervention. A uniform draw inherits the pool's short-baseline
    bulk and makes 'short baseline' an even stronger negative cue — moving the
    correlation the wrong way while looking like it was addressed."""
    pool = hosts()
    draw = draw_negative_hosts(pool, n=40, seed=1)
    positives_median = pool[pool["label"] == 1].pipe(
        lambda f: ((f["expected_transit_count"] - 1) * f["period"]).median()
    )
    pool_median = ((pool["expected_transit_count"] - 1) * pool["period"]).median()
    assert abs(draw.median_baseline - positives_median) < abs(draw.median_baseline - pool_median)


def test_the_draw_reports_the_distribution_it_hit():
    draw = draw_negative_hosts(hosts(), n=40, seed=1)
    assert "median baseline" in draw.report()
    assert draw.n_requested == 40


def test_a_stratum_the_pool_cannot_fill_raises_instead_of_backfilling():
    """Backfilling from the short-baseline bulk returns a clean number about a
    distribution that was never built."""
    with pytest.raises(ValueError, match="Backfilling"):
        draw_negative_hosts(hosts(n=40), n=400, seed=1)


def test_the_draw_refuses_a_pool_with_no_positives_to_match():
    pool = hosts()
    with pytest.raises(ValueError, match="no positives"):
        draw_negative_hosts(pool.assign(label=0), n=10)


def test_the_draw_refuses_a_frame_missing_the_ephemeris_it_needs():
    with pytest.raises(KeyError, match=r"cannot derive baseline_days|expected_transit_count"):
        draw_negative_hosts(pd.DataFrame({"tic_id": [1, 2], "label": [0, 1]}), n=1)


def test_the_draw_is_reproducible_from_its_seed():
    a = draw_negative_hosts(hosts(), n=40, seed=5).hosts["tic_id"].tolist()
    b = draw_negative_hosts(hosts(), n=40, seed=5).hosts["tic_id"].tolist()
    assert a == b
