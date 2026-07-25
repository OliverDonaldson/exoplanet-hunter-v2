"""Tests for the model-independent injection-recovery core."""

import numpy as np
import pytest

from exoplanet_hunter.eval.injection_recovery import (
    completeness_curve,
    count_transits,
    inject_box_transit,
    noise_ppm,
    transit_snr,
)


def test_inject_box_transit_dims_only_in_transit():
    time = np.arange(0, 10, 0.01)
    flux = np.ones_like(time)
    out = inject_box_transit(time, flux, period=2.0, t0=0.5, duration=0.1, depth=0.02)

    phase = np.abs(np.mod(time - 0.5 + 1.0, 2.0) - 1.0)
    in_tr = phase < 0.05
    assert np.allclose(out[in_tr], 0.98)  # scaled by 1 - depth
    assert np.allclose(out[~in_tr], 1.0)  # baseline untouched
    assert out is not flux  # returns a copy
    # ~5 transits over 10 d at P=2 d, each 0.1 d wide (0.01 sampling => ~10 pts)
    assert 40 <= in_tr.sum() <= 60


def test_inject_box_transit_guards_bad_ephemeris():
    time = np.arange(0, 5, 0.1)
    flux = np.ones_like(time)
    assert np.allclose(inject_box_transit(time, flux, 0.0, 0.0, 0.1, 0.02), flux)


def test_transit_snr():
    assert transit_snr(1000.0, 100.0, 9) == pytest.approx(30.0)  # 10 * 3
    assert transit_snr(1000.0, 0.0, 9) is None
    assert transit_snr(1000.0, 100.0, 0) is None


def test_completeness_curve_bins_recovery_fraction():
    # bin0 [0,1): 0.5 (T); bin1 [1,2): 1.2 (F), 1.8 (T); bin2 [2,3): empty.
    snr = np.array([0.5, 1.2, 1.8])
    recovered = np.array([True, False, True])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    centers, fraction, count = completeness_curve(snr, recovered, edges)

    assert centers.tolist() == [0.5, 1.5, 2.5]
    assert count.tolist() == [1, 2, 0]
    assert fraction[0] == pytest.approx(1.0)
    assert fraction[1] == pytest.approx(0.5)  # 1 of 2 recovered
    assert np.isnan(fraction[2])  # empty bin -> NaN


def test_noise_ppm_recovers_a_known_scatter():
    rng = np.random.default_rng(0)
    # 40 d of 2-min cadence, 500 ppm per-cadence white noise, binned to 0.1 d
    # (72 cadences) => bin scatter ~500/sqrt(72) ~ 59 ppm.
    time = np.arange(0, 40, 2 / 1440)
    flux = 1.0 + rng.normal(0, 500e-6, time.size)
    measured = noise_ppm(time, flux, duration=0.1)
    assert measured == pytest.approx(500 / np.sqrt(72), rel=0.25)


def test_noise_ppm_is_scale_free_and_guards_bad_input():
    time = np.arange(0, 10, 0.01)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1e-3, time.size)
    # Same fractional scatter on a different flux level -> same ppm.
    a = noise_ppm(time, 1.0 + noise, 0.5)
    b = noise_ppm(time, 5000.0 * (1.0 + noise), 0.5)
    assert a == pytest.approx(b, rel=1e-6)
    assert noise_ppm(time, np.ones_like(time), 0.0) is None
    assert noise_ppm(np.array([1.0]), np.array([1.0]), 0.5) is None


def test_count_transits_counts_distinct_epochs():
    time = np.arange(0, 10, 0.01)  # 10 d baseline
    assert count_transits(time, period=2.0, t0=0.5, duration=0.1) == 5
    # A gap removes the epochs it covers, not the ones either side.
    gapped = time[(time < 3.0) | (time > 7.0)]
    assert count_transits(gapped, period=2.0, t0=0.5, duration=0.1) == 3
    assert count_transits(time, period=0.0, t0=0.5, duration=0.1) == 0
    assert count_transits(time, period=2.0, t0=0.5, duration=0.0) == 0


def test_injected_transit_is_measurable_end_to_end():
    # The three core pieces compose: inject a transit, measure the host noise,
    # and the recovered S/N matches depth/CDPP*sqrt(n) within sampling error.
    rng = np.random.default_rng(7)
    time = np.arange(0, 27, 2 / 1440)
    flux = 1.0 + rng.normal(0, 1000e-6, time.size)
    period, t0, duration, depth = 3.0, 1.0, 0.1, 2000e-6

    injected = inject_box_transit(time, flux, period, t0, duration, depth)
    phase = np.abs(np.mod(time - t0 + 0.5 * period, period) - 0.5 * period)
    in_tr = phase < 0.5 * duration
    # The dip lands at the right depth, and only in transit.
    assert injected[in_tr].mean() == pytest.approx(flux[in_tr].mean() * (1 - depth), rel=1e-9)
    assert np.array_equal(injected[~in_tr], flux[~in_tr])

    n_tr = count_transits(time, period, t0, duration)
    assert n_tr == 9
    cdpp = noise_ppm(time, flux, duration)
    snr = transit_snr(depth * 1e6, cdpp, n_tr)
    assert snr is not None and snr > 10  # a 2000 ppm transit on this star is loud


def test_depth_for_snr_inverts_transit_snr():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from injection_recovery import depth_for_snr, snr_bin_edges

    cdpp, n_tr = 137.0, 11
    for target in (3.0, 7.5, 30.0):
        depth = depth_for_snr(target, cdpp, n_tr)
        assert transit_snr(depth, cdpp, n_tr) == pytest.approx(target)

    # Each requested level must land in its own bin, in order.
    levels = np.array([3.0, 5.0, 10.0, 30.0])
    edges = snr_bin_edges(levels)
    assert len(edges) == len(levels) + 1
    assert np.all(np.diff(edges) > 0)
    assert (np.digitize(levels, edges) - 1).tolist() == [0, 1, 2, 3]
