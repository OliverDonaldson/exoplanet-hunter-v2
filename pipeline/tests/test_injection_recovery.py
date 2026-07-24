"""Tests for the model-independent injection-recovery core."""

import numpy as np
import pytest

from exoplanet_hunter.eval.injection_recovery import (
    completeness_curve,
    inject_box_transit,
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
