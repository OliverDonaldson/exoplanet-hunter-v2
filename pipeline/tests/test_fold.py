"""Phase binning: the primitive under every view.

The equivalence test pins the median path against the original per-bin-mask
implementation — `fold_and_bin` feeds live serving, so a silent change there
moves every score.
"""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.preprocess.fold import bin_profile


def reference_binned(phase, flux, n_bins, phase_min=-0.5, phase_max=0.5):
    """The original per-bin-mask implementation, kept as the oracle."""
    mask = (phase >= phase_min) & (phase <= phase_max) & np.isfinite(flux)
    phase, flux = phase[mask], flux[mask]
    edges = np.linspace(phase_min, phase_max, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = flux[idx == b]
        if sel.size:
            binned[b] = np.median(sel)
    return binned


@pytest.mark.parametrize("n_bins", [31, 201, 301])
def test_median_matches_the_original_implementation(n_bins):
    rng = np.random.default_rng(305)
    phase = rng.uniform(-0.5, 0.5, size=5000)
    flux = rng.normal(1.0, 0.01, size=5000)
    flux[rng.integers(0, 5000, 50)] = np.nan  # gaps must not shift the medians
    got = bin_profile(phase, flux, n_bins)
    np.testing.assert_allclose(
        got.median, reference_binned(phase, flux, n_bins), equal_nan=True, rtol=0, atol=0
    )


def test_scatter_and_count_describe_bin_occupancy():
    # Two bins: the first holds four identical points, the second four spread.
    phase = np.array([-0.4, -0.4, -0.4, -0.4, 0.4, 0.4, 0.4, 0.4])
    flux = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.1])
    profile = bin_profile(phase, flux, 2)
    assert profile.count.tolist() == [4, 4]
    assert profile.scatter[0] == pytest.approx(0.0)
    assert profile.scatter[1] > 0.0
    np.testing.assert_allclose(profile.median, [1.0, 1.0])


def test_single_point_bin_has_no_scatter_rather_than_zero():
    # 0 would read as "perfectly consistent"; the truth is "unmeasured", and a
    # variance channel that cannot tell those apart is worse than none.
    profile = bin_profile(np.array([-0.4, 0.4, 0.4]), np.array([1.0, 1.0, 1.2]), 2)
    assert profile.count.tolist() == [1, 2]
    assert np.isnan(profile.scatter[0])
    assert profile.scatter[1] > 0


def test_empty_bins_stay_nan_and_are_never_interpolated():
    # A transit inside a data gap must stay invisible to the binner rather than
    # being smoothed over by a neighbour's value.
    profile = bin_profile(np.array([-0.45, 0.45]), np.array([1.0, 1.0]), 5)
    assert profile.count.tolist() == [1, 0, 0, 0, 1]
    assert np.isnan(profile.median[1:4]).all()


def test_empty_input_yields_an_all_nan_profile():
    profile = bin_profile(np.array([]), np.array([]), 4)
    assert np.isnan(profile.median).all()
    assert profile.count.tolist() == [0, 0, 0, 0]
    assert profile.centers.size == 4


def test_points_outside_the_window_are_dropped_not_clipped_into_edge_bins():
    phase = np.array([-0.9, -0.05, 0.05, 0.9])
    flux = np.array([5.0, 1.0, 1.0, 5.0])
    profile = bin_profile(phase, flux, 2, phase_min=-0.1, phase_max=0.1)
    assert profile.count.tolist() == [1, 1]
    np.testing.assert_allclose(profile.median, [1.0, 1.0])  # the 5.0s are gone
