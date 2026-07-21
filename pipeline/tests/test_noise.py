"""Pink-noise SNR (Kunimoto 2025 §2.1): white-noise limit and red-noise penalty."""

import numpy as np
import pytest

from exoplanet_hunter.features.noise import pink_noise_snr


def make_curve(depth=0.005, noise=1e-4, red_amp=0.0, red_period=0.7, n_periods=40):
    rng = np.random.default_rng(7)
    time = np.arange(0, 2.0 * n_periods, 2.0 / 400)
    flux = np.ones_like(time) + rng.normal(0, noise, len(time))
    if red_amp > 0:
        flux += red_amp * np.sin(2 * np.pi * time / red_period)
    phase = ((time + 1.0) % 2.0) - 1.0
    flux[np.abs(phase) < 0.05] -= depth
    return time, flux


def test_white_noise_limit_matches_analytic():
    time, flux = make_curve()
    result = pink_noise_snr(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.depth == pytest.approx(0.005, rel=0.05)
    assert result.sigma_red == pytest.approx(0.0, abs=3e-5)  # white-dominated
    # sigma_tr -> sigma_w / sqrt(n_in): SNR ~ depth * sqrt(n_in) / sigma_w.
    analytic = 0.005 * np.sqrt(result.n_in_transit) / result.sigma_white
    assert result.snr == pytest.approx(analytic, rel=0.05)


def test_red_noise_lowers_snr():
    time, flux = make_curve()
    white = pink_noise_snr(time, flux, period=2.0, t0=0.0, duration=0.1)
    time, flux = make_curve(red_amp=5e-4)
    red = pink_noise_snr(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert white is not None and red is not None
    assert red.sigma_red > 3 * red.sigma_white / np.sqrt(red.n_in_transit / red.n_transits)
    assert red.snr < 0.5 * white.snr


def test_degenerate_inputs_return_none():
    time = np.arange(0, 4, 0.005)
    flux = np.ones_like(time)
    assert pink_noise_snr(time, flux, period=-1.0, t0=0.0, duration=0.1) is None
    assert pink_noise_snr(time[:5], flux[:5], period=2.0, t0=0.0, duration=0.1) is None
