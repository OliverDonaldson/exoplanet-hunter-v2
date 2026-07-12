"""Calibration: Platt must correct the score shift that temperature cannot.

The full-scale expansion run produced raw scores shifted wholesale below the
positive base rate (ECE 0.136 with temperature-only calibration). These tests
pin the property that motivated the switch: a logit-space *shift* is exactly
recoverable by `PlattScaler`'s bias term and provably out of reach for
`TemperatureScaler`.
"""

from __future__ import annotations

import numpy as np

from exoplanet_hunter.training.calibration import (
    PlattScaler,
    TemperatureScaler,
    expected_calibration_error,
    fit_platt,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def shifted_scores(
    n: int = 4000, shift: float = -1.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from sigmoid(z); the 'model' reports sigmoid(z + shift)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < _sigmoid(z)).astype(float)
    return _sigmoid(z + shift), y


def test_fit_platt_recovers_a_pure_logit_shift():
    scores, y = shifted_scores()
    a, b = fit_platt(scores, y)
    assert abs(a - 1.0) < 0.15
    assert abs(b - 1.5) < 0.25  # undoes the -1.5 shift


def test_platt_fixes_the_shift_temperature_cannot():
    scores, y = shifted_scores()
    ece_raw = expected_calibration_error(y, scores)
    ece_temp = expected_calibration_error(
        y, TemperatureScaler.from_validation(scores, y).predict(scores)
    )
    ece_platt = expected_calibration_error(
        y, PlattScaler.from_validation(scores, y).predict(scores)
    )
    assert ece_raw > 0.10  # the failure mode is real in this fixture
    assert ece_platt < 0.02
    assert ece_platt < ece_temp


def test_platt_is_rank_preserving():
    scores, y = shifted_scores(n=500)
    calibrated = PlattScaler.from_validation(scores, y).predict(scores)
    order = np.argsort(scores)
    assert np.all(np.diff(calibrated[order]) >= 0)


def test_ece_near_zero_for_a_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.03
