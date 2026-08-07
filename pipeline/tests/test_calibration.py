"""Platt calibration must correct a score shift that temperature cannot."""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.training.calibration import (
    PlattScaler,
    TemperatureScaler,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
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
    assert ece_raw > 0.10
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


# ------------------------------------- refusing to fit what cannot be fitted --


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_platt_refuses_a_single_class_validation_split(label):
    """With one class the NLL is minimised at the extremes, so the optimiser
    converges and returns a scaler mapping every score to one end. That bundle
    is what gets written to disk as the servable calibrator."""
    scores, _ = shifted_scores(n=200)
    labels = np.full(len(scores), label)
    with pytest.raises(ValueError, match="both classes"):
        fit_platt(scores, labels)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_temperature_refuses_a_single_class_validation_split(label):
    scores, _ = shifted_scores(n=200)
    labels = np.full(len(scores), label)
    with pytest.raises(ValueError, match="both classes"):
        fit_temperature(scores, labels)


def test_a_healthy_split_still_fits():
    """The guards must not fire on the ordinary case."""
    scores, y = shifted_scores(n=500)
    a, b = fit_platt(scores, y)
    assert np.isfinite(a) and np.isfinite(b) and a > 0
    assert np.isfinite(fit_temperature(scores, y))
