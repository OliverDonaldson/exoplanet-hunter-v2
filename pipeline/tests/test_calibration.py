"""Platt calibration must correct a score shift that temperature cannot."""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.training import calibration
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


def test_the_convergence_guard_fires(monkeypatch):
    """`_assert_converged` shipped with no test that makes it fire, and a guard
    nobody has watched fail is not a guard. Its first contact with real data was
    an hour-long training run dying on fold 0."""
    scores, y = shifted_scores(n=200)

    class _Stalled:
        success = False
        message = "Desired error not necessarily achieved due to precision loss."
        x = np.array([0.0, 0.0])

    monkeypatch.setattr(calibration, "minimize", lambda *a, **k: _Stalled())
    with pytest.raises(RuntimeError, match="fit_platt did not converge"):
        fit_platt(scores, y)


def _saturating_split() -> tuple[np.ndarray, np.ndarray]:
    """Scores that push some rows past `_EPS`, as an ensemble mean of confident
    members does — 4 of fold 0's 868 real validation rows reached p = 3.5e-10."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.random(200) * 0.9 + 0.05, [0.0, 1.0, 1.0, 0.0]])
    labels = np.concatenate([(rng.random(200) < 0.5).astype(float), [1.0, 0.0, 1.0, 0.0]])
    return scores, labels


def test_the_objective_and_its_analytic_gradient_agree_under_saturation():
    """The stage 6 re-baseline died on fold 0 because they did not.

    The old objective clipped p to `[_EPS, 1 - _EPS]` while the analytic
    gradient handed to BFGS did not, so once any row saturated past the clip the
    two described different problems and the line search failed with "precision
    loss". A clipped row contributes a *constant* to the objective and a full
    residual to the gradient.

    This is the invariant. The convergence flag it happened to trip is
    downstream, and a test written against that flag passes on the broken code —
    checked, which is why this one is written against the gradient instead. At
    this point the old form gives an analytic u-component of 0.814 against a
    finite-difference 0.593: wrong by 27%."""
    scores, labels = _saturating_split()
    logits = calibration._logit(scores)
    params = np.array([np.log(1.4), 0.6])  # |z| ~ 22.6, so the clip is active

    def nll(x: np.ndarray) -> float:
        return calibration._nll(np.exp(x[0]) * logits + x[1], labels)

    def analytic(x: np.ndarray) -> np.ndarray:
        a = np.exp(x[0])
        residual = _sigmoid(a * logits + x[1]) - labels
        return np.array([float(np.mean(residual * logits)) * a, float(np.mean(residual))])

    step = 1e-6
    finite_difference = np.array(
        [(nll(params + step * e) - nll(params - step * e)) / (2 * step) for e in np.eye(2)]
    )
    assert finite_difference == pytest.approx(analytic(params), abs=1e-6)


def test_platt_still_fits_when_scores_saturate_past_the_logit_clip():
    """The end-to-end consequence of the invariant above: a stationary point
    rather than the last iterate before a failed line search."""
    scores, labels = _saturating_split()
    a, b = fit_platt(scores, labels)
    assert np.isfinite(a) and a > 0 and np.isfinite(b)

    logits = calibration._logit(scores)
    residual = _sigmoid(a * logits + b) - labels
    grad = np.array([float(np.mean(residual * logits)) * a, float(np.mean(residual))])
    assert np.linalg.norm(grad) < 1e-4, f"not a stationary point: ||grad||={grad}"


def test_the_objective_matches_the_textbook_cross_entropy_where_both_are_finite():
    """`softplus(z) - y*z` is only worth using if it is the same function. Away
    from the saturation that breaks the naive form, the two must agree."""
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 2.0, 500)
    y = (rng.random(500) < 0.5).astype(float)
    p = _sigmoid(z)
    textbook = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    assert calibration._nll(z, y) == pytest.approx(textbook, rel=1e-12)
