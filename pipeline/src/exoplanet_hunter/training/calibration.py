"""Post-hoc probability calibration for binary-sigmoid models.

Both calibrators are monotone in logit space (rank-preserving) and fitted
on validation scores by minimising NLL:

  * `PlattScaler` — affine, `sigmoid(a * logit + b)` (Platt 1999); the bias
    term corrects a shifted score distribution, which temperature cannot.
  * `TemperatureScaler` — the `a = 1/T, b = 0` special case (Guo et al.
    2017); kept for unpickling bundles from older runs.

Both mirror the sklearn `.predict(scores) -> scores` interface the scoring
path expects.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Minimise NLL(T) on validation `(scores, labels)` and return T*."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    logits = _logit(scores)

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        p = _sigmoid(logits / T)
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Minimise NLL over `sigmoid(a * logit(scores) + b)`; return (a*, b*).

    `a = exp(u)` keeps the slope positive (rank-preserving). BFGS with the
    analytic gradient — gradient-free search stalls on this surface.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    logits = _logit(scores)

    def nll(params: np.ndarray) -> float:
        u, b = params
        p = np.clip(_sigmoid(np.exp(u) * logits + b), _EPS, 1.0 - _EPS)
        return -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    def grad(params: np.ndarray) -> np.ndarray:
        u, b = params
        a = np.exp(u)
        residual = _sigmoid(a * logits + b) - labels
        return np.array([float(np.mean(residual * logits)) * a, float(np.mean(residual))])

    res = minimize(nll, x0=np.array([0.0, 0.0]), jac=grad, method="BFGS")
    return float(np.exp(res.x[0])), float(res.x[1])


class PlattScaler:
    """Sklearn-shaped wrapper around a learned logit-affine transform."""

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def from_validation(cls, scores: np.ndarray, labels: np.ndarray) -> PlattScaler:
        return cls(*fit_platt(scores, labels))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        return _sigmoid(self.a * _logit(scores) + self.b)

    def __repr__(self) -> str:
        return f"PlattScaler(a={self.a:.4f}, b={self.b:.4f})"


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Equal-width-binned ECE; same binning as the `/reliability` endpoint."""
    labels = np.asarray(labels, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = which == b
        if sel.any():
            ece += abs(probs[sel].mean() - labels[sel].mean()) * sel.sum() / len(probs)
    return float(ece)


class TemperatureScaler:
    """Sklearn-shaped wrapper around a learned scalar temperature."""

    def __init__(self, T: float = 1.0) -> None:
        self.T = float(T)

    @classmethod
    def from_validation(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> TemperatureScaler:
        return cls(T=fit_temperature(scores, labels))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        return _sigmoid(_logit(scores) / self.T)

    def __repr__(self) -> str:
        return f"TemperatureScaler(T={self.T:.4f})"
