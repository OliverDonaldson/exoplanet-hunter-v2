"""Injection-recovery completeness: how often the model recovers a real transit.

CV ROC-AUC measures *ranking*; it does not answer the detection-efficiency
question (Christiansen 2015; Coughlin 2015, KSCI-19096): at transit S/N X, what
fraction of genuine planets does the model actually pass at the serving
threshold? Inject synthetic transits of known depth/period into real light
curves, score them through the full preprocess -> ensemble path *with the
injected ephemeris*, and tally recoveries per S/N bin.

Because the ephemeris is supplied, this is the *classifier* analogue of the
pipeline injection tests — it measures our model's completeness, not a period
search. This module holds the model-independent core (injection physics, transit
S/N, per-bin aggregation); the runner that drives it through the ensemble lives
in a script so it can target whichever run the registry serves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
) -> np.ndarray:
    """Multiply a box transit of fractional ``depth`` into ``flux``.

    In-transit cadences (within ``duration/2`` of a midtime) are scaled by
    ``1 - depth``; a box is the standard first-order injection (Christiansen
    2015), a deliberate simplification of a limb-darkened profile. Multiplicative
    so it is correct on raw or normalised flux. Returns a copy.
    """
    t = np.asarray(time, dtype=float)
    injected = np.asarray(flux, dtype=float).copy()
    if period <= 0 or duration <= 0:
        return injected
    phase = np.abs(np.mod(t - t0 + 0.5 * period, period) - 0.5 * period)
    injected[phase < 0.5 * duration] *= 1.0 - depth
    return injected


def transit_snr(depth_ppm: float, cdpp_ppm: float, n_transits: int) -> float | None:
    """Transit S/N = depth / CDPP · sqrt(n_transits) (Coughlin 2015, Eq 17).

    ``depth_ppm`` and ``cdpp_ppm`` in the same (ppm) units; None when the inputs
    cannot form a ratio.
    """
    if not (cdpp_ppm > 0 and n_transits > 0 and np.isfinite(depth_ppm)):
        return None
    return float(depth_ppm / cdpp_ppm * np.sqrt(n_transits))


def noise_ppm(time: np.ndarray, flux: np.ndarray, duration: float) -> float | None:
    """Robust scatter of duration-length bins, in ppm — the CDPP analogue.

    CDPP is defined on the transit timescale (Christiansen 2012), so bin the
    flattened flux to ``duration`` and take a MAD-based sigma of the bin means.
    Computed here rather than via lightkurve so the units are unambiguous.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    if duration <= 0 or good.sum() < 2:
        return None
    t, f = t[good], f[good]
    bin_idx = np.floor((t - t.min()) / duration).astype(int)
    order = np.argsort(bin_idx, kind="stable")
    bin_idx, f = bin_idx[order], f[order]
    edges = np.flatnonzero(np.diff(bin_idx)) + 1
    means = np.array([chunk.mean() for chunk in np.split(f, edges) if len(chunk)])
    if len(means) < 2:
        return None
    median = float(np.median(means))
    if not np.isfinite(median) or median == 0:
        return None
    mad = float(np.median(np.abs(means - median)))
    return 1.4826 * mad / abs(median) * 1e6


def count_transits(time: np.ndarray, period: float, t0: float, duration: float) -> int:
    """Distinct transit epochs with at least one in-transit cadence."""
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if period <= 0 or duration <= 0 or t.size == 0:
        return 0
    phase = np.abs(np.mod(t - t0 + 0.5 * period, period) - 0.5 * period)
    in_transit = phase < 0.5 * duration
    if not in_transit.any():
        return 0
    epochs = np.round((t[in_transit] - t0) / period).astype(int)
    return int(np.unique(epochs).size)


@dataclass(frozen=True)
class RecoveryResult:
    """One injection outcome — the record the runner accumulates."""

    tic_id: int
    period_days: float
    depth_ppm: float
    snr: float
    prob: float
    recovered: bool


def completeness_curve(
    snr: np.ndarray, recovered: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recovery fraction per S/N bin.

    Returns ``(centers, fraction, count)`` for the bins defined by ``edges``;
    the recovery fraction of an empty bin is NaN. The completeness curve is the
    defensible statement of model sensitivity: "above S/N ~N the model passes
    >X% of injected planets".
    """
    snr = np.asarray(snr, dtype=float)
    recovered = np.asarray(recovered, dtype=bool)
    edges = np.asarray(edges, dtype=float)
    idx = np.digitize(snr, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    fraction = np.full(len(centers), np.nan)
    count = np.zeros(len(centers), dtype=int)
    for b in range(len(centers)):
        in_bin = idx == b
        count[b] = int(in_bin.sum())
        if count[b]:
            fraction[b] = float(recovered[in_bin].mean())
    return centers, fraction, count
