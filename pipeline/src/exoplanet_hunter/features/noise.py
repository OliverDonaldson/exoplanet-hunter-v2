"""Pink-noise transit SNR (Kunimoto 2025, AJ 170:280, §2.1, Eq 1-3).

Follows Pont et al. (2006) / Hartman & Bakos (2016): the effective noise over
the transit duration combines white noise per in-transit point with red noise
per transit event, so slow systematics are not mistaken for signal. Computed
from the light curve itself — available for every target at train and serve
time, unlike the catalogue transit SNR (NaN for non-TOI targets).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinkNoiseResult:
    snr: float
    depth: float
    sigma_white: float
    sigma_red: float
    n_in_transit: int
    n_transits: int


def pink_noise_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    min_points: int = 5,
) -> PinkNoiseResult | None:
    """SNR = depth / sigma_tr with sigma_tr = sqrt(sig_w²/n + sig_r²/N_tr).

    depth (Eq 1): mean out-of-transit minus mean of points within half a
    duration of the transit centre. sig_w: std of the flux after masking
    within one duration of the centres. sig_r (Eq 3): the light curve is
    binned in time with bin width = duration; sig_r² = std(bin means)² minus
    the value expected for uncorrelated noise, floored at 0 (white-dominated
    curves). Deviation from the paper: unweighted means — single-cadence
    SPOC/Kepler errors are near-homoscedastic per target and the serving
    path carries no per-point uncertainties.
    """
    ok = np.isfinite(time) & np.isfinite(flux)
    t, f = time[ok], flux[ok]
    if len(t) < 2 * min_points or period <= 0 or duration <= 0:
        return None

    phase_days = ((t - t0 + period / 2) % period) - period / 2
    in_transit = np.abs(phase_days) < duration / 2
    out_mask = np.abs(phase_days) >= duration  # Eq 2's masked baseline
    n_in = int(in_transit.sum())
    n_tr = len(np.unique(np.round((t[in_transit] - t0) / period).astype(int)))
    if n_in < min_points or not out_mask.any() or n_tr == 0:
        return None

    depth = float(np.mean(f[out_mask])) - float(np.mean(f[in_transit]))
    sigma_w = float(np.std(f[out_mask]))
    if sigma_w <= 0:
        return None

    # Duration-wide time bins over the masked baseline.
    t_out, f_out = t[out_mask], f[out_mask]
    bins = np.floor((t_out - t_out.min()) / duration).astype(int)
    means, counts = [], []
    for b in np.unique(bins):
        sel = bins == b
        n = int(sel.sum())
        if n >= 2:
            means.append(float(np.mean(f_out[sel])))
            counts.append(n)
    if len(means) >= 3:
        sigma_bin = float(np.std(means))
        expected = float(np.sqrt(np.mean(sigma_w**2 / np.asarray(counts))))
        sigma_r = float(np.sqrt(max(sigma_bin**2 - expected**2, 0.0)))
    else:
        sigma_r = 0.0

    sigma_tr = float(np.sqrt(sigma_w**2 / n_in + sigma_r**2 / n_tr))
    return PinkNoiseResult(
        snr=depth / max(sigma_tr, 1e-12),
        depth=depth,
        sigma_white=sigma_w,
        sigma_red=sigma_r,
        n_in_transit=n_in,
        n_transits=n_tr,
    )


__all__ = ["PinkNoiseResult", "pink_noise_snr"]
