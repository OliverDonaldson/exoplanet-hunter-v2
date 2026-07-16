"""Box Least Squares (BLS) period search.

Standard transit-finding algorithm: fits a flat-bottomed "box" dip at every
trial period and reports the strongest signal. Fast and robust, but assumes
the transit is rectangular — for U-shaped transits with limb-darkening
ingress/egress, see `tls.py` (TLS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class PeriodSearchResult:
    period: float
    t0: float
    duration: float
    power: float
    snr: float


def period_grid(
    baseline: float,
    *,
    period_min: float,
    period_max: float,
    min_duration: float,
    max_periods: int,
) -> np.ndarray:
    """Uniform-in-frequency trial periods, astropy-spaced, capped in count.

    astropy's `autoperiod` spacing is `df = min_duration / baseline**2`; on a
    multi-sector baseline (700+ days) that grid runs to millions of periods
    and the search to minutes, so the count is capped — trading long-baseline
    sensitivity for bounded latency. Below the cap the grid is the standard
    one.
    """
    f_min, f_max = 1.0 / period_max, 1.0 / period_min
    if baseline > 0:
        df = min_duration / baseline**2
        n = int(np.clip(np.ceil((f_max - f_min) / df), 2, max_periods))
    else:
        n = 2
    return 1.0 / np.linspace(f_min, f_max, n)


def bls_period_search(
    lc: lk.LightCurve,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    duration_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    max_periods: int = 5_000,
) -> PeriodSearchResult:
    """Run BLS over a period range and return the strongest peak."""
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    bls = BoxLeastSquares(time, flux)

    baseline = float(time.max() - time.min()) if time.size else 0.0
    periods = period_grid(
        baseline,
        period_min=period_min,
        period_max=period_max,
        min_duration=min(duration_grid),
        max_periods=max_periods,
    )

    result = bls.power(periods, list(duration_grid))
    best = int(np.argmax(result.power))
    period = float(result.period[best])
    t0 = float(result.transit_time[best])
    dur = float(result.duration[best])
    power = float(result.power[best])

    # Crude SNR estimate: peak power / median power.
    snr = power / float(np.median(result.power) + 1e-12)
    return PeriodSearchResult(period=period, t0=t0, duration=dur, power=power, snr=snr)


def bls_periodogram(
    lc: lk.LightCurve,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    duration_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    max_periods: int = 5_000,
    max_points: int = 400,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bounded BLS power spectrum for display: (periods, power, best_period).

    Downsampled by max-pooling so peaks survive; same capped grid as
    `bls_period_search`.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]

    baseline = float(time.max() - time.min()) if time.size else 0.0
    periods = period_grid(
        baseline,
        period_min=period_min,
        period_max=period_max,
        min_duration=min(duration_grid),
        max_periods=max_periods,
    )
    result = BoxLeastSquares(time, flux).power(periods, list(duration_grid))
    power = np.asarray(result.power, dtype=float)
    best = float(result.period[int(np.argmax(power))])

    if len(periods) > max_points:
        stride = int(np.ceil(len(periods) / max_points))
        n = (len(periods) // stride) * stride
        periods = periods[:n].reshape(-1, stride).mean(axis=1)
        power = power[:n].reshape(-1, stride).max(axis=1)
    return periods, power, best
