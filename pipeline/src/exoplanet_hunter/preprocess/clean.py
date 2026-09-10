"""Light-curve cleaning + detrending.

The two operations here happen on raw light curves before any phase-folding:

  * **clean_lightcurve** — drop NaNs and sigma-clip bright outliers (cosmic
    rays, momentum-dump artefacts, jumps). One-sided on the upper tail so
    deep transit dips aren't clipped as negative outliers.
  * **flatten_lightcurve** — fit and divide out long-term stellar variability
    via a Savitzky-Golay filter, with in-transit cadences masked out of the
    fit so the filter doesn't learn to interpolate through the very dip we
    want to preserve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


def clean_lightcurve(
    lc: lk.LightCurve,
    sigma_clip: float = 5.0,
    min_points: int = 1000,
) -> lk.LightCurve:
    """Drop NaNs and sigma-clip upper outliers only.

    Two-sided clipping — the lightkurve default — would treat deep transit dips as
    negative outliers and delete them. Only the upper tail is clipped (cosmic rays,
    scattered light, pointing jumps); anything real on the lower tail is kept and
    handled by flattening and masking downstream.

    Raises when fewer than `min_points` good cadences remain.
    """
    cleaned = lc.remove_nans().remove_outliers(sigma_upper=sigma_clip, sigma_lower=np.inf)
    if len(cleaned) < min_points:
        raise ValueError(
            f"only {len(cleaned)} good cadences after cleaning (required ≥{min_points})"
        )
    return cleaned


def _transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    pad: float = 1.0,
) -> np.ndarray:
    """Boolean mask: True inside a transit window, False in baseline.

    `duration` is the full transit duration in *days*. `pad` widens the
    in-transit window as a multiple of duration so ingress/egress tails are
    fully excluded from the out-of-transit fit.
    """
    half = 0.5 * pad * duration
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) <= half


def flatten_lightcurve(
    lc: lk.LightCurve,
    window_length: int = 301,
    polyorder: int = 2,
    *,
    period: float | None = None,
    t0: float | None = None,
    duration: float | None = None,
    mask_pad: float = 1.0,
) -> lk.LightCurve:
    """Remove long-term stellar variability with a Savitzky-Golay filter.

    `window_length` is in cadences, not days: at 2-min cadence, 301 is about 10
    hours, comfortably wider than a 1-6 h transit so the dip survives.

    Given `period`, `t0` and `duration`, the in-transit cadences are masked out of
    the fit so the spline cannot flatten the dip itself — the classic "filter learns
    the transit" failure. Without an ephemeris it falls back to unmasked flattening.
    """
    mask = None
    if period is not None and t0 is not None and duration is not None:
        time = np.asarray(lc.time.value, dtype=float)
        # lightkurve convention: `mask=True` means "exclude from the fit".
        mask = _transit_mask(time, period=period, t0=t0, duration=duration, pad=mask_pad)

    return lc.flatten(window_length=window_length, polyorder=polyorder, mask=mask)
