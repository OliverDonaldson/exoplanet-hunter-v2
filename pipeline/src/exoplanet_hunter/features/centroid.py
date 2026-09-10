"""Centroid-shift features for false-positive vetting.

A genuine transit shifts the photo-centre of the target pixel immeasurably; a
background eclipsing binary — a deep dip on a faint star inside the aperture —
shifts it clearly. This module measures that shift after detrending the raw
`MOM_CENTR1/2` columns for the systematics that swamp it: Kepler's quarterly
90-degree rolls and the per-quarter thermal drift.

Per Ansdell 2018: 5-sigma MAD outlier rejection, per-segment median subtraction
(segments from time gaps > 0.5 d, which works for both missions and survives
`stitch()`), a 1-day rolling-median detrend, then phase-fold and SNR per axis
against the robust out-of-transit scatter. Without the corrections a raw fold
captures the inter-quarter pixel jumps and returns SNRs orders too large.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import median_filter

if TYPE_CHECKING:
    import lightkurve as lk

_MAD_TO_STD = 1.4826
_DEFAULT_GAP_DAYS = 0.5
_DEFAULT_ROLL_DAYS = 1.0
_DEFAULT_OUTLIER_SIGMA = 5.0


def _robust_std(x: np.ndarray) -> float:
    """1.4826 · MAD — robust estimator of σ for a Gaussian sample."""
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return float("nan")
    xf = x[finite]
    return float(_MAD_TO_STD * np.median(np.abs(xf - np.median(xf))))


def _segment_by_time_gaps(
    t: np.ndarray, gap_days: float = _DEFAULT_GAP_DAYS
) -> list[tuple[int, int]]:
    """Return [(lo, hi), …] for each time-contiguous segment.

    Detects Kepler quarter / TESS sector boundaries from time gaps. Survives
    `LightCurveCollection.stitch()` which loses the `.quarter` metadata.
    """
    if t.size < 2:
        return [(0, t.size)]
    gaps = np.where(np.diff(t) > gap_days)[0]
    bounds = [0, *(gaps + 1).tolist(), t.size]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _detrend_axis(
    t: np.ndarray,
    c: np.ndarray,
    *,
    gap_days: float = _DEFAULT_GAP_DAYS,
    rolling_window_days: float = _DEFAULT_ROLL_DAYS,
    outlier_sigma: float = _DEFAULT_OUTLIER_SIGMA,
) -> np.ndarray:
    """Detrend a single centroid axis (MOM_CENTR1 *or* MOM_CENTR2).

    Steps per segment: 5σ MAD outlier rejection → median subtraction →
    time-based rolling-median detrend. Returns NaN for masked or
    too-short segments — the caller masks these out before in/out
    transit statistics.
    """
    out = np.full_like(c, np.nan, dtype=float)
    for lo, hi in _segment_by_time_gaps(t, gap_days):
        seg_t = t[lo:hi]
        seg_c = c[lo:hi].astype(float).copy()
        finite = np.isfinite(seg_c)
        if finite.sum() < 5:
            continue

        # 1. outlier rejection (5σ MAD)
        seg_med = float(np.median(seg_c[finite]))
        seg_std = _robust_std(seg_c)
        if np.isfinite(seg_std) and seg_std > 0:
            seg_c[np.abs(seg_c - seg_med) > outlier_sigma * seg_std] = np.nan
            finite = np.isfinite(seg_c)
        if finite.sum() < 5:
            continue

        # 2. per-segment median subtraction
        seg_c -= float(np.median(seg_c[finite]))

        # 3. rolling-median detrend via scipy.ndimage.median_filter, window sized
        # from the segment's own median cadence — far faster than pandas
        # time-based rolling at these segment lengths. NaN-safe: fill with the
        # segment median before filtering, restore the mask after. Sort first,
        # because stitched multi-quarter curves carry non-monotonic cadences that
        # would skew the window.
        order = np.argsort(seg_t, kind="stable")
        inv_order = np.argsort(order)
        seg_t_sorted = seg_t[order]
        seg_c_sorted = seg_c[order]

        diffs = np.diff(seg_t_sorted)
        diffs = diffs[diffs > 0]
        cadence_days = float(np.median(diffs)) if diffs.size else 0.0208  # ≈30 min
        if not np.isfinite(cadence_days) or cadence_days <= 0:
            cadence_days = 0.0208
        window_n = max(5, round(rolling_window_days / cadence_days))
        window_n = min(window_n, seg_c_sorted.size)
        if window_n % 2 == 0:
            window_n += 1  # odd for symmetric centring

        finite_sorted = np.isfinite(seg_c_sorted)
        if finite_sorted.sum() < 5:
            continue
        seg_filled = np.where(
            finite_sorted, seg_c_sorted, float(np.median(seg_c_sorted[finite_sorted]))
        )
        baseline = median_filter(seg_filled, size=window_n, mode="nearest")
        detrended_sorted = seg_c_sorted - baseline
        detrended_sorted[~finite_sorted] = np.nan
        out[lo:hi] = detrended_sorted[inv_order]
    return out


def extract_centroid_features(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> dict[str, float]:
    """Compute centroid-shift statistics during transit vs out-of-transit.

    Returns `centroid_shift_x` and `centroid_shift_y` in pixels after detrending,
    and `centroid_snr`, the quadrature sum of the per-axis SNRs. Genuine on-target
    transits give values below about 3; background eclipsing binaries give more.

    All three are NaN when MOM_CENTR1/2 are absent, when the detrended in- or
    out-of-transit masks are too small, or when the out-of-transit scatter collapses
    to zero. Those NaNs flow to the build pipeline's median imputer downstream.
    """
    cx_col = next((c for c in ("mom_centr1", "centroid_col") if c in lc.columns), None)
    cy_col = next((c for c in ("mom_centr2", "centroid_row") if c in lc.columns), None)
    nan_dict = {
        "centroid_shift_x": float("nan"),
        "centroid_shift_y": float("nan"),
        "centroid_snr": float("nan"),
    }
    if cx_col is None or cy_col is None:
        return nan_dict

    t = np.asarray(lc.time.value, dtype=float)
    cx_raw = np.asarray(lc[cx_col].value, dtype=float)
    cy_raw = np.asarray(lc[cy_col].value, dtype=float)

    # Detrend in time order (folding here would break the rolling window).
    cx = _detrend_axis(t, cx_raw)
    cy = _detrend_axis(t, cy_raw)

    # Phase-fold inline: φ ∈ [−0.5, 0.5], same convention as lk.LightCurve.fold.
    phase = ((t - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    half = (duration / period) / 2.0
    finite = np.isfinite(cx) & np.isfinite(cy)
    in_transit = (np.abs(phase) < half) & finite
    out_transit = (np.abs(phase) > 3 * half) & finite

    n_itr = int(in_transit.sum())
    n_oot = int(out_transit.sum())
    if n_itr < 3 or n_oot < 10:
        return nan_dict

    s_x = float(np.median(cx[in_transit]) - np.median(cx[out_transit]))
    s_y = float(np.median(cy[in_transit]) - np.median(cy[out_transit]))

    sigma_x = _robust_std(cx[out_transit])
    sigma_y = _robust_std(cy[out_transit])
    if not (np.isfinite(sigma_x) and sigma_x > 0 and np.isfinite(sigma_y) and sigma_y > 0):
        return {
            "centroid_shift_x": s_x,
            "centroid_shift_y": s_y,
            "centroid_snr": float("nan"),
        }

    n_sqrt = float(n_itr) ** 0.5
    snr_x = s_x / (sigma_x / n_sqrt)
    snr_y = s_y / (sigma_y / n_sqrt)
    return {
        "centroid_shift_x": s_x,
        "centroid_shift_y": s_y,
        "centroid_snr": float(np.hypot(snr_x, snr_y)),
    }


def extract_centroid_offset(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """Single-scalar wrapper of `extract_centroid_features` for the aux vector.

    Returns `centroid_snr`, the magnitude of the in-transit centroid shift in sigma.
    Genuine on-target transits sit below about 3; background eclipsing binaries
    above it.

    NaN when `MOM_CENTR1/2` are missing, the masks are too small, or the
    out-of-transit scatter collapses to zero; the build pipeline's median imputer
    fills those at training time.
    """
    return float(extract_centroid_features(lc, period, t0, duration)["centroid_snr"])


def centroid_phase_track(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
    *,
    n_bins: int = 61,
    window_durations: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Phase-binned detrended centroid offset magnitude around the transit.

    Flat for on-target transits; a bump at phase 0 flags a background
    eclipsing binary. Returns (phase_centers, offset_pixels) over
    ±window_durations transit durations, or None when centroid columns are
    missing or the window is empty.
    """
    cx_col = next((c for c in ("mom_centr1", "centroid_col") if c in lc.columns), None)
    cy_col = next((c for c in ("mom_centr2", "centroid_row") if c in lc.columns), None)
    if cx_col is None or cy_col is None:
        return None

    t = np.asarray(lc.time.value, dtype=float)
    cx = _detrend_axis(t, np.asarray(lc[cx_col].value, dtype=float))
    cy = _detrend_axis(t, np.asarray(lc[cy_col].value, dtype=float))

    phase = ((t - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half = float(min(max(window_durations * duration / period, 1e-3), 0.5))

    finite = np.isfinite(cx) & np.isfinite(cy) & (np.abs(phase) <= half)
    if int(finite.sum()) < n_bins:
        return None

    r = np.hypot(cx[finite], cy[finite])
    edges = np.linspace(-half, half, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(phase[finite], edges) - 1, 0, n_bins - 1)
    track = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = which == b
        if sel.any():
            track[b] = float(np.median(r[sel]))
    return centers, track
