"""Numeric vetting diagnostics for the serving path.

The six-panel figure (eval.vetting) draws these as plots; the API needs the
underlying numbers. Same physics, JSON-sized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Background-eclipsing-binary threshold on the centroid shift, in sigma.
BEB_THRESHOLD_SIGMA = 3.0


@dataclass(frozen=True)
class OddEvenResult:
    odd_depth_ppm: float
    even_depth_ppm: float
    depth_diff_sigma: float


def odd_even_depths(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    min_points: int = 5,
) -> OddEvenResult | None:
    """Depths of odd- vs even-numbered transits; a big difference means the
    "period" is really twice an eclipsing binary's true period.

    Depth = median(out-of-transit) - median(in-transit), in ppm of the
    normalised flux. The difference is expressed in sigma via the standard
    errors of the two in-transit medians. Returns None when either parity
    has too few in-transit points to say anything.
    """
    ok = np.isfinite(time) & np.isfinite(flux)
    t, f = time[ok], flux[ok]
    if len(t) == 0:
        return None

    phase_days = ((t - t0 + period / 2) % period) - period / 2
    transit_index = np.round((t - t0) / period).astype(int)
    in_transit = np.abs(phase_days) < duration / 2
    baseline = float(np.median(f[~in_transit])) if (~in_transit).any() else 1.0

    depths: dict[str, tuple[float, float]] = {}
    for name, parity in (("odd", 1), ("even", 0)):
        sel = in_transit & (transit_index % 2 == parity)
        if sel.sum() < min_points:
            return None
        depth = baseline - float(np.median(f[sel]))
        # Standard error of the median ~ 1.253 * sigma / sqrt(n).
        se = 1.253 * float(np.std(f[sel])) / np.sqrt(sel.sum())
        depths[name] = (depth, se)

    (odd_d, odd_se), (even_d, even_se) = depths["odd"], depths["even"]
    diff_sigma = abs(odd_d - even_d) / max(np.hypot(odd_se, even_se), 1e-12)
    return OddEvenResult(
        odd_depth_ppm=odd_d * 1e6,
        even_depth_ppm=even_d * 1e6,
        depth_diff_sigma=float(diff_sigma),
    )


def verdict(
    prob_calibrated: float,
    threshold: float,
    centroid_snr: float | None,
    odd_even: OddEvenResult | None,
) -> str:
    """Plain-language summary rendered in the vetting console."""
    concerns: list[str] = []
    if centroid_snr is not None and centroid_snr > BEB_THRESHOLD_SIGMA:
        concerns.append(
            f"centroid shift {centroid_snr:.1f}σ exceeds the {BEB_THRESHOLD_SIGMA:.0f}σ "
            "background-EB threshold"
        )
    if odd_even is not None and odd_even.depth_diff_sigma > 3.0:
        concerns.append(
            f"odd/even depths differ by {odd_even.depth_diff_sigma:.1f}σ "
            "(eclipsing-binary signature)"
        )

    if prob_calibrated >= 0.9 and not concerns:
        return (
            "Strong planet candidate: high calibrated probability with clean vetting diagnostics."
        )
    if prob_calibrated >= threshold and not concerns:
        return "Consistent with an on-target planetary transit; follow-up warranted."
    if concerns:
        return (
            f"Caution — {'; '.join(concerns)}. "
            f"Calibrated probability {prob_calibrated:.2f} should be discounted accordingly."
        )
    return "Unlikely to be a planetary transit at the current decision threshold."
