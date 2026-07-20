"""Numeric vetting diagnostics for the serving path.

The six-panel figure (eval.vetting) draws these as plots; the API needs the
underlying numbers. Same physics, JSON-sized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Background-eclipsing-binary threshold on the centroid shift, in sigma.
BEB_THRESHOLD_SIGMA = 3.0

#: Unphysical-duration caution triggers (Kunimoto 2025, AJ 170:280, §3.4).
DURATION_Q_MAX = 0.5
DURATION_Q_RATIO_MIN = 0.6
DURATION_A_OVER_RSTAR_MIN = 1.5

_G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
_R_SUN_CM = 6.957e10
_DAY_S = 86_400.0


@dataclass(frozen=True)
class OddEvenResult:
    odd_depth_ppm: float
    even_depth_ppm: float
    depth_diff_sigma: float


@dataclass(frozen=True)
class DurationResult:
    q: float
    q_circ: float | None
    q_ratio: float | None
    a_over_rstar: float | None
    suspicious: bool


def unphysical_duration(
    period: float,
    duration: float,
    *,
    stellar_radius: float | None,
    stellar_logg: float | None,
) -> DurationResult | None:
    """Unphysical transit duration test (Kunimoto 2025, AJ 170:280, §3.4).

    q = duration/period is checked against q_circ, the duration ratio of a
    central transit on a circular orbit. Caution when q > 0.5, q/q_circ < 0.6,
    or a/R* < 1.5 — the paper's single most effective false-alarm test.

    Deviation from the paper: a/R* comes from Kepler's third law with the
    stellar density (rho* = 3g / 4piGR* from the TIC logg + radius), not from
    a transit-model fit — the serving path fits no transit model. Without
    stellar params only the q > 0.5 condition can fire.
    """
    if period <= 0 or duration <= 0:
        return None
    q = duration / period

    q_circ = q_ratio = a_over_rstar = None
    if (
        stellar_radius is not None
        and stellar_logg is not None
        and stellar_radius > 0
        and np.isfinite(stellar_radius)
        and np.isfinite(stellar_logg)
    ):
        rho = 3.0 * 10.0**stellar_logg / (4.0 * np.pi * _G_CGS * stellar_radius * _R_SUN_CM)
        a_over_rstar = float((_G_CGS * (period * _DAY_S) ** 2 * rho / (3.0 * np.pi)) ** (1 / 3))
        q_circ = float(np.arcsin(min(1.0, 1.0 / a_over_rstar)) / np.pi)
        q_ratio = q / q_circ

    suspicious = (
        q > DURATION_Q_MAX
        or (q_ratio is not None and q_ratio < DURATION_Q_RATIO_MIN)
        or (a_over_rstar is not None and a_over_rstar < DURATION_A_OVER_RSTAR_MIN)
    )
    return DurationResult(
        q=float(q),
        q_circ=q_circ,
        q_ratio=q_ratio,
        a_over_rstar=a_over_rstar,
        suspicious=suspicious,
    )


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
    *,
    duration_check: DurationResult | None = None,
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
    if duration_check is not None and duration_check.suspicious:
        bits = [f"q={duration_check.q:.3g}"]
        if duration_check.q_ratio is not None:
            bits.append(f"q/q_circ={duration_check.q_ratio:.2f}")
        if duration_check.a_over_rstar is not None:
            bits.append(f"a/R*={duration_check.a_over_rstar:.1f}")
        concerns.append(f"transit duration is unphysical for this orbit ({', '.join(bits)})")

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
