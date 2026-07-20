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

#: Odd/even transit-timing caution trigger (Kunimoto 2025, §4.4, Eq 13).
ODD_EVEN_TIMING_SIGMA = 10.0

#: Occultation escape hatch (Kunimoto 2025, §4.3): a secondary this shallow
#: with a sub-unity implied albedo is consistent with planetary reflection.
SECONDARY_DEPTH_RATIO_MAX = 0.1
SECONDARY_ALBEDO_MAX = 1.0

_G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
_R_SUN_CM = 6.957e10
_DAY_S = 86_400.0


@dataclass(frozen=True)
class OddEvenResult:
    odd_depth_ppm: float
    even_depth_ppm: float
    depth_diff_sigma: float
    odd_timing_min: float | None = None
    even_timing_min: float | None = None
    timing_diff_sigma: float | None = None


@dataclass(frozen=True)
class DurationResult:
    q: float
    q_circ: float | None
    q_ratio: float | None
    a_over_rstar: float | None
    suspicious: bool


@dataclass(frozen=True)
class SecondaryResult:
    secondary_depth_ppm: float
    secondary_phase: float
    secondary_significance: float
    fa_threshold: float
    primary_depth_ppm: float
    depth_ratio: float
    albedo: float | None
    occultation_like: bool
    suspicious: bool


def _a_over_rstar(
    period: float, stellar_radius: float | None, stellar_logg: float | None
) -> float | None:
    """Scaled semimajor axis via Kepler's third law from the stellar density
    (rho* = 3g / 4piGR*); None when the TIC lacks radius or logg."""
    if (
        stellar_radius is None
        or stellar_logg is None
        or stellar_radius <= 0
        or not np.isfinite(stellar_radius)
        or not np.isfinite(stellar_logg)
    ):
        return None
    rho = 3.0 * 10.0**stellar_logg / (4.0 * np.pi * _G_CGS * stellar_radius * _R_SUN_CM)
    return float((_G_CGS * (period * _DAY_S) ** 2 * rho / (3.0 * np.pi)) ** (1 / 3))


def significant_secondary(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    stellar_radius: float | None = None,
    stellar_logg: float | None = None,
    min_points: int = 5,
) -> SecondaryResult | None:
    """Significant-secondary test (Kunimoto 2025, AJ 170:280, §3.9 + §4.3).

    Simplified Model-Shift: the folded light curve is scanned with a
    duration-wide box outside ±2 durations of the primary; each box's depth
    significance is depth / (sigma_w / sqrt(n)). The strongest dip is the
    secondary; following §4.3 (Eq 9) it is significant when MS4 > 0,
    MS5 > -1, MS6 > -1 with FA1 = FA2 = sqrt(2)·erfcinv(Tdur/P)
    (Thompson 2018, Eq 13-14, N_TCEs = 1 for single-target vetting).

    Simplifications vs the paper, by design: box depths on the folded curve
    instead of a transit-model MES series; F_red = 1 (white noise assumed, so
    the paper's F_red > 1.8 guard is moot); tertiary/positive comparisons are
    skipped when no valid box exists. Broad secondaries may also be partially
    attenuated by the transit-masked detrend upstream.

    Occultation escape hatch (§4.3): a significant secondary is not flagged
    when its depth is < 10% of the primary's and the geometric albedo needed
    to produce it, A = delta_sec·(a/Rp)² (Eq 10, with a/Rp from stellar
    density and Rp/R* = sqrt(delta_pri)), is < 1. The paper's additional
    impact-parameter < 0.95 and Rp < 22 R_Earth conditions are skipped — we
    fit neither; without stellar params the hatch cannot fire and the
    caution stands.
    """
    from scipy.special import erfcinv

    ok = np.isfinite(time) & np.isfinite(flux)
    t, f = time[ok], flux[ok]
    q = duration / period if period > 0 else 0.0
    if len(t) == 0 or q <= 0 or 1.0 - 4.0 * q <= 2.0 * q:  # no phase space left
        return None

    phase = ((t - t0) / period) % 1.0  # primary at 0
    dist_pri = np.minimum(phase, 1.0 - phase)
    masked = dist_pri >= 2.0 * q
    if masked.sum() < min_points:
        return None
    baseline = float(np.median(f[masked]))
    sigma_w = 1.4826 * float(np.median(np.abs(f[masked] - baseline)))
    if sigma_w <= 0:
        return None

    in_pri = dist_pri < q / 2
    if in_pri.sum() < min_points:
        return None
    primary_depth = baseline - float(np.mean(f[in_pri]))

    centers = np.arange(2.0 * q, 1.0 - 2.0 * q, q / 4)
    sigs, depths_by_center = {}, {}
    for c in centers:
        sel = masked & (np.abs(phase - c) < q / 2)
        n = int(sel.sum())
        if n < min_points:
            continue
        depth = baseline - float(np.mean(f[sel]))
        sigs[float(c)] = depth / (sigma_w / np.sqrt(n))
        depths_by_center[float(c)] = depth
    if not sigs:
        return None

    def wrap_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 1.0 - d)

    sec_phase = max(sigs, key=lambda c: sigs[c])
    sec_sig = sigs[sec_phase]
    sec_depth = depths_by_center[sec_phase]

    ter_sigs = [s for c, s in sigs.items() if wrap_dist(c, sec_phase) >= 2.0 * q]
    pos_sigs = [
        -s
        for c, s in sigs.items()
        if wrap_dist(c, sec_phase) >= 3.0 * q and min(c, 1.0 - c) >= 3.0 * q
    ]

    fa = float(np.sqrt(2.0) * erfcinv(q))
    ms4 = sec_sig - fa
    ms5 = (sec_sig - max(ter_sigs)) - fa if ter_sigs else None
    ms6 = (sec_sig - max(pos_sigs)) - fa if pos_sigs else None
    significant = ms4 > 0 and (ms5 is None or ms5 > -1) and (ms6 is None or ms6 > -1)

    depth_ratio = sec_depth / primary_depth if primary_depth > 0 else float("inf")
    albedo = None
    a_rs = _a_over_rstar(period, stellar_radius, stellar_logg)
    if a_rs is not None and primary_depth > 0 and sec_depth > 0:
        # A = delta_sec·(a/Rp)² with a/Rp = (a/R*)/sqrt(delta_pri) (Eq 10).
        albedo = float(sec_depth * a_rs**2 / primary_depth)

    occultation_like = bool(
        significant
        and 0 < depth_ratio < SECONDARY_DEPTH_RATIO_MAX
        and albedo is not None
        and albedo < SECONDARY_ALBEDO_MAX
    )
    return SecondaryResult(
        secondary_depth_ppm=float(sec_depth * 1e6),
        secondary_phase=float(sec_phase),
        secondary_significance=float(sec_sig),
        fa_threshold=fa,
        primary_depth_ppm=float(primary_depth * 1e6),
        depth_ratio=float(depth_ratio),
        albedo=albedo,
        occultation_like=occultation_like,
        suspicious=bool(significant and not occultation_like),
    )


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

    q_circ = q_ratio = None
    a_over_rstar = _a_over_rstar(period, stellar_radius, stellar_logg)
    if a_over_rstar is not None:
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
    """Depths and timings of odd- vs even-numbered transits; a big difference
    means the "period" is really twice an eclipsing binary's true period.

    Depth = median(out-of-transit) - median(in-transit), in ppm of the
    normalised flux. The difference is expressed in sigma via the standard
    errors of the two in-transit medians. Returns None when either parity
    has too few in-transit points to say anything.

    Timing (Kunimoto 2025, AJ 170:280, §4.4, Eq 13 — OE_trap,T analogue):
    catches eccentric EBs detected at half period whose primary and secondary
    depths match but whose eclipses are not separated by exactly half an
    orbit. Per-transit midtimes are flux-weighted centroids of the in-transit
    points (the paper fits trapezoids; we don't fit models in serving), so an
    offset larger than ~half a duration saturates at the window edge. Parity
    mean offsets from the linear ephemeris are compared in sigma via their
    standard errors; timing fields are None with fewer than two usable
    transits per parity.
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

    offsets: dict[int, list[float]] = {0: [], 1: []}
    for n in np.unique(transit_index[in_transit]):
        sel = in_transit & (transit_index == n)
        w = np.clip(baseline - f[sel], 0.0, None)
        if sel.sum() < 3 or w.sum() <= 0:
            continue
        midtime = float(np.sum(w * t[sel]) / w.sum())
        offsets[int(n) % 2].append(midtime - (t0 + float(n) * period))

    odd_t = even_t = timing_sigma = None
    if len(offsets[1]) >= 2 and len(offsets[0]) >= 2:
        means, ses = {}, {}
        for parity, vals in offsets.items():
            arr = np.asarray(vals)
            means[parity] = float(arr.mean())
            ses[parity] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        timing_sigma = float(abs(means[1] - means[0]) / max(np.hypot(ses[1], ses[0]), 1e-12))
        odd_t, even_t = means[1] * 1440.0, means[0] * 1440.0

    return OddEvenResult(
        odd_depth_ppm=odd_d * 1e6,
        even_depth_ppm=even_d * 1e6,
        depth_diff_sigma=float(diff_sigma),
        odd_timing_min=odd_t,
        even_timing_min=even_t,
        timing_diff_sigma=timing_sigma,
    )


def verdict(
    prob_calibrated: float,
    threshold: float,
    centroid_snr: float | None,
    odd_even: OddEvenResult | None,
    *,
    duration_check: DurationResult | None = None,
    secondary: SecondaryResult | None = None,
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
    if (
        odd_even is not None
        and odd_even.timing_diff_sigma is not None
        and odd_even.timing_diff_sigma > ODD_EVEN_TIMING_SIGMA
    ):
        concerns.append(
            f"odd/even transit timings differ by {odd_even.timing_diff_sigma:.1f}σ "
            "(eccentric eclipsing binary at half period)"
        )
    if secondary is not None and secondary.suspicious:
        concerns.append(
            f"significant secondary eclipse at phase {secondary.secondary_phase:.2f} "
            f"({secondary.secondary_significance:.1f}σ, "
            f"{secondary.secondary_depth_ppm:.0f} ppm — eclipsing-binary signature)"
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
