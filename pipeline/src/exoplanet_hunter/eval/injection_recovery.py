"""Injection-recovery completeness: how often the model recovers a real transit.

CV ROC-AUC measures *ranking*; it does not answer the detection-efficiency
question (Christiansen 2015; Coughlin 2015, KSCI-19096): at transit S/N X, what
fraction of genuine planets does the model pass at the serving threshold?
Inject synthetic transits of known depth and period into real light curves,
score them through the full preprocess -> ensemble path *with the injected
ephemeris*, and tally recoveries per S/N bin.

Because the ephemeris is supplied, this is the *classifier* analogue of the
pipeline injection tests — completeness, not a period search. The
model-independent core lives here; the runner that drives it through the
ensemble lives in a script, so it can target whichever run the registry serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

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


@dataclass(frozen=True)
class BlockedLift:
    """Recovery lift at one S/N level, measured within blocks."""

    snr: float
    lift: float
    se: float
    t: float
    p_value: float
    ci95: tuple[float, float]
    n_blocks: int


def within_host_lift(
    snr: np.ndarray,
    recovered: np.ndarray,
    host: np.ndarray,
    *,
    control: float = 0.0,
) -> list[BlockedLift]:
    """Lift over each host's own control cell, with the host as the block.

    The grid is a randomised block design, so the estimand is the treatment
    effect over the control and not the raw recovery rate: the null-injection
    floor is non-zero and differs by host class, and a curve that ignores it
    measures the star as much as the transit. Collapsing to one value per block
    before the test is what makes ``n_blocks - 1`` the honest degrees of freedom.
    """
    # The ignore is load-bearing under `mypy pipeline/src`, which finds no root
    # config and so never reads the scipy override; `cd pipeline && mypy` calls it
    # unused. That split is #55.
    from scipy import stats  # type: ignore[import-untyped]

    snr = np.asarray(snr, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    host = np.asarray(host)
    if not len(snr) == len(recovered) == len(host):
        raise ValueError(f"ragged inputs: {len(snr)}, {len(recovered)}, {len(host)}")

    control_rows = snr == control
    if not control_rows.any():
        raise ValueError(f"no rows at the control level S/N={control} — there is no baseline")
    floors = {h: recovered[control_rows & (host == h)].mean() for h in np.unique(host)}
    if any(np.isnan(v) for v in floors.values()):
        missing = sorted(str(h) for h, v in floors.items() if np.isnan(v))
        raise ValueError(f"hosts with no control cell, so no lift is defined: {missing}")

    out: list[BlockedLift] = []
    for level in sorted(set(snr[~control_rows])):
        at = snr == level
        blocks = np.array(
            [recovered[at & (host == h)].mean() - floors[h] for h in np.unique(host[at])]
        )
        n = len(blocks)
        if n < 2:
            raise ValueError(f"S/N={level} covers {n} block(s); a paired test needs at least 2")
        mean, se = float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(n))
        t = mean / se if se > 0 else np.inf * np.sign(mean)
        half = float(stats.t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
        out.append(
            BlockedLift(
                snr=float(level),
                lift=mean,
                se=se,
                t=float(t),
                p_value=float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else 0.0,
                ci95=(mean - half, mean + half),
                n_blocks=n,
            )
        )
    return out


def snr_at_half_max_lift(lifts: list[BlockedLift]) -> float:
    """The S/N reaching half the maximum lift, interpolated in log10 S/N.

    Not the S/N at 50% recovery: with a non-zero floor those are different
    quantities, and a two-parameter logistic forced through the origin
    estimates neither. Named because the two differ by a factor of ~2 here.
    """
    if len(lifts) < 2:
        raise ValueError("need at least two S/N levels to interpolate a half-maximum")
    target = max(x.lift for x in lifts) / 2.0
    for lo, hi in pairwise(lifts):
        if lo.lift <= target <= hi.lift and hi.lift > lo.lift:
            frac = (target - lo.lift) / (hi.lift - lo.lift)
            return float(10 ** (np.log10(lo.snr) + frac * (np.log10(hi.snr) - np.log10(lo.snr))))
    raise ValueError(f"no bracketing pair reaches half the maximum lift ({target:.4f})")
