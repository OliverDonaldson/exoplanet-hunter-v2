"""Per-diagnostic view set (stage 2 of the ExoMiner rebuild, old stage 1).

Separate from `views.py`, whose global/local pair feeds the live model.

Most views are `(bins, 3)` = `[flux, scatter, present]`: flux is the per-bin
median normalised so the primary's depth is -1, scatter the per-bin MAD on the
same scale, and `present` marks bins that held a cadence. Without `present` an
empty bin filled to baseline is indistinguishable from one measured flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exoplanet_hunter.preprocess.diffimage import (
    build_difference_views,
    empty_difference_views,
)
from exoplanet_hunter.preprocess.fold import bin_profile
from exoplanet_hunter.preprocess.momentum import (
    build_momentum_dump_view,
    empty_momentum_dump_view,
)

if TYPE_CHECKING:
    import lightkurve as lk

    from exoplanet_hunter.data.dv_xml import DVDifferenceImage

#: The champion's resolution, restored 2026-08-06. Stage 4 run 1 built these
#: at ExoMiner's 301/31 — on views ours oversample ~7x — and lost 0.0348 AUC on
#: Kepler, monotonically in transits caught, while TESS stayed level. That is the
#: signature of a bin count too coarse to hold what a four-year baseline
#: resolves. `docs/roadmap.md` carries the pre-registered reading of the re-run.
GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_DURATIONS = 3.0
MAX_TRANSITS = 20
PERIODOGRAM_BINS = 256
PERIODOGRAM_RANGE = (0.5, 30.0)  # days


@dataclass(frozen=True)
class ViewSet:
    """One target's views. Every array is (bins, 3) = [flux, scatter, present]."""

    global_view: np.ndarray  # (301, 3) full phase
    local_view: np.ndarray  # (31, 3)  ±3 durations around the transit
    odd_view: np.ndarray  # (31, 3)  odd-numbered transits only
    even_view: np.ndarray  # (31, 3)  even-numbered transits only
    secondary_view: np.ndarray  # (31, 3)  centred on the deepest non-primary phase
    trend_view: np.ndarray  # (301, 3) what the detrending removed
    unfolded_view: np.ndarray  # (MAX_TRANSITS, 31, 3) individual transits
    centroid_view: np.ndarray  # (31, 3)  offset in out-of-transit sigma
    gap_view: np.ndarray  # (301, 2) [missing fraction, present]
    periodogram_view: np.ndarray  # (256, 2) [BLS power, present], fixed grid
    periodogram_masked_view: np.ndarray  # (256, 2) same, transits removed
    #: (8, 17, 17, 4) per-sector difference-image stamps, and (8, 2) the quality
    #: DV assigns each one. The only view sourced from the DV report rather than
    #: the light curve, so it is absent for every Kepler and K2 row by
    #: construction — 58.9% of the set carries presence 0 here.
    difference_view: np.ndarray
    difference_quality_view: np.ndarray
    #: (201, 2) [dump fraction, present] over the local window. TESS-only: the
    #: spacecraft systematic has no Kepler or K2 analogue, so those rows carry
    #: presence 0 by construction rather than by a fetch failure.
    momentum_dump_view: np.ndarray
    #: Coverage the folded views cannot express: a single-transit candidate and
    #: a 40-transit one look identical once folded.
    observed_transit_count: int
    expected_transit_count: int
    secondary_phase: float


def _depth_of(median: np.ndarray) -> float:
    """Depth of a binned profile: distance from its baseline to its deepest bin."""
    if not np.isfinite(median).any():
        return 0.0
    return float(np.abs(np.nanmin(median - float(np.nanmedian(median)))))


def _normalise_pair(
    median: np.ndarray,
    scatter: np.ndarray,
    count: np.ndarray,
    depth: float | None = None,
) -> np.ndarray:
    """Stack (median, scatter, present) into a normalised (bins, 3) view.

    `depth` overrides the self-derived scale, and the comparison views **must**
    pass the primary's: normalising odd and even each by their own depth sends
    both to -1 and destroys the depth difference that is the whole EB
    signature. Same for the secondary and each unfolded transit.
    """
    present = (count > 0).astype(np.float32)
    if not np.isfinite(median).any():
        return np.stack([np.zeros_like(present), np.zeros_like(present), present], axis=-1)
    centred = median - float(np.nanmedian(median))
    scale = _depth_of(median) if depth is None else depth
    if scale < 1.0e-8:
        scale = 1.0
    return np.stack(
        [
            np.nan_to_num(centred / scale, nan=0.0),
            np.nan_to_num(scatter / scale, nan=0.0),
            present,
        ],
        axis=-1,
    ).astype(np.float32)


def _empty_view(n_bins: int) -> np.ndarray:
    return np.zeros((n_bins, 3), dtype=np.float32)


def _phase_of(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in [-0.5, 0.5), 0 at mid-transit."""
    return ((time - t0) / period + 0.5) % 1.0 - 0.5


def _transit_index(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Which transit each cadence belongs to, as an integer epoch number."""
    return np.round((time - t0) / period).astype(np.int64)


def _local_window(duration: float, period: float, durations: float) -> float:
    """Half-width of the local window in phase units, clamped to a sane range."""
    return float(min(max(durations * duration / period, 1e-3), 0.5))


def _binned_view(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    half: float | None = None,
    depth: float | None = None,
) -> np.ndarray:
    lo, hi = (-0.5, 0.5) if half is None else (-half, half)
    profile = bin_profile(phase, flux, n_bins, lo, hi)
    return _normalise_pair(profile.median, profile.scatter, profile.count, depth)


def _secondary_phase(phase: np.ndarray, flux: np.ndarray, half: float) -> float:
    """Phase of the deepest bin outside the primary transit.

    Searched on the global binning with the primary masked, so a grazing
    primary's wings cannot masquerade as a secondary eclipse.
    """
    profile = bin_profile(phase, flux, GLOBAL_BINS)
    outside = np.abs(profile.centers) > 3.0 * half
    if not outside.any() or not np.isfinite(profile.median[outside]).any():
        return 0.5
    candidates = np.where(outside, profile.median, np.nan)
    return float(profile.centers[int(np.nanargmin(candidates))])


def _unfolded(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    durations: float,
    max_transits: int,
    depth: float,
) -> tuple[np.ndarray, int, int]:
    """Per-transit views plus (observed, expected) transit counts.

    Expected counts every epoch the ephemeris predicts inside the baseline;
    observed counts those that caught a cadence. A single-transit candidate and
    a 40-transit one are indistinguishable once folded.
    """
    stack = np.zeros((max_transits, LOCAL_BINS, 3), dtype=np.float32)
    if time.size == 0:
        return stack, 0, 0

    half_days = durations * duration
    epochs = _transit_index(time, period, t0)
    first, last = int(epochs.min()), int(epochs.max())
    expected = last - first + 1

    # Which epoch each cadence falls in, if any — one pass over `time` rather
    # than one pass per epoch. A 4-year Kepler baseline at P=0.33 d gives 4,455
    # epochs, and scanning all 72,000 cadences for each took 0.20 s against
    # 0.0009 s here. Only the first `max_transits` epochs need binning at all.
    in_transit = np.abs(time - (t0 + epochs * period)) <= half_days
    caught = np.unique(epochs[in_transit])
    observed = int(caught.size)

    for slot, epoch in enumerate(caught[:max_transits]):
        centre = t0 + epoch * period
        window = in_transit & (epochs == epoch)
        offset = (time[window] - centre) / (2.0 * half_days)  # -0.5 .. 0.5
        stack[slot] = _binned_view(offset, flux[window], LOCAL_BINS, depth=depth)
    return stack, observed, expected


def _centroid_view(
    raw_lc: lk.LightCurve, period: float, t0: float, half: float, n_bins: int
) -> np.ndarray:
    """Phase-binned centroid offset in units of its out-of-transit scatter.

    Sigma rather than depth: this is a pixel shift, and normalising it to -1
    would say every target moved by the same amount.
    """
    from exoplanet_hunter.features.centroid import _detrend_axis

    columns = {c.lower() for c in raw_lc.columns}
    cx_col = next((c for c in ("mom_centr1", "centroid_col") if c in columns), None)
    cy_col = next((c for c in ("mom_centr2", "centroid_row") if c in columns), None)
    if cx_col is None or cy_col is None:
        return _empty_view(n_bins)

    time = np.asarray(raw_lc.time.value, dtype=float)
    try:
        cx = _detrend_axis(time, np.asarray(raw_lc[cx_col].value, dtype=float))
        cy = _detrend_axis(time, np.asarray(raw_lc[cy_col].value, dtype=float))
    except Exception:
        return _empty_view(n_bins)

    offset = np.hypot(cx, cy)
    phase = _phase_of(time, period, t0)
    finite = np.isfinite(offset)
    if finite.sum() < n_bins:
        return _empty_view(n_bins)

    # Scale by the robust scatter of cadences well away from the transit, so
    # the channel reads as "sigma of centroid shift".
    out_of_transit = finite & (np.abs(phase) > 3.0 * half)
    if out_of_transit.sum() > 3:
        sample = offset[out_of_transit]
        sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
    else:
        sigma = float("nan")
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = 1.0

    profile = bin_profile(phase[finite], offset[finite] / sigma, n_bins, -half, half)
    present = (profile.count > 0).astype(np.float32)
    baseline = float(np.nanmedian(profile.median)) if np.isfinite(profile.median).any() else 0.0
    return np.stack(
        [
            np.nan_to_num(profile.median - baseline, nan=0.0),
            np.nan_to_num(profile.scatter, nan=0.0),
            present,
        ],
        axis=-1,
    ).astype(np.float32)


def _gap_view(lc: lk.LightCurve, period: float, t0: float, n_bins: int) -> np.ndarray:
    """Fraction of expected cadences missing from each phase bin.

    The momentum-dump branch, measured as the hole a dump leaves. Reading
    `QUALITY` bit 5 directly gives zero for every target in the cache —
    lightkurve's default bitmask drops those cadences at download time — so the
    flag would have been an all-zero branch that looked like a feature.
    """
    from exoplanet_hunter.features.centroid import _segment_by_time_gaps

    time = np.asarray(lc.time.value, dtype=float)
    time = np.sort(time[np.isfinite(time)])
    if time.size < 2:
        return np.zeros((n_bins, 2), dtype=np.float32)
    cadence = float(np.median(np.diff(time)))
    if not np.isfinite(cadence) or cadence <= 0:
        return np.zeros((n_bins, 2), dtype=np.float32)

    # The ideal grid spans each *observed segment*, not the whole baseline.
    # Spanning the baseline counts the multi-year holes between sectors as
    # missing cadences, which pins every bin near the same large number — on
    # TIC 337385330 that was 0.823-0.902 across all 301 bins, a branch with no
    # discriminating power dressed up as a measurement.
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    ideal = np.concatenate(
        [
            np.arange(time[lo], time[hi - 1] + cadence, cadence)
            for lo, hi in _segment_by_time_gaps(time)
            if hi > lo
        ]
        or [np.empty(0)]
    )
    observed = np.bincount(
        np.clip(np.digitize(_phase_of(time, period, t0), edges) - 1, 0, n_bins - 1),
        minlength=n_bins,
    ).astype(float)
    expected = np.bincount(
        np.clip(np.digitize(_phase_of(ideal, period, t0), edges) - 1, 0, n_bins - 1),
        minlength=n_bins,
    ).astype(float)
    missing = np.divide(expected - observed, expected, out=np.zeros(n_bins), where=expected > 0)
    return np.stack([np.clip(missing, 0.0, 1.0), (expected > 0).astype(float)], axis=-1).astype(
        np.float32
    )


def _periodogram_view(lc: lk.LightCurve, n_bins: int) -> np.ndarray:
    """BLS power on a fixed log-period grid, so bin k is the same period always.

    `bls_periodogram`'s own grid depends on the target's baseline, which would
    make bin 40 a different period for every target.
    """
    from exoplanet_hunter.search.bls import bls_periodogram

    grid = np.geomspace(*PERIODOGRAM_RANGE, n_bins)
    try:
        periods, power, _ = bls_periodogram(
            lc, period_min=PERIODOGRAM_RANGE[0], period_max=PERIODOGRAM_RANGE[1], max_periods=2000
        )
    except Exception:
        return np.zeros((n_bins, 2), dtype=np.float32)
    if periods.size < 2 or not np.isfinite(power).any():
        return np.zeros((n_bins, 2), dtype=np.float32)

    order = np.argsort(periods)
    resampled = np.interp(grid, periods[order], power[order], left=np.nan, right=np.nan)
    present = np.isfinite(resampled).astype(np.float32)
    peak = float(np.nanmax(resampled)) if present.any() else 0.0
    scale = peak if peak > 1e-12 else 1.0
    return np.stack([np.nan_to_num(resampled / scale, nan=0.0), present], axis=-1).astype(
        np.float32
    )


def _mask_transits(lc: lk.LightCurve, period: float, t0: float, duration: float) -> lk.LightCurve:
    """Drop in-transit cadences so a second periodicity can surface."""
    time = np.asarray(lc.time.value, dtype=float)
    phase_days = (time - t0 + 0.5 * period) % period - 0.5 * period
    return lc[np.abs(phase_days) > duration]


def build_view_set(
    lc: lk.LightCurve,
    *,
    period: float,
    t0: float,
    duration: float,
    trend_lc: lk.LightCurve | None = None,
    raw_lc: lk.LightCurve | None = None,
    global_bins: int = GLOBAL_BINS,
    local_bins: int = LOCAL_BINS,
    local_durations: float = LOCAL_DURATIONS,
    max_transits: int = MAX_TRANSITS,
    periodogram_bins: int = PERIODOGRAM_BINS,
    difference_images: list[DVDifferenceImage] | None = None,
    momentum_dumps: np.ndarray | None = None,
) -> ViewSet:
    """Build every view for one (light curve, ephemeris).

    Parameters
    ----------
    lc              : cleaned and flattened light curve.
    period, t0      : ephemeris [days]; `t0` in the light curve's own time system.
    duration        : full transit duration [days], not hours.
    trend_lc        : the same target *before* flattening. The trend view shows
                      what detrending removed — a transit the spline absorbed
                      shows up here and nowhere else. Omitted, the branch is all
                      zeros with `present` 0, which is the honest encoding of
                      "not available" rather than "flat".
    raw_lc          : the *unprocessed* curve, for the centroid branch —
                      `clean_lightcurve` drops `MOM_CENTR1/2`, so it cannot be
                      recovered downstream. Omitted, that branch reads as absent
                      rather than as flat.
    momentum_dumps  : flagged cadence times from `momentum_dumps.parquet`, in
                      the light curve's own time system. Omitted, that branch is
                      all zeros with `present` 0 — correct for Kepler and K2,
                      which never saw a TESS reaction wheel, and the honest
                      encoding for a TESS row whose dump table was not supplied.
                      Folded on `raw_lc`'s cadence grid when one is given, so it
                      counts the cadences the target actually had rather than
                      the ones that survived cleaning.
    difference_images : this target's per-sector DV difference images. Omitted,
                      the branch is all zeros with `present` 0 — which is the
                      right encoding for the majority of the set, since Kepler
                      and K2 have no DV report at all. A target that *has* a
                      report but whose every sector DV declined also lands here,
                      and both are distinct from a stamp measured flat.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"invalid period: {period}")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration: {duration}")

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    half = _local_window(duration, period, local_durations)
    phase = _phase_of(time, period, t0)

    global_view = _binned_view(phase, flux, global_bins)

    # The primary's depth is the reference scale for every comparison view
    # below, so their depths stay meaningful relative to it.
    primary = bin_profile(phase, flux, local_bins, -half, half)
    primary_depth = _depth_of(primary.median)
    local_view = _normalise_pair(primary.median, primary.scatter, primary.count)

    # Odd vs even transits: a depth difference between them is the classic
    # eclipsing-binary signature, where the "period" is really half the true one.
    epochs = _transit_index(time, period, t0)
    odd = epochs % 2 == 1
    odd_view = _binned_view(phase[odd], flux[odd], local_bins, half, primary_depth)
    even_view = _binned_view(phase[~odd], flux[~odd], local_bins, half, primary_depth)

    sec_phase = _secondary_phase(phase, flux, half)
    sec_centred = (phase - sec_phase + 0.5) % 1.0 - 0.5
    secondary_view = _binned_view(sec_centred, flux, local_bins, half, primary_depth)

    if trend_lc is not None:
        trend_time = np.asarray(trend_lc.time.value, dtype=float)
        trend_flux = np.asarray(trend_lc.flux.value, dtype=float)
        trend_finite = np.isfinite(trend_time) & np.isfinite(trend_flux)
        trend_view = _binned_view(
            _phase_of(trend_time[trend_finite], period, t0),
            trend_flux[trend_finite],
            global_bins,
        )
    else:
        trend_view = _empty_view(global_bins)

    unfolded_view, observed, expected = _unfolded(
        time, flux, period, t0, duration, local_durations, max_transits, primary_depth
    )

    if raw_lc is not None:
        centroid_view = _centroid_view(raw_lc, period, t0, half, local_bins)
        gap_view = _gap_view(lc, period, t0, global_bins)
    else:
        centroid_view = _empty_view(local_bins)
        gap_view = _gap_view(lc, period, t0, global_bins)

    periodogram_view = _periodogram_view(lc, periodogram_bins)
    periodogram_masked_view = _periodogram_view(
        _mask_transits(lc, period, t0, duration), periodogram_bins
    )

    if difference_images:
        difference_view, difference_quality_view = build_difference_views(difference_images)
    else:
        difference_view, difference_quality_view = empty_difference_views()

    if momentum_dumps is not None and len(momentum_dumps):
        # The *raw* cadence grid, matching `_centroid_view`'s reason for taking
        # one: a dump fraction is a count over cadences, and cleaning removes
        # some, so measuring the denominator on the cleaned curve would report a
        # fraction of a grid the spacecraft never flew.
        cadence_source = raw_lc if raw_lc is not None else lc
        momentum_dump_view = build_momentum_dump_view(
            np.asarray(cadence_source.time.value, dtype=float),
            momentum_dumps,
            period=period,
            t0=t0,
            half_window=half,
            n_bins=local_bins,
        )
    else:
        momentum_dump_view = empty_momentum_dump_view(local_bins)

    return ViewSet(
        global_view=global_view,
        local_view=local_view,
        odd_view=odd_view,
        even_view=even_view,
        secondary_view=secondary_view,
        trend_view=trend_view,
        unfolded_view=unfolded_view,
        centroid_view=centroid_view,
        gap_view=gap_view,
        periodogram_view=periodogram_view,
        periodogram_masked_view=periodogram_masked_view,
        difference_view=difference_view,
        difference_quality_view=difference_quality_view,
        momentum_dump_view=momentum_dump_view,
        observed_transit_count=observed,
        expected_transit_count=expected,
        secondary_phase=sec_phase,
    )
