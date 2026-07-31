"""ExoMiner-grade view set: per-diagnostic views at 301/31 bins.

Stage 1 of the rebuild. Separate from `views.py` on purpose — that module's
2001/201 pair feeds the live `ca906040` model and must not move while this is
built.

Every view here is **(bins, 3)**: `[flux, scatter, present]`.

- `flux` is the median in each bin, baseline-subtracted and depth-normalised
  exactly as `views._normalise` does, so shape is what the model sees.
- `scatter` is the per-bin MAD on the *same* scale. Dividing it by the same
  depth matters: an unscaled scatter channel next to a normalised flux channel
  would make the model's notion of "noisy" depend on transit depth.
- `present` is 1 where the bin held a cadence and 0 where it did not. Without
  it, an empty bin filled to baseline is indistinguishable from a bin measured
  to be flat — and a transit falling in a data gap reads as no transit. The
  same reason every branch here carries a presence mask: Kepler has DV, K2 has
  none, FFI differs again, and a missing branch that looks like a measured zero
  poisons every row of its mission.

The unfolded branch is the one aimed squarely at the measured failure: score
correlates +0.211 with observation baseline and −0.003 with transit count, so
the model is reading how long a target was watched rather than how often it
dipped. Individual transits, plus observed/expected counts, are what make
repetition explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exoplanet_hunter.preprocess.fold import bin_profile

if TYPE_CHECKING:
    import lightkurve as lk

GLOBAL_BINS = 301
LOCAL_BINS = 31
LOCAL_DURATIONS = 3.0
MAX_TRANSITS = 20


@dataclass(frozen=True)
class ViewSet:
    """One target's views. Every array is (bins, 3) = [flux, scatter, present]."""

    global_view: np.ndarray  # (301, 3) full phase
    local_view: np.ndarray  # (31, 3)  ±3 durations around the transit
    odd_view: np.ndarray  # (31, 3)  odd-numbered transits only
    even_view: np.ndarray  # (31, 3)  even-numbered transits only
    secondary_view: np.ndarray  # (31, 3)  centred on the deepest non-primary phase
    trend_view: np.ndarray  # (301, 3) what the detrending removed
    #: (MAX_TRANSITS, 31, 3) — individual transits, newest bins last.
    unfolded_view: np.ndarray
    #: Transits with any cadence coverage, and how many the ephemeris predicts
    #: over the observed baseline. Their ratio is the completeness the folded
    #: views cannot express.
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

    Baseline at 0 and, by default, the deepest bin at -1 — so the model sees
    transit *shape* rather than magnitude. Scatter is divided by the same depth
    so the two channels stay commensurate.

    `depth` overrides that scale, and the comparison views **must** pass the
    primary's. Normalising the odd and even views each by their own depth sends
    both to exactly -1 and destroys the depth *difference* between them, which
    is the whole eclipsing-binary signature the branch exists to catch. The
    same argument applies to the secondary view (a shallow secondary would
    otherwise look as deep as the primary) and to each unfolded transit (depth
    varying transit to transit is what a blend looks like).
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

    Expected counts every epoch the ephemeris predicts inside the observed
    baseline; observed counts those that actually caught a cadence in-window.
    A single-transit candidate and a 40-transit one are indistinguishable once
    folded, which is exactly the blind spot being fixed.
    """
    stack = np.zeros((max_transits, LOCAL_BINS, 3), dtype=np.float32)
    if time.size == 0:
        return stack, 0, 0

    half_days = durations * duration
    epochs = _transit_index(time, period, t0)
    first, last = int(epochs.min()), int(epochs.max())
    expected = last - first + 1

    observed = 0
    for epoch in range(first, last + 1):
        centre = t0 + epoch * period
        window = np.abs(time - centre) <= half_days
        if not window.any():
            continue
        if observed < max_transits:
            offset = (time[window] - centre) / (2.0 * half_days)  # -0.5 .. 0.5
            stack[observed] = _binned_view(offset, flux[window], LOCAL_BINS, depth=depth)
        observed += 1
    return stack, observed, expected


def build_view_set(
    lc: lk.LightCurve,
    *,
    period: float,
    t0: float,
    duration: float,
    trend_lc: lk.LightCurve | None = None,
    global_bins: int = GLOBAL_BINS,
    local_bins: int = LOCAL_BINS,
    local_durations: float = LOCAL_DURATIONS,
    max_transits: int = MAX_TRANSITS,
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

    return ViewSet(
        global_view=global_view,
        local_view=local_view,
        odd_view=odd_view,
        even_view=even_view,
        secondary_view=secondary_view,
        trend_view=trend_view,
        unfolded_view=unfolded_view,
        observed_transit_count=observed,
        expected_transit_count=expected,
        secondary_phase=sec_phase,
    )
