"""The momentum-dump view — TESS reaction-wheel desaturation, folded on the transit.

TESS spins its reaction wheels down on a schedule; pointing moves while it does,
and every target's flux moves with it. If those cadences land at the candidate's
transit phase, the "transit" is partly the spacecraft. The view is TESS-specific
and absent by construction on Kepler and K2 rows, as `difference_view` is.

The DQ bit is zero on every cached curve — lightkurve's default bitmask removes
those cadences before the file is written — so the flag comes from unmasked
copies, one target per sector, the flag being a property of the spacecraft. The
removed cadences are put back at the target's own median cadence, or a fold
would find no dumps and leave an all-zero input that looked like a feature.
Measurements: `docs/experiments/phase-1-build-4-2d.md`.
"""

from __future__ import annotations

import numpy as np

#: `[dump fraction, present]`. The bin count is **not** declared here: it is
#: `viewset.LOCAL_BINS`, passed in by the caller, for the reason `viewset_io`
#: gives about two declarations of one bin count drifting apart.
MOMENTUM_CHANNELS = 2

#: Largest gap inside one dump event, in days. Events are 7 cadences (~14 min)
#: and at least 2.5 days apart in the tightest sector, so any threshold between
#: those two separates them; 0.05 d sits three orders of magnitude clear of both
#: boundaries rather than close to either.
_EVENT_GAP_DAYS = 0.05

#: Fallback cadence when a target's own spacing cannot be measured, in days.
#: 120 s — the SPOC 2-min cadence 96% of our TESS rows carry.
_FALLBACK_CADENCE_DAYS = 120.0 / 86400.0


def empty_momentum_dump_view(n_bins: int) -> np.ndarray:
    """All zeros, presence 0 — the honest encoding of a row with no dump data.

    Every Kepler and K2 row, and any TESS row whose ephemeris or cadence grid
    could not be established. Distinct from a bin measured and found free of
    dumps, which carries presence 1 on a zero.
    """
    return np.zeros((n_bins, MOMENTUM_CHANNELS), dtype=np.float32)


def dump_events(dump_times: np.ndarray) -> list[tuple[float, float]]:
    """Group flagged cadences into `(start, end)` intervals, one per dump."""
    times = np.sort(np.asarray(dump_times, dtype=float))
    times = times[np.isfinite(times)]
    if times.size == 0:
        return []
    breaks = np.where(np.diff(times) > _EVENT_GAP_DAYS)[0]
    return [(float(group[0]), float(group[-1])) for group in np.split(times, breaks + 1)]


def _observed_segments(time: np.ndarray, gap_days: float) -> list[tuple[float, float]]:
    """Time spans the target was actually observed over."""
    from exoplanet_hunter.features.centroid import _segment_by_time_gaps

    return [
        (float(time[lo]), float(time[hi - 1]))
        for lo, hi in _segment_by_time_gaps(time, gap_days=gap_days)
        if hi > lo
    ]


def _cadence_of(time: np.ndarray) -> float:
    """The target's own sampling interval, from the spacing it actually has."""
    if time.size < 2:
        return _FALLBACK_CADENCE_DAYS
    spacing = float(np.median(np.diff(time)))
    return spacing if np.isfinite(spacing) and spacing > 0 else _FALLBACK_CADENCE_DAYS


def _restored_dump_cadences(
    time: np.ndarray, dump_times: np.ndarray, *, gap_days: float
) -> np.ndarray:
    """The cadences this target lost to dumps, at this target's own cadence.

    Only dumps inside a span the target was observed over are restored. A dump
    from a sector the target was not on is not a cadence it lost, and adding it
    would put a systematic at a phase where the star was not being watched.
    """
    cadence = _cadence_of(time)
    segments = _observed_segments(time, gap_days)
    restored: list[np.ndarray] = []
    for start, end in dump_events(dump_times):
        if not any(lo <= start <= hi or lo <= end <= hi for lo, hi in segments):
            continue
        count = max(round((end - start) / cadence) + 1, 1)
        restored.append(np.linspace(start, end, count))
    return np.concatenate(restored) if restored else np.empty(0, dtype=float)


def build_momentum_dump_view(
    time: np.ndarray,
    dump_times: np.ndarray,
    *,
    period: float,
    t0: float,
    half_window: float,
    n_bins: int,
    gap_days: float = 0.5,
) -> np.ndarray:
    """`(n_bins, 2)` = `[dump fraction, present]` over the local transit window.

    `time` is the target's own cadence times with dumps already removed, and
    `period`/`t0` must be the ephemeris the *other* views were folded on: folded on a
    different epoch the dumps would sit at a phase the flux views disagree with, and
    the branch's whole question is whether they fall under the transit.

    Channel 0 is the fraction of cadences known at that phase that were dump-flagged;
    channel 1 marks bins that held any cadence at all. A bin with no cadence reads 0
    with presence 0, never 0 with presence 1 — the distinction every view in this
    package exists to keep.
    """
    time = np.asarray(time, dtype=float)
    time = np.sort(time[np.isfinite(time)])
    if time.size < 2 or not np.isfinite([period, t0, half_window]).all() or period <= 0:
        return empty_momentum_dump_view(n_bins)

    dumps = _restored_dump_cadences(time, dump_times, gap_days=gap_days)
    # The observed cadences and the ones the dumps took out, together: the
    # denominator is every cadence this target either has or provably lost.
    all_times = np.concatenate([time, dumps])
    flagged = np.concatenate([np.zeros(time.size, dtype=bool), np.ones(dumps.size, dtype=bool)])

    phase = ((all_times - t0) / period + 0.5) % 1.0 - 0.5
    edges = np.linspace(-half_window, half_window, n_bins + 1)
    index = np.digitize(phase, edges) - 1
    inside = (index >= 0) & (index < n_bins)
    if not inside.any():
        return empty_momentum_dump_view(n_bins)

    total = np.bincount(index[inside], minlength=n_bins).astype(float)
    dumped = np.bincount(index[inside & flagged], minlength=n_bins).astype(float)
    fraction = np.divide(dumped, total, out=np.zeros(n_bins), where=total > 0)
    return np.stack([fraction, (total > 0).astype(float)], axis=-1).astype(np.float32)
