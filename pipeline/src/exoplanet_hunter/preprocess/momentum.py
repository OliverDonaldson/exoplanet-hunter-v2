"""The momentum-dump view — TESS reaction-wheel desaturation, folded on the transit.

TESS spins its reaction wheels down on a schedule (every 2.5 days in the first
sectors, later every 5.5 and eventually rarer). The spacecraft's pointing moves
while it happens, and the flux of every target on the focal plane moves with it.
If those cadences land at the candidate's transit phase, the "transit" is partly
the spacecraft. ExoMiner++ is the only model of the eleven in their Table 1 that
carries this input, and it is TESS-specific: there is no Kepler or K2 analogue,
so the view is absent by construction on 55.8% of our rows exactly as
`difference_view` is.

**Where the flag comes from, and why not from our own light curves.** The DQ bit
is 32, and it is zero on every cadence of all 6,192 cached TESS curves: they were
downloaded through lightkurve's default quality bitmask, which *removes* those
cadences before the file is written. `viewset.py::_gap_view` already recorded
that finding — "reading `QUALITY` bit 5 directly gives zero for every target in
the cache". `scripts/fetch_momentum_dumps.py` fetches the flag from unmasked
copies instead, one representative target per sector, which is sound because the
flag is a property of the spacecraft: measured 2026-08-27, four independent
sector-1 targets carry the identical 70 flagged timestamps.

**The cadences the dump removed are put back, at the target's own cadence.** The
dump cadences are missing from the target's own time array — that is the same
masking — so a fold over the surviving times would find no dumps anywhere and
the branch would be an all-zero input that looked like a feature. Each dump is
re-expanded over its own measured interval at the target's median cadence, so a
120-s target and a 200-s FFI target get the number of cadences each would
actually have lost, rather than the number the representative curve lost.

**Why not carry a variance channel beside the mean, as ExoMiner does.** Their
`local_momentum_dump_view_var` is the spread of a 0/1 flag within a bin, which
for a Bernoulli mean `p` is `p(1-p)` — a deterministic function of the channel
already there. It would be a second copy of the first channel, and this project
has spent two stages removing inputs that looked like measurements and were not.
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

    Parameters
    ----------
    time        : the target's own cadence times [BTJD], dumps already removed.
    dump_times  : flagged cadence times from `momentum_dumps.parquet`.
    period, t0  : the ephemeris the *other* views were folded on. It must be that
                  one: a momentum view folded on a different epoch would put the
                  dumps at a phase the flux views disagree with, and the branch's
                  whole question is whether the dumps sit under the transit.
    half_window : half-width of the local window in phase units — `viewset.
                  _local_window(duration, period, LOCAL_DURATIONS)`.

    Channel 0 is the fraction of the cadences known at that phase that were
    dump-flagged; channel 1 marks bins that held any cadence at all. A bin with
    no cadence reads 0 with presence 0, never 0 with presence 1 — the
    distinction every view in this package exists to keep.
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
