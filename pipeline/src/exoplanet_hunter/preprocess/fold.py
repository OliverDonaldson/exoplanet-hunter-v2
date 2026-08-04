"""Phase-folding and binning for transit search.

Folding overlays every transit so the signal adds. `bin_profile` is the
primitive, returning per-bin median, scatter and count in one pass; the scatter
is the paired variance channel the new views feed alongside each flux channel.
`fold_and_bin` is the median-only wrapper the existing 2001/201 views use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class BinnedProfile:
    """Per-bin median, scatter and occupancy over a phase window."""

    centers: np.ndarray
    median: np.ndarray
    #: Median absolute deviation, scaled to a Gaussian sigma. NaN in bins with
    #: fewer than two points — one cadence has no scatter, and reporting 0
    #: there would read as "perfectly consistent" rather than "unmeasured".
    scatter: np.ndarray
    count: np.ndarray


def bin_profile(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
) -> BinnedProfile:
    """Bin (phase, flux) into `n_bins` over [phase_min, phase_max].

    Empty bins stay NaN rather than being interpolated: `np.interp` would
    manufacture smooth signal across data gaps, making a transit that falls
    inside a gap invisible. Callers decide how to fill.
    """
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    keep = (phase >= phase_min) & (phase <= phase_max) & np.isfinite(flux)
    phase, flux = phase[keep], flux[keep]

    edges = np.linspace(phase_min, phase_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(n_bins, np.nan)
    scatter = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int32)
    if phase.size == 0:
        return BinnedProfile(centers, median, scatter, count)

    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    # Sort once and split, rather than masking the whole array per bin: the old
    # per-bin mask was O(n_bins * n_points), which at 2001 bins over a
    # multi-sector curve is tens of millions of comparisons per view.
    order = np.argsort(idx, kind="stable")
    idx_sorted, flux_sorted = idx[order], flux[order]
    boundaries = np.searchsorted(idx_sorted, np.arange(n_bins + 1))
    for b in range(n_bins):
        chunk = flux_sorted[boundaries[b] : boundaries[b + 1]]
        count[b] = chunk.size
        if chunk.size:
            median[b] = np.median(chunk)
        if chunk.size > 1:
            scatter[b] = 1.4826 * np.median(np.abs(chunk - median[b]))
    return BinnedProfile(centers, median, scatter, count)


def fold_and_bin(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    n_bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold and bin a light curve.

    Returns
    -------
    bin_centers : phase values [phase_min, phase_max] of bin centres.
    binned_flux : median flux in each bin (NaN if empty).
    """
    profile = fold_to_profile(lc, period, t0, n_bins, phase_min, phase_max)
    return profile.centers, profile.median


def fold_to_profile(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    n_bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
) -> BinnedProfile:
    """Phase-fold a light curve and bin it into a `BinnedProfile`."""
    folded = lc.fold(period=period, epoch_time=t0)
    return bin_profile(
        np.asarray(folded.time.value, dtype=float),
        np.asarray(folded.flux.value, dtype=float),
        n_bins,
        phase_min,
        phase_max,
    )
