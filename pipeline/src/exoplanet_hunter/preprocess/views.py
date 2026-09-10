"""Global + local view extraction (Shallue & Vanderburg 2018).

For each (light curve, period, t0, duration) we produce two arrays:

  * **global view** — full phase-folded light curve at low resolution
    (default 2001 bins). Captures the planet's overall orbital phase
    relative to the star, including any secondary eclipse signature.
  * **local view** — zoomed-in window around phase 0 spanning
    ±N transit durations (default 3). Captures the transit shape at
    high resolution.

Both views are median-normalised so flux=0 corresponds to the out-of-transit
baseline and the transit dip is negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from exoplanet_hunter.preprocess.clean import flatten_lightcurve
from exoplanet_hunter.preprocess.fold import fold_and_bin

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class Views:
    global_view: np.ndarray
    local_view: np.ndarray


def _normalise(view: np.ndarray) -> np.ndarray:
    """Median-subtract and depth-divide.

    Subtracting the median puts the baseline at 0; dividing by |min - median| scales
    the deepest dip to -1 whatever the absolute depth, so the model sees transit
    *shape*, not magnitude.

    Nan-aware median and min keep empty bins — from long gaps after fold-and-bin —
    from poisoning the view, and any NaN left after normalisation is filled with 0,
    the baseline, which reads as "no flux deviation here". An all-NaN input raises:
    that is a fundamentally bad target for `build_dataset.py` to count as a
    preprocess error, rather than an all-NaN row poisoning gradients.
    """
    if not np.isfinite(view).any():
        raise ValueError("view is entirely NaN — no usable cadences after folding")
    med = float(np.nanmedian(view))
    centred = view - med
    depth = float(np.abs(np.nanmin(centred)))
    if depth < 1.0e-8:
        return np.nan_to_num(centred, nan=0.0)
    return np.nan_to_num(centred / depth, nan=0.0)


def build_views(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
    *,
    global_bins: int = 2001,
    local_bins: int = 201,
    local_durations: float = 3.0,
) -> Views:
    """Build the global + local views for a single (lc, period, t0, duration).

    `lc` is flattened and cleaned, `period` and `duration` are days (not hours), and
    `t0` is the transit midpoint in BTJD. `local_durations` is the half-width of the
    local window in transit durations.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"invalid period: {period}")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration: {duration}")

    # ----- global ---------------------------------------------------------
    _, gview = fold_and_bin(
        lc,
        period=period,
        t0=t0,
        n_bins=global_bins,
        phase_min=-0.5,
        phase_max=0.5,
    )

    # ----- local ----------------------------------------------------------
    half = local_durations * duration / period  # half-window in phase units
    half = float(min(max(half, 1e-3), 0.5))  # clamp to a sane range

    _, lview = fold_and_bin(
        lc,
        period=period,
        t0=t0,
        n_bins=local_bins,
        phase_min=-half,
        phase_max=+half,
    )

    return Views(
        global_view=_normalise(gview).astype(np.float32),
        local_view=_normalise(lview).astype(np.float32),
    )


def flatten_and_build_views(
    cleaned_lc: lk.LightCurve,
    *,
    period: float,
    t0: float,
    duration: float,
    preprocess_cfg: Any,
) -> Views:
    """Mask-flatten a cleaned light curve at a known ephemeris, then bin into views.

    The shared inference-time preprocessing tail used by both the API's
    `TargetScorer` and the bulk scorer: the transit is masked out of the
    Savitzky-Golay fit so the spline cannot absorb the dip, then the masked-flat
    curve is phase-folded into (global, local) views. One place, so inference
    preprocessing has a single source of truth.

    `preprocess_cfg` is the Hydra `preprocess` config node, mirroring the
    `model_cfg: Any` convention of `build_cnn_dualview`. The caller cleans first,
    since some callers also need the cleaned curve for an unmasked BLS search.
    """
    lc = flatten_lightcurve(
        cleaned_lc,
        window_length=int(preprocess_cfg.flatten.window_length),
        polyorder=int(preprocess_cfg.flatten.polyorder),
        period=period,
        t0=t0,
        duration=duration,
    )
    return build_views(
        lc,
        period=period,
        t0=t0,
        duration=duration,
        global_bins=int(preprocess_cfg.views.global_bins),
        local_bins=int(preprocess_cfg.views.local_bins),
        local_durations=float(preprocess_cfg.views.local_durations),
    )
