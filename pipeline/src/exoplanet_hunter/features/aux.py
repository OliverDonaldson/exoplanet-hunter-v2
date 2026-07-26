"""The aux feature vector layout — one implementation for every caller.

Training (build_dataset), serving (scoring.service) and batch scoring
(score_candidates) each used to spell this layout out separately, which is how
score_candidates ended up stuck on the legacy 9-dim vector after the model
moved to 13. The layout lives here now; callers supply the values.

    idx  0-3  teff, radius, logg, tmag      stellar context
    idx  4-6  depth, duration, log(period)  transit shape
    idx  7    pink_snr (>=13) | catalogue snr (8/9)
    idx  8    centroid_snr                  BEB diagnostic (>=9)
    idx  9-12 oe_depth_sigma, oe_timing_sigma, secondary_sig, q_ratio (>=13)

pink_snr (Kunimoto 2025 §2.1) replaces the catalogue transit SNR at idx 7 in
the 13-dim layout: it is computed from the light curve, so it exists for every
target at train *and* serve time, closing the non-TOI NaN->imputed mismatch.
"""

from __future__ import annotations

import numpy as np

#: The fitted aux pipeline indexes centroid here — it must not move.
CENTROID_COL = 8

#: Width of the vector the current training build emits.
TRAINING_AUX_DIM = 13

#: The 9-dim layout of promoted runs that predate the vetting features
#: (the live ca906040 serves it; the aux_dim branch keeps them loadable).
LEGACY_AUX_DIM = 9

AUX_NAMES_13 = (
    "teff",
    "radius",
    "logg",
    "tmag",
    "depth",
    "duration",
    "log_period",
    "pink_snr",
    "centroid_snr",
    "oe_depth_sigma",
    "oe_timing_sigma",
    "secondary_sig",
    "q_ratio",
)


def _f(value: object) -> float:
    """Coerce to float, mapping None/NA/unset to NaN."""
    if value is None:
        return float("nan")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


def build_aux_row(
    aux_dim: int,
    *,
    teff: object = None,
    radius: object = None,
    logg: object = None,
    tmag: object = None,
    depth: object = None,
    duration: object = None,
    period: object = None,
    centroid_snr: object = None,
    pink_snr: object = None,
    catalogue_snr: object = None,
    oe_depth_sigma: object = None,
    oe_timing_sigma: object = None,
    secondary_sig: object = None,
    q_ratio: object = None,
) -> np.ndarray:
    """Assemble the aux vector for a model trained with `aux_dim` features.

    Anything unknown may be omitted; it becomes NaN and the fitted aux
    pipeline imputes it, exactly as at training time.
    """
    period_f = _f(period)
    row = [
        _f(teff),
        _f(radius),
        _f(logg),
        _f(tmag),
        _f(depth),
        _f(duration),
        float(np.log(period_f)) if period_f > 0 else float("nan"),
    ]
    if aux_dim >= 13:
        row += [
            _f(pink_snr),
            _f(centroid_snr),
            _f(oe_depth_sigma),
            _f(oe_timing_sigma),
            _f(secondary_sig),
            _f(q_ratio),
        ]
    else:
        row.append(_f(catalogue_snr))
        if aux_dim >= 9:
            row.append(_f(centroid_snr))
    return np.array(row, dtype=np.float32)
