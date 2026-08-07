"""Stochastic augmentation for the view set — `augment.py`'s semantics, per view.

The incumbent augments two `(bins, 1)` tensors. The view set has eleven views of
`(bins, channels)` plus a `(20, 31, 3)` unfolded stack, so an exact reuse is not
possible. What is preserved is the operation set, their magnitudes and their
order: coherent phase shift, independent Gaussian noise, coherent depth scale,
independent bin masking.

Two things about the view set change what "per view" means.

**The presence channel is never augmented.** It is the last channel on every
view, and `_gated` in `cnn_branches.py` reads `reduce_max(present) > 0` to
decide whether a branch contributes at all. Noise on a genuinely-absent
branch's zeros flips it to "present" and silently disables the gating the whole
design rests on — for Kepler and K2, where `dv_usable` is 0%, that is 3,027 of
5,423 rows. Every op here writes to data channels only, and noise is multiplied
by `present` so a bin that held no cadence keeps the zero that says so.

**Not every axis is phase.** The periodogram views are indexed by *period*, so
rolling them shifts the peak onto a different period rather than shifting the
star in time. They take noise and masking but no phase shift. The unfolded
stack's phase axis is its second-to-last; axis 0 is transit number, and rolling
that would reorder the transits.

Which view is which lives in `VIEW_KINDS`, keyed by name — a view added to the
shard writer without a kind here raises rather than being silently augmented as
if it were folded flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import tensorflow as tf

from exoplanet_hunter.datasets.augment import AugmentConfig

__all__ = ["VIEW_KINDS", "AugmentConfig", "ViewKind", "augment_viewset"]


class ViewKind(Enum):
    """What a view's bin axis means, and whether its data scales with depth."""

    #: Phase-binned and depth-normalised: [flux, scatter, present].
    FOLDED_FLUX = "folded_flux"
    #: Phase-binned, one stack entry per transit: (transits, phase, 3).
    UNFOLDED_FLUX = "unfolded_flux"
    #: Phase-binned but not on the depth scale — a fraction in [0, 1].
    PHASE_FRACTION = "phase_fraction"
    #: Period-indexed and peak-normalised; a phase shift is meaningless here.
    PERIODOGRAM = "periodogram"


VIEW_KINDS: dict[str, ViewKind] = {
    "global_view": ViewKind.FOLDED_FLUX,
    "local_view": ViewKind.FOLDED_FLUX,
    "odd_view": ViewKind.FOLDED_FLUX,
    "even_view": ViewKind.FOLDED_FLUX,
    "secondary_view": ViewKind.FOLDED_FLUX,
    "trend_view": ViewKind.FOLDED_FLUX,
    "centroid_view": ViewKind.FOLDED_FLUX,
    "unfolded_view": ViewKind.UNFOLDED_FLUX,
    "gap_view": ViewKind.PHASE_FRACTION,
    "periodogram_view": ViewKind.PERIODOGRAM,
    "periodogram_masked_view": ViewKind.PERIODOGRAM,
}

#: Views whose bin axis is phase, so the coherent shift applies.
_SHIFTED = (ViewKind.FOLDED_FLUX, ViewKind.UNFOLDED_FLUX, ViewKind.PHASE_FRACTION)
#: Views whose data channels are on the primary's depth scale, so the coherent
#: depth scaling applies. A gap fraction and a peak-normalised periodogram are
#: not depths and do not move when the transit gets deeper.
_DEPTH_SCALED = (ViewKind.FOLDED_FLUX, ViewKind.UNFOLDED_FLUX)
#: Views where zero is the *neutral* value, so dropping a bin to it removes a
#: measurement rather than asserting a different one. On folded flux, zero is
#: the out-of-transit baseline and masking reads as "nothing to see here".
#: Elsewhere it is a claim: zero in a gap fraction says **no cadence was
#: missing**, and zero in a peak-normalised periodogram says **no power at this
#: period** — both stated confidently while `present` still says the bin was
#: measured. That is the flip this module's docstring exists to prevent, applied
#: by the module itself. Deliberately its own constant rather than reusing
#: `_DEPTH_SCALED`: the two sets coincide today for unrelated reasons.
_MASKABLE = (ViewKind.FOLDED_FLUX, ViewKind.UNFOLDED_FLUX)


@dataclass(frozen=True)
class _Draw:
    """One example's shared random draws — the coherent half of the augmentation."""

    shift_frac: tf.Tensor
    scale: tf.Tensor


def _phase_axis(kind: ViewKind) -> int:
    """The axis a phase shift rolls. For the unfolded stack it is the phase
    within each transit, not the transit index."""
    return -2 if kind is ViewKind.UNFOLDED_FLUX else 0


def _augment_view(view: tf.Tensor, kind: ViewKind, draw: _Draw, cfg: AugmentConfig) -> tf.Tensor:
    """One view, in `augment_views`' order: roll, noise, scale, mask."""
    data, present = view[..., :-1], view[..., -1:]

    if kind in _SHIFTED:
        axis = _phase_axis(kind)
        bins = tf.cast(tf.shape(view)[axis], tf.float32)
        # Truncation makes the shift a no-op below ~1/time_shift_frac bins: at
        # the default 0.005 it is +-10 bins on a 2001-bin view, +-1 on a 201-bin
        # one, and exactly 0 on a 31-bin one. The incumbent's 2001/201 gets the
        # augmentation it was tuned with; a coarser view set would silently lose
        # this op on its local views.
        shift = tf.cast(draw.shift_frac * bins, tf.int32)
        # The presence channel rolls with its data: which bins held a cadence
        # is a property of the phase grid, and leaving it behind would mark
        # measured bins empty. This moves presence, it does not create it.
        data = tf.roll(data, shift=shift, axis=axis)
        present = tf.roll(present, shift=shift, axis=axis)

    # Multiplied by presence: a bin that held no cadence is encoded as a zero
    # with present 0, and noise there invents a measurement that was never made.
    data = data + tf.random.normal(tf.shape(data), stddev=cfg.noise_std) * present

    if cfg.scale_range > 0 and kind in _DEPTH_SCALED:
        data = data * draw.scale

    if cfg.mask_prob > 0 and kind in _MASKABLE:
        # Masking drops a bin's measurement to the out-of-transit baseline,
        # exactly as the incumbent does. `present` is untouched: the bin was
        # measured, and claiming otherwise is the flip this module exists to
        # prevent. Gated on `_MASKABLE` because zero is only neutral on the flux
        # family — on a gap fraction or a periodogram it is a positive claim.
        keep = tf.random.uniform(tf.shape(data)[:-1]) > cfg.mask_prob
        data = data * tf.cast(keep, tf.float32)[..., None]

    return tf.concat([data, present], axis=-1)


def augment_viewset(views: dict[str, tf.Tensor], cfg: AugmentConfig) -> dict[str, tf.Tensor]:
    """Augment one example's views. Non-view entries (scalars, masks) pass through.

    The phase shift and the depth scale are drawn once and shared across views,
    because they are the same star at the same moment — the coherence the
    incumbent's `augment_views` gets from applying one draw to both of its views.
    """
    draw = _Draw(
        shift_frac=tf.random.uniform([], -cfg.time_shift_frac, cfg.time_shift_frac),
        scale=tf.random.uniform([], 1.0 - cfg.scale_range, 1.0 + cfg.scale_range),
    )
    out = dict(views)
    for name, view in views.items():
        if not name.endswith("_view"):
            continue
        kind = VIEW_KINDS.get(name)
        if kind is None:
            raise KeyError(
                f"{name!r} has no entry in VIEW_KINDS — augmenting it as folded flux "
                "would guess at whether its bin axis is phase and whether its data "
                "scales with depth"
            )
        out[name] = _augment_view(view, kind, draw, cfg)
    return out
