"""Stochastic per-example augmentation, ported from the V1 LightcurveDataset.

Semantics preserved exactly:

  * **Coherent phase shift** (±time_shift_frac of the view length) — the same
    random fraction rolls both views, because they are the same star at the
    same moment.
  * **Independent Gaussian noise** per view (sensor noise is uncorrelated
    between the two binnings).
  * **Coherent depth scaling** (±scale_range) — simulates depth/variability
    changes, again shared across views.
  * **Independent bin masking** (mask_prob) — simulates missing cadences.

These run *after* `cache()` in the input pipeline so every epoch sees fresh
draws (L6: cache the deterministic work, keep the stochastic work live).
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class AugmentConfig:
    time_shift_frac: float = 0.005
    noise_std: float = 0.0005
    scale_range: float = 0.05
    mask_prob: float = 0.02


def augment_views(g: tf.Tensor, l: tf.Tensor, cfg: AugmentConfig) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment one (global, local) pair; inputs are (bins, 1) float32."""
    shift_frac = tf.random.uniform([], -cfg.time_shift_frac, cfg.time_shift_frac)
    g_n = tf.cast(tf.shape(g)[0], tf.float32)
    l_n = tf.cast(tf.shape(l)[0], tf.float32)
    g = tf.roll(g, shift=tf.cast(shift_frac * g_n, tf.int32), axis=0)
    l = tf.roll(l, shift=tf.cast(shift_frac * l_n, tf.int32), axis=0)

    g = g + tf.random.normal(tf.shape(g), stddev=cfg.noise_std)
    l = l + tf.random.normal(tf.shape(l), stddev=cfg.noise_std)

    if cfg.scale_range > 0:
        scale = tf.random.uniform([], 1.0 - cfg.scale_range, 1.0 + cfg.scale_range)
        g = g * scale
        l = l * scale

    if cfg.mask_prob > 0:
        g = g * tf.cast(tf.random.uniform(tf.shape(g)) > cfg.mask_prob, tf.float32)
        l = l * tf.cast(tf.random.uniform(tf.shape(l)) > cfg.mask_prob, tf.float32)

    return g, l
