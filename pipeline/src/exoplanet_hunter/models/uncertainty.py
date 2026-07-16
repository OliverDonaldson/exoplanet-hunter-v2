"""Uncertainty estimation via MC Dropout (Gal & Ghahramani 2016).

Standard Keras dropout is disabled at inference. By calling the model with
`training=True` repeatedly and averaging predictions, we sample from the
*posterior predictive distribution* — giving us both a mean prediction and
a calibrated standard deviation.

This is critical when claiming "this signal could be a planet": a high mean
score with low std is a confident detection; the same mean with a wide std is
a candidate that needs follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class UncertaintyResult:
    mean: np.ndarray  # (N,)  mean predicted probability
    std: np.ndarray  # (N,)  std across MC samples
    samples: np.ndarray  # (T, N) raw samples


def mc_dropout_predict(
    model: tf.keras.Model,
    inputs: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray,
    n_samples: int = 50,
) -> UncertaintyResult:
    """Run T forward passes with dropout active and return mean + std.

    Parameters
    ----------
    model     : a trained Keras model whose Dropout layers were built with
                `training=None` (so we can pass `training=True` here to keep
                them active).
    inputs    : the same inputs you'd pass to `model.predict`.
    n_samples : T, number of stochastic forward passes. 50 is a good default;
                100+ for tighter intervals.

    For a single example the T samples are drawn in one batched forward pass
    (dropout masks are independent per batch element, so this is equivalent) —
    T sequential single-example calls pay T rounds of dispatch overhead, which
    dominates on CPU serving.
    """

    def _batch_size(x) -> int:
        arr = (
            next(iter(x.values())) if isinstance(x, dict) else (x[0] if isinstance(x, list) else x)
        )
        return int(np.asarray(arr).shape[0])

    if _batch_size(inputs) == 1:
        tiled: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray
        if isinstance(inputs, dict):
            tiled = {k: np.repeat(np.asarray(v), n_samples, axis=0) for k, v in inputs.items()}
        elif isinstance(inputs, list):
            tiled = [np.repeat(np.asarray(v), n_samples, axis=0) for v in inputs]
        else:
            tiled = np.repeat(np.asarray(inputs), n_samples, axis=0)
        samples = np.asarray(model(tiled, training=True)).reshape(n_samples)
    else:
        samples = np.stack(
            [np.asarray(model(inputs, training=True)).squeeze() for _ in range(n_samples)],
            axis=0,
        )
    return UncertaintyResult(
        mean=samples.mean(axis=0),
        std=samples.std(axis=0),
        samples=samples,
    )
