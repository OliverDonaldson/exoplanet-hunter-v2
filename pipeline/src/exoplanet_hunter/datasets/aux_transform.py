"""Aux-feature normalisation: one fitted sklearn pipeline, two execution paths.

The pipeline (median impute → log1p on the centroid column → standardise) is
fitted per fold on training rows and persisted in the calibration bundle —
that is the serving contract `score_target` relies on. Training, however,
streams examples through tf.data where a pickled sklearn object can't run,
so `tf_aux_transform` replays the *fitted constants* (medians, means, stds)
as tensor ops. `aux_constants_from_pipeline` is the only bridge between the
two, and `tests/test_datasets.py` pins their outputs to be identical — the
train/serve-skew guard the V2 doc calls for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: Index of centroid_snr in the 9-dim aux vector (absent in legacy 8-dim).
CENTROID_COL = 8


def _log1p_centroid(X: np.ndarray) -> np.ndarray:
    """log1p the centroid_snr column only.

    Centroid_snr is heavy-tailed on the FP cohort (q90=423, max=10436) while
    the planet body sits around ~1.1; without compression StandardScaler fits
    the tail and squashes the bulk. Module-level (not a lambda) so persisted
    pipelines unpickle anywhere.
    """
    if X.shape[1] <= CENTROID_COL:
        return X
    X = X.copy()
    X[:, CENTROID_COL] = np.log1p(X[:, CENTROID_COL])
    return X


def fit_aux_pipeline(train_aux: np.ndarray) -> Pipeline:
    """Fit impute → log1p(centroid) → standardise on training-fold aux rows.

    `keep_empty_features=True` is load-bearing: without it, a column that is
    all-NaN in some refresh (e.g. snr on a small fresh build) gets silently
    *dropped* by the imputer, the scaler fits one dimension short, and both
    the tf replay and the model input shape break. With it, dead columns
    impute to 0, standardise to 0 (unit scale on zero variance), and the aux
    dimension is stable across catalogue refreshes.
    """
    dead = np.isnan(train_aux).all(axis=0)
    if dead.any():
        log.warning(
            "[aux] columns %s are all-NaN in this training fold — imputing to a "
            "constant 0 (dead input). Check the catalogue build if unexpected.",
            np.where(dead)[0].tolist(),
        )
    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log_centroid", FunctionTransformer(_log1p_centroid, validate=False)),
            ("scale", StandardScaler()),
        ]
    )
    pipeline.fit(train_aux)
    return pipeline


@dataclass(frozen=True)
class AuxConstants:
    """Fitted constants extracted from the sklearn pipeline, for tf replay."""

    medians: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    aux_dim: int


def aux_constants_from_pipeline(pipeline: Pipeline) -> AuxConstants:
    imputer: SimpleImputer = pipeline.named_steps["impute"]
    scaler: StandardScaler = pipeline.named_steps["scale"]
    return AuxConstants(
        medians=tuple(float(v) for v in imputer.statistics_),
        means=tuple(float(v) for v in scaler.mean_),
        stds=tuple(float(v) for v in scaler.scale_),
        aux_dim=len(imputer.statistics_),
    )


def tf_aux_transform(aux: tf.Tensor, constants: AuxConstants) -> tf.Tensor:
    """Replay the fitted pipeline on one (aux_dim,) tensor.

    Order matches the sklearn pipeline exactly: impute NaNs with training
    medians, log1p the centroid column, then (x - mean) / std.
    """
    medians = tf.constant(constants.medians, dtype=tf.float32)
    means = tf.constant(constants.means, dtype=tf.float32)
    stds = tf.constant(constants.stds, dtype=tf.float32)

    x = tf.where(tf.math.is_nan(aux), medians, aux)
    if constants.aux_dim > CENTROID_COL:
        # tf.where selects rather than blends: log1p may be NaN on other
        # columns (they can legitimately be < -1), but those lanes are never
        # taken. centroid_snr itself is always >= 0.
        is_centroid = tf.range(constants.aux_dim) == CENTROID_COL
        x = tf.where(is_centroid, tf.math.log1p(x), x)
    return (x - means) / stds
