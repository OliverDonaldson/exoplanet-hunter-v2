"""The inner train/validation split, group-aware *and* class-aware.

Both trainers cut a validation split inside each CV fold, and that split feeds
early stopping **and** the Platt fit. `GroupShuffleSplit` is group-aware only:
it keeps one host's planets on a single side, which is the leakage rule, but it
makes no promise about class balance.

Calibration requires both classes. With every validation label identical the
NLL is minimised by pushing the fit to its extremes, so the optimiser converges
happily and returns a scaler that maps every score to one end — and that bundle
is written to disk as the *servable* calibrator.

`calibration._assert_both_classes` raises on that, which turns a silent bad
artefact into a failed run. This module removes the cause rather than catching
it: `StratifiedGroupKFold` keeps the grouping guarantee and adds the balance.

Applying it **changes the inner partition and therefore the numbers**, so it
belongs between experiments rather than between a run and its own control.
Landed 2026-08-08 alongside the unfolded-branch rebuild, which already forces a
fresh baseline — one re-baseline covering both changes rather than two.

Not to be confused with the *outer* split. `train.py`'s `GroupShuffleSplit` at
the random-forest holdout is a train/test cut that nothing calibrates on, and
is deliberately left alone.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

__all__ = ["stratified_inner_split"]

#: A split needs at least two parts before anything can be held out.
MIN_INNER_SPLITS = 2


def _assert_both_classes(labels: np.ndarray, side: str, val_frac: float) -> None:
    """A single-class side is a failed split, not a small one."""
    present = np.unique(labels)
    if len(present) < 2:
        raise ValueError(
            f"the inner {side} split holds only class {present.tolist()} over {len(labels)} "
            f"rows at val_frac={val_frac}; early stopping and the Platt fit both need "
            "both classes, and the fit would collapse every probability to one end"
        )


def stratified_inner_split(
    trainval: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one fold's train+validation rows into train and validation.

    Parameters
    ----------
    trainval : positions into `y` and `groups` that this fold may train on.
    y        : the full label vector.
    groups   : the full grouping vector, `tic_id` in both trainers.
    val_frac : share of `trainval` to hold out, as `1 / n_splits`.
    seed     : the fold's own seed.

    Returns positions into `y`, **not** offsets into `trainval`. The call sites
    used to index `trainval[tr_rel]` themselves, which is one more place for the
    frame of reference to be got wrong.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    n_splits = max(MIN_INNER_SPLITS, round(1.0 / val_frac))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_rel, val_rel = next(splitter.split(trainval, y[trainval], groups[trainval]))
    train_idx, val_idx = trainval[train_rel], trainval[val_rel]

    _assert_both_classes(y[val_idx], "validation", val_frac)
    _assert_both_classes(y[train_idx], "training", val_frac)
    return train_idx, val_idx
