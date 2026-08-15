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

**The outer split can also be pinned from a file, and stage 10.5 needs it to
be.** Both trainers build their own `StratifiedGroupKFold` over their own shard
set, and the two sets are not the same population — `data/processed/tfrecords`
holds 5,380 examples against `viewset_tfrecords`' 5,426, sharing 5,375. No seed
makes two different populations partition alike, so an ensemble measured across
them is comparing two different out-of-fold populations, which is the defect
stage 10.5 exists to avoid rather than reproduce. `build_fold_assignment` writes
one group→fold map; `assigned_group_kfold` replays it in either trainer.

Rows whose group the map does not cover are **dropped**, not assigned somewhere
convenient. That is the pre-registered behaviour (roadmap 4.1a point 3): a joint
measurement costs both models the rows the other cannot see, and silently
keeping them would mean the two models were scored on different populations
again while the summary claimed otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "assigned_group_kfold",
    "assignment_mask",
    "build_fold_assignment",
    "load_fold_assignment",
    "stratified_inner_split",
    "write_fold_assignment",
]

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


# --------------------------------------------------- the pinned outer split --


def build_fold_assignment(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> dict[int, int]:
    """Map every group to the one fold that holds it out.

    The same `StratifiedGroupKFold` both trainers already use, resolved once so
    two trainers over two shard sets can be made to agree. A group lands in
    exactly one test fold by construction; this asserts that rather than
    assuming it, because the whole point of the artefact is that the two
    trainers partition identically.
    """
    if n_splits < MIN_INNER_SPLITS:
        raise ValueError(f"n_splits must be at least {MIN_INNER_SPLITS}, got {n_splits}")
    if len(groups) != len(y):
        raise ValueError(f"groups and y disagree on length: {len(groups)} vs {len(y)}")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment: dict[int, int] = {}
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        for group in np.unique(groups[test_idx]):
            key = int(group)
            if key in assignment:
                raise ValueError(
                    f"group {key} is held out by fold {assignment[key]} and fold {fold}; "
                    "a grouped split that splits a group is not a grouped split"
                )
            assignment[key] = fold

    unique = {int(g) for g in np.unique(groups)}
    if set(assignment) != unique:
        missed = sorted(unique - set(assignment))[:5]
        raise ValueError(f"{len(unique) - len(assignment)} group(s) reached no fold, e.g. {missed}")
    return assignment


def extend_fold_assignment(
    existing: dict[int, int],
    groups: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[dict[int, int], int]:
    """Keep every already-assigned group where it is; place only the new ones.

    This is what lets a *self-refreshing* model be compared to itself. Pinning a
    fixed map across refreshes would silently drop every target the catalogue
    gained since it was written, because uncovered groups are dropped rather
    than placed somewhere convenient. Rebuilding the map each refresh instead
    re-partitions the whole population, so a candidate and the incumbent it is
    gated against are scored on different splits, and part of any margin is only
    which rows landed where.

    Extending gives both: a target keeps the fold that has always held it out,
    so the shared population is compared like for like, and new targets still
    enter training.

    New groups go to the fold currently holding the fewest of their own class,
    ties broken on the lowest fold index, after a seeded shuffle. That keeps
    fold sizes and class balance even without ever moving a group — moving one
    would place a target in a fold that had already trained on it.

    Returns the extended map and the number of groups added.
    """
    if n_splits < MIN_INNER_SPLITS:
        raise ValueError(f"n_splits must be at least {MIN_INNER_SPLITS}, got {n_splits}")
    if len(groups) != len(y):
        raise ValueError(f"groups and y disagree on length: {len(groups)} vs {len(y)}")

    assignment = dict(existing)
    folds = list(range(n_splits))
    label_of = {int(g): int(label) for g, label in zip(groups, y, strict=True)}

    # Per-class occupancy of the map as it stands, so a new row lands where its
    # own class is thinnest rather than where the fold is merely smallest.
    counts: dict[int, dict[int, int]] = {}
    for group, fold in assignment.items():
        label = label_of.get(int(group))
        if label is None:
            continue
        per_fold = counts.setdefault(label, dict.fromkeys(folds, 0))
        per_fold[fold] = per_fold.get(fold, 0) + 1

    new_groups = sorted({int(g) for g in np.unique(groups)} - set(assignment))
    rng = np.random.default_rng(seed)
    rng.shuffle(new_groups)

    for group in new_groups:
        label = label_of[group]
        per_fold = counts.setdefault(label, dict.fromkeys(folds, 0))
        fold = min(folds, key=lambda f: (per_fold.get(f, 0), f))
        assignment[group] = fold
        per_fold[fold] = per_fold.get(fold, 0) + 1

    moved = [g for g, fold in existing.items() if assignment[g] != fold]
    if moved:
        raise ValueError(
            f"{len(moved)} group(s) changed fold while extending, e.g. {moved[:5]}; "
            "a group that moves is then scored by a fold that trained on it"
        )
    return assignment, len(new_groups)


def write_fold_assignment(path: Path, assignment: dict[int, int], **provenance: Any) -> None:
    """Persist a group→fold map with the provenance needed to trust it.

    JSON rather than parquet: this artefact decides which rows every downstream
    number was computed over, and it should be readable in a diff without a
    pandas session. At ~5,400 groups it is ~130 KB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "n_groups": len(assignment),
            "n_folds": len(set(assignment.values())),
            **provenance,
        },
        # String keys: JSON has no integer keys, and round-tripping through
        # `load_fold_assignment` is what the trainers rely on.
        "assignment": {str(k): int(v) for k, v in sorted(assignment.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_fold_assignment(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    """Read a group→fold map back, with its provenance block."""
    if not path.exists():
        raise FileNotFoundError(
            f"no fold assignment at {path}. Build one with `build_fold_assignment` "
            "before asking two trainers to agree on folds"
        )
    payload = json.loads(path.read_text())
    assignment = {int(k): int(v) for k, v in payload["assignment"].items()}
    if not assignment:
        raise ValueError(f"{path} holds an empty assignment")
    return assignment, payload.get("provenance", {})


def assignment_mask(groups: np.ndarray, assignment: dict[int, int]) -> np.ndarray:
    """Rows the assignment covers. Everything else is dropped from CV.

    Returned rather than applied here so the caller can log how many rows it
    lost and record it in `run_config` — a joint run that silently trains on a
    different population than it reports is the failure this artefact exists to
    prevent.
    """
    covered = np.fromiter((int(g) in assignment for g in groups), dtype=bool, count=len(groups))
    return covered


def assigned_group_kfold(
    groups: np.ndarray,
    assignment: dict[int, int],
    *,
    n_splits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Replay a fixed group→fold map as `StratifiedGroupKFold.split` would.

    Yields `(trainval_idx, test_idx)` positions into `groups`, in fold order, so
    it is a drop-in for the splitter both trainers already call.

    Every guard here raises. A fold assignment that silently covers only part of
    the data, or collapses to fewer folds than asked for, produces a completely
    plausible set of metrics over the wrong population — which is this project's
    defining failure mode, and the reason none of these are warnings.
    """
    folds = np.fromiter(
        (assignment.get(int(g), -1) for g in groups), dtype=np.int64, count=len(groups)
    )
    if (missing := folds == -1).any():
        unknown = sorted({int(g) for g in groups[missing]})[:5]
        raise ValueError(
            f"{int(missing.sum())} row(s) over {len(unknown)}+ group(s) are not in the fold "
            f"assignment, e.g. {unknown}. Filter with `assignment_mask` first — dropping "
            "them is a decision the caller records, not one this function takes silently"
        )

    present = sorted(set(folds.tolist()))
    if present != list(range(n_splits)):
        raise ValueError(
            f"the assignment covers folds {present} but {n_splits} were requested; "
            "a missing fold would silently shrink CV to fewer folds than reported"
        )

    positions = np.arange(len(groups))
    for fold in range(n_splits):
        test_idx = positions[folds == fold]
        if len(test_idx) == 0:
            raise ValueError(f"fold {fold} holds out no rows")
        yield positions[folds != fold], test_idx
