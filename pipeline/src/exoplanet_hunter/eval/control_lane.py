"""The control lane: the served model re-scored on the population in front of it.

A candidate is scored on the current catalogue. The champion's stored summary
was measured on the catalogue it was trained against. Subtracting one from the
other gives a number that is a model difference and a population difference
added together, and the gate has been reporting that sum as if it were the
first.

This module measures the champion again, now, on the rows the candidate is
being judged on — so the difference the gate reads is attributable to the model.

**Which rows.** The shared out-of-fold population: rows in both the champion's
training set and the current shard set, each scored by the fold that held it
out. Rows added since the champion was trained are deliberately excluded from
the comparison. The champion never trained on them, so scoring them means
averaging all five folds — an ensemble — while the candidate scores each row
with the single fold that held it out. That would hand the champion a
five-model advantage on exactly the rows a refresh adds. They are counted and
reported; they never gate.

**What this cannot fix.** The shared population is pinned to the champion's
training set while the catalogue grows, so it covers a falling fraction of what
the model serves. Nothing here hides that: `PopulationOverlap` reports it every
run, and below `MIN_GATE_ROWS` the lane refuses rather than deciding on a
remainder too thin to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exoplanet_hunter.validation.promotion import GATE_MISSION

#: Below this the 1% FPR cut lands on fewer than about ten negatives and the
#: shortlist recall statistic is not worth reading. The lane reports that it
#: cannot decide rather than deciding on what is left — a thin slice returns a
#: number, which is the failure mode this project keeps finding.
MIN_GATE_ROWS = 1000

#: How far the lane may differ from the original scoring path **measured on the
#: same inputs on the same day** before it is considered to be computing
#: something other than what that path computes. Same weights, same shards, same
#: labels, same folds: the only expected difference is floating-point.
#:
#: 4.1c set this against a summary produced on a *different date*, which made it
#: a test of whether the population had moved — the one thing this lane exists to
#: stop assuming. 4.1d removed time from the comparison instead of loosening the
#: number: the tolerance is unchanged, what it ranges over is not.
REPRODUCTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PopulationOverlap:
    """How much of the current population the champion can be compared on."""

    shared: int
    added: int
    dropped: int

    @property
    def current(self) -> int:
        return self.shared + self.added

    @property
    def covered(self) -> float:
        """Fraction of the current population the comparison can reach."""
        return self.shared / self.current if self.current else 0.0

    def __str__(self) -> str:
        return (
            f"{self.shared} of {self.current} current rows are shared with the champion "
            f"({self.covered:.1%}); {self.added} added since it was trained and cannot be "
            f"compared like for like, {self.dropped} of its own rows are gone"
        )


def population_overlap(champion_tic_ids: set[int], current_tic_ids: set[int]) -> PopulationOverlap:
    """Split the current population against the one the champion was trained on."""
    return PopulationOverlap(
        shared=len(champion_tic_ids & current_tic_ids),
        added=len(current_tic_ids - champion_tic_ids),
        dropped=len(champion_tic_ids - current_tic_ids),
    )


def assert_gateable(summary: dict[str, Any]) -> None:
    """Refuse a control summary whose gate slice is too thin to read.

    Raises rather than returning a flag: a caller that forgot to check would
    otherwise gate on a recall figure set by where two or three negatives
    happened to land, and nothing downstream could tell.
    """
    slice_ = (summary.get("per_mission") or {}).get(GATE_MISSION)
    if slice_ is None:
        raise ValueError(
            f"the control summary carries no {GATE_MISSION} slice, so it cannot stand in "
            "for the champion in a comparison the gate decides on"
        )
    rows = int(slice_["n"])
    if rows < MIN_GATE_ROWS:
        raise ValueError(
            f"the shared {GATE_MISSION} population is down to {rows} rows, under the "
            f"{MIN_GATE_ROWS} this lane will decide on. A 1% FPR cut here lands on a "
            "handful of negatives and the recall margin would be set by where they fell. "
            "Re-baseline the champion on the current view set rather than narrowing the "
            "comparison further"
        )


def reproduces(
    control: dict[str, Any],
    stored: dict[str, Any],
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> list[str]:
    """Metrics where the lane disagrees with the original path on the same inputs.

    **Check A of 4.1d, at the summary level.** Every metric of every
    `per_mission` slice, not the gate slice alone — a lane that agreed on TESS
    while disagreeing on Kepler would have passed 4.1c's version of this and
    still been wrong.

    Both summaries must be measurements of the same model over the same shard
    set and the same labels, so the only difference left between them is the
    code path. That is the entire question this asks; it is not a test of
    whether the population has moved, and pointing it at a summary from another
    date turns it into one.
    """
    left_slices = control.get("per_mission") or {}
    right_slices = stored.get("per_mission") or {}
    drifted, compared = [], 0
    for name in sorted(set(left_slices) & set(right_slices)):
        left, right = left_slices[name] or {}, right_slices[name] or {}
        for metric in sorted(set(left) & set(right)):
            compared += 1
            if abs(float(left[metric]) - float(right[metric])) > tolerance:
                drifted.append(
                    f"{name}.{metric}: lane {float(left[metric]):.6f} "
                    f"vs original {float(right[metric]):.6f}"
                )
    if not compared:
        # An empty list reads as "reproduces exactly", so a comparison with
        # nothing in it must refuse rather than return the same value as a
        # comparison that passed.
        raise ValueError(
            "no per-mission metrics in common between the lane and the summary it is "
            "checked against, so the check that the lane computes the right thing cannot run"
        )
    return drifted


def rows_reproduce(
    lane: pd.DataFrame,
    original: pd.DataFrame,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> list[str]:
    """Row-level disagreements between two scorings of the same model.

    **Check A of 4.1d, at the row level, and the half that has teeth.** Slice
    metrics are means: two paths can average to the same AUC while disagreeing
    about individual objects, and the shortlist this system exists to produce is
    made of individual objects. Membership, fold assignment, ground-truth label
    and score are all compared, because a difference in any of them makes the
    two paths different measurements whatever the aggregates say.
    """
    left, right = lane.set_index("tic_id"), original.set_index("tic_id")
    problems: list[str] = []
    for missing, side in (
        (right.index.difference(left.index), "the lane"),
        (left.index.difference(right.index), "the original path"),
    ):
        if len(missing):
            problems.append(
                f"{len(missing)} rows are missing from {side}, e.g. {sorted(missing.tolist())[:3]}"
            )

    shared = left.index.intersection(right.index)
    if not len(shared):
        raise ValueError(
            "the lane and the original path share no rows, so the check that they compute "
            "the same thing cannot run"
        )
    left, right = left.loc[shared], right.loc[shared]

    # Fold and label before score: a row scored by a different fold, or measured
    # against a different label, is a different measurement even where the two
    # numbers happen to land within tolerance of each other.
    for column in ("fold", "label"):
        if column in left.columns and column in right.columns:
            moved = shared[left[column].to_numpy() != right[column].to_numpy()]
            if len(moved):
                problems.append(
                    f"{len(moved)} rows disagree on {column}, e.g. {sorted(moved.tolist())[:3]}"
                )

    delta = np.abs(left["score"].to_numpy(dtype=float) - right["score"].to_numpy(dtype=float))
    if over := int((delta > tolerance).sum()):
        worst = int(delta.argmax())
        problems.append(
            f"{over} of {len(shared)} rows differ in score beyond {tolerance:g}; "
            f"worst {delta[worst]:.3g} at tic_id {shared[worst]}"
        )
    return problems


#: Row counts, not metrics. Differencing them is a population statement and
#: reporting it beside AUC would read as a fourth score.
_NOT_A_METRIC = frozenset({"n", "n_positive"})


def _gate_slice(summary: dict[str, Any]) -> dict[str, Any]:
    return (summary.get("per_mission") or {}).get(GATE_MISSION) or {}


def deltas(
    candidate: dict[str, Any],
    control: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """The model effect, and the data effect it has been carrying.

    `model` is the candidate against the champion **on the same rows** — what
    the gate is entitled to call a model difference. `data` is the champion
    against its own previous measurement: same weights, a different population,
    and the quantity that has been inside every weekly margin with nothing
    separating it out. `data` is None on the first run, which has nothing to
    compare against.
    """
    cand, ctrl = _gate_slice(candidate), _gate_slice(control)
    metrics = sorted((set(cand) & set(ctrl)) - _NOT_A_METRIC)
    model = {m: float(cand[m]) - float(ctrl[m]) for m in metrics}
    if previous is None:
        return model, None
    prev = _gate_slice(previous)
    data = {m: float(ctrl[m]) - float(prev[m]) for m in metrics if m in prev}
    return model, data


def outweighed_by_data(model: dict[str, float], data: dict[str, float] | None) -> list[str]:
    """Metrics where the population moved further than the model did.

    Not blocking. A population that genuinely improved is not a fault — but a
    promotion taken while the data moved further than the model did is one that
    has to say so, because the headline number is the model's and the larger
    part of it is not.
    """
    if not data:
        return []
    return [
        f"{metric}: model {model[metric]:+.4f}, data {data[metric]:+.4f}"
        for metric in sorted(set(model) & set(data))
        if abs(data[metric]) > abs(model[metric])
    ]
