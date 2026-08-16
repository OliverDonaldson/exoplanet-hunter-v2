"""The control lane: the served model re-scored on the population in front of it.

A candidate is scored on the current catalogue. The incumbent's stored summary
was measured on the catalogue it was trained against. Subtracting one from the
other gives a number that is a model difference and a population difference
added together, and the gate has been reporting that sum as if it were the
first.

This module measures the incumbent again, now, on the rows the candidate is
being judged on — so the difference the gate reads is attributable to the model.

**Which rows.** The shared out-of-fold population: rows in both the incumbent's
training set and the current shard set, each scored by the fold that held it
out. Rows added since the incumbent was trained are deliberately excluded from
the comparison. The incumbent never trained on them, so scoring them means
averaging all five folds — an ensemble — while the candidate scores each row
with the single fold that held it out. That would hand the incumbent a
five-model advantage on exactly the rows a refresh adds. They are counted and
reported; they never gate.

**What this cannot fix.** The shared population is pinned to the incumbent's
training set while the catalogue grows, so it covers a falling fraction of what
the model serves. Nothing here hides that: `PopulationOverlap` reports it every
run, and below `MIN_GATE_ROWS` the lane refuses rather than deciding on a
remainder too thin to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exoplanet_hunter.validation.promotion import GATE_MISSION

#: Below this the 1% FPR cut lands on fewer than about ten negatives and the
#: shortlist recall statistic is not worth reading. The lane reports that it
#: cannot decide rather than deciding on what is left — a thin slice returns a
#: number, which is the failure mode this project keeps finding.
MIN_GATE_ROWS = 1000

#: How far the re-scored incumbent may differ from a stored measurement of the
#: same model on the same rows before the lane is considered to be measuring
#: something other than what it claims. Same weights, same folds, same rows:
#: the only expected difference is floating-point.
REPRODUCTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PopulationOverlap:
    """How much of the current population the incumbent can be compared on."""

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
            f"{self.shared} of {self.current} current rows are shared with the incumbent "
            f"({self.covered:.1%}); {self.added} added since it was trained and cannot be "
            f"compared like for like, {self.dropped} of its own rows are gone"
        )


def population_overlap(incumbent_tic_ids: set[int], current_tic_ids: set[int]) -> PopulationOverlap:
    """Split the current population against the one the incumbent was trained on."""
    return PopulationOverlap(
        shared=len(incumbent_tic_ids & current_tic_ids),
        added=len(current_tic_ids - incumbent_tic_ids),
        dropped=len(incumbent_tic_ids - current_tic_ids),
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
            "for the incumbent in a comparison the gate decides on"
        )
    rows = int(slice_["n"])
    if rows < MIN_GATE_ROWS:
        raise ValueError(
            f"the shared {GATE_MISSION} population is down to {rows} rows, under the "
            f"{MIN_GATE_ROWS} this lane will decide on. A 1% FPR cut here lands on a "
            "handful of negatives and the recall margin would be set by where they fell. "
            "Re-baseline the incumbent on the current view set rather than narrowing the "
            "comparison further"
        )


def reproduces(control: dict[str, Any], stored: dict[str, Any]) -> list[str]:
    """Metrics where a re-score disagrees with a stored measurement of the same model.

    The lane's own correctness check. Same weights, same folds, the same rows:
    a disagreement beyond floating point means the re-score is not measuring the
    model the registry names, and every delta built on it is void.
    """
    left = (control.get("per_mission") or {}).get(GATE_MISSION) or {}
    right = (stored.get("per_mission") or {}).get(GATE_MISSION) or {}
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError(
            f"no {GATE_MISSION} metrics in common between the re-score and the stored "
            "summary, so the check that the lane measures the right model cannot run"
        )
    return [
        f"{metric}: re-scored {left[metric]:.6f} vs stored {right[metric]:.6f}"
        for metric in shared
        if abs(float(left[metric]) - float(right[metric])) > REPRODUCTION_TOLERANCE
    ]


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

    `model` is the candidate against the incumbent **on the same rows** — what
    the gate is entitled to call a model difference. `data` is the incumbent
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
