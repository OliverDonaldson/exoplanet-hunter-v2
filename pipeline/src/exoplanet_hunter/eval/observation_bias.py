"""Does the model score the transit, or the observation?

The original finding, over 3,919 scored candidates: probability correlated
**+0.211** with observation baseline and **-0.003** with transit count.

`candidate_bias.py` reproduced both on 3,908 of those rows and established which
column each was measured against:

- **+0.211 was right.** Baseline in days recomputes to **+0.208**.
- **-0.003 was the wrong column.** It is the correlation with
  `expected_transit_count` — the transits the ephemeris *predicts*. Against the
  transits actually *captured* it is **-0.048**.

`expected_transit_count` is baseline / period, so it is neither a baseline nor a
count of anything caught. An earlier version of this module then compounded the
confusion by using it as the *baseline* proxy, where it reported -0.068 against
the +0.239 that days reports on the same predictions. The two errors ran in
opposite directions on the same column, which is why each looked plausible.

`baseline_days` is the default and is derived explicitly. Passing the count is
still possible, but only by naming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ObservationBias:
    transit_sensitivity: float
    baseline_sensitivity: float
    completeness_sensitivity: float
    n: int

    def improved_over(self, other: ObservationBias, *, margin: float = 0.05) -> bool:
        """True when transit count matters more and baseline matters less."""
        return (
            abs(self.transit_sensitivity) > abs(other.transit_sensitivity) + margin
            and abs(self.baseline_sensitivity) < abs(other.baseline_sensitivity) - margin
        )


BASELINE_DAYS = "baseline_days"


def baseline_days(index: pd.DataFrame) -> np.ndarray:
    """Observation baseline in days, reconstructed from the ephemeris.

    `expected_transit_count` is `last_epoch - first_epoch + 1` over the cadences
    present, so the span it covers is `(expected - 1) * period`. Two properties
    worth knowing before reading a correlation off this: it is quantised to whole
    periods, and it is exactly zero whenever only one transit is predicted, which
    floors the long-period tail rather than ordering it.
    """
    missing = {"expected_transit_count", "period"} - set(index.columns)
    if missing:
        raise KeyError(f"cannot derive {BASELINE_DAYS} without {sorted(missing)}")
    expected = index["expected_transit_count"].to_numpy(dtype=float)
    return (expected - 1.0) * index["period"].to_numpy(dtype=float)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, or NaN when a column has no spread to rank."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    xs, ys = x[keep], y[keep]
    if np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return float("nan")
    return float(spearmanr(xs, ys).statistic)


def measure_observation_bias(
    scores: np.ndarray,
    index: pd.DataFrame,
    *,
    transit_column: str = "observed_transit_count",
    baseline_column: str = BASELINE_DAYS,
    completeness_column: str = "transit_completeness",
) -> ObservationBias:
    """Rank correlations of score against transit count, baseline, completeness.

    Spearman rather than Pearson: transit counts are heavily skewed (median 3,
    max 881 over the FFI targets), and a Pearson coefficient there mostly
    reports the tail.

    `baseline_column` defaults to `baseline_days` and is derived by
    `baseline_days()` when the frame does not already carry it. Passing
    `expected_transit_count` here measures a different thing — see the module
    docstring — and is left available only so that the old number can be
    reproduced deliberately.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) != len(index):
        raise ValueError(f"{len(scores)} scores but {len(index)} index rows")
    baseline = (
        baseline_days(index)
        if baseline_column == BASELINE_DAYS and BASELINE_DAYS not in index.columns
        else index[baseline_column].to_numpy(dtype=float)
    )
    return ObservationBias(
        transit_sensitivity=_spearman(scores, index[transit_column].to_numpy(dtype=float)),
        baseline_sensitivity=_spearman(scores, baseline),
        completeness_sensitivity=_spearman(scores, index[completeness_column].to_numpy(dtype=float))
        if completeness_column in index.columns
        else float("nan"),
        n=len(scores),
    )
