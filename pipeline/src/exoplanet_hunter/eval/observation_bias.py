"""Does the model score the transit, or the observation?

The measured failure the rebuild exists to fix: over 3,919 scored candidates,
probability correlated **+0.211** with observation baseline and **-0.003** with
the number of transits actually captured. The model was reading how long a
target was watched, not how often it dipped.

That makes stage 2(b) falsifiable. `transit_sensitivity` must move clearly away
from zero, and `baseline_sensitivity` must fall. A model that improves AUC while
leaving these unchanged has learnt the same shortcut with more parameters.
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
    baseline_column: str = "expected_transit_count",
    completeness_column: str = "transit_completeness",
) -> ObservationBias:
    """Rank correlations of score against transit count, baseline, completeness.

    Spearman rather than Pearson: transit counts are heavily skewed (median 3,
    max 881 over the FFI targets), and a Pearson coefficient there mostly
    reports the tail.

    `expected_transit_count` stands in for observation baseline — it is how many
    transits the ephemeris predicts over the observed span, so it grows with
    baseline and is independent of whether any were caught.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) != len(index):
        raise ValueError(f"{len(scores)} scores but {len(index)} index rows")
    return ObservationBias(
        transit_sensitivity=_spearman(scores, index[transit_column].to_numpy(dtype=float)),
        baseline_sensitivity=_spearman(scores, index[baseline_column].to_numpy(dtype=float)),
        completeness_sensitivity=_spearman(scores, index[completeness_column].to_numpy(dtype=float))
        if completeness_column in index.columns
        else float("nan"),
        n=len(scores),
    )
