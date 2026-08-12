"""Two of stage 8's three interventions against observation-baseline dependence.

The defect, measured on TESS: `Spearman(label, baseline_days) = +0.3874`, and the
branch model reads it at **+0.5155** — above its own labels. The catalogue
promotes what it has looked at for longest, so in the training labels the
association is real, and a model that learns it is learning something true about
the catalogue and useless about the sky. For the deployment use — ranking
candidates for follow-up — it is worse than useless: it promotes targets that
already received attention over under-observed ones that may deserve it.

`synthetic_negatives` attacks it by adding rows whose labels carry no baseline
association. This module attacks the *existing* rows two ways:

- **propensity weighting** (`propensity_weights`) — reweight each example by the
  inverse of how likely its label was given its baseline, so the fitted loss sees
  a population in which baseline and label are independent. Nothing is discarded.
- **stratified negative sampling** (`stratified_negative_sample`) — resample the
  negatives so their baseline distribution matches the positives'. Rows are
  discarded, and the count is reported rather than absorbed.

**Which to prefer is an empirical question and stage 8 pre-registered it as one:
each runs as its own arm against a common control.** They are not
interchangeable. Weighting keeps every row and pays in variance — a handful of
rare-stratum examples can end up carrying most of the gradient. Sampling keeps
the effective sample size honest and pays in discarded data.

**The failure mode both share.** An intervention that silently does nothing —
weights that are all 1.0 because the propensity model saw one stratum, a
"stratified" sample that is the original negatives in a different order — leaves
training unchanged while the run is recorded as an arm that was tried and
failed. Both entry points therefore measure their own effect on the correlation
and **raise when it has not moved**, rather than trusting that they ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exoplanet_hunter.eval.observation_bias import BASELINE_DAYS, _spearman, baseline_days
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: Weights are clipped to this ratio of the median before use. An inverse
#: propensity is unbounded as the propensity approaches zero, and a single
#: example carrying a thousand times the gradient of its neighbours is not a
#: reweighted population — it is a one-row training set with extra steps.
MAX_WEIGHT_RATIO = 10.0

#: The residual correlation an intervention must get below to count as having
#: run at all. Deliberately loose: this is a did-anything-happen check, not the
#: stage's success criterion, which is pre-registered in roadmap.md and read
#: against a measured floor.
MAX_RESIDUAL_CORRELATION = 0.05


def _strata(values: np.ndarray, n_strata: int) -> np.ndarray:
    """Quantile bins of `values`, ranked first so ties cannot collapse an edge.

    `baseline_days` is quantised to whole periods and floors at exactly 0 for a
    single predicted transit, so its raw quantiles are not guaranteed distinct
    and `qcut` raises on tied edges. Ranking first spreads the ties across the
    bins they belong to rather than dropping a boundary.
    """
    ranked = pd.Series(values).rank(method="first")
    return pd.qcut(ranked, q=n_strata, labels=False, duplicates="drop").to_numpy(int)


@dataclass(frozen=True)
class Reweighting:
    """Per-example weights, and what they did to the correlation."""

    weights: np.ndarray
    before: float
    after: float
    n_strata_used: int
    n_clipped: int

    @property
    def removed(self) -> float:
        return self.before - self.after

    def report(self) -> str:
        return (
            f"baseline-label correlation {self.before:+.4f} -> {self.after:+.4f} "
            f"({self.removed:+.4f}) over {self.n_strata_used} strata; "
            f"{self.n_clipped} weight(s) clipped at {MAX_WEIGHT_RATIO}x median"
        )


def propensity_weights(
    frame: pd.DataFrame,
    *,
    n_strata: int = 8,
    max_ratio: float = MAX_WEIGHT_RATIO,
    max_residual: float = MAX_RESIDUAL_CORRELATION,
) -> Reweighting:
    """Inverse-propensity weights that decorrelate label from observation baseline.

    The propensity is `P(label = 1 | baseline stratum)`, estimated as the stratum's
    own positive rate — a saturated model rather than a fitted logistic. With one
    covariate and quantile strata there is nothing for a parametric fit to add,
    and a saturated estimate cannot be misspecified, which matters when the whole
    point is that the reweighted population be *exactly* balanced rather than
    approximately so.

    Weights are `1 / p` for positives and `1 / (1 - p)` for negatives, normalised
    to mean 1 so the effective learning rate does not move with `n_strata`.

    Raises when the reweighted correlation has not collapsed. A weighting that
    silently did nothing is the expensive failure here: training runs, the arm is
    recorded, and nothing was ever intervened upon.
    """
    if "label" not in frame.columns:
        raise KeyError("propensity weighting needs a 'label' column")
    if n_strata < 2:
        raise ValueError(f"n_strata must be at least 2 to reweight anything, got {n_strata}")

    baselines = (
        frame[BASELINE_DAYS].to_numpy(float)
        if BASELINE_DAYS in frame.columns
        else baseline_days(frame)
    )
    labels = frame["label"].to_numpy(int)
    finite = np.isfinite(baselines)
    if not finite.all():
        raise ValueError(
            f"{int((~finite).sum())} row(s) carry no finite baseline_days. Dropping them here "
            "would silently change the population the weights describe — drop them upstream, "
            "where the count travels with the dataset"
        )
    if len(np.unique(labels)) < 2:
        raise ValueError("propensity weighting needs both labels present")

    stratum = _strata(baselines, n_strata)
    weights = np.ones(len(frame), dtype=float)
    for value in np.unique(stratum):
        block = stratum == value
        p = float(labels[block].mean())
        # A pure stratum has no propensity to invert: 1/0 is infinite and
        # 1/(1-1) likewise. Left at weight 1 and reported through n_strata_used,
        # because dropping the rows would change the population silently and
        # inventing a shrunk rate would put a fabricated number in the gradient.
        if p <= 0.0 or p >= 1.0:
            log.info(
                "[baseline-bias] stratum %d is single-class (p=%.3f); left unweighted", value, p
            )
            continue
        weights[block & (labels == 1)] = 1.0 / p
        weights[block & (labels == 0)] = 1.0 / (1.0 - p)

    median = float(np.median(weights))
    ceiling = max_ratio * median
    n_clipped = int((weights > ceiling).sum())
    weights = np.minimum(weights, ceiling)
    weights *= len(weights) / weights.sum()

    before = _spearman(labels.astype(float), baselines)
    after = _weighted_spearman(labels.astype(float), baselines, weights)
    used = int(sum(1 for v in np.unique(stratum) if 0.0 < float(labels[stratum == v].mean()) < 1.0))
    result = Reweighting(weights, before, after, used, n_clipped)
    if abs(after) > max_residual:
        raise ValueError(
            f"propensity weighting left {after:+.4f} of baseline-label correlation "
            f"(limit {max_residual}). {result.report()}. The weights are not doing what "
            "the arm claims, and training on them would record an intervention that never "
            "happened"
        )
    log.info("[baseline-bias] propensity weighting: %s", result.report())
    return result


def _weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Spearman rho under `weights`, computed on ranks as the unweighted one is.

    Ranks first, then a weighted Pearson over them — which is what Spearman is.
    Ranking before weighting rather than after is deliberate: weighting the raw
    values first would change the ranks themselves, and the statistic reported
    would no longer be the one the roadmap's +0.3874 is measured with.
    """
    rx = pd.Series(x).rank().to_numpy(float)
    ry = pd.Series(y).rank().to_numpy(float)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    mx, my = (w * rx).sum() / total, (w * ry).sum() / total
    cov = (w * (rx - mx) * (ry - my)).sum() / total
    sx = np.sqrt((w * (rx - mx) ** 2).sum() / total)
    sy = np.sqrt((w * (ry - my) ** 2).sum() / total)
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(cov / (sx * sy))


@dataclass(frozen=True)
class StratifiedSample:
    """The kept rows, and what matching cost in discarded negatives."""

    index: np.ndarray
    before: float
    after: float
    n_dropped: int
    n_strata_used: int

    def report(self) -> str:
        return (
            f"baseline-label correlation {self.before:+.4f} -> {self.after:+.4f}; "
            f"{self.n_dropped} negative(s) dropped over {self.n_strata_used} strata"
        )


def stratified_negative_sample(
    frame: pd.DataFrame,
    *,
    n_strata: int = 8,
    seed: int = 42,
    max_residual: float = MAX_RESIDUAL_CORRELATION,
) -> StratifiedSample:
    """Resample negatives so their baseline distribution matches the positives'.

    Returns positional indices into `frame` — every positive, plus the negatives
    kept. Within each baseline stratum the negatives are cut to the positive
    count, so the joint distribution of (baseline, label) becomes a product by
    construction.

    Strata that hold positives but no negatives keep their positives and are
    counted. The alternative — dropping those positives too — buys a cleaner
    correlation by deleting the long-baseline planets that are the population
    the product exists to rank, which would be optimising the metric against the
    purpose.
    """
    if "label" not in frame.columns:
        raise KeyError("stratified sampling needs a 'label' column")
    if n_strata < 2:
        raise ValueError(f"n_strata must be at least 2 to stratify anything, got {n_strata}")

    baselines = (
        frame[BASELINE_DAYS].to_numpy(float)
        if BASELINE_DAYS in frame.columns
        else baseline_days(frame)
    )
    labels = frame["label"].to_numpy(int)
    if not np.isfinite(baselines).all():
        raise ValueError(
            f"{int((~np.isfinite(baselines)).sum())} row(s) carry no finite baseline_days; "
            "drop them upstream where the count travels with the dataset"
        )
    if len(np.unique(labels)) < 2:
        raise ValueError("stratified sampling needs both labels present")

    stratum = _strata(baselines, n_strata)
    values = np.unique(stratum)
    counts = {
        value: (
            np.flatnonzero((stratum == value) & (labels == 1)),
            np.flatnonzero((stratum == value) & (labels == 0)),
        )
        for value in values
    }

    # A GLOBAL negative-to-positive ratio, not a per-stratum cap. Capping each
    # stratum's negatives at its own positive count only balances the strata
    # where negatives dominate: on the real shape of this catalogue the
    # long-baseline strata hold ~228 planets against ~22 false positives, so a
    # per-stratum cap leaves them at a ratio of 0.10 while the short-baseline
    # strata are cut to 1.00 — and P(label | baseline) is still a staircase. It
    # took the residual from +0.696 to +0.244 and stopped, which the guard below
    # caught. Equalising the ratio across strata is what makes label and stratum
    # independent, and only the scarcest stratum can set it.
    ratios = []
    for value, (positives, negatives) in counts.items():
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError(
                f"baseline stratum {value} holds {len(positives)} planet / {len(negatives)} FP "
                "host(s). A stratum missing a label cannot be balanced at any ratio, and "
                "dropping it would delete part of the population rather than rebalance it — "
                "reduce n_strata so the classes meet"
            )
        ratios.append(len(negatives) / len(positives))
    ratio = min(ratios)

    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for positives, negatives in counts.values():
        take = max(1, round(ratio * len(positives)))
        keep.append(positives)
        keep.append(rng.choice(negatives, size=min(take, len(negatives)), replace=False))
    used = len(values)

    index = np.sort(np.concatenate(keep))
    before = _spearman(labels.astype(float), baselines)
    after = _spearman(labels[index].astype(float), baselines[index])
    result = StratifiedSample(index, before, after, len(frame) - len(index), used)
    if abs(after) > max_residual:
        raise ValueError(
            f"stratified sampling left {after:+.4f} of baseline-label correlation "
            f"(limit {max_residual}). {result.report()}. Most likely the strata are too "
            "coarse to separate the confound — raise n_strata rather than accepting this"
        )
    log.info("[baseline-bias] stratified negatives: %s", result.report())
    return result
