"""The offline control arm: does the model score the star, or the transit?

With **no injection at all**, 26.4% of hosts passed threshold through the live
dual-view path — 46.7% of planet hosts against 12.3% of false-positive hosts.
That is the model recognising the kind of star that tends to have planets, which
is not vetting, and driving it down is stage 7's success criterion.

**It cannot be measured for a branch model through the serving path.**
`ScoringEnsemble.from_registry` loads `cnn_dualview.keras` and `TargetScorer`
builds two views with `preprocess.views`; a branch model needs the eleven that
`preprocess.viewset` builds. Making a branch model scoreable from a light curve
is stage 11. This module is the agreed way round it: score a run directory
*offline*, over a shard set built the same way training's was.

    clean -> flatten -> inject_box_transit(depth=0) -> build_view_set
          -> write_viewset_shards -> make_viewset_dataset
          -> the run directory's fold members and calibrator

Writing a shard and reading it back is the construction rather than a shortcut:
it puts the control arm through the same parse and scalar-normalisation path
training used, instead of a second implementation that agrees with it by
inspection.

**Three things are pre-registered in `roadmap.md` and implemented here.**

1. **Out-of-fold routing, or the host is dropped.** A control-arm host is a real
   labelled target that was in training, so the only honest protocol is the fold
   that held it out. Mixing in zero-shot rows across one population is the
   defect stage 3 closed, so an unroutable host is dropped and counted, never
   scored by an averaged ensemble.
2. **Two operating points, both reported.** A branch run directory stores no
   threshold — `bundle["threshold"]` is the *legacy serving* bundle's field, and
   the branch bundle carries `calibrator`, `platt_a`, `platt_b` and
   `scalar_constants`. So one is derived: recall @1% FPR is primary, because it
   is where every gate decision in this project is made, and the F1-optimal
   point is reported beside it because that is what produced the original 26.4%.
3. **Baseline-matched hosts.** The statistic of interest is the 46.7 / 12.3
   split, and observation baseline is the confound it is most exposed to
   (+0.387 against the label on TESS). Hosts are stratified on `baseline_days`
   and drawn equally per label within each stratum.

**Two limits, recorded before any number exists.** The offline arm injects at a
synthetic ephemeris, so no DV report exists for it and the `detection` / `ghost`
branches run masked — the model handles that by design, but it is a real
difference from how 56% of training rows were built. And this does **not**
restore comparability with 26.4%: that came through the dual-view path, so the
champion is re-measured here and the comparison is made within protocol. No
number from this module can support a claim about *serving*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from exoplanet_hunter.eval.observation_bias import BASELINE_DAYS, baseline_days
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: The gating mission. Kepler and K2 carry no serving consequence.
GATE_MISSION = "TESS"
#: The shortlist's false-positive budget — the operating point the gate reads.
SHORTLIST_FPR = 0.01


@dataclass(frozen=True)
class OperatingPoints:
    """The two thresholds a run is read at, derived from its own predictions."""

    shortlist: float
    f1_optimal: float
    n: int
    n_positive: int

    def as_dict(self) -> dict[str, float]:
        return {
            "shortlist_threshold": self.shortlist,
            "f1_optimal_threshold": self.f1_optimal,
            "n_reference": float(self.n),
            "n_positive_reference": float(self.n_positive),
        }


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float) -> float:
    """The score cut achieving the best TPR without exceeding `fpr`.

    Sibling to `comparison.recall_at_fpr`, which returns the recall at this
    point and not the cut. Read off the ROC curve for the same reason: ties move
    a block of rows across a threshold together, and a count-based cut splits
    that block, naming an operating point no threshold actually achieves.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(y_score)} scores")
    if len(np.unique(y_true)) < 2:
        raise ValueError("a threshold needs both classes present in the reference set")
    curve_fpr, _, thresholds = roc_curve(y_true, y_score)
    within = curve_fpr <= fpr
    if not within.any():
        raise ValueError(f"no threshold reaches an FPR of {fpr}")
    # `roc_curve` sorts by decreasing threshold, so the last index inside the
    # budget is the most permissive cut that still respects it.
    return float(thresholds[np.flatnonzero(within)[-1]])


def f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The cut maximising F1 — the live path's operating point.

    Reported alongside the shortlist cut rather than instead of it: the original
    26.4% was measured at this point, so dropping it would discard the only
    continuity the offline protocol has with the number it is compared against.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(y_score)
    if candidates.size == 0:
        raise ValueError("no scores to threshold")
    best_f1, best_cut = -1.0, float(candidates[0])
    positives = y_true.sum()
    for cut in candidates:
        predicted = y_score >= cut
        tp = float(np.sum(predicted & (y_true == 1)))
        if tp == 0:
            continue
        precision = tp / float(predicted.sum())
        recall = tp / float(positives)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_cut = f1, float(cut)
    return best_cut


def operating_points(
    predictions: pd.DataFrame, *, mission: str = GATE_MISSION, score_column: str = "score"
) -> OperatingPoints:
    """Derive both thresholds from a run's own out-of-fold predictions.

    Restricted to the gating mission: a threshold fitted over all three missions
    is set partly by Kepler, whose base rate and score distribution have no
    bearing on what a TESS shortlist should cut at.
    """
    if "mission" in predictions.columns:
        frame = predictions[predictions["mission"] == mission]
    else:
        frame = predictions
    if frame.empty:
        raise ValueError(f"no {mission} rows to derive an operating point from")
    y_true = frame["label"].to_numpy().astype(int)
    y_score = frame[score_column].to_numpy(dtype=float)
    return OperatingPoints(
        shortlist=threshold_at_fpr(y_true, y_score, SHORTLIST_FPR),
        f1_optimal=f1_optimal_threshold(y_true, y_score),
        n=len(frame),
        n_positive=int(y_true.sum()),
    )


@dataclass(frozen=True)
class MatchedHosts:
    """A baseline-matched host draw, with what matching cost it."""

    hosts: pd.DataFrame
    n_strata_used: int
    n_strata_dropped: int
    n_available: int

    @property
    def n(self) -> int:
        return len(self.hosts)

    def report(self) -> str:
        by_label = self.hosts["label"].value_counts().to_dict()
        return (
            f"{self.n} hosts ({by_label.get(1, 0)} planet / {by_label.get(0, 0)} FP) "
            f"from {self.n_available} eligible; {self.n_strata_used} strata used, "
            f"{self.n_strata_dropped} dropped for want of both labels"
        )


def baseline_matched_hosts(
    candidates: pd.DataFrame,
    *,
    per_label_per_stratum: int,
    n_strata: int = 4,
    seed: int = 42,
) -> MatchedHosts:
    """Draw planet and false-positive hosts matched on observation baseline.

    `candidates` needs `tic_id`, `label`, and either `baseline_days` or the
    `expected_transit_count` + `period` it is derived from. **`labels.parquet`
    carries neither**, so the caller joins the viewset scalars first — this
    raises rather than inventing a baseline, because a control arm silently
    matched on nothing is indistinguishable from one matched correctly.

    Strata are quantile bins of `baseline_days`, and a stratum that cannot
    supply both labels is **dropped, not backfilled**. Backfilling from the
    unmatched pool would return a clean number about an easier population, which
    is the exact confound the matching exists to remove — so the count of
    dropped strata travels with the result.
    """
    required = {"tic_id", "label"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"host candidates are missing {sorted(missing)}")
    if per_label_per_stratum < 1:
        raise ValueError(f"per_label_per_stratum must be at least 1, got {per_label_per_stratum}")
    if n_strata < 1:
        raise ValueError(f"n_strata must be at least 1, got {n_strata}")

    frame = candidates.drop_duplicates("tic_id").copy()
    if BASELINE_DAYS not in frame.columns:
        # Raises via `baseline_days` when the ephemeris columns are absent.
        frame[BASELINE_DAYS] = baseline_days(frame)
    frame = frame[np.isfinite(frame[BASELINE_DAYS].to_numpy(dtype=float))]
    if frame.empty:
        raise ValueError("no host candidates carry a finite baseline_days")

    # `duplicates="drop"` because baseline_days is quantised to whole periods
    # and floors at exactly 0 for a single predicted transit, so its quantiles
    # are not guaranteed distinct; qcut raises on tied edges otherwise.
    frame["stratum"] = pd.qcut(
        frame[BASELINE_DAYS].rank(method="first"), q=n_strata, labels=False, duplicates="drop"
    )

    rng = np.random.default_rng(seed)
    drawn: list[pd.DataFrame] = []
    used = dropped = 0
    for stratum, block in frame.groupby("stratum", sort=True):
        sides = {label: block[block["label"] == label] for label in (0, 1)}
        take = min(per_label_per_stratum, *(len(side) for side in sides.values()))
        if take < 1:
            dropped += 1
            log.info(
                "[control-arm] stratum %s dropped: %d planet / %d FP hosts",
                stratum,
                len(sides[1]),
                len(sides[0]),
            )
            continue
        used += 1
        for side in sides.values():
            drawn.append(side.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    hosts = (
        pd.concat(drawn, ignore_index=True)
        if drawn
        else frame.iloc[[]].assign(stratum=pd.Series(dtype=float))
    )
    return MatchedHosts(
        hosts=hosts,
        n_strata_used=used,
        n_strata_dropped=dropped,
        n_available=len(frame),
    )


def fold_assignment(predictions: pd.DataFrame) -> dict[int, int]:
    """`tic_id -> fold` from a run's own out-of-fold predictions.

    Every row is tested exactly once across folds, so this is the fold that held
    each host out. A host absent from the map has no honest fold and is dropped
    by `control_arm_rate`'s caller rather than scored by an averaged ensemble.
    """
    for column in ("tic_id", "fold"):
        if column not in predictions.columns:
            raise KeyError(f"predictions carry no {column!r} column")
    duplicated = predictions["tic_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} tic_id(s) appear in more than one fold; "
            "out-of-fold routing is ambiguous and would silently pick one"
        )
    return {int(t): int(f) for t, f in zip(predictions["tic_id"], predictions["fold"], strict=True)}


@dataclass(frozen=True)
class ControlArmRate:
    """The host-pass rate, split by label — the 46.7 / 12.3 statistic."""

    threshold_name: str
    threshold: float
    n: int
    overall: float
    planet_hosts: float
    fp_hosts: float
    n_planet: int
    n_fp: int

    @property
    def split(self) -> float:
        """Planet-host rate minus FP-host rate.

        The headline 26.4% conflates two populations; this difference is what
        "the model scores the star" actually predicts, and it is what an
        architecture that vets the transit should drive toward zero.
        """
        return self.planet_hosts - self.fp_hosts

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "threshold_name": self.threshold_name,
            "threshold": self.threshold,
            "n": self.n,
            "pass_rate": self.overall,
            "pass_rate_planet_hosts": self.planet_hosts,
            "pass_rate_fp_hosts": self.fp_hosts,
            "planet_minus_fp": self.split,
            "n_planet": self.n_planet,
            "n_fp": self.n_fp,
        }


def control_arm_rate(
    scores: np.ndarray, labels: np.ndarray, *, threshold: float, threshold_name: str
) -> ControlArmRate:
    """Fraction of un-injected hosts passing `threshold`, overall and per label."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores but {len(labels)} labels")
    if len(scores) == 0:
        raise ValueError("no scored hosts — the control arm has nothing to report")
    if not np.isfinite(scores).all():
        # An uninitialised or failed score reaching here would be averaged into
        # a plausible pass rate. This project's defining failure mode.
        raise ValueError(f"{int((~np.isfinite(scores)).sum())} non-finite score(s)")

    passed = scores >= threshold
    is_planet = labels == 1
    return ControlArmRate(
        threshold_name=threshold_name,
        threshold=float(threshold),
        n=len(scores),
        overall=float(passed.mean()),
        planet_hosts=float(passed[is_planet].mean()) if is_planet.any() else float("nan"),
        fp_hosts=float(passed[~is_planet].mean()) if (~is_planet).any() else float("nan"),
        n_planet=int(is_planet.sum()),
        n_fp=int((~is_planet).sum()),
    )
