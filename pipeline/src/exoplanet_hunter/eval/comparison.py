"""Comparing two prediction sets — per mission, and never silently.

Two models are compared on the rows they share, and an inner join is a lossy
operation that reports nothing about what it dropped. Comparing the champion
`ca906040` against the stage 4 (old 2(a)) branch model matched 4,605 rows and looked
complete; it had silently discarded **all 527 K2 examples**, 9.7% of training,
because the champion's run predates K2 entirely. The surviving weights were
Kepler 48.6% / TESS 51.4% / K2 0%, in a decision whose consequences are 100%
TESS.

So coverage is computed first and returned alongside the metrics, and
`MissionCoverage.dropped` names any mission the join lost outright.

`recall_at_fpr` lives here because it is the other half of the same lesson: AUC
scores ranking at every threshold, and a follow-up shortlist lives at exactly
one. The two models' TESS AUCs differ by 0.002 while their recall at 1% FPR
differs by 0.069.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from exoplanet_hunter.eval.metrics import ClassificationMetrics, classification_metrics
from exoplanet_hunter.training.calibration import expected_calibration_error
from exoplanet_hunter.training.mlflow_utils import MEMBER_SCORE_PREFIX

MISSION_COLUMN = "mission"
#: The mission a promotion is decided on, and the FPR its shortlist sits at.
#: Both trainers read these so a floor means the same thing in either summary.
GATE_MISSION = "TESS"
GATE_FPR = 0.01


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float) -> float:
    """Best true-positive rate reachable without exceeding a false-positive
    rate of `fpr` — the shortlist's operating point.

    Read off the ROC curve rather than by counting negatives: ties in the score
    move a whole block of rows across a threshold at once, and a count-based
    cut splits that block, reporting an operating point no threshold achieves.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(y_score)} scores")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    curve_fpr, curve_tpr, _ = roc_curve(y_true, y_score)
    return float(curve_tpr[curve_fpr <= fpr].max())


@dataclass(frozen=True)
class SliceMetrics:
    """One population's metrics, ranking and operating point together."""

    n: int
    n_positive: int
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    recall_at_1pct_fpr: float
    recall_at_5pct_fpr: float
    recall_at_10pct_fpr: float

    @classmethod
    def measure(cls, y_true: np.ndarray, y_score: np.ndarray) -> SliceMetrics:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        single_class = len(np.unique(y_true)) < 2
        metrics = (
            ClassificationMetrics(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            if single_class
            else classification_metrics(y_true, y_score)
        )
        return cls(
            n=len(y_true),
            n_positive=int(y_true.sum()),
            roc_auc=metrics.roc_auc,
            pr_auc=metrics.pr_auc,
            brier=float(np.mean((y_score - y_true) ** 2)),
            ece=float("nan")
            if single_class
            else float(expected_calibration_error(y_true, y_score)),
            recall_at_1pct_fpr=recall_at_fpr(y_true, y_score, 0.01),
            recall_at_5pct_fpr=recall_at_fpr(y_true, y_score, 0.05),
            recall_at_10pct_fpr=recall_at_fpr(y_true, y_score, 0.10),
        )


@dataclass(frozen=True)
class MissionCoverage:
    """What an inner join over two prediction sets kept, and what it lost."""

    shared: dict[str, int]
    left_only: dict[str, int]
    right_only: dict[str, int]

    @property
    def dropped(self) -> list[str]:
        """Missions present in either set that the join retained none of."""
        present = set(self.shared) | set(self.left_only) | set(self.right_only)
        return sorted(m for m in present if self.shared.get(m, 0) == 0)

    @property
    def shared_weights(self) -> dict[str, float]:
        total = sum(self.shared.values())
        return {m: n / total for m, n in self.shared.items()} if total else {}

    def report(self) -> str:
        rows = ["mission      shared  left-only  right-only   share"]
        weights = self.shared_weights
        for mission in sorted(set(self.shared) | set(self.left_only) | set(self.right_only)):
            rows.append(
                f"{mission:<10} {self.shared.get(mission, 0):>7} "
                f"{self.left_only.get(mission, 0):>10} {self.right_only.get(mission, 0):>11} "
                f"{weights.get(mission, 0.0):>7.1%}"
            )
        if self.dropped:
            rows.append(f"DROPPED ENTIRELY: {', '.join(self.dropped)} — 0 rows survive the join")
        return "\n".join(rows)


def _counts(frame: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    return frame.loc[mask, MISSION_COLUMN].value_counts().to_dict()


def _mission_lookup(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """One tic_id -> mission table from whichever frames carry the column."""
    sources = [f[[key, MISSION_COLUMN]] for f in (left, right) if MISSION_COLUMN in f.columns]
    if not sources:
        raise ValueError(f"neither frame carries a {MISSION_COLUMN!r} column")
    return pd.concat(sources).dropna(subset=[MISSION_COLUMN]).drop_duplicates(subset=key)


def mission_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str = "tic_id",
) -> MissionCoverage:
    """Per-mission accounting of an inner join between two prediction sets.

    Either frame may carry the mission column; a set that predates it (the
    champion's `predictions.parquet` has only row/tic_id/fold/y_true/prob) is
    labelled from whichever frame does, and rows neither can label become
    `"unknown"` rather than vanishing from the count.
    """
    missions = _mission_lookup(left, right, key)

    def labelled(frame: pd.DataFrame) -> pd.DataFrame:
        out = (
            frame[[key]]
            .drop_duplicates()
            .merge(missions, on=key, how="left", validate="one_to_one")
        )
        out[MISSION_COLUMN] = out[MISSION_COLUMN].fillna("unknown")
        return out

    a, b = labelled(left), labelled(right)
    in_b, in_a = a[key].isin(set(b[key])), b[key].isin(set(a[key]))
    return MissionCoverage(
        shared=_counts(a, in_b), left_only=_counts(a, ~in_b), right_only=_counts(b, ~in_a)
    )


@dataclass(frozen=True)
class PredictionSetComparison:
    coverage: MissionCoverage
    left: dict[str, SliceMetrics]
    right: dict[str, SliceMetrics]

    def auc_gaps(self) -> dict[str, float]:
        """left minus right, per mission and for the pooled shared rows."""
        return {m: self.left[m].roc_auc - self.right[m].roc_auc for m in self.left}

    def report(self) -> str:
        lines = [self.coverage.report(), "", "slice        n    left AUC   right AUC        gap"]
        for mission, gap in self.auc_gaps().items():
            lines.append(
                f"{mission:<10} {self.left[mission].n:>5} "
                f"{self.left[mission].roc_auc:>10.4f} {self.right[mission].roc_auc:>11.4f} "
                f"{gap:>+10.4f}"
            )
        return "\n".join(lines)


def compare_prediction_sets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str = "tic_id",
    label_column: str = "label",
    score_column: str = "score",
    strict: bool = False,
) -> PredictionSetComparison:
    """Compare two prediction sets on shared rows, per mission and pooled.

    `left`/`right` each need `key`, `label_column` and `score_column`; at least
    one needs `mission`. With `strict=True` a mission that the join drops
    entirely raises instead of being reported — use it where a silently missing
    mission would corrupt a decision.
    """
    coverage = mission_coverage(left, right, key=key)
    if strict and coverage.dropped:
        raise ValueError(
            f"the join dropped every row of {', '.join(coverage.dropped)}; "
            "the comparison would be weighted by which missions happen to overlap"
        )

    missions = _mission_lookup(left, right, key)
    columns = [key, label_column, score_column]
    # `validate` on both: a repeated tic_id on either side turns this into a
    # many-to-many join that multiplies rows, and every metric below would then
    # be computed over a population that does not exist. pandas does that
    # silently, and the row count is the only trace.
    merged = (
        left[columns]
        .merge(right[columns], on=key, suffixes=("_left", "_right"), validate="one_to_one")
        .merge(missions, on=key, how="left", validate="many_to_one")
    )
    merged[MISSION_COLUMN] = merged[MISSION_COLUMN].fillna("unknown")
    if not merged[f"{label_column}_left"].equals(merged[f"{label_column}_right"]):
        raise ValueError("the two prediction sets disagree on labels for shared rows")

    def by_slice(side: str) -> dict[str, SliceMetrics]:
        out = {
            str(mission): SliceMetrics.measure(
                group[f"{label_column}_left"].to_numpy(), group[f"{score_column}_{side}"].to_numpy()
            )
            for mission, group in merged.groupby(MISSION_COLUMN)
        }
        out["all"] = SliceMetrics.measure(
            merged[f"{label_column}_left"].to_numpy(), merged[f"{score_column}_{side}"].to_numpy()
        )
        return out

    return PredictionSetComparison(coverage, by_slice("left"), by_slice("right"))


def paired_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
    *,
    key: str = "tic_id",
    label_column: str = "label",
    score_column: str = "score",
) -> pd.DataFrame:
    """The shared rows of two prediction sets, with covariates attached.

    Every stratified reading below starts from this frame, so the join is
    written once instead of once per cut. Scores always land in `score_left` and
    `score_right` whatever they were called on the way in, which is the contract
    `gap_table` reads.
    """
    merged = (
        left[[key, label_column, score_column]]
        .rename(columns={label_column: "label", score_column: "score_left"})
        .merge(
            right[[key, score_column]].rename(columns={score_column: "score_right"}),
            on=key,
        )
    )
    return merged if covariates is None else merged.merge(covariates, on=key, how="left")


def band_labels(values: pd.Series, bands: Sequence[tuple[float, float]]) -> pd.Categorical:
    """Left-closed band label per value, ordered so a grouped table reads low to high.

    Values outside every band become NaN and are dropped by `gap_table` rather
    than collapsing into the first bin.
    """
    names = [f"{lo}-{hi if hi < np.inf else '+'}" for lo, hi in bands]
    out = pd.Series(pd.NA, index=values.index, dtype=object)
    for (lo, hi), name in zip(bands, names, strict=True):
        out = out.mask(values.between(lo, hi, inclusive="left"), name)
    return pd.Categorical(out, categories=names, ordered=True)


def gap_table(
    paired: pd.DataFrame,
    by: str | Sequence[str],
    *,
    min_rows: int = 1,
    medians: Sequence[str] = (),
) -> pd.DataFrame:
    """Each side's AUC within each group of `by`, and left minus right.

    Quartiles, absolute transit bands and the span-by-count cross differ only in
    how rows are binned, so binning is the caller's job and the measuring is not
    written once per cut. A group thinner than `min_rows`, or holding one class,
    yields NaN rather than an AUC no threshold supports.
    """
    keys = [by] if isinstance(by, str) else list(by)
    rows: list[dict[str, object]] = []
    for value, group in paired.groupby(keys, observed=True, dropna=True):
        labels = group["label"].to_numpy()
        measurable = len(group) >= min_rows and len(np.unique(labels)) > 1
        row: dict[str, object] = dict(
            zip(keys, value if isinstance(value, tuple) else (value,), strict=True)
        )
        row["n"] = len(group)
        for column in medians:
            row[f"median_{column}"] = float(group[column].median()) if len(group) else np.nan
        for side in ("left", "right"):
            row[side] = (
                SliceMetrics.measure(labels, group[f"score_{side}"].to_numpy()).roc_auc
                if measurable
                else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["gap"] = out["left"] - out["right"]
    return out


# ---------------------------------------------------------------------------
# Summary pieces shared by BOTH trainers.
#
# These lived in train_branches.py, which meant the dual-view trainer wrote no
# per-mission block and no recall floor — so the promotion gate could not slice
# a dual-view candidate's population and refused to compare it at all, and
# `decision_floor` fell back to a constant already measured as too tight. One
# definition, both trainers delegating, for the same reason
# `member_checkpoint_name` was centralised: two that drifted would change what a
# bar was computed from.
# ---------------------------------------------------------------------------


def pooled_member_draws(predictions: pd.DataFrame) -> dict[str, Any]:
    """Independent draws of the *pooled* gate statistic, one per ensemble member.

    `gate_recall_seed_sd` is measured on a fold's TESS slice — ~215 negatives, so
    a 1% FPR cut of two rows, and a statistic set by where the third-highest
    negative lands. The gate reads the pooled out-of-fold set instead: ~1,074
    negatives, a cut of ten rows, and a materially better-conditioned number.
    Bounding the second with the first only ever overstates the noise.

    Member `i`'s score column is filled by whichever fold held each row, so
    stacking them re-forms a complete out-of-fold prediction set for member `i`
    alone — the same protocol as the ensemble, one seed instead of three. Their
    spread is the run-level reseeding sd directly, with no `sqrt(n)` argument in
    the way. Three draws is a thin sd and it is reported with its `n`.
    """
    # Sorted on the integer, not the string: `member_score_10` sorts before
    # `member_score_2` lexicographically, which would reorder the draws. The
    # order does not change their sd, but it changes which draw is which in the
    # recorded list, and that list is read against the folds' own members.
    columns = sorted(
        (c for c in predictions.columns if c.startswith(MEMBER_SCORE_PREFIX)),
        key=lambda c: int(c.removeprefix(MEMBER_SCORE_PREFIX)),
    )
    gate = (
        predictions[predictions[MISSION_COLUMN] == GATE_MISSION]
        if (MISSION_COLUMN in predictions.columns)
        else predictions.iloc[:0]
    )
    if not columns or gate.empty:
        # The same keys either way. A summary whose schema depends on whether a
        # measurement succeeded is one where a missing key and a null mean the
        # same thing to a reader and different things to a program.
        return {
            "pooled_gate_recall": [],
            "pooled_gate_recall_seed_sd": None,
            "pooled_gate_recall_n_draws": 0,
            "pooled_gate_n": len(gate),
        }
    labels = gate["label"].to_numpy()
    draws = []
    for column in columns:
        scores = gate[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(scores)):
            # A fold that trained fewer members leaves NaN here rather than a
            # short column, and a NaN score silently sinks those rows to the
            # bottom of the ranking — a plausible number over a population that
            # is not the one named.
            raise ValueError(
                f"{column} is not finite on every gate-mission row; the folds did "
                f"not all train the same number of members"
            )
        draws.append(float(recall_at_fpr(labels, scores, GATE_FPR)))
    if not np.all(np.isfinite(draws)):
        # `recall_at_fpr` returns NaN on a single-class slice rather than
        # raising, and an sd over NaN is NaN — which loses every inequality it
        # would later be compared with. Empty was handled above; this is the
        # other way the statistic can fail to exist.
        raise ValueError(
            f"the pooled {GATE_MISSION} slice yielded non-finite recall {draws} over "
            f"{len(labels)} rows with {int(labels.sum())} positives; it is single-class"
        )
    return {
        "pooled_gate_recall": draws,
        "pooled_gate_recall_seed_sd": float(np.std(draws, ddof=1)) if len(draws) > 1 else None,
        "pooled_gate_recall_n_draws": len(draws),
        "pooled_gate_n": len(labels),
    }


def per_mission_summary(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pooled out-of-fold metrics per mission, plus the pooled aggregate.

    The gate reads TESS from here; Kepler and K2 are reported diagnostics with
    an alarm threshold, and `all` never gates. See `evaluate_promotion` and the
    roadmap's gate-population pre-commitment for why the aggregate is unfit to
    decide anything: its mission weights are a sampling decision.
    """
    slices = {
        str(mission): asdict(
            SliceMetrics.measure(group["label"].to_numpy(), group["score"].to_numpy())
        )
        for mission, group in predictions.groupby("mission")
    }
    slices["all"] = asdict(
        SliceMetrics.measure(predictions["label"].to_numpy(), predictions["score"].to_numpy())
    )
    return slices
