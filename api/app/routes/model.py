"""`GET /model` — the served run's own metrics, so the console follows
`registry.json` instead of carrying literals that go stale on promotion.

Per-mission is computed, not read: `cv_summary.json` has no `per_mission`
block, and predictions carry `tic_id` but not mission, so the split joins the
labelled catalogue. Missions the run never evaluated are omitted rather than
emitted empty — the console builds cards from the list, so `ca906040` gets no
K2 card.

Error bars are the metric recomputed per fold, matching what ± means elsewhere
on the page; a bootstrap mixed in would not be comparable.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import (
    MetricSummary,
    MissionMetrics,
    ModelSummaryResponse,
    NoiseFloor,
    RocPoint,
)

router = APIRouter()

_ROOT = Path(__file__).resolve().parents[3]
_N_BINS = 10

#: `cv_summary.json` key → published name. An allowlist, not a passthrough: the
#: summary also holds Platt coefficients, which are calibration internals and
#: would invite a client to plot them beside the metrics.
_METRICS = {
    "test_roc_auc": "roc_auc",
    "test_pr_auc": "pr_auc",
    "test_f1": "f1",
    "test_brier": "brier",
    "test_ece": "ece",
    "best_threshold": "threshold",
}

#: The mission whose numbers decide promotion. Everything else is diagnostic.
_GATING_MISSION = "TESS"

#: The floor rule, `2 * sd / sqrt(n_models_per_fold)`, over the per-member
#: spread the trainer records in `summary.variance`. Stage 6 measured it on the
#: branch model; 4.1a measured the dual-view protocol separately. The two are
#: different numbers for different architectures, so this is read from the run
#: rather than held as a constant — the constants that used to sit here were
#: branch-model figures published under a dual-view run.
_FLOOR_MULTIPLIER = 2.0


def _noise_floor(doc: dict) -> NoiseFloor:
    """This run's own floor, or an explicit "not measured".

    `summary.variance` is written only where a run trains more than one model
    per fold; the served champion trains one, so it has no seed spread of its
    own and gets nulls with the reason attached. Deriving one from another run
    would publish a floor for an architecture and a protocol this run did not
    use, which is the defect this function replaced.
    """
    variance = (doc.get("summary") or {}).get("variance") or {}
    n = variance.get("n_models_per_fold")
    if not isinstance(n, int) or n < 2:
        return NoiseFloor(
            measured=False,
            n_models_per_fold=n if isinstance(n, int) else 1,
            source=(
                "not measured: this run trains one model per fold, so it has no "
                "seed spread of its own"
            ),
        )

    def floor(sd: object) -> float | None:
        return (
            _FLOOR_MULTIPLIER * float(sd) / math.sqrt(n)
            if isinstance(sd, int | float) and math.isfinite(float(sd))
            else None
        )

    # The pooled gate draw is the one the promotion gate reads, so it is the one
    # published; `gate_recall_seed_sd` is the per-fold version of the same thing.
    recall_sd = variance.get("pooled_gate_recall_seed_sd", variance.get("gate_recall_seed_sd"))
    return NoiseFloor(
        auc=floor(variance.get("seed_sd")),
        recall=floor(recall_sd),
        measured=True,
        n_models_per_fold=n,
        source=f"this run's own members, 2 x sd / sqrt({n})",
    )


def _confusion_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float = 0.01) -> dict | None:
    """The shortlist operating point and the four cells it produces.

    The threshold is a quantile of the negatives, so `fpr` of them pass by
    construction — but only in the limit. On a finite slice the realised rate
    lands near the target rather than on it, which is why `fpr_actual` is
    returned and not assumed: 11 of 1,067 TESS negatives is 1.03%, and a panel
    captioned "1% FPR" over cells cut at 1.03% would misdescribe them.

    None when either class is absent. Recall against no negatives is undefined
    rather than small, and a matrix with an empty row is not a matrix.
    """
    neg, pos = p[y == 0], p[y == 1]
    if neg.size == 0 or pos.size == 0:
        return None
    threshold = float(np.quantile(neg, 1.0 - fpr))
    tp = int((pos > threshold).sum())
    fp = int((neg > threshold).sum())
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": int(pos.size) - tp,
        "tn": int(neg.size) - fp,
        "fpr_actual": fp / float(neg.size),
    }


def _recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float = 0.01) -> float | None:
    """Recall at a fixed FPR — the shortlist criterion. Delegates so that this
    and the confusion matrix beside it can never be cut at different
    thresholds and disagree on screen."""
    cells = _confusion_at_fpr(y, p, fpr)
    if cells is None:
        return None
    return cells["tp"] / float(cells["tp"] + cells["fn"])


def _precision_f1(cells: dict | None) -> tuple[float | None, float | None]:
    """Precision and F1 from the cells, or nulls. Both are undefined when the
    operating point selects nothing, which a strict enough threshold on a small
    fold can do."""
    if cells is None:
        return None, None
    called = cells["tp"] + cells["fp"]
    if called == 0 or cells["tp"] == 0:
        return (0.0 if called else None), (0.0 if called else None)
    precision = cells["tp"] / float(called)
    recall = cells["tp"] / float(cells["tp"] + cells["fn"])
    return precision, 2 * precision * recall / (precision + recall)


#: Points on the published ROC. The full curve has one point per distinct score
#: — 4,802 of them on the served run — which is a 200 kB payload to draw a
#: 400 px chart. Thinned by even spacing along the curve, with both endpoints
#: and the shortlist operating point always kept.
_ROC_POINTS = 160


def _roc_curve(y: np.ndarray, p: np.ndarray, keep: float | None = None) -> list[RocPoint] | None:
    """The measured ROC, thinned. None when either class is absent.

    Thresholds descend, so the curve runs from (0,0) to (1,1). `keep` is a
    threshold that must survive thinning — the operating point the confusion
    matrix is cut at, so the chart can mark the number the page decides on.
    """
    neg, pos = p[y == 0], p[y == 1]
    if neg.size == 0 or pos.size == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order].astype(bool)
    p_sorted = p[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    # One point per distinct score: a threshold inside a run of ties is not a
    # threshold anyone could set.
    last = np.r_[np.flatnonzero(np.diff(p_sorted)), p_sorted.size - 1]
    fpr, tpr, thr = fps[last] / neg.size, tps[last] / pos.size, p_sorted[last]

    idx = np.unique(np.linspace(0, last.size - 1, min(_ROC_POINTS, last.size)).astype(int))
    if keep is not None:
        idx = np.unique(np.r_[idx, int(np.abs(thr - keep).argmin())])
    points = [RocPoint(fpr=float(fpr[i]), tpr=float(tpr[i]), threshold=float(thr[i])) for i in idx]
    # The sweep starts at the highest score, which is already one call, so the
    # origin is not in it and a chart drawn without it starts mid-air.
    return [RocPoint(fpr=0.0, tpr=0.0, threshold=float(thr[0]) + 1.0), *points]


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC by rank, so the endpoint does not depend on scikit-learn."""
    if np.unique(y).size < 2:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties, or tied scores would bias the statistic
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ece(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, _N_BINS + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, _N_BINS - 1)
    total = 0.0
    for b in range(_N_BINS):
        sel = which == b
        if sel.any():
            total += abs(y[sel].mean() - p[sel].mean()) * sel.sum() / len(p)
    return float(total)


def _spread(values: list[float | None]) -> float | None:
    """Std across folds, or None when too few folds produced the metric."""
    clean = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(clean)) if len(clean) >= 2 else None


def _mission_metrics(frame: pd.DataFrame, mission: str) -> MissionMetrics:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame["prob_calibrated"].to_numpy(dtype=float)

    per_fold: dict[str, list[float | None]] = {
        "auc": [],
        "recall": [],
        "brier": [],
        "ece": [],
        "precision": [],
        "f1": [],
    }
    for _, g in frame.groupby("fold"):
        gy = g["y_true"].to_numpy(dtype=int)
        gp = g["prob_calibrated"].to_numpy(dtype=float)
        fold_cells = _confusion_at_fpr(gy, gp)
        fold_precision, fold_f1 = _precision_f1(fold_cells)
        per_fold["auc"].append(_roc_auc(gy, gp))
        per_fold["recall"].append(_recall_at_fpr(gy, gp))
        per_fold["brier"].append(float(np.mean((gp - gy) ** 2)))
        per_fold["ece"].append(_ece(gy, gp))
        per_fold["precision"].append(fold_precision)
        per_fold["f1"].append(fold_f1)

    # Pooled, at the same cut the pooled recall uses, so the matrix and the
    # recall printed beside it are the same measurement.
    cells = _confusion_at_fpr(y, p)
    precision, f1 = _precision_f1(cells)

    return MissionMetrics(
        mission=mission,
        role="gating" if mission == _GATING_MISSION else "diagnostic",
        # Every mission in this run's predictions was in a training fold; a
        # zero-shot one would need its own label, being incomparable to these.
        evaluation="out-of-fold",
        n=len(frame),
        auc=_roc_auc(y, p),
        aucErr=_spread(per_fold["auc"]),
        recall=_recall_at_fpr(y, p),
        recallErr=_spread(per_fold["recall"]),
        brier=float(np.mean((p - y) ** 2)),
        brierErr=_spread(per_fold["brier"]),
        ece=_ece(y, p),
        eceErr=_spread(per_fold["ece"]),
        nPositive=int((y == 1).sum()),
        threshold=cells["threshold"] if cells else None,
        fprActual=cells["fpr_actual"] if cells else None,
        tp=cells["tp"] if cells else None,
        fp=cells["fp"] if cells else None,
        fn=cells["fn"] if cells else None,
        tn=cells["tn"] if cells else None,
        precision=precision,
        precisionErr=_spread(per_fold["precision"]),
        f1=f1,
        f1Err=_spread(per_fold["f1"]),
        roc=_roc_curve(y, p, keep=cells["threshold"] if cells else None),
    )


def _mission_lookup(models_dir: Path) -> pd.DataFrame | None:
    """tic_id → mission, from the labelled catalogue."""
    labels = models_dir.parent / "data" / "tables" / "labels" / "labels.parquet"
    if not labels.exists():
        return None
    df = pd.read_parquet(labels, columns=["tic_id", "mission"])
    return df.dropna(subset=["tic_id", "mission"]).drop_duplicates("tic_id")


@router.get("/model", response_model=ModelSummaryResponse)
def model_summary() -> ModelSummaryResponse:
    models_dir = Path(os.environ.get("MODEL_DIR", _ROOT / "models"))
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        raise HTTPException(503, detail="No promoted model in the registry yet.")
    registry = json.loads(registry_path.read_text())

    summary_path = Path(registry["cv_summary"])
    if not summary_path.is_absolute():
        summary_path = models_dir.parent / summary_path
    if not summary_path.exists():
        raise HTTPException(
            503, detail=f"The promoted run names {summary_path.name}, which is not on disk."
        )

    doc = json.loads(summary_path.read_text())
    summary = doc.get("summary", {})

    metrics: dict[str, MetricSummary] = {}
    for key, published in _METRICS.items():
        entry = summary.get(key)
        # A metric the run did not record is omitted, never defaulted to 0.0 —
        # a zero ECE would read as perfect calibration rather than as absent.
        if isinstance(entry, dict) and "mean" in entry:
            metrics[published] = MetricSummary(
                mean=float(entry["mean"]), std=float(entry.get("std", 0.0))
            )

    per_mission: list[MissionMetrics] = []
    n_scored = 0
    n_high_confidence = 0
    cv_dir = Path(registry["cv_dir"])
    if not cv_dir.is_absolute():
        cv_dir = models_dir.parent / cv_dir
    predictions_path = cv_dir / "predictions.parquet"
    if predictions_path.exists():
        preds = pd.read_parquet(predictions_path)
        n_scored = len(preds)
        n_high_confidence = int((preds["prob_calibrated"] >= 0.85).sum())
        lookup = _mission_lookup(models_dir)
        if lookup is not None:
            joined = preds.merge(lookup, on="tic_id", how="left")
            for mission, group in joined.dropna(subset=["mission"]).groupby("mission"):
                # A single-class slice yields no AUC and no recall; it would
                # render as a card of dashes, which is noise rather than a fact.
                if group["y_true"].min() != group["y_true"].max():
                    per_mission.append(_mission_metrics(group, str(mission)))
            per_mission.sort(key=lambda m: (m.role != "gating", m.mission))

    folds = doc.get("folds")
    return ModelSummaryResponse(
        run_id=str(registry["run_id"]),
        model_version=f"cnn_dualview-cv-{str(registry['run_id'])[:8]}",
        promoted_at=registry.get("promoted_at"),
        n_folds=len(folds) if isinstance(folds, list) else None,
        metrics=metrics,
        per_mission=per_mission or None,
        noise_floor=_noise_floor(doc),
        n_scored=n_scored,
        n_high_confidence=n_high_confidence,
    )
