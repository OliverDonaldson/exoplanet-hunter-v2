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
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import MetricSummary, MissionMetrics, ModelSummaryResponse, NoiseFloor

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

#: Seed-to-seed spread measured in stage 6 (2026-08-09): a margin under this is
#: not a decision. A property of the training procedure, not of one run, hence a
#: constant rather than a `cv_summary.json` read.
_NOISE_FLOOR_AUC = 0.0070
_NOISE_FLOOR_RECALL = 0.0337


def _recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float = 0.01) -> float | None:
    """Recall at a fixed FPR — the shortlist criterion. The threshold sits on
    the negatives so `fpr` of them pass; None when either class is absent,
    since recall against no negatives is undefined rather than small."""
    neg, pos = p[y == 0], p[y == 1]
    if neg.size == 0 or pos.size == 0:
        return None
    return float((pos > np.quantile(neg, 1.0 - fpr)).mean())


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

    per_fold: dict[str, list[float | None]] = {"auc": [], "recall": [], "brier": [], "ece": []}
    for _, g in frame.groupby("fold"):
        gy = g["y_true"].to_numpy(dtype=int)
        gp = g["prob_calibrated"].to_numpy(dtype=float)
        per_fold["auc"].append(_roc_auc(gy, gp))
        per_fold["recall"].append(_recall_at_fpr(gy, gp))
        per_fold["brier"].append(float(np.mean((gp - gy) ** 2)))
        per_fold["ece"].append(_ece(gy, gp))

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
        noise_floor=NoiseFloor(auc=_NOISE_FLOOR_AUC, recall=_NOISE_FLOOR_RECALL),
        n_scored=n_scored,
        n_high_confidence=n_high_confidence,
    )
