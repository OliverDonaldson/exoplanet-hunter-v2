"""`GET /reliability` — calibration quality of the promoted model.

Bins the promoted run's per-example CV test predictions (written by the
trainer, or backfilled by scripts/export_predictions.py) into a reliability
diagram: mean predicted probability vs observed positive fraction per bin,
plus ECE and Brier. This is the chart that shows whether "0.9" from this
model actually means 90% — the project's central claim, made inspectable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import ReliabilityBin, ReliabilityResponse

router = APIRouter()

_ROOT = Path(__file__).resolve().parents[3]
_N_BINS = 10


@router.get("/reliability", response_model=ReliabilityResponse)
def reliability() -> ReliabilityResponse:
    models_dir = Path(os.environ.get("MODEL_DIR", _ROOT / "models"))
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        raise HTTPException(503, detail="No promoted model in the registry yet.")
    registry = json.loads(registry_path.read_text())
    cv_dir = Path(registry["cv_dir"])
    if not cv_dir.is_absolute():
        cv_dir = models_dir.parent / cv_dir
    predictions_path = cv_dir / "predictions.parquet"
    if not predictions_path.exists():
        raise HTTPException(
            503,
            detail=(
                "The promoted run has no predictions.parquet — run "
                "pipeline/scripts/export_predictions.py to backfill it."
            ),
        )

    preds = pd.read_parquet(predictions_path)
    p = preds["prob_calibrated"].to_numpy(dtype=float)
    y = preds["y_true"].to_numpy(dtype=int)

    edges = np.linspace(0.0, 1.0, _N_BINS + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, _N_BINS - 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(_N_BINS):
        sel = which == b
        if not sel.any():
            continue
        conf = float(p[sel].mean())
        acc = float(y[sel].mean())
        bins.append(ReliabilityBin(prob_mean=conf, frac_positive=acc, count=int(sel.sum())))
        ece += abs(acc - conf) * sel.sum() / len(p)

    return ReliabilityResponse(
        run_id=str(registry["run_id"]),
        n_examples=len(p),
        bins=bins,
        ece=float(ece),
        brier=float(np.mean((p - y) ** 2)),
    )
