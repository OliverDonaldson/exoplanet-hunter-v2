"""`GET /runs` — the CV runs on disk, active first, so the history table
follows the registry instead of a hardcoded list.

`verdict` and `reason` come from the `promotion_log.json` the promotion gate
writes into each run directory. Both stay Optional and both stay absent rather
than being derived here: a run gated before the log existed legitimately has no
verdict, and reconstructing one from the metrics on disk would fabricate an
audit trail rather than report one. A row without a log still renders.

Only runs with a pooled ROC-AUC are listed — the directory also holds branch
experiments and tuning trials that were never promotion candidates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.routes.model import _mission_lookup, _recall_at_fpr, _roc_auc
from app.schemas import RunRecord, RunsResponse

router = APIRouter()

_ROOT = Path(__file__).resolve().parents[3]

#: Mirrors `exoplanet_hunter.validation.promotion.PROMOTION_LOG_NAME`. Named
#: again rather than imported: reading it needs the filename and nothing else,
#: and importing the validation package costs a pandera import on a route that
#: is otherwise pure JSON and parquet.
_PROMOTION_LOG = "promotion_log.json"


def _promotion(run_dir: Path) -> tuple[str | None, str | None]:
    """The gate's verdict for this run, and its reasons as one readable string.

    Parsed as plain JSON rather than through `validation.read_decision`, which
    raises on an unrecognised verdict: a corrupt or hand-edited log would take
    the whole history table down with it, and on a display path eleven readable
    rows beat a 500. Every failure here degrades to the same nulls a run with no
    log produces.

    The verdict string is passed through as written. Only the gate produces
    these files, so an unfamiliar value means a corrupt or newer log, and
    showing it is more use to whoever has to explain it than dropping it.

    Reasons only. Alarms are advisory, and wherever one actually changed the
    verdict the gate already says so in the reasons — including them too would
    print them twice in the single column the console has for this.
    """
    path = run_dir / _PROMOTION_LOG
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    verdict = payload.get("verdict")
    # `isinstance`, not truthiness: a corrupt log whose `reasons` is a bare
    # string would iterate character by character and publish it letter-spaced.
    raw = payload.get("reasons")
    reasons = [str(r) for r in raw] if isinstance(raw, list) else []
    return (str(verdict) if verdict else None), ("; ".join(reasons) or None)


def _num(value: object) -> float | None:
    """Null is absent, not zero: single-fold runs write `std: null`, and 0.0
    would publish no measured spread as perfect agreement."""
    return float(value) if isinstance(value, (int, float)) else None


def _metric(summary: dict, key: str) -> tuple[float | None, float | None]:
    entry = summary.get(key)
    if isinstance(entry, dict):
        return _num(entry.get("mean")), _num(entry.get("std"))
    return None, None


@router.get("/runs", response_model=RunsResponse)
def runs(limit: int = 12) -> RunsResponse:
    models_dir = Path(os.environ.get("MODEL_DIR", _ROOT / "models"))
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        raise HTTPException(503, detail="No promoted model in the registry yet.")
    registry = json.loads(registry_path.read_text())
    active = str(registry["run_id"])
    promoted_at = registry.get("promoted_at")
    promoted_date = str(promoted_at)[:10] if promoted_at else None

    cv_root = models_dir / "cv"
    if not cv_root.is_dir():
        return RunsResponse(active_run_id=active, runs=[])

    lookup = _mission_lookup(models_dir)

    def tess_slice(run_dir: Path) -> tuple[float | None, float | None]:
        """TESS AUC and recall @ 1% FPR for one run — the columns are labelled
        TESS because TESS gates, and `cv_summary.json` holds only pooled
        figures. Nulls where no predictions or no TESS slice resolve."""
        path = run_dir / "predictions.parquet"
        if lookup is None or not path.is_file():
            return None, None
        try:
            preds = pd.read_parquet(path, columns=["tic_id", "y_true", "prob_calibrated"])
        except (OSError, ValueError, KeyError):
            return None, None
        tess = preds.merge(lookup, on="tic_id", how="left")
        tess = tess[tess["mission"] == "TESS"]
        if tess.empty or tess["y_true"].min() == tess["y_true"].max():
            return None, None
        y = tess["y_true"].to_numpy(dtype=int)
        p = tess["prob_calibrated"].to_numpy(dtype=float)
        return _roc_auc(y, p), _recall_at_fpr(y, p)

    records: list[RunRecord] = []
    for run_dir in cv_root.iterdir():
        summary_path = run_dir / "cv_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text()).get("summary", {})
        except (OSError, json.JSONDecodeError):
            continue
        auc, auc_err = _metric(summary, "test_roc_auc")
        if auc is None:
            continue
        brier, _ = _metric(summary, "test_brier")
        tess_auc, tess_recall = tess_slice(run_dir)
        verdict, reason = _promotion(run_dir)
        # An eight-character truncation suits a hex digest, not a named dir.
        name = run_dir.name
        short = name[:8] if len(name) == 32 and all(c in "0123456789abcdef" for c in name) else name
        records.append(
            RunRecord(
                run_id=name,
                short_id=short,
                # No run records its completion time, and the summary's mtime
                # is the DVC pull time in the container — identical across runs
                # and equal to the last deploy. Only the registry date is real.
                date=promoted_date if name == active else None,
                auc=tess_auc if tess_auc is not None else auc,
                aucErr=auc_err if tess_auc is None else None,
                recall=tess_recall,
                brier=brier,
                status="active" if name == active else "archived",
                verdict=verdict,
                reason=reason,
            )
        )

    # By id, not date: with no trustworthy timestamp any order would imply a
    # chronology nobody recorded.
    records = [r for r in records if r.status == "active"] + sorted(
        [r for r in records if r.status != "active"], key=lambda r: r.short_id
    )
    return RunsResponse(active_run_id=active, runs=records[:limit])
