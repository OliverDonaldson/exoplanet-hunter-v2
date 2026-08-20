"""`GET /runs` — the CV runs on disk, newest first.

The Model Performance page shows a run history so a reader can see what the
current champion was promoted over. Serving it means the table follows the
registry: promote a new run and it becomes `active` here without an edit.

**The verdict column has no source and is returned null.** `registry.json`
records only the run being served — there is no promotion log, so nothing on
disk says why any earlier run was rejected. Those sentences existed in the
design prototype as hand-written narrative. Synthesising them here by
comparing metrics would manufacture a decision record the project never kept,
and a fabricated audit trail is worse than an absent one. Writing a real
`promotion_log.json` on each gate run is the fix; until then the column is
honestly empty.

Only runs whose `cv_summary.json` carries a pooled ROC-AUC are listed. The
directory also holds branch-architecture experiments and tuning trials that
were never promotion candidates; a table that mixed them with champions would
misrepresent what was actually in contention.
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


def _num(value: object) -> float | None:
    """A summary field that exists but is null is absent, not zero.

    Single-fold runs write `std: null`, and coercing that to 0.0 would publish
    a run with no measured spread as one with perfect agreement.
    """
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
        """TESS AUC and recall @ 1% FPR for one run.

        The history table's columns are labelled TESS, because TESS is the
        gating mission. `cv_summary.json` only holds pooled figures, so the
        slice is recomputed from the run's own predictions. A run without a
        predictions file, or without a resolvable TESS slice, returns nulls
        and the cells read as not measured.
        """
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
        # Truncating a run id to eight characters is right for a 32-hex digest
        # and wrong for a named directory, where it cuts mid-word.
        name = run_dir.name
        short = name[:8] if len(name) == 32 and all(c in "0123456789abcdef" for c in name) else name
        records.append(
            RunRecord(
                run_id=name,
                short_id=short,
                # Only the promoted run has a recorded date. No run writes its
                # own completion time, and the summary's mtime is the DVC pull
                # time inside the serving container -- identical for every run
                # and equal to the last deploy, which would read as a real date
                # and be wrong. Archived runs therefore carry no date until the
                # trainer records one.
                date=promoted_date if name == active else None,
                auc=tess_auc if tess_auc is not None else auc,
                aucErr=auc_err if tess_auc is None else None,
                recall=tess_recall,
                brier=brier,
                status="active" if name == active else "archived",
                verdict=None,  # see the module docstring
                reason=None,
            )
        )

    # Active first, then the rest by id — with no trustworthy timestamp there
    # is nothing to order archived runs by, and a fabricated order would imply
    # a chronology that is not recorded.
    records = [r for r in records if r.status == "active"] + sorted(
        [r for r in records if r.status != "active"], key=lambda r: r.short_id
    )
    return RunsResponse(active_run_id=active, runs=records[:limit])
