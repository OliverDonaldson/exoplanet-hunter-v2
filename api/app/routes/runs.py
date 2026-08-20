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
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

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
    active = str(json.loads(registry_path.read_text())["run_id"])

    cv_root = models_dir / "cv"
    if not cv_root.is_dir():
        return RunsResponse(active_run_id=active, runs=[])

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
        # No run records its own completion time, so the summary's mtime is the
        # closest honest stand-in. Labelled `date` rather than `promoted_at`
        # because for every archived run those are different things.
        stamp = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
        records.append(
            RunRecord(
                run_id=run_dir.name,
                short_id=run_dir.name[:8],
                date=stamp.date().isoformat(),
                auc=auc,
                aucErr=auc_err,
                brier=brier,
                status="active" if run_dir.name == active else "archived",
                verdict=None,  # see the module docstring
                reason=None,
            )
        )

    records.sort(key=lambda r: (r.status != "active", r.date), reverse=False)
    records = [r for r in records if r.status == "active"] + sorted(
        [r for r in records if r.status != "active"], key=lambda r: r.date, reverse=True
    )
    return RunsResponse(active_run_id=active, runs=records[:limit])
