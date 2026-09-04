"""`GET /runs` — the promotion log behind the console's Verdict and Reason.

The registry records only what is currently served, so for every run it does
not name, this file is the only thing on disk that says the run was ever judged.
It postdates almost every run in `models/cv/`, which is why both columns stay
Optional and why "no log" has to render rather than raise.
"""

import json
from pathlib import Path

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _run_dir(models_dir: Path, name: str, auc: float = 0.91) -> Path:
    """A run the route will list: it needs a pooled ROC-AUC and nothing else.

    No `predictions.parquet`, so the TESS slice resolves to null — this suite is
    about the promotion log, and a run with predictions would also exercise the
    parquet merge in `tess_slice`.
    """
    run_dir = models_dir / "cv" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cv_summary.json").write_text(
        json.dumps({"summary": {"test_roc_auc": {"mean": auc, "std": 0.005}}})
    )
    return run_dir


def _registry(models_dir: Path, active: str) -> None:
    (models_dir / "registry.json").write_text(
        json.dumps({"run_id": active, "promoted_at": "2026-07-19T00:00:00+00:00"})
    )


def _rows(models_dir: Path, monkeypatch) -> dict[str, dict]:
    monkeypatch.setenv("MODEL_DIR", str(models_dir))
    body = client.get("/runs").json()
    return {row["run_id"]: row for row in body["runs"]}


def test_a_run_with_a_log_serves_its_verdict_and_reason(tmp_path, monkeypatch):
    """The whole point of the log. Hardcoded nulls here are what made the console
    print "no promotion log is written yet" beside a verdict the gate had in fact
    computed, every week, into a directory it then deleted."""
    _registry(tmp_path, "served")
    run_dir = _run_dir(tmp_path, "rejected")
    (run_dir / "promotion_log.json").write_text(
        json.dumps(
            {
                "verdict": "REJECT",
                "reasons": ["gated on TESS (n=2399)", "recall @1% FPR 0.238 vs champion 0.307"],
                "alarms": [],
                "thresholds": {"recall_tolerance": 0.0337},
                "candidate_run_id": "rejected",
                "champion_run_id": "served",
                "decided_at": "2026-08-28T01:00:00+00:00",
            }
        )
    )
    row = _rows(tmp_path, monkeypatch)["rejected"]
    assert row["verdict"] == "REJECT"
    # One readable string, in the gate's own order: the console has a single
    # Reason cell and joining is the route's job, not the reader's.
    assert row["reason"] == "gated on TESS (n=2399); recall @1% FPR 0.238 vs champion 0.307"


def test_a_run_without_a_log_still_renders(tmp_path, monkeypatch):
    """Every run on disk today predates the log. Dropping those rows, or 500ing
    on them, would empty the history table to publish one new column."""
    _registry(tmp_path, "served")
    _run_dir(tmp_path, "served")
    _run_dir(tmp_path, "legacy")
    rows = _rows(tmp_path, monkeypatch)
    assert set(rows) == {"served", "legacy"}
    assert rows["legacy"]["verdict"] is None
    assert rows["legacy"]["reason"] is None


def test_an_unreadable_log_loses_the_verdict_and_not_the_run(tmp_path, monkeypatch):
    """A half-written or hand-edited log is indistinguishable from none at all
    for display purposes, and taking the table down over one is a worse trade
    than showing eleven rows and a blank cell."""
    _registry(tmp_path, "served")
    run_dir = _run_dir(tmp_path, "corrupt")
    (run_dir / "promotion_log.json").write_text("{not json")
    row = _rows(tmp_path, monkeypatch)["corrupt"]
    assert row["verdict"] is None
    assert row["reason"] is None


def test_a_log_with_no_reasons_reports_no_reason_rather_than_an_empty_string(tmp_path, monkeypatch):
    """The console renders a fallback sentence on a null and the literal cell on
    a string, so an empty string would print an empty Reason beside a verdict."""
    _registry(tmp_path, "served")
    run_dir = _run_dir(tmp_path, "terse")
    (run_dir / "promotion_log.json").write_text(json.dumps({"verdict": "PROMOTE", "reasons": []}))
    row = _rows(tmp_path, monkeypatch)["terse"]
    assert row["verdict"] == "PROMOTE"
    assert row["reason"] is None
