"""`GET /runs` — the promotion log behind the console's Verdict and Reason.

The registry records only what is currently served, so for every run it does
not name, this file is the only thing on disk that says the run was ever judged.
It postdates almost every run in `models/cv/`, which is why both columns stay
Optional and why "no log" has to render rather than raise.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from app.main import app
from app.routes import runs as runs_route
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


def test_log_with_no_reasons_reports_none_not_empty(tmp_path, monkeypatch):
    """The console renders a fallback sentence on a null and the literal cell on
    a string, so an empty string would print an empty Reason beside a verdict."""
    _registry(tmp_path, "served")
    run_dir = _run_dir(tmp_path, "terse")
    (run_dir / "promotion_log.json").write_text(json.dumps({"verdict": "PROMOTE", "reasons": []}))
    row = _rows(tmp_path, monkeypatch)["terse"]
    assert row["verdict"] == "PROMOTE"
    assert row["reason"] is None


# --------------------------------------------------------------------------
# The scan is cached and `limit` is bounded (#21).
# --------------------------------------------------------------------------


def test_a_repeat_request_does_not_reread_the_run_directories(tmp_path, monkeypatch):
    """The route opened every cv_summary.json and predictions.parquet on every
    request — 23 files and 17 MB today — and applied `limit` only afterwards,
    so the limit bounded the response and not the work."""
    _registry(tmp_path, "a" * 32)
    _run_dir(tmp_path, "a" * 32)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from app.routes import runs as runs_module

    runs_module._cache = None
    assert client.get("/runs").status_code == 200

    def forbidden(*args, **kwargs):
        raise AssertionError("re-read the run directories on a cache hit")

    monkeypatch.setattr(runs_module, "_mission_lookup", forbidden)
    assert client.get("/runs").status_code == 200


def test_a_rewritten_summary_invalidates_the_cache(tmp_path, monkeypatch):
    """A refresh rewrites these files in place, and a cache that outlived that
    would serve last week's numbers under this week's registry."""
    import os

    _registry(tmp_path, "a" * 32)
    run = _run_dir(tmp_path, "a" * 32, auc=0.91)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from app.routes import runs as runs_module

    runs_module._cache = None
    assert client.get("/runs").json()["runs"][0]["auc"] == 0.91

    (run / "cv_summary.json").write_text(
        json.dumps({"summary": {"test_roc_auc": {"mean": 0.95, "std": 0.005}}})
    )
    os.utime(run / "cv_summary.json", (0, 0))
    assert client.get("/runs").json()["runs"][0]["auc"] == 0.95


def test_a_new_run_appearing_invalidates_the_cache(tmp_path, monkeypatch):
    _registry(tmp_path, "a" * 32)
    _run_dir(tmp_path, "a" * 32)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from app.routes import runs as runs_module

    runs_module._cache = None
    assert len(client.get("/runs").json()["runs"]) == 1
    _run_dir(tmp_path, "b" * 32)
    assert len(client.get("/runs").json()["runs"]) == 2


def test_limit_is_bounded(tmp_path, monkeypatch):
    """Unbounded before: `?limit=100000` was accepted."""
    _registry(tmp_path, "a" * 32)
    _run_dir(tmp_path, "a" * 32)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert client.get("/runs?limit=100000").status_code == 422
    assert client.get("/runs?limit=0").status_code == 422
    assert client.get("/runs?limit=50").status_code == 200


def test_limit_still_slices_the_cached_list(tmp_path, monkeypatch):
    """One cache entry has to serve every limit, or the cache is per-limit."""
    _registry(tmp_path, "a" * 32)
    for name in ("a" * 32, "b" * 32, "c" * 32):
        _run_dir(tmp_path, name)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from app.routes import runs as runs_module

    runs_module._cache = None
    assert len(client.get("/runs?limit=3").json()["runs"]) == 3
    assert len(client.get("/runs?limit=1").json()["runs"]) == 1


def test_a_run_without_predictions_is_labelled_pooled(tmp_path, monkeypatch):
    """#96: the Brier was pooled while the AUC beside it was TESS, and the
    console rendered both under a "TESS AUC" heading. A row now names the
    population all three of its metrics came from.
    """
    _run_dir(tmp_path, "abc", auc=0.95)
    _registry(tmp_path, "abc")
    row = _rows(tmp_path, monkeypatch)["abc"]
    assert row["slice"] == "pooled"
    assert row["auc"] == pytest.approx(0.95)
    assert row["aucErr"] == pytest.approx(0.005)
    assert row["recall"] is None


def test_a_tess_row_carries_the_tess_brier_not_the_pooled_one(tmp_path, monkeypatch):
    """On the served run the two differ by 0.04, so mixing them was not rounding."""
    run_dir = _run_dir(tmp_path, "abc", auc=0.95)
    (run_dir / "cv_summary.json").write_text(
        json.dumps({"summary": {"test_roc_auc": {"mean": 0.95}, "test_brier": {"mean": 0.079}}})
    )
    pd.DataFrame(
        {
            "tic_id": [1, 2, 3, 4],
            "y_true": [0, 1, 0, 1],
            "prob_calibrated": [0.1, 0.9, 0.2, 0.8],
        }
    ).to_parquet(run_dir / "predictions.parquet")
    _registry(tmp_path, "abc")
    monkeypatch.setattr(
        runs_route,
        "_mission_lookup",
        lambda _: pd.DataFrame({"tic_id": [1, 2, 3, 4], "mission": ["TESS"] * 4}),
    )
    row = _rows(tmp_path, monkeypatch)["abc"]
    assert row["slice"] == "TESS"
    # mean((p - y)^2) over the four rows, not the 0.079 in the summary
    assert row["brier"] == pytest.approx(0.025)
    assert row["aucErr"] is None
