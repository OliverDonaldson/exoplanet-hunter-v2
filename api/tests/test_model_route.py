"""`GET /model` — the noise floor the console prints under the served metrics.

A floor belongs to the architecture and the run it was measured on. This route
published two constants from a *branch*-model calibration under the *dual-view*
champion's numbers for a month, which is a category error rather than a stale
figure: no amount of re-measuring the branch model makes those the champion's
floor. These tests pin the two outcomes that replaced them — a run's own floor
where the run can produce one, and an explicit "not measured" where it cannot.
"""

import json
from pathlib import Path

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

RUN = "ca906040cdb74ba6b07353a500244777"


def _served(models_dir: Path, summary: dict) -> None:
    """A registry naming one run, and that run's summary. No predictions, so
    `per_mission` stays empty and the response is the floor and the metrics."""
    run_dir = models_dir / "cv" / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cv_summary.json").write_text(json.dumps(summary))
    # Absolute, because the route resolves a relative `cv_summary` against the
    # models directory's PARENT — correct for the repo layout, wrong for a
    # tmp_path whose parent holds nothing.
    (models_dir / "registry.json").write_text(
        json.dumps(
            {
                "run_id": RUN,
                "cv_summary": str(run_dir / "cv_summary.json"),
                "cv_dir": str(run_dir),
                "promoted_at": "2026-07-19T11:14:12+00:00",
            }
        )
    )


def _floor(models_dir: Path, monkeypatch) -> dict:
    monkeypatch.setenv("MODEL_DIR", str(models_dir))
    response = client.get("/model")
    assert response.status_code == 200, response.text
    return response.json()["noise_floor"]


def test_a_single_member_run_reports_no_floor_rather_than_a_number(tmp_path, monkeypatch):
    """The served champion's case. One model per fold has no seed spread, so
    `2 * sd / sqrt(n)` has nothing to average over. Nulls plus a reason, never a
    figure borrowed from a run that did measure one."""
    _served(tmp_path, {"summary": {"test_roc_auc": {"mean": 0.958, "std": 0.006}}})
    floor = _floor(tmp_path, monkeypatch)

    assert floor["measured"] is False
    assert floor["auc"] is None
    assert floor["recall"] is None
    assert floor["source"]


def test_a_multi_member_run_reports_the_floor_its_own_members_measured(tmp_path, monkeypatch):
    """`2 * sd / sqrt(n_models_per_fold)`, over the spread the trainer wrote.
    The figures below are Phase 1a seed 44's, whose own log reports
    `auc_floor seed_sd 0.0143` and `pooled gate floor seed_sd 0.0625` at n=3."""
    _served(
        tmp_path,
        {
            "summary": {
                "test_roc_auc": {"mean": 0.9317, "std": 0.014},
                "variance": {
                    "seed_sd": 0.014336329217202667,
                    "pooled_gate_recall_seed_sd": 0.06253204306798563,
                    "n_models_per_fold": 3,
                },
            }
        },
    )
    floor = _floor(tmp_path, monkeypatch)

    assert floor["measured"] is True
    assert floor["n_models_per_fold"] == 3
    assert floor["auc"] == 0.016554167065486115
    assert floor["recall"] == 0.07220578379655755


def test_the_pooled_gate_draw_is_preferred_over_the_per_fold_one(tmp_path, monkeypatch):
    """Both keys exist and they are different measurements. The promotion gate
    decides on the pooled draw, so that is the one published; publishing the
    per-fold spread would print a floor no decision is read against."""
    _served(
        tmp_path,
        {
            "summary": {
                "test_roc_auc": {"mean": 0.93, "std": 0.01},
                "variance": {
                    "seed_sd": 0.01,
                    "gate_recall_seed_sd": 0.02,
                    "pooled_gate_recall_seed_sd": 0.05,
                    "n_models_per_fold": 4,
                },
            }
        },
    )
    floor = _floor(tmp_path, monkeypatch)

    assert floor["recall"] == 0.05  # 2 * 0.05 / sqrt(4), not 2 * 0.02 / sqrt(4)


def test_a_variance_block_missing_a_statistic_nulls_that_one_and_keeps_the_other(
    tmp_path, monkeypatch
):
    """Partial is not absent. A run that recorded an AUC spread and no recall
    spread has a measured AUC floor, and saying otherwise would discard a real
    measurement to keep the pair tidy."""
    _served(
        tmp_path,
        {
            "summary": {
                "test_roc_auc": {"mean": 0.93, "std": 0.01},
                "variance": {"seed_sd": 0.02, "n_models_per_fold": 3},
            }
        },
    )
    floor = _floor(tmp_path, monkeypatch)

    assert floor["measured"] is True
    assert floor["auc"] is not None
    assert floor["recall"] is None
