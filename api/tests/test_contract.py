"""Contract tests for the serving layer.

These pin the `/score/{tic_id}` JSON shape so the FastAPI endpoint and the
React console evolve against the same target instead of drifting apart.
Route behaviour is covered in test_score_route.py; here we only exercise
the schema and the health/degraded states, with MODEL_DIR pointed at
controlled directories so results don't depend on what's on this machine.
"""

import json

from app.main import app
from app.schemas import ScoreResponse
from fastapi.testclient import TestClient

client = TestClient(app)


def test_healthz_degraded_without_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_healthz_ok_with_registry(tmp_path, monkeypatch):
    (tmp_path / "registry.json").write_text(json.dumps({"run_id": "abcdef1234567890"}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "cnn_dualview-cv-abcdef12"


def test_score_response_schema_roundtrips() -> None:
    """A representative payload validates — the frontend types mirror this."""
    example = {
        "tic_id": 307210830,
        "ephemeris": {
            "period_days": 703.0,
            "t0_btjd": 1400.5,
            "duration_days": 0.35,
            "source": "bls",
        },
        "prob_calibrated": 0.91,
        "prob_mean": 0.88,
        "prob_std": 0.04,
        "per_fold": [{"fold": i, "prob": p} for i, p in enumerate([0.90, 0.85, 0.87, 0.89, 0.91])],
        "decision_threshold": 0.5,
        "centroid": {"centroid_snr": 1.2, "beb_threshold_sigma": 3.0, "suspicious": False},
        "odd_even": {"odd_depth_ppm": 950.0, "even_depth_ppm": 940.0, "depth_diff_sigma": 0.3},
        "duration_check": {
            "q": 0.0005,
            "q_circ": 0.0015,
            "q_ratio": 0.33,
            "a_over_rstar": 215.0,
            "suspicious": True,
        },
        "global_view": {"phase": [-0.5, 0.0, 0.5], "flux": [0.0, -0.001, None]},
        "local_view": {"phase": [-0.02, 0.0, 0.02], "flux": [0.0, -0.001, 0.0]},
        "verdict": "Consistent with an on-target planetary transit.",
        "model_version": "cnn_dualview-cv-e5388ed9",
        "n_mc_samples": 50,
    }
    parsed = ScoreResponse.model_validate(example)
    assert parsed.tic_id == 307210830
    assert len(parsed.per_fold) == 5
    assert parsed.duration_check is not None and parsed.duration_check.suspicious

    # New cautions are optional: a payload without them still validates.
    del example["duration_check"]
    assert ScoreResponse.model_validate(example).duration_check is None
