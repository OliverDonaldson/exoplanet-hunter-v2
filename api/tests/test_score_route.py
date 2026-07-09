"""Route behaviour for /score/{tic_id}: registry states and outcome mapping.

The real scoring path (MAST fetch + TF ensemble) is exercised by the
network-marked integration test in the pipeline suite; here the scorer is
stubbed so the tests pin the HTTP semantics without heavy dependencies.
"""

import pytest
from app.main import app
from app.routes import score as score_module
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_scorer_singleton():
    score_module._scorer = None
    yield
    score_module._scorer = None


def stub_outcome(tic_id: int):
    from exoplanet_hunter.scoring import PhaseSeries, ScoreOutcome
    from exoplanet_hunter.scoring.diagnostics import OddEvenResult

    return ScoreOutcome(
        tic_id=tic_id,
        period_days=703.79,
        t0_btjd=1400.0,
        duration_days=0.35,
        ephemeris_source="user",
        per_fold=[0.90, 0.85, 0.88, 0.90, 0.87],
        prob_calibrated=0.88,
        prob_mean=0.90,
        prob_std=0.05,
        threshold=0.31,
        centroid_snr=4.2,
        odd_even=OddEvenResult(950.0, 940.0, 0.3),
        global_view=PhaseSeries(phase=[-0.5, 0.0, 0.5], flux=[0.0, -1.0, None]),
        local_view=PhaseSeries(phase=[-0.02, 0.0, 0.02], flux=[0.0, -1.0, 0.0]),
        verdict="stub verdict",
        model_version="cnn_dualview-cv-stub",
        n_mc_samples=50,
    )


def test_score_503_without_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    resp = client.get("/score/123")
    assert resp.status_code == 503
    assert "registry" in resp.json()["detail"]


def test_score_maps_outcome_to_contract(monkeypatch):
    class StubScorer:
        def score(self, tic_id, **kwargs):
            return stub_outcome(tic_id)

    monkeypatch.setattr(score_module, "get_scorer", lambda: StubScorer())
    body = client.get("/score/77175217").json()
    assert body["tic_id"] == 77175217
    assert body["ephemeris"]["source"] == "user"
    assert len(body["per_fold"]) == 5
    assert body["centroid"]["suspicious"] is True  # 4.2σ > 3σ BEB threshold
    assert body["odd_even"]["depth_diff_sigma"] == 0.3
    assert body["global_view"]["flux"][2] is None  # empty bins survive as null
    assert body["verdict"] == "stub verdict"


def test_score_404_when_no_lightcurve(monkeypatch):
    from exoplanet_hunter.scoring import NoLightCurveError

    class NoDataScorer:
        def score(self, tic_id, **kwargs):
            raise NoLightCurveError(f"no SPOC light curve for TIC {tic_id}")

    monkeypatch.setattr(score_module, "get_scorer", lambda: NoDataScorer())
    resp = client.get("/score/1")
    assert resp.status_code == 404
