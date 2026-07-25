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
    # The response cache is process-lifetime, so without this a TIC scored by
    # one test is served from cache to the next and never reaches its stub.
    score_module._scorer = None
    score_module._cache.clear()
    yield
    score_module._scorer = None
    score_module._cache.clear()


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


# ---------------------------------------------------------------- hardening --


class _StubScorer:
    def score(self, tic_id, **kwargs):
        return stub_outcome(tic_id)


@pytest.mark.parametrize("tic_id", ["0", "-1", "10000000001"])
def test_score_rejects_out_of_range_tic_id(monkeypatch, tic_id):
    """An unbounded path segment reaches MAST and the download manifest."""
    monkeypatch.setattr(score_module, "get_scorer", lambda: _StubScorer())
    assert client.get(f"/score/{tic_id}").status_code == 422


@pytest.mark.parametrize("tic_id", ["1", "77175217", "10000000000"])
def test_score_accepts_real_tic_id_range(monkeypatch, tic_id):
    monkeypatch.setattr(score_module, "get_scorer", lambda: _StubScorer())
    assert client.get(f"/score/{tic_id}").status_code == 200


@pytest.mark.parametrize(
    "query",
    [
        "period_days=inf&t0_btjd=0&duration_hours=1",
        "period_days=1&t0_btjd=inf&duration_hours=1",
        "period_days=1&t0_btjd=0&duration_hours=inf",
        "period_days=1e300",
    ],
)
def test_score_rejects_non_finite_ephemeris(monkeypatch, query):
    """`inf` satisfies a bare `gt=0` and poisons the phase-fold arithmetic."""
    monkeypatch.setattr(score_module, "get_scorer", lambda: _StubScorer())
    assert client.get(f"/score/12345?{query}").status_code == 422


def test_score_503_detail_carries_no_server_path(tmp_path, monkeypatch):
    def raiser():
        raise FileNotFoundError(f"{tmp_path}/models/registry.json")

    monkeypatch.setattr(score_module, "get_scorer", raiser)
    detail = client.get("/score/123").json()["detail"]
    assert str(tmp_path) not in detail
    assert "/" not in detail


def test_score_404_detail_carries_no_server_path(monkeypatch):
    """Download reasons interpolate the OSError, which carries its path."""
    from exoplanet_hunter.scoring import NoLightCurveError

    oserror = OSError(28, "No space left on device", "/srv/data/raw/.lightkurve/x.fits")

    class FailingScorer:
        def score(self, tic_id, **kwargs):
            raise NoLightCurveError(
                f"no SPOC light curve for TIC {tic_id} (fits write error: {oserror})"
            )

    monkeypatch.setattr(score_module, "get_scorer", lambda: FailingScorer())
    resp = client.get("/score/77175217")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "/srv" not in detail
    assert "x.fits" not in detail
    # The diagnosis survives; only the layout goes.
    assert "77175217" in detail and "No space left on device" in detail


@pytest.mark.parametrize(
    "reason, kept",
    [
        ("no pipeline data", "no pipeline data"),
        (
            "search error: HTTPError 500 for https://mast.stsci.edu/api/v0/x",
            "https://mast.stsci.edu/api/v0/x",
        ),
    ],
)
def test_score_404_keeps_path_free_detail(monkeypatch, reason, kept):
    """Redaction must not eat the useful reasons, including MAST URLs."""
    from exoplanet_hunter.scoring import NoLightCurveError

    class FailingScorer:
        def score(self, tic_id, **kwargs):
            raise NoLightCurveError(f"no SPOC light curve for TIC {tic_id} ({reason})")

    monkeypatch.setattr(score_module, "get_scorer", lambda: FailingScorer())
    assert kept in client.get("/score/77175217").json()["detail"]


def test_score_cache_is_bounded_and_thread_safe(monkeypatch):
    """Concurrent hits and evictions must not raise, and must respect the cap.

    The cache-hit touch and the evict-then-insert are both multi-step; two
    request threads interleaving in either used to raise KeyError (hit) or
    RuntimeError from the eviction's iterator (insert).
    """
    import threading

    monkeypatch.setattr(score_module, "get_scorer", lambda: _StubScorer())
    score_module._cache.clear()
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def hammer(worker: int) -> None:
        barrier.wait()
        for i in range(40):
            try:
                # Mixed traffic: a shared hot key (touch path) and unique keys
                # (eviction path) once the cache is at capacity.
                client.get("/score/4242")
                client.get(f"/score/12345?n_mc={10 + (worker * 40 + i) % 400}")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(score_module._cache) <= score_module._CACHE_MAX
    score_module._cache.clear()
