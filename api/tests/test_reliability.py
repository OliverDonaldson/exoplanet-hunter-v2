"""Tests for the /reliability endpoint against fixture predictions."""

import json

import numpy as np
import pandas as pd
import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture()
def promoted_run(tmp_path, monkeypatch):
    cv_dir = tmp_path / "cv" / "runX"
    cv_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    # A perfectly calibrated toy model: y ~ Bernoulli(p).
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    pd.DataFrame(
        {
            "row": np.arange(2000),
            "tic_id": np.arange(2000),
            "fold": 0,
            "y_true": y,
            "prob_raw": p,
            "prob_calibrated": p,
        }
    ).to_parquet(cv_dir / "predictions.parquet", index=False)
    (tmp_path / "registry.json").write_text(json.dumps({"run_id": "runX", "cv_dir": str(cv_dir)}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    return tmp_path


def test_reliability_503_without_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert client.get("/reliability").status_code == 503


def test_reliability_503_without_predictions(tmp_path, monkeypatch):
    (tmp_path / "registry.json").write_text(
        json.dumps({"run_id": "r", "cv_dir": str(tmp_path / "cv" / "r")})
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    resp = client.get("/reliability")
    assert resp.status_code == 503
    assert "export_predictions" in resp.json()["detail"]


def test_reliability_bins_of_calibrated_model(promoted_run):
    body = client.get("/reliability").json()
    assert body["run_id"] == "runX"
    assert body["n_examples"] == 2000
    assert len(body["bins"]) == 10
    # A calibrated model sits near the diagonal with tiny ECE.
    for b in body["bins"]:
        assert abs(b["frac_positive"] - b["prob_mean"]) < 0.12
    assert body["ece"] < 0.05
    assert sum(b["count"] for b in body["bins"]) == 2000
