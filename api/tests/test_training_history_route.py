"""`GET /model/training-history` — the panel that told visitors the data was gone.

`app.pages.js` said for months that per-epoch metrics "are not persisted by the
training job yet". They were, in the served run's MLflow child runs. What was
missing was a path to the browser: the MLflow store is 33 MB of local
development state and never enters the serving image, so the route reads a
small export instead. Issue #25.

The two cases that matter are a run with an export and a run without: the
console renders those differently and must be able to tell them apart.
"""

import json
from pathlib import Path

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

RUN = "ca906040cdb74ba6b07353a500244777"


def _registry(models_dir: Path) -> None:
    (models_dir).mkdir(parents=True, exist_ok=True)
    (models_dir / "registry.json").write_text(json.dumps({"run_id": RUN}))


def _history(models_dir: Path, folds: list[dict]) -> None:
    (models_dir / "history").mkdir(parents=True, exist_ok=True)
    (models_dir / "history" / f"{RUN}.json").write_text(
        json.dumps(
            {
                "run_id": RUN,
                "monitor": "val_auc",
                "monitor_mode": "max",
                "patience": 25,
                "n_folds": len(folds),
                "n_folds_with_history": sum(1 for f in folds if f["epochs"]),
                "folds": folds,
            }
        )
    )


def _fold(index: int, epochs: int) -> dict:
    series = [float(i) for i in range(epochs)]
    return {
        "fold": index,
        "epochs": epochs,
        "loss": series,
        "val_loss": series,
        "auc": series,
        "val_auc": series,
        "learning_rate": series,
        "restored_epoch": epochs - 26,
        "stopped_epoch": epochs - 1,
    }


def test_the_served_runs_curves_are_returned(tmp_path, monkeypatch):
    _registry(tmp_path)
    _history(tmp_path, [_fold(1, 77), _fold(2, 78)])
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    body = client.get("/model/training-history").json()

    assert body["run_id"] == RUN
    assert body["monitor"] == "val_auc"
    assert [f["epochs"] for f in body["folds"]] == [77, 78]
    assert len(body["folds"][0]["val_auc"]) == 77


def test_a_fold_with_no_history_survives_the_response(tmp_path, monkeypatch):
    """The served run's fold 0. Dropping it would leave the console drawing
    four curves for a five-fold run with nothing on screen saying so."""
    _registry(tmp_path)
    empty = {"fold": 0, "epochs": 0, "note": "no per-epoch history was logged for this fold"}
    _history(tmp_path, [empty, _fold(1, 77)])
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    body = client.get("/model/training-history").json()

    assert body["n_folds"] == 2
    assert body["n_folds_with_history"] == 1
    assert body["folds"][0]["epochs"] == 0
    assert body["folds"][0]["val_auc"] is None
    assert body["folds"][0]["note"]


def test_a_run_with_no_export_404s_and_names_the_script(tmp_path, monkeypatch):
    """Not an empty body. "No history for this run" and "the endpoint returned
    nothing" are different facts and the panel states them differently."""
    _registry(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    response = client.get("/model/training-history")

    assert response.status_code == 404
    assert "export_training_history.py" in response.json()["detail"]


def test_no_registry_is_a_503_not_a_404(tmp_path, monkeypatch):
    """A service with no promoted model is unavailable, not missing a file."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))

    assert client.get("/model/training-history").status_code == 503
