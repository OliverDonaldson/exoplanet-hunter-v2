"""Tuning-harness seams: MLflow run nesting and search-space key validity."""

import mlflow
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from exoplanet_hunter.training.mlflow_utils import start_root_run
from exoplanet_hunter.training.tune import run_study


@pytest.fixture()
def tmp_tracking(tmp_path):
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-tuning")
    yield
    mlflow.set_tracking_uri(old)


def test_start_root_run_standalone(tmp_tracking):
    with start_root_run("solo") as run:
        pass
    tags = mlflow.get_run(run.info.run_id).data.tags
    assert "mlflow.parentRunId" not in tags


def test_start_root_run_nests_under_active_run(tmp_tracking):
    with mlflow.start_run(run_name="outer") as outer, start_root_run("inner") as inner:
        pass
    tags = mlflow.get_run(inner.info.run_id).data.tags
    assert tags.get("mlflow.parentRunId") == outer.info.run_id


def test_tune_search_space_keys_exist():
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="config", overrides=["train=tune"])
    for key in cfg.train.search_space:
        assert OmegaConf.select(cfg, str(key)) is not None, (
            f"search-space key {key!r} is not a real config path — "
            "the tuner would silently no-op on it"
        )


def test_run_study_loop(tmp_path, tmp_tracking, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="config", overrides=["train=tune"])
    # Literal root: ${hydra:runtime.cwd} (the oc.env default) needs a real app.
    cfg.paths.root = str(tmp_path)
    cfg.train.optuna.n_trials = 3

    seen: list = []

    def fake_train(trial_cfg):
        seen.append(trial_cfg)
        return float(trial_cfg.model.head.dropout)

    best = run_study(cfg, train_fn=fake_train)

    assert len(seen) == 3
    for trial_cfg in seen:
        # Suggested values must land on the real keys, not force-added strays.
        assert 0.1 <= float(trial_cfg.model.head.dropout) <= 0.5
        assert int(trial_cfg.train.batch_size) in (32, 64, 128, 256)
        assert 1e-5 <= float(trial_cfg.train.optimizer.learning_rate) <= 1e-2
    assert best == max(float(c.model.head.dropout) for c in seen)
    assert (tmp_path / "results" / "tune" / "trials.parquet").exists()
    assert (tmp_path / "optuna.db").exists()
