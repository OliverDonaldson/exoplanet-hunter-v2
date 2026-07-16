"""Optuna hyperparameter search for the dual-view CNN (V1 port).

Each trial calls `train.run` on a config copy with suggested overrides and
logs to MLflow as a nested run. Current hyperparameters date from the
881-example era; the full pool is 5.5x larger.

A full 5-fold trial is expensive — cheapen trials on the CLI:

    python -m exoplanet_hunter.training.tune train=tune \\
        model.cross_validation.n_splits=2 train.epochs=60

The study persists to `train.optuna.storage` (sqlite); re-running the same
command resumes it, adding up to `n_trials` more trials.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf

from exoplanet_hunter.training.mlflow_utils import setup_mlflow
from exoplanet_hunter.training.train import run as train_run
from exoplanet_hunter.utils import get_logger, set_global_seed

log = get_logger(__name__)


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Translate a YAML search-space entry into an Optuna `suggest_*` call."""
    kind = spec["type"]
    if kind == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if kind == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"unknown search-space type: {kind}")


def run_study(
    cfg: DictConfig,
    train_fn: Callable[[DictConfig], float] = train_run,
) -> float:
    """Optimize the search space; `train_fn` scores one config (injectable for tests)."""
    if cfg.train.name != "tune":
        raise SystemExit("tune entry point requires train=tune")

    set_global_seed(int(cfg.seed))
    setup_mlflow(cfg)

    pruner = hydra.utils.instantiate(cfg.train.optuna.pruner)
    study = optuna.create_study(
        direction=str(cfg.train.optuna.direction),
        pruner=pruner,
        study_name=f"{cfg.project_name}-{cfg.model.name}-{cfg.data.name}",
        storage=str(cfg.train.optuna.storage),
        load_if_exists=True,
    )

    parent_run = mlflow.start_run(run_name=f"tune-{cfg.model.name}-{cfg.data.name}")

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_yaml(cfg)))
        for key, spec in cfg.train.search_space.items():
            value = _suggest(trial, key, spec)
            OmegaConf.update(trial_cfg, key, value, force_add=True)

        log.info("[tune] trial %d -> %s", trial.number, trial.params)
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params({f"trial.{k}": v for k, v in trial.params.items()})
            # The non-@hydra.main `run`, so trials don't re-parse sys.argv.
            score = float(train_fn(trial_cfg))
            mlflow.log_metric(str(cfg.train.optuna.metric), score)
            return score

    try:
        study.optimize(
            objective,
            n_trials=int(cfg.train.optuna.n_trials),
            timeout=int(cfg.train.optuna.timeout) if cfg.train.optuna.timeout else None,
        )
        log.info("[tune] best score = %.4f", study.best_value)
        log.info("[tune] best params = %s", study.best_params)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best.{k}", v)
        mlflow.log_metric("best_value", float(study.best_value))
        return float(study.best_value)
    finally:
        if study.trials:
            out_dir = Path(str(cfg.paths.results)) / "tune"
            out_dir.mkdir(parents=True, exist_ok=True)
            study.trials_dataframe().to_parquet(out_dir / "trials.parquet", index=False)
        mlflow.end_run()
        if parent_run:
            mlflow.end_run()


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    return run_study(cfg)


if __name__ == "__main__":
    main()
