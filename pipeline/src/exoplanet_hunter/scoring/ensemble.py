"""Load the registered 5-fold ensemble and aggregate its predictions.

The registry (`models/registry.json`, written by the promotion gate) points
at a CV run directory of fold_*/ subdirs, each holding the fold's Keras
checkpoint and its calibration bundle (calibrator, threshold, aux_pipeline —
the V1 bundle contract). Serving loads all folds once and scores each target
with every member.

Aggregation, chosen to match what the vetting console displays:

  * per-fold prob   — the fold's MC-Dropout mean, temperature-calibrated
                      (the "five dots").
  * prob_calibrated — mean of the per-fold calibrated probs (the headline).
  * prob_mean       — mean of the raw (uncalibrated) fold means.
  * prob_std        — total uncertainty: sqrt(mean within-fold MC variance
                      + across-fold variance of the means). Both epistemic
                      terms the report cares about, in one band.
  * threshold       — mean of the folds' F1-optimal thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FoldMember:
    fold: int
    model: Any  # keras model (Any: keras import deferred to load time)
    calibrator: Any  # TemperatureScaler (sklearn-shaped .predict)
    threshold: float
    aux_pipeline: Any | None
    aux_dim: int | None


@dataclass(frozen=True)
class EnsemblePrediction:
    per_fold: list[float]  # calibrated per-fold probabilities
    prob_calibrated: float
    prob_mean: float
    prob_std: float
    threshold: float


class ScoringEnsemble:
    def __init__(self, members: list[FoldMember], run_id: str) -> None:
        if not members:
            raise ValueError("ensemble has no members")
        self.members = members
        self.run_id = run_id

    @property
    def aux_dim(self) -> int | None:
        return self.members[0].aux_dim

    @classmethod
    def from_registry(cls, models_dir: Path) -> ScoringEnsemble:
        """Load the promoted run's fold models + bundles. Raises FileNotFoundError."""
        import tensorflow as tf

        registry_path = models_dir / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"no model registry at {registry_path}")
        registry = json.loads(registry_path.read_text())
        cv_dir = Path(registry["cv_dir"])
        if not cv_dir.is_absolute():
            # The registry stores repo-relative paths; resolve against the
            # models dir's parent (the repo root) so serving works from any cwd.
            cv_dir = models_dir.parent / cv_dir

        members: list[FoldMember] = []
        for fold_dir in sorted(cv_dir.glob("fold_*")):
            bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
            members.append(
                FoldMember(
                    fold=int(fold_dir.name.split("_")[1]),
                    model=tf.keras.models.load_model(
                        str(fold_dir / "cnn_dualview.keras"), compile=False
                    ),
                    calibrator=bundle["calibrator"],
                    threshold=float(bundle["threshold"]),
                    aux_pipeline=bundle.get("aux_pipeline"),
                    aux_dim=bundle.get("aux_dim"),
                )
            )
        log.info("[ensemble] loaded %d folds from run %s", len(members), registry["run_id"])
        return cls(members, run_id=str(registry["run_id"]))

    def predict(
        self,
        global_view: np.ndarray,
        local_view: np.ndarray,
        aux_raw: np.ndarray | None,
        *,
        n_mc: int = 50,
    ) -> EnsemblePrediction:
        """Score one target: views are (bins,) float32, aux_raw is (aux_dim,) or None."""
        from exoplanet_hunter.models.uncertainty import mc_dropout_predict

        raw_means: list[float] = []
        mc_vars: list[float] = []
        calibrated: list[float] = []
        for member in self.members:
            inputs: dict[str, np.ndarray] = {
                "global_view": global_view[None, :, None].astype(np.float32),
                "local_view": local_view[None, :, None].astype(np.float32),
            }
            if member.aux_pipeline is not None:
                if aux_raw is None:
                    raise ValueError("ensemble expects aux features but none were provided")
                inputs["aux_features"] = member.aux_pipeline.transform(
                    aux_raw[None, :].astype(np.float32)
                ).astype(np.float32)
            result = mc_dropout_predict(member.model, inputs, n_samples=n_mc)
            mean = float(np.asarray(result.mean).squeeze())
            raw_means.append(mean)
            mc_vars.append(float(np.asarray(result.std).squeeze()) ** 2)
            calibrated.append(float(member.calibrator.predict(np.array([mean]))[0]))

        return EnsemblePrediction(
            per_fold=calibrated,
            prob_calibrated=float(np.mean(calibrated)),
            prob_mean=float(np.mean(raw_means)),
            prob_std=float(np.sqrt(np.mean(mc_vars) + np.var(raw_means))),
            threshold=float(np.mean([m.threshold for m in self.members])),
        )
