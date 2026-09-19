"""Load the registered 5-fold ensemble and aggregate its predictions.

`models/registry.json`, written by the promotion gate, points at a CV run
directory of `fold_*/` subdirs, each holding that fold's Keras checkpoint and
its calibration bundle (calibrator, threshold, aux_pipeline — the V1 bundle
contract). Serving loads all folds once, scoring each target with every member.

Aggregation matches what the vetting console displays: a per-fold calibrated
prob (the "five dots"), `prob_calibrated` as their mean (the headline),
`prob_mean` over the raw scores, `prob_std` combining within-fold MC-Dropout
variance with across-fold variance, and `threshold` as the mean of the folds'
F1-optimal thresholds. Calibrators are fitted on deterministic scores, so MC
means never feed them.
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
    models: list[Any]  # keras models (Any: keras import deferred to load time)
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
    # Seed-to-seed spread within a fold, 0.0 on a single-member run. Kept out of
    # `prob_std` because it is the quantity the promotion gate measures, and
    # folding it into MC noise would hide it.
    prob_std_member: float = 0.0


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
        registry_path = models_dir / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"no model registry at {registry_path}")
        registry = json.loads(registry_path.read_text())
        cv_dir = Path(registry["cv_dir"])
        if not cv_dir.is_absolute():
            # The registry stores repo-relative paths; resolve against the
            # models dir's parent (the repo root) so serving works from any cwd.
            cv_dir = models_dir.parent / cv_dir
        return cls.from_cv_dir(cv_dir, run_id=str(registry["run_id"]))

    @classmethod
    def from_cv_dir(cls, cv_dir: Path, run_id: str | None = None) -> ScoringEnsemble:
        """Load fold models + bundles from any CV run dir — the path that reads an
        off-registry arm without promoting it. Raises FileNotFoundError."""
        import tensorflow as tf

        fold_dirs = sorted(cv_dir.glob("fold_*"))
        if not fold_dirs:
            raise FileNotFoundError(f"no fold_* directories under {cv_dir}")

        members: list[FoldMember] = []
        for fold_dir in fold_dirs:
            # `member_checkpoint_name` keeps the bare name at one member per fold
            # and numbers it past that, so both layouts are on disk in the wild.
            bare = fold_dir / "cnn_dualview.keras"
            ckpts = [bare] if bare.exists() else sorted(fold_dir.glob("model_*_cnn_dualview.keras"))
            if not ckpts:
                found = sorted(q.name for q in fold_dir.glob("*.keras"))
                raise FileNotFoundError(
                    f"no cnn_dualview checkpoint in {fold_dir} "
                    f"(found: {found or 'no .keras files'}); this loader serves the "
                    "dual-view architecture, and a branch run needs its own"
                )
            bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
            members.append(
                FoldMember(
                    fold=int(fold_dir.name.split("_")[1]),
                    models=[tf.keras.models.load_model(str(c), compile=False) for c in ckpts],
                    calibrator=bundle["calibrator"],
                    threshold=float(bundle["threshold"]),
                    aux_pipeline=bundle.get("aux_pipeline"),
                    aux_dim=bundle.get("aux_dim"),
                )
            )
        run = run_id or cv_dir.name
        log.info(
            "[ensemble] loaded %d folds x %d members from run %s",
            len(members),
            len(members[0].models) if members else 0,
            run,
        )
        return cls(members, run_id=run)

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
        member_vars: list[float] = []
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
            # Calibrated headline from the deterministic pass: calibrators are
            # fitted on deterministic scores, and feeding them MC means costs
            # ~0.08 ECE. MC sampling contributes only the uncertainty band.
            dets = [float(np.asarray(m(inputs, training=False)).squeeze()) for m in member.models]
            mcs = [
                float(np.asarray(mc_dropout_predict(m, inputs, n_samples=n_mc).std).squeeze())
                for m in member.models
            ]
            # `train.py` averages the members' RAW scores and fits one Platt on
            # that average, so serving has to calibrate the same quantity.
            n = len(dets)
            det = float(np.mean(dets))
            raw_means.append(det)
            # MC variance of a mean of n members, which is their mean variance
            # over n. At n=1 this is the single-model term it replaces.
            mc_vars.append(float(np.mean(np.square(mcs))) / n)
            if n > 1:
                member_vars.append(float(np.var(dets, ddof=1)))
            calibrated.append(float(member.calibrator.predict(np.array([det]))[0]))

        return EnsemblePrediction(
            per_fold=calibrated,
            prob_calibrated=float(np.mean(calibrated)),
            prob_mean=float(np.mean(raw_means)),
            prob_std=float(np.sqrt(np.mean(mc_vars) + np.var(raw_means))),
            threshold=float(np.mean([m.threshold for m in self.members])),
            prob_std_member=float(np.sqrt(np.mean(member_vars))) if member_vars else 0.0,
        )
