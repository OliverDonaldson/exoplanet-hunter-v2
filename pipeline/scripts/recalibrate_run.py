"""Refit an existing CV run's calibration bundles in place — no retraining.

Why this exists: the full-scale expansion run (cebb0fe6) shipped with
temperature-only calibration, which cannot correct the wholesale downward
shift its raw scores exhibited (ECE 0.136 against the incumbent's 0.031).
The trainer now fits a `PlattScaler` per fold; this script backfills a run
that predates that change so it serves honest probabilities without paying
for a retrain.

Per-fold validation scores are not persisted, so each fold's calibrator is
fitted on the pooled out-of-fold test predictions of the *other* folds
(cross-fitted). That is ~8x more examples than a single validation split,
drawn from the same held-out-score distribution, and fold f's own test rows
never touch fold f's fit — the rewritten per-fold test metrics stay
leakage-clean. The F1 threshold is re-swept on the same fit rows in
calibrated space, matching the trainer.

Rewrites, for each fold: `cnn_calibrator.joblib` (calibrator + threshold;
aux pipeline untouched) and `predictions.parquet` (prob_calibrated), then
the run-level `predictions.parquet` and `cv_summary.json` (now including
`test_ece`). ROC/PR-AUC are unchanged by construction — Platt is monotone.

Usage (from the repository root):

    python pipeline/scripts/recalibrate_run.py models/cv/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from exoplanet_hunter.training.calibration import PlattScaler, expected_calibration_error
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def recalibrate_fold(fold_dir: Path, preds: pd.DataFrame, fold_idx: int) -> dict[str, float]:
    """Fit on the other folds' OOF rows, rewrite this fold's bundle + parquet."""
    fit_rows = preds[preds["fold"] != fold_idx]
    calibrator = PlattScaler.from_validation(
        fit_rows["prob_raw"].to_numpy(), fit_rows["y_true"].to_numpy()
    )

    fit_cal = calibrator.predict(fit_rows["prob_raw"].to_numpy())
    fit_y = fit_rows["y_true"].to_numpy()
    f1s = [f1_score(fit_y, (fit_cal >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    best_threshold = float(THRESHOLDS[int(np.argmax(f1s))])

    bundle_path = fold_dir / "cnn_calibrator.joblib"
    bundle = joblib.load(bundle_path)
    bundle.pop("temperature", None)
    bundle.update(
        calibrator=calibrator,
        platt_a=calibrator.a,
        platt_b=calibrator.b,
        threshold=best_threshold,
    )
    joblib.dump(bundle, bundle_path)

    fold_preds = pd.read_parquet(fold_dir / "predictions.parquet")
    fold_preds["prob_calibrated"] = calibrator.predict(fold_preds["prob_raw"].to_numpy())
    fold_preds.to_parquet(fold_dir / "predictions.parquet", index=False)

    test_y = fold_preds["y_true"].to_numpy()
    test_cal = fold_preds["prob_calibrated"].to_numpy()
    log.info(
        "[fold %d] platt a=%.4f b=%.4f  threshold=%.2f",
        fold_idx,
        calibrator.a,
        calibrator.b,
        best_threshold,
    )
    return {
        "test_roc_auc": float(roc_auc_score(test_y, test_cal)),
        "test_pr_auc": float(average_precision_score(test_y, test_cal)),
        "test_f1": float(
            f1_score(test_y, (test_cal >= best_threshold).astype(int), zero_division=0)
        ),
        "test_brier": float(brier_score_loss(test_y, test_cal)),
        "test_ece": float(expected_calibration_error(test_y, test_cal)),
        "best_threshold": best_threshold,
        "platt_a": float(calibrator.a),
        "platt_b": float(calibrator.b),
        "fold": fold_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="models/cv/<run_id> directory to recalibrate")
    args = parser.parse_args()

    fold_dirs = sorted(args.run_dir.glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"no fold_* directories under {args.run_dir}")

    preds = pd.concat(
        [pd.read_parquet(d / "predictions.parquet") for d in fold_dirs], ignore_index=True
    )
    before_ece = expected_calibration_error(
        preds["y_true"].to_numpy(), preds["prob_calibrated"].to_numpy()
    )
    before_brier = float(
        brier_score_loss(preds["y_true"].to_numpy(), preds["prob_calibrated"].to_numpy())
    )

    fold_rows = [recalibrate_fold(d, preds, int(d.name.removeprefix("fold_"))) for d in fold_dirs]

    # Same aggregation keys as the trainer's _aggregate_cv (kept in sync).
    keys = (
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_brier",
        "test_ece",
        "best_threshold",
        "platt_a",
        "platt_b",
    )
    summary = {
        k: {
            "mean": float(np.mean([m[k] for m in fold_rows])),
            "std": float(np.std([m[k] for m in fold_rows])),
        }
        for k in keys
    }
    (args.run_dir / "cv_summary.json").write_text(
        json.dumps({"folds": fold_rows, "summary": summary}, indent=2)
    )

    new_preds = pd.concat(
        [pd.read_parquet(d / "predictions.parquet") for d in fold_dirs], ignore_index=True
    )
    new_preds.to_parquet(args.run_dir / "predictions.parquet", index=False)

    after_ece = expected_calibration_error(
        new_preds["y_true"].to_numpy(), new_preds["prob_calibrated"].to_numpy()
    )
    log.info(
        "[recalibrate] ECE %.4f -> %.4f  Brier %.4f -> %.4f  (ROC-AUC %.4f, unchanged)",
        before_ece,
        after_ece,
        before_brier,
        summary["test_brier"]["mean"],
        summary["test_roc_auc"]["mean"],
    )


if __name__ == "__main__":
    main()
