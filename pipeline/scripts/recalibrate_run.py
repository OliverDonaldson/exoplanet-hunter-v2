"""Refit an existing CV run's calibration bundles in place — no retraining.

Validation scores are not persisted, so each fold's Platt calibrator is
fitted on the pooled out-of-fold predictions of the *other* folds; fold f's
own test rows never touch its fit, keeping the rewritten metrics clean.

`--rescore` first regenerates each fold's prob_raw from its saved checkpoint
(deterministic pass over the shards), for runs whose parquet was scored with
in-memory weights that drifted from the shipped checkpoint.

Usage (from the repository root):

    python pipeline/scripts/recalibrate_run.py models/cv/<run_id> [--rescore]
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


def rescore_fold(fold_dir: Path, shard_dir: Path) -> None:
    """Overwrite the fold parquet's prob_raw with the checkpoint's scores."""
    import tensorflow as tf

    from exoplanet_hunter.datasets import (
        ShardMetadata,
        Split,
        aux_constants_from_pipeline,
        list_shards,
        load_index,
        make_dataset,
        make_split_table,
    )

    fold_preds = pd.read_parquet(fold_dir / "predictions.parquet").sort_values("row")
    index = load_index(shard_dir)
    codes = np.full(len(index), int(Split.TRAIN), dtype=np.int64)
    codes[fold_preds["row"].to_numpy()] = int(Split.TEST)

    bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
    aux_pipeline = bundle.get("aux_pipeline")
    ds = make_dataset(
        list_shards(shard_dir),
        split=Split.TEST,
        metadata=ShardMetadata.load(shard_dir),
        split_table=make_split_table(index["tic_id"].to_numpy(), codes),
        aux_constants=aux_constants_from_pipeline(aux_pipeline) if aux_pipeline else None,
        use_aux=aux_pipeline is not None,
        batch_size=256,
        seed=0,
    )
    model = tf.keras.models.load_model(str(fold_dir / "cnn_dualview.keras"), compile=False)
    scores, labels = [], []
    for feats, y in ds:
        scores.append(np.asarray(model(feats, training=False)).squeeze())
        labels.append(y.numpy())
    if not np.array_equal(np.concatenate(labels), fold_preds["y_true"].to_numpy()):
        raise SystemExit(f"{fold_dir.name}: stream order does not match the parquet — aborting")
    drift = float(np.abs(np.concatenate(scores) - fold_preds["prob_raw"].to_numpy()).max())
    fold_preds["prob_raw"] = np.concatenate(scores)
    fold_preds.to_parquet(fold_dir / "predictions.parquet", index=False)
    log.info(
        "[%s] rescored from checkpoint (max drift from old prob_raw: %.4f)", fold_dir.name, drift
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="models/cv/<run_id> directory to recalibrate")
    parser.add_argument(
        "--rescore", action="store_true", help="Regenerate prob_raw from checkpoints"
    )
    parser.add_argument("--shards", type=Path, default=Path("data/processed/tfrecords"))
    args = parser.parse_args()

    fold_dirs = sorted(args.run_dir.glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"no fold_* directories under {args.run_dir}")

    if args.rescore:
        for d in fold_dirs:
            rescore_fold(d, args.shards)

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
