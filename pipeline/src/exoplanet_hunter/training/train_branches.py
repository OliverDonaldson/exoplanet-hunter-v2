"""Cross-validated training for the per-diagnostic branch model.

Same evaluation contract as `train.py`: StratifiedGroupKFold with group =
tic_id, an inner GroupShuffleSplit for early stopping and the Platt fit, and a
`cv_summary.json` in the schema the promotion gate reads. Reusing that schema
matters — two implementations of it would drift, and the gate is what decides
whether a run is better than the incumbent.

Nothing here promotes anything. It writes a summary; comparing it to the
incumbent is `promotion_gate.py`'s job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from exoplanet_hunter.datasets.viewset_pipeline import (
    Split,
    fit_scalar_constants,
    make_split_table,
    make_viewset_dataset,
)
from exoplanet_hunter.datasets.viewset_tfrecords import list_shards, load_index, load_metadata
from exoplanet_hunter.eval.metrics import classification_metrics
from exoplanet_hunter.eval.observation_bias import measure_observation_bias
from exoplanet_hunter.models.cnn_branches import build_cnn_branches
from exoplanet_hunter.training.calibration import PlattScaler, expected_calibration_error
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

SUMMARY_KEYS = ("test_roc_auc", "test_pr_auc", "test_f1", "test_brier", "test_ece")


@dataclass
class CVConfig:
    n_splits: int = 5
    val_frac: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    patience: int = 8
    learning_rate: float = 1e-3
    seed: int = 42


def _split_codes(n: int, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> np.ndarray:
    codes = np.full(n, -1, dtype=np.int64)
    codes[train] = int(Split.TRAIN)
    codes[val] = int(Split.VAL)
    codes[test] = int(Split.TEST)
    return codes


def _predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels in stream order — unshuffled, so they stay aligned."""
    scores, labels = [], []
    for inputs, y in dataset:
        scores.append(model(inputs, training=False).numpy().ravel())
        labels.append(y.numpy().ravel())
    return np.concatenate(scores), np.concatenate(labels)


def run_fold(
    shard_dir: Path,
    index: pd.DataFrame,
    metadata: dict,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: CVConfig,
    model_cfg: object,
) -> tuple[dict, pd.DataFrame]:
    """Train one fold; return its metrics and its test-row predictions."""
    shards = list_shards(shard_dir)
    tic_ids = index["tic_id"].to_numpy()
    table = make_split_table(tic_ids, _split_codes(len(index), train_idx, val_idx, test_idx))
    # Fitted on the training rows only: a validation row must never influence
    # the scale a training row is measured against.
    constants = fit_scalar_constants(index.iloc[train_idx], list(metadata["scalar_columns"]))

    def stream(split: Split, *, shuffle: bool) -> tf.data.Dataset:
        return make_viewset_dataset(
            shards,
            metadata,
            split_table=table,
            split=split,
            scalar_constants=constants,
            batch_size=config.batch_size,
            shuffle=shuffle,
            seed=config.seed,
        )

    model = build_cnn_branches(
        model_cfg,
        scalar_columns=list(metadata["scalar_columns"]),
        mask_columns=list(metadata["mask_columns"]),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    model.fit(
        stream(Split.TRAIN, shuffle=True),
        validation_data=stream(Split.VAL, shuffle=False),
        epochs=config.epochs,
        callbacks=[
            # AUC-PR rather than loss: the ExoMiner papers stop on it, and it
            # is the metric that tracks the minority class we care about.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=config.patience,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    val_scores, val_labels = _predict(model, stream(Split.VAL, shuffle=False))
    test_scores, test_labels = _predict(model, stream(Split.TEST, shuffle=False))
    calibrator = PlattScaler.from_validation(val_scores, val_labels)
    calibrated = calibrator.predict(test_scores)

    metrics = classification_metrics(test_labels, calibrated)
    # Test rows in stream order, which is index order — the observation-bias
    # measurement needs a score per row, and without it stage 2(b)'s success
    # criterion cannot be evaluated at all.
    predictions = index.iloc[np.sort(test_idx)].copy()
    predictions["score"] = calibrated
    # The stream yields test rows in ascending index position, so they line up
    # with sorted(test_idx). Asserted rather than assumed: a silent
    # misalignment would attach every score to the wrong target and still
    # produce a plausible AUC.
    if not np.array_equal(predictions["label"].to_numpy(), test_labels.astype(int)):
        raise RuntimeError("test predictions are not aligned with the index rows")
    return {
        "test_roc_auc": metrics.roc_auc,
        "test_pr_auc": metrics.pr_auc,
        "test_f1": metrics.f1,
        "test_brier": metrics.brier,
        "test_ece": float(expected_calibration_error(test_labels, calibrated)),
        "n_test": len(test_labels),
    }, predictions


def run_cv(
    shard_dir: Path,
    out_dir: Path,
    *,
    config: CVConfig | None = None,
    model_cfg: object | None = None,
) -> dict:
    """Full CV over a view-set shard set; writes `cv_summary.json`."""
    config = config or CVConfig()
    model_cfg = model_cfg if model_cfg is not None else object()
    metadata = load_metadata(shard_dir)
    index = load_index(shard_dir)
    y = index["label"].to_numpy().astype(int)
    groups = index["tic_id"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    positions = np.arange(len(y))
    for fold, (trainval, test_idx) in enumerate(splitter.split(positions, y, groups)):
        inner = GroupShuffleSplit(
            n_splits=1, test_size=config.val_frac, random_state=config.seed * 1000 + fold
        )
        tr_rel, va_rel = next(inner.split(trainval, y[trainval], groups[trainval]))
        metrics, fold_predictions = run_fold(
            shard_dir,
            index,
            metadata,
            train_idx=trainval[tr_rel],
            val_idx=trainval[va_rel],
            test_idx=test_idx,
            config=config,
            model_cfg=model_cfg,
        )
        metrics["fold"] = fold
        rows.append(metrics)
        fold_predictions["fold"] = fold
        predictions.append(fold_predictions)
        log.info(
            "[train-branches] fold %d  AUC %.4f  Brier %.4f  ECE %.4f  (n=%d)",
            fold,
            metrics["test_roc_auc"],
            metrics["test_brier"],
            metrics["test_ece"],
            metrics["n_test"],
        )

    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "std": float(np.std([r[key] for r in rows])),
        }
        for key in SUMMARY_KEYS
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"folds": rows, "summary": summary}
    (out_dir / "cv_summary.json").write_text(json.dumps(payload, indent=2))

    # Every row is tested exactly once across folds, so this is a full
    # out-of-fold prediction set — what the observation-bias metric reads.
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    bias = measure_observation_bias(all_predictions["score"].to_numpy(), all_predictions)
    (out_dir / "observation_bias.json").write_text(json.dumps(asdict(bias), indent=2))
    log.info(
        "[train-branches] observation bias: transit %+.3f  baseline %+.3f  "
        "completeness %+.3f  (labelled set: incumbent -0.087 / +0.238, "
        "label itself -0.073 / +0.278)",
        bias.transit_sensitivity,
        bias.baseline_sensitivity,
        bias.completeness_sensitivity,
    )
    log.info(
        "[train-branches] ROC-AUC %.4f ± %.4f  Brier %.4f ± %.4f  ECE %.4f ± %.4f -> %s",
        summary["test_roc_auc"]["mean"],
        summary["test_roc_auc"]["std"],
        summary["test_brier"]["mean"],
        summary["test_brier"]["std"],
        summary["test_ece"]["mean"],
        summary["test_ece"]["std"],
        out_dir / "cv_summary.json",
    )
    return payload
