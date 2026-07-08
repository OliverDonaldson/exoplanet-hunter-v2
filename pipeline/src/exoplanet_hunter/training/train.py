"""Hydra-driven training entry point — V2, streaming from TFRecord shards.

Usage (from the repository root):

    python -m exoplanet_hunter.training.train                      # 5-fold CV CNN
    python -m exoplanet_hunter.training.train model=random_forest  # RF baseline
    python -m exoplanet_hunter.training.train train.mixed_precision=true  # GPU burst

Differences from the V1 trainer this replaces:

  * Input is a TFRecord shard set (`scripts/shard_views.py`) streamed via
    `datasets.make_dataset` — parse/filter/normalise cached, augmentation
    fresh each epoch — instead of NumPy arrays held in RAM.
  * Fold membership is routed by TIC ID through a StaticHashTable filter,
    so the StratifiedGroupKFold leakage guarantee survives streaming.
  * Aux normalisation is still a fitted sklearn pipeline persisted in the
    calibration bundle (the serving contract); training replays its fitted
    constants as tensor ops (`datasets.aux_transform`, parity-tested).
  * Optional mixed_float16 policy for the GPU burst.

Split semantics, callbacks, threshold sweep, temperature scaling, and the
bundle format are unchanged from V1 — same keys, same behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from exoplanet_hunter.datasets import (
    AugmentConfig,
    ShardMetadata,
    Split,
    aux_constants_from_pipeline,
    fit_aux_pipeline,
    list_shards,
    load_index,
    load_views,
    make_dataset,
    make_split_table,
)
from exoplanet_hunter.models import (
    binary_focal_loss,
    build_cnn_dualview,
    build_random_forest,
)
from exoplanet_hunter.training.calibration import TemperatureScaler
from exoplanet_hunter.training.mlflow_utils import (
    keras_callbacks,
    log_classification_artifacts,
    log_config,
    log_history,
    setup_mlflow,
)
from exoplanet_hunter.training.precision import apply_precision_policy
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


def run(cfg: DictConfig) -> float:
    """Train one model from a fully-resolved Hydra config."""
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)
    setup_mlflow(cfg)

    if cfg.model.type == "sklearn":
        return _train_rf(cfg, paths)
    if cfg.model.type == "keras":
        apply_precision_policy(bool(OmegaConf.select(cfg.train, "mixed_precision") or False))
        return _train_cnn_cv(cfg, paths)
    raise ValueError(f"unknown model type: {cfg.model.type}")


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    return run(cfg)


# ------------------------------------------------------------- shard access --


def _shard_dir(paths: ProjectPaths) -> Path:
    d = paths.data_processed / "tfrecords"
    if not (d / "metadata.json").exists():
        raise FileNotFoundError(
            f"No TFRecord shard set at {d}. Run `python scripts/shard_views.py` "
            "after building views."
        )
    return d


# -------------------------------------------------------------- RF baseline --


def _train_rf(cfg: DictConfig, paths: ProjectPaths) -> float:
    """Random-forest baseline on handcrafted features (the promotion bar).

    Small data by construction (a feature matrix), so it reads the views
    .npz directly rather than streaming shards.
    """
    from sklearn.metrics import classification_report

    from exoplanet_hunter.features import extract_features

    views = load_views(paths.data_processed / "views.npz")
    groups = views.tic_ids
    y = views.labels.astype(int)

    # Group-aware holdout: same leakage rule as the CNN path.
    gss = GroupShuffleSplit(
        n_splits=1, test_size=float(cfg.data.split.test), random_state=int(cfg.seed)
    )
    trainval_idx, test_idx = next(gss.split(np.arange(len(y)), y, groups))

    log.info("[train-rf] extracting handcrafted features ...")
    X_all = np.array([extract_features(v) for v in views.global_views])

    pipeline = build_random_forest(cfg.model)

    with mlflow.start_run(run_name=f"rf-{cfg.data.name}"):
        log_config(cfg)
        skf = StratifiedKFold(
            n_splits=int(cfg.model.cross_validation.n_splits),
            shuffle=bool(cfg.model.cross_validation.shuffle),
            random_state=int(cfg.model.cross_validation.random_state),
        )
        cv_aucs: list[float] = []
        for fold, (tr, va) in enumerate(skf.split(X_all[trainval_idx], y[trainval_idx])):
            pipeline.fit(X_all[trainval_idx][tr], y[trainval_idx][tr])
            score = pipeline.predict_proba(X_all[trainval_idx][va])[:, 1]
            auc = roc_auc_score(y[trainval_idx][va], score)
            cv_aucs.append(auc)
            mlflow.log_metric(f"cv_auc_fold_{fold}", float(auc))
        mlflow.log_metric("cv_auc_mean", float(np.mean(cv_aucs)))
        mlflow.log_metric("cv_auc_std", float(np.std(cv_aucs)))
        log.info("[train-rf] CV AUC %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

        pipeline.fit(X_all[trainval_idx], y[trainval_idx])
        test_score = pipeline.predict_proba(X_all[test_idx])[:, 1]
        report = classification_report(
            y[test_idx], (test_score >= 0.5).astype(int), zero_division=0
        )
        log.info("[train-rf] test classification report:\n%s", report)
        log_classification_artifacts(
            y[test_idx], test_score, threshold=0.5, out_dir=paths.results / "rf"
        )

        artifact = paths.models / "random_forest.joblib"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact)
        mlflow.log_artifact(str(artifact))
        return float(np.mean(cv_aucs))


# ----------------------------------------------------------------- CNN + CV --


def _train_cnn_cv(cfg: DictConfig, paths: ProjectPaths) -> float:
    """Stratified group k-fold CV over the shard set. Group = tic_id.

    Per fold: outer split -> fold-test; inner GroupShuffleSplit -> train/val
    (val drives EarlyStopping, the threshold sweep, and the calibrator fit).
    Returns mean test ROC-AUC across folds.
    """
    shard_dir = _shard_dir(paths)
    metadata = ShardMetadata.load(shard_dir)
    shards = list_shards(shard_dir)
    index = load_index(shard_dir)
    y = index["label"].to_numpy().astype(int)
    groups = index["tic_id"].to_numpy()
    aux_cols = [c for c in index.columns if c.startswith("aux_")]
    aux_all = index[aux_cols].to_numpy(dtype=np.float32) if aux_cols else None

    cv_cfg = cfg.model.cross_validation
    n_splits = int(cv_cfg.n_splits)
    val_frac = float(cv_cfg.val_frac_within_fold)
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=bool(cv_cfg.shuffle),
        random_state=int(cv_cfg.random_state),
    )

    with mlflow.start_run(run_name=f"cnn-cv-{cfg.data.name}") as parent:
        log_config(cfg)
        run_id = parent.info.run_id
        cv_root = paths.models / "cv" / run_id
        results_root = paths.results / "cnn" / "cv" / run_id
        log.info(
            "[train-cnn-cv] %d folds over %d examples (%d shards), run_id=%s",
            n_splits,
            metadata.n_examples,
            metadata.n_shards,
            run_id,
        )

        fold_rows: list[dict] = []
        idx = np.arange(len(y))
        for fold_idx, (trainval_idx, test_idx) in enumerate(sgkf.split(idx, y, groups)):
            inner_seed = int(cfg.seed) * 1000 + fold_idx
            inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=inner_seed)
            tr_rel, va_rel = next(inner.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
            train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]

            fold_dir = cv_root / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            with mlflow.start_run(run_name=f"fold-{fold_idx}", nested=True):
                metrics = _run_cnn_fold(
                    cfg=cfg,
                    shards=shards,
                    metadata=metadata,
                    y=y,
                    groups=groups,
                    aux_all=aux_all,
                    split_indices=(train_idx, val_idx, test_idx),
                    ckpt_path=fold_dir / "cnn_dualview.keras",
                    cal_path=fold_dir / "cnn_calibrator.joblib",
                    results_dir=results_root / f"fold_{fold_idx}",
                    fold_idx=fold_idx,
                )
            metrics["fold"] = fold_idx
            fold_rows.append(metrics)

        _aggregate_cv(fold_rows, cv_root)
        return float(np.mean([m["test_roc_auc"] for m in fold_rows]))


def _run_cnn_fold(
    *,
    cfg: DictConfig,
    shards: list[str],
    metadata: ShardMetadata,
    y: np.ndarray,
    groups: np.ndarray,
    aux_all: np.ndarray | None,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ckpt_path: Path,
    cal_path: Path,
    results_dir: Path,
    fold_idx: int,
) -> dict[str, float]:
    """Train one CNN on one fold, streaming all three splits from the shards."""
    import tensorflow as tf

    train_idx, val_idx, test_idx = split_indices
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, split_idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        pos = int((y[split_idx] == 1).sum())
        n = len(split_idx)
        log.info("[fold %d] %s: n=%d pos=%d (%.3f)", fold_idx, name, n, pos, pos / n if n else 0)
        mlflow.log_metrics(
            {
                f"fold_{fold_idx}_{name}_n": float(n),
                f"fold_{fold_idx}_{name}_pos": float(pos),
                f"fold_{fold_idx}_{name}_pos_frac": float(pos / n) if n else 0.0,
            }
        )

    # tic_id -> split routing table for the stream filter.
    split_codes = np.full(len(y), -1, dtype=np.int64)
    split_codes[train_idx] = int(Split.TRAIN)
    split_codes[val_idx] = int(Split.VAL)
    split_codes[test_idx] = int(Split.TEST)
    # Every row of a TIC carries the same code by construction (group splits).
    table = make_split_table(groups, split_codes)

    # Aux pipeline: fit on training rows only (from the index sidecar); the
    # fitted sklearn object ships in the bundle, its constants run in-stream.
    use_aux = bool(getattr(cfg.model, "use_aux_features", False)) and aux_all is not None
    aux_pipeline = None
    aux_constants = None
    aux_dim = None
    if use_aux:
        assert aux_all is not None
        aux_pipeline = fit_aux_pipeline(aux_all[train_idx])
        aux_constants = aux_constants_from_pipeline(aux_pipeline)
        aux_dim = aux_constants.aux_dim
        log.info("[fold %d] aux pipeline fitted (dim=%d)", fold_idx, aux_dim)

    aug_cfg = cfg.preprocess.augmentation
    augment = (
        AugmentConfig(
            time_shift_frac=float(aug_cfg.time_shift_frac),
            noise_std=float(aug_cfg.noise_std),
            scale_range=float(aug_cfg.get("scale_range", 0.0)),
            mask_prob=float(aug_cfg.get("mask_prob", 0.0)),
        )
        if bool(aug_cfg.enabled)
        else None
    )

    common: dict[str, Any] = {
        "metadata": metadata,
        "split_table": table,
        "aux_constants": aux_constants,
        "use_aux": use_aux,
        "batch_size": int(cfg.train.batch_size),
        "seed": int(cfg.seed),
    }
    train_ds = make_dataset(
        shards,
        split=Split.TRAIN,
        shuffle=True,
        shuffle_buffer=int(cfg.train.shuffle_buffer),
        augment=augment,
        **common,
    )
    val_ds = make_dataset(shards, split=Split.VAL, **common)
    test_ds = make_dataset(shards, split=Split.TEST, **common)

    model = build_cnn_dualview(
        cfg.model,
        global_input_length=metadata.global_bins,
        local_input_length=metadata.local_bins,
        aux_input_dim=aux_dim,
    )
    optimizer = instantiate(cfg.train.optimizer)

    if cfg.train.loss.type == "binary_crossentropy":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif cfg.train.loss.type == "focal":
        loss = binary_focal_loss(
            gamma=float(cfg.train.loss.focal_gamma),
            alpha=float(cfg.train.loss.focal_alpha),
        )
    else:
        raise ValueError(f"unknown loss: {cfg.train.loss.type}")

    # Focal loss already rebalances via alpha; stacking class_weight on top
    # double-counts the minority class (V1 lesson, preserved).
    class_weight = None
    if cfg.train.loss.type != "focal" and str(cfg.train.class_weight) == "auto":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y[train_idx])
        class_weight = dict(zip(classes.tolist(), weights.tolist(), strict=False))
        log.info("[fold %d] class_weight=%s", fold_idx, class_weight)

    metrics_map = {
        "accuracy": "accuracy",
        "auc": tf.keras.metrics.AUC(name="auc"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
    }
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metrics_map[m] for m in cfg.train.metrics if m in metrics_map],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.train.epochs),
        callbacks=keras_callbacks(cfg.train, ckpt_path),
        class_weight=class_weight,
        verbose=2,
    )
    log_history(history.history, results_dir)

    # Unshuffled streams yield examples in ascending index-row order, so
    # sort the fold indices the same way for positional alignment with
    # predict() output.
    val_y = y[np.sort(val_idx)]
    test_y = y[np.sort(test_idx)]
    val_score = model.predict(val_ds, verbose=0).squeeze()
    test_score = model.predict(test_ds, verbose=0).squeeze()

    thresholds = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(val_y, (val_score >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    mlflow.log_metric("best_threshold", best_threshold)

    calibrator = TemperatureScaler.from_validation(val_score, val_y)
    test_score_cal = calibrator.predict(test_score)
    T_star = float(calibrator.T)
    mlflow.log_metric("temperature_T_star", T_star)
    log.info("[fold %d] threshold=%.2f  T*=%.4f", fold_idx, best_threshold, T_star)

    # Same bundle keys as V1 — the scoring path's contract.
    joblib.dump(
        {
            "calibrator": calibrator,
            "temperature": T_star,
            "threshold": best_threshold,
            "aux_pipeline": aux_pipeline,
            "aux_dim": aux_dim,
        },
        cal_path,
    )

    log_classification_artifacts(
        test_y, test_score_cal, threshold=best_threshold, out_dir=results_dir
    )
    mlflow.log_artifact(str(ckpt_path))
    mlflow.log_artifact(str(cal_path))

    return {
        "test_roc_auc": float(roc_auc_score(test_y, test_score_cal)),
        "test_pr_auc": float(average_precision_score(test_y, test_score_cal)),
        "test_f1": float(
            f1_score(test_y, (test_score_cal >= best_threshold).astype(int), zero_division=0)
        ),
        "test_brier": float(brier_score_loss(test_y, test_score_cal)),
        "best_threshold": best_threshold,
        "temperature": T_star,
    }


def _aggregate_cv(fold_rows: list[dict], cv_root: Path) -> None:
    """Log mean/std across folds and write the summary table artifact."""
    keys = ("test_roc_auc", "test_pr_auc", "test_f1", "test_brier", "best_threshold", "temperature")
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_rows], dtype=float)
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        mlflow.log_metric(f"cv_{k}_mean", summary[k]["mean"])
        mlflow.log_metric(f"cv_{k}_std", summary[k]["std"])

    cv_root.mkdir(parents=True, exist_ok=True)
    summary_path = cv_root / "cv_summary.json"
    summary_path.write_text(json.dumps({"folds": fold_rows, "summary": summary}, indent=2))
    mlflow.log_artifact(str(summary_path))
    log.info(
        "[train-cnn-cv] ROC-AUC %.4f ± %.4f  Brier %.4f ± %.4f",
        summary["test_roc_auc"]["mean"],
        summary["test_roc_auc"]["std"],
        summary["test_brier"]["mean"],
        summary["test_brier"]["std"],
    )


if __name__ == "__main__":
    main()
