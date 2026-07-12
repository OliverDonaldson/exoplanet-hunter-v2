"""Render the promoted run's performance figures for the docs.

Four figures, written to docs/figures/ (committed — they are the README's
evidence that the system does what it claims):

  * training_curves.png — per-fold loss + ROC-AUC by epoch (train faint,
    validation solid), pulled from the MLflow sqlite metric store.
  * roc_pr.png          — per-fold ROC and precision-recall curves with the
    pooled out-of-fold curve on top.
  * calibration.png     — reliability before (raw sigmoid outputs) and after
    (served Platt calibration), plus the calibrated score distributions by
    true class. This is the shift the 2026-07-13 recalibration corrected.
  * embedding_3d.png    — the fold-0 network's penultimate-layer activations
    for its out-of-fold test targets, PCA-projected to 3D: what the CNN's
    learned representation looks like, planets vs false positives.

Usage (from the repository root):

    python pipeline/scripts/make_performance_figures.py               # promoted run
    python pipeline/scripts/make_performance_figures.py --run <run_id>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from exoplanet_hunter.training.calibration import expected_calibration_error
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

FOLD_COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, 5))


def _epoch_series(con: sqlite3.Connection, run_uuid: str, key: str) -> np.ndarray:
    rows = con.execute(
        "select value from metrics where run_uuid=? and key=? order by step", (run_uuid, key)
    ).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def fig_training_curves(db_path: Path, run_id: str, out: Path) -> None:
    con = sqlite3.connect(db_path)
    folds = dict(
        con.execute(
            "select name, run_uuid from runs where run_uuid in "
            "(select run_uuid from tags where key='mlflow.parentRunId' and value=?)",
            (run_id,),
        ).fetchall()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, ax, ylabel in (
        (("loss", "val_loss"), axes[0], "loss"),
        (("auc", "val_auc"), axes[1], "ROC-AUC"),
    ):
        for i in range(5):
            uuid = folds.get(f"fold-{i}")
            if uuid is None:
                continue
            train, val = _epoch_series(con, uuid, name[0]), _epoch_series(con, uuid, name[1])
            ax.plot(train, color=FOLD_COLORS[i], alpha=0.25)
            ax.plot(val, color=FOLD_COLORS[i], label=f"fold {i}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Loss by epoch (faint = train, solid = validation)")
    axes[1].set_title("ROC-AUC by epoch")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote %s", out)


def fig_roc_pr(preds: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fold, g in preds.groupby("fold"):
        fpr, tpr, _ = roc_curve(g.y_true, g.prob_calibrated)
        axes[0].plot(fpr, tpr, color=FOLD_COLORS[int(fold)], alpha=0.55, lw=1)
        prec, rec, _ = precision_recall_curve(g.y_true, g.prob_calibrated)
        axes[1].plot(rec, prec, color=FOLD_COLORS[int(fold)], alpha=0.55, lw=1)
    fpr, tpr, _ = roc_curve(preds.y_true, preds.prob_calibrated)
    axes[0].plot(fpr, tpr, "k", lw=2, label=f"pooled OOF (AUC {auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="grey")
    prec, rec, _ = precision_recall_curve(preds.y_true, preds.prob_calibrated)
    axes[1].plot(rec, prec, "k", lw=2, label=f"pooled OOF (AP {auc(rec[::-1], prec[::-1]):.3f})")
    axes[0].set(
        xlabel="false positive rate",
        ylabel="true positive rate",
        title="ROC — thin lines are folds",
    )
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision–recall")
    for ax in axes:
        ax.legend(fontsize=9, loc="lower right" if ax is axes[0] else "lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote %s", out)


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        sel = which == b
        if sel.any():
            conf.append(p[sel].mean())
            acc.append(y[sel].mean())
    return np.array(conf), np.array(acc)


def fig_calibration(preds: pd.DataFrame, out: Path) -> None:
    y = preds.y_true.to_numpy()
    raw, cal = preds.prob_raw.to_numpy(), preds.prob_calibrated.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for p, label, color in (
        (raw, "raw sigmoid output", "tab:red"),
        (cal, "Platt-calibrated (served)", "tab:blue"),
    ):
        conf, acc = _reliability(y, p)
        axes[0].plot(
            conf,
            acc,
            "o-",
            color=color,
            label=f"{label} — ECE {expected_calibration_error(y, p):.3f}",
        )
    axes[0].plot([0, 1], [0, 1], ":", color="grey")
    axes[0].set(
        xlabel="predicted probability",
        ylabel="observed planet fraction",
        title="Reliability — the shift Platt corrected",
    )
    axes[0].legend(fontsize=9)

    bins = np.linspace(0, 1, 41)
    axes[1].hist(cal[y == 1], bins=bins, alpha=0.6, color="tab:blue", label="true planets")
    axes[1].hist(cal[y == 0], bins=bins, alpha=0.6, color="tab:orange", label="false positives")
    axes[1].set(
        xlabel="calibrated probability",
        ylabel="targets",
        title="Served score distribution by true class",
    )
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote %s", out)


def fig_embedding_3d(run_dir: Path, shard_dir: Path, preds: pd.DataFrame, out: Path) -> None:
    """Penultimate activations of the fold-0 CNN on its OOF test targets."""
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

    fold = preds[preds.fold == 0].sort_values("row")
    metadata = ShardMetadata.load(shard_dir)
    index = load_index(shard_dir)
    groups = index["tic_id"].to_numpy()

    split_codes = np.full(len(index), int(Split.TRAIN), dtype=np.int64)
    split_codes[fold["row"].to_numpy()] = int(Split.TEST)

    bundle = joblib.load(run_dir / "fold_0" / "cnn_calibrator.joblib")
    aux_pipeline = bundle.get("aux_pipeline")
    ds = make_dataset(
        list_shards(shard_dir),
        split=Split.TEST,
        metadata=metadata,
        split_table=make_split_table(groups, split_codes),
        aux_constants=aux_constants_from_pipeline(aux_pipeline) if aux_pipeline else None,
        use_aux=aux_pipeline is not None,
        batch_size=256,
        seed=0,
    )

    model = tf.keras.models.load_model(
        str(run_dir / "fold_0" / "cnn_dualview.keras"), compile=False
    )
    penultimate = next(
        layer
        for layer in reversed(model.layers)
        if len(layer.output.shape) == 2 and layer.output.shape[-1] > 1
    )
    extractor = tf.keras.Model(model.inputs, penultimate.output)
    emb = extractor.predict(ds, verbose=0)

    # Standardise units before PCA so a few high-variance activations don't
    # own the projection, and clip the view to the central 99% so outliers
    # don't crush the structure into a corner.
    emb_std = (emb - emb.mean(axis=0)) / (emb.std(axis=0) + 1e-9)
    xyz = PCA(n_components=3, random_state=0).fit_transform(emb_std)
    y = fold.y_true.to_numpy()
    wrong = (fold.prob_calibrated.to_numpy() >= float(bundle["threshold"])).astype(int) != y

    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(projection="3d")
    for label, color, name in (
        (1, "tab:blue", "true planets"),
        (0, "tab:orange", "false positives"),
    ):
        sel = y == label
        ax.scatter(*xyz[sel].T, s=10, alpha=0.55, color=color, label=name)
    ax.scatter(
        *xyz[wrong].T,
        s=42,
        facecolors="none",
        edgecolors="red",
        lw=0.8,
        label=f"misclassified ({int(wrong.sum())})",
    )
    for axis, setter in zip(range(3), (ax.set_xlim, ax.set_ylim, ax.set_zlim), strict=True):
        lo, hi = np.percentile(xyz[:, axis], [0.5, 99.5])
        setter(lo, hi)
    ax.view_init(elev=18, azim=35)
    ax.set(
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3",
        title=f"Learned representation — fold-0 penultimate layer ({emb.shape[1]}-dim), OOF targets, PCA→3D",
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote %s (embeddings %s)", out, emb.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run id (default: the promoted run)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--shards", type=Path, default=Path("data/processed/tfrecords"))
    parser.add_argument("--mlflow-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    run_id = args.run or json.loads((args.models_dir / "registry.json").read_text())["run_id"]
    run_dir = args.models_dir / "cv" / run_id
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("figures for run %s (%d OOF predictions)", run_id[:8], len(preds))

    fig_training_curves(args.mlflow_db, run_id, args.out / "training_curves.png")
    fig_roc_pr(preds, args.out / "roc_pr.png")
    fig_calibration(preds, args.out / "calibration.png")
    fig_embedding_3d(run_dir, args.shards, preds, args.out / "embedding_3d.png")


if __name__ == "__main__":
    main()
