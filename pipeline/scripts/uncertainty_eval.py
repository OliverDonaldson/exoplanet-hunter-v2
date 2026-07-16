"""Does MC-Dropout uncertainty predict errors? Risk-coverage for prob_std.

Streams each fold's OOF test set, draws T dropout samples per example
(batched), and asks whether high per-example std concentrates the model's
mistakes — the justification (or not) for an "abstain" band in the console.
Writes docs/figures/risk_coverage.png and prints the headline numbers.

Usage (from the repository root):

    python pipeline/scripts/uncertainty_eval.py [--samples 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def fold_mc_std(fold_dir: Path, shard_dir: Path, n_samples: int) -> pd.DataFrame:
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

    preds = pd.read_parquet(fold_dir / "predictions.parquet").sort_values("row")
    index = load_index(shard_dir)
    codes = np.full(len(index), int(Split.TRAIN), dtype=np.int64)
    codes[preds["row"].to_numpy()] = int(Split.TEST)

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

    batches = [feats for feats, _ in ds]
    draws = np.stack(
        [
            np.concatenate([np.asarray(model(f, training=True)).squeeze() for f in batches])
            for _ in range(n_samples)
        ]
    )
    preds["mc_std"] = draws.std(axis=0)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run id (default: promoted)")
    parser.add_argument("--shards", type=Path, default=Path("data/processed/tfrecords"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/risk_coverage.png"))
    args = parser.parse_args()

    run_id = args.run or json.loads(Path("models/registry.json").read_text())["run_id"]
    run_dir = Path("models/cv") / run_id

    frames = [fold_mc_std(d, args.shards, args.samples) for d in sorted(run_dir.glob("fold_*"))]
    df = pd.concat(frames, ignore_index=True)

    thr = float(
        np.mean(
            [
                float(joblib.load(d / "cnn_calibrator.joblib")["threshold"])
                for d in sorted(run_dir.glob("fold_*"))
            ]
        )
    )
    df["error"] = ((df["prob_calibrated"] >= thr) != df["y_true"].astype(bool)).astype(int)
    # Fold std scales differ; rank within fold before pooling.
    df["std_rank"] = df.groupby("fold")["mc_std"].rank(pct=True)
    df["thr_proximity"] = -np.abs(df["prob_calibrated"] - thr)

    from sklearn.metrics import roc_auc_score

    n = len(df)
    base = float(df["error"].mean())

    def risk_curve(uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(uncertainty)  # most-certain first
        errors = df["error"].to_numpy()[order]
        return np.arange(1, n + 1) / n, np.cumsum(errors) / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, signal in (
        ("MC-Dropout std (fold-ranked)", df["std_rank"].to_numpy()),
        ("distance to threshold", df["thr_proximity"].to_numpy()),
    ):
        coverage, risk = risk_curve(signal)
        auroc = float(roc_auc_score(df["error"], signal))
        at90 = float(risk[int(0.9 * n) - 1])
        ax.plot(coverage, risk, lw=2, label=f"{label} — AUROC {auroc:.3f}, err@90% {at90:.3f}")
        log.info(
            "[uncertainty] %s: AUROC(->error)=%.3f  error@90%%=%.4f  (base %.4f, n=%d)",
            label,
            auroc,
            at90,
            base,
            n,
        )

    ax.axhline(base, ls=":", color="grey", label=f"full-coverage error {base:.3f}")
    ax.set(
        xlabel="coverage (fraction answered, most-certain first)",
        ylabel="error rate at serving threshold",
        title=f"Risk-coverage — run {run_id[:8]}",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    log.info("[uncertainty] wrote %s", args.out)


if __name__ == "__main__":
    main()
