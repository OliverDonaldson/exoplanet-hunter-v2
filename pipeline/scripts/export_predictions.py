"""Backfill per-example CV test predictions for an already-trained run.

Newer trainer runs write <cv_dir>/predictions.parquet themselves; this
script regenerates it for runs trained before that existed (e.g. the first
registered incumbent). Fold membership is reconstructed exactly as the
trainer computed it — same StratifiedGroupKFold parameters over the same
shard index — and each fold's saved model + calibrator re-scores its own
held-out test split.

Usage (from the repository root):

    python pipeline/scripts/export_predictions.py                  # registered run
    python pipeline/scripts/export_predictions.py --cv-dir models/cv/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet_hunter.datasets import (
    ShardMetadata,
    Split,
    aux_constants_from_pipeline,
    list_shards,
    load_index,
    make_dataset,
    make_split_table,
)
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-dir", type=Path, default=None, help="CV run directory; defaults to the registered run"
    )
    parser.add_argument("--shard-dir", type=Path, default=Path("data/processed/tfrecords"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cv_dir = args.cv_dir
    if cv_dir is None:
        registry = json.loads(Path("models/registry.json").read_text())
        cv_dir = Path(registry["cv_dir"])

    import tensorflow as tf

    metadata = ShardMetadata.load(args.shard_dir)
    shards = list_shards(args.shard_dir)
    index = load_index(args.shard_dir)
    y = index["label"].to_numpy().astype(int)
    groups = index["tic_id"].to_numpy()

    # Must match conf/model/cnn_dualview.yaml cross_validation exactly.
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    frames: list[pd.DataFrame] = []
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(np.arange(len(y)), y, groups)):
        fold_dir = cv_dir / f"fold_{fold_idx}"
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        model = tf.keras.models.load_model(str(fold_dir / "cnn_dualview.keras"), compile=False)

        split_codes = np.full(len(y), -1, dtype=np.int64)
        split_codes[test_idx] = int(Split.TEST)
        aux_constants = (
            aux_constants_from_pipeline(bundle["aux_pipeline"])
            if bundle.get("aux_pipeline") is not None
            else None
        )
        test_ds = make_dataset(
            shards,
            metadata,
            split_table=make_split_table(groups, split_codes),
            split=Split.TEST,
            aux_constants=aux_constants,
            use_aux=aux_constants is not None,
            batch_size=64,
        )
        prob_raw = model.predict(test_ds, verbose=0).squeeze()
        prob_cal = bundle["calibrator"].predict(prob_raw)

        rows = np.sort(test_idx)
        frames.append(
            pd.DataFrame(
                {
                    "row": rows,
                    "tic_id": groups[rows],
                    "fold": fold_idx,
                    "y_true": y[rows],
                    "prob_raw": np.atleast_1d(prob_raw),
                    "prob_calibrated": np.atleast_1d(prob_cal),
                }
            )
        )
        log.info("[export] fold %d: %d test predictions", fold_idx, len(rows))

    out = cv_dir / "predictions.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("[export] wrote %s (%d rows)", out, sum(len(f) for f in frames))


if __name__ == "__main__":
    main()
