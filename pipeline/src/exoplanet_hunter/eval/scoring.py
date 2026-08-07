"""Scoring a promoted run against a shard set, under a stated protocol.

The incumbent's own training inputs no longer exist. The legacy shards were
rebuilt on 2026-07-25, six days after run `ca906040` trained on 2026-07-19, and
its 9-dim aux is not a slice of the 13-dim shard aux: index 7 is `pink_snr`
there and the catalogue transit SNR in the model. Taking 13 -> 9 by slicing
would feed one into the lane the model learned as the other and return a
confident wrong number, so index 7 is rebuilt from the catalogue and the rest
taken directly.

**The two protocols are not interchangeable, and the difference is the whole
reason this module states it in the return value.**

- `OUT_OF_FOLD` scores each row with the single fold that held it out. It is the
  only honest protocol for rows the run trained on, and it reproduces the run's
  own `predictions.parquet`.
- `ZERO_SHOT` has every fold score every row and takes the mean. Valid only
  where no fold trained on the row. Ranking stays comparable; calibration does
  not, because the Platt scalers were fitted on other missions' validation rows.

Silently mixing them across one population is itself a comparability defect, so
`score_run` refuses to guess and `Scored.protocol` travels with the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.tfrecords import (
    ShardMetadata,
    list_shards,
    load_index,
    make_parse_fn,
)
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: The one aux lane whose meaning differs between the 13-dim shards and the
#: 9-dim layout the incumbent learned.
CATALOGUE_SNR_COLUMN = 7


class Protocol(StrEnum):
    OUT_OF_FOLD = "oof"
    ZERO_SHOT = "zeroshot"


@dataclass(frozen=True)
class Scored:
    """Predictions and the protocol that produced them, kept together."""

    predictions: pd.DataFrame
    protocol: Protocol

    def __post_init__(self) -> None:
        missing = {"tic_id", "label", "score"} - set(self.predictions.columns)
        if missing:
            raise ValueError(f"scored frame is missing {sorted(missing)}")


def legacy_aux(index: pd.DataFrame, labels: pd.DataFrame, aux_dim: int) -> np.ndarray:
    """The incumbent's 9-dim aux, rebuilt from the wider shard index.

    Indices 0-6 and 8 mean the same thing in both layouts and are taken as they
    are. Index 7 comes from the catalogue; where the catalogue has no SNR it
    stays NaN and the fold's own aux pipeline imputes it, which is the path a
    non-TOI already takes at serve time.
    """
    wide = index[[f"aux_{k}" for k in range(aux_dim)]].to_numpy(dtype=np.float64)
    snr = index[["tic_id"]].merge(
        labels[["tic_id", "snr"]].drop_duplicates("tic_id"), on="tic_id", how="left"
    )["snr"]
    return np.column_stack(
        [wide[:, :CATALOGUE_SNR_COLUMN], snr.to_numpy(dtype=np.float64), wide[:, 8]]
    )


def read_views(
    shard_dir: Path, metadata: ShardMetadata, index: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Both views in shard order, asserted to line up with the index."""
    import tensorflow as tf

    dataset = (
        tf.data.TFRecordDataset(list_shards(shard_dir)).map(make_parse_fn(metadata)).batch(256)
    )
    batches: dict[str, list[np.ndarray]] = {"global_view": [], "local_view": [], "tic_id": []}
    for features, _ in dataset:
        for key in batches:
            batches[key].append(features[key].numpy())
    views = {key: np.concatenate(value) for key, value in batches.items()}
    if not np.array_equal(views["tic_id"], index["tic_id"].to_numpy()):
        raise RuntimeError("shard stream order does not match the index — scores would misalign")
    return views


def score_run(
    run_dir: Path,
    shard_dir: Path,
    *,
    labels: pd.DataFrame,
    protocol: Protocol,
    fold_of: pd.Series | None = None,
    subset: np.ndarray | None = None,
    allow_untracked_rows: bool = False,
) -> Scored:
    """Score `run_dir`'s folds over `shard_dir`, restricted to `subset`.

    `fold_of` maps tic_id to the fold that held that row out and is required for
    `OUT_OF_FOLD`; supplying it under `ZERO_SHOT` is refused rather than ignored,
    since the caller clearly meant one thing and asked for the other.
    """
    import tensorflow as tf

    if (protocol is Protocol.OUT_OF_FOLD) != (fold_of is not None):
        raise ValueError(f"{protocol} needs fold_of={protocol is Protocol.OUT_OF_FOLD}")

    metadata = ShardMetadata.load(shard_dir)
    index = load_index(shard_dir)
    views = read_views(shard_dir, metadata, index)
    aux = legacy_aux(index, labels, metadata.aux_dim)

    keep = np.ones(len(index), dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    folds = sorted(run_dir.glob("fold_*"))
    if not folds:
        raise ValueError(f"no fold_* directories under {run_dir}")

    def predict(fold_dir: Path, rows: np.ndarray) -> np.ndarray:
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        model = tf.keras.models.load_model(fold_dir / "cnn_dualview.keras", compile=False)
        raw = model.predict(
            {
                "global_view": views["global_view"][rows],
                "local_view": views["local_view"][rows],
                "aux_features": bundle["aux_pipeline"].transform(aux[rows]).astype(np.float32),
            },
            verbose=0,
        ).squeeze()
        return np.asarray(bundle["calibrator"].predict(raw), dtype=float)

    tic_ids = index["tic_id"].to_numpy()
    if protocol is Protocol.ZERO_SHOT:
        # Averaging all five folds over a row one of them trained on is not a
        # held-out measurement, and nothing downstream could tell. Rows the run
        # already scored out-of-fold are dropped here rather than trusted to the
        # caller's mission filter.
        trained = run_dir / "predictions.parquet"
        if not trained.exists() and not allow_untracked_rows:
            raise ValueError(
                f"{run_dir} has no predictions.parquet, so the rows it trained on are unknown "
                "and the contamination filter cannot run. Zero-shot over rows a fold trained on "
                "returns an optimistic number nothing downstream can detect — pass "
                "allow_untracked_rows=True only if you know every row here is held out."
            )
        if trained.exists():
            seen = pd.Series(tic_ids).isin(set(pd.read_parquet(trained)["tic_id"])).to_numpy()
            if (keep & seen).any():
                log.info(
                    "[score] zero-shot: dropped %d rows this run trained on", (keep & seen).sum()
                )
            keep = keep & ~seen
        rows = np.flatnonzero(keep)
        scores = np.mean([predict(fold, rows) for fold in folds], axis=0)
        assigned = np.full(len(rows), -1)
    else:
        assert fold_of is not None
        wanted = keep & pd.Series(tic_ids).isin(fold_of.index).to_numpy()
        rows = np.flatnonzero(wanted)
        assigned = fold_of.loc[tic_ids[rows]].to_numpy()
        # NaN, not np.empty: `fold_of` comes from the run's predictions.parquet
        # while `folds` comes from globbing the directory, and nothing ties the
        # two together. A row whose fold has no checkpoint would otherwise keep
        # uninitialised memory and be returned as a calibrated probability.
        scores = np.full(len(rows), np.nan)
        for fold_dir in folds:
            fold = int(fold_dir.name.removeprefix("fold_"))
            within = np.flatnonzero(assigned == fold)
            if len(within):
                scores[within] = predict(fold_dir, rows[within])
            log.info("[score] %s: %d rows", fold_dir.name, len(within))
        if not np.isfinite(scores).all():
            unscored = sorted(set(assigned[~np.isfinite(scores)]))
            raise RuntimeError(
                f"{np.isnan(scores).sum()} rows reference fold(s) {unscored} with no checkpoint "
                f"under {run_dir} — scores would be uninitialised memory"
            )

    return Scored(
        pd.DataFrame(
            {
                "tic_id": tic_ids[rows],
                "label": index["label"].to_numpy()[rows].astype(int),
                "score": scores,
                "fold": assigned,
            }
        ),
        protocol,
    )
