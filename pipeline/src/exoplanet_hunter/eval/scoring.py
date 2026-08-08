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

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.tfrecords import (
    ShardMetadata,
    list_shards,
    load_index,
    make_parse_fn,
)
from exoplanet_hunter.eval.comparison import SliceMetrics
from exoplanet_hunter.utils.logging import get_logger
from exoplanet_hunter.validation.promotion import AGGREGATE_SLICE, GATE_MISSION

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


#: The only protocol a gate slice may be measured under. `ZERO_SHOT` rows are a
#: real measurement of cross-mission transfer, but they are not the same
#: quantity, so they are reported apart rather than averaged in.
GATE_PROTOCOL = Protocol.OUT_OF_FOLD

#: `SliceMetrics` field -> the `summary` key the gate's pooled fallback reads.
_SUMMARY_METRICS = {"roc_auc": "test_roc_auc", "brier": "test_brier", "ece": "test_ece"}


def _measure(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    """One slice's metrics, refusing to measure across two protocols.

    Every slice this function returns is single-protocol by construction, so the
    check can only fire if a caller reaches past `summarise_scored`'s filter.
    It is kept because that is exactly how the original defect arrived: the
    invariant was stated in this module's docstring and enforced nowhere.
    """
    protocols = sorted(set(frame["protocol"]))
    if len(protocols) != 1:
        raise ValueError(
            f"slice {name!r} spans protocols {protocols} — an out-of-fold score and a "
            "zero-shot score are not the same measurement, and pooling them reports a "
            "population no model was asked about"
        )
    return asdict(SliceMetrics.measure(frame["label"].to_numpy(), frame["score"].to_numpy()))


def summarise_scored(
    scored: pd.DataFrame, *, source: str, exclude_unresolved: bool = False
) -> dict[str, Any]:
    """A gate-readable summary from a scored frame, one protocol per slice.

    `evaluate_promotion` gates on `per_mission[GATE_MISSION]`. A summary without
    that block makes it fall through to comparing pooled means over populations
    that may differ, which is how every stage 4 decision before 2026-08-07 was
    silently taken. The live incumbent `ca906040` has no such block, and the
    re-baseline exists only as predictions — so this is what lets the gate
    engage at all.

    **Slices are out-of-fold only, and that is the point.** A re-baselined
    incumbent carries both protocols in one frame: measured 2026-08-08, the live
    set holds 2,238 out-of-fold Kepler rows plus 243 zero-shot ones that are
    **all negatives**. Pooling them moves the Kepler figure for reasons that have
    nothing to do with the model. It happens to be worth only +0.0001 there
    (0.9915 pooled against 0.9914 out-of-fold, because the incumbent already
    ranks those negatives low) — which makes it more dangerous rather than less,
    since a plausible right answer is one nobody re-checks. Zero-shot rows are
    kept, labelled, under `zero_shot`, and never reach the gate.
    """
    required = {"tic_id", "label", "score", "mission", "protocol"}
    if missing := sorted(required - set(scored.columns)):
        raise ValueError(f"scored frame is missing {missing} — cannot build a gate summary")

    # `groupby` drops null keys by default, so an unresolved mission would leave
    # the per-mission slices summing to less than the aggregate with nothing
    # saying so. Refuse the frame instead of measuring part of it.
    #
    # A row reaches here when it was scored against a shard set the mission
    # source does not cover. On 2026-08-08 that was five rows: the incumbent was
    # re-scored on the legacy 9-dim shards (5,380 rows, no mission column) while
    # missions resolve from the current view set (5,426), and five confirmed
    # planets present in the former dropped out of the stage 2 rebuild. They are
    # outside the comparison population — no candidate can score them — so
    # excluding them is right, but it is a decision the caller makes explicitly.
    unresolved = scored[scored["mission"].isna()]
    if len(unresolved):
        if not exclude_unresolved:
            raise ValueError(
                f"{len(unresolved)} row(s) carry no mission, so they would vanish from every "
                "per-mission slice while still counting toward the aggregate. They were "
                "scored against a shard set the mission source does not cover; pass "
                "exclude_unresolved=True to drop them from the comparison population, "
                "which records each one."
            )
        log.info(
            "[summarise] excluded %d row(s) with no mission: %s",
            len(unresolved),
            sorted(int(t) for t in unresolved["tic_id"]),
        )
        scored = scored[scored["mission"].notna()]

    held_out = scored[scored["protocol"] == GATE_PROTOCOL]
    if held_out.empty:
        raise ValueError(
            f"no {GATE_PROTOCOL} rows in {source} — every slice here would be zero-shot, "
            "which is not a held-out measurement and cannot gate"
        )

    per_mission = {
        str(mission): _measure(group, str(mission))
        for mission, group in held_out.groupby("mission")
    }
    per_mission[AGGREGATE_SLICE] = _measure(held_out, AGGREGATE_SLICE)

    # The aggregate has to be exactly the missions that make it up. This is the
    # arithmetic that would have caught a mission silently leaving a comparison.
    counted = sum(v["n"] for k, v in per_mission.items() if k != AGGREGATE_SLICE)
    if counted != per_mission[AGGREGATE_SLICE]["n"]:
        raise ValueError(
            f"per-mission rows sum to {counted} but the aggregate holds "
            f"{per_mission[AGGREGATE_SLICE]['n']} — a slice is being dropped"
        )
    if GATE_MISSION not in per_mission:
        raise ValueError(
            f"{source} has no out-of-fold {GATE_MISSION} rows, so it cannot gate: "
            f"{GATE_MISSION} is 100% of the deployment population"
        )

    zero_shot = {
        str(mission): _measure(group, str(mission))
        for mission, group in scored[scored["protocol"] != GATE_PROTOCOL].groupby("mission")
    }

    return {
        # No `folds` block: this run's folds are a different split from any
        # candidate's, so pairing on fold index would compare fold *k* of one
        # partition against fold *k* of another. `paired_folds` returns None on a
        # missing block, which is the honest outcome rather than a fabricated one.
        "summary": {
            key: {"mean": per_mission[AGGREGATE_SLICE][field], "std": None}
            for field, key in _SUMMARY_METRICS.items()
        },
        "per_mission": per_mission,
        "zero_shot": zero_shot,
        "provenance": {
            "source": source,
            "protocol": str(GATE_PROTOCOL),
            "n_rows": len(scored),
            "n_gated": len(held_out),
            "excluded_unresolved": sorted(int(t) for t in unresolved["tic_id"]),
            "note": (
                "per_mission and summary are pooled out-of-fold metrics, not fold means; "
                "zero_shot is reported for coverage and never gates"
            ),
        },
    }
