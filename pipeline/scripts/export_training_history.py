"""Export a run's per-epoch training history from MLflow into models/history/.

The history was always persisted — it is in the run's MLflow *child* runs, one
per fold, tagged `mlflow.parentRunId`. What was missing is a path from there to
the console: `mlflow.db` is 33 MB of local development state and is excluded
from the serving image, so the API cannot read it. This writes the five series
the Model page plots into a small file git carries, so the container has it
without a DVC pull. See report.md §5.2 and issue #25.

Usage (from the repository root):

    python pipeline/scripts/export_training_history.py [--run <run_id>]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

#: Keras metric name → the name the wire contract uses. `val_*` is the series
#: with real step numbers; MLflow's autologger also writes a `validation_*`
#: duplicate with every row at step 0, which sorts into nonsense. Reading the
#: wrong one is silent, so the allowlist names only the good copy.
_SERIES = {
    "loss": "loss",
    "val_loss": "val_loss",
    "auc": "auc",
    "val_auc": "val_auc",
    "learning_rate": "learning_rate",
}

#: Single-value metrics EarlyStopping writes at the end of a fold.
_SCALARS = ("restored_epoch", "stopped_epoch")

#: The run's own early-stopping settings, read rather than assumed: whether a
#: trailing restore record exists at all depends on `restore_best_weights`, and
#: what `restored_epoch` is the best of depends on `monitor`.
_EARLY_STOPPING = "train.callbacks.early_stopping"


def _series(con: sqlite3.Connection, run_uuid: str, key: str) -> list[float]:
    rows = con.execute(
        "select value from metrics where run_uuid=? and key=? order by step", (run_uuid, key)
    ).fetchall()
    return [float(r[0]) for r in rows]


def _scalar(con: sqlite3.Connection, run_uuid: str, key: str) -> int | None:
    row = con.execute(
        "select value from metrics where run_uuid=? and key=? order by step desc limit 1",
        (run_uuid, key),
    ).fetchone()
    return None if row is None else int(row[0])


def _drop_restore_record(series: dict[str, list[float]], restored: int, stopped: int) -> None:
    """Drop the trailing sample, which is not an epoch.

    With `restore_best_weights=True` the autologger re-logs every metric for
    the restored weights at `step = stopped_epoch + 1`, milliseconds after the
    last real epoch. It is a bitwise copy of the best epoch's row, so a chart
    that plots it draws an epoch that never ran and ends every curve exactly on
    its own best value. `docs/figures/training_curves.png` did until 2026-09-11.

    Raises rather than trimming on anything that does not match that shape: a
    fold that ran to its epoch cap never restores, and silently deleting its
    last real epoch would be the same class of defect pointed the other way.
    """
    n = len(series["val_auc"])
    if n != stopped + 2:
        raise SystemExit(
            f"expected {stopped + 2} samples for a fold stopped at epoch {stopped} "
            f"(epochs 0-{stopped} plus one restore record), found {n}; the "
            "autologger's shape has changed and the trim would cut a real epoch"
        )
    for key, values in series.items():
        if values[-1] != values[restored]:
            raise SystemExit(
                f"the trailing {key} sample is {values[-1]} and epoch {restored} is "
                f"{values[restored]}; only an exact copy is the restore record"
            )
        del values[-1]


def _early_stopping(con: sqlite3.Connection, run_id: str) -> dict:
    """The parent run's recorded early-stopping configuration."""
    params = dict(
        con.execute(
            "select key, value from params where run_uuid=? and key like ?",
            (run_id, f"{_EARLY_STOPPING}.%"),
        ).fetchall()
    )
    missing = [
        k
        for k in ("monitor", "mode", "patience", "restore_best_weights")
        if f"{_EARLY_STOPPING}.{k}" not in params
    ]
    if missing:
        raise SystemExit(f"run {run_id} records no {', '.join(missing)} for early stopping")
    return {
        "monitor": params[f"{_EARLY_STOPPING}.monitor"],
        "monitor_mode": params[f"{_EARLY_STOPPING}.mode"],
        "patience": int(params[f"{_EARLY_STOPPING}.patience"]),
        "restore_best_weights": params[f"{_EARLY_STOPPING}.restore_best_weights"] == "True",
    }


def _child_runs(con: sqlite3.Connection, run_id: str) -> dict[str, str]:
    folds = dict(
        con.execute(
            "select name, run_uuid from runs where run_uuid in "
            "(select run_uuid from tags where key='mlflow.parentRunId' and value=?)",
            (run_id,),
        ).fetchall()
    )
    if not folds:
        raise SystemExit(f"no MLflow child runs carry mlflow.parentRunId={run_id}")
    return folds


def export(db_path: Path, run_id: str) -> dict:
    if not db_path.exists():
        raise SystemExit(f"{db_path} does not exist; the history lives in the MLflow store")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        settings = _early_stopping(con, run_id)
        children = _child_runs(con, run_id)
        folds = []
        for name, uuid in sorted(children.items()):
            series = {wire: _series(con, uuid, key) for key, wire in _SERIES.items()}
            n = len(series["val_loss"])
            fold: dict = {"fold": int(name.rsplit("-", 1)[1]), "epochs": n}
            # A fold that logged no epoch series is recorded as a fold with
            # none, not dropped: ca906040's fold 0 is exactly that case, and a
            # panel that silently plots four curves for a five-fold run is the
            # defect this file exists to fix.
            if n:
                scalars = {k: _scalar(con, uuid, k) for k in _SCALARS}
                restored, stopped = scalars["restored_epoch"], scalars["stopped_epoch"]
                if restored is None or stopped is None:
                    raise SystemExit(f"{name} logged an epoch series and no early-stopping record")
                if settings["restore_best_weights"]:
                    _drop_restore_record(series, restored, stopped)
                fold["epochs"] = len(series["val_auc"])
                fold.update(series)
                fold.update(scalars)
            else:
                fold["note"] = "no per-epoch history was logged for this fold"
            folds.append(fold)
    finally:
        con.close()
    drawn = [f for f in folds if f["epochs"]]
    if not drawn:
        raise SystemExit(f"run {run_id} has child runs but not one per-epoch series")
    return {
        "run_id": run_id,
        "monitor": settings["monitor"],
        "monitor_mode": settings["monitor_mode"],
        "patience": settings["patience"],
        "n_folds": len(folds),
        "n_folds_with_history": len(drawn),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run id (default: the promoted run)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--mlflow-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_id = args.run or json.loads((args.models_dir / "registry.json").read_text())["run_id"]
    out = args.out or args.models_dir / "history" / f"{run_id}.json"
    history = export(args.mlflow_db, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, separators=(",", ":")) + "\n")
    log.info(
        "wrote %s — %d of %d folds carry epoch history",
        out,
        history["n_folds_with_history"],
        history["n_folds"],
    )


if __name__ == "__main__":
    main()
