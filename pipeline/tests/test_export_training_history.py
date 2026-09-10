"""The training-history exporter, and the trailing sample that is not an epoch.

MLflow's autologger re-logs every metric for the restored weights one step past
the last epoch. It is a bitwise copy of the best epoch, so a chart that plots it
draws an epoch that never ran and ends every curve exactly on its own best
value — which is what `docs/figures/training_curves.png` did until 2026-09-11
and what the served run's report table counted. These cover the drop and, more
importantly, that it refuses to drop anything else.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "_export_training_history",
        Path(__file__).resolve().parents[1] / "scripts" / "export_training_history.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = _exporter()

_SERIES = ("loss", "val_loss", "auc", "val_auc", "learning_rate")


_ES = "train.callbacks.early_stopping"


def _db(tmp_path: Path, folds: dict[str, dict], *, restore: str = "True") -> Path:
    """An MLflow store holding one parent run and the child runs described."""
    path = tmp_path / "mlflow.db"
    con = sqlite3.connect(path)
    con.execute("create table runs (run_uuid text, name text)")
    con.execute("create table tags (run_uuid text, key text, value text)")
    con.execute("create table params (run_uuid text, key text, value text)")
    con.execute("create table metrics (run_uuid text, key text, value real, step integer)")
    for key, value in (
        ("monitor", "val_auc"),
        ("mode", "max"),
        ("patience", "25"),
        ("restore_best_weights", restore),
    ):
        con.execute("insert into params values ('parent', ?, ?)", (f"{_ES}.{key}", value))
    for name, spec in folds.items():
        uuid = f"uuid-{name}"
        con.execute("insert into runs values (?, ?)", (uuid, name))
        con.execute(
            "insert into tags values (?, 'mlflow.parentRunId', 'parent')",
            (uuid,),
        )
        for key, values in spec.get("series", {}).items():
            for step, value in enumerate(values):
                con.execute("insert into metrics values (?, ?, ?, ?)", (uuid, key, value, step))
        for key in ("restored_epoch", "stopped_epoch"):
            if key in spec:
                con.execute("insert into metrics values (?, ?, ?, 0)", (uuid, key, spec[key]))
    con.commit()
    con.close()
    return path


def _fold(epochs: int, restored: int) -> dict:
    """A fold that ran `epochs` epochs and carries the restore record."""
    series = {k: [float(i) for i in range(epochs)] for k in _SERIES}
    for values in series.values():
        values.append(values[restored])
    return {"series": series, "restored_epoch": restored, "stopped_epoch": epochs - 1}


def test_the_restore_record_is_dropped_from_every_series(tmp_path):
    db = _db(tmp_path, {"fold-0": _fold(10, 4)})
    fold = exporter.export(db, "parent")["folds"][0]
    assert fold["epochs"] == 10
    for key in _SERIES:
        assert len(fold[key]) == 10
        assert fold[key][-1] == 9.0


def test_the_restored_epoch_survives_the_drop(tmp_path):
    """The scalar is what the console marks, so trimming must not disturb it."""
    db = _db(tmp_path, {"fold-0": _fold(10, 4)})
    fold = exporter.export(db, "parent")["folds"][0]
    assert fold["restored_epoch"] == 4
    assert fold["stopped_epoch"] == 9
    assert fold["val_auc"][4] == 4.0


def test_a_trailing_sample_that_is_not_a_copy_raises(tmp_path):
    """The guard on the drop: a real epoch must never be trimmed away."""
    spec = _fold(10, 4)
    spec["series"]["val_auc"][-1] = 99.0
    db = _db(tmp_path, {"fold-0": spec})
    with pytest.raises(SystemExit, match="only an exact copy is the restore record"):
        exporter.export(db, "parent")


def test_a_fold_that_never_early_stopped_raises(tmp_path):
    """No restore record means the last sample is an epoch, so nothing is cut."""
    spec = _fold(10, 4)
    for values in spec["series"].values():
        values.pop()
    db = _db(tmp_path, {"fold-0": spec})
    with pytest.raises(SystemExit, match="would cut a real epoch"):
        exporter.export(db, "parent")


def test_a_fold_with_no_epoch_series_is_recorded_not_dropped(tmp_path):
    """ca906040's fold 0. Four curves for a five-fold run must say so."""
    db = _db(tmp_path, {"fold-0": {}, "fold-1": _fold(10, 4)})
    history = exporter.export(db, "parent")
    assert history["n_folds"] == 2
    assert history["n_folds_with_history"] == 1
    empty = next(f for f in history["folds"] if f["fold"] == 0)
    assert empty["epochs"] == 0
    assert "val_auc" not in empty
    assert "no per-epoch history" in empty["note"]


def test_a_run_with_no_child_runs_raises(tmp_path):
    db = _db(tmp_path, {})
    with pytest.raises(SystemExit, match="no MLflow child runs"):
        exporter.export(db, "parent")


def test_a_run_where_no_fold_logged_a_series_raises(tmp_path):
    """An export of nothing would serve a panel that says the data exists."""
    db = _db(tmp_path, {"fold-0": {}, "fold-1": {}})
    with pytest.raises(SystemExit, match="not one per-epoch series"):
        exporter.export(db, "parent")


def test_the_exported_file_is_json_the_api_can_read(tmp_path):
    db = _db(tmp_path, {"fold-0": _fold(10, 4)})
    out = tmp_path / "history.json"
    out.write_text(json.dumps(exporter.export(db, "parent")))
    loaded = json.loads(out.read_text())
    assert loaded["monitor"] == "val_auc"
    assert loaded["monitor_mode"] == "max"


def test_the_early_stopping_settings_are_read_from_the_run(tmp_path):
    """Not constants in the exporter. What `restored_epoch` is the best of is a
    property of the run, and a file that assumed it would be wrong in silence
    the first time the config changed."""
    db = _db(tmp_path, {"fold-0": _fold(10, 4)})
    history = exporter.export(db, "parent")
    assert (history["monitor"], history["monitor_mode"], history["patience"]) == (
        "val_auc",
        "max",
        25,
    )


def test_nothing_is_trimmed_when_weights_were_never_restored(tmp_path):
    """`restore_best_weights=False` logs no restore record, so the last sample
    is a real epoch and the trim must not run at all."""
    spec = _fold(10, 4)
    for values in spec["series"].values():
        values.pop()
    db = _db(tmp_path, {"fold-0": spec}, restore="False")
    fold = exporter.export(db, "parent")["folds"][0]
    assert fold["epochs"] == 10
    assert fold["val_auc"][-1] == 9.0


def test_a_run_with_no_early_stopping_params_raises(tmp_path):
    db = _db(tmp_path, {"fold-0": _fold(10, 4)})
    con = sqlite3.connect(db)
    con.execute("delete from params where key like ?", (f"{_ES}.monitor",))
    con.commit()
    con.close()
    with pytest.raises(SystemExit, match="records no monitor"):
        exporter.export(db, "parent")
