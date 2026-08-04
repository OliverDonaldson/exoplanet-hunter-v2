"""Generate synthetic artefacts for the CI gate jobs.

CI has no R2 credentials, so it cannot pull the real data — but the gate
*scripts* can still run end-to-end against small synthetic artefacts that
satisfy the real schemas. The generator self-validates: everything it writes
is checked with the same code the gates run, so a schema change that breaks
the fixtures fails here, loudly, not downstream.

Usage:
    python .github/scripts/make_gate_fixtures.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from exoplanet_hunter.datasets.views_io import load_views
from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES, ViewSetArrays
from exoplanet_hunter.validation.promotion import evaluate_promotion, promote
from exoplanet_hunter.validation.schemas import (
    candidate_catalogue_schema,
    check_dv_archive,
    check_view_set,
    check_views,
    label_catalogue_schema,
)


def _labels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # TESS-majority, like the real catalogue: the DV gate's coverage
            # floor is a ratio, so two TESS rows cannot express "most targets
            # have DV products and a few genuinely do not".
            "tic_id": [1001, 1002, 1003, 1004, 1005, 757450, 211390903],
            "period": [3.5, 1.2, 5.4, 11.0, 0.9, 8.9, 2.6],
            "t0": [1325.0, 1330.5, 1412.0, 1500.25, 1290.0, 131.5, -1614.0],
            "duration": [0.12, 0.08, 0.15, 0.3, 0.05, 0.25, 0.1],
            "depth": [0.001, 0.02, 0.003, 0.0008, 0.04, 0.005, 0.0],
            "disposition": ["CP", "FP", "CP", "KP", "FP", "CONFIRMED", "CANDIDATE"],
            "label": [1, 0, 1, 1, 0, 1, -1],
            "mission": ["TESS", "TESS", "TESS", "TESS", "TESS", "Kepler", "K2"],
        }
    )


def _candidates_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["TOI", "CTOI"],
            "name": ["TOI-1000.01", "TIC 33840779.01"],
            "tic_id": [50365310, 33840779],
            "disposition": ["FP", None],
            "ra_deg": [112.5, 210.1],
            "dec_deg": [-12.3, 45.6],
            "period_days": [2.17, 3.12],
            "duration_hours": [2.03, 1.9],
            "depth_ppm": [1400.0, 800.0],
            "tess_mag": [9.1, 11.4],
        }
    )


def _views_npz(path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 4
    np.savez_compressed(
        path,
        global_views=rng.normal(0.0, 1.0, size=(n, 2001)).astype(np.float32),
        local_views=rng.normal(0.0, 1.0, size=(n, 201)).astype(np.float32),
        labels=np.array([1, 0, 1, 0], dtype=np.int8),
        tic_ids=np.array([1001, 1002, 757450, 211390903], dtype=np.int64),
        aux_features=rng.normal(0.0, 1.0, size=(n, 13)).astype(np.float32),
    )


def _dv_archive(root: Path, tic_ids: list[int]) -> Path:
    """A DV archive covering `tic_ids`, one target deliberately without products.

    The absent target is the point: ~20% of real targets have no DV, and the
    gate has to accept that while still catching a target that was never
    queried at all.
    """
    cache = root / "raw_dv"
    cache.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for i, tic in enumerate(tic_ids):
        if i == len(tic_ids) - 1:
            manifest[str(tic)] = {"success": False, "n_available": 0, "reason": "no DV products"}
            continue
        path = cache / f"tic_{tic}" / f"tess-s0001-s0009-{tic:016d}-00001_dvr.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<?xml version="1.0"?><dv:dvTargetResults/>')
        manifest[str(tic)] = {"success": True, "n_available": 1, "paths": [str(path)]}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return cache


def _view_set(labels: pd.DataFrame) -> ViewSetArrays:
    """A view set whose presence channels vary, since a mask stuck at 1 fails."""
    rng = np.random.default_rng(7)
    n = len(labels)
    views = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(0.0, 1.0, size=(n, *shape)).astype(np.float32)
        arr[..., -1] = (rng.random((n, *shape[:-1])) > 0.2).astype(np.float32)
        views[name] = arr
    scalars = pd.DataFrame(
        {
            "tic_id": labels["tic_id"].to_numpy(),
            "mission": labels["mission"].to_numpy(),
            "label": labels["label"].to_numpy(),
            "observed_transit_count": rng.integers(1, 20, n),
            "expected_transit_count": rng.integers(20, 40, n),
            "transit_completeness": rng.random(n),
            "secondary_phase": rng.random(n) - 0.5,
            "ruwe": rng.normal(1.0, 0.2, n),
            "dv_usable": rng.random(n) > 0.2,
            "has_ruwe": [True] * n,
        }
    )
    return ViewSetArrays(views=views, scalars=scalars)


def _cv_summary(auc: float, brier: float, ece: float) -> dict:
    return {
        "folds": [],
        "summary": {
            "test_roc_auc": {"mean": auc, "std": 0.005},
            "test_brier": {"mean": brier, "std": 0.003},
            "test_ece": {"mean": ece, "std": 0.003},
        },
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _labels_frame()
    label_catalogue_schema.validate(labels, lazy=True)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    # Previous == current: the shrink and leakage gates run and pass.
    labels.to_parquet(out_dir / "labels_previous.parquet", index=False)

    candidates = _candidates_frame()
    candidate_catalogue_schema.validate(candidates, lazy=True)
    candidates.to_parquet(out_dir / "candidates.parquet", index=False)

    _views_npz(out_dir / "views.npz")
    problems = check_views(load_views(out_dir / "views.npz"))
    if problems:
        raise SystemExit(f"views fixture fails its own gate: {problems}")

    trainable = labels[labels["label"].isin([0, 1])].reset_index(drop=True)
    view_set = _view_set(trainable)
    problems = check_view_set(view_set)
    if problems:
        raise SystemExit(f"view-set fixture fails its own gate: {problems}")
    view_set.save(out_dir / "viewset")

    tess_tics = labels.loc[labels["mission"] == "TESS", "tic_id"].astype(int).tolist()
    dv_dir = _dv_archive(out_dir, tess_tics)
    problems = check_dv_archive(dv_dir, tess_tics)
    if problems:
        raise SystemExit(f"DV fixture fails its own gate: {problems}")

    # Incumbent registry via the real promote() path, plus one candidate the
    # gate must accept and one it must reject.
    models_dir = out_dir / "models"
    cv_dir = models_dir / "cv" / "incumbent"
    cv_dir.mkdir(parents=True, exist_ok=True)
    incumbent = _cv_summary(auc=0.90, brier=0.10, ece=0.02)
    incumbent_path = cv_dir / "cv_summary.json"
    incumbent_path.write_text(json.dumps(incumbent))
    promote(models_dir, "incumbent", incumbent_path)

    better = _cv_summary(auc=0.93, brier=0.09, ece=0.02)
    worse = _cv_summary(auc=0.88, brier=0.11, ece=0.02)
    (out_dir / "candidate_better.json").write_text(json.dumps(better))
    (out_dir / "candidate_worse.json").write_text(json.dumps(worse))
    assert evaluate_promotion(better, incumbent).promoted
    assert not evaluate_promotion(worse, incumbent).promoted

    print(f"gate fixtures written to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
