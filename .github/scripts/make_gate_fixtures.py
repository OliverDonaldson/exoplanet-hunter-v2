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
from exoplanet_hunter.validation.promotion import evaluate_promotion, promote
from exoplanet_hunter.validation.schemas import (
    candidate_catalogue_schema,
    check_views,
    label_catalogue_schema,
)


def _labels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tic_id": [1001, 1002, 757450, 211390903],
            "period": [3.5, 1.2, 8.9, 2.6],
            "t0": [1325.0, 1330.5, 131.5, -1614.0],
            "duration": [0.12, 0.08, 0.25, 0.1],
            "depth": [0.001, 0.02, 0.005, 0.0],
            "disposition": ["CP", "FP", "CONFIRMED", "CANDIDATE"],
            "label": [1, 0, 1, -1],
            "mission": ["TESS", "TESS", "Kepler", "K2"],
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
