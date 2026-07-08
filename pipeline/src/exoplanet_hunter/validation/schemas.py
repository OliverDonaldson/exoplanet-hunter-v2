"""Pandera schemas + array checks: the data half of the validation gates.

Three artefacts get validated before anything trains or serves:

  * the **label catalogue** (`data/labels/labels.parquet`) that training
    consumes — column types, disposition/label domains, ephemeris sanity;
  * the **candidate catalogue** (`data/catalogue/candidates.parquet`) that
    the API serves — the browse-table contract;
  * the **processed views** (views.npz / shard sets) — no all-NaN folds,
    label domain, shape consistency.

Schemas are deliberately strict on domains and lenient on physical values
that ExoFOP legitimately leaves blank (nullable=True): the gate's job is to
catch *structural* corruption from a refresh, not to second-guess astronomy.
"""

from __future__ import annotations

import numpy as np
import pandera.pandas as pa

from exoplanet_hunter.datasets.views_io import ViewArrays

#: TFOPWG working-group codes, as mapped by data.catalog / data.exofop.
DISPOSITIONS = ["CP", "KP", "PC", "FP", "FA", "APC"]

label_catalogue_schema = pa.DataFrameSchema(
    name="label_catalogue",
    columns={
        "tic_id": pa.Column(int, pa.Check.gt(0)),
        "period": pa.Column(float, pa.Check.gt(0), nullable=True),
        "t0": pa.Column(float, nullable=True),
        "duration": pa.Column(float, pa.Check.gt(0), nullable=True),
        "depth": pa.Column(float, pa.Check.ge(0), nullable=True),
        "disposition": pa.Column(str, pa.Check.isin(DISPOSITIONS)),
        # 1 = confirmed, 0 = false positive, -1 = held-out candidate (PC).
        "label": pa.Column(int, pa.Check.isin([-1, 0, 1])),
        "mission": pa.Column(str, pa.Check.isin(["TESS", "Kepler"])),
    },
    checks=[
        # KIC and TIC numbering overlap, so uniqueness is per mission.
        pa.Check(
            lambda df: ~df.duplicated(subset=["mission", "tic_id"]),
            name="unique_target_per_mission",
            error="duplicate (mission, tic_id) rows",
        ),
        # A training catalogue with one class only is a refresh gone wrong.
        pa.Check(
            lambda df: df[df["label"] >= 0]["label"].nunique() == 2,
            name="both_classes_present",
            error="labelled rows must include both classes",
        ),
    ],
    strict=False,  # extra columns (snr, stellar params) are welcome
    coerce=True,
)

candidate_catalogue_schema = pa.DataFrameSchema(
    name="candidate_catalogue",
    columns={
        "source": pa.Column(str, pa.Check.isin(["TOI", "CTOI"])),
        "name": pa.Column(str, nullable=False),
        "tic_id": pa.Column(int, pa.Check.gt(0)),
        "disposition": pa.Column(str, pa.Check.isin(DISPOSITIONS), nullable=True),
        "ra_deg": pa.Column(float, pa.Check.in_range(0.0, 360.0), nullable=True),
        "dec_deg": pa.Column(float, pa.Check.in_range(-90.0, 90.0), nullable=True),
        # ExoFOP publishes 0.0 for "period unknown" on some CTOIs.
        "period_days": pa.Column(float, pa.Check.ge(0), nullable=True),
        "duration_hours": pa.Column(float, pa.Check.ge(0), nullable=True),
        "depth_ppm": pa.Column(float, pa.Check.ge(0), nullable=True),
        "tess_mag": pa.Column(float, pa.Check.in_range(-5.0, 30.0), nullable=True),
        # Follow-up metrics: NExScI-published (TOI) or computed (CTOI).
        "teq_k": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
        "tsm": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
        "esm": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
        "predicted_mass_me": pa.Column(float, pa.Check.gt(0), nullable=True, required=False),
        "predicted_k_ms": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(subset=["source", "name"]),
            name="unique_candidate_name",
            error="duplicate (source, name) rows",
        ),
    ],
    strict=False,
    coerce=True,
)


def check_views(views: ViewArrays, *, max_nan_frac: float = 0.5) -> list[str]:
    """Structural checks on a processed view set; returns problems (empty = pass).

    The headline check is the V2 doc's "no all-NaN folds": a target whose
    phase-folded view binned to nothing but NaN made it through preprocessing
    without data — training on it is training on imputation artefacts.
    """
    problems: list[str] = []
    n = len(views.labels)

    for name, arr in (("global_views", views.global_views), ("local_views", views.local_views)):
        if len(arr) != n:
            problems.append(f"{name}: {len(arr)} rows but {n} labels")
            continue
        all_nan = np.isnan(arr).all(axis=1)
        if all_nan.any():
            problems.append(
                f"{name}: {int(all_nan.sum())} all-NaN views (rows {np.where(all_nan)[0][:5].tolist()}…)"
            )
        nan_frac = np.isnan(arr).mean(axis=1)
        too_sparse = nan_frac > max_nan_frac
        if too_sparse.any():
            problems.append(f"{name}: {int(too_sparse.sum())} views over {max_nan_frac:.0%} NaN")

    labels = np.asarray(views.labels)
    bad_labels = ~np.isin(labels, [0, 1])
    if bad_labels.any():
        problems.append(f"labels: {int(bad_labels.sum())} values outside {{0, 1}}")
    if len(np.unique(labels[~bad_labels])) < 2:
        problems.append("labels: only one class present")

    if (np.asarray(views.tic_ids) <= 0).any():
        problems.append("tic_ids: non-positive IDs present")

    if views.aux_features is not None:
        aux = views.aux_features
        if len(aux) != n:
            problems.append(f"aux_features: {len(aux)} rows but {n} labels")
        elif np.isnan(aux).all(axis=0).any():
            dead = np.where(np.isnan(aux).all(axis=0))[0].tolist()
            problems.append(f"aux_features: columns {dead} are all-NaN")

    return problems
