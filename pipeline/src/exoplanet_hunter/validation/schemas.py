"""Pandera schemas + array checks: the data half of the validation gates.

Four artefacts get validated before anything trains or serves:

  * the **label catalogue** (`data/labels/labels.parquet`) that training
    consumes — column types, disposition/label domains, ephemeris sanity;
  * the **candidate catalogue** (`data/catalogue/candidates.parquet`) that
    the API serves — the browse-table contract;
  * the **processed views** (views.npz / shard sets) — no all-NaN folds,
    label domain, shape consistency;
  * the **DV archive** (`data/raw_dv/`) — presence-mask integrity, so that
    "never queried" can never be read as "this target has no DV products".

Schemas are deliberately strict on domains and lenient on physical values
that ExoFOP legitimately leaves blank (nullable=True): the gate's job is to
catch *structural* corruption from a refresh, not to second-guess astronomy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandera.pandas as pa

from exoplanet_hunter.datasets.views_io import ViewArrays

#: TFOPWG working-group codes (TESS), as mapped by data.catalog / data.exofop.
DISPOSITIONS = ["CP", "KP", "PC", "FP", "FA", "APC"]

#: Kepler KOI vocabulary (koi_disposition) — the label catalogue is
#: multi-mission, so its disposition domain is the union of all three.
KOI_DISPOSITIONS = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]

#: K2 (k2pandc) vocabulary: the KOI strings plus REFUTED, a published planet
#: since retracted. Mirrors data.catalog.K2_DISPOSITION_LABELS.
K2_DISPOSITIONS = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE", "REFUTED"]

#: Missions the label catalogue may carry. `tic_id` holds the mission's native
#: target id — TIC, KIC, or EPIC — so it is only unique per mission.
MISSIONS = ["TESS", "Kepler", "K2"]

label_catalogue_schema = pa.DataFrameSchema(
    name="label_catalogue",
    columns={
        "tic_id": pa.Column(int, pa.Check.gt(0)),
        "period": pa.Column(float, pa.Check.gt(0), nullable=True),
        "t0": pa.Column(float, nullable=True),
        "duration": pa.Column(float, pa.Check.gt(0), nullable=True),
        "depth": pa.Column(float, pa.Check.ge(0), nullable=True),
        "disposition": pa.Column(
            str, pa.Check.isin(sorted({*DISPOSITIONS, *KOI_DISPOSITIONS, *K2_DISPOSITIONS}))
        ),
        # 1 = confirmed, 0 = false positive, -1 = held-out candidate (PC).
        "label": pa.Column(int, pa.Check.isin([-1, 0, 1])),
        "mission": pa.Column(str, pa.Check.isin(MISSIONS)),
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


def check_dv_archive(
    cache_dir: Path,
    expected_tics: Iterable[int] | None = None,
    *,
    min_coverage: float = 0.60,
    sample: int = 20,
) -> list[str]:
    """Structural checks on a fetched DV archive; returns problems (empty = pass).

    The headline check is **presence-mask integrity**. 18.5% of TESS targets
    genuinely have no DV products, and the difference-image branch masks them
    out. A target that was never queried looks identical to one that was
    queried and has none — so an interrupted fetch would silently mask out real
    data for every target after the interruption. Only the manifest can tell
    those apart, which is why it records both outcomes explicitly.
    """
    problems: list[str] = []
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return [f"no manifest at {manifest_path}"]
    try:
        manifest: dict[str, dict] = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"manifest is not readable JSON: {exc}"]
    if not manifest:
        return ["manifest is empty"]

    if expected_tics is not None:
        expected = {int(t) for t in expected_tics}
        unasked = expected - {int(k) for k in manifest}
        if unasked:
            problems.append(
                f"{len(unasked)} expected targets absent from the manifest "
                f"(e.g. {sorted(unasked)[:5]}) — never queried, not 'no DV'"
            )

    fetched = {t: e for t, e in manifest.items() if e.get("success")}
    coverage = len(fetched) / len(manifest)
    if coverage < min_coverage:
        problems.append(
            f"only {coverage:.1%} of queried targets have DV products "
            f"(floor {min_coverage:.0%}) — suspect the query, not the archive"
        )

    missing = [t for t, e in fetched.items() if not e.get("paths")]
    if missing:
        problems.append(f"{len(missing)} targets marked fetched with no paths (e.g. {missing[:5]})")

    # Sample rather than stat every file: the point is to catch a systematic
    # truncation, and a full pass over 7,199 targets would make the gate slow
    # enough that people skip it.
    checked = 0
    for tic, entry in list(fetched.items())[:sample]:
        for raw in entry.get("paths", []):
            path = Path(raw)
            checked += 1
            if not path.exists():
                problems.append(f"TIC {tic}: {path.name} recorded but missing")
            elif path.stat().st_size == 0:
                problems.append(f"TIC {tic}: {path.name} is empty")
    if fetched and checked == 0:
        problems.append("no DV files to check despite successful entries")

    return problems
