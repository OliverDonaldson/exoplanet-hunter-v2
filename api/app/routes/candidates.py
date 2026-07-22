"""Candidate-catalogue endpoints: browse as JSON, export as CSV.

Serves the table produced by `pipeline/scripts/ingest_exofop.py`. Both
endpoints accept the same filters, so "download CSV" from the console
exports exactly what the table shows. The parquet is re-read only when its
mtime changes; at ~11k rows the in-memory copy is trivial. Once
`feat/dashboard` lands DuckDB, this loader becomes a DuckDB view over
scores.parquet joined onto the catalogue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response

from app.schemas import CandidatesPage

router = APIRouter()

# repo-root/data/catalogue/candidates.parquet, both locally and in the
# container (where the repo root is /srv); override with CATALOGUE_PATH.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "catalogue" / "candidates.parquet"

_SORTABLE = {
    "name",
    "tic_id",
    "disposition",
    "tess_mag",
    "period_days",
    "duration_hours",
    "depth_ppm",
    "planet_radius_re",
    "planet_snr",
    "teq_k",
    "tsm",
    "esm",
    "insolation_earth",
    "predicted_mass_me",
    "predicted_k_ms",
    "stellar_teff_k",
    "stellar_distance_pc",
}

_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _load_catalogue() -> pd.DataFrame:
    path = Path(os.environ.get("CATALOGUE_PATH", _DEFAULT_PATH))
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Candidate catalogue not found at {path}. "
                "Run pipeline/scripts/ingest_exofop.py first."
            ),
        )
    mtime = path.stat().st_mtime
    key = str(path)
    cached = _cache.get(key)
    if cached is None or cached[0] != mtime:
        _cache[key] = (mtime, pd.read_parquet(path))
    return _cache[key][1]


def _apply_filters(
    catalogue: pd.DataFrame,
    search: str | None,
    disposition: str | None,
    source: str | None,
) -> pd.DataFrame:
    out = catalogue
    if source:
        out = out[out["source"] == source.upper()]
    if disposition:
        if disposition == "none":
            out = out[out["disposition"].isna()]
        else:
            out = out[out["disposition"] == disposition.upper()]
    if search:
        needle = search.strip().lower()
        hay = (
            out["name"].astype(str).str.lower().str.contains(needle, regex=False)
            | out["tic_id"].astype(str).str.contains(needle, regex=False)
            | out["comments"].astype(str).str.lower().str.contains(needle, regex=False)
        )
        out = out[hay]
    return out


@router.get("/candidates", response_model=CandidatesPage)
def list_candidates(
    search: str | None = Query(None, description="Substring match on name / TIC ID / comments"),
    disposition: str | None = Query(
        None, description="TFOPWG code (PC, CP, KP, FP, FA, APC) or 'none'"
    ),
    source: str | None = Query(None, description="TOI or CTOI"),
    sort_by: str = Query("tess_mag"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> CandidatesPage:
    if sort_by not in _SORTABLE:
        raise HTTPException(422, detail=f"sort_by must be one of {sorted(_SORTABLE)}")
    filtered = _apply_filters(_load_catalogue(), search, disposition, source)
    filtered = filtered.sort_values(sort_by, ascending=order == "asc", na_position="last")
    page = filtered.iloc[offset : offset + limit]
    # to_json handles NaN -> null and numpy scalar -> plain JSON types.
    rows = json.loads(page.to_json(orient="records"))
    return CandidatesPage(total=len(filtered), offset=offset, rows=rows)


@router.get("/candidates.csv")
def download_candidates_csv(
    search: str | None = Query(None),
    disposition: str | None = Query(None),
    source: str | None = Query(None),
) -> Response:
    filtered = _apply_filters(_load_catalogue(), search, disposition, source)
    return Response(
        content=filtered.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="candidates.csv"'},
    )
