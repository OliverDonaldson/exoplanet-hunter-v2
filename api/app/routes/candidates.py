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
import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response

from app.schemas import CandidatesPage

router = APIRouter()
log = logging.getLogger(__name__)

# repo-root/data/tables/catalogue/candidates.parquet, both locally and in the
# container (where the repo root is /srv); override with CATALOGUE_PATH.
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "tables" / "catalogue" / "candidates.parquet"
)

#: repo-root/results/candidates_scored.parquet — written by
#: pipeline/scripts/score_candidates.py. Absent on a fresh checkout, in which
#: case every row is simply unscored.
_DEFAULT_SCORES = Path(__file__).resolve().parents[3] / "results" / "candidates_scored.parquet"

_SORTABLE = {
    "prob_mean",
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

# Keyed on (catalogue mtime, scores mtime): the frame is a join of the two, so
# either changing on disk has to invalidate it. Re-running the bulk scorer
# therefore shows up without a restart.
_cache: dict[str, tuple[tuple[float, float], pd.DataFrame]] = {}


def _load_catalogue() -> pd.DataFrame:
    path = Path(os.environ.get("CATALOGUE_PATH", _DEFAULT_PATH))
    if not path.exists():
        # The server's absolute layout is the operator's business, not the
        # client's — the path goes to the log, the remedy to the response.
        log.warning("[candidates] catalogue missing at %s", path)
        raise HTTPException(
            status_code=503,
            detail="Candidate catalogue is not available on this server yet.",
        )
    mtime = path.stat().st_mtime
    key = str(path)
    scores_path = Path(os.environ.get("SCORES_PATH", _DEFAULT_SCORES))
    scores_mtime = scores_path.stat().st_mtime if scores_path.exists() else 0.0
    cached = _cache.get(key)
    if cached is None or cached[0] != (mtime, scores_mtime):
        frame = pd.read_parquet(path)
        _cache[key] = ((mtime, scores_mtime), _attach_scores(frame, scores_path))
    return _cache[key][1]


def _attach_scores(catalogue: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    """Join the bulk-scored candidates onto the catalogue.

    `scripts/score_candidates.py` scores the held-out pool offline, because a
    live score is a MAST fetch plus five model passes and cannot be done for a
    page of rows on load. Rows it has not reached keep a null score and the
    console renders them as not scored.

    **These are ensemble means, not the Platt-calibrated number `/score`
    returns.** The calibrators live inside the fold bundle and cannot be
    applied to a stored mean after the fact, so the same target can show a
    different figure here and on the vetting page. `score_source` and
    `scored_at` travel with every row so a reader can tell which is which, and
    re-running the scorer is what closes the gap.
    """
    catalogue = catalogue.copy()
    for column in ("prob_mean", "prob_std", "scored_at", "score_source"):
        catalogue[column] = None
    if not scores_path.exists():
        return catalogue
    try:
        scored = pd.read_parquet(
            scores_path, columns=["tic_id", "status", "prob_mean", "prob_std", "scored_at"]
        )
    except (OSError, ValueError, KeyError):
        log.warning("[candidates] could not read scores at %s", scores_path)
        return catalogue
    # Only rows the scorer actually completed. `no_fits` and `preprocess_fail`
    # carry no probability and must not read as a low one.
    scored = scored[(scored["status"] == "ok") & scored["prob_mean"].notna()]
    scored = scored.drop_duplicates("tic_id")
    merged = catalogue.drop(columns=["prob_mean", "prob_std", "scored_at"]).merge(
        scored.drop(columns=["status"]), on="tic_id", how="left"
    )
    merged["scored_at"] = (
        merged["scored_at"].astype(object).where(merged["scored_at"].notna(), None)
    )
    merged["score_source"] = (
        merged["prob_mean"].notna().map({True: "bulk-ensemble-mean", False: None})
    )
    return merged


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
