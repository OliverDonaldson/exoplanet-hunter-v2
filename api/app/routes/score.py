"""`GET /score/{tic_id}` — the live inference endpoint.

The scoring implementation lands in `feat/fastapi-serving` by refactoring
`pipeline/scripts/score_target.py` into a library service:

    fetch light curve from MAST -> clean -> (BLS if no ephemeris) ->
    transit-masked flatten -> global/local views -> 5-fold ensemble +
    MC-Dropout -> temperature-scaled probability -> vetting diagnostics.

Until a trained V2 model bundle exists (fresh data only — V1 artefacts are
not ported), the route answers 503 so the contract is exercisable end-to-end
by the frontend and tests without pretending to score.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ScoreResponse

router = APIRouter()


@router.get("/score/{tic_id}", response_model=ScoreResponse)
def score_target(
    tic_id: int,
    period_days: float | None = Query(None, gt=0, description="Override BLS period search"),
    t0_btjd: float | None = Query(None, description="Transit epoch (BTJD); requires period_days"),
    duration_hours: float | None = Query(None, gt=0),
    n_mc: int = Query(50, ge=10, le=500, description="MC-Dropout samples"),
) -> ScoreResponse:
    raise HTTPException(
        status_code=503,
        detail=(
            "No model bundle deployed yet. The V2 pipeline trains on fresh "
            "preprocessed data before this endpoint goes live "
            "(feat/tfdata-pipeline -> feat/fastapi-serving)."
        ),
    )
