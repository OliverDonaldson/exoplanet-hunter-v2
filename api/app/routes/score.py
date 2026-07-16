"""`GET /score/{tic_id}` — the live inference endpoint.

Wraps `exoplanet_hunter.scoring.TargetScorer`: fetch light curve (MAST or
local FITS cache) → clean → ephemeris (user > catalogue > BLS search) →
transit-masked flatten → global/local views → registered 5-fold ensemble +
MC-Dropout → calibrated probability + vetting diagnostics.

The scorer loads the promoted ensemble lazily on first request (Keras +
5 fold models — a cold start of a few seconds) and is cached for the
process lifetime. 503 until a model has been promoted to the registry;
404 when the target has no SPOC light curve.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    CentroidDiagnostics,
    Ephemeris,
    FoldPrediction,
    OddEvenDiagnostics,
    PhaseView,
    ScoreResponse,
)

router = APIRouter()

# Repo root both locally and in the container (/srv); override via env.
_ROOT = Path(__file__).resolve().parents[3]
_lock = threading.Lock()
_scorer = None


def get_scorer():
    global _scorer
    with _lock:
        if _scorer is None:
            from exoplanet_hunter.scoring import TargetScorer

            _scorer = TargetScorer(
                models_dir=Path(os.environ.get("MODEL_DIR", _ROOT / "models")),
                data_raw=Path(os.environ.get("DATA_RAW_DIR", _ROOT / "data" / "raw")),
                candidates_path=Path(
                    os.environ.get(
                        "CATALOGUE_PATH", _ROOT / "data" / "catalogue" / "candidates.parquet"
                    )
                ),
            )
        return _scorer


@router.get("/score/{tic_id}", response_model=ScoreResponse)
def score_target(
    tic_id: int,
    period_days: float | None = Query(None, gt=0, description="Override BLS period search"),
    t0_btjd: float | None = Query(None, description="Transit epoch (BTJD); requires period_days"),
    duration_hours: float | None = Query(None, gt=0),
    n_mc: int = Query(20, ge=10, le=500, description="MC-Dropout samples"),
    force_download: bool = Query(False),
    force_bls: bool = Query(
        False, description="Ignore the catalogue ephemeris; run the BLS search"
    ),
) -> ScoreResponse:
    from exoplanet_hunter.scoring import BEB_THRESHOLD_SIGMA, NoLightCurveError

    try:
        scorer = get_scorer()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No promoted model in the registry yet: {exc}",
        ) from exc

    try:
        outcome = scorer.score(
            tic_id,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            n_mc=n_mc,
            force_download=force_download,
            force_bls=force_bls,
        )
    except NoLightCurveError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ScoreResponse(
        tic_id=outcome.tic_id,
        ephemeris=Ephemeris(
            period_days=outcome.period_days,
            t0_btjd=outcome.t0_btjd,
            duration_days=outcome.duration_days,
            source=outcome.ephemeris_source,  # type: ignore[arg-type]
        ),
        prob_calibrated=outcome.prob_calibrated,
        prob_mean=outcome.prob_mean,
        prob_std=outcome.prob_std,
        per_fold=[FoldPrediction(fold=i, prob=p) for i, p in enumerate(outcome.per_fold)],
        decision_threshold=outcome.threshold,
        centroid=(
            CentroidDiagnostics(
                centroid_snr=outcome.centroid_snr,
                beb_threshold_sigma=BEB_THRESHOLD_SIGMA,
                suspicious=outcome.centroid_snr > BEB_THRESHOLD_SIGMA,
            )
            if outcome.centroid_snr is not None
            else None
        ),
        odd_even=(
            OddEvenDiagnostics(
                odd_depth_ppm=outcome.odd_even.odd_depth_ppm,
                even_depth_ppm=outcome.odd_even.even_depth_ppm,
                depth_diff_sigma=outcome.odd_even.depth_diff_sigma,
            )
            if outcome.odd_even is not None
            else None
        ),
        global_view=PhaseView(phase=outcome.global_view.phase, flux=outcome.global_view.flux),
        local_view=PhaseView(phase=outcome.local_view.phase, flux=outcome.local_view.flux),
        verdict=outcome.verdict,
        model_version=outcome.model_version,
        n_mc_samples=outcome.n_mc_samples,
    )
