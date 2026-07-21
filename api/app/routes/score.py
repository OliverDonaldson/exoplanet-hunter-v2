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
    CentroidTrack,
    DurationDiagnostics,
    Ephemeris,
    FalseAlarmDiagnostics,
    FoldPrediction,
    OddEvenDiagnostics,
    Periodogram,
    PhaseView,
    ScoreResponse,
    SecondaryDiagnostics,
)

router = APIRouter()

# Repo root both locally and in the container (/srv); override via env.
_ROOT = Path(__file__).resolve().parents[3]
_lock = threading.Lock()
# One score at a time: concurrent requests thrash the single serving CPU, and
# two scores of the same TIC can rewrite a FITS under the other's memory-map.
_score_lock = threading.Lock()
_scorer = None

# Process-lifetime response cache — a score is deterministic given the
# ephemeris and n_mc, and the console re-requests a target on every click.
_cache: dict[tuple, ScoreResponse] = {}
_CACHE_MAX = 128


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
    include_periodogram: bool = Query(
        False, description="Also run a bounded BLS and return its power spectrum"
    ),
) -> ScoreResponse:
    from exoplanet_hunter.scoring import BEB_THRESHOLD_SIGMA, NoLightCurveError
    from exoplanet_hunter.scoring.diagnostics import ODD_EVEN_TIMING_SIGMA

    cache_key = (tic_id, period_days, t0_btjd, duration_hours, n_mc, force_bls, include_periodogram)
    if not force_download and cache_key in _cache:
        return _cache[cache_key]

    try:
        scorer = get_scorer()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No promoted model in the registry yet: {exc}",
        ) from exc

    try:
        with _score_lock:
            outcome = scorer.score(
                tic_id,
                period_days=period_days,
                t0_btjd=t0_btjd,
                duration_hours=duration_hours,
                n_mc=n_mc,
                force_download=force_download,
                force_bls=force_bls,
                include_periodogram=include_periodogram,
            )
    except NoLightCurveError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = ScoreResponse(
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
                odd_timing_min=outcome.odd_even.odd_timing_min,
                even_timing_min=outcome.odd_even.even_timing_min,
                timing_diff_sigma=outcome.odd_even.timing_diff_sigma,
                timing_suspicious=(
                    outcome.odd_even.timing_diff_sigma > ODD_EVEN_TIMING_SIGMA
                    if outcome.odd_even.timing_diff_sigma is not None
                    else None
                ),
            )
            if outcome.odd_even is not None
            else None
        ),
        global_view=PhaseView(phase=outcome.global_view.phase, flux=outcome.global_view.flux),
        local_view=PhaseView(phase=outcome.local_view.phase, flux=outcome.local_view.flux),
        odd_view=(
            PhaseView(phase=outcome.odd_view.phase, flux=outcome.odd_view.flux)
            if outcome.odd_view
            else None
        ),
        even_view=(
            PhaseView(phase=outcome.even_view.phase, flux=outcome.even_view.flux)
            if outcome.even_view
            else None
        ),
        centroid_track=(
            CentroidTrack(
                phase=outcome.centroid_track.phase,
                offset_pixels=outcome.centroid_track.flux,
            )
            if outcome.centroid_track
            else None
        ),
        periodogram=(
            Periodogram(
                period_days=outcome.periodogram.period_days,
                power=outcome.periodogram.power,
                best_period_days=outcome.periodogram.best_period_days,
            )
            if outcome.periodogram
            else None
        ),
        duration_check=(
            DurationDiagnostics(
                q=outcome.duration_check.q,
                q_circ=outcome.duration_check.q_circ,
                q_ratio=outcome.duration_check.q_ratio,
                a_over_rstar=outcome.duration_check.a_over_rstar,
                suspicious=outcome.duration_check.suspicious,
            )
            if outcome.duration_check is not None
            else None
        ),
        secondary=(
            SecondaryDiagnostics(
                secondary_depth_ppm=outcome.secondary.secondary_depth_ppm,
                secondary_phase=outcome.secondary.secondary_phase,
                secondary_significance=outcome.secondary.secondary_significance,
                fa_threshold=outcome.secondary.fa_threshold,
                primary_depth_ppm=outcome.secondary.primary_depth_ppm,
                depth_ratio=outcome.secondary.depth_ratio,
                albedo=outcome.secondary.albedo,
                occultation_like=outcome.secondary.occultation_like,
                suspicious=outcome.secondary.suspicious,
                f_red=outcome.secondary.f_red,
            )
            if outcome.secondary is not None
            else None
        ),
        false_alarms=(
            FalseAlarmDiagnostics(
                sweet_significance=outcome.false_alarms.sweet_significance,
                sweet_suspicious=outcome.false_alarms.sweet_suspicious,
                asymmetry_sigma=outcome.false_alarms.asymmetry_sigma,
                asymmetry_suspicious=outcome.false_alarms.asymmetry_suspicious,
                depth_mean_median_ratio=outcome.false_alarms.depth_mean_median_ratio,
                dmm_suspicious=outcome.false_alarms.dmm_suspicious,
                gap_fraction=outcome.false_alarms.gap_fraction,
                gap_suspicious=outcome.false_alarms.gap_suspicious,
                suspicious=outcome.false_alarms.suspicious,
            )
            if outcome.false_alarms is not None
            else None
        ),
        verdict=outcome.verdict,
        model_version=outcome.model_version,
        n_mc_samples=outcome.n_mc_samples,
    )
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[cache_key] = response
    return response
