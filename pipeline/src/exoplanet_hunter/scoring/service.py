"""TIC ID -> calibrated score + vetting data: the live serving path.

This is `scripts/score_target.py` refactored into a library so the FastAPI
endpoint and the CLI share one implementation — fetch from MAST (or the
local FITS cache) → clean → BLS if no ephemeris given → transit-masked
flatten → global/local views → 5-fold ensemble + MC-Dropout → temperature
calibration → centroid & odd/even diagnostics → plain-language verdict.

Preprocessing parameters default to conf/preprocess/default.yaml's values so
the service needs no Hydra at runtime; the dataclass keeps them overridable
and in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.data.stellar import fetch_stellar_params
from exoplanet_hunter.features.centroid import extract_centroid_offset
from exoplanet_hunter.preprocess import (
    clean_lightcurve,
    flatten_lightcurve,
)
from exoplanet_hunter.preprocess.views import build_views
from exoplanet_hunter.scoring.diagnostics import (
    BEB_THRESHOLD_SIGMA,
    OddEvenResult,
    odd_even_depths,
    verdict,
)
from exoplanet_hunter.scoring.ensemble import ScoringEnsemble
from exoplanet_hunter.search import bls_period_search
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


class NoLightCurveError(LookupError):
    """The target has no SPOC light curve on MAST (or the fetch failed)."""


@dataclass(frozen=True)
class PreprocessParams:
    """Mirror of conf/preprocess/default.yaml — keep the two in sync."""

    sigma_clip: float = 5.0
    window_length: int = 301
    polyorder: int = 2
    global_bins: int = 2001
    local_bins: int = 201
    local_durations: float = 3.0


@dataclass(frozen=True)
class PhaseSeries:
    phase: list[float]
    flux: list[float | None]


@dataclass(frozen=True)
class ScoreOutcome:
    tic_id: int
    period_days: float
    t0_btjd: float
    duration_days: float
    ephemeris_source: str  # "bls" | "user"
    per_fold: list[float]
    prob_calibrated: float
    prob_mean: float
    prob_std: float
    threshold: float
    centroid_snr: float | None
    odd_even: OddEvenResult | None
    global_view: PhaseSeries
    local_view: PhaseSeries
    verdict: str
    model_version: str
    n_mc_samples: int


def _phase_series(view: np.ndarray, lo: float, hi: float) -> PhaseSeries:
    edges = np.linspace(lo, hi, len(view) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    flux = [None if not np.isfinite(v) else float(v) for v in view]
    return PhaseSeries(phase=[float(p) for p in centers], flux=flux)


class TargetScorer:
    """Loads the registered ensemble once; scores targets on demand."""

    def __init__(
        self,
        models_dir: Path,
        data_raw: Path,
        *,
        candidates_path: Path | None = None,
        preprocess: PreprocessParams | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.data_raw = data_raw
        self.candidates_path = candidates_path
        self.preprocess = preprocess or PreprocessParams()
        self.ensemble = ScoringEnsemble.from_registry(models_dir)
        self.downloader = LightCurveDownloader(data_raw, author="SPOC", cadence=120)
        self._snr_series: Any | None = None
        # Stellar params are immutable per TIC — cache the TAP round-trip
        # (several seconds) so re-scores of a target are dominated by MC only.
        self._fetch_stellar = lru_cache(maxsize=4096)(fetch_stellar_params)

    # ------------------------------------------------------------- aux row --

    def _exofop_snr(self, tic_id: int) -> float:
        if self.candidates_path is None or not self.candidates_path.exists():
            return float("nan")
        if self._snr_series is None:
            # One parquet read per process, not per score request.
            from exoplanet_hunter.data.exofop import toi_snr_by_tic

            self._snr_series = toi_snr_by_tic(self.candidates_path)
        return float(self._snr_series.get(tic_id, float("nan")))

    def _aux_row(
        self, tic_id: int, raw_lc: Any, period: float, t0: float, duration: float
    ) -> np.ndarray | None:
        aux_dim = self.ensemble.aux_dim
        if not aux_dim:
            return None
        sp = self._fetch_stellar(tic_id)
        row = [
            sp.teff if sp.teff is not None else np.nan,
            sp.radius if sp.radius is not None else np.nan,
            sp.logg if sp.logg is not None else np.nan,
            sp.tmag if sp.tmag is not None else np.nan,
            np.nan,  # depth — unknown for an ad-hoc target (as in training)
            float(duration),
            float(np.log(period)) if period > 0 else np.nan,
            self._exofop_snr(tic_id),
        ]
        if aux_dim >= 9:
            try:
                row.append(float(extract_centroid_offset(raw_lc, period, t0, duration)))
            except Exception as exc:  # centroid columns often missing
                log.warning("[score] centroid extraction for aux failed: %s", exc)
                row.append(float("nan"))
        return np.array(row, dtype=np.float32)

    # --------------------------------------------------------------- score --

    def score(
        self,
        tic_id: int,
        *,
        period_days: float | None = None,
        t0_btjd: float | None = None,
        duration_hours: float | None = None,
        n_mc: int = 50,
        force_download: bool = False,
    ) -> ScoreOutcome:
        import lightkurve as lk

        res = self.downloader.download_one(tic_id, force=force_download)
        if not res.success or res.path is None:
            raise NoLightCurveError(f"no SPOC light curve for TIC {tic_id} ({res.reason})")
        raw = lk.read(str(res.path))
        cleaned = clean_lightcurve(raw, sigma_clip=self.preprocess.sigma_clip)

        if period_days is None or t0_btjd is None or duration_hours is None:
            # Unmasked flatten just for the search; re-flattened with a
            # transit mask below once the ephemeris is known.
            lc_search = flatten_lightcurve(
                cleaned,
                window_length=self.preprocess.window_length,
                polyorder=self.preprocess.polyorder,
            )
            bls = bls_period_search(lc_search)
            period, t0, duration = float(bls.period), float(bls.t0), float(bls.duration)
            source = "bls"
            log.info(
                "[score] TIC %d BLS: P=%.4f d t0=%.4f dur=%.3f d SNR=%.1f",
                tic_id,
                period,
                t0,
                duration,
                float(bls.snr),
            )
        else:
            period, t0 = float(period_days), float(t0_btjd)
            duration = float(duration_hours) / 24.0
            source = "user"

        flat = flatten_lightcurve(
            cleaned,
            window_length=self.preprocess.window_length,
            polyorder=self.preprocess.polyorder,
            period=period,
            t0=t0,
            duration=duration,
        )
        views = build_views(
            flat,
            period=period,
            t0=t0,
            duration=duration,
            global_bins=self.preprocess.global_bins,
            local_bins=self.preprocess.local_bins,
            local_durations=self.preprocess.local_durations,
        )

        prediction = self.ensemble.predict(
            views.global_view,
            views.local_view,
            self._aux_row(tic_id, raw, period, t0, duration),
            n_mc=n_mc,
        )

        try:
            centroid_snr: float | None = float(extract_centroid_offset(raw, period, t0, duration))
        except Exception as exc:
            log.warning("[score] centroid extraction failed: %s", exc)
            centroid_snr = None
        oe = odd_even_depths(
            np.asarray(flat.time.value, dtype=float),
            np.asarray(flat.flux.value, dtype=float),
            period,
            t0,
            duration,
        )

        half = float(min(max(self.preprocess.local_durations * duration / period, 1e-3), 0.5))
        return ScoreOutcome(
            tic_id=tic_id,
            period_days=period,
            t0_btjd=t0,
            duration_days=duration,
            ephemeris_source=source,
            per_fold=prediction.per_fold,
            prob_calibrated=prediction.prob_calibrated,
            prob_mean=prediction.prob_mean,
            prob_std=prediction.prob_std,
            threshold=prediction.threshold,
            centroid_snr=centroid_snr,
            odd_even=oe,
            global_view=_phase_series(views.global_view, -0.5, 0.5),
            local_view=_phase_series(views.local_view, -half, half),
            verdict=verdict(prediction.prob_calibrated, prediction.threshold, centroid_snr, oe),
            model_version=f"cnn_dualview-cv-{self.ensemble.run_id[:8]}",
            n_mc_samples=n_mc,
        )


__all__ = [
    "BEB_THRESHOLD_SIGMA",
    "NoLightCurveError",
    "PhaseSeries",
    "PreprocessParams",
    "ScoreOutcome",
    "TargetScorer",
]
