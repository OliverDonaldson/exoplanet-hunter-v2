"""TIC ID -> calibrated score + vetting data: the live serving path.

This is `scripts/score_target.py` refactored into a library so the FastAPI
endpoint and the CLI share one implementation — fetch from MAST (or the
local FITS cache) → clean → ephemeris (user > catalogue > BLS search) →
transit-masked flatten → global/local views → 5-fold ensemble + MC-Dropout
→ calibration → centroid & odd/even diagnostics → plain-language verdict.

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
import pandas as pd

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.data.stellar import fetch_stellar_params
from exoplanet_hunter.features.centroid import centroid_phase_track, extract_centroid_offset
from exoplanet_hunter.features.noise import pink_noise_snr
from exoplanet_hunter.preprocess import (
    clean_lightcurve,
    flatten_lightcurve,
)
from exoplanet_hunter.preprocess.views import build_views
from exoplanet_hunter.scoring.diagnostics import (
    BEB_THRESHOLD_SIGMA,
    DurationResult,
    FalseAlarmResult,
    OddEvenResult,
    SecondaryResult,
    false_alarm_checks,
    odd_even_depths,
    significant_secondary,
    unphysical_duration,
    verdict,
)
from exoplanet_hunter.scoring.ensemble import ScoringEnsemble
from exoplanet_hunter.search import bls_period_search
from exoplanet_hunter.search.bls import bls_periodogram
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

BTJD_OFFSET = 2_457_000.0
# BLS cost is (n_periods x n_cadences x n_durations). bls_period_search caps
# n_periods; this caps n_cadences for the search only (the final phase-fold
# still uses every cadence). Together they bound an otherwise multi-minute
# search on a 180k-cadence multi-sector target.
MAX_SEARCH_CADENCES = 20_000


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
class Periodogram:
    period_days: list[float]
    power: list[float]
    best_period_days: float


@dataclass(frozen=True)
class ScoreOutcome:
    tic_id: int
    period_days: float
    t0_btjd: float
    duration_days: float
    ephemeris_source: str  # "bls" | "user" | "catalogue"
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
    odd_view: PhaseSeries | None = None
    even_view: PhaseSeries | None = None
    centroid_track: PhaseSeries | None = None  # flux carries the offset in pixels
    periodogram: Periodogram | None = None
    duration_check: DurationResult | None = None
    secondary: SecondaryResult | None = None
    false_alarms: FalseAlarmResult | None = None


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
        self._ephemeris: pd.DataFrame | None = None
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

    def _catalogue_ephemeris(self, tic_id: int) -> tuple[float, float, float] | None:
        """Published (period, t0_btjd, duration_days) for a known TOI/CTOI.

        Lets the endpoint skip the BLS search for the ~11k catalogued
        targets — the search is the dominant cost on multi-sector light
        curves. Returns None when the target is absent or its row lacks a
        usable period/epoch/duration.
        """
        if self.candidates_path is None or not self.candidates_path.exists():
            return None
        if self._ephemeris is None:
            cols = ["tic_id", "period_days", "epoch_bjd", "duration_hours"]
            df = pd.read_parquet(self.candidates_path, columns=cols)
            self._ephemeris = df.dropna(subset=cols).set_index("tic_id")
        if tic_id not in self._ephemeris.index:
            return None
        row = self._ephemeris.loc[tic_id]
        if isinstance(row, pd.DataFrame):  # duplicate TIC — take the first
            row = row.iloc[0]
        period = float(row["period_days"])
        t0 = float(row["epoch_bjd"]) - BTJD_OFFSET
        duration = float(row["duration_hours"]) / 24.0
        # A valid TESS-era epoch lands in BTJD ~1000-5000; reject dirty rows.
        if period <= 0 or duration <= 0 or not 1_000.0 < t0 < 5_000.0:
            return None
        return period, t0, duration

    def _aux_row(
        self,
        tic_id: int,
        period: float,
        t0: float,
        duration: float,
        *,
        flat_time: np.ndarray,
        flat_flux: np.ndarray,
        centroid_snr: float | None,
        odd_even: OddEvenResult | None,
        secondary: SecondaryResult | None,
        duration_check: DurationResult | None,
    ) -> np.ndarray | None:
        """Aux features matching the served model's training-time layout.

        aux_dim >= 13 is the vetting-aux layout (see build_dataset.py):
        light-curve pink-noise SNR at index 7 plus the vetting diagnostics.
        Legacy models (aux_dim 8/9) get the catalogue ExoFOP snr there
        instead — the branch keeps old bundles serving byte-identically.
        """
        aux_dim = self.ensemble.aux_dim
        if not aux_dim:
            return None
        sp = self._fetch_stellar(tic_id)
        centroid = centroid_snr if centroid_snr is not None else float("nan")
        row = [
            sp.teff if sp.teff is not None else np.nan,
            sp.radius if sp.radius is not None else np.nan,
            sp.logg if sp.logg is not None else np.nan,
            sp.tmag if sp.tmag is not None else np.nan,
            np.nan,  # depth — unknown for an ad-hoc target (as in training)
            float(duration),
            float(np.log(period)) if period > 0 else np.nan,
        ]
        if aux_dim >= 13:
            pn = pink_noise_snr(flat_time, flat_flux, period, t0, duration)
            row += [
                pn.snr if pn is not None else np.nan,
                centroid,
                odd_even.depth_diff_sigma if odd_even is not None else np.nan,
                odd_even.timing_diff_sigma
                if odd_even is not None and odd_even.timing_diff_sigma is not None
                else np.nan,
                secondary.secondary_significance if secondary is not None else np.nan,
                duration_check.q_ratio
                if duration_check is not None and duration_check.q_ratio is not None
                else np.nan,
            ]
        else:
            row.append(self._exofop_snr(tic_id))
            if aux_dim >= 9:
                row.append(centroid)
        return np.array(row, dtype=np.float32)

    def _search_lightcurve(self, cleaned, tic_id: int):
        """Unmasked flatten + decimation for BLS (search only, not the fold)."""
        lc = flatten_lightcurve(
            cleaned,
            window_length=self.preprocess.window_length,
            polyorder=self.preprocess.polyorder,
        )
        if len(lc) > MAX_SEARCH_CADENCES:
            stride = len(lc) // MAX_SEARCH_CADENCES + 1
            log.info(
                "[score] TIC %d: decimating BLS search %d->%d cadences",
                tic_id,
                len(lc),
                len(lc[::stride]),
            )
            lc = lc[::stride]
        return lc

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
        force_bls: bool = False,
        include_periodogram: bool = False,
    ) -> ScoreOutcome:
        import lightkurve as lk

        res = self.downloader.download_one(tic_id, force=force_download)
        if not res.success or res.path is None:
            raise NoLightCurveError(f"no SPOC light curve for TIC {tic_id} ({res.reason})")
        raw = lk.read(str(res.path))
        cleaned = clean_lightcurve(raw, sigma_clip=self.preprocess.sigma_clip)

        catalogue = None if force_bls else self._catalogue_ephemeris(tic_id)
        if period_days is not None and t0_btjd is not None and duration_hours is not None:
            period, t0 = float(period_days), float(t0_btjd)
            duration = float(duration_hours) / 24.0
            source = "user"
        elif catalogue is not None:
            period, t0, duration = catalogue
            source = "catalogue"
            log.info("[score] TIC %d catalogue ephemeris: P=%.4f d t0=%.4f", tic_id, period, t0)
        else:
            bls = bls_period_search(self._search_lightcurve(cleaned, tic_id))
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

        flat_time = np.asarray(flat.time.value, dtype=float)
        flat_flux = np.asarray(flat.flux.value, dtype=float)

        # Vetting diagnostics come before the model pass: models trained on
        # the 13-dim vetting-aux layout consume them as features.
        try:
            centroid_snr: float | None = float(extract_centroid_offset(raw, period, t0, duration))
        except Exception as exc:
            log.warning("[score] centroid extraction failed: %s", exc)
            centroid_snr = None
        oe = odd_even_depths(flat_time, flat_flux, period, t0, duration)
        sp = self._fetch_stellar(tic_id)
        duration_check = unphysical_duration(
            period, duration, stellar_radius=sp.radius, stellar_logg=sp.logg
        )
        secondary = significant_secondary(
            flat_time,
            flat_flux,
            period,
            t0,
            duration,
            stellar_radius=sp.radius,
            stellar_logg=sp.logg,
            stellar_teff=sp.teff,
        )
        # The model never trained on junk detections — a search-sourced
        # ephemeris gets the noise/systematic false-alarm bundle too.
        false_alarms = None
        if source == "bls":
            false_alarms = false_alarm_checks(flat_time, flat_flux, period, t0, duration)

        prediction = self.ensemble.predict(
            views.global_view,
            views.local_view,
            self._aux_row(
                tic_id,
                period,
                t0,
                duration,
                flat_time=flat_time,
                flat_flux=flat_flux,
                centroid_snr=centroid_snr,
                odd_even=oe,
                secondary=secondary,
                duration_check=duration_check,
            ),
            n_mc=n_mc,
        )

        half = float(min(max(self.preprocess.local_durations * duration / period, 1e-3), 0.5))

        # Odd/even local views: same fold, split by transit parity — the
        # overlay makes an eclipsing binary's alternating depths visible.
        odd_view = even_view = None
        parity = np.round((np.asarray(flat.time.value, dtype=float) - t0) / period).astype(int)
        for is_odd in (True, False):
            mask = (parity % 2 != 0) == is_odd
            if int(mask.sum()) < 50:
                continue
            v = build_views(
                flat[mask],
                period=period,
                t0=t0,
                duration=duration,
                global_bins=self.preprocess.global_bins,
                local_bins=self.preprocess.local_bins,
                local_durations=self.preprocess.local_durations,
            )
            series = _phase_series(v.local_view, -half, half)
            if is_odd:
                odd_view = series
            else:
                even_view = series

        track = None
        try:
            ct = centroid_phase_track(raw, period, t0, duration)
        except Exception as exc:
            log.warning("[score] centroid track failed: %s", exc)
            ct = None
        if ct is not None:
            centers, offsets = ct
            track = PhaseSeries(
                phase=[float(p) for p in centers],
                flux=[None if not np.isfinite(v) else float(v) for v in offsets],
            )

        pgram = None
        if include_periodogram:
            periods, power, best = bls_periodogram(self._search_lightcurve(cleaned, tic_id))
            pgram = Periodogram(
                period_days=[float(x) for x in periods],
                power=[float(x) for x in power],
                best_period_days=best,
            )

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
            odd_view=odd_view,
            even_view=even_view,
            centroid_track=track,
            periodogram=pgram,
            duration_check=duration_check,
            secondary=secondary,
            false_alarms=false_alarms,
            verdict=verdict(
                prediction.prob_calibrated,
                prediction.threshold,
                centroid_snr,
                oe,
                duration_check=duration_check,
                secondary=secondary,
                false_alarms=false_alarms,
            ),
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
