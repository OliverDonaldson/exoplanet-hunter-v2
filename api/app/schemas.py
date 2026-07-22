"""Pinned JSON contract for the serving layer.

This module is the single source of truth for the shape of the
`/score/{tic_id}` response. The React console's `src/api/types.ts` mirrors it
field-for-field; change them together or not at all. Field names follow the
V1 quantities they carry: `prob_mean`/`prob_std` from `mc_dropout_predict`,
`centroid_snr` from `extract_centroid_offset`, the BLS ephemeris from
`bls_period_search`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Ephemeris(BaseModel):
    period_days: float = Field(gt=0)
    t0_btjd: float
    duration_days: float = Field(gt=0)
    source: Literal["bls", "user", "catalogue"]


class FoldPrediction(BaseModel):
    """One member of the 5-fold CV ensemble."""

    fold: int = Field(ge=0)
    prob: float = Field(ge=0, le=1)


class CentroidDiagnostics(BaseModel):
    """Centroid-shift vetting against the background-EB threshold."""

    centroid_snr: float
    beb_threshold_sigma: float = 3.0
    suspicious: bool


class OddEvenDiagnostics(BaseModel):
    """Odd vs even transit depths and timings; a large difference in either
    flags an eclipsing binary (timing: LEO-Vetter §4.4, catches eccentric EBs
    at half period whose depths match). Timing fields are optional: absent on
    older serving builds or when too few transits are usable."""

    odd_depth_ppm: float
    even_depth_ppm: float
    depth_diff_sigma: float
    odd_timing_min: float | None = None
    even_timing_min: float | None = None
    timing_diff_sigma: float | None = None
    timing_suspicious: bool | None = None


class SecondaryDiagnostics(BaseModel):
    """Significant-secondary test (LEO-Vetter §3.9 + §4.3, simplified
    Model-Shift): strongest box-depth dip outside ±2 durations of the
    primary. occultation_like means the depth ratio + implied albedo are
    consistent with a planetary occultation, so no caution is raised."""

    secondary_depth_ppm: float
    secondary_phase: float
    secondary_significance: float
    fa_threshold: float
    primary_depth_ppm: float
    depth_ratio: float
    albedo: float | None
    occultation_like: bool
    suspicious: bool
    f_red: float | None = None  # red/white noise ratio at the transit duration


class FalseAlarmDiagnostics(BaseModel):
    """Noise/systematic false-alarm bundle, computed only for BLS-found
    ephemerides (LEO-Vetter §3.3/§3.5/§3.6/§3.12) — a grouped "low-trust
    detection" caution. Individual metrics are None with too little data."""

    sweet_significance: float | None
    sweet_suspicious: bool
    asymmetry_sigma: float | None
    asymmetry_suspicious: bool
    depth_mean_median_ratio: float | None
    dmm_suspicious: bool
    gap_fraction: float | None
    gap_suspicious: bool
    suspicious: bool


class DurationDiagnostics(BaseModel):
    """Unphysical-duration test (LEO-Vetter §3.4): q = duration/period vs the
    circular-orbit expectation from stellar density. Nullable fields are absent
    when the TIC lacks radius/logg."""

    q: float
    q_circ: float | None
    q_ratio: float | None
    a_over_rstar: float | None
    suspicious: bool


class PhaseView(BaseModel):
    """A binned, phase-folded view (global: 2001 bins, local: 201 bins)."""

    phase: list[float]
    flux: list[float | None]  # None where a phase bin is empty


class CentroidTrack(BaseModel):
    """Phase-binned detrended centroid offset: flat when the transit is on
    target, a bump at phase 0 for a background eclipsing binary."""

    phase: list[float]
    offset_pixels: list[float | None]


class Periodogram(BaseModel):
    """Bounded BLS power spectrum (opt-in via ?include_periodogram=true)."""

    period_days: list[float]
    power: list[float]
    best_period_days: float


class ScoreResponse(BaseModel):
    tic_id: int
    ephemeris: Ephemeris

    # Headline numbers: calibrated ensemble mean with MC-Dropout uncertainty.
    prob_calibrated: float = Field(ge=0, le=1)
    prob_mean: float = Field(ge=0, le=1)
    prob_std: float = Field(ge=0)
    per_fold: list[FoldPrediction]
    decision_threshold: float = Field(ge=0, le=1)

    # Vetting diagnostics (None when the raw light curve lacks the columns).
    centroid: CentroidDiagnostics | None
    odd_even: OddEvenDiagnostics | None

    # Data for the console's phase-fold panels.
    global_view: PhaseView
    local_view: PhaseView

    # Optional vetting series (panel parity with V1's six-panel figure).
    odd_view: PhaseView | None = None
    even_view: PhaseView | None = None
    centroid_track: CentroidTrack | None = None
    periodogram: Periodogram | None = None

    # LEO-Vetter-style cautions (optional: absent on older serving builds).
    duration_check: DurationDiagnostics | None = None
    secondary: SecondaryDiagnostics | None = None
    false_alarms: FalseAlarmDiagnostics | None = None  # BLS-found ephemerides only

    # Plain-language verdict rendered in the console.
    verdict: str

    # Provenance: which model bundle produced this score.
    model_version: str
    n_mc_samples: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None


class ReliabilityBin(BaseModel):
    """One bin of the reliability diagram: predicted vs observed frequency."""

    prob_mean: float = Field(ge=0, le=1)  # mean calibrated prediction in bin
    frac_positive: float = Field(ge=0, le=1)  # observed positive fraction
    count: int = Field(ge=0)


class ReliabilityResponse(BaseModel):
    """Calibration quality of the promoted model, from its CV test predictions."""

    run_id: str
    n_examples: int
    bins: list[ReliabilityBin]
    ece: float = Field(ge=0, description="Expected calibration error (count-weighted)")
    brier: float = Field(ge=0)


class CandidateRow(BaseModel):
    """One row of the candidate catalogue (normalised ExoFOP TOI/CTOI export).

    Mirrors `exoplanet_hunter.data.exofop.CATALOGUE_COLUMNS`; most physical
    parameters are nullable because ExoFOP rows are incomplete for many
    community candidates.
    """

    source: Literal["TOI", "CTOI"]
    name: str
    tic_id: int
    disposition: str | None = None
    tess_mag: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    epoch_bjd: float | None = None
    period_days: float | None = None
    duration_hours: float | None = None
    depth_ppm: float | None = None
    planet_radius_re: float | None = None
    planet_snr: float | None = None
    # Follow-up prioritisation (Kempton 2018): NExScI values for TOIs,
    # computed via features.followup for CTOIs (TSM/ESM need J/K mags and
    # stay null there).
    teq_k: float | None = None
    tsm: float | None = None
    esm: float | None = None
    # POE observables (NASA archive formulae), computed for every row from the
    # stellar parameters + period: insolation in Earth flux, habitable-zone
    # edges in AU (luminosity-scaled recent-Venus / early-Mars).
    insolation_earth: float | None = None
    hz_inner_au: float | None = None
    hz_outer_au: float | None = None
    predicted_mass_me: float | None = None
    predicted_k_ms: float | None = None
    stellar_teff_k: float | None = None
    stellar_logg: float | None = None
    stellar_radius_rsun: float | None = None
    stellar_distance_pc: float | None = None
    sectors: str | None = None
    promoted_to_toi: str | None = None
    comments: str | None = None
    date_modified: str | None = None


class CandidatesPage(BaseModel):
    total: int
    offset: int
    rows: list[CandidateRow]
