/**
 * Mirror of api/app/schemas.py — the pinned /score/{tic_id} contract.
 * Change the two files together or not at all.
 */

export interface Ephemeris {
  period_days: number;
  t0_btjd: number;
  duration_days: number;
  source: "bls" | "user" | "catalogue";
}

export interface FoldPrediction {
  fold: number;
  prob: number;
}

export interface CentroidDiagnostics {
  centroid_snr: number;
  beb_threshold_sigma: number;
  suspicious: boolean;
}

export interface OddEvenDiagnostics {
  odd_depth_ppm: number;
  even_depth_ppm: number;
  depth_diff_sigma: number;
  odd_timing_min?: number | null;
  even_timing_min?: number | null;
  timing_diff_sigma?: number | null;
  timing_suspicious?: boolean | null;
}

export interface DurationDiagnostics {
  q: number;
  q_circ: number | null;
  q_ratio: number | null;
  a_over_rstar: number | null;
  suspicious: boolean;
}

export interface PhaseView {
  phase: number[];
  flux: (number | null)[];
}

export interface CentroidTrack {
  phase: number[];
  offset_pixels: (number | null)[];
}

export interface Periodogram {
  period_days: number[];
  power: number[];
  best_period_days: number;
}

export interface ScoreResponse {
  tic_id: number;
  ephemeris: Ephemeris;
  prob_calibrated: number;
  prob_mean: number;
  prob_std: number;
  per_fold: FoldPrediction[];
  decision_threshold: number;
  centroid: CentroidDiagnostics | null;
  odd_even: OddEvenDiagnostics | null;
  global_view: PhaseView;
  local_view: PhaseView;
  odd_view: PhaseView | null;
  even_view: PhaseView | null;
  centroid_track: CentroidTrack | null;
  periodogram: Periodogram | null;
  duration_check?: DurationDiagnostics | null;
  verdict: string;
  model_version: string;
  n_mc_samples: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string | null;
}

export interface CandidateRow {
  source: "TOI" | "CTOI";
  name: string;
  tic_id: number;
  disposition: string | null;
  tess_mag: number | null;
  ra_deg: number | null;
  dec_deg: number | null;
  epoch_bjd: number | null;
  period_days: number | null;
  duration_hours: number | null;
  depth_ppm: number | null;
  planet_radius_re: number | null;
  planet_snr: number | null;
  teq_k: number | null;
  tsm: number | null;
  esm: number | null;
  predicted_mass_me: number | null;
  predicted_k_ms: number | null;
  stellar_teff_k: number | null;
  stellar_logg: number | null;
  stellar_radius_rsun: number | null;
  stellar_distance_pc: number | null;
  sectors: string | null;
  promoted_to_toi: string | null;
  comments: string | null;
  date_modified: string | null;
}

export interface ReliabilityBin {
  prob_mean: number;
  frac_positive: number;
  count: number;
}

export interface ReliabilityResponse {
  run_id: string;
  n_examples: number;
  bins: ReliabilityBin[];
  ece: number;
  brier: number;
}

export interface CandidatesPage {
  total: number;
  offset: number;
  rows: CandidateRow[];
}

export interface CandidateQuery {
  search?: string;
  disposition?: string;
  source?: string;
  sort_by?: string;
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}
