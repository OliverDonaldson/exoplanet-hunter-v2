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
}

export interface PhaseView {
  phase: number[];
  flux: (number | null)[];
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
  stellar_teff_k: number | null;
  stellar_logg: number | null;
  stellar_radius_rsun: number | null;
  stellar_distance_pc: number | null;
  sectors: string | null;
  promoted_to_toi: string | null;
  comments: string | null;
  date_modified: string | null;
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
