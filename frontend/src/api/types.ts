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
