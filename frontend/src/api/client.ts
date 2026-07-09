import type {
  CandidateQuery,
  CandidatesPage,
  HealthResponse,
  ReliabilityResponse,
  ScoreResponse,
} from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `${resp.status} ${resp.statusText}`);
  }
  return resp.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return get<HealthResponse>("/healthz");
}

export function fetchReliability(): Promise<ReliabilityResponse> {
  return get<ReliabilityResponse>("/reliability");
}

function candidateParams(query: CandidateQuery): URLSearchParams {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  return params;
}

export function fetchCandidates(query: CandidateQuery = {}): Promise<CandidatesPage> {
  const params = candidateParams(query);
  const qs = params.size ? `?${params}` : "";
  return get<CandidatesPage>(`/candidates${qs}`);
}

/** URL for the CSV export matching the current filters (used as a download link). */
export function candidatesCsvUrl(query: CandidateQuery = {}): string {
  const params = candidateParams({
    search: query.search,
    disposition: query.disposition,
    source: query.source,
  });
  const qs = params.size ? `?${params}` : "";
  return `${BASE}/candidates.csv${qs}`;
}

export function fetchScore(
  ticId: number,
  opts: { periodDays?: number; t0Btjd?: number; durationHours?: number; nMc?: number } = {},
): Promise<ScoreResponse> {
  const params = new URLSearchParams();
  if (opts.periodDays !== undefined) params.set("period_days", String(opts.periodDays));
  if (opts.t0Btjd !== undefined) params.set("t0_btjd", String(opts.t0Btjd));
  if (opts.durationHours !== undefined) params.set("duration_hours", String(opts.durationHours));
  if (opts.nMc !== undefined) params.set("n_mc", String(opts.nMc));
  const qs = params.size ? `?${params}` : "";
  return get<ScoreResponse>(`/score/${ticId}${qs}`);
}
