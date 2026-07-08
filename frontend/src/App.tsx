import { useEffect, useState } from "react";
import { fetchHealth } from "./api/client";
import type { HealthResponse } from "./api/types";
import CandidatesTable from "./components/CandidatesTable";

/**
 * Vetting-console shell — the two-pane layout lands in `feat/dashboard`:
 * left, the ranked candidate list read from scores.parquet via the API;
 * right, the per-candidate vetting view fed by /score/{tic_id} with the
 * calibrated probability, its ±MC-dropout band, and the five per-fold dots.
 * For now it proves the frontend ↔ API wiring end to end via /healthz.
 */
export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 1080, margin: "3rem auto", padding: "0 1rem" }}>
      <h1>Exoplanet Vetting Console</h1>
      <p>
        Interactive triage for transit candidates: calibrated probabilities with
        MC-Dropout uncertainty, centroid and odd/even diagnostics, phase-folded views.
      </p>
      {error && <p style={{ color: "#b91c1c" }}>API unreachable: {error}</p>}
      {health && (
        <p>
          API status: <strong>{health.status}</strong>
          {" — "}
          {health.model_loaded
            ? `model ${health.model_version} loaded`
            : "no model bundle deployed yet (awaiting first V2 training run)"}
        </p>
      )}
      <CandidatesTable />
    </main>
  );
}
