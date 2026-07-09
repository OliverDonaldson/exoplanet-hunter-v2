import { useEffect, useState } from "react";
import { fetchHealth } from "./api/client";
import type { CandidateRow, HealthResponse } from "./api/types";
import CandidatesTable from "./components/CandidatesTable";
import VettingPanel from "./components/VettingPanel";

/**
 * Two-pane vetting console: ranked candidate list left, per-candidate
 * vetting right. Selecting a row scores the target live through
 * /score/{tic_id} — calibrated probability with its ±MC-dropout band and
 * per-fold dots, phase-folded views, centroid and odd/even diagnostics.
 */
export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CandidateRow | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 1500, margin: "2rem auto", padding: "0 1rem" }}>
      <h1 style={{ marginBottom: "0.25rem" }}>Exoplanet Vetting Console</h1>
      <p style={{ marginTop: 0 }}>
        Interactive triage for transit candidates: calibrated probabilities with
        MC-Dropout uncertainty, centroid and odd/even diagnostics, phase-folded views.
      </p>
      {error && <p style={{ color: "#b91c1c" }}>API unreachable: {error}</p>}
      {health && (
        <p style={{ fontSize: "0.85rem", opacity: 0.8 }}>
          API status: <strong>{health.status}</strong>
          {" — "}
          {health.model_loaded
            ? `serving ${health.model_version}`
            : "no model bundle deployed yet (awaiting first V2 training run)"}
        </p>
      )}
      <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 640px", minWidth: 0 }}>
          <CandidatesTable
            onSelect={setSelected}
            selectedKey={selected ? `${selected.source}-${selected.name}` : null}
          />
        </div>
        <div style={{ flex: "1 1 380px", position: "sticky", top: "1rem", maxWidth: 560 }}>
          <VettingPanel candidate={selected} />
        </div>
      </div>
    </main>
  );
}
