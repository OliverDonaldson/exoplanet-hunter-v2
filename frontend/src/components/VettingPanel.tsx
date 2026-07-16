import { useEffect, useState } from "react";
import { fetchScore } from "../api/client";
import type { CandidateRow, ScoreResponse } from "../api/types";

/** Cool→warm scale shared with the list: blue (unlikely) → red (planet). */
export function probColor(p: number): string {
  return `hsl(${Math.round(220 * (1 - p))}, 72%, 44%)`;
}

function PhaseChart({
  title,
  phase,
  flux,
  height = 150,
}: {
  title: string;
  phase: number[];
  flux: (number | null)[];
  height?: number;
}) {
  const width = 460;
  const pad = { l: 8, r: 8, t: 6, b: 18 };
  const pts: { x: number; y: number }[] = [];
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < phase.length; i++) {
    const f = flux[i];
    if (f === null || !isFinite(f)) continue;
    if (f < yMin) yMin = f;
    if (f > yMax) yMax = f;
    pts.push({ x: phase[i], y: f });
  }
  if (pts.length === 0) return <p>{title}: no data</p>;
  const xMin = phase[0];
  const xMax = phase[phase.length - 1];
  const ySpan = yMax - yMin || 1;
  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin)) * (width - pad.l - pad.r);
  const sy = (y: number) =>
    pad.t + (1 - (y - yMin) / ySpan) * (height - pad.t - pad.b);
  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ fontSize: "0.8rem", opacity: 0.75 }}>{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto" }}>
        <line x1={sx(0)} y1={pad.t} x2={sx(0)} y2={height - pad.b} stroke="#8884" />
        {pts.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={1.1} fill="currentColor" opacity={0.65} />
        ))}
        <text x={sx(xMin)} y={height - 4} fontSize="9" fill="currentColor" opacity={0.6}>
          {xMin.toFixed(2)}
        </text>
        <text x={sx(0)} y={height - 4} fontSize="9" fill="currentColor" opacity={0.6} textAnchor="middle">
          phase 0
        </text>
        <text x={sx(xMax)} y={height - 4} fontSize="9" fill="currentColor" opacity={0.6} textAnchor="end">
          {xMax.toFixed(2)}
        </text>
      </svg>
    </figure>
  );
}

function ProbabilityBar({ score }: { score: ScoreResponse }) {
  const width = 460;
  const h = 64;
  const track = { x: 10, y: 34, w: width - 20, h: 8 };
  const sx = (p: number) => track.x + Math.min(Math.max(p, 0), 1) * track.w;
  const p = score.prob_calibrated;
  const lo = Math.max(0, p - score.prob_std);
  const hi = Math.min(1, p + score.prob_std);
  return (
    <svg viewBox={`0 0 ${width} ${h}`} style={{ width: "100%", height: "auto" }}>
      <rect {...track} width={track.w} height={track.h} rx={4} fill="#8883" />
      {/* ±MC-dropout band */}
      <rect
        x={sx(lo)}
        y={track.y}
        width={sx(hi) - sx(lo)}
        height={track.h}
        rx={4}
        fill={probColor(p)}
        opacity={0.35}
      />
      {/* decision threshold */}
      <line x1={sx(score.decision_threshold)} y1={track.y - 8} x2={sx(score.decision_threshold)} y2={track.y + track.h + 8} stroke="#888" strokeDasharray="3 2" />
      <text x={sx(score.decision_threshold)} y={track.y + track.h + 20} fontSize="9" fill="currentColor" opacity={0.6} textAnchor="middle">
        thr {score.decision_threshold.toFixed(2)}
      </text>
      {/* five per-fold dots — the ensemble's spread, visible */}
      {score.per_fold.map((f) => (
        <circle key={f.fold} cx={sx(f.prob)} cy={track.y + track.h / 2} r={4} fill={probColor(f.prob)} stroke="#fff8" />
      ))}
      {/* ensemble mean */}
      <line x1={sx(p)} y1={track.y - 6} x2={sx(p)} y2={track.y + track.h + 6} stroke={probColor(p)} strokeWidth={2.5} />
      <text x={sx(p)} y={track.y - 12} fontSize="13" fontWeight="bold" fill={probColor(p)} textAnchor="middle">
        {p.toFixed(2)} ± {score.prob_std.toFixed(2)}
      </text>
    </svg>
  );
}

function Readout({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "0.25rem 0", borderBottom: "1px solid #8882", fontSize: "0.85rem" }}>
      <span style={{ opacity: 0.75 }}>{label}</span>
      <span style={{ fontWeight: 600, color: warn ? "#b91c1c" : "inherit" }}>{value}</span>
    </div>
  );
}

export default function VettingPanel({ candidate }: { candidate: CandidateRow | null }) {
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usedBls, setUsedBls] = useState(false);

  useEffect(() => {
    if (!candidate) return;
    const hasEphemeris =
      candidate.period_days != null &&
      candidate.period_days > 0 &&
      candidate.epoch_bjd != null &&
      candidate.duration_hours != null &&
      candidate.duration_hours > 0;
    setUsedBls(!hasEphemeris);
    setBusy(true);
    setScore(null);
    setError(null);
    let cancelled = false;
    fetchScore(
      candidate.tic_id,
      hasEphemeris
        ? {
            periodDays: candidate.period_days!,
            // Catalogue epochs are full BJD; the API speaks BTJD (BJD − 2457000).
            t0Btjd:
              candidate.epoch_bjd! > 2_440_000
                ? candidate.epoch_bjd! - 2_457_000
                : candidate.epoch_bjd!,
            durationHours: candidate.duration_hours!,
          }
        : {},
    )
      .then((s) => {
        if (!cancelled) setScore(s);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [candidate]);

  if (!candidate) {
    return (
      <aside>
        <h2>Vetting</h2>
        <p style={{ opacity: 0.7 }}>Select a candidate from the list to score it live.</p>
      </aside>
    );
  }

  return (
    <aside>
      <h2 style={{ marginBottom: "0.25rem" }}>{candidate.name}</h2>
      <p style={{ marginTop: 0, fontSize: "0.85rem", opacity: 0.75 }}>
        TIC {candidate.tic_id} · {candidate.source}
        {candidate.disposition ? ` · TFOPWG ${candidate.disposition}` : ""}
      </p>

      {busy && (
        <p style={{ fontStyle: "italic", opacity: 0.8 }}>
          Scoring live — light curve → preprocessing → 5-fold ensemble with MC-Dropout…
          {usedBls
            ? " (no catalogue ephemeris: running a BLS period search too — allow a few minutes)"
            : " (seconds when warm; up to a few minutes if the server just woke or the light curve isn't cached yet)"}
        </p>
      )}
      {error && <p style={{ color: "#b91c1c" }}>Scoring failed: {error}</p>}

      {score && (
        <>
          <ProbabilityBar score={score} />
          <p style={{ fontSize: "0.9rem", borderLeft: `3px solid ${probColor(score.prob_calibrated)}`, paddingLeft: "0.6rem" }}>
            {score.verdict}
          </p>
          <PhaseChart title="Global view (full phase)" phase={score.global_view.phase} flux={score.global_view.flux} />
          <PhaseChart title="Local view (±3 transit durations)" phase={score.local_view.phase} flux={score.local_view.flux} />
          <div style={{ marginTop: "0.5rem" }}>
            <Readout
              label="Ephemeris"
              value={`P=${score.ephemeris.period_days.toFixed(3)} d · dur=${(score.ephemeris.duration_days * 24).toFixed(2)} h · ${score.ephemeris.source}`}
            />
            {score.centroid && (
              <Readout
                label="Centroid shift"
                value={`${score.centroid.centroid_snr.toFixed(2)}σ vs ${score.centroid.beb_threshold_sigma.toFixed(0)}σ BEB threshold`}
                warn={score.centroid.suspicious}
              />
            )}
            {score.odd_even && (
              <Readout
                label="Odd/even depth"
                value={`${score.odd_even.odd_depth_ppm.toFixed(0)} / ${score.odd_even.even_depth_ppm.toFixed(0)} ppm (Δ ${score.odd_even.depth_diff_sigma.toFixed(1)}σ)`}
                warn={score.odd_even.depth_diff_sigma > 3}
              />
            )}
            <Readout label="Raw ensemble mean" value={score.prob_mean.toFixed(3)} />
            <Readout label="Model" value={`${score.model_version} · ${score.n_mc_samples} MC samples`} />
          </div>
        </>
      )}
    </aside>
  );
}
