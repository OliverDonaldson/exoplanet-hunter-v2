import { useEffect, useState } from "react";
import { fetchReliability } from "../api/client";
import type { ReliabilityResponse } from "../api/types";
import { probColor } from "./VettingPanel";

/**
 * Reliability diagram of the promoted model: mean predicted probability vs
 * observed planet fraction, per bin, from the run's own CV test predictions.
 * On the diagonal = "0.9 really means 90%" — the project's central claim.
 */
export default function ReliabilityChart() {
  const [data, setData] = useState<ReliabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchReliability().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <p style={{ fontSize: "0.8rem", opacity: 0.7 }}>Reliability diagram unavailable: {error}</p>;
  if (!data) return null;

  const size = 260;
  const pad = { l: 34, r: 10, t: 10, b: 32 };
  const plot = { w: size - pad.l - pad.r, h: size - pad.t - pad.b };
  const sx = (p: number) => pad.l + p * plot.w;
  const sy = (p: number) => pad.t + (1 - p) * plot.h;
  const maxCount = Math.max(...data.bins.map((b) => b.count));

  return (
    <section style={{ marginTop: "1.5rem" }}>
      <h3 style={{ marginBottom: "0.25rem" }}>Calibration — reliability diagram</h3>
      <p style={{ marginTop: 0, fontSize: "0.8rem", opacity: 0.75 }}>
        Promoted run {data.run_id.slice(0, 8)} · {data.n_examples} CV test predictions ·
        ECE {data.ece.toFixed(3)} · Brier {data.brier.toFixed(3)}
      </p>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxWidth: 320, height: "auto" }}>
        {/* frame + diagonal (perfect calibration) */}
        <rect x={pad.l} y={pad.t} width={plot.w} height={plot.h} fill="none" stroke="#8885" />
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#888" strokeDasharray="4 3" />
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <text x={sx(v)} y={size - 12} fontSize="9" fill="currentColor" opacity={0.6} textAnchor="middle">
              {v}
            </text>
            <text x={pad.l - 6} y={sy(v) + 3} fontSize="9" fill="currentColor" opacity={0.6} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        <text x={sx(0.5)} y={size - 1} fontSize="9" fill="currentColor" opacity={0.75} textAnchor="middle">
          mean predicted probability
        </text>
        <text x={9} y={sy(0.5)} fontSize="9" fill="currentColor" opacity={0.75} textAnchor="middle" transform={`rotate(-90 9 ${sy(0.5)})`}>
          observed planet fraction
        </text>
        {/* bins: dot area encodes bin population */}
        {data.bins.map((b, i) => (
          <circle
            key={i}
            cx={sx(b.prob_mean)}
            cy={sy(b.frac_positive)}
            r={3 + 6 * Math.sqrt(b.count / maxCount)}
            fill={probColor(b.prob_mean)}
            opacity={0.8}
            stroke="#fff6"
          />
        ))}
      </svg>
    </section>
  );
}
