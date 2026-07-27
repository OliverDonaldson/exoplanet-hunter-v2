# Console

React + Vite vetting console. Live at
`https://exoplanet-hunter-console.onrender.com`.

```bash
make frontend     # Vite dev server on :5173
npm run build     # what Render deploys
```

Talks to the API via `VITE_API_BASE` (defaults to the local `:8000`).

## Layout

| file | what it is |
|---|---|
| `src/App.tsx` | shell and routing |
| `src/api/client.ts` | every call to the API |
| `src/api/types.ts` | **the pinned wire contract** — mirrors `api/app/schemas.py` |
| `src/components/CandidatesTable.tsx` | sortable catalogue with the follow-up columns |
| `src/components/VettingPanel.tsx` | per-target vetting: phase views, diagnostics, verdict |
| `src/components/ReliabilityChart.tsx` | calibration curve for the promoted run |

## Two things that will bite

**`types.ts` and `api/app/schemas.py` change together.** An optional field
added server-side without its TypeScript is invisible here — the panel just
stops rendering that piece.

**The console converts BJD to BTJD before calling `/score`**
(`VettingPanel.tsx`, `epoch_bjd - 2_457_000`). The catalogue stores full BJD;
the API takes BTJD. Three CTOIs carry malformed epochs that convert to ~2.2e7
BTJD, which is why the API's `t0` bound is 1e8 and not something tighter — it
exists to reject `inf`, not to filter data quality.

## Status

This is the functional console, not the final design. The Mission Control
redesign is Stage 5 of [the roadmap](../docs/roadmap.md) — deliberately last,
so it lands on top of the per-branch vetting evidence the model rebuild
produces rather than being rebuilt around it twice.
