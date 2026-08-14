# Design console

A self-contained design prototype of the Exoplanet Hunter vetting console:
Mission, Catalogue, Vetting, Model Performance and Upload, built as one HTML
file with no network dependencies.

This is **not** the shipping frontend (that is `frontend/src`). It is the
reference the shipping frontend is meant to be built against — the design
system, the page structures, the copy, and the interaction states, in a form
you can open in a browser and click through.

All data is mock data. The numbers are shaped to match the real pipeline's
contracts and orders of magnitude, but nothing here talks to the API.

## Build

```bash
cd frontend && npm install          # provides animejs@4
cd design-console && python3 build.py
open dist/preview.html
```

`build.py` inlines the anime.js ESM bundle from `../node_modules/animejs` and
base64-inlines the three woff2 subsets in `assets/fonts/`, then concatenates
`src/app.*.js` in a fixed order into a single `<script type="module">`.

Two outputs:

| file | purpose |
| --- | --- |
| `dist/exoplanet-hunter.html` | body-only; what gets published as the Claude artifact |
| `dist/preview.html` | same content wrapped in `<html>`/`<body>` for opening from disk |

The fonts are vendored rather than linked because the artifact host applies a
strict CSP that blocks external font hosts — a `<link>` to Google Fonts would
silently fall back to system fonts and the design would look wrong.

## Source layout

| file | contents |
| --- | --- |
| `src/shell.html` | design tokens, all CSS, static markup (nav, ticker, boot overlay) |
| `src/app.data.js` | candidate data, branch/fold/diagnostic/follow-up models, chart renderer, router |
| `src/app.health.js` | the `/healthz` state machine behind the service-status panel |
| `src/app.home.js` | Mission (home) and Catalogue |
| `src/app.pages.js` | Vetting, Model Performance, Upload |
| `src/app.boot.js` | boot preloader, then the initial `route()` |

## What the design asserts

Points worth carrying into the real console, because each one is a claim about
how the product should behave:

- **Branch evidence is a first-class view.** The eleven input views each get a
  signed contribution and a plain-English reading. Contributions are
  attributions on the probability scale, summing from the catalogue mean to the
  calibrated score, so the arithmetic on screen is checkable.
- **Diagnostics have three states, not two.** pass / fail / **not measured**.
  About 7% of TESS rows carry no Data Validation report; rendering an unmeasured
  flag as a green tick would reintroduce the exact defect the presence-mask
  machinery exists to prevent.
- **No pooled headline metric.** Per-mission cards driven by the served model's
  own evaluation list, with out-of-fold and zero-shot labelled differently and
  never compared. The card count comes from the data, not from a fixed layout.
- **Every headline number carries its interval.** The project measured its noise
  floor (AUC ±0.0070, shortlist recall ±0.0337); a bare figure invites reading
  noise as improvement.
- **Recall @ 1% FPR is a headline**, not a footnote — it is the promotion
  criterion and the thing "would this reach the shortlist" actually means.
- **Follow-up priority is surfaced.** TSM/ESM against Kempton (2018) thresholds,
  because ranking candidates for telescope time is the product's stated purpose.
- **Observation baseline is visible** wherever a score is, since long baselines
  inflate detectability and scores are not comparable across very different ones.
- **The console is never gated on a warm model.** See below.

## Service status panel

`src/app.health.js`. Polls `/healthz` and renders four states. The copy matters
because three of the four are fast:

| state | condition | UI |
| --- | --- | --- |
| Connecting | request in flight, < 2 s | small spinner — most visits never leave this |
| Waking | request in flight, > 2 s | "Waking the observatory…" — the suspend-resume path |
| Warming model | 200 with `ensemble_ready: false` | determinate bar against `uptime_s / 90` |
| Ready | `ensemble_ready: true` | teal dot, the served `model_version` |

A fifth state, Degraded, covers `status: "degraded"` (no promoted run in the
registry): scoring unavailable, catalogue still browsable.

Three rules the panel follows:

1. **Browsing is never gated.** `/candidates` and `/reliability` are parquet
   reads that answer in ~0.8 s whatever the ensemble is doing. Only Score waits.
2. **Name the served model once known**, so the console cannot quietly drift
   from what is actually deployed.
3. **Determinate bar only for the ~90 s ensemble load.** A bar that completes in
   2 s reads as broken; a spinner that runs for 90 s reads as hung.

### Measured timings this is designed around

| state | TTFB | what happened |
| --- | --- | --- |
| resume from suspend | 2.0 s | RAM snapshot restored, TF ensemble already in memory |
| warm | 0.14 s | steady state |
| `/candidates` | 0.78 s | parquet read |
| `/reliability` | 0.87 s | parquet read |

`fly.toml` uses `auto_stop_machines = "suspend"`, which snapshots RAM rather
than killing the process, so the common idle path is a ~2 s resume with the
model still loaded — not a cold start. A true cold start (stopped machine, or
post-deploy) runs `dvc pull` from R2, then FastAPI starts, then a background
thread imports TensorFlow and five folds: 60–180 s, and the Fly health check
allows a 180 s grace period.

### ⚠ Contract gap

The deployed `GET /healthz` currently returns only:

```json
{ "status": "ok", "model_loaded": true, "model_version": "cnn_dualview-cv-ca906040" }
```

The **Warming** state needs two fields that do not exist yet:

- `ensemble_ready: bool` — `model_loaded` is not a substitute; it only means a
  promoted run exists in `registry.json`, and the TF ensemble is loaded lazily
  on the first `/score` request.
- `uptime_s: number` — to drive the determinate bar against the ~90 s load.

Until those land, the client degrades to connecting → waking → ready, which is
correct for the suspend-resume path but skips cold-start progress entirely.
`phaseFor()` handles the absence explicitly; no API change is assumed.

### Wiring it to the real endpoint

The prototype runs a scripted clock so the states can be seen. To point it at
the live service, set `HEALTH.live = true` in `src/app.health.js` — `fetchHealth()`
is already written and sits next to the simulation. The prototype plays one
honest cold start per session (`sessionStorage`), then behaves like a warm
service, which is what the real thing does; the ↻ button replays the sequence.

## Boot preloader

`src/app.boot.js`. An instrument dial in the shape of the anime.js hero
animation — glowing segmented outer ring, dense rotating tick ring, glass disc
with a specular highlight, thin concentric arcs drawn on with a stagger, and a
particle field — rendered in our palette, with our motif at the centre (a
planet transiting its star, tracing the light-curve dip).

anime.js features used: `svg.createDrawable` with `draw` for the ring segments,
arcs and the transit trace; `svg.createMotionPath` for the transiting planet;
`stagger` with a `grid` for the particle field; `composition: 'blend'` so the
per-particle drifts add rather than replace; `text.scrambleText` for the status
readout.

Skipped entirely under `prefers-reduced-motion`. A 9 s hard timeout guarantees
the console is never left behind the overlay if the animation stalls.
