# Operating Guide

Plain-language manual for running Exoplanet Hunter V2. For the design
rationale see `architecture.md`; this is the *how*.

## The one rule

**Always `conda activate exoplanet-hunter-v2` first.** The V1 environment
(`exoplanet-hunter`) contains V1's code under the same package name — in
that env, commands silently run last year's pipeline. Your prompt should
say `(exoplanet-hunter-v2)` before anything else.

## What this system is, in one paragraph

NASA's TESS telescope measures the brightness of stars over months. When a
planet crosses ("transits") its star, the star dims by a fraction of a
percent, periodically. This repo turns those raw brightness series into a
ranked, *calibrated* list of "how likely is this candidate a real planet",
and serves it as a website where each candidate can be scored live and
inspected. Calibrated means the probabilities are trustworthy as numbers:
of all candidates scored 0.9, about 90% really are planets — which is what
lets the scores prioritise scarce follow-up telescope time.

## The moving parts

| Piece | What it is | Where |
|---|---|---|
| Candidate catalogue | Every TOI/CTOI the community tracks (~11k), with TSM/ESM follow-up metrics | `data/catalogue/` |
| Label catalogue | Targets with known answers (confirmed planet / false positive) — training ground truth | `data/labels/` |
| Views | Each target's light curve cleaned, flattened, phase-folded into a 2001-bin global + 201-bin local "picture", plus 9 auxiliary numbers | `data/processed/` |
| Model | Five 1D-CNNs (one per CV fold) + a Platt calibrator each | `models/cv/<run_id>/` |
| Registry | A tiny JSON pointing at the *promoted* run — the one being served | `models/registry.json` (in git) |
| API | FastAPI: `/candidates`, `/score/{tic_id}`, `/reliability`, `/healthz` | `api/` |
| Console | React site: sortable catalogue, live vetting pane, reliability diagram | `frontend/` |
| Orchestrator | The one-command loop that refreshes, validates, retrains, promotes | `orchestration/` |

## Where the data actually lives

- **Code** → git (GitHub `exoplanet-hunter-v2`).
- **Data + models** → DVC-tracked; git holds tiny `.dvc` pointer files, the
  bytes live in the Cloudflare R2 bucket. `dvc pull` materialises them,
  `dvc push` uploads changes (`make data-pull` / `make data-push`).
- **Raw FITS light curves** (`data/raw/`, 60+ GB) → *deliberately untracked
  cache*. NASA hosts the originals forever; delete this whenever you need
  disk and it re-downloads on demand.
- **Experiment history** → MLflow, sqlite store. `make mlflow` →
  http://localhost:5001.

Fresh machine: clone repo → create env → `dvc pull` → everything works.

## Everyday commands

```bash
make test          # full fast test suite (pipeline + API)
make validate      # data gates: catalogue schemas, no dead columns
make api           # FastAPI on :8000  (docs at /docs)
make frontend      # console on :5173  (needs the api running)
make mlflow        # experiment UI on :5001
make refresh       # THE loop: refresh → gates → train-if-warranted → publish
```

## The loop, step by step

`make refresh` (or `python orchestration/flows/refresh_pipeline.py`) runs:

1. **Download** the latest TOI/CTOI tables from ExoFOP.
2. **Ingest** them into the candidate catalogue (with TSM/ESM computed).
3. **Rebuild** the label catalogue from NASA's archive, keeping the old
   one aside.
4. **Gates**: schema checks + the *leakage guard* — any target whose label
   changed since last time is reported and quarantined (it joins the
   "since-confirmed" holdout; it never quietly enters training).
5. **Trigger**: train only if ≥25 genuinely new labelled targets arrived
   (`--force-train` overrides; `--no-train` stops here).
6. **Build + shard + train**: preprocess everything, write TFRecord
   shards, run 5-fold CV training (`--data-config full` = whole pool).
7. **Promotion gate**: the new model must beat the registered champion's
   CV ROC-AUC *without* degrading calibration — neither Brier nor ECE may
   worsen beyond tolerance — or it is rejected and the registry stays put.
8. **Publish**: version everything with DVC and push to R2.

A promoted model is served automatically the next time the API starts.

## Scoring a single target by hand

```bash
curl "localhost:8000/score/307210830"                      # BLS finds the period (~1 min)
curl "localhost:8000/score/307210830?period_days=3.55&t0_btjd=1400.2&duration_hours=2.4"
```
Or click any row in the console. Under the hood: fetch light curve
(cache/MAST) → same preprocessing as training → 5 models × 50
dropout-active forward passes → calibrated probability ± uncertainty →
centroid & odd/even checks → plain-language verdict.

## Running the refresh on a schedule

The refresh + retrain loop needs this Mac (data, GPU), so scheduling is a
launchd job, not a cloud cron:

```bash
cp scripts-dev/com.exoplanet-hunter.refresh.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.exoplanet-hunter.refresh.plist
```

Runs Saturdays 09:00 (edit StartCalendarInterval to taste; the Mac must be
awake). Logs append to `outputs/refresh-cron.log`. Set `NOTIFY_WEBHOOK_URL`
(uncomment in the plist) to get a Discord/Slack ping with the promotion
verdict at the end of each run. Unload with `launchctl unload …`.

## Reading the numbers

- **prob_calibrated ± prob_std** — the headline. The ± band is real model
  uncertainty (MC-Dropout spread + fold disagreement); a wide band means
  "needs follow-up", however high the mean.
- **five fold dots** — if they scatter widely, the ensemble disagrees.
- **centroid > 3σ** — the dip may come from a background eclipsing binary,
  not the target star. Discount the probability.
- **odd/even Δ > 3σ** — alternating depths: classic eclipsing-binary sign.
- **Reliability diagram (ECE)** — dots on the diagonal = "0.9 means 90%".
  Current promoted run: ECE 0.031.

## When something fails

- **A gate FAILs** — read its message; that's the system refusing bad data
  *before* you burn compute. Fix the data problem, re-run.
- **`zsh: command not found: dvc/prefect/...`** — wrong conda env.
- **"file may be corrupt / interrupted download"** — self-heals: the
  downloader evicts the bad file and retries. Rerun if it happened
  mid-build; nothing is poisoned.
- **Promotion REJECTED** — not a failure. The challenger lost; the
  champion keeps serving. The run is still in MLflow for analysis.

## Safe to delete / never delete

Safe: `data/raw*` (re-downloads), `mlruns/`, `outputs/`, `results/`,
anything `dvc pull` can restore. Never: `models/registry.json`, `.dvc`
pointer files, `.dvc/config.local` (your R2 keys, exists only on this
machine — losing it means minting a new R2 token).

## Docker?

Not needed for anything local. The Dockerfiles exist for the *deployed*
API (Fly.io/Render, pending) and the future GPU-burst training image.
Keep Docker Desktop closed until then.
