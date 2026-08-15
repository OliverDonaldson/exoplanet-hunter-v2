# Getting Started

How to run Exoplanet Hunter. For what it produces see
[overview.md](overview.md); for why it is shaped this way,
[model_pipeline.md](model_pipeline.md).

## 1. The one rule

**Activate the environment first.**

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate exoplanet-hunter-v2
```

The V1 environment contains V1's code under the same package name. In that
environment, commands silently run last year's pipeline. The prompt should read
`(exoplanet-hunter-v2)` before anything else.

## 2. Setting up a fresh machine

```bash
git clone <repo> && cd v2
conda env create -f environment.yml
conda activate exoplanet-hunter-v2
dvc pull                     # or: make data-pull
```

Code lives in git. Data and models are DVC-tracked — git holds small `.dvc`
pointer files and the bytes live in a Cloudflare R2 bucket. Raw FITS light
curves are a deliberately untracked cache; the mission archives host the
originals permanently, so the cache can be deleted whenever disk is needed and
it re-downloads on demand.

## 3. The moving parts

| Piece | What it is | Where |
|---|---|---|
| Candidate catalogue | Every TOI/CTOI tracked, with follow-up metrics | `data/catalogue/` |
| Label catalogue | Targets with known dispositions — training ground truth | `data/labels/` |
| Processed views | Cleaned, flattened, phase-folded inputs | `data/processed/` |
| Models | One CNN per CV fold, each with a Platt calibrator | `models/cv/<run_id>/` |
| Registry | JSON pointing at the promoted run — the one being served | `models/registry.json` |
| API | FastAPI: `/candidates`, `/score/{tic_id}`, `/reliability`, `/healthz` | `api/` |
| Console | React: catalogue table, vetting pane, reliability diagram | `frontend/` |
| Orchestrator | The loop that refreshes, validates, retrains, promotes | `orchestration/` |

## 4. Everyday commands

```bash
make test          # full fast test suite (pipeline + API)
make validate      # data gates: catalogue schemas, no dead columns
make api           # FastAPI on :8000  (docs at /docs)
make frontend      # console on :5173  (needs the api running)
make mlflow        # experiment UI on :5001
make refresh       # the loop: refresh -> gates -> train-if-warranted -> publish
```

Run the test suite **one process per file**. Repeated heavy fixtures in a single
process slow without bound.

## 5. The refresh loop

`make refresh` runs, in order:

1. **Download** the latest TOI/CTOI tables from ExoFOP.
2. **Ingest** them into the candidate catalogue.
3. **Rebuild** the label catalogue from NASA's archive, keeping the previous one
   aside as `labels.previous.parquet`.
4. **Gates** — schema checks plus the leakage guard: any target whose label
   changed since last time is quarantined into the prospective holdout rather
   than quietly entering training.
5. **Trigger** — train only if at least 25 genuinely new labelled targets
   arrived. `--force-train` overrides; `--no-train` stops here.
6. **Build, shard, train** — preprocess, write TFRecord shards, run 5-fold CV.
7. **Promotion gate** — the challenger must beat the champion's CV ROC-AUC
   without degrading Brier or ECE, or the registry stays put.
8. **Publish** — version everything with DVC and push to R2.

A promoted model is served the next time the API starts.

## 6. Scoring a single target

```bash
curl "localhost:8000/score/307210830"
```

BLS finds the period, which takes about a minute. Supplying the ephemeris skips
that:

```bash
curl "localhost:8000/score/307210830?period_days=3.55&t0_btjd=1400.2&duration_hours=2.4"
```

Or click any row in the console.

## 7. Statistical validation

The model reads the light curve but never the pixels, so it cannot separate a
transit on the target from an eclipse on a nearby star in the same aperture.
TRICERATOPS computes a false-positive probability (FPP) and nearby-FPP (NFPP)
from the pixels and the surrounding star field. It is slow and network-bound, so
it runs offline over a scored shortlist rather than inside `/score`.

```bash
pip install -e 'pipeline[validation]'
python scripts/validate_candidates.py \
    --shortlist results/candidates_scored.parquet --top 20 \
    --insecure-trilegal \
    --out results/candidates_validated.csv
```

`--insecure-trilegal` skips SSL verification for the background-star query to
the TRILEGAL server, whose certificate chain is broken and unverifiable even
with a current CA bundle. The query is a public, unauthenticated star-count
lookup sending only coordinates. Omitting the flag keeps verification on, and
the request will then fail against that server. A pre-downloaded TRILEGAL table
can be passed instead as a fully verified alternative.

Thresholds: **validated** = FPP < 0.015 and NFPP < 1e-3; **likely planet** =
FPP < 0.5 and NFPP < 1e-3; **likely nearby FP** = NFPP > 0.1. FPP is unreliable
below transit S/N ≈ 15, reported per row. SAP flux is used deliberately, not
PDCSAP — PDC removes the nearby-star contamination NFPP exists to detect.

## 8. Scheduling the refresh

The loop needs this machine's data and GPU, so it is a launchd job rather than a
cloud cron.

```bash
cp scripts-dev/com.exoplanet-hunter.refresh.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.exoplanet-hunter.refresh.plist
```

Runs Saturdays at 09:00; the machine must be awake. Logs append to
`outputs/refresh-cron.log`. Setting `NOTIFY_WEBHOOK_URL` posts the promotion
verdict at the end of each run. Unload with `launchctl unload`.

## 9. Deploying

The console is a static site on Render. The API is a multi-fold TensorFlow
ensemble needing roughly 2 GB, so it runs on Fly.io.

### 9.1 Build and test the image locally

```bash
docker build -f docker/api.Dockerfile -t exoplanet-hunter-api .
```

Smoke-test without R2 credentials by mounting the artefacts and skipping the
pull — without credentials, botocore's credential-chain probing hangs rather
than failing fast:

```bash
docker run --rm -p 8010:8000 -e SKIP_DVC_PULL=1 \
  -v "$(pwd)/models:/srv/models:ro" \
  -v "$(pwd)/data/catalogue:/srv/data/catalogue:ro" \
  exoplanet-hunter-api
```

Allow about 60 s for the TensorFlow import and model load, then:

```bash
curl -s localhost:8010/healthz
```

### 9.2 Deploy

```bash
fly auth login
fly secrets set AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
fly deploy --remote-only
```

The secret **names** must be exactly `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` — boto3's names, not the field names used inside
`.dvc/config.local`. Values only, no quotes.

`--remote-only` builds on Fly's builder, so a broken local Docker network does
not block the deploy and a multi-gigabyte image is never pushed from home.

Notes from deployments so far:

- A deploy replaces the machine and kills in-flight scores. Do not push mid-demo
  while auto-deploy is connected.
- `auto_stop_machines = "suspend"` snapshots RAM on idle, so the machine resumes
  in seconds with the ensemble still loaded.
- The ensemble preloads in a background thread at boot; the console's page-load
  requests wake the machine, so the model is usually ready by the time a score
  is requested.

## 10. Reading the numbers

- **`prob_calibrated` ± `prob_std`** — the headline. The band is real model
  uncertainty; a wide band means "needs follow-up" however high the mean.
- **Per-fold values** — if they scatter widely, the ensemble disagrees.
- **Centroid > 3σ** — the dip may come from a background eclipsing binary.
- **Odd/even Δ > 3σ** — alternating depths, the classic eclipsing-binary sign.
- **Reliability diagram** — points on the diagonal mean a stated 0.9 really is
  about 90%.

When something goes wrong, see [troubleshooting.md](troubleshooting.md).
