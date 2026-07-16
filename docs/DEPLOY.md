# Deploying the serving API

The React console is a static site on Render. The API — a 5-fold TensorFlow
ensemble — needs a ~2 GB always-warmable box, so it goes on Fly.io. This is
the step that clears the console's "API unreachable" banner.

## 1. Build and test the image locally

The daemon needs outbound network to Docker Hub + PyPI. If pulls hang right
after Docker Desktop starts, give its VM a minute to bring networking up and
retry.

```bash
cd /Users/ollie/Project/v2
docker build -f docker/api.Dockerfile -t exoplanet-hunter-api .
```

Smoke-test without R2 credentials by mounting the artefacts directly and
skipping the pull (without credentials, botocore's credential-chain probing
hangs rather than failing fast):

```bash
docker run --rm -p 8010:8000 -e SKIP_DVC_PULL=1 \
  -v "$(pwd)/models:/srv/models:ro" \
  -v "$(pwd)/data/catalogue:/srv/data/catalogue:ro" \
  exoplanet-hunter-api &
# wait ~60s for TF import + model load, then:
curl -s localhost:8010/healthz      # {"status":"ok","model_loaded":true,...}
curl -s localhost:8010/reliability  # proves the model artefacts read back
```

## 2. Deploy to Fly.io

`fly.toml` at the repo root is ready — edit `app` to a unique name first.
`fly deploy` builds on Fly's **remote builder**, so a broken local Docker
network doesn't block the deploy.

```bash
brew install flyctl          # or: curl -L https://fly.io/install.sh | sh
fly auth login

# R2 credentials for the boot-time dvc pull. The env-var NAMES must be
# exactly AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (boto3's names), NOT
# the field names used inside .dvc/config.local. Values only — no quotes.
fly secrets set AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

fly deploy --remote-only     # remote builder: don't push a 3.8 GB image from home
fly logs                     # watch: entrypoint dvc pull -> uvicorn
curl -s https://<app>.fly.dev/healthz
```

Notes from the first deployment:

- A deploy replaces the machine and kills in-flight scores — don't push to
  `main` mid-demo (the GitHub connection auto-deploys).
- `auto_stop_machines = "suspend"` snapshots RAM on idle: the machine
  resumes in seconds with the TF ensemble still loaded, and the downloaded
  FITS cache survives until the next deploy.
- The ensemble preloads in a background thread at boot; the console's
  page-load requests wake the machine, so the model is typically ready by
  the time a target is clicked.

## 3. Point the console at the API

- **Render** → console service → env var `VITE_API_BASE = https://<app>.fly.dev`
  → redeploy (Vite bakes it in at build time).
- The API already reads `FRONTEND_ORIGIN` for CORS; keep it matching the
  console's Render URL (set in `fly.toml`).

## Cost (Sydney, shared-cpu-1x @ 2 GB)

256 MB — the Fly calculator's default — cannot import TensorFlow; 2 GB is the
real floor (1 GB risks OOM on model load).

| Mode | Monthly | Notes |
|---|---|---|
| Suspend-on-idle (current config) | **~$1-4** | resumes in seconds with the model in RAM |
| Always-on (`min_machines_running = 1`) | **~$12.76** | no wake-ups at all |

Egress is negligible (small JSON responses, $0.04/GB). Optional upgrades,
each a one-liner: a 3 GB volume (`fly volumes create raw_cache -s 3` + a
`[mounts]` block to `/srv/data/raw`, $0.45/mo) makes the FITS cache survive
deploys; `shared-cpu-2x` (~2x inference) costs ~$0.9/mo more under
suspend-on-idle usage. Fly's "run 2+ machines" banner is production-HA
advice — for a single-user demo it doubles cost for redundancy, and the
per-machine FITS cache works best on one machine.
