# Deploying the serving API

The React console is a static site on Render. The API — a 5-fold TensorFlow
ensemble — needs a ~2 GB always-warmable box, so it goes on Fly.io. This is
the step that clears the console's "API unreachable" banner.

## 1. Build and test the image locally

The daemon needs outbound network to Docker Hub + PyPI; run these in your own
terminal (a sandboxed session may block the Docker VM's egress).

```bash
cd /Users/ollie/Project/v2
docker build -f docker/api.Dockerfile -t exoplanet-hunter-api .
```

Smoke-test without R2 credentials by mounting the local DVC cache, so the
entrypoint's `dvc pull` resolves from cache instead of the remote:

```bash
docker run --rm -p 8010:8000 \
  -v "$(pwd)/.dvc/cache:/srv/.dvc/cache:ro" \
  exoplanet-hunter-api &
# wait ~30s for TF import + model load, then:
curl -s localhost:8010/healthz      # {"status":"ok","model_loaded":true,...}
curl -s localhost:8010/reliability  # proves the pulled model artefacts read back
```

If `pip install` fails on a source build (no manylinux wheel for some dep),
add `build-essential` to the `apt-get install` line in the Dockerfile.

## 2. Deploy to Fly.io

`fly.toml` at the repo root is ready — edit `app` to a unique name first.
`fly deploy` builds on Fly's **remote builder**, so a broken local Docker
network doesn't block the deploy.

```bash
brew install flyctl          # or: curl -L https://fly.io/install.sh | sh
fly auth login

# R2 credentials for the boot-time dvc pull (from .dvc/config.local).
# Set them yourself — never commit or echo them.
fly secrets set AWS_ACCESS_KEY_ID=<r2-access-key> AWS_SECRET_ACCESS_KEY=<r2-secret>

fly deploy
fly logs                     # watch: entrypoint dvc pull -> uvicorn
curl -s https://<app>.fly.dev/healthz
```

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
| Scale-to-zero (`min_machines_running = 0`) | **~$1-4** | idle most of the month; ~20-30s cold start on first hit after idle |
| Always-on | **~$12.76** | snappy every request |

Egress is negligible (small JSON responses, $0.04/GB). No Postgres, no
volume, no dedicated IP needed — the MAST download cache is regenerable, so
skip a Fly volume unless cold-start re-downloads become annoying (then a
1-2 GB volume at $0.15/GB/mo persists `/srv/data/raw`).
