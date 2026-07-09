# Infra notes

## Cloudflare R2 (zero-egress object store)
Bucket `exoplanet-hunter-v2` is the DVC remote (endpoint committed in
`.dvc/config`; the account ID in the URL is not a secret). DVC lays out its
own content-addressed structure inside the bucket.

Credential setup (once per machine — keys live in `.dvc/config.local`,
which DVC gitignores):

1. Cloudflare dashboard → R2 → create bucket `exoplanet-hunter-v2`
   (or `dvc remote modify r2 url s3://<your-name>` to match).
2. R2 Overview → **Manage R2 API Tokens** → Create API Token →
   permission **Object Read & Write**, scoped to that bucket → copy the
   Access Key ID and Secret Access Key (shown once).
3. ```
   dvc remote modify --local r2 access_key_id     <ACCESS_KEY_ID>
   dvc remote modify --local r2 secret_access_key <SECRET_ACCESS_KEY>
   dvc push
   ```
For CI / the GPU burst, the same two values go in GitHub Secrets
(`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`), never in the repo.

## Secrets policy
R2 keys, DB URLs, and host tokens live in GitHub Secrets or the serving
host's secret store — never in the repo, never in a committed Hydra config.

## Verify before building on them (free tiers shift)
- Serving host terms (Hugging Face Spaces / Fly.io / Render) and cold-start
  behaviour — decide whether the demo tolerates a sleeping API.
- R2 free-tier limits; GitHub Actions minutes.
- GPU-burst provider (Lambda/RunPod/Modal/etc.) pricing per run.
