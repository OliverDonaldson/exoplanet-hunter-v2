# Infra notes

## Cloudflare R2 (zero-egress object store)
Buckets/prefixes: `catalogue/` (parquet), `views/` (TFRecord shards),
`models/` (model + calibration bundles), `scores/` (scores.parquet read by
the console via DuckDB).

## Secrets policy
R2 keys, DB URLs, and host tokens live in GitHub Secrets or the serving
host's secret store — never in the repo, never in a committed Hydra config.

## Verify before building on them (free tiers shift)
- Serving host terms (Hugging Face Spaces / Fly.io / Render) and cold-start
  behaviour — decide whether the demo tolerates a sleeping API.
- R2 free-tier limits; GitHub Actions minutes.
- GPU-burst provider (Lambda/RunPod/Modal/etc.) pricing per run.
