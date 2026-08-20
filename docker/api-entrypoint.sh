#!/bin/sh
# Boot sequence for the serving container: materialise the promoted model
# and catalogue from R2 via DVC, then start uvicorn.
#
# Credentials arrive as AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
# (DVC's s3 remote reads the standard boto3 chain), set in the host's
# secret store — never baked into the image.
set -e

# DVC wants an SCM root; cloud build contexts may strip .git.
if [ ! -d .git ]; then
    git init -q
fi

# SKIP_DVC_PULL=1 serves pre-mounted artefacts (local smoke tests) — without
# credentials, botocore's credential-chain probing hangs for minutes.
if [ "${SKIP_DVC_PULL:-0}" = "1" ]; then
    echo "[entrypoint] SKIP_DVC_PULL=1 — serving mounted artefacts"
else
    echo "[entrypoint] pulling DVC artefacts from R2 (creds present: $([ -n "$AWS_ACCESS_KEY_ID" ] && echo yes || echo NO)) ..."
    dvc pull -v data/tables/catalogue.dvc models/cv/*.dvc || {
        echo "[entrypoint] FATAL: dvc pull failed — check AWS_* secrets and R2 access"
        exit 1
    }
    # Non-fatal: the API degrades honestly without these. No labels means
    # /model reports no per-mission split; no scores means the catalogue's
    # P(planet) column reads "not scored". Both are better than refusing to
    # boot, because /score and /healthz do not depend on either.
    dvc pull -v data/tables/labels.dvc results/candidates_scored.parquet.dvc || \
        echo "[entrypoint] WARNING: optional artefacts unavailable — per-mission metrics and catalogue scores will be absent"
fi
echo "[entrypoint] artefacts ready:"
ls models/cv/ data/tables/catalogue/
ls data/tables/labels/ results/ 2>/dev/null || true

exec uvicorn app.main:app --app-dir api --host 0.0.0.0 --port "${PORT:-8000}"
