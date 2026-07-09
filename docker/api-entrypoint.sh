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

echo "[entrypoint] pulling DVC artefacts from R2 ..."
dvc pull -q data/catalogue.dvc models/cv/*.dvc
echo "[entrypoint] artefacts ready:"
ls models/cv/ data/catalogue/

exec uvicorn app.main:app --app-dir api --host 0.0.0.0 --port "${PORT:-8000}"
