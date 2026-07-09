#!/bin/zsh
# Dev-only launcher used by the editor preview; `make api` is the normal path.
cd /Users/ollie/Project/v2/api
export PYTHONPATH=/Users/ollie/Project/v2/api:/Users/ollie/Project/v2/pipeline/src
exec /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python -m uvicorn app.main:app --port 8000
