#!/bin/zsh
# Dev-only launcher used by the editor preview; `make api` is the normal path.
# Uses the V1 conda env until exoplanet-hunter-v2 finishes creating.
cd /Users/ollie/Project/v2/api
export PYTHONPATH=/Users/ollie/Project/v2/api:/Users/ollie/Project/v2/pipeline/src
exec /opt/anaconda3/envs/exoplanet-hunter/bin/python -m uvicorn app.main:app --port 8000
