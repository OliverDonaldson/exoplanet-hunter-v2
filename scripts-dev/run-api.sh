#!/bin/zsh
# Dev-only launcher used by the editor preview; `make api` is the normal path.
# Portable: repo root comes from this script's location, and the conda env
# python is discovered (override with $EXO_PYTHON if your install is unusual).
set -euo pipefail

ROOT="${0:a:h:h}"

PY="${EXO_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/anaconda3 /opt/miniconda3 \
      /opt/homebrew/Caskroom/miniconda/base /usr/local/Caskroom/miniconda/base; do
    if [[ -x "$base/envs/exoplanet-hunter-v2/bin/python" ]]; then
      PY="$base/envs/exoplanet-hunter-v2/bin/python"
      break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "run-api.sh: conda env 'exoplanet-hunter-v2' not found — set EXO_PYTHON" >&2
  exit 1
fi

cd "$ROOT/api"
export PYTHONPATH="$ROOT/api:$ROOT/pipeline/src"
exec "$PY" -m uvicorn app.main:app --port 8000
