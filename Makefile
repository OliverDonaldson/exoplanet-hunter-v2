# Exoplanet Hunter V2 — developer entry points.
#
# Every target that runs project code goes through $(PYTHON), so it follows
# the active environment. Override for a specific interpreter:
#   make PYTHON=/opt/anaconda3/envs/exoplanet-hunter-v2/bin/python data-push
PYTHON ?= python

# ai-slop-detector lives in its own venv, built with --system-site-packages on
# top of exoplanet-hunter-v2. That is deliberate and the same coupling problem
# as `dvc` below, in reverse: the tool's phantom_import check resolves imports
# against whatever interpreter runs it, so it MUST see numpy/pandas/sklearn/
# exoplanet_hunter or it reports them as critical "phantom imports" (measured
# 2026-08-16: 47 false criticals, score 11.9 instead of 2.47). Building the venv
# on the conda env gives it read access to those without letting pip resolve
# anything into the env that owns the TensorFlow pins.
#   Rebuild: conda activate exoplanet-hunter-v2 && \
#     python -m venv --system-site-packages ~/.local/venvs/ai-slop-detector && \
#     ~/.local/venvs/ai-slop-detector/bin/pip install "ai-slop-detector==3.8.7"
SLOP ?= $(HOME)/.local/venvs/ai-slop-detector/bin/slop-detector

.PHONY: env install lint type test validate refresh data-push data-pull mlflow api frontend up slop slop-report

env:            ## Create the conda env (pipeline + api, editable, dev extras)
	conda env create -f environment.yml

install:        ## Editable install into the *active* environment
	pip install -e ./pipeline[dev] -e ./api[dev]

lint:
	ruff check pipeline api

type:
	mypy pipeline/src

test:           ## Fast tests only (network/slow markers excluded)
	pytest pipeline/tests -m "not network and not slow"
	pytest api/tests

slop:           ## Structural-risk scan; same gate CI runs (fails above 3.5)
	$(SLOP) --project . --no-history --fail-threshold 3.5

slop-report:    ## Full audit incl. cross-file duplicates and JS/TS -> slop.json
	$(SLOP) --project . --cross-file --js --no-history --json --output slop.json

validate:       ## Run the data validation gates on whatever artefacts exist
	$(PYTHON) pipeline/scripts/validate_data.py

refresh:        ## Run the full refresh DAG (trains only if warranted)
	$(PYTHON) orchestration/flows/refresh_pipeline.py

# `dvc` as a bare command is what broke the weekly cron for weeks (0898939):
# it is absent from launchd's PATH, and a `command -v` probe can even resolve
# to a stale path that no longer exists. `python -m dvc` binds it to the
# interpreter that owns the install, which is the coupling we actually want.
data-push:      ## Sync DVC-tracked artefacts to R2
	$(PYTHON) -m dvc push

data-pull:      ## Materialise DVC-tracked artefacts from R2
	$(PYTHON) -m dvc pull

mlflow:         ## MLflow UI on :5001 (5000 collides with macOS AirPlay)
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

api:            ## Run the FastAPI dev server on :8000
	cd api && uvicorn app.main:app --reload --port 8000

frontend:       ## Build the console into frontend/design-console/dist, serve it on :5173
	cd frontend && npm install --no-audit --no-fund && python3 design-console/build.py
	@echo "open http://localhost:5173/?api=http://localhost:8000  (with 'make api' running)"
	cd frontend/design-console/dist && python3 -m http.server 5173

up:             ## The API container via Docker (the console is a static build)
	docker compose up --build

figures:        ## Regenerate docs/figures/ from the promoted run
	python pipeline/scripts/make_performance_figures.py

report:         ## Render docs/report.md to docs/report.pdf (needs pandoc + xelatex)
	pandoc docs/report.md -o docs/report.pdf \
	  --resource-path=docs --pdf-engine=xelatex --number-sections
	@echo "wrote docs/report.pdf"

ready:          ## Is the project fit to show? Prints LOOKS GOOD or NOT YET
	python pipeline/scripts/check_showcase_ready.py

ready-live:     ## The same, plus the deployed API and console
	python pipeline/scripts/check_showcase_ready.py --live
