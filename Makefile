# Exoplanet Hunter V2 — developer entry points.
#
# Every target that runs project code goes through $(PYTHON), so it follows
# the active environment. Override for a specific interpreter:
#   make PYTHON=/opt/anaconda3/envs/exoplanet-hunter-v2/bin/python data-push
PYTHON ?= python

.PHONY: env install lint type test validate refresh data-push data-pull mlflow api frontend up

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

frontend:       ## Run the Vite dev server on :5173 (proxies /api -> :8000)
	cd frontend && npm run dev

up:             ## Full local stack via Docker
	docker compose up --build
