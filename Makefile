# Exoplanet Hunter V2 — developer entry points.

.PHONY: env install lint type test api frontend up

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

api:            ## Run the FastAPI dev server on :8000
	cd api && uvicorn app.main:app --reload --port 8000

frontend:       ## Run the Vite dev server on :5173 (proxies /api -> :8000)
	cd frontend && npm run dev

up:             ## Full local stack via Docker
	docker compose up --build
