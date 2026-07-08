"""Exoplanet Hunter V2 serving app.

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive OpenAPI docs at /docs — this replaces GUI API clients entirely;
contract tests live in tests/test_contract.py.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.candidates import router as candidates_router
from app.routes.score import router as score_router
from app.schemas import HealthResponse

app = FastAPI(
    title="Exoplanet Hunter V2",
    description="TIC ID -> live calibrated transit probability + vetting diagnostics.",
    version="2.0.0.dev0",
)

# The React dev server (vite, port 5173) calls the API cross-origin during
# development; production serves both behind one host so this list stays short.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(score_router)
app.include_router(candidates_router)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="degraded", model_loaded=False, model_version=None)
