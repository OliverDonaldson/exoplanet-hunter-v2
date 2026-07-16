"""Exoplanet Hunter V2 serving app.

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive OpenAPI docs at /docs — this replaces GUI API clients entirely;
contract tests live in tests/test_contract.py.
"""

from __future__ import annotations

import contextlib
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.candidates import router as candidates_router
from app.routes.reliability import router as reliability_router
from app.routes.score import router as score_router
from app.schemas import HealthResponse


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    # Warm the TF ensemble off the request path: the console wakes the
    # machine on page load, so the ~90 s model load runs while the user is
    # still browsing the catalogue instead of on their first score click.
    from app.routes.score import get_scorer

    def _warm() -> None:
        try:
            get_scorer()
        except Exception as exc:  # no registry yet — lazy path still applies
            print(f"[warmup] ensemble preload skipped: {exc}")

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="Exoplanet Hunter V2",
    description="TIC ID -> live calibrated transit probability + vetting diagnostics.",
    version="2.0.0.dev0",
    lifespan=_lifespan,
)

# The React dev server (vite, port 5173) calls the API cross-origin during
# development; a deployed static console announces its origin via
# FRONTEND_ORIGIN (set in the host's environment, see render.yaml).
_origins = ["http://localhost:5173"]
if frontend_origin := os.environ.get("FRONTEND_ORIGIN"):
    _origins.append(frontend_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(score_router)
app.include_router(candidates_router)
app.include_router(reliability_router)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Servable = a promoted run exists in the registry (loaded lazily on
    first /score request), so this stays cheap — a JSON stat, no TF."""
    import json
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    registry = Path(os.environ.get("MODEL_DIR", root / "models")) / "registry.json"
    if registry.exists():
        run_id = str(json.loads(registry.read_text())["run_id"])
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=f"cnn_dualview-cv-{run_id[:8]}",
        )
    return HealthResponse(status="degraded", model_loaded=False, model_version=None)
