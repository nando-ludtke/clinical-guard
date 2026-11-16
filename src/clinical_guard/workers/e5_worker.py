"""Dedicated FastAPI worker service wrapping E5SafetyGuard."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from clinical_guard.config import get_settings
from clinical_guard.logging_config import configure_logging, get_logger
from clinical_guard.safety.content_safety import E5Action, E5SafetyGuard
from clinical_guard.schemas import RiskCategory

logger = get_logger(__name__)


class E5PredictRequest(BaseModel):
    """Incoming query payload for E5 content safety inference."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=8000)


class E5PredictResponse(BaseModel):
    """Serialized E5GuardResult returned by the E5 worker."""

    model_config = ConfigDict(extra="forbid")

    action: E5Action
    risk_category: RiskCategory
    top_category: str
    top_score: float
    benign_score: float
    margin: float
    latency_ms: float


class WorkerHealthResponse(BaseModel):
    """Health check payload for the E5 worker."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    loaded: bool


def get_guard(request: Request) -> Any:
    """Return the app-scoped E5SafetyGuard instance."""
    guard = getattr(request.app.state, "guard", None)
    if guard is None:
        guard = E5SafetyGuard()
        request.app.state.guard = guard
    return guard


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and pre-load the E5 model before serving traffic."""
    settings = get_settings()
    configure_logging(settings.log_level)

    guard = getattr(fastapi_app.state, "guard", None)
    if guard is None:
        guard = E5SafetyGuard()
        fastapi_app.state.guard = guard
    if not guard.is_loaded:
        guard.load()
        guard.evaluate("warmup check")

    logger.info(
        "worker_startup",
        extra={"event": "worker_startup", "service": "e5-guard", "loaded": guard.is_loaded},
    )
    yield
    logger.info("worker_shutdown", extra={"event": "worker_shutdown", "service": "e5-guard"})


app = FastAPI(title="clinical-guard-e5-worker", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=WorkerHealthResponse, tags=["system"])
async def health(guard: Annotated[Any, Depends(get_guard)]) -> WorkerHealthResponse:
    """Return worker liveness and model readiness status."""
    return WorkerHealthResponse(
        status="ok",
        service="e5-guard",
        loaded=bool(guard.is_loaded),
    )


@app.post("/predict", response_model=E5PredictResponse, tags=["predict"])
def predict(
    payload: E5PredictRequest,
    guard: Annotated[Any, Depends(get_guard)],
) -> E5PredictResponse:
    """Evaluate a query with E5SafetyGuard and return E5GuardResult JSON."""
    result = guard.evaluate(payload.query)
    return E5PredictResponse(**asdict(result))
