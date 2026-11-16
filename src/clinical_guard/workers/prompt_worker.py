"""Dedicated FastAPI worker service wrapping PromptInjectionGuard."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from clinical_guard.config import get_settings
from clinical_guard.logging_config import configure_logging, get_logger
from clinical_guard.safety.prompt_safety import PromptInjectionGuard

logger = get_logger(__name__)


class PromptPredictRequest(BaseModel):
    """Incoming query payload for prompt-injection inference."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=16000)


class PromptPredictResponse(BaseModel):
    """Serialized PromptGuardResult returned by the prompt-guard worker."""

    model_config = ConfigDict(extra="forbid")

    is_injection: bool
    label: str
    score: float
    latency_ms: float


class WorkerHealthResponse(BaseModel):
    """Health check payload for the prompt-guard worker."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    loaded: bool


def get_guard(request: Request) -> Any:
    """Return the app-scoped PromptInjectionGuard instance."""
    guard = getattr(request.app.state, "guard", None)
    if guard is None:
        guard = PromptInjectionGuard()
        request.app.state.guard = guard
    return guard


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and pre-load the Prompt Guard model before serving traffic."""
    settings = get_settings()
    configure_logging(settings.log_level)

    guard = getattr(fastapi_app.state, "guard", None)
    if guard is None:
        guard = PromptInjectionGuard()
        fastapi_app.state.guard = guard
    if not guard.is_loaded:
        guard.load()
        guard.evaluate("warmup check")

    logger.info(
        "worker_startup",
        extra={"event": "worker_startup", "service": "prompt-guard", "loaded": guard.is_loaded},
    )
    yield
    logger.info("worker_shutdown", extra={"event": "worker_shutdown", "service": "prompt-guard"})


app = FastAPI(title="clinical-guard-prompt-worker", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=WorkerHealthResponse, tags=["system"])
async def health(guard: Annotated[Any, Depends(get_guard)]) -> WorkerHealthResponse:
    """Return worker liveness and model readiness status."""
    return WorkerHealthResponse(
        status="ok",
        service="prompt-guard",
        loaded=bool(guard.is_loaded),
    )


@app.post("/predict", response_model=PromptPredictResponse, tags=["predict"])
def predict(
    payload: PromptPredictRequest,
    guard: Annotated[Any, Depends(get_guard)],
) -> PromptPredictResponse:
    """Evaluate a query with PromptInjectionGuard and return PromptGuardResult JSON."""
    result = guard.evaluate(payload.query)
    return PromptPredictResponse(**asdict(result))
