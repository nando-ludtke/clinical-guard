"""Dedicated FastAPI worker service wrapping OnnxPiiDetector."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from clinical_guard.config import get_settings
from clinical_guard.logging_config import configure_logging, get_logger
from clinical_guard.privacy.anonymizer import OnnxPiiDetector

logger = get_logger(__name__)


class NerPredictRequest(BaseModel):
    """Incoming text payload for ONNX PII NER inference."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., max_length=32000)


class NerSpanItem(BaseModel):
    """Serialized PII entity span returned by the NER worker."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    entity_type: str
    text: str


class NerPredictResponse(BaseModel):
    """Response payload containing detected PII entity spans."""

    model_config = ConfigDict(extra="forbid")

    spans: list[NerSpanItem]


class WorkerHealthResponse(BaseModel):
    """Health check payload for the NER PII worker."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    loaded: bool


def get_detector(request: Request) -> Any:
    """Return the app-scoped OnnxPiiDetector instance."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        detector = OnnxPiiDetector()
        request.app.state.detector = detector
    return detector


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and pre-load the ONNX NER pipeline before serving traffic."""
    settings = get_settings()
    configure_logging(settings.log_level)

    detector = getattr(fastapi_app.state, "detector", None)
    if detector is None:
        detector = OnnxPiiDetector()
        fastapi_app.state.detector = detector
    if not detector.is_loaded:
        detector.load()
        detector.detect("warmup check")

    logger.info(
        "worker_startup",
        extra={
            "event": "worker_startup",
            "service": "ner-pii-guard",
            "loaded": detector.is_loaded,
        },
    )
    yield
    logger.info("worker_shutdown", extra={"event": "worker_shutdown", "service": "ner-pii-guard"})


app = FastAPI(title="clinical-guard-ner-worker", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=WorkerHealthResponse, tags=["system"])
async def health(detector: Annotated[Any, Depends(get_detector)]) -> WorkerHealthResponse:
    """Return worker liveness and model readiness status."""
    return WorkerHealthResponse(
        status="ok",
        service="ner-pii-guard",
        loaded=bool(detector.is_loaded),
    )


@app.post("/predict", response_model=NerPredictResponse, tags=["predict"])
def predict(
    payload: NerPredictRequest,
    detector: Annotated[Any, Depends(get_detector)],
) -> NerPredictResponse:
    """Detect PII entities in text using OnnxPiiDetector."""
    spans = detector.detect(payload.text)
    return NerPredictResponse(
        spans=[
            NerSpanItem(
                start=span.start,
                end=span.end,
                entity_type=span.entity_type,
                text=payload.text[span.start : span.end],
            )
            for span in spans
        ]
    )
