"""FastAPI application entrypoint for clinical-guard."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from clinical_guard.benchmarks import load_benchmark_suites
from clinical_guard.clients.worker_clients import (
    HttpE5SafetyGuard,
    HttpNerDetector,
    HttpPromptInjectionGuard,
)
from clinical_guard.config import Settings, get_settings
from clinical_guard.context.context_check import ContextGroundednessChecker
from clinical_guard.logging_config import (
    configure_logging,
    get_logger,
    redact_query_metadata,
)
from clinical_guard.pipelines.context_pipeline import run_context_pipeline
from clinical_guard.pipelines.safety_pipeline import run_safety_pipeline
from clinical_guard.privacy.anonymizer import LocalAnonymizer
from clinical_guard.rate_limit import (
    SlidingWindowRateLimiter,
    acquire_inference_slot,
    get_client_ip,
)
from clinical_guard.safety.content_safety import E5SafetyGuard
from clinical_guard.safety.prompt_safety import PromptInjectionGuard
from clinical_guard.schemas import (
    ContextEvaluateRequest,
    ContextEvaluateResponse,
    GuardDiagnostics,
    GuardRequest,
    GuardResponse,
    HealthResponse,
    PromptBlockedBy,
    PromptCheckResponse,
    UserRole,
)

logger = get_logger(__name__)
_STATIC_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"


def _build_prompt_guard(current_settings: Settings) -> Any:
    if current_settings.prompt_guard_url:
        return HttpPromptInjectionGuard(
            base_url=current_settings.prompt_guard_url,
            timeout=current_settings.worker_timeout_seconds,
        )
    return PromptInjectionGuard()


def _build_e5_guard(current_settings: Settings) -> Any:
    if current_settings.e5_guard_url:
        return HttpE5SafetyGuard(
            base_url=current_settings.e5_guard_url,
            timeout=current_settings.worker_timeout_seconds,
        )
    return E5SafetyGuard()


def _build_anonymizer(current_settings: Settings) -> LocalAnonymizer:
    if current_settings.ner_guard_url:
        return LocalAnonymizer(
            ner_detector=HttpNerDetector(
                base_url=current_settings.ner_guard_url,
                timeout=current_settings.worker_timeout_seconds,
            )
        )
    return LocalAnonymizer(enable_ner=current_settings.enable_local_ner)


def get_prompt_guard(request: Request) -> PromptInjectionGuard | HttpPromptInjectionGuard:
    """Return the app-scoped prompt guard (HTTP worker client or local model)."""
    guard = getattr(request.app.state, "prompt_guard", None)
    if guard is None:
        guard = _build_prompt_guard(get_settings())
        request.app.state.prompt_guard = guard
    return guard


def get_e5_guard(request: Request) -> E5SafetyGuard | HttpE5SafetyGuard:
    """Return the app-scoped E5 guard (HTTP worker client or local model)."""
    guard = getattr(request.app.state, "e5_guard", None)
    if guard is None:
        guard = _build_e5_guard(get_settings())
        request.app.state.e5_guard = guard
    return guard


def get_groundedness_checker(request: Request) -> ContextGroundednessChecker:
    """Return the app-scoped ContextGroundednessChecker instance."""
    checker = getattr(request.app.state, "groundedness_checker", None)
    if checker is None:
        checker = ContextGroundednessChecker()
        request.app.state.groundedness_checker = checker
    return checker


def get_local_anonymizer(request: Request) -> LocalAnonymizer:
    """Return the app-scoped LocalAnonymizer instance (remote NER worker or local)."""
    anonymizer = getattr(request.app.state, "local_anonymizer", None)
    if anonymizer is None:
        anonymizer = _build_anonymizer(get_settings())
        request.app.state.local_anonymizer = anonymizer
    return anonymizer


get_anonymizer = get_local_anonymizer


def get_rate_limiter(request: Request) -> SlidingWindowRateLimiter:
    """Return the app-scoped IP sliding-window rate limiter."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        current_settings = get_settings()
        limiter = SlidingWindowRateLimiter(
            max_requests=current_settings.rate_limit_requests,
            window_seconds=current_settings.rate_limit_window_seconds,
        )
        request.app.state.rate_limiter = limiter
    return limiter


def get_inference_semaphore(request: Request) -> asyncio.Semaphore:
    """Return the app-scoped inference concurrency semaphore."""
    semaphore = getattr(request.app.state, "inference_semaphore", None)
    if semaphore is None:
        current_settings = get_settings()
        semaphore = asyncio.Semaphore(current_settings.max_concurrent_inferences)
        request.app.state.inference_semaphore = semaphore
    return semaphore


def enforce_rate_limit(
    request: Request,
    rate_limiter: Annotated[SlidingWindowRateLimiter, Depends(get_rate_limiter)],
) -> None:
    """Enforce per-IP rate limiting on guard endpoints when enabled."""
    current_settings = get_settings()
    if not current_settings.rate_limit_enabled:
        return

    client_ip = get_client_ip(request)
    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        logger.warning(
            "rate_limit_exceeded",
            extra={
                "event": "rate_limit_exceeded",
                "client_ip": client_ip,
                "path": request.url.path,
                "retry_after_seconds": retry_after,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Initialize structured logging, guard instances, and emit lifecycle audit events."""
    current_settings = get_settings()
    configure_logging(current_settings.log_level)

    if getattr(fastapi_app.state, "prompt_guard", None) is None:
        fastapi_app.state.prompt_guard = _build_prompt_guard(current_settings)
    if getattr(fastapi_app.state, "e5_guard", None) is None:
        fastapi_app.state.e5_guard = _build_e5_guard(current_settings)
    if getattr(fastapi_app.state, "groundedness_checker", None) is None:
        fastapi_app.state.groundedness_checker = ContextGroundednessChecker()
    if getattr(fastapi_app.state, "local_anonymizer", None) is None:
        fastapi_app.state.local_anonymizer = _build_anonymizer(current_settings)
    if getattr(fastapi_app.state, "rate_limiter", None) is None:
        fastapi_app.state.rate_limiter = SlidingWindowRateLimiter(
            max_requests=current_settings.rate_limit_requests,
            window_seconds=current_settings.rate_limit_window_seconds,
        )
    if getattr(fastapi_app.state, "inference_semaphore", None) is None:
        fastapi_app.state.inference_semaphore = asyncio.Semaphore(
            current_settings.max_concurrent_inferences
        )

    logger.info(
        "service_startup",
        extra={
            "event": "service_startup",
            "environment": current_settings.environment,
            "version": current_settings.app_version,
            "remote_e5": bool(current_settings.e5_guard_url),
            "remote_prompt": bool(current_settings.prompt_guard_url),
            "remote_ner": bool(current_settings.ner_guard_url),
        },
    )
    yield
    for state_attr in ("prompt_guard", "e5_guard"):
        client = getattr(fastapi_app.state, state_attr, None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()
    anonymizer = getattr(fastapi_app.state, "local_anonymizer", None)
    ner_detector = getattr(anonymizer, "ner_detector", None) if anonymizer is not None else None
    if ner_detector is not None and hasattr(ner_detector, "aclose"):
        await ner_detector.aclose()
    logger.info(
        "service_shutdown",
        extra={"event": "service_shutdown", "version": current_settings.app_version},
    )


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def showcase_index() -> HTMLResponse:
    """Serve the Clinical Guard Showcase UI."""
    return HTMLResponse(content=_STATIC_INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return service liveness metadata."""
    current_settings = get_settings()
    return HealthResponse(
        status="ok",
        service=current_settings.app_name,
        version=current_settings.app_version,
    )


@app.get("/v1/config/status", tags=["system"])
async def config_status() -> dict[str, bool]:
    """Report non-sensitive runtime configuration status (never returns secrets)."""
    return {"openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip())}


@app.get("/v1/benchmarks", tags=["benchmarks"])
async def list_benchmarks() -> dict[str, list[dict[str, Any]]]:
    """Return normalized evaluation benchmark suites from data/eval/."""
    return {"benchmarks": load_benchmark_suites()}


@app.post(
    "/v1/guard/evaluate",
    response_model=GuardResponse,
    tags=["guard"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def evaluate_guard(
    payload: GuardRequest,
    x_user_role: Annotated[UserRole, Header(alias="X-User-Role")],
    prompt_guard: Annotated[PromptInjectionGuard, Depends(get_prompt_guard)],
    e5_guard: Annotated[E5SafetyGuard, Depends(get_e5_guard)],
    inference_semaphore: Annotated[asyncio.Semaphore, Depends(get_inference_semaphore)],
) -> GuardResponse:
    """Evaluate prompt-injection and E5 content safety guards, then resolve routing policy."""
    async with acquire_inference_slot(
        inference_semaphore, get_settings().inference_queue_timeout_seconds
    ):
        result = await run_safety_pipeline(
            query=payload.query,
            user_role=x_user_role,
            prompt_guard=prompt_guard,
            e5_guard=e5_guard,
        )
    routing_decision = result.routing_decision
    prompt_result = result.prompt_result
    e5_result = result.e5_result

    logger.info(
        "guard_evaluated",
        extra={
            "event": "guard_evaluated",
            "session_id": payload.session_id,
            "user_role": x_user_role,
            "decision": routing_decision.decision,
            "risk_category": routing_decision.risk_category,
            "routing_target": routing_decision.routing_target,
            "rule_id": routing_decision.rule_id,
            "audit_flag": routing_decision.audit_flag,
            "simplify_output": routing_decision.simplify_output,
            "prompt_is_injection": prompt_result.is_injection,
            "prompt_score": prompt_result.score,
            "e5_action": e5_result.action,
            "e5_margin": e5_result.margin,
            "latency_ms": result.latency_ms,
            **redact_query_metadata(payload.query),
        },
    )

    return GuardResponse(
        decision=routing_decision.decision,
        risk_category=routing_decision.risk_category,
        routing_target=routing_decision.routing_target,
        response_text=routing_decision.response_text,
        disclaimer=routing_decision.disclaimer,
        simplify_output=routing_decision.simplify_output,
        rule_id=routing_decision.rule_id,
        latency_ms=result.latency_ms,
        session_id=payload.session_id,
        user_role=x_user_role,
        diagnostics=GuardDiagnostics(
            prompt_guard_score=prompt_result.score,
            prompt_guard_label=prompt_result.label,
            prompt_guard_is_injection=prompt_result.is_injection,
            prompt_guard_latency_ms=prompt_result.latency_ms,
            e5_top_category=e5_result.top_category,
            e5_top_score=e5_result.top_score,
            e5_benign_score=e5_result.benign_score,
            e5_margin=e5_result.margin,
            e5_latency_ms=e5_result.latency_ms,
        ),
    )


@app.post(
    "/v1/guard/prompt",
    response_model=PromptCheckResponse,
    tags=["guard"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def check_prompt_injection(
    payload: GuardRequest,
    prompt_guard: Annotated[PromptInjectionGuard, Depends(get_prompt_guard)],
    e5_guard: Annotated[E5SafetyGuard, Depends(get_e5_guard)],
    inference_semaphore: Annotated[asyncio.Semaphore, Depends(get_inference_semaphore)],
) -> PromptCheckResponse:
    """Screen a query for prompt injection using PromptInjectionGuard + E5 defense-in-depth."""
    async with acquire_inference_slot(
        inference_semaphore, get_settings().inference_queue_timeout_seconds
    ):
        result = await run_safety_pipeline(
            query=payload.query,
            user_role="layman",
            prompt_guard=prompt_guard,
            e5_guard=e5_guard,
        )

    prompt_result = result.prompt_result
    e5_result = result.e5_result
    routing_decision = result.routing_decision

    is_injection = routing_decision.decision == "block_security"
    blocked_by: PromptBlockedBy = "none"
    if prompt_result.is_injection:
        blocked_by = "prompt_guard"
    elif is_injection:
        blocked_by = "e5_prompt_injection"

    logger.info(
        "prompt_checked",
        extra={
            "event": "prompt_checked",
            "session_id": payload.session_id,
            "is_injection": is_injection,
            "blocked_by": blocked_by,
            "rule_id": routing_decision.rule_id if is_injection else None,
            "prompt_score": prompt_result.score,
            "e5_margin": e5_result.margin,
            "latency_ms": result.latency_ms,
            **redact_query_metadata(payload.query),
        },
    )

    return PromptCheckResponse(
        is_injection=is_injection,
        blocked_by=blocked_by,
        rule_id=routing_decision.rule_id if is_injection else None,
        prompt_guard_score=prompt_result.score,
        prompt_guard_label=prompt_result.label,
        e5_margin=e5_result.margin,
        latency_ms=result.latency_ms,
        session_id=payload.session_id,
    )


@app.post(
    "/v1/guard/evaluate_context",
    response_model=ContextEvaluateResponse,
    tags=["context"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def evaluate_context(
    payload: ContextEvaluateRequest,
    prompt_guard: Annotated[PromptInjectionGuard, Depends(get_prompt_guard)],
    groundedness_checker: Annotated[
        ContextGroundednessChecker, Depends(get_groundedness_checker)
    ],
    anonymizer: Annotated[LocalAnonymizer, Depends(get_local_anonymizer)],
    inference_semaphore: Annotated[asyncio.Semaphore, Depends(get_inference_semaphore)],
    x_openai_api_key: Annotated[str | None, Header(alias="X-OpenAI-Api-Key")] = None,
) -> ContextEvaluateResponse:
    """Run parallel injection gate + local anonymization, then groundedness with citations."""
    async with acquire_inference_slot(
        inference_semaphore, get_settings().inference_queue_timeout_seconds
    ):
        result = await run_context_pipeline(
            query=payload.query,
            response_text=payload.response_text,
            contexts=payload.contexts,
            conversation_history=payload.conversation_history,
            prompt_guard=prompt_guard,
            groundedness_checker=groundedness_checker,
            anonymizer=anonymizer,
            api_key=x_openai_api_key,
        )

    logger.info(
        "context_evaluated",
        extra={
            "event": "context_evaluated",
            "session_id": payload.session_id,
            "security_blocked": result.security_blocked,
            "routing_target": result.routing_target,
            "flagged_doc_ids": result.flagged_doc_ids,
            "grounded": (
                result.groundedness.grounded if result.groundedness is not None else None
            ),
            "entity_count": len(result.entity_table),
            "latency_ms": result.latency_ms,
            **redact_query_metadata(payload.query),
        },
    )

    return ContextEvaluateResponse(
        security_blocked=result.security_blocked,
        routing_target=result.routing_target,
        flagged_doc_ids=result.flagged_doc_ids,
        groundedness=result.groundedness,
        latency_ms=result.latency_ms,
        session_id=payload.session_id,
    )
