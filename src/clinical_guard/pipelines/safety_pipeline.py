"""Concurrent System 1 safety evaluation pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import anyio

from clinical_guard.routing.policy import RoutingDecision, resolve_policy
from clinical_guard.safety.content_safety import E5GuardResult, E5SafetyGuard
from clinical_guard.safety.prompt_safety import PromptGuardResult, PromptInjectionGuard
from clinical_guard.schemas import UserRole


@dataclass(frozen=True, slots=True)
class SafetyPipelineResult:
    """Combined output from concurrent System 1 guards and policy resolution."""

    routing_decision: RoutingDecision
    prompt_result: PromptGuardResult
    e5_result: E5GuardResult
    latency_ms: float


async def run_safety_pipeline(
    query: str,
    user_role: UserRole,
    prompt_guard: PromptInjectionGuard,
    e5_guard: E5SafetyGuard,
) -> SafetyPipelineResult:
    """Run PromptInjectionGuard and E5SafetyGuard concurrently, then resolve policy."""
    start_time = time.perf_counter()
    prompt_result: PromptGuardResult | None = None
    e5_result: E5GuardResult | None = None

    async def _run_prompt() -> None:
        nonlocal prompt_result
        prompt_result = await anyio.to_thread.run_sync(prompt_guard.evaluate, query)

    async def _run_e5() -> None:
        nonlocal e5_result
        e5_result = await anyio.to_thread.run_sync(e5_guard.evaluate, query)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_prompt)
        tg.start_soon(_run_e5)

    assert prompt_result is not None
    assert e5_result is not None

    routing_decision = resolve_policy(
        user_role=user_role,
        e5_result=e5_result,
        prompt_result=prompt_result,
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

    return SafetyPipelineResult(
        routing_decision=routing_decision,
        prompt_result=prompt_result,
        e5_result=e5_result,
        latency_ms=latency_ms,
    )
