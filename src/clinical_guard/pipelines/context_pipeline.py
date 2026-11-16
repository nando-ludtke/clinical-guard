"""Context groundedness pipeline with parallel injection gate and entity anonymization."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import anyio

from clinical_guard.context.context_check import (
    ContextGroundednessChecker,
    normalize_context_documents,
)
from clinical_guard.privacy.anonymizer import EntityTable, LocalAnonymizer
from clinical_guard.safety.prompt_safety import PromptInjectionGuard
from clinical_guard.schemas import (
    ContextDocument,
    ContextRoutingTarget,
    GroundednessResult,
)


@dataclass(frozen=True, slots=True)
class ContextPipelineResult:
    """Result of the parallel injection-gate + anonymization + groundedness pipeline."""

    security_blocked: bool
    routing_target: ContextRoutingTarget
    flagged_doc_ids: list[str]
    groundedness: GroundednessResult | None
    entity_table: EntityTable
    latency_ms: float


async def run_context_pipeline(
    query: str,
    response_text: str,
    contexts: Sequence[ContextDocument | str],
    conversation_history: Sequence[dict[str, str]] | None,
    prompt_guard: PromptInjectionGuard,
    groundedness_checker: ContextGroundednessChecker,
    anonymizer: LocalAnonymizer,
    api_key: str | None = None,
) -> ContextPipelineResult:
    """Run PromptInjectionGuard and LocalAnonymizer in parallel, then check groundedness."""
    start_time = time.perf_counter()
    normalized_docs = normalize_context_documents(contexts)
    entity_table = anonymizer.create_table()

    input_flagged = False
    doc_flags: dict[str, bool] = {}

    def _populate_entity_table() -> None:
        entity_table.anonymize(query)
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "")
                if content:
                    entity_table.anonymize(content)
        entity_table.anonymize(response_text)
        for doc in normalized_docs:
            entity_table.anonymize(doc.text)

    async def _scan_input(text: str) -> None:
        nonlocal input_flagged
        if not text:
            return
        res = await anyio.to_thread.run_sync(prompt_guard.evaluate, text)
        if res.is_injection:
            input_flagged = True

    async def _scan_doc(doc: ContextDocument) -> None:
        if not doc.text:
            return
        res = await anyio.to_thread.run_sync(prompt_guard.evaluate, doc.text)
        if res.is_injection:
            doc_flags[doc.doc_id] = True

    async with anyio.create_task_group() as tg:
        tg.start_soon(anyio.to_thread.run_sync, _populate_entity_table)
        tg.start_soon(_scan_input, query)
        tg.start_soon(_scan_input, response_text)
        if conversation_history:
            for msg in conversation_history:
                tg.start_soon(_scan_input, msg.get("content", ""))
        for doc in normalized_docs:
            tg.start_soon(_scan_doc, doc)

    flagged_doc_ids = [doc.doc_id for doc in normalized_docs if doc_flags.get(doc.doc_id)]

    if input_flagged or flagged_doc_ids:
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return ContextPipelineResult(
            security_blocked=True,
            routing_target="security_block",
            flagged_doc_ids=flagged_doc_ids,
            groundedness=None,
            entity_table=entity_table,
            latency_ms=latency_ms,
        )

    groundedness = await groundedness_checker.check(
        query=query,
        response_text=response_text,
        contexts=normalized_docs,
        conversation_history=conversation_history,
        entity_table=entity_table,
        api_key=api_key,
    )
    routing_target: ContextRoutingTarget = (
        "context_grounded" if groundedness.grounded else "context_ungrounded"
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

    return ContextPipelineResult(
        security_blocked=False,
        routing_target=routing_target,
        flagged_doc_ids=[],
        groundedness=groundedness,
        entity_table=entity_table,
        latency_ms=latency_ms,
    )
