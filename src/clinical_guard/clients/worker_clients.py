"""HTTP client adapters implementing the guard and NER detector interfaces."""

from __future__ import annotations

from typing import Any

import httpx

from clinical_guard.privacy.anonymizer import EntitySpan
from clinical_guard.safety.content_safety import E5GuardResult
from clinical_guard.safety.prompt_safety import PromptGuardResult


class _BaseWorkerClient:
    """Shared synchronous HTTP client wrapper for model worker services."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def close(self) -> None:
        """Close the underlying httpx.Client connection pool."""
        self._client.close()

    async def aclose(self) -> None:
        """Async-compatible close hook for FastAPI lifespan shutdown."""
        self.close()


class HttpE5SafetyGuard(_BaseWorkerClient):
    """HTTP adapter for the remote E5SafetyGuard worker service."""

    def evaluate(self, query: str) -> E5GuardResult:
        """Evaluate query safety via POST /predict on the E5 worker."""
        data = self._post_json("/predict", {"query": query})
        return E5GuardResult(
            action=data["action"],
            risk_category=data["risk_category"],
            top_category=data["top_category"],
            top_score=float(data["top_score"]),
            benign_score=float(data["benign_score"]),
            margin=float(data["margin"]),
            latency_ms=float(data["latency_ms"]),
        )


class HttpPromptInjectionGuard(_BaseWorkerClient):
    """HTTP adapter for the remote PromptInjectionGuard worker service."""

    def evaluate(self, query: str) -> PromptGuardResult:
        """Evaluate prompt injection risk via POST /predict on the prompt worker."""
        data = self._post_json("/predict", {"query": query})
        return PromptGuardResult(
            is_injection=bool(data["is_injection"]),
            label=str(data["label"]),
            score=float(data["score"]),
            latency_ms=float(data["latency_ms"]),
        )


class HttpNerDetector(_BaseWorkerClient):
    """HTTP adapter for the remote OnnxPiiDetector worker service."""

    def detect(self, text: str) -> list[EntitySpan]:
        """Detect PII entity spans via POST /predict on the NER worker."""
        if not text or not text.strip():
            return []
        data = self._post_json("/predict", {"text": text})
        spans: list[EntitySpan] = []
        for item in data.get("spans", []):
            spans.append(
                EntitySpan(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    entity_type=str(item["entity_type"]),
                )
            )
        return spans

    def __call__(self, text: str) -> list[EntitySpan]:
        """Allow HttpNerDetector to be used directly as a CustomEntityDetector."""
        return self.detect(text)
