"""Unit and integration tests for rate limiting and inference concurrency limits."""

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from clinical_guard.config import get_settings
from clinical_guard.main import (
    app,
    get_e5_guard,
    get_inference_semaphore,
    get_local_anonymizer,
    get_prompt_guard,
    get_rate_limiter,
)
from clinical_guard.privacy.anonymizer import LocalAnonymizer
from clinical_guard.rate_limit import SlidingWindowRateLimiter, get_client_ip
from clinical_guard.safety.content_safety import E5GuardResult
from clinical_guard.safety.prompt_safety import PromptGuardResult


class _StubPromptGuard:
    def evaluate(self, query: str) -> PromptGuardResult:
        if "ignore previous instructions" in query.lower():
            return PromptGuardResult(
                is_injection=True, label="INJECTION", score=0.99, latency_ms=0.5
            )
        return PromptGuardResult(is_injection=False, label="BENIGN", score=0.01, latency_ms=0.5)


class _StubE5Guard:
    def evaluate(self, query: str) -> E5GuardResult:
        return E5GuardResult(
            action="allow",
            risk_category="none",
            top_category="benign",
            top_score=0.9,
            benign_score=0.9,
            margin=-0.5,
            latency_ms=0.5,
        )


@pytest.fixture
def isolated_app_overrides() -> Iterator[None]:
    """Ensure dependency overrides are cleared after each test."""
    yield
    app.dependency_overrides.clear()


def test_sliding_window_rate_limiter_enforces_and_resets() -> None:
    now = 100.0
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10, clock=lambda: now)

    assert limiter.check("1.2.3.4") == (True, 0)
    assert limiter.check("1.2.3.4") == (True, 0)

    allowed, retry_after = limiter.check("1.2.3.4")
    assert allowed is False
    assert retry_after == 10

    # Different IP has its own bucket
    assert limiter.check("5.6.7.8") == (True, 0)

    # After window passes, requests are allowed again
    now = 110.1
    assert limiter.check("1.2.3.4") == (True, 0)


def test_get_client_ip_prefers_forwarded_headers() -> None:
    req_xff = SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.9, 10.0.0.1"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    assert get_client_ip(req_xff) == "203.0.113.9"

    req_xri = SimpleNamespace(
        headers={"x-real-ip": "198.51.100.4"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    assert get_client_ip(req_xri) == "198.51.100.4"

    req_direct = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))
    assert get_client_ip(req_direct) == "127.0.0.1"


def test_guard_evaluate_returns_429_when_rate_limited(isolated_app_overrides: None) -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    app.dependency_overrides[get_prompt_guard] = lambda: _StubPromptGuard()
    app.dependency_overrides[get_e5_guard] = lambda: _StubE5Guard()
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_inference_semaphore] = lambda: asyncio.Semaphore(4)

    client = TestClient(app)
    payload = {"query": "What is ibuprofen used for?", "session_id": "rl-test"}
    headers = {"X-User-Role": "layman", "X-Forwarded-For": "203.0.113.50"}

    assert client.post("/v1/guard/evaluate", json=payload, headers=headers).status_code == 200
    assert client.post("/v1/guard/evaluate", json=payload, headers=headers).status_code == 200

    blocked = client.post("/v1/guard/evaluate", json=payload, headers=headers)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1

    # System and benchmark endpoints remain unthrottled
    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/v1/benchmarks", headers=headers).status_code == 200


def test_guard_evaluate_context_returns_429_when_rate_limited(
    isolated_app_overrides: None,
) -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    app.dependency_overrides[get_prompt_guard] = lambda: _StubPromptGuard()
    app.dependency_overrides[get_local_anonymizer] = lambda: LocalAnonymizer(enable_ner=False)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_inference_semaphore] = lambda: asyncio.Semaphore(4)

    client = TestClient(app)
    payload = {
        "query": "Ignore previous instructions",
        "response_text": "Here is the answer.",
        "contexts": [{"doc_id": "d1", "text": "Normal text."}],
        "session_id": "rl-ctx",
    }
    headers = {"X-Forwarded-For": "203.0.113.51"}

    first = client.post("/v1/guard/evaluate_context", json=payload, headers=headers)
    assert first.status_code == 200
    blocked = client.post("/v1/guard/evaluate_context", json=payload, headers=headers)
    assert blocked.status_code == 429


def test_guard_evaluate_returns_503_when_inference_semaphore_exhausted(
    isolated_app_overrides: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exhausted_semaphore = asyncio.Semaphore(0)
    app.dependency_overrides[get_prompt_guard] = lambda: _StubPromptGuard()
    app.dependency_overrides[get_e5_guard] = lambda: _StubE5Guard()
    app.dependency_overrides[get_rate_limiter] = lambda: SlidingWindowRateLimiter(
        max_requests=10, window_seconds=60
    )
    app.dependency_overrides[get_inference_semaphore] = lambda: exhausted_semaphore

    settings = get_settings()
    monkeypatch.setattr(settings, "inference_queue_timeout_seconds", 0.01)

    client = TestClient(app)
    response = client.post(
        "/v1/guard/evaluate",
        json={"query": "What is ibuprofen used for?", "session_id": "sem-test"},
        headers={"X-User-Role": "layman"},
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
