"""In-memory IP sliding-window rate limiter and inference concurrency semaphore."""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request, status

_MAX_TRACKED_IPS = 10_000


def get_client_ip(request: Request) -> str:
    """Extract the client IP, honoring reverse-proxy headers such as X-Forwarded-For."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


class SlidingWindowRateLimiter:
    """Thread-safe/async-safe per-IP sliding-window rate limiter."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request for `key` if allowed, returning `(allowed, retry_after_seconds)`."""
        now = self._clock()
        cutoff = now - self.window_seconds

        if len(self._hits) >= _MAX_TRACKED_IPS and key not in self._hits:
            self._prune_stale_keys(cutoff)

        bucket = self._hits.get(key)
        if bucket is None:
            bucket = deque()
            self._hits[key] = bucket

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            retry_after = max(1, math.ceil(bucket[0] + self.window_seconds - now))
            return False, retry_after

        bucket.append(now)
        return True, 0

    def _prune_stale_keys(self, cutoff: float) -> None:
        stale_keys = [
            ip
            for ip, timestamps in self._hits.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for ip in stale_keys:
            del self._hits[ip]


@asynccontextmanager
async def acquire_inference_slot(
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
) -> AsyncIterator[None]:
    """Acquire an inference concurrency slot or fail fast with HTTP 503."""
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference queue is busy; please retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc

    try:
        yield
    finally:
        semaphore.release()
