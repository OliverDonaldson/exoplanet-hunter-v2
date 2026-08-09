"""Per-client token-bucket rate limiting for the public scoring endpoint.

`/score` is the one expensive route: a miss costs a MAST download plus a
five-fold TF inference on a 2 GB scale-to-zero machine. Two mitigations already
exist and this is deliberately the third rather than the first — `_score_lock`
serialises scoring so concurrent callers queue instead of thrashing the CPU, and
the process-lifetime response cache makes a repeated TIC free. What neither
bounds is a caller walking *distinct* TIC IDs: every one is a cache miss and a
fresh download, serialised into a slow drain of the machine's wall clock and the
account's egress.

So the exposure this closes is **cost and availability, not correctness**, and
the default is set to be invisible to a human using the console.

Two decisions worth stating, because getting either wrong is worse than having
no limiter at all:

**Client identity behind a proxy.** `request.client.host` is the *proxy's*
address once this runs on Fly, so keying on it would put every user in the world
into one bucket — a limiter that reliably throttles everyone at once. The client
address therefore comes from a proxy header when one is configured.

**Spoofing.** A caller can set `X-Forwarded-For` freely, so trusting it
unconditionally hands out a fresh bucket per request and the limiter does
nothing. The header is read **only** when `TRUSTED_CLIENT_IP_HEADER` names it,
which is a deployment assertion that a proxy overwrites it inbound. Fly's
`Fly-Client-IP` is set by their edge and cannot be injected from outside, so that
is the value to configure there. Unset — as in local development — the socket
address is used and no header is believed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

#: Requests per window, per client. A console session scores a handful of
#: targets a minute at most; this leaves an order of magnitude of headroom.
DEFAULT_CAPACITY = 20
#: Seconds over which a spent bucket refills completely.
DEFAULT_WINDOW_SECONDS = 60.0
#: Buckets idle for longer than this are dropped. Without it the map is an
#: unbounded dict keyed on attacker-supplied identity — a memory-exhaustion
#: vector inside the thing built to bound resource use.
IDLE_EVICTION_SECONDS = 900.0
#: Evicting on every request would be O(clients) per call; amortise it.
EVICTION_INTERVAL_SECONDS = 60.0


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token bucket per client identity, refilled continuously.

    A continuous refill rather than a fixed window: a fixed window lets a caller
    spend a full allowance at 0:59 and another at 1:01, so the real burst is
    twice the configured one at the boundary.

    `clock` is injected so the tests can advance time instead of sleeping —
    a limiter tested with `sleep` is a limiter tested at one timing.
    """

    capacity: int = DEFAULT_CAPACITY
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    clock: object = time.monotonic
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_eviction: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {self.capacity}")
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {self.window_seconds}")

    def _now(self) -> float:
        return float(self.clock())  # type: ignore[operator]

    def _evict_idle(self, now: float) -> None:
        """Drop buckets untouched for `IDLE_EVICTION_SECONDS`. Caller holds the lock."""
        if now - self._last_eviction < EVICTION_INTERVAL_SECONDS:
            return
        self._last_eviction = now
        stale = [key for key, b in self._buckets.items() if now - b.updated > IDLE_EVICTION_SECONDS]
        for key in stale:
            del self._buckets[key]

    def check(self, client: str) -> tuple[bool, float]:
        """Spend one token for `client`.

        Returns `(allowed, retry_after_seconds)`. `retry_after` is 0.0 when
        allowed, and otherwise the wait until one whole token exists — reported
        rather than guessed, so the 429 carries a header a client can obey.
        """
        rate = self.capacity / self.window_seconds
        now = self._now()
        with self._lock:
            self._evict_idle(now)
            bucket = self._buckets.get(client)
            if bucket is None:
                # A new client starts full, so a first request is never refused.
                bucket = _Bucket(tokens=float(self.capacity), updated=now)
                self._buckets[client] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated)
                bucket.tokens = min(float(self.capacity), bucket.tokens + elapsed * rate)
                bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - bucket.tokens) / rate


def client_identity(request: Request) -> str:
    """The address to key a bucket on — see the module docstring on spoofing.

    Only the header named by `TRUSTED_CLIENT_IP_HEADER` is believed, and only
    its first entry: `X-Forwarded-For` is a comma-separated chain and a caller
    controls everything after the proxy's own append.
    """
    header = os.environ.get("TRUSTED_CLIENT_IP_HEADER", "").strip()
    if header:
        forwarded = request.headers.get(header)
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    if request.client is not None and request.client.host:
        return request.client.host
    # No socket address and no trusted header: one shared bucket is the
    # conservative reading. Returning a unique key here would silently disable
    # the limiter for exactly the callers it cannot identify.
    return "unknown"


def _limiter_from_env() -> RateLimiter | None:
    """`None` when disabled. Set `RATE_LIMIT_PER_MINUTE=0` to turn it off."""
    raw = os.environ.get("RATE_LIMIT_PER_MINUTE")
    capacity = DEFAULT_CAPACITY if raw is None else int(raw)
    if capacity <= 0:
        return None
    return RateLimiter(capacity=capacity, window_seconds=DEFAULT_WINDOW_SECONDS)


_limiter = _limiter_from_env()


def rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 with `Retry-After` once a client is over budget."""
    if _limiter is None:
        return
    allowed, retry_after = _limiter.check(client_identity(request))
    if allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="rate limit exceeded; retry shortly",
        headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
    )
