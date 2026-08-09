"""Rate limiter behaviour, including the two ways it fails silently.

Every guard here was made to fire before it was trusted: a limiter that never
refuses, and a limiter that refuses everyone at once, both look like "the
limiter is installed" from the outside.
"""

from __future__ import annotations

import pytest
from app.ratelimit import RateLimiter, client_identity
from fastapi import HTTPException


class _Clock:
    """Injected time, so a window is tested at an exact boundary, not a slept one."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    def __init__(self, host: str | None = "1.2.3.4", headers: dict | None = None) -> None:
        self.client = _FakeClient(host) if host is not None else None
        self.headers = headers or {}


def test_a_fresh_client_is_never_refused_its_first_request():
    limiter = RateLimiter(capacity=3, window_seconds=60.0, clock=_Clock())
    allowed, retry_after = limiter.check("a")
    assert allowed and retry_after == 0.0


def test_the_bucket_empties_and_then_refuses():
    clock = _Clock()
    limiter = RateLimiter(capacity=3, window_seconds=60.0, clock=clock)
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]

    allowed, retry_after = limiter.check("a")
    assert not allowed
    # One token at 3 per 60 s is 20 s, and the client is told so rather than
    # left to guess — a 429 without Retry-After invites a tight retry loop.
    assert retry_after == pytest.approx(20.0)


def test_it_refills_continuously_rather_than_on_a_window_boundary():
    """A fixed window lets a caller spend twice the allowance across the edge."""
    clock = _Clock()
    limiter = RateLimiter(capacity=2, window_seconds=60.0, clock=clock)
    limiter.check("a")
    limiter.check("a")
    assert not limiter.check("a")[0]

    clock.advance(30.0)  # half a window -> exactly one token
    assert limiter.check("a")[0]
    assert not limiter.check("a")[0]


def test_it_never_accrues_more_than_capacity_while_idle():
    clock = _Clock()
    limiter = RateLimiter(capacity=2, window_seconds=60.0, clock=clock)
    clock.advance(10_000.0)
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, False]


def test_clients_do_not_share_a_bucket():
    """The failure this pins: keying every caller on one identity throttles all."""
    limiter = RateLimiter(capacity=1, window_seconds=60.0, clock=_Clock())
    assert limiter.check("a")[0]
    assert not limiter.check("a")[0]
    assert limiter.check("b")[0], "a second client must not inherit the first's spend"


def test_idle_buckets_are_evicted_so_the_map_cannot_grow_without_bound():
    clock = _Clock()
    limiter = RateLimiter(capacity=5, window_seconds=60.0, clock=clock)
    for i in range(50):
        limiter.check(f"client-{i}")
    assert len(limiter._buckets) == 50

    # Past the idle horizon, and past the eviction interval so the sweep runs.
    clock.advance(1000.0)
    limiter.check("fresh")
    assert len(limiter._buckets) == 1, "idle buckets are a memory-exhaustion vector"


def test_a_degenerate_configuration_raises_rather_than_disabling_the_limiter():
    with pytest.raises(ValueError, match="capacity"):
        RateLimiter(capacity=0)
    with pytest.raises(ValueError, match="window_seconds"):
        RateLimiter(window_seconds=0.0)


def test_an_untrusted_forwarded_header_is_ignored(monkeypatch):
    """Believing X-Forwarded-For unconditionally hands out a bucket per request."""
    monkeypatch.delenv("TRUSTED_CLIENT_IP_HEADER", raising=False)
    request = _FakeRequest(host="10.0.0.1", headers={"X-Forwarded-For": "9.9.9.9"})
    assert client_identity(request) == "10.0.0.1"


def test_the_configured_header_is_trusted_and_only_its_first_entry(monkeypatch):
    monkeypatch.setenv("TRUSTED_CLIENT_IP_HEADER", "X-Forwarded-For")
    request = _FakeRequest(host="10.0.0.1", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})
    # Everything after the proxy's own append is caller-controlled.
    assert client_identity(request) == "9.9.9.9"


def test_an_unidentifiable_caller_shares_one_bucket_rather_than_bypassing(monkeypatch):
    monkeypatch.delenv("TRUSTED_CLIENT_IP_HEADER", raising=False)
    assert client_identity(_FakeRequest(host=None)) == "unknown"


def test_the_dependency_raises_429_with_a_retry_after_header(monkeypatch):
    import app.ratelimit as rl

    monkeypatch.setattr(rl, "_limiter", RateLimiter(capacity=1, window_seconds=60.0))
    request = _FakeRequest(host="1.2.3.4")
    rl.rate_limit(request)  # first call spends the only token

    with pytest.raises(HTTPException) as excinfo:
        rl.rate_limit(request)
    assert excinfo.value.status_code == 429
    assert int(excinfo.value.headers["Retry-After"]) >= 1


def test_it_can_be_turned_off_explicitly(monkeypatch):
    import app.ratelimit as rl

    monkeypatch.setattr(rl, "_limiter", None)
    for _ in range(100):
        rl.rate_limit(_FakeRequest(host="1.2.3.4"))
