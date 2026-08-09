"""Shared test setup for the API suite.

The rate limiter is disabled for every test *except* its own file. It is
process-global and stateful, so a suite that grows past the per-client budget
would start failing on a 429 in whichever test happened to be 21st — a failure
that points at the wrong route and reproduces only in full-suite order.
`test_ratelimit.py` installs its own limiter explicitly, so nothing is left
untested by this.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limit(request, monkeypatch):
    if request.node.path.name == "test_ratelimit.py":
        return
    import app.ratelimit as rl

    monkeypatch.setattr(rl, "_limiter", None)
