"""`make ready`'s own checks — the gate this project trusts to answer one question.

Only the refresh-health check is covered here. It is the one check that reports
on something outside the repository, and the failure it exists to catch is an
*absence*: launchd skips the Saturday interval whenever the Mac is off, and a
week that never ran leaves nothing behind to look wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def _module():
    """The gate is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "_check_showcase_ready",
        Path(__file__).resolve().parents[2] / "pipeline" / "scripts" / "check_showcase_ready.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and @dataclass
    # resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ready = _module()


@pytest.fixture
def host(tmp_path: Path, monkeypatch):
    """A host with the agent installed and a writable status path.

    Returns `write(state, age_days)`; not calling it leaves the status absent.
    """
    agent = tmp_path / "com.exoplanet-hunter.refresh.plist"
    agent.write_text("<plist/>")
    status = tmp_path / "refresh-status.json"
    monkeypatch.setattr(ready, "REFRESH_AGENT", agent)
    monkeypatch.setattr(ready, "REFRESH_STATUS", status)

    def write(state: str = "COMPLETED", age_days: float = 0.0, detail: str = "") -> None:
        stamp = datetime.now(UTC) - timedelta(days=age_days)
        status.write_text(
            json.dumps({"state": state, "detail": detail, "finished_at": stamp.isoformat()})
        )

    return write


def test_a_fresh_completed_run_passes(host):
    host("COMPLETED", 1.0)
    assert ready.check_refresh_healthy().ok


def test_a_failed_run_fails_the_gate(host):
    host("FAILED", 0.5, detail="dv-archive FAIL: 2 expected targets absent")
    result = ready.check_refresh_healthy()
    assert not result.ok
    assert "dv-archive" in result.detail


def test_a_run_older_than_the_limit_fails(host):
    """The missed-week case. 2026-09-12 never ran and nothing said so."""
    host("COMPLETED", ready.REFRESH_MAX_AGE_DAYS + 1)
    assert not ready.check_refresh_healthy().ok


def test_a_run_inside_the_limit_still_passes(host):
    host("COMPLETED", ready.REFRESH_MAX_AGE_DAYS - 1)
    assert ready.check_refresh_healthy().ok


def test_no_status_at_all_is_a_failure_not_a_pass(host):
    """An absent file is the shape a never-run agent leaves. Reading it as a
    pass would restore exactly the silence this check was added to break."""
    result = ready.check_refresh_healthy()
    assert not result.ok
    assert "no run has ever recorded" in result.detail


def test_unreadable_status_is_a_failure(host, tmp_path):
    (tmp_path / "refresh-status.json").write_text("{not json")
    assert not ready.check_refresh_healthy().ok


def test_status_missing_its_fields_is_a_failure(host, tmp_path):
    (tmp_path / "refresh-status.json").write_text('{"state": "COMPLETED"}')
    assert not ready.check_refresh_healthy().ok


def test_a_host_without_the_agent_is_not_judged(tmp_path, monkeypatch):
    """A stranger's clone has no refresh to be stale, and a red check there
    would train the reader to ignore this line."""
    monkeypatch.setattr(ready, "REFRESH_AGENT", tmp_path / "absent.plist")
    monkeypatch.setattr(ready, "REFRESH_STATUS", tmp_path / "absent.json")
    result = ready.check_refresh_healthy()
    assert result.ok
    assert "n/a" in result.detail


def test_the_check_is_in_the_default_run():
    """A check nothing calls is a document, not a gate."""
    source = (
        Path(__file__).resolve().parents[2] / "pipeline" / "scripts" / "check_showcase_ready.py"
    ).read_text()
    assert "        check_refresh_healthy,\n" in source
