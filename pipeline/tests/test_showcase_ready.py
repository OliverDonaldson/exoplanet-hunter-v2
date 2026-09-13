"""`make ready`'s own checks — the gate this project trusts to answer one question.

Two failure modes, both of them a check asserting something it never measured.
A command that timed out or could not be started produced no verdict, and
"the tests failed" on that evidence is the category error the third promotion
verdict exists to avoid. The refresh-health check reports on something outside
the repository entirely, where the failure is an *absence*: launchd skips the
Saturday interval whenever the Mac is off, and a week that never ran leaves
nothing behind to look wrong.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


PASSING = "12 passed in 3.10s\n"
FAILING = "1 failed, 11 passed in 3.20s\n"


@pytest.fixture
def suites(monkeypatch):
    """Drive `check_tests` with a scripted result per pytest invocation.

    Returns `set(pipeline, api)`, each a `(code, output)` pair, in the order
    `check_tests` runs them.
    """

    def install(pipeline: tuple[int, str], api: tuple[int, str]) -> None:
        results = iter([pipeline, api])

        def fake_run(cmd, cwd=None, timeout=900):
            return next(results)

        monkeypatch.setattr(ready, "run", fake_run)

    return install


def test_a_timed_out_suite_is_not_reported_as_a_failure(suites):
    """Issue #79. The pipeline suite trips the deadline and the API suite passes;
    the old code filtered both outputs for a summary line, found only the API's,
    and printed "12 passed" as the detail of a failing check."""
    suites((ready.TIMED_OUT, "timed out after 900s\n"), (0, PASSING))
    result = ready.check_tests()
    assert not result.ok
    assert "did not finish" in result.detail
    assert "passed" not in result.detail


def test_a_timed_out_api_suite_is_caught_too(suites):
    """The pipeline suite is the slow one, so the reversed case is the one that
    would go unnoticed."""
    suites((0, PASSING), (ready.TIMED_OUT, "timed out after 900s\n"))
    result = ready.check_tests()
    assert not result.ok
    assert "did not finish" in result.detail
    assert "passed" not in result.detail


def test_a_missing_pytest_is_not_a_test_failure(suites):
    """A binary that is not installed measured nothing about the tests."""
    suites((ready.NOT_FOUND, "No such file or directory: 'pytest'"), (0, PASSING))
    result = ready.check_tests()
    assert not result.ok
    assert "could not run" in result.detail


def test_a_real_failure_still_reads_as_a_failure(suites):
    """The fix must not make the check unable to report the thing it is for."""
    suites((1, FAILING), (0, PASSING))
    result = ready.check_tests()
    assert not result.ok
    assert "1 failed" in result.detail


def test_two_green_suites_still_pass(suites):
    suites((0, PASSING), (0, PASSING))
    result = ready.check_tests()
    assert result.ok
    assert "passed" in result.detail


def test_timeout_and_not_found_are_distinct_from_exit_codes(suites):
    """Both were 127 before, which is also a real shell exit code for "command
    not found" — so a command that genuinely exited 127 was indistinguishable
    from one that timed out."""
    assert ready.TIMED_OUT != ready.NOT_FOUND
    assert ready.TIMED_OUT not in (0, 1, 127)
    assert ready.NOT_FOUND not in (0, 1, 127)


def test_no_verdict_is_silent_when_commands_ran(suites):
    """It must not intercept an ordinary non-zero exit."""
    assert ready._no_verdict("x", (0, "fine"), (1, "findings")) is None


def test_run_keeps_what_a_timed_out_command_said(monkeypatch):
    """The partial output is the only evidence of where it got to."""

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(
            cmd=["pytest"], timeout=900, output="collected 900 items\n", stderr=""
        )

    monkeypatch.setattr(ready.subprocess, "run", boom)
    code, out = ready.run(["pytest"])
    assert code == ready.TIMED_OUT
    assert "collected 900 items" in out
    assert "timed out after 900s" in out


def test_run_separates_a_missing_binary_from_a_timeout(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ruff")

    monkeypatch.setattr(ready.subprocess, "run", boom)
    code, _ = ready.run(["ruff"])
    assert code == ready.NOT_FOUND


@pytest.mark.parametrize(
    ("check", "name"),
    [
        ("check_lint", "ruff clean"),
        ("check_types", "mypy at or under baseline"),
        ("check_git_clean", "working tree is clean"),
    ],
)
def test_other_checks_do_not_misreport_a_timeout(monkeypatch, check, name):
    """The same conflation sat under every check that shells out. mypy's was the
    quietest: a timeout read as "could not read mypy output"."""
    monkeypatch.setattr(ready, "run", lambda *a, **k: (ready.TIMED_OUT, "timed out after 900s\n"))
    result = getattr(ready, check)()
    assert not result.ok
    assert result.name == name
    assert "did not finish" in result.detail


def test_the_test_timeout_clears_the_measured_runtime():
    """594 s measured on 2026-09-13. A deadline under it would make the timeout
    path the normal outcome, which is a different bug with the same symptom."""
    assert ready.TEST_TIMEOUT >= 2 * 594
