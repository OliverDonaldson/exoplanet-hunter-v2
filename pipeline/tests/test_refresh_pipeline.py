"""The weekly refresh flow's promotion step, and what it reports.

The flow runs unattended from launchd, so the webhook message is the entire
report. Anything the verdict fails to distinguish there is indistinguishable
for good — nobody is reading the Prefect logs on a Saturday morning.

**No test here runs the real gate.** `promotion_gate` builds its command with
`--promote` hard-wired, so `subprocess.run` is replaced in every test below and
the registry is never a participant.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import pytest

# The flow is a Prefect flow and imports `prefect` at module scope. Prefect
# lives in the optional `orchestration` extra — deliberately out of the core
# install, since nothing but this flow needs it — so a plain `pipeline[dev]`
# environment cannot import the module at all. Same treatment the vendored
# TRICERATOPS tests get for `pipeline[validation]`: skip, rather than fail a
# whole collection run on an extra that was never meant to be present.
pytest.importorskip("prefect", reason="pipeline[orchestration] not installed")


def _flow():
    """The flow as a module. It is not importable as a package path."""
    spec = importlib.util.spec_from_file_location(
        "_refresh_pipeline",
        Path(__file__).resolve().parents[2] / "orchestration" / "flows" / "refresh_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = _flow()


@pytest.fixture
def gate(tmp_path: Path, monkeypatch):
    """Run `promotion_gate` against a fake repository and a fake gate process.

    Returns a `run(verdict, *, exit_code=None, reasons=..., alarms=...)` that
    drives the task and hands back both the decision it returned and the command
    it would have executed.
    """
    monkeypatch.setattr(flow, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(flow, "get_run_logger", lambda: logging.getLogger("test"))
    cv_root = tmp_path / "models" / "cv" / "candidate"
    cv_root.mkdir(parents=True)
    (cv_root / "cv_summary.json").write_text("{}")

    recorded: dict[str, list[str]] = {}

    def run(verdict, *, exit_code=None, reasons=("because",), alarms=(), write=True):
        from exoplanet_hunter.validation import VERDICT_EXIT_CODES, Verdict

        code = VERDICT_EXIT_CODES[Verdict(verdict)] if exit_code is None else exit_code

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = list(cmd)
            if write:
                out = Path(cmd[cmd.index("--verdict-out") + 1])
                out.write_text(
                    json.dumps(
                        {"verdict": verdict, "reasons": list(reasons), "alarms": list(alarms)}
                    )
                )
            return subprocess.CompletedProcess(cmd, code)

        # The task gates whichever summary is newest, and a test that writes a
        # champion after the candidate would silently gate the champion instead.
        summary = cv_root / "cv_summary.json"
        os.utime(summary, (summary.stat().st_atime, time.time() + 10))
        monkeypatch.setattr(flow.subprocess, "run", fake_run)
        return flow.promotion_gate.fn(control_summary(tmp_path)), recorded

    return run


def control_summary(root: Path) -> Path:
    """What the control lane hands the gate: the served model, measured now."""
    path = root / "models" / "cv" / "control-lane" / "cv_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {}, "per_mission": {"TESS": {"roc_auc": 0.91}}}))
    return path


@pytest.fixture
def lane(tmp_path: Path, monkeypatch):
    """Run `control_lane` against a fake repository and a fake lane process.

    Returns `run(exit_code, *, write=True)`, giving back whatever the task
    returned — a path, or None where the lane refused.
    """
    monkeypatch.setattr(flow, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(flow, "get_run_logger", lambda: logging.getLogger("test"))
    out = tmp_path / "models" / "cv" / "control-lane" / "cv_summary.json"

    def run(exit_code: int, *, write: bool = True):
        def fake_run(cmd, **kwargs):
            if write:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"per_mission": {"TESS": {"roc_auc": 0.91}}}))
            return subprocess.CompletedProcess(cmd, exit_code)

        monkeypatch.setattr(flow.subprocess, "run", fake_run)
        return flow.control_lane.fn()

    return run


def rebaselined(root: Path) -> None:
    """The standing fallback summary, measured on the current view set."""
    path = root / "models" / "cv" / "incumbent-rebaselined" / "cv_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {}, "per_mission": {"TESS": {"roc_auc": 0.91}}}))


# --------------------------------------------------------------------------
# The verdict reaches the caller intact.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["PROMOTE", "REJECT", "UNRESOLVED"])
def test_every_verdict_survives_the_subprocess_boundary(gate, verdict):
    decision, _ = gate(verdict)
    assert decision.verdict.value == verdict


def test_the_reasons_come_back_with_the_verdict(gate):
    """The verdict alone cannot say why. A candidate that lost on recall and one
    whose population could not be matched are both REJECT, and the difference is
    the only actionable part of the message."""
    decision, _ = gate("REJECT", reasons=["recall fell", "populations differ"])
    assert decision.reasons == ["recall fell", "populations differ"]


def test_alarms_come_back_too(gate):
    decision, _ = gate("PROMOTE", alarms=["Kepler AUC fell"])
    assert decision.alarms == ["Kepler AUC fell"]


def test_the_verdict_is_written_somewhere_that_outlives_the_flow(gate, tmp_path):
    """This pointed at a TemporaryDirectory, so the gate computed a full verdict
    with its reasons every week and deleted it — `/runs` had nothing to serve and
    the console said so on every row. The flow must name the candidate's own run
    directory, which is where the API looks."""
    from exoplanet_hunter.validation import PROMOTION_LOG_NAME

    _, recorded = gate("REJECT")
    out = Path(recorded["cmd"][recorded["cmd"].index("--verdict-out") + 1])
    assert out == tmp_path / "models" / "cv" / "candidate" / PROMOTION_LOG_NAME
    assert out.is_file()


def test_a_stale_log_from_a_previous_gating_is_not_read_as_this_run_s_verdict(gate, tmp_path):
    """The durable path persists between runs, unlike the tempdir it replaced.
    A leftover file would satisfy the existence check that is the flow's only
    way to tell a decision from a gate that died before reaching one."""
    from exoplanet_hunter.validation import PROMOTION_LOG_NAME

    stale = tmp_path / "models" / "cv" / "candidate" / PROMOTION_LOG_NAME
    stale.write_text(json.dumps({"verdict": "PROMOTE", "reasons": ["last week"], "alarms": []}))
    with pytest.raises(RuntimeError, match="without reaching a verdict"):
        gate("REJECT", exit_code=1, write=False)


def test_only_promote_reads_as_a_promotion(gate):
    """`promoted` is what decides whether the message claims a new model is
    served, and UNRESOLVED must not reach it."""
    assert gate("PROMOTE")[0].promoted is True
    assert gate("REJECT")[0].promoted is False
    assert gate("UNRESOLVED")[0].promoted is False


# --------------------------------------------------------------------------
# A gate that never decided is not a rejection.
# --------------------------------------------------------------------------


def test_a_gate_that_crashed_before_deciding_raises_rather_than_reporting_a_rejection(gate):
    """An uncaught exception exits 1, which is also REJECT's code. Reading the
    code alone reported every crash as a candidate that lost on quality."""
    with pytest.raises(RuntimeError, match="without reaching a verdict"):
        gate("REJECT", exit_code=1, write=False)


def test_the_crash_says_in_words_that_it_is_not_a_quality_rejection(gate):
    with pytest.raises(RuntimeError, match="NOT a quality rejection"):
        gate("REJECT", exit_code=3, write=False)


def test_a_decision_that_failed_to_apply_is_not_reported_as_the_decision(gate):
    """The gate decided PROMOTE and then died updating the registry. Reporting
    PROMOTE would claim a model is served that is not."""
    with pytest.raises(RuntimeError, match="failed to apply"):
        gate("PROMOTE", exit_code=1)


def test_a_crash_is_notified_before_it_raises(gate, monkeypatch):
    """launchd watches nothing. A task that raises in silence is a week with no
    result and no notice, so the webhook has to hear about it first."""
    sent: list[str] = []
    monkeypatch.setattr(flow, "_notify", sent.append)
    with pytest.raises(RuntimeError):
        gate("REJECT", exit_code=1, write=False)
    assert sent and "NOT a quality rejection" in sent[0]


# --------------------------------------------------------------------------
# How each verdict reads in the one message anybody sees.
# --------------------------------------------------------------------------


def test_unresolved_does_not_read_as_a_rejection():
    headline = flow._gate_headline("UNRESOLVED")
    assert "UNRESOLVED" in headline
    assert "NOT a quality rejection" in headline


def test_the_three_headlines_are_distinct():
    headlines = {flow._gate_headline(v) for v in ("PROMOTE", "REJECT", "UNRESOLVED")}
    assert len(headlines) == 3


def test_every_verdict_the_library_defines_has_a_headline():
    """A verdict added without a headline would raise inside the notification
    step, after the gate has already run and possibly promoted."""
    from exoplanet_hunter.validation import Verdict

    for verdict in Verdict:
        assert flow._gate_headline(verdict.value)


# --------------------------------------------------------------------------
# The gate is read against the control lane, not a stored summary. The lane
# re-derives the champion from the registry every run, so the reference follows
# each promotion — the property the old conditional fallback existed to protect,
# now held by construction instead of by a branch.
# --------------------------------------------------------------------------


def test_the_gate_is_always_given_the_lane_s_measurement(gate, tmp_path):
    """Unconditional, where a stored summary had to be conditional. A stored one
    was pinned to a single model measured on a single past population, so passing
    it always would freeze the comparison. This is regenerated every run."""
    _, recorded = gate("REJECT")
    assert "--champion-summary" in recorded["cmd"]
    passed = recorded["cmd"][recorded["cmd"].index("--champion-summary") + 1]
    assert passed.endswith("control-lane/cv_summary.json")


def test_the_lane_s_own_summary_is_never_selected_as_the_candidate(gate, tmp_path):
    """The lane writes into models/cv/ like every run does. Picked up by the
    newest-summary glob it would be gated against itself — a guaranteed dead heat
    reported as a decision. Written newest here precisely because mtime ordering
    is what would have made this bite."""
    control = control_summary(tmp_path)
    os.utime(control, (control.stat().st_atime, time.time() + 3600))
    _, recorded = gate("REJECT")
    assert not recorded["cmd"][2].startswith("models/cv/control-lane")
    assert recorded["cmd"][2] == "models/cv/candidate/cv_summary.json"


def test_nothing_gateable_at_all_is_a_failure_rather_than_a_verdict(tmp_path, monkeypatch):
    """If the only summary present is the lane's, there is no candidate. Raising
    beats gating the control against itself and calling the tie a result."""
    monkeypatch.setattr(flow, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(flow, "get_run_logger", lambda: logging.getLogger("test"))
    control = control_summary(tmp_path)
    with pytest.raises(RuntimeError, match="nothing to gate"):
        flow.promotion_gate.fn(control)


def test_the_stale_rebaselined_summary_is_no_longer_reachable(gate, tmp_path):
    """The 2026-08-07 measurement must not come back as a silent substitute. It
    stays on disk — recorded commands name it — but nothing routes to it."""
    rebaselined(tmp_path)
    _, recorded = gate("REJECT")
    assert not any("incumbent-rebaselined" in str(arg) for arg in recorded["cmd"])


# --------------------------------------------------------------------------
# A lane that refused is not a lane that broke.
# --------------------------------------------------------------------------


def test_a_lane_that_produced_a_summary_hands_back_its_path(lane):
    assert lane(0) is not None


def test_a_refusal_is_reported_as_no_reference_rather_than_a_crash(lane):
    """Too thin to measure is a verdict. It shares UNRESOLVED's exit code because
    it is UNRESOLVED, reached one step before the gate."""
    assert lane(2) is None


def test_a_lane_that_crashed_does_not_read_as_a_refusal(lane):
    """The distinction the gate's own exit codes could not make: a fault and a
    decision must not arrive as the same non-zero."""
    with pytest.raises(RuntimeError, match="crashed rather than deciding"):
        lane(1, write=False)


def test_a_crashed_lane_says_it_is_not_a_quality_rejection(lane):
    with pytest.raises(RuntimeError, match="NOT a quality rejection"):
        lane(3, write=False)


def test_exit_zero_with_no_summary_is_a_failure_not_last_week_s_control(lane):
    """The dangerous case: succeed, write nothing, and let every later step read
    whatever an earlier run left at that path and call it this week's control."""
    with pytest.raises(RuntimeError, match="wrote no summary"):
        lane(0, write=False)


# --------------------------------------------------------------------------
# What the unattended run is invoked with.
# --------------------------------------------------------------------------


def test_the_flow_still_promotes_autonomously(gate):
    """The loop deciding for itself is the point of the project. If this ever
    fails because --promote was removed, the weekly run has become a report."""
    _, recorded = gate("PROMOTE")
    assert "--promote" in recorded["cmd"]


def test_the_flow_gates_strictly(gate):
    """An alarm owes a written explanation before promotion and nobody is here
    to give one, so advisory alarms would be promoted straight past."""
    _, recorded = gate("PROMOTE")
    assert "--strict" in recorded["cmd"]
