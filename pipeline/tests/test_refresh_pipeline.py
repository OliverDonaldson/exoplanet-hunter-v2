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
        return flow.promotion_gate.fn(), recorded

    return run


def champion(root: Path, run_id: str, *, sliceable: bool) -> None:
    """Point the registry at a served run, with or without a per_mission block."""
    summary: dict = {"summary": {}}
    if sliceable:
        summary["per_mission"] = {"TESS": {"roc_auc": 0.91}}
    cv = root / "models" / "cv" / run_id
    cv.mkdir(parents=True, exist_ok=True)
    (cv / "cv_summary.json").write_text(json.dumps(summary))
    (root / "models" / "registry.json").write_text(json.dumps({"run_id": run_id}))


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
# The champion follows the registry. This is the property that keeps the weekly
# comparison moving: --champion-summary is a FALLBACK for a served model whose
# summary cannot be sliced, not a fixed reference. Passed unconditionally it
# would freeze the comparison — promote a new model and the next candidate would
# still be gated against the old baseline for ever.
# --------------------------------------------------------------------------


def test_a_sliceable_champion_is_gated_against_directly(gate, tmp_path):
    """Once anything with a per_mission block is served, the override disengages
    permanently and the comparison tracks whatever the registry names."""
    champion(tmp_path, "served", sliceable=True)
    rebaselined(tmp_path)
    _, recorded = gate("REJECT")
    assert "--champion-summary" not in recorded["cmd"]


def test_a_champion_that_cannot_be_sliced_falls_back(gate, tmp_path):
    """The served model's summary predates the block, so gating against it
    directly would refuse every candidate on provenance before comparing one
    metric. The re-baselined measurement of that same model stands in."""
    champion(tmp_path, "served", sliceable=False)
    rebaselined(tmp_path)
    _, recorded = gate("REJECT")
    assert "--champion-summary" in recorded["cmd"]
    fallback = recorded["cmd"][recorded["cmd"].index("--champion-summary") + 1]
    assert fallback.endswith("incumbent-rebaselined/cv_summary.json")


def test_no_registry_at_all_still_falls_back(gate, tmp_path):
    rebaselined(tmp_path)
    _, recorded = gate("REJECT")
    assert "--champion-summary" in recorded["cmd"]


def test_nothing_to_fall_back_to_does_not_silently_pin_a_reference(gate, tmp_path):
    """No registry and no re-baseline. The gate will refuse on population
    mismatch, which is correct — what must not happen is a reference being
    invented to get a decision out of it."""
    _, recorded = gate("REJECT")
    assert "--champion-summary" not in recorded["cmd"]


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
