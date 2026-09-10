"""`--live` asks the deployed API whether it can serve the console; this checks
it asks for the right things.

The probe used to request `/health`, which no version of this API has ever
declared, so `make ready-live` reported the deployment broken whether it was
current or five commits behind — a check that always fails carries no
information, and the one time it mattered (the console shipped ahead of the
API on 2026-09-11, leaving a panel reading "Not Found") it said nothing new.

Same shape as `test_client_contract.py` and for the same reason: the console
and the API are deployed separately, so anything the console assumes about the
API has to be asserted somewhere that runs. Both directions are checked, plus
a floor, because an extraction that silently matches nothing would pass.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
from app.main import app

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "frontend" / "design-console" / "src"
_GATE = _ROOT / "pipeline" / "scripts" / "check_showcase_ready.py"
_BUILD = _ROOT / "frontend" / "design-console" / "build.py"

#: The console fetched six endpoints at load time when this was written. The
#: floor is well under that, so ordinary edits do not trip it and a regex that
#: stops matching does.
_MIN_ENDPOINTS = 4

#: Fetched on demand from the Upload page, not at load, and a cold call runs a
#: MAST pull and a BLS search — minutes. A readiness gate cannot wait for it.
_ON_DEMAND = "/score/"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("check_showcase_ready", _GATE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The script is `from __future__ import annotations`, so @dataclass resolves
    # its field types through sys.modules and raises if the module is not there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _console_paths() -> set[str]:
    """Every `apiFetch('/x')` and ``apiFetch(`/x?y=${z}`)`` in the client."""
    found = set()
    for path in sorted(_SRC.glob("app.*.js")):
        for raw in re.findall(r"apiFetch\(\s*[`'\"]([^`'\"]+)[`'\"]", path.read_text()):
            found.add(raw.split("?")[0])
    return found


def _declared_paths() -> set[str]:
    return set(app.openapi()["paths"])


def test_the_probe_list_is_not_empty(gate):
    assert len(gate.CONSOLE_ENDPOINTS) >= _MIN_ENDPOINTS
    assert len(_console_paths()) >= _MIN_ENDPOINTS


def test_every_probed_path_is_a_route_the_api_declares(gate):
    declared = _declared_paths()
    unknown = [p for p in gate.CONSOLE_ENDPOINTS if p.split("?")[0] not in declared]
    assert not unknown, f"probed but not served: {unknown}; declared: {sorted(declared)}"


def test_every_load_time_console_call_is_probed(gate):
    probed = {p.split("?")[0] for p in gate.CONSOLE_ENDPOINTS}
    missed = [p for p in _console_paths() if not p.startswith(_ON_DEMAND) and p not in probed]
    assert not missed, f"the console loads these and --live never asks for them: {missed}"


def test_a_missing_endpoint_fails_the_check(gate, monkeypatch):
    absent = f"{gate.CONSOLE_ENDPOINTS[-1]}"
    monkeypatch.setattr(
        gate,
        "_get",
        lambda url, timeout=25, retry_slow=False: (200, "{}") if absent not in url else (404, ""),
    )
    assert not gate.check_api_live().ok


def test_all_endpoints_answering_passes_the_check(gate, monkeypatch):
    monkeypatch.setattr(gate, "_get", lambda url, timeout=25, retry_slow=False: (200, "{}"))
    assert gate.check_api_live().ok


class _Calls:
    """A stand-in for _get that records what it was asked and answers a script."""

    def __init__(self, *answers):
        self.answers, self.urls = list(answers), []

    def __call__(self, url, timeout=25, retry_slow=False):
        self.urls.append(url)
        return self.answers[min(len(self.urls) - 1, len(self.answers) - 1)]


def test_a_slow_first_probe_is_retried_once(gate, monkeypatch):
    tries = []

    def boom(url, timeout=25):
        tries.append(url)
        raise TimeoutError("the read operation timed out")

    monkeypatch.setattr(gate.urllib.request, "urlopen", boom)
    status, _ = gate._get("https://example.invalid/healthz", retry_slow=True)
    assert status == 0 and len(tries) == 2


def test_a_refused_connection_is_not_retried(gate, monkeypatch):
    tries = []

    def refused(url, timeout=25):
        tries.append(url)
        raise OSError("connection refused")

    monkeypatch.setattr(gate.urllib.request, "urlopen", refused)
    gate._get("https://example.invalid/healthz", retry_slow=True)
    assert len(tries) == 1


def test_only_the_first_endpoint_pays_the_resume(gate, monkeypatch):
    seen = []

    def spy(url, timeout=25, retry_slow=False):
        seen.append(retry_slow)
        return 200, "{}"

    monkeypatch.setattr(gate, "_get", spy)
    gate.check_api_live()
    assert seen[0] is True and not any(seen[1:])


def test_a_head_with_no_og_image_fails_the_preview_check(gate, monkeypatch):
    monkeypatch.setattr(gate, "_get", _Calls((200, "<head><title>x</title></head>")))
    assert not gate.check_link_preview().ok


def test_an_og_image_that_does_not_answer_fails_the_check(gate, monkeypatch):
    page = '<meta property="og:image" content="https://example.test/og.png">'
    monkeypatch.setattr(gate, "_get", _Calls((200, page), (404, "")))
    assert not gate.check_link_preview().ok


def test_a_card_that_answers_passes_the_check(gate, monkeypatch):
    page = '<meta property="og:image" content="https://example.test/og.png">'
    calls = _Calls((200, page), (200, "PNG"))
    monkeypatch.setattr(gate, "_get", calls)
    assert gate.check_link_preview().ok
    assert calls.urls[1] == "https://example.test/og.png"


def test_the_gate_reads_the_tag_the_console_actually_writes(gate, monkeypatch):
    """The check greps the served HTML, so a reformatted tag in build.py would
    silently stop matching and the card would go unchecked."""
    emitted = '\'<meta property="og:image" content="{console_url}/og.png">\''
    assert emitted in _BUILD.read_text()

    page = '<meta property="og:image" content="https://example.test/og.png">'
    monkeypatch.setattr(gate, "_get", _Calls((200, page), (200, "PNG")))
    assert gate.check_link_preview().ok
