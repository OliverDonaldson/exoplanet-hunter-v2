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
        gate, "_get", lambda url, timeout=25: (200, "{}") if absent not in url else (404, "")
    )
    assert not gate.check_api_live().ok


def test_all_endpoints_answering_passes_the_check(gate, monkeypatch):
    monkeypatch.setattr(gate, "_get", lambda url, timeout=25: (200, "{}"))
    assert gate.check_api_live().ok
