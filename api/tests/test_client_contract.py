"""The console reads the wire contract; this checks it can.

The pinned contract used to be `api/app/schemas.py` mirrored by
`frontend/src/api/types.ts`, and that mirror was deleted with the React console
on 2026-09-04. The shipping client is
`frontend/design-console/src/app.api.js`, and nothing checked it against the
models — so a field renamed here would surface as a panel quietly rendering
"not measured", which is the one failure mode this console is built to avoid.

The check is deliberately crude: read the `row.<field>` and `body.<field>`
accesses out of the client with a regex and assert each names a field the
matching Pydantic model declares. A JavaScript parser would be more precise and
is not worth a dependency for a file that is one flat mapping function.

**What this must not become.** A test that finds no accesses and passes is
worse than no test, because it reads as coverage. Both directions are asserted
below: the extraction must find a plausible number of fields, and the client
must not reference one the model lacks.
"""

import re
from pathlib import Path

import pytest
from app.schemas import CandidateRow, ScoreResponse

_CLIENT = Path(__file__).resolve().parents[2] / "frontend" / "design-console" / "src" / "app.api.js"

#: Below this, the extraction has broken rather than the client having got
#: smaller. It read 31 catalogue fields when this was written; the floor is set
#: well under that so ordinary edits do not trip it, and a regex that silently
#: stops matching does.
_MIN_ROW_FIELDS = 20


def _accessed(prefix: str) -> set[str]:
    source = _CLIENT.read_text()
    return set(re.findall(rf"\b{prefix}\.([a-z][a-z0-9_]*)\b", source))


@pytest.mark.skipif(not _CLIENT.exists(), reason="console source not in this checkout")
def test_the_client_only_reads_catalogue_fields_the_contract_declares():
    """`mapCandidate` maps /candidates rows field by field."""
    accessed = _accessed("row")
    assert len(accessed) >= _MIN_ROW_FIELDS, (
        f"only {len(accessed)} row.<field> accesses found in {_CLIENT.name}; "
        "the extraction has broken, and a contract test that reads nothing passes "
        "for the wrong reason"
    )
    declared = set(CandidateRow.model_fields)
    missing = sorted(accessed - declared)
    assert not missing, (
        f"the console reads {missing} off a /candidates row and CandidateRow does not "
        "declare them — the panel that shows them will render 'not measured' forever"
    )


@pytest.mark.skipif(not _CLIENT.exists(), reason="console source not in this checkout")
def test_the_contract_test_would_notice_a_removed_field():
    """The guard on the guard.

    A field the client reads and the model has is the passing case above; this
    pins that the comparison is the one being made, by checking a field known
    to be on both sides really is on both sides. Without it, an accessor regex
    that matched nothing and a model that declared everything would look
    identical to a healthy contract.
    """
    accessed = _accessed("row")
    assert "baseline_days" in accessed, "the client should read the observation baseline"
    assert "baseline_days" in CandidateRow.model_fields


def test_the_score_response_still_carries_what_the_vetting_page_needs():
    """The Vetting page's panels each rest on one of these. They are named here
    rather than extracted because the client reaches them through `mapScore`'s
    nested shapes, which a flat regex cannot follow honestly."""
    required = {
        "prob_calibrated",
        "prob_mean",
        "prob_std",
        "per_fold",
        "decision_threshold",
        "centroid",
        "odd_even",
        "secondary",
        "duration_check",
        "false_alarms",
        "global_view",
        "local_view",
        "odd_view",
        "even_view",
        "centroid_track",
        "periodogram",
        "ephemeris",
        "n_mc_samples",
        "model_version",
    }
    missing = sorted(required - set(ScoreResponse.model_fields))
    assert not missing, f"ScoreResponse no longer carries {missing}"
