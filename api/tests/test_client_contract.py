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
from app.main import app
from app.schemas import CandidateRow, ScoreResponse, TrainingFoldHistory
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parents[2] / "frontend" / "design-console" / "src"
_CLIENT = _SRC / "app.api.js"
#: The Model page reads /model/training-history field by field. It lives in
#: the page file rather than the client, so it is extracted separately —
#: `foldHistory` is spelled out for exactly this reason, since a one-letter
#: loop variable would make the accessor regex match half the file.
_PAGES = _SRC / "app.pages.js"

#: Below this, the extraction has broken rather than the client having got
#: smaller. It read 31 catalogue fields when this was written; the floor is set
#: well under that so ordinary edits do not trip it, and a regex that silently
#: stops matching does.
_MIN_ROW_FIELDS = 20


def _accessed(prefix: str, path: Path = _CLIENT) -> set[str]:
    source = path.read_text()
    return set(re.findall(rf"\b{prefix}\.([a-z][a-z0-9_]*)\b", source))


@pytest.mark.skipif(not _CLIENT.exists(), reason="console source not in this checkout")
def test_client_reads_only_declared_catalogue_fields():
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


def test_score_response_carries_what_the_vetting_page_needs():
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


@pytest.mark.skipif(not _PAGES.exists(), reason="console source not in this checkout")
def test_the_model_page_reads_only_declared_history_fields():
    """The Training History panel maps a fold record field by field.

    This panel replaced a tile that told visitors the data did not exist. A
    field renamed here would put it straight back to rendering nothing, which
    is the one failure mode worth a contract test on this route in particular.
    """
    accessed = _accessed("foldHistory", _PAGES)
    assert accessed, (
        "no foldHistory.<field> accesses found in app.pages.js; the extraction has "
        "broken, and a contract test that reads nothing passes for the wrong reason"
    )
    declared = set(TrainingFoldHistory.model_fields)
    missing = sorted(accessed - declared)
    assert not missing, (
        f"the Model page reads {missing} off a training-history fold and "
        "TrainingFoldHistory does not declare them"
    )
    # The two the panel cannot render without: the curve and the epoch it marks.
    assert {"val_auc", "restored_epoch"} <= accessed


#: Every page that renders a figure from SERVED, CANDIDATES or GATING. When the
#: probe falls back to the stand-in set these show prototype numbers, and each
#: one has to say so — the Model page's TESS recall reads 0.6120 against the
#: served 0.3113, and the Catalogue carries a K2 card the served run cannot
#: have. Issue #75.
_FIGURE_PAGES = {
    "app.pages.js": ("About", "ModelPerformance", "Vetting"),
    "app.home.js": ("Home", "Catalogue"),
}


@pytest.mark.skipif(not _SRC.exists(), reason="console source not in this checkout")
def test_every_page_that_shows_a_figure_labels_prototype_data():
    """A page rendering the stand-in set unlabelled is a wrong number, not a gap.

    The console gives /healthz about 16 s before falling back, and a Fly machine
    returning from a full stop is allowed 180 s. The window in between is what a
    first visitor lands in, so the label is the only thing standing between them
    and a measurement that was never made.
    """
    helper = (_SRC / "app.data.js").read_text()
    assert "const prototypeNote =" in helper, (
        "prototypeNote has been renamed or removed from app.data.js; the pages "
        "below call it and this test would otherwise pass by finding nothing"
    )
    assert "API.mode === 'live' ? ''" in helper, (
        "prototypeNote no longer gates on API.mode, so it renders in a live "
        "session or never renders at all"
    )

    for filename, pages in _FIGURE_PAGES.items():
        source = (_SRC / filename).read_text()
        for page in pages:
            body = re.search(rf"\nfunction {page}\(.*?\n(?=\nfunction |\Z)", source, re.S)
            assert body, f"{page}() not found in {filename}; this test is reading nothing"
            assert "prototypeNote(" in body.group(), (
                f"{page}() renders figures without calling prototypeNote(); a "
                "visitor arriving during a cold start reads the stand-in set as "
                "the served run"
            )


def test_a_rejected_tic_detail_is_a_list_the_client_flattens():
    """FastAPI's 422 `detail` is a list, and `new Error(list)` is "[object Object]".

    That string is what the Upload page printed for a malformed TIC, so this
    pins both halves: the wire shape that makes flattening necessary, and the
    client still having the flattener. Issue #73.
    """
    body = TestClient(app).get("/score/notanumber").json()
    assert isinstance(body["detail"], list), (
        "422 detail is no longer a list; if FastAPI changed this, detailText still "
        "handles it, but the comment explaining why it exists is now wrong"
    )
    assert body["detail"][0]["msg"], "the 422 entry carries no msg to render"

    client_src = _CLIENT.read_text()
    assert "const detailText =" in client_src, (
        "detailText has gone from app.api.js; a list detail will render as "
        "'[object Object]' on the Upload page again"
    )
    assert "new Error(detailText(" in client_src, (
        "apiFetch no longer routes the error body through detailText"
    )
