"""Tests for `eval.scoring` — the run-scoring layer and its protocol invariant.

`eval/scoring.py` states in its own module docstring that mixing out-of-fold and
zero-shot scores across one population is a comparability defect. That invariant
was enforced nowhere until 2026-08-08, which is the class of defect this project
keeps finding: a stated rule with no code behind it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from exoplanet_hunter.eval.scoring import Protocol, summarise_scored


def scored(**overrides) -> pd.DataFrame:
    """A scored frame: TESS and Kepler out-of-fold, K2 zero-shot."""
    rows = {
        "tic_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
        "score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        "mission": ["TESS"] * 4 + ["Kepler"] * 2 + ["K2"] * 2,
        "protocol": [Protocol.OUT_OF_FOLD] * 6 + [Protocol.ZERO_SHOT] * 2,
    }
    return pd.DataFrame({**rows, **overrides})


def test_slices_are_out_of_fold_only_and_zero_shot_is_reported_apart():
    result = summarise_scored(scored(), source="test")
    assert set(result["per_mission"]) == {"TESS", "Kepler", "all"}
    assert result["per_mission"]["all"]["n"] == 6
    assert set(result["zero_shot"]) == {"K2"}
    assert result["zero_shot"]["K2"]["n"] == 2


def test_a_zero_shot_block_of_pure_negatives_cannot_reach_the_gate():
    """The live case, measured 2026-08-08: the re-baselined incumbent carries
    243 zero-shot Kepler rows with no positives at all. Pooling them into the
    out-of-fold Kepler slice measures a population no model was asked about."""
    frame = scored()
    frame.loc[frame["mission"] == "Kepler", "protocol"] = Protocol.ZERO_SHOT
    frame.loc[frame["mission"] == "Kepler", "label"] = 0

    result = summarise_scored(frame, source="test")
    assert "Kepler" not in result["per_mission"]
    assert result["zero_shot"]["Kepler"]["n_positive"] == 0
    assert result["per_mission"]["all"]["n"] == 4


def test_an_unresolved_mission_is_refused_rather_than_dropped():
    frame = scored()
    frame.loc[0, "mission"] = None
    with pytest.raises(ValueError, match="carry no mission"):
        summarise_scored(frame, source="test")


def test_unresolved_rows_can_be_excluded_but_are_recorded():
    frame = scored()
    frame.loc[0, "mission"] = None
    result = summarise_scored(frame, source="test", exclude_unresolved=True)
    assert result["provenance"]["excluded_unresolved"] == [1]
    assert result["per_mission"]["all"]["n"] == 5


def test_the_aggregate_must_equal_the_missions_that_make_it_up():
    result = summarise_scored(scored(), source="test")
    per_mission = result["per_mission"]
    counted = sum(v["n"] for k, v in per_mission.items() if k != "all")
    assert counted == per_mission["all"]["n"]


def test_a_frame_with_no_held_out_rows_cannot_gate():
    frame = scored()
    frame["protocol"] = Protocol.ZERO_SHOT
    with pytest.raises(ValueError, match="not a held-out measurement"):
        summarise_scored(frame, source="test")


def test_a_frame_without_the_gate_mission_is_refused():
    frame = scored()
    frame = frame[frame["mission"] != "TESS"]
    with pytest.raises(ValueError, match="cannot gate"):
        summarise_scored(frame, source="test")


def test_a_missing_column_names_itself():
    with pytest.raises(ValueError, match=r"\['protocol'\]"):
        summarise_scored(scored().drop(columns=["protocol"]), source="test")


def test_the_summary_block_the_pooled_fallback_reads_is_present():
    result = summarise_scored(scored(), source="test")
    assert set(result["summary"]) == {"test_roc_auc", "test_brier", "test_ece"}
    assert result["summary"]["test_roc_auc"]["mean"] == result["per_mission"]["all"]["roc_auc"]


def test_no_folds_block_so_pairing_reports_nothing_rather_than_something_wrong():
    """The incumbent's folds are a different partition from any candidate's, so
    pairing on fold index would compare fold k of one split against fold k of
    another. `paired_folds` returns None on a missing block."""
    from exoplanet_hunter.validation.promotion import paired_folds

    result = summarise_scored(scored(), source="test")
    assert "folds" not in result
    assert paired_folds({"folds": [{"test_roc_auc": 0.9}]}, result) is None
