"""The control lane's arithmetic and its refusals.

`scripts/control_lane.py` is the driver — loading weights, scoring shards. What
is here is everything that decides whether the resulting number may be used:
which rows can be compared, when the slice is too thin to read, whether the
re-score is measuring the model it claims, and how a delta is split into the
part the gate may attribute to the model and the part it may not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from exoplanet_hunter.eval.control_lane import (
    MIN_GATE_ROWS,
    assert_gateable,
    deltas,
    outweighed_by_data,
    population_overlap,
    reproduces,
    rows_reproduce,
)


def summary(**tess) -> dict:
    base = {"n": 2367, "n_positive": 1300, "roc_auc": 0.91, "recall_at_1pct_fpr": 0.307}
    return {"per_mission": {"TESS": {**base, **tess}}}


# --------------------------------------------------------------------------
# Which rows can be compared at all.
# --------------------------------------------------------------------------


def test_the_overlap_splits_the_current_population_three_ways():
    overlap = population_overlap({1, 2, 3, 4}, {3, 4, 5})
    assert (overlap.shared, overlap.added, overlap.dropped) == (2, 1, 2)
    assert overlap.current == 3


def test_coverage_is_the_fraction_of_today_the_comparison_reaches():
    """The shared set is pinned to what the champion trained on while the
    catalogue grows, so this falls over time. It is reported every run rather
    than discovered once a margin is being read against a quarter of the data."""
    assert population_overlap({1, 2, 3, 4}, {1, 2, 3, 4}).covered == 1.0
    assert population_overlap({1, 2}, {1, 2, 3, 4}).covered == 0.5
    assert population_overlap(set(), set()).covered == 0.0


def test_rows_the_champion_never_saw_are_counted_as_added_not_shared():
    """They are excluded from the gating comparison on purpose: scoring them
    would average every fold and hand the champion an ensemble the candidate
    does not get, on exactly the rows a refresh adds."""
    overlap = population_overlap({1, 2}, {1, 2, 3, 4, 5})
    assert overlap.shared == 2
    assert overlap.added == 3


# --------------------------------------------------------------------------
# When the slice is too thin to decide on.
# --------------------------------------------------------------------------


def test_a_healthy_slice_passes():
    """Returns nothing and raises nothing; the refusal below is the behaviour."""
    assert_gateable(summary())


def test_a_slice_under_the_floor_refuses_rather_than_deciding_on_the_remainder():
    with pytest.raises(ValueError, match="rather than narrowing"):
        assert_gateable(summary(n=MIN_GATE_ROWS - 1))


def test_the_refusal_names_the_size_it_found():
    with pytest.raises(ValueError, match=str(MIN_GATE_ROWS - 1)):
        assert_gateable(summary(n=MIN_GATE_ROWS - 1))


def test_a_summary_with_no_gate_slice_cannot_stand_in_for_the_champion():
    with pytest.raises(ValueError, match="no TESS slice"):
        assert_gateable({"per_mission": {"Kepler": {"n": 5000}}})


# --------------------------------------------------------------------------
# Is the re-score measuring the model the registry names?
# --------------------------------------------------------------------------


def test_an_exact_reproduction_reports_nothing():
    assert reproduces(summary(), summary()) == []


def test_a_drifted_metric_is_named_with_both_values():
    drifted = reproduces(summary(roc_auc=0.92), summary(roc_auc=0.91))
    assert len(drifted) == 1
    assert "roc_auc" in drifted[0] and "0.92" in drifted[0] and "0.91" in drifted[0]


def test_floating_point_noise_is_not_drift():
    assert reproduces(summary(roc_auc=0.91 + 1e-9), summary(roc_auc=0.91)) == []


def test_nothing_in_common_is_refused_rather_than_passing_vacuously():
    """Zero comparable metrics would otherwise return an empty list, which reads
    as "reproduces exactly" — a check that passes because it ran on nothing."""
    with pytest.raises(ValueError, match="cannot run"):
        reproduces({"per_mission": {"TESS": {}}}, {"per_mission": {"TESS": {}}})


def test_a_slice_outside_the_gate_is_checked_too():
    """4.1c looked at TESS alone. A lane that agreed there and disagreed on
    Kepler would have passed a check on the gate slice and still been wrong
    about what it computes."""
    drifted = reproduces(
        {"per_mission": {"TESS": {"roc_auc": 0.91}, "Kepler": {"roc_auc": 0.95}}},
        {"per_mission": {"TESS": {"roc_auc": 0.91}, "Kepler": {"roc_auc": 0.88}}},
    )
    assert len(drifted) == 1
    assert drifted[0].startswith("Kepler.roc_auc")


# --------------------------------------------------------------------------
# The same check at the row level, which is the half with teeth.
# --------------------------------------------------------------------------


def scored(tic_ids=(1, 2, 3), scores=(0.1, 0.5, 0.9), folds=(0, 1, 2), labels=(0, 1, 1)):
    return pd.DataFrame(
        {"tic_id": list(tic_ids), "score": list(scores), "fold": list(folds), "label": list(labels)}
    )


def test_two_identical_scorings_agree_row_for_row():
    assert rows_reproduce(scored(), scored()) == []


def test_floating_point_noise_is_not_a_row_disagreement():
    assert rows_reproduce(scored(scores=(0.1 + 1e-9, 0.5, 0.9)), scored()) == []


def test_one_moved_row_is_caught_and_located():
    """The case slice means hide: a single object scored differently by the two
    paths while the averages stay within tolerance."""
    problems = rows_reproduce(scored(scores=(0.1, 0.5, 0.8)), scored())
    assert len(problems) == 1
    assert "1 of 3 rows" in problems[0] and "tic_id 3" in problems[0]


def test_a_row_only_one_path_scored_is_named():
    problems = rows_reproduce(
        scored(tic_ids=(1, 2), scores=(0.1, 0.5), folds=(0, 1), labels=(0, 1)), scored()
    )
    assert len(problems) == 1
    assert "missing from the lane" in problems[0]


def test_a_row_scored_by_a_different_fold_is_a_disagreement_even_at_the_same_score():
    """Same number, different measurement. An out-of-fold score from a fold that
    trained on the row is not the quantity the gate thinks it is reading."""
    problems = rows_reproduce(scored(folds=(0, 1, 3)), scored())
    assert len(problems) == 1
    assert "disagree on fold" in problems[0]


def test_a_row_measured_against_a_different_label_is_a_disagreement():
    """Both paths read ground truth from the same frozen index, so a label that
    differs means they are not reading the same index."""
    problems = rows_reproduce(scored(labels=(1, 1, 1)), scored())
    assert len(problems) == 1
    assert "disagree on label" in problems[0]


def test_no_shared_rows_is_refused_rather_than_passing_vacuously():
    with pytest.raises(ValueError, match="cannot run"):
        rows_reproduce(scored(tic_ids=(1, 2, 3)), scored(tic_ids=(4, 5, 6)))


# --------------------------------------------------------------------------
# Splitting the delta into the model's part and the data's.
# --------------------------------------------------------------------------


def test_the_model_effect_compares_the_two_models_on_one_population():
    model, data = deltas(summary(roc_auc=0.93), summary(roc_auc=0.91))
    assert model["roc_auc"] == pytest.approx(0.02)
    assert data is None


def test_the_data_effect_compares_one_model_against_two_populations():
    model, data = deltas(
        summary(roc_auc=0.93), summary(roc_auc=0.91), previous=summary(roc_auc=0.88)
    )
    assert model["roc_auc"] == pytest.approx(0.02)
    assert data is not None
    assert data["roc_auc"] == pytest.approx(0.03)


def test_row_counts_are_not_reported_as_a_delta():
    """`n` differencing is a population statement. Beside AUC and recall it
    would read as a fourth score."""
    model, _ = deltas(summary(), summary())
    assert "n" not in model and "n_positive" not in model


def test_a_population_that_moved_further_than_the_model_is_named():
    outweighed = outweighed_by_data({"roc_auc": 0.002}, {"roc_auc": 0.03})
    assert len(outweighed) == 1
    assert "roc_auc" in outweighed[0]


def test_a_model_that_moved_further_than_the_population_is_not():
    assert outweighed_by_data({"roc_auc": 0.03}, {"roc_auc": 0.002}) == []


def test_the_first_run_has_no_data_effect_to_be_outweighed_by():
    assert outweighed_by_data({"roc_auc": 0.002}, None) == []


def test_the_sign_of_the_data_effect_does_not_excuse_its_size():
    """A population that moved further in the model's favour is the same problem
    as one that moved against it: the headline number is the model's and the
    larger part of it is not."""
    assert outweighed_by_data({"roc_auc": 0.002}, {"roc_auc": -0.03})
