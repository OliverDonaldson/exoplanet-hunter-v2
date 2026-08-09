"""Offline control-arm harness.

Each guard here is made to fire. The failures this module has to prevent all
return a plausible number: a threshold fitted on the wrong population, a matched
draw quietly backfilled from the easy stratum, a host scored by the fold that
trained on it, and a non-finite score averaged into a pass rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.eval.comparison import recall_at_fpr
from exoplanet_hunter.eval.control_arm import (
    SHORTLIST_FPR,
    baseline_matched_hosts,
    control_arm_rate,
    f1_optimal_threshold,
    fold_assignment,
    operating_points,
    threshold_at_fpr,
)


def _separable(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2))
    scores = np.where(labels == 1, rng.normal(0.7, 0.15, n), rng.normal(0.3, 0.15, n))
    return labels, np.clip(scores, 0.0, 1.0)


def test_the_threshold_and_the_recall_at_1pct_fpr_describe_the_same_cut():
    """Pins this module's cut against the gate's own statistic in comparison.py."""
    labels, scores = _separable()
    cut = threshold_at_fpr(labels, scores, SHORTLIST_FPR)

    achieved_tpr = float((scores[labels == 1] >= cut).mean())
    achieved_fpr = float((scores[labels == 0] >= cut).mean())
    assert achieved_fpr <= SHORTLIST_FPR + 1e-12, "the cut must respect the budget it names"
    assert achieved_tpr == pytest.approx(recall_at_fpr(labels, scores, SHORTLIST_FPR), abs=1e-9)


def test_a_threshold_needs_both_classes_rather_than_returning_something():
    labels, scores = _separable()
    with pytest.raises(ValueError, match="both classes"):
        threshold_at_fpr(np.ones_like(labels), scores, SHORTLIST_FPR)
    with pytest.raises(ValueError, match="length mismatch"):
        threshold_at_fpr(labels[:10], scores, SHORTLIST_FPR)


def test_the_f1_cut_beats_its_neighbours_on_f1():
    labels, scores = _separable()
    cut = f1_optimal_threshold(labels, scores)

    def f1_at(t: float) -> float:
        predicted = scores >= t
        tp = float(np.sum(predicted & (labels == 1)))
        if tp == 0 or predicted.sum() == 0:
            return 0.0
        precision, recall = tp / predicted.sum(), tp / labels.sum()
        return 2 * precision * recall / (precision + recall)

    best = f1_at(cut)
    assert best >= max(f1_at(cut - 0.05), f1_at(cut + 0.05))


def test_the_two_operating_points_differ_and_the_shortlist_one_is_stricter():
    """If they collapsed to one number, reporting both would be theatre."""
    labels, scores = _separable()
    frame = pd.DataFrame({"mission": "TESS", "label": labels, "score": scores})
    points = operating_points(frame)
    assert points.shortlist > points.f1_optimal
    assert points.n == len(frame) and points.n_positive == int(labels.sum())


def test_the_operating_point_is_derived_from_the_gating_mission_alone():
    """A Kepler-weighted threshold is set by a population with no serving stake."""
    labels, tess = _separable(seed=1)
    kepler = np.where(labels == 1, 0.99, 0.01)  # trivially separable, unlike TESS
    frame = pd.DataFrame(
        {
            "mission": ["TESS"] * len(labels) + ["Kepler"] * len(labels),
            "label": np.concatenate([labels, labels]),
            "score": np.concatenate([tess, kepler]),
        }
    )
    assert operating_points(frame).shortlist == pytest.approx(
        operating_points(frame[frame["mission"] == "TESS"]).shortlist
    )
    with pytest.raises(ValueError, match="no K2 rows"):
        operating_points(frame, mission="K2")


def _host_pool(n_per_cell: int = 6) -> pd.DataFrame:
    """Planet hosts with long baselines, FP hosts with short ones — the real bias."""
    rows = []
    tic = 1
    for label, centre in ((1, 1500.0), (0, 430.0)):
        for _ in range(n_per_cell * 4):
            rows.append({"tic_id": tic, "label": label, "baseline_days": centre})
            tic += 1
    frame = pd.DataFrame(rows)
    # Spread within each label so quantile bins are not degenerate.
    frame["baseline_days"] += np.linspace(0, 1200, len(frame))
    return frame


def test_matching_draws_equal_numbers_of_each_label_in_every_stratum():
    matched = baseline_matched_hosts(_host_pool(), per_label_per_stratum=2, n_strata=4)
    counts = matched.hosts.pivot_table(
        index="stratum", columns="label", values="tic_id", aggfunc="size", fill_value=0
    )
    assert (counts[0] == counts[1]).all(), "a stratum must not be label-imbalanced"
    assert matched.n == 2 * counts.sum().sum() / 2


def test_a_stratum_without_both_labels_is_dropped_and_counted_not_backfilled():
    """Backfilling returns a clean number about an easier population."""
    pool = _host_pool()
    # Make the longest-baseline stratum planet-only, which is how the real bias
    # presents: nothing observed for 2,500 d is a dispositioned false positive.
    pool.loc[pool["baseline_days"] > pool["baseline_days"].quantile(0.75), "label"] = 1
    matched = baseline_matched_hosts(pool, per_label_per_stratum=2, n_strata=4)

    assert matched.n_strata_dropped >= 1
    assert matched.n_strata_used + matched.n_strata_dropped == 4
    counts = matched.hosts.pivot_table(
        index="stratum", columns="label", values="tic_id", aggfunc="size", fill_value=0
    )
    assert (counts[0] == counts[1]).all(), "the surviving strata stay matched"
    assert "dropped" in matched.report()


def test_matching_refuses_to_invent_a_baseline_it_was_not_given():
    """labels.parquet has no expected_transit_count; a silent default is the bug."""
    pool = _host_pool().drop(columns=["baseline_days"])
    with pytest.raises(KeyError, match="expected_transit_count"):
        baseline_matched_hosts(pool, per_label_per_stratum=2)


def test_matching_derives_baseline_days_from_the_ephemeris_when_not_supplied():
    pool = _host_pool().drop(columns=["baseline_days"])
    # Alternating rather than in row order: `_host_pool` is label-blocked, so a
    # monotone assignment gives every planet host a short baseline and every FP
    # a long one, and *both* strata come out single-label. That is a real
    # matcher refusing a perfectly-separated pool, not a bug — but it tests the
    # dropping path, which the test above already owns.
    pool["expected_transit_count"] = np.tile([2.0, 20.0, 40.0, 60.0], len(pool) // 4)
    pool["period"] = 10.0
    matched = baseline_matched_hosts(pool, per_label_per_stratum=1, n_strata=2)
    assert matched.n > 0
    assert (matched.hosts["baseline_days"] > 0).all()

    with pytest.raises(KeyError, match="tic_id"):
        baseline_matched_hosts(pool.drop(columns=["tic_id"]), per_label_per_stratum=1)


def test_matching_rejects_a_degenerate_configuration():
    with pytest.raises(ValueError, match="per_label_per_stratum"):
        baseline_matched_hosts(_host_pool(), per_label_per_stratum=0)
    with pytest.raises(ValueError, match="n_strata"):
        baseline_matched_hosts(_host_pool(), per_label_per_stratum=1, n_strata=0)


def test_matching_is_reproducible_from_its_seed():
    a = baseline_matched_hosts(_host_pool(), per_label_per_stratum=2, seed=7)
    b = baseline_matched_hosts(_host_pool(), per_label_per_stratum=2, seed=7)
    c = baseline_matched_hosts(_host_pool(), per_label_per_stratum=2, seed=8)
    assert sorted(a.hosts["tic_id"]) == sorted(b.hosts["tic_id"])
    assert sorted(a.hosts["tic_id"]) != sorted(c.hosts["tic_id"])


def test_fold_routing_maps_each_host_to_the_fold_that_held_it_out():
    predictions = pd.DataFrame({"tic_id": [10, 11, 12], "fold": [0, 3, 1], "label": [1, 0, 1]})
    assert fold_assignment(predictions) == {10: 0, 11: 3, 12: 1}


def test_a_host_in_two_folds_raises_rather_than_picking_one():
    """Ambiguous routing would silently score a host with a fold that trained on it."""
    predictions = pd.DataFrame({"tic_id": [10, 10], "fold": [0, 2], "label": [1, 1]})
    with pytest.raises(ValueError, match="more than one fold"):
        fold_assignment(predictions)
    with pytest.raises(KeyError, match="fold"):
        fold_assignment(predictions.drop(columns=["fold"]))


def test_the_pass_rate_splits_by_label_because_the_headline_conflates_two():
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.95, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])
    rate = control_arm_rate(scores, labels, threshold=0.5, threshold_name="shortlist")

    assert rate.overall == pytest.approx(3 / 6)
    assert rate.planet_hosts == pytest.approx(2 / 3)
    assert rate.fp_hosts == pytest.approx(1 / 3)
    assert rate.split == pytest.approx(1 / 3)
    assert rate.as_dict()["threshold_name"] == "shortlist"


def test_a_non_finite_score_raises_instead_of_being_averaged_into_a_rate():
    scores = np.array([0.9, np.nan, 0.2])
    with pytest.raises(ValueError, match="non-finite"):
        control_arm_rate(scores, np.array([1, 0, 1]), threshold=0.5, threshold_name="shortlist")

    with pytest.raises(ValueError, match="no scored hosts"):
        control_arm_rate(np.array([]), np.array([]), threshold=0.5, threshold_name="shortlist")

    with pytest.raises(ValueError, match="3 scores but 2 labels"):
        control_arm_rate(
            np.array([0.1, 0.2, 0.3]), np.array([1, 0]), threshold=0.5, threshold_name="s"
        )
