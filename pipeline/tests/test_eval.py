"""Comparing two prediction sets, and the cuts that read the result.

These numbers decide whether a run promotes and whether stage 7 proceeds. A
silently wrong cut produces a plausible table, which is the failure mode this
project keeps meeting — so the guards against silence are tested as hard as the
arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from exoplanet_hunter.eval.comparison import (
    SliceMetrics,
    band_labels,
    compare_prediction_sets,
    gap_table,
    mission_coverage,
    paired_frame,
    recall_at_fpr,
)

BANDS = ((0, 10), (10, 100), (100, np.inf))


def frame(tic_ids, missions=None, *, labels=None, scores=None, column="score"):
    n = len(tic_ids)
    data = {
        "tic_id": tic_ids,
        "label": labels if labels is not None else [i % 2 for i in range(n)],
        column: scores if scores is not None else np.linspace(0.1, 0.9, n),
    }
    if missions is not None:
        data["mission"] = missions
    return pd.DataFrame(data)


def covariates(tic_ids, missions, transits, period=10.0, duration=0.2):
    return pd.DataFrame(
        {
            "tic_id": tic_ids,
            "mission": missions,
            "observed_transit_count": transits,
            "period": period,
            "duration": duration,
        }
    )


def probability(x):
    """Scores must be calibrated probabilities — Brier and ECE reject anything
    outside [0, 1], and a real `predictions.parquet` is Platt-scaled."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def separable(n, rng, strength):
    """Labels and calibrated scores whose AUC rises with `strength`."""
    labels = np.tile([0, 1], n // 2)
    return labels, probability(labels * strength + rng.normal(0, 1.0, n) - strength / 2)


# ------------------------------------------------------------------ coverage --


def test_a_mission_the_join_drops_is_named():
    """The stage 4 case: the champion predates K2, so an inner join keeps
    none of it while reporting a healthy 4,605 rows."""
    branches = frame(list(range(10)), ["K2"] * 4 + ["TESS"] * 3 + ["Kepler"] * 3)
    champion = frame(list(range(4, 10)))

    coverage = mission_coverage(branches, champion)

    assert coverage.dropped == ["K2"]
    assert coverage.shared == {"TESS": 3, "Kepler": 3}
    assert coverage.left_only == {"K2": 4}
    assert "DROPPED ENTIRELY: K2" in coverage.report()


def test_shared_weights_expose_the_sampling_artefact():
    coverage = mission_coverage(
        frame(list(range(10)), ["Kepler"] * 8 + ["TESS"] * 2), frame(list(range(10)))
    )
    assert coverage.shared_weights == {"Kepler": 0.8, "TESS": 0.2}


def test_mission_labels_come_from_whichever_side_has_them():
    """The champion's predictions.parquet carries no mission column."""
    coverage = mission_coverage(frame([1, 2], None), frame([1, 2], ["TESS", "K2"]))
    assert coverage.shared == {"TESS": 1, "K2": 1}


def test_rows_no_frame_can_label_are_counted_as_unknown():
    left = frame([1, 2, 3], ["TESS", "TESS", None])
    coverage = mission_coverage(left, frame([1, 2, 3]))
    assert coverage.shared == {"TESS": 2, "unknown": 1}


def test_coverage_needs_a_mission_column_somewhere():
    with pytest.raises(ValueError, match="mission"):
        mission_coverage(frame([1, 2]), frame([1, 2]))


def test_strict_refuses_a_comparison_that_lost_a_mission():
    left = frame(list(range(8)), ["K2"] * 4 + ["TESS"] * 4)
    right = frame(list(range(4, 8)))
    with pytest.raises(ValueError, match="dropped every row of K2"):
        compare_prediction_sets(left, right, strict=True)


def test_comparison_reports_every_mission_and_the_pool():
    rng = np.random.default_rng(0)
    ids = list(range(200))
    missions = ["TESS"] * 100 + ["Kepler"] * 100
    labels = [i % 2 for i in range(200)]
    left = frame(ids, missions, labels=labels, scores=rng.random(200))
    right = frame(ids, missions, labels=labels, scores=rng.random(200))

    result = compare_prediction_sets(left, right)

    assert set(result.left) == {"TESS", "Kepler", "all"}
    assert result.left["all"].n == 200
    assert set(result.auc_gaps()) == {"TESS", "Kepler", "all"}
    assert not result.coverage.dropped


def test_disagreeing_labels_are_an_error_not_a_silent_join():
    left = frame([1, 2, 3], ["TESS"] * 3, labels=[0, 1, 1])
    right = frame([1, 2, 3], ["TESS"] * 3, labels=[0, 1, 0])
    with pytest.raises(ValueError, match="disagree on labels"):
        compare_prediction_sets(left, right)


def test_a_single_class_slice_is_nan_not_a_crash():
    left = frame([1, 2], ["K2"] * 2, labels=[1, 1])
    right = frame([1, 2], ["K2"] * 2, labels=[1, 1])
    result = compare_prediction_sets(left, right)
    assert np.isnan(result.left["K2"].roc_auc)
    assert result.left["K2"].n == 2


# --------------------------------------------------------------- recall@FPR --


def test_recall_at_fpr_respects_the_budget_without_splitting_tied_scores():
    """Ten tied negatives outrank every positive, so the only operating points
    are FPR 0 and FPR 0.10. A count-based cut would admit one of the ten and
    report a recall no threshold achieves."""
    y = np.array([1] * 10 + [0] * 100)
    scores = np.concatenate([np.full(10, 0.5), np.full(10, 0.9), np.zeros(90)])
    assert recall_at_fpr(y, scores, 0.01) == 0.0
    assert recall_at_fpr(y, scores, 0.05) == 0.0
    assert recall_at_fpr(y, scores, 0.10) == 1.0


def test_recall_at_fpr_is_one_for_a_perfect_ranker():
    y = np.array([1] * 50 + [0] * 100)
    scores = np.concatenate([np.ones(50), np.zeros(100)])
    assert recall_at_fpr(y, scores, 0.01) == 1.0


def test_recall_at_fpr_needs_both_classes():
    assert np.isnan(recall_at_fpr(np.ones(5), np.linspace(0, 1, 5), 0.01))


def test_recall_at_fpr_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        recall_at_fpr(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), 0.01)


def test_slice_metrics_agree_with_reference_implementations():
    """Every headline number in the roadmap comes through SliceMetrics, so it is
    pinned against sklearn and against a brute-force sweep of every achievable
    threshold — not just against itself."""
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 500)
    s = probability(y * 1.5 + rng.normal(0, 1.0, 1000) - 0.75)
    measured = SliceMetrics.measure(y, s)

    assert measured.roc_auc == pytest.approx(roc_auc_score(y, s))
    assert measured.pr_auc == pytest.approx(average_precision_score(y, s))
    assert measured.brier == pytest.approx(brier_score_loss(y, s))

    for budget, got in (
        (0.01, measured.recall_at_1pct_fpr),
        (0.05, measured.recall_at_5pct_fpr),
        (0.10, measured.recall_at_10pct_fpr),
    ):
        reachable = [
            (s >= t)[y == 1].mean() for t in np.unique(s) if (s >= t)[y == 0].mean() <= budget
        ]
        assert got == pytest.approx(max(reachable))


# ------------------------------------------------------------- paired frames --


def test_paired_frame_normalises_score_names_whatever_they_arrived_as():
    """`gap_table` reads score_left/score_right, so the rename is the contract —
    a caller passing `prob` must not silently produce an empty table."""
    left = frame([1, 2, 3], ["TESS"] * 3, column="prob")
    right = frame([1, 2, 3], ["TESS"] * 3, column="prob")
    paired = paired_frame(left, right, score_column="prob")
    assert {"score_left", "score_right", "label"} <= set(paired.columns)


def test_paired_frame_keeps_only_shared_rows():
    paired = paired_frame(frame([1, 2, 3], ["TESS"] * 3), frame([2, 3, 4], ["TESS"] * 3))
    assert sorted(paired["tic_id"]) == [2, 3]


# -------------------------------------------------------------- band labels --


def test_bands_are_left_closed_and_ordered_low_to_high():
    labelled = band_labels(pd.Series([0, 9.999, 10, 99, 100, 5000]), BANDS)
    assert list(labelled) == ["0-10", "0-10", "10-100", "10-100", "100-+", "100-+"]
    assert list(labelled.categories) == ["0-10", "10-100", "100-+"]
    assert labelled.ordered


def test_a_value_outside_every_band_is_nan_not_the_first_bin():
    assert pd.isna(band_labels(pd.Series([-5.0]), BANDS)[0])


# --------------------------------------------------------------- gap tables --


def test_gap_table_reports_left_minus_right_per_group():
    ids = np.arange(200)
    labels = np.tile([0, 1], 100)
    perfect = probability(labels * 6.0 - 3.0)
    paired = paired_frame(
        frame(ids, ["Kepler"] * 200, labels=labels, scores=perfect),
        frame(ids, ["Kepler"] * 200, labels=labels, scores=probability(np.zeros(200))),
        covariates(ids, ["Kepler"] * 200, np.full(200, 5)),
    ).assign(band=lambda f: band_labels(f["observed_transit_count"], BANDS))

    table = gap_table(paired, "band")

    assert list(table["band"]) == ["0-10"]
    assert table.iloc[0]["left"] == pytest.approx(1.0)
    assert table.iloc[0]["gap"] == pytest.approx(0.5, abs=0.01)


def test_a_group_thinner_than_min_rows_is_nan_not_noise():
    ids = np.arange(200)
    labels, scores = separable(200, np.random.default_rng(2), 2.0)
    transits = np.where(np.arange(200) < 195, 5, 900)  # only 5 rows in the top band
    paired = paired_frame(
        frame(ids, ["Kepler"] * 200, labels=labels, scores=scores),
        frame(ids, ["Kepler"] * 200, labels=labels, scores=scores),
        covariates(ids, ["Kepler"] * 200, transits),
    ).assign(band=lambda f: band_labels(f["observed_transit_count"], BANDS))

    table = gap_table(paired, "band", min_rows=60)
    assert not np.isfinite(table.set_index("band").loc["100-+", "gap"])


def test_a_single_class_group_is_nan_not_a_spurious_auc():
    ids = np.arange(100)
    paired = paired_frame(
        frame(ids, ["Kepler"] * 100, labels=np.ones(100, int), scores=np.linspace(0, 1, 100)),
        frame(ids, ["Kepler"] * 100, labels=np.ones(100, int), scores=np.linspace(1, 0, 100)),
        covariates(ids, ["Kepler"] * 100, np.full(100, 5)),
    ).assign(band=lambda f: band_labels(f["observed_transit_count"], BANDS))

    assert not np.isfinite(gap_table(paired, "band")["gap"]).any()


def test_absolute_bands_put_both_missions_on_one_cut():
    """Quartiles are cut per mission, so the same rank means a different transit
    count on each. Fixed bands are what make the two comparable."""
    rng = np.random.default_rng(1)
    n = 800
    ids = np.arange(n)
    labels, scores = separable(n, rng, 2.0)
    transits = np.tile([5, 20, 50, 200, 500, 5, 20, 50], n // 8)
    missions = np.tile(["Kepler", "TESS"], n // 2)
    paired = paired_frame(
        frame(ids, missions, labels=labels, scores=scores),
        frame(ids, missions, labels=labels, scores=probability(rng.normal(0, 1, n))),
        covariates(ids, missions, transits),
    ).assign(band=lambda f: band_labels(f["observed_transit_count"], BANDS))

    table = gap_table(paired, ["band", "mission"], min_rows=10)

    assert set(table["mission"]) == {"Kepler", "TESS"}
    assert table["n"].sum() == n


def test_the_span_by_count_cross_separates_the_two_variables():
    """Two cells identical in span and differing only in transit count must land
    in different rows — that separation is what showed the gap tracks count."""
    rng = np.random.default_rng(4)
    n = 400
    ids = np.arange(n)
    labels = np.tile([0, 1], n // 2)
    transits = np.where(np.arange(n) < n // 2, 10, 500)
    perfect = probability(labels * 6.0 - 3.0)
    # The candidate is only bad on the many-transit half.
    degraded = np.where(transits < 100, perfect, probability(rng.normal(0, 1, n)))

    paired = paired_frame(
        frame(ids, ["Kepler"] * n, labels=labels, scores=perfect),
        frame(ids, ["Kepler"] * n, labels=labels, scores=degraded),
        covariates(ids, ["Kepler"] * n, transits, period=10.0, duration=0.1),
    ).assign(
        count_band=lambda f: band_labels(f["observed_transit_count"], ((0, 100), (100, np.inf)))
    )

    table = gap_table(paired, "count_band", min_rows=10).set_index("count_band")

    assert table.loc["0-100", "gap"] == pytest.approx(0.0, abs=0.01)
    assert table.loc["100-+", "gap"] > 0.3


def test_rows_without_a_usable_ephemeris_drop_out_rather_than_binning_at_zero():
    ids = np.arange(200)
    labels, scores = separable(200, np.random.default_rng(5), 2.0)
    cov = covariates(ids, ["Kepler"] * 200, np.full(200, 200), period=10.0, duration=0.1)
    cov.loc[:99, "period"] = np.nan

    paired = paired_frame(
        frame(ids, ["Kepler"] * 200, labels=labels, scores=scores),
        frame(ids, ["Kepler"] * 200, labels=labels, scores=scores),
        cov,
    )
    paired["span"] = paired["duration"] / paired["period"] * 301
    paired = paired[np.isfinite(paired["span"])].assign(
        span_band=lambda f: band_labels(f["span"], ((0, 4), (4, np.inf)))
    )

    table = gap_table(paired, "span_band", medians=["span"])
    assert table["n"].sum() == 100
    assert table.iloc[0]["median_span"] == pytest.approx(3.01, abs=0.01)
