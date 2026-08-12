"""Propensity weighting and stratified negative sampling — every guard fired.

The failure both interventions share is doing nothing quietly: weights that are
all 1.0, a "stratified" sample that is the original rows reordered. Training then
runs, the arm is recorded against the pre-registration, and nothing was ever
intervened upon. So most of what is tested here is the modules noticing that
about themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.datasets.baseline_bias import (
    MAX_WEIGHT_RATIO,
    Reweighting,
    _strata,
    propensity_weights,
    stratified_negative_sample,
)
from exoplanet_hunter.eval.observation_bias import _spearman, baseline_days


def confounded(n: int = 2000, strength: float = 0.9, seed: int = 0) -> pd.DataFrame:
    """A catalogue with the real defect in it: long baseline -> more likely positive.

    `strength` is the probability that a long-baseline target is labelled
    positive, so it dials the confound from severe (0.9) down to absent (0.5).
    """
    rng = np.random.default_rng(seed)
    period = rng.uniform(1.0, 20.0, n)
    expected = rng.integers(2, 300, n)
    span = (expected - 1) * period
    long_lived = span > np.median(span)
    p = np.where(long_lived, strength, 1.0 - strength)
    return pd.DataFrame(
        {
            "tic_id": np.arange(n),
            "label": rng.binomial(1, p),
            "period": period,
            "expected_transit_count": expected,
        }
    )


def correlation(frame: pd.DataFrame) -> float:
    return _spearman(frame["label"].to_numpy(float), baseline_days(frame))


def test_the_fixture_actually_carries_the_confound():
    """A test suite that reweights an uncorrelated population proves nothing."""
    assert correlation(confounded()) > 0.4


# --------------------------------------------------------- propensity weighting --


def test_weighting_removes_the_correlation_it_was_built_for():
    result = propensity_weights(confounded())
    assert abs(result.before) > 0.4
    assert abs(result.after) < 0.05
    assert result.removed > 0.35


def test_weights_are_normalised_to_mean_one():
    """Otherwise the effective learning rate moves with n_strata, and an arm
    that changed the step size gets read as an arm that changed the labels."""
    result = propensity_weights(confounded())
    assert result.weights.mean() == pytest.approx(1.0)


def test_no_weight_dominates_the_gradient():
    """An inverse propensity is unbounded as p approaches 0. One example
    carrying a thousand times its neighbours is a one-row training set."""
    # max_residual is relaxed because this is a test about the clip, not about
    # decorrelation: at strength=0.99 a stratum comes out single-class, which has
    # no invertible propensity, so the residual legitimately survives and the
    # decorrelation guard would fire first and mask what is being measured.
    result = propensity_weights(confounded(strength=0.99), max_residual=1.0)
    assert result.weights.max() <= MAX_WEIGHT_RATIO * np.median(result.weights) * 1.5


def test_the_clip_count_is_reported_rather_than_absorbed():
    result = propensity_weights(confounded(strength=0.99), max_residual=1.0)
    assert "clipped" in result.report()
    assert result.n_clipped >= 0


def test_weighting_that_did_nothing_raises_instead_of_training_on_ones():
    """The expensive failure: a run happens, an arm is recorded, and the weights
    were flat the whole time. Forced by asking for a residual of zero, which no
    finite weighting attains."""
    with pytest.raises(ValueError, match="not doing what the arm claims"):
        propensity_weights(confounded(), max_residual=0.0)


def test_weighting_refuses_a_single_class_population():
    frame = confounded().assign(label=1)
    with pytest.raises(ValueError, match="both labels"):
        propensity_weights(frame)


def test_weighting_refuses_to_silently_drop_rows_with_no_baseline():
    """Dropping them here changes the population the weights describe, and
    nothing downstream carries the count."""
    frame = confounded()
    frame.loc[:5, "period"] = np.nan
    with pytest.raises(ValueError, match="no finite baseline_days"):
        propensity_weights(frame)


def test_weighting_refuses_a_degenerate_stratum_count():
    with pytest.raises(ValueError, match="at least 2"):
        propensity_weights(confounded(), n_strata=1)


def test_a_single_class_stratum_is_left_unweighted_and_counted():
    """1/0 has no inverse. Left at weight 1 and excluded from n_strata_used,
    rather than dropped (which changes the population) or shrunk (which puts a
    fabricated rate into the gradient)."""
    result = propensity_weights(confounded(strength=1.0), max_residual=1.0)
    assert result.n_strata_used < 8


def test_weighting_is_reproducible():
    a = propensity_weights(confounded()).weights
    b = propensity_weights(confounded()).weights
    assert np.array_equal(a, b)


def test_the_weighted_statistic_matches_the_unweighted_one_at_equal_weights():
    """Pins `_weighted_spearman` against the module the roadmap's +0.3874 came
    from. If they disagree at weight 1, every reported 'after' is a different
    statistic from the 'before' it is compared with."""
    frame = confounded()
    flat = Reweighting(np.ones(len(frame)), 0.0, 0.0, 0, 0)
    from exoplanet_hunter.datasets.baseline_bias import _weighted_spearman

    labels = frame["label"].to_numpy(float)
    spans = baseline_days(frame)
    assert _weighted_spearman(labels, spans, flat.weights) == pytest.approx(
        _spearman(labels, spans), abs=1e-9
    )


# ------------------------------------------------------ stratified negative draw --


def test_sampling_removes_the_correlation_it_was_built_for():
    result = stratified_negative_sample(confounded())
    assert abs(result.before) > 0.4
    assert abs(result.after) < 0.05


def test_sampling_keeps_every_positive():
    """Dropping long-baseline positives buys a clean correlation by deleting the
    population the product exists to rank."""
    frame = confounded()
    result = stratified_negative_sample(frame)
    kept = frame.iloc[result.index]
    assert int(kept["label"].sum()) == int(frame["label"].sum())


def test_sampling_only_ever_discards_negatives():
    frame = confounded()
    result = stratified_negative_sample(frame)
    dropped = frame.drop(index=frame.index[result.index])
    assert (dropped["label"] == 0).all()


def test_the_discard_count_travels_with_the_result():
    frame = confounded()
    result = stratified_negative_sample(frame)
    assert result.n_dropped == len(frame) - len(result.index)
    assert "dropped" in result.report()


def test_sampling_that_did_nothing_raises():
    with pytest.raises(ValueError, match=r"left .* of baseline-label correlation"):
        stratified_negative_sample(confounded(), max_residual=0.0)


def test_sampling_refuses_a_single_class_population():
    with pytest.raises(ValueError, match="both labels"):
        stratified_negative_sample(confounded().assign(label=0))


def test_sampling_refuses_rows_with_no_baseline():
    frame = confounded()
    frame.loc[:5, "expected_transit_count"] = np.nan
    with pytest.raises(ValueError, match="no finite baseline_days"):
        stratified_negative_sample(frame)


def test_sampling_is_reproducible_from_its_seed():
    a = stratified_negative_sample(confounded(), seed=3).index
    b = stratified_negative_sample(confounded(), seed=3).index
    assert np.array_equal(a, b)


def test_a_different_seed_draws_a_different_negative_set():
    a = stratified_negative_sample(confounded(), seed=3).index
    b = stratified_negative_sample(confounded(), seed=4).index
    assert not np.array_equal(a, b)


def test_the_index_is_positional_and_in_range():
    """Returned as positions, not labels — a caller applying them with .loc on a
    non-default index would silently select the wrong rows."""
    frame = confounded()
    result = stratified_negative_sample(frame)
    assert result.index.min() >= 0
    assert result.index.max() < len(frame)
    assert len(np.unique(result.index)) == len(result.index)


def test_the_kept_ratio_is_equal_across_every_stratum():
    """The property the correlation check only sees the shadow of.

    The first implementation capped each stratum's negatives at its own positive
    count. That balances the strata where negatives dominate and leaves the
    others untouched: on this catalogue's real shape the long-baseline strata
    hold ~228 planets to ~22 false positives, so they stayed at a ratio of 0.10
    while the short-baseline strata were cut to 1.00, and P(label | baseline) was
    still a staircase. It moved the residual from +0.696 to +0.244 and stopped.

    Asserted on the ratio rather than only on the residual because a coarse
    stratification can hide a surviving staircase inside a correlation that
    happens to land under the limit.
    """
    frame = confounded()
    result = stratified_negative_sample(frame, n_strata=8)
    kept = frame.iloc[result.index]
    stratum = _strata(baseline_days(kept), 8)
    ratios = [
        (kept["label"].to_numpy()[stratum == v] == 0).sum()
        / max((kept["label"].to_numpy()[stratum == v] == 1).sum(), 1)
        for v in np.unique(stratum)
    ]
    assert max(ratios) - min(ratios) < 0.35


def test_a_stratum_missing_a_label_refuses_rather_than_balancing_at_zero():
    """The scarcest stratum sets the global ratio, so one with no negatives sets
    it to zero and would discard every negative in the catalogue while reporting
    a beautifully decorrelated population."""
    frame = confounded(n=400, strength=1.0)
    with pytest.raises(ValueError, match="cannot be balanced at any ratio"):
        stratified_negative_sample(frame, n_strata=8)
