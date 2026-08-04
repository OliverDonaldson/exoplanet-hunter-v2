"""The falsifiable test for stage 2(b): does the score track transits or time?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.eval.observation_bias import ObservationBias, measure_observation_bias


def make_index(n: int = 200, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = rng.integers(1, 40, n)
    expected = observed + rng.integers(0, 400, n)
    return pd.DataFrame(
        {
            "observed_transit_count": observed,
            "expected_transit_count": expected,
            "transit_completeness": observed / expected,
        }
    )


def test_a_model_scoring_the_baseline_is_caught():
    # The measured 2026-07-26 failure: +0.211 with baseline, -0.003 with transit
    # count. A score that is just the baseline must show exactly that shape.
    index = make_index()
    bias = measure_observation_bias(index["expected_transit_count"].to_numpy(dtype=float), index)
    assert bias.baseline_sensitivity > 0.9
    assert abs(bias.transit_sensitivity) < 0.3


def test_a_model_scoring_transit_count_reads_the_other_way():
    index = make_index()
    bias = measure_observation_bias(index["observed_transit_count"].to_numpy(dtype=float), index)
    assert bias.transit_sensitivity > 0.9


def test_improvement_needs_both_directions_to_move():
    before = ObservationBias(
        transit_sensitivity=-0.003, baseline_sensitivity=0.211, completeness_sensitivity=0.0, n=100
    )
    fixed = ObservationBias(
        transit_sensitivity=0.42, baseline_sensitivity=0.05, completeness_sensitivity=0.3, n=100
    )
    assert fixed.improved_over(before)

    # Better transit sensitivity but the baseline shortcut intact is not a fix.
    half = ObservationBias(
        transit_sensitivity=0.42, baseline_sensitivity=0.30, completeness_sensitivity=0.3, n=100
    )
    assert not half.improved_over(before)
    # ...and neither is dropping the baseline while still ignoring transits.
    other_half = ObservationBias(
        transit_sensitivity=0.01, baseline_sensitivity=0.02, completeness_sensitivity=0.0, n=100
    )
    assert not other_half.improved_over(before)


def test_constant_scores_give_nan_not_a_spurious_zero():
    # A degenerate model has no measurable sensitivity; reporting 0.0 would read
    # as "correctly ignores the baseline".
    index = make_index()
    bias = measure_observation_bias(np.full(len(index), 0.5), index)
    assert np.isnan(bias.transit_sensitivity)
    assert np.isnan(bias.baseline_sensitivity)


def test_length_mismatch_is_an_error():
    index = make_index(n=10)
    with pytest.raises(ValueError):
        measure_observation_bias(np.zeros(9), index)


def test_missing_completeness_column_is_nan_not_a_crash():
    index = make_index().drop(columns=["transit_completeness"])
    bias = measure_observation_bias(np.random.default_rng(1).random(len(index)), index)
    assert np.isnan(bias.completeness_sensitivity)
    assert np.isfinite(bias.transit_sensitivity)
