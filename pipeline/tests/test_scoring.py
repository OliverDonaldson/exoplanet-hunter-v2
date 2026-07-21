"""Tests for the scoring layer: ensemble aggregation and vetting diagnostics."""

import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.scoring import (
    FoldMember,
    ScoringEnsemble,
    false_alarm_checks,
    odd_even_depths,
    significant_secondary,
    unphysical_duration,
    verdict,
)
from exoplanet_hunter.training.calibration import TemperatureScaler


class ConstantModel:
    """Stands in for a Keras model under mc_dropout_predict."""

    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, inputs, training=False):
        first = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
        n = int(tf.shape(first)[0])
        return tf.fill((n, 1), tf.constant(self.p, dtype=tf.float32))


def member(fold: int, p: float, threshold: float, temperature: float) -> FoldMember:
    return FoldMember(
        fold=fold,
        model=ConstantModel(p),
        calibrator=TemperatureScaler(T=temperature),
        threshold=threshold,
        aux_pipeline=None,
        aux_dim=None,
    )


def test_ensemble_aggregation():
    ensemble = ScoringEnsemble(
        [member(0, 0.8, 0.2, 1.0), member(1, 0.6, 0.4, 1.0)], run_id="testrun"
    )
    pred = ensemble.predict(np.zeros(16, np.float32), np.zeros(8, np.float32), None, n_mc=5)

    # T=1 calibration is the identity, so per-fold == raw means.
    assert pred.per_fold == pytest.approx([0.8, 0.6], abs=1e-6)
    assert pred.prob_calibrated == pytest.approx(0.7, abs=1e-6)
    assert pred.prob_mean == pytest.approx(0.7, abs=1e-6)
    # Constant models have zero MC variance -> std is purely across-fold.
    assert pred.prob_std == pytest.approx(np.std([0.8, 0.6]), abs=1e-6)
    assert pred.threshold == pytest.approx(0.3, abs=1e-6)


def test_ensemble_calibration_applied():
    # T > 1 pulls probabilities toward 0.5 (overconfidence correction).
    ensemble = ScoringEnsemble([member(0, 0.9, 0.5, 2.0)], run_id="t")
    pred = ensemble.predict(np.zeros(4, np.float32), np.zeros(4, np.float32), None, n_mc=3)
    assert 0.5 < pred.per_fold[0] < 0.9
    assert pred.prob_mean == pytest.approx(0.9, abs=1e-6)  # raw mean unchanged


def test_ensemble_requires_aux_when_bundled():
    m = member(0, 0.5, 0.5, 1.0)
    m.aux_pipeline = object()  # anything non-None
    ensemble = ScoringEnsemble([m], run_id="t")
    with pytest.raises(ValueError, match="aux"):
        ensemble.predict(np.zeros(4, np.float32), np.zeros(4, np.float32), None)


# ------------------------------------------------------------- diagnostics --


def synthetic_transits(odd_depth: float, even_depth: float, n_periods: int = 40):
    """Box transits at P=2 d with alternating depths and mild noise."""
    rng = np.random.default_rng(0)
    time = np.arange(0, 2.0 * n_periods, 2.0 / 400)  # 400 cadences per period
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    phase = ((time + 1.0) % 2.0) - 1.0
    idx = np.round(time / 2.0).astype(int)
    in_tr = np.abs(phase) < 0.05
    flux[in_tr & (idx % 2 == 1)] -= odd_depth
    flux[in_tr & (idx % 2 == 0)] -= even_depth
    return time, flux


def test_odd_even_flags_alternating_depths():
    time, flux = synthetic_transits(0.010, 0.002)
    result = odd_even_depths(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.odd_depth_ppm == pytest.approx(10_000, rel=0.1)
    assert result.even_depth_ppm == pytest.approx(2_000, rel=0.2)
    assert result.depth_diff_sigma > 5


def test_odd_even_consistent_for_genuine_transit():
    time, flux = synthetic_transits(0.005, 0.005)
    result = odd_even_depths(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.depth_diff_sigma < 3


def synthetic_timed_transits(odd_shift: float, even_shift: float, n_periods: int = 40):
    """Equal-depth box transits at P=2 d whose midtimes are shifted per parity —
    an eccentric EB folded at half its true period."""
    rng = np.random.default_rng(1)
    time = np.arange(0, 2.0 * n_periods, 2.0 / 400)
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    for n in range(n_periods):
        center = 2.0 * n + (odd_shift if n % 2 else even_shift)
        flux[np.abs(time - center) < 0.05] -= 0.005
    return time, flux


def test_odd_even_flags_offset_timings():
    # ±0.02 d (~29 min) parity shifts; depths identical, so only timing fires.
    time, flux = synthetic_timed_transits(0.02, -0.02)
    result = odd_even_depths(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.depth_diff_sigma < 3
    assert result.timing_diff_sigma is not None and result.timing_diff_sigma > 10
    assert result.odd_timing_min is not None and result.odd_timing_min > 0
    assert result.even_timing_min is not None and result.even_timing_min < 0


def test_odd_even_timing_consistent_for_genuine_transit():
    time, flux = synthetic_timed_transits(0.0, 0.0)
    result = odd_even_depths(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.timing_diff_sigma is not None and result.timing_diff_sigma < 3


def test_batched_mc_samples_are_independent():
    from exoplanet_hunter.models.uncertainty import mc_dropout_predict

    class DropoutishModel:
        def __call__(self, inputs, training=False):
            first = next(iter(inputs.values()))
            n = int(tf.shape(first)[0])
            return tf.random.stateless_uniform((n, 1), seed=(n, 7))

    result = mc_dropout_predict(
        DropoutishModel(), {"global_view": np.zeros((1, 8, 1), np.float32)}, n_samples=32
    )
    assert result.samples.shape == (32,)
    assert len(np.unique(result.samples)) > 1  # one batched pass, distinct draws
    assert 0.0 <= result.mean <= 1.0


def test_verdict_language():
    assert "Strong planet candidate" in verdict(0.95, 0.3, centroid_snr=1.0, odd_even=None)
    assert "background-EB" in verdict(0.95, 0.3, centroid_snr=5.0, odd_even=None)
    assert "Unlikely" in verdict(0.05, 0.3, centroid_snr=1.0, odd_even=None)


# ----------------------------------------------- false-alarm bundle (BLS only) --


def test_false_alarms_sweet_flags_sinusoid():
    time = np.arange(0, 30, 0.01)
    rng = np.random.default_rng(3)
    flux = 1 + 5e-3 * np.sin(2 * np.pi * time / 3.0) + rng.normal(0, 1e-4, len(time))
    result = false_alarm_checks(time, flux, period=3.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.sweet_significance is not None and result.sweet_significance > 15
    assert result.sweet_suspicious
    assert result.suspicious


def test_false_alarms_quiet_for_genuine_transit():
    time, flux = synthetic_transits(0.005, 0.005)
    result = false_alarm_checks(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert not result.sweet_suspicious
    assert not result.asymmetry_suspicious
    assert not result.dmm_suspicious
    assert not result.gap_suspicious
    assert not result.suspicious


def test_false_alarms_flags_asymmetric_ramp():
    rng = np.random.default_rng(4)
    time = np.arange(0, 80, 2.0 / 400)
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    phase = ((time + 1.0) % 2.0) - 1.0
    in_tr = np.abs(phase) < 0.05
    flux[in_tr & (phase < 0)] -= 0.002
    flux[in_tr & (phase >= 0)] -= 0.010  # ramp-like: right side far deeper
    result = false_alarm_checks(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.asymmetry_sigma is not None and result.asymmetry_sigma > 10
    assert result.asymmetry_suspicious


def test_false_alarms_flags_outlier_depths():
    rng = np.random.default_rng(5)
    time = np.arange(0, 24, 2.0 / 400)
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    phase = ((time + 1.0) % 2.0) - 1.0
    idx = np.round(time / 2.0).astype(int)
    in_tr = np.abs(phase) < 0.05
    flux[in_tr] -= 0.005
    flux[in_tr & np.isin(idx, [3, 7])] -= 0.045  # two events dominate the mean
    result = false_alarm_checks(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.depth_mean_median_ratio is not None
    assert result.depth_mean_median_ratio > 1.5
    assert result.dmm_suspicious


def test_false_alarms_flags_transits_near_gaps():
    rng = np.random.default_rng(6)
    time = np.arange(0, 20, 0.005)
    # Carve a 0.35 d hole starting 0.15 d after every transit midtime.
    keep = (time % 1.0 < 0.15) | (time % 1.0 >= 0.5)
    time = time[keep]
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    phase = ((time + 0.5) % 1.0) - 0.5
    flux[np.abs(phase) < 0.05] -= 0.005
    result = false_alarm_checks(time, flux, period=1.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.gap_fraction is not None and result.gap_fraction >= 0.5
    assert result.gap_suspicious


def test_false_alarms_verdict_language():
    time = np.arange(0, 30, 0.01)
    flux = 1 + 5e-3 * np.sin(2 * np.pi * time / 3.0)
    result = false_alarm_checks(time, flux, period=3.0, t0=0.0, duration=0.1)
    text = verdict(0.9, 0.3, centroid_snr=None, odd_even=None, false_alarms=result)
    assert "low-trust BLS detection" in text
    assert "sinusoidal variability" in text


# ------------------------------------------------ significant secondary (§4.3) --


def synthetic_with_secondary(
    primary_depth: float,
    secondary_depth: float,
    period: float = 2.0,
    secondary_phase: float = 0.5,
    n_periods: int = 40,
):
    """Box primary at phase 0 plus an optional secondary dip elsewhere."""
    rng = np.random.default_rng(2)
    time = np.arange(0, period * n_periods, period / 400)
    flux = np.ones_like(time) + rng.normal(0, 1e-4, len(time))
    phase = (time / period) % 1.0
    flux[np.minimum(phase, 1 - phase) < 0.025] -= primary_depth
    if secondary_depth > 0:
        flux[np.abs(phase - secondary_phase) < 0.025] -= secondary_depth
    return time, flux


def test_secondary_flags_injected_eb_eclipse():
    time, flux = synthetic_with_secondary(0.01, 0.003)
    result = significant_secondary(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.secondary_phase == pytest.approx(0.5, abs=0.02)
    assert result.secondary_depth_ppm == pytest.approx(3_000, rel=0.2)
    assert result.secondary_significance > result.fa_threshold
    assert result.depth_ratio == pytest.approx(0.3, rel=0.2)
    assert result.suspicious


def test_secondary_f_red_near_unity_on_white_noise():
    time, flux = synthetic_with_secondary(0.01, 0.0)
    result = significant_secondary(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert result.f_red is not None
    assert 0.5 < result.f_red < 1.8  # white-noise curve: sig scatter ~1


def test_secondary_quiet_for_clean_transit():
    time, flux = synthetic_with_secondary(0.01, 0.0)
    result = significant_secondary(time, flux, period=2.0, t0=0.0, duration=0.1)
    assert result is not None
    assert not result.suspicious


def test_secondary_occultation_escape_hatch():
    # Hot Jupiter: 1% primary, 250 ppm occultation at P=1.5 d on a Sun-like
    # star -> depth ratio 2.5% and implied albedo ~0.8 < 1: no caution.
    time, flux = synthetic_with_secondary(0.01, 0.00025, period=1.5)
    result = significant_secondary(
        time, flux, period=1.5, t0=0.0, duration=0.075, stellar_radius=1.0, stellar_logg=4.44
    )
    assert result is not None
    assert result.secondary_significance > result.fa_threshold  # detected...
    assert result.albedo is not None and result.albedo < 1.0
    assert result.occultation_like
    assert not result.suspicious  # ...but excused as an occultation

    # Same secondary without stellar params: hatch cannot fire, caution stands.
    no_stellar = significant_secondary(time, flux, period=1.5, t0=0.0, duration=0.075)
    assert no_stellar is not None
    assert no_stellar.albedo is None
    assert no_stellar.suspicious


def test_secondary_verdict_language():
    time, flux = synthetic_with_secondary(0.01, 0.003)
    result = significant_secondary(time, flux, period=2.0, t0=0.0, duration=0.1)
    text = verdict(0.9, 0.3, centroid_snr=None, odd_even=None, secondary=result)
    assert "secondary eclipse" in text
    assert "Caution" in text


# ------------------------------------------------- unphysical duration (§3.4) --

# Sun-like star: logg 4.44, R* 1.0 -> a/R* ~215 at P=1yr, central duration ~13h.
SUN = dict(stellar_radius=1.0, stellar_logg=4.44)


def test_duration_earth_analog_is_clean():
    result = unphysical_duration(365.25, 13.0 / 24.0, **SUN)
    assert result is not None
    assert result.a_over_rstar == pytest.approx(215, rel=0.02)
    assert result.q_ratio == pytest.approx(1.0, rel=0.05)
    assert not result.suspicious


def test_duration_flags_q_above_half():
    # Half the "period" spent in transit — sinusoidal variability, not a planet.
    result = unphysical_duration(1.0, 0.55, **SUN)
    assert result is not None
    assert result.q > 0.5
    assert result.suspicious


def test_duration_flags_too_short_for_circular_orbit():
    # 1.3h event at P=1yr on a Sun-like star: q/q_circ ~0.1.
    result = unphysical_duration(365.25, 1.3 / 24.0, **SUN)
    assert result is not None
    assert result.q_ratio is not None and result.q_ratio < 0.6
    assert result.suspicious


def test_duration_without_stellar_params_uses_q_only():
    result = unphysical_duration(365.25, 1.3 / 24.0, stellar_radius=None, stellar_logg=None)
    assert result is not None
    assert result.q_circ is None and result.q_ratio is None and result.a_over_rstar is None
    assert not result.suspicious  # q is fine; density conditions can't fire

    long_q = unphysical_duration(1.0, 0.55, stellar_radius=None, stellar_logg=None)
    assert long_q is not None and long_q.suspicious


def test_duration_verdict_language():
    check = unphysical_duration(1.0, 0.55, **SUN)
    text = verdict(0.95, 0.3, centroid_snr=1.0, odd_even=None, duration_check=check)
    assert "duration" in text
    assert "Caution" in text


def _bare_scorer(candidates_path):
    from exoplanet_hunter.scoring.service import TargetScorer

    scorer = object.__new__(TargetScorer)  # skip the heavy ensemble load
    scorer.candidates_path = candidates_path
    scorer._ephemeris = None
    return scorer


def test_aux_row_layouts_by_aux_dim():
    """aux_dim >= 13 gets the vetting-aux layout; 9 keeps the legacy one."""
    from exoplanet_hunter.data.stellar import StellarParams
    from exoplanet_hunter.features.noise import pink_noise_snr
    from exoplanet_hunter.scoring.diagnostics import unphysical_duration

    scorer = _bare_scorer(None)
    scorer._snr_series = None
    scorer._fetch_stellar = lambda tic: StellarParams(
        tic_id=tic, teff=5800.0, radius=1.0, logg=4.44, tmag=10.0
    )

    class StubEnsemble:
        aux_dim = 13

    scorer.ensemble = StubEnsemble()

    time, flux = synthetic_transits(0.005, 0.005)
    oe = odd_even_depths(time, flux, period=2.0, t0=0.0, duration=0.1)
    dc = unphysical_duration(2.0, 0.1, stellar_radius=1.0, stellar_logg=4.44)
    kwargs = dict(
        flat_time=time,
        flat_flux=flux,
        centroid_snr=1.5,
        odd_even=oe,
        secondary=None,
        duration_check=dc,
    )

    row = scorer._aux_row(1, 2.0, 0.0, 0.1, **kwargs)
    assert row.shape == (13,)
    pn = pink_noise_snr(time, flux, 2.0, 0.0, 0.1)
    assert row[7] == pytest.approx(pn.snr, rel=1e-5)
    assert row[8] == pytest.approx(1.5)  # centroid stays at CENTROID_COL
    assert row[9] == pytest.approx(oe.depth_diff_sigma, rel=1e-5)
    assert np.isnan(row[11])  # no secondary result -> NaN
    assert row[12] == pytest.approx(dc.q_ratio, rel=1e-5)

    StubEnsemble.aux_dim = 9
    legacy = scorer._aux_row(1, 2.0, 0.0, 0.1, **kwargs)
    assert legacy.shape == (9,)
    assert np.isnan(legacy[7])  # no candidates.parquet -> catalogue snr NaN
    assert legacy[8] == pytest.approx(1.5)


def test_catalogue_ephemeris_converts_bjd_to_btjd(tmp_path):
    import pandas as pd

    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "tic_id": [111],
            "period_days": [2.47],
            "epoch_bjd": [2459013.0],  # BTJD 2013
            "duration_hours": [1.8],
        }
    ).to_parquet(path)

    period, t0, duration = _bare_scorer(path)._catalogue_ephemeris(111)
    assert period == pytest.approx(2.47)
    assert t0 == pytest.approx(2013.0)
    assert duration == pytest.approx(1.8 / 24.0)


def test_catalogue_ephemeris_rejects_dirty_and_missing(tmp_path):
    import pandas as pd

    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "tic_id": [1, 2, 3],
            "period_days": [2.47, -1.0, 3.0],
            "epoch_bjd": [2459013.0, 2459013.0, 24581371.0],  # row 3 epoch malformed
            "duration_hours": [1.8, 1.8, None],  # row 3 duration missing
        }
    ).to_parquet(path)

    scorer = _bare_scorer(path)
    assert scorer._catalogue_ephemeris(1) is not None
    assert scorer._catalogue_ephemeris(2) is None  # negative period
    assert scorer._catalogue_ephemeris(3) is None  # dropped by dropna
    assert scorer._catalogue_ephemeris(999) is None  # absent
