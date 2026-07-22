"""Follow-up metrics pinned to the NExScI worked example (TOI-664.01).

Every expected value below comes from the provenance table in
"Supplementary ExoFOP Calculations" (docs/exofop_calculations.pdf):
M*=1.516 Msun, log g=3.72765, R*=2.79 Rsun, P=4.736 d, a=0.0634 AU,
T*=5302 K, Teq=1699 K, Rp=14.03 Re, mJ=6.616, mK=6.122, Mp=128 Me,
K=36.9 m/s, TSM scale=1.15, TSM=257, ESM=132.
"""

import numpy as np
import pytest
from astropy import units as u

from exoplanet_hunter.features.followup import (
    _TEFF_SUN_K,
    equilibrium_temperature_k,
    esm,
    habitable_zone_au,
    insolation_flux_earth,
    predict_planet_mass_me,
    rv_semi_amplitude_ms,
    semi_major_axis_au,
    stellar_luminosity_lsun,
    stellar_mass_from_logg,
    tsm,
    tsm_scale_factor,
)

LOGG, R_STAR, PERIOD, TEFF = 3.72765, 2.79, 4.736, 5302.0
RP, MJ, MK = 14.03, 6.616, 6.122


def test_stellar_mass_from_logg():
    assert stellar_mass_from_logg(LOGG, R_STAR) == pytest.approx(1.516, rel=1e-3)


def test_semi_major_axis():
    assert semi_major_axis_au(1.516, PERIOD) == pytest.approx(0.0634, rel=2e-3)


def test_equilibrium_temperature():
    teq = equilibrium_temperature_k(1.516, PERIOD, R_STAR, TEFF)
    assert teq == pytest.approx(1699, rel=3e-3)


def test_predicted_mass_regimes():
    assert predict_planet_mass_me(RP) == pytest.approx(128.0, rel=5e-3)
    assert predict_planet_mass_me(1.0) == pytest.approx(0.9718, rel=1e-6)  # rocky branch
    assert predict_planet_mass_me(20.0) == 317.0  # Jovian pin


def test_tsm_scale_factor_bins():
    assert list(tsm_scale_factor(np.array([1.0, 2.0, 3.5, 8.0, RP]))) == [
        0.19,
        1.26,
        1.28,
        1.15,
        1.15,
    ]


def test_tsm_worked_example():
    teq = equilibrium_temperature_k(1.516, PERIOD, R_STAR, TEFF)
    assert tsm(RP, teq, R_STAR, MJ) == pytest.approx(257, rel=0.01)


def test_esm_worked_example():
    teq = equilibrium_temperature_k(1.516, PERIOD, R_STAR, TEFF)
    assert esm(teq, TEFF, RP, R_STAR, MK) == pytest.approx(132, rel=0.015)


def test_rv_semi_amplitude_worked_example():
    assert rv_semi_amplitude_ms(PERIOD, 1.516, 128.0) == pytest.approx(36.9, rel=0.01)


def test_stellar_luminosity_lsun():
    assert stellar_luminosity_lsun(1.0, _TEFF_SUN_K) == pytest.approx(1.0, rel=1e-9)  # the Sun
    # TOI-664: L = R*^2 (Teff/Tsun)^4 = 2.79^2 (5302/5772)^4
    assert stellar_luminosity_lsun(R_STAR, TEFF) == pytest.approx(5.54, rel=5e-3)


def test_insolation_and_habitable_zone_earth_sun():
    assert insolation_flux_earth(1.0, 1.0) == pytest.approx(1.0, rel=1e-9)  # Earth around the Sun
    inner, outer = habitable_zone_au(1.0)
    assert inner == pytest.approx(0.75) and outer == pytest.approx(1.77)
    # HZ scales as sqrt(L): a 4x-luminosity star pushes both edges out 2x.
    inner4, outer4 = habitable_zone_au(4.0)
    assert inner4 == pytest.approx(1.5) and outer4 == pytest.approx(3.54)


def test_insolation_consistent_with_equilibrium_temperature():
    """Cross-check the new POE insolation against the independent Teq recipe:
    Teq = Teq_earth · S^(1/4), with Teq_earth the zero-albedo full-redistribution
    value at 1 AU. Agreement ties L*, a and S to the pinned TOI-664 Teq."""
    lum = stellar_luminosity_lsun(R_STAR, TEFF)
    a = semi_major_axis_au(1.516, PERIOD)
    s = insolation_flux_earth(lum, a)
    teq_earth = _TEFF_SUN_K * np.sqrt((1 * u.Rsun).to(u.AU).value) * 0.25**0.25
    assert teq_earth * s**0.25 == pytest.approx(
        equilibrium_temperature_k(1.516, PERIOD, R_STAR, TEFF), rel=2e-3
    )


def test_nan_and_zero_period_propagate():
    assert np.isnan(semi_major_axis_au(1.0, 0.0))  # ExoFOP's "period unknown"
    assert np.isnan(insolation_flux_earth(1.0, semi_major_axis_au(1.0, 0.0)))
    assert np.isnan(predict_planet_mass_me(np.nan))
    out = esm(np.array([np.nan, 1699.0]), TEFF, RP, R_STAR, MK)
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_vectorised_over_catalogue_columns():
    n = 50
    rng = np.random.default_rng(0)
    teq = equilibrium_temperature_k(
        rng.uniform(0.5, 2.0, n),
        rng.uniform(1.0, 100.0, n),
        rng.uniform(0.5, 3.0, n),
        rng.uniform(3000, 7000, n),
    )
    values = tsm(rng.uniform(0.8, 20.0, n), teq, rng.uniform(0.5, 3.0, n), rng.uniform(6, 14, n))
    assert values.shape == (n,) and np.isfinite(values).all()
