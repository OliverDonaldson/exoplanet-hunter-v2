"""Follow-up prioritisation metrics: TSM, ESM, predicted mass, predicted K.

Implements the NExScI "Supplementary ExoFOP Calculations" recipes (Kempton
et al. 2018 metrics; Chen & Kipping 2017 mass-radius relation as applied by
Louie et al. 2018) so we can compute these numbers for candidates that
ExoFOP doesn't cover: CTOIs, and our own model's discoveries scored ad hoc.
For TOIs the NExScI-published values are used as-is (they draw on TFOP
working-group spreadsheets we can't see); this module exists for everything
else, and its outputs are pinned against the document's TOI-664.01 worked
example in tests.

All functions are float/ndarray-vectorised. Units are handled by astropy
(`units`, `constants`, `modeling.BlackBody`) exactly as in the reference
implementation; NaN inputs propagate to NaN outputs so callers can compute
column-wise over incomplete catalogues.
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import BlackBody

FloatArray = float | np.ndarray

#: (Rearth / Rsun)^2 — converts (Rp/Rs) in mixed units to a true depth ratio.
_DEPTH_CONVERSION = float(((1 * u.earthRad) / (1 * u.solRad)).decompose().value) ** 2

#: ESM is evaluated at 7.5 micron (Kempton et al. 2018, Eq. 4).
_ESM_WAVELENGTH = 7.5 * u.micron


def stellar_mass_from_logg(logg_cgs: FloatArray, r_star_rsun: FloatArray) -> FloatArray:
    """M* [Msun] from surface gravity [log10 cm/s^2] and stellar radius [Rsun]."""
    g = 10.0 ** np.asanyarray(logg_cgs, dtype=float) * u.cm / u.s**2
    m = g * (np.asanyarray(r_star_rsun, dtype=float) * u.Rsun) ** 2 / const.G
    return m.to(u.Msun).value


def semi_major_axis_au(m_star_msun: FloatArray, period_days: FloatArray) -> FloatArray:
    """a [AU] from Kepler's third law, planet mass neglected (Eq. 4)."""
    period = np.asanyarray(period_days, dtype=float)
    period = np.where(period > 0, period, np.nan)  # ExoFOP uses 0 for unknown
    a3 = (
        const.G
        * (np.asanyarray(m_star_msun, dtype=float) * u.Msun)
        * (period * u.d) ** 2
        / (4 * np.pi**2)
    )
    return (a3 ** (1 / 3)).to(u.AU).value


def equilibrium_temperature_k(
    m_star_msun: FloatArray,
    period_days: FloatArray,
    r_star_rsun: FloatArray,
    teff_k: FloatArray,
) -> FloatArray:
    """Teq [K] assuming zero albedo and full heat redistribution (Eq. 3)."""
    a = semi_major_axis_au(m_star_msun, period_days) * u.AU
    ratio = ((np.asanyarray(r_star_rsun, dtype=float) * u.Rsun) / a).decompose().value
    return np.asanyarray(teff_k, dtype=float) * np.sqrt(ratio) * 0.25**0.25


#: IAU 2015 nominal solar effective temperature [K] — the Teff that reproduces
#: the nominal solar luminosity at the nominal solar radius, so the
#: Stefan-Boltzmann constants cancel in the Sun-normalised luminosity ratio.
_TEFF_SUN_K = 5772.0

#: Kasting et al. (1993) empirical habitable-zone edges for the Sun [AU]:
#: recent-Venus (inner) and early-Mars (outer), scaled by sqrt(L*/Lsun). These
#: are the archive's POE bounds — the conservative Kopparapu (2013) edges add a
#: Teff term we deliberately omit to match the published recent-Venus/early-Mars.
_HZ_RECENT_VENUS_AU = 0.75
_HZ_EARLY_MARS_AU = 1.77


def stellar_luminosity_lsun(r_star_rsun: FloatArray, teff_k: FloatArray) -> FloatArray:
    """L* [Lsun] from Stefan-Boltzmann, L = 4π R*² σ Teff⁴.

    Written as the Sun-normalised ratio (R*/Rsun)² (Teff/Teff_sun)⁴ so the 4π σ
    constants cancel exactly; a Sun (R*=1 Rsun, Teff=5772 K) returns 1.0.
    """
    r = np.asanyarray(r_star_rsun, dtype=float)
    teff = np.asanyarray(teff_k, dtype=float)
    return r**2 * (teff / _TEFF_SUN_K) ** 4


def insolation_flux_earth(luminosity_lsun: FloatArray, a_au: FloatArray) -> FloatArray:
    """Insolation S [S_earth] = (L*/Lsun) / (a/AU)² — inverse-square law,
    Earth-normalised so an Earth (L*=1 Lsun, a=1 AU) returns 1.0."""
    a = np.asanyarray(a_au, dtype=float)
    a = np.where(a > 0, a, np.nan)  # ExoFOP uses 0 for unknown period -> a
    return np.asanyarray(luminosity_lsun, dtype=float) / a**2


def habitable_zone_au(luminosity_lsun: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(inner, outer) habitable-zone radii [AU] via the luminosity-scaled Kasting
    edges r = r_sun · sqrt(L*/Lsun): recent-Venus inner, early-Mars outer. A Sun
    (L*=1 Lsun) returns (0.75, 1.77) AU."""
    scale = np.sqrt(np.asanyarray(luminosity_lsun, dtype=float))
    return _HZ_RECENT_VENUS_AU * scale, _HZ_EARLY_MARS_AU * scale


def predict_planet_mass_me(radius_re: FloatArray) -> FloatArray:
    """Mp [Mearth] from radius via Chen & Kipping 2017 / Louie et al. 2018.

    Jovian regime (> 14.26 Rearth) pinned to 317 Mearth (~1 Mjup) per the
    NExScI extension of Louie et al.
    """
    r = np.asanyarray(radius_re, dtype=float)
    return np.select(
        [r > 14.26, r < 1.23],
        [np.full_like(r, 317.0), 0.9718 * r**3.58],
        default=1.436 * r**1.70,
    )


def tsm_scale_factor(radius_re: FloatArray) -> FloatArray:
    """Kempton et al. 2018 Table 1 scale factor; > 10 Rearth reuses the 4-10 bin."""
    r = np.asanyarray(radius_re, dtype=float)
    return np.select(
        [r <= 1.5, r <= 2.75, r <= 4.0],
        [np.full_like(r, 0.19), np.full_like(r, 1.26), np.full_like(r, 1.28)],
        default=1.15,
    )


def tsm(
    radius_re: FloatArray,
    teq_k: FloatArray,
    r_star_rsun: FloatArray,
    j_mag: FloatArray,
    m_planet_me: FloatArray | None = None,
) -> FloatArray:
    """Transmission spectroscopy metric (Eq. 1); mass predicted if not given."""
    r = np.asanyarray(radius_re, dtype=float)
    mp = predict_planet_mass_me(r) if m_planet_me is None else np.asanyarray(m_planet_me, float)
    return (
        tsm_scale_factor(r)
        * np.asanyarray(teq_k, dtype=float)
        * r**3
        / (mp * np.asanyarray(r_star_rsun, dtype=float) ** 2)
        * 10.0 ** (-np.asanyarray(j_mag, dtype=float) / 5.0)
    )


def esm(
    teq_k: FloatArray,
    teff_k: FloatArray,
    radius_re: FloatArray,
    r_star_rsun: FloatArray,
    k_mag: FloatArray,
) -> FloatArray:
    """Emission spectroscopy metric (Eq. 2): 7.5 µm blackbody ratio at Tday=1.1 Teq."""
    teq = np.asanyarray(teq_k, dtype=float)
    teff = np.asanyarray(teff_k, dtype=float)
    # BlackBody rejects non-positive/NaN temperatures; mask then restore NaN.
    valid = np.isfinite(teq) & np.isfinite(teff) & (teq > 0) & (teff > 0)
    tday = np.where(valid, 1.1 * teq, 1000.0)
    tstar = np.where(valid, teff, 1000.0)
    bb_ratio = (
        BlackBody(temperature=tday * u.K)(_ESM_WAVELENGTH)
        / BlackBody(temperature=tstar * u.K)(_ESM_WAVELENGTH)
    ).value
    bb_ratio = np.where(valid, bb_ratio, np.nan)

    depth_ppm = (
        1e6
        * _DEPTH_CONVERSION
        * (np.asanyarray(radius_re, dtype=float) / np.asanyarray(r_star_rsun, dtype=float)) ** 2
    )
    return 4.29 * bb_ratio * depth_ppm * 10.0 ** (-np.asanyarray(k_mag, dtype=float) / 5.0)


def rv_semi_amplitude_ms(
    period_days: FloatArray,
    m_star_msun: FloatArray,
    m_planet_me: FloatArray,
    inclination_deg: FloatArray = 90.0,
    eccentricity: FloatArray = 0.0,
) -> FloatArray:
    """Predicted RV semi-amplitude K [m/s] (Eq. 6); circular edge-on by default."""
    period = np.asanyarray(period_days, dtype=float)
    period = np.where(period > 0, period, np.nan)
    part1 = ((2 * np.pi * const.G) / (period * u.d)) ** (1 / 3)
    part2 = (np.asanyarray(m_planet_me, dtype=float) * u.Mearth) / (
        np.asanyarray(m_star_msun, dtype=float) * u.Msun
    ) ** (2 / 3)
    part3 = np.sin(np.deg2rad(np.asanyarray(inclination_deg, dtype=float))) / np.sqrt(
        1 - np.asanyarray(eccentricity, dtype=float) ** 2
    )
    return (part1 * part2 * part3).to(u.m / u.s).value
