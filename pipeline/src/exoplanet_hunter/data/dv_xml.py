"""Parse a SPOC DV report XML into difference images and DV scalars.

Reader half of `data/dv.py`. Three traps, each yielding a plausible wrong
number rather than an error:

- one `planetResults` per TCE, not per target — the catalogue period picks the
  nearest `@orbitalPeriodInDays` and the mismatch is returned;
- difference images are a sparse CCD-pixel list sized to the target's aperture,
  not Kepler's fixed 33x33 — re-gridding is the consumer's job;
- `-1.0` is DV's "attempted, undefined" sentinel, not a measurement.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

NS = "{http://www.nasa.gov/2018/TESS/DV}"

#: DV writes this where a statistic was attempted but is undefined.
_SENTINEL = -1.0


def _f(element: ET.Element | None, attr: str, *, sentinel: bool = False) -> float | None:
    """Read a float attribute; None when absent, unparseable, or a sentinel."""
    if element is None:
        return None
    raw = element.get(attr)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    if sentinel and value == _SENTINEL:
        return None
    return value


def _i(element: ET.Element | None, attr: str) -> int | None:
    value = _f(element, attr)
    return None if value is None else int(value)


@dataclass(frozen=True)
class DVDifferenceImage:
    """One sector's in-transit minus out-of-transit image, as sparse pixels."""

    sector: int
    ccd_rows: np.ndarray
    ccd_cols: np.ndarray
    flux_difference: np.ndarray
    flux_in_transit: np.ndarray
    flux_out_of_transit: np.ndarray
    quality_metric: float | None
    quality_valid: bool
    n_transits: int | None
    n_cadences_in_transit: int | None

    @property
    def n_pixels(self) -> int:
        return int(self.ccd_rows.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Bounding box of the aperture — variable per target, hence not fixed."""
        if not self.n_pixels:
            return (0, 0)
        return (
            int(self.ccd_rows.max() - self.ccd_rows.min() + 1),
            int(self.ccd_cols.max() - self.ccd_cols.min() + 1),
        )


@dataclass(frozen=True)
class DVResult:
    tic_id: int
    sectors_observed: list[int]
    n_planet_candidates: int
    matched_period_days: float | None
    period_mismatch_frac: float | None
    # --- transit counts: the pair the -0.003 correlation finding calls for ---
    observed_transit_count: int | None = None
    expected_transit_count: int | None = None
    # --- detection strength ---
    max_multiple_event_sigma: float | None = None
    max_single_event_sigma: float | None = None
    max_ses_in_mes: float | None = None
    robust_statistic: float | None = None
    # --- bootstrap false-alarm ---
    bootstrap_significance: float | None = None
    bootstrap_threshold_pfa: float | None = None
    # --- ghost diagnostic (core vs halo aperture correlation) ---
    ghost_core_statistic: float | None = None
    ghost_core_significance: float | None = None
    ghost_halo_statistic: float | None = None
    ghost_halo_significance: float | None = None
    # --- fit quality ---
    chi_square_gof: float | None = None
    chi_square_gof_dof: float | None = None
    model_fit_snr: float | None = None
    # --- binary discrimination ---
    odd_even_statistic: float | None = None
    odd_even_significance: float | None = None
    longer_period_statistic: float | None = None
    shorter_period_statistic: float | None = None
    # --- weak secondary ---
    weak_secondary_max_mes: float | None = None
    weak_secondary_depth_ppm: float | None = None
    weak_secondary_robust_statistic: float | None = None
    albedo_comparison_statistic: float | None = None
    # --- centroid motion (arcsec; TIC-relative and control-relative) ---
    mean_sky_offset: float | None = None
    mean_sky_offset_uncertainty: float | None = None
    control_sky_offset: float | None = None
    control_sky_offset_uncertainty: float | None = None
    # --- stellar (TIC-8, as DV saw it) ---
    effective_temp: float | None = None
    log_g: float | None = None
    log_metallicity: float | None = None
    stellar_density: float | None = None
    stellar_radius: float | None = None
    tess_mag: float | None = None
    # --- difference-image quality ---
    summary_quality_fraction: float | None = None
    difference_images: list[DVDifferenceImage] = field(default_factory=list)

    @property
    def n_difference_images(self) -> int:
        return len(self.difference_images)

    @property
    def stellar_mass(self) -> float | None:
        """Mass from density and radius — DV publishes both but not the mass.

        rho = M / (4/3 pi R^3), in solar units where rho_sun = 1.408 g/cm^3.
        """
        if self.stellar_density is None or self.stellar_radius is None:
            return None
        if self.stellar_density <= 0 or self.stellar_radius <= 0:
            return None
        return float(self.stellar_density / 1.408 * self.stellar_radius**3)


def _sectors_from_bitmask(bitmask: str | None) -> list[int]:
    """Decode `sectorsObserved` — a 0/1 string indexed *directly* by sector.

    Position 0 is an unused slot (there is no sector 0), so character `i` is
    sector `i`, not `i + 1`. Verified against the s0042-s0046 report for TIC
    337385330: bits 44/45/46 set, and its three `differenceImageResults` carry
    `sector="44"`, `"45"`, `"46"`. An off-by-one here would mislabel every
    difference image by one sector, which is wrong in a way nothing downstream
    would flag.
    """
    if not bitmask:
        return []
    return [i for i, c in enumerate(bitmask) if c == "1"]


def _difference_images(planet: ET.Element) -> list[DVDifferenceImage]:
    images: list[DVDifferenceImage] = []
    for block in planet.findall(f"{NS}differenceImageResults"):
        rows: list[int] = []
        cols: list[int] = []
        diff: list[float] = []
        in_tr: list[float] = []
        out_tr: list[float] = []
        for pixel in block.findall(f"{NS}differenceImagePixelData"):
            row, col = _i(pixel, "ccdRow"), _i(pixel, "ccdColumn")
            if row is None or col is None:
                continue
            rows.append(row)
            cols.append(col)
            diff.append(_f(pixel.find(f"{NS}meanFluxDifference"), "value") or np.nan)
            in_tr.append(_f(pixel.find(f"{NS}meanFluxInTransit"), "value") or np.nan)
            out_tr.append(_f(pixel.find(f"{NS}meanFluxOutOfTransit"), "value") or np.nan)
        quality = block.find(f"{NS}qualityMetric")
        images.append(
            DVDifferenceImage(
                sector=_i(block, "sector") or -1,
                ccd_rows=np.asarray(rows, dtype=np.int32),
                ccd_cols=np.asarray(cols, dtype=np.int32),
                flux_difference=np.asarray(diff, dtype=np.float32),
                flux_in_transit=np.asarray(in_tr, dtype=np.float32),
                flux_out_of_transit=np.asarray(out_tr, dtype=np.float32),
                quality_metric=_f(quality, "value"),
                quality_valid=(quality is not None and quality.get("valid") == "true"),
                n_transits=_i(block, "numberOfTransits"),
                n_cadences_in_transit=_i(block, "numberOfCadencesInTransit"),
            )
        )
    return images


def _pick_planet(
    planets: list[ET.Element], period_days: float | None
) -> tuple[ET.Element | None, float | None, float | None]:
    """Choose the TCE matching our catalogue period; return (element, period, mismatch)."""
    if not planets:
        return None, None, None
    if period_days is None or not np.isfinite(period_days) or period_days <= 0:
        candidate = planets[0].find(f"{NS}planetCandidate")
        return planets[0], _f(candidate, "orbitalPeriodInDays"), None
    best: tuple[float, ET.Element, float | None] | None = None
    for planet in planets:
        found = _f(planet.find(f"{NS}planetCandidate"), "orbitalPeriodInDays")
        if found is None or found <= 0:
            continue
        mismatch = abs(found - period_days) / period_days
        if best is None or mismatch < best[0]:
            best = (mismatch, planet, found)
    if best is None:
        return planets[0], None, None
    return best[1], best[2], best[0]


def parse_dv_xml(path: Path, *, period_days: float | None = None) -> DVResult:
    """Parse one `*_dvr.xml`.

    Parameters
    ----------
    path        : the DV report.
    period_days : catalogue orbital period, used to pick the right TCE when the
                  report covers several. Omit it and the first TCE is used —
                  correct only for single-candidate targets.
    """
    root = ET.parse(path).getroot()
    tic_id = _i(root, "ticId") or 0
    planets = root.findall(f"{NS}planetResults")
    planet, matched_period, mismatch = _pick_planet(planets, period_days)
    if mismatch is not None and mismatch > 0.01:
        # Not fatal — the caller may still want the target-level scalars — but
        # never silent: a 5% period mismatch means these diagnostics describe a
        # different signal than our row does.
        log.warning(
            "[dv] TIC %d: best TCE period %.6f d is %.1f%% from the catalogue %.6f d",
            tic_id,
            matched_period or float("nan"),
            mismatch * 100,
            period_days or float("nan"),
        )

    result_kwargs: dict[str, object] = {
        "tic_id": tic_id,
        "sectors_observed": _sectors_from_bitmask(root.get("sectorsObserved")),
        "n_planet_candidates": len(planets),
        "matched_period_days": matched_period,
        "period_mismatch_frac": mismatch,
        "effective_temp": _f(root.find(f".//{NS}effectiveTemp"), "value"),
        "log_g": _f(root.find(f".//{NS}log10SurfaceGravity"), "value"),
        "log_metallicity": _f(root.find(f".//{NS}log10Metallicity"), "value"),
        "stellar_density": _f(root.find(f".//{NS}stellarDensity"), "value"),
        "stellar_radius": _f(root.find(f".//{NS}radius"), "value"),
        "tess_mag": _f(root.find(f".//{NS}tessMag"), "value"),
    }
    if planet is None:
        return DVResult(**result_kwargs)  # type: ignore[arg-type]

    candidate = planet.find(f"{NS}planetCandidate")
    bootstrap = planet.find(f"{NS}bootstrapResults")
    ghost = planet.find(f"{NS}ghostDiagnosticResults")
    binary = planet.find(f"{NS}binaryDiscriminationResults")
    # `weakSecondary` hangs off planetCandidate, and the albedo test off
    # secondaryEventResults/comparisonTests — neither is a direct child of
    # planetResults, and looking for them there returns None for every target.
    secondary = planet.find(f".//{NS}weakSecondary")
    all_fit = planet.find(f"{NS}allTransitsFit")
    summary = planet.find(f".//{NS}summaryQualityMetric")
    # Offset of the difference-image centroid from the catalogue position
    # (`msTicCentroidOffsets`) and from the out-of-transit centroid
    # (`msControlCentroidOffsets`). Both are needed: the control offset is what
    # calibrates a systematic shared by the pair, so the TIC offset alone can
    # read as a large centroid shift when nothing moved.
    tic_offset = planet.find(f".//{NS}msTicCentroidOffsets/{NS}meanSkyOffset")
    control_offset = planet.find(f".//{NS}msControlCentroidOffsets/{NS}meanSkyOffset")

    result_kwargs.update(
        {
            "observed_transit_count": _i(candidate, "observedTransitCount"),
            "expected_transit_count": _i(candidate, "expectedTransitCount"),
            "max_multiple_event_sigma": _f(candidate, "maxMultipleEventSigma"),
            "max_single_event_sigma": _f(candidate, "maxSingleEventSigma"),
            "max_ses_in_mes": _f(candidate, "maxSesInMes"),
            "robust_statistic": _f(candidate, "robustStatistic"),
            "chi_square_gof": _f(candidate, "chiSquareGof"),
            "chi_square_gof_dof": _f(candidate, "chiSquareGofDof"),
            "bootstrap_significance": _f(bootstrap, "significance", sentinel=True),
            "bootstrap_threshold_pfa": _f(bootstrap, "bootstrapThresholdForDesiredPfa"),
            "model_fit_snr": _f(all_fit, "modelFitSnr"),
            "weak_secondary_max_mes": _f(secondary, "maxMes"),
            "weak_secondary_robust_statistic": _f(secondary, "robustStatistic"),
            "summary_quality_fraction": _f(summary, "fractionOfGoodMetrics", sentinel=True),
            "mean_sky_offset": _f(tic_offset, "value"),
            "mean_sky_offset_uncertainty": _f(tic_offset, "uncertainty"),
            "control_sky_offset": _f(control_offset, "value"),
            "control_sky_offset_uncertainty": _f(control_offset, "uncertainty"),
            "difference_images": _difference_images(planet),
        }
    )
    if ghost is not None:
        core = ghost.find(f"{NS}coreApertureCorrelationStatistic")
        halo = ghost.find(f"{NS}haloApertureCorrelationStatistic")
        result_kwargs.update(
            {
                "ghost_core_statistic": _f(core, "value"),
                "ghost_core_significance": _f(core, "significance", sentinel=True),
                "ghost_halo_statistic": _f(halo, "value"),
                "ghost_halo_significance": _f(halo, "significance", sentinel=True),
            }
        )
    if binary is not None:
        odd_even = binary.find(f"{NS}oddEvenTransitDepthComparisonStatistic")
        result_kwargs.update(
            {
                "odd_even_statistic": _f(odd_even, "value"),
                "odd_even_significance": _f(odd_even, "significance", sentinel=True),
                # These are -1.0 whenever there is no comparison planet, which
                # is most targets. Sentinel-aware so "no comparison" cannot be
                # read as "a significance of -1".
                "longer_period_statistic": _f(
                    binary.find(f"{NS}longerPeriodComparisonStatistic"),
                    "significance",
                    sentinel=True,
                ),
                "shorter_period_statistic": _f(
                    binary.find(f"{NS}shorterPeriodComparisonStatistic"),
                    "significance",
                    sentinel=True,
                ),
            }
        )
    if secondary is not None:
        result_kwargs["weak_secondary_depth_ppm"] = _f(secondary.find(f"{NS}depthPpm"), "value")
    result_kwargs["albedo_comparison_statistic"] = _f(
        planet.find(f".//{NS}albedoComparisonStatistic"), "value"
    )

    return DVResult(**result_kwargs)  # type: ignore[arg-type]
