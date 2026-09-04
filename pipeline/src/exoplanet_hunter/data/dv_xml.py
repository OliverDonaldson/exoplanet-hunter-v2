"""Parse a SPOC DV report XML into difference images and DV scalars.

Reader half of `data/dv.py`. Three traps, each yielding a plausible wrong
number rather than an error:

- one `planetResults` per TCE, not per target — the catalogue period picks the
  nearest `@orbitalPeriodInDays` and the mismatch is returned;
- difference images are a CCD-pixel list sized to the target's aperture, not
  Kepler's fixed 33x33 — re-gridding is the consumer's job. Measured 2026-08-17
  over the whole archive, the list is *dense*: it fills its bounding box
  exactly, with no gaps and no repeated coordinates, so reconstructing the
  rectangle is exact rather than an interpolation. 95.8% are 11x11 and the full
  range is 11-25 px;
- `-1.0` is DV's "attempted, undefined" sentinel, not a measurement — including
  per pixel, where it marks a whole sector DV declined to measure while writing
  its flux as a plausible 0.0. See `DVDifferenceImage.declined`.
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


def _nan(value: float | None) -> float:
    """None -> NaN, keeping a measured 0.0 as 0.0.

    Spelled out rather than `value or np.nan`, which agrees with this for every
    input except zero and is silently wrong for that one. Exactly 0.0 is how DV
    writes a pixel it declined to measure, so the shorter form collapsed
    "declined" into "unreadable" — and a consumer that cannot tell those apart
    cannot build an honest presence mask.
    """
    return float("nan") if value is None else value


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
    #: Per-pixel uncertainty on `flux_difference`. DV's -1.0 sentinel here is
    #: what marks a pixel it did not measure, and it is the only unambiguous
    #: marker: the value itself is written as a plausible 0.0.
    flux_difference_uncertainty: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    #: The target star's catalogue position on the CCD for this sector, from
    #: `ticReferenceCentroid`, in the same row/column frame as `ccd_rows` and
    #: `ccd_cols` and with sub-pixel precision. None when DV did not define it.
    #:
    #: This is the **origin** the difference image is read against. Without it a
    #: stamp says only that flux moved, not whether it moved away from the star,
    #: and roadmap 4.2b finding 2 records its absence as the mechanism behind
    #: stage 9's null. Measured 2026-08-27 over all 38,964 non-declined images in
    #: the archive: the target sits a median 0.84 px from the bounding box's
    #: centre, sd 0.61 px in row and 0.64 in column, and lands in a **different
    #: pixel** than that centre on 77.8% of stamps — so the centred placement
    #: `preprocess/diffimage.py` performs is not a stand-in for it.
    target_row: float | None = None
    target_col: float | None = None

    @property
    def n_pixels(self) -> int:
        return int(self.ccd_rows.size)

    @property
    def declined(self) -> bool:
        """True when DV produced no difference image for this sector at all.

        Measured 2026-08-17 over the 53,118 difference images in the archive:
        **26.6% are declined**, and the state is all-or-nothing — every pixel of
        an image carries the sentinel or none of them does, with nothing in
        between. Every declined image also reports `quality_metric` exactly 0.0
        and `quality_valid` false, so DV is consistent about it in two places.

        This is a third state, and the reason it is named here rather than
        inferred downstream: a declined sector is *not* a measurement of no
        centroid shift. Fed to a model as zeros it would be indistinguishable
        from a star that genuinely did not move, which is the strongest evidence
        this diagnostic can give.
        """
        if not self.n_pixels or not self.flux_difference_uncertainty.size:
            return True
        return bool(np.all(self.flux_difference_uncertainty == _SENTINEL))

    @property
    def target_position(self) -> tuple[float, float] | None:
        """`(row, column)` of the target on the CCD, or None if DV left it undefined.

        Both halves must be defined for the pair to locate anything, so this is
        all-or-nothing rather than two independently-missing floats.
        """
        if self.target_row is None or self.target_col is None:
            return None
        return (self.target_row, self.target_col)

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


def _centroid_coordinate(block: ET.Element | None, tag: str) -> float | None:
    """One axis of a DV centroid, or None where DV attempted it and failed.

    **The sentinel here is on the uncertainty, not the value.** On a sector DV
    declined to measure it writes `ticReferenceCentroid` as row 0.0, column 0.0,
    uncertainty -1.0 — verified on every one of the 14,154 declined images in
    the archive, where the row value is *exactly* 0.0 and never anything else.
    Reading the value alone would put the target at CCD row 0 of a stamp whose
    pixels start near row 2000, which is a confident placement 2,000 px away
    rather than a missing one. The same trap `_nan` exists for, one level up.
    """
    if block is None:
        return None
    element = block.find(f"{NS}{tag}")
    if element is None:
        return None
    if _f(element, "uncertainty") == _SENTINEL:
        return None
    return _f(element, "value")


def _difference_images(planet: ET.Element) -> list[DVDifferenceImage]:
    images: list[DVDifferenceImage] = []
    for block in planet.findall(f"{NS}differenceImageResults"):
        rows: list[int] = []
        cols: list[int] = []
        diff: list[float] = []
        diff_unc: list[float] = []
        in_tr: list[float] = []
        out_tr: list[float] = []
        for pixel in block.findall(f"{NS}differenceImagePixelData"):
            row, col = _i(pixel, "ccdRow"), _i(pixel, "ccdColumn")
            if row is None or col is None:
                continue
            rows.append(row)
            cols.append(col)
            difference = pixel.find(f"{NS}meanFluxDifference")
            diff.append(_nan(_f(difference, "value")))
            # Read raw: `_f(..., sentinel=True)` would map -1.0 to None, which is
            # the one value this column exists to preserve.
            diff_unc.append(_nan(_f(difference, "uncertainty")))
            in_tr.append(_nan(_f(pixel.find(f"{NS}meanFluxInTransit"), "value")))
            out_tr.append(_nan(_f(pixel.find(f"{NS}meanFluxOutOfTransit"), "value")))
        quality = block.find(f"{NS}qualityMetric")
        # The catalogue position of the star this stamp is of — the frame the
        # difference is read against. `ticReferenceCentroid` rather than
        # `controlImageCentroid`: the control centroid is *measured from the
        # out-of-transit image*, so an offset computed against it would be
        # partly the thing being measured, while the TIC reference is
        # independent of this sector's photometry.
        reference = block.find(f"{NS}ticReferenceCentroid")
        images.append(
            DVDifferenceImage(
                sector=_i(block, "sector") or -1,
                ccd_rows=np.asarray(rows, dtype=np.int32),
                ccd_cols=np.asarray(cols, dtype=np.int32),
                flux_difference=np.asarray(diff, dtype=np.float32),
                flux_difference_uncertainty=np.asarray(diff_unc, dtype=np.float32),
                flux_in_transit=np.asarray(in_tr, dtype=np.float32),
                flux_out_of_transit=np.asarray(out_tr, dtype=np.float32),
                quality_metric=_f(quality, "value"),
                quality_valid=(quality is not None and quality.get("valid") == "true"),
                n_transits=_i(block, "numberOfTransits"),
                n_cadences_in_transit=_i(block, "numberOfCadencesInTransit"),
                target_row=_centroid_coordinate(reference, "row"),
                target_col=_centroid_coordinate(reference, "column"),
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
