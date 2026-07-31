"""DV report parsing: TCE matching, sparse difference images, sentinels.

The fixture mirrors the real report for TIC 337385330 (2026-08-01) — same
nesting, same attribute names, same `sectorsObserved` convention — shrunk to
two TCEs and a 2x2 aperture. Each test pins a way the parse could return a
plausible wrong number: the wrong TCE's diagnostics, sectors off by one, or a
`-1.0` "not computed" read as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.data.dv_xml import parse_dv_xml

NS = "http://www.nasa.gov/2018/TESS/DV"


def _pixel(row: int, col: int, diff: float) -> str:
    return (
        f'<dv:differenceImagePixelData ccdRow="{row}" ccdColumn="{col}">'
        f'<dv:meanFluxInTransit value="10.0" uncertainty="1.0"/>'
        f'<dv:meanFluxOutOfTransit value="12.0" uncertainty="1.0"/>'
        f'<dv:meanFluxDifference value="{diff}" uncertainty="1.0"/>'
        f'<dv:meanFluxForTargetTable value="11.0" uncertainty="1.0"/>'
        f"</dv:differenceImagePixelData>"
    )


def _planet(number: int, period: float, observed: int, expected: int, mes: float) -> str:
    pixels = "".join(_pixel(r, c, 1.0 * (r + c)) for r in (100, 101) for c in (200, 201))
    return f"""
  <dv:planetResults planetNumber="{number}">
    <dv:planetCandidate planetNumber="{number}" orbitalPeriodInDays="{period}"
        observedTransitCount="{observed}" expectedTransitCount="{expected}"
        maxMultipleEventSigma="{mes}" maxSingleEventSigma="7.16" maxSesInMes="6.89"
        robustStatistic="9.77" chiSquareGof="480.44" chiSquareGofDof="524.0"
        suspectedEclipsingBinary="false">
      <dv:weakSecondary maxMes="2.72" robustStatistic="2.18" medianMes="-0.03">
        <dv:depthPpm value="345.33" uncertainty="141.57"/>
      </dv:weakSecondary>
    </dv:planetCandidate>
    <dv:allTransitsFit type="ALL" modelFitSnr="11.54"/>
    <dv:bootstrapResults significance="6.17E-39" bootstrapThresholdForDesiredPfa="6.39"/>
    <dv:ghostDiagnosticResults>
      <dv:coreApertureCorrelationStatistic value="11.95" significance="1.0"/>
      <dv:haloApertureCorrelationStatistic value="-0.99" significance="0.162"/>
    </dv:ghostDiagnosticResults>
    <dv:binaryDiscriminationResults>
      <dv:oddEvenTransitDepthComparisonStatistic value="1.41" significance="0.236"/>
      <dv:longerPeriodComparisonStatistic planetNumber="0" value="0.0" significance="-1.0"/>
      <dv:shorterPeriodComparisonStatistic planetNumber="0" value="0.0" significance="-1.0"/>
    </dv:binaryDiscriminationResults>
    <dv:secondaryEventResults>
      <dv:comparisonTests>
        <dv:albedoComparisonStatistic value="1.23" significance="0.5"/>
      </dv:comparisonTests>
    </dv:secondaryEventResults>
    <dv:centroidResults>
      <dv:differenceImageMotionResults>
        <dv:msTicCentroidOffsets>
          <dv:meanSkyOffset value="114.33" uncertainty="26.36"/>
        </dv:msTicCentroidOffsets>
        <dv:msControlCentroidOffsets>
          <dv:meanSkyOffset value="112.18" uncertainty="33.59"/>
        </dv:msControlCentroidOffsets>
      </dv:differenceImageMotionResults>
    </dv:centroidResults>
    <dv:differenceImageResults sector="44" numberOfTransits="1" numberOfCadencesInTransit="88">
      <dv:qualityMetric attempted="true" valid="true" value="0.983"/>
      {pixels}
    </dv:differenceImageResults>
    <dv:differenceImageResults sector="46" numberOfTransits="2" numberOfCadencesInTransit="90">
      <dv:qualityMetric attempted="true" valid="false" value="0.42"/>
      {pixels}
    </dv:differenceImageResults>
    <dv:summaryQualityMetric qualityThreshold="0.7" numberOfMetrics="2"
        numberOfGoodMetrics="1" fractionOfGoodMetrics="0.5"/>
  </dv:planetResults>"""


#: bits 44 and 46 set, index == sector (position 0 is an unused slot).
_BITMASK = "".join("1" if i in (44, 46) else "0" for i in range(47))


@pytest.fixture
def report(tmp_path):
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<dv:dvTargetResults xmlns:dv="{NS}" ticId="337385330" planetCandidateCount="2"
    sectorsObserved="{_BITMASK}">
  <dv:effectiveTemp value="3973.0" uncertainty="157.0"/>
  <dv:log10SurfaceGravity value="4.646" uncertainty="0.008"/>
  <dv:log10Metallicity value="0.0" uncertainty="0.5"/>
  <dv:stellarDensity value="2.666" uncertainty="0.1"/>
  <dv:radius value="0.6059" uncertainty="0.017"/>
  <dv:tessMag value="10.1857" uncertainty="0.007"/>
{_planet(1, 15.534716, observed=5, expected=6, mes=11.57)}
{_planet(2, 3.211, observed=20, expected=21, mes=8.10)}
</dv:dvTargetResults>"""
    path = tmp_path / "tess-s0042-s0046-0000000337385330-00550_dvr.xml"
    path.write_text(xml)
    return path


def test_matches_the_tce_with_the_nearest_period(report):
    # A report holds one planetResults per TCE. Taking the first would attribute
    # the 3.2 d signal's diagnostics to a 15.5 d candidate.
    r = parse_dv_xml(report, period_days=15.532806)
    assert r.n_planet_candidates == 2
    assert r.matched_period_days == pytest.approx(15.534716)
    assert r.period_mismatch_frac < 1e-3
    assert r.max_multiple_event_sigma == pytest.approx(11.57)
    assert (r.observed_transit_count, r.expected_transit_count) == (5, 6)

    other = parse_dv_xml(report, period_days=3.2)
    assert other.matched_period_days == pytest.approx(3.211)
    assert other.observed_transit_count == 20


def test_without_a_period_the_first_tce_is_used(report):
    r = parse_dv_xml(report)
    assert r.matched_period_days == pytest.approx(15.534716)
    assert r.period_mismatch_frac is None


def test_sector_bitmask_is_indexed_by_sector_not_offset(report):
    # Off by one here mislabels every difference image by a sector, and nothing
    # downstream would flag it.
    r = parse_dv_xml(report, period_days=15.53)
    assert r.sectors_observed == [44, 46]
    assert [im.sector for im in r.difference_images] == [44, 46]


def test_difference_images_are_sparse_with_a_bounding_box(report):
    r = parse_dv_xml(report, period_days=15.53)
    first = r.difference_images[0]
    assert first.n_pixels == 4
    assert first.shape == (2, 2)  # aperture extent, not a fixed 33x33
    assert np.array_equal(first.ccd_rows, np.array([100, 100, 101, 101]))
    assert first.flux_difference.tolist() == [300.0, 301.0, 301.0, 302.0]
    assert first.quality_metric == pytest.approx(0.983) and first.quality_valid
    # An attempted-but-invalid metric keeps its value and loses its validity —
    # that pair is what the difference-image quality mask is built from.
    assert not r.difference_images[1].quality_valid
    assert r.summary_quality_fraction == pytest.approx(0.5)


def test_sentinels_become_none_not_measurements(report):
    r = parse_dv_xml(report, period_days=15.53)
    # -1.0 means "no comparison planet", which is most targets; as a number it
    # would poison any aggregate it entered.
    assert r.longer_period_statistic is None
    assert r.shorter_period_statistic is None
    assert r.odd_even_significance == pytest.approx(0.236)


def test_reads_nested_weak_secondary_and_albedo(report):
    # Both hang below planetResults, not off it: weakSecondary under
    # planetCandidate, albedo under secondaryEventResults/comparisonTests.
    r = parse_dv_xml(report, period_days=15.53)
    assert r.weak_secondary_max_mes == pytest.approx(2.72)
    assert r.weak_secondary_depth_ppm == pytest.approx(345.33)
    assert r.albedo_comparison_statistic == pytest.approx(1.23)


def test_keeps_both_tic_and_control_centroid_offsets(report):
    # The control offset calibrates a systematic shared by the pair; the TIC
    # offset alone reads as a large centroid shift when nothing moved.
    r = parse_dv_xml(report, period_days=15.53)
    assert r.mean_sky_offset == pytest.approx(114.33)
    assert r.control_sky_offset == pytest.approx(112.18)
    assert r.mean_sky_offset_uncertainty == pytest.approx(26.36)


def test_stellar_mass_is_derived_from_density_and_radius(report):
    # DV publishes density and radius but not mass.
    r = parse_dv_xml(report, period_days=15.53)
    assert r.stellar_mass == pytest.approx(2.666 / 1.408 * 0.6059**3, rel=1e-6)


def test_a_target_with_no_tce_still_yields_stellar_scalars(tmp_path):
    path = tmp_path / "empty_dvr.xml"
    path.write_text(
        f'<?xml version="1.0"?><dv:dvTargetResults xmlns:dv="{NS}" ticId="1" '
        f'sectorsObserved="0011"><dv:effectiveTemp value="5000.0"/>'
        f"</dv:dvTargetResults>"
    )
    r = parse_dv_xml(path, period_days=10.0)
    assert r.tic_id == 1 and r.n_planet_candidates == 0
    assert r.effective_temp == pytest.approx(5000.0)
    assert r.n_difference_images == 0
    assert r.max_multiple_event_sigma is None
