"""Re-gridding DV difference-image stamps onto a fixed grid.

The tests that matter are the ones separating the three states that all look
like zeros — no product, a sector DV declined, and a sector measured flat. A
re-grid that collapses them passes every shape assertion and quietly tells the
model a star did not move when nobody looked.

Grid and slot counts are read from the module rather than restated, so a
deliberate change to either does not need the tests edited to match.
"""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.data.dv_xml import DVDifferenceImage
from exoplanet_hunter.preprocess.diffimage import (
    DIFF_CHANNELS,
    DIFF_GRID,
    MAX_DIFF_SECTORS,
    build_difference_views,
    empty_difference_views,
    regrid_stamp,
)

SENTINEL = -1.0


def make_image(
    *,
    height: int = 11,
    width: int = 11,
    sector: int = 1,
    quality: float | None = 0.9,
    quality_valid: bool = True,
    difference: np.ndarray | None = None,
    out_of_transit: np.ndarray | None = None,
    uncertainty: float = 0.1,
    row0: int = 400,
    col0: int = 700,
) -> DVDifferenceImage:
    """A dense stamp in the shape DV actually publishes: a filled bounding box."""
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    n = height * width
    diff = np.ones(n, dtype=np.float32) if difference is None else difference.ravel()
    out = np.full(n, 10.0, dtype=np.float32) if out_of_transit is None else out_of_transit.ravel()
    return DVDifferenceImage(
        sector=sector,
        ccd_rows=(rows.ravel() + row0).astype(np.int32),
        ccd_cols=(cols.ravel() + col0).astype(np.int32),
        flux_difference=diff.astype(np.float32),
        flux_difference_uncertainty=np.full(n, uncertainty, dtype=np.float32),
        flux_in_transit=np.zeros(n, dtype=np.float32),
        flux_out_of_transit=out.astype(np.float32),
        quality_metric=quality,
        quality_valid=quality_valid,
        n_transits=3,
        n_cadences_in_transit=30,
    )


class TestRegridStamp:
    def test_an_11px_stamp_lands_centred_and_the_rest_is_marked_absent(self):
        stamp = regrid_stamp(make_image(height=11, width=11))
        assert stamp.shape == (DIFF_GRID, DIFF_GRID, DIFF_CHANNELS)
        present = stamp[..., 2]
        assert present.sum() == 11 * 11
        edge = (DIFF_GRID - 11) // 2
        assert present[edge : edge + 11, edge : edge + 11].all()
        # Padding is absent, not a measured zero.
        assert present[0, 0] == 0.0

    def test_the_regrid_recovers_the_original_pixels_exactly(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 1.0, size=(11, 11)).astype(np.float32)
        image = make_image(difference=values, out_of_transit=np.full((11, 11), 4.0, np.float32))
        stamp = regrid_stamp(image)
        edge = (DIFF_GRID - 11) // 2
        recovered = stamp[edge : edge + 11, edge : edge + 11, 0] * 4.0
        # Exact rather than approximate: nothing is interpolated onto the grid.
        assert np.allclose(recovered, values, rtol=0, atol=1e-5)

    def test_both_channels_share_one_scale_so_depth_survives(self):
        # Scaling each channel by its own peak would send both to 1.0 and throw
        # away how deep the difference is relative to the star.
        image = make_image(
            difference=np.full((11, 11), 2.0, np.float32),
            out_of_transit=np.full((11, 11), 8.0, np.float32),
        )
        stamp = regrid_stamp(image)
        assert stamp[..., 0].max() == pytest.approx(0.25)
        assert stamp[..., 1].max() == pytest.approx(1.0)

    def test_a_declined_sector_is_not_a_measurement(self):
        declined = make_image(difference=np.zeros((11, 11), np.float32), uncertainty=SENTINEL)
        assert declined.declined
        assert regrid_stamp(declined) is None

    def test_a_measured_flat_stamp_is_a_measurement(self):
        # The whole point: identical zeros to the declined case above, opposite
        # meaning, and the presence channel is what carries the difference.
        flat = make_image(difference=np.zeros((11, 11), np.float32), uncertainty=0.1)
        assert not flat.declined
        stamp = regrid_stamp(flat)
        assert stamp is not None
        assert (stamp[..., 0] == 0.0).all()
        assert stamp[..., 2].sum() == 11 * 11

    def test_no_usable_brightness_scale_reads_as_absent(self):
        image = make_image(out_of_transit=np.zeros((11, 11), np.float32))
        assert regrid_stamp(image) is None

    @pytest.mark.parametrize("height,width", [(13, 11), (15, 15), (17, 11), (25, 25)])
    def test_every_shape_in_the_archive_regrids(self, height, width):
        stamp = regrid_stamp(make_image(height=height, width=width))
        assert stamp.shape == (DIFF_GRID, DIFF_GRID, DIFF_CHANNELS)
        kept = min(height, DIFF_GRID) * min(width, DIFF_GRID)
        assert stamp[..., 2].sum() == kept

    def test_an_oversized_stamp_is_cropped_about_its_centre(self):
        values = np.zeros((25, 25), dtype=np.float32)
        values[12, 12] = 5.0  # centre survives a centred crop
        values[0, 0] = 9.0  # corner does not
        stamp = regrid_stamp(
            make_image(
                height=25,
                width=25,
                difference=values,
                out_of_transit=np.full((25, 25), 1.0, np.float32),
            )
        )
        assert stamp[..., 0].max() == pytest.approx(5.0)

    def test_a_pixel_list_with_a_hole_raises(self):
        # The dense-box property the module is built on. A hole scattered into
        # zeros would read as a measured zero pixel.
        image = make_image()
        holed = DVDifferenceImage(
            sector=image.sector,
            ccd_rows=image.ccd_rows[:-1],
            ccd_cols=image.ccd_cols[:-1],
            flux_difference=image.flux_difference[:-1],
            flux_difference_uncertainty=image.flux_difference_uncertainty[:-1],
            flux_in_transit=image.flux_in_transit[:-1],
            flux_out_of_transit=image.flux_out_of_transit[:-1],
            quality_metric=image.quality_metric,
            quality_valid=image.quality_valid,
            n_transits=image.n_transits,
            n_cadences_in_transit=image.n_cadences_in_transit,
        )
        with pytest.raises(ValueError, match="fills its box exactly"):
            regrid_stamp(holed)


class TestBuildDifferenceViews:
    def test_shapes_and_slot_presence(self):
        stamps, quality = build_difference_views([make_image(sector=s) for s in (1, 2, 3)])
        assert stamps.shape == (MAX_DIFF_SECTORS, DIFF_GRID, DIFF_GRID, DIFF_CHANNELS)
        assert quality.shape == (MAX_DIFF_SECTORS, 2)
        assert quality[:, 1].tolist() == [1.0, 1.0, 1.0] + [0.0] * (MAX_DIFF_SECTORS - 3)
        # An unused slot is all zeros, stamp and quality alike.
        assert (stamps[3:] == 0.0).all()

    def test_sectors_are_kept_highest_quality_first(self):
        images = [
            make_image(sector=1, quality=0.2),
            make_image(sector=2, quality=0.9),
            make_image(sector=3, quality=0.5),
        ]
        _stamps, quality = build_difference_views(images)
        assert quality[:3, 0].tolist() == pytest.approx([0.9, 0.5, 0.2])

    def test_the_cap_drops_the_worst_sectors_not_the_latest(self):
        images = [make_image(sector=s, quality=s / 100.0) for s in range(1, MAX_DIFF_SECTORS + 4)]
        _stamps, quality = build_difference_views(images)
        assert int(quality[:, 1].sum()) == MAX_DIFF_SECTORS
        best = sorted((s / 100.0 for s in range(1, MAX_DIFF_SECTORS + 4)), reverse=True)
        assert quality[:, 0].tolist() == pytest.approx(best[:MAX_DIFF_SECTORS])

    def test_ordering_does_not_depend_on_input_order(self):
        images = [make_image(sector=s, quality=0.5) for s in (5, 1, 3)]
        _stamps, quality = build_difference_views(images)
        # Ties break on sector number, so the result is reproducible.
        _stamps2, quality2 = build_difference_views(list(reversed(images)))
        assert np.array_equal(quality, quality2)

    def test_declined_sectors_take_no_slot(self):
        images = [
            make_image(sector=1),
            make_image(sector=2, difference=np.zeros((11, 11), np.float32), uncertainty=SENTINEL),
            make_image(sector=3),
        ]
        _stamps, quality = build_difference_views(images)
        assert int(quality[:, 1].sum()) == 2

    def test_an_unmeasurable_quality_still_counts_as_present(self):
        # "measured, trust it least" is not the same as "absent", and the branch
        # has to be able to tell them apart.
        _stamps, quality = build_difference_views(
            [make_image(sector=1, quality=None, quality_valid=False)]
        )
        assert quality[0].tolist() == [0.0, 1.0]

    def test_a_target_with_no_images_is_absent_everywhere(self):
        stamps, quality = build_difference_views([])
        empty_stamps, empty_quality = empty_difference_views()
        assert np.array_equal(stamps, empty_stamps)
        assert np.array_equal(quality, empty_quality)
        assert stamps[..., 2].sum() == 0.0
