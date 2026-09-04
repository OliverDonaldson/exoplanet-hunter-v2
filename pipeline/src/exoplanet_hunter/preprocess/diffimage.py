"""Difference-image stamps, re-gridded from CCD pixels onto a fixed grid.

DV publishes one difference image per sector as a list of CCD pixels sized to
that target's aperture. A branch needs a fixed shape, so the stamps are placed
on a common grid here.

**The re-grid is exact, not a resampling.** Measured 2026-08-17 over all 33,540
difference images in the archive, a stamp's pixel list fills its bounding box
completely — fill fraction 1.000 at the minimum, zero repeated coordinates — so
scattering it back into a dense rectangle recovers the original array rather
than approximating it. `regrid_stamp` asserts that rather than assuming it.

**Nothing is interpolated onto the grid, deliberately.** Resampling an 11x11
aperture up to the grid size would make one grid cell a different number of CCD
pixels on every target, and the branch's whole subject is *where* flux moved,
in pixels. The same argument stops the periodogram view resampling onto a
per-target period grid and stops the centroid view normalising to a fixed depth.
So a stamp is placed at native scale and the rest of the grid is padding, marked
absent in the presence channel.

**Three states, kept apart.** A target with no DV product at all, a sector DV
declined to measure (`DVDifferenceImage.declined`), and a sector measured and
genuinely featureless are three different things that all look like zeros. Only
the third is evidence. The first two carry `present` 0 and the third carries
`present` 1 on a flat stamp.

**The stamp carries its own origin, added 2026-08-27 (roadmap 4.2c Phase 1).**
Until then the branch was fed a difference and a reference image and no target
position, so the only thing marking the star was `_centred_slice` putting the
bounding box in the middle of the grid — a *placement*, not a measurement.
Roadmap 4.2b finding 2 named that as the mechanism behind stage 9's null:
ExoMiner++ feeds three tensors where we fed one, and the missing one is
`target_imgs`, a marker at the star's own pixel. Channel 2 is now that marker.
Measured over all 38,964 non-declined images in the archive, the star sits a
median **0.84 px** from the box centre and lands in a **different pixel** than
it on **77.8%** of stamps, so the placement was never a stand-in for the
measurement.
"""

from __future__ import annotations

import numpy as np

from exoplanet_hunter.data.dv_xml import DVDifferenceImage

#: Side of the fixed grid, in CCD pixels.
#:
#: Chosen as **the smallest grid on which no stamp loses its peak pixel**, which
#: is the pixel the diagnostic is about — a centroid shift is read from where the
#: difference is brightest, so a crop that keeps most of the flux but moves the
#: peak out of frame has removed the measurement while looking almost lossless.
#: Measured 2026-08-17 over the whole archive, peak pixels lost by a centred crop:
#: 11x11 -> 366 stamps, 13x13 -> 109, 15x15 -> 24, **17x17 -> 0**. Going wider
#: buys nothing but padding: 17x17 already crops only 39 stamps of 33,540 (0.12%),
#: and 25x25 — the smallest fully lossless grid — would leave a typical 11x11
#: stamp sitting in 19% of its own view.
DIFF_GRID = 17

#: Sectors kept per target. The count is long-tailed — median 3, 90th percentile
#: 11, maximum 43 — and every extra slot costs the shard set 1.2 kB on every row
#: including the 58.9% that have no difference image at all. Eight covers 86.0%
#: of targets whole; the rest keep their eight highest-quality sectors.
MAX_DIFF_SECTORS = 8

#: (grid, grid, [difference, out-of-transit, target, present]) per sector.
#:
#: **`present` stays last.** `cnn_branches._gated`, `SectorPresence` and
#: `viewset_augment._augment_view` all read the presence channel as `[..., -1]`,
#: which is the project's convention on every view. The target marker is
#: inserted *before* it rather than appended, so adding a channel here does not
#: silently make three unrelated modules read a target marker as a presence flag.
DIFF_CHANNELS = 4

#: Index of the target-position marker within a stamp's channels.
TARGET_CHANNEL = 2


def _dense(image: DVDifferenceImage, values: np.ndarray) -> np.ndarray:
    """Scatter one sparse pixel list into its dense bounding box."""
    rows = image.ccd_rows - int(image.ccd_rows.min())
    cols = image.ccd_cols - int(image.ccd_cols.min())
    height, width = image.shape
    grid = np.zeros((height, width), dtype=np.float64)
    grid[rows, cols] = values
    return grid


def _centred_slice(size: int, grid: int) -> tuple[slice, slice]:
    """Where a `size`-long axis sits on a `grid`-long one, cropping if it must."""
    if size <= grid:
        start = (grid - size) // 2
        return slice(start, start + size), slice(0, size)
    start = (size - grid) // 2
    return slice(0, grid), slice(start, start + grid)


def _target_marker(
    position: tuple[float, float],
    origin: tuple[int, int],
    rows: tuple[slice, slice],
    cols: tuple[slice, slice],
    grid: int,
) -> np.ndarray | None:
    """Unit mass at the star's sub-pixel position on the fixed grid, or None.

    ExoMiner writes a hard one-hot at the *rounded* target pixel
    (`utils_diff_img.py::create_target_image`) and recovers the sub-pixel part by
    up-sampling the whole stamp by an integer factor first, so the rounded pixel
    of the up-sampled grid encodes a fraction of an original one.

    **We deposit the mass bilinearly instead, and the reason is the grid.**
    `DIFF_GRID` is one CCD pixel per cell by construction — the same refusal to
    resample that `_periodogram_view` and `_centroid_view` make — so
    up-sampling to carry sub-pixel information would undo the property the whole
    module rests on. Splitting one unit of mass over the (up to) four pixels the
    position falls between carries the same information on the native grid: at
    an exact pixel centre it reduces to ExoMiner's one-hot, and off-centre the
    ratio between the four weights *is* the offset. It is not free information
    either — measured over the archive, the residual after rounding is uniform
    (sd 0.2884 against 1/sqrt(12) = 0.2887), so a hard one-hot would discard a
    full pixel's worth of position on every stamp.

    **The mass is not renormalised when part of it falls off the grid.** A
    marker summing to less than one says the star is near the frame edge, which
    is true; rescaling it back to one would move the implied centroid inwards
    and state a position the data does not support. None is returned only when
    *no* mass lands at all — the star is outside the stamp entirely, and a
    difference image with no origin in frame is exactly what this channel
    exists to stop the branch being fed. Measured 2026-08-27: that happens on
    **0** of the archive's 38,964 non-declined images, the largest offset from
    the box centre being 3.24 px on a grid 17 wide.
    """
    rows_to, rows_from = rows
    cols_to, cols_from = cols
    # Box frame -> grid frame, through exactly the placement `_centred_slice`
    # applies to the pixels, so the marker moves with the stamp it belongs to.
    row = (position[0] - origin[0]) - rows_from.start + rows_to.start
    col = (position[1] - origin[1]) - cols_from.start + cols_to.start

    marker = np.zeros((grid, grid), dtype=np.float32)
    row_floor, col_floor = int(np.floor(row)), int(np.floor(col))
    row_frac, col_frac = row - row_floor, col - col_floor
    landed = 0.0
    for row_step, row_weight in ((0, 1.0 - row_frac), (1, row_frac)):
        for col_step, col_weight in ((0, 1.0 - col_frac), (1, col_frac)):
            weight = row_weight * col_weight
            r, c = row_floor + row_step, col_floor + col_step
            if weight > 0.0 and 0 <= r < grid and 0 <= c < grid:
                marker[r, c] = weight
                landed += weight
    return marker if landed > 0.0 else None


def regrid_stamp(image: DVDifferenceImage) -> np.ndarray | None:
    """One sector's stamp on the fixed grid, or None if it is not a measurement.

    Returns `(DIFF_GRID, DIFF_GRID, 4)` =
    `[difference, out-of-transit, target, present]`. Both flux channels are
    divided by the out-of-transit peak, so the difference reads as a fraction of
    the star's own brightness and is comparable between a bright star and a
    faint one. Normalising each channel by its *own* peak would send every
    difference image to 1.0 and destroy the depth information, the same trap the
    odd/even views document.

    The `target` channel is the star's own position on this grid, at sub-pixel
    resolution — see `_target_marker`. It is what makes a *centroid offset*
    computable from the branch's inputs: with it, "the flux moved two pixels
    north-east of the star" is a subtraction the branch can learn; without it,
    the only reference point is where the bounding box happened to be placed.

    None means "not a measurement", and the caller must encode that as absent
    rather than as a flat stamp: no pixels, a sector DV declined, an
    out-of-transit image with no usable scale, or a target position DV left
    undefined or placed outside the frame.
    """
    if not image.n_pixels or image.declined:
        return None
    position = image.target_position
    if position is None:
        # A stamp with no origin is the state this branch was built without and
        # roadmap 4.2b finding 2 identifies as the reason it did nothing. It is
        # absent rather than centred-by-default: a marker at the middle would be
        # the placement artefact again, restated as if it were a measurement.
        return None

    height, width = image.shape
    if image.n_pixels != height * width:
        # The dense-box property this module is built on. It holds on every
        # image in the archive, and if a future product breaks it the scatter
        # below would silently leave holes reading as measured zeros.
        raise ValueError(
            f"difference image for sector {image.sector} has {image.n_pixels} pixels "
            f"in a {height}x{width} bounding box; re-gridding assumes the pixel list "
            "fills its box exactly"
        )

    difference = _dense(image, np.nan_to_num(image.flux_difference, nan=0.0))
    out_of_transit = _dense(image, np.nan_to_num(image.flux_out_of_transit, nan=0.0))

    scale = float(np.nanmax(np.abs(out_of_transit))) if out_of_transit.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        # No brightness to measure the difference against. A stamp scaled by a
        # fallback 1.0 would be in raw electrons on a handful of targets and in
        # fractional units on every other, which is worse than absent.
        return None

    rows_to, rows_from = _centred_slice(height, DIFF_GRID)
    cols_to, cols_from = _centred_slice(width, DIFF_GRID)
    marker = _target_marker(
        position,
        (int(image.ccd_rows.min()), int(image.ccd_cols.min())),
        (rows_to, rows_from),
        (cols_to, cols_from),
        DIFF_GRID,
    )
    if marker is None:
        return None

    stamp = np.zeros((DIFF_GRID, DIFF_GRID, DIFF_CHANNELS), dtype=np.float32)
    stamp[rows_to, cols_to, 0] = difference[rows_from, cols_from] / scale
    stamp[rows_to, cols_to, 1] = out_of_transit[rows_from, cols_from] / scale
    stamp[..., TARGET_CHANNEL] = marker
    stamp[rows_to, cols_to, -1] = 1.0
    return stamp


def build_difference_views(
    images: list[DVDifferenceImage],
    *,
    max_sectors: int = MAX_DIFF_SECTORS,
    grid: int = DIFF_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Every sector's stamp, plus the quality view the branch attends with.

    Returns `(stamps, quality)`:

    - `stamps`  : `(max_sectors, grid, grid, 4)`, unused slots all zero.
    - `quality` : `(max_sectors, 2)` = `[quality_metric, present]`.

    Sectors are kept **highest quality first**, so a target with more sectors
    than slots keeps its most trustworthy ones rather than its earliest. That
    choice is not free: it means the retained quality distribution depends on how
    many sectors a target had, and a 40-sector target's kept eight are better
    than an 8-sector target's. The alternative — keeping the first eight by
    sector number — trades that for feeding the branch images DV itself flags as
    untrustworthy, which is the failure this branch's attention exists to avoid.
    Ties break on sector number so the result does not depend on input order.

    `quality_metric` is passed through unscaled. It is *not* a 0-1 score despite
    reading like one — measured over the archive it runs -0.72 to 1.0, median
    0.86 — and it is only ever used as a relative weight between one target's own
    sectors, so the offset does not matter and rescaling it would only invent a
    range DV does not claim.
    """
    stamps = np.zeros((max_sectors, grid, grid, DIFF_CHANNELS), dtype=np.float32)
    quality = np.zeros((max_sectors, 2), dtype=np.float32)

    usable = []
    for image in images:
        stamp = regrid_stamp(image)
        if stamp is None:
            continue
        # A sector whose quality DV could not compute still carries a stamp; it
        # sorts last and the branch sees a weight of zero beside a present flag
        # of one, which is "measured, trust it least" rather than "absent".
        metric = image.quality_metric if image.quality_valid else None
        usable.append((stamp, metric, image.sector))

    usable.sort(key=lambda item: (-(item[1] if item[1] is not None else -np.inf), item[2]))
    for slot, (stamp, metric, _sector) in enumerate(usable[:max_sectors]):
        stamps[slot] = stamp
        quality[slot] = (float(metric) if metric is not None else 0.0, 1.0)
    return stamps, quality


def empty_difference_views(
    *, max_sectors: int = MAX_DIFF_SECTORS, grid: int = DIFF_GRID
) -> tuple[np.ndarray, np.ndarray]:
    """The honest encoding of a target with no difference image at all.

    All zeros, presence 0 — the same convention every other view uses for a
    branch it could not build, and distinct from a measured stamp that happens
    to be flat, which carries presence 1.
    """
    return (
        np.zeros((max_sectors, grid, grid, DIFF_CHANNELS), dtype=np.float32),
        np.zeros((max_sectors, 2), dtype=np.float32),
    )
