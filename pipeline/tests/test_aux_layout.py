"""The aux layout is one implementation — training, serving and batch scoring
all go through build_aux_row, so a layout change cannot drift between them."""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.features.aux import (
    CENTROID_COL,
    LEGACY_AUX_DIM,
    TRAINING_AUX_DIM,
    build_aux_row,
)

BASE = dict(
    teff=5800.0,
    radius=1.0,
    logg=4.44,
    tmag=10.5,
    depth=0.0012,
    duration=0.12,
    period=3.5,
    centroid_snr=1.7,
)


def test_legacy_layout_is_unchanged():
    row = build_aux_row(LEGACY_AUX_DIM, catalogue_snr=22.0, **BASE)
    assert row.dtype == np.float32
    assert row.tolist() == pytest.approx(
        [5800.0, 1.0, 4.44, 10.5, 0.0012, 0.12, float(np.log(3.5)), 22.0, 1.7], rel=1e-6
    )


def test_eight_dim_drops_centroid_only():
    nine = build_aux_row(9, catalogue_snr=22.0, **BASE)
    eight = build_aux_row(8, catalogue_snr=22.0, **BASE)
    assert eight.tolist() == nine[:8].tolist()


def test_vetting_layout_puts_pink_snr_at_the_snr_slot():
    row = build_aux_row(
        TRAINING_AUX_DIM,
        pink_snr=9.1,
        catalogue_snr=22.0,
        oe_depth_sigma=0.4,
        oe_timing_sigma=1.1,
        secondary_sig=2.2,
        q_ratio=0.9,
        **BASE,
    )
    assert len(row) == 13
    assert row[7] == np.float32(9.1)  # pink_snr replaces the catalogue snr
    assert row[9:].tolist() == pytest.approx([0.4, 1.1, 2.2, 0.9], rel=1e-6)


def test_centroid_column_is_stable_across_widths():
    # The fitted aux pipeline indexes this column; moving it silently
    # rescales the wrong feature at serve time.
    for dim in (9, TRAINING_AUX_DIM):
        row = build_aux_row(dim, catalogue_snr=1.0, pink_snr=1.0, **BASE)
        assert row[CENTROID_COL] == np.float32(1.7)


def test_absent_and_invalid_values_become_nan():
    row = build_aux_row(TRAINING_AUX_DIM, period=3.5)
    assert np.isnan(row[[0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12]]).all()
    assert np.isnan(build_aux_row(9, period=0.0)[6])  # log of a bad period
    assert np.isnan(build_aux_row(9, period=3.5, teff="not a number")[0])
