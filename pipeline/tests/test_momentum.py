"""The momentum-dump view — TESS reaction-wheel desaturation, folded on the transit.

The tests that matter are the ones about the *denominator*. The flagged cadences
are missing from every cached light curve, so the naive fold finds no dumps
anywhere and produces an all-zero branch that looks like a feature — the same
class of defect `_gap_view` documents and `DVDifferenceImage.declined` exists
for. What is asserted here is that the lost cadences come back, that they come
back at the target's own cadence rather than the reference curve's, and that a
bin nobody observed stays distinguishable from a bin observed and found clean.
"""

from __future__ import annotations

import numpy as np
import pytest

from exoplanet_hunter.preprocess.momentum import (
    MOMENTUM_CHANNELS,
    build_momentum_dump_view,
    dump_events,
    empty_momentum_dump_view,
)

BINS = 201
CADENCE = 120.0 / 86400.0


def sector_times(start: float = 1325.0, days: float = 27.0, cadence: float = CADENCE):
    return np.arange(start, start + days, cadence)


def dumps_every(period_days: float, start: float, stop: float, n_cadences: int = 7):
    """Flagged cadences shaped like the archive: short bursts on a fixed spacing."""
    return np.concatenate(
        [t + np.arange(n_cadences) * CADENCE for t in np.arange(start, stop, period_days)]
    )


class TestDumpEvents:
    def test_bursts_group_into_one_event_each(self):
        times = dumps_every(2.5, 1327.8, 1352.0)
        events = dump_events(times)
        assert len(events) == 10
        for start, end in events:
            assert end - start == pytest.approx(6 * CADENCE, abs=1e-9)

    def test_no_flagged_cadences_is_no_events_rather_than_an_error(self):
        assert dump_events(np.empty(0)) == []


class TestBuildMomentumDumpView:
    def test_shape_and_the_absent_encoding(self):
        empty = empty_momentum_dump_view(BINS)
        assert empty.shape == (BINS, MOMENTUM_CHANNELS)
        assert not empty.any()

    def test_a_dump_under_the_transit_shows_up_at_phase_zero(self):
        # Period locked to the dump spacing, epoch on a dump: every dump lands
        # at phase 0 and the view has to say so.
        time = sector_times()
        dumps = dumps_every(2.5, 1327.8, 1352.0)
        view = build_momentum_dump_view(
            time, dumps, period=2.5, t0=1327.8 + 3 * CADENCE, half_window=0.05, n_bins=BINS
        )
        centre = BINS // 2
        assert view[centre, 0] > 0.0
        assert view[centre, 0] == view[:, 0].max()
        # Away from the transit there is nothing. Some of those bins are
        # genuinely empty rather than clean — at P=2.5 d a 201-bin window is
        # 107 s wide against a 120 s cadence, so a few bins hold no cadence at
        # all, exactly as they do in `local_view` at the same period. Presence
        # is what keeps the two zeros apart.
        edge = np.r_[0:20, BINS - 20 : BINS]
        assert view[edge, 0].max() == 0.0
        assert view[:, 1].max() == 1.0
        assert view[:, 1].sum() < BINS

    def test_a_dump_away_from_the_transit_does_not_reach_phase_zero(self):
        time = sector_times()
        dumps = dumps_every(2.5, 1327.8, 1352.0)
        # Same dumps, epoch shifted half a period: they now fold to the window edge.
        view = build_momentum_dump_view(
            time, dumps, period=2.5, t0=1327.8 + 1.25, half_window=0.5, n_bins=BINS
        )
        centre = BINS // 2
        assert view[centre, 0] == 0.0
        assert view[:, 0].max() > 0.0

    def test_the_lost_cadences_are_restored_or_the_branch_is_all_zero(self):
        # The defect this module exists for: our cached curves have the dump
        # cadences removed, so a fold over the surviving times alone finds
        # nothing at any phase.
        time = sector_times()
        dumps = dumps_every(2.5, 1327.8, 1352.0)
        with_restore = build_momentum_dump_view(
            time, dumps, period=2.5, t0=1327.8, half_window=0.05, n_bins=BINS
        )
        without_any = build_momentum_dump_view(
            time, np.empty(0), period=2.5, t0=1327.8, half_window=0.05, n_bins=BINS
        )
        assert with_restore[:, 0].max() > 0.0
        assert without_any[:, 0].max() == 0.0
        assert without_any[:, 1].max() == 1.0  # measured, and clean

    def test_restored_cadences_use_the_target_cadence_not_the_reference_one(self):
        # A 200-s FFI target lost fewer cadences to the same dump than a 120-s
        # target did. Counting the reference curve's cadences would overstate it.
        dumps = dumps_every(2.5, 1327.8, 1352.0)
        fine = build_momentum_dump_view(
            sector_times(cadence=CADENCE),
            dumps,
            period=2.5,
            t0=1327.8 + 3 * CADENCE,
            half_window=0.05,
            n_bins=BINS,
        )
        coarse = build_momentum_dump_view(
            sector_times(cadence=200.0 / 86400.0),
            dumps,
            period=2.5,
            t0=1327.8 + 3 * CADENCE,
            half_window=0.05,
            n_bins=BINS,
        )
        # Both see the dump; neither reports the other's cadence count.
        assert fine[:, 0].max() > 0.0
        assert coarse[:, 0].max() > 0.0
        assert fine[:, 0].sum() != pytest.approx(coarse[:, 0].sum())

    def test_dumps_from_a_sector_the_target_was_not_on_are_not_its_dumps(self):
        # A dump 300 days after the target's last cadence is not a cadence it
        # lost, and folding it in would put a systematic at a phase where nobody
        # was watching this star.
        time = sector_times()
        elsewhere = dumps_every(2.5, 1625.0, 1650.0)
        view = build_momentum_dump_view(
            time, elsewhere, period=2.5, t0=1327.8, half_window=0.05, n_bins=BINS
        )
        assert view[:, 0].max() == 0.0
        assert view[:, 1].max() == 1.0

    def test_an_unobserved_bin_is_absent_and_a_clean_bin_is_present(self):
        # One narrow segment and a period that leaves most of the local window
        # unsampled: the two zeros must not be the same zero.
        time = np.arange(1325.0, 1325.4, CADENCE)
        view = build_momentum_dump_view(
            time, np.empty(0), period=40.0, t0=1325.2, half_window=0.5, n_bins=BINS
        )
        assert view[:, 1].min() == 0.0
        assert view[:, 1].max() == 1.0
        assert ((view[:, 0] == 0.0) | (view[:, 1] == 1.0)).all()

    def test_the_fraction_never_leaves_zero_one(self):
        view = build_momentum_dump_view(
            sector_times(),
            dumps_every(2.5, 1327.8, 1352.0),
            period=2.5,
            t0=1327.8 + 3 * CADENCE,
            half_window=0.05,
            n_bins=BINS,
        )
        assert view[:, 0].min() >= 0.0
        assert view[:, 0].max() <= 1.0

    @pytest.mark.parametrize(
        "period,t0", [(float("nan"), 1325.0), (0.0, 1325.0), (2.5, float("nan"))]
    )
    def test_an_unusable_ephemeris_is_absent_rather_than_folded_on_a_guess(self, period, t0):
        view = build_momentum_dump_view(
            sector_times(),
            dumps_every(2.5, 1327.8, 1352.0),
            period=period,
            t0=t0,
            half_window=0.05,
            n_bins=BINS,
        )
        assert not view.any()

    def test_a_target_with_no_cadences_is_absent(self):
        view = build_momentum_dump_view(
            np.empty(0),
            dumps_every(2.5, 1327.8, 1352.0),
            period=2.5,
            t0=1327.8,
            half_window=0.05,
            n_bins=BINS,
        )
        assert not view.any()
