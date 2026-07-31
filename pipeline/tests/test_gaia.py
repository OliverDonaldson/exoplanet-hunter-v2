"""Gaia DR2 -> DR3 RUWE matching.

`dr2_neighbourhood` is many-to-many: where DR3 resolved a blend, one DR2
source maps to several DR3 sources. Taking an arbitrary row attaches a
*neighbour's* RUWE to our target — a plausible number for the wrong star.
"""

from __future__ import annotations

import pandas as pd

from exoplanet_hunter.data.gaia import _best_match


def neighbours() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dr2_source_id": [111, 111, 111, 222],
            "dr3_source_id": [1001, 1002, 1003, 2001],
            "ruwe": [1.02, 8.90, 4.40, 1.15],
            "angular_distance": [0.01, 0.55, 1.20, 0.02],
            "magnitude_difference": [0.001, 2.5, 3.1, 0.0],
        }
    )


def test_keeps_the_nearest_dr3_match():
    best = _best_match(neighbours()).set_index("dr2_source_id")
    assert best.loc[111, "dr3_source_id"] == 1001
    assert best.loc[111, "ruwe"] == 1.02  # not the 8.90 of a 0.55" neighbour
    assert best.loc[222, "dr3_source_id"] == 2001


def test_records_how_many_candidates_there_were():
    # >1 means DR3 split what DR2 saw as one source — itself a blend hint, and
    # something a consumer should be able to mask on rather than inherit blind.
    best = _best_match(neighbours()).set_index("dr2_source_id")
    assert best.loc[111, "n_dr3_candidates"] == 3
    assert best.loc[222, "n_dr3_candidates"] == 1


def test_one_row_per_dr2_source():
    best = _best_match(neighbours())
    assert len(best) == best["dr2_source_id"].nunique() == 2
