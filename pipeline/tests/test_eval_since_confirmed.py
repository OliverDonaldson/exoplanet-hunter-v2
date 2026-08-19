"""Prospective evaluation: candidates unlabelled at training time, labelled since.

The test that matters is the column collision. Both frames carry `disposition`,
and the held-out one is `PC` on every row by definition — so a merge that lets
pandas suffix them silently labels every prospective target negative and the
whole instrument returns zeros that look like a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_since_confirmed import flipped_holdout


def write(tmp_path: Path, holdout: pd.DataFrame, catalogue: pd.DataFrame) -> tuple[Path, Path]:
    h, c = tmp_path / "holdout.parquet", tmp_path / "catalogue.parquet"
    holdout.to_parquet(h)
    catalogue.to_parquet(c)
    return h, c


def holdout_frame(n: int = 3) -> pd.DataFrame:
    """Held-out candidates as the training build wrote them: all still `PC`."""
    return pd.DataFrame(
        {
            "tic_id": range(1, n + 1),
            "period": [4.0] * n,
            "t0": [1.0] * n,
            "duration": [0.1] * n,
            "disposition": ["PC"] * n,
            "name": [f"TOI-{i}" for i in range(1, n + 1)],
        }
    )


def test_the_label_comes_from_the_disposition_that_arrived_later(tmp_path):
    """The bug this file exists for. Read from the held-out column instead and
    every row is `PC`, so `y_true` is 0 everywhere — including for the confirmed
    planets, which is exactly what the 2026-07-20 artefact recorded."""
    catalogue = pd.DataFrame(
        {
            "tic_id": [1, 2, 3],
            "period_days": [4.0, 4.0, 4.0],
            "disposition": ["CP", "FP", "KP"],
        }
    )
    out = flipped_holdout(*write(tmp_path, holdout_frame(), catalogue))
    assert out["y_true"].tolist() == [1, 0, 1]
    # The training-time value is still available and still says PC — the point is
    # that it is not what the label is taken from.
    assert out["disposition"].unique().tolist() == ["PC"]


def test_a_single_class_set_raises_rather_than_scoring(tmp_path):
    """AUC is undefined and recall is 0 or 1 by construction. Returning it would
    be a number where the honest answer is "nothing to rank"."""
    catalogue = pd.DataFrame(
        {"tic_id": [1, 2, 3], "period_days": [4.0] * 3, "disposition": ["FP", "FP", "FP"]}
    )
    with pytest.raises(ValueError, match="nothing to rank"):
        flipped_holdout(*write(tmp_path, holdout_frame(), catalogue))


def test_a_different_planet_on_the_same_star_is_not_matched(tmp_path):
    """Same TIC, unrelated period: a second planet's disposition says nothing
    about this candidate."""
    catalogue = pd.DataFrame(
        {"tic_id": [1, 2, 3], "period_days": [4.0, 91.3, 4.0], "disposition": ["CP", "CP", "FP"]}
    )
    out = flipped_holdout(*write(tmp_path, holdout_frame(), catalogue))
    assert out["tic_id"].tolist() == [1, 3]


def test_candidates_still_awaiting_a_disposition_are_excluded(tmp_path):
    """A target still `PC` in the refreshed catalogue has not been followed up,
    so it carries no ground truth to score against."""
    catalogue = pd.DataFrame(
        {"tic_id": [1, 2, 3], "period_days": [4.0] * 3, "disposition": ["CP", "PC", "FP"]}
    )
    out = flipped_holdout(*write(tmp_path, holdout_frame(), catalogue))
    assert out["tic_id"].tolist() == [1, 3]
