"""Interchange format for the view set.

Two files, mirroring the `views.npz` + `index.parquet` split the legacy set
uses:

    viewset.npz            named view arrays, each (N, ...) float32
    viewset_scalars.parquet  one row per example: label, tic_id, mission,
                             transit counts, DV scalars, RUWE, provenance

Kept separate from `views_io.py` so the legacy artefact that feeds the live
model is untouched while this is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoplanet_hunter.preprocess.diffimage import (
    DIFF_CHANNELS,
    DIFF_GRID,
    MAX_DIFF_SECTORS,
)
from exoplanet_hunter.preprocess.viewset import (
    GLOBAL_BINS,
    LOCAL_BINS,
    MAX_TRANSITS,
    PERIODOGRAM_BINS,
)

#: Every view array, with the per-example shape the schema guarantees. Derived
#: from the builder's own constants rather than restated: two declarations of
#: the same bin counts is the shape of every silent-shard bug this project has
#: had, and a resolution change has to be one edit or it is a mismatch waiting.
VIEW_SHAPES: dict[str, tuple[int, ...]] = {
    "global_view": (GLOBAL_BINS, 3),
    "local_view": (LOCAL_BINS, 3),
    "odd_view": (LOCAL_BINS, 3),
    "even_view": (LOCAL_BINS, 3),
    "secondary_view": (LOCAL_BINS, 3),
    "trend_view": (GLOBAL_BINS, 3),
    "centroid_view": (LOCAL_BINS, 3),
    "unfolded_view": (MAX_TRANSITS, LOCAL_BINS, 3),
    "gap_view": (GLOBAL_BINS, 2),
    "periodogram_view": (PERIODOGRAM_BINS, 2),
    "periodogram_masked_view": (PERIODOGRAM_BINS, 2),
    "difference_view": (MAX_DIFF_SECTORS, DIFF_GRID, DIFF_GRID, DIFF_CHANNELS),
    # Not a branch of its own — it is how the difference branch weights its
    # sectors, and `cnn_branches.ATTENTION_VIEWS` keeps it out of the fusion.
    "difference_quality_view": (MAX_DIFF_SECTORS, 2),
}

#: Views built from the DV report rather than from the light curve. They depend
#: on neither the bin resolution nor the epoch and duration a curve is folded
#: on, so they are **not** part of the per-target light-curve cache — which is
#: keyed on exactly those things. Keeping them out of it means adding this
#: branch costs a DV re-parse (minutes) instead of re-deriving every folded view
#: from the FITS files again (~95 min), and it stops a cache keyed on an
#: ephemeris from implying these arrays were folded on one.
DV_VIEWS: frozenset[str] = frozenset({"difference_view", "difference_quality_view"})
#: The rest: everything a `_build_one` pass produces from a light curve.
LIGHTCURVE_VIEWS: tuple[str, ...] = tuple(n for n in VIEW_SHAPES if n not in DV_VIEWS)

NPZ_NAME = "viewset.npz"
SCALARS_NAME = "viewset_scalars.parquet"


@dataclass
class ViewSetArrays:
    views: dict[str, np.ndarray]
    scalars: pd.DataFrame

    def __len__(self) -> int:
        return len(self.scalars)

    def validate(self) -> list[str]:
        """Shape and length agreement; empty list means the set is well-formed."""
        problems: list[str] = []
        n = len(self.scalars)
        for name, shape in VIEW_SHAPES.items():
            arr = self.views.get(name)
            if arr is None:
                problems.append(f"{name}: missing")
                continue
            if arr.shape != (n, *shape):
                problems.append(f"{name}: {arr.shape} but expected {(n, *shape)}")
        for column in ("label", "tic_id", "mission"):
            if column not in self.scalars.columns:
                problems.append(f"scalars: missing {column}")
        return problems

    def save(self, out_dir: Path | str) -> None:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # Typed Any because numpy's stub declares savez_compressed's second
        # positional as `allow_pickle: bool`, which a **dict of arrays trips.
        arrays: dict[str, Any] = {k: v.astype(np.float32) for k, v in self.views.items()}
        np.savez_compressed(directory / NPZ_NAME, **arrays)
        self.scalars.to_parquet(directory / SCALARS_NAME, index=False)

    @classmethod
    def load(cls, out_dir: Path | str) -> ViewSetArrays:
        directory = Path(out_dir)
        with np.load(directory / NPZ_NAME) as f:
            views = {k: f[k] for k in f.files}
        return cls(views=views, scalars=pd.read_parquet(directory / SCALARS_NAME))


def stack_view_sets(view_sets: list, scalars: pd.DataFrame) -> ViewSetArrays:
    """Stack per-target `ViewSet` objects into arrays keyed by view name."""
    views = {
        name: np.stack([getattr(vs, name) for vs in view_sets]).astype(np.float32)
        for name in VIEW_SHAPES
    }
    return ViewSetArrays(views=views, scalars=scalars.reset_index(drop=True))
