"""Interchange format for the 301/31 view set.

Two files, mirroring the `views.npz` + `index.parquet` split the legacy set
uses:

    viewset.npz            named view arrays, each (N, ...) float32
    viewset_scalars.parquet  one row per example: label, tic_id, mission,
                             transit counts, DV scalars, RUWE, provenance

Kept separate from `views_io.py` so the legacy 2001/201 artefact that feeds the
live model is untouched while this is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Every view array, with the per-example shape the schema guarantees.
VIEW_SHAPES: dict[str, tuple[int, ...]] = {
    "global_view": (301, 3),
    "local_view": (31, 3),
    "odd_view": (31, 3),
    "even_view": (31, 3),
    "secondary_view": (31, 3),
    "trend_view": (301, 3),
    "centroid_view": (31, 3),
    "unfolded_view": (20, 31, 3),
    "gap_view": (301, 2),
    "periodogram_view": (256, 2),
    "periodogram_masked_view": (256, 2),
}

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

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Typed Any because numpy's stub declares savez_compressed's second
        # positional as `allow_pickle: bool`, which a **dict of arrays trips.
        arrays: dict[str, Any] = {k: v.astype(np.float32) for k, v in self.views.items()}
        np.savez_compressed(out_dir / NPZ_NAME, **arrays)
        self.scalars.to_parquet(out_dir / SCALARS_NAME, index=False)

    @classmethod
    def load(cls, out_dir: Path) -> ViewSetArrays:
        with np.load(out_dir / NPZ_NAME) as f:
            views = {k: f[k] for k in f.files}
        return cls(views=views, scalars=pd.read_parquet(out_dir / SCALARS_NAME))


def stack_view_sets(view_sets: list, scalars: pd.DataFrame) -> ViewSetArrays:
    """Stack per-target `ViewSet` objects into arrays keyed by view name."""
    views = {
        name: np.stack([getattr(vs, name) for vs in view_sets]).astype(np.float32)
        for name in VIEW_SHAPES
    }
    return ViewSetArrays(views=views, scalars=scalars.reset_index(drop=True))
