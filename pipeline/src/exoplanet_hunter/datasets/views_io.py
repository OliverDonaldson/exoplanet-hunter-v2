"""The .npz interchange format for processed views.

`build_dataset.py` / `preprocess_only.py` write a single views.npz:

    global_views   (N, 2001) float32
    local_views    (N, 201)  float32
    labels         (N,)      int8     {0, 1}
    tic_ids        (N,)      int64
    aux_features   (N, A)    float32  (optional)

This module is the reader side of that contract (ported from the V1
data_module). The in-RAM `LightcurveDataset` that used to live next to it is
gone — training streams from TFRecord shards via `datasets.pipeline` instead.
`shard_views.py` converts an .npz into shards; the RF baseline still loads
the .npz directly (its handcrafted features want plain arrays).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ViewArrays:
    global_views: np.ndarray
    local_views: np.ndarray
    labels: np.ndarray
    tic_ids: np.ndarray
    aux_features: np.ndarray | None


def load_views(path: Path) -> ViewArrays:
    """Read a processed views .npz built by `build_dataset.py`."""
    with np.load(path) as f:
        aux = f["aux_features"] if "aux_features" in f.files else None
        return ViewArrays(
            global_views=f["global_views"].astype(np.float32),
            local_views=f["local_views"].astype(np.float32),
            labels=f["labels"].astype(np.int8),
            tic_ids=f["tic_ids"].astype(np.int64),
            aux_features=aux.astype(np.float32) if aux is not None else None,
        )


def slice_views(v: ViewArrays, idx: np.ndarray) -> ViewArrays:
    """Index into a `ViewArrays` and return a new container with the slices."""
    return ViewArrays(
        global_views=v.global_views[idx],
        local_views=v.local_views[idx],
        labels=v.labels[idx],
        tic_ids=v.tic_ids[idx],
        aux_features=None if v.aux_features is None else v.aux_features[idx],
    )
