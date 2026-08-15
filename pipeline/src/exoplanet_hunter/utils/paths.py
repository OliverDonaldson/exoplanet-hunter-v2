"""Centralised path resolution.

Hydra resolves paths from `conf/config.yaml` into an `omegaconf.DictConfig`;
this module wraps that into a typed convenience class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    #: TESS light curves. The raw cache is one directory per mission under
    #: `data/raw/`, because a single flat directory made the three missions
    #: indistinguishable except by filename prefix.
    data_raw: Path
    data_raw_kepler: Path
    data_raw_k2: Path
    data_interim: Path
    data_processed: Path
    data_labels: Path
    models: Path
    results: Path

    @classmethod
    def from_cfg(cls, cfg: Any) -> ProjectPaths:
        """Build from a Hydra/OmegaConf `paths` group."""
        p = cfg.paths
        # Siblings of the TESS directory, never string-concatenated onto it. The
        # old fallback was `p.data_raw + "_kepler"`, which produced
        # `data/raw/tess/lightcurves_kepler` the moment the caches were split by
        # mission — a path that exists nowhere and fails by re-downloading 43 GB
        # rather than by raising.
        raw = Path(p.data_raw)
        paths = cls(
            root=Path(p.root),
            data_raw=raw,
            data_raw_kepler=Path(
                getattr(p, "data_raw_kepler", None) or raw.parent.parent / "kepler" / raw.name
            ),
            data_raw_k2=Path(
                getattr(p, "data_raw_k2", None) or raw.parent.parent / "k2" / raw.name
            ),
            data_interim=Path(p.data_interim),
            data_processed=Path(p.data_processed),
            data_labels=Path(p.data_labels),
            models=Path(p.models),
            results=Path(p.results),
        )
        paths.ensure()
        return paths

    def ensure(self) -> None:
        """Create every directory if it doesn't exist."""
        for path in (
            self.data_raw,
            self.data_raw_kepler,
            self.data_raw_k2,
            self.data_interim,
            self.data_processed,
            self.data_labels,
            self.models,
            self.results,
        ):
            path.mkdir(parents=True, exist_ok=True)
