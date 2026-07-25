"""Rebuild the labelled catalogue only — stage 1 of build_dataset, no downloads.

Hydra entry point. Usage:

    python pipeline/scripts/refresh_labels.py data=full
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from exoplanet_hunter.data.catalog import build_labels_from_cfg
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)
    catalog = build_labels_from_cfg(
        cfg.data,
        paths.data_labels,
        paths.root / "data" / "catalogue" / "candidates.parquet",
    )
    log.info(
        "[refresh-labels] %d rows (data=%s) by mission: %s",
        len(catalog),
        cfg.data.get("name", "?"),
        catalog["mission"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
