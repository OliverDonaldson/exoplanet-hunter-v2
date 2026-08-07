"""Smoke test: every salvaged module imports cleanly in the V2 layout.

This is the seed of the V2 test suite — it exists so CI fails loudly if the
extraction left a dangling import. Behavioural tests for preprocessing and
the model arrive with `feat/tfdata-pipeline`.
"""

import importlib
from pathlib import Path

import pytest

MODULES = [
    "exoplanet_hunter",
    "exoplanet_hunter.data",
    "exoplanet_hunter.data.catalog",
    "exoplanet_hunter.data.download",
    "exoplanet_hunter.data.stellar",
    "exoplanet_hunter.preprocess",
    "exoplanet_hunter.search",
    "exoplanet_hunter.features",
    "exoplanet_hunter.features.centroid",
    "exoplanet_hunter.models",
    "exoplanet_hunter.training",
    "exoplanet_hunter.training.tune",
    "exoplanet_hunter.eval",
    "exoplanet_hunter.datasets",
    "exoplanet_hunter.validation",
    "exoplanet_hunter.utils",
]


#: Deleted in stage 0 because each could silently produce wrong data of record.
#: `preprocess_only.py` wrote a 9-dim aux vector where `build_dataset.py` writes
#: 13 and bypassed `build_labels_from_cfg`, dropping every K2 row; nothing
#: downstream errored. `score_target.py` was the last hand-rolled aux layout,
#: capped at 9 dims. Both were *documented* recovery paths, which is what made
#: them dangerous.
DELETED_SCRIPTS = ["preprocess_only.py", "score_target.py"]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("script", DELETED_SCRIPTS)
def test_scripts_that_could_write_wrong_data_stay_deleted(script: str) -> None:
    assert not (Path(__file__).resolve().parents[1] / "scripts" / script).exists()
