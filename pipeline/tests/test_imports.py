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


#: Deleted in stage 1 because each could silently produce wrong data of record.
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


#: Resolved by NAME out of persisted sklearn pipelines, never called by our own
#: code. `e5388ed9`, `cebb0fe6` and the live `ca906040` pickle these by
#: reference, so unpickling a served bundle looks them up in the module at
#: serve time. That makes them unreferenced by design and indistinguishable from
#: dead code to any static sweep — deleting or renaming one breaks serving, and
#: breaks it at model-load rather than at import, which is the worst place.
PICKLE_RESOLVED_BY_NAME = [
    ("exoplanet_hunter.datasets.aux_transform", "_log1p_centroid"),
]


@pytest.mark.parametrize(("module", "attribute"), PICKLE_RESOLVED_BY_NAME)
def test_functions_persisted_pickles_resolve_by_name_stay_put(module: str, attribute: str) -> None:
    """Not a behaviour test — a name test, which is the whole point.

    The signature and the module path are the contract, because that is what
    pickle stored. `getattr` here is exactly what `joblib.load` does.
    """
    import inspect

    resolved = getattr(importlib.import_module(module), attribute)
    assert callable(resolved)
    assert list(inspect.signature(resolved).parameters) == ["X"], (
        f"{module}.{attribute} is resolved by name out of persisted pipelines; "
        "its signature is a serving contract, not an implementation detail"
    )
