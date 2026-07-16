"""Smoke test: every salvaged module imports cleanly in the V2 layout.

This is the seed of the V2 test suite — it exists so CI fails loudly if the
extraction left a dangling import. Behavioural tests for preprocessing and
the model arrive with `feat/tfdata-pipeline`.
"""

import importlib

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


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
