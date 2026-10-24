"""Smoke test: the package skeleton imports (Step 1 only)."""

import importlib

import pytest

MODULES = [
    "config", "grid", "chromosome", "fitness", "operators",
    "genetic_algorithm", "random_search", "visualization", "experiments",
]


def test_package_has_version():
    import uav_ga

    assert uav_ga.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_submodule_imports(name):
    importlib.import_module(f"uav_ga.{name}")
