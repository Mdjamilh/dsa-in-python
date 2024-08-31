import importlib
import importlib.resources
import pkgutil
from importlib.metadata import version

import pytest

import dsa
import dsa.algorithms
import dsa.structures

MODULES = sorted(
    name
    for pkg in (dsa.structures, dsa.algorithms)
    for _, name, _ in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")
)


def test_version_matches_metadata() -> None:
    assert dsa.__version__ == version("dsa-in-python")


def test_top_level_exports_everything_public() -> None:
    sub = set(dsa.structures.__all__) | set(dsa.algorithms.__all__)
    assert sub | {"__version__"} == set(dsa.__all__)
    for name in dsa.__all__:
        assert hasattr(dsa, name), name


@pytest.mark.parametrize("name", MODULES)
def test_every_module_documents_its_complexity(name: str) -> None:
    module = importlib.import_module(name)
    doc = module.__doc__ or ""
    assert "O(" in doc, f"{name} should state its complexity in the module docstring"
    for symbol in getattr(module, "__all__", []):
        obj = getattr(module, symbol)
        if callable(obj):
            assert obj.__doc__, f"{name}.{symbol} needs a docstring"


def test_package_is_typed() -> None:
    files = importlib.resources.files("dsa")
    assert files.joinpath("py.typed").is_file()
