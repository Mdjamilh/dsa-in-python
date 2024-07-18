from importlib.metadata import version

import dsa


def test_version_matches_metadata() -> None:
    assert dsa.__version__ == version("dsa-in-python")
