"""Shared typing helpers.

Ordered structures (heaps, search trees, sorts) only ever compare values with
``<``. ``SupportsLessThan`` captures exactly that, so ``int``, ``str``,
``float``, tuples of those, and any class with ``__lt__`` all qualify.
"""

from typing import Any, Protocol


class SupportsLessThan(Protocol):
    """Anything that can be ordered with ``<``."""

    def __lt__(self, other: Any, /) -> bool: ...
