"""Union-find (disjoint set union) with path halving and union by size.

Each set is a tree of elements, and the root is the set's representative.
``find`` walks up to the root; ``union`` hangs one root under the other.

Two tricks keep the trees nearly flat:

- **Union by size.** The smaller tree goes under the larger, so an element's
  depth grows only when its tree at least doubles in size. Depth stays
  O(log n) even without the next trick.
- **Path halving.** During ``find``, every other node on the path is pointed
  at its grandparent. Later finds on that path take about half the steps.
  It is a one-pass version of full path compression and is just as good.

Together they make any sequence of ``m`` operations cost
O(m alpha(n)), where alpha is the inverse Ackermann function. Alpha is at most
4 for any n that fits in this universe, so each operation is effectively O(1).

Elements are added on first sight, so any hashable value can be used.

==================================  ======================
Operation                           Time
==================================  ======================
``find``, ``union``, ``connected``  O(alpha(n)) amortized
``size_of``                         O(alpha(n)) amortized
``groups``                          O(n alpha(n))
==================================  ======================

Space: O(n).
"""

from collections.abc import Hashable, Iterable, Iterator

__all__ = ["UnionFind"]


class UnionFind[T: Hashable]:
    """Disjoint sets over hashable elements.

    >>> uf = UnionFind(range(5))
    >>> uf.union(0, 1), uf.union(3, 4), uf.union(1, 0)
    (True, True, False)
    >>> uf.connected(0, 1), uf.connected(1, 3), uf.count
    (True, False, 3)
    """

    __slots__ = ("_count", "_parent", "_size")

    def __init__(self, elements: Iterable[T] = ()) -> None:
        self._parent: dict[T, T] = {}
        self._size: dict[T, int] = {}  # only meaningful for roots
        self._count = 0
        for x in elements:
            self.add(x)

    def add(self, x: T) -> None:
        """Add ``x`` as a singleton set if it is new. O(1)."""
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1
            self._count += 1

    def find(self, x: T) -> T:
        """Return the representative of the set containing ``x`` (adding ``x`` if new)."""
        parent = self._parent
        if x not in parent:
            self.add(x)
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: T, b: T) -> bool:
        """Merge the sets containing ``a`` and ``b``. Return ``False`` if already together."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size.pop(rb)
        self._count -= 1
        return True

    def connected(self, a: T, b: T) -> bool:
        """Return whether ``a`` and ``b`` are in the same set."""
        return self.find(a) == self.find(b)

    def size_of(self, x: T) -> int:
        """Return the number of elements in the set containing ``x``."""
        return self._size[self.find(x)]

    @property
    def count(self) -> int:
        """Number of disjoint sets."""
        return self._count

    def groups(self) -> list[set[T]]:
        """Return every set, as a list of Python sets."""
        by_root: dict[T, set[T]] = {}
        for x in self._parent:
            by_root.setdefault(self.find(x), set()).add(x)
        return list(by_root.values())

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, x: object) -> bool:
        return x in self._parent

    def __iter__(self) -> Iterator[T]:
        return iter(self._parent)

    def __repr__(self) -> str:
        return f"UnionFind(elements={len(self)}, sets={self._count})"
