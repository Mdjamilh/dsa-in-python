"""An unbalanced binary search tree, and the read-only machinery shared with the AVL tree.

A binary search tree keeps one key per node, with every key in the left
subtree smaller and every key in the right subtree larger. Searching compares
against the root and goes left or right, so each step discards a subtree.

The cost of every operation is the height ``h`` of the tree. With random
insertion order the expected height is about ``2.99 log2 n``, but inserting
keys in sorted order builds a linked list with ``h = n``. ``AVLTree`` fixes
that; this class is here to show the problem and the plain algorithms.

All loops here are iterative, so a degenerate tree of any height works without
hitting Python's recursion limit.

Deletion follows Hibbard: a node with two children is replaced by its
successor (the smallest key in its right subtree), which has at most one child
and is easy to unlink.

==========================================  =========================
Operation                                   Time
==========================================  =========================
get, set, delete, ``in``                    O(h): O(log n) average, O(n) worst
``floor``, ``ceiling``, ``min``, ``max``    O(h)
iteration (sorted), ``height``              O(n)
==========================================  =========================

Space: O(n), plus O(h) for an iterator's stack.
"""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import overload

from dsa._typing import SupportsLessThan

__all__ = ["BinarySearchTree"]


class _Node[K, V]:
    __slots__ = ("height", "key", "left", "right", "size", "value")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.left: _Node[K, V] | None = None
        self.right: _Node[K, V] | None = None
        # Used by AVLTree only; kept here so both trees share one node type.
        self.height = 1
        self.size = 1


class _SearchTree[K: SupportsLessThan, V](MutableMapping[K, V]):
    """Lookups, ordered queries and iteration for any binary search tree.

    Subclasses provide ``__setitem__`` and ``__delitem__`` and keep ``_root``
    and ``_size`` up to date.
    """

    __slots__ = ("_root", "_size")

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, items: Mapping[K, V]) -> None: ...
    @overload
    def __init__(self, items: Iterable[tuple[K, V]]) -> None: ...
    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        self._root: _Node[K, V] | None = None
        self._size = 0
        self.update(items)

    def __len__(self) -> int:
        return self._size

    def _find(self, key: K) -> _Node[K, V] | None:
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif node.key < key:
                node = node.right
            else:
                return node
        return None

    def __getitem__(self, key: K) -> V:
        node = self._find(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def __contains__(self, key: object) -> bool:
        try:
            return self._find(key) is not None  # type: ignore[arg-type]
        except TypeError:  # a key that cannot be compared with ours is not here
            return False

    def __iter__(self) -> Iterator[K]:
        for node in self._in_order():
            yield node.key

    def __reversed__(self) -> Iterator[K]:
        stack: list[_Node[K, V]] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.right
            node = stack.pop()
            yield node.key
            node = node.left

    def _in_order(self) -> Iterator[_Node[K, V]]:
        stack: list[_Node[K, V]] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node
            node = node.right

    def min(self) -> K:
        """Return the smallest key. Raises ``KeyError`` when empty. O(h)."""
        node = self._root
        if node is None:
            raise KeyError("min of an empty tree")
        while node.left is not None:
            node = node.left
        return node.key

    def max(self) -> K:
        """Return the largest key. Raises ``KeyError`` when empty. O(h)."""
        node = self._root
        if node is None:
            raise KeyError("max of an empty tree")
        while node.right is not None:
            node = node.right
        return node.key

    def floor(self, key: K) -> K | None:
        """Return the largest key ``<= key``, or ``None`` if there is none. O(h)."""
        best: K | None = None
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif node.key < key:
                best = node.key
                node = node.right
            else:
                return node.key
        return best

    def ceiling(self, key: K) -> K | None:
        """Return the smallest key ``>= key``, or ``None`` if there is none. O(h)."""
        best: K | None = None
        node = self._root
        while node is not None:
            if node.key < key:
                node = node.right
            elif key < node.key:
                best = node.key
                node = node.left
            else:
                return node.key
        return best

    def height(self) -> int:
        """Return the number of nodes on the longest root-to-leaf path. O(n), iterative."""
        best = 0
        stack: list[tuple[_Node[K, V] | None, int]] = [(self._root, 1)]
        while stack:
            node, depth = stack.pop()
            if node is None:
                continue
            best = max(best, depth)
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
        return best

    def clear(self) -> None:
        """Remove every entry. O(1)."""
        self._root = None
        self._size = 0

    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


class BinarySearchTree[K: SupportsLessThan, V](_SearchTree[K, V]):
    """An ordered ``MutableMapping`` on a plain, unbalanced binary search tree.

    >>> t = BinarySearchTree({5: "e", 2: "b", 8: "h"})
    >>> list(t), t.floor(6), t.ceiling(6)
    ([2, 5, 8], 5, 8)
    """

    __slots__ = ()

    def __setitem__(self, key: K, value: V) -> None:
        if self._root is None:
            self._root = _Node(key, value)
            self._size = 1
            return
        node = self._root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = _Node(key, value)
                    break
                node = node.left
            elif node.key < key:
                if node.right is None:
                    node.right = _Node(key, value)
                    break
                node = node.right
            else:
                node.value = value
                return
        self._size += 1

    def __delitem__(self, key: K) -> None:
        parent: _Node[K, V] | None = None
        node = self._root
        while node is not None and (key < node.key or node.key < key):
            parent = node
            node = node.left if key < node.key else node.right
        if node is None:
            raise KeyError(key)

        if node.left is not None and node.right is not None:
            # Two children: copy the successor up, then unlink the successor,
            # which has no left child.
            succ_parent, succ = node, node.right
            while succ.left is not None:
                succ_parent, succ = succ, succ.left
            node.key, node.value = succ.key, succ.value
            parent, node = succ_parent, succ

        child = node.left if node.left is not None else node.right
        if parent is None:
            self._root = child
        elif parent.left is node:
            parent.left = child
        else:
            parent.right = child
        self._size -= 1
