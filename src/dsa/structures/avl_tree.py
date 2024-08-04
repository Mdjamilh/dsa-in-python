"""A self-balancing AVL tree with order statistics.

An AVL tree is a binary search tree that keeps one invariant at every node:
the heights of its two subtrees differ by at most one. A tree like that has
height at most about ``1.44 log2 n``, so every operation is O(log n) in the
worst case, including for sorted input that turns a plain BST into a list.

After an insert or delete, heights are recomputed on the way back up the
search path. A node whose balance factor (left height minus right height)
reaches +2 or -2 is fixed with one or two rotations:

- left-left (heavy on the left, and the left child leans left or is even):
  rotate right once.
- left-right (heavy on the left, but the left child leans right): rotate the
  child left, then the node right.
- right-right and right-left mirror those.

A rotation changes a constant number of links and keeps the in-order sequence
of keys, so the tree stays a valid BST.

Each node also stores the size of its subtree. That gives ``rank(key)`` (how
many keys are smaller) and ``select(i)`` (the ``i``-th smallest key) by
walking one path, using the left subtree's size to decide which way to go.

The recursion depth is the tree height, O(log n), so recursion is safe here.

==============================================  ==========
Operation                                       Time
==============================================  ==========
get, set, delete, ``in``                        O(log n)
``floor``, ``ceiling``, ``min``, ``max``        O(log n)
``rank``, ``select``                            O(log n)
``range(lo, hi)``                               O(log n + k) for k results
iteration (sorted)                              O(n)
==============================================  ==========

Space: O(n).
"""

from collections.abc import Iterator

from dsa._typing import SupportsLessThan
from dsa.structures.bst import _Node, _SearchTree

__all__ = ["AVLTree"]


def _h[K, V](node: _Node[K, V] | None) -> int:
    return 0 if node is None else node.height


def _s[K, V](node: _Node[K, V] | None) -> int:
    return 0 if node is None else node.size


class AVLTree[K: SupportsLessThan, V](_SearchTree[K, V]):
    """An ordered ``MutableMapping`` that stays balanced, with ``rank`` and ``select``.

    >>> t = AVLTree((k, str(k)) for k in range(1, 8))
    >>> t.height(), t.rank(4), t.select(0), list(t.range(3, 6))
    (3, 3, 1, [3, 4, 5])
    """

    __slots__ = ()

    @staticmethod
    def _update(node: _Node[K, V]) -> None:
        node.height = 1 + max(_h(node.left), _h(node.right))
        node.size = 1 + _s(node.left) + _s(node.right)

    def _rotate_right(self, node: _Node[K, V]) -> _Node[K, V]:
        pivot = node.left
        assert pivot is not None
        node.left = pivot.right
        pivot.right = node
        self._update(node)
        self._update(pivot)
        return pivot

    def _rotate_left(self, node: _Node[K, V]) -> _Node[K, V]:
        pivot = node.right
        assert pivot is not None
        node.right = pivot.left
        pivot.left = node
        self._update(node)
        self._update(pivot)
        return pivot

    def _rebalance(self, node: _Node[K, V]) -> _Node[K, V]:
        self._update(node)
        balance = _h(node.left) - _h(node.right)
        if balance > 1:
            assert node.left is not None
            if _h(node.left.left) < _h(node.left.right):
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1:
            assert node.right is not None
            if _h(node.right.right) < _h(node.right.left):
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def __setitem__(self, key: K, value: V) -> None:
        self._root = self._insert(self._root, key, value)

    def _insert(self, node: _Node[K, V] | None, key: K, value: V) -> _Node[K, V]:
        if node is None:
            self._size += 1
            return _Node(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif node.key < key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value
            return node
        return self._rebalance(node)

    def __delitem__(self, key: K) -> None:
        self._root = self._delete(self._root, key)
        self._size -= 1

    def _delete(self, node: _Node[K, V] | None, key: K) -> _Node[K, V] | None:
        if node is None:
            raise KeyError(key)
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif node.key < key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.value = successor.key, successor.value
            node.right = self._delete_min(node.right)
        return self._rebalance(node)

    def _delete_min(self, node: _Node[K, V]) -> _Node[K, V] | None:
        if node.left is None:
            return node.right
        node.left = self._delete_min(node.left)
        return self._rebalance(node)

    def rank(self, key: K) -> int:
        """Return how many keys are strictly less than ``key``. O(log n).

        ``key`` does not have to be in the tree.
        """
        rank = 0
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif node.key < key:
                rank += _s(node.left) + 1
                node = node.right
            else:
                return rank + _s(node.left)
        return rank

    def select(self, index: int) -> K:
        """Return the key at position ``index`` in sorted order. Negative indices count back.

        Raises ``IndexError`` when out of range. O(log n).
        """
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("select index out of range")
        node = self._root
        while node is not None:
            left = _s(node.left)
            if index < left:
                node = node.left
            elif index > left:
                index -= left + 1
                node = node.right
            else:
                return node.key
        raise AssertionError("subtree sizes are inconsistent")  # pragma: no cover

    def range(self, lo: K, hi: K) -> Iterator[K]:
        """Yield keys ``k`` with ``lo <= k < hi`` in order. O(log n + k).

        Subtrees that lie entirely outside the range are never visited.
        """
        stack: list[_Node[K, V]] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                if node.key < lo:
                    node = node.right  # everything to the left is smaller still
                else:
                    stack.append(node)
                    node = node.left
            if not stack:
                return
            node = stack.pop()
            if not node.key < hi:
                return
            yield node.key
            node = node.right
