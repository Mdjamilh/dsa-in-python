import bisect

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from dsa.structures import BinarySearchTree
from dsa.structures.bst import _Node


def assert_bst_ordered[V](node: _Node[int, V] | None) -> None:
    """Check the BST invariant with explicit (low, high) bounds, iteratively."""
    stack: list[tuple[_Node[int, V] | None, float, float]] = [(node, float("-inf"), float("inf"))]
    while stack:
        n, lo, hi = stack.pop()
        if n is None:
            continue
        assert lo < n.key < hi
        stack.append((n.left, lo, n.key))
        stack.append((n.right, n.key, hi))


def test_mapping_basics_and_order() -> None:
    t = BinarySearchTree({5: "e", 2: "b", 8: "h", 1: "a"})
    t[9] = "i"
    t[5] = "E"
    assert list(t) == [1, 2, 5, 8, 9]
    assert list(reversed(t)) == [9, 8, 5, 2, 1]
    assert t[5] == "E"
    assert len(t) == 5
    assert repr(BinarySearchTree({1: "a"})) == "BinarySearchTree({1: 'a'})"
    with pytest.raises(KeyError):
        t[3]


def test_floor_ceiling_min_max() -> None:
    t = BinarySearchTree((k, None) for k in [20, 10, 30, 5, 15, 25, 35])
    assert (t.min(), t.max()) == (5, 35)
    assert t.floor(17) == 15
    assert t.floor(15) == 15
    assert t.floor(4) is None
    assert t.ceiling(17) == 20
    assert t.ceiling(30) == 30
    assert t.ceiling(36) is None


def test_empty_tree() -> None:
    t: BinarySearchTree[int, int] = BinarySearchTree()
    with pytest.raises(KeyError):
        t.min()
    with pytest.raises(KeyError):
        t.max()
    with pytest.raises(KeyError):
        del t[1]
    assert t.height() == 0
    assert t.floor(1) is None


@pytest.mark.parametrize("key", [20, 10, 30, 25, 5])
def test_delete_each_shape(key: int) -> None:
    """Root with two children, inner node, leaf, and node with one child."""
    keys = [20, 10, 30, 25, 5, 27]
    t = BinarySearchTree((k, k) for k in keys)
    del t[key]
    assert list(t) == sorted(k for k in keys if k != key)
    assert_bst_ordered(t._root)


def test_contains_with_incomparable_key_is_false() -> None:
    t = BinarySearchTree({1: "a"})
    assert 1 in t
    assert "1" not in t  # type: ignore[comparison-overlap]


def test_sorted_input_degenerates_but_still_works() -> None:
    n = 3000  # far deeper than the default recursion limit of 1000
    t = BinarySearchTree((i, i) for i in range(n))
    assert t.height() == n
    assert t[n - 1] == n - 1
    assert list(t)[-3:] == [n - 3, n - 2, n - 1]
    for i in range(0, n, 2):
        del t[i]
    assert len(t) == n // 2
    t.clear()
    assert len(t) == 0


@given(st.lists(st.integers(-1000, 1000)), st.integers(-1100, 1100))
def test_floor_and_ceiling_match_bisect(keys: list[int], probe: int) -> None:
    t = BinarySearchTree((k, None) for k in keys)
    ordered = sorted(set(keys))
    i = bisect.bisect_right(ordered, probe)
    assert t.floor(probe) == (ordered[i - 1] if i else None)
    j = bisect.bisect_left(ordered, probe)
    assert t.ceiling(probe) == (ordered[j] if j < len(ordered) else None)


class BSTVsDict(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.t: BinarySearchTree[int, int] = BinarySearchTree()
        self.model: dict[int, int] = {}

    @rule(k=st.integers(0, 40), v=st.integers())
    def set(self, k: int, v: int) -> None:
        self.t[k] = v
        self.model[k] = v

    @rule(k=st.integers(0, 40))
    def delete(self, k: int) -> None:
        if k in self.model:
            del self.t[k]
            del self.model[k]
        else:
            with pytest.raises(KeyError):
                del self.t[k]

    @invariant()
    def sorted_like_dict(self) -> None:
        assert list(self.t.items()) == sorted(self.model.items())
        assert len(self.t) == len(self.model)
        assert_bst_ordered(self.t._root)


TestBSTVsDict = BSTVsDict.TestCase
