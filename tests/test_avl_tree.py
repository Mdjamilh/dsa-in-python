import bisect

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from dsa.structures import AVLTree
from dsa.structures.bst import _Node


def check_avl(node: _Node[int, int] | None) -> tuple[int, int]:
    """Recompute height and size bottom-up and check every stored value and balance."""
    if node is None:
        return 0, 0
    lh, ls = check_avl(node.left)
    rh, rs = check_avl(node.right)
    if node.left is not None:
        assert node.left.key < node.key
    if node.right is not None:
        assert node.key < node.right.key
    assert abs(lh - rh) <= 1, f"unbalanced at {node.key}"
    assert node.height == 1 + max(lh, rh)
    assert node.size == 1 + ls + rs
    return node.height, node.size


@pytest.mark.parametrize(
    ("keys", "root"),
    [
        ([3, 2, 1], 2),  # left-left: one right rotation
        ([1, 2, 3], 2),  # right-right: one left rotation
        ([3, 1, 2], 2),  # left-right: double rotation
        ([1, 3, 2], 2),  # right-left: double rotation
    ],
)
def test_four_rotation_cases(keys: list[int], root: int) -> None:
    t = AVLTree((k, k) for k in keys)
    assert t._root is not None
    assert t._root.key == root
    assert t.height() == 2
    check_avl(t._root)


def test_sorted_input_stays_logarithmic() -> None:
    n = 10_000
    t = AVLTree((i, i) for i in range(n))
    assert t.height() <= 1.45 * n.bit_length()
    check_avl(t._root)
    for i in range(0, n, 3):
        del t[i]
    check_avl(t._root)
    assert list(t) == [i for i in range(n) if i % 3]


def test_rank_select_and_range() -> None:
    t = AVLTree((k, k) for k in [10, 20, 30, 40, 50])
    assert [t.rank(k) for k in [5, 10, 25, 50, 99]] == [0, 0, 2, 4, 5]
    assert [t.select(i) for i in range(5)] == [10, 20, 30, 40, 50]
    assert t.select(-1) == 50
    with pytest.raises(IndexError):
        t.select(5)
    with pytest.raises(IndexError):
        t.select(-6)
    assert list(t.range(15, 45)) == [20, 30, 40]
    assert list(t.range(20, 20)) == []
    assert list(t.range(60, 70)) == []
    assert list(t.range(0, 100)) == [10, 20, 30, 40, 50]


def test_delete_missing_key_leaves_tree_intact() -> None:
    t = AVLTree((k, k) for k in range(10))
    with pytest.raises(KeyError):
        del t[42]
    assert len(t) == 10
    check_avl(t._root)


def test_shared_queries() -> None:
    t = AVLTree({"b": 2, "a": 1, "c": 3})
    assert (t.min(), t.max(), t.floor("bb"), t.ceiling("bb")) == ("a", "c", "b", "c")
    assert repr(t) == "AVLTree({'a': 1, 'b': 2, 'c': 3})"
    t["b"] = 20
    assert t["b"] == 20
    assert len(t) == 3


@given(st.lists(st.integers(-50, 50)), st.integers(-60, 60), st.integers(-60, 60))
def test_rank_and_range_match_bisect(keys: list[int], lo: int, hi: int) -> None:
    t = AVLTree((k, k) for k in keys)
    ordered = sorted(set(keys))
    assert t.rank(lo) == bisect.bisect_left(ordered, lo)
    assert list(t.range(lo, hi)) == [k for k in ordered if lo <= k < hi]
    assert [t.select(i) for i in range(len(ordered))] == ordered


class AVLVsDict(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.t: AVLTree[int, int] = AVLTree()
        self.model: dict[int, int] = {}

    @rule(k=st.integers(0, 60), v=st.integers())
    def set(self, k: int, v: int) -> None:
        self.t[k] = v
        self.model[k] = v

    @rule(k=st.integers(0, 60))
    def delete(self, k: int) -> None:
        if k in self.model:
            del self.t[k]
            del self.model[k]
        else:
            with pytest.raises(KeyError):
                del self.t[k]

    @invariant()
    def balanced_and_sorted(self) -> None:
        _, size = check_avl(self.t._root)
        assert size == len(self.t) == len(self.model)
        assert list(self.t.items()) == sorted(self.model.items())


TestAVLVsDict = AVLVsDict.TestCase
