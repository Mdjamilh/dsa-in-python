from hypothesis import given
from hypothesis import strategies as st

from dsa.structures import UnionFind


def test_basic_union_and_find() -> None:
    uf = UnionFind("abcde")
    assert uf.count == 5
    assert uf.union("a", "b")
    assert uf.union("c", "d")
    assert not uf.union("b", "a")
    assert uf.connected("a", "b")
    assert not uf.connected("a", "c")
    assert uf.union("b", "d")
    assert uf.connected("a", "c")
    assert uf.count == 2
    assert uf.size_of("d") == 4
    assert sorted(sorted(g) for g in uf.groups()) == [["a", "b", "c", "d"], ["e"]]
    assert repr(uf) == "UnionFind(elements=5, sets=2)"


def test_elements_are_added_on_first_sight() -> None:
    uf: UnionFind[int] = UnionFind()
    assert uf.find(7) == 7
    assert 7 in uf
    uf.union(8, 9)
    assert len(uf) == 3
    assert set(uf) == {7, 8, 9}
    uf.add(7)  # already there: no change
    assert uf.count == 2


def test_union_by_size_keeps_trees_shallow() -> None:
    n = 1 << 12
    uf = UnionFind(range(n))
    step = 1
    while step < n:  # merge pairs, then pairs of pairs: the worst case for naive linking
        for i in range(0, n, 2 * step):
            uf.union(i, i + step)
        step *= 2
    # With union by size, no element is deeper than log2(n).
    depth = 0
    for start in range(n):
        x, d = start, 0
        while uf._parent[x] != x:
            x = uf._parent[x]
            d += 1
        depth = max(depth, d)
    assert depth <= 12
    assert uf.count == 1


def test_long_chain_is_flattened_by_find() -> None:
    uf = UnionFind(range(1000))
    for i in range(999):
        uf._parent[i] = i + 1  # build a worst-case chain by hand
    uf._size = {999: 1000}
    uf._count = 1
    root = uf.find(0)
    assert root == 999
    # Path halving pointed every other node at its grandparent.
    assert uf._parent[0] == 2


@given(st.integers(1, 30), st.lists(st.tuples(st.integers(0, 29), st.integers(0, 29))))
def test_matches_naive_relabelling(n: int, pairs: list[tuple[int, int]]) -> None:
    uf = UnionFind(range(n))
    label = list(range(n))  # the model: each element stores its set label
    for raw_a, raw_b in pairs:
        a, b = raw_a % n, raw_b % n
        merged = uf.union(a, b)
        assert merged == (label[a] != label[b])
        if merged:
            old, new = label[b], label[a]
            label = [new if x == old else x for x in label]
        for x in range(n):
            assert uf.connected(a, x) == (label[a] == label[x])
    assert uf.count == len(set(label))
    for x in range(n):
        assert uf.size_of(x) == label.count(label[x])
