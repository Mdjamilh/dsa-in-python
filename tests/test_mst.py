import itertools

import pytest
from hypothesis import given

from dsa.algorithms import connected_components, kruskal, prim, total_weight
from dsa.structures import Edge, Graph, UnionFind
from tests.strategies import graphs


def brute_force_forest_weight(g: Graph[int]) -> float:
    """Try every subset of the right size and keep the lightest acyclic one."""
    edges = list(g.edges())
    size = g.vertex_count - len(connected_components(g))
    best = float("inf")
    for subset in itertools.combinations(edges, size):
        uf = UnionFind(g.vertices())
        if all(uf.union(u, v) for u, v, _ in subset):
            best = min(best, total_weight(list(subset)))
    return best


def assert_spanning_forest(g: Graph[int], forest: list[Edge[int]]) -> None:
    uf = UnionFind(g.vertices())
    for u, v, w in forest:
        assert g.weight(u, v) == w
        assert uf.union(u, v), "forest contains a cycle"
    assert uf.count == len(connected_components(g))


def test_known_example() -> None:
    g = Graph.from_edges(
        [("a", "b", 4), ("a", "h", 8), ("b", "h", 11), ("b", "c", 8), ("h", "i", 7),
         ("h", "g", 1), ("i", "g", 6), ("c", "i", 2), ("c", "f", 4), ("g", "f", 2),
         ("c", "d", 7), ("d", "f", 14), ("d", "e", 9), ("e", "f", 10)]
    )  # fmt: skip
    assert total_weight(kruskal(g)) == 37
    assert total_weight(prim(g)) == 37
    assert len(kruskal(g)) == len(prim(g)) == 8


def test_disconnected_graph_gives_a_forest() -> None:
    g = Graph.from_edges([(1, 2, 3), (3, 4, 1)], vertices=[5])
    assert sorted(kruskal(g)) == [(1, 2, 3), (3, 4, 1)]
    assert len(prim(g)) == 2


def test_rejects_directed() -> None:
    g = Graph.from_edges([(1, 2)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        kruskal(g)
    with pytest.raises(ValueError, match="undirected"):
        prim(g)


@given(graphs(directed=False, max_vertices=6, max_edges=9))
def test_kruskal_and_prim_match_brute_force(g: Graph[int]) -> None:
    expected = brute_force_forest_weight(g)
    for algorithm in (kruskal, prim):
        forest = algorithm(g)
        assert_spanning_forest(g, forest)
        assert total_weight(forest) == expected
