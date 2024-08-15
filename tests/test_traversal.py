from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.algorithms import bfs, bfs_distances, connected_components, dfs, shortest_path
from dsa.structures import Graph
from tests.strategies import graphs


def recursive_dfs(graph: Graph[int], start: int) -> list[int]:
    """The textbook recursive DFS, as the reference for visiting order."""
    order: list[int] = []
    seen: set[int] = set()

    def visit(v: int) -> None:
        seen.add(v)
        order.append(v)
        for w in graph.neighbors(v):
            if w not in seen:
                visit(w)

    visit(start)
    return order


def relaxation_distances(graph: Graph[int], start: int) -> dict[int, int]:
    """Bellman-Ford style: relax every edge V times with weight 1. Slow, obviously right."""
    dist = {start: 0}
    for _ in range(graph.vertex_count):
        for u, v, _w in graph.edges():
            for a, b in [(u, v)] if graph.directed else [(u, v), (v, u)]:
                if a in dist and dist[a] + 1 < dist.get(b, 10**9):
                    dist[b] = dist[a] + 1
    return dist


def test_bfs_and_dfs_orders() -> None:
    g = Graph.from_edges([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    assert list(bfs(g, 1)) == [1, 2, 3, 4, 5]
    assert list(dfs(g, 1)) == [1, 2, 4, 3, 5]
    assert bfs_distances(g, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}


def test_shortest_path() -> None:
    g = Graph.from_edges([("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")], directed=True)
    assert shortest_path(g, "a", "d") == ["a", "c", "d"]
    assert shortest_path(g, "a", "a") == ["a"]
    assert shortest_path(g, "d", "a") is None


def test_unknown_start_raises() -> None:
    g = Graph.from_edges([(1, 2)])
    with pytest.raises(KeyError):
        next(bfs(g, 9))
    with pytest.raises(KeyError):
        next(dfs(g, 9))
    with pytest.raises(KeyError):
        bfs_distances(g, 9)
    with pytest.raises(KeyError):
        shortest_path(g, 9, 1)


def test_dfs_handles_a_long_path_without_recursion() -> None:
    n = 20_000
    g = Graph.from_edges([(i, i + 1) for i in range(n)], directed=True)
    assert sum(1 for _ in dfs(g, 0)) == n + 1


def test_generators_are_lazy() -> None:
    g = Graph.from_edges([(i, i + 1) for i in range(1000)])
    it = bfs(g, 0)
    assert [next(it) for _ in range(3)] == [0, 1, 2]


def test_connected_components() -> None:
    g = Graph.from_edges([(1, 2), (3, 4), (4, 5)], vertices=[6])
    assert connected_components(g) == [[6], [1, 2], [3, 4, 5]]  # vertices= come first
    with pytest.raises(ValueError, match="undirected"):
        connected_components(Graph(directed=True))


@given(graphs(), st.data())
def test_dfs_matches_recursive_order(g: Graph[int], data: st.DataObject) -> None:
    start = data.draw(st.sampled_from(list(g.vertices())))
    assert list(dfs(g, start)) == recursive_dfs(g, start)


@given(graphs(), st.data())
def test_bfs_distances_match_relaxation(g: Graph[int], data: st.DataObject) -> None:
    start = data.draw(st.sampled_from(list(g.vertices())))
    expected = relaxation_distances(g, start)
    assert bfs_distances(g, start) == expected
    order = list(bfs(g, start))
    assert set(order) == set(expected)
    assert [expected[v] for v in order] == sorted(expected[v] for v in order)  # by layer


@given(graphs(), st.data())
def test_shortest_path_is_valid_and_shortest(g: Graph[int], data: st.DataObject) -> None:
    vs = list(g.vertices())
    s, t = data.draw(st.sampled_from(vs)), data.draw(st.sampled_from(vs))
    dist = relaxation_distances(g, s)
    path = shortest_path(g, s, t)
    if t not in dist:
        assert path is None
        return
    assert path is not None
    assert path[0] == s
    assert path[-1] == t
    assert len(path) - 1 == dist[t]
    assert all(g.has_edge(a, b) for a, b in pairwise(path))


@given(graphs(directed=False))
def test_components_partition_the_vertices(g: Graph[int]) -> None:
    comps = connected_components(g)
    flat = [v for c in comps for v in c]
    assert sorted(flat) == sorted(g.vertices())
    label = {v: i for i, c in enumerate(comps) for v in c}
    assert all(label[u] == label[v] for u, v, _ in g.edges())
