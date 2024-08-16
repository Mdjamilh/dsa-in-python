import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.algorithms import dijkstra
from dsa.structures import Graph
from tests.strategies import graphs


def floyd_warshall(graph: Graph[int]) -> dict[tuple[int, int], float]:
    """All-pairs shortest distances in O(V^3). Simple enough to trust as the oracle."""
    vs = list(graph.vertices())
    dist = {(a, b): (0.0 if a == b else math.inf) for a in vs for b in vs}
    for u, v, w in graph.edges():
        dist[u, v] = min(dist[u, v], w)
        if not graph.directed:
            dist[v, u] = min(dist[v, u], w)
    for k in vs:
        for i in vs:
            for j in vs:
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist


def test_classic_example() -> None:
    g = Graph.from_edges(
        [("s", "a", 7), ("s", "b", 2), ("b", "a", 3), ("a", "t", 1), ("b", "t", 8)],
        directed=True,
    )
    paths = dijkstra(g, "s")
    assert paths.distance == {"s": 0, "a": 5, "b": 2, "t": 6}
    assert paths.path_to("t") == ["s", "b", "a", "t"]
    assert paths.path_to("s") == ["s"]


def test_unreachable_and_errors() -> None:
    g = Graph.from_edges([(1, 2, 1.0)], directed=True, vertices=[3])
    paths = dijkstra(g, 1)
    assert paths.distance_to(3) == math.inf
    assert paths.path_to(3) is None
    with pytest.raises(KeyError):
        dijkstra(g, 9)
    with pytest.raises(ValueError, match="negative"):
        dijkstra(Graph.from_edges([(1, 2, -1.0)]), 1)


def test_zero_weight_edges_and_cycles() -> None:
    g = Graph.from_edges([(1, 2, 0), (2, 3, 0), (3, 1, 0), (3, 4, 5)], directed=True)
    assert dijkstra(g, 1).distance == {1: 0, 2: 0, 3: 0, 4: 5}


@given(graphs(), st.data())
def test_matches_floyd_warshall(g: Graph[int], data: st.DataObject) -> None:
    source = data.draw(st.sampled_from(list(g.vertices())))
    expected = floyd_warshall(g)
    paths = dijkstra(g, source)
    for v in g.vertices():
        assert paths.distance_to(v) == expected[source, v]
        path = paths.path_to(v)
        if path is None:
            assert expected[source, v] == math.inf
            continue
        # The reconstructed path must use real edges and add up to the distance.
        assert path[0] == source
        assert path[-1] == v
        assert sum(g.weight(a, b) for a, b in pairwise(path)) == (paths.distance_to(v))


@given(graphs(integer_weights=False), st.data())
def test_float_weights_close_to_floyd_warshall(g: Graph[int], data: st.DataObject) -> None:
    source = data.draw(st.sampled_from(list(g.vertices())))
    expected = floyd_warshall(g)
    paths = dijkstra(g, source)
    for v in g.vertices():
        assert math.isclose(paths.distance_to(v), expected[source, v], rel_tol=1e-9, abs_tol=1e-9)
