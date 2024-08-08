import math

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from dsa.structures import Graph


def test_undirected_basics() -> None:
    g = Graph.from_edges([("a", "b", 2.0), ("b", "c", 1.0)])
    assert g.vertex_count == 3
    assert g.edge_count == 2
    assert g.has_edge("b", "a")
    assert g.weight("c", "b") == 1.0
    assert sorted(g.neighbors("b")) == ["a", "c"]
    assert g.degree("b") == 2
    assert list(g.edges()) == [("a", "b", 2.0), ("b", "c", 1.0)]
    assert repr(g) == "Graph(undirected, vertices=3, edges=2)"
    assert "a" in g
    assert "z" not in g
    assert len(g) == 3
    assert list(g) == ["a", "b", "c"]


def test_directed_basics_and_reverse() -> None:
    g = Graph.from_edges([(1, 2), (2, 3)], directed=True)
    assert g.has_edge(1, 2)
    assert not g.has_edge(2, 1)
    assert not g.has_edge(9, 1)
    r = g.reversed()
    assert list(r.edges()) == [(2, 1, 1.0), (3, 2, 1.0)]
    assert r.directed


def test_readding_an_edge_replaces_weight() -> None:
    g: Graph[str] = Graph()
    g.add_edge("a", "b", 1)
    g.add_edge("b", "a", 5)
    assert g.edge_count == 1
    assert g.weight("a", "b") == 5


def test_errors() -> None:
    g = Graph.from_edges([(1, 2)])
    with pytest.raises(ValueError, match="NaN"):
        g.add_edge(1, 3, math.nan)
    with pytest.raises(KeyError):
        g.weight(1, 3)
    with pytest.raises(KeyError):
        g.remove_edge(1, 3)
    with pytest.raises(KeyError):
        g.remove_vertex(9)


def test_adjacency_is_read_only_view() -> None:
    g = Graph.from_edges([(1, 2, 3.0)])
    view = g.adjacency(1)
    assert dict(view) == {2: 3.0}
    with pytest.raises(TypeError):
        view[5] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("directed", [False, True])
def test_self_loop_is_one_edge(directed: bool) -> None:
    g: Graph[int] = Graph(directed=directed)
    g.add_edge(1, 1)
    g.add_edge(1, 2)
    assert g.edge_count == 2
    assert g.degree(1) == 2
    g.remove_edge(1, 1)
    assert g.edge_count == 1
    g.add_edge(1, 1)
    g.remove_vertex(1)
    assert g.edge_count == 0
    assert list(g) == [2]


def test_remove_vertex_directed_drops_incoming_edges() -> None:
    g = Graph.from_edges([(1, 2), (3, 2), (2, 4), (2, 2)], directed=True)
    g.remove_vertex(2)
    assert g.edge_count == 0
    assert all(g.degree(v) == 0 for v in g)


class GraphVsEdgeSet(RuleBasedStateMachine):
    """The model is a dict of (u, v) -> weight, with undirected pairs stored sorted."""

    @initialize(directed=st.booleans())
    def setup(self, directed: bool) -> None:
        self.g: Graph[int] = Graph(directed=directed)
        self.vertices: set[int] = set()
        self.edges: dict[tuple[int, int], float] = {}

    def _key(self, u: int, v: int) -> tuple[int, int]:
        return (u, v) if self.g.directed else (min(u, v), max(u, v))

    @rule(u=st.integers(0, 5), v=st.integers(0, 5), w=st.integers(0, 9))
    def add_edge(self, u: int, v: int, w: int) -> None:
        self.g.add_edge(u, v, w)
        self.vertices |= {u, v}
        self.edges[self._key(u, v)] = w

    @rule(u=st.integers(0, 5), v=st.integers(0, 5))
    def remove_edge(self, u: int, v: int) -> None:
        if self._key(u, v) in self.edges:
            self.g.remove_edge(u, v)
            del self.edges[self._key(u, v)]

    @rule(v=st.integers(0, 5))
    def remove_vertex(self, v: int) -> None:
        if v in self.vertices:
            self.g.remove_vertex(v)
            self.vertices.discard(v)
            self.edges = {k: w for k, w in self.edges.items() if v not in k}

    @invariant()
    def agree(self) -> None:
        if not hasattr(self, "g"):
            return
        assert set(self.g.vertices()) == self.vertices
        assert self.g.edge_count == len(self.edges)
        got = {self._key(u, v): w for u, v, w in self.g.edges()}
        assert got == self.edges
        assert len(list(self.g.edges())) == len(self.edges)
        for (u, v), w in self.edges.items():
            assert self.g.weight(u, v) == w
            assert self.g.directed or self.g.weight(v, u) == w


TestGraphVsEdgeSet = GraphVsEdgeSet.TestCase


def test_isolated_vertices() -> None:
    g: Graph[int] = Graph.from_edges([], vertices=[1, 2])
    assert g.has_vertex(1)
    assert not g.has_vertex(3)
    assert g.edge_count == 0
    assert list(g.neighbors(2)) == []
