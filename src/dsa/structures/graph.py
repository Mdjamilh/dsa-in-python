"""A weighted graph as an adjacency map, directed or undirected.

Each vertex maps to a dict of ``neighbour -> weight``. That gives O(1) edge
lookup (like an adjacency matrix) while using space proportional to the edges
(like an adjacency list), and neighbours come out in insertion order, which
keeps every traversal deterministic.

An undirected edge is stored in both directions but counted once. There is at
most one edge per ordered pair: adding the same edge again replaces its
weight. Self-loops are allowed.

==================================  ==============
Operation                           Time
==================================  ==============
``add_vertex``, ``add_edge``        O(1)
``has_edge``, ``weight``            O(1)
``remove_edge``                     O(1)
``remove_vertex``                   O(degree) undirected, O(V) directed
``neighbors``                       O(1) to start, O(degree) to exhaust
``edges``                           O(V + E)
==================================  ==============

Space: O(V + E).
"""

import math
from collections.abc import Hashable, Iterable, Iterator, Mapping
from types import MappingProxyType

__all__ = ["Edge", "Graph"]

type Edge[V] = tuple[V, V, float]
"""An edge as ``(source, target, weight)``."""


class Graph[V: Hashable]:
    """A graph whose vertices are any hashable values.

    >>> g = Graph.from_edges([("a", "b", 2.0), ("b", "c", 1.0)])
    >>> sorted(g.neighbors("b")), g.weight("a", "b"), g.edge_count
    (['a', 'c'], 2.0, 2)
    """

    __slots__ = ("_adj", "_edge_count", "directed")

    def __init__(self, *, directed: bool = False) -> None:
        self.directed = directed
        self._adj: dict[V, dict[V, float]] = {}
        self._edge_count = 0

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[V, V] | tuple[V, V, float]],
        *,
        directed: bool = False,
        vertices: Iterable[V] = (),
    ) -> "Graph[V]":
        """Build a graph from ``(u, v)`` or ``(u, v, weight)`` tuples. Weight defaults to 1."""
        g: Graph[V] = cls(directed=directed)
        for v in vertices:
            g.add_vertex(v)
        for edge in edges:
            if len(edge) == 2:
                g.add_edge(edge[0], edge[1])
            else:
                g.add_edge(edge[0], edge[1], edge[2])
        return g

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return len(self._adj)

    @property
    def edge_count(self) -> int:
        """Number of edges (an undirected edge counts once)."""
        return self._edge_count

    def add_vertex(self, v: V) -> None:
        """Add ``v`` with no edges, if it is new."""
        self._adj.setdefault(v, {})

    def add_edge(self, u: V, v: V, weight: float = 1.0) -> None:
        """Add the edge ``u -> v`` (both ways if undirected), creating vertices as needed.

        Raises ``ValueError`` for a NaN weight, which would break every comparison.
        """
        if math.isnan(weight):
            raise ValueError("edge weight must not be NaN")
        self.add_vertex(u)
        self.add_vertex(v)
        if v not in self._adj[u]:
            self._edge_count += 1
        self._adj[u][v] = weight
        if not self.directed:
            self._adj[v][u] = weight

    def has_vertex(self, v: V) -> bool:
        """Return whether ``v`` is a vertex."""
        return v in self._adj

    def has_edge(self, u: V, v: V) -> bool:
        """Return whether the edge ``u -> v`` exists."""
        return v in self._adj.get(u, {})

    def weight(self, u: V, v: V) -> float:
        """Return the weight of ``u -> v``. Raises ``KeyError`` if there is no such edge."""
        try:
            return self._adj[u][v]
        except KeyError:
            raise KeyError((u, v)) from None

    def remove_edge(self, u: V, v: V) -> None:
        """Remove the edge ``u -> v`` (both ways if undirected). Raises ``KeyError`` if absent."""
        if not self.has_edge(u, v):
            raise KeyError((u, v))
        del self._adj[u][v]
        if not self.directed and u != v:
            del self._adj[v][u]
        self._edge_count -= 1

    def remove_vertex(self, v: V) -> None:
        """Remove ``v`` and every edge touching it. Raises ``KeyError`` if absent."""
        out = self._adj.pop(v)
        if self.directed:
            self._edge_count -= len(out)
            for nbrs in self._adj.values():
                if v in nbrs:
                    del nbrs[v]
                    self._edge_count -= 1
        else:
            for u in out:
                if u != v:
                    del self._adj[u][v]
            self._edge_count -= len(out)

    def neighbors(self, v: V) -> Iterator[V]:
        """Iterate the vertices that ``v`` has an edge to. Raises ``KeyError`` if absent."""
        return iter(self._adj[v])

    def adjacency(self, v: V) -> Mapping[V, float]:
        """Return a read-only view of ``v``'s neighbours and edge weights."""
        return MappingProxyType(self._adj[v])

    def degree(self, v: V) -> int:
        """Return the out-degree of ``v`` (a self-loop counts once)."""
        return len(self._adj[v])

    def vertices(self) -> Iterator[V]:
        """Iterate vertices in insertion order."""
        return iter(self._adj)

    def edges(self) -> Iterator[Edge[V]]:
        """Iterate edges as ``(u, v, weight)``. Undirected edges appear once."""
        seen: set[V] = set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                if self.directed or v not in seen:
                    yield (u, v, w)
            if not self.directed:
                seen.add(u)

    def reversed(self) -> "Graph[V]":
        """Return a copy with every edge flipped (the same graph if undirected)."""
        g: Graph[V] = Graph(directed=self.directed)
        for v in self._adj:
            g.add_vertex(v)
        for u, v, w in self.edges():
            g.add_edge(v, u, w)
        return g

    def __contains__(self, v: object) -> bool:
        return v in self._adj

    def __len__(self) -> int:
        return len(self._adj)

    def __iter__(self) -> Iterator[V]:
        return iter(self._adj)

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        return f"Graph({kind}, vertices={self.vertex_count}, edges={self.edge_count})"
