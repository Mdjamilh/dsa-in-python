"""Topological sort with Kahn's algorithm, reporting a concrete cycle if there is one.

A topological order lists the vertices of a directed graph so that every edge
goes from an earlier vertex to a later one. It exists exactly when the graph
has no directed cycle.

Kahn's algorithm counts each vertex's incoming edges. Vertices with none can
go first; output one, remove its out-edges by decrementing its neighbours'
counts, and any neighbour that drops to zero becomes ready. Ties are broken
in insertion order, so the result is deterministic.

If vertices remain when nothing is ready, every one of them still has an
incoming edge from another remaining vertex. Starting anywhere among them and
repeatedly stepping to such a predecessor must eventually revisit a vertex,
and the loop found that way, read in reverse, is a cycle. Reporting it is far
more useful than "graph has a cycle".

Time: O(V + E). Space: O(V).
"""

from collections.abc import Hashable

from dsa.structures.deque import Deque
from dsa.structures.graph import Graph

__all__ = ["CycleError", "topological_sort"]


class CycleError[V](ValueError):
    """Raised when a directed graph has a cycle.

    ``cycle`` lists the vertices in order: each has an edge to the next, and the
    last has an edge back to the first.
    """

    def __init__(self, cycle: list[V]) -> None:
        self.cycle = cycle
        shown = " -> ".join(map(repr, [*cycle, cycle[0]]))
        super().__init__(f"graph has a cycle: {shown}")


def topological_sort[V: Hashable](graph: Graph[V]) -> list[V]:
    """Return the vertices in an order where every edge points forward.

    Raises ``CycleError`` (a ``ValueError``) naming a cycle when there is none,
    and ``ValueError`` for an undirected graph.

    >>> steps = Graph.from_edges([("install", "build"), ("build", "test")], directed=True)
    >>> topological_sort(steps)
    ['install', 'build', 'test']
    """
    if not graph.directed:
        raise ValueError("topological_sort needs a directed graph")
    indegree = dict.fromkeys(graph.vertices(), 0)
    for _, v, _ in graph.edges():
        indegree[v] += 1
    ready = Deque(v for v, d in indegree.items() if d == 0)
    order: list[V] = []
    while ready:
        v = ready.popleft()
        order.append(v)
        for w in graph.neighbors(v):
            indegree[w] -= 1
            if indegree[w] == 0:
                ready.append(w)
    if len(order) < len(indegree):
        raise CycleError(_find_cycle(graph, {v for v, d in indegree.items() if d > 0}))
    return order


def _find_cycle[V: Hashable](graph: Graph[V], remaining: set[V]) -> list[V]:
    """Walk predecessors inside ``remaining`` until a vertex repeats. O(V + E)."""
    predecessor: dict[V, V] = {}
    for u, v, _ in graph.edges():
        if u in remaining and v in remaining:
            predecessor.setdefault(v, u)
    # Start from the first remaining vertex in insertion order, not set order, so
    # the reported cycle does not depend on hash randomisation.
    v = next(u for u in graph.vertices() if u in remaining)
    position: dict[V, int] = {}
    walk: list[V] = []
    while v not in position:
        position[v] = len(walk)
        walk.append(v)
        v = predecessor[v]
    loop = walk[position[v] :]  # each step went to a predecessor, so this runs backwards
    loop.reverse()
    return loop
