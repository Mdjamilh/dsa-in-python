"""Dijkstra's single-source shortest paths, with a binary heap and lazy deletion.

Dijkstra grows a set of vertices whose shortest distance is final. Each round
it takes the closest vertex not yet final, finalises it, and *relaxes* its
out-edges: if going through it gives a neighbour a shorter distance, record
that. It is correct because with non-negative weights no later path can make
the closest unfinished vertex any closer.

The textbook version needs a heap with decrease-key. This one uses the
simpler "lazy deletion" approach: when a distance improves, push a new heap
entry and leave the old one in place. When a stale entry is popped (the
vertex is already final) it is skipped. The heap can hold up to E entries
instead of V, which changes O(log V) to O(log E) = O(log V) per operation.

Time: O((V + E) log V). Space: O(V + E).
"""

import math
from collections.abc import Hashable
from dataclasses import dataclass

from dsa.structures.graph import Graph
from dsa.structures.heap import PriorityQueue

__all__ = ["ShortestPaths", "dijkstra"]


@dataclass(frozen=True, slots=True)
class ShortestPaths[V: Hashable]:
    """Distances and shortest-path tree from one source.

    ``distance`` holds only reachable vertices. ``previous[v]`` is the vertex
    before ``v`` on a shortest path (absent for the source).
    """

    source: V
    distance: dict[V, float]
    previous: dict[V, V]

    def distance_to(self, v: V) -> float:
        """Return the shortest distance to ``v``, or ``math.inf`` if unreachable."""
        return self.distance.get(v, math.inf)

    def path_to(self, v: V) -> list[V] | None:
        """Return the vertices on a shortest path from the source to ``v``, or ``None``."""
        if v not in self.distance:
            return None
        path = [v]
        while path[-1] != self.source:
            path.append(self.previous[path[-1]])
        path.reverse()
        return path


def dijkstra[V: Hashable](graph: Graph[V], source: V) -> ShortestPaths[V]:
    """Compute shortest paths from ``source``. Raises ``ValueError`` on a negative edge weight.

    >>> roads = Graph.from_edges(
    ...     [("home", "cafe", 4), ("home", "park", 1), ("park", "cafe", 2), ("cafe", "office", 5)]
    ... )
    >>> paths = dijkstra(roads, "home")
    >>> paths.distance_to("office"), paths.path_to("office")
    (8, ['home', 'park', 'cafe', 'office'])
    """
    if source not in graph:
        raise KeyError(source)
    distance: dict[V, float] = {source: 0}
    previous: dict[V, V] = {}
    done: set[V] = set()
    frontier: PriorityQueue[V, float] = PriorityQueue()
    frontier.push(source, 0)
    while frontier:
        v, d = frontier.pop_with_priority()
        if v in done:
            continue  # a stale entry: v was already finalised with a smaller distance
        done.add(v)
        for w, weight in graph.adjacency(v).items():
            if weight < 0:
                raise ValueError(f"negative edge weight {weight} on {v!r} -> {w!r}")
            candidate = d + weight
            if candidate < distance.get(w, math.inf):
                distance[w] = candidate
                previous[w] = v
                frontier.push(w, candidate)
    return ShortestPaths(source, distance, previous)
