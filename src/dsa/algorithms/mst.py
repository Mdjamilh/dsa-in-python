"""Minimum spanning trees: Kruskal's and Prim's algorithms.

A minimum spanning tree connects every vertex of a connected, undirected,
weighted graph using edges of the least possible total weight. On a graph that
is not connected, both functions return a minimum spanning *forest*: one tree
per component.

Both rely on the cut property: for any split of the vertices into two groups,
the lightest edge crossing the split belongs to some minimum spanning tree.

- **Kruskal** sorts all edges by weight and takes each one that joins two
  different trees, which union-find answers in near O(1). O(E log E).
- **Prim** grows one tree from a start vertex, always adding the lightest edge
  leaving it, with a heap of candidate edges and lazy deletion. Restarting
  from each untouched vertex covers every component. O(E log V).

Space: O(V + E) for both.
"""

from collections.abc import Hashable

from dsa.structures.graph import Edge, Graph
from dsa.structures.heap import PriorityQueue
from dsa.structures.union_find import UnionFind

__all__ = ["kruskal", "prim", "total_weight"]


def _require_undirected[V: Hashable](graph: Graph[V]) -> None:
    if graph.directed:
        raise ValueError("minimum spanning trees need an undirected graph")


def kruskal[V: Hashable](graph: Graph[V]) -> list[Edge[V]]:
    """Return the edges of a minimum spanning forest, in the order Kruskal takes them.

    >>> g = Graph.from_edges([("a", "b", 4), ("b", "c", 1), ("a", "c", 2), ("c", "d", 7)])
    >>> kruskal(g)
    [('b', 'c', 1), ('a', 'c', 2), ('c', 'd', 7)]
    """
    _require_undirected(graph)
    forest = UnionFind(graph.vertices())
    chosen: list[Edge[V]] = []
    for u, v, w in sorted(graph.edges(), key=lambda e: e[2]):
        if forest.union(u, v):  # False means u and v are already connected
            chosen.append((u, v, w))
            if len(chosen) == graph.vertex_count - 1:
                break  # a spanning tree is complete, so no later edge can join anything
    return chosen


def prim[V: Hashable](graph: Graph[V]) -> list[Edge[V]]:
    """Return the edges of a minimum spanning forest, grown tree by tree.

    >>> g = Graph.from_edges([("a", "b", 4), ("b", "c", 1), ("a", "c", 2), ("c", "d", 7)])
    >>> prim(g)
    [('a', 'c', 2), ('c', 'b', 1), ('c', 'd', 7)]
    """
    _require_undirected(graph)
    in_tree: set[V] = set()
    chosen: list[Edge[V]] = []
    for root in graph.vertices():
        if root in in_tree:
            continue
        in_tree.add(root)
        candidates: PriorityQueue[Edge[V], float] = PriorityQueue()
        for v, w in graph.adjacency(root).items():
            candidates.push((root, v, w), w)
        while candidates:
            u, v, w = candidates.pop()
            if v in in_tree:
                continue  # stale: both ends were already joined
            in_tree.add(v)
            chosen.append((u, v, w))
            for x, wx in graph.adjacency(v).items():
                if x not in in_tree:
                    candidates.push((v, x, wx), wx)
    return chosen


def total_weight[V](edges: list[Edge[V]]) -> float:
    """Return the sum of the edge weights."""
    return sum(w for _, _, w in edges)
