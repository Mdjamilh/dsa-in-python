"""Breadth-first and depth-first search, and what they give you for free.

Both visit every vertex reachable from ``start`` once and look at every edge
out of those vertices once, so both are O(V + E) time and O(V) space.

- **BFS** uses a FIFO queue, so it visits vertices in order of their distance
  (in edges) from ``start``. The first time it reaches a vertex is along a
  shortest path, which makes it the shortest-path algorithm for unweighted
  graphs.
- **DFS** uses a stack and follows one path as deep as it goes before backing
  up. The version here is iterative, so a long path cannot overflow Python's
  recursion limit, and it visits vertices in exactly the order the textbook
  recursive version would: neighbours are pushed in reverse so the first
  neighbour is popped first, and a vertex is marked when popped, not pushed.

Both are generators, so stopping early (``next(...)``, ``break``) does no
extra work.
"""

from collections.abc import Hashable, Iterator

from dsa.structures.deque import Deque
from dsa.structures.graph import Graph

__all__ = ["bfs", "bfs_distances", "connected_components", "dfs", "shortest_path"]


def bfs[V: Hashable](graph: Graph[V], start: V) -> Iterator[V]:
    """Yield vertices reachable from ``start`` in breadth-first order. O(V + E).

    >>> g = Graph.from_edges([(1, 2), (1, 3), (2, 4), (3, 4)])
    >>> list(bfs(g, 1))
    [1, 2, 3, 4]
    """
    if start not in graph:
        raise KeyError(start)
    seen = {start}
    queue = Deque([start])
    while queue:
        v = queue.popleft()
        yield v
        for w in graph.neighbors(v):
            if w not in seen:
                seen.add(w)  # mark on enqueue, so no vertex is queued twice
                queue.append(w)


def bfs_distances[V: Hashable](graph: Graph[V], start: V) -> dict[V, int]:
    """Return the number of edges on a shortest path from ``start`` to each reachable vertex."""
    if start not in graph:
        raise KeyError(start)
    dist = {start: 0}
    queue = Deque([start])
    while queue:
        v = queue.popleft()
        for w in graph.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


def shortest_path[V: Hashable](graph: Graph[V], start: V, goal: V) -> list[V] | None:
    """Return a path with the fewest edges from ``start`` to ``goal``, or ``None``. O(V + E).

    Weights are ignored; use ``dijkstra`` for weighted shortest paths.

    >>> g = Graph.from_edges([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")])
    >>> shortest_path(g, "a", "c")
    ['a', 'b', 'c']
    """
    if start not in graph:
        raise KeyError(start)
    parent: dict[V, V | None] = {start: None}
    queue = Deque([start])
    while queue:
        v = queue.popleft()
        if v == goal:
            path = [v]
            while (p := parent[path[-1]]) is not None:
                path.append(p)
            path.reverse()
            return path
        for w in graph.neighbors(v):
            if w not in parent:
                parent[w] = v
                queue.append(w)
    return None


def dfs[V: Hashable](graph: Graph[V], start: V) -> Iterator[V]:
    """Yield vertices reachable from ``start`` in depth-first preorder. O(V + E).

    >>> g = Graph.from_edges([(1, 2), (1, 3), (2, 4), (3, 4)])
    >>> list(dfs(g, 1))
    [1, 2, 4, 3]
    """
    if start not in graph:
        raise KeyError(start)
    seen: set[V] = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        yield v
        stack.extend(w for w in reversed(list(graph.neighbors(v))) if w not in seen)


def connected_components[V: Hashable](graph: Graph[V]) -> list[list[V]]:
    """Return the connected components of an undirected graph, each in BFS order. O(V + E).

    Raises ``ValueError`` for a directed graph, where "connected" needs a choice
    between weak and strong connectivity.
    """
    if graph.directed:
        raise ValueError("connected_components needs an undirected graph")
    seen: set[V] = set()
    components = []
    for v in graph.vertices():
        if v not in seen:
            component = list(bfs(graph, v))
            seen.update(component)
            components.append(component)
    return components
