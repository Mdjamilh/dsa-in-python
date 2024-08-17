"""Algorithms."""

from dsa.algorithms.binary_search import (
    binary_search,
    count_equal,
    first_true,
    lower_bound,
    search_rotated,
    upper_bound,
)
from dsa.algorithms.dijkstra import ShortestPaths, dijkstra
from dsa.algorithms.mst import kruskal, prim, total_weight
from dsa.algorithms.sorting import (
    counting_sort,
    heap_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    radix_sort,
)
from dsa.algorithms.topological_sort import CycleError, topological_sort
from dsa.algorithms.traversal import (
    bfs,
    bfs_distances,
    connected_components,
    dfs,
    shortest_path,
)

__all__ = [
    "CycleError",
    "ShortestPaths",
    "bfs",
    "bfs_distances",
    "binary_search",
    "connected_components",
    "count_equal",
    "counting_sort",
    "dfs",
    "dijkstra",
    "first_true",
    "heap_sort",
    "insertion_sort",
    "kruskal",
    "lower_bound",
    "merge_sort",
    "prim",
    "quick_sort",
    "radix_sort",
    "search_rotated",
    "shortest_path",
    "topological_sort",
    "total_weight",
    "upper_bound",
]
