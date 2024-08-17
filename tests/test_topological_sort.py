import itertools

import pytest
from hypothesis import given

from dsa.algorithms import CycleError, topological_sort
from dsa.structures import Graph
from tests.strategies import graphs


def has_cycle_brute_force(g: Graph[int]) -> bool:
    """A directed graph is acyclic iff some permutation of its vertices puts every edge forward."""
    vs = list(g.vertices())
    edges = [(u, v) for u, v, _ in g.edges()]
    for perm in itertools.permutations(vs):
        pos = {v: i for i, v in enumerate(perm)}
        if all(pos[u] < pos[v] for u, v in edges):
            return False
    return True


def test_simple_order_is_deterministic() -> None:
    g = Graph.from_edges(
        [("shirt", "tie"), ("tie", "jacket"), ("trousers", "shoes"), ("trousers", "belt")],
        directed=True,
    )
    assert topological_sort(g) == ["shirt", "trousers", "tie", "shoes", "belt", "jacket"]


def test_reports_the_cycle() -> None:
    g = Graph.from_edges(
        [("install", "build"), ("build", "test"), ("test", "build")], directed=True
    )
    with pytest.raises(CycleError) as info:
        topological_sort(g)
    cycle = info.value.cycle
    assert sorted(cycle) == ["build", "test"]
    assert "graph has a cycle: " in str(info.value)
    assert isinstance(info.value, ValueError)


def test_self_loop_is_a_cycle() -> None:
    g = Graph.from_edges([(1, 1)], directed=True)
    with pytest.raises(CycleError) as info:
        topological_sort(g)
    assert info.value.cycle == [1]
    assert str(info.value) == "graph has a cycle: 1 -> 1"


def test_rejects_undirected() -> None:
    with pytest.raises(ValueError, match="directed"):
        topological_sort(Graph.from_edges([(1, 2)]))


@given(graphs(directed=True, max_vertices=6, max_edges=10))
def test_order_or_cycle_is_genuine(g: Graph[int]) -> None:
    try:
        order = topological_sort(g)
    except CycleError as e:
        assert has_cycle_brute_force(g)
        cycle = e.cycle
        assert len(set(cycle)) == len(cycle)
        for a, b in zip(cycle, [*cycle[1:], cycle[0]], strict=True):
            assert g.has_edge(a, b)
    else:
        assert not has_cycle_brute_force(g)
        assert sorted(order) == sorted(g.vertices())
        pos = {v: i for i, v in enumerate(order)}
        assert all(pos[u] < pos[v] for u, v, _ in g.edges())
