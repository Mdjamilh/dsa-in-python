"""Hypothesis strategies shared by the graph tests."""

from hypothesis import strategies as st

from dsa.structures import Graph


@st.composite
def graphs(
    draw: st.DrawFn,
    *,
    directed: bool | None = None,
    max_vertices: int = 8,
    max_edges: int = 20,
    min_weight: float = 0.0,
    max_weight: float = 10.0,
    integer_weights: bool = True,
) -> Graph[int]:
    """Draw a small graph on vertices ``0..n-1`` with random edges.

    Integer weights by default, so sums are exact and distances can be compared
    with ``==`` instead of a tolerance.
    """
    is_directed = draw(st.booleans()) if directed is None else directed
    n = draw(st.integers(1, max_vertices))
    weight = (
        st.integers(int(min_weight), int(max_weight)).map(float)
        if integer_weights
        else st.floats(min_weight, max_weight)
    )
    edges = draw(
        st.lists(
            st.tuples(st.integers(0, n - 1), st.integers(0, n - 1), weight), max_size=max_edges
        )
    )
    return Graph.from_edges(edges, directed=is_directed, vertices=range(n))
