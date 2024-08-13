import random
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.algorithms import (
    counting_sort,
    heap_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    radix_sort,
)

type Sorter = Callable[..., list[Any]]

COMPARISON_SORTS: list[Sorter] = [insertion_sort, merge_sort, quick_sort, heap_sort]
STABLE_SORTS: list[Sorter] = [insertion_sort, merge_sort]


def ids(fns: Iterable[Sorter]) -> list[str]:
    return [f.__name__ for f in fns]


@pytest.mark.parametrize("sort", COMPARISON_SORTS, ids=ids(COMPARISON_SORTS))
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1],
        [2, 1],
        list(range(100)),
        list(range(100, 0, -1)),
        [5] * 50,
        [3, 1, 2] * 30,
        list(range(50)) + list(range(50, 0, -1)),  # organ pipe
    ],
    ids=["empty", "one", "two", "sorted", "reversed", "all-equal", "repeating", "organ-pipe"],
)
def test_edge_case_inputs(sort: Sorter, data: list[int]) -> None:
    assert sort(data) == sorted(data)
    assert sort(data, reverse=True) == sorted(data, reverse=True)


@pytest.mark.parametrize("sort", COMPARISON_SORTS, ids=ids(COMPARISON_SORTS))
def test_does_not_modify_input(sort: Sorter) -> None:
    data = [3, 1, 2]
    sort(data)
    assert data == [3, 1, 2]


@pytest.mark.parametrize("sort", COMPARISON_SORTS, ids=ids(COMPARISON_SORTS))
def test_key_is_called_once_per_element(sort: Sorter) -> None:
    calls = Counter[int]()

    def key(x: int) -> int:
        calls[x] += 1
        return -x

    data = list(range(200))
    random.Random(1).shuffle(data)
    assert sort(data, key=key) == sorted(data, reverse=True)
    assert set(calls.values()) == {1}


@pytest.mark.parametrize("sort", COMPARISON_SORTS, ids=ids(COMPARISON_SORTS))
@given(data=st.lists(st.integers()))
def test_matches_sorted(sort: Sorter, data: list[int]) -> None:
    assert sort(data) == sorted(data)


@pytest.mark.parametrize("sort", COMPARISON_SORTS, ids=ids(COMPARISON_SORTS))
@given(data=st.lists(st.text(max_size=3)), reverse=st.booleans())
def test_matches_sorted_with_key_and_reverse(sort: Sorter, data: list[str], reverse: bool) -> None:
    result = sort(data, key=len, reverse=reverse)
    assert [len(s) for s in result] == sorted((len(s) for s in data), reverse=reverse)
    assert Counter(result) == Counter(data)


@pytest.mark.parametrize("sort", STABLE_SORTS, ids=ids(STABLE_SORTS))
@given(
    data=st.lists(st.tuples(st.integers(0, 5), st.integers()), max_size=200),
    reverse=st.booleans(),
)
def test_stable_sorts_match_sorted_exactly(
    sort: Sorter, data: list[tuple[int, int]], reverse: bool
) -> None:
    # sorted() is stable, including with reverse=True, so equal keys must keep their order.
    def first(p: tuple[int, int]) -> int:
        return p[0]

    assert sort(data, key=first, reverse=reverse) == sorted(data, key=first, reverse=reverse)


def test_merge_sort_skips_merges_on_sorted_input() -> None:
    comparisons = 0

    def key(x: int) -> "Counted":
        return Counted(x)

    class Counted:
        def __init__(self, v: int) -> None:
            self.v = v

        def __lt__(self, other: "Counted") -> bool:
            nonlocal comparisons
            comparisons += 1
            return self.v < other.v

    n = 4096
    merge_sort(range(n), key=key)
    assert comparisons < 2 * n  # linear, not n log n (about 49,000 here)


def test_quick_sort_is_reproducible_with_a_seed() -> None:
    data = [(i % 7, i) for i in range(500)]

    def first(p: tuple[int, int]) -> int:
        return p[0]

    a = quick_sort(data, key=first, rng=random.Random(42))
    b = quick_sort(data, key=first, rng=random.Random(42))
    assert a == b


def test_quick_sort_handles_large_equal_runs_quickly() -> None:
    data = [1, 2] * 50_000
    assert quick_sort(data) == sorted(data)


def test_sorts_floats_and_mixed_numeric_types() -> None:
    data: list[float] = [2.5, -1, 0, 3, 1e-9, -0.0]
    for sort in COMPARISON_SORTS:
        assert sort(data) == sorted(data)


# -- integer sorts -------------------------------------------------------------


@given(st.lists(st.integers(-1000, 1000)))
def test_counting_sort_matches_sorted(data: list[int]) -> None:
    assert counting_sort(data) == sorted(data)


@given(st.lists(st.integers(-(2**64), 2**64)), st.sampled_from([2, 10, 16, 256, 4096]))
def test_radix_sort_matches_sorted(data: list[int], base: int) -> None:
    assert radix_sort(data, base=base) == sorted(data)


def test_integer_sort_edge_cases() -> None:
    assert counting_sort([]) == []
    assert radix_sort([]) == []
    assert radix_sort([0, 0, -1]) == [-1, 0, 0]
    assert radix_sort([2**70, 1, -(2**70)]) == [-(2**70), 1, 2**70]
    with pytest.raises(ValueError, match="base"):
        radix_sort([1], base=1)
