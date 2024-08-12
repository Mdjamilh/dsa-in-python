import bisect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.algorithms import (
    binary_search,
    count_equal,
    first_true,
    lower_bound,
    search_rotated,
    upper_bound,
)

sorted_lists = st.lists(st.integers(-20, 20)).map(sorted)


def test_bounds_on_small_cases() -> None:
    a = [1, 2, 2, 2, 5]
    assert [lower_bound(a, x) for x in [0, 1, 2, 3, 5, 6]] == [0, 0, 1, 4, 4, 5]
    assert [upper_bound(a, x) for x in [0, 1, 2, 3, 5, 6]] == [0, 1, 4, 4, 5, 5]
    assert lower_bound([], 1) == 0
    assert upper_bound([], 1) == 0


def test_lo_hi_window() -> None:
    a = [1, 2, 3, 4, 5, 6]
    assert lower_bound(a, 1, lo=2) == 2
    assert upper_bound(a, 6, hi=3) == 3


def test_key_applies_to_elements_like_bisect() -> None:
    people = [("ana", 20), ("bo", 25), ("cy", 25), ("di", 31)]

    def age(p: tuple[str, int]) -> int:
        return p[1]

    assert lower_bound(people, 25, key=age) == bisect.bisect_left(people, 25, key=age) == 1
    assert upper_bound(people, 25, key=age) == bisect.bisect_right(people, 25, key=age) == 3


def test_first_true_binary_search_on_the_answer() -> None:
    # Smallest ship capacity that ships these weights, in order, within 5 days.
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def fits(capacity: int) -> bool:
        days, load = 1, 0
        for w in weights:
            if load + w > capacity:
                days, load = days + 1, 0
            load += w
        return days <= 5

    assert first_true(max(weights), sum(weights) + 1, fits) == 15
    assert first_true(0, 10, lambda _: False) == 10
    assert first_true(3, 3, lambda _: True) == 3


@pytest.mark.parametrize(("x", "expected"), [(1, 0), (3, 1), (7, 3), (4, None), (0, None)])
def test_binary_search(x: int, expected: int | None) -> None:
    assert binary_search([1, 3, 5, 7], x) == expected


def test_search_rotated_cases() -> None:
    a = [4, 5, 6, 7, 0, 1, 2]
    assert [search_rotated(a, x) for x in a] == list(range(len(a)))
    assert search_rotated(a, 3) is None
    assert search_rotated([], 1) is None
    assert search_rotated([1], 1) == 0


@given(sorted_lists, st.integers(-25, 25))
def test_bounds_match_bisect(a: list[int], x: int) -> None:
    assert lower_bound(a, x) == bisect.bisect_left(a, x)
    assert upper_bound(a, x) == bisect.bisect_right(a, x)
    assert count_equal(a, x) == a.count(x)


@given(sorted_lists, st.integers(-25, 25))
def test_binary_search_finds_first_occurrence(a: list[int], x: int) -> None:
    assert binary_search(a, x) == (a.index(x) if x in a else None)


@given(st.sets(st.integers(-50, 50)), st.integers(0, 100), st.integers(-55, 55))
def test_search_rotated_matches_index(values: set[int], shift: int, x: int) -> None:
    a = sorted(values)
    if a:
        k = shift % len(a)
        a = a[k:] + a[:k]
    assert search_rotated(a, x) == (a.index(x) if x in a else None)


@given(st.integers(-100, 100), st.integers(0, 200), st.integers(-100, 300))
def test_first_true_matches_linear_scan(lo: int, width: int, threshold: int) -> None:
    hi = lo + width
    expected = next((i for i in range(lo, hi) if i >= threshold), hi)
    assert first_true(lo, hi, lambda i: i >= threshold) == expected
