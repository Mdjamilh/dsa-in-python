import itertools
from collections.abc import Sequence
from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.algorithms import (
    count_change,
    edit_distance,
    knapsack,
    longest_common_subsequence,
    longest_increasing_subsequence,
    max_subarray,
    min_coins,
)

short_text = st.text(alphabet="abc", max_size=8)


def is_subsequence[T](sub: Sequence[T], seq: Sequence[T]) -> bool:
    it = iter(seq)
    return all(ch in it for ch in sub)


def edit_distance_reference(a: str, b: str) -> int:
    """The textbook recursion, memoised. Exponential without the cache, obviously right."""

    @cache
    def d(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return i + j
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))

    return d(len(a), len(b))


# -- edit distance -------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("", "abc", 3), ("kitten", "sitting", 3), ("flaw", "lawn", 2), ("abc", "abc", 0)],
)
def test_edit_distance_examples(a: str, b: str, expected: int) -> None:
    assert edit_distance(a, b) == expected


def test_edit_distance_on_lists() -> None:
    assert edit_distance(["the", "cat", "sat"], ["the", "dog", "sat", "down"]) == 2


@given(short_text, short_text)
def test_edit_distance_matches_recursion(a: str, b: str) -> None:
    assert edit_distance(a, b) == edit_distance_reference(a, b)


@given(short_text, short_text, short_text)
def test_edit_distance_is_a_metric(a: str, b: str, c: str) -> None:
    assert edit_distance(a, b) == edit_distance(b, a)
    assert (edit_distance(a, b) == 0) == (a == b)
    assert edit_distance(a, c) <= edit_distance(a, b) + edit_distance(b, c)
    assert abs(len(a) - len(b)) <= edit_distance(a, b) <= max(len(a), len(b))


# -- LCS -----------------------------------------------------------------------


def test_lcs_examples() -> None:
    assert "".join(longest_common_subsequence("ABCBDAB", "BDCABA")) == "BDAB"
    assert longest_common_subsequence("", "abc") == []
    assert len(longest_common_subsequence([1, 2, 3], [3, 2, 1])) == 1  # any one element


@given(short_text, short_text)
def test_lcs_is_common_and_longest(a: str, b: str) -> None:
    lcs = longest_common_subsequence(a, b)
    assert is_subsequence(lcs, a)
    assert is_subsequence(lcs, b)
    longest = max(
        (
            r
            for r in range(len(a) + 1)
            for sub in itertools.combinations(a, r)
            if is_subsequence(sub, b)
        ),
        default=0,
    )
    assert len(lcs) == longest


# -- LIS -----------------------------------------------------------------------


def test_lis_examples() -> None:
    assert longest_increasing_subsequence([]) == []
    assert longest_increasing_subsequence([5, 4, 3]) == [3]
    assert longest_increasing_subsequence([1, 1, 1]) == [1]  # strictly increasing
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == [2, 3, 7, 18]


@given(st.lists(st.integers(-5, 5), max_size=10))
def test_lis_matches_brute_force(xs: list[int]) -> None:
    lis = longest_increasing_subsequence(xs)
    assert all(a < b for a, b in itertools.pairwise(lis))
    assert is_subsequence(lis, xs)
    best = max(
        (
            r
            for r in range(len(xs) + 1)
            for sub in itertools.combinations(xs, r)
            if all(a < b for a, b in itertools.pairwise(sub))
        ),
        default=0,
    )
    assert len(lis) == best


# -- knapsack ------------------------------------------------------------------


def test_knapsack_examples_and_errors() -> None:
    assert knapsack([], 10) == (0, [])
    assert knapsack([(5, 10)], 4) == (0, [])
    assert knapsack([(0, 3), (2, 2)], 0) == (3, [0])
    with pytest.raises(ValueError, match="capacity"):
        knapsack([(1, 1)], -1)
    with pytest.raises(ValueError, match="weights"):
        knapsack([(-1, 1)], 1)


@given(
    st.lists(st.tuples(st.integers(0, 10), st.integers(0, 20)), max_size=8),
    st.integers(0, 30),
)
def test_knapsack_matches_brute_force(items: list[tuple[int, int]], capacity: int) -> None:
    value, chosen = knapsack(items, capacity)
    assert sorted(set(chosen)) == chosen
    assert sum(items[i][0] for i in chosen) <= capacity
    assert sum(items[i][1] for i in chosen) == value
    best = max(
        sum(v for _, v in subset)
        for r in range(len(items) + 1)
        for subset in itertools.combinations(items, r)
        if sum(w for w, _ in subset) <= capacity
    )
    assert value == best


# -- coins ---------------------------------------------------------------------


def test_coin_examples_and_errors() -> None:
    assert min_coins([1, 3, 4], 6) == 2  # greedy would give 3
    assert min_coins([1, 5, 10, 25], 63) == 6
    assert min_coins([2], 3) is None
    assert min_coins([], 0) == 0
    assert min_coins([0, -1, 2], 4) == 2  # non-positive coins are ignored
    assert count_change([1, 2, 5], 5) == 4
    assert count_change([2, 2, 0], 4) == 1  # duplicates and zero ignored
    assert count_change([], 0) == 1
    for fn in (min_coins, count_change):
        with pytest.raises(ValueError, match="amount"):
            fn([1], -1)


def coin_combinations(coins: list[int], amount: int) -> list[tuple[int, ...]]:
    """Every multiset of coins summing to ``amount``, by recursion over sorted coins."""
    coins = sorted({c for c in coins if c > 0})
    out: list[tuple[int, ...]] = []

    def go(i: int, left: int, picked: tuple[int, ...]) -> None:
        if left == 0:
            out.append(picked)
            return
        for j in range(i, len(coins)):
            if coins[j] <= left:
                go(j, left - coins[j], (*picked, coins[j]))

    go(0, amount, ())
    return out


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4), st.integers(0, 25))
def test_coins_match_enumeration(coins: list[int], amount: int) -> None:
    combos = coin_combinations(coins, amount)
    assert count_change(coins, amount) == len(combos)
    assert min_coins(coins, amount) == (min(map(len, combos)) if combos else None)


# -- max subarray --------------------------------------------------------------


def test_max_subarray_examples() -> None:
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == (6, 3, 7)
    assert max_subarray([-3, -1, -2]) == (-1, 1, 2)  # all negative: the best single element
    with pytest.raises(ValueError, match="empty"):
        max_subarray([])


@given(st.lists(st.integers(-20, 20), min_size=1, max_size=30))
def test_max_subarray_matches_all_slices(xs: list[int]) -> None:
    total, start, stop = max_subarray(xs)
    assert sum(xs[start:stop]) == total
    assert start < stop
    best = max(sum(xs[i:j]) for i in range(len(xs)) for j in range(i + 1, len(xs) + 1))
    assert total == best
