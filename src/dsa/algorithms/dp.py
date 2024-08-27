"""Dynamic programming classics.

Dynamic programming solves a problem by solving smaller versions of it once
each and building up. Each function below says what its table means, because
that sentence is the whole algorithm; the loops just fill the table in an
order where every entry's dependencies are ready.

=========================================  ==============  ==================
Function                                   Time            Space
=========================================  ==============  ==================
``edit_distance(a, b)``                    O(nm)           O(min(n, m))
``longest_common_subsequence(a, b)``       O(nm)           O(nm), to rebuild it
``longest_increasing_subsequence(xs)``     O(n log n)      O(n)
``knapsack(items, capacity)``              O(n W)          O(n W), to rebuild the choice
``min_coins(coins, amount)``               O(k A)          O(A)
``count_change(coins, amount)``            O(k A)          O(A)
``max_subarray(xs)``                       O(n)            O(1)
=========================================  ==============  ==================

n and m are input lengths, W the knapsack capacity, k the number of coin
denominations and A the amount.
"""

from bisect import bisect_left
from collections.abc import Hashable, Sequence

from dsa._typing import SupportsLessThan

__all__ = [
    "count_change",
    "edit_distance",
    "knapsack",
    "longest_common_subsequence",
    "longest_increasing_subsequence",
    "max_subarray",
    "min_coins",
]


def edit_distance[T: Hashable](a: Sequence[T], b: Sequence[T]) -> int:
    """Return the Levenshtein distance: fewest insertions, deletions and substitutions.

    ``row[j]`` is the distance between the current prefix of ``a`` and
    ``b[:j]``. Only the previous row is needed, so the shorter input is put on
    the row axis and space is O(min(n, m)).

    Works on any sequences, so strings compare by character (code point).

    >>> edit_distance("kitten", "sitting")
    3
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,  # delete x
                    cur[j - 1] + 1,  # insert y
                    prev[j - 1] + (x != y),  # substitute, free if equal
                )
            )
        prev = cur
    return prev[-1]


def longest_common_subsequence[T: Hashable](a: Sequence[T], b: Sequence[T]) -> list[T]:
    """Return a longest sequence that appears in both ``a`` and ``b`` in order (not contiguous).

    ``t[i][j]`` is the LCS length of ``a[i:]`` and ``b[j:]``. Building it from the
    end lets the answer be read off from the start with a single forward walk.

    >>> "".join(longest_common_subsequence("ABCBDAB", "BDCABA"))
    'BDAB'
    """
    n, m = len(a), len(b)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            t[i][j] = t[i + 1][j + 1] + 1 if a[i] == b[j] else max(t[i + 1][j], t[i][j + 1])
    out: list[T] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif t[i + 1][j] >= t[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def longest_increasing_subsequence[T: SupportsLessThan](xs: Sequence[T]) -> list[T]:
    """Return a longest strictly increasing subsequence, by patience sorting. O(n log n).

    ``tails[k]`` is the smallest possible last value of an increasing
    subsequence of length ``k + 1`` seen so far. It is always sorted, so each new
    value finds its place by binary search: it either extends the longest run
    or lowers one tail. ``parent`` links let the subsequence be rebuilt.

    >>> longest_increasing_subsequence([3, 1, 4, 1, 5, 9, 2, 6])
    [1, 4, 5, 6]
    """
    tails: list[T] = []
    tail_index: list[int] = []
    parent = [-1] * len(xs)
    for i, x in enumerate(xs):
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            tail_index.append(i)
        else:
            tails[k] = x
            tail_index[k] = i
        parent[i] = tail_index[k - 1] if k else -1
    out: list[T] = []
    i = tail_index[-1] if tail_index else -1
    while i >= 0:
        out.append(xs[i])
        i = parent[i]
    out.reverse()
    return out


def knapsack(items: Sequence[tuple[int, float]], capacity: int) -> tuple[float, list[int]]:
    """Solve 0/1 knapsack. ``items`` are ``(weight, value)``; return ``(best value, indices)``.

    ``t[i][c]`` is the best value using the first ``i`` items with capacity ``c``.
    Item ``i`` is either skipped (``t[i-1][c]``) or taken
    (``t[i-1][c - w] + v``). Walking back through the table recovers which.

    Weights must be non-negative integers; O(n W) is pseudo-polynomial, fine
    for capacities in the thousands, not for 10**12.

    >>> knapsack([(1, 1), (3, 4), (4, 5), (5, 7)], 7)
    (9, [1, 2])
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if any(w < 0 for w, _ in items):
        raise ValueError("weights must be non-negative")
    n = len(items)
    t: list[list[float]] = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i, (w, v) in enumerate(items, 1):
        prev, row = t[i - 1], t[i]
        for c in range(capacity + 1):
            row[c] = max(prev[c], prev[c - w] + v) if w <= c else prev[c]
    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if t[i][c] != t[i - 1][c]:
            chosen.append(i - 1)
            c -= items[i - 1][0]
    chosen.reverse()
    return t[n][capacity], chosen


def min_coins(coins: Sequence[int], amount: int) -> int | None:
    """Return the fewest coins that make ``amount`` (unlimited supply), or ``None`` if impossible.

    ``best[a]`` is the fewest coins summing to ``a``: one coin ``c`` plus the best
    for ``a - c``, over every coin. Greedy fails on sets like ``[1, 3, 4]`` for 6
    (greedy gives 4+1+1, the answer is 3+3); this does not.

    >>> min_coins([1, 3, 4], 6), min_coins([2], 3)
    (2, None)
    """
    if amount < 0:
        raise ValueError("amount must be non-negative")
    coins = [c for c in coins if c > 0]
    unreachable = amount + 1
    best = [0] + [unreachable] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    return None if best[amount] == unreachable else best[amount]


def count_change(coins: Sequence[int], amount: int) -> int:
    """Return how many multisets of ``coins`` sum to ``amount`` (order does not matter).

    ``ways[a]`` counts the ways to make ``a`` from the coins processed so far.
    Looping over coins on the outside means each combination is counted once,
    not once per ordering.

    >>> count_change([1, 2, 5], 5)
    4
    """
    if amount < 0:
        raise ValueError("amount must be non-negative")
    ways = [1] + [0] * amount
    for c in set(coins):
        if c <= 0:
            continue
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]


def max_subarray(xs: Sequence[float]) -> tuple[float, int, int]:
    """Return ``(sum, start, stop)`` of a non-empty contiguous slice with the largest sum (Kadane).

    ``here`` is the best sum of a slice ending at the current index: either
    extend the previous best slice or start fresh. ``xs[start:stop]`` is the
    slice. Raises ``ValueError`` for an empty input.

    >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    (6, 3, 7)
    """
    if not xs:
        raise ValueError("max_subarray of an empty sequence")
    best, best_start, best_stop = xs[0], 0, 1
    here, here_start = xs[0], 0
    for i in range(1, len(xs)):
        if here < 0:
            here, here_start = xs[i], i
        else:
            here += xs[i]
        if here > best:
            best, best_start, best_stop = here, here_start, i + 1
    return best, best_start, best_stop
