"""Comparison sorts (merge, quick, heap, insertion) and integer sorts (counting, radix).

Every comparison sort here has the same interface as ``sorted()``: it takes
any iterable, an optional ``key`` and ``reverse``, and returns a new list. When
``key`` is given it is called once per element, as ``sorted()`` does, and the
results are carried alongside the elements ("decorate, sort, undecorate").
``reverse=True`` flips the comparison rather than reversing the output, so
stable sorts stay stable in both directions, again like ``sorted()``.

==================  ======================  =========  =======  ==============================
Algorithm           Time                    Space      Stable   Notes
==================  ======================  =========  =======  ==============================
``insertion_sort``  O(n^2), O(n) if sorted  O(1)       yes      fastest for tiny inputs
``merge_sort``      O(n log n)              O(n)       yes      O(n) if already sorted
``quick_sort``      O(n log n) expected     O(log n)   no       random pivot, three-way split
``heap_sort``       O(n log n) worst        O(1)       no       in place
``counting_sort``   O(n + k)                O(n + k)   yes      ints in a range of size k
``radix_sort``      O(d (n + b))            O(n + b)   yes      ints; d digits in base b
==================  ======================  =========  =======  ==============================

(Space is the extra space beyond the output list.) None of these beat the
built-in ``sorted()``, which is Timsort in C. See the benchmarks in the README.
"""

import operator
import random
from collections.abc import Callable, Iterable
from typing import Any, overload

from dsa._typing import SupportsLessThan

__all__ = [
    "counting_sort",
    "heap_sort",
    "insertion_sort",
    "merge_sort",
    "quick_sort",
    "radix_sort",
]

type _Less = Callable[[Any, Any], bool]

_INSERTION_CUTOFF = 16


def _run(
    items: Iterable[Any],
    key: Callable[[Any], SupportsLessThan] | None,
    reverse: bool,
    algorithm: Callable[[list[Any], _Less], None],
) -> list[Any]:
    """Apply an in-place ``algorithm(list, less)`` with ``sorted()``-style key and reverse."""
    if key is None:
        values = list(items)
        algorithm(values, _flip(operator.lt) if reverse else operator.lt)
        return values
    decorated = [(key(x), x) for x in items]

    def by_key(a: tuple[SupportsLessThan, Any], b: tuple[SupportsLessThan, Any]) -> bool:
        return a[0] < b[0]

    algorithm(decorated, _flip(by_key) if reverse else by_key)
    return [x for _, x in decorated]


def _flip(less: _Less) -> _Less:
    return lambda a, b: less(b, a)


# -- insertion sort ------------------------------------------------------------


def _insertion(a: list[Any], less: _Less, lo: int = 0, hi: int | None = None) -> None:
    """Sort ``a[lo:hi]`` in place. Shifts larger elements right, then drops the item in."""
    hi = len(a) if hi is None else hi
    for i in range(lo + 1, hi):
        item = a[i]
        j = i
        while j > lo and less(item, a[j - 1]):
            a[j] = a[j - 1]
            j -= 1
        a[j] = item


@overload
def insertion_sort[T: SupportsLessThan](
    items: Iterable[T], *, key: None = None, reverse: bool = False
) -> list[T]: ...
@overload
def insertion_sort[T](
    items: Iterable[T], *, key: Callable[[T], SupportsLessThan], reverse: bool = False
) -> list[T]: ...
def insertion_sort(
    items: Iterable[Any],
    *,
    key: Callable[[Any], SupportsLessThan] | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Return a sorted list using insertion sort. Stable. O(n^2) time, O(n) on sorted input.

    >>> insertion_sort([3, 1, 2])
    [1, 2, 3]
    """
    return _run(items, key, reverse, _insertion)


# -- merge sort ----------------------------------------------------------------


def _merge_sort(a: list[Any], less: _Less) -> None:
    """Top-down merge sort with one shared buffer.

    Small ranges use insertion sort. If the two sorted halves are already in
    order (the last of the left is not greater than the first of the right) the
    merge is skipped, which makes sorted input O(n). Recursion depth is log2 n.
    """
    buffer = a.copy()

    def sort(lo: int, hi: int) -> None:
        if hi - lo <= _INSERTION_CUTOFF:
            _insertion(a, less, lo, hi)
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        if not less(a[mid], a[mid - 1]):
            return
        buffer[lo:hi] = a[lo:hi]
        i, j = lo, mid
        for k in range(lo, hi):
            # Take from the right only when strictly smaller: that is what keeps it stable.
            if j < hi and (i >= mid or less(buffer[j], buffer[i])):
                a[k] = buffer[j]
                j += 1
            else:
                a[k] = buffer[i]
                i += 1

    sort(0, len(a))


@overload
def merge_sort[T: SupportsLessThan](
    items: Iterable[T], *, key: None = None, reverse: bool = False
) -> list[T]: ...
@overload
def merge_sort[T](
    items: Iterable[T], *, key: Callable[[T], SupportsLessThan], reverse: bool = False
) -> list[T]: ...
def merge_sort(
    items: Iterable[Any],
    *,
    key: Callable[[Any], SupportsLessThan] | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Return a sorted list using merge sort. Stable. O(n log n) time, O(n) extra space.

    >>> merge_sort(["bb", "a", "cc", "d"], key=len)
    ['a', 'd', 'bb', 'cc']
    """
    return _run(items, key, reverse, _merge_sort)


# -- quick sort ----------------------------------------------------------------


def _quick_sort(a: list[Any], less: _Less, rng: random.Random) -> None:
    """Randomised quick sort with a three-way partition and no recursion.

    The pivot is chosen at random, so no fixed input (sorted, reversed,
    organ-pipe) can force the O(n^2) case. Partitioning splits the range into
    ``< pivot``, ``== pivot`` and ``> pivot`` (Dijkstra's Dutch national flag),
    so runs of equal keys are finished in one pass. Of the two remaining parts,
    the larger is pushed on the stack and the smaller processed next, which
    bounds the stack at O(log n).
    """
    stack = [(0, len(a))]
    while stack:
        lo, hi = stack.pop()
        while hi - lo > _INSERTION_CUTOFF:
            pivot = a[rng.randrange(lo, hi)]
            lt, i, gt = lo, lo, hi  # a[lo:lt] < p, a[lt:i] == p, a[gt:hi] > p
            while i < gt:
                if less(a[i], pivot):
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                elif less(pivot, a[i]):
                    gt -= 1
                    a[gt], a[i] = a[i], a[gt]
                else:
                    i += 1
            if lt - lo < hi - gt:
                stack.append((gt, hi))
                hi = lt
            else:
                stack.append((lo, lt))
                lo = gt
        _insertion(a, less, lo, hi)


@overload
def quick_sort[T: SupportsLessThan](
    items: Iterable[T],
    *,
    key: None = None,
    reverse: bool = False,
    rng: random.Random | None = None,
) -> list[T]: ...
@overload
def quick_sort[T](
    items: Iterable[T],
    *,
    key: Callable[[T], SupportsLessThan],
    reverse: bool = False,
    rng: random.Random | None = None,
) -> list[T]: ...
def quick_sort(
    items: Iterable[Any],
    *,
    key: Callable[[Any], SupportsLessThan] | None = None,
    reverse: bool = False,
    rng: random.Random | None = None,
) -> list[Any]:
    """Return a sorted list using quick sort. Not stable. O(n log n) expected time.

    Pass ``rng`` (a seeded ``random.Random``) for reproducible pivot choices.

    >>> quick_sort([5, 3, 5, 1, 5], reverse=True)
    [5, 5, 5, 3, 1]
    """
    generator = rng if rng is not None else random.Random()
    return _run(items, key, reverse, lambda a, less: _quick_sort(a, less, generator))


# -- heap sort -----------------------------------------------------------------


def _heap_sort(a: list[Any], less: _Less) -> None:
    """In-place heap sort: build a max-heap, then swap the max to the end n times."""

    def sift_down(i: int, n: int) -> None:
        item = a[i]
        while (child := 2 * i + 1) < n:
            if child + 1 < n and less(a[child], a[child + 1]):
                child += 1
            if not less(item, a[child]):
                break
            a[i] = a[child]
            i = child
        a[i] = item

    n = len(a)
    for i in reversed(range(n // 2)):
        sift_down(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(0, end)


@overload
def heap_sort[T: SupportsLessThan](
    items: Iterable[T], *, key: None = None, reverse: bool = False
) -> list[T]: ...
@overload
def heap_sort[T](
    items: Iterable[T], *, key: Callable[[T], SupportsLessThan], reverse: bool = False
) -> list[T]: ...
def heap_sort(
    items: Iterable[Any],
    *,
    key: Callable[[Any], SupportsLessThan] | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Return a sorted list using heap sort. Not stable. O(n log n) worst case, O(1) extra.

    >>> heap_sort([4, 1, 3, 2])
    [1, 2, 3, 4]
    """
    return _run(items, key, reverse, _heap_sort)


# -- integer sorts -------------------------------------------------------------


def counting_sort(items: Iterable[int]) -> list[int]:
    """Sort integers by counting how many times each value occurs. O(n + k).

    ``k`` is ``max - min + 1``, so this is only sensible when the values sit in a
    narrow range (ages, scores, bytes). Negative numbers are fine.

    >>> counting_sort([3, -1, 2, 3, 0])
    [-1, 0, 2, 3, 3]
    """
    values = list(items)
    if not values:
        return []
    lo, hi = min(values), max(values)
    counts = [0] * (hi - lo + 1)
    for v in values:
        counts[v - lo] += 1
    out: list[int] = []
    for offset, c in enumerate(counts):
        out.extend([lo + offset] * c)
    return out


def _radix_non_negative(values: list[int], base: int) -> list[int]:
    """LSD radix sort: one stable bucket pass per digit, least significant first."""
    if not values:
        return values
    largest = max(values)
    place = 1
    while place <= largest:
        buckets: list[list[int]] = [[] for _ in range(base)]
        for v in values:
            buckets[(v // place) % base].append(v)
        values = [v for bucket in buckets for v in bucket]
        place *= base
    return values


def radix_sort(items: Iterable[int], *, base: int = 256) -> list[int]:
    """Sort integers digit by digit (least significant digit first). O(d (n + base)).

    ``d`` is the number of base-``base`` digits in the largest magnitude. Each
    pass is a stable bucket sort, so after the pass for digit ``i`` the list is
    sorted by its lowest ``i + 1`` digits. Negative numbers are sorted by
    magnitude separately and placed first, reversed.

    >>> radix_sort([170, -45, 75, 90, -802, 24, 2, 66])
    [-802, -45, 2, 24, 66, 75, 90, 170]
    """
    if base < 2:
        raise ValueError("base must be at least 2")
    values = list(items)
    negatives = _radix_non_negative([-v for v in values if v < 0], base)
    positives = _radix_non_negative([v for v in values if v >= 0], base)
    return [-v for v in reversed(negatives)] + positives
