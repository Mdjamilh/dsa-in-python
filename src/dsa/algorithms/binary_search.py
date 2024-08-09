"""Binary search and its variants.

All of these keep a half-open window ``[lo, hi)`` that is known to contain the
answer, and halve it each step. Writing every variant as "find the first
position where a condition becomes true" avoids the classic off-by-one bugs:

- ``lower_bound(a, x)``: first index ``i`` with ``a[i] >= x`` (``bisect_left``).
- ``upper_bound(a, x)``: first index ``i`` with ``a[i] > x`` (``bisect_right``).
- ``first_true(lo, hi, pred)``: first integer in ``[lo, hi)`` where a
  monotone predicate (false, ..., false, true, ..., true) is true. This is
  "binary search on the answer", for problems like "smallest capacity that
  ships everything in D days".

The sequence must be sorted (by ``key``, if given). Like ``bisect``, ``key`` is
applied to the elements, not to ``x``.

==================================  ===============================  =====
Function                            Time                             Space
==================================  ===============================  =====
``lower_bound``, ``upper_bound``    O(log n)                         O(1)
``binary_search``, ``count_equal``  O(log n)                         O(1)
``first_true``                      O(log(hi - lo)) predicate calls  O(1)
``search_rotated``                  O(log n)                         O(1)
==================================  ===============================  =====
"""

from collections.abc import Callable, Sequence
from typing import Any, overload

from dsa._typing import SupportsLessThan

__all__ = [
    "binary_search",
    "count_equal",
    "first_true",
    "lower_bound",
    "search_rotated",
    "upper_bound",
]


def first_true(lo: int, hi: int, predicate: Callable[[int], bool]) -> int:
    """Return the smallest ``i`` in ``[lo, hi)`` with ``predicate(i)`` true, or ``hi`` if none.

    ``predicate`` must be monotone over the range: once true, true from then on.

    >>> first_true(0, 100, lambda i: i * i >= 50)
    8
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid  # mid might be the answer; keep it in the window
        else:
            lo = mid + 1  # mid is false, so the answer is to the right
    return lo


@overload
def lower_bound[T: SupportsLessThan](
    a: Sequence[T], x: T, *, key: None = None, lo: int = 0, hi: int | None = None
) -> int: ...
@overload
def lower_bound[T, K: SupportsLessThan](
    a: Sequence[T], x: K, *, key: Callable[[T], K], lo: int = 0, hi: int | None = None
) -> int: ...
def lower_bound(
    a: Sequence[Any],
    x: Any,
    *,
    key: Callable[[Any], Any] | None = None,
    lo: int = 0,
    hi: int | None = None,
) -> int:
    """Return the first index ``i`` in ``[lo, hi)`` with ``a[i] >= x``. Same as ``bisect_left``.

    >>> lower_bound([1, 2, 2, 2, 5], 2), lower_bound([1, 2, 5], 3)
    (1, 2)
    """
    end = len(a) if hi is None else hi
    if key is None:
        return first_true(lo, end, lambda i: not a[i] < x)
    return first_true(lo, end, lambda i: not key(a[i]) < x)


@overload
def upper_bound[T: SupportsLessThan](
    a: Sequence[T], x: T, *, key: None = None, lo: int = 0, hi: int | None = None
) -> int: ...
@overload
def upper_bound[T, K: SupportsLessThan](
    a: Sequence[T], x: K, *, key: Callable[[T], K], lo: int = 0, hi: int | None = None
) -> int: ...
def upper_bound(
    a: Sequence[Any],
    x: Any,
    *,
    key: Callable[[Any], Any] | None = None,
    lo: int = 0,
    hi: int | None = None,
) -> int:
    """Return the first index ``i`` in ``[lo, hi)`` with ``a[i] > x``. Same as ``bisect_right``.

    >>> upper_bound([1, 2, 2, 2, 5], 2)
    4
    """
    end = len(a) if hi is None else hi
    if key is None:
        return first_true(lo, end, lambda i: x < a[i])
    return first_true(lo, end, lambda i: x < key(a[i]))


def binary_search[T: SupportsLessThan](a: Sequence[T], x: T) -> int | None:
    """Return an index of ``x`` in sorted ``a`` (the first, if repeated), or ``None``.

    >>> binary_search([1, 3, 5, 7], 5), binary_search([1, 3, 5, 7], 4)
    (2, None)
    """
    i = lower_bound(a, x)
    if i < len(a) and not x < a[i]:
        return i
    return None


def count_equal[T: SupportsLessThan](a: Sequence[T], x: T) -> int:
    """Return how many elements of sorted ``a`` equal ``x``, with two searches.

    >>> count_equal([1, 2, 2, 2, 5], 2)
    3
    """
    return upper_bound(a, x) - lower_bound(a, x)


def search_rotated[T: SupportsLessThan](a: Sequence[T], x: T) -> int | None:
    """Find ``x`` in a rotated sorted sequence of distinct values, like ``[4, 5, 1, 2, 3]``.

    At every step at least one half of the window is in sorted order. If ``x``
    lies inside that half's range, search it, otherwise search the other half.
    Returns an index or ``None``. Distinct values are required: with
    duplicates, the sorted half cannot always be identified in O(1).

    >>> search_rotated([4, 5, 6, 1, 2, 3], 2), search_rotated([4, 5, 6, 1, 2, 3], 7)
    (4, None)
    """
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if not (a[mid] < x or x < a[mid]):
            return mid
        if not a[mid] < a[lo]:  # a[lo..mid] is sorted
            if not x < a[lo] and x < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        elif a[mid] < x and not a[hi] < x:  # a[mid..hi] is sorted and holds x
            lo = mid + 1
        else:
            hi = mid - 1
    return None
