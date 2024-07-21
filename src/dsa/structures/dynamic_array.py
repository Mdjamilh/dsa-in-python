"""A growable array on top of fixed-size storage.

Python's ``list`` is already a dynamic array. This one makes the mechanism
visible: elements live in a fixed-capacity block of slots, and when the block
is full it is replaced by one twice the size and everything is copied across.

Why doubling gives O(1) amortized appends: after growing to capacity ``c``, the
next ``c / 2`` appends are free before another copy. The copies across the whole
life of the array add up to ``1 + 2 + 4 + ... + n < 2n`` element moves, so ``n``
appends cost O(n) in total.

Shrinking happens when the array falls to a quarter full, not half. With a
half-full rule, alternating append and pop at the boundary would copy on every
operation. The gap between "grow at full" and "shrink at a quarter" means at
least ``c / 4`` cheap operations separate any two resizes.

Complexity, with n elements:

======================  ==================
Operation               Time
======================  ==================
``append``, ``pop()``   O(1) amortized
``a[i]``, ``a[i] = x``  O(1)
``insert``, ``pop(i)``  O(n - i)
``del a[i]``            O(n - i)
slicing                 O(k) for k items
======================  ==================

Space: O(n). Capacity is always at most ``4n`` (or the minimum capacity).
"""

from collections.abc import Iterable, Iterator, MutableSequence
from typing import cast, overload

__all__ = ["DynamicArray"]

_MIN_CAPACITY = 4


class DynamicArray[T](MutableSequence[T]):
    """A list-like sequence that grows by doubling and shrinks at a quarter full.

    Supports everything ``MutableSequence`` does, including negative indices and
    slices (slices return a new ``DynamicArray``).

    >>> a = DynamicArray([1, 2, 3])
    >>> a.append(4)
    >>> a[-1], len(a), a.capacity
    (4, 4, 4)
    >>> a.append(5)
    >>> a.capacity
    8
    """

    __slots__ = ("_size", "_slots")

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._slots: list[T | None] = [None] * _MIN_CAPACITY
        self._size = 0
        for item in items:
            self.append(item)

    @property
    def capacity(self) -> int:
        """Number of slots currently allocated."""
        return len(self._slots)

    def __len__(self) -> int:
        return self._size

    def _index(self, index: int) -> int:
        """Normalise a possibly negative index, raising ``IndexError`` if out of range."""
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("DynamicArray index out of range")
        return index

    def _resize(self, capacity: int) -> None:
        """Copy the live elements into a new block of ``capacity`` slots. O(n)."""
        slots: list[T | None] = [None] * capacity
        slots[: self._size] = self._slots[: self._size]
        self._slots = slots

    def _get(self, i: int) -> T:
        # Slots below _size always hold a real element (which may itself be None
        # when T allows it), so the Optional in the storage type does not apply.
        return cast("T", self._slots[i])

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> "DynamicArray[T]": ...
    def __getitem__(self, index: int | slice) -> "T | DynamicArray[T]":
        if isinstance(index, slice):
            return DynamicArray(self._get(i) for i in range(*index.indices(self._size)))
        return self._get(self._index(index))

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            # Slice assignment can change the length. Rebuilding is O(n) and
            # reuses list's well-tested slice semantics, including extended slices.
            items = list(self)
            items[index] = value  # type: ignore[assignment]
            self.clear()
            self.extend(items)
            return
        self._slots[self._index(index)] = value  # type: ignore[assignment]

    @overload
    def __delitem__(self, index: int) -> None: ...
    @overload
    def __delitem__(self, index: slice) -> None: ...
    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            items = list(self)
            del items[index]
            self.clear()
            self.extend(items)
            return
        i = self._index(index)
        self._slots[i : self._size - 1] = self._slots[i + 1 : self._size]
        self._size -= 1
        self._slots[self._size] = None  # drop the reference so it can be collected
        self._maybe_shrink()

    def _maybe_shrink(self) -> None:
        if self._size <= len(self._slots) // 4 and len(self._slots) > _MIN_CAPACITY:
            self._resize(max(_MIN_CAPACITY, len(self._slots) // 2))

    def insert(self, index: int, value: T) -> None:
        """Insert ``value`` before ``index``. O(n - index).

        Like ``list.insert``, an index past either end is clamped, so it never raises.
        """
        if index < 0:
            index = max(0, index + self._size)
        index = min(index, self._size)
        if self._size == len(self._slots):
            self._resize(2 * len(self._slots))
        self._slots[index + 1 : self._size + 1] = self._slots[index : self._size]
        self._slots[index] = value
        self._size += 1

    def append(self, value: T) -> None:
        """Add ``value`` at the end. O(1) amortized."""
        if self._size == len(self._slots):
            self._resize(2 * len(self._slots))
        self._slots[self._size] = value
        self._size += 1

    def pop(self, index: int = -1) -> T:
        """Remove and return the item at ``index`` (default last). O(1) at the end."""
        i = self._index(index)
        value = self._get(i)
        del self[i]
        return value

    def clear(self) -> None:
        """Remove every element and release the storage. O(capacity)."""
        self._slots = [None] * _MIN_CAPACITY
        self._size = 0

    def __iter__(self) -> Iterator[T]:
        for i in range(self._size):
            yield self._get(i)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DynamicArray):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # mutable, like list

    def __repr__(self) -> str:
        return f"DynamicArray({list(self)!r})"
