"""A double-ended queue and a FIFO queue on a ring buffer.

A ring buffer is a fixed block of slots with a ``head`` index and a size. The
element at logical position ``i`` lives in slot ``(head + i) % capacity``, so
the live region can wrap around the end of the block. Adding at the front just
moves ``head`` back one slot, and nothing is ever shifted. When the block is
full, the elements are copied, unwrapped, into a block twice the size.

Capacity is kept at a power of two so that ``% capacity`` can be written as
``& (capacity - 1)``. In CPython that is a minor speed-up; the main point is
that it is the usual trick, and worth seeing once.

``Deque`` operations:

====================================  ===============
Operation                             Time
====================================  ===============
``append``, ``appendleft``            O(1) amortized
``pop``, ``popleft``                  O(1) amortized
``d[i]``, ``d[i] = x``                O(1)
``rotate(k)``                         O(min(k, n - k))
====================================  ===============

The buffer shrinks to half when it falls to a quarter full, for the same reason
as ``DynamicArray``. ``Queue`` is a thin FIFO view over a ``Deque``.

Space: O(n).
"""

from collections.abc import Iterable, Iterator
from typing import cast

__all__ = ["Deque", "Queue"]

_MIN_CAPACITY = 8


class Deque[T]:
    """A double-ended queue with O(1) operations at both ends and O(1) indexing.

    The method names follow ``collections.deque``.

    >>> d = Deque([2, 3])
    >>> d.appendleft(1)
    >>> d.append(4)
    >>> d.rotate(1)
    >>> list(d), d[0], d[-1]
    ([4, 1, 2, 3], 4, 3)
    """

    __slots__ = ("_head", "_size", "_slots")

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._slots: list[T | None] = [None] * _MIN_CAPACITY
        self._head = 0
        self._size = 0
        for item in items:
            self.append(item)

    @property
    def capacity(self) -> int:
        """Number of slots currently allocated (always a power of two)."""
        return len(self._slots)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def _slot(self, i: int) -> int:
        return (self._head + i) & (len(self._slots) - 1)

    def _resize(self, capacity: int) -> None:
        slots: list[T | None] = [None] * capacity
        for i in range(self._size):
            slots[i] = self._slots[self._slot(i)]
        self._slots = slots
        self._head = 0

    def _grow_if_full(self) -> None:
        if self._size == len(self._slots):
            self._resize(2 * len(self._slots))

    def _shrink_if_sparse(self) -> None:
        if len(self._slots) > _MIN_CAPACITY and self._size <= len(self._slots) // 4:
            self._resize(len(self._slots) // 2)

    def append(self, value: T) -> None:
        """Add ``value`` at the back. O(1) amortized."""
        self._grow_if_full()
        self._slots[self._slot(self._size)] = value
        self._size += 1

    def appendleft(self, value: T) -> None:
        """Add ``value`` at the front. O(1) amortized."""
        self._grow_if_full()
        self._head = (self._head - 1) & (len(self._slots) - 1)
        self._slots[self._head] = value
        self._size += 1

    def pop(self) -> T:
        """Remove and return the back value. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("pop from an empty deque")
        self._size -= 1
        slot = self._slot(self._size)
        value = cast("T", self._slots[slot])
        self._slots[slot] = None
        self._shrink_if_sparse()
        return value

    def popleft(self) -> T:
        """Remove and return the front value. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("pop from an empty deque")
        value = cast("T", self._slots[self._head])
        self._slots[self._head] = None
        self._head = self._slot(1)
        self._size -= 1
        self._shrink_if_sparse()
        return value

    def _index(self, index: int) -> int:
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("deque index out of range")
        return self._slot(index)

    def __getitem__(self, index: int) -> T:
        return cast("T", self._slots[self._index(index)])

    def __setitem__(self, index: int, value: T) -> None:
        self._slots[self._index(index)] = value

    def peek(self) -> T:
        """Return the back value. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("peek at an empty deque")
        return self[-1]

    def peekleft(self) -> T:
        """Return the front value. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("peek at an empty deque")
        return self[0]

    def rotate(self, k: int = 1) -> None:
        """Rotate ``k`` steps to the right (negative ``k`` rotates left).

        Like ``collections.deque.rotate``: after ``rotate(1)`` the last element
        is first. It moves whichever is fewer, ``k`` or ``n - k`` elements.
        """
        if self._size <= 1:
            return
        k %= self._size
        if k <= self._size - k:
            for _ in range(k):
                self.appendleft(self.pop())
        else:
            for _ in range(self._size - k):
                self.append(self.popleft())

    def clear(self) -> None:
        """Remove every element and release the storage."""
        self._slots = [None] * _MIN_CAPACITY
        self._head = 0
        self._size = 0

    def __iter__(self) -> Iterator[T]:
        for i in range(self._size):
            yield cast("T", self._slots[self._slot(i)])

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(range(self._size)):
            yield cast("T", self._slots[self._slot(i)])

    def __repr__(self) -> str:
        return f"Deque({list(self)!r})"


class Queue[T]:
    """A first-in, first-out queue.

    >>> q = Queue(["a"])
    >>> q.enqueue("b")
    >>> q.dequeue(), q.peek(), len(q)
    ('a', 'b', 1)
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items: Deque[T] = Deque(items)

    def enqueue(self, value: T) -> None:
        """Add ``value`` at the back. O(1) amortized."""
        self._items.append(value)

    def dequeue(self) -> T:
        """Remove and return the front value. Raises ``IndexError`` when empty. O(1) amortized."""
        if not self._items:
            raise IndexError("dequeue from an empty queue")
        return self._items.popleft()

    def peek(self) -> T:
        """Return the front value. Raises ``IndexError`` when empty. O(1)."""
        if not self._items:
            raise IndexError("peek at an empty queue")
        return self._items[0]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Queue({list(self._items)!r})"
