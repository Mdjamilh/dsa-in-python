"""A binary min-heap and a stable priority queue built on it.

A binary heap is a complete binary tree stored in a list: the children of the
node at index ``i`` are at ``2i + 1`` and ``2i + 2``, and its parent is at
``(i - 1) // 2``. The heap property says every parent is less than or equal to
its children, so the minimum is always at index 0.

- ``push`` appends at the end and *sifts up*: swap with the parent while smaller.
- ``pop`` moves the last element to the root and *sifts down*: swap with the
  smaller child while larger. Both walk one root-to-leaf path, so O(log n).
- ``heapify`` sifts down every internal node, from the last one back to the
  root. Most nodes are near the bottom and move only a little, and the sum
  works out to O(n), better than n separate pushes at O(n log n).

The sifts here move a "hole" instead of swapping: the moving element is held
aside and each step copies one element, which halves the writes.

``BinaryHeap`` operations:

========================  ===========
Operation                 Time
========================  ===========
``push``                  O(log n)
``pop``, ``pushpop``      O(log n)
``peek``                  O(1)
``BinaryHeap(items)``     O(n)
========================  ===========

``PriorityQueue`` pairs each item with a priority and an insertion counter.
Comparing ``(priority, counter)`` makes ties come out first-in, first-out,
which a bare heap does not promise. The item itself is never compared, so it
does not need to be orderable.

Space: O(n).
"""

from collections.abc import Callable, Iterable, Iterator
from itertools import count

from dsa._typing import SupportsLessThan

__all__ = ["BinaryHeap", "PriorityQueue"]


class BinaryHeap[T: SupportsLessThan]:
    """A min-heap. For a max-heap, push negated numbers or a wrapper with a reversed ``<``.

    >>> h = BinaryHeap([5, 1, 4])
    >>> h.push(2)
    >>> [h.pop() for _ in range(len(h))]
    [1, 2, 4, 5]
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items = list(items)
        for i in reversed(range(len(self._items) // 2)):
            self._sift_down(i)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"BinaryHeap({self._items!r})"

    def __iter__(self) -> Iterator[T]:
        """Iterate in storage order (a valid heap order, not sorted order)."""
        return iter(self._items)

    def peek(self) -> T:
        """Return the smallest item. Raises ``IndexError`` when empty. O(1)."""
        if not self._items:
            raise IndexError("peek at an empty heap")
        return self._items[0]

    def push(self, item: T) -> None:
        """Add ``item``. O(log n)."""
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def pop(self) -> T:
        """Remove and return the smallest item. Raises ``IndexError`` when empty. O(log n)."""
        items = self._items
        if not items:
            raise IndexError("pop from an empty heap")
        last = items.pop()
        if not items:
            return last
        top = items[0]
        items[0] = last
        self._sift_down(0)
        return top

    def pushpop(self, item: T) -> T:
        """Push ``item`` then pop the smallest, in one sift. O(log n).

        If ``item`` is no larger than the current minimum it comes straight back
        and the heap is untouched.
        """
        items = self._items
        if items and items[0] < item:
            item, items[0] = items[0], item
            self._sift_down(0)
        return item

    def replace(self, item: T) -> T:
        """Pop the smallest, then push ``item``. Raises ``IndexError`` when empty. O(log n)."""
        items = self._items
        if not items:
            raise IndexError("replace on an empty heap")
        top = items[0]
        items[0] = item
        self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        items = self._items
        moving = items[i]
        while i > 0:
            parent = (i - 1) >> 1
            if not moving < items[parent]:
                break
            items[i] = items[parent]
            i = parent
        items[i] = moving

    def _sift_down(self, i: int) -> None:
        items = self._items
        n = len(items)
        moving = items[i]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and items[child + 1] < items[child]:
                child += 1
            if not items[child] < moving:
                break
            items[i] = items[child]
            i = child
        items[i] = moving


class _Entry[T, P: SupportsLessThan]:
    """A heap entry that orders by ``(priority, sequence)`` and never compares the item."""

    __slots__ = ("item", "priority", "seq")

    def __init__(self, priority: P, seq: int, item: T) -> None:
        self.priority = priority
        self.seq = seq
        self.item = item

    def __lt__(self, other: "_Entry[T, P]") -> bool:
        if self.priority < other.priority:
            return True
        if other.priority < self.priority:
            return False
        return self.seq < other.seq


class PriorityQueue[T, P: SupportsLessThan]:
    """A min-priority queue where equal priorities come out in insertion order.

    >>> jobs: PriorityQueue[str, int] = PriorityQueue()
    >>> jobs.push("send newsletter", 3)
    >>> jobs.push("page on-call", 1)
    >>> jobs.push("rotate logs", 3)
    >>> [jobs.pop() for _ in range(3)]
    ['page on-call', 'send newsletter', 'rotate logs']

    Pass ``key`` to derive the priority from the item instead:

    >>> words = PriorityQueue(["ccc", "a", "bb"], key=len)
    >>> words.pop()
    'a'
    """

    __slots__ = ("_counter", "_heap", "_key")

    def __init__(self, items: Iterable[T] = (), *, key: Callable[[T], P] | None = None) -> None:
        self._key = key
        self._counter = count()
        entries = []
        for item in items:
            if key is None:
                raise TypeError("initial items need a key function to give them priorities")
            entries.append(_Entry(key(item), next(self._counter), item))
        self._heap: BinaryHeap[_Entry[T, P]] = BinaryHeap(entries)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        return f"PriorityQueue(size={len(self)})"

    def push(self, item: T, priority: P | None = None) -> None:
        """Add ``item`` with ``priority`` (or ``key(item)`` if omitted). O(log n)."""
        if priority is None:
            if self._key is None:
                raise TypeError("push needs a priority when the queue has no key function")
            priority = self._key(item)
        self._heap.push(_Entry(priority, next(self._counter), item))

    def pop(self) -> T:
        """Remove and return the item with the lowest priority. Raises ``IndexError`` when empty."""
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        return self._heap.pop().item

    def pop_with_priority(self) -> tuple[T, P]:
        """Like ``pop``, but also return the priority. O(log n)."""
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        entry = self._heap.pop()
        return entry.item, entry.priority

    def peek(self) -> T:
        """Return the item that ``pop`` would return, without removing it. O(1)."""
        if not self._heap:
            raise IndexError("peek at an empty priority queue")
        return self._heap.peek().item
