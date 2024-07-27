"""A LIFO stack.

The top of the stack is the end of a Python ``list``, so ``push`` and ``pop``
are the list's own amortized O(1) ``append`` and ``pop``. The class exists to
give the structure a narrow, intention-revealing interface: you can only touch
the top.

==================  ===============
Operation           Time
==================  ===============
``push``            O(1) amortized
``pop``, ``peek``   O(1)
==================  ===============

Space: O(n).
"""

from collections.abc import Iterable, Iterator

__all__ = ["Stack", "balanced_brackets"]


class Stack[T]:
    """A last-in, first-out stack.

    Iteration goes from the top down, which is the order ``pop`` would return.

    >>> s = Stack([1, 2])
    >>> s.push(3)
    >>> s.pop(), s.peek(), len(s)
    (3, 2, 2)
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items = list(items)

    def push(self, value: T) -> None:
        """Put ``value`` on top. O(1) amortized."""
        self._items.append(value)

    def pop(self) -> T:
        """Remove and return the top value. Raises ``IndexError`` when empty. O(1)."""
        if not self._items:
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def peek(self) -> T:
        """Return the top value without removing it. Raises ``IndexError`` when empty. O(1)."""
        if not self._items:
            raise IndexError("peek at an empty stack")
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[T]:
        return reversed(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items!r})"


_PAIRS = {")": "(", "]": "[", "}": "{"}


def balanced_brackets(text: str) -> bool:
    """Return whether every bracket in ``text`` is closed in the right order. O(n).

    The classic stack exercise: push each opener, and each closer must match the
    opener on top. Characters other than ``()[]{}`` are ignored.

    >>> balanced_brackets("f(a[1], {b: 2})"), balanced_brackets("(]")
    (True, False)
    """
    openers: Stack[str] = Stack()
    for ch in text:
        if ch in "([{":
            openers.push(ch)
        elif ch in _PAIRS and (not openers or openers.pop() != _PAIRS[ch]):
            return False
    return not openers
