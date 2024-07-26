"""Singly and doubly linked lists.

A linked list stores each element in its own node, and each node points at the
next one (and, in a doubly linked list, the previous one). Nothing is ever
copied or shifted, so work at a known position is O(1). The price is that
reaching position ``i`` means walking ``i`` links, and every element carries
one or two extra pointers.

``SinglyLinkedList`` keeps a head and a tail pointer:

======================================  ======
Operation                               Time
======================================  ======
``push_front``, ``push_back``           O(1)
``pop_front``                           O(1)
``pop_back``                            O(n)   (must find the node before the tail)
``reverse``                             O(n), O(1) extra space
``remove`` (first match), ``in``        O(n)
======================================  ======

``DoublyLinkedList`` hands out its ``Node`` objects, so a caller that keeps a
node can unlink or move it in O(1). This is what an LRU cache needs. It uses a
sentinel node, which makes the list circular around a dummy element: every real
node always has a real ``prev`` and ``next``, so insertion and removal have no
special cases for the ends.

======================================  ======
Operation                               Time
======================================  ======
push and pop at either end              O(1)
``remove(node)``, ``move_to_front``     O(1)
``insert_after(node, value)``           O(1)
iteration, ``reversed``                 O(n)
======================================  ======

Space for both: O(n).
"""

from collections.abc import Iterable, Iterator
from typing import Self

__all__ = ["DoublyLinkedList", "Node", "SinglyLinkedList"]


class _SNode[T]:
    __slots__ = ("next", "value")

    def __init__(self, value: T, next: "_SNode[T] | None" = None) -> None:
        self.value = value
        self.next = next


class SinglyLinkedList[T]:
    """A singly linked list with O(1) push at both ends and O(1) pop at the front.

    >>> xs = SinglyLinkedList([2, 3])
    >>> xs.push_front(1)
    >>> xs.reverse()
    >>> list(xs)
    [3, 2, 1]
    """

    __slots__ = ("_head", "_size", "_tail")

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._head: _SNode[T] | None = None
        self._tail: _SNode[T] | None = None
        self._size = 0
        for item in items:
            self.push_back(item)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        node = self._head
        while node is not None:
            yield node.value
            node = node.next

    def __repr__(self) -> str:
        return f"SinglyLinkedList({list(self)!r})"

    def push_front(self, value: T) -> None:
        """Add ``value`` before the first element. O(1)."""
        self._head = _SNode(value, self._head)
        if self._tail is None:
            self._tail = self._head
        self._size += 1

    def push_back(self, value: T) -> None:
        """Add ``value`` after the last element. O(1)."""
        node = _SNode(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def peek_front(self) -> T:
        """Return the first element. Raises ``IndexError`` when empty. O(1)."""
        if self._head is None:
            raise IndexError("peek at an empty list")
        return self._head.value

    def peek_back(self) -> T:
        """Return the last element. Raises ``IndexError`` when empty. O(1)."""
        if self._tail is None:
            raise IndexError("peek at an empty list")
        return self._tail.value

    def pop_front(self) -> T:
        """Remove and return the first element. Raises ``IndexError`` when empty. O(1)."""
        head = self._head
        if head is None:
            raise IndexError("pop from an empty list")
        self._head = head.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return head.value

    def pop_back(self) -> T:
        """Remove and return the last element. Raises ``IndexError`` when empty. O(n)."""
        if self._head is None:
            raise IndexError("pop from an empty list")
        if self._head is self._tail:
            return self.pop_front()
        node = self._head
        while node.next is not self._tail:
            assert node.next is not None
            node = node.next
        assert self._tail is not None
        value = self._tail.value
        node.next = None
        self._tail = node
        self._size -= 1
        return value

    def remove(self, value: T) -> None:
        """Remove the first element equal to ``value``. Raises ``ValueError`` if absent. O(n)."""
        prev: _SNode[T] | None = None
        node = self._head
        while node is not None:
            if node.value == value:
                if prev is None:
                    self._head = node.next
                else:
                    prev.next = node.next
                if node is self._tail:
                    self._tail = prev
                self._size -= 1
                return
            prev, node = node, node.next
        raise ValueError(f"{value!r} not in list")

    def reverse(self) -> None:
        """Reverse the list in place by flipping every ``next`` pointer. O(n) time, O(1) space."""
        prev: _SNode[T] | None = None
        node = self._head
        self._tail = node
        while node is not None:
            node.next, prev, node = prev, node, node.next
        self._head = prev

    def clear(self) -> None:
        """Remove every element. O(1)."""
        self._head = self._tail = None
        self._size = 0


class Node[T]:
    """A node of a ``DoublyLinkedList``. Hold on to it to unlink it in O(1) later."""

    __slots__ = ("next", "owner", "prev", "value")

    def __init__(self, value: T) -> None:
        self.value = value
        self.prev: Node[T] = self
        self.next: Node[T] = self
        # The list that currently contains this node, or None once removed.
        # Checking it stops a node from one list being unlinked through another.
        self.owner: DoublyLinkedList[T] | None = None

    def __repr__(self) -> str:
        return f"Node({self.value!r})"


class DoublyLinkedList[T]:
    """A circular doubly linked list with a sentinel, exposing its nodes.

    >>> xs = DoublyLinkedList("abc")
    >>> b = xs.first_node().next
    >>> xs.move_to_front(b)
    >>> "".join(xs), "".join(reversed(xs))
    ('bac', 'cab')
    """

    __slots__ = ("_sentinel", "_size")

    def __init__(self, items: Iterable[T] = ()) -> None:
        # The sentinel's value is never read. It is typed as T only so that the
        # links between real nodes and the sentinel have one node type.
        self._sentinel: Node[T] = Node.__new__(Node)
        self._sentinel.prev = self._sentinel.next = self._sentinel
        self._sentinel.owner = None
        self._size = 0
        for item in items:
            self.push_back(item)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        node = self._sentinel.next
        while node is not self._sentinel:
            yield node.value
            node = node.next

    def __reversed__(self) -> Iterator[T]:
        node = self._sentinel.prev
        while node is not self._sentinel:
            yield node.value
            node = node.prev

    def __repr__(self) -> str:
        return f"DoublyLinkedList({list(self)!r})"

    def nodes(self) -> Iterator[Node[T]]:
        """Iterate over the nodes from front to back."""
        node = self._sentinel.next
        while node is not self._sentinel:
            nxt = node.next  # read first, so the caller may remove ``node``
            yield node
            node = nxt

    def _link_after(self, anchor: Node[T], node: Node[T]) -> Node[T]:
        node.prev = anchor
        node.next = anchor.next
        anchor.next.prev = node
        anchor.next = node
        node.owner = self
        self._size += 1
        return node

    def _check(self, node: Node[T]) -> None:
        if node.owner is not self:
            raise ValueError("node does not belong to this list")

    def push_front(self, value: T) -> Node[T]:
        """Insert ``value`` at the front and return its node. O(1)."""
        return self._link_after(self._sentinel, Node(value))

    def push_back(self, value: T) -> Node[T]:
        """Insert ``value`` at the back and return its node. O(1)."""
        return self._link_after(self._sentinel.prev, Node(value))

    def insert_after(self, node: Node[T], value: T) -> Node[T]:
        """Insert ``value`` right after ``node`` and return the new node. O(1)."""
        self._check(node)
        return self._link_after(node, Node(value))

    def first_node(self) -> Node[T]:
        """Return the front node. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("empty list has no nodes")
        return self._sentinel.next

    def last_node(self) -> Node[T]:
        """Return the back node. Raises ``IndexError`` when empty."""
        if not self._size:
            raise IndexError("empty list has no nodes")
        return self._sentinel.prev

    def remove(self, node: Node[T]) -> T:
        """Unlink ``node`` and return its value. O(1)."""
        self._check(node)
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = node
        node.owner = None
        self._size -= 1
        return node.value

    def pop_front(self) -> T:
        """Remove and return the front value. Raises ``IndexError`` when empty. O(1)."""
        if not self._size:
            raise IndexError("pop from an empty list")
        return self.remove(self._sentinel.next)

    def pop_back(self) -> T:
        """Remove and return the back value. Raises ``IndexError`` when empty. O(1)."""
        if not self._size:
            raise IndexError("pop from an empty list")
        return self.remove(self._sentinel.prev)

    def move_to_front(self, node: Node[T]) -> None:
        """Move ``node`` to the front without allocating. O(1)."""
        self._check(node)
        if self._sentinel.next is node:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1
        self._link_after(self._sentinel, node)

    def move_to_back(self, node: Node[T]) -> None:
        """Move ``node`` to the back without allocating. O(1)."""
        self._check(node)
        if self._sentinel.prev is node:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1
        self._link_after(self._sentinel.prev, node)

    def clear(self) -> None:
        """Remove every element. O(n), because each node is detached from this list."""
        for node in self.nodes():
            node.prev = node.next = node
            node.owner = None
        self._sentinel.prev = self._sentinel.next = self._sentinel
        self._size = 0

    def copy(self) -> Self:
        """Return a shallow copy with new nodes. O(n)."""
        return type(self)(self)
