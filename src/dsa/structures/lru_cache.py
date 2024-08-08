"""A least-recently-used cache from a dict and a doubly linked list.

The dict maps each key to its node in a ``DoublyLinkedList``. The list holds
entries in recency order, most recent at the front. A hit moves the node to
the front; an insert past capacity evicts the node at the back. Every step is
a dict lookup or a constant number of pointer changes.

(``functools.lru_cache`` does the same thing in C for memoising functions.
This class is the general key-value version, with the mechanism in view.)

========================================  ======
Operation                                 Time
========================================  ======
``get``, ``put``, ``peek``, ``delete``    O(1)
========================================  ======

Space: O(capacity).
"""

from collections.abc import Callable, Hashable, Iterator

from dsa.structures.linked_list import DoublyLinkedList, Node

__all__ = ["LRUCache"]


class LRUCache[K: Hashable, V]:
    """A fixed-capacity mapping that evicts the least recently used entry.

    ``get`` and ``put`` count as uses; ``peek`` and ``in`` do not.

    >>> cache: LRUCache[str, int] = LRUCache(2)
    >>> cache.put("a", 1)
    >>> cache.put("b", 2)
    >>> cache.get("a")
    1
    >>> cache.put("c", 3)  # evicts "b", the least recently used
    >>> list(cache)
    ['c', 'a']
    """

    __slots__ = ("_capacity", "_nodes", "_on_evict", "_order")

    def __init__(self, capacity: int, *, on_evict: Callable[[K, V], None] | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._on_evict = on_evict
        self._nodes: dict[K, Node[tuple[K, V]]] = {}
        self._order: DoublyLinkedList[tuple[K, V]] = DoublyLinkedList()

    @property
    def capacity(self) -> int:
        """Maximum number of entries."""
        return self._capacity

    def get[D](self, key: K, default: D | None = None) -> V | D | None:
        """Return the value for ``key`` and mark it most recent, or ``default``."""
        node = self._nodes.get(key)
        if node is None:
            return default
        self._order.move_to_front(node)
        return node.value[1]

    def peek[D](self, key: K, default: D | None = None) -> V | D | None:
        """Return the value for ``key`` without changing its recency."""
        node = self._nodes.get(key)
        return default if node is None else node.value[1]

    def put(self, key: K, value: V) -> None:
        """Insert or update ``key``, mark it most recent, and evict if over capacity."""
        node = self._nodes.get(key)
        if node is not None:
            node.value = (key, value)
            self._order.move_to_front(node)
            return
        self._nodes[key] = self._order.push_front((key, value))
        if len(self._nodes) > self._capacity:
            old_key, old_value = self._order.pop_back()
            del self._nodes[old_key]
            if self._on_evict is not None:
                self._on_evict(old_key, old_value)

    def delete(self, key: K) -> bool:
        """Remove ``key``. Return whether it was present."""
        node = self._nodes.pop(key, None)
        if node is None:
            return False
        self._order.remove(node)
        return True

    def __contains__(self, key: object) -> bool:
        return key in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[K]:
        """Iterate keys from most to least recently used."""
        for key, _ in self._order:
            yield key

    def items(self) -> Iterator[tuple[K, V]]:
        """Iterate ``(key, value)`` pairs from most to least recently used."""
        return iter(self._order)

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self._capacity}, items={list(self.items())!r})"
