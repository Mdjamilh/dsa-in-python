"""A hash map with open addressing and linear probing.

All entries live in one flat table of slots. To find a key, hash it to a home
slot and walk forward one slot at a time (wrapping at the end) until the key or
an empty slot turns up. An empty slot ends the search, because an insert would
have used it.

Deleting cannot simply empty a slot: that would cut the probe path for any key
that was pushed past it. Deleted slots become *tombstones* instead. A lookup
walks over them; an insert may reuse the first one it passed. Tombstones count
toward the load, and a resize drops them all.

Two details keep the probe sequences short:

- **Fibonacci hashing.** Python hashes small integers to themselves, so keys
  like ``0, 8, 16, ...`` would all land in slot 0 of an 8-slot table. The hash
  is multiplied by ``2**64 / phi`` and the top bits are taken, which spreads
  such patterns across the table.
- **Load factor at most 2/3** (live entries plus tombstones). Linear probing
  degrades sharply as the table fills. The table doubles when the limit is
  reached, and halves when it falls below 1/8 live.

Keys match the way they do in ``dict``: the same object, or equal with equal
hashes. That is why ``nan`` can be looked up by the same ``nan`` object.

==============================  =================================
Operation                       Time
==============================  =================================
get, set, delete, ``in``        O(1) expected, O(n) worst case
iteration                       O(capacity)
resize                          O(n), amortized O(1) per insert
==============================  =================================

Space: O(n). Iteration order is slot order, not insertion order.
There is no defence against adversarial keys chosen to collide.
"""

from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from typing import Final, cast, overload

__all__ = ["HashMap"]

_MIN_CAPACITY: Final = 8
_MASK64: Final = (1 << 64) - 1
_FIBONACCI: Final = 0x9E3779B97F4A7C15  # 2**64 / golden ratio, rounded to odd


class _Tombstone:
    """Marks a deleted slot. There is exactly one instance, ``_DELETED``."""

    __slots__ = ()


_DELETED: Final = _Tombstone()


class _Slot[K, V]:
    __slots__ = ("hash", "key", "value")

    def __init__(self, h: int, key: K, value: V) -> None:
        self.hash = h
        self.key = key
        self.value = value


class HashMap[K: Hashable, V](MutableMapping[K, V]):
    """A ``MutableMapping`` backed by an open-addressing table.

    >>> m = HashMap({"a": 1})
    >>> m["b"] = 2
    >>> del m["a"]
    >>> dict(m), "a" in m, len(m)
    ({'b': 2}, False, 1)
    """

    __slots__ = ("_bits", "_size", "_table", "_used")

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, items: Mapping[K, V]) -> None: ...
    @overload
    def __init__(self, items: Iterable[tuple[K, V]]) -> None: ...
    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        self._reset(_MIN_CAPACITY)
        self.update(items)

    def _reset(self, capacity: int) -> None:
        self._table: list[_Slot[K, V] | _Tombstone | None] = [None] * capacity
        self._bits = capacity.bit_length() - 1
        self._size = 0  # live entries
        self._used = 0  # live entries plus tombstones

    @property
    def capacity(self) -> int:
        """Number of slots in the table (always a power of two)."""
        return len(self._table)

    def _home(self, h: int) -> int:
        return ((h * _FIBONACCI) & _MASK64) >> (64 - self._bits)

    def _find(self, key: K, h: int) -> tuple[int, bool]:
        """Return ``(slot, found)``.

        If the key is present, ``slot`` holds it. Otherwise ``slot`` is where an
        insert should go: the first tombstone on the probe path, or the empty
        slot that ended it. The load limit guarantees an empty slot exists.
        """
        table = self._table
        mask = len(table) - 1
        i = self._home(h)
        first_free = -1
        while True:
            slot = table[i]
            if slot is None:
                return (i if first_free < 0 else first_free), False
            if slot is _DELETED:
                if first_free < 0:
                    first_free = i
            else:
                entry = cast("_Slot[K, V]", slot)
                if entry.hash == h and (entry.key is key or entry.key == key):
                    return i, True
            i = (i + 1) & mask

    def __getitem__(self, key: K) -> V:
        i, found = self._find(key, hash(key))
        if not found:
            raise KeyError(key)
        return cast("_Slot[K, V]", self._table[i]).value

    def __setitem__(self, key: K, value: V) -> None:
        h = hash(key)
        i, found = self._find(key, h)
        if found:
            cast("_Slot[K, V]", self._table[i]).value = value
            return
        reused_tombstone = self._table[i] is _DELETED
        self._table[i] = _Slot(h, key, value)
        self._size += 1
        if not reused_tombstone:
            self._used += 1
            if 3 * self._used > 2 * len(self._table):
                self._resize(self._target_capacity())

    def __delitem__(self, key: K) -> None:
        i, found = self._find(key, hash(key))
        if not found:
            raise KeyError(key)
        self._table[i] = _DELETED
        self._size -= 1
        if len(self._table) > _MIN_CAPACITY and 8 * self._size < len(self._table):
            self._resize(self._target_capacity())

    def _target_capacity(self) -> int:
        """Smallest power of two that keeps the live entries at most half the table."""
        capacity = _MIN_CAPACITY
        while capacity < 2 * self._size:
            capacity *= 2
        return capacity

    def _resize(self, capacity: int) -> None:
        old = self._table
        self._reset(capacity)
        table = self._table
        mask = capacity - 1
        for slot in old:
            if slot is None or slot is _DELETED:
                continue
            entry = cast("_Slot[K, V]", slot)
            # Keys are known to be distinct, so just find the first empty slot.
            i = self._home(entry.hash)
            while table[i] is not None:
                i = (i + 1) & mask
            table[i] = entry
            self._size += 1
        self._used = self._size

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[K]:
        for slot in self._table:
            if slot is not None and slot is not _DELETED:
                yield cast("_Slot[K, V]", slot).key

    def clear(self) -> None:
        """Remove every entry and shrink to the minimum capacity. O(1)."""
        self._reset(_MIN_CAPACITY)

    def probe_lengths(self) -> list[int]:
        """Return, for each key, how many slots a lookup inspects. For tests and teaching."""
        lengths = []
        mask = len(self._table) - 1
        for i, slot in enumerate(self._table):
            if slot is None or slot is _DELETED:
                continue
            home = self._home(cast("_Slot[K, V]", slot).hash)
            lengths.append(((i - home) & mask) + 1)
        return lengths

    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"HashMap({{{body}}})"
