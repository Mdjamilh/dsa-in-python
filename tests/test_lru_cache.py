import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from dsa.structures import LRUCache


def test_eviction_order_and_callback() -> None:
    evicted: list[tuple[str, int]] = []
    cache: LRUCache[str, int] = LRUCache(2, on_evict=lambda k, v: evicted.append((k, v)))
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a is now most recent
    cache.put("c", 3)
    assert evicted == [("b", 2)]
    assert list(cache) == ["c", "a"]
    cache.put("a", 10)  # update counts as a use, no eviction
    assert list(cache.items()) == [("a", 10), ("c", 3)]
    assert evicted == [("b", 2)]


def test_peek_and_contains_do_not_touch_recency() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.peek("a") == 1
    assert "a" in cache
    cache.put("c", 3)
    assert "a" not in cache  # a was still least recent
    assert cache.peek("a", -1) == -1
    assert cache.get("a", -1) == -1
    assert cache.get("a") is None


def test_delete_and_validation() -> None:
    cache: LRUCache[int, int] = LRUCache(3)
    cache.put(1, 1)
    assert cache.delete(1)
    assert not cache.delete(1)
    assert len(cache) == 0
    assert cache.capacity == 3
    assert repr(cache) == "LRUCache(capacity=3, items=[])"
    with pytest.raises(ValueError, match="at least 1"):
        LRUCache(0)


class ModelLRU:
    """The reference: a list of (key, value), most recent first. O(n) and easy to check by eye."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[tuple[int, int]] = []
        self.evicted: list[tuple[int, int]] = []

    def _take(self, key: int) -> tuple[int, int] | None:
        for i, (k, v) in enumerate(self.entries):
            if k == key:
                del self.entries[i]
                return k, v
        return None

    def get(self, key: int) -> int | None:
        entry = self._take(key)
        if entry is None:
            return None
        self.entries.insert(0, entry)
        return entry[1]

    def put(self, key: int, value: int) -> None:
        self._take(key)
        self.entries.insert(0, (key, value))
        if len(self.entries) > self.capacity:
            self.evicted.append(self.entries.pop())

    def delete(self, key: int) -> bool:
        return self._take(key) is not None


class LRUVsModel(RuleBasedStateMachine):
    @initialize(capacity=st.integers(1, 5))
    def setup(self, capacity: int) -> None:
        self.evicted: list[tuple[int, int]] = []
        self.cache: LRUCache[int, int] = LRUCache(
            capacity, on_evict=lambda k, v: self.evicted.append((k, v))
        )
        self.model = ModelLRU(capacity)

    @rule(k=st.integers(0, 8), v=st.integers())
    def put(self, k: int, v: int) -> None:
        self.cache.put(k, v)
        self.model.put(k, v)

    @rule(k=st.integers(0, 8))
    def get(self, k: int) -> None:
        assert self.cache.get(k) == self.model.get(k)

    @rule(k=st.integers(0, 8))
    def delete(self, k: int) -> None:
        assert self.cache.delete(k) == self.model.delete(k)

    @invariant()
    def same_order_and_evictions(self) -> None:
        if not hasattr(self, "cache"):
            return
        assert list(self.cache.items()) == self.model.entries
        assert self.evicted == self.model.evicted


TestLRUVsModel = LRUVsModel.TestCase
