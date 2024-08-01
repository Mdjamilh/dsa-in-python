import heapq

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from dsa.structures import BinaryHeap, PriorityQueue


def _is_heap(items: list[int]) -> bool:
    return all(not items[i] < items[(i - 1) // 2] for i in range(1, len(items)))


def test_pop_order_and_peek() -> None:
    h = BinaryHeap([5, 3, 8, 1])
    assert h.peek() == 1
    h.push(0)
    assert [h.pop() for _ in range(5)] == [0, 1, 3, 5, 8]
    assert not h


def test_empty_errors_and_repr() -> None:
    h: BinaryHeap[int] = BinaryHeap()
    with pytest.raises(IndexError):
        h.pop()
    with pytest.raises(IndexError):
        h.peek()
    with pytest.raises(IndexError):
        h.replace(1)
    assert h.pushpop(7) == 7  # empty heap: the item comes straight back
    assert repr(BinaryHeap([2, 1])) == "BinaryHeap([1, 2])"


def test_pushpop_and_replace() -> None:
    h = BinaryHeap([3, 5])
    assert h.pushpop(1) == 1  # smaller than the min: heap untouched
    assert list(h) == [3, 5]
    assert h.pushpop(4) == 3
    assert sorted(h) == [4, 5]
    assert h.replace(0) == 4  # replace pops first, so 0 goes in after
    assert h.pop() == 0


def test_heapify_does_linear_work() -> None:
    comparisons = 0

    class Counted:
        def __init__(self, v: int) -> None:
            self.v = v

        def __lt__(self, other: "Counted") -> bool:
            nonlocal comparisons
            comparisons += 1
            return self.v < other.v

    n = 4096
    BinaryHeap(Counted(v) for v in range(n, 0, -1))  # reversed input: worst case for a min-heap
    assert comparisons < 2 * n  # bound is about 2n; n pushes would need about n log n


@given(st.lists(st.integers()))
def test_heapify_gives_a_valid_heap(items: list[int]) -> None:
    assert _is_heap(list(BinaryHeap(items)))


@given(st.lists(st.floats(allow_nan=False)))
def test_heap_sorts_like_sorted(items: list[float]) -> None:
    h = BinaryHeap(items)
    assert [h.pop() for _ in items] == sorted(items)


class HeapVsHeapq(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.h: BinaryHeap[int] = BinaryHeap()
        self.model: list[int] = []

    @rule(v=st.integers(-100, 100))
    def push(self, v: int) -> None:
        self.h.push(v)
        heapq.heappush(self.model, v)

    @precondition(lambda self: self.model)
    @rule()
    def pop(self) -> None:
        assert self.h.pop() == heapq.heappop(self.model)

    @rule(v=st.integers(-100, 100))
    def pushpop(self, v: int) -> None:
        assert self.h.pushpop(v) == heapq.heappushpop(self.model, v)

    @precondition(lambda self: self.model)
    @rule(v=st.integers(-100, 100))
    def replace(self, v: int) -> None:
        assert self.h.replace(v) == heapq.heapreplace(self.model, v)

    @invariant()
    def same_multiset_and_heap_shape(self) -> None:
        assert sorted(self.h) == sorted(self.model)
        assert _is_heap(list(self.h))
        if self.model:
            assert self.h.peek() == self.model[0]


TestHeapVsHeapq = HeapVsHeapq.TestCase

# -- priority queue -----------------------------------------------------------


def test_priority_queue_ties_are_fifo() -> None:
    q: PriorityQueue[str, int] = PriorityQueue()
    for name, prio in [("a", 2), ("b", 1), ("c", 2), ("d", 1), ("e", 2)]:
        q.push(name, prio)
    assert q.peek() == "b"
    assert [q.pop() for _ in range(5)] == ["b", "d", "a", "c", "e"]


def test_priority_queue_items_need_not_be_comparable() -> None:
    q: PriorityQueue[dict[str, int], int] = PriorityQueue()
    q.push({"x": 1}, 1)
    q.push({"x": 2}, 1)
    assert q.pop_with_priority() == ({"x": 1}, 1)


def test_priority_queue_key_function() -> None:
    q = PriorityQueue(["ccc", "a", "bb", "b"], key=len)
    q.push("dddd")
    q.push("zz", 0)  # an explicit priority overrides the key
    assert [q.pop() for _ in range(len(q))] == ["zz", "a", "b", "bb", "ccc", "dddd"]


def test_priority_queue_errors_and_repr() -> None:
    q: PriorityQueue[str, int] = PriorityQueue()
    with pytest.raises(TypeError, match="needs a priority"):
        q.push("x")
    with pytest.raises(TypeError, match="key function"):
        PriorityQueue(["x"])
    for op in (q.pop, q.peek, q.pop_with_priority):
        with pytest.raises(IndexError):
            op()
    q.push("x", 1)
    assert repr(q) == "PriorityQueue(size=1)"
    assert len(q) == 1
    assert q


@given(st.lists(st.tuples(st.integers(0, 5), st.text(max_size=3))))
def test_priority_queue_matches_stable_sort(pairs: list[tuple[int, str]]) -> None:
    q: PriorityQueue[tuple[int, str], int] = PriorityQueue()
    for i, (prio, label) in enumerate(pairs):
        q.push((i, label), prio)
    out = [q.pop() for _ in pairs]
    # sorted() is stable, so sorting indices by priority gives FIFO order on ties.
    expected = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    assert [i for i, _ in out] == expected
