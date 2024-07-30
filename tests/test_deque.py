from collections import deque

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from dsa.structures import Deque, Queue


def test_wraps_around_without_growing() -> None:
    d = Deque(range(6))
    for _ in range(6):
        d.popleft()
    for i in range(8):  # head is now at slot 6, so these wrap past the end
        d.append(i)
    assert d.capacity == 8
    assert list(d) == list(range(8))
    d.append(8)  # full: grow and unwrap
    assert d.capacity == 16
    assert list(d) == list(range(9))


def test_appendleft_wraps_from_slot_zero() -> None:
    d: Deque[int] = Deque()
    d.appendleft(1)
    d.appendleft(0)
    d.append(2)
    assert list(d) == [0, 1, 2]
    assert list(reversed(d)) == [2, 1, 0]


def test_indexing_and_peeks() -> None:
    d = Deque("abcd")
    assert (d[0], d[-1], d[2]) == ("a", "d", "c")
    d[1] = "B"
    assert (d.peekleft(), d.peek()) == ("a", "d")
    assert "".join(d) == "aBcd"
    with pytest.raises(IndexError):
        d[4]
    with pytest.raises(IndexError):
        d[-5]


def test_empty_errors() -> None:
    d: Deque[int] = Deque()
    for op in (d.pop, d.popleft, d.peek, d.peekleft):
        with pytest.raises(IndexError):
            op()


def test_shrinks_but_not_below_minimum() -> None:
    d = Deque(range(64))
    assert d.capacity == 64
    while len(d) > 1:
        d.pop()
    assert d.capacity == 8
    d.clear()
    assert d.capacity == 8
    assert not d
    assert repr(d) == "Deque([])"


def test_rotate_matches_collections_deque_cases() -> None:
    for k in range(-7, 8):
        ours, theirs = Deque(range(5)), deque(range(5))
        ours.rotate(k)
        theirs.rotate(k)
        assert list(ours) == list(theirs), k
    single = Deque([1])
    single.rotate(3)
    assert list(single) == [1]


def test_queue_fifo() -> None:
    q = Queue([1])
    q.enqueue(2)
    q.enqueue(3)
    assert list(q) == [1, 2, 3]
    assert q.peek() == 1
    assert [q.dequeue(), q.dequeue()] == [1, 2]
    assert len(q) == 1
    assert repr(q) == "Queue([3])"
    q.dequeue()
    assert not q
    with pytest.raises(IndexError):
        q.dequeue()
    with pytest.raises(IndexError):
        q.peek()


@given(st.lists(st.integers()), st.integers(-50, 50))
def test_rotate_matches_collections_deque(items: list[int], k: int) -> None:
    ours, theirs = Deque(items), deque(items)
    ours.rotate(k)
    theirs.rotate(k)
    assert list(ours) == list(theirs)


class DequeVsCollections(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.d: Deque[int] = Deque()
        self.model: deque[int] = deque()

    @rule(v=st.integers(), n=st.integers(1, 20))
    def append_many(self, v: int, n: int) -> None:
        for _ in range(n):
            self.d.append(v)
            self.model.append(v)

    @rule(v=st.integers())
    def appendleft(self, v: int) -> None:
        self.d.appendleft(v)
        self.model.appendleft(v)

    @precondition(lambda self: self.model)
    @rule(n=st.integers(1, 20), left=st.booleans())
    def pop_many(self, n: int, left: bool) -> None:
        for _ in range(min(n, len(self.model))):
            if left:
                assert self.d.popleft() == self.model.popleft()
            else:
                assert self.d.pop() == self.model.pop()

    @rule(k=st.integers(-10, 10))
    def rotate(self, k: int) -> None:
        self.d.rotate(k)
        self.model.rotate(k)

    @precondition(lambda self: self.model)
    @rule(data=st.data(), v=st.integers())
    def set_item(self, data: st.DataObject, v: int) -> None:
        i = data.draw(st.integers(-len(self.model), len(self.model) - 1))
        self.d[i] = v
        self.model[i] = v

    @invariant()
    def agree(self) -> None:
        assert list(self.d) == list(self.model)
        assert list(reversed(self.d)) == list(reversed(self.model))
        assert len(self.d) == len(self.model)

    @invariant()
    def capacity_is_power_of_two_and_bounded(self) -> None:
        cap = self.d.capacity
        assert cap & (cap - 1) == 0
        assert len(self.model) <= cap
        assert cap == 8 or cap <= 4 * len(self.model)


TestDequeVsCollections = DequeVsCollections.TestCase
