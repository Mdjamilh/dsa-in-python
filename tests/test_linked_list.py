from collections import deque

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from dsa.structures import DoublyLinkedList, Node, SinglyLinkedList

# -- singly linked ------------------------------------------------------------


def test_singly_basic_operations() -> None:
    xs = SinglyLinkedList([2, 3])
    xs.push_front(1)
    xs.push_back(4)
    assert list(xs) == [1, 2, 3, 4]
    assert (xs.peek_front(), xs.peek_back()) == (1, 4)
    assert xs.pop_front() == 1
    assert xs.pop_back() == 4
    assert list(xs) == [2, 3]
    assert len(xs) == 2
    assert repr(xs) == "SinglyLinkedList([2, 3])"


def test_singly_empty_errors() -> None:
    xs: SinglyLinkedList[int] = SinglyLinkedList()
    for op in (xs.pop_front, xs.pop_back, xs.peek_front, xs.peek_back):
        with pytest.raises(IndexError):
            op()
    with pytest.raises(ValueError, match="not in list"):
        xs.remove(1)


def test_singly_pop_back_down_to_empty_resets_tail() -> None:
    xs = SinglyLinkedList([1, 2])
    assert xs.pop_back() == 2
    assert xs.pop_back() == 1
    xs.push_back(3)
    assert list(xs) == [3]
    assert xs.peek_front() == xs.peek_back() == 3


def test_singly_remove_updates_tail() -> None:
    xs = SinglyLinkedList([1, 2, 3])
    xs.remove(3)
    xs.push_back(4)
    xs.remove(1)
    assert list(xs) == [2, 4]
    assert xs.peek_back() == 4


def test_singly_reverse_keeps_head_and_tail_right() -> None:
    xs = SinglyLinkedList([1, 2, 3])
    xs.reverse()
    assert list(xs) == [3, 2, 1]
    xs.push_back(0)
    xs.push_front(4)
    assert list(xs) == [4, 3, 2, 1, 0]
    empty: SinglyLinkedList[int] = SinglyLinkedList()
    empty.reverse()
    assert list(empty) == []


def test_singly_clear() -> None:
    xs = SinglyLinkedList([1, 2])
    xs.clear()
    assert len(xs) == 0
    xs.push_back(5)
    assert list(xs) == [5]


class SinglyVsDeque(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.xs: SinglyLinkedList[int] = SinglyLinkedList()
        self.model: deque[int] = deque()

    @rule(v=st.integers(0, 9))
    def push_front(self, v: int) -> None:
        self.xs.push_front(v)
        self.model.appendleft(v)

    @rule(v=st.integers(0, 9))
    def push_back(self, v: int) -> None:
        self.xs.push_back(v)
        self.model.append(v)

    @precondition(lambda self: self.model)
    @rule()
    def pop_front(self) -> None:
        assert self.xs.pop_front() == self.model.popleft()

    @precondition(lambda self: self.model)
    @rule()
    def pop_back(self) -> None:
        assert self.xs.pop_back() == self.model.pop()

    @rule(v=st.integers(0, 9))
    def remove(self, v: int) -> None:
        if v in self.model:
            self.xs.remove(v)
            self.model.remove(v)

    @rule()
    def reverse(self) -> None:
        self.xs.reverse()
        self.model.reverse()

    @invariant()
    def agree(self) -> None:
        assert list(self.xs) == list(self.model)
        assert len(self.xs) == len(self.model)
        if self.model:
            assert self.xs.peek_front() == self.model[0]
            assert self.xs.peek_back() == self.model[-1]


TestSinglyVsDeque = SinglyVsDeque.TestCase

# -- doubly linked ------------------------------------------------------------


def test_doubly_push_pop_both_ends() -> None:
    xs = DoublyLinkedList([2])
    xs.push_front(1)
    xs.push_back(3)
    assert list(xs) == [1, 2, 3]
    assert list(reversed(xs)) == [3, 2, 1]
    assert xs.pop_front() == 1
    assert xs.pop_back() == 3
    assert list(xs) == [2]
    assert repr(xs) == "DoublyLinkedList([2])"


def test_doubly_node_handles() -> None:
    xs: DoublyLinkedList[str] = DoublyLinkedList()
    a = xs.push_back("a")
    b = xs.push_back("b")
    c = xs.push_back("c")
    assert xs.first_node() is a
    assert xs.last_node() is c
    xs.move_to_front(c)
    assert "".join(xs) == "cab"
    xs.move_to_back(c)
    assert "".join(xs) == "abc"
    xs.move_to_front(a)  # already first: no change
    xs.move_to_back(c)  # already last: no change
    assert "".join(xs) == "abc"
    assert xs.remove(b) == "b"
    xs.insert_after(a, "B")
    assert "".join(xs) == "aBc"
    assert repr(a) == "Node('a')"


def test_doubly_rejects_foreign_or_removed_nodes() -> None:
    xs = DoublyLinkedList([1])
    ys = DoublyLinkedList([2])
    node = ys.first_node()
    with pytest.raises(ValueError, match="does not belong"):
        xs.remove(node)
    ys.remove(node)
    with pytest.raises(ValueError, match="does not belong"):
        ys.remove(node)
    with pytest.raises(ValueError, match="does not belong"):
        xs.move_to_front(Node(3))


def test_doubly_empty_errors() -> None:
    xs: DoublyLinkedList[int] = DoublyLinkedList()
    for op in (xs.pop_front, xs.pop_back, xs.first_node, xs.last_node):
        with pytest.raises(IndexError):
            op()


def test_doubly_clear_detaches_nodes_and_copy_is_independent() -> None:
    xs = DoublyLinkedList([1, 2, 3])
    first = xs.first_node()
    ys = xs.copy()
    xs.clear()
    assert len(xs) == 0
    assert list(xs) == []
    assert list(ys) == [1, 2, 3]
    with pytest.raises(ValueError, match="does not belong"):
        xs.remove(first)


def test_doubly_nodes_allows_removal_while_iterating() -> None:
    xs = DoublyLinkedList(range(10))
    for node in xs.nodes():
        if node.value % 2:
            xs.remove(node)
    assert list(xs) == [0, 2, 4, 6, 8]


class DoublyVsList(RuleBasedStateMachine):
    """Keep a list of (value, node) pairs as the model, so node operations can be checked."""

    def __init__(self) -> None:
        super().__init__()
        self.xs: DoublyLinkedList[int] = DoublyLinkedList()
        self.model: list[tuple[int, Node[int]]] = []
        self.counter = 0

    def _fresh(self) -> int:
        self.counter += 1
        return self.counter

    @rule()
    def push_front(self) -> None:
        v = self._fresh()
        self.model.insert(0, (v, self.xs.push_front(v)))

    @rule()
    def push_back(self) -> None:
        v = self._fresh()
        self.model.append((v, self.xs.push_back(v)))

    @precondition(lambda self: self.model)
    @rule(data=st.data())
    def insert_after(self, data: st.DataObject) -> None:
        i = data.draw(st.integers(0, len(self.model) - 1))
        v = self._fresh()
        self.model.insert(i + 1, (v, self.xs.insert_after(self.model[i][1], v)))

    @precondition(lambda self: self.model)
    @rule(data=st.data())
    def remove(self, data: st.DataObject) -> None:
        i = data.draw(st.integers(0, len(self.model) - 1))
        v, node = self.model.pop(i)
        assert self.xs.remove(node) == v

    @precondition(lambda self: self.model)
    @rule(data=st.data(), front=st.booleans())
    def move(self, data: st.DataObject, front: bool) -> None:
        i = data.draw(st.integers(0, len(self.model) - 1))
        entry = self.model.pop(i)
        if front:
            self.xs.move_to_front(entry[1])
            self.model.insert(0, entry)
        else:
            self.xs.move_to_back(entry[1])
            self.model.append(entry)

    @precondition(lambda self: self.model)
    @rule(front=st.booleans())
    def pop(self, front: bool) -> None:
        if front:
            assert self.xs.pop_front() == self.model.pop(0)[0]
        else:
            assert self.xs.pop_back() == self.model.pop()[0]

    @invariant()
    def links_agree_both_ways(self) -> None:
        values = [v for v, _ in self.model]
        assert list(self.xs) == values
        assert list(reversed(self.xs)) == values[::-1]
        assert len(self.xs) == len(values)
        assert list(self.xs.nodes()) == [n for _, n in self.model]


TestDoublyVsList = DoublyVsList.TestCase
