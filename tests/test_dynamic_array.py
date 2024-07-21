import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from dsa.structures import DynamicArray


def test_starts_empty_with_minimum_capacity() -> None:
    a: DynamicArray[int] = DynamicArray()
    assert len(a) == 0
    assert a.capacity == 4
    assert list(a) == []


def test_doubles_when_full() -> None:
    a = DynamicArray(range(4))
    assert a.capacity == 4
    a.append(4)
    assert a.capacity == 8
    for i in range(5, 9):
        a.append(i)
    assert a.capacity == 16
    assert list(a) == list(range(9))


def test_shrinks_at_a_quarter_not_a_half() -> None:
    a = DynamicArray(range(16))
    assert a.capacity == 16
    while len(a) > 5:
        a.pop()
    assert a.capacity == 16  # 5 of 16 is above a quarter
    a.pop()  # 4 of 16 is a quarter
    assert a.capacity == 8
    assert list(a) == [0, 1, 2, 3]


def test_no_thrashing_at_the_boundary() -> None:
    a = DynamicArray(range(8))
    capacities = set()
    for _ in range(50):
        a.append(0)
        capacities.add(a.capacity)
        a.pop()
        capacities.add(a.capacity)
    assert capacities == {16}


def test_negative_indices_and_errors() -> None:
    a = DynamicArray("abc")
    assert a[-1] == "c"
    assert a[-3] == "a"
    a[-2] = "B"
    assert list(a) == ["a", "B", "c"]
    with pytest.raises(IndexError):
        a[3]
    with pytest.raises(IndexError):
        a[-4]
    with pytest.raises(IndexError):
        DynamicArray[int]().pop()


def test_slices() -> None:
    a = DynamicArray(range(10))
    assert a[2:5] == DynamicArray([2, 3, 4])
    assert list(a[::-3]) == [9, 6, 3, 0]
    a[1:3] = ["x", "y", "z"]  # type: ignore[list-item]
    assert list(a) == [0, "x", "y", "z", 3, 4, 5, 6, 7, 8, 9]
    del a[::2]
    assert list(a) == ["x", "z", 4, 6, 8]


def test_insert_clamps_like_list() -> None:
    a = DynamicArray([1, 2])
    a.insert(100, 3)
    a.insert(-100, 0)
    a.insert(-1, 9)
    assert list(a) == [0, 1, 2, 9, 3]


def test_mutable_sequence_mixins_work() -> None:
    a = DynamicArray([3, 1, 2])
    a.extend([5, 4])
    a.reverse()
    assert list(a) == [4, 5, 2, 1, 3]
    assert a.index(2) == 2
    assert a.count(1) == 1
    assert 5 in a
    a.remove(5)
    assert list(a) == [4, 2, 1, 3]
    a += [7]
    assert list(a) == [4, 2, 1, 3, 7]


def test_equality_repr_and_unhashable() -> None:
    assert DynamicArray([1, 2]) == DynamicArray([1, 2])
    assert DynamicArray([1, 2]) != DynamicArray([1])
    assert DynamicArray([1]) != [1]
    assert repr(DynamicArray([1, 2])) == "DynamicArray([1, 2])"
    with pytest.raises(TypeError):
        hash(DynamicArray[int]())


def test_none_is_a_valid_element() -> None:
    a: DynamicArray[int | None] = DynamicArray([None, 1, None])
    assert a.pop(0) is None
    assert list(a) == [1, None]


@given(st.lists(st.integers()), st.slices(20))
def test_slicing_matches_list(items: list[int], s: slice) -> None:
    assert list(DynamicArray(items)[s]) == items[s]


class DynamicArrayVsList(RuleBasedStateMachine):
    """Drive a DynamicArray and a list with the same random operations."""

    def __init__(self) -> None:
        super().__init__()
        self.array: DynamicArray[int] = DynamicArray()
        self.model: list[int] = []

    @rule(value=st.integers())
    def append(self, value: int) -> None:
        self.array.append(value)
        self.model.append(value)

    @rule(index=st.integers(-30, 30), value=st.integers())
    def insert(self, index: int, value: int) -> None:
        self.array.insert(index, value)
        self.model.insert(index, value)

    @precondition(lambda self: self.model)
    @rule(data=st.data())
    def pop_at(self, data: st.DataObject) -> None:
        i = data.draw(st.integers(-len(self.model), len(self.model) - 1))
        assert self.array.pop(i) == self.model.pop(i)

    @precondition(lambda self: self.model)
    @rule(data=st.data(), value=st.integers())
    def set_item(self, data: st.DataObject, value: int) -> None:
        i = data.draw(st.integers(-len(self.model), len(self.model) - 1))
        self.array[i] = value
        self.model[i] = value

    @rule(s=st.slices(30))
    def delete_slice(self, s: slice) -> None:
        del self.array[s]
        del self.model[s]

    @invariant()
    def same_contents(self) -> None:
        assert list(self.array) == self.model
        assert len(self.array) == len(self.model)

    @invariant()
    def capacity_bounds(self) -> None:
        cap = self.array.capacity
        assert cap >= len(self.model)
        assert cap == 4 or cap <= 4 * max(len(self.model), 1)


TestDynamicArrayVsList = DynamicArrayVsList.TestCase
