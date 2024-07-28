import pytest
from hypothesis import given
from hypothesis import strategies as st

from dsa.structures import Stack, balanced_brackets


def test_lifo_order() -> None:
    s = Stack([1, 2])
    s.push(3)
    assert list(s) == [3, 2, 1]
    assert s.peek() == 3
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]
    assert not s
    assert len(s) == 0


def test_empty_errors_and_repr() -> None:
    s: Stack[int] = Stack()
    with pytest.raises(IndexError):
        s.pop()
    with pytest.raises(IndexError):
        s.peek()
    s.push(1)
    assert repr(s) == "Stack([1])"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("()[]{}", True),
        ("{[()()]}", True),
        ("def f(x): return [x, {1: 2}]", True),
        ("(", False),
        (")", False),
        ("(]", False),
        ("([)]", False),
        ("(()", False),
    ],
)
def test_balanced_brackets(text: str, expected: bool) -> None:
    assert balanced_brackets(text) is expected


def _balanced_reference(text: str) -> bool:
    """Repeatedly delete adjacent matching pairs. Quadratic, but obviously right."""
    s = "".join(ch for ch in text if ch in "()[]{}")
    while True:
        shorter = s.replace("()", "").replace("[]", "").replace("{}", "")
        if shorter == s:
            return not s
        s = shorter


@given(st.text(alphabet="()[]{}a", max_size=30))
def test_balanced_brackets_matches_pair_deletion(text: str) -> None:
    assert balanced_brackets(text) == _balanced_reference(text)


@given(st.lists(st.one_of(st.integers(), st.none())))
def test_push_all_then_pop_all_reverses(items: list[int | None]) -> None:
    s: Stack[int | None] = Stack()
    for x in items:
        s.push(x)
    assert [s.pop() for _ in items] == items[::-1]
