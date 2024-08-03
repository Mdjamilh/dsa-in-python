import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from dsa.structures import HashMap


class Collides:
    """Every instance has the same hash, so every lookup is a probe walk."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return 42

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Collides) and other.label == self.label

    def __repr__(self) -> str:
        return f"Collides({self.label!r})"


def test_basic_mapping_behaviour() -> None:
    m: HashMap[str, int] = HashMap()
    m["a"] = 1
    m["b"] = 2
    m["a"] = 10
    assert m["a"] == 10
    assert len(m) == 2
    assert "b" in m
    assert "z" not in m
    assert m.get("z") is None
    del m["a"]
    assert dict(m) == {"b": 2}
    with pytest.raises(KeyError):
        m["a"]
    with pytest.raises(KeyError):
        del m["a"]


def test_constructors_and_repr() -> None:
    assert dict(HashMap({"x": 1})) == {"x": 1}
    assert dict(HashMap([("x", 1), ("y", 2)])) == {"x": 1, "y": 2}
    assert repr(HashMap({"x": 1})) == "HashMap({'x': 1})"
    assert HashMap({"x": 1}) == {"x": 1}  # Mapping equality compares items


def test_grows_at_two_thirds_and_keeps_power_of_two() -> None:
    m: HashMap[int, int] = HashMap()
    assert m.capacity == 8
    for i in range(5):
        m[i] = i
    assert m.capacity == 8  # 5 of 8 is under 2/3
    m[5] = 5  # 6 of 8 is over 2/3
    assert m.capacity == 16
    assert dict(m) == {i: i for i in range(6)}


def test_shrinks_when_sparse() -> None:
    m = HashMap((i, i) for i in range(1000))
    big = m.capacity
    for i in range(990):
        del m[i]
    assert m.capacity < big
    assert dict(m) == {i: i for i in range(990, 1000)}
    m.clear()
    assert m.capacity == 8
    assert len(m) == 0


def test_tombstones_are_reused_and_do_not_break_probing() -> None:
    a, b, c = Collides("a"), Collides("b"), Collides("c")
    m = HashMap({a: 1, b: 2, c: 3})
    del m[b]  # leaves a tombstone between a and c on the probe path
    assert m[c] == 3  # lookup must walk past the tombstone
    capacity = m.capacity
    m[Collides("d")] = 4  # reuses the tombstone
    assert m.capacity == capacity
    assert sorted(k.label for k in m) == ["a", "c", "d"]


def test_churn_with_tombstones_triggers_cleanup_not_growth() -> None:
    m: HashMap[int, int] = HashMap()
    for i in range(10_000):
        m[i] = i
        del m[i]
    assert m.capacity == 8
    assert len(m) == 0


def test_nan_key_follows_dict_identity_rule() -> None:
    nan = math.nan
    m = HashMap({nan: "same object"})
    assert m[nan] == "same object"
    other_nan = float("nan")  # a different nan object, and nan != nan
    assert other_nan not in m
    assert (other_nan in {nan: 1}) == (other_nan in m)


def test_sequential_ints_spread_out() -> None:
    m = HashMap((i * 1024, i) for i in range(500))  # identity hashes that share low bits
    lengths = m.probe_lengths()
    assert len(lengths) == 500
    assert sum(lengths) / len(lengths) < 3


def test_mapping_mixins() -> None:
    m = HashMap({"a": 1})
    assert m.setdefault("b", 2) == 2
    assert m.pop("a") == 1
    assert m.popitem() == ("b", 2)
    m.update(c=3)
    assert list(m.items()) == [("c", 3)]


keys = st.one_of(
    st.integers(-20, 20),
    st.sampled_from(["", "a", "b", "ab", "ba"]),
    st.sampled_from([Collides("p"), Collides("q"), Collides("r")]),
    st.just(math.nan),  # always the same object, so it can be found
    st.sampled_from([0.0, -0.0, 1.0, True, False]),  # 0 == 0.0 == -0.0 == False in dict
)


class HashMapVsDict(RuleBasedStateMachine):
    stored = Bundle("stored")

    def __init__(self) -> None:
        super().__init__()
        self.m: HashMap[object, int] = HashMap()
        self.model: dict[object, int] = {}

    @rule(target=stored, k=keys, v=st.integers())
    def set(self, k: object, v: int) -> object:
        self.m[k] = v
        self.model[k] = v
        return k

    @rule(k=st.one_of(stored, keys))
    def delete(self, k: object) -> None:
        if k in self.model:
            del self.m[k]
            del self.model[k]
        else:
            with pytest.raises(KeyError):
                del self.m[k]

    @rule(k=st.one_of(stored, keys))
    def get(self, k: object) -> None:
        assert self.m.get(k, -1) == self.model.get(k, -1)

    @invariant()
    def same_items(self) -> None:
        assert len(self.m) == len(self.model)
        assert dict(self.m.items()) == self.model

    @invariant()
    def load_factor_bounded(self) -> None:
        assert 3 * len(self.m) <= 2 * self.m.capacity


TestHashMapVsDict = HashMapVsDict.TestCase


@given(st.dictionaries(st.text(max_size=5), st.integers()))
def test_round_trips_any_str_dict(d: dict[str, int]) -> None:
    m = HashMap(d)
    assert dict(m) == d
    assert all(m[k] == v for k, v in d.items())
