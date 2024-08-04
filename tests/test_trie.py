from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from dsa.structures import Trie


def test_add_contains_and_sorted_iteration() -> None:
    t = Trie(["cat", "car", "cart", "dog", "car"])
    assert len(t) == 4
    assert list(t) == ["car", "cart", "cat", "dog"]
    assert "car" in t
    assert "ca" not in t
    assert 3 not in t  # type: ignore[comparison-overlap]
    assert repr(Trie(["b", "a"])) == "Trie(['a', 'b'])"


def test_prefix_queries() -> None:
    t = Trie(["car", "cart", "carbon", "cat", "dog"])
    assert t.count_prefix("car") == 3
    assert t.count_prefix("") == 5
    assert t.count_prefix("x") == 0
    assert t.with_prefix("car") == ["car", "carbon", "cart"]
    assert t.with_prefix("z") == []


def test_discard_prunes_and_keeps_shared_prefixes() -> None:
    t = Trie(["car", "cart"])
    t.discard("car")
    assert list(t) == ["cart"]
    assert t.count_prefix("car") == 1
    t.discard("cart")
    assert len(t) == 0
    assert t._root.children == {}  # nothing left behind
    t.discard("never added")
    assert len(t) == 0


def test_empty_string_is_a_word() -> None:
    t = Trie([""])
    assert "" in t
    assert list(t) == [""]
    assert t.longest_prefix_of("abc") == ""
    t.remove("")
    assert "" not in t


def test_longest_prefix_of() -> None:
    t = Trie(["car", "cart", "c"])
    assert t.longest_prefix_of("cartwheel") == "cart"
    assert t.longest_prefix_of("carp") == "car"
    assert t.longest_prefix_of("cab") == "c"
    assert t.longest_prefix_of("dog") is None


def test_unicode_order_matches_sorted() -> None:
    words = ["éclair", "eclair", "Zebra", "zebra", "日本", "🙂"]
    assert list(Trie(words)) == sorted(words)


def test_set_operations_from_mutable_set() -> None:
    a, b = Trie(["x", "y"]), Trie(["y", "z"])
    assert set(a & b) == {"y"}
    assert set(a | b) == {"x", "y", "z"}
    assert a == {"x", "y"}  # type: ignore[comparison-overlap]


def test_very_long_word_does_not_recurse() -> None:
    word = "a" * 5000
    t = Trie([word])
    assert list(t) == [word]


words = st.text(alphabet="abc", max_size=6)


@given(st.lists(words), st.text(alphabet="abc", max_size=3))
def test_prefix_queries_match_startswith(items: list[str], prefix: str) -> None:
    t = Trie(items)
    expected = sorted({w for w in items if w.startswith(prefix)})
    assert t.with_prefix(prefix) == expected
    assert t.count_prefix(prefix) == len(expected)


@given(st.lists(words), st.text(alphabet="abc", max_size=8))
def test_longest_prefix_matches_brute_force(items: list[str], text: str) -> None:
    t = Trie(items)
    matches = [w for w in set(items) if text.startswith(w)]
    assert t.longest_prefix_of(text) == (max(matches, key=len) if matches else None)


class TrieVsSet(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.t = Trie()
        self.model: set[str] = set()

    @rule(w=words)
    def add(self, w: str) -> None:
        self.t.add(w)
        self.model.add(w)

    @rule(w=words)
    def discard(self, w: str) -> None:
        self.t.discard(w)
        self.model.discard(w)

    @invariant()
    def agree(self) -> None:
        assert list(self.t) == sorted(self.model)
        assert len(self.t) == len(self.model)

    @invariant()
    def counts_are_consistent(self) -> None:
        # Every node's count equals the number of stored words under it.
        stack = [(self.t._root, "")]
        while stack:
            node, prefix = stack.pop()
            assert node.count == sum(w.startswith(prefix) for w in self.model)
            assert node.count > 0 or not prefix  # no dead branches
            stack.extend((child, prefix + ch) for ch, child in node.children.items())


TestTrieVsSet = TrieVsSet.TestCase
