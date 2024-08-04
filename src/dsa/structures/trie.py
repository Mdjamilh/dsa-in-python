"""A prefix tree (trie) over strings.

Each node represents a prefix. Its children are keyed by the next character,
and a flag marks whether the prefix is itself a stored word. Looking up a word
of length ``L`` follows ``L`` links, no matter how many words are stored, and
every word that shares a prefix shares the nodes for it.

Each node also counts how many stored words pass through it. That makes
``count_prefix`` O(prefix length) instead of a walk over the whole subtree,
and lets ``discard`` prune nodes that no word uses any more.

Characters are Python ``str`` characters, which are Unicode code points, so
iteration order matches ``sorted()`` on the same words.

==================================  ==============================
Operation                           Time
==================================  ==============================
``add``, ``discard``, ``in``        O(L)
``count_prefix``                    O(P) for a prefix of length P
``with_prefix``                     O(P + size of the output)
``longest_prefix_of(text)``         O(length of the match)
iteration (sorted)                  O(total characters, times log of the alphabet)
==================================  ==============================

Space: O(total characters stored), less when words share prefixes.
"""

from collections.abc import Iterable, Iterator, MutableSet

__all__ = ["Trie"]


class _TrieNode:
    __slots__ = ("children", "count", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.terminal = False
        self.count = 0  # stored words in this subtree, including this node's own


class Trie(MutableSet[str]):
    """A set of strings with fast prefix queries.

    >>> t = Trie(["car", "cart", "cat", "dog"])
    >>> t.with_prefix("car"), t.count_prefix("ca"), "ca" in t
    (['car', 'cart'], 3, False)
    """

    __slots__ = ("_root",)

    def __init__(self, words: Iterable[str] = ()) -> None:
        self._root = _TrieNode()
        for word in words:
            self.add(word)

    def __len__(self) -> int:
        return self._root.count

    def _walk(self, prefix: str) -> _TrieNode | None:
        node = self._root
        for ch in prefix:
            child = node.children.get(ch)
            if child is None:
                return None
            node = child
        return node

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        node = self._walk(word)
        return node is not None and node.terminal

    def add(self, word: str) -> None:
        """Insert ``word``. Adding a word that is already present does nothing. O(L)."""
        if word in self:
            return
        node = self._root
        node.count += 1
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
            node.count += 1
        node.terminal = True

    def discard(self, word: str) -> None:
        """Remove ``word`` if present, pruning nodes no other word needs. O(L)."""
        if word not in self:
            return
        node = self._root
        node.count -= 1
        for ch in word:
            child = node.children[ch]
            if child.count == 1:
                # Only this word passes through here: drop the whole branch.
                del node.children[ch]
                return
            child.count -= 1
            node = child
        node.terminal = False

    def count_prefix(self, prefix: str) -> int:
        """Return how many stored words start with ``prefix``. O(P)."""
        node = self._walk(prefix)
        return 0 if node is None else node.count

    def with_prefix(self, prefix: str) -> list[str]:
        """Return every stored word that starts with ``prefix``, sorted."""
        node = self._walk(prefix)
        return [] if node is None else list(self._words(node, prefix))

    def longest_prefix_of(self, text: str) -> str | None:
        """Return the longest stored word that is a prefix of ``text``, or ``None``.

        This is the lookup a tokenizer or a router does: given ``"cartwheel"``
        and the words ``car`` and ``cart``, the answer is ``"cart"``.
        """
        node = self._root
        best = 0 if node.terminal else -1
        for i, ch in enumerate(text):
            child = node.children.get(ch)
            if child is None:
                break
            node = child
            if node.terminal:
                best = i + 1
        return None if best < 0 else text[:best]

    def _words(self, node: _TrieNode, prefix: str) -> Iterator[str]:
        # An explicit stack, so very long words cannot hit the recursion limit.
        # Children are pushed in reverse sorted order so they pop in sorted order.
        stack = [(node, prefix)]
        while stack:
            current, word = stack.pop()
            if current.terminal:
                yield word
            children = current.children
            stack.extend((children[ch], word + ch) for ch in sorted(children, reverse=True))

    def __iter__(self) -> Iterator[str]:
        return self._words(self._root, "")

    def __repr__(self) -> str:
        return f"Trie({list(self)!r})"
