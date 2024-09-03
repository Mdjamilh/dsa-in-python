# dsa-in-python

[![CI](https://github.com/Mdjamilh/dsa-in-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Mdjamilh/dsa-in-python/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25%20enforced-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

Textbook data structures and algorithms in typed Python, written so you can read them.

Every module starts with a docstring that explains how the thing works and what each operation costs in time and space. The code follows that explanation. Each structure is tested with Hypothesis against the standard library (`list`, `dict`, `collections.deque`, `heapq`, `bisect`, `sorted`) or against a slow, obviously correct model. The speed claims come with benchmark numbers from a real machine.

It is a learning reference and a small library with no runtime dependencies. It is not a replacement for `dict`, `list` or `heapq`, and the benchmarks below show why.

- Python 3.12+ generics (`class AVLTree[K, V]`), checked with `mypy --strict`. A `py.typed` marker ships with the package.
- The collections plug into `collections.abc`: `HashMap`, `BinarySearchTree` and `AVLTree` are `MutableMapping`s, `Trie` is a `MutableSet`, `DynamicArray` is a `MutableSequence`. So `dict(tree)`, `tree.items()`, `trie & other` and slicing all work.
- No recursion where the depth depends on the input, except in the AVL tree and merge sort, where depth is O(log n).

## Contents

- [What is inside](#what-is-inside)
- [Install from GitHub](#install-from-github)
- [Using it](#using-it)
- [How the tests work](#how-the-tests-work)
- [Benchmarks](#benchmarks)
- [Design notes and limits](#design-notes-and-limits)
- [Development](#development)

## What is inside

n is the number of elements, h the tree height, V and E the vertices and edges, L a word's length. "am." means amortized.

### Data structures

| Structure | Module | Key operations | Time | Space |
| --- | --- | --- | --- | --- |
| [Dynamic array](src/dsa/structures/dynamic_array.py) | `structures.dynamic_array` | append, pop / get, set / insert, delete at i | O(1) am. / O(1) / O(n) | O(n) |
| [Singly linked list](src/dsa/structures/linked_list.py) | `structures.linked_list` | push_front, push_back, pop_front / pop_back, reverse | O(1) / O(n) | O(n) |
| [Doubly linked list](src/dsa/structures/linked_list.py) | `structures.linked_list` | push and pop at both ends, remove(node), move_to_front | O(1) | O(n) |
| [Stack](src/dsa/structures/stack.py) | `structures.stack` | push, pop, peek | O(1) am. | O(n) |
| [Deque and queue (ring buffer)](src/dsa/structures/deque.py) | `structures.deque` | append, appendleft, pop, popleft / index / rotate(k) | O(1) am. / O(1) / O(min(k, n - k)) | O(n) |
| [Binary heap](src/dsa/structures/heap.py) | `structures.heap` | push, pop, pushpop / peek / heapify | O(log n) / O(1) / O(n) | O(n) |
| [Priority queue (FIFO on ties)](src/dsa/structures/heap.py) | `structures.heap` | push, pop | O(log n) | O(n) |
| [Hash map (open addressing)](src/dsa/structures/hash_map.py) | `structures.hash_map` | get, set, delete, `in` | O(1) expected, O(n) worst | O(n) |
| [Binary search tree](src/dsa/structures/bst.py) | `structures.bst` | get, set, delete, floor, ceiling | O(h): O(log n) average, O(n) worst | O(n) |
| [AVL tree](src/dsa/structures/avl_tree.py) | `structures.avl_tree` | get, set, delete, floor, ceiling, rank, select / range | O(log n) worst / O(log n + k) | O(n) |
| [Trie](src/dsa/structures/trie.py) | `structures.trie` | add, discard, `in` / count_prefix / longest_prefix_of | O(L) / O(prefix length) / O(match length) | O(total characters) |
| [Union-find](src/dsa/structures/union_find.py) | `structures.union_find` | find, union, connected | O(alpha(n)) am. | O(n) |
| [LRU cache](src/dsa/structures/lru_cache.py) | `structures.lru_cache` | get, put, peek, delete, eviction | O(1) | O(capacity) |
| [Graph (adjacency map)](src/dsa/structures/graph.py) | `structures.graph` | add_edge, has_edge, weight, remove_edge / neighbors | O(1) / O(degree) | O(V + E) |

### Algorithms

| Algorithm | Module | Time | Space | Notes |
| --- | --- | --- | --- | --- |
| [Insertion sort](src/dsa/algorithms/sorting.py) | `algorithms.sorting` | O(n^2), O(n) if sorted | O(1) | Stable |
| [Merge sort](src/dsa/algorithms/sorting.py) | `algorithms.sorting` | O(n log n), O(n) if sorted | O(n) | Stable; insertion sort below 16 elements |
| [Quick sort](src/dsa/algorithms/sorting.py) | `algorithms.sorting` | O(n log n) expected | O(log n) | Random pivot, three-way partition, no recursion |
| [Heap sort](src/dsa/algorithms/sorting.py) | `algorithms.sorting` | O(n log n) worst | O(1) | In place, not stable |
| [Counting sort, radix sort](src/dsa/algorithms/sorting.py) | `algorithms.sorting` | O(n + k), O(d (n + b)) | O(n + k), O(n + b) | Integers only, negatives allowed |
| [lower_bound, upper_bound, first_true](src/dsa/algorithms/binary_search.py) | `algorithms.binary_search` | O(log n) | O(1) | Half-open window; also `count_equal` and rotated-array search |
| [BFS, DFS, shortest unweighted path, components](src/dsa/algorithms/traversal.py) | `algorithms.traversal` | O(V + E) | O(V) | Lazy generators; iterative DFS in recursive order |
| [Dijkstra](src/dsa/algorithms/dijkstra.py) | `algorithms.dijkstra` | O((V + E) log V) | O(V + E) | Uses the priority queue with lazy deletion |
| [Topological sort (Kahn)](src/dsa/algorithms/topological_sort.py) | `algorithms.topological_sort` | O(V + E) | O(V) | Raises `CycleError` naming a concrete cycle |
| [Kruskal and Prim MST](src/dsa/algorithms/mst.py) | `algorithms.mst` | O(E log E), O(E log V) | O(V + E) | Spanning forest on disconnected graphs |
| [Edit distance](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(nm) | O(min(n, m)) | Any sequences, not only strings |
| [Longest common subsequence](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(nm) | O(nm) | Returns the subsequence |
| [Longest increasing subsequence](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(n log n) | O(n) | Patience sorting, returns the subsequence |
| [0/1 knapsack](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(n W) | O(n W) | Returns the best value and the chosen items |
| [Coin change (fewest coins, number of ways)](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(k A) | O(A) | Works where greedy fails, e.g. coins 1, 3, 4 |
| [Maximum subarray (Kadane)](src/dsa/algorithms/dp.py) | `algorithms.dp` | O(n) | O(1) | Returns the sum and the slice bounds |

The comparison sorts take `key` and `reverse` exactly like `sorted()`, call `key` once per element, and return a new list.

## Install from GitHub

The package is not on PyPI. Install it straight from this repository:

```sh
pip install git+https://github.com/Mdjamilh/dsa-in-python
```

You need Python 3.12 or later. To pin a version, add a commit or tag, for example `git+https://github.com/Mdjamilh/dsa-in-python@<sha>`. The distribution is called `dsa-in-python` and the import name is `dsa`.

## Using it

Import everything from the top level, or from `dsa.structures`, `dsa.algorithms` or the module itself:

```python
from dsa import AVLTree, Graph, dijkstra
from dsa.structures.heap import PriorityQueue
```

A priority queue whose ties come out in the order they went in:

```python
from dsa import PriorityQueue

jobs: PriorityQueue[str, int] = PriorityQueue()
jobs.push("send newsletter", 3)
jobs.push("page on-call", 1)
jobs.push("rotate logs", 3)

jobs.pop()  # 'page on-call'
jobs.pop()  # 'send newsletter' (same priority as 'rotate logs', pushed first)
```

Shortest paths on a weighted graph:

```python
from dsa import Graph, dijkstra

roads = Graph.from_edges(
    [
        ("home", "cafe", 4),
        ("home", "park", 1),
        ("park", "cafe", 2),
        ("cafe", "office", 5),
    ]
)

paths = dijkstra(roads, "home")
paths.distance_to("office")  # 8
paths.path_to("office")  # ['home', 'park', 'cafe', 'office']
```

An ordered map that stays balanced, with order statistics:

```python
from dsa import AVLTree

scores = AVLTree({72: "ana", 95: "bo", 88: "cy"})

scores.floor(90)  # 88
scores.rank(88)  # 1, because one score is lower
scores.select(0)  # 72, the lowest score
list(scores.range(80, 100))  # [88, 95]
dict(scores)  # it is a MutableMapping: {72: 'ana', 88: 'cy', 95: 'bo'}
```

Dependency order with cycle reporting:

```python
from dsa import CycleError, Graph, topological_sort

steps = Graph.from_edges(
    [("install", "build"), ("build", "test"), ("test", "build")], directed=True
)
try:
    topological_sort(steps)
except CycleError as e:
    print(e)  # graph has a cycle: 'test' -> 'build' -> 'test'
```

An LRU cache, a trie and union-find:

```python
from dsa import LRUCache, Trie, UnionFind

cache: LRUCache[str, int] = LRUCache(2, on_evict=lambda k, v: print("evicted", k))
cache.put("a", 1)
cache.put("b", 2)
cache.get("a")
cache.put("c", 3)  # evicted b

words = Trie(["car", "cart", "cat"])
words.with_prefix("car")  # ['car', 'cart']
words.longest_prefix_of("cartwheel")  # 'cart'

groups = UnionFind(range(5))
groups.union(0, 1)
groups.connected(1, 0)  # True
groups.count  # 4
```

Sorts behave like `sorted()`:

```python
from dsa import merge_sort, quick_sort

people = [("ana", 31), ("bo", 25), ("cy", 31)]
merge_sort(people, key=lambda p: p[1])  # stable: ana stays before cy
quick_sort([5, 3, 5, 1], reverse=True)  # [5, 5, 3, 1]
```

Dynamic programming:

```python
from dsa import edit_distance, knapsack, min_coins

edit_distance("kitten", "sitting")  # 3
knapsack([(1, 1), (3, 4), (4, 5), (5, 7)], 7)  # (9, [1, 2]): weights 3 + 4, values 4 + 5
min_coins([1, 3, 4], 6)  # 2 (3 + 3; greedy would use 4 + 1 + 1)
```

Every example in a docstring is run as a doctest, so the examples in the source stay correct.

## How the tests work

There are 300 tests: 258 in `tests/` across 20 files, plus 42 doctests. They include 37 Hypothesis property tests and 11 Hypothesis state machines. CI fails if branch coverage drops under 95%. It is currently 100% of lines and branches.

A property test does not check a few hand-picked inputs. It describes a whole space of inputs, and Hypothesis generates hundreds of cases from it. For the mutable structures the tests are *state machines*: Hypothesis generates random sequences of operations (insert, delete, pop, rotate, ...), applies each one to this library's structure and to a reference, and checks after every step that the two agree. Here is the hash map test in outline ([full version](tests/test_hash_map.py)):

```python
class HashMapVsDict(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.m: HashMap[object, int] = HashMap()
        self.model: dict[object, int] = {}

    @rule(k=keys, v=st.integers())
    def set(self, k, v):
        self.m[k] = v
        self.model[k] = v

    # ... delete and get rules ...

    @invariant()
    def same_items(self):
        assert dict(self.m.items()) == self.model
```

The key strategy mixes small integers, short strings, objects that all hash to 42, `nan`, and values like `0`, `0.0`, `-0.0` and `False` that `dict` treats as the same key.

When a test fails, Hypothesis shrinks the input to a minimal case. To check that the tests can actually catch bugs, I put three plausible bugs into the code by hand and ran the suite. All three were caught, and the shrunk counterexamples explain the bug on their own:

1. **Hash map: treat a tombstone like an empty slot during lookup.** This is the classic open-addressing mistake. The state machine reduced it to three steps: `set(0)`, `set('')`, `delete(0)`, after which looking up `''` raises `KeyError`. In CPython both `0` and `''` hash to 0, so they share a home slot, and deleting the first one cut the probe path to the second.
2. **AVL tree: skip the inner rotation of the left-right case.** Shrunk to: insert 2, then 0, then 1. That is the smallest left-right shape, and the invariant check reports `unbalanced at 0`.
3. **Merge sort: take from the right half on ties.** The stability property failed on a list of 17 pairs. It needs 17 because below 16 elements merge sort hands off to insertion sort, which never runs the broken merge.

What each test compares against:

| Module | Reference ("oracle") |
| --- | --- |
| Dynamic array | `list`, including slice reads, writes and deletes |
| Linked lists, deque, queue | `collections.deque` or `list`, driven by the same operations; for the doubly linked list, node handles too |
| Binary heap | `heapq` (`heappush`, `heappop`, `heappushpop`, `heapreplace`), plus the heap shape after every step |
| Priority queue | `sorted()` by priority, which is stable, so ties must come out in FIFO order |
| Hash map | `dict`, with colliding, `nan` and cross-type-equal keys |
| BST, AVL tree | `dict` for contents and `bisect` for floor, ceiling and rank. The AVL test recomputes every height and subtree size and checks the balance at every node. |
| Trie | a `set`, with `str.startswith` for prefix queries; also checks that no empty branches are left behind |
| Union-find | naive relabelling (each element stores its set label, and a union rewrites them all) |
| LRU cache | a list model, including the exact eviction sequence |
| Graph | a dict of edges; checks counts through edge and vertex removal, including self-loops |
| Sorts | `sorted()` with the same `key` and `reverse`; for the stable sorts, exact equality with `sorted()` on records with many equal keys |
| Binary search | `bisect_left`, `bisect_right`, `list.index` and a linear scan |
| BFS, DFS | a recursive DFS for visiting order, and Bellman-Ford-style relaxation for BFS distances |
| Dijkstra | Floyd-Warshall all-pairs distances, with integer and float weights; every reconstructed path must add up to its distance |
| Topological sort | brute force over every vertex permutation; a returned cycle must use real edges |
| Kruskal, Prim | brute force over every subset of edges, for the true minimum spanning forest weight |
| DP functions | the memoised textbook recursion (edit distance), exhaustive search over subsequences, subsets or slices, and metric laws for edit distance |

There are also tests for the claims that are easy to state and easy to get wrong: heapify does fewer than 2n comparisons, merge sort on sorted input does fewer than 2n, a degenerate 3,000-deep BST and a 20,000-step DFS path work without hitting the recursion limit, and `key` is called exactly once per element.

The honest result: the property tests did not find a bug in a finished implementation. What failed during development was mostly my own expectations: a dynamic-array test assumed the wrong number of pops before a shrink, two examples expected one longest common or increasing subsequence when the code correctly returned another of the same length, a test expected components in the wrong order, and the radix sort property hit Hypothesis's deadline with base 65,536 under load. One real issue turned up while checking the usage examples: the cycle that `topological_sort` reported depended on set iteration order, so it changed with `PYTHONHASHSEED`. It was always a valid cycle, but not a reproducible one. The walk now starts in insertion order, and a test pins the exact cycle. The mutation check above shows the suite does catch the kind of bug that matters. Treat that as evidence, not proof.

## Benchmarks

Run them with `python bench/run.py` (or `--quick` for a smoke run, which CI does). Each case runs five times and the fastest run is reported. Inputs come from a seeded `random.Random`, so every run uses the same data.

**Machine:** Intel Core i5-8350U (4 cores, 8 threads, 1.7 GHz base), 16 GB RAM, Windows 11 Pro. **Python:** CPython 3.13.7. **Date:** 2024-09-03. The laptop was running other work at the time, so read the ratios, not the second digit.

**Sorting 20,000 integers** (ms)

| Input | `sorted()` | merge_sort | quick_sort | heap_sort | radix_sort | counting_sort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random (0 to 999,999) | **3.22** | 49.6 | 55.6 | 90.9 | 10.4 | 226 |
| already sorted | **0.20** | 4.39 | 71.3 | 79.7 | 8.01 | 5.80 |
| reversed | **0.16** | 46.8 | 62.9 | 64.7 | 6.68 | 5.78 |
| only 8 distinct values | **1.41** | 47.9 | 12.5 | 71.2 | 3.33 | 2.15 |

**Priority queue**: 20,000 pushes with a pop after every second push, then drain (ms)

| BinaryHeap | `heapq` | sorted list with `bisect.insort` and `pop(0)` |
| ---: | ---: | ---: |
| 66.2 | **10.1** | 47.3 |

**Maps**: insert 50,000 keys, then look each one up (ms)

| Keys | `dict` | HashMap | AVLTree | HashMap vs `dict` |
| --- | ---: | ---: | ---: | ---: |
| integers | **12.3** | 235 | 861 | 19x slower |
| strings | **14.6** | 204 | 873 | 14x slower |

**Search trees**: insert 2,000 keys (ms, and the resulting height)

| Insert order | BST time | BST height | AVL time | AVL height |
| --- | ---: | ---: | ---: | ---: |
| random | **6.96** | 26 | 54.5 | 13 |
| sorted | 317 | 2,000 | **38.4** | 11 |

**Deque**: 20,000 pushes at each end, then pop both ends until empty (ms)

| Deque (ring buffer) | `collections.deque` | `list` with `insert(0)` and `pop(0)` |
| ---: | ---: | ---: |
| 127 | **3.73** | 427 |

What the numbers show:

- **The built-ins win, and should.** `sorted()`, `dict`, `heapq` and `deque` are written in C. Here, pure Python costs 10 to 30 times as much for the same algorithm. That is the price of every comparison and index being an interpreted bytecode, and it is the main reason to use the standard library in real code.
- **Algorithm still matters more than language.** The sorted-order BST row is a linked list of 2,000 nodes, and the AVL tree is 8 times faster on it. On random input the BST is faster, because it does no rebalancing work. The `list` deque does an O(n) shift on every front operation and is more than 3 times slower than the pure-Python ring buffer, even though the shift happens in C.
- **Constant factors can beat asymptotics at this size.** The sorted list with `insort` beat `BinaryHeap` on the priority-queue workload. Its O(n) insert is one C `memmove` of a few thousand pointers, while each heap operation is about 15 interpreted comparisons. The heap would pull ahead as n grows. `heapq` beats both.
- **Sorting.** Merge sort's "halves already in order" check makes sorted input about 10 times cheaper. Three-way quick sort finishes runs of equal keys in one pass, so it is fastest of the comparison sorts when there are only 8 distinct values. Heap sort is the slowest despite its worst-case guarantee. Radix sort with base 256 needs only three passes for numbers under a million and is the fastest pure-Python sort here. Counting sort is the fastest on a narrow range and the slowest on a wide one, because it allocates a slot for every value between the minimum and the maximum.

## Design notes and limits

- **Open addressing, not chaining.** `HashMap` stores entries in one flat table with linear probing, tombstones for deletes, and Fibonacci hashing of the key's `hash()`. The multiplication matters because CPython hashes small integers to themselves, so keys like `0, 1024, 2048, ...` would otherwise share a home slot. The load factor (live entries plus tombstones) stays at most 2/3. Keys match as in `dict`: the same object, or equal with equal hashes, which is why `nan` can be found by the same `nan` object. Iteration is in slot order, not insertion order. There is no defence against keys chosen to collide.
- **AVL rather than red-black.** AVL trees are more strictly balanced (height at most about 1.44 log2 n), and their invariant is easy to state and check in a test. Red-black trees do fewer rotations on writes. Each node also stores its subtree size, which gives `rank`, `select` and `range` in O(log n).
- **No recursion where depth depends on input.** The BST, quick sort, DFS and trie iteration use explicit stacks. The AVL tree and merge sort recurse, but only to depth O(log n).
- **Lazy deletion in Dijkstra and Prim.** Instead of decrease-key, a better distance pushes a new heap entry and stale entries are skipped when popped. The heap can grow to O(E) entries, but the code is simpler.
- **Graphs allow one edge per ordered pair.** Adding the same edge again replaces its weight, so there are no multigraphs. NaN weights are rejected, and Dijkstra rejects negative weights.
- **Iteration while mutating** is not guarded, except `DoublyLinkedList.nodes()`, which allows removing the current node. Changing any other structure during a `for` loop over it gives unspecified results.
- **Strings.** The trie and edit distance work on Python `str` characters, which are Unicode code points, so trie iteration order matches `sorted()`.

## Development

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e . --group dev   # needs pip 25.1 or later

pytest                             # unit, property and doctests
pytest --cov                       # the same, failing under 95% coverage
HYPOTHESIS_PROFILE=ci pytest       # 400 examples per property instead of 100
ruff check . && ruff format --check .
mypy                               # strict, over src, tests and bench
python bench/run.py                # benchmarks, about a minute
```

CI runs ruff and `mypy --strict`, then the coverage-gated tests with the `ci` Hypothesis profile on Python 3.12 and 3.13 on Ubuntu and Python 3.13 on Windows. Each test job also smoke-runs the benchmarks, builds the sdist and wheel, and imports the wheel from a clean virtual environment.

## License

[MIT](LICENSE)
