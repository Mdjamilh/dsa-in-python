"""Small benchmarks: this library against the standard library.

Run with ``python bench/run.py`` (add ``--quick`` for a fast smoke run). Each
case runs several times and the fastest run is reported, which is the number
least disturbed by other work on the machine. Inputs come from a seeded
``random.Random``, so every run uses the same data.

The point is not to win. The built-ins are written in C and these are not.
The point is to see the shape: which inputs each algorithm likes, and how big
the constant-factor gap to C is.
"""

import argparse
import bisect
import heapq
import platform
import random
import time
from collections import deque
from collections.abc import Callable
from functools import partial

from dsa.algorithms import counting_sort, heap_sort, merge_sort, quick_sort, radix_sort
from dsa.structures import AVLTree, BinaryHeap, BinarySearchTree, Deque, HashMap


def best_of(fn: Callable[[], object], repeats: int) -> float:
    """Return the fastest of ``repeats`` runs of ``fn``, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n**{title}** (ms, fastest of the runs; lower is better)\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def fmt(ms: float) -> str:
    return f"{ms:.2f}" if ms < 100 else f"{ms:.0f}"


def bench_sorting(n: int, repeats: int) -> None:
    rng = random.Random(1)
    inputs = {
        "random": [rng.randrange(1_000_000) for _ in range(n)],
        "already sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "8 distinct values": [rng.randrange(8) for _ in range(n)],
    }
    sorts: dict[str, Callable[[list[int]], list[int]]] = {
        "`sorted()`": sorted,
        "merge_sort": merge_sort,
        "quick_sort": lambda a: quick_sort(a, rng=random.Random(0)),
        "heap_sort": heap_sort,
        "radix_sort": radix_sort,
        "counting_sort": counting_sort,
    }
    rows = []
    for label, data in inputs.items():
        rows.append([label] + [fmt(best_of(partial(f, data), repeats)) for f in sorts.values()])
    table(f"Sorting {n:,} integers", ["Input", *sorts], rows)


def bench_heap(n: int, repeats: int) -> None:
    rng = random.Random(2)
    values = [rng.random() for _ in range(n)]

    def ours() -> None:
        h: BinaryHeap[float] = BinaryHeap()
        for i, v in enumerate(values):
            h.push(v)
            if i % 2:
                h.pop()
        while h:
            h.pop()

    def stdlib() -> None:
        h: list[float] = []
        for i, v in enumerate(values):
            heapq.heappush(h, v)
            if i % 2:
                heapq.heappop(h)
        while h:
            heapq.heappop(h)

    def sorted_list() -> None:
        a: list[float] = []
        for i, v in enumerate(values):
            bisect.insort(a, v)  # keep sorted, pop the smallest from the front
            if i % 2:
                a.pop(0)
        while a:
            a.pop(0)

    table(
        f"Priority queue: {n:,} pushes with a pop after every second push, then drain",
        ["BinaryHeap", "`heapq`", "sorted list + `insort`"],
        [
            [
                fmt(best_of(ours, repeats)),
                fmt(best_of(stdlib, repeats)),
                fmt(best_of(sorted_list, repeats)),
            ]
        ],
    )


def _fill_dict[K: (int, str)](keys: list[K]) -> None:
    d: dict[K, int] = {}
    for i, k in enumerate(keys):
        d[k] = i
    for k in keys:
        d[k]


def _fill_hash_map[K: (int, str)](keys: list[K]) -> None:
    m: HashMap[K, int] = HashMap()
    for i, k in enumerate(keys):
        m[k] = i
    for k in keys:
        m[k]


def _fill_avl[K: (int, str)](keys: list[K]) -> None:
    t: AVLTree[K, int] = AVLTree()
    for i, k in enumerate(keys):
        t[k] = i
    for k in keys:
        t[k]


def _map_row[K: (int, str)](label: str, keys: list[K], repeats: int) -> list[str]:
    d = best_of(partial(_fill_dict, keys), repeats)
    h = best_of(partial(_fill_hash_map, keys), repeats)
    a = best_of(partial(_fill_avl, keys), repeats)
    return [label, fmt(d), fmt(h), fmt(a), f"{h / d:.1f}x"]


def bench_maps(n: int, repeats: int) -> None:
    rng = random.Random(3)
    int_keys = rng.sample(range(10 * n), n)
    str_keys = [f"key-{k}" for k in int_keys]
    table(
        f"Maps: insert {n:,} keys, then look each one up",
        ["Keys", "`dict`", "HashMap", "AVLTree", "HashMap / dict"],
        [_map_row("int keys", int_keys, repeats), _map_row("str keys", str_keys, repeats)],
    )


def _build(
    tree: type[BinarySearchTree[int, int]] | type[AVLTree[int, int]], keys: list[int]
) -> BinarySearchTree[int, int] | AVLTree[int, int]:
    return tree((k, k) for k in keys)


def bench_trees(n: int, repeats: int) -> None:
    rng = random.Random(4)
    rows = []
    for label, keys in [
        ("random order", rng.sample(range(n), n)),
        ("sorted order", list(range(n))),
    ]:
        bst_ms = best_of(partial(_build, BinarySearchTree, keys), repeats)
        avl_ms = best_of(partial(_build, AVLTree, keys), repeats)
        bst_height = _build(BinarySearchTree, keys).height()
        avl_height = _build(AVLTree, keys).height()
        rows.append([label, fmt(bst_ms), str(bst_height), fmt(avl_ms), str(avl_height)])
    table(
        f"Search trees: insert {n:,} keys",
        ["Insert order", "BST time", "BST height", "AVL time", "AVL height"],
        rows,
    )


def bench_deque(n: int, repeats: int) -> None:
    def ours() -> None:
        d: Deque[int] = Deque()
        for i in range(n):
            d.append(i)
            d.appendleft(i)
        while d:
            d.pop()
            d.popleft()

    def stdlib() -> None:
        d: deque[int] = deque()
        for i in range(n):
            d.append(i)
            d.appendleft(i)
        while d:
            d.pop()
            d.popleft()

    def plain_list() -> None:
        a: list[int] = []
        for i in range(n):
            a.append(i)
            a.insert(0, i)  # O(n): shifts everything
        while a:
            a.pop()
            a.pop(0)

    table(
        f"Deque: {n:,} pushes at each end, then pop both ends until empty",
        ["Deque (ring buffer)", "`collections.deque`", "`list` with insert(0) / pop(0)"],
        [
            [
                fmt(best_of(ours, repeats)),
                fmt(best_of(stdlib, repeats)),
                fmt(best_of(plain_list, repeats)),
            ]
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny inputs, one run each")
    args = parser.parse_args()
    scale, repeats = (0.05, 1) if args.quick else (1.0, 5)

    print(f"Python {platform.python_version()} ({platform.python_implementation()})")
    print(f"{platform.system()} {platform.release()}, {platform.machine()}")
    print(f"Processor: {platform.processor() or 'unknown'}")

    bench_sorting(int(20_000 * scale), repeats)
    bench_heap(int(20_000 * scale), repeats)
    bench_maps(int(50_000 * scale), repeats)
    bench_trees(int(2_000 * scale), repeats)
    bench_deque(int(20_000 * scale), repeats)


if __name__ == "__main__":
    main()
