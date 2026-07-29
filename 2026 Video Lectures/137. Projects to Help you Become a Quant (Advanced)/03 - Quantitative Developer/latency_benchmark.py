"""Algorithm latency benchmark: implementation choice vs asymptotic complexity.

The point of this script is to make two ideas concrete for a quant developer:

1. **Implementation matters as much as big-O.** Two algorithms with the *same*
   asymptotic complexity can differ by 10-100× in wall-clock latency because of
   constant factors: interpreter overhead, memory layout, and whether the work
   happens in optimised C or in the Python interpreter.

2. **Latency scales with input size in a way that mirrors big-O.** On a log-log
   plot, an O(n) routine has slope ~1, an O(n log n) routine is slightly steeper,
   and an O(log n) / O(1) lookup is nearly flat. Seeing the slopes makes the
   theory tangible.

We benchmark three families a quant dev meets constantly:

* **Sorting**   — pure-Python insertion-ish sort (O(n^2)) vs built-in Timsort
                  (``sorted``, O(n log n)) vs ``numpy.sort`` (O(n log n) in C).
* **Search**    — linear scan (O(n)) vs ``bisect`` binary search on a sorted
                  list (O(log n)) vs ``set`` membership (O(1) average).
* **Reduction** — summing with a Python ``for`` loop (O(n)) vs ``numpy.sum``
                  (O(n) but vectorised in C).

Each case is timed with :func:`timeit.repeat` (multiple repetitions, best-of via
median), and the results are printed as a table and plotted on log-log axes and
saved to ``latency_results.png``.

Run headlessly::

    python latency_benchmark.py
"""

from __future__ import annotations

import statistics
import sys
import timeit
from bisect import bisect_left

import numpy as np

# Windows consoles may default to cp1252 and fail on µ / × glyphs in the table.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def time_call(stmt, *, repeat: int = 7, number: int = 1) -> float:
    """Return the median per-call time (seconds) of ``stmt`` over ``repeat`` runs.

    ``timeit`` disables the garbage collector and uses ``perf_counter`` under the
    hood. We take the **median** of ``repeat`` measurements (each the total time
    of ``number`` calls) to shrug off occasional OS-scheduling spikes, which is
    more representative than the mean and less brittle than a single ``min``.
    """
    timer = timeit.Timer(stmt)
    samples = timer.repeat(repeat=repeat, number=number)
    return statistics.median(samples) / number


# ---------------------------------------------------------------------------
# Pure-Python reference algorithms (deliberately un-optimised)
# ---------------------------------------------------------------------------
def python_insertion_sort(data: list) -> list:
    """O(n^2) insertion sort in pure Python -- the slow baseline for sorting."""
    a = list(data)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def python_linear_search(data: list, target) -> int:
    """O(n) linear scan; returns index or -1."""
    for i, x in enumerate(data):
        if x == target:
            return i
    return -1


def python_binary_search(sorted_data: list, target) -> int:
    """O(log n) binary search on an already-sorted list via ``bisect``."""
    i = bisect_left(sorted_data, target)
    if i != len(sorted_data) and sorted_data[i] == target:
        return i
    return -1


def python_loop_sum(data) -> float:
    """O(n) reduction with an explicit Python loop."""
    total = 0.0
    for x in data:
        total += x
    return total


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------
def benchmark_sorting(sizes: list[int], rng: np.random.Generator) -> dict[str, list[float]]:
    results = {"python_insertion (O(n^2))": [], "sorted/Timsort (O(n log n))": [], "numpy.sort (O(n log n))": []}
    for n in sizes:
        arr = rng.standard_normal(n)
        py = arr.tolist()
        # Insertion sort is O(n^2); skip it for large n so the script stays fast.
        if n <= 5000:
            results["python_insertion (O(n^2))"].append(time_call(lambda: python_insertion_sort(py), repeat=5))
        else:
            results["python_insertion (O(n^2))"].append(float("nan"))
        results["sorted/Timsort (O(n log n))"].append(time_call(lambda: sorted(py)))
        results["numpy.sort (O(n log n))"].append(time_call(lambda: np.sort(arr)))
    return results


def benchmark_search(sizes: list[int], rng: np.random.Generator) -> dict[str, list[float]]:
    results = {"linear scan (O(n))": [], "bisect / binary (O(log n))": [], "set lookup (O(1))": []}
    for n in sizes:
        # Distinct sorted integers; search for a value that is present.
        data = list(range(n))
        target = n - 1  # worst case for linear scan (last element)
        data_set = set(data)
        results["linear scan (O(n))"].append(time_call(lambda: python_linear_search(data, target)))
        results["bisect / binary (O(log n))"].append(time_call(lambda: python_binary_search(data, target)))
        results["set lookup (O(1))"].append(time_call(lambda: target in data_set))
    return results


def benchmark_reduction(sizes: list[int], rng: np.random.Generator) -> dict[str, list[float]]:
    results = {"python loop sum (O(n))": [], "numpy.sum (O(n), vectorised)": []}
    for n in sizes:
        arr = rng.standard_normal(n)
        py = arr.tolist()
        results["python loop sum (O(n))"].append(time_call(lambda: python_loop_sum(py)))
        results["numpy.sum (O(n), vectorised)"].append(time_call(lambda: np.sum(arr)))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(title: str, sizes: list[int], results: dict[str, list[float]]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    header = f"{'n':>10} | " + " | ".join(f"{name:>28}" for name in results)
    print(header)
    print("-" * len(header))
    for idx, n in enumerate(sizes):
        row = f"{n:>10,} | "
        cells = []
        for name in results:
            t = results[name][idx]
            cells.append("        —          " if (t != t) else f"{t * 1e6:>24.2f} µs")
        print(row + " | ".join(cells))


def plot_results(sizes, sorting, search, reduction, out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (axes[0], "Sorting", sorting),
        (axes[1], "Search (element present)", search),
        (axes[2], "Reduction (sum)", reduction),
    ]
    x = np.asarray(sizes, dtype=float)
    for ax, title, results in panels:
        for name, times in results.items():
            y = np.asarray(times, dtype=float)
            mask = ~np.isnan(y)
            ax.loglog(x[mask], y[mask] * 1e6, marker="o", markersize=4, label=name)
        ax.set_title(title)
        ax.set_xlabel("input size n")
        ax.set_ylabel("median latency (µs)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Algorithm latency vs input size (log-log): implementation beats big-O constants", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved latency chart to {out_png}")


def main() -> None:
    rng = np.random.default_rng(7)

    sort_search_sizes = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    reduction_sizes = [1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]

    print("Benchmarking... (this takes a few seconds)")
    sorting = benchmark_sorting(sort_search_sizes, rng)
    search = benchmark_search(sort_search_sizes, rng)
    reduction = benchmark_reduction(reduction_sizes, rng)

    print_table("SORTING: latency (µs) vs input size", sort_search_sizes, sorting)
    print_table("SEARCH: latency (µs) vs input size", sort_search_sizes, search)
    print_table("REDUCTION: latency (µs) vs input size", reduction_sizes, reduction)

    # A couple of headline takeaways, computed from the data.
    print("\n" + "=" * 78)
    print("TAKEAWAYS")
    print("=" * 78)
    i = sort_search_sizes.index(1000)
    py_ins = sorting["python_insertion (O(n^2))"][i]
    timsort = sorting["sorted/Timsort (O(n log n))"][i]
    print(f"* At n=1,000, built-in Timsort is ~{py_ins / timsort:.0f}× faster than a pure-Python O(n^2) sort.")
    j = sort_search_sizes.index(100_000)
    lin = search["linear scan (O(n))"][j]
    setl = search["set lookup (O(1))"][j]
    print(f"* At n=100,000, a set lookup is ~{lin / setl:.0f}× faster than a linear scan (O(1) vs O(n)).")
    k = reduction_sizes.index(1_000_000)
    pyloop = reduction["python loop sum (O(n))"][k]
    npsum = reduction["numpy.sum (O(n), vectorised)"][k]
    print(f"* At n=1,000,000, numpy.sum is ~{pyloop / npsum:.0f}× faster than a Python loop (same O(n), better constants).")

    plot_results(sort_search_sizes, sorting, search, reduction, "latency_results.png")


if __name__ == "__main__":
    main()
