"""Benchmark & cross-check the Monte Carlo pricers (the centrepiece).

What this script does
---------------------
1. Builds one shared array of standard-normal shocks ``Z`` from a fixed seed, so
   every implementation prices the *identical* set of paths.
2. Runs every *available* implementation (pure Python, numpy, Cython if built,
   numba if installed) on the same parameters.
3. Verifies they all agree on the option price to within a tiny tolerance, and
   sanity-checks against the closed-form Black-Scholes value.
4. Times each implementation carefully (warm-ups + repetitions + median) and
   prints a speed-up table relative to pure Python.
5. Saves a bar chart of the speed-ups to ``mc_speedup.png``.

Robustness
----------
The Cython extension requires a C compiler to have been built first
(``python setup.py build_ext --inplace``). If it was not built, this script
does **not** crash: it skips Cython, prints a clear note with build
instructions, and reports results for the remaining implementations.

Run headlessly::

    python benchmark.py
    python benchmark.py --paths 40000 --steps 50 --repeats 5
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from typing import Callable

import numpy as np

# Some Windows consoles default to cp1252 and choke on nice glyphs (Δ, ×, µ).
# Reconfigure stdout to UTF-8 so the report prints cleanly everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Use a non-interactive backend so the script works over SSH / in CI / headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

import mc_python
import mc_numpy

# --- Optional implementations, imported defensively -------------------------

# Cython: only importable once the extension has been compiled.
try:
    import mc_cython

    CYTHON_AVAILABLE = True
    CYTHON_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or ABI mismatch, etc.
    CYTHON_AVAILABLE = False
    CYTHON_IMPORT_ERROR = exc

# Numba: optional dependency.
try:
    import mc_numba

    NUMBA_AVAILABLE = mc_numba.NUMBA_AVAILABLE
except Exception:
    NUMBA_AVAILABLE = False


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form Black-Scholes price of a European call (reference value)."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S0 * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)


def bench(fn: Callable[[], float], repeats: int, warmups: int = 1) -> dict:
    """Time ``fn`` carefully and return timing statistics (seconds).

    Methodology
    -----------
    * Run ``warmups`` untimed calls first. This pays one-off costs (JIT
      compilation for numba, cache warming, first-touch page faults) so they do
      not pollute the measured numbers.
    * Then run ``repeats`` timed calls with ``time.perf_counter`` (a high
      resolution, monotonic clock).
    * Report the **median** (robust to occasional OS-scheduling spikes) plus the
      min and the spread, rather than a single noisy sample or the mean.
    """
    result = None
    for _ in range(warmups):
        result = fn()

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)

    samples.sort()
    return {
        "result": result,
        "median": statistics.median(samples),
        "min": samples[0],
        "max": samples[-1],
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=40_000, help="number of Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=50, help="time steps per path")
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions per implementation")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for the shared shocks")
    args = parser.parse_args()

    # --- Model parameters ---------------------------------------------------
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n_paths, n_steps = args.paths, args.steps

    print("=" * 74)
    print("Monte Carlo European call — implementation benchmark")
    print("=" * 74)
    print(f"S0={S0}  K={K}  r={r}  sigma={sigma}  T={T}")
    print(f"paths={n_paths:,}  steps={n_steps}  (kernel iterations = {n_paths * n_steps:,})")
    print(f"repeats={args.repeats}  seed={args.seed}")
    print()

    # --- Shared random shocks: identical paths for every implementation -----
    rng = np.random.default_rng(args.seed)
    Z = rng.standard_normal(size=(n_paths, n_steps)).astype(np.float64)

    bs_price = black_scholes_call(S0, K, r, sigma, T)

    # --- Assemble the implementations that are actually available -----------
    implementations: list[tuple[str, Callable[[], float]]] = [
        ("pure_python", lambda: mc_python.price_european_option(S0, K, r, sigma, T, Z, "call")),
        ("numpy", lambda: mc_numpy.price_european_option(S0, K, r, sigma, T, Z, "call")),
    ]
    if CYTHON_AVAILABLE:
        implementations.append(
            ("cython", lambda: mc_cython.price_european_option(S0, K, r, sigma, T, Z, "call"))
        )
    if NUMBA_AVAILABLE:
        implementations.append(
            ("numba", lambda: mc_numba.price_european_option(S0, K, r, sigma, T, Z, "call"))
        )

    # Report which optional pieces are in play.
    if not CYTHON_AVAILABLE:
        print("[note] Cython extension not built — skipping the 'cython' implementation.")
        print("       Build it with:  python setup.py build_ext --inplace")
        print(f"       (import error: {CYTHON_IMPORT_ERROR})")
    if not NUMBA_AVAILABLE:
        print("[note] numba not installed — skipping the 'numba' implementation.")
        print("       Install it with:  pip install numba")
    print()

    # --- Run, verify agreement, and time -----------------------------------
    # Pure Python is slow; cap its repeats so the whole run stays snappy while
    # still taking a median.
    results: dict[str, dict] = {}
    for name, fn in implementations:
        repeats = min(args.repeats, 3) if name == "pure_python" else args.repeats
        print(f"  running {name:<12} ... ", end="", flush=True)
        stats = bench(fn, repeats=repeats)
        results[name] = stats
        print(f"price={stats['result']:.6f}   median={stats['median'] * 1e3:.3f} ms")

    print()
    print(f"Black-Scholes closed form:            {bs_price:.6f}")
    print(f"Monte Carlo (numpy) estimate:         {results['numpy']['result']:.6f}")
    mc_err = abs(results["numpy"]["result"] - bs_price)
    print(f"|MC - BS| (finite-sample MC error):   {mc_err:.6f}")
    print()

    # --- Cross-check: every implementation must agree (shared Z) ------------
    # They price identical paths, so differences are only floating-point
    # summation order → a very tight tolerance is appropriate.
    ref = results["numpy"]["result"]
    tol = 1e-6 * max(1.0, abs(ref))
    print(f"Agreement check (tolerance = {tol:.2e}, all share the same paths):")
    all_agree = True
    for name in results:
        diff = abs(results[name]["result"] - ref)
        ok = diff <= tol
        all_agree &= ok
        print(f"  {name:<12} price={results[name]['result']:.10f}  |Δ vs numpy|={diff:.2e}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n  ==> {'ALL IMPLEMENTATIONS AGREE' if all_agree else 'DISAGREEMENT DETECTED'}")
    print()

    # --- Speed-up table (relative to pure Python) ---------------------------
    base = results["pure_python"]["median"]
    print("=" * 74)
    print("Speed-up table (higher is faster; baseline = pure Python)")
    print("=" * 74)
    print(f"{'implementation':<14}{'median (ms)':>14}{'stdev (ms)':>14}{'speedup vs python':>20}")
    print("-" * 74)
    speedups: dict[str, float] = {}
    for name in results:
        med = results[name]["median"]
        sd = results[name]["stdev"]
        speedup = base / med if med > 0 else float("inf")
        speedups[name] = speedup
        print(f"{name:<14}{med * 1e3:>14.3f}{sd * 1e3:>14.3f}{speedup:>19.1f}x")
    print("=" * 74)
    print()

    # --- Bar chart of speed-ups --------------------------------------------
    names = list(speedups.keys())
    values = [speedups[n] for n in names]
    colors = ["#888888", "#1f77b4", "#d62728", "#2ca02c"][: len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Speed-up vs pure Python (×)")
    ax.set_title(
        f"Monte Carlo pricer: implementation speed-up\n"
        f"({n_paths:,} paths × {n_steps} steps = {n_paths * n_steps:,} loop iterations)"
    )
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.5)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.1f}×",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    out_png = "mc_speedup.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved speed-up chart to {out_png}")


if __name__ == "__main__":
    main()
