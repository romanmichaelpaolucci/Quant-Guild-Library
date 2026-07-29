"""Numpy-vectorised Monte Carlo European option pricer.

Same model and same shared-``Z`` contract as :mod:`mc_python`, but the explicit
Python double loop is replaced by array operations. The time dimension is
collapsed with a single cumulative sum along the steps axis, so the whole
simulation is a handful of numpy calls that run in optimised C.

This is usually the *pragmatic* winner for a quant: it is fast, memory-bound,
and requires no compiler. The Cython/numba versions exist to show what further
speedups (and memory savings) are possible when you drop into typed, compiled
loops.
"""

from __future__ import annotations

import numpy as np


def price_european_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    Z: np.ndarray,
    option: str = "call",
) -> float:
    """Price a European option using numpy vectorisation.

    Parameters
    ----------
    S0, K, r, sigma, T : float
        Standard Black-Scholes/GBM parameters (see :mod:`mc_python`).
    Z : numpy.ndarray, shape (n_paths, n_steps)
        Pre-generated standard-normal shocks (shared across all implementations).
    option : {"call", "put"}
        Option type.

    Returns
    -------
    float
        The Monte Carlo price estimate.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n_steps = Z.shape[1]
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    # Log-increments per step, then cumulative sum along the time axis gives the
    # log-price path. We only need the terminal value, so take the last column.
    log_increments = drift + vol * Z
    terminal_log_return = log_increments.sum(axis=1)
    S_T = S0 * np.exp(terminal_log_return)

    if option == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    return float(discount * payoff.mean())
