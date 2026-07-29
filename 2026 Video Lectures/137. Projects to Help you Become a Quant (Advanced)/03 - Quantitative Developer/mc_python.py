"""Pure-Python Monte Carlo European option pricer (the slow baseline).

This module prices a European call (or put) under geometric Brownian motion
(GBM) using an *explicit* double loop over paths and time steps. It is written
deliberately in plain Python -- no numpy vectorisation, no compiled code -- so
that it can serve as the honest, slow baseline against which the numpy, Cython
and numba implementations are compared in ``benchmark.py``.

Model
-----
Under the risk-neutral measure the asset follows

    S_{t+dt} = S_t * exp((r - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z)

where ``Z`` is a standard normal draw. The discounted expected payoff

    price = exp(-r*T) * E[max(S_T - K, 0)]      (call)
    price = exp(-r*T) * E[max(K - S_T, 0)]      (put)

is estimated by averaging over many simulated paths.

Why pass in the random numbers?
-------------------------------
Every implementation in this project receives the *same* pre-generated array of
standard-normal shocks ``Z`` (shape ``(n_paths, n_steps)``). Sharing the random
numbers means all implementations price the *identical* set of paths, so they
must agree on the answer up to floating-point summation order. That lets us
verify correctness with a tiny tolerance and ensures the benchmark measures the
*pricing loop* itself rather than differences in random-number generation.
"""

from __future__ import annotations

import math


def price_european_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    Z,
    option: str = "call",
) -> float:
    """Price a European option with an explicit Python loop over paths/steps.

    Parameters
    ----------
    S0 : float
        Spot price at t=0.
    K : float
        Strike.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised).
    T : float
        Time to maturity in years.
    Z : 2D array-like of float, shape (n_paths, n_steps)
        Pre-generated standard-normal shocks. A numpy array is accepted; it is
        converted to nested Python lists first so the hot loop works on native
        Python floats (indexing a numpy array element-by-element would add
        scalar-boxing overhead that has nothing to do with the algorithm).
    option : {"call", "put"}
        Option type.

    Returns
    -------
    float
        The Monte Carlo price estimate.
    """
    # Convert to nested Python lists once. tolist() runs in C and is cheap
    # relative to the O(n_paths * n_steps) Python loop below; doing this keeps
    # the loop honestly "pure Python" (no numpy scalar boxing per access).
    if hasattr(Z, "tolist"):
        Z = Z.tolist()

    n_paths = len(Z)
    n_steps = len(Z[0]) if n_paths else 0

    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    is_call = option == "call"

    payoff_sum = 0.0
    for i in range(n_paths):
        s = S0
        row = Z[i]
        for j in range(n_steps):
            s *= math.exp(drift + vol * row[j])
        if is_call:
            payoff = s - K
        else:
            payoff = K - s
        if payoff > 0.0:
            payoff_sum += payoff

    discount = math.exp(-r * T)
    return discount * payoff_sum / n_paths
