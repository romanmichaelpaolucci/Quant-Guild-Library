"""Numba JIT Monte Carlo European option pricer (optional).

If ``numba`` is installed, this module exposes a ``@njit`` compiled version of
the same explicit path/step loop used by :mod:`mc_python`. Numba compiles the
Python function to machine code at runtime (LLVM), giving Cython-like speed with
zero build step and no separate ``.pyx`` file.

The module is import-guarded: importing it never fails just because numba is
missing. Callers should check :data:`NUMBA_AVAILABLE` before using
:func:`price_european_option`.

Note on timing: the first call to a ``@njit`` function pays a one-off
*compilation* cost. ``benchmark.py`` deliberately warms the function up before
timing so that the reported numbers reflect steady-state execution, not the JIT
compile.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - environment without numba
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _price_kernel(S0, K, r, sigma, T, Z, is_call):
        n_paths = Z.shape[0]
        n_steps = Z.shape[1]
        dt = T / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)

        payoff_sum = 0.0
        for i in range(n_paths):
            s = S0
            for j in range(n_steps):
                s *= np.exp(drift + vol * Z[i, j])
            if is_call:
                payoff = s - K
            else:
                payoff = K - s
            if payoff > 0.0:
                payoff_sum += payoff

        discount = np.exp(-r * T)
        return discount * payoff_sum / n_paths

    def price_european_option(S0, K, r, sigma, T, Z, option="call"):
        """Numba-compiled European option price (see :mod:`mc_python`)."""
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        return float(_price_kernel(S0, K, r, sigma, T, Z, option == "call"))

else:  # pragma: no cover - only exercised when numba is absent

    def price_european_option(*args, **kwargs):
        raise RuntimeError(
            "numba is not installed; install it with `pip install numba` "
            "to use the numba implementation."
        )
