# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython Monte Carlo European option pricer -- the compiled hot loop.

This is the *same* explicit path/step loop as :mod:`mc_python`, but with the
compiler-level knobs that make Cython fast:

* ``cdef double`` locals and a typed memoryview ``double[:, ::1]`` for ``Z`` so
  there is no Python object boxing inside the loop -- everything is native C.
* ``boundscheck(False)`` / ``wraparound(False)`` remove per-index safety checks
  (we guarantee valid indices ourselves).
* ``cdivision(True)`` uses C division semantics (no Python zero-division check).
* ``exp`` / ``sqrt`` come from ``libc.math`` (the C standard library), avoiding
  a Python-level function call per step.

The result is a tight machine-code loop with essentially zero interpreter
overhead, which is where the large speedup over pure Python comes from.

Build with::

    python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt


def price_european_option(
    double S0,
    double K,
    double r,
    double sigma,
    double T,
    Z,
    str option="call",
):
    """Price a European option with a typed, compiled path/step loop.

    Parameters mirror the pure-Python implementation. ``Z`` must be convertible
    to a C-contiguous ``float64`` array of shape ``(n_paths, n_steps)``.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Z_arr = np.ascontiguousarray(Z, dtype=np.float64)
    cdef double[:, ::1] z = Z_arr

    cdef Py_ssize_t n_paths = z.shape[0]
    cdef Py_ssize_t n_steps = z.shape[1]
    cdef Py_ssize_t i, j

    cdef double dt = T / n_steps
    cdef double drift = (r - 0.5 * sigma * sigma) * dt
    cdef double vol = sigma * sqrt(dt)
    cdef bint is_call = (option == "call")

    cdef double payoff_sum = 0.0
    cdef double s, payoff

    for i in range(n_paths):
        s = S0
        for j in range(n_steps):
            s *= exp(drift + vol * z[i, j])
        if is_call:
            payoff = s - K
        else:
            payoff = K - s
        if payoff > 0.0:
            payoff_sum += payoff

    cdef double discount = exp(-r * T)
    return discount * payoff_sum / n_paths
