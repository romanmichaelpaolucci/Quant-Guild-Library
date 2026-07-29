"""
markovian_lifting.py
====================

**Markovian lifting** of a rough volatility model: approximate the singular
fractional kernel by a finite sum of exponentials, turning a non-Markovian
Volterra process into a high-dimensional but *Markovian* system of
Ornstein-Uhlenbeck factors.

Why bother?
-----------
The rough kernel ``K(t) = t^{H - 1/2}`` (H < 1/2) has infinite memory: the
Volterra process ``Y_t = int_0^t K(t-s) dW_s`` is *not* Markovian, so it has no
finite-dimensional state and is expensive to simulate and impossible to price
with a PDE. Abi Jaber & El Euch (2019) and Cuchiero-Teichmann (2019) observed
that the kernel admits a completely monotone integral representation

    t^{H - 1/2}  =  int_0^infty  e^{-x t}  mu(dx),
    mu(dx)       =  x^{-H - 1/2} / Gamma(1/2 - H)  dx           (H < 1/2),

which follows from Gamma(s) t^{-s} = int_0^infty x^{s-1} e^{-x t} dx with
s = 1/2 - H > 0. Discretising the *measure* ``mu`` by a finite sum of point
masses ``mu ~ sum_j w_j delta_{x_j}`` yields a sum-of-exponentials kernel

    K_N(t) = sum_{j=1}^N  w_j  e^{-x_j t}  ~  K(t),

and each mode ``U^j_t = int_0^t e^{-x_j (t-s)} dW_s`` is a mean-reverting OU
factor solving ``dU^j = -x_j U^j dt + dW``. The lifted process
``Y^N_t = sum_j w_j U^j_t`` is driven by the ``N``-dimensional Markov state
``(U^1, ..., U^N)`` and reproduces the rough behaviour as ``N -> infinity``.

Choosing the nodes and weights
------------------------------
We use the transparent quadrature of Abi Jaber (2019), "Lifting the Heston
model": pick geometric breakpoints ``0 < eta_0 < eta_1 < ... < eta_N`` and set

    w_j = int_{eta_{j-1}}^{eta_j} mu(dx),
    x_j = (1 / w_j) int_{eta_{j-1}}^{eta_j} x mu(dx),

i.e. each ``w_j`` is the total mass of ``mu`` on an interval and ``x_j`` its
mass-weighted centroid. Both integrals are elementary because ``mu`` is a power
law. Widening the geometric range and increasing ``N`` drives ``K_N -> K``.

References
----------
* E. Abi Jaber, O. El Euch, "Multifactor approximation of rough volatility
  models", SIAM J. Financial Math., 2019.
* E. Abi Jaber, "Lifting the Heston model", Quantitative Finance, 2019.
* C. Cuchiero, J. Teichmann, "Markovian lifts of positive semidefinite affine
  Volterra-type processes", Decisions in Economics and Finance, 2019.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def fractional_kernel(t: np.ndarray, H: float) -> np.ndarray:
    """The (unnormalised) rough kernel K(t) = t^{H - 1/2}, H < 1/2."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = t[pos] ** (H - 0.5)
    return out


@dataclass
class SumOfExponentials:
    """A sum-of-exponentials approximation ``sum_j w_j exp(-x_j t)`` of the
    fractional kernel, plus the OU factors that realise the Markovian lift."""

    H: float
    weights: np.ndarray   # w_j
    nodes: np.ndarray     # x_j (mean-reversion speeds)

    @property
    def n_factors(self) -> int:
        return len(self.nodes)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate K_N(t) = sum_j w_j exp(-x_j t)."""
        t = np.asarray(t, dtype=float)
        return (self.weights[None, :] * np.exp(-self.nodes[None, :] * t[:, None])).sum(axis=1)

    def l2_kernel_error(self, T: float = 1.0, n: int = 2000) -> float:
        """Relative L2 error of K_N vs K on (0, T].

        The true kernel is (mildly) singular at 0 but square-integrable for
        H > 0, so a fine grid gives a faithful error estimate.
        """
        t = np.linspace(T / n, T, n)
        true = fractional_kernel(t, self.H)
        approx = self.evaluate(t)
        num = np.sqrt(np.trapezoid((approx - true) ** 2, t))
        den = np.sqrt(np.trapezoid(true ** 2, t))
        return float(num / den)


def build_lift(H: float, n_factors: int, x_min: float = 1e-2,
               x_max: float = 1e4) -> SumOfExponentials:
    """Construct the sum-of-exponentials lift with ``n_factors`` OU modes.

    Uses geometric breakpoints on ``[x_min, x_max]`` and the mass / centroid
    quadrature of the spectral measure ``mu(dx) = x^{-H-1/2}/Gamma(1/2-H) dx``.

    Closed forms on an interval [a, b] (let ``p = 1/2 - H`` so mu ~ x^{-p-1/2}...):
    with alpha = -H - 1/2,
        int_a^b x^{alpha}   dx = (b^{alpha+1}   - a^{alpha+1})   / (alpha + 1),
        int_a^b x^{alpha+1} dx = (b^{alpha+2}   - a^{alpha+2})   / (alpha + 2).
    """
    if not (0.0 < H < 0.5):
        raise ValueError("H must lie in (0, 1/2) for a rough kernel.")

    alpha = -H - 0.5                      # exponent of mu's density
    c = 1.0 / math.gamma(0.5 - H)         # normalising constant of mu
    breaks = np.geomspace(x_min, x_max, n_factors + 1)

    weights = np.empty(n_factors)
    nodes = np.empty(n_factors)
    for j in range(n_factors):
        a, b = breaks[j], breaks[j + 1]
        mass = c * (b ** (alpha + 1) - a ** (alpha + 1)) / (alpha + 1)
        first_moment = c * (b ** (alpha + 2) - a ** (alpha + 2)) / (alpha + 2)
        weights[j] = mass
        nodes[j] = first_moment / mass    # mass-weighted centroid
    return SumOfExponentials(H=H, weights=weights, nodes=nodes)


def error_vs_n(H: float, n_list, T: float = 1.0, **kwargs):
    """Relative L2 kernel error for a range of factor counts ``N``.

    Demonstrates the core convergence claim: error decreases as ``N`` grows.
    """
    n_list = list(n_list)
    errors = [build_lift(H, n, **kwargs).l2_kernel_error(T=T) for n in n_list]
    return {"n_factors": np.array(n_list), "l2_error": np.array(errors)}


def simulate_lifted_volterra(lift: SumOfExponentials, T: float, n_steps: int,
                             seed: int | None = 0):
    """Simulate the lifted Markovian process and compare with the true Volterra
    process on the *same* Brownian path.

    We drive both with one shared increment sequence ``dW``:
      * True (non-Markovian) reference via direct convolution,
            Y_{t_k} = sum_i K(t_k - t_{i-1}) dW_i.
      * Lifted (Markovian) OU factors via exact-in-law one-step recursion
            U^j_{k} = e^{-x_j dt} U^j_{k-1} + eps^j_k,
        with the stochastic integral over the step approximated by the shared
        increment scaled by the step's kernel weight, and
            Y^N_{t_k} = sum_j w_j U^j_k.
    Both use the same ``dW`` so the paths are directly comparable.

    Returns dict with ``t``, ``Y_true``, ``Y_lift`` and the number of factors.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    dW = rng.standard_normal(n_steps) * math.sqrt(dt)

    # --- true Volterra process by direct (left-point) convolution ----------
    Y_true = np.zeros(n_steps + 1)
    lags = (np.arange(1, n_steps + 1) - 0.0) * dt   # t_k - t_{i-1} style lags
    kernel_lags = fractional_kernel(lags, lift.H)
    for k in range(1, n_steps + 1):
        # contributions of increments dW_1..dW_k with kernel at lag t_k-t_{i-1}
        Y_true[k] = np.dot(kernel_lags[:k][::-1], dW[:k])

    # --- lifted Markovian OU factors, same driving increments --------------
    x = lift.nodes
    w = lift.weights
    decay = np.exp(-x * dt)
    U = np.zeros(len(x))
    Y_lift = np.zeros(n_steps + 1)
    for k in range(1, n_steps + 1):
        # OU exact mean recursion driven by the shared Brownian increment.
        U = decay * U + dW[k - 1]
        Y_lift[k] = np.dot(w, U)

    return {"t": t, "Y_true": Y_true, "Y_lift": Y_lift,
            "n_factors": lift.n_factors}


def _smoke_test() -> None:
    """Headless self-check: error must fall as N grows; sims must be finite."""
    H = 0.1
    res = error_vs_n(H, [1, 2, 4, 8, 16, 32])
    errs = res["l2_error"]
    assert np.all(np.isfinite(errs))
    # Monotone-ish decrease; require the endpoint to be clearly better.
    assert errs[-1] < errs[0], "kernel error did not decrease with N"
    assert errs[-1] < 0.5, "32-factor lift unexpectedly inaccurate"

    lift = build_lift(H, 20)
    sim = simulate_lifted_volterra(lift, T=1.0, n_steps=200, seed=3)
    assert np.all(np.isfinite(sim["Y_true"]))
    assert np.all(np.isfinite(sim["Y_lift"]))
    print(f"[markovian_lifting] OK  N=1 err={errs[0]:.3f} -> "
          f"N=32 err={errs[-1]:.3f}")


if __name__ == "__main__":
    _smoke_test()
