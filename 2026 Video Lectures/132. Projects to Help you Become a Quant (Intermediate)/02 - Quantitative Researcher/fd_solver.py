"""
fd_solver.py
============

Finite-difference (FD) engine for the Black-Scholes family of pricing PDEs.

The Black-Scholes PDE
---------------------
An option value ``V(S, t)`` under the risk-neutral geometric Brownian motion
``dS = (r - q) S dt + sigma S dW`` satisfies

    V_t + 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0,        (BS-PDE)

a backward parabolic PDE with a *terminal* condition at expiry ``t = T`` (the
payoff) and boundary conditions in ``S``. Substituting time-to-maturity
``tau = T - t`` turns it into a forward heat-like equation

    V_tau = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V =: L[V],

which we integrate forward in ``tau`` from ``tau = 0`` (the payoff) up to
``tau = T`` (today).

Discretisation (theta-scheme)
-----------------------------
On a uniform grid ``S_i = i * dS`` (i = 0..M) and ``tau_n = n * dtau``
(n = 0..N) we approximate ``L`` at interior nodes by central differences:

    (L V)_i = alpha_i V_{i-1} + beta_i V_i + gamma_i V_{i+1},
    alpha_i = 0.5 sigma^2 S_i^2 / dS^2 - 0.5 (r - q) S_i / dS,
    beta_i  = -sigma^2 S_i^2 / dS^2 - r,
    gamma_i = 0.5 sigma^2 S_i^2 / dS^2 + 0.5 (r - q) S_i / dS.

The theta time-stepping blends explicit and implicit evaluations of ``L``:

    (I - theta dtau L) V^{n+1} = (I + (1 - theta) dtau L) V^n.

    * theta = 0   -> fully explicit  (fast, but only conditionally stable:
                     needs dtau <= dS^2 / (sigma^2 S_max^2)).
    * theta = 1   -> fully implicit  (unconditionally stable, first-order in tau).
    * theta = 1/2 -> Crank-Nicolson  (unconditionally stable, second-order in
                     tau AND in S; the default and recommended scheme).

Each implicit step is a tridiagonal linear system solved in O(M) by the
Thomas algorithm. American options replace the linear solve by a *projected*
SOR sweep that enforces ``V >= payoff`` (the early-exercise constraint),
turning the equation into a linear complementarity problem.

The solver returns the full price surface ``V(S, t)`` for visualisation plus
the interpolated price at the spot ``S0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["FDResult", "solve", "FRAMEWORKS"]


# ---------------------------------------------------------------------------
# Tridiagonal (Thomas) solver
# ---------------------------------------------------------------------------

def thomas_solve(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                 rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system ``A x = rhs`` by the Thomas algorithm.

    ``diag`` is the main diagonal (length n); ``sub`` is the sub-diagonal
    (``sub[i]`` multiplies ``x[i-1]``, ``sub[0]`` unused); ``sup`` is the
    super-diagonal (``sup[i]`` multiplies ``x[i+1]``, ``sup[-1]`` unused).
    Runs in O(n) and is the workhorse of every implicit time step.
    """
    n = len(diag)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / m if i < n - 1 else 0.0
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def psor_solve(sub, diag, sup, rhs, constraint, omega=1.5,
               tol=1e-8, max_iter=10000):
    """Projected SOR solve of the linear complementarity problem.

    Finds ``x`` satisfying ``x >= constraint`` and, componentwise, either the
    tridiagonal equation holds or the constraint is active. This encodes the
    American early-exercise condition (value never below intrinsic). ``omega``
    is the over-relaxation factor (1 < omega < 2 accelerates convergence).
    """
    n = len(diag)
    x = np.maximum(rhs / diag, constraint)  # warm start
    for _ in range(max_iter):
        err = 0.0
        for i in range(n):
            lower = sub[i] * x[i - 1] if i > 0 else 0.0
            upper = sup[i] * x[i + 1] if i < n - 1 else 0.0
            gs = (rhs[i] - lower - upper) / diag[i]
            x_new = max(constraint[i], x[i] + omega * (gs - x[i]))
            err = max(err, abs(x_new - x[i]))
            x[i] = x_new
        if err < tol:
            break
    return x


# ---------------------------------------------------------------------------
# Instrument specification
# ---------------------------------------------------------------------------

@dataclass
class Instrument:
    """Everything the generic theta-scheme needs to know about an instrument.

    The solver is deliberately payoff-agnostic: an :class:`Instrument` supplies
    the spatial domain, the terminal payoff and the two Dirichlet boundary
    values as functions of time-to-maturity ``tau``. Barrier knock-outs simply
    shrink the domain and pin the value to zero at the barrier boundary;
    American options add an ``early_exercise`` intrinsic that activates PSOR.
    """

    S_min: float
    S_max: float
    terminal: Callable[[np.ndarray], np.ndarray]
    lower_bc: Callable[[float], float]
    upper_bc: Callable[[float], float]
    american: bool = False
    intrinsic: Callable[[np.ndarray], np.ndarray] | None = None
    label: str = ""


@dataclass
class FDResult:
    """Container for a finite-difference solve."""

    S: np.ndarray                 # spatial grid, shape (M+1,)
    t: np.ndarray                 # calendar-time grid 0..T, shape (N+1,)
    V: np.ndarray                 # surface V(S, t), shape (N+1, M+1)
    price: float                  # interpolated value at (S0, t=0)
    scheme: str
    stable: bool = True           # explicit-scheme stability flag
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instrument builders (organised "by framework")
# ---------------------------------------------------------------------------

def _make_vanilla(K, r, q, sigma, T, option_type, S_max):
    call = option_type == "call"

    def terminal(S):
        return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)

    if call:
        lower = lambda tau: 0.0
        upper = lambda tau: S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
    else:
        lower = lambda tau: K * np.exp(-r * tau)
        upper = lambda tau: 0.0
    return Instrument(0.0, S_max, terminal, lower, upper,
                      label=f"European {option_type}")


def _make_digital(K, r, q, sigma, T, option_type, cash, S_max):
    call = option_type == "call"

    def terminal(S):
        return np.where(S > K, cash, 0.0) if call else np.where(S < K, cash, 0.0)

    if call:
        lower = lambda tau: 0.0
        upper = lambda tau: cash * np.exp(-r * tau)
    else:
        lower = lambda tau: cash * np.exp(-r * tau)
        upper = lambda tau: 0.0
    return Instrument(0.0, S_max, terminal, lower, upper,
                      label=f"Cash-or-nothing {option_type}")


def _make_barrier(K, r, q, sigma, T, option_type, H, barrier_type, S_max):
    call = option_type == "call"

    def payoff(S):
        return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)

    if barrier_type == "up-and-out":
        # Domain [0, H]; knocked out (V=0) at the barrier H (upper boundary).
        S_lo, S_hi = 0.0, H
        lower = lambda tau: (0.0 if call else K * np.exp(-r * tau))
        upper = lambda tau: 0.0  # knock-out at barrier
    elif barrier_type == "down-and-out":
        # Domain [H, S_max]; knocked out (V=0) at the barrier H (lower boundary).
        S_lo, S_hi = H, S_max
        lower = lambda tau: 0.0  # knock-out at barrier
        upper = lambda tau: (S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
                             if call else 0.0)
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

    def terminal(S):
        return payoff(S)

    return Instrument(S_lo, S_hi, terminal, lower, upper,
                      label=f"{barrier_type} {option_type}")


def _make_american(K, r, q, sigma, T, option_type, S_max):
    call = option_type == "call"

    def payoff(S):
        return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)

    if call:
        lower = lambda tau: 0.0
        upper = lambda tau: S_max - K  # deep ITM American call ~ intrinsic
    else:
        lower = lambda tau: K          # deep ITM American put ~ intrinsic K - 0
        upper = lambda tau: 0.0
    return Instrument(0.0, S_max, payoff, lower, upper,
                      american=True, intrinsic=payoff,
                      label=f"American {option_type}")


def build_instrument(framework: str, p: dict, S_max: float) -> Instrument:
    """Dispatch to the right :class:`Instrument` builder for ``framework``."""
    K = p["K"]; r = p["r"]; q = p.get("q", 0.0)
    sigma = p["sigma"]; T = p["T"]; opt = p.get("option_type", "call")
    if framework == "vanilla":
        return _make_vanilla(K, r, q, sigma, T, opt, S_max)
    if framework == "digital":
        return _make_digital(K, r, q, sigma, T, opt, p.get("cash", 1.0), S_max)
    if framework == "barrier":
        return _make_barrier(K, r, q, sigma, T, opt, p["barrier"],
                             p.get("barrier_type", "up-and-out"), S_max)
    if framework == "american":
        return _make_american(K, r, q, sigma, T, opt, S_max)
    raise ValueError(f"Unknown framework: {framework!r}")


# ---------------------------------------------------------------------------
# Core theta-scheme solver
# ---------------------------------------------------------------------------

def solve(framework: str, params: dict, M: int = 200, N: int = 200,
          scheme: str = "crank-nicolson", S_max: float | None = None,
          rannacher: bool = True) -> FDResult:
    """Solve the pricing PDE for ``framework`` and return the price surface.

    Parameters
    ----------
    framework : "vanilla" | "barrier" | "digital" | "american".
    params    : dict with S0, K, r, sigma, T and, per framework, option_type,
                barrier / barrier_type, cash, q.
    M, N      : number of spatial (S) and temporal (tau) steps.
    scheme    : "explicit" (theta=0), "implicit" (theta=1) or
                "crank-nicolson" (theta=0.5, default).
    S_max     : upper truncation of the S domain (auto-chosen if None).
    rannacher : if True, replace the first two Crank-Nicolson steps by fully
                implicit ones. Crank-Nicolson barely damps the high-frequency
                error introduced by non-smooth payoffs (the digital step, the
                vanilla kink), producing spurious oscillations near the strike.
                Two implicit "Rannacher" start-up steps annihilate them while
                preserving second-order accuracy overall. No effect for the
                explicit / implicit schemes.

    Returns
    -------
    :class:`FDResult` with the grid, the surface ``V(S, t)`` and the spot price.
    """
    theta = {"explicit": 0.0, "implicit": 1.0, "crank-nicolson": 0.5}[scheme]

    S0 = params["S0"]; K = params["K"]; r = params["r"]
    sigma = params["sigma"]; T = params["T"]; q = params.get("q", 0.0)

    # Choose the domain. Cover the spot and strike with generous headroom so the
    # truncation boundary does not contaminate the interior solution.
    if S_max is None:
        S_max = 4.0 * max(S0, K)
    inst = build_instrument(framework, params, S_max)

    S = np.linspace(inst.S_min, inst.S_max, M + 1)
    dS = S[1] - S[0]
    dtau = T / N

    # Interior spatial operator coefficients L[V]_i (i = 1..M-1).
    Si = S[1:-1]
    a = 0.5 * sigma * sigma * Si * Si / (dS * dS)
    b = 0.5 * (r - q) * Si / dS
    alpha = a - b                       # coeff of V_{i-1}
    beta = -2.0 * a - r                 # coeff of V_i
    gamma = a + b                       # coeff of V_{i+1}

    # Explicit-scheme stability diagnostic (von Neumann, worst node).
    stable = True
    if scheme == "explicit":
        stable = bool(dtau <= 1.0 / np.max(2.0 * a + r)) if np.max(2.0 * a + r) > 0 else True

    def operators(th):
        """Return (LHS sub/diag/sup, RHS sub/diag/sup) for time-weight ``th``."""
        return (
            -th * dtau * alpha, 1.0 - th * dtau * beta, -th * dtau * gamma,
            (1.0 - th) * dtau * alpha, 1.0 + (1.0 - th) * dtau * beta,
            (1.0 - th) * dtau * gamma,
        )

    # Surface storage: row n corresponds to tau_n; we flip to calendar t later.
    V_tau = np.empty((N + 1, M + 1))
    # Cell-average the terminal payoff (project it onto the grid). Sampling a
    # discontinuous payoff (digital) or a kink (vanilla) at a single node
    # introduces an O(dS) bias exactly where accuracy matters most; averaging
    # the payoff over each grid cell [S_i - dS/2, S_i + dS/2] removes it, e.g.
    # a digital node sitting on the strike correctly gets cash/2.
    _sub = 21
    _off = ((np.arange(_sub) + 0.5) / _sub - 0.5) * dS
    _samples = S[:, None] + _off[None, :]
    V = np.asarray(inst.terminal(_samples), dtype=float).mean(axis=1)
    # Pin boundaries at tau = 0 to be consistent with the payoff endpoints.
    V[0] = inst.lower_bc(0.0)
    V[-1] = inst.upper_bc(0.0)
    V_tau[0] = V

    intrinsic_int = None
    if inst.american and inst.intrinsic is not None:
        intrinsic_int = inst.intrinsic(S[1:-1])

    for n in range(1, N + 1):
        tau_new = n * dtau
        tau_old = (n - 1) * dtau
        lo_new = inst.lower_bc(tau_new); hi_new = inst.upper_bc(tau_new)
        lo_old = inst.lower_bc(tau_old); hi_old = inst.upper_bc(tau_old)

        # Rannacher start-up: first two Crank-Nicolson steps run fully implicit.
        th = theta
        if scheme == "crank-nicolson" and rannacher and n <= 2:
            th = 1.0
        lhs_sub, lhs_diag, lhs_sup, rhs_sub, rhs_diag, rhs_sup = operators(th)

        Vi = V[1:-1]
        # RHS = (I + (1-theta) dtau L) V^n on the interior.
        rhs = rhs_diag * Vi
        rhs[1:] += rhs_sub[1:] * Vi[:-1]
        rhs[:-1] += rhs_sup[:-1] * Vi[1:]
        # Add the known Dirichlet boundary contributions. The explicit part
        # uses the old boundary value, the implicit part the new one.
        rhs[0] += (1.0 - th) * dtau * alpha[0] * lo_old + th * dtau * alpha[0] * lo_new
        rhs[-1] += (1.0 - th) * dtau * gamma[-1] * hi_old + th * dtau * gamma[-1] * hi_new

        if inst.american:
            new_int = psor_solve(lhs_sub, lhs_diag, lhs_sup, rhs, intrinsic_int)
        elif scheme == "explicit":
            new_int = rhs  # theta = 0 => LHS is identity
        else:
            new_int = thomas_solve(lhs_sub, lhs_diag, lhs_sup, rhs)

        V = np.empty(M + 1)
        V[0] = lo_new
        V[-1] = hi_new
        V[1:-1] = new_int
        V_tau[n] = V

    # Convert tau-indexed surface to calendar time t = T - tau (ascending t).
    # V_tau[n] is at tau_n = n*dtau => t = T - n*dtau. Row n=N is t=0 (today).
    t = T - np.arange(N + 1) * dtau      # descending: T .. 0
    order = np.argsort(t)                # ascending t
    V_surface = V_tau[order]
    t_sorted = t[order]

    price = float(np.interp(S0, S, V_tau[N])) if inst.S_min <= S0 <= inst.S_max else 0.0

    return FDResult(S=S, t=t_sorted, V=V_surface, price=price, scheme=scheme,
                    stable=stable,
                    meta={"M": M, "N": N, "S_max": inst.S_max,
                          "S_min": inst.S_min, "dS": dS, "dtau": dtau,
                          "label": inst.label, "theta": theta})


# Human-readable catalogue consumed by the web UI (organised by framework).
FRAMEWORKS = {
    "vanilla": {
        "name": "Black-Scholes PDE - Vanilla European",
        "pde": r"\frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0",
        "terminal": r"V(S,T) = \max(S-K,0)\ \text{(call)},\quad \max(K-S,0)\ \text{(put)}",
        "boundary": r"V(0,t)=0,\ V(S_{\max},t)=S_{\max}e^{-q\tau}-Ke^{-r\tau}\ \text{(call)}",
        "needs_barrier": False,
    },
    "barrier": {
        "name": "Barrier PDE - Knock-out (Dirichlet at barrier)",
        "pde": r"\frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0",
        "terminal": r"V(S,T)=\max(S-K,0)\ \text{for } S \text{ not knocked out}",
        "boundary": r"V(H,t)=0\ \text{at the barrier } H\ \text{(knock-out)}",
        "needs_barrier": True,
    },
    "digital": {
        "name": "Digital PDE - Cash-or-nothing",
        "pde": r"\frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0",
        "terminal": r"V(S,T)=Q\,\mathbf{1}\{S>K\}\ \text{(call)},\quad Q\,\mathbf{1}\{S<K\}\ \text{(put)}",
        "boundary": r"V(0,t)=0,\ V(S_{\max},t)=Qe^{-r\tau}\ \text{(call)}",
        "needs_barrier": False,
    },
    "american": {
        "name": "American PDE - Linear Complementarity (PSOR)",
        "pde": r"\min\!\Big(\!-\big(V_t+\tfrac12\sigma^2 S^2 V_{SS}+rSV_S-rV\big),\ V-\text{payoff}\Big)=0",
        "terminal": r"V(S,T)=\max(S-K,0)\ \text{(call)},\quad \max(K-S,0)\ \text{(put)}",
        "boundary": r"V(S,t)\ge \text{intrinsic value everywhere (early exercise)}",
        "needs_barrier": False,
    },
}


if __name__ == "__main__":
    import analytics
    p = {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0,
         "option_type": "call"}
    for sch in ("explicit", "implicit", "crank-nicolson"):
        M = 200
        N = 200 if sch != "explicit" else 20000
        res = solve("vanilla", p, M=M, N=N, scheme=sch)
        bs = analytics.bs_price(p["S0"], p["K"], p["r"], p["sigma"], p["T"], "call")
        print(f"{sch:15s} FD={res.price:.4f}  BS={bs:.4f}  "
              f"err={abs(res.price - bs):.2e}  stable={res.stable}")
