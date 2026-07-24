"""
mc_pricer.py
============

Monte-Carlo (MC) pricing engine — the *independent* numerical check on the
finite-difference PDE solver.

Why price the same option two completely different ways? The PDE and the
expectation are two faces of the same coin: the Feynman-Kac theorem says the
solution of the Black-Scholes PDE equals a discounted risk-neutral
expectation of the payoff,

    V(S_0, 0) = e^{-rT} E^Q[ payoff(S_T, path) | S_0 ].

Finite differences attack the left-hand (PDE) view; Monte-Carlo attacks the
right-hand (expectation) view by simulating the risk-neutral GBM

    S_{t+dt} = S_t * exp[(r - q - sigma^2/2) dt + sigma sqrt(dt) Z],  Z ~ N(0,1).

If both are implemented correctly they must agree to within Monte-Carlo
sampling error, which we quantify with a 95% confidence interval. Divergence
beyond the CI is a red flag that one of the two engines is wrong.

Each pricer returns a :class:`MCResult` with the price, the standard error and
the CI so the dashboard can show whether FD lands inside the MC interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MCResult", "price"]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass
class MCResult:
    """Result of a Monte-Carlo pricing run."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    meta: dict


def _discounted_stats(payoffs: np.ndarray, discount: float, n_paths: int,
                      n_steps: int, meta: dict) -> MCResult:
    """Turn an array of (undiscounted) payoffs into a :class:`MCResult`."""
    disc_payoffs = discount * payoffs
    price = float(disc_payoffs.mean())
    se = float(disc_payoffs.std(ddof=1) / np.sqrt(len(disc_payoffs)))
    return MCResult(price=price, std_error=se,
                    ci_low=price - _Z95 * se, ci_high=price + _Z95 * se,
                    n_paths=n_paths, n_steps=n_steps, meta=meta)


# ---------------------------------------------------------------------------
# Terminal-only pricers (vanilla, digital) — need one draw of S_T
# ---------------------------------------------------------------------------

def _terminal_prices(S0, r, q, sigma, T, n_paths, rng, antithetic=True):
    """Draw ``n_paths`` risk-neutral terminal prices ``S_T`` in one shot.

    Uses antithetic variates (pairing ``Z`` with ``-Z``) to halve variance for
    free. Only the terminal value is needed for path-independent payoffs.
    """
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma * sigma) * T
    diff = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diff)


def _price_vanilla(p, n_paths, rng):
    S0, K, r, sigma, T = p["S0"], p["K"], p["r"], p["sigma"], p["T"]
    q = p.get("q", 0.0); call = p.get("option_type", "call") == "call"
    ST = _terminal_prices(S0, r, q, sigma, T, n_paths, rng)
    payoff = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    return _discounted_stats(payoff, np.exp(-r * T), n_paths, 1,
                             {"instrument": "vanilla"})


def _price_digital(p, n_paths, rng):
    S0, K, r, sigma, T = p["S0"], p["K"], p["r"], p["sigma"], p["T"]
    q = p.get("q", 0.0); call = p.get("option_type", "call") == "call"
    cash = p.get("cash", 1.0)
    ST = _terminal_prices(S0, r, q, sigma, T, n_paths, rng)
    hit = (ST > K) if call else (ST < K)
    payoff = np.where(hit, cash, 0.0)
    return _discounted_stats(payoff, np.exp(-r * T), n_paths, 1,
                             {"instrument": "digital"})


# ---------------------------------------------------------------------------
# Path-dependent pricer (barrier) — needs the whole path, monitored discretely
# ---------------------------------------------------------------------------

def _price_barrier(p, n_paths, rng, n_steps=252):
    """Price a knock-out barrier option by simulating monitored GBM paths.

    The barrier is checked at each of ``n_steps`` time points (discrete
    monitoring). A path that ever touches or crosses the barrier is knocked
    out and pays zero. Finer monitoring (more steps) converges to the
    continuously-monitored analytic price. Paths are generated in blocks to
    keep memory bounded for large ``n_paths``.
    """
    S0, K, r, sigma, T = p["S0"], p["K"], p["r"], p["sigma"], p["T"]
    q = p.get("q", 0.0); call = p.get("option_type", "call") == "call"
    H = p["barrier"]; btype = p.get("barrier_type", "up-and-out")
    up = btype == "up-and-out"

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    block = 20000
    disc_payoffs = np.empty(n_paths)
    done = 0
    while done < n_paths:
        m = min(block, n_paths - done)
        S = np.full(m, S0)
        alive = np.ones(m, dtype=bool)
        # Immediate knock-out if the spot already sits beyond the barrier.
        if up:
            alive &= S0 < H
        else:
            alive &= S0 > H
        for _ in range(n_steps):
            z = rng.standard_normal(m)
            S = S * np.exp(drift + vol * z)
            if up:
                alive &= S < H
            else:
                alive &= S > H
        payoff = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
        payoff = np.where(alive, payoff, 0.0)
        disc_payoffs[done:done + m] = discount * payoff
        done += m

    price = float(disc_payoffs.mean())
    se = float(disc_payoffs.std(ddof=1) / np.sqrt(n_paths))
    return MCResult(price=price, std_error=se,
                    ci_low=price - _Z95 * se, ci_high=price + _Z95 * se,
                    n_paths=n_paths, n_steps=n_steps,
                    meta={"instrument": "barrier", "barrier_type": btype})


# ---------------------------------------------------------------------------
# American pricer — Longstaff-Schwartz least-squares Monte-Carlo
# ---------------------------------------------------------------------------

def _price_american(p, n_paths, rng, n_steps=50):
    """Price an American option by Longstaff-Schwartz (2001) LSM.

    American options require an optimal-exercise *policy*, which plain MC cannot
    express. LSM approximates the continuation value at each exercise date by
    regressing discounted future cash-flows on a polynomial basis of the spot,
    then exercises whenever the immediate payoff beats the fitted continuation.
    This is the standard MC analogue of the PDE's free-boundary (PSOR) solve.
    """
    S0, K, r, sigma, T = p["S0"], p["K"], p["r"], p["sigma"], p["T"]
    q = p.get("q", 0.0); call = p.get("option_type", "call") == "call"

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * dt)

    # Simulate the full path matrix (n_paths x n_steps+1) with antithetics.
    half = (n_paths + 1) // 2
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)[:n_paths]
    log_incr = drift + vol * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1)
    S = S0 * np.exp(log_paths)

    def payoff(s):
        return np.maximum(s - K, 0.0) if call else np.maximum(K - s, 0.0)

    # Cash-flow = payoff at the currently-optimal stopping time, discounted back
    # one step at a time as we roll from expiry to today.
    cash = payoff(S[:, -1])
    for t in range(n_steps - 1, 0, -1):
        cash *= disc  # discount previous cash-flow one step toward this date
        exercise = payoff(S[:, t])
        itm = exercise > 0
        if itm.sum() >= 3:
            x = S[itm, t]
            y = cash[itm]
            # Regress continuation value on {1, x, x^2}.
            A = np.vstack([np.ones_like(x), x, x * x]).T
            coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
            continuation = A @ coeff
            do_ex = exercise[itm] > continuation
            idx = np.where(itm)[0][do_ex]
            cash[idx] = exercise[itm][do_ex]
    cash *= disc  # final discount from t=1 to t=0

    price = float(cash.mean())
    se = float(cash.std(ddof=1) / np.sqrt(n_paths))
    return MCResult(price=price, std_error=se,
                    ci_low=price - _Z95 * se, ci_high=price + _Z95 * se,
                    n_paths=n_paths, n_steps=n_steps,
                    meta={"instrument": "american", "method": "Longstaff-Schwartz"})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def price(framework: str, params: dict, n_paths: int = 100000,
          n_steps: int = 252, seed: int | None = 42) -> MCResult:
    """Monte-Carlo price for ``framework`` with a 95% confidence interval.

    ``n_steps`` is used by the path-dependent (barrier) and American engines;
    the terminal-only vanilla/digital pricers ignore it. A fixed ``seed`` makes
    results reproducible for the smoke tests.
    """
    rng = np.random.default_rng(seed)
    if framework == "vanilla":
        return _price_vanilla(params, n_paths, rng)
    if framework == "digital":
        return _price_digital(params, n_paths, rng)
    if framework == "barrier":
        return _price_barrier(params, n_paths, rng, n_steps=n_steps)
    if framework == "american":
        # Fewer exercise dates suffice; cap for speed/stability of regression.
        return _price_american(params, n_paths, rng, n_steps=min(n_steps, 50))
    raise ValueError(f"Unknown framework: {framework!r}")


if __name__ == "__main__":
    import analytics
    p = {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0,
         "option_type": "call"}
    mc = price("vanilla", p, n_paths=200000)
    bs = analytics.bs_price(100, 100, 0.05, 0.2, 1.0, "call")
    print(f"MC vanilla={mc.price:.4f} +/- {mc.std_error:.4f} "
          f"[{mc.ci_low:.4f}, {mc.ci_high:.4f}]  BS={bs:.4f}")
