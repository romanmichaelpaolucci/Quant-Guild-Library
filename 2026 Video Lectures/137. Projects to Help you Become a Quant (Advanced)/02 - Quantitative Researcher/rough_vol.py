"""
rough_vol.py
============

Simulation of a **rough volatility** model (rough Bergomi) and the implied
volatility smile / skew it produces.

The idea in one paragraph
-------------------------
Classical stochastic-volatility models (Heston, SABR, ...) drive the variance
with a *standard* Brownian motion, whose paths have Holder regularity ~1/2.
Gatheral, Jaisson & Rosenbaum (2018), "Volatility is rough", measured realised
volatility across thousands of assets and found the log-volatility behaves like
a fractional Brownian motion with Hurst exponent ``H`` far *below* 1/2 (often
H ~ 0.1). In other words volatility is much *rougher* than Brownian motion.
Bayer, Friz & Gatheral (2016) turned this observation into the **rough Bergomi**
pricing model, which -- with only a handful of parameters -- reproduces the
steep, negative, short-maturity ATM skew of index options that classical
diffusions structurally cannot generate.

The model
---------
Let ``W`` be a standard Brownian motion. Define the (Riemann-Liouville)
Volterra / fractional process

    Y_t = integral_0^t  K(t - s) dW_s ,      K(u) = sqrt(2H) * u^{H - 1/2}.

The normalisation ``sqrt(2H)`` makes ``Var(Y_t) = t^{2H}`` exactly (proved in
``_volterra_cov`` below). For H < 1/2 the kernel is singular at u = 0, which is
precisely what makes ``Y`` -- and hence the variance -- rough.

The instantaneous variance is a log-normal (Bergomi) functional of ``Y``:

    v_t = xi0 * exp( eta * Y_t  -  0.5 * eta^2 * t^{2H} ) ,

where ``xi0`` is the (flat) forward variance curve and ``eta`` the vol-of-vol.
The Wick / martingale correction ``-0.5 eta^2 Var(Y_t)`` makes ``E[v_t] = xi0``.

The spot follows, with leverage correlation ``rho`` (usually strongly negative)

    dS_t / S_t = sqrt(v_t) ( rho dW_t + sqrt(1 - rho^2) dB_t ) ,

where ``B`` is a Brownian motion independent of ``W``.

How we simulate
---------------
We use an **exact** (up to the time grid) scheme: on the grid t_1 < ... < t_n we
build the joint Gaussian covariance of the vector ``(W_{t_i}, Y_{t_i})`` -- both
are Wiener integrals of the same ``W`` so all covariances are available in
closed form (see ``build_joint_covariance``) -- and sample it via a Cholesky
factor. This avoids the discretisation bias of naive Euler schemes near the
singularity and is transparently correct for a teaching illustration. For very
long grids one would prefer the hybrid scheme of Bennedsen-Lunde-Pakkanen
(2017) or an FFT of the covariance; Cholesky is O(n^3) but n is small here.

References
----------
* J. Gatheral, T. Jaisson, M. Rosenbaum, "Volatility is rough",
  Quantitative Finance 18(6), 2018.
* C. Bayer, P. Friz, J. Gatheral, "Pricing under rough volatility",
  Quantitative Finance 16(6), 2016.
* M. Bennedsen, A. Lunde, M. Pakkanen, "Hybrid scheme for Brownian
  semistationary processes", Finance & Stochastics, 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers (used to turn Monte-Carlo prices into implied vols)
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes price of a European call (no dividends)."""
    if T <= 0 or sigma <= 0:
        return float(max(S - K * math.exp(-r * T), 0.0))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def implied_vol_call(price: float, S: float, K: float, T: float,
                     r: float = 0.0) -> float:
    """Invert Black-Scholes for the implied volatility of a call price.

    Uses Brent's method on a generous bracket. Returns ``nan`` when the price
    is outside the no-arbitrage bounds (below intrinsic or above spot), which
    can happen for deep wings estimated from a finite Monte-Carlo sample.
    """
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12 or price >= S:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_call_price(S, K, T, sigma, r) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, maxiter=200, xtol=1e-8))
    except (ValueError, RuntimeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Volterra covariance structure (closed form)
# ---------------------------------------------------------------------------

def _volterra_cov(s: float, t: float, H: float) -> float:
    """Cov(Y_s, Y_t) for the Riemann-Liouville process with kernel
    K(u) = sqrt(2H) u^{H-1/2}.

    Derivation (s <= t). With Ito isometry,
        Cov(Y_s, Y_t) = 2H * integral_0^s (s-u)^{H-1/2} (t-u)^{H-1/2} du.
    Substituting u = s w and using Euler's integral for the Gauss
    hypergeometric function 2F1 gives the closed form

        Cov(Y_s, Y_t) = 2H/(H+1/2) * s^{H+1/2} t^{H-1/2}
                        * 2F1(1/2 - H, 1; H + 3/2; s/t).

    A sanity check: setting s = t collapses 2F1(.,.;.;1) to (H+1/2)/(2H) and
    hence ``Var(Y_t) = t^{2H}`` exactly (see the module tests).
    """
    if s <= 0.0 or t <= 0.0:
        return 0.0
    lo, hi = (s, t) if s <= t else (t, s)
    coeff = 2.0 * H / (H + 0.5)
    return float(coeff * lo ** (H + 0.5) * hi ** (H - 0.5)
                 * hyp2f1(0.5 - H, 1.0, H + 1.5, lo / hi))


def build_joint_covariance(times: np.ndarray, H: float) -> np.ndarray:
    """Assemble the (2n x 2n) covariance of ``(W_{t_i}, Y_{t_i})``.

    Blocks, using Ito isometry for Wiener integrals of the same ``W``:
      * Cov(W_s, W_t)  = min(s, t).
      * Cov(Y_s, Y_t)  = ``_volterra_cov`` above.
      * Cov(Y_{t_i}, W_{t_j}) = sqrt(2H)/(H+1/2)
            * ( t_i^{H+1/2} - (t_i - m)^{H+1/2} ),   m = min(t_i, t_j),
        obtained by integrating the kernel against the indicator of [0, m].
    """
    n = len(times)
    cov = np.zeros((2 * n, 2 * n))

    # W-W block.
    ti = times[:, None]
    tj = times[None, :]
    cov[:n, :n] = np.minimum(ti, tj)

    # Y-Y block.
    yy = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            c = _volterra_cov(times[i], times[j], H)
            yy[i, j] = c
            yy[j, i] = c
    cov[n:, n:] = yy

    # Y-W cross block.
    c0 = math.sqrt(2.0 * H) / (H + 0.5)
    m = np.minimum(ti, tj)                     # m[i, j] = min(t_i, t_j)
    yw = c0 * (times[:, None] ** (H + 0.5) - (times[:, None] - m) ** (H + 0.5))
    cov[n:, :n] = yw
    cov[:n, n:] = yw.T
    return cov


# ---------------------------------------------------------------------------
# Rough Bergomi simulation
# ---------------------------------------------------------------------------

@dataclass
class RoughBergomi:
    """Rough Bergomi model configuration and simulator.

    Parameters
    ----------
    H : Hurst exponent in (0, 1/2). Smaller = rougher.
    eta : vol-of-vol (how strongly Y moves the variance).
    rho : spot/vol correlation (leverage), typically strongly negative.
    xi0 : flat forward variance (so at-the-money vol ~ sqrt(xi0)).
    S0 : initial spot.
    r : risk-free rate (kept at 0 for clean forward-measure illustrations).
    """

    H: float = 0.1
    eta: float = 1.5
    rho: float = -0.7
    xi0: float = 0.04
    S0: float = 100.0
    r: float = 0.0

    # Cached Cholesky factor keyed by the grid it was built for.
    _chol: np.ndarray | None = field(default=None, repr=False, compare=False)
    _grid_key: tuple | None = field(default=None, repr=False, compare=False)

    def _cholesky(self, times: np.ndarray) -> np.ndarray:
        key = (len(times), float(times[-1]), self.H)
        if self._chol is None or self._grid_key != key:
            cov = build_joint_covariance(times, self.H)
            # Symmetric PSD matrix; add a whisper of jitter for numerical PD.
            jitter = 1e-12 * np.trace(cov) / cov.shape[0]
            self._chol = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            self._grid_key = key
        return self._chol

    def simulate(self, T: float, n_steps: int, n_paths: int,
                 seed: int | None = None):
        """Simulate variance and price paths on [0, T].

        Returns a dict with keys:
          ``t``      : time grid including 0, shape (n_steps + 1,)
          ``S``      : price paths, shape (n_paths, n_steps + 1)
          ``v``      : variance paths, shape (n_paths, n_steps + 1)
          ``Y``      : Volterra process paths, shape (n_paths, n_steps + 1)
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        times = np.linspace(dt, T, n_steps)          # strictly positive grid
        L = self._cholesky(times)
        n = n_steps

        # Draw the joint Gaussian (W, Y) at the grid points for every path.
        z = rng.standard_normal((2 * n, n_paths))
        joint = L @ z                                # shape (2n, n_paths)
        W = joint[:n, :].T                           # (n_paths, n)
        Y = joint[n:, :].T                           # (n_paths, n)

        # Prepend the t = 0 state (W_0 = Y_0 = 0).
        zeros = np.zeros((n_paths, 1))
        W = np.hstack([zeros, W])
        Y = np.hstack([zeros, Y])
        t_full = np.concatenate([[0.0], times])

        # Variance (Bergomi log-normal functional of Y).
        var_correction = 0.5 * self.eta ** 2 * t_full ** (2.0 * self.H)
        v = self.xi0 * np.exp(self.eta * Y - var_correction[None, :])

        # Spot via log-Euler. dW increments come from the sampled W path; the
        # orthogonal component uses fresh independent normals.
        dW = np.diff(W, axis=1)                       # (n_paths, n)
        dB = rng.standard_normal((n_paths, n)) * math.sqrt(dt)
        sqrt_v = np.sqrt(np.maximum(v[:, :-1], 0.0))  # left-point variance
        incr = (self.r * dt
                - 0.5 * v[:, :-1] * dt
                + sqrt_v * (self.rho * dW + math.sqrt(1.0 - self.rho ** 2) * dB))
        log_paths = np.log(self.S0) + np.cumsum(incr, axis=1)
        S = np.hstack([np.full((n_paths, 1), self.S0), np.exp(log_paths)])

        return {"t": t_full, "S": S, "v": v, "Y": Y}

    # ------------------------------------------------------------------
    # Implied-volatility smile / skew from Monte-Carlo option prices
    # ------------------------------------------------------------------

    def implied_smile(self, T: float, strikes: np.ndarray,
                      n_steps: int = 100, n_paths: int = 40000,
                      seed: int | None = 0):
        """Monte-Carlo European-call prices across ``strikes`` at maturity ``T``,
        inverted to Black-Scholes implied vols.

        Returns dict with ``strikes``, ``log_moneyness`` (ln K / S0),
        ``prices``, ``iv`` and Monte-Carlo standard errors ``iv_se_proxy``.
        """
        sim = self.simulate(T, n_steps, n_paths, seed=seed)
        ST = sim["S"][:, -1]
        disc = math.exp(-self.r * T)

        strikes = np.asarray(strikes, dtype=float)
        prices = np.empty_like(strikes)
        stderr = np.empty_like(strikes)
        for i, K in enumerate(strikes):
            payoff = disc * np.maximum(ST - K, 0.0)
            prices[i] = payoff.mean()
            stderr[i] = payoff.std(ddof=1) / math.sqrt(len(payoff))

        iv = np.array([implied_vol_call(p, self.S0, K, T, self.r)
                       for p, K in zip(prices, strikes)])
        return {
            "strikes": strikes,
            "log_moneyness": np.log(strikes / self.S0),
            "prices": prices,
            "price_se": stderr,
            "iv": iv,
        }

    def atm_skew_term_structure(self, maturities, n_steps: int = 100,
                                n_paths: int = 40000, seed: int | None = 0):
        """ATM volatility skew d(IV)/d(log-moneyness) at k = 0 for each maturity.

        Rough volatility's signature is that this skew *blows up* like
        ``T^{H - 1/2}`` as ``T -> 0``, matching short-dated index options -- a
        power law that classical diffusions (with a finite short-maturity skew)
        cannot reproduce.
        """
        maturities = np.asarray(maturities, dtype=float)
        skews = np.empty_like(maturities)
        # A tight symmetric strike triple around ATM gives a stable slope.
        for i, T in enumerate(maturities):
            eps = 0.05
            strikes = self.S0 * np.exp(np.array([-eps, 0.0, eps]))
            smile = self.implied_smile(T, strikes, n_steps=n_steps,
                                        n_paths=n_paths, seed=seed)
            k = smile["log_moneyness"]
            iv = smile["iv"]
            # Central finite difference in log-moneyness.
            skews[i] = (iv[2] - iv[0]) / (k[2] - k[0])
        return {"maturities": maturities, "skew": skews}


def _smoke_test() -> None:
    """Tiny headless self-check exercised by the test harness and __main__."""
    # Var(Y_t) = t^{2H} check via the closed-form covariance.
    for H in (0.1, 0.3):
        for t in (0.25, 1.0, 2.0):
            assert abs(_volterra_cov(t, t, H) - t ** (2 * H)) < 1e-9

    model = RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    sim = model.simulate(T=1.0, n_steps=50, n_paths=200, seed=1)
    assert np.all(np.isfinite(sim["S"])) and np.all(sim["S"] > 0)
    assert np.all(np.isfinite(sim["v"])) and np.all(sim["v"] > 0)

    strikes = model.S0 * np.exp(np.linspace(-0.2, 0.2, 7))
    smile = model.implied_smile(T=0.1, strikes=strikes, n_steps=50,
                                n_paths=8000, seed=2)
    finite_iv = smile["iv"][np.isfinite(smile["iv"])]
    assert finite_iv.size >= 5, "smile inversion produced too few finite IVs"
    print("[rough_vol] OK  "
          f"mean IV~{np.nanmean(smile['iv']):.3f}  "
          f"S_T mean={sim['S'][:, -1].mean():.2f}")


if __name__ == "__main__":
    _smoke_test()
