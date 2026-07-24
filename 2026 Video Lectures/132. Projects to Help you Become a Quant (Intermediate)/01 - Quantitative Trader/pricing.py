"""
pricing.py
==========

Option pricing library for the Market Making Simulator.

This module implements two European option pricers and their greeks:

1. Black-Scholes-Merton (BSM)
   - Assumes the underlying follows a Geometric Brownian Motion (GBM):
         dS = mu*S*dt + sigma*S*dW
   - Closed-form price and greeks (delta, gamma, vega, theta).

2. Merton Jump-Diffusion (MJD)
   - Assumes the underlying follows GBM *plus* a compound-Poisson jump
     component: between jumps the log-price diffuses; jumps arrive at rate
     `lam` (per year) and multiply the price by exp(Y), where
     Y ~ Normal(mu_j, sigma_j^2).
   - Merton (1976) showed the option price is a Poisson-weighted sum of
     Black-Scholes prices, each evaluated with a jump-adjusted spot,
     volatility and risk-free rate. We truncate the infinite series.

Why this matters for the simulator
-----------------------------------
A market maker must convert a *belief about dynamics* into a *quote*. The
pricer is that belief. If the MM prices with Black-Scholes but the real world
jumps (Merton), the MM's delta hedge and fair value are systematically wrong,
and the "guaranteed" spread capture can be overwhelmed by jump losses. This
file is deliberately model-agnostic: the simulator asks a pricer for
(price, delta) and does not care which model produced them.

All prices are for European options; time is in years; rates/vols are annual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------

def _bs_d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    """Return the Black-Scholes d1, d2 terms.

    d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    """
    # Guard the degenerate limits (expiry, zero vol) that would divide by zero.
    if T <= 0 or sigma <= 0:
        # At/after expiry (or with no vol) the option is a deterministic
        # forward-intrinsic payoff; d1/d2 collapse to +/- infinity depending
        # on moneyness. Returning large signed numbers reproduces that limit.
        eps = 1e-12
        T = max(T, eps)
        sigma = max(sigma, eps)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike
    r : continuously-compounded risk-free rate (annual)
    sigma : annualised volatility of returns
    T : time to maturity in years
    option_type : "call" or "put"
    """
    if T <= 0:
        # Intrinsic value at expiry.
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)

    d1, d2 = _bs_d1_d2(S, K, r, sigma, T)
    disc = math.exp(-r * T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(price)


def bs_delta(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Black-Scholes delta: dPrice/dS. The market maker hedges with this."""
    if T <= 0:
        # Step function of moneyness at expiry.
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _bs_d1_d2(S, K, r, sigma, T)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float,
              option_type: str = "call") -> dict:
    """Return the standard first/second-order greeks used for risk display.

    delta : dV/dS
    gamma : d2V/dS2 (same for calls and puts)
    vega  : dV/dsigma (per 1.0 change in vol, i.e. per 100 vol points)
    theta : dV/dt (per year; negative for long options typically)
    """
    if T <= 0 or sigma <= 0:
        return {
            "delta": bs_delta(S, K, r, sigma, T, option_type),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    d1, d2 = _bs_d1_d2(S, K, r, sigma, T)
    sqrtT = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc = math.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT

    if option_type == "call":
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * disc * norm.cdf(d2))
    else:
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * disc * norm.cdf(-d2))

    return {
        "delta": bs_delta(S, K, r, sigma, T, option_type),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


# ---------------------------------------------------------------------------
# Merton Jump-Diffusion
# ---------------------------------------------------------------------------

def merton_price(S: float, K: float, r: float, sigma: float, T: float,
                 lam: float, mu_j: float, sigma_j: float,
                 option_type: str = "call", n_terms: int = 60) -> float:
    """Merton (1976) jump-diffusion price of a European option.

    The model adds log-normal jumps to GBM. Jumps arrive as a Poisson
    process with intensity `lam` (jumps per year). Each jump multiplies the
    spot by exp(Y) with Y ~ N(mu_j, sigma_j^2).

    Merton's insight: conditional on exactly `n` jumps occurring before
    maturity, the terminal price is again log-normal, so the option is worth
    a Black-Scholes price with a shifted spot, variance and rate. Averaging
    over the Poisson distribution of `n` gives:

        Price = sum_{n=0}^inf  exp(-lam' T) (lam' T)^n / n!  *  BS(S_n, sigma_n, r_n)

    where, writing k = exp(mu_j + sigma_j^2/2) - 1 (mean jump size),
        lam'    = lam * (1 + k)          (risk-neutral jump intensity)
        sigma_n^2 = sigma^2 + n*sigma_j^2 / T
        r_n     = r - lam*k + n*(mu_j + sigma_j^2/2) / T

    Parameters mirror `bs_price` with the extra jump parameters. The infinite
    sum is truncated at `n_terms`, which is far beyond machine precision for
    realistic intensities.
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)

    # If there are effectively no jumps, fall back to Black-Scholes exactly.
    if lam <= 0:
        return bs_price(S, K, r, sigma, T, option_type)

    # Mean proportional jump size k = E[e^Y] - 1.
    k = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    lam_prime = lam * (1.0 + k)
    lamT = lam_prime * T

    price = 0.0
    # Poisson weights exp(-lamT) lamT^n / n! computed iteratively for stability.
    log_lamT = math.log(lamT) if lamT > 0 else float("-inf")
    for n in range(n_terms):
        # log Poisson weight = -lamT + n*log(lamT) - log(n!)
        log_w = -lamT + n * log_lamT - math.lgamma(n + 1)
        weight = math.exp(log_w)
        if weight < 1e-15 and n > lamT + 10:
            break  # tail is negligible

        # Variance and drift adjustments given n jumps.
        sigma_n = math.sqrt(sigma * sigma + n * sigma_j * sigma_j / T)
        r_n = r - lam * k + n * (mu_j + 0.5 * sigma_j * sigma_j) / T
        price += weight * bs_price(S, K, r_n, sigma_n, T, option_type)

    return float(price)


def merton_delta(S: float, K: float, r: float, sigma: float, T: float,
                 lam: float, mu_j: float, sigma_j: float,
                 option_type: str = "call", n_terms: int = 60) -> float:
    """Merton jump-diffusion delta.

    Because each Poisson term is a Black-Scholes price in `S`, the delta is
    the same Poisson-weighted sum of the corresponding Black-Scholes deltas
    (the weights and vol/rate adjustments do not depend on S).
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    if lam <= 0:
        return bs_delta(S, K, r, sigma, T, option_type)

    k = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    lam_prime = lam * (1.0 + k)
    lamT = lam_prime * T

    delta = 0.0
    log_lamT = math.log(lamT) if lamT > 0 else float("-inf")
    for n in range(n_terms):
        log_w = -lamT + n * log_lamT - math.lgamma(n + 1)
        weight = math.exp(log_w)
        if weight < 1e-15 and n > lamT + 10:
            break
        sigma_n = math.sqrt(sigma * sigma + n * sigma_j * sigma_j / T)
        r_n = r - lam * k + n * (mu_j + 0.5 * sigma_j * sigma_j) / T
        delta += weight * bs_delta(S, K, r_n, sigma_n, T, option_type)

    return float(delta)


# ---------------------------------------------------------------------------
# Unified pricer interface used by the simulator
# ---------------------------------------------------------------------------

@dataclass
class PricingModel:
    """A thin adapter that gives the simulator a uniform (price, delta) API.

    The simulator only ever calls `.price(...)` and `.delta(...)`; it does not
    need to know whether the numbers come from Black-Scholes or Merton. This
    is what lets us mismatch the *pricing* model against the *true* market
    path model and observe the resulting model risk.
    """

    name: str  # "black_scholes" or "merton"
    r: float = 0.0
    lam: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    def price(self, S: float, K: float, sigma: float, T: float,
              option_type: str = "call") -> float:
        if self.name == "merton":
            return merton_price(S, K, self.r, sigma, T,
                                 self.lam, self.mu_j, self.sigma_j, option_type)
        return bs_price(S, K, self.r, sigma, T, option_type)

    def delta(self, S: float, K: float, sigma: float, T: float,
              option_type: str = "call") -> float:
        if self.name == "merton":
            return merton_delta(S, K, self.r, sigma, T,
                                self.lam, self.mu_j, self.sigma_j, option_type)
        return bs_delta(S, K, self.r, sigma, T, option_type)


if __name__ == "__main__":
    # Quick self-check / put-call parity sanity test.
    S, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 0.5
    c = bs_price(S, K, r, sigma, T, "call")
    p = bs_price(S, K, r, sigma, T, "put")
    parity = c - p - (S - K * math.exp(-r * T))
    print(f"BS call={c:.4f} put={p:.4f} parity_resid={parity:.2e}")

    mc = merton_price(S, K, r, sigma, T, lam=1.0, mu_j=-0.05, sigma_j=0.10)
    print(f"Merton call={mc:.4f} (>= BS call: {mc >= c})")
    print(f"BS delta call={bs_delta(S, K, r, sigma, T, 'call'):.4f}")
    print(f"Merton delta call={merton_delta(S, K, r, sigma, T, 1.0, -0.05, 0.10):.4f}")
