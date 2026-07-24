"""
analytics.py
============

Closed-form (analytic) option prices used as an *independent ground truth*
against which the finite-difference (FD) and Monte-Carlo (MC) engines are
checked.

Every instrument priced numerically in this project has, where one exists, a
closed-form counterpart here:

* Vanilla European call/put ................ Black-Scholes-Merton (1973).
* Cash-or-nothing digital call/put ......... discounted N(d2) (Reiner-Rubinstein).
* Knock-out barrier call/put ............... Merton (1973) / Reiner-Rubinstein (1991)
                                             analytic barrier formulas, computed as
                                             ``vanilla - knock_in``.
* American call (no dividends) ............. equals the European call (never
                                             optimal to exercise early), so the
                                             Black-Scholes call is the closed form.
* American put ............................. no closed form (left to FD/MC).

All models assume the risk-neutral geometric Brownian motion

    dS = (r - q) S dt + sigma S dW,

with continuous dividend yield ``q`` (default 0). Time is in years; ``r``,
``q`` and ``sigma`` are annualised. These formulas are the yardstick: if the
FD PDE solver and the MC simulator are both correct they must reproduce the
numbers below to within discretisation / sampling error.
"""

from __future__ import annotations

import math

from scipy.stats import norm

__all__ = [
    "bs_price",
    "bs_greeks",
    "digital_cash_or_nothing",
    "barrier_price",
    "american_call_analytic",
    "analytic_price",
]


# ---------------------------------------------------------------------------
# Black-Scholes-Merton vanilla
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float):
    """Return the Black-Scholes ``d1``, ``d2`` auxiliary quantities.

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    """
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European vanilla option.

    This is the analytic solution of the Black-Scholes PDE

        V_t + 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0

    with terminal condition ``V(S, T) = max(S - K, 0)`` (call) or
    ``max(K - S, 0)`` (put). The FD solver should reproduce it for the vanilla
    framework.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)

    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if option_type == "call":
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    return float(price)


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float,
              option_type: str = "call", q: float = 0.0) -> dict:
    """Return delta, gamma, vega, theta and rho for a European vanilla option."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    sqrtT = math.sqrt(T)
    pdf = norm.pdf(d1)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    gamma = df_q * pdf / (S * sigma * sqrtT)
    vega = S * df_q * pdf * sqrtT
    if option_type == "call":
        delta = df_q * norm.cdf(d1)
        theta = (-S * df_q * pdf * sigma / (2 * sqrtT)
                 - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1))
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2 * sqrtT)
                 + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1))
        rho = -K * T * df_r * norm.cdf(-d2)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


# ---------------------------------------------------------------------------
# Cash-or-nothing digital
# ---------------------------------------------------------------------------

def digital_cash_or_nothing(S: float, K: float, r: float, sigma: float,
                            T: float, option_type: str = "call",
                            cash: float = 1.0, q: float = 0.0) -> float:
    """Price of a cash-or-nothing digital that pays ``cash`` if it expires ITM.

    A cash-or-nothing call pays ``cash`` when ``S_T > K`` and nothing otherwise.
    Under the risk-neutral measure the price is the discounted probability of
    finishing in the money:

        call = cash * e^{-rT} * N(d2),     put = cash * e^{-rT} * N(-d2).

    The FD solver reproduces this with the terminal condition being a step of
    height ``cash`` at the strike.
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return float(cash if S > K else 0.0)
        return float(cash if S < K else 0.0)

    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    df = math.exp(-r * T)
    if option_type == "call":
        return float(cash * df * norm.cdf(d2))
    return float(cash * df * norm.cdf(-d2))


# ---------------------------------------------------------------------------
# Knock-out barrier options (Reiner-Rubinstein / Merton)
# ---------------------------------------------------------------------------

def _barrier_ABCD(S, K, H, r, q, sigma, T, phi, eta):
    """Return the four Reiner-Rubinstein building blocks A, B, C, D.

    ``phi`` = +1 for a call, -1 for a put.
    ``eta`` = +1 for a *down* barrier, -1 for an *up* barrier.

    These blocks combine into the eight standard knock-in prices; knock-out
    prices are then ``vanilla - knock_in`` (rebate assumed zero).
    """
    sqrtT = math.sqrt(T)
    mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma)
    sigsqT = sigma * sqrtT

    x1 = math.log(S / K) / sigsqT + (1 + mu) * sigsqT
    x2 = math.log(S / H) / sigsqT + (1 + mu) * sigsqT
    y1 = math.log(H * H / (S * K)) / sigsqT + (1 + mu) * sigsqT
    y2 = math.log(H / S) / sigsqT + (1 + mu) * sigsqT

    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    HS_pow_plus = (H / S) ** (2 * (mu + 1))
    HS_pow = (H / S) ** (2 * mu)

    A = (phi * S * df_q * norm.cdf(phi * x1)
         - phi * K * df_r * norm.cdf(phi * x1 - phi * sigsqT))
    B = (phi * S * df_q * norm.cdf(phi * x2)
         - phi * K * df_r * norm.cdf(phi * x2 - phi * sigsqT))
    C = (phi * S * df_q * HS_pow_plus * norm.cdf(eta * y1)
         - phi * K * df_r * HS_pow * norm.cdf(eta * y1 - eta * sigsqT))
    D = (phi * S * df_q * HS_pow_plus * norm.cdf(eta * y2)
         - phi * K * df_r * HS_pow * norm.cdf(eta * y2 - eta * sigsqT))
    return A, B, C, D


def barrier_price(S: float, K: float, H: float, r: float, sigma: float,
                  T: float, option_type: str = "call",
                  barrier_type: str = "up-and-out", q: float = 0.0) -> float:
    """Analytic price of a single-barrier knock-out option (zero rebate).

    Parameters
    ----------
    H : barrier level.
    barrier_type : one of ``"up-and-out"``, ``"down-and-out"``.

    Method
    ------
    We compute the corresponding *knock-in* value from the Reiner-Rubinstein
    A/B/C/D blocks and use in-out parity ``out = vanilla - in`` (valid with no
    rebate). Returns ``0`` if the spot has already breached the barrier.
    """
    call = option_type == "call"
    phi = 1.0 if call else -1.0

    # Already knocked out?
    if barrier_type == "up-and-out" and S >= H:
        return 0.0
    if barrier_type == "down-and-out" and S <= H:
        return 0.0

    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
        return float(intrinsic)

    vanilla = bs_price(S, K, r, sigma, T, option_type, q)

    if barrier_type == "down-and-out":
        eta = 1.0
        A, B, C, D = _barrier_ABCD(S, K, H, r, q, sigma, T, phi, eta)
        if call:
            knock_in = C if K >= H else (A - B + D)
        else:  # down-and-in put
            knock_in = (B - C + D) if K >= H else A
    elif barrier_type == "up-and-out":
        eta = -1.0
        A, B, C, D = _barrier_ABCD(S, K, H, r, q, sigma, T, phi, eta)
        if call:
            knock_in = A if K > H else (B - C + D)
        else:  # up-and-in put
            knock_in = (A - B + D) if K > H else C
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

    return float(max(vanilla - knock_in, 0.0))


# ---------------------------------------------------------------------------
# American
# ---------------------------------------------------------------------------

def american_call_analytic(S: float, K: float, r: float, sigma: float,
                           T: float, q: float = 0.0) -> float:
    """American call price when there are no dividends.

    With ``q = 0`` early exercise of a call is never optimal (the time value is
    always positive), so the American call equals the European call. With
    dividends this no longer holds and there is no simple closed form.
    """
    if q > 0:
        raise ValueError("No closed form for an American call with dividends.")
    return bs_price(S, K, r, sigma, T, "call", q)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def analytic_price(framework: str, params: dict):
    """Return the analytic price for ``framework`` given ``params`` or ``None``.

    ``params`` is expected to contain S0, K, r, sigma, T and, where relevant,
    ``option_type``, ``barrier`` / ``barrier_type``, ``cash`` and ``q``.
    Returns ``None`` when no closed form exists (e.g. the American put), so the
    UI can display "n/a".
    """
    S0 = params["S0"]; K = params["K"]; r = params["r"]
    sigma = params["sigma"]; T = params["T"]
    q = params.get("q", 0.0)
    opt = params.get("option_type", "call")

    if framework == "vanilla":
        return bs_price(S0, K, r, sigma, T, opt, q)
    if framework == "digital":
        return digital_cash_or_nothing(S0, K, r, sigma, T, opt,
                                       params.get("cash", 1.0), q)
    if framework == "barrier":
        return barrier_price(S0, K, params["barrier"], r, sigma, T, opt,
                             params.get("barrier_type", "up-and-out"), q)
    if framework == "american":
        if opt == "call" and q == 0.0:
            return american_call_analytic(S0, K, r, sigma, T, q)
        return None  # American put: no closed form.
    return None


if __name__ == "__main__":
    # Sanity checks / put-call parity.
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
    c = bs_price(S, K, r, sig, T, "call")
    p = bs_price(S, K, r, sig, T, "put")
    print(f"BS call={c:.4f} put={p:.4f} "
          f"parity_resid={c - p - (S - K * math.exp(-r * T)):.2e}")
    print(f"digital call={digital_cash_or_nothing(S, K, r, sig, T, 'call'):.4f}")
    uo = barrier_price(S, K, 130.0, r, sig, T, "call", "up-and-out")
    print(f"up-and-out call (H=130)={uo:.4f}  (< vanilla {c:.4f}: {uo < c})")
    do = barrier_price(S, K, 90.0, r, sig, T, "call", "down-and-out")
    print(f"down-and-out call (H=90)={do:.4f}  (< vanilla: {do < c})")
