"""
simulator.py
============

Tick-by-tick options market-making simulation.

The story
---------
A market maker (MM) runs a rolling, constant-maturity quoting business: every
tick it makes a market in a *fresh* X-DTE option (constant time-to-maturity `T`,
fixed strike `K`) -- it is NOT aging one single contract toward expiry. Think of
it as quoting today's 3-month option, then tomorrow's brand-new 3-month option,
and so on. This keeps the desk's risk profile stationary so the P/L story is
about *pricing edge* and *model risk*, not theta decay.

Crucially, we **realize the P/L of each option at the moment it trades**. When a
client trades against a quote, the MM books that contract and we immediately
simulate its entire delta-hedged life to expiry *under the true underlying
model*, discount the outcome back, and bank it as realized cash. So each fill is
resolved on the spot rather than being warehoused as a growing marked-to-market
inventory position (which would make the equity curve a story about a big open
directional bet instead of about spread capture).

Every tick the MM:

  1. Observes the current underlying price S_t (driven by the *true* market path
     model, which the MM may not know).
  2. Computes their *model* fair value V_model = pricer.price(S_t, ..., T).
  3. Posts a bid and an ask around that fair value:
         bid = V_model - half_spread
         ask = V_model + half_spread
  4. Receives random client order flow. Clients buy at the MM's ask (MM sells an
     option) or sell at the MM's bid (MM buys an option). Fill probability scales
     with quote size / fill intensity.
  5. For every contract that trades, we realize its P/L: simulate the option's
     hedged life to expiry under the *true* dynamics (the MM delta-hedges with
     *their own* model delta) and bank the discounted result.

Why this gives positive expectancy from the spread
---------------------------------------------------
A continuously delta-hedged option's P/L is (in the limit) independent of the
real-world drift: it equals the premium collected minus the option's fair value.
So a contract SOLD at ``ask = fair + half_spread`` realizes, in expectation,
``+half_spread`` of edge; a contract BOUGHT at ``bid = fair - half_spread``
realizes ``+half_spread`` too. Buying low and selling high around fair value is
exactly what banks the spread. The *realized* curve is stochastic (discrete
hedging error + whatever the underlying actually did), but on average it climbs
by the spread per contract -- positive expectancy.

We track three P/L curves:

  * realized_pnl - the actual, stochastic cash realized as each option's hedged
                   life plays out under the true model. This is the equity curve.
  * exp_model    - cumulative edge the MM *believes* it is banking, measured
                   against the MM's own fair value: sign*(premium - V_model).
                   With unbiased pricing this is ~ half_spread per contract.
  * exp_truth    - cumulative edge measured against the *true* fair value:
                   sign*(premium - V_truth). This is the genuine expected edge.

The teaching moment: when pricing model == truth model, exp_model ~= exp_truth
and the realized curve wobbles upward around them -- clean spread capture. When
the MM prices with Black-Scholes/GBM but the world has Merton jumps, exp_model
still looks healthy (the MM thinks every fill banks the spread) while exp_truth
sags -- the MM is systematically selling too cheap / buying too dear -- and the
realized curve takes fat-tailed jump losses its delta hedge never saw coming:
the classic "picking up pennies in front of a steamroller."

Everything is intentionally simple and transparent so the mechanics are visible;
it is a teaching simulator, not an exchange matching engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from pricing import PricingModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All knobs the front-end exposes. Sensible defaults included."""

    # --- Option contract --------------------------------------------------
    # The MM quotes a *rolling, constant-maturity* option: every tick it makes
    # a market in a *fresh* contract with a fixed parameterisation (constant
    # time-to-maturity `T`, fixed strike `K`), rather than aging one single
    # option toward expiry. This keeps theta/vega exposure stationary so the
    # P/L story is driven by pricing edge and model risk, not by decay.
    S0: float = 100.0            # initial underlying price
    K: float = 100.0             # option strike (fixed parameterisation)
    T: float = 0.25              # CONSTANT time-to-maturity quoted every tick
    option_type: str = "call"    # "call" or "put"

    # Simulation horizon in years: how long the desk runs quoting the rolling
    # contract. Decoupled from `T` (the contract's constant maturity) so a desk
    # can, e.g., trade a rolling 3-month option (T=0.25) across a 1-year run.
    horizon: float = 1.0

    # --- True market-path model ------------------------------------------
    path_model: str = "gbm"      # "gbm" or "merton"
    mu: float = 0.05             # real-world drift of the underlying (annual)
    sigma: float = 0.20          # diffusive volatility (annual)
    lam: float = 1.0             # Merton jump intensity (jumps/year)
    jump_mean: float = -0.05     # mean of log jump size Y
    jump_std: float = 0.10       # std of log jump size Y

    # --- Pricing model used by the MM ------------------------------------
    pricing_model: str = "black_scholes"  # "black_scholes" or "merton"
    r: float = 0.02              # risk-free rate used for discounting/hedging
    # Vol the MM plugs into their pricer. Defaults to the true sigma so that,
    # with matching models, the MM is "correct"; the mismatch experiment is
    # about the *model form*, not a vol misestimate (though you can set this).
    pricing_sigma: Optional[float] = None

    # --- Market making behaviour -----------------------------------------
    half_spread: float = 0.15    # half the quoted bid/ask spread, in option $
    quote_size: float = 1.0      # contracts quoted per side per tick
    # Baseline probability that a resting quote gets hit/lifted in one tick,
    # per unit quote size. Higher => busier market => more fills => the law of
    # large numbers reveals the spread edge sooner.
    fill_intensity: float = 0.35
    # Reserved knob (kept for UI compatibility). Because every fill is realized
    # on the spot, the desk never warehouses inventory, so there is no running
    # position to skew against; this currently has no effect on quoting.
    risk_aversion: float = 0.02

    # --- Realization of each trade ---------------------------------------
    # When a contract trades we simulate its delta-hedged life to expiry under
    # the true model to realize its P/L. `hedge_steps` is the number of
    # rebalancing points over the option's life T. More steps => tighter hedge
    # => smaller realized variance (so the spread edge is easier to see).
    hedge_steps: int = 30

    # --- Simulation grid --------------------------------------------------
    n_steps: int = 250           # number of ticks
    seed: Optional[int] = 42     # RNG seed; None => nondeterministic

    def effective_pricing_sigma(self) -> float:
        return self.sigma if self.pricing_sigma is None else self.pricing_sigma


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Time series produced by a run. All arrays have length n_steps+1."""

    t: list = field(default_factory=list)            # calendar time (years)
    S: list = field(default_factory=list)            # underlying price
    bid: list = field(default_factory=list)          # MM bid on the option
    ask: list = field(default_factory=list)          # MM ask on the option
    model_price: list = field(default_factory=list)  # MM model fair value
    truth_price: list = field(default_factory=list)  # true fair value
    inventory: list = field(default_factory=list)    # cumulative contracts traded
    hedge: list = field(default_factory=list)        # avg realized edge / contract
    realized_pnl: list = field(default_factory=list) # cumulative realized P/L (equity)
    mtm_model: list = field(default_factory=list)     # cumulative MM-believed edge
    mtm_truth: list = field(default_factory=list)     # cumulative true expected edge
    fills_buy: list = field(default_factory=list)     # cumulative contracts MM bought
    fills_sell: list = field(default_factory=list)    # cumulative contracts MM sold

    # Scalar summary stats filled in at the end.
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Path generation for the TRUE market model
# ---------------------------------------------------------------------------

def _simulate_true_path(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate the true underlying price path S_0..S_n on the *quoting* grid.

    This is the calendar spine of the desk: the spot the MM observes (and quotes
    a fresh option against) at each tick. It is separate from the per-option
    hedge sub-paths used to realize each trade.

    GBM:    exact log-Euler update  S_{t+dt} = S_t * exp((mu-sigma^2/2)dt + sigma sqrt(dt) Z)
    Merton: same diffusion, plus in each step add N~Poisson(lam*dt) jumps whose
            summed log size is Normal(N*jump_mean, N*jump_std^2). A drift
            compensator -lam*k*dt keeps the *diffusive* drift interpretable.
    """
    n = cfg.n_steps
    dt = cfg.horizon / n           # step size is set by the run horizon
    S = np.empty(n + 1)
    S[0] = cfg.S0

    # Compensator so E[jump contribution] is centred (keeps mu the net drift).
    k = np.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0

    for i in range(n):
        Z = rng.standard_normal()
        drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * dt
        diffusion = cfg.sigma * np.sqrt(dt) * Z
        log_move = drift + diffusion

        if cfg.path_model == "merton":
            n_jumps = rng.poisson(cfg.lam * dt)
            if n_jumps > 0:
                jump = rng.normal(cfg.jump_mean * n_jumps,
                                  cfg.jump_std * np.sqrt(n_jumps))
            else:
                jump = 0.0
            log_move += jump - cfg.lam * k * dt  # subtract drift compensator

        S[i + 1] = S[i] * np.exp(log_move)

    return S


def _true_pricer(cfg: SimConfig) -> PricingModel:
    """Pricer representing the TRUE dynamics, used only for mark-to-truth."""
    if cfg.path_model == "merton":
        return PricingModel(name="merton", r=cfg.r, lam=cfg.lam,
                            mu_j=cfg.jump_mean, sigma_j=cfg.jump_std)
    return PricingModel(name="black_scholes", r=cfg.r)


def _mm_pricer(cfg: SimConfig) -> PricingModel:
    """Pricer the MM actually quotes with (their belief)."""
    if cfg.pricing_model == "merton":
        return PricingModel(name="merton", r=cfg.r, lam=cfg.lam,
                            mu_j=cfg.jump_mean, sigma_j=cfg.jump_std)
    return PricingModel(name="black_scholes", r=cfg.r)


# ---------------------------------------------------------------------------
# Realizing a single trade's P/L
# ---------------------------------------------------------------------------

def _simulate_option_life(cfg: SimConfig, S_start: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Simulate the underlying over ONE option's life [0, T] under the true model.

    Returns an array of length ``hedge_steps + 1`` starting at ``S_start``. Uses
    the same GBM/Merton dynamics as the quoting spine but on the option's own
    (finer) maturity grid so we can delta-hedge it.
    """
    m = max(1, int(cfg.hedge_steps))
    dth = cfg.T / m
    S = np.empty(m + 1)
    S[0] = S_start
    k = np.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
    sqrt_dth = np.sqrt(dth)

    for j in range(m):
        log_move = ((cfg.mu - 0.5 * cfg.sigma ** 2) * dth
                    + cfg.sigma * sqrt_dth * rng.standard_normal())
        if cfg.path_model == "merton":
            n_jumps = rng.poisson(cfg.lam * dth)
            if n_jumps > 0:
                log_move += rng.normal(cfg.jump_mean * n_jumps,
                                       cfg.jump_std * np.sqrt(n_jumps))
            log_move -= cfg.lam * k * dth
        S[j + 1] = S[j] * np.exp(log_move)
    return S


def _realize_trade_pnl(cfg: SimConfig, mm: PricingModel, sig_mm: float,
                       S_start: float, premium: float, side: str,
                       qty: float, rng: np.random.Generator) -> float:
    """Realize the full-life, delta-hedged P/L of one option trade.

    The MM has just traded ``qty`` contracts at ``premium`` (``side`` is
    ``"sell"`` if the client lifted the ask, ``"buy"`` if the client hit the
    bid). We simulate the option's life to expiry under the TRUE model and let
    the MM delta-hedge with THEIR OWN model delta, then return the discounted
    (present-value at trade time) realized P/L of the whole package.

    Why this yields the spread in expectation
    ------------------------------------------
    A continuously delta-hedged option's discounted P/L equals
    ``premium - fair_value`` regardless of the real-world drift. So selling at
    ``ask`` banks ``ask - fair = +half_spread`` and buying at ``bid`` banks
    ``fair - bid = +half_spread``, on average. Discrete hedging adds mean-~zero
    noise; model mismatch (e.g. BS-hedging a jumpy world) breaks the hedge on
    jumps and can turn the edge negative -- which is the whole lesson.
    """
    m = max(1, int(cfg.hedge_steps))
    tau = cfg.T
    dth = tau / m
    r = cfg.r
    growth = np.exp(r * dth)              # one-step financing factor for the hedge

    path = _simulate_option_life(cfg, S_start, rng)

    # Signed option position: long (+) if we bought, short (-) if we sold.
    n_opt = qty if side == "buy" else -qty

    # Premium cash flow at trade time (PV @ t=0). Sold => receive; bought => pay.
    pnl = (premium * qty) if side == "sell" else (-premium * qty)

    # Delta-hedge over the option's life. At each step we hold `h` underlying
    # units to be (model-)delta-neutral; the step's hedge P/L is financed at r
    # and discounted back to trade time. Using financing-correct increments is
    # what makes the hedged P/L drift-independent.
    for j in range(m):
        s_rem = tau - j * dth
        delta = mm.delta(path[j], cfg.K, sig_mm, s_rem, cfg.option_type)
        h = -n_opt * delta
        step_pnl = h * (path[j + 1] - path[j] * growth)
        pnl += np.exp(-r * (j * dth)) * step_pnl

    # Option settlement at expiry, discounted to trade time.
    if cfg.option_type == "call":
        payoff = max(path[m] - cfg.K, 0.0)
    else:
        payoff = max(cfg.K - path[m], 0.0)
    pnl += np.exp(-r * tau) * n_opt * payoff

    return float(pnl)


# ---------------------------------------------------------------------------
# The main simulation
# ---------------------------------------------------------------------------

def run_simulation(cfg: SimConfig) -> SimResult:
    """Run the full tick-by-tick market-making simulation.

    Returns a `SimResult` with aligned time series suitable for animation.
    """
    rng = np.random.default_rng(cfg.seed)

    S = _simulate_true_path(cfg, rng)
    n = cfg.n_steps
    dt = cfg.horizon / n

    mm = _mm_pricer(cfg)
    truth = _true_pricer(cfg)
    sig_mm = cfg.effective_pricing_sigma()

    res = SimResult()

    # --- Running tallies --------------------------------------------------
    realized = 0.0      # cumulative realized P/L (the equity curve)
    exp_model = 0.0     # cumulative edge vs the MM's own fair value
    exp_truth = 0.0     # cumulative edge vs the true fair value
    cum_buy = 0.0       # cumulative contracts the MM BOUGHT (clients hit bid)
    cum_sell = 0.0      # cumulative contracts the MM SOLD (clients lift ask)

    for i in range(n + 1):
        t = i * dt
        # ROLLING CONTRACT: every tick we quote a *fresh* option with the same
        # fixed parameterisation, so time-to-maturity stays constant at `T`.
        tau = cfg.T
        St = S[i]

        # MM fair value + true fair value for the current rolling contract.
        v_model = mm.price(St, cfg.K, sig_mm, tau, cfg.option_type)
        v_truth = truth.price(St, cfg.K, cfg.sigma, tau, cfg.option_type)

        # Symmetric two-sided quote around the MM's fair value.
        bid = v_model - cfg.half_spread
        ask = v_model + cfg.half_spread

        # Record state at the top of the tick (reflects all trades so far).
        total_contracts = cum_buy + cum_sell
        res.t.append(t)
        res.S.append(float(St))
        res.bid.append(float(bid))
        res.ask.append(float(ask))
        res.model_price.append(float(v_model))
        res.truth_price.append(float(v_truth))
        res.inventory.append(float(total_contracts))
        res.hedge.append(float(realized / total_contracts) if total_contracts else 0.0)
        res.realized_pnl.append(float(realized))
        res.mtm_model.append(float(exp_model))
        res.mtm_truth.append(float(exp_truth))
        res.fills_buy.append(float(cum_buy))
        res.fills_sell.append(float(cum_sell))

        if i == n:
            break  # last point is a pure mark; no trading after the horizon

        # --- Client order flow --------------------------------------------
        # Independent Bernoulli fills on each side; probability grows with quote
        # size / fill intensity. No inventory skew because every fill is
        # realized on the spot (the desk carries no running position).
        p = float(np.clip(cfg.fill_intensity * cfg.quote_size, 0.02, 0.98))
        lifted = rng.random() < p   # client BUYS from us at ask -> we SELL
        hit = rng.random() < p      # client SELLS to us at bid -> we BUY

        if lifted:
            cum_sell += cfg.quote_size
            exp_model += (ask - v_model) * cfg.quote_size   # = +half_spread
            exp_truth += (ask - v_truth) * cfg.quote_size
            realized += _realize_trade_pnl(cfg, mm, sig_mm, St, ask,
                                           "sell", cfg.quote_size, rng)
        if hit:
            cum_buy += cfg.quote_size
            exp_model += (v_model - bid) * cfg.quote_size   # = +half_spread
            exp_truth += (v_truth - bid) * cfg.quote_size
            realized += _realize_trade_pnl(cfg, mm, sig_mm, St, bid,
                                           "buy", cfg.quote_size, rng)

    _finalize_summary(cfg, res)
    return res


def _finalize_summary(cfg: SimConfig, res: SimResult) -> None:
    """Compute scalar summary statistics for the stats panel."""
    realized = np.asarray(res.realized_pnl)
    exp_model = np.asarray(res.mtm_model)
    exp_truth = np.asarray(res.mtm_truth)

    total_fills = res.fills_buy[-1] + res.fills_sell[-1]
    # Theoretical edge if every contract banked the full half-spread.
    ev_spread = total_fills * cfg.half_spread

    # Per-tick change in realized P/L, for a volatility-of-P/L (risk) read.
    dpnl = np.diff(realized)
    pnl_vol = float(np.std(dpnl)) if dpnl.size else 0.0

    avg_edge = float(realized[-1] / total_fills) if total_fills > 0 else 0.0

    res.summary = {
        "final_realized_pnl": float(realized[-1]),
        "final_expected_model": float(exp_model[-1]),
        "final_expected_truth": float(exp_truth[-1]),
        "model_vs_truth_gap": float(exp_model[-1] - exp_truth[-1]),
        "realized_vs_truth_gap": float(realized[-1] - exp_truth[-1]),
        "total_contracts_traded": float(total_fills),
        "contracts_bought": float(res.fills_buy[-1]),
        "contracts_sold": float(res.fills_sell[-1]),
        "expected_spread_capture": float(ev_spread),
        "avg_realized_edge": avg_edge,
        "half_spread": float(cfg.half_spread),
        "pnl_step_vol": pnl_vol,
        "max_drawdown_truth": float(_max_drawdown(realized)),
        "models_match": cfg.pricing_model == (
            "merton" if cfg.path_model == "merton" else "black_scholes"),
        "pricing_model": cfg.pricing_model,
        "path_model": cfg.path_model,
    }


def _max_drawdown(series: np.ndarray) -> float:
    """Largest peak-to-trough drop in a P/L series (positive number)."""
    if series.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(series)
    drawdowns = running_max - series
    return float(np.max(drawdowns))


# ---------------------------------------------------------------------------
# Return-distribution comparison (GBM vs Merton)
# ---------------------------------------------------------------------------

def _simulate_paths_vectorized(
    cfg: "SimConfig",
    model: str,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate ``n_paths`` full underlying paths under ``model`` at once.

    Returns an array of shape ``(n_paths, n_steps + 1)`` of prices. The maths
    mirrors ``_simulate_true_path`` but is fully vectorised for speed so we can
    draw tens of thousands of paths to build a smooth return histogram.
    """
    n = cfg.n_steps
    dt = cfg.T / n

    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * dt
    diffusion = cfg.sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n))
    log_incr = drift + diffusion

    if model == "merton":
        # Compensator so the diffusive drift stays interpretable as ``mu``.
        k = np.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
        n_jumps = rng.poisson(cfg.lam * dt, size=(n_paths, n))
        # Sum of ``n_jumps`` iid Normal(mu_j, sigma_j^2) jump sizes is
        # Normal(n_jumps*mu_j, n_jumps*sigma_j^2) -- sample it in one shot.
        jump = (n_jumps * cfg.jump_mean
                + np.sqrt(n_jumps) * cfg.jump_std * rng.standard_normal((n_paths, n)))
        log_incr += jump - cfg.lam * k * dt

    log_paths = np.cumsum(log_incr, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return cfg.S0 * np.exp(log_paths)


def _distribution_moments(returns: np.ndarray) -> dict:
    """Summary moments of a terminal-return sample (returns are simple, decimal)."""
    r = np.asarray(returns, dtype=float)
    mean = float(np.mean(r))
    std = float(np.std(r))
    if std > 1e-12:
        z = (r - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)  # excess kurtosis
    else:
        skew = kurt = 0.0
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": kurt,
        "worst": float(np.min(r)),
        "best": float(np.max(r)),
        "var_1pct": float(np.percentile(r, 1.0)),   # 1% left-tail return
    }


def simulate_return_distributions(
    cfg: "SimConfig",
    n_paths: int = 15000,
    n_sample_paths: int = 8,
    n_bins: int = 70,
) -> dict:
    """Simulate GBM *and* Merton under the same config and compare their
    terminal-return distributions.

    The point: both models share the same diffusive ``mu``/``sigma``, yet
    adding a Poisson jump component gives Merton a fat, (typically) left-skewed
    tail. That extra tail risk is exactly why an option priced under Merton
    costs more than under Black-Scholes/GBM. Seeing the two histograms side by
    side makes the pricing difference intuitive.

    Returns a JSON-serialisable dict with, per model, the terminal simple
    returns, summary moments, and a handful of sample paths; plus a shared time
    grid and common histogram bin spec so the two histograms overlay fairly.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    dt = cfg.T / n
    t_grid = [i * dt for i in range(n + 1)]

    out: dict = {"t": t_grid, "models": {}}
    all_returns = []

    for model in ("gbm", "merton"):
        paths = _simulate_paths_vectorized(cfg, model, n_paths, rng)
        terminal_returns = paths[:, -1] / cfg.S0 - 1.0   # simple return over T
        all_returns.append(terminal_returns)
        out["models"][model] = {
            "returns": terminal_returns.tolist(),
            "paths": paths[:n_sample_paths].tolist(),
            "stats": _distribution_moments(terminal_returns),
        }

    # Shared bins from robust combined range (trim extreme Merton tails so the
    # bulk of both distributions is legible), padded a touch.
    combined = np.concatenate(all_returns)
    lo = float(np.percentile(combined, 0.3))
    hi = float(np.percentile(combined, 99.7))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    start, end = lo - pad, hi + pad
    out["bins"] = {"start": start, "end": end, "size": (end - start) / n_bins}
    out["n_paths"] = n_paths
    return out


if __name__ == "__main__":
    # Headless smoke test: run two scenarios and print summaries.
    print("=== Matched model (BS prices a GBM world) ===")
    cfg_match = SimConfig(path_model="gbm", pricing_model="black_scholes",
                          seed=7)
    r1 = run_simulation(cfg_match)
    for k, v in r1.summary.items():
        print(f"  {k}: {v}")

    print("\n=== Mismatched model (BS prices a Merton-jump world) ===")
    cfg_mismatch = SimConfig(path_model="merton", pricing_model="black_scholes",
                             lam=3.0, jump_mean=-0.08, jump_std=0.15, seed=7)
    r2 = run_simulation(cfg_mismatch)
    for k, v in r2.summary.items():
        print(f"  {k}: {v}")

    # Expectancy check: average realized edge per contract in the matched world
    # should be positive and close to the quoted half-spread across seeds.
    print("\n=== Positive-expectancy check (matched, averaged over seeds) ===")
    edges = []
    for s in range(40):
        c = SimConfig(path_model="gbm", pricing_model="black_scholes",
                      fill_intensity=0.9, seed=s)
        edges.append(run_simulation(c).summary["avg_realized_edge"])
    print(f"  mean avg_realized_edge over 40 seeds: {np.mean(edges):.4f}"
          f"  (quoted half_spread = {SimConfig().half_spread})")

    assert len(r1.S) == cfg_match.n_steps + 1
    assert np.all(np.isfinite(r1.mtm_truth))
    assert np.all(np.isfinite(r2.mtm_truth))
    assert np.all(np.isfinite(r1.realized_pnl))
    print("\nSmoke test OK: arrays finite and correctly sized.")
