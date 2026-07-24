"""
market_sim.py
=============

A tiny **stochastic order-flow generator** that keeps the limit order book in
:mod:`order_book` *alive*, so the GUI has something to visualise.

This is deliberately a *toy* microstructure model -- not a calibrated one. Its
job is to produce a plausible, continuously churning book: liquidity providers
posting limit orders around a drifting fair value, liquidity takers crossing the
spread with market orders, and traders cancelling stale quotes.

--------------------------------------------------------------------------
The agent model
--------------------------------------------------------------------------
A single latent **fair value** ``fv`` follows an arithmetic random walk with a
small drift, so the mid price wanders around and the book never stalls::

    fv <- fv + drift + Normal(0, sigma)

On every ``step`` the simulator fires three independent kinds of flow, each
governed by a Poisson-style rate (an expected count per step):

* **Limit orders** (``limit_rate``) -- passive liquidity. A side is chosen, then
  the order is priced a random number of ticks *away from* the fair value using
  an exponential offset (most quotes land near the touch, a few sit deep). These
  usually rest, building depth. A quote priced through the fair value can be
  marketable and trade immediately.
* **Market orders** (``market_rate``) -- aggressive liquidity takers that cross
  the spread and generate trades / partial fills.
* **Cancellations** (``cancel_rate``) -- a random resting order the simulator
  still owns is pulled, mimicking quote churn.

Everything is driven by the standard-library :mod:`random` module (seedable for
reproducible tests); numpy is not required.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from order_book import OrderBook, Side, Trade


@dataclass
class SimConfig:
    """Tunable knobs for the order-flow model.

    All *rate* fields are expected counts per :meth:`MarketSimulator.step`; the
    actual count each step is Poisson-distributed around them, so fractional
    rates below 1.0 are meaningful (e.g. ``market_rate=0.6`` ~ a market order
    roughly every other step).
    """

    fair_value: float = 100.0
    tick_size: float = 0.01

    # Fair-value random walk (per step).
    drift: float = 0.0
    sigma: float = 0.05

    # Flow intensities (expected events per step).
    limit_rate: float = 6.0
    market_rate: float = 1.2
    cancel_rate: float = 3.0

    # Limit-order placement: offset from fair value ~ Exponential(mean ticks).
    mean_offset_ticks: float = 8.0
    # A small chance a limit order is priced aggressively *through* the fair
    # value, so it crosses and takes liquidity instead of resting.
    aggressive_limit_prob: float = 0.15

    # Order sizes (uniform integer lots).
    min_size: int = 1
    max_size: int = 12

    # "Aggressiveness" scales market-order rate and size; 1.0 == as configured.
    aggressiveness: float = 1.0

    seed: int | None = None


class MarketSimulator:
    """Drives synthetic order flow into an :class:`~order_book.OrderBook`."""

    def __init__(self, book: OrderBook, config: SimConfig | None = None) -> None:
        self.book = book
        self.cfg = config or SimConfig()
        self.rng = random.Random(self.cfg.seed)
        self.fair_value = self.cfg.fair_value
        # Resting order ids the simulator still owns (candidates to cancel).
        self._resting: set[int] = set()
        self.steps = 0

    # -- helpers ------------------------------------------------------------
    def _poisson(self, lam: float) -> int:
        """Draw a Poisson count via Knuth's algorithm (no numpy needed)."""
        if lam <= 0:
            return 0
        import math

        l_bound = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rng.random()
            if p <= l_bound:
                return k - 1

    def _random_size(self, scale: float = 1.0) -> int:
        lo, hi = self.cfg.min_size, max(self.cfg.min_size, self.cfg.max_size)
        base = self.rng.randint(lo, hi)
        return max(1, int(round(base * scale)))

    def _seed_book(self) -> None:
        """Populate an initially empty book with a symmetric ladder of quotes."""
        fv = self.fair_value
        tick = self.cfg.tick_size
        for i in range(1, 11):
            bid_px = fv - i * tick * 2
            ask_px = fv + i * tick * 2
            b, _ = self.book.submit_limit(Side.BUY, bid_px, self._random_size())
            a, _ = self.book.submit_limit(Side.SELL, ask_px, self._random_size())
            if not b.is_filled:
                self._resting.add(b.id)
            if not a.is_filled:
                self._resting.add(a.id)

    # -- main tick ----------------------------------------------------------
    def step(self) -> list[Trade]:
        """Advance the simulation by one tick; return trades generated this tick."""
        self.steps += 1
        if self.steps == 1 and self.book.best_bid() is None:
            self._seed_book()

        # 1) Drift the latent fair value.
        self.fair_value += self.cfg.drift + self.rng.gauss(0.0, self.cfg.sigma)

        trades: list[Trade] = []
        trades += self._do_cancels()
        trades += self._do_limits()
        trades += self._do_markets()

        self._prune_resting()
        return trades

    def _do_limits(self) -> list[Trade]:
        trades: list[Trade] = []
        n = self._poisson(self.cfg.limit_rate)
        tick = self.cfg.tick_size
        for _ in range(n):
            side = self.rng.choice((Side.BUY, Side.SELL))
            offset_ticks = self.rng.expovariate(1.0 / self.cfg.mean_offset_ticks)
            aggressive = self.rng.random() < self.cfg.aggressive_limit_prob
            sign = -1.0 if aggressive else 1.0  # aggressive => price through fv
            if side is Side.BUY:
                price = self.fair_value - sign * offset_ticks * tick
            else:
                price = self.fair_value + sign * offset_ticks * tick
            price = max(tick, price)
            order, t = self.book.submit_limit(side, price, self._random_size())
            trades += t
            if not order.is_filled:
                self._resting.add(order.id)
        return trades

    def _do_markets(self) -> list[Trade]:
        trades: list[Trade] = []
        rate = self.cfg.market_rate * self.cfg.aggressiveness
        n = self._poisson(rate)
        for _ in range(n):
            side = self.rng.choice((Side.BUY, Side.SELL))
            # Only send if there is something to hit on the far side.
            far = self.book.best_ask() if side is Side.BUY else self.book.best_bid()
            if far is None:
                continue
            size = self._random_size(scale=self.cfg.aggressiveness)
            _, t = self.book.submit_market(side, size)
            trades += t
        return trades

    def _do_cancels(self) -> list[Trade]:
        n = self._poisson(self.cfg.cancel_rate)
        for _ in range(n):
            if not self._resting:
                break
            oid = self.rng.choice(tuple(self._resting))
            self._resting.discard(oid)
            self.book.cancel(oid)
        return []  # cancels never create trades

    def _prune_resting(self) -> None:
        """Drop ids the book no longer holds (filled or already cancelled)."""
        self._resting = {oid for oid in self._resting if oid in self.book}


__all__ = ["SimConfig", "MarketSimulator"]
