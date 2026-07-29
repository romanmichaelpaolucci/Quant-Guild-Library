"""
strategies.py
=============

Strategy framework for the Algorithmic Trading System cockpit.

Design
------
Every trading strategy is a small, self-contained object that looks at a rolling
window of prices for a single symbol and emits a *directional signal* in the
range [-1.0, +1.0]:

    +1.0  ->  maximally long
     0.0  ->  flat / no opinion
    -1.0  ->  maximally short

The signal is intentionally *dimensionless*. It says "how bullish/bearish am I",
not "how many shares". Position sizing is a separate concern handled by the
portfolio layer so that risk/capital allocation is decoupled from alpha logic.

How signals become target positions
------------------------------------
Each strategy is assigned a *percent allocation* of the portfolio (e.g. 25%).
When a strategy is enabled (ON) its contribution to a symbol's target weight is:

    contribution = signal * (allocation_pct / 100)

The portfolio aggregates contributions per symbol across all *enabled*
strategies (a form of signal blending / capital budgeting):

    target_weight[symbol] = sum(contribution_i for enabled strategies on symbol)

Target weights are clamped to [-1, +1] per symbol so a single symbol can never
demand more than 100% of net liquidation (before leverage). The dollar target
and share target are then:

    target_dollars[symbol] = net_liq * target_weight[symbol]
    target_shares[symbol]  = round(target_dollars[symbol] / price[symbol])

The order to send is the *delta* between target shares and the currently held
position, so the system is self-correcting toward its target book on each rebalance.

This module is pure Python/numpy and has NO tkinter or ibapi dependencies, so it
can be unit-tested headlessly. Run `python strategies.py` for a self-test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------
class Strategy(ABC):
    """Abstract base for all strategies.

    Subclasses implement :meth:`generate_signal`, returning a float in
    ``[-1.0, +1.0]``. Strategies are parameterized via keyword arguments that
    are validated in ``__init__`` and exposed through :attr:`params` so the UI
    can display and edit them.
    """

    #: Human-readable strategy type shown in the UI dropdown / table.
    type_name: str = "Strategy"

    #: Ordered list of (param_key, label, default) used by the UI to build the
    #: "add strategy" form dynamically. Subclasses override this.
    param_spec: List[tuple] = []

    def __init__(self, symbol: str, **params):
        self.symbol = symbol.upper().strip()
        # Merge provided params over defaults declared in param_spec.
        self.params: Dict[str, float] = {k: default for (k, _label, default) in self.param_spec}
        for key, value in params.items():
            if key not in self.params:
                raise ValueError(f"{self.type_name}: unknown parameter '{key}'")
            self.params[key] = value
        self.validate()

    # -- overridable hooks ---------------------------------------------------
    def validate(self) -> None:
        """Validate parameters. Override to enforce constraints."""

    @property
    def min_history(self) -> int:
        """Minimum number of price points required to produce a signal."""
        return 2

    @abstractmethod
    def generate_signal(self, prices: Sequence[float]) -> float:
        """Return a directional signal in [-1, +1] for the latest bar.

        Parameters
        ----------
        prices:
            Chronologically ordered price history (oldest first, newest last)
            for :attr:`symbol`. Implementations must tolerate short histories
            by returning ``0.0`` (flat) until enough data has accumulated.
        """

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _clean(prices: Sequence[float]) -> np.ndarray:
        arr = np.asarray(prices, dtype=float)
        return arr[np.isfinite(arr)]

    def describe_params(self) -> str:
        """Compact parameter string for the strategy table (e.g. 'fast=20 slow=50')."""
        parts = []
        for key in self.params:
            val = self.params[key]
            parts.append(f"{key}={int(val) if float(val).is_integer() else round(val, 3)}")
        return " ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.type_name} {self.symbol} {self.describe_params()}>"


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------
class TrendFollowing(Strategy):
    """Moving-average crossover trend follower.

    Goes long when the fast simple moving average is above the slow one and
    short when it is below. The signal magnitude scales with the *normalized
    spread* between the two averages (a wider, more established trend produces a
    stronger conviction), squashed into [-1, 1] via ``tanh``.
    """

    type_name = "Trend Following"
    param_spec = [
        ("fast", "Fast MA", 20),
        ("slow", "Slow MA", 50),
        ("scale", "Sensitivity", 25.0),
    ]

    def validate(self) -> None:
        if self.params["fast"] < 1 or self.params["slow"] < 1:
            raise ValueError("MA windows must be >= 1")
        if self.params["fast"] >= self.params["slow"]:
            raise ValueError("fast MA window must be < slow MA window")

    @property
    def min_history(self) -> int:
        return int(self.params["slow"])

    def generate_signal(self, prices: Sequence[float]) -> float:
        arr = self._clean(prices)
        slow_n = int(self.params["slow"])
        fast_n = int(self.params["fast"])
        if arr.size < slow_n:
            return 0.0
        fast_ma = arr[-fast_n:].mean()
        slow_ma = arr[-slow_n:].mean()
        # Normalize spread by price so the signal is scale-invariant.
        spread = (fast_ma - slow_ma) / slow_ma
        return float(np.tanh(spread * self.params["scale"]))


class MeanReversion(Strategy):
    """Z-score mean-reversion strategy.

    Computes the z-score of the latest price versus its rolling mean/std over
    ``lookback`` bars and *fades* extremes: a high positive z-score (price
    stretched above its mean) yields a short signal, and vice versa. The signal
    is the negative z-score scaled by ``entry_z`` and clamped to [-1, 1].
    """

    type_name = "Mean Reversion"
    param_spec = [
        ("lookback", "Lookback", 20),
        ("entry_z", "Entry Z", 2.0),
    ]

    def validate(self) -> None:
        if self.params["lookback"] < 3:
            raise ValueError("lookback must be >= 3")
        if self.params["entry_z"] <= 0:
            raise ValueError("entry_z must be > 0")

    @property
    def min_history(self) -> int:
        return int(self.params["lookback"])

    def generate_signal(self, prices: Sequence[float]) -> float:
        arr = self._clean(prices)
        n = int(self.params["lookback"])
        if arr.size < n:
            return 0.0
        window = arr[-n:]
        mean = window.mean()
        std = window.std(ddof=1)
        if std <= 1e-12:
            return 0.0
        z = (arr[-1] - mean) / std
        # Fade the move: short when stretched up, long when stretched down.
        return float(np.clip(-z / self.params["entry_z"], -1.0, 1.0))


class Momentum(Strategy):
    """Time-series momentum strategy.

    Uses the total return over the ``lookback`` window. Positive momentum ->
    long, negative -> short. Magnitude scales with the size of the move via
    ``tanh`` so small drifts produce small positions.
    """

    type_name = "Momentum"
    param_spec = [
        ("lookback", "Lookback", 60),
        ("scale", "Sensitivity", 8.0),
    ]

    def validate(self) -> None:
        if self.params["lookback"] < 2:
            raise ValueError("lookback must be >= 2")

    @property
    def min_history(self) -> int:
        return int(self.params["lookback"])

    def generate_signal(self, prices: Sequence[float]) -> float:
        arr = self._clean(prices)
        n = int(self.params["lookback"])
        if arr.size < n:
            return 0.0
        past = arr[-n]
        if past <= 1e-12:
            return 0.0
        ret = (arr[-1] - past) / past
        return float(np.tanh(ret * self.params["scale"]))


# Registry used by the UI dropdown and by tests to instantiate by type name.
STRATEGY_REGISTRY: Dict[str, type] = {
    cls.type_name: cls for cls in (TrendFollowing, MeanReversion, Momentum)
}


def create_strategy(type_name: str, symbol: str, **params) -> Strategy:
    """Factory: build a strategy from its display type name."""
    if type_name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy type: {type_name!r}")
    return STRATEGY_REGISTRY[type_name](symbol, **params)


# ---------------------------------------------------------------------------
# Portfolio allocation record + aggregation / sizing engine
# ---------------------------------------------------------------------------
@dataclass
class StrategyAllocation:
    """A row in the strategy table: a strategy + its capital allocation + state.

    Attributes
    ----------
    strategy:
        The concrete :class:`Strategy` instance.
    allocation_pct:
        Percent of net liquidation this strategy may command (0-100).
    enabled:
        Whether the strategy is switched ON. Disabled strategies contribute
        nothing to target weights.
    """

    strategy: Strategy
    allocation_pct: float = 10.0
    enabled: bool = True
    id: int = field(default=0)

    @property
    def symbol(self) -> str:
        return self.strategy.symbol

    def contribution(self, prices: Sequence[float]) -> float:
        """Signed target-weight contribution for this strategy's symbol."""
        if not self.enabled:
            return 0.0
        signal = self.strategy.generate_signal(prices)
        return signal * (self.allocation_pct / 100.0)


def aggregate_target_weights(
    allocations: Sequence[StrategyAllocation],
    price_history: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """Blend enabled strategies into a target weight per symbol.

    For each symbol we sum the (signal * allocation) contributions of every
    enabled strategy trading that symbol, then clamp the net weight to
    ``[-1, +1]`` so no single symbol can exceed 100% of net liq (pre-leverage).
    """
    weights: Dict[str, float] = {}
    for alloc in allocations:
        if not alloc.enabled:
            continue
        prices = price_history.get(alloc.symbol, [])
        weights[alloc.symbol] = weights.get(alloc.symbol, 0.0) + alloc.contribution(prices)
    return {sym: float(np.clip(w, -1.0, 1.0)) for sym, w in weights.items()}


def target_shares_from_weight(target_weight: float, net_liq: float, price: float) -> int:
    """Convert a target weight into a whole-share target given capital & price.

        target_dollars = net_liq * target_weight
        target_shares  = round(target_dollars / price)
    """
    if price is None or price <= 0 or net_liq <= 0:
        return 0
    return int(round((net_liq * target_weight) / price))


@dataclass
class OrderPlan:
    """A proposed rebalance order for a single symbol."""

    symbol: str
    target_weight: float
    target_shares: int
    current_shares: int
    price: float

    @property
    def delta_shares(self) -> int:
        return self.target_shares - self.current_shares

    @property
    def action(self) -> str:
        return "BUY" if self.delta_shares > 0 else ("SELL" if self.delta_shares < 0 else "HOLD")

    @property
    def notional(self) -> float:
        return abs(self.delta_shares) * (self.price or 0.0)


def build_order_plans(
    allocations: Sequence[StrategyAllocation],
    price_history: Dict[str, Sequence[float]],
    latest_price: Dict[str, float],
    current_positions: Dict[str, int],
    net_liq: float,
) -> List[OrderPlan]:
    """Full pipeline: signals -> target weights -> target shares -> order deltas."""
    weights = aggregate_target_weights(allocations, price_history)
    plans: List[OrderPlan] = []
    for symbol, weight in weights.items():
        price = latest_price.get(symbol, 0.0)
        tgt = target_shares_from_weight(weight, net_liq, price)
        cur = int(current_positions.get(symbol, 0))
        plans.append(
            OrderPlan(
                symbol=symbol,
                target_weight=weight,
                target_shares=tgt,
                current_shares=cur,
                price=price,
            )
        )
    return plans


# ---------------------------------------------------------------------------
# Headless self-test (no GUI, no IBKR)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(7)

    # 1) Uptrending series -> trend follower long, momentum long.
    up = np.cumsum(np.abs(rng.normal(0.5, 0.2, 200))) + 100
    tf = TrendFollowing("AAPL", fast=10, slow=30)
    mom = Momentum("AAPL", lookback=50)
    assert tf.generate_signal(up) > 0.2, tf.generate_signal(up)
    assert mom.generate_signal(up) > 0.2, mom.generate_signal(up)

    # 2) Downtrending series -> short signals.
    down = up[::-1].copy()
    assert tf.generate_signal(down) < -0.2
    assert mom.generate_signal(down) < -0.2

    # 3) Mean reversion: spike above mean -> short (negative) signal.
    flat = np.full(50, 100.0)
    spiked = np.append(flat, [110.0])
    mr = MeanReversion("AAPL", lookback=30, entry_z=2.0)
    assert mr.generate_signal(spiked) < 0, mr.generate_signal(spiked)
    dipped = np.append(flat, [90.0])
    assert mr.generate_signal(dipped) > 0

    # 4) Short history -> flat.
    assert tf.generate_signal([100, 101]) == 0.0
    assert mr.generate_signal([100, 101]) == 0.0

    # 5) Sizing.
    assert target_shares_from_weight(0.5, 1_000_000, 100.0) == 5000
    assert target_shares_from_weight(-0.25, 1_000_000, 50.0) == -5000
    assert target_shares_from_weight(0.5, 1_000_000, 0.0) == 0  # guard bad price

    # 6) Aggregation of enabled strategies + clamping.
    allocs = [
        StrategyAllocation(TrendFollowing("AAPL", fast=10, slow=30), allocation_pct=60, enabled=True),
        StrategyAllocation(Momentum("AAPL", lookback=50), allocation_pct=60, enabled=True),
        StrategyAllocation(MeanReversion("MSFT", lookback=30, entry_z=2.0), allocation_pct=40, enabled=False),
    ]
    hist = {"AAPL": up, "MSFT": spiked}
    weights = aggregate_target_weights(allocs, hist)
    assert "MSFT" not in weights, "disabled strategy must not contribute"
    assert 0.0 < weights["AAPL"] <= 1.0, weights  # two long strategies blended

    # Clamping: three overlapping longs at 60% each must clamp net weight to 1.0.
    clamp_allocs = [
        StrategyAllocation(Momentum("AAPL", lookback=50), allocation_pct=60, enabled=True)
        for _ in range(3)
    ]
    clamped = aggregate_target_weights(clamp_allocs, hist)
    assert clamped["AAPL"] == 1.0, clamped

    # 7) Full order plan pipeline (expectations derived from the actual weight).
    plans = build_order_plans(
        allocs, hist,
        latest_price={"AAPL": 120.0},
        current_positions={"AAPL": 1000},
        net_liq=1_000_000,
    )
    plan = plans[0]
    assert plan.symbol == "AAPL"
    assert plan.target_shares == round(1_000_000 * weights["AAPL"] / 120.0)
    assert plan.delta_shares == plan.target_shares - 1000
    assert plan.action in ("BUY", "SELL", "HOLD")

    print("strategies.py self-test: ALL PASSED")


if __name__ == "__main__":
    _self_test()
