"""
order_book.py
=============

A **limit order book (LOB)** with a **price-time-priority matching engine** for
the *Order Book Simulator* educational project.

This module is *pure* (no tkinter, no I/O, no global state) so it can be unit
tested headlessly and reused anywhere. The GUI in ``app.py`` and the flow
generator in ``market_sim.py`` are thin clients on top of the API defined here.

--------------------------------------------------------------------------
What is a limit order book?
--------------------------------------------------------------------------
An exchange keeps, for a single instrument, two sorted collections of resting
orders:

* the **bids** -- orders to *buy* at a stated price or better, sorted so the
  **highest** price is most aggressive (best bid), and
* the **asks** (offers) -- orders to *sell* at a stated price or better, sorted
  so the **lowest** price is most aggressive (best ask).

The gap between the best bid and best ask is the **spread**; their average is
the **mid** price. An order that would trade immediately (a buy priced at or
above the best ask, a sell priced at or below the best bid, or any market
order) is *marketable* and is matched against the resting book. Anything that
cannot trade immediately *rests* in the book, adding liquidity.

--------------------------------------------------------------------------
Price-time priority (the matching rule)
--------------------------------------------------------------------------
When an incoming (aggressive / *taker*) order crosses the spread it is matched
against resting (*maker*) orders in a strict order:

1. **Price priority** -- the best price is served first. A buyer takes the
   cheapest offers first; a seller hits the highest bids first.
2. **Time priority** -- within one price level, the order that arrived *first*
   is filled first (FIFO). Resting early is rewarded.

Trades print at the **resting (maker) order's price**, so an aggressor can
receive *price improvement* relative to its own limit.

--------------------------------------------------------------------------
Data structures & complexity  (L = number of distinct price levels)
--------------------------------------------------------------------------
* Price levels live in two ``SortedDict`` maps (one per side) keyed by an
  integer number of ticks. This keeps levels sorted and gives O(log L) access
  to the best price and O(log L) insertion of a brand-new level.
* Each price level (:class:`PriceLevel`) holds its resting orders in an
  ``OrderedDict`` keyed by order id. Insertion order *is* time priority, so the
  FIFO front is ``next(iter(...))`` and cancellation by id is O(1) within the
  level.
* A global ``_orders`` index maps ``order_id -> Order`` so cancel/lookup is
  O(1) plus the O(log L) to reach the level.

Resulting big-O per operation:

    add resting limit order   : O(log L)      (find/create the level, append)
    cancel by id              : O(log L)      (find level, O(1) OrderedDict pop)
    match a marketable order  : O(k + m log L) (k orders touched, m levels
                                                fully consumed and removed)
    best bid / best ask       : O(log L)      (``SortedDict.peekitem``)

Every fill conserves shares: the quantity bought exactly equals the quantity
sold, which the test-suite checks explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Iterable, Iterator, Optional

from sortedcontainers import SortedDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Side(Enum):
    """Which side of the book an order sits on."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    """Limit orders carry a price; market orders take whatever is resting."""

    LIMIT = "limit"
    MARKET = "market"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class Order:
    """A single order tracked by the book.

    ``remaining`` shrinks as the order is (partially) filled. ``seq`` is a
    monotonically increasing sequence number assigned on arrival; it is the
    tie-breaker that encodes *time priority* independent of any wall clock.
    """

    id: int
    side: Side
    type: OrderType
    quantity: int
    remaining: int
    price: Optional[float] = None  # None for market orders
    seq: int = 0
    owner: str = "sim"  # "sim", "user", ... purely for display/attribution

    @property
    def is_filled(self) -> bool:
        return self.remaining <= 0


@dataclass
class Trade:
    """An execution produced when an aggressor crosses the spread.

    The trade prints at ``price`` (the resting maker's price). ``aggressor``
    is the side of the incoming order that caused the match, which is what
    the trade tape colours by ("who lifted/hit whom").
    """

    price: float
    size: int
    aggressor: Side
    maker_id: int
    taker_id: int
    seq: int


@dataclass
class PriceLevel:
    """All resting orders at one price, in strict time (FIFO) order.

    Backed by an ``OrderedDict`` so that iteration yields orders oldest-first
    (time priority) while cancellation by id stays O(1). ``total`` caches the
    summed ``remaining`` quantity so depth queries don't rescan the level.
    """

    price: float
    orders: "OrderedDict[int, Order]" = field(default_factory=OrderedDict)
    total: int = 0

    def append(self, order: Order) -> None:
        self.orders[order.id] = order
        self.total += order.remaining

    def remove(self, order: Order) -> None:
        if order.id in self.orders:
            del self.orders[order.id]
            self.total -= order.remaining

    @property
    def is_empty(self) -> bool:
        return not self.orders


# ---------------------------------------------------------------------------
# The matching engine
# ---------------------------------------------------------------------------
class OrderBook:
    """A single-instrument limit order book with a matching engine.

    Prices are quantised to an integer number of ``tick_size`` ticks internally
    so that price levels are exact dictionary keys (no floating-point drift).
    The public API always speaks in real prices (floats).
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        self.tick_size = tick_size

        # Both sides keyed by integer ticks. Bids: best == largest key;
        # asks: best == smallest key. SortedDict keeps them ordered for us.
        self._bids: "SortedDict[int, PriceLevel]" = SortedDict()
        self._asks: "SortedDict[int, PriceLevel]" = SortedDict()

        # id -> Order, for O(1) lookup and cancellation.
        self._orders: dict[int, Order] = {}

        self._id_gen = count(1)
        self._seq_gen = count(1)

    # -- tick <-> price helpers --------------------------------------------
    def _to_ticks(self, price: float) -> int:
        return int(round(price / self.tick_size))

    def _to_price(self, ticks: int) -> float:
        return round(ticks * self.tick_size, 10)

    def _side_map(self, side: Side) -> "SortedDict[int, PriceLevel]":
        return self._bids if side is Side.BUY else self._asks

    # -- top-of-book & summary statistics ----------------------------------
    def best_bid(self) -> Optional[float]:
        """Highest bid price, or ``None`` if there are no bids."""
        if not self._bids:
            return None
        return self._to_price(self._bids.keys()[-1])

    def best_ask(self) -> Optional[float]:
        """Lowest ask price, or ``None`` if there are no asks."""
        if not self._asks:
            return None
        return self._to_price(self._asks.keys()[0])

    def spread(self) -> Optional[float]:
        """Best ask minus best bid, or ``None`` if either side is empty."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return round(ask - bid, 10)

    def mid(self) -> Optional[float]:
        """Arithmetic mid price, or ``None`` if either side is empty."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return round((ask + bid) / 2.0, 10)

    def depth(
        self, side: Side, levels: Optional[int] = None
    ) -> list[tuple[float, int]]:
        """Return ``[(price, total_qty), ...]`` most-aggressive-first.

        Bids come back highest-price-first, asks lowest-price-first. Pass
        ``levels`` to cap how many price levels are returned (e.g. for a
        depth-of-book ladder).
        """
        book = self._side_map(side)
        # Bids: iterate keys from the top (highest); asks from the bottom.
        keys = reversed(book.keys()) if side is Side.BUY else iter(book.keys())
        out: list[tuple[float, int]] = []
        for k in keys:
            out.append((self._to_price(k), book[k].total))
            if levels is not None and len(out) >= levels:
                break
        return out

    def total_volume(self, side: Side) -> int:
        """Sum of all resting quantity on one side of the book."""
        book = self._side_map(side)
        return sum(level.total for level in book.values())

    def get_order(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._orders

    # -- order entry --------------------------------------------------------
    def submit_limit(
        self, side: Side, price: float, quantity: int, owner: str = "sim"
    ) -> tuple[Order, list[Trade]]:
        """Submit a limit order.

        The order first matches against any resting liquidity it crosses
        (price-time priority); whatever quantity is left over *rests* in the
        book as passive liquidity. Returns the created :class:`Order` (with its
        assigned id and residual ``remaining``) and the list of trades it
        generated.
        """
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer.")
        order = Order(
            id=next(self._id_gen),
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            remaining=quantity,
            price=self._to_price(self._to_ticks(price)),
            seq=next(self._seq_gen),
            owner=owner,
        )
        trades = self._match(order)
        if not order.is_filled:
            self._rest(order)
        return order, trades

    def submit_market(
        self, side: Side, quantity: int, owner: str = "sim"
    ) -> tuple[Order, list[Trade]]:
        """Submit a market order.

        A market order sweeps the opposite side for as much as it can, ignoring
        price. Any quantity that cannot be filled (book exhausted) is dropped --
        market orders never rest. Returns the order (its ``remaining`` reveals
        any unfilled shortfall) and the trades generated.
        """
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer.")
        order = Order(
            id=next(self._id_gen),
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            remaining=quantity,
            price=None,
            seq=next(self._seq_gen),
            owner=owner,
        )
        trades = self._match(order)
        # Unfilled remainder of a market order is simply not booked.
        return order, trades

    def cancel(self, order_id: int) -> Optional[Order]:
        """Cancel a resting order by id.

        Returns the removed :class:`Order`, or ``None`` if the id is unknown
        (already filled, already cancelled, or never existed). O(log L).
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        book = self._side_map(order.side)
        ticks = self._to_ticks(order.price)  # type: ignore[arg-type]
        level = book.get(ticks)
        if level is not None:
            level.remove(order)
            if level.is_empty:
                del book[ticks]
        return order

    # -- internal: matching & resting --------------------------------------
    def _rest(self, order: Order) -> None:
        """Insert an order's residual into the book as passive liquidity."""
        book = self._side_map(order.side)
        ticks = self._to_ticks(order.price)  # type: ignore[arg-type]
        level = book.get(ticks)
        if level is None:
            level = PriceLevel(price=self._to_price(ticks))
            book[ticks] = level
        level.append(order)
        self._orders[order.id] = order

    def _crosses(self, taker: Order, best_price: float) -> bool:
        """Would ``taker`` trade against a resting order at ``best_price``?

        Market orders always cross. A limit buy crosses when its price is at or
        above the resting ask; a limit sell crosses at or below the resting bid.
        """
        if taker.type is OrderType.MARKET:
            return True
        assert taker.price is not None
        if taker.side is Side.BUY:
            return taker.price >= best_price
        return taker.price <= best_price

    def _match(self, taker: Order) -> list[Trade]:
        """Match an incoming order against the opposite side of the book.

        Walks resting price levels from the most aggressive inward, filling
        FIFO within each level, until the taker is exhausted or no more resting
        orders cross its price. Fully consumed levels and fully filled maker
        orders are removed as we go.
        """
        trades: list[Trade] = []
        opposite = self._side_map(taker.side.opposite)

        while taker.remaining > 0 and opposite:
            # Best resting price on the opposite side.
            if taker.side is Side.BUY:
                best_ticks = opposite.keys()[0]          # lowest ask
            else:
                best_ticks = opposite.keys()[-1]         # highest bid
            level = opposite[best_ticks]

            if not self._crosses(taker, level.price):
                break  # price priority: nothing left that we can trade with

            # Time priority: fill oldest resting orders at this level first.
            while taker.remaining > 0 and level.orders:
                maker_id, maker = next(iter(level.orders.items()))
                fill = min(taker.remaining, maker.remaining)

                trades.append(
                    Trade(
                        price=level.price,   # trade prints at the maker's price
                        size=fill,
                        aggressor=taker.side,
                        maker_id=maker_id,
                        taker_id=taker.id,
                        seq=next(self._seq_gen),
                    )
                )

                taker.remaining -= fill
                maker.remaining -= fill
                level.total -= fill

                if maker.remaining <= 0:
                    del level.orders[maker_id]
                    self._orders.pop(maker_id, None)

            if level.is_empty:
                del opposite[best_ticks]

        return trades

    # -- convenience / snapshots -------------------------------------------
    def snapshot(self, levels: int = 10) -> dict:
        """A plain-dict snapshot of the top of book (handy for the GUI/tests)."""
        return {
            "bids": self.depth(Side.BUY, levels),
            "asks": self.depth(Side.SELL, levels),
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "spread": self.spread(),
            "mid": self.mid(),
        }

    def resting_orders(self, side: Side) -> Iterator[Order]:
        """Yield resting orders on ``side`` in (price, time) priority order."""
        book = self._side_map(side)
        keys = reversed(book.keys()) if side is Side.BUY else iter(book.keys())
        for k in keys:
            yield from book[k].orders.values()

    def clear(self) -> None:
        """Reset the book to empty (ids/sequence counters keep advancing)."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()


__all__ = [
    "Side",
    "OrderType",
    "Order",
    "Trade",
    "PriceLevel",
    "OrderBook",
]
