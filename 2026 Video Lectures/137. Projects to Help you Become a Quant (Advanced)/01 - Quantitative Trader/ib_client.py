"""
ib_client.py
============

Connectivity layer for the Algorithmic Trading System cockpit.

Responsibilities
----------------
1. Wrap the Interactive Brokers ``ibapi`` EWrapper/EClient and run its blocking
   message loop on a *background thread* so the tkinter main loop is never
   blocked.
2. Expose a single, thread-safe :class:`TradingBackend` facade that the GUI
   talks to. The GUI never touches ibapi directly.
3. Provide a fully functional **DEMO / disconnected mode**: if TWS / IB Gateway
   is unavailable (connection refused or handshake times out), the backend
   transparently falls back to a simulated account and an internal random-walk
   price feed so the entire application demos end-to-end offline.

Thread-safety model
-------------------
* All mutable state (prices, price history, account, positions) lives inside
  :class:`TradingBackend` and is guarded by a single re-entrant lock.
* Callbacks from the ibapi reader thread and ticks from the demo feed thread
  both mutate that state *under the lock*.
* The backend also pushes lightweight, human-readable events (status changes,
  logs, fills) onto a ``queue.Queue``. The GUI drains this queue from the
  tkinter main thread via ``root.after(...)`` — the classic, safe pattern for
  moving data from worker threads to Tk. The GUI reads numeric state via
  :meth:`TradingBackend.snapshot`, which returns an immutable copy.

Nothing in this module imports tkinter, so it can be smoke/unit tested headless.
Run ``python ib_client.py`` for a headless self-test of the demo backend.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

# ibapi is required for live connectivity but the app must still run without a
# live TWS. We import lazily-guarded so a missing/broken install degrades to
# DEMO mode instead of crashing the whole app on import.
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order

    IBAPI_AVAILABLE = True
    IBAPI_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - only hit when ibapi is absent
    IBAPI_AVAILABLE = False
    IBAPI_IMPORT_ERROR = str(exc)

    # Minimal stand-ins so the rest of the module still imports/type-checks.
    class EClient:  # type: ignore
        def __init__(self, *a, **k):
            pass

    class EWrapper:  # type: ignore
        pass

    class Contract:  # type: ignore
        pass

    class Order:  # type: ignore
        pass


# Account summary tags we care about (maps directly to IBKR reqAccountSummary).
ACCOUNT_TAGS = (
    "NetLiquidation",
    "TotalCashValue",
    "BuyingPower",
    "GrossPositionValue",
    "UnrealizedPnL",
    "RealizedPnL",
)

# Error codes that IBKR emits as informational connection notices, not errors.
BENIGN_ERROR_CODES = {2104, 2106, 2158, 2107, 2119, 2100}

# Request id namespace.
REQ_ACCOUNT_SUMMARY = 9001
MKT_DATA_BASE = 1000  # per-symbol market data reqId = base + index


# ---------------------------------------------------------------------------
# Event + snapshot value objects passed to the GUI
# ---------------------------------------------------------------------------
@dataclass
class BackendEvent:
    """A message from the backend to the GUI (drained via root.after)."""

    kind: str  # 'status' | 'log' | 'fill' | 'error'
    message: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.price - self.avg_cost)


# ---------------------------------------------------------------------------
# ibapi wrapper (only used when a live connection is established)
# ---------------------------------------------------------------------------
class _IBApi(EWrapper, EClient):
    """Thin ibapi wrapper that forwards callbacks into a :class:`TradingBackend`.

    Kept deliberately small: it does no business logic, it only translates
    ibapi callbacks into ``backend`` method calls (which are themselves locked).
    """

    def __init__(self, backend: "TradingBackend"):
        EClient.__init__(self, self)
        self.backend = backend
        self.next_order_id: Optional[int] = None
        self._connected_evt = threading.Event()

    # -- connection lifecycle ------------------------------------------------
    def nextValidId(self, orderId: int):
        # Called once the API handshake completes; the signal that we're live.
        self.next_order_id = orderId
        self._connected_evt.set()
        self.backend._log(f"IBKR handshake complete. next valid order id = {orderId}")

    def connectionClosed(self):  # pragma: no cover - needs live socket
        self.backend._on_disconnect("IBKR connection closed by peer")

    def managedAccounts(self, accountsList: str):  # pragma: no cover
        self.backend._log(f"Managed accounts: {accountsList}")

    # -- robust error handler (supports old AND new ibapi signatures) --------
    def error(self, *args, **kwargs):
        """Handle ``error`` across ibapi versions.

        Old (<=9.81):  error(reqId, errorCode, errorString)
        4-arg variant: error(reqId, errorCode, errorString, advancedOrderRejectJson)
        New (>=10.19): error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
        """
        req_id = args[0] if args else -1
        code: Optional[int] = None
        msg: str = ""
        rest = list(args[1:])

        if len(rest) >= 3:
            # New style: (errorTime, errorCode, errorString, [adv])
            code, msg = rest[1], rest[2]
        elif len(rest) == 2:
            # Old 3-arg style: (errorCode, errorString)
            code, msg = rest[0], rest[1]
        elif len(rest) == 1:
            msg = str(rest[0])
        else:
            code = kwargs.get("errorCode")
            msg = kwargs.get("errorString", "")

        try:
            code_int = int(code) if code is not None else None
        except (TypeError, ValueError):
            code_int = None

        if code_int in BENIGN_ERROR_CODES:
            self.backend._log(f"IBKR info {code_int}: {msg}")
            return
        self.backend._on_error(req_id, code_int, str(msg))

    # -- account summary -----------------------------------------------------
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        self.backend._on_account_value(tag, value)

    def accountSummaryEnd(self, reqId: int):  # pragma: no cover
        self.backend._log("Account summary snapshot complete.")

    # updateAccountValue is delivered when using reqAccountUpdates (streaming).
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        self.backend._on_account_value(key, val)

    def updatePortfolio(self, contract, position, marketPrice, marketValue,
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        self.backend._on_position(
            getattr(contract, "symbol", "?"), float(position), float(averageCost),
            price=float(marketPrice),
        )

    # -- positions -----------------------------------------------------------
    def position(self, account: str, contract, position, avgCost: float):
        self.backend._on_position(getattr(contract, "symbol", "?"), float(position), float(avgCost))

    def positionEnd(self):  # pragma: no cover
        self.backend._log("Positions snapshot complete.")

    # -- market data ---------------------------------------------------------
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        # 1=bid, 2=ask, 4=last, 9=close, 68=delayed last. Prefer last/close.
        if tickType in (4, 68, 9) and price and price > 0:
            symbol = self.backend._symbol_for_req(reqId)
            if symbol:
                self.backend._on_price(symbol, float(price))


# ---------------------------------------------------------------------------
# TradingBackend: the single facade used by the GUI
# ---------------------------------------------------------------------------
class TradingBackend:
    """Unified data/trading backend with transparent DEMO fallback.

    Parameters
    ----------
    symbols:
        Universe of tradable symbols to track prices for.
    host, port, client_id:
        IBKR TWS / Gateway connection params (default paper port 7497).
    connect_timeout:
        Seconds to wait for the IBKR handshake before falling back to demo.
    demo_start_cash:
        Starting cash for the simulated account in DEMO mode.
    """

    def __init__(
        self,
        symbols: List[str],
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        connect_timeout: float = 5.0,
        demo_start_cash: float = 1_000_000.0,
        price_history_len: int = 750,
        seed: Optional[int] = None,
    ):
        self.symbols = [s.upper().strip() for s in symbols]
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connect_timeout = connect_timeout
        self.demo_start_cash = demo_start_cash

        self._lock = threading.RLock()
        self.events: "queue.Queue[BackendEvent]" = queue.Queue()

        # Connection state.
        self.connected = False
        self.mode = "STARTING"  # 'IBKR' | 'DEMO'
        self._api: Optional[_IBApi] = None
        self._api_thread: Optional[threading.Thread] = None

        # Market state.
        self._rng = random.Random(seed)
        self._seed_prices = {s: round(self._rng.uniform(50, 400), 2) for s in self.symbols}
        self.prices: Dict[str, float] = dict(self._seed_prices)
        self.history: Dict[str, Deque[float]] = {
            s: deque([self._seed_prices[s]], maxlen=price_history_len) for s in self.symbols
        }
        self._req_to_symbol: Dict[int, str] = {}

        # Account + positions (authoritative in DEMO; mirrored in IBKR mode).
        self.account: Dict[str, float] = {tag: 0.0 for tag in ACCOUNT_TAGS}
        self.account["TotalCashValue"] = demo_start_cash
        self.account["NetLiquidation"] = demo_start_cash
        self.account["BuyingPower"] = demo_start_cash * 4
        self.positions: Dict[str, Position] = {}
        self._realized_pnl = 0.0

        # Demo feed thread control.
        self._stop = threading.Event()
        self._demo_thread: Optional[threading.Thread] = None
        self.demo_volatility = 0.004  # per-tick lognormal-ish sigma

    # ================= public API used by the GUI =========================
    def start(self) -> None:
        """Attempt an IBKR connection; on failure, start the DEMO feed."""
        if IBAPI_AVAILABLE:
            ok = self._try_connect_ibkr()
        else:
            ok = False
            self._log(f"ibapi unavailable ({IBAPI_IMPORT_ERROR or 'not installed'}); using DEMO mode.")
        if not ok:
            self._start_demo()

    def stop(self) -> None:
        """Tear everything down cleanly (idempotent)."""
        self._stop.set()
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
        if self._demo_thread and self._demo_thread.is_alive():
            self._demo_thread.join(timeout=2.0)

    def snapshot(self) -> dict:
        """Return an immutable copy of all state the GUI needs to render."""
        with self._lock:
            self._recompute_account_locked()
            positions = [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "price": self.prices.get(p.symbol, p.price),
                    "market_value": p.quantity * self.prices.get(p.symbol, p.price),
                    "unrealized_pnl": p.quantity * (self.prices.get(p.symbol, p.price) - p.avg_cost),
                }
                for p in self.positions.values()
                if abs(p.quantity) > 1e-9
            ]
            return {
                "connected": self.connected,
                "mode": self.mode,
                "prices": dict(self.prices),
                "history": {s: list(h) for s, h in self.history.items()},
                "account": dict(self.account),
                "positions": positions,
            }

    def get_price_history(self, symbol: str) -> List[float]:
        with self._lock:
            return list(self.history.get(symbol.upper(), []))

    def add_symbol(self, symbol: str) -> None:
        """Register a new symbol (used when a strategy on a new ticker is added)."""
        symbol = symbol.upper().strip()
        with self._lock:
            if symbol in self.history:
                return
            seed = round(self._rng.uniform(50, 400), 2)
            self._seed_prices[symbol] = seed
            self.prices[symbol] = seed
            self.history[symbol] = deque([seed], maxlen=750)
            self.symbols.append(symbol)
        if self.connected and self._api is not None:  # pragma: no cover - live only
            self._subscribe_market_data(symbol)

    def submit_orders(self, order_plans, guarded: bool = True) -> List[str]:
        """Execute (or simulate) a list of OrderPlan objects.

        In DEMO mode the fills update the simulated account/positions instantly
        at the current price. In IBKR mode the ibapi Contract + Order are built
        and ``placeOrder`` is called. ``guarded`` is a caller-supplied safety
        flag: when True we refuse to place live orders (dry-run) so nothing ever
        fires unintentionally during development.
        """
        messages: List[str] = []
        for plan in order_plans:
            if plan.delta_shares == 0:
                continue
            if self.mode == "IBKR" and self.connected:
                messages.append(self._place_live_order(plan, guarded))
            else:
                messages.append(self._fill_demo_order(plan))
        return messages

    # ================= IBKR connection ====================================
    def _try_connect_ibkr(self) -> bool:
        """Try to connect to TWS/Gateway. Returns True on success."""
        self._log(f"Connecting to IBKR at {self.host}:{self.port} (clientId={self.client_id})...")
        try:
            self._api = _IBApi(self)
            self._api.connect(self.host, self.port, clientId=self.client_id)
        except Exception as exc:
            self._log(f"IBKR connect() raised: {exc!r}")
            return False

        # Run the blocking message loop on a background thread.
        self._api_thread = threading.Thread(
            target=self._run_api_loop, name="ibapi-reader", daemon=True
        )
        self._api_thread.start()

        # Wait for the handshake (nextValidId) or timeout -> demo fallback.
        if not self._api._connected_evt.wait(timeout=self.connect_timeout):
            self._log("IBKR handshake timed out; falling back to DEMO mode.")
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None
            return False

        with self._lock:
            self.connected = True
            self.mode = "IBKR"
        self._emit_status()
        self._request_streams()
        return True

    def _run_api_loop(self):  # pragma: no cover - needs live socket
        try:
            self._api.run()
        except Exception as exc:
            self._on_disconnect(f"ibapi reader loop error: {exc!r}")

    def _request_streams(self):  # pragma: no cover - needs live socket
        api = self._api
        if api is None:
            return
        api.reqAccountSummary(REQ_ACCOUNT_SUMMARY, "All", ",".join(ACCOUNT_TAGS))
        api.reqPositions()
        for idx, symbol in enumerate(self.symbols):
            self._subscribe_market_data(symbol, idx)

    def _subscribe_market_data(self, symbol: str, idx: Optional[int] = None):  # pragma: no cover
        api = self._api
        if api is None:
            return
        if idx is None:
            idx = self.symbols.index(symbol)
        req_id = MKT_DATA_BASE + idx
        with self._lock:
            self._req_to_symbol[req_id] = symbol
        contract = self._make_contract(symbol)
        # Fall back to delayed data (type 3) if no live subscription.
        try:
            api.reqMarketDataType(3)
        except Exception:
            pass
        api.reqMktData(req_id, contract, "", False, False, [])

    @staticmethod
    def _make_contract(symbol: str) -> "Contract":
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def _place_live_order(self, plan, guarded: bool) -> str:  # pragma: no cover - live only
        api = self._api
        if api is None or api.next_order_id is None:
            return f"{plan.symbol}: no order id available"
        contract = self._make_contract(plan.symbol)
        order = Order()
        order.action = plan.action  # BUY / SELL
        order.orderType = "MKT"
        order.totalQuantity = abs(plan.delta_shares)
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        if guarded:
            return (f"[DRY-RUN] would {order.action} {order.totalQuantity} {plan.symbol} "
                    f"(guard ON; live order suppressed)")
        oid = api.next_order_id
        api.next_order_id += 1
        api.placeOrder(oid, contract, order)
        return f"LIVE {order.action} {order.totalQuantity} {plan.symbol} (orderId={oid})"

    # ================= DEMO feed ==========================================
    def _start_demo(self):
        with self._lock:
            self.connected = False
            self.mode = "DEMO"
        self._log("Running in DEMO mode: simulated account + internal random-walk price feed.")
        self._emit_status()
        self._stop.clear()
        self._demo_thread = threading.Thread(target=self._demo_loop, name="demo-feed", daemon=True)
        self._demo_thread.start()

    def _demo_loop(self):
        """Random-walk price generator running ~1 Hz on a background thread."""
        while not self._stop.is_set():
            with self._lock:
                for symbol in list(self.history.keys()):
                    last = self.prices.get(symbol, 100.0)
                    shock = self._rng.gauss(0.0, self.demo_volatility)
                    drift = 0.00005  # slight upward drift for realism
                    new_price = max(0.5, last * (1.0 + drift + shock))
                    new_price = round(new_price, 2)
                    self.prices[symbol] = new_price
                    self.history[symbol].append(new_price)
            self._stop.wait(1.0)

    def _fill_demo_order(self, plan) -> str:
        """Instantly fill a rebalance order against the simulated book."""
        with self._lock:
            price = self.prices.get(plan.symbol, plan.price) or plan.price
            qty = plan.delta_shares
            pos = self.positions.get(plan.symbol) or Position(plan.symbol, 0.0, price)

            old_qty = pos.quantity
            new_qty = old_qty + qty

            # Cash impact: buying spends cash, selling adds cash.
            self.account["TotalCashValue"] -= qty * price

            # Realized P/L when reducing/closing an existing position.
            if old_qty != 0 and (old_qty > 0) != (qty > 0):
                closing = min(abs(qty), abs(old_qty))
                direction = 1 if old_qty > 0 else -1
                self._realized_pnl += direction * closing * (price - pos.avg_cost)

            # New average cost (only changes when adding in the same direction).
            if new_qty == 0:
                pos.avg_cost = 0.0
            elif (old_qty >= 0 and qty > 0) or (old_qty <= 0 and qty < 0):
                # Adding to (or opening) the position: weighted average cost.
                denom = old_qty + qty
                pos.avg_cost = ((old_qty * pos.avg_cost) + (qty * price)) / denom if denom else price
            elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                # Flipped through zero: the residual opens at the fill price.
                pos.avg_cost = price

            pos.quantity = new_qty
            pos.price = price
            self.positions[plan.symbol] = pos
            self.account["RealizedPnL"] = self._realized_pnl
            self._recompute_account_locked()

        msg = f"[DEMO FILL] {plan.action} {abs(qty)} {plan.symbol} @ {price:.2f}"
        self.events.put(BackendEvent("fill", msg, {"symbol": plan.symbol, "qty": qty, "price": price}))
        return msg

    # ================= locked state mutators (called by callbacks) =========
    def _on_price(self, symbol: str, price: float):
        with self._lock:
            self.prices[symbol] = price
            hist = self.history.setdefault(symbol, deque(maxlen=750))
            hist.append(price)

    def _on_account_value(self, tag: str, value: str):
        if tag not in ACCOUNT_TAGS:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        with self._lock:
            self.account[tag] = v

    def _on_position(self, symbol: str, quantity: float, avg_cost: float, price: float = 0.0):
        with self._lock:
            pos = self.positions.get(symbol) or Position(symbol, 0.0, avg_cost)
            pos.quantity = quantity
            pos.avg_cost = avg_cost
            if price:
                pos.price = price
                self.prices[symbol] = price
            self.positions[symbol] = pos

    def _recompute_account_locked(self):
        """Recompute derived account fields (used in DEMO; harmless in IBKR)."""
        if self.mode != "DEMO":
            return
        gross = 0.0
        unrealized = 0.0
        market_val = 0.0
        for p in self.positions.values():
            px = self.prices.get(p.symbol, p.price)
            gross += abs(p.quantity) * px
            unrealized += p.quantity * (px - p.avg_cost)
            market_val += p.quantity * px
        cash = self.account["TotalCashValue"]
        self.account["GrossPositionValue"] = gross
        self.account["UnrealizedPnL"] = unrealized
        self.account["NetLiquidation"] = cash + market_val
        self.account["BuyingPower"] = max(0.0, (cash + market_val) * 4 - gross)

    # ================= helpers ============================================
    def _symbol_for_req(self, req_id: int) -> Optional[str]:
        with self._lock:
            return self._req_to_symbol.get(req_id)

    def _emit_status(self):
        label = "Connected to IBKR" if self.connected else "DEMO mode (disconnected)"
        self.events.put(BackendEvent("status", label, {"connected": self.connected, "mode": self.mode}))

    def _on_disconnect(self, reason: str):  # pragma: no cover - live only
        with self._lock:
            self.connected = False
        self._log(reason)
        self._emit_status()

    def _on_error(self, req_id, code, msg):
        self.events.put(BackendEvent("error", f"IBKR error {code} (req {req_id}): {msg}",
                                     {"code": code, "req_id": req_id}))

    def _log(self, message: str):
        self.events.put(BackendEvent("log", message))


# ---------------------------------------------------------------------------
# Headless self-test of the DEMO backend (no GUI, no TWS)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from strategies import Momentum, StrategyAllocation, build_order_plans

    backend = TradingBackend(["AAPL", "MSFT"], connect_timeout=0.2, seed=42)
    # Force demo (no TWS in test env) and start the feed.
    backend._start_demo()
    try:
        time.sleep(2.2)  # let the random walk produce a few ticks
        snap = backend.snapshot()
        assert snap["mode"] == "DEMO"
        assert len(snap["history"]["AAPL"]) >= 2, snap["history"]["AAPL"]
        assert snap["account"]["NetLiquidation"] > 0

        # Simulate buying 100 AAPL via the order pipeline.
        from types import SimpleNamespace

        price = snap["prices"]["AAPL"]
        buy = SimpleNamespace(symbol="AAPL", delta_shares=100, action="BUY", price=price)

        cash_before = snap["account"]["TotalCashValue"]
        backend.submit_orders([buy], guarded=True)
        snap2 = backend.snapshot()
        pos = [p for p in snap2["positions"] if p["symbol"] == "AAPL"]
        assert pos and abs(pos[0]["quantity"] - 100) < 1e-6, snap2["positions"]
        assert snap2["account"]["TotalCashValue"] < cash_before, "cash should drop after a buy"

        # Sell 100 back -> flat, realized pnl recorded.
        sell = SimpleNamespace(symbol="AAPL", delta_shares=-100, action="SELL",
                               price=snap2["prices"]["AAPL"])
        backend.submit_orders([sell], guarded=True)
        snap3 = backend.snapshot()
        pos = [p for p in snap3["positions"] if p["symbol"] == "AAPL"]
        assert not pos, f"position should be flat, got {pos}"

        # Drain a few events to confirm the queue works.
        drained = 0
        while not backend.events.empty():
            backend.events.get_nowait()
            drained += 1
        assert drained > 0

        print("ib_client.py DEMO self-test: ALL PASSED "
              f"(ibapi_available={IBAPI_AVAILABLE}, ticks={len(snap['history']['AAPL'])})")
    finally:
        backend.stop()


if __name__ == "__main__":
    _self_test()
