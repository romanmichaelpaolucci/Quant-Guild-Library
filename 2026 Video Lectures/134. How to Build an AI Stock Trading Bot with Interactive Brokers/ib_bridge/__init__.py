"""Interactive Brokers session + request helpers (ib_insync over EClient/EWrapper)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import queue
import threading
import time
from typing import Any, Callable, TypeVar

# Python 3.14+: eventkit/ib_insync require a running loop at import time.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, LimitOrder, MarketOrder, Stock

from config import (
    get_ib_request_timeout,
    get_ib_settings,
    order_mode_status,
    orders_allowed,
)

F = TypeVar("F", bound=Callable[..., Any])

_ib: IB | None = None
# Settings used for the current singleton connect (detect readonly/env flips).
_ib_connect_key: tuple[str, int, int, bool] | None = None
_last_order_mono: float = 0.0
_ORDER_PACE_S = 2.0  # min gap between placeOrder calls
# Only contended on the IB worker thread (nested helpers); Flask never touches IB directly.
_lock = threading.RLock()


class _IBExecutor:
    """
    All ib_insync traffic MUST run on one dedicated thread.

    Flask is multi-threaded; crossing threads with a shared IB client leaves
    subsequent placeOrder calls stuck in PendingSubmit forever (first may work).
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="astb-ib-worker", daemon=True
        )
        self._ready = threading.Event()
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("IB worker thread failed to start")

    def _run(self) -> None:
        # Own the asyncio loop for this thread only.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            fn, args, kwargs, fut = job
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                fut.set_exception(exc)

    def call(self, fn: Callable[..., Any], *args: Any, timeout: float = 180.0, **kwargs: Any) -> Any:
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, args, kwargs, fut))
        return fut.result(timeout=timeout)


_executor: _IBExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> _IBExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = _IBExecutor()
        return _executor


def _on_ib_thread(fn: F) -> F:
    """Public IB APIs: always execute on the dedicated IB worker thread."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _get_executor().call(fn, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _ensure_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def disconnect_ib() -> None:
    def _disconnect() -> None:
        global _ib, _ib_connect_key
        if _ib is not None:
            try:
                if _ib.isConnected():
                    _ib.disconnect()
            finally:
                _ib = None
                _ib_connect_key = None

    _get_executor().call(_disconnect)


def get_ib() -> IB:
    """Return a connected IB session. Call only from the IB worker thread."""
    global _ib, _ib_connect_key
    worker = _get_executor()._thread
    if threading.current_thread() is not worker:
        raise RuntimeError(
            "get_ib() called off the IB worker thread — use get_portfolio / "
            "get_account_summary / place_equity_order (they route correctly)."
        )
    _ensure_loop()
    host, port, client_id, readonly = get_ib_settings()
    want_key = (host, port, client_id, readonly)

    if _ib is not None and _ib.isConnected():
        if _ib_connect_key == want_key:
            _ib.RequestTimeout = get_ib_request_timeout()
            return _ib
        try:
            _ib.disconnect()
        except Exception:
            pass
        _ib = None
        _ib_connect_key = None

    ib = IB()
    ib.RequestTimeout = get_ib_request_timeout()
    print(
        f"[IB] connecting {host}:{port} clientId={client_id} readonly={readonly}",
        flush=True,
    )
    ib.connect(
        host,
        port,
        clientId=client_id,
        readonly=readonly,
        timeout=15,
    )
    if not ib.isConnected():
        raise RuntimeError(
            f"Failed to connect to TWS/Gateway at {host}:{port}"
        )
    _ib = ib
    _ib_connect_key = want_key
    return _ib


def _ib_num(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        # IB uses large sentinel values for "not set"
        if f != f or abs(f) > 1e10:  # NaN or IB unset
            return None
        return f
    except (TypeError, ValueError):
        return None


def _ticker_has_price(ticker: Any) -> bool:
    for attr in ("last", "close", "bid", "ask", "marketPrice"):
        v = _ib_num(getattr(ticker, attr, None))
        if v is not None and v > 0:
            return True
    return False


def _stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    return Stock(symbol.upper().strip(), exchange, currency)


def _contract_from_book(ib: IB, symbol: str) -> Any | None:
    """Already-qualified contract from portfolio or positions, if any."""
    sym = symbol.upper().strip()
    for item in ib.portfolio() or []:
        c = item.contract
        if str(getattr(c, "symbol", "")).upper() != sym:
            continue
        if str(getattr(c, "secType", "STK") or "STK") not in {"STK", "ETF", "WAR"}:
            continue
        if getattr(c, "conId", 0):
            return c
    for p in ib.positions() or []:
        c = p.contract
        if str(getattr(c, "symbol", "")).upper() != sym:
            continue
        if str(getattr(c, "secType", "STK") or "STK") not in {"STK", "ETF", "WAR"}:
            continue
        if getattr(c, "conId", 0):
            return c
    return None


def _resolve_equity_contract(ib: IB, symbol: str) -> Any:
    """
    Resolve a stock contract without depending solely on qualifyContracts.

    Prefer an already-qualified contract from portfolio/positions (fast, reliable
    for exits). Fall back to qualifyContracts; if that times out, use SMART stub.
    """
    sym = symbol.upper().strip()
    existing = _contract_from_book(ib, sym)
    if existing is not None:
        return _prepare_order_contract(existing)

    # Portfolio may be empty until account updates flow.
    try:
        _refresh_account_portfolio(ib)
    except Exception:
        pass
    existing = _contract_from_book(ib, sym)
    if existing is not None:
        return _prepare_order_contract(existing)

    contract = _stock(sym)
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            return _prepare_order_contract(qualified[0])
    except Exception:
        pass
    # Liquid US names usually accept SMART stock even if qualify timed out.
    return _prepare_order_contract(contract)


def _prepare_order_contract(contract: Any) -> Any:
    """
    Portfolio/position contracts often omit `exchange`, which makes IB reject
    placeOrder with Error 321 "Missing order exchange". Always route SMART.
    """
    try:
        # Copy so we don't mutate IB's live portfolio objects.
        c = contract if not hasattr(contract, "copy") else contract.copy()
        if c is contract:
            # Fallback shallow clone via Stock fields when copy() missing
            from ib_insync import Contract

            c = Contract()
            for attr in (
                "conId",
                "symbol",
                "secType",
                "lastTradeDateOrContractMonth",
                "strike",
                "right",
                "multiplier",
                "exchange",
                "primaryExchange",
                "currency",
                "localSymbol",
                "tradingClass",
                "includeExpired",
                "secIdType",
                "secId",
            ):
                if hasattr(contract, attr):
                    setattr(c, attr, getattr(contract, attr))
    except Exception:
        c = contract

    exch = str(getattr(c, "exchange", "") or "").strip()
    if not exch or exch.upper() in {"", "0", "NONE"}:
        c.exchange = "SMART"
    # Portfolio stubs sometimes set right='0' which is invalid for stocks.
    right = str(getattr(c, "right", "") or "").strip()
    if right in {"0", "None", "none"}:
        c.right = ""
    if not getattr(c, "currency", None):
        c.currency = "USD"
    if not getattr(c, "secType", None):
        c.secType = "STK"
    return c


def _portfolio_mark(ib: IB, symbol: str) -> float | None:
    sym = symbol.upper().strip()
    for item in ib.portfolio() or []:
        c = item.contract
        if str(getattr(c, "symbol", "")).upper() != sym:
            continue
        px = float(item.marketPrice or 0)
        if px > 0:
            return px
    return None


@_on_ib_thread
def qualify_stock(
    symbol: str, exchange: str = "SMART", currency: str = "USD"
) -> dict[str, Any]:
    """
    Resolve contract details. Prefer portfolio/positions conId so exits never
    depend on a flaky qualifyContracts round-trip.
    """
    with _lock:
        ib = get_ib()
        sym = symbol.upper().strip()

        existing = _contract_from_book(ib, sym)
        if existing is None:
            try:
                _refresh_account_portfolio(ib)
            except Exception:
                pass
            existing = _contract_from_book(ib, sym)

        if existing is not None:
            c = existing
            return {
                "ok": True,
                "source": "portfolio",
                "symbol": c.symbol,
                "conId": c.conId,
                "exchange": c.exchange,
                "primaryExchange": getattr(c, "primaryExchange", None),
                "currency": c.currency,
                "localSymbol": getattr(c, "localSymbol", None),
                "secType": c.secType,
                "note": (
                    "Resolved from live portfolio/positions — qualifyContracts "
                    "not required. For exits call propose_position_size(liquidate=true) "
                    "then place_equity_order(confirm=true)."
                ),
            }

        contract = _stock(sym, exchange, currency)
        try:
            qualified = ib.qualifyContracts(contract)
        except Exception as exc:
            # Still return a usable SMART stub so the agent can continue to place
            # (especially if the name is already held — book refresh may lag).
            return {
                "ok": True,
                "source": "smart_stub",
                "symbol": sym,
                "conId": 0,
                "exchange": exchange,
                "currency": currency,
                "secType": "STK",
                "warning": f"qualifyContracts timed out/failed: {type(exc).__name__}: {exc}",
                "note": (
                    "Do NOT stop. For held names use get_portfolio then "
                    "propose_position_size(liquidate=true) → place_equity_order(confirm=true); "
                    "place_equity_order resolves conId from the book."
                ),
            }
        if not qualified:
            return {
                "ok": True,
                "source": "smart_stub",
                "symbol": sym,
                "conId": 0,
                "exchange": exchange,
                "currency": currency,
                "secType": "STK",
                "warning": f"Could not qualify contract for {sym}",
                "note": (
                    "Do NOT stop. Proceed with place_equity_order for held positions."
                ),
            }
        c = qualified[0]
        return {
            "ok": True,
            "source": "qualifyContracts",
            "symbol": c.symbol,
            "conId": c.conId,
            "exchange": c.exchange,
            "primaryExchange": c.primaryExchange,
            "currency": c.currency,
            "localSymbol": c.localSymbol,
            "secType": c.secType,
        }


@_on_ib_thread
def get_news_providers() -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        try:
            providers = ib.reqNewsProviders()
        except Exception as exc:
            return {
                "ok": False,
                "error": f"reqNewsProviders failed: {type(exc).__name__}: {exc}",
            }
        return {
            "ok": True,
            "providers": [{"code": p.code, "name": p.name} for p in providers],
        }


@_on_ib_thread
def get_historical_news(
    symbol: str,
    provider_codes: str | None = None,
    start_datetime: str = "",
    end_datetime: str = "",
    max_results: int = 10,
) -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        contract = _stock(symbol)
        try:
            if not ib.qualifyContracts(contract):
                return {"ok": False, "error": f"Could not qualify {symbol}"}

            if not provider_codes:
                providers = ib.reqNewsProviders()
                if not providers:
                    return {"ok": False, "error": "No news providers available"}
                provider_codes = "+".join(p.code for p in providers)

            headlines = ib.reqHistoricalNews(
                contract.conId,
                provider_codes,
                start_datetime,
                end_datetime,
                min(max(1, max_results), 100),
            )
        except Exception as exc:
            return {
                "ok": False,
                "symbol": symbol.upper(),
                "error": f"historical news failed: {type(exc).__name__}: {exc}",
            }
        rows = []
        for h in headlines or []:
            rows.append(
                {
                    "time": str(h.time),
                    "providerCode": h.providerCode,
                    "articleId": h.articleId,
                    "headline": h.headline,
                }
            )
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "conId": contract.conId,
            "providerCodes": provider_codes,
            "headlines": rows,
        }


@_on_ib_thread
def get_news_article(provider_code: str, article_id: str) -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        try:
            article = ib.reqNewsArticle(provider_code, article_id)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"reqNewsArticle failed: {type(exc).__name__}: {exc}",
            }
        text = article.articleText or ""
        # Cap size so tool results stay within model context budgets
        max_chars = 12_000
        truncated = len(text) > max_chars
        return {
            "ok": True,
            "providerCode": provider_code,
            "articleId": article_id,
            "articleType": getattr(article, "articleType", None),
            "truncated": truncated,
            "articleText": text[:max_chars],
        }


@_on_ib_thread
def get_historical_bars(
    symbol: str,
    duration: str = "1 M",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    end_datetime: str = "",
) -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        contract = _stock(symbol)
        try:
            if not ib.qualifyContracts(contract):
                return {"ok": False, "error": f"Could not qualify {symbol}"}

            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )
        except Exception as exc:
            return {
                "ok": False,
                "symbol": symbol.upper(),
                "error": f"historical bars failed: {type(exc).__name__}: {exc}",
            }
        rows = []
        for b in bars or []:
            rows.append(
                {
                    "date": str(b.date),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "average": getattr(b, "average", None),
                    "barCount": getattr(b, "barCount", None),
                }
            )
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "conId": contract.conId,
            "duration": duration,
            "barSize": bar_size,
            "whatToShow": what_to_show,
            "useRTH": use_rth,
            "bars": rows,
        }


@_on_ib_thread
def get_market_snapshot(symbol: str) -> dict[str, Any]:
    """One-shot market data snapshot (EClient.reqMktData → ticker fields)."""
    sym = symbol.upper().strip()
    with _lock:
        ib = get_ib()
        contract = _stock(sym)
        try:
            qualified = ib.qualifyContracts(contract)
        except Exception as exc:
            return {
                "ok": False,
                "symbol": sym,
                "error": f"qualifyContracts failed: {type(exc).__name__}: {exc}",
            }
        if not qualified:
            return {"ok": False, "symbol": sym, "error": f"Could not qualify {sym}"}

        ticker = None
        try:
            ticker = ib.reqMktData(
                contract,
                genericTickList="",
                snapshot=True,
                regulatorySnapshot=False,
            )
            # Bound wait: poll until a usable price arrives or deadline hits.
            # Blind ib.sleep(2) can hang forever if another Flask thread races the IB client.
            wait_s = min(5.0, max(1.0, get_ib_request_timeout() or 5.0))
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if _ticker_has_price(ticker):
                    break
                ib.sleep(0.2)

            result = {
                "ok": True,
                "symbol": sym,
                "conId": contract.conId,
                "bid": _ib_num(ticker.bid),
                "ask": _ib_num(ticker.ask),
                "last": _ib_num(ticker.last),
                "close": _ib_num(ticker.close),
                "high": _ib_num(ticker.high),
                "low": _ib_num(ticker.low),
                "volume": _ib_num(ticker.volume),
                "bidSize": _ib_num(ticker.bidSize),
                "askSize": _ib_num(ticker.askSize),
                "lastSize": _ib_num(ticker.lastSize),
                "time": str(ticker.time) if ticker.time else None,
            }
            if not _ticker_has_price(ticker):
                result["ok"] = False
                result["error"] = (
                    "Market snapshot timed out with no bid/ask/last/close. "
                    "Check TWS market-data subscriptions, or use portfolio marketPrice / historical bars."
                )
            return result
        except Exception as exc:
            return {
                "ok": False,
                "symbol": sym,
                "error": f"market snapshot failed: {type(exc).__name__}: {exc}",
            }
        finally:
            if ticker is not None:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass


@_on_ib_thread
def get_account_summary() -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        rows = ib.accountSummary()
        if not rows:
            # Force a refresh
            ib.reqAccountSummary()
            ib.sleep(1)
            rows = ib.accountSummary()
        summary = [
            {
                "account": r.account,
                "tag": r.tag,
                "value": r.value,
                "currency": r.currency,
            }
            for r in rows
        ]
        return {"ok": True, "accounts": ib.managedAccounts(), "summary": summary}


@_on_ib_thread
def get_positions() -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        positions = []
        for p in ib.positions():
            c = p.contract
            positions.append(
                {
                    "account": p.account,
                    "symbol": c.symbol,
                    "secType": c.secType,
                    "currency": c.currency,
                    "exchange": c.exchange,
                    "conId": c.conId,
                    "position": p.position,
                    "avgCost": p.avgCost,
                }
            )
        return {"ok": True, "positions": positions}


def _refresh_account_portfolio(ib: IB) -> None:
    """Ensure account updates are flowing so portfolio() is populated."""
    accounts = ib.managedAccounts()
    if not accounts:
        ib.sleep(0.5)
        accounts = ib.managedAccounts()
    for acct in accounts or []:
        ib.reqAccountUpdates(True, acct)
    if accounts:
        ib.sleep(1.0)


@_on_ib_thread
def get_portfolio() -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        portfolio_items = list(ib.portfolio())
        if not portfolio_items:
            _refresh_account_portfolio(ib)
            portfolio_items = list(ib.portfolio())

        items = []
        for item in portfolio_items:
            c = item.contract
            items.append(
                {
                    "account": item.account,
                    "symbol": c.symbol,
                    "secType": c.secType,
                    "conId": c.conId,
                    "position": item.position,
                    "marketPrice": item.marketPrice,
                    "marketValue": item.marketValue,
                    "averageCost": item.averageCost,
                    "unrealizedPNL": item.unrealizedPNL,
                    "realizedPNL": item.realizedPNL,
                }
            )
        return {"ok": True, "portfolio": items}


def _trade_snapshot(trade: Any) -> dict[str, Any]:
    status = trade.orderStatus
    return {
        "orderId": trade.order.orderId,
        "permId": getattr(trade.order, "permId", None),
        "action": trade.order.action,
        "totalQuantity": float(trade.order.totalQuantity),
        "orderType": trade.order.orderType,
        "lmtPrice": getattr(trade.order, "lmtPrice", None),
        "status": status.status,
        "filled": float(status.filled or 0),
        "remaining": float(status.remaining or 0),
        "avgFillPrice": float(status.avgFillPrice or 0),
        "whyHeld": status.whyHeld or "",
        "symbol": trade.contract.symbol if trade.contract else None,
    }


# Live / in-flight statuses — do not stack a same-side replacement on these.
_WORKING_ORDER_STATUSES = frozenset(
    {
        "PendingSubmit",
        "PendingCancel",
        "PreSubmitted",
        "Submitted",
        "ApiPending",
        "PartiallyFilled",
    }
)


def _is_working_order_snapshot(snap: dict[str, Any]) -> bool:
    status = str(snap.get("status") or "")
    remaining = float(snap.get("remaining") or 0)
    if remaining <= 0 and status not in _WORKING_ORDER_STATUSES:
        return False
    if status in {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}:
        return False
    return status in _WORKING_ORDER_STATUSES or remaining > 0


def _working_orders_for_symbol_side(
    ib: Any, symbol: str, side: str
) -> list[dict[str, Any]]:
    """Open/working trades for symbol+side (caller must hold IB lock)."""
    sym = symbol.upper().strip()
    side_u = side.upper().strip()
    try:
        ib.reqOpenOrders()
        ib.sleep(0.35)
    except Exception:
        pass
    out: list[dict[str, Any]] = []
    for trade in ib.openTrades():
        snap = _trade_snapshot(trade)
        if str(snap.get("symbol") or "").upper() != sym:
            continue
        if str(snap.get("action") or "").upper() != side_u:
            continue
        if _is_working_order_snapshot(snap):
            out.append(snap)
    return out


@_on_ib_thread
def get_open_orders() -> dict[str, Any]:
    with _lock:
        ib = get_ib()
        ib.reqOpenOrders()
        ib.sleep(0.5)
        orders = [_trade_snapshot(t) for t in ib.openTrades()]
        working = [o for o in orders if _is_working_order_snapshot(o)]
        return {"ok": True, "open_orders": orders, "working_orders": working}


@_on_ib_thread
def cancel_order(order_id: int, confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        return {
            "ok": False,
            "error": "confirm=true is required to cancel an order",
        }
    if not orders_allowed():
        return {
            "ok": False,
            "error": (
                "Order cancellation blocked by current env flags. "
                "Set IB_ALLOW_ORDERS=true and IB_READONLY=false in .env."
            ),
            "order_mode": order_mode_status(),
        }

    with _lock:
        ib = get_ib()
        trade = next((t for t in ib.trades() if t.order.orderId == int(order_id)), None)
        if trade is None:
            # Try open trades only
            trade = next(
                (t for t in ib.openTrades() if t.order.orderId == int(order_id)), None
            )
        if trade is None:
            return {"ok": False, "error": f"No trade found for orderId={order_id}"}

        ib.cancelOrder(trade.order)
        ib.sleep(1)
        return {"ok": True, "cancelled": _trade_snapshot(trade)}


@_on_ib_thread
def place_equity_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    confirm: bool = False,
    thesis_note: str = "",
) -> dict[str, Any]:
    """
    Place a stock order via IB (EClient.placeOrder).

    Safety: requires confirm=true AND IB_ALLOW_ORDERS=true with IB_READONLY=false.
    """
    if not confirm:
        return {
            "ok": False,
            "error": (
                "confirm=true is required to place an order. "
                "Call propose_position_size first, then resubmit with confirm=true."
            ),
            "preview": {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "quantity": quantity,
                "order_type": order_type.upper(),
                "limit_price": limit_price,
                "tif": tif,
                "thesis_note": thesis_note,
            },
        }

    if not orders_allowed():
        mode = order_mode_status()
        return {
            "ok": False,
            "error": (
                "Order placement blocked by current env flags "
                f"(IB_ALLOW_ORDERS={mode['IB_ALLOW_ORDERS']}, "
                f"IB_READONLY={mode['IB_READONLY']}, "
                f"orders_allowed={mode['orders_allowed']}). "
                "Set IB_ALLOW_ORDERS=true and IB_READONLY=false in .env; "
                "the IB session will reconnect trading-enabled on the next call."
            ),
            "order_mode": mode,
        }

    side_u = side.upper().strip()
    if side_u not in {"BUY", "SELL"}:
        return {"ok": False, "error": "side must be BUY or SELL"}

    qty = int(quantity)
    if qty <= 0:
        return {"ok": False, "error": "quantity must be a positive integer"}

    otype = order_type.upper().strip()
    if otype not in {"MKT", "LMT"}:
        return {"ok": False, "error": "order_type must be MKT or LMT"}
    if otype == "LMT" and (limit_price is None or float(limit_price) <= 0):
        return {"ok": False, "error": "limit_price required for LMT orders"}

    from config import get_max_order_notional

    global _last_order_mono

    with _lock:
        ib = get_ib()
        contract = _resolve_equity_contract(ib, symbol)
        # Belt-and-suspenders: IB Error 321 if exchange blank on portfolio stubs.
        contract = _prepare_order_contract(contract)

        # Do not stack a duplicate same-side order while one is awaiting fill.
        working = _working_orders_for_symbol_side(ib, symbol, side_u)
        if working:
            return {
                "ok": False,
                "error": (
                    f"Working {side_u} order already open for {symbol.upper()} "
                    f"(orderId={working[0].get('orderId')}, "
                    f"status={working[0].get('status')}, "
                    f"remaining={working[0].get('remaining')}). "
                    "Do not replace or duplicate — wait for fill/cancel, or "
                    "cancel_order(confirm=true) only if intentionally superseding."
                ),
                "blocked_by_open_order": True,
                "working_orders": working,
                "hint": (
                    "Skip this symbol this cycle; leave the working order alone."
                ),
            }

        # Soft notional guard — BUY only. Use portfolio mark only (never nested
        # get_market_snapshot here — that races qualify/mktData against the next order).
        max_notional = get_max_order_notional()
        if side_u == "BUY" and max_notional > 0:
            px = None
            if otype == "LMT" and limit_price:
                px = float(limit_price)
            if px is None:
                px = _portfolio_mark(ib, symbol)
            if px:
                notional = qty * px
                if notional > max_notional:
                    return {
                        "ok": False,
                        "error": (
                            f"Order notional ${notional:,.2f} exceeds "
                            f"IB_MAX_ORDER_NOTIONAL=${max_notional:,.2f}"
                        ),
                    }

        # Pace sequential agent orders so IB can ack orderIds / status callbacks.
        gap = time.monotonic() - _last_order_mono
        if _last_order_mono > 0 and gap < _ORDER_PACE_S:
            ib.sleep(_ORDER_PACE_S - gap)

        if otype == "MKT":
            order = MarketOrder(side_u, qty)
        else:
            order = LimitOrder(side_u, qty, float(limit_price))

        order.tif = tif.upper()
        order.outsideRth = bool(outside_rth)
        # Force a fresh order id — reusing 0 lets ib_insync allocate; never reuse prior.
        order.orderId = 0
        if thesis_note:
            # IB orderRef max length is short; keep a compact tag
            order.orderRef = thesis_note[:32]

        try:
            trade = ib.placeOrder(contract, order)
            _last_order_mono = time.monotonic()
            # Wait for IB ack on THIS thread's event loop (worker). PendingSubmit
            # forever = callback never arrived (classic cross-thread IB bug).
            deadline = time.monotonic() + 8.0
            status = ""
            while time.monotonic() < deadline:
                ib.sleep(0.25)
                status = str(trade.orderStatus.status or "")
                if status and status not in {"PendingSubmit", "PendingCancel"}:
                    break
            # One flush attempt if still pending
            if status in {"", "PendingSubmit"}:
                try:
                    ib.reqOpenOrders()
                    ib.sleep(0.75)
                    status = str(trade.orderStatus.status or status)
                except Exception:
                    pass
            # Let TWS settle before the next symbol in an agent burst.
            ib.sleep(0.5)
        except Exception as exc:
            return {
                "ok": False,
                "symbol": symbol.upper(),
                "error": f"placeOrder failed: {type(exc).__name__}: {exc}",
                "contract": {
                    "symbol": getattr(contract, "symbol", symbol),
                    "conId": getattr(contract, "conId", None),
                    "exchange": getattr(contract, "exchange", None),
                },
            }

        snap = _trade_snapshot(trade)
        status = str(snap.get("status") or status or "")
        print(
            f"[IB] place_equity_order {side_u} {qty} {symbol.upper()} "
            f"orderId={snap.get('orderId')} status={status} "
            f"exchange={getattr(contract, 'exchange', None)} "
            f"conId={getattr(contract, 'conId', None)} "
            f"thread={threading.current_thread().name}",
            flush=True,
        )

        # Surface IB validation cancels (e.g. Error 321 missing exchange) as failures.
        if status.lower() in {"cancelled", "apicancelled", "inactive"}:
            why = ""
            try:
                for entry in reversed(list(trade.log or [])):
                    msg = getattr(entry, "message", "") or ""
                    if msg:
                        why = msg
                        break
            except Exception:
                pass
            return {
                "ok": False,
                "submitted": False,
                "symbol": symbol.upper(),
                "error": why or f"Order {status} by IB",
                "order": snap,
                "contract": {
                    "symbol": contract.symbol,
                    "conId": contract.conId,
                    "exchange": contract.exchange,
                    "currency": contract.currency,
                },
            }

        # PreSubmitted / Submitted = accepted, may wait for fill (normal for MKT).
        if status in {"PreSubmitted", "Submitted", "Filled", "PartiallyFilled"}:
            return {
                "ok": True,
                "submitted": True,
                "thesis_note": thesis_note,
                "order": snap,
                "contract": {
                    "symbol": contract.symbol,
                    "conId": contract.conId,
                    "exchange": contract.exchange,
                    "currency": contract.currency,
                },
            }

        if not status or status == "PendingSubmit":
            return {
                "ok": False,
                "submitted": False,
                "symbol": symbol.upper(),
                "error": (
                    "Order stuck PendingSubmit — IB never acked (often a clientId "
                    "clash or cross-thread IB use). Retry once; if it persists, "
                    "restart the app so the IB worker reconnects with a fresh clientId."
                ),
                "order": snap,
                "contract": {
                    "symbol": contract.symbol,
                    "conId": contract.conId,
                    "exchange": contract.exchange,
                    "currency": contract.currency,
                },
            }

        return {
            "ok": True,
            "submitted": True,
            "thesis_note": thesis_note,
            "order": snap,
            "contract": {
                "symbol": contract.symbol,
                "conId": contract.conId,
                "exchange": contract.exchange,
                "currency": contract.currency,
            },
        }
