"""Map live IB account/portfolio data into the Flask terminal UI shape."""

from __future__ import annotations

from typing import Any

from config import get_ib_settings

from . import get_account_summary, get_portfolio


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0:
        return f"-${abs(value):,.0f}"
    return f"${value:,.0f}"


def _fmt_money2(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    abs_s = f"${abs(value):,.2f}"
    if not signed:
        return f"-{abs_s}" if value < 0 else abs_s
    if value > 0:
        return f"+{abs_s}"
    if value < 0:
        return f"-{abs_s}"
    return abs_s


def _fmt_pct(value: float | None, *, signed: bool = True) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def _summary_map(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Prefer USD tags; fall back to first numeric value per tag."""
    out: dict[str, float] = {}
    for row in summary_rows:
        tag = row.get("tag")
        if not tag:
            continue
        try:
            val = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        currency = (row.get("currency") or "").upper()
        if tag not in out or currency == "USD":
            out[tag] = val
    return out


def _build_holdings(
    portfolio_items: list[dict[str, Any]], nav: float | None
) -> list[dict[str, Any]]:
    """Position book rows only — theses are joined later via brain.enrich_holdings."""
    holdings: list[dict[str, Any]] = []
    for item in portfolio_items:
        if (item.get("secType") or "STK") not in {"STK", "ETF", "WAR"}:
            continue
        qty = float(item.get("position") or 0)
        if qty == 0:
            continue

        last = float(item.get("marketPrice") or 0)
        avg_cost = float(item.get("averageCost") or 0)
        mkt_value = float(item.get("marketValue") or (qty * last))
        unrealized = float(item.get("unrealizedPNL") or 0)

        cost_basis = abs(avg_cost * qty)
        if cost_basis > 0:
            pnl_pct = (unrealized / cost_basis) * 100.0
        elif avg_cost > 0 and last > 0:
            pnl_pct = ((last - avg_cost) / avg_cost) * 100.0
        else:
            pnl_pct = 0.0

        weight = (mkt_value / nav * 100.0) if nav and nav > 0 else 0.0
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue

        holdings.append(
            {
                "symbol": symbol,
                "name": symbol,
                "qty": qty,
                "last": round(last, 4),
                "avg_cost": round(avg_cost, 4),
                "mkt_value": round(mkt_value, 2),
                "pnl_pct": round(pnl_pct, 2),
                "weight": round(weight, 2),
                "sector": "",
                "unrealized_pnl": round(unrealized, 2),
            }
        )

    holdings.sort(key=lambda h: abs(h["mkt_value"]), reverse=True)
    return holdings


def _build_metrics(tags: dict[str, float]) -> list[dict[str, Any]]:
    nav = tags.get("NetLiquidation")
    cash = tags.get("TotalCashValue")
    buying_power = tags.get("BuyingPower")
    unrealized = tags.get("UnrealizedPnL")
    realized = tags.get("RealizedPnL")
    gross = tags.get("GrossPositionValue")
    excess = tags.get("ExcessLiquidity")

    day_pnl = tags.get("DailyPnL")
    day_tone = "neutral"
    if day_pnl is not None:
        day_tone = "up" if day_pnl >= 0 else "down"

    unreal_tone = "neutral"
    if unrealized is not None:
        unreal_tone = "up" if unrealized >= 0 else "down"

    return [
        {
            "id": "nav",
            "label": "PORTFOLIO NAV",
            "value": _fmt_money(nav),
            "delta": "NetLiquidation · TWS",
            "tone": "up" if (nav or 0) >= 0 else "down",
        },
        {
            "id": "cash",
            "label": "CASH",
            "value": _fmt_money(cash),
            "delta": "TotalCashValue",
            "tone": "neutral",
        },
        {
            "id": "buying_power",
            "label": "BUYING POWER",
            "value": _fmt_money(buying_power),
            "delta": "BuyingPower",
            "tone": "neutral",
        },
        {
            "id": "unrealized",
            "label": "UNREALIZED P&L",
            "value": _fmt_money2(unrealized, signed=True),
            "delta": _fmt_pct(
                (unrealized / nav * 100.0) if nav and unrealized is not None else None
            ),
            "tone": unreal_tone,
        },
        {
            "id": "day_pnl",
            "label": "DAY P&L",
            "value": (
                _fmt_money2(day_pnl, signed=True)
                if day_pnl is not None
                else (
                    _fmt_money2(realized, signed=True)
                    if realized is not None
                    else "—"
                )
            ),
            "delta": "DailyPnL" if day_pnl is not None else "RealizedPnL",
            "tone": day_tone
            if day_pnl is not None
            else ("up" if (realized or 0) >= 0 else "down"),
        },
        {
            "id": "gross",
            "label": "GROSS EXPOSURE",
            "value": _fmt_money(gross),
            "delta": (
                f"Excess liq {_fmt_money(excess)}"
                if excess is not None
                else "GrossPositionValue"
            ),
            "tone": "neutral",
        },
    ]


def connection_info(*, connected: bool, error: str | None = None) -> dict[str, Any]:
    from config import order_mode_status

    host, port, client_id, readonly = get_ib_settings()
    mode = (
        "PAPER"
        if port in {7497, 4002}
        else "LIVE"
        if port in {7496, 4001}
        else "CUSTOM"
    )
    order_mode = order_mode_status()
    return {
        "connected": connected,
        "host": host,
        "port": port,
        "client_id": client_id,
        "readonly": readonly,
        "orders_allowed": bool(order_mode.get("orders_allowed")),
        "order_mode": order_mode,
        "mode": mode,
        "label": f"TWS {port}",
        "status_label": f"{'LIVE' if connected else 'OFFLINE'} {mode}",
        "error": error,
    }


def fetch_live_portfolio_for_ui() -> dict[str, Any]:
    """
    Pull live account + portfolio from TWS and shape for the terminal UI.

    Holdings are position-book rows only (no thesis). Returns ok=False when
    TWS is unreachable.

    IMPORTANT: never call get_ib() from this Flask-facing helper. All IB I/O
    must go through @_on_ib_thread wrappers (get_account_summary / get_portfolio)
    so the dedicated IB worker owns the socket.
    """
    try:
        summary = get_account_summary()
        if not summary.get("ok"):
            raise RuntimeError(
                summary.get("error") or "get_account_summary failed"
            )
        portfolio = get_portfolio()
        if not portfolio.get("ok"):
            raise RuntimeError(portfolio.get("error") or "get_portfolio failed")

        tags = _summary_map(summary.get("summary") or [])
        nav = tags.get("NetLiquidation")
        cash = tags.get("TotalCashValue") or tags.get("AvailableFunds") or tags.get("CashBalance")
        holdings = _build_holdings(portfolio.get("portfolio") or [], nav)
        metrics = _build_metrics(tags)

        return {
            "ok": True,
            "source": "tws",
            "holdings": holdings,
            "metrics": metrics,
            "nav": nav,
            "cash": cash,
            "accounts": summary.get("accounts") or [],
            "connection": connection_info(connected=True),
        }
    except Exception as exc:
        return {
            "ok": False,
            "source": "error",
            "holdings": [],
            "metrics": [],
            "nav": None,
            "cash": None,
            "accounts": [],
            "connection": connection_info(
                connected=False, error=f"{type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
