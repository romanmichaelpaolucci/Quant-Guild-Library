"""Position sizing relative to portfolio objectives and live IB account state."""

from __future__ import annotations

import re
from typing import Any

from ib_bridge import get_account_summary, get_market_snapshot, get_portfolio
from risk.objectives import load_objectives

CONVICTION_SCALE = {
    "low": 0.40,
    "medium": 0.70,
    "med": 0.70,
    "high": 1.00,
}

# Fraction of NAV risked per trade when a stop is provided, by risk tolerance label.
RISK_PER_TRADE = {
    "conservative": 0.005,
    "moderate": 0.010,
    "moderate-aggressive": 0.0125,
    "aggressive": 0.020,
}


def _parse_pct(value: Any, default: float) -> float:
    """Parse '12%', '12% NAV', 12, 0.12 → fraction of NAV."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 100.0 if v > 1 else v
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%?", s)
    if not m:
        return default
    v = float(m.group(1))
    return v / 100.0 if v > 1 or "%" in s else v


def _summary_map(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in summary_rows:
        tag = row.get("tag")
        try:
            out[str(tag)] = float(row.get("value"))
        except (TypeError, ValueError):
            continue
    return out


def _price_from_snapshot(snap: dict[str, Any]) -> float | None:
    for key in ("last", "close", "bid", "ask"):
        v = snap.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def propose_position_size(
    symbol: str,
    side: str = "BUY",
    conviction: str = "medium",
    target_weight_pct: float | None = None,
    stop_price: float | None = None,
    entry_price: float | None = None,
    liquidate: bool = False,
) -> dict[str, Any]:
    """
    Propose share quantity for a trade given live NAV/cash/positions and objectives.

    Caps:
      - max_single_name (% of NAV) after the trade
      - cash_floor (% of NAV) remaining after a BUY
      - optional stop-based risk budget (% NAV per trade by risk_tolerance)

    Set liquidate=True (SELL) to exit 100% of the position (take-profit / invalidation).
    For mandate risk trims, pass target_weight_pct equal to max_single_name (e.g. 12).
    """
    symbol = symbol.upper().strip()
    side_u = side.upper().strip()
    if side_u not in {"BUY", "SELL"}:
        return {"ok": False, "error": "side must be BUY or SELL"}

    if liquidate:
        side_u = "SELL"
        target_weight_pct = 0.0

    objectives = load_objectives()
    max_single = _parse_pct(objectives.get("max_single_name"), 0.12)
    cash_floor = _parse_pct(objectives.get("cash_floor"), 0.05)
    risk_label = str(objectives.get("risk_tolerance", "moderate")).strip().lower()
    risk_frac = RISK_PER_TRADE.get(risk_label, 0.01)

    acct = get_account_summary()
    if not acct.get("ok"):
        return acct
    tags = _summary_map(acct.get("summary") or [])
    nav = tags.get("NetLiquidation") or tags.get("EquityWithLoanValue")
    cash = tags.get("TotalCashValue") or tags.get("AvailableFunds") or tags.get("CashBalance")
    if not nav or nav <= 0:
        return {
            "ok": False,
            "error": "Could not read NetLiquidation from account summary",
            "tags_seen": sorted(tags.keys()),
        }
    if cash is None:
        cash = 0.0

    port = get_portfolio()
    if not port.get("ok"):
        return port

    current_mv = 0.0
    current_qty = 0.0
    for item in port.get("portfolio") or []:
        if str(item.get("symbol", "")).upper() == symbol:
            current_mv += float(item.get("marketValue") or 0)
            current_qty += float(item.get("position") or 0)

    if entry_price is None or entry_price <= 0:
        # Prefer portfolio mark first (avoids hung/empty snapshots during reevaluation).
        for item in port.get("portfolio") or []:
            if str(item.get("symbol", "")).upper() == symbol:
                px = float(item.get("marketPrice") or 0)
                if px > 0:
                    entry_price = px
                    break
        if entry_price is None or entry_price <= 0:
            snap = get_market_snapshot(symbol)
            if snap.get("ok"):
                entry_price = _price_from_snapshot(snap)
            if entry_price is None or entry_price <= 0:
                return {
                    "ok": False,
                    "error": f"No usable price for {symbol}",
                    "snapshot": snap if "snap" in locals() else None,
                }

    scale = CONVICTION_SCALE.get(conviction.strip().lower(), 0.70)
    if target_weight_pct is not None:
        desired_weight = _parse_pct(target_weight_pct, max_single * scale)
    elif side_u == "SELL":
        # Default SELL without liquidate → trim toward mandate max_single_name (Path R)
        desired_weight = max_single
    else:
        desired_weight = max_single * scale

    if not liquidate:
        desired_weight = min(desired_weight, max_single)
    desired_mv = desired_weight * nav

    constraints: list[str] = []
    blockers: list[str] = []

    if side_u == "BUY":
        # Additional notional needed to reach desired weight (or full desired if flat)
        delta_mv = max(0.0, desired_mv - max(current_mv, 0.0))
        # Headroom under max single-name
        max_mv = max_single * nav
        headroom_mv = max(0.0, max_mv - max(current_mv, 0.0))
        if delta_mv > headroom_mv:
            delta_mv = headroom_mv
            constraints.append(
                f"Capped to max_single_name={max_single:.1%} NAV "
                f"(headroom ${headroom_mv:,.2f})"
            )

        # Cash floor: spend at most cash - cash_floor*NAV
        spendable = max(0.0, float(cash) - cash_floor * nav)
        if delta_mv > spendable:
            delta_mv = spendable
            constraints.append(
                f"Capped by cash_floor={cash_floor:.1%} "
                f"(spendable ${spendable:,.2f})"
            )

        # Optional stop-based risk sizing
        if stop_price is not None and stop_price > 0:
            risk_per_share = abs(entry_price - float(stop_price))
            if risk_per_share <= 0:
                blockers.append("stop_price equals entry; risk sizing undefined")
            else:
                risk_budget = risk_frac * nav
                risk_shares = int(risk_budget // risk_per_share)
                risk_notional = risk_shares * entry_price
                if risk_notional < delta_mv:
                    delta_mv = risk_notional
                    constraints.append(
                        f"Capped by stop risk budget {risk_frac:.2%} NAV "
                        f"(${risk_budget:,.2f}, stop={stop_price})"
                    )

        shares = int(delta_mv // entry_price)
        if shares <= 0:
            blockers.append(
                "Proposed BUY size is 0 shares after constraints "
                "(at weight cap, cash floor, or price too high)"
            )
        signed_shares = shares
        notional = shares * entry_price
        weight_after = (current_mv + notional) / nav
        cash_after = float(cash) - notional

    else:  # SELL
        long_qty = max(current_qty, 0.0)
        if long_qty <= 0:
            return {
                "ok": False,
                "error": f"No long position in {symbol} to sell",
                "current_qty": current_qty,
            }

        if liquidate or desired_weight <= 0:
            shares = int(long_qty)
            constraints.append("Full liquidation (100% of position)")
        else:
            target_mv = desired_weight * nav
            reduce_mv = max(0.0, current_mv - target_mv)
            shares = min(int(long_qty), int(reduce_mv // entry_price) if entry_price else 0)
            if shares <= 0:
                return {
                    "ok": False,
                    "error": (
                        f"{symbol} already at/under desired weight "
                        f"{desired_weight:.1%}; no trim needed. "
                        "Pass liquidate=true for a full exit if taking profit / invalidating."
                    ),
                    "symbol": symbol,
                    "side": "SELL",
                    "shares": 0,
                    "current_qty": current_qty,
                    "current_weight_pct": round((current_mv / nav) * 100, 3) if nav else None,
                    "desired_weight_pct": round(desired_weight * 100, 3),
                    "hint": "Use liquidate=true for 100% exit, or hold",
                }
            constraints.append(
                f"Risk trim toward {desired_weight:.1%} NAV "
                f"(max_single_name={max_single:.1%})"
            )

        signed_shares = shares
        notional = shares * entry_price
        weight_after = max(0.0, (current_mv - notional) / nav)
        cash_after = float(cash) + notional

    return {
        "ok": len(blockers) == 0 and signed_shares > 0,
        "symbol": symbol,
        "side": side_u,
        "shares": signed_shares,
        "entry_price": round(entry_price, 4),
        "notional": round(notional, 2),
        "conviction": conviction,
        "liquidate": bool(liquidate) or desired_weight <= 0,
        "desired_weight_pct": round(desired_weight * 100, 3),
        "current_weight_pct": round((current_mv / nav) * 100, 3) if nav else None,
        "weight_after_pct": round(weight_after * 100, 3),
        "nav": round(nav, 2),
        "cash": round(float(cash), 2),
        "cash_after": round(cash_after, 2),
        "cash_floor_pct": round(cash_floor * 100, 3),
        "max_single_name_pct": round(max_single * 100, 3),
        "current_qty": current_qty,
        "current_market_value": round(current_mv, 2),
        "stop_price": stop_price,
        "risk_tolerance": objectives.get("risk_tolerance"),
        "constraints_applied": constraints,
        "blockers": blockers,
        "objectives_snapshot": {
            "max_single_name": objectives.get("max_single_name"),
            "cash_floor": objectives.get("cash_floor"),
            "risk_tolerance": objectives.get("risk_tolerance"),
            "max_drawdown": objectives.get("max_drawdown"),
        },
    }
