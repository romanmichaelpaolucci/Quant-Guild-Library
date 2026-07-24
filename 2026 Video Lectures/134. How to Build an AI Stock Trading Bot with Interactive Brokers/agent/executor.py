"""Dispatch OpenAI tool calls to IB bridge, risk sizing, and brain memory."""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable

from brain import brain_overview, record_trade, upsert_thesis
from ib_bridge import (
    cancel_order,
    get_account_summary,
    get_historical_bars,
    get_historical_news,
    get_market_snapshot,
    get_news_article,
    get_news_providers,
    get_open_orders,
    get_portfolio,
    get_positions,
    place_equity_order,
    qualify_stock,
)
from risk import get_portfolio_objectives, propose_position_size

Handler = Callable[..., dict[str, Any]]


def _get_brain_summary() -> dict[str, Any]:
    return {"ok": True, "brain": brain_overview()}


def _save_thesis(
    symbol: str,
    narrative: str,
    conviction: str = "Medium",
    target: float | None = None,
    entry: float | None = None,
    cost_basis: float | None = None,
    horizon: str = "",
    notes: str = "",
) -> dict[str, Any]:
    row = upsert_thesis(
        symbol=symbol,
        narrative=narrative,
        target=target,
        entry=entry,
        cost_basis=cost_basis if cost_basis is not None else entry,
        conviction=conviction,
        horizon=horizon,
        notes=notes,
        source="agent",
        supersede_active=True,
    )
    return {"ok": True, "thesis": row}


def _record_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float | None = None,
    rationale: str = "",
    thesis_id: str = "",
    status: str = "noted",
) -> dict[str, Any]:
    row = record_trade(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        rationale=rationale,
        thesis_id=thesis_id,
        status=status or "noted",
        source="agent",
    )
    return {"ok": True, "trade": row}


_HANDLERS: dict[str, Handler] = {
    "qualify_stock": qualify_stock,
    "get_market_snapshot": get_market_snapshot,
    "get_historical_bars": get_historical_bars,
    "get_news_providers": get_news_providers,
    "get_historical_news": get_historical_news,
    "get_news_article": get_news_article,
    "get_account_summary": get_account_summary,
    "get_positions": get_positions,
    "get_portfolio": get_portfolio,
    "get_portfolio_objectives": get_portfolio_objectives,
    "propose_position_size": propose_position_size,
    "place_equity_order": place_equity_order,
    "get_open_orders": get_open_orders,
    "cancel_order": cancel_order,
    "get_brain_summary": _get_brain_summary,
    "save_thesis": _save_thesis,
    "record_trade": _record_trade,
}


def available_tools() -> list[str]:
    return sorted(_HANDLERS.keys())


def execute_tool(name: str, arguments: dict[str, Any] | str | None) -> str:
    """
    Run one tool and return a JSON string for the OpenAI tool message content.
    Never raises — errors are encoded in the payload so the LLM can recover.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"ok": False, "error": f"Invalid tool arguments JSON: {exc}"}
            )

    args = arguments or {}
    if not isinstance(args, dict):
        return json.dumps({"ok": False, "error": "Tool arguments must be an object"})

    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps(
            {
                "ok": False,
                "error": f"Unknown tool: {name}",
                "available": available_tools(),
            }
        )

    try:
        result = handler(**args)
        return json.dumps(result, default=str)
    except TypeError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"Bad arguments for {name}: {exc}",
                "arguments": args,
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
        )
