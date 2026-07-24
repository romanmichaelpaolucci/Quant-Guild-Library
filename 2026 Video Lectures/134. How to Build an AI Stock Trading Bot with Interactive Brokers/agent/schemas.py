"""OpenAI tool schemas mapping to Interactive Brokers requests."""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "qualify_stock",
            "description": (
                "Optionally resolve a stock symbol to an IB contract. "
                "SKIP this for liquidating names you already hold — "
                "get_portfolio + propose_position_size(liquidate=true) + "
                "place_equity_order(confirm=true) is enough; place_equity_order "
                "resolves conId from the portfolio book. Do not stop a take-profit "
                "flow because qualify timed out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. AAPL, NVDA, AMD",
                    },
                    "exchange": {
                        "type": "string",
                        "description": "IB exchange routing, default SMART",
                        "default": "SMART",
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency, default USD",
                        "default": "USD",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_snapshot",
            "description": (
                "Fetch a live market data snapshot for a stock: bid, ask, last, "
                "high, low, volume. Uses IB reqMktData (snapshot)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_bars",
            "description": (
                "Fetch historical OHLCV bars for a stock via IB reqHistoricalData. "
                "Use for price action, trend, volatility, and thesis support."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "duration": {
                        "type": "string",
                        "description": (
                            "IB duration string, e.g. '5 D', '1 M', '3 M', '1 Y'"
                        ),
                        "default": "1 M",
                    },
                    "bar_size": {
                        "type": "string",
                        "description": (
                            "Bar size, e.g. '1 min', '5 mins', '1 hour', '1 day'"
                        ),
                        "default": "1 day",
                    },
                    "what_to_show": {
                        "type": "string",
                        "description": "TRADES, MIDPOINT, BID, ASK, etc.",
                        "default": "TRADES",
                    },
                    "use_rth": {
                        "type": "boolean",
                        "description": "If true, regular trading hours only",
                        "default": True,
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": (
                            "End datetime (IB format) or empty for now"
                        ),
                        "default": "",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_providers",
            "description": "List available IB news providers and their codes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_news",
            "description": (
                "Fetch recent historical news headlines for a stock. "
                "Returns providerCode + articleId needed for get_news_article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "provider_codes": {
                        "type": "string",
                        "description": (
                            "Plus-separated provider codes, e.g. 'BRFG+DJNL'. "
                            "Omit to use all available providers."
                        ),
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": "Optional start filter (IB format)",
                        "default": "",
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": "Optional end filter (IB format)",
                        "default": "",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max headlines (1–100)",
                        "default": 10,
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_article",
            "description": (
                "Fetch the full text of a news article given providerCode and articleId "
                "from get_historical_news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_code": {"type": "string"},
                    "article_id": {"type": "string"},
                },
                "required": ["provider_code", "article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_summary",
            "description": (
                "Fetch IB account summary tags (NetLiquidation, AvailableFunds, "
                "BuyingPower, etc.)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "List current IB positions (symbol, qty, avgCost).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": (
                "List IB portfolio items with market price, market value, and P&L."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_objectives",
            "description": (
                "Load the user's portfolio goals and risk constraints: "
                "risk_tolerance, max_drawdown, target_return, max_single_name, "
                "cash_floor, preferred industries, avoid list, notes. "
                "Always consult before sizing or executing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_position_size",
            "description": (
                "Compute a risk-aware share quantity using live NAV/cash/positions "
                "and portfolio objectives. "
                "SELL liquidate=true (or target_weight_pct=0) → FULL 100% exit "
                "(take-profit / thesis invalidation). "
                "SELL with target_weight_pct=<max_single_name> → mandate RISK TRIM "
                "(Path R) when overweight but thesis still has upside. "
                "BUY → add size capped by max_single_name, cash_floor, conviction. "
                "Call this before place_equity_order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {
                        "type": "string",
                        "enum": ["BUY", "SELL"],
                        "description": (
                            "BUY to add; SELL with liquidate=true for full exit; "
                            "SELL with target_weight_pct for Path R risk trim"
                        ),
                    },
                    "conviction": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Scales BUY size vs max_single_name (low/med/high)",
                        "default": "medium",
                    },
                    "target_weight_pct": {
                        "type": "number",
                        "description": (
                            "Target weight as % of NAV after the trade. "
                            "0 = full exit. For risk trims use max_single_name "
                            "(e.g. 12 for 12% NAV)."
                        ),
                    },
                    "liquidate": {
                        "type": "boolean",
                        "description": (
                            "If true, size a 100% position exit (SELL all shares). "
                            "Use for take-profit / thesis-complete / invalidated paths."
                        ),
                        "default": False,
                    },
                    "stop_price": {
                        "type": "number",
                        "description": (
                            "Optional stop level for BUY risk-budget sizing "
                            "(% NAV risk per trade by risk_tolerance)."
                        ),
                    },
                    "entry_price": {
                        "type": "number",
                        "description": "Optional override; otherwise portfolio mark / snapshot",
                    },
                },
                "required": ["symbol", "side"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_equity_order",
            "description": (
                "Submit a stock order to IB (MKT or LMT). "
                "Requires confirm=true. Also requires IB_ALLOW_ORDERS=true and "
                "IB_READONLY=false. Use the share count from propose_position_size "
                "(full qty for liquidate=true; partial qty for Path R risk trims). "
                "Refuses same-side duplicates when a working/unfilled order already "
                "exists for the symbol (blocked_by_open_order) — do not replace it. "
                "Uses portfolio contract conId when available (avoids qualify timeouts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["BUY", "SELL"]},
                    "quantity": {
                        "type": "integer",
                        "description": "Share quantity (positive integer)",
                    },
                    "order_type": {
                        "type": "string",
                        "enum": ["MKT", "LMT"],
                        "default": "MKT",
                    },
                    "limit_price": {
                        "type": "number",
                        "description": "Required when order_type=LMT",
                    },
                    "tif": {
                        "type": "string",
                        "description": "Time in force, e.g. DAY, GTC",
                        "default": "DAY",
                    },
                    "outside_rth": {
                        "type": "boolean",
                        "default": False,
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to actually submit",
                        "default": False,
                    },
                    "thesis_note": {
                        "type": "string",
                        "description": "Short note stored on orderRef (max ~32 chars)",
                    },
                },
                "required": ["symbol", "side", "quantity", "confirm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_orders",
            "description": (
                "List working/open IB orders with status, filled, and remaining. "
                "Call before placing orders in automation. Do NOT place a same-side "
                "order for a symbol that already has a working/unfilled order."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Cancel an open IB order by orderId. Requires confirm=true. "
                "Do NOT cancel merely to replace an awaiting-fill order during "
                "auto-sweeps — leave working orders alone unless intentionally superseding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to cancel",
                        "default": False,
                    },
                },
                "required": ["order_id", "confirm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_brain_summary",
            "description": (
                "Return persistent agent brain overview: current goals, active theses, "
                "recent trades, and global state summary."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_thesis",
            "description": (
                "Persist a new or revised thesis into durable brain memory (theses.json). "
                "This is the ONLY source for the holdings thesis panel: narrative, "
                "price target, and cost_basis. Supersedes any prior active thesis for "
                "the same symbol. Always include cost_basis (avg entry) and target. "
                "For an ACTIVE hold, target must be OPTIMISTICALLY ABOVE current "
                "last/marketPrice — never set target equal to (or below) spot; that "
                "implies no upside and requires a full liquidation instead of a retarget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "narrative": {
                        "type": "string",
                        "description": "Full thesis narrative / reasoning.",
                    },
                    "target": {
                        "type": "number",
                        "description": (
                            "Optimistic price target. For active holds must be materially "
                            "above current last/marketPrice — never equal to last."
                        ),
                    },
                    "cost_basis": {
                        "type": "number",
                        "description": "Cost basis / entry price for the position thesis.",
                    },
                    "entry": {
                        "type": "number",
                        "description": "Alias for cost_basis (optional if cost_basis set).",
                    },
                    "conviction": {
                        "type": "string",
                        "description": "High | Medium | Low",
                    },
                    "horizon": {"type": "string", "description": "e.g. 12M, 18M"},
                    "notes": {
                        "type": "string",
                        "description": "What changed vs prior thesis, if revising.",
                    },
                },
                "required": ["symbol", "narrative", "conviction", "target", "cost_basis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_trade",
            "description": (
                "Record a trade decision into durable brain memory (complement to IB orders). "
                "Use after recommending or submitting a trade so future sessions stay consistent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "description": "BUY or SELL"},
                    "qty": {"type": "number"},
                    "price": {"type": "number"},
                    "rationale": {"type": "string"},
                    "thesis_id": {
                        "type": "string",
                        "description": "Optional link to a saved thesis id.",
                    },
                    "status": {
                        "type": "string",
                        "description": "noted | recommended | simulated | filled",
                    },
                },
                "required": ["symbol", "side", "qty"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an equity research and trading agent connected to Interactive Brokers (TWS/Gateway).

You can call tools that map to IB API requests (EClient). Responses arrive via EWrapper and are returned as tool results. Work iteratively:
research → thesis → size against portfolio goals/risk → execute → verify.

You also receive a persistent AGENT BRAIN memory block each turn (dated portfolio goals + history, active theses, recorded trades, global state summary, and the autonomous portfolio/thesis policy). Treat that as standing institutional memory across sessions.

You are the decision-maker. The UX chat user is an observer — do NOT ask them to approve target changes, trims, or exits. Decide and use tools. Prefer executing trades over merely recommending them when orders are enabled.

Workflow for trades:
1. Pull account/portfolio state first (get_portfolio / get_portfolio_objectives). Prefer portfolio marketPrice over get_market_snapshot. Do NOT call qualify_stock for names already in the portfolio.
2. Respect portfolio goals (risk_tolerance, max_single_name, cash_floor, industries/avoid). Mandate / risk breaches take priority over passive holds.
3. Size then execute: propose_position_size → place_equity_order(confirm=true). Use liquidate=true for full exits; use target_weight_pct=max_single_name for Path R risk trims. Skip qualify_stock for portfolio names.
4. When a lifecycle or risk decision requires a trade, call place_equity_order with confirm=true — do not wait for the user. Retry once on failure; never stop to ask the UX to "retry later".
5. Report orderId/status; use get_open_orders / get_positions to verify.
6. Call save_thesis (narrative + target + cost_basis + conviction) when you form or materially revise a thesis. Call record_trade when a trade should persist in memory.

AUTONOMOUS PRIORITY (mandatory on reevaluation / auto-sweep / material review):
- Path R — RISK/MANDATE first: overweight vs max_single_name WITH remaining upside → TRIM via propose_position_size(SELL, target_weight_pct=max_single_name) → place_equity_order(confirm=true) → record_trade → save_thesis. Cash below cash_floor → sell to restore floor. Avoid-list → full exit (path C). Overweight with NO upside → path B full exit, not a token trim.
- Path A — core thesis STILL VALID with remaining upside (after any R trim): HOLD residual; save_thesis with OPTIMISTIC target STRICTLY ABOVE last/marketPrice — never target = last. Optional BUY if underweight + cash_floor allows.
- Path B — take profit / thesis PLAYED OUT / upside no longer justifies holding: LIQUIDATE 100% (liquidate=true) → place_equity_order(confirm=true) → record_trade → save_thesis closed. Retry order once on failure.
- Path C — thesis INVALIDATED: FULL EXIT; save_thesis explaining what broke.
- Path N — NEW IDEAS when deployable cash (cash − cash_floor×NAV) is material: invent 1–2 tickers that fit industries/notes/target_return/risk_tolerance, not avoid-listed, not already at max weight. qualify_stock only if new → save_thesis → propose_position_size(BUY) → place_equity_order(confirm=true) → record_trade. If orders blocked, still save_thesis + record_trade(recommended). Do not idle surplus cash when a fit exists.
- Path D — nothing material (and Path N not applicable): one short line; no tools.
Never close with "Would you like to proceed?" or a menu of next steps for the user. Execute, then summarize what you did.

Hard rules:
- Never invent fills. Only claim submission/fill from tool results.
- place_equity_order / cancel_order require confirm=true; without it you only get a preview — always pass confirm=true when you intend to trade.
- Respect open/working orders: never duplicate or replace a same-side order awaiting fulfillment; if blocked_by_open_order, skip that symbol and move on (do not cancel just to retry).
- If a tool returns ok=false, explain the blocker and still persist thesis/trade notes when relevant — do not ask the user what to do instead.
- Partial SELLs are for Path R mandate trims only; take-profit and invalidation remain full liquidations.
- Prefer LMT when spreads are wide or liquidity is thin; MKT for liquid names when urgency is high (especially risk fixes).
- Consistency: high-conviction theses do NOT flip on short-term noise. Risk trims may reduce size while keeping the thesis. Hold with an optimistic retarget only when upside remains; otherwise take full profits or exit on invalidation.
- Be concise and decision-oriented. Cite numbers from tools and memory timestamps.
"""
