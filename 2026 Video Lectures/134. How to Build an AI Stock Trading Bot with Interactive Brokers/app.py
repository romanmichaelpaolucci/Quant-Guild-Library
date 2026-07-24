"""AI Stock Trading Bot — Bloomberg-style terminal UI + LLM/IB agent."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from agent import TradingAgent
from brain import (
    append_turn,
    brain_overview,
    chat_for_agent,
    chat_for_ui,
    current_goals,
    enrich_holdings,
    ensure_brain_initialized,
    ensure_ux_state,
    format_memory_for_llm,
    get_auto_status,
    load_goals,
    load_holdings_book,
    load_ux_state,
    mark_auto_run,
    notify_ux_changed,
    prepare_chat_for_session,
    run_auto_review_now,
    save_goals,
    save_holdings_book,
    save_ux_state,
    start_auto_scheduler,
)
from brain.policy import auto_sweep_prompt, build_auto_sweep_context
from ib_bridge.ui_portfolio import connection_info, fetch_live_portfolio_for_ui
from risk.objectives import load_objectives

app = Flask(__name__)

# Offline fallback metrics only — replaced by TWS account tags when connected.
DEMO_PORTFOLIO_METRICS = [
    {"id": "nav", "label": "PORTFOLIO NAV", "value": "—", "delta": "TWS offline", "tone": "neutral"},
    {"id": "cash", "label": "CASH", "value": "—", "delta": "TWS offline", "tone": "neutral"},
    {"id": "buying_power", "label": "BUYING POWER", "value": "—", "delta": "TWS offline", "tone": "neutral"},
    {"id": "unrealized", "label": "UNREALIZED P&L", "value": "—", "delta": "TWS offline", "tone": "neutral"},
    {"id": "day_pnl", "label": "DAY P&L", "value": "—", "delta": "TWS offline", "tone": "neutral"},
    {"id": "gross", "label": "GROSS EXPOSURE", "value": "—", "delta": "TWS offline", "tone": "neutral"},
]


def _parse_tag_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


CHAT_SEED_FALLBACK = (
    "Terminal online. Brain memory loaded. Ask me to generate or revise a thesis "
    "for any holding — narrative, target, and cost basis are stored in theses.json "
    "and drive the left-hand thesis panel."
)


def _chat_seed(*, tws_note: str | None = None) -> list[dict]:
    """Brain greeting + restored compressed chat history."""
    overview = brain_overview()
    counts = overview.get("counts") or {}
    goals = overview.get("goals") or {}
    greeting = (
        f"Terminal online — brain synced @ {overview.get('updated_at', 'n/a')}. "
        f"Goals saved {goals.get('saved_at', 'n/a')} "
        f"(history={counts.get('goal_history', 0)}). "
        f"Active theses={counts.get('active_theses', 0)}, "
        f"recorded trades={counts.get('trades', 0)}.\n\n"
        f"{overview.get('summary') or CHAT_SEED_FALLBACK}\n\n"
        "Thesis / target / cost basis for each holding come only from saved JSON "
        "(via save_thesis). High-conviction theses stay sticky unless the core thesis breaks."
    )
    if tws_note:
        greeting = f"{tws_note}\n\n{greeting}"
    restored = chat_for_ui()
    if restored:
        return [{"role": "assistant", "content": greeting}, *restored]
    return [{"role": "assistant", "content": greeting}]


def _merge_book_metadata(
    live_rows: list[dict], prior_book: list[dict]
) -> list[dict]:
    """Keep name/sector from the prior book when TWS only returns the symbol."""
    prior = {str(h.get("symbol") or "").upper(): h for h in prior_book}
    merged = []
    for row in live_rows:
        out = dict(row)
        prev = prior.get(out["symbol"])
        if prev:
            if prev.get("name") and (
                not out.get("name") or out.get("name") == out["symbol"]
            ):
                out["name"] = prev["name"]
            if prev.get("sector") and not out.get("sector"):
                out["sector"] = prev["sector"]
        merged.append(out)
    return merged


def load_portfolio_view(*, persist: bool = True) -> dict:
    """
    Prefer live TWS positions + account metrics.
    Fall back to holdings.json book when TWS is down.
    """
    prior = load_holdings_book()
    live = fetch_live_portfolio_for_ui()

    if live.get("ok"):
        book = _merge_book_metadata(live["holdings"], prior)
        if persist:
            try:
                save_holdings_book(book)
            except Exception:
                pass
        try:
            ensure_brain_initialized(book, live["metrics"])
        except Exception:
            pass
        return {
            "source": "tws",
            "book": book,
            "holdings": enrich_holdings(book),
            "metrics": live["metrics"],
            "nav": live.get("nav"),
            "cash": live.get("cash"),
            "connection": live["connection"],
            "error": None,
        }

    book = prior
    try:
        ensure_brain_initialized(book, DEMO_PORTFOLIO_METRICS)
    except Exception:
        pass
    return {
        "source": "demo",
        "book": book,
        "holdings": enrich_holdings(book),
        "metrics": DEMO_PORTFOLIO_METRICS,
        "nav": None,
        "cash": None,
        "connection": live.get("connection")
        or connection_info(connected=False, error=live.get("error")),
        "error": live.get("error"),
    }


# Initialize dated JSON brain on startup (holdings book / chat files).
ensure_brain_initialized(load_holdings_book(), DEMO_PORTFOLIO_METRICS)
ensure_ux_state()
start_auto_scheduler()
OBJECTIVES = load_objectives()


@app.route("/")
def index():
    global OBJECTIVES
    OBJECTIVES = load_objectives()
    prepare_chat_for_session()
    view = load_portfolio_view(persist=True)
    holdings = view["holdings"]
    conn = view["connection"]
    ux = load_ux_state()

    tws_note = None
    if view["source"] == "demo" and view.get("error"):
        tws_note = (
            f"TWS unreachable ({view['error']}). Showing last saved holdings book. "
            f"Start TWS/Gateway on {conn.get('host')}:{conn.get('port')} "
            "with API enabled, then refresh."
        )
    elif view["source"] == "tws" and not view["book"]:
        tws_note = (
            f"Connected to TWS {conn.get('port')} ({conn.get('mode')}), "
            "but no stock positions were returned."
        )

    # Empty table placeholder so the thesis panel still renders.
    if not holdings:
        holdings = [
            {
                "symbol": "—",
                "name": "",
                "qty": 0,
                "last": 0.0,
                "avg_cost": 0.0,
                "mkt_value": 0.0,
                "pnl_pct": 0.0,
                "weight": 0.0,
                "sector": "",
                "thesis": {
                    "narrative": "",
                    "target": None,
                    "cost_basis": None,
                    "entry": None,
                    "conviction": "",
                    "horizon": "",
                    "saved_at": None,
                    "missing": True,
                },
            }
        ]

    return render_template(
        "index.html",
        metrics=view["metrics"],
        objectives=current_goals(),
        holdings=holdings,
        chat_seed=_chat_seed(tws_note=tws_note),
        brain=brain_overview(),
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        data_source=view["source"],
        connection=conn,
        ux=ux,
    )


@app.get("/api/ux")
def api_get_ux():
    return jsonify({**load_ux_state(), **get_auto_status()})


@app.put("/api/ux")
@app.post("/api/ux")
def api_put_ux():
    payload = request.get_json(silent=True) or {}
    # sendBeacon may post as text; try raw JSON parse fallback
    if not payload and request.data:
        try:
            import json as _json

            payload = _json.loads(request.data.decode("utf-8"))
        except Exception:
            payload = {}
    state = save_ux_state(payload)
    notify_ux_changed()
    return jsonify({**state, **get_auto_status()})


@app.get("/api/auto/status")
def api_auto_status():
    return jsonify(get_auto_status())


@app.post("/api/auto/run")
def api_auto_run():
    """Non-blocking: queue a background auto-review if idle."""
    force = bool((request.get_json(silent=True) or {}).get("force"))
    return jsonify(run_auto_review_now(force=force or True))


@app.get("/api/status")
def api_status():
    view = load_portfolio_view(persist=False)
    return jsonify(
        {
            "ok": view["source"] == "tws",
            "source": view["source"],
            "connection": view["connection"],
            "holdings_count": len(view["book"]),
            "error": view.get("error"),
            "brain": brain_overview(),
        }
    )


@app.get("/api/objectives")
def api_get_objectives():
    doc = load_goals()
    current = doc["current"]
    return jsonify(
        {
            **{k: current[k] for k in current if k != "source"},
            "history": doc.get("history") or [],
            "history_count": len(doc.get("history") or []),
        }
    )


@app.put("/api/objectives")
def api_put_objectives():
    global OBJECTIVES
    payload = request.get_json(silent=True) or {}
    updated = dict(load_objectives())

    for key in (
        "risk_tolerance",
        "max_drawdown",
        "target_return",
        "horizon",
        "max_single_name",
        "cash_floor",
        "notes",
    ):
        if key in payload:
            value = payload[key]
            if not isinstance(value, str):
                return jsonify({"error": f"{key} must be a string"}), 400
            updated[key] = value.strip()

    for key in ("industries", "avoid"):
        if key in payload:
            updated[key] = _parse_tag_list(payload[key])

    doc = save_goals(updated, source="ux")
    OBJECTIVES = {k: doc["current"][k] for k in updated}
    current = doc["current"]
    return jsonify(
        {
            **{k: current[k] for k in current if k != "source"},
            "history": doc.get("history") or [],
            "history_count": len(doc.get("history") or []),
        }
    )


@app.get("/api/brain")
def api_brain():
    return jsonify(brain_overview())


@app.get("/api/chat/history")
def api_chat_history():
    return jsonify({"messages": chat_for_ui(), "brain": brain_overview()})


@app.get("/api/holdings")
def api_holdings():
    view = load_portfolio_view(persist=True)
    return jsonify(
        {
            "source": view["source"],
            "holdings": view["holdings"],
            "connection": view["connection"],
            "error": view.get("error"),
        }
    )


@app.get("/api/metrics")
def api_metrics():
    view = load_portfolio_view(persist=False)
    return jsonify(
        {
            "source": view["source"],
            "metrics": view["metrics"],
            "connection": view["connection"],
            "error": view.get("error"),
        }
    )


@app.get("/api/holdings/<symbol>")
def api_holding(symbol: str):
    view = load_portfolio_view(persist=False)
    row = next(
        (h for h in view["holdings"] if h["symbol"].upper() == symbol.upper()),
        None,
    )
    if not row:
        return jsonify({"error": "not found", "source": view["source"]}), 404
    return jsonify({**row, "source": view["source"]})


@app.post("/api/chat")
def api_chat():
    """
    Run the iterative LLM ↔ IB tool loop with persistent brain + compressed chat memory.

    Set Accept: text/event-stream (or ?stream=1) for SSE token streaming.
    Optional JSON field mode="auto_review" for the 5-minute automated sweep.
    """
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode") or "").strip().lower()
    message = (payload.get("message") or "").strip()
    symbol = (payload.get("symbol") or "").strip().upper()
    client_history = payload.get("history") or []
    want_stream = (
        request.args.get("stream") in {"1", "true", "yes"}
        or "text/event-stream" in (request.headers.get("Accept") or "")
    )

    if mode == "auto_review":
        ux = load_ux_state()
        message = auto_sweep_prompt(interval_label=ux.get("interval_label"))
    elif not message:
        return jsonify({"error": "empty message"}), 400

    view = load_portfolio_view(persist=True)
    holdings = view["holdings"]
    holding = next((h for h in holdings if h["symbol"] == symbol), None)
    extra_bits = []
    if mode == "auto_review":
        goals = (load_goals() or {}).get("current") or {}
        open_orders = []
        try:
            from ib_bridge import get_open_orders

            oo = get_open_orders()
            if oo.get("ok"):
                open_orders = oo.get("working_orders") or oo.get("open_orders") or []
        except Exception:
            pass
        extra_bits.append(
            build_auto_sweep_context(
                holdings,
                goals=goals,
                nav=view.get("nav"),
                cash=view.get("cash"),
                open_orders=open_orders,
            )
        )
    elif holding:
        thesis = holding.get("thesis") or {}
        if thesis.get("missing"):
            extra_bits.append(
                f"UI holding {holding['symbol']} ({holding.get('name', '')}) has NO saved thesis yet. "
                f"qty={holding['qty']}, last=${float(holding['last']):.2f}, "
                f"broker_avg_cost=${holding.get('avg_cost')}. "
                f"Data source={view['source']}. "
                "Call save_thesis with narrative, target, cost_basis, and conviction."
            )
        else:
            extra_bits.append(
                f"UI holding context for {holding['symbol']} ({holding.get('name', '')}): "
                f"qty={holding['qty']}, last=${float(holding['last']):.2f}, "
                f"broker_avg_cost=${holding.get('avg_cost')}, "
                f"weight={float(holding['weight']):.2f}%, "
                f"thesis_target={thesis.get('target')}, "
                f"cost_basis={thesis.get('cost_basis')}, "
                f"conviction={thesis.get('conviction')}, "
                f"thesis_saved_at={thesis.get('saved_at', 'n/a')}. "
                f"Data source={view['source']}."
            )
    extra_context = "\n".join(extra_bits) if extra_bits else None

    summary, stored_recent = chat_for_agent(limit=12)
    memory = format_memory_for_llm(symbol=symbol or None)
    if summary:
        memory = memory + "\n\n[Compressed prior chat memory]\n" + summary

    history = stored_recent or (client_history if isinstance(client_history, list) else [])
    if mode == "auto_review":
        history = []

    display_user = (
        f"[AUTO] portfolio sweep · every {load_ux_state().get('interval_label', '5M')}"
        if mode == "auto_review"
        else message
    )

    if want_stream:

        def event_stream():
            import json as _json

            agent = TradingAgent()
            final_content = ""
            tool_calls = []
            try:
                for event in agent.run_stream(
                    message,
                    symbol=symbol or None,
                    history=history,
                    extra_context=extra_context,
                    memory=memory,
                ):
                    et = event.get("type")
                    if et == "delta":
                        final_content += event.get("text") or ""
                    elif et == "done":
                        final_content = event.get("content") or final_content
                        tool_calls = event.get("tool_calls") or []
                    yield f"data: {_json.dumps(event, default=str)}\n\n"

                    if et == "error":
                        return

                chat_doc = append_turn(display_user, final_content)
                if mode == "auto_review":
                    mark_auto_run()
                refreshed = load_portfolio_view(persist=True)
                done_meta = {
                    "type": "meta",
                    "brain": brain_overview(),
                    "holdings": refreshed["holdings"],
                    "metrics": refreshed["metrics"],
                    "source": refreshed["source"],
                    "connection": refreshed["connection"],
                    "ux": load_ux_state(),
                    "chat": {
                        "updated_at": chat_doc.get("updated_at"),
                        "compression_count": chat_doc.get("compression_count"),
                        "message_count": len(chat_doc.get("messages") or []),
                        "has_summary": bool(chat_doc.get("summary")),
                    },
                    "tool_calls": tool_calls,
                }
                yield f"data: {_json.dumps(done_meta, default=str)}\n\n"
            except Exception as exc:
                err = {
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                yield f"data: {_json.dumps(err)}\n\n"

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        agent = TradingAgent()
        result = agent.run(
            message,
            symbol=symbol or None,
            history=history,
            extra_context=extra_context,
            memory=memory,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "role": "assistant",
                    "content": (
                        f"Agent failed: {type(exc).__name__}: {exc}. "
                        "Check OPENAI_API_KEY, TWS on IB_PORT, and API permissions."
                    ),
                }
            ),
            500,
        )

    chat_doc = append_turn(display_user, result.content)
    if mode == "auto_review":
        mark_auto_run()
    refreshed = load_portfolio_view(persist=True)

    return jsonify(
        {
            "role": "assistant",
            "content": result.content,
            "iterations": result.iterations,
            "model": result.model,
            "finish_reason": result.finish_reason,
            "brain": brain_overview(),
            "holdings": refreshed["holdings"],
            "metrics": refreshed["metrics"],
            "source": refreshed["source"],
            "connection": refreshed["connection"],
            "ux": load_ux_state(),
            "chat": {
                "updated_at": chat_doc.get("updated_at"),
                "compression_count": chat_doc.get("compression_count"),
                "message_count": len(chat_doc.get("messages") or []),
                "has_summary": bool(chat_doc.get("summary")),
            },
            "tool_calls": [
                {
                    "name": t.name,
                    "arguments": t.arguments,
                    "result_preview": t.result_preview,
                }
                for t in result.tool_traces
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)