"""Persistent agent brain — dated JSON memory for goals, theses, trades, state."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GOALS_PATH = DATA_DIR / "goals.json"
THESES_PATH = DATA_DIR / "theses.json"
TRADES_PATH = DATA_DIR / "trades.json"
STATE_PATH = DATA_DIR / "state.json"
HOLDINGS_PATH = DATA_DIR / "holdings.json"
LEGACY_OBJECTIVES_PATH = DATA_DIR / "objectives.json"

MAX_GOAL_HISTORY = 10
MAX_THESES = 50
MAX_TRADES = 100

GOAL_FIELDS = (
    "risk_tolerance",
    "max_drawdown",
    "target_return",
    "horizon",
    "max_single_name",
    "cash_floor",
    "industries",
    "avoid",
    "notes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def default_goals_fields() -> dict[str, Any]:
    return {
        "risk_tolerance": "Moderate-Aggressive",
        "max_drawdown": "15%",
        "target_return": "18% annualized",
        "horizon": "12–24 months",
        "max_single_name": "12% NAV",
        "cash_floor": "5%",
        "industries": ["Semiconductors", "Software", "Healthcare", "Industrials"],
        "avoid": ["Pure tobacco", "SPACs"],
        "notes": "Prefer liquid large/mid-cap; allow selective AI infrastructure overweights.",
    }


def _normalize_goal_fields(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = default_goals_fields()
    if not raw:
        return base
    for key in GOAL_FIELDS:
        if key not in raw:
            continue
        if key in {"industries", "avoid"}:
            value = raw[key]
            if isinstance(value, list):
                base[key] = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str):
                base[key] = [p.strip() for p in value.split(",") if p.strip()]
        else:
            base[key] = str(raw[key]).strip()
    return base


def _goal_snapshot(fields: dict[str, Any], saved_at: str | None = None) -> dict[str, Any]:
    snap = {k: deepcopy(fields[k]) for k in GOAL_FIELDS}
    snap["saved_at"] = saved_at or _utc_now()
    return snap


def _goals_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in GOAL_FIELDS)


# --- Goals -----------------------------------------------------------------


def load_goals() -> dict[str, Any]:
    """Return { current, history } with dated snapshots (history ≤ 10)."""
    default_current = _goal_snapshot(default_goals_fields(), saved_at=_utc_now())
    default_doc = {"current": default_current, "history": []}

    if GOALS_PATH.exists():
        doc = _read_json(GOALS_PATH, default_doc)
    elif LEGACY_OBJECTIVES_PATH.exists():
        legacy = _read_json(LEGACY_OBJECTIVES_PATH, {})
        fields = _normalize_goal_fields(legacy if isinstance(legacy, dict) else {})
        doc = {
            "current": _goal_snapshot(fields, saved_at=_utc_now()),
            "history": [],
        }
        save_goals_document(doc)
    else:
        doc = default_doc
        save_goals_document(doc)

    current_fields = _normalize_goal_fields(doc.get("current") or {})
    saved_at = (doc.get("current") or {}).get("saved_at") or _utc_now()
    history_raw = doc.get("history") or []
    history: list[dict[str, Any]] = []
    for item in history_raw:
        if not isinstance(item, dict):
            continue
        snap = _goal_snapshot(_normalize_goal_fields(item), saved_at=item.get("saved_at") or _utc_now())
        history.append(snap)

    return {
        "current": _goal_snapshot(current_fields, saved_at=saved_at),
        "history": history[-MAX_GOAL_HISTORY:],
    }


def save_goals_document(doc: dict[str, Any]) -> None:
    current = _goal_snapshot(
        _normalize_goal_fields(doc.get("current") or {}),
        saved_at=(doc.get("current") or {}).get("saved_at") or _utc_now(),
    )
    history = []
    for item in doc.get("history") or []:
        if isinstance(item, dict):
            history.append(
                _goal_snapshot(
                    _normalize_goal_fields(item),
                    saved_at=item.get("saved_at") or _utc_now(),
                )
            )
    payload = {"current": current, "history": history[-MAX_GOAL_HISTORY:]}
    _write_json(GOALS_PATH, payload)
    # Keep legacy flat file in sync for any older readers
    flat = {k: current[k] for k in GOAL_FIELDS}
    _write_json(LEGACY_OBJECTIVES_PATH, flat)


def save_goals(fields: dict[str, Any], *, source: str = "ux") -> dict[str, Any]:
    """
    Save a new goals snapshot from the UX (or agent).

    Pushes the previous current into history (max 10 prior versions).
    Skips a history push if content is unchanged (still refreshes saved_at).
    """
    doc = load_goals()
    new_fields = _normalize_goal_fields(fields)
    now = _utc_now()
    prev = doc["current"]

    if not _goals_equal(prev, new_fields):
        # Archive prior current (without duplicating identical consecutive snaps)
        archived = _goal_snapshot(prev, saved_at=prev.get("saved_at") or now)
        history = [archived] + list(doc.get("history") or [])
        doc["history"] = history[:MAX_GOAL_HISTORY]

    doc["current"] = _goal_snapshot(new_fields, saved_at=now)
    doc["current"]["source"] = source
    save_goals_document(doc)
    rebuild_state_summary()
    return doc


def current_goals() -> dict[str, Any]:
    return load_goals()["current"]


# --- Theses ----------------------------------------------------------------


def _empty_theses() -> dict[str, Any]:
    return {"items": []}


def load_theses() -> dict[str, Any]:
    doc = _read_json(THESES_PATH, _empty_theses())
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict)]
    return {"items": items[-MAX_THESES:]}


def save_theses_document(doc: dict[str, Any]) -> None:
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict)]
    _write_json(THESES_PATH, {"items": items[-MAX_THESES:]})


def upsert_thesis(
    *,
    symbol: str,
    narrative: str,
    target: float | int | str | None = None,
    entry: float | int | str | None = None,
    cost_basis: float | int | str | None = None,
    conviction: str = "Medium",
    horizon: str = "",
    status: str = "active",
    source: str = "agent",
    notes: str = "",
    supersede_active: bool = True,
) -> dict[str, Any]:
    """
    Append a thesis owned by the LLM/brain JSON.

    cost_basis is the canonical entry/cost field shown in the UI; `entry` is
    accepted as an alias for backwards compatibility.
    """
    symbol = symbol.upper().strip()
    doc = load_theses()
    now = _utc_now()

    if supersede_active:
        for item in doc["items"]:
            if item.get("symbol") == symbol and item.get("status") == "active":
                item["status"] = "superseded"
                item["superseded_at"] = now

    def _num(v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    basis = _num(cost_basis if cost_basis is not None else entry)
    entry_row = {
        "id": _new_id("thesis"),
        "saved_at": now,
        "symbol": symbol,
        "narrative": (narrative or "").strip(),
        "target": _num(target),
        "cost_basis": basis,
        "entry": basis,  # alias for older readers / prompts
        "conviction": (conviction or "Medium").strip(),
        "horizon": (horizon or "").strip(),
        "status": status,
        "source": source,
        "notes": (notes or "").strip(),
    }
    doc["items"].append(entry_row)
    save_theses_document(doc)
    rebuild_state_summary()
    return entry_row


def thesis_view(stored: dict[str, Any] | None) -> dict[str, Any]:
    """UI/API shape for a holding's thesis panel (JSON-backed only)."""
    if not stored:
        return {
            "narrative": "",
            "target": None,
            "cost_basis": None,
            "entry": None,
            "conviction": "",
            "horizon": "",
            "saved_at": None,
            "id": None,
            "missing": True,
        }
    basis = stored.get("cost_basis")
    if basis is None:
        basis = stored.get("entry")
    return {
        "narrative": stored.get("narrative") or "",
        "target": stored.get("target"),
        "cost_basis": basis,
        "entry": basis,
        "conviction": stored.get("conviction") or "",
        "horizon": stored.get("horizon") or "",
        "saved_at": stored.get("saved_at"),
        "id": stored.get("id"),
        "source": stored.get("source"),
        "missing": False,
    }


def load_holdings_book() -> list[dict[str, Any]]:
    """Position book only — no embedded thesis text."""
    doc = _read_json(HOLDINGS_PATH, {"items": []})
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict) and i.get("symbol")]
    return items


def save_holdings_book(items: list[dict[str, Any]]) -> None:
    _write_json(HOLDINGS_PATH, {"items": items})


def enrich_holdings(items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Join position book with LLM-persisted theses from theses.json."""
    book = deepcopy(items if items is not None else load_holdings_book())
    rows = []
    for h in book:
        row = deepcopy(h)
        row.pop("thesis", None)
        row["thesis"] = thesis_view(thesis_for_symbol(str(h.get("symbol") or "")))
        # Prefer thesis cost basis for display when present; else broker avg_cost
        if row["thesis"].get("cost_basis") is None and h.get("avg_cost") is not None:
            # Do not invent a thesis — leave missing; avg_cost stays on the row
            pass
        rows.append(row)
    return rows


def migrate_theses_cost_basis() -> int:
    """Backfill cost_basis from entry on older thesis rows."""
    doc = load_theses()
    changed = 0
    for item in doc["items"]:
        if item.get("cost_basis") is None and item.get("entry") is not None:
            item["cost_basis"] = item["entry"]
            changed += 1
        elif item.get("entry") is None and item.get("cost_basis") is not None:
            item["entry"] = item["cost_basis"]
            changed += 1
    if changed:
        save_theses_document(doc)
    return changed


def seed_theses_from_holdings(holdings: list[dict[str, Any]]) -> int:
    """Deprecated for LLM-owned theses — kept as no-op for API compatibility."""
    return 0


def active_theses() -> list[dict[str, Any]]:
    """Latest active thesis per symbol (by saved_at)."""
    by_symbol: dict[str, dict[str, Any]] = {}
    for item in load_theses()["items"]:
        if item.get("status") != "active":
            continue
        sym = str(item.get("symbol") or "").upper()
        if not sym:
            continue
        # normalize cost_basis alias
        if item.get("cost_basis") is None and item.get("entry") is not None:
            item = dict(item)
            item["cost_basis"] = item["entry"]
        prev = by_symbol.get(sym)
        if prev is None or str(item.get("saved_at") or "") >= str(prev.get("saved_at") or ""):
            by_symbol[sym] = item
    return sorted(by_symbol.values(), key=lambda x: x.get("symbol") or "")


def thesis_for_symbol(symbol: str) -> dict[str, Any] | None:
    symbol = symbol.upper().strip()
    for item in active_theses():
        if item.get("symbol") == symbol:
            return item
    return None


# --- Trades ----------------------------------------------------------------


def _empty_trades() -> dict[str, Any]:
    return {"items": []}


def load_trades() -> dict[str, Any]:
    doc = _read_json(TRADES_PATH, _empty_trades())
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict)]
    return {"items": items[-MAX_TRADES:]}


def save_trades_document(doc: dict[str, Any]) -> None:
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict)]
    _write_json(TRADES_PATH, {"items": items[-MAX_TRADES:]})


def record_trade(
    *,
    symbol: str,
    side: str,
    qty: float | int | str,
    price: float | int | str | None = None,
    rationale: str = "",
    thesis_id: str = "",
    status: str = "noted",
    source: str = "agent",
) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    side_u = side.upper().strip()
    try:
        qty_f = float(qty)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid qty: {qty}") from exc

    price_f: float | None
    try:
        price_f = float(price) if price is not None and price != "" else None
    except (TypeError, ValueError):
        price_f = None

    row = {
        "id": _new_id("trade"),
        "saved_at": _utc_now(),
        "symbol": symbol,
        "side": side_u,
        "qty": qty_f,
        "price": price_f,
        "rationale": (rationale or "").strip(),
        "thesis_id": (thesis_id or "").strip(),
        "status": (status or "noted").strip(),
        "source": source,
    }
    doc = load_trades()
    doc["items"].append(row)
    save_trades_document(doc)
    rebuild_state_summary()
    return row


def recent_trades(limit: int = 15) -> list[dict[str, Any]]:
    items = load_trades()["items"]
    return list(reversed(items[-limit:]))


# --- Global state summary --------------------------------------------------


def rebuild_state_summary(
    *,
    metrics: list[dict[str, Any]] | None = None,
    holdings_count: int | None = None,
) -> dict[str, Any]:
    """Refresh data/state.json — compact standing memory for the model."""
    goals = current_goals()
    theses = active_theses()
    trades = recent_trades(10)

    thesis_lines = []
    for t in theses:
        target = t.get("target")
        target_s = f"${target:.2f}" if isinstance(target, (int, float)) else "n/a"
        thesis_lines.append(
            f"{t.get('symbol')}: conviction={t.get('conviction')}, target={target_s}, "
            f"status={t.get('status')}, saved={t.get('saved_at')}"
        )

    trade_lines = []
    for tr in trades[:8]:
        px = tr.get("price")
        px_s = f" @ ${px:.2f}" if isinstance(px, (int, float)) else ""
        trade_lines.append(
            f"{tr.get('saved_at')} {tr.get('side')} {tr.get('qty')} {tr.get('symbol')}{px_s} "
            f"[{tr.get('status')}]"
        )

    summary_parts = [
        f"Goals ({goals.get('saved_at')}): risk={goals.get('risk_tolerance')}, "
        f"maxDD={goals.get('max_drawdown')}, targetRet={goals.get('target_return')}, "
        f"horizon={goals.get('horizon')}, maxName={goals.get('max_single_name')}, "
        f"cashFloor={goals.get('cash_floor')}.",
        f"Industries: {', '.join(goals.get('industries') or [])}. "
        f"Avoid: {', '.join(goals.get('avoid') or [])}.",
        f"Active theses ({len(theses)}): " + ("; ".join(thesis_lines) if thesis_lines else "none."),
        f"Recent trades: " + ("; ".join(trade_lines) if trade_lines else "none recorded."),
    ]
    if holdings_count is not None:
        summary_parts.insert(0, f"Tracked holdings in UI: {holdings_count}.")
    if metrics:
        metric_bits = [f"{m.get('label')}={m.get('value')}" for m in metrics[:6]]
        summary_parts.append("UI risk/metrics snapshot: " + "; ".join(metric_bits) + ".")

    summary_parts.append(
        "Standing rule: do not abandon a high-conviction thesis on short-term noise; "
        "only revise conviction or supersede when the core thesis is invalidated."
    )

    state = {
        "updated_at": _utc_now(),
        "summary": " ".join(summary_parts),
        "goals": {k: goals.get(k) for k in (*GOAL_FIELDS, "saved_at")},
        "active_theses": [
            {
                "id": t.get("id"),
                "symbol": t.get("symbol"),
                "conviction": t.get("conviction"),
                "target": t.get("target"),
                "entry": t.get("entry") if t.get("entry") is not None else t.get("cost_basis"),
                "cost_basis": t.get("cost_basis") if t.get("cost_basis") is not None else t.get("entry"),
                "horizon": t.get("horizon"),
                "saved_at": t.get("saved_at"),
                "narrative_preview": (t.get("narrative") or "")[:280],
            }
            for t in theses
        ],
        "recent_trades": trades[:10],
        "goal_history_count": len(load_goals().get("history") or []),
        "counts": {
            "active_theses": len(theses),
            "trades": len(load_trades()["items"]),
            "goal_history": len(load_goals().get("history") or []),
        },
    }
    _write_json(STATE_PATH, state)
    return state


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return rebuild_state_summary()
    return _read_json(STATE_PATH, {})


def format_memory_for_llm(
    *,
    symbol: str | None = None,
    include_goal_history: bool = True,
    max_narrative_chars: int = 600,
) -> str:
    """
    Compact structured memory block injected as a system message each chat turn.
    """
    state = load_state()
    goals_doc = load_goals()
    goals = goals_doc["current"]
    theses = active_theses()
    trades = recent_trades(12)

    lines: list[str] = [
        "=== AGENT BRAIN (persistent memory) ===",
        "Use this as standing truth. IB tools refresh market facts; this brain holds "
        "portfolio intent, prior theses, and trade memory. Re-evaluate carefully: "
        "short-term price/news noise alone should NOT break a high-conviction thesis "
        "unless the underlying thesis premises are invalidated OR the thesis has "
        "fully played out.",
        "",
    ]

    try:
        from brain.policy import THESIS_AUTONOMY_POLICY

        lines.append(THESIS_AUTONOMY_POLICY)
        lines.append("")
    except Exception:
        pass

    lines.extend(
        [
        f"[Global state summary @ {state.get('updated_at', 'n/a')}]",
        state.get("summary") or "(empty)",
        "",
        ]
    )

    try:
        from config import order_mode_status

        om = order_mode_status()
        lines.append("[Order mode — live env]")
        lines.append(
            f"  orders_allowed={om.get('orders_allowed')} | "
            f"IB_ALLOW_ORDERS={om.get('IB_ALLOW_ORDERS')} | "
            f"IB_READONLY={om.get('IB_READONLY')} | "
            f"connect_readonly={om.get('ib_connect_readonly')} | "
            f"{om.get('host')}:{om.get('port')} clientId={om.get('client_id')}"
        )
        if om.get("orders_allowed"):
            lines.append(
                "  Orders ARE enabled. Prefer place_equity_order(confirm=true) for "
                "risk trims, take-profits, invalidation exits, adds, and NEW IDEAS "
                "(Path N) when deployable cash exists — do not only recommend trades."
            )
        else:
            lines.append(
                "  Orders DISABLED — save_thesis/record_trade only; report blocker."
            )
        lines.append("")
    except Exception:
        pass

    lines.extend(
        [
        "[Current portfolio goals]",
        f"  saved_at: {goals.get('saved_at')}",
        f"  risk_tolerance: {goals.get('risk_tolerance')}",
        f"  max_drawdown: {goals.get('max_drawdown')}",
        f"  target_return: {goals.get('target_return')}",
        f"  horizon: {goals.get('horizon')}",
        f"  max_single_name: {goals.get('max_single_name')}",
        f"  cash_floor: {goals.get('cash_floor')}",
        f"  industries: {', '.join(goals.get('industries') or [])}",
        f"  avoid: {', '.join(goals.get('avoid') or [])}",
        f"  notes: {goals.get('notes')}",
        ]
    )

    if include_goal_history and goals_doc.get("history"):
        lines.append("")
        lines.append(f"[Goals history — last {len(goals_doc['history'])} prior saves]")
        for snap in goals_doc["history"][:5]:
            lines.append(
                f"  - {snap.get('saved_at')}: risk={snap.get('risk_tolerance')}, "
                f"maxDD={snap.get('max_drawdown')}, targetRet={snap.get('target_return')}"
            )

    lines.append("")
    lines.append(f"[Active theses — {len(theses)}]")
    if not theses:
        lines.append("  (none)")
    for t in theses:
        narr = (t.get("narrative") or "").strip()
        if len(narr) > max_narrative_chars:
            narr = narr[:max_narrative_chars] + "…"
        highlight = "  << ACTIVE UI SYMBOL" if symbol and t.get("symbol") == symbol.upper() else ""
        lines.append(
            f"  - {t.get('symbol')} | conviction={t.get('conviction')} | "
            f"target={t.get('target')} | cost_basis={t.get('cost_basis', t.get('entry'))} | "
            f"horizon={t.get('horizon')} | saved={t.get('saved_at')} | "
            f"id={t.get('id')}{highlight}"
        )
        lines.append(f"    narrative: {narr}")

    if symbol:
        focused = thesis_for_symbol(symbol)
        if focused:
            lines.append("")
            lines.append(f"[Focus thesis for {symbol.upper()}]")
            lines.append(json.dumps(focused, default=str))

    lines.append("")
    lines.append(f"[Recent trades — showing {len(trades)}]")
    if not trades:
        lines.append("  (none recorded yet)")
    for tr in trades:
        lines.append(
            f"  - {tr.get('saved_at')} {tr.get('side')} qty={tr.get('qty')} "
            f"{tr.get('symbol')} px={tr.get('price')} status={tr.get('status')} "
            f"id={tr.get('id')}"
        )
        if tr.get("rationale"):
            lines.append(f"    rationale: {tr['rationale'][:240]}")

    lines.append("")
    lines.append(
        "When you form or revise a thesis, call save_thesis with narrative, target, "
        "and cost_basis so the holdings thesis panel stays in sync. "
        "When you recommend a concrete trade decision worth remembering, call record_trade. "
        "Call get_brain_summary if you need a refreshed compact view."
    )
    lines.append("=== END AGENT BRAIN ===")
    return "\n".join(lines)


def ensure_brain_initialized(
    holdings: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ensure all brain files exist; theses are LLM-owned (not seeded from UI copy)."""
    load_goals()
    if not THESES_PATH.exists():
        _write_json(THESES_PATH, _empty_theses())
    if not TRADES_PATH.exists():
        _write_json(TRADES_PATH, _empty_trades())
    if not HOLDINGS_PATH.exists() and holdings:
        # Persist position book without embedded thesis blobs
        clean = []
        for h in holdings:
            row = {k: v for k, v in h.items() if k != "thesis"}
            clean.append(row)
        save_holdings_book(clean)
    migrate_theses_cost_basis()
    from brain.chat_store import ensure_chat_file

    ensure_chat_file()
    book = holdings or load_holdings_book()
    return rebuild_state_summary(
        metrics=metrics,
        holdings_count=len(book),
    )


def brain_overview() -> dict[str, Any]:
    """Lightweight payload for UX / initial connection."""
    state = load_state()
    goals = current_goals()
    return {
        "updated_at": state.get("updated_at"),
        "summary": state.get("summary"),
        "goals_saved_at": goals.get("saved_at"),
        "counts": state.get("counts")
        or {
            "active_theses": len(active_theses()),
            "trades": len(load_trades()["items"]),
            "goal_history": len(load_goals().get("history") or []),
        },
        "active_theses": state.get("active_theses") or [],
        "recent_trades": state.get("recent_trades") or [],
        "goals": {k: goals.get(k) for k in (*GOAL_FIELDS, "saved_at")},
    }
