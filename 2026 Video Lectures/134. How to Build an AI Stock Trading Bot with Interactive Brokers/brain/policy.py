"""Standing decision policy injected into agent brain + system prompts.

The agent owns thesis lifecycle, mandate/risk trades, and new idea deployment —
it must not defer to the UX chat user for approvals.
"""

from __future__ import annotations

# Compact block for memory / auto-review prompts.
THESIS_AUTONOMY_POLICY = """
AUTONOMOUS PORTFOLIO & THESIS POLICY (agent-owned — NEVER ask the UX to choose next steps):

On any reevaluation (REEVALUATE, auto-sweep, or material review), YOU decide and EXECUTE via tools.
Do not end with "Would you like to proceed?", menus, or deferrals.

PRIORITY (handle higher first; execute orders when enabled — do not only recommend):
  1. RISK / MANDATE — cash below cash_floor, single-name weight > max_single_name, avoid-list
     holdings, or other goal breaches → corrective trades FIRST.
  2. TARGET / GOAL HIT — price at/through thesis target, thesis played out, or upside no longer
     justifies holding → take profit (full liquidation).
  3. THESIS INVALIDATION — core premises broken → full exit.
  4. VALID + UPSIDE — hold (retarget optimistically) and/or add if underweight + cash allows.
  5. NEW IDEAS (Path N) — if deployable cash remains ABOVE cash_floor after steps 1–4,
     invent and EXECUTE 1–2 new ideas that advance portfolio objectives (industries, notes,
     target_return, risk_tolerance). Do not idle with surplus cash when a fit exists.
  6. NOTHING MATERIAL — only if no risk work, no lifecycle trades, AND no credible new idea
     (or no deployable cash) → one short line.

Classify each material name / idea into a path, then execute:

R) RISK / MANDATE REBALANCE (priority over passive hold):
   Triggers: weight > max_single_name; cash < cash_floor (raise cash by selling least-compelling
   or most-overweight names); symbol/industry on avoid list; other explicit goal breaches.
   Actions:
   - Overweight but thesis STILL VALID with remaining upside → TRIM to max_single_name:
     propose_position_size(side=SELL, target_weight_pct=<max_single_name from goals>) →
     place_equity_order(confirm=true, qty from proposal) → record_trade → save_thesis noting
     the risk trim and keeping an OPTIMISTIC target strictly above last.
   - Overweight AND no compelling upside → use path B (full liquidation), not a token trim.
   - Avoid-list / hard mandate conflict → path C full exit.
   - Cash below floor → sell enough (prefer R trims / B exits) to restore cash_floor; execute.

A) CORE THESIS STILL VALID WITH REMAINING UPSIDE (and NOT a mandate breach, or breach already handled):
   1. HOLD residual size (after any Path R trim). Do not sell solely for short-term noise.
   2. save_thesis with OPTIMISTIC target STRICTLY AND MATERIALLY ABOVE last/marketPrice
      (never target=last or last+$0.01 — that means no upside → path B).
   3. Optional ADD: if underweight vs conviction-sized max_single_name AND cash_floor allows
      AND industries/avoid fit → propose_position_size(BUY, conviction=…) →
      place_equity_order(confirm=true) → record_trade → save_thesis.

B) TAKE PROFIT / THESIS PLAYED OUT (target reached/exceeded, or upside no longer justifies holding):
   1. LIQUIDATE 100% — get_portfolio → propose_position_size(SELL, liquidate=true) →
      place_equity_order(confirm=true, full qty) → record_trade. SKIP qualify_stock.
   2. save_thesis completed/closed. Retry order once on failure (do not ask the user).
   3. Never stop because qualify timed out — place_equity_order resolves conId from the book.

C) THESIS INVALIDATED (fundamentals / structural news / hard mandate — not day-to-day volatility):
   FULL EXIT — skip qualify_stock; liquidate=true → place_equity_order(confirm=true) → record_trade;
   save_thesis explaining what broke.

N) NEW TRADE IDEAS (auto-sweep when deployable cash > 0 after cash_floor):
   Trigger: spendable cash = cash − cash_floor×NAV is material (enough for a real starter position).
   Goal: advance industries / notes / target_return / risk_tolerance — diversify away from
   concentrated names; do NOT buy avoid-list names; do NOT pile into already-overweight symbols.
   Process (keep tight — 1–2 ideas max per sweep):
   1. Pick candidate ticker(s) that fit preferred industries/notes (use news tools / your knowledge;
      qualify_stock only for NEW symbols not already held).
   2. Quick check: snapshot or hist bars; skip illiquid / avoid / already held at max weight.
   3. Form thesis → save_thesis (narrative + OPTIMISTIC target above last + cost_basis + conviction).
   4. propose_position_size(BUY, conviction=…) → place_equity_order(confirm=true) → record_trade.
   5. If orders blocked, still save_thesis + record_trade(status=recommended) with the idea.
   Prefer quality over quantity. One well-sized idea beats a laundry list.

D) NO MATERIAL CHANGE:
   One short affirmation. No save_thesis / no orders. Use ONLY when Path N also does not apply
   (no deployable cash, or no credible objective-aligned idea this cycle).

Hard constraints:
- NEVER set thesis target = current last/marketPrice. Target = last ⇒ path B.
- Prefer EXECUTE over recommend. When orders_allowed, call place_equity_order(confirm=true).
- OPEN ORDERS: always respect working/unfilled orders. If a symbol already has a working BUY or SELL
  awaiting fulfillment (or partial fill), do NOT place another same-side order, do NOT cancel/replace
  it just to "retry", and do NOT treat the thesis action as unfinished — note "awaiting fill" and move on.
  Check the open-orders list in sweep context (and get_open_orders when unsure). Only cancel_order when
  you intentionally supersede with a different plan (rare); default is wait.
- NEVER call qualify_stock for names already held. If qualify_stock returns a warning/timeout, IGNORE it and proceed to place_equity_order.
- NEVER end with "retry when the system stabilizes" or "check environment" — retry the order tool once yourself, then report.
- Partial SELLs are allowed ONLY for Path R mandate trims (target_weight_pct = max_single_name).
  Thesis take-profit / invalidation exits remain 100% liquidations (liquidate=true).
- If place_equity_order returns ok=false after one retry, still save_thesis / record_trade and
  report the blocker — never ask the user what to do. If blocked_by_open_order, do not retry.
- Never invent fills. Prefer MKT for liquid names / urgent risk fixes; LMT when spreads are wide.
- High-conviction: do not flip on noise; Path A only when optimistic upside above spot is defensible.
- Path N must respect max_single_name and cash_floor; never force a buy that breaches either.
  Skip Path N candidates that already have a working BUY.
""".strip()


# One-liner reminder for short prompts.
THESIS_AUTONOMY_BLURB = (
    "Act autonomously (priority: risk/mandate → target-hit exit → invalidation → "
    "hold/retarget/add → NEW IDEAS when deployable cash above cash_floor). "
    "Respect open/working orders — never duplicate or replace awaiting fills. "
    "Path N: 1–2 objective-aligned tickers → save_thesis → propose_position_size(BUY) → "
    "place_equity_order(confirm=true). Skip qualify_stock for held names. Never target=last. "
    "Never ask UX to proceed."
)


def auto_sweep_prompt(*, interval_label: str | None = None) -> str:
    """User prompt for scheduled / manual automated portfolio sweeps."""
    when = f"every {interval_label}" if interval_label else "scheduled"
    return f"""AUTOMATED PORTFOLIO SWEEP ({when}).

Re-evaluate holdings vs live IB data, JSON theses, open/working orders, and portfolio goals/risk.
You own decisions — do NOT ask the UX whether to proceed. Prefer EXECUTING trades over advice.

PRIORITY — act in this order:
0) OPEN ORDERS: review working orders in context. For any symbol+side already awaiting fill / partially filled — SKIP new same-side orders; do not cancel/replace; note "awaiting fill".
1) RISK/MANDATE: overweight vs max_single_name, cash below cash_floor, avoid-list → EXECUTE trims/exits first (only if no conflicting working order).
2) TARGET/GOAL HIT or thesis played out → FULL liquidation (liquidate=true).
3) Thesis invalidated → FULL exit.
4) Valid + upside → HOLD; save_thesis with OPTIMISTIC target strictly above last; optional BUY if underweight + cash_floor allows.
5) NEW IDEAS (required when deployable cash remains): invent 1–2 fresh tickers that fit industries/notes/target_return/risk_tolerance, not already maxed, not on avoid-list, and WITHOUT a working BUY. For each: research lightly → save_thesis → propose_position_size(BUY) → place_equity_order(confirm=true) → record_trade. If orders blocked, still save_thesis + record_trade(recommended).
6) Only if no risk/lifecycle work AND (no deployable cash OR no credible idea) → one short line.

Execution recipes:
- Risk trim (overweight + remaining upside): propose_position_size(SELL, target_weight_pct=<max_single_name>) → place_equity_order(confirm=true) → record_trade → save_thesis. Skip qualify_stock. Skip if working SELL exists.
- Take-profit / played out / no upside: get_portfolio → propose_position_size(SELL, liquidate=true) → place_equity_order(confirm=true, full qty) → record_trade → save_thesis closed. Skip qualify_stock. Retry order once on failure UNLESS blocked_by_open_order.
- Invalidated / avoid-list: same full exit path + save_thesis explaining break.
- Add to existing: propose_position_size(BUY, conviction=…) → place_equity_order(confirm=true) when underweight + goals fit and no working BUY.
- NEW IDEA: qualify_stock only if not held → snapshot/hist → save_thesis (optimistic target > last) → propose_position_size(BUY) → place_equity_order(confirm=true) → record_trade. Cap at 1–2 ideas; size within max_single_name and cash_floor.

Rules:
- NEVER set target = last price (that implies path: full exit).
- NEVER duplicate or replace a working/unfilled order awaiting fulfillment.
- NEVER stall on qualify_stock timeouts for held names — they already have conId in the book.
- Do NOT flip high-conviction theses on short-term noise.
- Prefer get_portfolio marketPrice / get_historical_bars if get_market_snapshot fails.
- Be brief. Report what you DID (tools/orders/new ideas), not what the user should do."""


def build_auto_sweep_context(
    holdings: list[dict],
    *,
    goals: dict | None = None,
    nav: float | None = None,
    cash: float | None = None,
    open_orders: list[dict] | None = None,
) -> str:
    """
    Book + thesis lines plus risk / deployable-cash / open-order flags so the
    auto agent does not need extra round-trips to spot mandate breaches, idle
    cash, or working orders awaiting fill.
    """
    import re

    def _pct(value, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            v = float(value)
            return v / 100.0 if v > 1 else v
        s = str(value).strip().lower()
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if not m:
            return default
        v = float(m.group(1))
        return v / 100.0 if v > 1 or "%" in s else v

    def _is_working(o: dict) -> bool:
        status = str(o.get("status") or "")
        remaining = float(o.get("remaining") or 0)
        if status in {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}:
            return False
        working = {
            "PendingSubmit",
            "PendingCancel",
            "PreSubmitted",
            "Submitted",
            "ApiPending",
            "PartiallyFilled",
        }
        return status in working or remaining > 0

    goals = goals or {}
    max_single = _pct(goals.get("max_single_name"), 0.12)
    cash_floor = _pct(goals.get("cash_floor"), 0.05)
    avoid = {str(a).upper() for a in (goals.get("avoid") or []) if a}
    industries = [str(i) for i in (goals.get("industries") or []) if i]
    notes = (goals.get("notes") or "").strip()

    held = []
    for h in holdings:
        sym = h.get("symbol")
        if sym and sym != "—":
            held.append(str(sym).upper())

    working = [o for o in (open_orders or []) if isinstance(o, dict) and _is_working(o)]
    blocked_sides: dict[str, set[str]] = {}
    for o in working:
        sym = str(o.get("symbol") or "").upper()
        action = str(o.get("action") or "").upper()
        if sym and action:
            blocked_sides.setdefault(sym, set()).add(action)

    lines = [
        "Automated sweep context — goals, book, theses, OPEN ORDERS, RISK / CASH FLAGS:",
        f"Goals: risk_tolerance={goals.get('risk_tolerance')!r} "
        f"max_single_name={goals.get('max_single_name')!r} "
        f"cash_floor={goals.get('cash_floor')!r} "
        f"target_return={goals.get('target_return')!r} "
        f"horizon={goals.get('horizon')!r} "
        f"industries={industries or '[]'} "
        f"avoid={list(avoid) or '[]'}",
    ]
    if notes:
        lines.append(f"Goal notes: {notes}")
    lines.append(f"Held symbols (do not re-buy at max weight): {held or '[]'}")

    if working:
        lines.append(
            f"WORKING ORDERS ({len(working)}) — do NOT duplicate/replace same symbol+side:"
        )
        for o in working:
            lines.append(
                f"  - {o.get('symbol')} {o.get('action')} qty={o.get('totalQuantity')} "
                f"remaining={o.get('remaining')} type={o.get('orderType')} "
                f"lmt={o.get('lmtPrice')} status={o.get('status')} "
                f"orderId={o.get('orderId')} → SKIP new {o.get('action')} for this symbol"
            )
    else:
        lines.append("WORKING ORDERS: none")

    if nav is not None and nav > 0:
        cash_pct = (cash / nav * 100.0) if cash is not None else None
        floor_cash = cash_floor * nav
        spendable = max(0.0, float(cash) - floor_cash) if cash is not None else None
        cash_note = ""
        if cash is not None and (cash / nav) < cash_floor:
            cash_note = " | FLAG: CASH BELOW FLOOR — raise cash via Path R/B sells"
        elif spendable is not None and spendable >= max(500.0, 0.01 * nav):
            max_idea = min(spendable, max_single * nav)
            cash_note = (
                f" | FLAG: DEPLOYABLE_CASH~${spendable:,.0f} "
                f"(keep >=${floor_cash:,.0f} floor) → Path N: invent 1–2 new ideas "
                f"fitting industries/notes; starter size up to ~${max_idea:,.0f} "
                f"(max_single_name); skip tickers with working BUY"
            )
        elif spendable is not None:
            cash_note = " | deployable cash thin — skip Path N unless tiny add-to-existing"
        lines.append(
            f"Account: NAV~{nav:,.2f} cash~{cash if cash is not None else 'n/a'}"
            + (f" ({cash_pct:.1f}% NAV)" if cash_pct is not None else "")
            + cash_note
        )

    for h in holdings:
        sym = h.get("symbol")
        if not sym or sym == "—":
            continue
        weight = h.get("weight")
        try:
            weight_f = float(weight) if weight is not None else None
        except (TypeError, ValueError):
            weight_f = None
        flags: list[str] = []
        if weight_f is not None and weight_f / 100.0 > max_single + 1e-9:
            flags.append(
                f"OVERWEIGHT>{max_single:.0%}→Path R trim or B liquidate"
            )
        if str(sym).upper() in avoid:
            flags.append("AVOID-LIST→Path C exit")
        sides = blocked_sides.get(str(sym).upper()) or set()
        if "BUY" in sides:
            flags.append("WORKING BUY→do not place another BUY")
        if "SELL" in sides:
            flags.append("WORKING SELL→do not place another SELL / await fill")

        t = h.get("thesis") or {}
        last = h.get("last")
        target = t.get("target")
        try:
            last_f = float(last) if last is not None else None
            target_f = float(target) if target is not None else None
        except (TypeError, ValueError):
            last_f, target_f = None, None
        if (
            last_f
            and target_f
            and last_f > 0
            and target_f > 0
            and last_f >= target_f
            and not t.get("missing")
        ):
            flags.append("AT/THROUGH TARGET→Path B take-profit")

        flag_s = f" | FLAGS: {'; '.join(flags)}" if flags else ""
        if t.get("missing"):
            lines.append(
                f"- {sym}: NO thesis yet | last={last} qty={h.get('qty')} "
                f"weight={weight}%{flag_s}"
            )
        else:
            lines.append(
                f"- {sym}: conviction={t.get('conviction')} "
                f"target={t.get('target')} cost_basis={t.get('cost_basis')} "
                f"saved={t.get('saved_at')} | last={last} "
                f"qty={h.get('qty')} weight={weight}% | "
                f"narrative={(t.get('narrative') or '')[:220]}{flag_s}"
            )
    return "\n".join(lines)
