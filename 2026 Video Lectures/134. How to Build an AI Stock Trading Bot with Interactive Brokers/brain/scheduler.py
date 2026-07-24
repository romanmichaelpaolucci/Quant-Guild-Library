"""Background automated thesis reevaluation (daemon thread, non-blocking)."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from brain.ux_state import format_interval_label, load_ux_state, mark_auto_run

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_STARTED = False

_STATUS: dict[str, Any] = {
    "running": False,
    "last_error": None,
    "last_started_at": None,
    "last_finished_at": None,
    "last_content_preview": None,
}


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_auto_status() -> dict[str, Any]:
    ux = load_ux_state()
    last_ts = _parse_iso(ux.get("last_auto_run_at"))
    interval = int(ux.get("auto_interval_sec") or 300)
    next_due = None
    if ux.get("automated") and last_ts is not None:
        next_due = datetime.fromtimestamp(last_ts + interval, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    elif ux.get("automated") and last_ts is None:
        next_due = "due"

    return {
        **ux,
        "running": bool(_STATUS["running"]),
        "last_error": _STATUS["last_error"],
        "last_started_at": _STATUS["last_started_at"],
        "last_finished_at": _STATUS["last_finished_at"],
        "last_content_preview": _STATUS["last_content_preview"],
        "next_due_at": next_due,
        "scheduler_alive": bool(_THREAD and _THREAD.is_alive()),
    }


def _execute_auto_review() -> None:
    """
    Heavy work lives here — never call from a Flask request thread path
    except via the scheduler worker (already off-request).
    """
    # Lazy imports keep Flask startup light and avoid circular imports.
    from agent import TradingAgent
    from brain import (
        append_turn,
        enrich_holdings,
        format_memory_for_llm,
        load_goals,
        load_holdings_book,
    )
    from brain.policy import auto_sweep_prompt, build_auto_sweep_context
    from ib_bridge.ui_portfolio import fetch_live_portfolio_for_ui

    ux = load_ux_state()
    label = format_interval_label(ux.get("auto_every", 5), ux.get("auto_unit", "minutes"))
    prompt = auto_sweep_prompt(interval_label=label)

    live = fetch_live_portfolio_for_ui()
    if live.get("ok") and live.get("holdings"):
        book = [{k: v for k, v in h.items() if k != "thesis"} for h in live["holdings"]]
        nav = live.get("nav")
        cash = live.get("cash")
    else:
        book = load_holdings_book()
        nav = None
        cash = None
    holdings = enrich_holdings(book)
    goals = (load_goals() or {}).get("current") or {}

    open_orders: list = []
    try:
        from ib_bridge import get_open_orders

        oo = get_open_orders()
        if oo.get("ok"):
            open_orders = oo.get("working_orders") or oo.get("open_orders") or []
    except Exception:
        log.exception("Failed to load open orders for auto-sweep context")

    extra = build_auto_sweep_context(
        holdings,
        goals=goals,
        nav=nav,
        cash=cash,
        open_orders=open_orders,
    )
    memory = format_memory_for_llm(symbol=None)

    agent = TradingAgent()
    result = agent.run(
        prompt,
        symbol=None,
        history=None,
        extra_context=extra,
        memory=memory,
    )
    display_user = f"[AUTO] portfolio sweep · every {label}"
    append_turn(display_user, result.content)
    mark_auto_run()
    preview = (result.content or "").strip()
    _STATUS["last_content_preview"] = preview[:280]


def run_auto_review_now(*, force: bool = False) -> dict[str, Any]:
    """
    Trigger one auto-review on a worker thread if idle.
    Returns immediately with acceptance status (non-blocking).
    """
    ux = load_ux_state()
    if not force and not ux.get("automated"):
        return {"ok": False, "accepted": False, "reason": "automated_off"}

    if not _RUN_LOCK.acquire(blocking=False):
        return {"ok": True, "accepted": False, "reason": "already_running", **get_auto_status()}

    def _worker() -> None:
        _STATUS["running"] = True
        _STATUS["last_error"] = None
        _STATUS["last_started_at"] = _utc_iso()
        try:
            _execute_auto_review()
        except Exception as exc:
            log.exception("Auto-review failed")
            _STATUS["last_error"] = f"{type(exc).__name__}: {exc}"
            # Still advance last_auto_run_at so we don't tight-loop on hard failures
            try:
                mark_auto_run()
            except Exception:
                pass
        finally:
            _STATUS["running"] = False
            _STATUS["last_finished_at"] = _utc_iso()
            _RUN_LOCK.release()

    threading.Thread(target=_worker, name="auto-review-worker", daemon=True).start()
    return {"ok": True, "accepted": True, "reason": "started", **get_auto_status()}


def _scheduler_loop() -> None:
    """
    Lightweight tick loop (~1s). Does not hold IB/OpenAI work on this thread —
    it only decides when to spawn a worker.
    """
    log.info("Auto-review scheduler started")
    while not _STOP.is_set():
        try:
            ux = load_ux_state()
            if ux.get("automated") and not _STATUS["running"]:
                interval = int(ux.get("auto_interval_sec") or 300)
                last_ts = _parse_iso(ux.get("last_auto_run_at"))
                now = time.time()
                due = last_ts is None or (now - last_ts) >= interval
                if due:
                    run_auto_review_now(force=False)
        except Exception:
            log.exception("Scheduler tick failed")
        _STOP.wait(1.0)
    log.info("Auto-review scheduler stopped")


def start_auto_scheduler() -> bool:
    """
    Start the daemon scheduler once per process.
    Skips the Flask reloader parent process to avoid duplicate workers.
    """
    global _THREAD, _STARTED
    with _LOCK:
        if _STARTED and _THREAD and _THREAD.is_alive():
            return False
        # Werkzeug reloader spawns a parent that should not run jobs.
        if os.environ.get("WERKZEUG_RUN_MAIN") == "false":
            return False
        _STOP.clear()
        _THREAD = threading.Thread(
            target=_scheduler_loop,
            name="auto-review-scheduler",
            daemon=True,
        )
        _THREAD.start()
        _STARTED = True
        return True


def stop_auto_scheduler() -> None:
    _STOP.set()


def notify_ux_changed() -> None:
    """
    Called after UX saves. If automated just turned on and never ran,
    kick a worker soon without blocking the request.
    """
    ux = load_ux_state()
    if not ux.get("automated"):
        return
    if ux.get("last_auto_run_at") is None and not _STATUS["running"]:
        # Defer slightly so the HTTP response returns first
        def _deferred() -> None:
            time.sleep(2.0)
            if load_ux_state().get("automated"):
                run_auto_review_now(force=False)

        threading.Thread(target=_deferred, name="auto-review-kick", daemon=True).start()
