"""Persisted UX preferences (toggles, schedule) for the terminal UI."""

from __future__ import annotations

from typing import Any

from brain.store import DATA_DIR, _read_json, _utc_now, _write_json

UX_PATH = DATA_DIR / "ux_state.json"

ALLOWED_EVERY = (1, 5, 10, 15, 30)
ALLOWED_UNITS = ("minutes", "hours", "days")

_UNIT_SECONDS = {
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
}

DEFAULT_UX = {
    "automated": False,
    "auto_every": 5,
    "auto_unit": "minutes",
    "auto_interval_sec": 300,
    "last_auto_run_at": None,
    "updated_at": None,
}


def default_ux() -> dict[str, Any]:
    return dict(DEFAULT_UX)


def normalize_every(value: Any, default: int = 5) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n in ALLOWED_EVERY else default


def normalize_unit(value: Any, default: str = "minutes") -> str:
    unit = str(value or default).strip().lower()
    return unit if unit in ALLOWED_UNITS else default


def interval_seconds(every: int, unit: str) -> int:
    every_n = normalize_every(every)
    unit_n = normalize_unit(unit)
    return every_n * _UNIT_SECONDS[unit_n]


def format_interval_label(every: int, unit: str) -> str:
    every_n = normalize_every(every)
    unit_n = normalize_unit(unit)
    short = {"minutes": "M", "hours": "H", "days": "D"}[unit_n]
    return f"{every_n}{short}"


def load_ux_state() -> dict[str, Any]:
    raw = _read_json(UX_PATH, default_ux())
    if not isinstance(raw, dict):
        raw = {}
    state = default_ux()
    state["automated"] = bool(raw.get("automated", False))

    # Prefer explicit every/unit; fall back from legacy auto_interval_sec
    if "auto_every" in raw or "auto_unit" in raw:
        every = normalize_every(raw.get("auto_every"), 5)
        unit = normalize_unit(raw.get("auto_unit"), "minutes")
    else:
        try:
            sec = int(raw.get("auto_interval_sec") or 300)
        except (TypeError, ValueError):
            sec = 300
        # Best-effort reverse map for legacy seconds
        if sec % 86400 == 0 and (sec // 86400) in ALLOWED_EVERY:
            every, unit = sec // 86400, "days"
        elif sec % 3600 == 0 and (sec // 3600) in ALLOWED_EVERY:
            every, unit = sec // 3600, "hours"
        elif sec % 60 == 0 and (sec // 60) in ALLOWED_EVERY:
            every, unit = sec // 60, "minutes"
        else:
            every, unit = 5, "minutes"

    state["auto_every"] = every
    state["auto_unit"] = unit
    state["auto_interval_sec"] = interval_seconds(every, unit)
    state["last_auto_run_at"] = raw.get("last_auto_run_at")
    state["updated_at"] = raw.get("updated_at") or _utc_now()
    state["interval_label"] = format_interval_label(every, unit)
    return state


def save_ux_state(patch: dict[str, Any] | None = None) -> dict[str, Any]:
    state = load_ux_state()
    patch = patch or {}

    if "automated" in patch:
        state["automated"] = bool(patch["automated"])

    every = state["auto_every"]
    unit = state["auto_unit"]
    if "auto_every" in patch:
        every = normalize_every(patch["auto_every"], every)
    if "auto_unit" in patch:
        unit = normalize_unit(patch["auto_unit"], unit)

    # Legacy: allow setting seconds directly → snap to nearest allowed combo
    if "auto_interval_sec" in patch and "auto_every" not in patch and "auto_unit" not in patch:
        try:
            sec = max(60, int(patch["auto_interval_sec"]))
        except (TypeError, ValueError):
            sec = state["auto_interval_sec"]
        if sec % 86400 == 0 and (sec // 86400) in ALLOWED_EVERY:
            every, unit = sec // 86400, "days"
        elif sec % 3600 == 0 and (sec // 3600) in ALLOWED_EVERY:
            every, unit = sec // 3600, "hours"
        elif sec % 60 == 0 and (sec // 60) in ALLOWED_EVERY:
            every, unit = sec // 60, "minutes"

    state["auto_every"] = every
    state["auto_unit"] = unit
    state["auto_interval_sec"] = interval_seconds(every, unit)
    state["interval_label"] = format_interval_label(every, unit)

    if "last_auto_run_at" in patch:
        state["last_auto_run_at"] = patch["last_auto_run_at"]

    state["updated_at"] = _utc_now()
    payload = {
        "automated": state["automated"],
        "auto_every": state["auto_every"],
        "auto_unit": state["auto_unit"],
        "auto_interval_sec": state["auto_interval_sec"],
        "last_auto_run_at": state["last_auto_run_at"],
        "updated_at": state["updated_at"],
    }
    _write_json(UX_PATH, payload)
    return state


def mark_auto_run() -> dict[str, Any]:
    return save_ux_state({"last_auto_run_at": _utc_now()})


def ensure_ux_state() -> dict[str, Any]:
    if not UX_PATH.exists():
        return save_ux_state({})
    return load_ux_state()
