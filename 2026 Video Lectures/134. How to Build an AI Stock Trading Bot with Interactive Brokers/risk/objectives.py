"""Portfolio objectives loader (shared by UI, sizing, and agent tools)."""

from __future__ import annotations

from typing import Any

from brain.store import GOAL_FIELDS, current_goals, default_goals_fields, save_goals


def default_objectives() -> dict[str, Any]:
    return default_goals_fields()


def load_objectives() -> dict[str, Any]:
    """Current goals fields (no history) for sizing / UI compatibility."""
    goals = current_goals()
    return {k: goals[k] for k in GOAL_FIELDS}


def save_objectives(objectives: dict[str, Any], *, source: str = "ux") -> dict[str, Any]:
    """Persist goals with dated history via the brain store."""
    return save_goals(objectives, source=source)


def get_portfolio_objectives() -> dict[str, Any]:
    """Tool-facing wrapper — includes saved_at for the model."""
    goals = current_goals()
    return {
        "ok": True,
        "objectives": {k: goals.get(k) for k in (*GOAL_FIELDS, "saved_at")},
    }
