"""Compressed, durable chat history for the strategy desk."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from brain.store import DATA_DIR, _read_json, _utc_now, _write_json
from config import get_settings

CHAT_PATH = DATA_DIR / "chat_history.json"

# Keep this many recent turns verbatim after compression.
MAX_RECENT_MESSAGES = 12
# Trigger LLM compression when raw message count exceeds this.
COMPRESS_AFTER = 20
# Hard cap on stored verbatim messages even if compression fails.
MAX_MESSAGES = 40
# Max chars of the rolling summary.
MAX_SUMMARY_CHARS = 4000


def _empty_chat() -> dict[str, Any]:
    return {
        "updated_at": _utc_now(),
        "summary": "",
        "messages": [],
        "compression_count": 0,
    }


def load_chat() -> dict[str, Any]:
    doc = _read_json(CHAT_PATH, _empty_chat())
    messages = [
        m
        for m in (doc.get("messages") or [])
        if isinstance(m, dict) and m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    return {
        "updated_at": doc.get("updated_at") or _utc_now(),
        "summary": str(doc.get("summary") or ""),
        "messages": messages[-MAX_MESSAGES:],
        "compression_count": int(doc.get("compression_count") or 0),
    }


def save_chat(doc: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "updated_at": _utc_now(),
        "summary": str(doc.get("summary") or "")[:MAX_SUMMARY_CHARS],
        "messages": (doc.get("messages") or [])[-MAX_MESSAGES:],
        "compression_count": int(doc.get("compression_count") or 0),
    }
    _write_json(CHAT_PATH, payload)
    return payload


def append_turn(user_message: str, assistant_message: str) -> dict[str, Any]:
    """Append a user/assistant pair and compress if over threshold."""
    doc = load_chat()
    now = _utc_now()
    doc["messages"].append(
        {"role": "user", "content": user_message.strip(), "saved_at": now}
    )
    doc["messages"].append(
        {
            "role": "assistant",
            "content": (assistant_message or "").strip(),
            "saved_at": now,
        }
    )
    doc["messages"] = doc["messages"][-MAX_MESSAGES:]
    if len(doc["messages"]) > COMPRESS_AFTER:
        doc = compress_chat(doc)
    else:
        doc = save_chat(doc)
    return doc


def _heuristic_compress(messages: list[dict[str, Any]]) -> str:
    bits = []
    for m in messages:
        role = "U" if m.get("role") == "user" else "A"
        text = " ".join(str(m.get("content") or "").split())
        if len(text) > 220:
            text = text[:220] + "…"
        bits.append(f"{role}: {text}")
    blob = " | ".join(bits)
    return blob[:MAX_SUMMARY_CHARS]


def compress_chat(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Fold older messages into `summary`, keep the newest MAX_RECENT_MESSAGES raw.
    Uses the LLM when available; falls back to heuristic compression.
    """
    doc = dict(doc or load_chat())
    messages = list(doc.get("messages") or [])
    if len(messages) <= MAX_RECENT_MESSAGES:
        return save_chat(doc)

    older = messages[:-MAX_RECENT_MESSAGES]
    recent = messages[-MAX_RECENT_MESSAGES:]
    prior_summary = str(doc.get("summary") or "").strip()

    transcript = "\n".join(
        f"{m.get('role', '').upper()}: {m.get('content', '')}" for m in older
    )
    prompt = (
        "Compress the following trading-desk chat into a durable memory brief "
        "for a portfolio AI. Preserve: symbols discussed, theses/conviction, "
        "targets, cost basis, trade decisions, risk concerns, and open questions. "
        "Drop pleasantries and repeated tool chatter. Max ~400 words.\n\n"
    )
    if prior_summary:
        prompt += f"EXISTING SUMMARY:\n{prior_summary}\n\n"
    prompt += f"OLDER MESSAGES TO MERGE:\n{transcript}"

    new_summary = None
    try:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You compress trading chat history into factual memory briefs.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        new_summary = (resp.choices[0].message.content or "").strip()
    except Exception:
        new_summary = None

    if not new_summary:
        merged = (prior_summary + "\n" if prior_summary else "") + _heuristic_compress(older)
        new_summary = merged.strip()[:MAX_SUMMARY_CHARS]
    else:
        new_summary = new_summary[:MAX_SUMMARY_CHARS]

    doc["summary"] = new_summary
    doc["messages"] = recent
    doc["compression_count"] = int(doc.get("compression_count") or 0) + 1
    return save_chat(doc)


def chat_for_ui() -> list[dict[str, str]]:
    """Messages to render on page load (summary banner + recent turns)."""
    doc = load_chat()
    out: list[dict[str, str]] = []
    summary = str(doc.get("summary") or "").strip()
    if summary:
        out.append(
            {
                "role": "assistant",
                "content": f"[COMPRESSED MEMORY]\n{summary}",
            }
        )
    for m in doc.get("messages") or []:
        out.append({"role": m["role"], "content": m["content"]})
    return out


def chat_for_agent(limit: int = 12) -> tuple[str | None, list[dict[str, str]]]:
    """
    Return (summary_block, recent_messages) for injection into the agent.
    """
    doc = load_chat()
    summary = str(doc.get("summary") or "").strip() or None
    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in (doc.get("messages") or [])[-limit:]
    ]
    return summary, recent


def ensure_chat_file() -> None:
    if not CHAT_PATH.exists():
        save_chat(_empty_chat())


def prepare_chat_for_session() -> dict[str, Any]:
    """
    Called on app reload: ensure file exists and compress if the raw transcript
    has grown past the threshold so reloads always start from a tight memory.
    """
    ensure_chat_file()
    doc = load_chat()
    if len(doc.get("messages") or []) > COMPRESS_AFTER:
        return compress_chat(doc)
    return doc
