"""Iterative ChatGPT ↔ IB tool-calling agent loop."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from agent.executor import execute_tool
from agent.schemas import SYSTEM_PROMPT, TOOL_DEFINITIONS
from config import Settings, get_settings


@dataclass
class ToolTrace:
    """One tool invocation within an agent turn."""

    name: str
    arguments: dict[str, Any]
    result_preview: str
    tool_call_id: str


@dataclass
class AgentResult:
    """Final assistant reply plus audit trail of IB tool calls."""

    content: str
    iterations: int
    tool_traces: list[ToolTrace] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


def _merge_tool_call_delta(
    acc: dict[int, dict[str, Any]], tc_delta: Any
) -> None:
    """Accumulate streamed tool_call deltas by index."""
    idx = tc_delta.index if tc_delta.index is not None else 0
    slot = acc.setdefault(
        idx,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if tc_delta.id:
        slot["id"] = tc_delta.id
    if tc_delta.type:
        slot["type"] = tc_delta.type
    if tc_delta.function:
        if tc_delta.function.name:
            slot["function"]["name"] = (
                (slot["function"].get("name") or "") + tc_delta.function.name
            )
        if tc_delta.function.arguments:
            slot["function"]["arguments"] = (
                (slot["function"].get("arguments") or "")
                + tc_delta.function.arguments
            )


class TradingAgent:
    """
    Runs a multi-step OpenAI chat completion loop.

    Flow:
      1. Model may emit tool_calls (IB EClient-equivalent requests)
      2. We execute each tool; IB EWrapper data is collected into JSON
      3. Tool results are appended as role=tool messages
      4. Repeat until the model returns a final text answer (or max iterations)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)

    def _build_messages(
        self,
        user_message: str,
        *,
        symbol: str | None = None,
        history: list[dict[str, Any]] | None = None,
        extra_context: str | None = None,
        memory: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if memory:
            messages.append({"role": "system", "content": memory})

        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        context_bits = []
        if symbol:
            context_bits.append(f"Active UI symbol: {symbol.upper()}.")
        if extra_context:
            context_bits.append(extra_context)
        if context_bits:
            messages.append(
                {
                    "role": "system",
                    "content": "Session context:\n" + "\n".join(context_bits),
                }
            )

        messages.append({"role": "user", "content": user_message})
        return messages

    def run(
        self,
        user_message: str,
        *,
        symbol: str | None = None,
        history: list[dict[str, Any]] | None = None,
        extra_context: str | None = None,
        memory: str | None = None,
    ) -> AgentResult:
        content = ""
        iterations = 0
        traces: list[ToolTrace] = []
        finish_reason = ""
        for event in self.run_stream(
            user_message,
            symbol=symbol,
            history=history,
            extra_context=extra_context,
            memory=memory,
        ):
            et = event.get("type")
            if et == "delta":
                content += event.get("text") or ""
            elif et == "tool":
                traces.append(
                    ToolTrace(
                        name=event.get("name") or "",
                        arguments=event.get("arguments") or {},
                        result_preview=event.get("result_preview") or "",
                        tool_call_id=event.get("tool_call_id") or "",
                    )
                )
            elif et == "done":
                iterations = int(event.get("iterations") or 0)
                finish_reason = event.get("finish_reason") or ""
                content = event.get("content") or content
            elif et == "error":
                raise RuntimeError(event.get("error") or "agent error")

        return AgentResult(
            content=(content or "").strip() or "(empty response)",
            iterations=iterations or 1,
            tool_traces=traces,
            model=self.settings.openai_model,
            finish_reason=finish_reason,
        )

    def run_stream(
        self,
        user_message: str,
        *,
        symbol: str | None = None,
        history: list[dict[str, Any]] | None = None,
        extra_context: str | None = None,
        memory: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield SSE-friendly events: status | tool | delta | done | error.

        Each model pass is streamed. Content tokens are emitted as `delta`.
        Tool-call passes emit `status`/`tool` instead of answer text.
        """
        messages = self._build_messages(
            user_message,
            symbol=symbol,
            history=history,
            extra_context=extra_context,
            memory=memory,
        )
        traces: list[ToolTrace] = []
        finish_reason = ""
        max_iters = self.settings.agent_max_iterations
        model = self.settings.openai_model

        try:
            yield {"type": "status", "text": "thinking…"}

            for iteration in range(1, max_iters + 1):
                yield {
                    "type": "status",
                    "text": f"model pass {iteration}/{max_iters}…",
                }

                content_parts: list[str] = []
                tool_acc: dict[int, dict[str, Any]] = {}
                saw_tools = False

                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.2,
                    stream=True,
                )

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    if delta and delta.tool_calls:
                        saw_tools = True
                        for tc_delta in delta.tool_calls:
                            _merge_tool_call_delta(tool_acc, tc_delta)

                    if delta and delta.content:
                        # Only stream visible answer tokens when not building tools
                        if not saw_tools and not tool_acc:
                            content_parts.append(delta.content)
                            yield {"type": "delta", "text": delta.content}
                        else:
                            content_parts.append(delta.content)

                content_text = "".join(content_parts)
                tool_calls = [tool_acc[i] for i in sorted(tool_acc.keys())] if tool_acc else []

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_text or "",
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                if not tool_calls:
                    final = content_text.strip() or "(empty response)"
                    # If somehow no deltas were yielded (empty stream), push once
                    if not content_text:
                        yield {"type": "delta", "text": final}
                    yield {
                        "type": "done",
                        "content": final,
                        "iterations": iteration,
                        "model": model,
                        "finish_reason": finish_reason or "stop",
                        "tool_calls": [
                            {
                                "name": t.name,
                                "arguments": t.arguments,
                                "result_preview": t.result_preview,
                            }
                            for t in traces
                        ],
                    }
                    return

                for tc in tool_calls:
                    name = (tc.get("function") or {}).get("name") or ""
                    raw_args = (tc.get("function") or {}).get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}

                    yield {"type": "status", "text": f"tool: {name}…"}
                    result_str = execute_tool(name, raw_args)
                    preview = (
                        result_str if len(result_str) <= 500 else result_str[:500] + "…"
                    )
                    trace = ToolTrace(
                        name=name,
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        result_preview=preview,
                        tool_call_id=tc.get("id") or "",
                    )
                    traces.append(trace)
                    yield {
                        "type": "tool",
                        "name": name,
                        "arguments": trace.arguments,
                        "result_preview": preview,
                        "tool_call_id": trace.tool_call_id,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id") or "",
                            "content": result_str,
                        }
                    )

            # Iteration cap — stream a concluding answer without tools
            yield {"type": "status", "text": "max iterations — concluding…"}
            cap_messages = messages + [
                {
                    "role": "system",
                    "content": (
                        "You have reached the maximum tool-call iterations. "
                        "Provide your best thesis/decision using data gathered so far."
                    ),
                }
            ]
            streamed = ""
            stream = self.client.chat.completions.create(
                model=model,
                messages=cap_messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    streamed += piece
                    yield {"type": "delta", "text": piece}

            final = streamed.strip() or "(stopped: max iterations)"
            yield {
                "type": "done",
                "content": final,
                "iterations": max_iters,
                "model": model,
                "finish_reason": "max_iterations",
                "tool_calls": [
                    {
                        "name": t.name,
                        "arguments": t.arguments,
                        "result_preview": t.result_preview,
                    }
                    for t in traces
                ],
            }
        except Exception as exc:
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
