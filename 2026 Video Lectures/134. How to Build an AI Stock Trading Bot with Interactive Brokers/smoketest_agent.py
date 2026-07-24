"""Smoke test: LLM tool-calling loop against live TWS (news + snapshot)."""

from __future__ import annotations

import asyncio

# Python 3.14+: create a loop before importing ib_insync/eventkit
asyncio.set_event_loop(asyncio.new_event_loop())

from agent import TradingAgent
from ib_bridge import disconnect_ib


def main() -> None:
    print("Running TradingAgent with IB tools (requires TWS + OPENAI_API_KEY)...")
    agent = TradingAgent()
    result = agent.run(
        (
            "For AMD: pull a market snapshot and the 3 most recent news headlines. "
            "Summarize what you found in 5 bullets. Do not invent prices."
        ),
        symbol="AMD",
    )

    print(f"\nModel: {result.model}")
    print(f"Iterations: {result.iterations}")
    print(f"Finish: {result.finish_reason}")
    print(f"Tool calls ({len(result.tool_traces)}):")
    for t in result.tool_traces:
        print(f"  - {t.name}({t.arguments})")
    print("\n--- Assistant reply ---")
    print(result.content)
    print("\nSmoke test PASSED — agent tool loop completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSmoke test FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    finally:
        disconnect_ib()
