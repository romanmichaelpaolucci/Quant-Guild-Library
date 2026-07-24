"""Smoke test: verify OpenAI API key can communicate."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (copy .env.example → .env)")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    print("1) Listing models...")
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f"   OK — received {len(model_ids)} models")
    print(f"   Sample: {model_ids[:5]}")

    print(f"\n2) Chat completion ({model})...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=10,
    )
    reply = response.choices[0].message.content
    print(f"   OK — reply: {reply!r}")
    print(f"   Usage: {response.usage}")

    print("\nSmoke test PASSED — OpenAI key is working.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSmoke test FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
