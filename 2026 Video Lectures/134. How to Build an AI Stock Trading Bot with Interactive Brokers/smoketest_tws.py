"""Smoke test: verify TWS API on port 7497 can connect and request news."""

import asyncio

# Python 3.14+: create a loop before importing ib_insync/eventkit
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock

HOST = "127.0.0.1"
PORT = 7497  # TWS paper trading
CLIENT_ID = 97
SYMBOL = "AMD"


def main() -> None:
    ib = IB()

    print(f"1) Connecting to TWS at {HOST}:{PORT}...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID, readonly=True, timeout=10)
    if not ib.isConnected():
        raise RuntimeError("Connected flag is False after connect()")
    accounts = ib.managedAccounts()
    print(f"   OK — connected (accounts: {accounts})")

    print("\n2) Requesting news providers...")
    providers = ib.reqNewsProviders()
    if not providers:
        raise RuntimeError("No news providers returned")
    print(f"   OK — {len(providers)} providers")
    for p in providers:
        print(f"      {p.code}: {p.name}")

    codes = "+".join(p.code for p in providers)

    print(f"\n3) Requesting historical news for {SYMBOL}...")
    contract = Stock(SYMBOL, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"Could not qualify contract for {SYMBOL}")
    print(f"   OK — conId={contract.conId}")

    headlines = ib.reqHistoricalNews(contract.conId, codes, "", "", 5)
    if not headlines:
        raise RuntimeError("No historical news headlines returned")
    print(f"   OK — {len(headlines)} headlines")
    latest = headlines[0]
    print(f"   Latest: [{latest.providerCode}] {latest.headline}")

    print("\n4) Fetching full article for latest headline...")
    article = ib.reqNewsArticle(latest.providerCode, latest.articleId)
    body_preview = (article.articleText or "")[:200].replace("\n", " ")
    print(f"   OK — articleId={latest.articleId}")
    print(f"   Preview: {body_preview!r}...")

    ib.disconnect()
    print("\nSmoke test PASSED — TWS news API is working.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSmoke test FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
