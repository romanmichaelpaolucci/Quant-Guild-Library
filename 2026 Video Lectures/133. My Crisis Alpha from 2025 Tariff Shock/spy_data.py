"""
Fetch daily OHLCV from IBKR for SPY from the start of 2025 through today and save to CSV.

Requires TWS or IB Gateway running on localhost with API enabled (default TWS port 7497).

Since IBKR limits a single historical request to about 1 year of daily bars, 
we request data chunk-by-chunk for each year, ending on Dec 31 (or today for the latest year).
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from threading import Thread

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

SYMBOL = "SPY"
START_YEAR = 2025

HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 114
CHUNK_DURATION_STR = "1 Y"
REQUEST_TIMEOUT_S = 120
PACING_SLEEP_S = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "spy_prices_2025.csv"


def _parse_bar_date(s: str) -> pd.Timestamp:
    if isinstance(s, str) and len(s) == 8 and s.isdigit():
        return pd.Timestamp(s)
    return pd.to_datetime(s)


def run_loop(app: EClient) -> None:
    app.run()


def stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


class IBKRHistoricalApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.data: list[dict] = []
        self.finished = False
        self._errors: list[tuple[int, str]] = []

    def historicalData(self, reqId, bar) -> None:
        self.data.append(
            {
                "date": _parse_bar_date(bar.date),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

    def historicalDataEnd(self, reqId, start, end) -> None:
        self.finished = True

    def error(self, reqId, errorCode, errorString, advancedOrderReject="") -> None:
        if errorCode in (2104, 2106, 2158, 2119):
            return
        self._errors.append((errorCode, errorString))
        if reqId >= 0:
            self.finished = True


def fetch_daily_bars(
    app: IBKRHistoricalApp,
    req_id: int,
    symbol: str,
    end_date_time: str = "",
    duration_str: str = CHUNK_DURATION_STR,
) -> pd.DataFrame:
    app.data = []
    app.finished = False
    app._errors.clear()

    app.reqHistoricalData(
        reqId=req_id,
        contract=stock_contract(symbol),
        endDateTime=end_date_time,
        durationStr=duration_str,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    deadline = time.time() + REQUEST_TIMEOUT_S
    while not app.finished and time.time() < deadline:
        time.sleep(0.25)

    if not app.finished:
        try:
            app.cancelHistoricalData(req_id)
        except Exception:
            pass
        return pd.DataFrame()

    if app._errors and not app.data:
        return pd.DataFrame()

    df = pd.DataFrame(app.data)
    if df.empty:
        return df

    return df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def fetch_symbol_history(
    app: IBKRHistoricalApp,
    symbol: str,
    start_year: int,
) -> pd.DataFrame:
    """Fetch daily bars from start_year-01-01 to today by 1-year chunks."""
    today = date.today()
    end_dates: list[str] = []

    # Most recent chunk ends "now" (empty string == latest available bar).
    end_dates.append("")

    # Then walk back year-by-year using Dec 31 anchors.
    for year in range(today.year - 1, start_year - 1, -1):
        end_dates.append(f"{year}1231 23:59:59 US/Eastern")

    frames: list[pd.DataFrame] = []
    for i, end_dt in enumerate(end_dates, start=1):
        label = end_dt or f"{today.isoformat()} (latest)"
        print(f"  [{i}/{len(end_dates)}] {symbol} chunk ending {label}", flush=True)
        df = fetch_daily_bars(
            app,
            req_id=i,
            symbol=symbol,
            end_date_time=end_dt,
            duration_str=CHUNK_DURATION_STR,
        )
        if df.empty:
            err = app._errors[-1] if app._errors else "no bars returned"
            print(f"    no data for chunk: {err}", flush=True)
        else:
            print(
                f"    {len(df)} rows "
                f"({df['date'].min().date()} to {df['date'].max().date()})",
                flush=True,
            )
            frames.append(df)
        time.sleep(PACING_SLEEP_S)

    if not frames:
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    cutoff = pd.Timestamp(year=start_year, month=1, day=1)
    combined = combined[combined["date"] >= cutoff].reset_index(drop=True)
    return combined


def fetch_and_save(
    host: str = HOST,
    port: int = PORT,
    client_id: int = CLIENT_ID,
    symbol: str = SYMBOL,
    start_year: int = START_YEAR,
    output_path: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    app = IBKRHistoricalApp()
    app.connect(host, port, client_id)
    thread = Thread(target=run_loop, args=(app,), daemon=True)
    thread.start()
    time.sleep(1)

    try:
        print(f"Fetching {symbol} daily bars from {start_year} to today…", flush=True)
        df = fetch_symbol_history(app, symbol=symbol, start_year=start_year)
    finally:
        app.disconnect()

    if df.empty:
        print("No data retrieved; check TWS/Gateway and market data subscriptions.")
        return df

    out = df.rename(columns={"date": "Date"})
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {len(out)} rows "
        f"({out['Date'].min().date()} to {out['Date'].max().date()}) "
        f"-> {output_path}"
    )
    return out


if __name__ == "__main__":
    fetch_and_save()
