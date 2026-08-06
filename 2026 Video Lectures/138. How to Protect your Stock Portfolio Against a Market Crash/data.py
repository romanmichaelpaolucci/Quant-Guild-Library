import time
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Thread

# Script to fetch and save NVDA, CAT, and SPY daily prices for 2025 to CSV

class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        self.nvda_data = []
        self.cat_data = []
        self.spy_data = []
        self.done = False
        self.req_error = False
        self.mode = None  # "nvda", "cat", "spy"

    def error(self, reqId, errorCode, errorString):
        print(f"IBKR ERROR {reqId} {errorCode} {errorString}")
        if errorCode in [10314, 200]:
            self.done = True
            self.req_error = True

    def historicalData(self, reqId, bar):
        record = {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }
        if self.mode == "nvda":
            self.nvda_data.append(record)
        elif self.mode == "cat":
            self.cat_data.append(record)
        elif self.mode == "spy":
            self.spy_data.append(record)

    def historicalDataEnd(self, reqId, start, end):
        self.done = True

def run_loop(app):
    app.run()

def fetch_nvda_cat_spy_2025(
    nvda_filename="NVDA_2025.csv",
    cat_filename="CAT_2025.csv",
    spy_filename="SPY_2025.csv",
    durationStr="1 Y"
):
    app = IBKRApp()
    app.connect("127.0.0.1", 7497, 1)
    api_thread = Thread(target=run_loop, args=(app,))
    api_thread.start()
    time.sleep(1)

    # Contract for NVDA stock
    nvda_contract = Contract()
    nvda_contract.symbol = "NVDA"
    nvda_contract.secType = "STK"
    nvda_contract.exchange = "SMART"
    nvda_contract.currency = "USD"

    app.mode = "nvda"
    try:
        app.reqHistoricalData(
            reqId=1,
            contract=nvda_contract,
            endDateTime="20251231 23:59:59",
            durationStr=durationStr,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
    except Exception as e:
        print(f"Error requesting NVDA TRADES data:", e)
        app.done = True

    timeout = 90
    t0 = time.time()
    while not app.done and (time.time() - t0 < timeout):
        time.sleep(0.5)
    if not app.done:
        print("Timeout waiting for IBKR NVDA price data.")

    app.done = False  # reset for CAT

    # Contract for CAT stock
    cat_contract = Contract()
    cat_contract.symbol = "CAT"
    cat_contract.secType = "STK"
    cat_contract.exchange = "SMART"
    cat_contract.currency = "USD"

    app.mode = "cat"
    try:
        app.reqHistoricalData(
            reqId=2,
            contract=cat_contract,
            endDateTime="20251231 23:59:59",
            durationStr=durationStr,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
    except Exception as e:
        print(f"Error requesting CAT TRADES data:", e)
        app.done = True

    t1 = time.time()
    while not app.done and (time.time() - t1 < timeout):
        time.sleep(0.5)
    if not app.done:
        print("Timeout waiting for IBKR CAT price data.")

    app.done = False  # reset for SPY

    # Contract for SPY ETF
    spy_contract = Contract()
    spy_contract.symbol = "SPY"
    spy_contract.secType = "STK"
    spy_contract.exchange = "ARCA"  # SPY is on ARCA, can also use SMART
    spy_contract.currency = "USD"

    app.mode = "spy"
    try:
        app.reqHistoricalData(
            reqId=3,
            contract=spy_contract,
            endDateTime="20251231 23:59:59",
            durationStr=durationStr,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
    except Exception as e:
        print(f"Error requesting SPY TRADES data:", e)
        app.done = True

    t2 = time.time()
    while not app.done and (time.time() - t2 < timeout):
        time.sleep(0.5)
    if not app.done:
        print("Timeout waiting for IBKR SPY price data.")

    app.disconnect()
    time.sleep(1)

    nvda_df = pd.DataFrame(app.nvda_data)
    cat_df = pd.DataFrame(app.cat_data)
    spy_df = pd.DataFrame(app.spy_data)

    # Only keep bars within 2025
    if not nvda_df.empty:
        nvda_df["date"] = pd.to_datetime(nvda_df["date"].astype(str).str[:10])
        nvda_df = nvda_df[(nvda_df["date"] >= "2025-01-01") & (nvda_df["date"] <= "2025-12-31")]
        nvda_df.to_csv(nvda_filename, index=False)
        print(f"Saved NVDA daily prices (2025) as {nvda_filename}")
    else:
        print("No TRADES price data for NVDA.")

    if not cat_df.empty:
        cat_df["date"] = pd.to_datetime(cat_df["date"].astype(str).str[:10])
        cat_df = cat_df[(cat_df["date"] >= "2025-01-01") & (cat_df["date"] <= "2025-12-31")]
        cat_df.to_csv(cat_filename, index=False)
        print(f"Saved CAT daily prices (2025) as {cat_filename}")
    else:
        print("No TRADES price data for CAT.")

    if not spy_df.empty:
        spy_df["date"] = pd.to_datetime(spy_df["date"].astype(str).str[:10])
        spy_df = spy_df[(spy_df["date"] >= "2025-01-01") & (spy_df["date"] <= "2025-12-31")]
        spy_df.to_csv(spy_filename, index=False)
        print(f"Saved SPY daily prices (2025) as {spy_filename}")
    else:
        print("No TRADES price data for SPY.")

if __name__ == "__main__":
    fetch_nvda_cat_spy_2025()