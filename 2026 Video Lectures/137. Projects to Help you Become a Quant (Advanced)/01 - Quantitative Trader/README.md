# Algorithmic Trading System — Quant Guild Cockpit

**Role/Level:** Quant Trader — Advanced
**Stack:** Interactive Brokers Python API (`ibapi`) + Python `tkinter` (stdlib) + `numpy` / `pandas`.
**No web, no Flask, no other GUI toolkit** — this is a self-contained desktop trading cockpit.

A tkinter desktop application that connects to Interactive Brokers **TWS / IB Gateway**,
lets you build a *book of algorithmic strategies* (each with its own parameters, capital
allocation and ON/OFF switch), blends their signals into target positions, sizes orders
against your live account, and reads back live portfolio state. If TWS is not reachable it
transparently drops into a fully functional **DEMO mode** driven by a simulated account and
an internal random-walk price feed, so the whole thing demos end-to-end offline.

---

## 1. Quick start

```powershell
# From this folder:
pip install -r requirements.txt        # ibapi may already be installed
python app.py                          # launches the cockpit window
```

* If TWS / IB Gateway is running with the API enabled (see §4), the status pill turns
  green: **CONNECTED TO IBKR**.
* If not, the pill turns amber: **DEMO MODE (SIMULATED)** — everything still works.

> `ibapi` is optional at runtime. If it is missing or broken, the app logs the reason and
> runs in DEMO mode instead of crashing.

### Headless checks (no window)

```powershell
python strategies.py     # unit self-test of the strategy framework + sizing
python ib_client.py      # self-test of the DEMO backend + order simulation
python app.py --check    # import/wiring smoke test (does NOT open a window)
```

---

## 2. Architecture

```
+------------------ app.py (tkinter view) --------------------+
|  TradingCockpit                                             |
|   • Strategy book (ttk.Treeview) + add/edit/toggle form     |
|   • Portfolio readouts + positions table                    |
|   • Rebalance preview + guarded "Send Orders"               |
|   • two root.after() polling loops (events + snapshot)      |
+-----------------------------+-------------------------------+
                              | thread-safe facade
                              v
+-------------------- ib_client.py ---------------------------+
|  TradingBackend                                             |
|   • _IBApi(EWrapper, EClient)  → runs on a background thread |
|   • DEMO random-walk price feed → its own background thread  |
|   • locked state: prices, history, account, positions       |
|   • queue.Queue of events for the GUI                        |
+-----------------------------+-------------------------------+
                              | pure functions (no I/O)
                              v
+-------------------- strategies.py --------------------------+
|  Strategy (ABC) → TrendFollowing / MeanReversion / Momentum |
|  StrategyAllocation, aggregate_target_weights,              |
|  target_shares_from_weight, build_order_plans               |
+-------------------------------------------------------------+
```

**Separation of concerns.** `strategies.py` is pure Python/numpy with no I/O and is fully
unit-tested headlessly. `ib_client.py` owns all connectivity and threading and exposes one
thread-safe facade. `app.py` is a thin view that never touches `ibapi` or worker threads
directly.

### Threading model

* The `ibapi` **message loop** (`EClient.run()`) runs on a daemon background thread so it
  never blocks Tk.
* The **DEMO price feed** runs on its own daemon thread, ticking ~1 Hz.
* All shared state lives inside `TradingBackend` behind a single `threading.RLock`.
* Worker threads never call Tk. They push events onto a `queue.Queue` and update locked
  state; the GUI **polls** from the Tk main thread via `root.after(...)`:
  * `_drain_events()` every 200 ms → status/log/fill/error messages.
  * `_refresh()` every 1000 ms → repaints tables from an immutable `backend.snapshot()`.

### Robust `error()` callback

`_IBApi.error()` accepts `*args` and decodes **both** the old signature
`error(reqId, errorCode, errorString)` and the newer
`error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)`. Benign
connection notices (**2104, 2106, 2158**, and a few related codes) are logged as info and
ignored rather than surfaced as errors.

---

## 3. The strategy framework

Every strategy subclasses `Strategy` and implements
`generate_signal(prices) -> float` returning a **dimensionless directional signal** in
`[-1, +1]` (`+1` = max long, `0` = flat, `-1` = max short). Signals describe *conviction*,
not size.

| Strategy | Idea | Key parameters |
|---|---|---|
| **Trend Following** | Fast SMA vs slow SMA crossover; signal = `tanh(normalized spread · sensitivity)` | `fast`, `slow`, `scale` |
| **Mean Reversion** | Fade the z-score of price vs its rolling mean; signal = `clip(-z / entry_z, -1, 1)` | `lookback`, `entry_z` |
| **Momentum** | Sign & size of trailing return over `lookback`; signal = `tanh(return · sensitivity)` | `lookback`, `scale` |

New strategy types are added by subclassing `Strategy`, declaring `type_name` +
`param_spec`, and they automatically appear in the UI dropdown via `STRATEGY_REGISTRY`.

### How allocations + toggles combine into target positions

1. Each strategy row has a **percent allocation** and an **ON/OFF** switch.
2. For every **enabled** strategy, its contribution to its symbol's target weight is
   `signal × (allocation% / 100)`.
3. Contributions are **summed per symbol** across all enabled strategies (signal blending /
   capital budgeting), then **clamped to `[-1, +1]`** so no symbol exceeds 100% of net
   liquidation (pre-leverage). Disabled strategies contribute nothing.

```
target_weight[sym]  = clip( Σ signal_i · alloc_i%/100 , -1, +1 )   # enabled strategies on sym
```

### Order sizing

```
target_dollars[sym] = NetLiquidation × target_weight[sym]
target_shares[sym]  = round( target_dollars[sym] / price[sym] )
delta_shares        = target_shares − current_position          # the order to send
```

The system trades the **delta** toward target, so it self-corrects to the target book on
each rebalance. See `build_order_plans()`.

### Order placement (guarded)

* **DEMO mode:** orders are simulated instantly at the current price — cash, average cost,
  positions and realized/unrealized P/L all update in the simulated account.
* **IBKR mode:** a proper `ibapi` `Contract` (STK / SMART / USD) and `Order` (`MKT`) are
  built. Live transmission is **double-guarded**: you must tick **"Arm LIVE orders"** *and*
  confirm a dialog. With the guard on, the app performs a `[DRY-RUN]` and transmits nothing,
  so orders never fire unintentionally.

---

## 4. Connecting to TWS / IB Gateway (paper trading)

1. Launch **TWS** or **IB Gateway** and log in to your **paper** account.
2. `Configure → API → Settings`:
   * ✅ *Enable ActiveX and Socket Clients*
   * *Socket port*: **7497** (paper TWS default; live TWS is 7496; Gateway paper is 4002,
     live 4001)
   * ✅ *Allow connections from localhost only* (recommended)
   * Add `127.0.0.1` to *Trusted IPs* if prompted.
3. In the cockpit top bar set **Host** `127.0.0.1`, **Port** `7497`, a unique **Client Id**
   (e.g. `17`), then click **Reconnect**.
4. On success the status pill shows **CONNECTED TO IBKR** and the portfolio panel fills from
   `reqAccountSummary` / `reqPositions`, with live prices via `reqMktData` (delayed data is
   requested as a fallback when you lack a live market-data subscription).

If the handshake does not complete within the timeout (default 5 s), the app falls back to
DEMO mode automatically.

---

## 5. Using the cockpit

* **Add a strategy:** pick a *Type* (the parameter fields update automatically), enter a
  *Symbol*, an *Alloc %*, tweak parameters, then **＋ Add**.
* **Toggle ON/OFF:** select a row and click **Toggle ON/OFF**, or **double-click** the row.
  Disabled rows are greyed out and contribute nothing.
* **Edit allocation:** select a row → **Edit Allocation…**. The footer shows gross enabled
  allocation (turns red if it exceeds 100%).
* **Signal / Target columns** update live: `Signal` is the raw strategy signal, `Target %`
  is the blended, clamped target weight for that symbol.
* **Rebalance:** **↻ Compute Rebalance** shows the order plan (target vs current vs delta).
  **▶ Send Orders** executes it (simulated in DEMO; guarded live in IBKR).

---

## 6. Files

| File | Purpose |
|---|---|
| `app.py` | tkinter entry point / cockpit view. `python app.py` to run. |
| `ib_client.py` | `ibapi` EWrapper/EClient wrapper + thread-safe `TradingBackend` + DEMO fallback feed. |
| `strategies.py` | Strategy framework, signal blending, position sizing, order plans. |
| `requirements.txt` | `ibapi`, `numpy`, `pandas` (tkinter is stdlib). |
| `README.md` | This document. |

---

## 7. Disclaimer

Educational software. Signals and sizing are illustrative, not investment advice. Always
test against a **paper** account first. Live order transmission is intentionally guarded.
