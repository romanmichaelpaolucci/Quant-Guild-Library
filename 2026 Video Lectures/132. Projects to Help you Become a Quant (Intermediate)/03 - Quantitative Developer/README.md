# Order Book Simulator

**Quant Developer · Intermediate**

A from-scratch **limit order book** with a **price-time-priority matching
engine**, visualized live in a `tkinter` desktop app. It is a hands-on tour of
*market microstructure* (how exchanges actually match orders) and the *data
structures & algorithms* that make matching fast.

```
┌────────────────────────────────────────────────────────────────────────┐
│ ORDER BOOK                         BID 99.98  ASK 100.02  SPREAD 0.04 …  │
├───────────────┬──────────────────────────────────┬──────────────────────┤
│ DEPTH OF MKT  │        CUMULATIVE DEPTH           │     TIME & SALES     │
│  Bid   Px  Ask│      (green bids | red asks)      │  09:31:02 100.02  4 ▲│
│         100.04│         staircase chart           │  09:31:02  99.98  9 ▼│
│         100.02│                                   │        …             │
│  ─ 0.04 ─     │                                   ├──────────────────────┤
│  12    99.98  │                                   │  MANUAL ORDER ENTRY  │
│  …            │                                   │  SIMULATION CONTROLS │
└───────────────┴──────────────────────────────────┴──────────────────────┘
```

---

## Run it

```powershell
# from this folder
pip install -r requirements.txt
python app.py
```

Requirements are minimal: `sortedcontainers` for the sorted price levels, and
the standard-library `tkinter` for the GUI (bundled with CPython on
Windows/macOS; `sudo apt install python3-tk` on Debian/Ubuntu). `numpy` is
**optional** and not used by the engine, simulator, or GUI.

Run the tests (no pytest required):

```powershell
python test_order_book.py      # built-in runner
# or, if you have pytest:
pytest -q
```

---

## What's in the box

| File | Role | tkinter? |
|------|------|:--------:|
| `order_book.py` | The matching engine — pure, testable, no I/O. | no |
| `market_sim.py` | Stochastic order-flow generator that keeps the book alive. | no |
| `app.py`        | The live GUI (ladder, depth chart, tape, order entry, controls). | **yes** |
| `test_order_book.py` | Headless unit tests of the matching semantics. | no |
| `requirements.txt` | Dependencies. | — |

The separation is deliberate: **all matching logic lives in `order_book.py`**
with zero GUI or randomness, so it can be unit-tested and reused. `app.py` is a
thin view layer.

---

## Crash course: the limit order book

An exchange keeps, per instrument, two sorted collections of resting orders:

- **Bids** — orders to *buy*, sorted so the **highest** price is most
  aggressive. The top is the **best bid**.
- **Asks** (offers) — orders to *sell*, sorted so the **lowest** price is most
  aggressive. The top is the **best ask**.

Derived quantities the UI tracks live:

- **Spread** = best ask − best bid (the cost of immediacy).
- **Mid** = (best ask + best bid) / 2 (a fair-price proxy).
- **Depth** = resting size available at each price level.

### Order types

- **Limit order** — has a price. The part that can trade *immediately* does so;
  the rest **rests** in the book, providing liquidity (you become a *maker*).
- **Market order** — no price; it **takes** whatever liquidity exists on the far
  side until filled. Any unfilled remainder is dropped (market orders never
  rest).

An order that would trade immediately — a buy priced ≥ best ask, a sell priced
≤ best bid, or any market order — is **marketable** and crosses the spread. A
limit priced *away from* the market on the passive side (a buy below it, a sell
above it) is **not** marketable: it simply **rests and sits there** until the
market comes to it.

### Price band (fat-finger protection)

A manual limit priced far *through* the market — a buy well above it or a sell
well below it — is marketable, so it would sweep the whole book and leave its
remainder resting at that absurd price (a phantom level far from the market).
Real venues reject or **collar** such orders, so the manual entry panel warns
you and, if you confirm, collars the order to a **10% band** around the mid: it
fills up to the band edge and any remainder rests near the market, never at the
typed price. Passive far-away quotes are untouched — they just rest.

### Partial fills

A single order rarely meets an exactly-equal counterparty. A large taker eats
several resting orders (and several price levels); a resting maker gets nibbled
by many small takers. The engine tracks each order's `remaining` quantity, so
partial fills leave the correct residual — which for a limit order rests, and
for a market order is discarded.

---

## The matching rule: **price-time priority**

When a marketable (aggressive / *taker*) order arrives, it is matched against
resting (*maker*) orders in a strict order:

1. **Price priority.** The best price is filled first. A buyer sweeps the
   cheapest asks; a seller hits the highest bids. This is what makes markets
   competitive — quote a better price and you jump the queue.
2. **Time priority.** *Within* a single price level, orders are filled
   **first-in, first-out**. Resting earlier at a given price earns you a place
   at the front of the queue. This is why "being early" has value even without
   a better price.

Trades **print at the resting maker's price**, so a taker can receive *price
improvement*: a buy limit at 100.10 that hits a resting ask at 99.90 trades at
**99.90**, not 100.10.

> Why price-time priority? It is simple, deterministic, and *fair*: it rewards
> the two things an exchange wants to encourage — **better prices** and
> **earlier liquidity** — and it can't be gamed by order splitting the way a
> pro-rata scheme can.

Every fill **conserves shares**: the quantity bought equals the quantity sold.
The book never creates or destroys inventory, and it must never end up
*crossed* (best bid ≥ best ask). Both invariants are asserted in the tests.

---

## Data structures & complexity

Let **L** = number of distinct price levels currently in the book.

### Price levels — one `SortedDict` per side

Each side (bids/asks) is a
[`sortedcontainers.SortedDict`](https://grantjenks.com/docs/sortedcontainers/)
keyed by an **integer number of ticks** (prices are quantised to `tick_size` so
dictionary keys are exact — no floating-point drift).

- Best bid / best ask = the largest / smallest key → **O(log L)**.
- Insert a brand-new price level → **O(log L)**.

### Within a level — an `OrderedDict` FIFO queue

Each price level stores its resting orders in a `collections.OrderedDict` keyed
by order id. **Insertion order *is* time priority**, so:

- The FIFO front (oldest order) is `next(iter(...))` → **O(1)**.
- Cancel a specific order by id → **O(1)** within the level (dict delete).

A global `order_id → Order` index makes lookup/cancel **O(1)** to find the
order, plus **O(log L)** to reach its level.

### Summary

| Operation | Cost | Why |
|-----------|------|-----|
| Best bid / ask | `O(log L)` | `SortedDict` peek at an end key. |
| Add a resting limit order | `O(log L)` | Find/create the level, append to its FIFO queue (O(1)). |
| Cancel by id | `O(log L)` | O(1) index + O(1) `OrderedDict` delete; O(log L) if the level empties. |
| Match a marketable order | `O(k + m·log L)` | `k` = resting orders touched; `m` = price levels fully consumed and removed. |

> **Why not just sort a list on every insert?** That would be `O(L)` (or
> `O(L log L)`) per order — fine for a demo, hopeless for a real feed doing
> hundreds of thousands of messages a second. A balanced sorted structure plus
> per-level FIFO queues is the standard production shape, and it's exactly what
> this engine implements.

Alternatives you could swap in (and their trade-offs):

- **Binary heaps** per side give `O(log L)` best-price and insert, but make
  *cancellation* and *depth iteration* awkward (heaps aren't ordered for
  traversal, and arbitrary deletion is `O(L)` without extra bookkeeping).
- **Plain `dict` of price→level + tracked best price** gives `O(1)` average
  add/cancel but `O(L)` to find the *next* best price after a level empties.

`SortedDict` gives the best all-round ergonomics for an educational engine:
fast best-price, cheap next-level, easy ordered depth traversal for the ladder.

---

## The simulator (`market_sim.py`)

To make the book *alive*, a toy agent model injects order flow each tick:

- A latent **fair value** follows a random walk (`fv += drift + N(0, σ)`), so
  the mid wanders.
- **Limit orders** are posted around the fair value with an exponential price
  offset (most near the touch, a few deep) — this builds depth.
- **Market orders** cross the spread and generate trades / partial fills.
- **Cancellations** pull random resting orders, mimicking quote churn.

Each flow's intensity is a Poisson rate you can tune (`limit_rate`,
`market_rate`, `cancel_rate`, `aggressiveness`, …). It uses only the standard
library `random` (seedable for reproducible tests).

This is intentionally *not* a calibrated market model — it exists to exercise
the engine and produce a realistic-looking, continuously churning book.

---

## Using the GUI

**Layout**

- **Depth of Market ladder** — asks (red) above, bids (green) below, with the
  best bid/ask highlighted and the spread on the divider line. Your own resting
  orders are tinted blue.
- **Cumulative depth chart** — the classic order-book *staircase*: cumulative
  bid size building left from the mid (green), cumulative ask size building
  right (red). Fatter, closer walls = more liquidity.
- **Time & Sales tape** — recent executions (time, price, size, aggressor).
  Green ▲ = a buyer lifted the offer; red ▼ = a seller hit the bid.

**Controls**

| Control | Effect |
|---------|--------|
| **Start / Pause Flow** | Toggle the simulated order flow. |
| **Speed** | Simulation ticks per second (drives the `root.after` loop). |
| **Aggressiveness** | Scales market-order rate & size (more crossing = more trades). |
| **Reset** | Clear the book and restart the simulator. |
| **Manual Order Entry** | Submit your own Buy/Sell × Limit/Market order. |
| **Your resting orders → Cancel Selected** | Cancel a limit order you posted. |
| **How matching works** | In-app popup of the rules and big-O. |

**Try this**

1. Start Flow and watch the ladder and depth staircase churn.
2. Post a **limit** buy *inside* the book (e.g. a couple of ticks below the
   ask) and watch it rest in blue, then get filled as the market drifts down to
   it — appearing on the tape.
3. Send a **market** buy and watch it *walk the book*: it prints against
   successive ask levels at increasing prices (that's slippage), and the best
   ask jumps up.
4. Post a limit **far away on the passive side** (a buy well *below* the market)
   and watch it just **sit there** and rest — it never trades until price falls
   to it.
5. Now fat-finger a buy limit **far above** the market (e.g. `100000`): the
   price-band guard warns you and collars it to a 10% band instead of letting it
   sweep the whole book and park at an absurd price.

> The real-time loop uses `root.after` (never `time.sleep` on the main thread),
> so the UI stays responsive. `app.py` is import-safe: creating the window
> happens only inside `main()` under the `__main__` guard.

---

## Tests

`test_order_book.py` covers the semantics an exchange must get right:

- market order fills the **best** resting price for the **right** size;
- market order **walks multiple levels** and reports any unfilled shortfall;
- **time priority** (FIFO) fills the oldest order at a level first;
- **price priority** hits the best price first;
- a **marketable limit** crosses and then **rests its residual**;
- a taker gets **price improvement** at the maker's price;
- **partial fills** leave the correct residual;
- **cancel** removes the right order and empties the level cleanly;
- best bid/ask/**spread update** as levels are consumed;
- **conservation of shares** and a **never-crossed book** hold under 1000s of
  random simulated steps.

```
19/19 tests passed.
```

---

Built for the [Quant Guild](https://quantguild.com) by Roman Paolucci.
