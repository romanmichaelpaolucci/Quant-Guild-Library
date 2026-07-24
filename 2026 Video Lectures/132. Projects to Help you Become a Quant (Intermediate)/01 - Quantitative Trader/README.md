# Market Making Simulator

**Quant Trader track · Intermediate project**

An interactive Flask web app that shows what happens when you make an options
market **without knowing the true dynamics of the underlying**. You choose the
model the market maker (MM) *prices* with, and — separately — the model that
*actually* drives the underlying. Every tick the MM quotes a bid/ask around its
model fair value on a **fresh, rolling X-DTE option** (constant maturity `T`),
gets filled by random client flow, and **realizes each trade's P/L on the
spot**: the moment a contract trades we simulate its delta-hedged life to expiry
under the *true* model and bank the discounted result. Two live-ticking charts
let you watch the spread edge get captured — or bled away by **model risk**.

> **Why realize at the trade?** We are NOT aging one option toward 0DTE. The desk
> quotes today's X-DTE option, tomorrow's brand-new X-DTE option, and so on. A
> continuously delta-hedged option's P/L is independent of drift and equals
> `premium − fair`, so selling at `ask = fair + ½-spread` (and buying at
> `bid = fair − ½-spread`) banks the half-spread *on average*. Buying low and
> selling high around fair value is exactly what realizes positive expectancy.

---

## The lesson in one sentence

> When your pricing model matches reality, you collect the bid/ask spread with
> controlled variance; when it's wrong (e.g. you price Black-Scholes but the
> world has jumps), you can lose money even though every quote looked
> "positive expected value."

---

## The finance & math

### 1. Market making, bid/ask, and spread capture
A market maker continuously posts a two-sided quote on the option:

```
bid = fair_value − half_spread        (price at which the MM will BUY)
ask = fair_value + half_spread         (price at which the MM will SELL)
```

Clients who want to buy **lift the ask** (MM sells the option, receives `+ask`);
clients who want to sell **hit the bid** (MM buys the option, pays `−bid`). Each
such fill is a fresh X-DTE contract, and we realize its P/L immediately (see
below). Buying below fair and selling above fair is the MM's **edge** — its
compensation for providing liquidity and warehousing risk.

### 2. Realizing each trade + expected vs realized P/L
The instant a contract trades, we simulate that option's **delta-hedged life to
expiry under the true model** and bank the discounted P/L. Because a hedged
option's P/L equals `premium − fair` regardless of drift, each fill realizes the
half-spread *in expectation*, plus stochastic hedging/jump noise. The app tracks
three P/L curves so you can see edge and model risk separately:

| Curve | Meaning |
|-------|---------|
| **Realized (equity)** | The actual, stochastic cash realized as each trade's hedged life plays out under the true model. This is the equity curve. |
| **Expected (model)** | Cumulative edge the MM *believes* it banks, measured against its own fair value: `sign × (premium − V_model)` ≈ half-spread per contract. |
| **Expected (truth)** | Cumulative edge measured against the **true** fair value: `sign × (premium − V_truth)` — the genuine expected edge. |

When your model is right, expected-model ≈ expected-truth and the realized curve
wobbles upward around them. When it's wrong, the gap between expected-model and
expected-truth **is** your mispricing, and the realized curve additionally takes
fat-tailed jump losses — model risk expressed in dollars.

### 3. Delta hedging
An option's value moves with the underlying. When we realize a trade, the MM
neutralises that first-order risk by holding `−position × delta` units of the
underlying and rebalancing `hedge_steps` times over the option's life, so
diffusion is (largely) hedged away and what remains is the priced edge.
Crucially, the MM can only hedge the delta **it can compute from its own
model**. If the model is wrong, the hedge is wrong, and moves the model says are
"impossible" (jumps) blow straight through it — which is exactly how the
matched-vs-mismatched experiment diverges.

### 4. GBM vs Merton jump-diffusion
- **Geometric Brownian Motion (GBM)** — the Black-Scholes world. Log-returns are
  normal and continuous:  `dS = μ·S·dt + σ·S·dW`. No surprises, so a
  continuously-rebalanced delta hedge is (in theory) perfect.
- **Merton Jump-Diffusion** — GBM **plus** sudden jumps arriving as a Poisson
  process (intensity `λ` per year); each jump multiplies the price by `exp(Y)`
  with `Y ~ N(jump_mean, jump_std²)`. Returns have **fat tails and skew**.
  Merton (1976) showed the option price is a Poisson-weighted sum of
  Black-Scholes prices, which `pricing.py` implements as a truncated series.

The interesting experiment: **price with Black-Scholes while the true path is
Merton with frequent negative jumps.** Your expected-model line still climbs by
the full half-spread on every fill, so you *think* you are banking the spread.
But your expected-truth line sags (you are quoting too cheap for the real tail
risk), and — worse — your realized equity gets gapped by jumps your delta hedge
never saw coming. Across seeds the average edge per contract collapses and
drawdowns explode: *picking up pennies in front of a steamroller.*

### 5. How the animation demonstrates it
Both charts stream **one tick at a time** (via Plotly `extendTraces` on a
timer), so you *feel* the market ticking:

- **Underlying & quotes** — the true price path (left axis) with the MM's live
  option bid/ask band on a twin right axis. You can literally watch the spread
  the MM is quoting.
- **Cumulative P/L** — the realized equity curve (each trade banked at fill)
  streaming tick by tick, overlaid with the smooth **expected-model** and
  **expected-truth** edge lines so you can see the realized curve wobble upward
  around the truth line — and, when mispricing, diverge below what the MM thinks.

Use the **"Preset: matched"** and **"Preset: steamroller"** buttons to jump
straight to the two canonical scenarios.

### 6. Return Distributions tab
A second tab simulates **both GBM and Merton** under the same drift/vol from the
control panel (thousands of vectorized paths) and overlays their **terminal
return histograms** alongside a few sample paths. Merton's Poisson jumps produce
visibly **fatter tails, more skew, and higher excess kurtosis** than GBM. Because
an option payoff is convex, that heavier tail raises its expected value — which
is *why* the Merton price sits above the Black-Scholes price for the same σ. The
moments table quantifies the difference (volatility, skew, kurtosis, 1% VaR).

---

## Project structure

```
02 - Intermediate - Market Making Simulator/
├── app.py             # Flask server (port 5001): /api/simulate + /api/distributions
├── simulator.py       # Tick-by-tick MM engine + GBM/Merton distribution sim
├── pricing.py         # Black-Scholes & Merton pricers + greeks
├── requirements.txt
├── README.md
├── templates/
│   └── index.html     # Single-page trading-desk UI
└── static/
    ├── style.css      # Dark theme
    └── app.js         # Plotly live-tick animation + controls
```

## Controls
- **Models**: MM pricing model (Black-Scholes / Merton) and the true market-path
  model (GBM / Merton) — set them equal or different.
- **Contract**: spot `S₀`, strike `K`, maturity `T`, call/put.
- **True dynamics**: drift `μ`, vol `σ`, rate `r`, jump intensity `λ`, jump
  mean/std.
- **Market making**: half-spread, quote size, fill intensity, and (optionally)
  the vol the MM plugs into their pricer. *(Risk aversion is retained for
  compatibility but has no effect — every fill is realized on the spot, so the
  desk never warehouses inventory to skew against.)*
- **Simulation**: number of ticks, **hedge rebalances** per option (more
  rebalances ⇒ tighter hedge ⇒ smaller realized variance), RNG **seed**
  (blank = random), animation speed.

## Reproducibility
Set a **seed** (default `42`) for a deterministic path and P/L. Leave it blank
for a fresh random run each time. The whole simulation is computed once
server-side from the seed; the front-end only replays it.

---

## How to run (Windows / PowerShell)

```powershell
cd "C:\Users\Roman\Desktop\Quant-Guild-Library\2026 Video Lectures\130. Projects to Help you Become a Quant (by Role and Level)\01 - Quant Trader\02 - Intermediate - Market Making Simulator"

pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5001> in your browser, tweak the controls, and hit
**▶ Run & Animate**.

### Quick headless checks
Run the engine without the browser to sanity-check P/L arrays:

```powershell
python simulator.py     # prints matched vs mismatched run summaries
python pricing.py       # prints put-call parity + Merton >= BS checks
```

---

## Suggested experiments
1. **Matched baseline** — Black-Scholes pricing + GBM path. Run several seeds:
   the average realized edge per contract should sit right on the quoted
   half-spread and realized equity climbs steadily with small variance.
2. **Steamroller** — Black-Scholes pricing + Merton path, `λ=4`,
   `jump_mean=−0.12`. The expected-model line still climbs, but realized equity
   is battered by jumps: average edge/contract collapses and drawdowns blow out.
3. **Right model, jumpy world** — Merton pricing + Merton path. The MM now
   *knows* about jumps: quotes and hedges account for them, expected-model ≈
   expected-truth, and realized edge is restored toward the half-spread.
4. **Over-cautious** — Merton pricing + GBM path. The MM prices jumps that never
   come; quotes are effectively too defensive and give up some edge.
5. **Hedge frequency** — lower **hedge rebalances** and watch the realized curve
   get noisier around the expected lines; raise it to tighten variance.
6. **Spread vs risk** — widen the half-spread and see how much cushion you need
   to survive the mismatched case.

> Educational simulator — simplified fills, single option, discrete hedging. Not
> trading or investment advice.
