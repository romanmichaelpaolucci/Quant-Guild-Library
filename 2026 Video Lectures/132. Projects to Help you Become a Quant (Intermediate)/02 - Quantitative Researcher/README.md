# PDE Solver — Option Pricing by Finite Differences

**Quant Researcher · Intermediate**

An interactive research dashboard that demonstrates a foundational fact of
mathematical finance: **option prices solve partial differential equations
(PDEs)**, and those PDEs can be solved numerically with **finite-difference
(FD)** methods. To prove the FD solver is correct, every price is
cross-checked against an **independent Monte-Carlo (MC)** simulation and,
where one exists, a **closed-form** solution.

Pick a pricing framework, enter market and grid parameters, and watch three
completely different methods land on the same number.

![frameworks: vanilla · barrier · digital · american](https://img.shields.io/badge/frameworks-4-4fd1c5) ![scheme: Crank–Nicolson](https://img.shields.io/badge/scheme-Crank--Nicolson-a78bfa)

---

## 1. Why an option price is a PDE

Under the risk-neutral measure the underlying follows a geometric Brownian
motion (GBM):

```
dS = (r - q) S dt + sigma S dW.
```

Form a self-financing portfolio that is **long one option and short `Δ = ∂V/∂S`
units of stock**. Over `dt`, Itô's lemma gives the option's change as

```
dV = ( V_t + 0.5 sigma^2 S^2 V_SS ) dt + V_S dS.
```

Choosing `Δ = V_S` cancels the random `dW` term, so the hedged portfolio is
**riskless** and must therefore earn the risk-free rate `r`. Equating the two
expressions for the portfolio's growth and rearranging yields the
**Black-Scholes PDE**:

```
V_t + 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0.        (BS-PDE)
```

The magic is that *every* European-style derivative on this underlying obeys
the **same** PDE — only the **terminal condition** (the payoff at expiry) and
the **boundary conditions** change. That is exactly what makes a single FD
engine able to price a whole family of instruments.

### Terminal & boundary conditions by framework

| Framework | Terminal condition `V(S,T)` | Key boundary condition |
|---|---|---|
| **Vanilla** call/put | `max(S−K,0)` / `max(K−S,0)` | `V(0,t)=0`, `V(S_max,t)=S_max e^{−qτ}−K e^{−rτ}` (call) |
| **Barrier** knock-out | vanilla payoff, on the un-knocked domain | **Dirichlet `V(H,t)=0`** at the barrier `H` |
| **Digital** cash-or-nothing | `Q·1{S>K}` (call) / `Q·1{S<K}` (put) | `V(S_max,t)=Q e^{−rτ}` (call) |
| **American** call/put | vanilla payoff | free boundary: `V ≥ intrinsic` everywhere |

Here `τ = T − t` is time to maturity.

---

## 2. Finite differences

Substituting `τ = T − t` turns the backward PDE into a forward, heat-like
equation `V_τ = L[V]` where

```
L[V] = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V.
```

On a uniform grid `S_i = i·dS`, `τ_n = n·dτ`, central differences approximate
`L` at each interior node as a **tridiagonal** operator
`(L V)_i = α_i V_{i−1} + β_i V_i + γ_i V_{i+1}` with

```
α_i = 0.5 sigma^2 S_i^2 / dS^2 − 0.5 (r−q) S_i / dS
β_i = − sigma^2 S_i^2 / dS^2 − r
γ_i = 0.5 sigma^2 S_i^2 / dS^2 + 0.5 (r−q) S_i / dS.
```

### Explicit vs implicit vs Crank-Nicolson

Time-stepping uses the **θ-scheme**, blending explicit and implicit evaluations:

```
(I − θ dτ L) V^{n+1} = (I + (1−θ) dτ L) V^n.
```

| θ | Scheme | Stability | Accuracy | Notes |
|---|---|---|---|---|
| `0` | **Explicit** | *conditional*: needs `dτ ≤ dS² / (σ² S_max²)` | O(dτ)+O(dS²) | cheap per step but can blow up |
| `1` | **Implicit** | **unconditional** | O(dτ)+O(dS²) | one tridiagonal solve per step |
| `½` | **Crank-Nicolson** | **unconditional** | **O(dτ²)+O(dS²)** | the default; most accurate |

Each implicit step is solved in `O(M)` by the **Thomas algorithm** (a
specialised tridiagonal Gaussian elimination) in `fd_solver.py`.

The dashboard reports a **stability flag** for the explicit scheme so you can
see it break the CFL condition and watch the price diverge — a great teaching
moment.

### Barrier PDEs

A knock-out barrier option is priced on a **truncated domain** with a
**Dirichlet boundary of zero at the barrier**: the instant `S` touches `H`,
the option is worthless. For an *up-and-out* we solve on `[0, H]` with
`V(H,t)=0`; for a *down-and-out* we solve on `[H, S_max]` with `V(H,t)=0`.
This is the entire modelling change — same PDE, different boundary.

### American options (free boundary)

An American option can be exercised early, so its value can never fall below
its intrinsic payoff. This turns the PDE into a **linear complementarity
problem (LCP)**:

```
min( −(V_t + L[V]),  V − payoff ) = 0.
```

We solve it with **Projected SOR (PSOR)**: an iterative sweep that, after each
implicit update, *projects* the value back up to the payoff wherever early
exercise is optimal. The MC counterpart is **Longstaff-Schwartz** least-squares
Monte-Carlo.

---

## 3. Why Monte-Carlo must agree

The **Feynman-Kac theorem** says the solution of the BS-PDE equals a discounted
risk-neutral expectation:

```
V(S_0, 0) = e^{−rT} · E^Q[ payoff(S_T, path) ].
```

FD attacks the **PDE (left) side**; MC estimates the **expectation (right)
side** by simulating GBM paths:

```
S_{t+dt} = S_t · exp[ (r − q − σ²/2) dt + σ √dt · Z ],   Z ~ N(0,1).
```

Two entirely different numerical philosophies must therefore converge to the
same price. The dashboard shows the **MC 95% confidence interval** and flags
whether the FD price falls **inside** it. If it does, both engines corroborate
each other; if not, something is wrong. Path-dependent barriers are simulated
with fine time steps and discrete barrier monitoring; American options use
Longstaff-Schwartz.

---

## 4. Project layout

```
02 - Intermediate - PDE Solver/
├── app.py            # Flask server (port 5003): wires FD + MC + analytics to the UI
├── fd_solver.py      # θ-scheme FD engine (explicit/implicit/Crank-Nicolson), Thomas, PSOR
├── mc_pricer.py      # risk-neutral Monte-Carlo (+ barrier monitoring, Longstaff-Schwartz)
├── analytics.py      # closed forms: BS, cash-or-nothing, Reiner-Rubinstein barriers
├── smoke_test.py     # headless correctness checks (FD ≈ MC ≈ analytic)
├── requirements.txt
├── templates/index.html
└── static/css/style.css, static/js/app.js
```

Each module has a rich docstring explaining the mathematics it implements and
can be run standalone (`python fd_solver.py`, etc.) for a quick self-check.

---

## 5. Running it

From this folder (Windows PowerShell shown; use the analogous commands on
macOS/Linux):

```powershell
# 1. (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. run the headless smoke test (correctness checks, no server)
python smoke_test.py

# 4. launch the dashboard
python app.py
```

Then open **http://127.0.0.1:5003** in your browser.

### Using the dashboard

1. Choose a **framework** (Vanilla / Barrier / Digital / American).
2. Set market & contract inputs (`S₀, K, r, q, σ, T`, plus barrier or cash
   payout where relevant).
3. Tune the **FD grid** (scheme, `M` spatial steps, `N` time steps) and the
   **MC** settings (paths, steps).
4. Click **Solve PDE**. You'll see:
   - the governing PDE rendered in LaTeX with its terminal/boundary conditions,
   - **FD vs MC vs analytic** price cards, with the MC 95% CI and an
     in/out-of-CI badge,
   - the FD **price surface `V(S,t)`** as a 3D surface or heatmap,
   - the **value-vs-spot slice at `t=0`** with the payoff overlaid and `S₀`
     marked,
   - a **bar chart** comparing the three methods (MC shown with error bars).

---

## 6. Things to try (teaching prompts)

- **Convergence:** raise `M` and `N` and watch the FD price march toward the
  closed form.
- **Break the explicit scheme:** select *Explicit* with a coarse `N` — the
  stability flag turns red and the price diverges. Increase `N` to restore it.
- **Barrier ≤ vanilla:** a knock-out is always **cheaper** than the vanilla
  (it can expire worthless from a knock-out). Move `H` toward the spot and the
  price collapses toward zero.
- **Digital sanity:** a cash-or-nothing call with `Q=1` prices the discounted
  risk-neutral probability of finishing ITM, `e^{−rT} N(d₂)`.
- **Early exercise premium:** compare the **American put** FD price against the
  European put — the difference is the early-exercise premium (no closed form
  exists, so MC via Longstaff-Schwartz is the independent check).

---

## 7. Notes & assumptions

- Continuous dividend yield `q` supported (default 0). The American-call
  closed form (`= European call`) is only valid when `q = 0`.
- Barrier MC uses **discrete** monitoring; it converges to the
  continuously-monitored analytic price as the number of steps grows (expect a
  small positive bias for coarse monitoring).
- Grid/MC sizes are capped server-side to keep the dashboard responsive.
- This is an **educational** project: clarity and correctness are prioritised
  over raw performance.
