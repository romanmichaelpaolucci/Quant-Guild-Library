"""
app.py
======

Flask dashboard for the **Pricing Library** (Quant Researcher / Advanced).

Three sections, each pairing a rendered-LaTeX explanation with interactive
Plotly charts:

  1. Rough Volatility  -- simulate rough Bergomi; view rough vol paths, the
     induced price paths, the implied-vol smile, and the exploding
     short-maturity ATM skew.
  2. Markovian Lifting -- approximate the fractional kernel by a sum of
     exponentials (OU factors); watch the approximation converge as N grows.
  3. Path Signatures   -- truncated signatures as features; a signature-based
     linear regression prices an Asian option in a model-free spirit.

Run
---
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5004

The heavy lifting lives in ``rough_vol.py``, ``markovian_lifting.py`` and
``signatures.py``; this file only wires computation to charts.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template_string, request

import markovian_lifting as ml
import rough_vol as rv
import signatures as sg

app = Flask(__name__)

# --- shared "frontier research" chart theme --------------------------------
_PAPER = "#0f1420"
_PANEL = "#151b2b"
_GRID = "#2a3350"
_FONT = "#c9d4ee"
_ACCENT = ["#5eead4", "#a78bfa", "#f472b6", "#fbbf24", "#60a5fa", "#f87171"]


def _layout(title: str, xaxis: str, yaxis: str, **kwargs) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color=_FONT)),
        paper_bgcolor=_PAPER, plot_bgcolor=_PANEL,
        font=dict(color=_FONT, family="Inter, system-ui, sans-serif", size=12),
        xaxis=dict(title=xaxis, gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(title=yaxis, gridcolor=_GRID, zerolinecolor=_GRID),
        margin=dict(l=60, r=25, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **kwargs,
    )


def _fig(fig: go.Figure) -> dict:
    """Serialise a Plotly figure to a plain dict for the front end."""
    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# Rough volatility
# ---------------------------------------------------------------------------

@app.route("/api/rough_vol", methods=["POST"])
def api_rough_vol():
    p = request.get_json(force=True)
    H = float(p.get("H", 0.1))
    eta = float(p.get("eta", 1.5))
    rho = float(p.get("rho", -0.7))
    xi0 = float(p.get("xi0", 0.04))

    model = rv.RoughBergomi(H=H, eta=eta, rho=rho, xi0=xi0)

    # --- sample vol and price paths (few paths, for visual clarity) --------
    sim = model.simulate(T=1.0, n_steps=200, n_paths=6, seed=7)
    t = sim["t"]
    vol_fig = go.Figure(layout=_layout(
        "Rough volatility paths  \u221av\u209c", "time (years)", "instantaneous vol"))
    price_fig = go.Figure(layout=_layout(
        "Induced price paths S\u209c", "time (years)", "price"))
    for i in range(sim["S"].shape[0]):
        c = _ACCENT[i % len(_ACCENT)]
        vol_fig.add_trace(go.Scatter(x=t, y=np.sqrt(sim["v"][i]), mode="lines",
                                     line=dict(width=1.4, color=c),
                                     name=f"path {i+1}"))
        price_fig.add_trace(go.Scatter(x=t, y=sim["S"][i], mode="lines",
                                       line=dict(width=1.4, color=c),
                                       name=f"path {i+1}"))

    # --- implied-vol smile at two maturities -------------------------------
    smile_fig = go.Figure(layout=_layout(
        "Implied-volatility smile", "log-moneyness  ln(K/S\u2080)",
        "Black\u2013Scholes implied vol"))
    strikes = model.S0 * np.exp(np.linspace(-0.25, 0.25, 15))
    for j, T in enumerate([0.1, 0.5]):
        smile = model.implied_smile(T, strikes, n_steps=120, n_paths=20000,
                                    seed=11)
        smile_fig.add_trace(go.Scatter(
            x=smile["log_moneyness"], y=smile["iv"], mode="lines+markers",
            line=dict(width=2, color=_ACCENT[j]),
            marker=dict(size=6), name=f"T = {T:g}y"))

    # --- ATM skew term structure vs the T^{H-1/2} power law ----------------
    mats = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    skew = model.atm_skew_term_structure(mats, n_steps=120, n_paths=20000,
                                         seed=13)
    sk = np.abs(skew["skew"])
    # Fit the theoretical slope power-law |skew| ~ c * T^{H-1/2} to the data.
    ref = mats ** (H - 0.5)
    scale = np.nanmedian(sk / ref)
    skew_fig = go.Figure(layout=_layout(
        "ATM skew term structure (log-log)", "maturity T (years)",
        "|ATM skew|  |d IV / d k|"))
    skew_fig.update_xaxes(type="log")
    skew_fig.update_yaxes(type="log")
    skew_fig.add_trace(go.Scatter(x=mats, y=sk, mode="lines+markers",
                                  line=dict(width=2, color=_ACCENT[2]),
                                  marker=dict(size=7), name="rough Bergomi (MC)"))
    skew_fig.add_trace(go.Scatter(x=mats, y=scale * ref, mode="lines",
                                  line=dict(width=2, dash="dash", color=_FONT),
                                  name=f"c\u00b7T^(H\u22121/2), H={H:g}"))

    return jsonify(vol=_fig(vol_fig), price=_fig(price_fig),
                   smile=_fig(smile_fig), skew=_fig(skew_fig))


# ---------------------------------------------------------------------------
# Markovian lifting
# ---------------------------------------------------------------------------

@app.route("/api/lifting", methods=["POST"])
def api_lifting():
    p = request.get_json(force=True)
    H = float(p.get("H", 0.1))
    n_factors = int(p.get("n_factors", 8))

    lift = ml.build_lift(H, n_factors)
    t = np.linspace(1e-3, 1.0, 400)
    kernel_fig = go.Figure(layout=_layout(
        f"Fractional kernel vs {n_factors}-factor sum of exponentials",
        "t", "K(t)"))
    kernel_fig.add_trace(go.Scatter(x=t, y=ml.fractional_kernel(t, H),
                                    mode="lines", name="true  t^(H\u22121/2)",
                                    line=dict(width=3, color=_ACCENT[0])))
    kernel_fig.add_trace(go.Scatter(x=t, y=lift.evaluate(t), mode="lines",
                                    name=f"sum of {n_factors} exp",
                                    line=dict(width=2, dash="dash",
                                              color=_ACCENT[2])))
    # Show a few individual OU modes to make the decomposition tangible.
    for j in range(min(n_factors, 5)):
        kernel_fig.add_trace(go.Scatter(
            x=t, y=lift.weights[j] * np.exp(-lift.nodes[j] * t), mode="lines",
            line=dict(width=1, color="rgba(167,139,250,0.35)"),
            showlegend=(j == 0), name="individual OU modes"))

    # --- error vs N --------------------------------------------------------
    ns = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    ev = ml.error_vs_n(H, ns)
    err_fig = go.Figure(layout=_layout(
        "Kernel approximation error vs number of factors N",
        "N (number of OU factors)", "relative L2 error"))
    err_fig.update_yaxes(type="log")
    err_fig.update_xaxes(type="log")
    err_fig.add_trace(go.Scatter(x=ev["n_factors"], y=ev["l2_error"],
                                 mode="lines+markers",
                                 line=dict(width=2, color=_ACCENT[4]),
                                 marker=dict(size=7), name="rel. L2 error"))

    # --- lifted vs true Volterra sample path -------------------------------
    sim = ml.simulate_lifted_volterra(lift, T=1.0, n_steps=400, seed=5)
    path_fig = go.Figure(layout=_layout(
        "Lifted Markovian path vs true (non-Markovian) Volterra path",
        "time", "Y\u209c"))
    path_fig.add_trace(go.Scatter(x=sim["t"], y=sim["Y_true"], mode="lines",
                                  name="true Volterra",
                                  line=dict(width=2.5, color=_ACCENT[0])))
    path_fig.add_trace(go.Scatter(x=sim["t"], y=sim["Y_lift"], mode="lines",
                                  name=f"lifted ({n_factors} OU factors)",
                                  line=dict(width=1.6, dash="dash",
                                            color=_ACCENT[2])))

    return jsonify(kernel=_fig(kernel_fig), error=_fig(err_fig),
                   path=_fig(path_fig))


# ---------------------------------------------------------------------------
# Path signatures
# ---------------------------------------------------------------------------

@app.route("/api/signatures", methods=["POST"])
def api_signatures():
    p = request.get_json(force=True)
    sigma = float(p.get("sigma", 0.2))
    K = float(p.get("K", 100.0))
    level = int(p.get("level", 3))

    res = sg.price_asian_with_signatures(
        S0=100.0, K=K, r=0.03, sigma=sigma, T=1.0,
        n_steps=50, n_paths=6000, level=level, seed=1)

    # --- predicted vs realised payoff scatter ------------------------------
    fit_fig = go.Figure(layout=_layout(
        "Signature regression: predicted vs realised Asian payoff",
        "realised discounted payoff", "signature-predicted payoff"))
    lim = float(max(res.scatter_realized.max(), res.scatter_predicted.max(), 1.0))
    fit_fig.add_trace(go.Scatter(
        x=res.scatter_realized, y=res.scatter_predicted, mode="markers",
        marker=dict(size=5, color=_ACCENT[1], opacity=0.5), name="test paths"))
    fit_fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 line=dict(width=2, dash="dash", color=_FONT),
                                 name="y = x"))

    # --- price comparison bar ----------------------------------------------
    price_fig = go.Figure(layout=_layout(
        "Asian option price: signature functional vs Monte-Carlo",
        "", "price"))
    price_fig.add_trace(go.Bar(
        x=["Monte-Carlo", "Signature functional"],
        y=[res.mc_price, res.sig_price],
        marker_color=[_ACCENT[4], _ACCENT[0]],
        error_y=dict(type="data", array=[1.96 * res.mc_se, 0], visible=True),
        text=[f"{res.mc_price:.3f}", f"{res.sig_price:.3f}"],
        textposition="outside"))

    # --- expected signature bar (the "market" object) ----------------------
    esig = res.expected_signature
    labels = [f"s{i+1}" for i in range(len(esig))]
    exp_fig = go.Figure(layout=_layout(
        "Expected signature  E[S(X)]  (level 1\u2013{})".format(level),
        "signature coordinate", "value"))
    exp_fig.add_trace(go.Bar(x=labels, y=esig, marker_color=_ACCENT[3]))

    stats = dict(r2_train=res.r2_train, r2_test=res.r2_test,
                 mc_price=res.mc_price, mc_se=res.mc_se,
                 sig_price=res.sig_price, n_features=res.n_features,
                 have_iisignature=sg.HAVE_IISIGNATURE)
    return jsonify(fit=_fig(fit_fig), price=_fig(price_fig),
                   expected=_fig(exp_fig), stats=stats)


@app.route("/")
def index():
    return render_template_string(_PAGE)


# ---------------------------------------------------------------------------
# Front end (single self-contained page)
# ---------------------------------------------------------------------------

_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pricing Library &mdash; Rough Vol, Markovian Lifts &amp; Signatures</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>
window.MathJax = { tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
<style>
  :root{
    --bg:#0b0f1a; --panel:#0f1420; --panel2:#151b2b; --line:#243050;
    --fg:#e6ecff; --muted:#8b97b8; --accent:#5eead4; --accent2:#a78bfa;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#182238,transparent),var(--bg);
       color:var(--fg);font-family:Inter,system-ui,Segoe UI,sans-serif;line-height:1.55}
  header{padding:34px 40px 18px;border-bottom:1px solid var(--line)}
  header h1{margin:0;font-size:26px;font-weight:700;letter-spacing:-.02em}
  header h1 .g{background:linear-gradient(90deg,var(--accent),var(--accent2));
       -webkit-background-clip:text;background-clip:text;color:transparent}
  header p{margin:8px 0 0;color:var(--muted);max-width:900px}
  nav{display:flex;gap:6px;padding:0 40px;margin-top:16px;flex-wrap:wrap}
  nav button{background:var(--panel);color:var(--muted);border:1px solid var(--line);
       padding:10px 18px;border-radius:10px 10px 0 0;cursor:pointer;font-size:14px;font-weight:600}
  nav button.active{background:var(--panel2);color:var(--fg);border-bottom-color:var(--panel2)}
  main{padding:26px 40px 60px}
  .tab{display:none} .tab.active{display:block}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
  @media(max-width:1050px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}
  .chart{width:100%;height:340px}
  .explain{background:var(--panel);border:1px solid var(--line);border-radius:14px;
       padding:20px 24px}
  .explain h2{margin:0 0 6px;font-size:19px}
  .explain h3{margin:16px 0 4px;font-size:15px;color:var(--accent)}
  .explain p{color:#cdd6f4;margin:8px 0}
  .controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin:18px 0 4px;
       background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 20px}
  .ctl{display:flex;flex-direction:column;gap:6px}
  .ctl label{font-size:12px;color:var(--muted)}
  .ctl input,.ctl select{background:var(--panel2);border:1px solid var(--line);color:var(--fg);
       border-radius:8px;padding:8px 10px;width:140px;font-size:14px}
  .ctl output{font-size:12px;color:var(--accent)}
  button.run{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#08121f;
       border:none;border-radius:10px;padding:11px 22px;font-weight:700;cursor:pointer;font-size:14px}
  .stat{display:inline-block;margin:4px 18px 4px 0;color:var(--muted);font-size:13px}
  .stat b{color:var(--fg);font-size:15px}
  .refs{color:var(--muted);font-size:12px;margin-top:14px}
  .badge{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;
       font-size:11px;color:var(--muted);margin-left:8px}
  .spinner{color:var(--accent);font-size:13px;margin-left:12px}
</style>
</head>
<body>
<header>
  <h1><span class="g">Pricing Library</span> &nbsp;&mdash;&nbsp; beyond Black&ndash;Scholes</h1>
  <p>A frontier-research tour for quant researchers: <b>rough volatility</b>,
     the <b>Markovian lifting</b> that makes it tractable, and <b>path
     signatures</b> for model-free pricing. Every claim is illustrated with a
     live Monte-Carlo or numerical experiment &mdash; not a black box.</p>
</header>

<nav>
  <button class="active" data-tab="rough">1 &middot; Rough Volatility</button>
  <button data-tab="lift">2 &middot; Markovian Lifting</button>
  <button data-tab="sig">3 &middot; Path Signatures</button>
</nav>

<main>

<!-- ============================ ROUGH VOL ============================ -->
<section class="tab active" id="rough">
  <div class="explain">
    <h2>Volatility is rough</h2>
    <p>Empirically, log-volatility behaves like a fractional Brownian motion
    with Hurst exponent \(H \approx 0.1\) &mdash; far below the \(H = 1/2\) of
    standard Brownian motion (Gatheral&ndash;Jaisson&ndash;Rosenbaum, 2018). The
    <b>rough Bergomi</b> model (Bayer&ndash;Friz&ndash;Gatheral, 2016) builds
    variance from a singular Volterra kernel:</p>
    \[ Y_t = \int_0^t K(t-s)\,dW_s,\qquad K(u)=\sqrt{2H}\,u^{H-1/2}, \]
    \[ v_t = \xi_0\,\exp\!\Big(\eta\,Y_t-\tfrac12\eta^2 t^{2H}\Big),\qquad
       \frac{dS_t}{S_t}=\sqrt{v_t}\,\big(\rho\,dW_t+\sqrt{1-\rho^2}\,dB_t\big). \]
    <p>For \(H<1/2\) the kernel is singular at \(0\), making the variance path
    <i>rough</i>. The tell-tale signature is the <b>ATM skew</b>, which explodes
    like \(T^{H-1/2}\) as maturity \(T\to 0\) &mdash; matching short-dated index
    options, which classical diffusions structurally cannot fit. We simulate
    \((W,Y)\) jointly and exactly (up to the grid) via a Cholesky factor of
    their closed-form covariance.</p>
    <p class="refs">Toggle \(H\) below and watch both the roughness of the vol
    paths and the steepness of the short-maturity skew respond.</p>
  </div>

  <div class="controls">
    <div class="ctl"><label>Hurst H <output id="oH">0.10</output></label>
      <input type="range" id="H" min="0.05" max="0.45" step="0.01" value="0.10"></div>
    <div class="ctl"><label>vol-of-vol &eta; <output id="oeta">1.5</output></label>
      <input type="range" id="eta" min="0.5" max="3.0" step="0.1" value="1.5"></div>
    <div class="ctl"><label>correlation &rho; <output id="orho">-0.70</output></label>
      <input type="range" id="rho" min="-0.95" max="0.0" step="0.05" value="-0.70"></div>
    <div class="ctl"><label>forward variance &xi;&#8320; <output id="oxi">0.040</output></label>
      <input type="range" id="xi0" min="0.01" max="0.12" step="0.005" value="0.040"></div>
    <button class="run" onclick="runRough()">Run simulation</button>
    <span class="spinner" id="spinRough"></span>
  </div>

  <div class="grid">
    <div class="card"><div class="chart" id="c_vol"></div></div>
    <div class="card"><div class="chart" id="c_price"></div></div>
    <div class="card"><div class="chart" id="c_smile"></div></div>
    <div class="card"><div class="chart" id="c_skew"></div></div>
  </div>
</section>

<!-- ========================= MARKOVIAN LIFT ========================= -->
<section class="tab" id="lift">
  <div class="explain">
    <h2>Markovian lifting: rough, but tractable</h2>
    <p>The rough kernel has infinite memory, so \(Y\) is <b>not Markovian</b>
    &mdash; painful to simulate and impossible to solve with a PDE. The fix uses
    the completely monotone representation</p>
    \[ t^{H-1/2}=\int_0^\infty e^{-xt}\,\mu(dx),\qquad
       \mu(dx)=\frac{x^{-H-1/2}}{\Gamma(1/2-H)}\,dx . \]
    <p>Discretising the measure \(\mu\approx\sum_j w_j\,\delta_{x_j}\) yields a
    <b>sum of exponentials</b> \(K_N(t)=\sum_j w_j e^{-x_j t}\), and each mode</p>
    \[ U^j_t=\int_0^t e^{-x_j(t-s)}\,dW_s,\qquad dU^j=-x_j U^j\,dt+dW \]
    <p>is a mean-reverting <b>Ornstein&ndash;Uhlenbeck factor</b>. The lifted
    process \(Y^N_t=\sum_j w_j U^j_t\) is driven by the finite Markov state
    \((U^1,\dots,U^N)\) and converges to the rough process as \(N\to\infty\)
    (Abi Jaber&ndash;El Euch, 2019; Cuchiero&ndash;Teichmann, 2019). We pick
    \((w_j,x_j)\) by mass / centroid quadrature of \(\mu\) on a geometric grid.</p>
  </div>

  <div class="controls">
    <div class="ctl"><label>Hurst H <output id="oH2">0.10</output></label>
      <input type="range" id="H2" min="0.05" max="0.45" step="0.01" value="0.10"></div>
    <div class="ctl"><label>factors N <output id="oN">8</output></label>
      <input type="range" id="N" min="1" max="64" step="1" value="8"></div>
    <button class="run" onclick="runLift()">Rebuild lift</button>
    <span class="spinner" id="spinLift"></span>
  </div>

  <div class="grid">
    <div class="card"><div class="chart" id="c_kernel"></div></div>
    <div class="card"><div class="chart" id="c_err"></div></div>
    <div class="card" style="grid-column:1/-1"><div class="chart" id="c_path"></div></div>
  </div>
</section>

<!-- =========================== SIGNATURES =========================== -->
<section class="tab" id="sig">
  <div class="explain">
    <h2>Path signatures &amp; model-free pricing</h2>
    <p>The <b>signature</b> of a path \(X:[0,T]\to\mathbb R^d\) is the graded
    collection of iterated integrals</p>
    \[ S(X)=\Big(1,\ \int dX^{i},\ \iint dX^{i}dX^{j},\
       \iiint dX^{i}dX^{j}dX^{k},\ \dots\Big), \]
    <p>truncated at level \(M\) to a vector of dimension
    \(\tfrac{d^{M+1}-1}{d-1}\). By <b>universality</b>, any continuous payoff is
    approximately <i>linear</i> in the signature, \(F(X)\approx\langle \ell,
    S(X)\rangle\). Pricing then <b>factorises</b>:</p>
    \[ \text{price}=\mathbb E[F(X)]\approx\big\langle \ell,\ \mathbb E[S(X)]\big\rangle, \]
    <p>into a payoff object \(\ell\) (learned once) and the <b>expected
    signature</b> \(\mathbb E[S(X)]\) &mdash; a property of the market. We learn
    \(\ell\) for an arithmetic <b>Asian option</b> by linear regression on
    simulated signatures, then price it as a linear functional of the expected
    signature and check it against plain Monte-Carlo. Signatures are computed
    with a pure-NumPy tensor-algebra routine (Chen's identity); an optional
    <code>iisignature</code> backend is auto-detected.
    <span class="badge" id="sigBackend">backend: &hellip;</span></p>
  </div>

  <div class="controls">
    <div class="ctl"><label>volatility &sigma; <output id="osig">0.20</output></label>
      <input type="range" id="sigma" min="0.05" max="0.6" step="0.01" value="0.20"></div>
    <div class="ctl"><label>strike K <output id="oK">100</output></label>
      <input type="range" id="K" min="80" max="120" step="1" value="100"></div>
    <div class="ctl"><label>signature level M</label>
      <select id="level"><option>2</option><option selected>3</option></select></div>
    <button class="run" onclick="runSig()">Learn &amp; price</button>
    <span class="spinner" id="spinSig"></span>
  </div>

  <div style="margin:16px 0 4px">
    <span class="stat">train R<sup>2</sup>: <b id="s_r2tr">&ndash;</b></span>
    <span class="stat">test R<sup>2</sup>: <b id="s_r2te">&ndash;</b></span>
    <span class="stat">MC price: <b id="s_mc">&ndash;</b></span>
    <span class="stat">signature price: <b id="s_sig">&ndash;</b></span>
    <span class="stat">features: <b id="s_nf">&ndash;</b></span>
  </div>

  <div class="grid">
    <div class="card"><div class="chart" id="c_fit"></div></div>
    <div class="card"><div class="chart" id="c_priceS"></div></div>
    <div class="card" style="grid-column:1/-1"><div class="chart" id="c_exp"></div></div>
  </div>
</section>

</main>

<script>
const PCFG = {responsive:true, displayModeBar:false};

// tabs
document.querySelectorAll('nav button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById(b.dataset.tab).classList.add('active');
  };
});

// live slider read-outs
const bind=(id,out,fmt)=>{const el=document.getElementById(id),o=document.getElementById(out);
  const upd=()=>o.textContent=fmt(parseFloat(el.value));el.addEventListener('input',upd);upd();};
bind('H','oH',v=>v.toFixed(2)); bind('eta','oeta',v=>v.toFixed(1));
bind('rho','orho',v=>v.toFixed(2)); bind('xi0','oxi',v=>v.toFixed(3));
bind('H2','oH2',v=>v.toFixed(2)); bind('N','oN',v=>v.toFixed(0));
bind('sigma','osig',v=>v.toFixed(2)); bind('K','oK',v=>v.toFixed(0));

async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  return r.json();
}
function draw(id,fig){Plotly.newPlot(id,fig.data,fig.layout,PCFG);}

async function runRough(){
  const s=document.getElementById('spinRough'); s.textContent='simulating\u2026';
  const d=await post('/api/rough_vol',{H:+H.value,eta:+eta.value,rho:+rho.value,xi0:+xi0.value});
  draw('c_vol',d.vol);draw('c_price',d.price);draw('c_smile',d.smile);draw('c_skew',d.skew);
  s.textContent='';
}
async function runLift(){
  const s=document.getElementById('spinLift'); s.textContent='building\u2026';
  const d=await post('/api/lifting',{H:+H2.value,n_factors:+N.value});
  draw('c_kernel',d.kernel);draw('c_err',d.error);draw('c_path',d.path);
  s.textContent='';
}
async function runSig(){
  const s=document.getElementById('spinSig'); s.textContent='learning\u2026';
  const d=await post('/api/signatures',{sigma:+sigma.value,K:+K.value,level:+level.value});
  draw('c_fit',d.fit);draw('c_priceS',d.price);draw('c_exp',d.expected);
  const st=d.stats;
  s_r2tr.textContent=st.r2_train.toFixed(3); s_r2te.textContent=st.r2_test.toFixed(3);
  s_mc.textContent=st.mc_price.toFixed(3)+' \u00b1 '+(1.96*st.mc_se).toFixed(3);
  s_sig.textContent=st.sig_price.toFixed(3); s_nf.textContent=st.n_features;
  document.getElementById('sigBackend').textContent=
    'backend: '+(st.have_iisignature?'iisignature':'NumPy fallback');
  s.textContent='';
}

// initial render
runRough(); runLift(); runSig();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5004)
