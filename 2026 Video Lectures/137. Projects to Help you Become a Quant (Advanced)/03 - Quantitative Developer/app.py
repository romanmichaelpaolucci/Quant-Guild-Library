"""
app.py
======

Flask dashboard for **Latency & Cythonization** (Quant Developer / Advanced).

Two sections, each pairing a rendered explanation with interactive Plotly charts:

  1. Monte Carlo Speed-up -- the same European call pricer written four ways
     (pure Python, numpy, Cython, numba); verify agreement, measure median
     latency, and show speed-ups vs the interpreter baseline.
  2. Algorithm Latency    -- sorting / search / reduction across input sizes
     on a log-log plot; see big-O slopes and constant-factor gaps side by side.

Run
---
    pip install -r requirements.txt
    python setup.py build_ext --inplace   # optional; enables Cython
    python app.py
    # open http://127.0.0.1:5005

The heavy lifting lives in ``benchmark.py``, ``latency_benchmark.py`` and the
``mc_*.py`` / ``mc_cython`` modules; this file only wires measurement to charts.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template_string, request

import latency_benchmark as lb
import mc_numpy
import mc_python
from benchmark import (
    CYTHON_AVAILABLE,
    NUMBA_AVAILABLE,
    bench,
    black_scholes_call,
)

if CYTHON_AVAILABLE:
    import mc_cython
if NUMBA_AVAILABLE:
    import mc_numba

app = Flask(__name__)

# --- shared "frontier engineering" chart theme -----------------------------
_PAPER = "#0f1420"
_PANEL = "#151b2b"
_GRID = "#2a3350"
_FONT = "#c9d4ee"
_ACCENT = ["#5eead4", "#a78bfa", "#f472b6", "#fbbf24", "#60a5fa", "#f87171"]

_IMPL_COLORS = {
    "pure_python": "#94a3b8",
    "numpy": "#60a5fa",
    "cython": "#f472b6",
    "numba": "#5eead4",
}
_IMPL_LABELS = {
    "pure_python": "pure Python",
    "numpy": "numpy",
    "cython": "Cython",
    "numba": "numba",
}


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
    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# Monte Carlo implementation benchmark
# ---------------------------------------------------------------------------

@app.route("/api/mc_bench", methods=["POST"])
def api_mc_bench():
    p = request.get_json(force=True)
    n_paths = int(p.get("paths", 20_000))
    n_steps = int(p.get("steps", 50))
    repeats = int(p.get("repeats", 5))
    seed = int(p.get("seed", 12345))

    # Cap so a browser request stays interactive even on slow machines.
    n_paths = max(1_000, min(n_paths, 80_000))
    n_steps = max(10, min(n_steps, 200))
    repeats = max(2, min(repeats, 9))

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps)).astype(np.float64)
    bs_price = black_scholes_call(S0, K, r, sigma, T)

    implementations: list[tuple[str, Callable[[], float]]] = [
        ("pure_python", lambda: mc_python.price_european_option(
            S0, K, r, sigma, T, Z, "call")),
        ("numpy", lambda: mc_numpy.price_european_option(
            S0, K, r, sigma, T, Z, "call")),
    ]
    if CYTHON_AVAILABLE:
        implementations.append(
            ("cython", lambda: mc_cython.price_european_option(
                S0, K, r, sigma, T, Z, "call"))
        )
    if NUMBA_AVAILABLE:
        implementations.append(
            ("numba", lambda: mc_numba.price_european_option(
                S0, K, r, sigma, T, Z, "call"))
        )

    results: dict[str, dict] = {}
    for name, fn in implementations:
        n_rep = min(repeats, 3) if name == "pure_python" else repeats
        results[name] = bench(fn, repeats=n_rep, warmups=1)

    ref = results["numpy"]["result"]
    tol = 1e-6 * max(1.0, abs(ref))
    base = results["pure_python"]["median"]

    rows = []
    for name, stats in results.items():
        med = stats["median"]
        speedup = base / med if med > 0 else float("inf")
        rows.append({
            "name": name,
            "label": _IMPL_LABELS[name],
            "price": float(stats["result"]),
            "median_ms": med * 1e3,
            "stdev_ms": stats["stdev"] * 1e3,
            "min_ms": stats["min"] * 1e3,
            "max_ms": stats["max"] * 1e3,
            "speedup": speedup,
            "agree": abs(stats["result"] - ref) <= tol,
            "delta": abs(stats["result"] - ref),
        })

    # --- speed-up bars -----------------------------------------------------
    speed_fig = go.Figure(layout=_layout(
        f"Speed-up vs pure Python  ({n_paths:,} paths \u00d7 {n_steps} steps)",
        "", "speed-up (\u00d7)"))
    speed_fig.update_yaxes(type="log")
    speed_fig.add_trace(go.Bar(
        x=[r["label"] for r in rows],
        y=[r["speedup"] for r in rows],
        marker_color=[_IMPL_COLORS[r["name"]] for r in rows],
        text=[f"{r['speedup']:.1f}\u00d7" for r in rows],
        textposition="outside",
        textfont=dict(color=_FONT, size=13),
        hovertemplate="%{x}<br>%{y:.2f}\u00d7 faster<extra></extra>",
    ))
    speed_fig.update_layout(yaxis=dict(title="speed-up (\u00d7)", gridcolor=_GRID))

    # --- median latency bars -----------------------------------------------
    lat_fig = go.Figure(layout=_layout(
        "Median wall-clock latency", "", "median time (ms)"))
    lat_fig.add_trace(go.Bar(
        x=[r["label"] for r in rows],
        y=[r["median_ms"] for r in rows],
        marker_color=[_IMPL_COLORS[r["name"]] for r in rows],
        error_y=dict(
            type="data",
            array=[r["stdev_ms"] for r in rows],
            visible=True,
            color="rgba(201,212,238,0.45)",
        ),
        text=[f"{r['median_ms']:.2f} ms" for r in rows],
        textposition="outside",
        textfont=dict(color=_FONT, size=12),
        hovertemplate="%{x}<br>%{y:.3f} ms \u00b1 %{error_y.array:.3f}<extra></extra>",
    ))

    # --- price agreement ---------------------------------------------------
    agree_fig = go.Figure(layout=_layout(
        "Price agreement (shared paths Z)", "", "Monte Carlo price"))
    agree_fig.add_trace(go.Bar(
        x=[r["label"] for r in rows],
        y=[r["price"] for r in rows],
        marker_color=[_IMPL_COLORS[r["name"]] for r in rows],
        text=[f"{r['price']:.4f}" for r in rows],
        textposition="outside",
        textfont=dict(color=_FONT, size=12),
    ))
    agree_fig.add_hline(
        y=bs_price, line=dict(width=2, dash="dash", color=_ACCENT[3]),
        annotation_text=f"Black\u2013Scholes {bs_price:.4f}",
        annotation_font_color=_ACCENT[3],
    )

    available = {
        "cython": CYTHON_AVAILABLE,
        "numba": NUMBA_AVAILABLE,
        "paths": n_paths,
        "steps": n_steps,
        "iterations": n_paths * n_steps,
        "bs_price": bs_price,
        "mc_price": float(ref),
        "mc_err": abs(float(ref) - bs_price),
        "all_agree": all(r["agree"] for r in rows),
        "tol": tol,
    }
    return jsonify(
        speedup=_fig(speed_fig),
        latency=_fig(lat_fig),
        agree=_fig(agree_fig),
        rows=rows,
        meta=available,
    )


# ---------------------------------------------------------------------------
# Algorithm latency vs input size
# ---------------------------------------------------------------------------

@app.route("/api/latency", methods=["POST"])
def api_latency():
    p = request.get_json(force=True)
    # "quick" keeps the interactive demo snappy; "full" matches the CLI script.
    mode = str(p.get("mode", "quick"))

    if mode == "full":
        sort_search_sizes = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
        reduction_sizes = [1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    else:
        sort_search_sizes = [100, 300, 1000, 3000, 10_000, 30_000]
        reduction_sizes = [1000, 3000, 10_000, 30_000, 100_000, 300_000]

    rng = np.random.default_rng(7)
    sorting = lb.benchmark_sorting(sort_search_sizes, rng)
    search = lb.benchmark_search(sort_search_sizes, rng)
    reduction = lb.benchmark_reduction(reduction_sizes, rng)

    def series_fig(title: str, sizes: list[int], results: dict[str, list[float]]) -> go.Figure:
        fig = go.Figure(layout=_layout(title, "input size n", "median latency (\u00b5s)"))
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        x = np.asarray(sizes, dtype=float)
        for i, (name, times) in enumerate(results.items()):
            y = np.asarray(times, dtype=float) * 1e6
            mask = ~np.isnan(y)
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="lines+markers",
                name=name,
                line=dict(width=2.2, color=_ACCENT[i % len(_ACCENT)]),
                marker=dict(size=7),
            ))
        return fig

    sort_fig = series_fig("Sorting latency vs n (log-log)", sort_search_sizes, sorting)
    search_fig = series_fig("Search latency vs n (log-log)", sort_search_sizes, search)
    red_fig = series_fig("Reduction latency vs n (log-log)", reduction_sizes, reduction)

    # Headline takeaways (guard sizes that may be absent in quick mode).
    takeaways = []
    if 1000 in sort_search_sizes:
        i = sort_search_sizes.index(1000)
        py_ins = sorting["python_insertion (O(n^2))"][i]
        timsort = sorting["sorted/Timsort (O(n log n))"][i]
        if py_ins == py_ins and timsort > 0:
            takeaways.append({
                "label": "Timsort vs O(n\u00b2) at n=1,000",
                "value": f"{py_ins / timsort:.0f}\u00d7",
            })
    if 30_000 in sort_search_sizes:
        j = sort_search_sizes.index(30_000)
        lin = search["linear scan (O(n))"][j]
        setl = search["set lookup (O(1))"][j]
        if setl > 0:
            takeaways.append({
                "label": "set vs linear scan at n=30,000",
                "value": f"{lin / setl:.0f}\u00d7",
            })
    if 100_000 in sort_search_sizes:
        j = sort_search_sizes.index(100_000)
        lin = search["linear scan (O(n))"][j]
        setl = search["set lookup (O(1))"][j]
        if setl > 0:
            takeaways.append({
                "label": "set vs linear scan at n=100,000",
                "value": f"{lin / setl:.0f}\u00d7",
            })
    if 300_000 in reduction_sizes:
        k = reduction_sizes.index(300_000)
        pyloop = reduction["python loop sum (O(n))"][k]
        npsum = reduction["numpy.sum (O(n), vectorised)"][k]
        if npsum > 0:
            takeaways.append({
                "label": "numpy.sum vs loop at n=300,000",
                "value": f"{pyloop / npsum:.0f}\u00d7",
            })
    if 1_000_000 in reduction_sizes:
        k = reduction_sizes.index(1_000_000)
        pyloop = reduction["python loop sum (O(n))"][k]
        npsum = reduction["numpy.sum (O(n), vectorised)"][k]
        if npsum > 0:
            takeaways.append({
                "label": "numpy.sum vs loop at n=1,000,000",
                "value": f"{pyloop / npsum:.0f}\u00d7",
            })

    def pack(sizes, results):
        return {
            "sizes": sizes,
            "series": {
                name: [None if (t != t) else t * 1e6 for t in times]
                for name, times in results.items()
            },
        }

    return jsonify(
        sorting=_fig(sort_fig),
        search=_fig(search_fig),
        reduction=_fig(red_fig),
        takeaways=takeaways,
        tables={
            "sorting": pack(sort_search_sizes, sorting),
            "search": pack(sort_search_sizes, search),
            "reduction": pack(reduction_sizes, reduction),
        },
        mode=mode,
    )


@app.route("/api/status")
def api_status():
    return jsonify(
        cython=CYTHON_AVAILABLE,
        numba=NUMBA_AVAILABLE,
        python=f"{__import__('sys').version_info.major}."
               f"{__import__('sys').version_info.minor}",
        numpy=np.__version__,
    )


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
<title>Latency Lab &mdash; Cythonization &amp; Big-O in Practice</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>
window.MathJax = { tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
<style>
  :root{
    --bg:#0b0f1a; --panel:#0f1420; --panel2:#151b2b; --line:#243050;
    --fg:#e6ecff; --muted:#8b97b8; --accent:#5eead4; --accent2:#a78bfa;
    --warn:#fbbf24; --ok:#5eead4; --bad:#f87171;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#182238,transparent),var(--bg);
       color:var(--fg);font-family:Inter,system-ui,Segoe UI,sans-serif;line-height:1.55}
  header{padding:34px 40px 18px;border-bottom:1px solid var(--line)}
  header h1{margin:0;font-size:26px;font-weight:700;letter-spacing:-.02em}
  header h1 .g{background:linear-gradient(90deg,var(--accent),var(--accent2));
       -webkit-background-clip:text;background-clip:text;color:transparent}
  header p{margin:8px 0 0;color:var(--muted);max-width:920px}
  .pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
  .pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
       border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted);
       background:rgba(15,20,32,.6)}
  .pill .dot{width:7px;height:7px;border-radius:50%;background:var(--bad)}
  .pill.on .dot{background:var(--ok);box-shadow:0 0 8px rgba(94,234,212,.55)}
  nav{display:flex;gap:6px;padding:0 40px;margin-top:16px;flex-wrap:wrap}
  nav button{background:var(--panel);color:var(--muted);border:1px solid var(--line);
       padding:10px 18px;border-radius:10px 10px 0 0;cursor:pointer;font-size:14px;font-weight:600}
  nav button.active{background:var(--panel2);color:var(--fg);border-bottom-color:var(--panel2)}
  main{padding:26px 40px 60px}
  .tab{display:none} .tab.active{display:block}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:18px}
  @media(max-width:1100px){.grid,.grid3{grid-template-columns:1fr}}
  .card{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}
  .chart{width:100%;height:360px}
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
       border-radius:8px;padding:8px 10px;width:150px;font-size:14px}
  .ctl output{font-size:12px;color:var(--accent)}
  button.run{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#08121f;
       border:none;border-radius:10px;padding:11px 22px;font-weight:700;cursor:pointer;font-size:14px}
  button.run:disabled{opacity:.55;cursor:wait}
  .stat{display:inline-block;margin:4px 18px 4px 0;color:var(--muted);font-size:13px}
  .stat b{color:var(--fg);font-size:15px}
  .refs{color:var(--muted);font-size:12px;margin-top:14px}
  .badge{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;
       font-size:11px;color:var(--muted);margin-left:8px}
  .spinner{color:var(--accent);font-size:13px;margin-left:12px}
  .takeaways{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 4px}
  .tk{background:var(--panel);border:1px solid var(--line);border-radius:12px;
       padding:12px 16px;min-width:180px}
  .tk .v{font-size:22px;font-weight:700;color:var(--accent);letter-spacing:-.02em}
  .tk .l{font-size:12px;color:var(--muted);margin-top:2px}
  table.res{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
  table.res th,table.res td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
  table.res th:first-child,table.res td:first-child{text-align:left}
  table.res th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
       letter-spacing:.04em}
  .ok{color:var(--ok)} .bad{color:var(--bad)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
       font-size:12px;background:rgba(94,234,212,.08);padding:1px 6px;border-radius:5px;
       color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1><span class="g">Latency Lab</span> &nbsp;&mdash;&nbsp; measure, then accelerate</h1>
  <p>A hands-on performance lab for quant developers: compare the <b>same</b> Monte
     Carlo pricer across pure Python, numpy, Cython and numba, then watch how
     algorithm latency scales with input size. Warm-ups, medians, shared paths
     &mdash; the methodology that makes benchmarks trustworthy.</p>
  <div class="pills">
    <span class="pill" id="pillCy"><span class="dot"></span>Cython</span>
    <span class="pill" id="pillNb"><span class="dot"></span>numba</span>
    <span class="pill" id="pillPy"><span class="dot" style="background:#60a5fa"></span>Python <span id="pyVer"></span></span>
    <span class="pill" id="pillNp"><span class="dot" style="background:#a78bfa"></span>numpy <span id="npVer"></span></span>
  </div>
</header>

<nav>
  <button class="active" data-tab="mc">1 &middot; Monte Carlo Speed-up</button>
  <button data-tab="lat">2 &middot; Algorithm Latency</button>
</nav>

<main>

<!-- ======================== MONTE CARLO ======================== -->
<section class="tab active" id="mc">
  <div class="explain">
    <h2>One pricer, four implementations</h2>
    <p>Under the risk-neutral measure a European call is</p>
    \[
      S_{t+\Delta t}=S_t\exp\!\Big((r-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z\Big),
      \qquad
      \widehat C=e^{-rT}\frac1N\sum_{i=1}^N\max(S_T^{(i)}-K,0).
    \]
    <p>Every implementation prices the <b>identical</b> shared shocks \(Z\), so
    disagreement can only come from floating-point summation order. Timing uses
    warm-up calls (JIT / cache) then the <b>median</b> of several
    <code>perf_counter</code> samples &mdash; never a single noisy wall-clock
    reading.</p>
    <h3>What Cython / numba remove</h3>
    <p>Pure Python pays interpreter overhead on every arithmetic op and array
    index. Cython turns typed memoryviews into C pointer arithmetic and calls
    <code>libc</code> <code>exp</code>/<code>sqrt</code>; numba does the same via
    LLVM JIT at runtime. Numpy vectorises but materialises intermediate arrays
    &mdash; often memory-bandwidth bound.</p>
    <p class="refs">Tip: start with ~20k paths for a snappy demo; raise paths
    when you want the speed-up ratios to stabilise.</p>
  </div>

  <div class="controls">
    <div class="ctl"><label>paths N <output id="oPaths">20000</output></label>
      <input type="range" id="paths" min="5000" max="60000" step="5000" value="20000"></div>
    <div class="ctl"><label>steps <output id="oSteps">50</output></label>
      <input type="range" id="steps" min="20" max="100" step="10" value="50"></div>
    <div class="ctl"><label>repeats <output id="oRep">5</output></label>
      <input type="range" id="repeats" min="3" max="9" step="1" value="5"></div>
    <button class="run" id="btnMc" onclick="runMc()">Run benchmark</button>
    <span class="spinner" id="spinMc"></span>
  </div>

  <div style="margin:16px 0 4px">
    <span class="stat">iterations: <b id="s_iters">&ndash;</b></span>
    <span class="stat">MC price: <b id="s_mc">&ndash;</b></span>
    <span class="stat">Black&ndash;Scholes: <b id="s_bs">&ndash;</b></span>
    <span class="stat">|MC &minus; BS|: <b id="s_err">&ndash;</b></span>
    <span class="stat">agreement: <b id="s_agree">&ndash;</b></span>
  </div>

  <div class="grid">
    <div class="card"><div class="chart" id="c_speed"></div></div>
    <div class="card"><div class="chart" id="c_lat"></div></div>
    <div class="card" style="grid-column:1/-1"><div class="chart" id="c_agree" style="height:300px"></div></div>
  </div>

  <div class="card" style="margin-top:18px">
    <table class="res" id="mcTable">
      <thead><tr>
        <th>implementation</th><th>median (ms)</th><th>stdev (ms)</th>
        <th>speed-up</th><th>price</th><th>vs numpy</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</section>

<!-- ======================== LATENCY ======================== -->
<section class="tab" id="lat">
  <div class="explain">
    <h2>Big-O on a log-log plot</h2>
    <p>Asymptotic complexity predicts the <i>shape</i> of latency vs input size.
    On log-log axes:</p>
    \[
      T(n)\propto n^{\alpha}
      \quad\Rightarrow\quad
      \log T \approx \alpha\log n + c
    \]
    <p>so slope \(\approx\alpha\). An \(O(1)\) / \(O(\log n)\) lookup is nearly
    flat; \(O(n)\) has slope \(\approx 1\); \(O(n^2)\) is the steep climb.
    Constant factors decide which of two same-order algorithms wins in practice
    &mdash; <code>numpy.sum</code> and a Python loop are both \(O(n)\), yet
    differ by tens of times.</p>
    <h3>Three families every quant meets</h3>
    <p><b>Sorting</b> &mdash; pure-Python insertion (\(O(n^2)\)) vs Timsort vs
    <code>numpy.sort</code>. <b>Search</b> &mdash; linear scan vs
    <code>bisect</code> vs <code>set</code>. <b>Reduction</b> &mdash; Python
    loop sum vs vectorised <code>numpy.sum</code>.</p>
  </div>

  <div class="controls">
    <div class="ctl"><label>benchmark depth</label>
      <select id="latMode">
        <option value="quick" selected>quick (interactive)</option>
        <option value="full">full (matches CLI)</option>
      </select></div>
    <button class="run" id="btnLat" onclick="runLat()">Measure latency</button>
    <span class="spinner" id="spinLat"></span>
  </div>

  <div class="takeaways" id="takeaways"></div>

  <div class="grid3">
    <div class="card"><div class="chart" id="c_sort"></div></div>
    <div class="card"><div class="chart" id="c_search"></div></div>
    <div class="card"><div class="chart" id="c_red"></div></div>
  </div>
</section>

</main>

<script>
const PCFG = {responsive:true, displayModeBar:false};

document.querySelectorAll('nav button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById(b.dataset.tab).classList.add('active');
  };
});

const bind=(id,out,fmt)=>{
  const el=document.getElementById(id), o=document.getElementById(out);
  const upd=()=>o.textContent=fmt(parseFloat(el.value));
  el.addEventListener('input',upd); upd();
};
bind('paths','oPaths',v=>v.toLocaleString('en-US',{maximumFractionDigits:0}));
bind('steps','oSteps',v=>v.toFixed(0));
bind('repeats','oRep',v=>v.toFixed(0));

async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function draw(id,fig){Plotly.newPlot(id,fig.data,fig.layout,PCFG);}

async function loadStatus(){
  const s=await fetch('/api/status').then(r=>r.json());
  const cy=document.getElementById('pillCy');
  const nb=document.getElementById('pillNb');
  if(s.cython) cy.classList.add('on');
  else cy.title='Build with: python setup.py build_ext --inplace';
  if(s.numba) nb.classList.add('on');
  else nb.title='Install with: pip install numba';
  document.getElementById('pyVer').textContent=s.python;
  document.getElementById('npVer').textContent=s.numpy;
}

async function runMc(){
  const btn=document.getElementById('btnMc');
  const s=document.getElementById('spinMc');
  btn.disabled=true; s.textContent='benchmarking\u2026 (pure Python is slow)';
  try{
    const d=await post('/api/mc_bench',{
      paths:+paths.value, steps:+steps.value, repeats:+repeats.value
    });
    draw('c_speed',d.speedup); draw('c_lat',d.latency); draw('c_agree',d.agree);
    const m=d.meta;
    s_iters.textContent=m.iterations.toLocaleString();
    s_mc.textContent=m.mc_price.toFixed(4);
    s_bs.textContent=m.bs_price.toFixed(4);
    s_err.textContent=m.mc_err.toFixed(4);
    s_agree.textContent=m.all_agree?'ALL AGREE':'MISMATCH';
    s_agree.className=m.all_agree?'ok':'bad';
    const tb=document.querySelector('#mcTable tbody');
    tb.innerHTML=d.rows.map(r=>`<tr>
      <td>${r.label}</td>
      <td>${r.median_ms.toFixed(3)}</td>
      <td>${r.stdev_ms.toFixed(3)}</td>
      <td>${r.speedup.toFixed(1)}\u00d7</td>
      <td>${r.price.toFixed(6)}</td>
      <td class="${r.agree?'ok':'bad'}">${r.delta.toExponential(1)}</td>
    </tr>`).join('');
  }catch(e){ s.textContent='error: '+e.message; }
  finally{ btn.disabled=false; if(s.textContent.startsWith('benchmarking')) s.textContent=''; }
}

async function runLat(){
  const btn=document.getElementById('btnLat');
  const s=document.getElementById('spinLat');
  btn.disabled=true; s.textContent='measuring\u2026';
  try{
    const d=await post('/api/latency',{mode:latMode.value});
    draw('c_sort',d.sorting); draw('c_search',d.search); draw('c_red',d.reduction);
    document.getElementById('takeaways').innerHTML=d.takeaways.map(t=>
      `<div class="tk"><div class="v">${t.value}</div><div class="l">${t.label}</div></div>`
    ).join('');
  }catch(e){ s.textContent='error: '+e.message; }
  finally{ btn.disabled=false; if(s.textContent.startsWith('measuring')) s.textContent=''; }
}

loadStatus();
runMc();
runLat();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5005)
