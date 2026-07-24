/* ===========================================================================
   Market Making Simulator — front-end controller
   ---------------------------------------------------------------------------
   Responsibilities:
     1. Read the control panel, POST it to /api/simulate.
     2. Receive the full time series and REPLAY it tick-by-tick with Plotly,
        using extendTraces() on a timer so it looks like a live market feed.
     3. Update the live stat strip and, at the end, render a post-mortem.
   The numerical work is done server-side; this file is pure presentation.
   =========================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Fields that map 1:1 to SimConfig keys.
  const FIELDS = [
    "S0", "K", "T", "horizon", "option_type", "path_model", "mu", "sigma", "r",
    "lam", "jump_mean", "jump_std", "pricing_model", "half_spread", "quote_size",
    "fill_intensity", "risk_aversion", "pricing_sigma", "hedge_steps", "n_steps", "seed",
  ];

  const COLORS = {
    price: "#4aa8ff",
    band: "rgba(61,220,151,0.18)",
    bid: "#3ddc97",
    ask: "#ff5c72",
    realized: "#8b98ad",
    model: "#4aa8ff",
    truth: "#ffb347",
    grid: "#1e2635",
    gbm: "#4aa8ff",          // Black-Scholes / GBM world (blue)
    merton: "#ffb347",       // Merton jump-diffusion world (amber)
  };

  let timer = null;          // animation interval handle
  let data = null;           // last simulation payload
  let distData = null;       // last return-distribution payload

  // ---- Prefill controls from server defaults --------------------------
  function prefill() {
    const d = window.SIM_DEFAULTS || {};
    FIELDS.forEach((f) => {
      const el = $(f);
      if (!el) return;
      let v = d[f];
      if (f === "pricing_sigma" && (v === null || v === undefined)) v = "";
      if (v !== undefined && v !== null) el.value = v;
    });
    updateMatchHint();
  }

  function readConfig() {
    const cfg = {};
    FIELDS.forEach((f) => {
      const el = $(f);
      if (el) cfg[f] = el.value;
    });
    return cfg;
  }

  // ---- Match/mismatch teaching hint -----------------------------------
  function updateMatchHint() {
    const pm = $("pricing_model").value;
    const path = $("path_model").value;
    const pmIsJump = pm === "merton";
    const pathIsJump = path === "merton";
    const hint = $("matchHint");
    if (pmIsJump === pathIsJump) {
      hint.className = "hint match";
      hint.textContent = "✓ Model matches reality — the MM should capture the spread with controlled variance.";
    } else if (!pmIsJump && pathIsJump) {
      hint.className = "hint mismatch";
      hint.textContent = "⚠ Pricing GBM in a jumpy world — classic 'pennies in front of a steamroller'. Watch mark-to-truth.";
    } else {
      hint.className = "hint mismatch";
      hint.textContent = "⚠ Pricing jumps that aren't there — quotes may be too wide/defensive, giving up edge.";
    }
  }

  // ---- Chart scaffolding ----------------------------------------------
  const baseLayout = (title) => ({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#8b98ad", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 55, r: 20, t: 10, b: 35 },
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
    xaxis: { gridcolor: COLORS.grid, zeroline: false, title: "time (yrs)" },
    yaxis: { gridcolor: COLORS.grid, zeroline: false, title: title },
  });

  // The underlying (~$100) and the option quotes (~$1-10) live on totally
  // different scales, so the bid/ask band gets its own right-hand y-axis.
  const priceLayout = () => ({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#8b98ad", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 55, r: 62, t: 10, b: 35 },
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
    xaxis: { gridcolor: COLORS.grid, zeroline: false, title: "time (yrs)" },
    yaxis: {
      gridcolor: COLORS.grid, zeroline: false,
      title: { text: "underlying", font: { color: COLORS.price } },
      tickfont: { color: COLORS.price },
    },
    yaxis2: {
      title: { text: "option quote ($)", font: { color: COLORS.bid } },
      tickfont: { color: COLORS.bid },
      overlaying: "y", side: "right", zeroline: false, showgrid: false,
    },
  });

  function initPriceChart() {
    // ask/bid quote the OPTION and are drawn on the secondary (right) axis y2;
    // the underlying stays on the primary (left) axis y.
    const traces = [
      { x: [], y: [], name: "ask", yaxis: "y2", mode: "lines", line: { color: COLORS.ask, width: 1, dash: "dot" }, hoverinfo: "skip" },
      { x: [], y: [], name: "bid", yaxis: "y2", mode: "lines", line: { color: COLORS.bid, width: 1, dash: "dot" }, fill: "tonexty", fillcolor: COLORS.band, hoverinfo: "skip" },
      { x: [], y: [], name: "underlying", yaxis: "y", mode: "lines", line: { color: COLORS.price, width: 2 } },
    ];
    Plotly.newPlot("priceChart", traces, priceLayout(), { displayModeBar: false, responsive: true });
  }

  function initPnlChart() {
    // realized = the actual, stochastic equity curve (each trade's hedged life
    // realized under the true model). The two smooth dashed lines are the
    // cumulative *expected* edge measured against the MM's model price vs the
    // true price -- realized should wobble upward around the truth line.
    const traces = [
      { x: [], y: [], name: "expected (model)", mode: "lines", line: { color: COLORS.model, width: 1.5, dash: "dot" } },
      { x: [], y: [], name: "expected (truth)", mode: "lines", line: { color: COLORS.truth, width: 1.5, dash: "dot" } },
      { x: [], y: [], name: "realized", mode: "lines", line: { color: COLORS.bid, width: 2 } },
    ];
    Plotly.newPlot("pnlChart", traces, baseLayout("cumulative P/L ($)"), { displayModeBar: false, responsive: true });
  }

  // ---- Stats strip -----------------------------------------------------
  function fmt(x, dp = 2) {
    if (x === undefined || x === null || Number.isNaN(x)) return "–";
    return Number(x).toFixed(dp);
  }
  function setSigned(el, x, dp = 2) {
    el.textContent = fmt(x, dp);
    el.classList.remove("pos", "neg");
    if (x > 1e-9) el.classList.add("pos");
    else if (x < -1e-9) el.classList.add("neg");
  }

  function updateStats(i) {
    const r = data.result;
    $("s_tick").textContent = i + " / " + (r.t.length - 1);
    $("s_spot").textContent = fmt(r.S[i]);
    // inventory[] carries cumulative contracts traded; hedge[] carries the
    // running average realized edge per contract (should approach half_spread).
    $("s_inv").textContent = fmt(r.inventory[i], 0);
    setSigned($("s_hedge"), r.hedge[i], 3);
    setSigned($("s_real"), r.realized_pnl[i]);
    setSigned($("s_model"), r.mtm_model[i]);
    setSigned($("s_truth"), r.mtm_truth[i]);
    setSigned($("s_gap"), r.mtm_model[i] - r.mtm_truth[i]);
  }

  // ---- Animation replay -----------------------------------------------
  function stopAnim() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  function setStatus(text, cls) {
    const pill = $("statusPill");
    pill.textContent = text;
    pill.className = "ticker" + (cls ? " " + cls : "");
  }

  function animate() {
    const r = data.result;
    const n = r.t.length;
    const speed = Math.max(1, parseInt($("speed").value || "30", 10));

    initPriceChart();
    initPnlChart();
    setStatus("● LIVE", "live");

    let i = 0;
    stopAnim();
    timer = setInterval(() => {
      if (i >= n) {
        stopAnim();
        setStatus("DONE", "done");
        renderSummary();
        return;
      }
      // Stream one tick into every trace via extendTraces.
      Plotly.extendTraces("priceChart", {
        x: [[r.t[i]], [r.t[i]], [r.t[i]]],
        y: [[r.ask[i]], [r.bid[i]], [r.S[i]]],
      }, [0, 1, 2]);

      Plotly.extendTraces("pnlChart", {
        x: [[r.t[i]], [r.t[i]], [r.t[i]]],
        y: [[r.mtm_model[i]], [r.mtm_truth[i]], [r.realized_pnl[i]]],
      }, [0, 1, 2]);

      updateStats(i);
      i++;
    }, speed);
  }

  // ---- Post-mortem summary --------------------------------------------
  function renderSummary() {
    const s = data.result.summary;
    const row = (k, v) => `<span class="k">${k.padEnd(28, " ")}</span><span class="v">${v}</span>`;
    const lines = [
      row("Pricing model", s.pricing_model),
      row("True path model", s.path_model),
      row("Contracts traded", fmt(s.total_contracts_traded, 0) + "  (bought " + fmt(s.contracts_bought, 0) + " / sold " + fmt(s.contracts_sold, 0) + ")"),
      row("Quoted spread capture", "$" + fmt(s.expected_spread_capture) + "  (½-spread × contracts)"),
      row("Realized P/L (equity)", "$" + fmt(s.final_realized_pnl)),
      row("Avg realized edge / ct", "$" + fmt(s.avg_realized_edge, 3) + "  vs quoted $" + fmt(s.half_spread, 3)),
      row("Expected edge (model)", "$" + fmt(s.final_expected_model)),
      row("Expected edge (truth)", "$" + fmt(s.final_expected_truth)),
      row("Model − Truth gap", "$" + fmt(s.model_vs_truth_gap)),
      row("Realized − Truth gap", "$" + fmt(s.realized_vs_truth_gap)),
      row("P/L step volatility", "$" + fmt(s.pnl_step_vol, 3)),
      row("Max drawdown", "$" + fmt(s.max_drawdown_truth)),
    ].join("\n");

    // Each fill is realized on the spot: we simulate that option's delta-hedged
    // life to expiry under the TRUE model and bank the discounted result. A
    // hedged option's P/L is drift-independent and equals premium − fair, so on
    // average every contract banks the half-spread — positive expectancy. When
    // the pricing model is wrong, the MM still *thinks* it banks the spread
    // (expected-model line) while the true edge sags (expected-truth line) and
    // the realized curve takes fat-tailed jump losses the delta hedge never saw.
    let verdict;
    const matched = (s.pricing_model === "merton") === (s.path_model === "merton");
    const dd = s.max_drawdown_truth;
    const vol = s.pnl_step_vol;
    const gap = s.model_vs_truth_gap;
    if (matched) {
      verdict = `<span class="verdict good">✓ Matched model: realized P/L ≈ $${fmt(s.final_realized_pnl)}, and the average realized edge is $${fmt(s.avg_realized_edge, 3)}/contract vs the $${fmt(s.half_spread, 3)} half-spread you quoted — you are banking the spread. Low P/L step-vol $${fmt(vol, 3)} and shallow drawdown $${fmt(dd)}. Buying below fair and selling above fair realizes positive expectancy: repeat over seeds and the mean stays positive with tight variance.</span>`;
    } else {
      verdict = `<span class="verdict bad">⚠ Mismatched model: on your own marks you *think* you banked $${fmt(s.final_expected_model)} of spread (Model−Truth gap $${fmt(gap)} vs true fair), yet you actually realized $${fmt(s.final_realized_pnl)} — an average of $${fmt(s.avg_realized_edge, 3)}/contract against the $${fmt(s.half_spread, 3)} you quoted. P/L step-vol ($${fmt(vol, 3)}) and max drawdown ($${fmt(dd)}) blow out vs the matched case: your delta hedge can't contain the jumps, so the "guaranteed" spread gets swamped by fat-tailed losses. Pennies in front of a steamroller.</span>`;
    }

    $("summaryBox").innerHTML = lines + "\n" + verdict;
  }

  // ---- Run -------------------------------------------------------------
  async function run() {
    stopAnim();
    setStatus("… computing", "");
    $("runBtn").disabled = true;
    try {
      const resp = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readConfig()),
      });
      if (!resp.ok) throw new Error("server " + resp.status);
      data = await resp.json();
      animate();
    } catch (err) {
      setStatus("ERROR", "");
      $("summaryBox").textContent = "Simulation failed: " + err.message;
    } finally {
      $("runBtn").disabled = false;
    }
  }

  // ---- Return distributions (GBM vs Merton) ---------------------------
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  function renderPaths(res) {
    const t = res.t;
    const traces = [];
    ["gbm", "merton"].forEach((m) => {
      const paths = res.models[m].paths;
      paths.forEach((p, idx) => {
        traces.push({
          x: t, y: p, mode: "lines",
          line: { color: COLORS[m], width: 1 },
          opacity: 0.55,
          name: m === "gbm" ? "GBM" : "Merton",
          legendgroup: m,
          showlegend: idx === 0,
          hoverinfo: "skip",
        });
      });
    });
    const layout = baseLayout("underlying price");
    layout.legend = { orientation: "h", y: 1.12, x: 0, font: { size: 10 } };
    Plotly.newPlot("pathChart", traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderHist(res) {
    const b = res.bins;
    const xbins = { start: b.start, end: b.end, size: b.size };
    const mk = (m, label) => ({
      x: res.models[m].returns,
      type: "histogram",
      histnorm: "probability density",
      xbins,
      name: label,
      marker: { color: rgba(COLORS[m], 0.5), line: { color: COLORS[m], width: 1 } },
      opacity: 0.75,
    });
    const traces = [mk("gbm", "GBM (Black-Scholes)"), mk("merton", "Merton jumps")];
    const layout = baseLayout("probability density");
    layout.barmode = "overlay";
    layout.xaxis = { gridcolor: COLORS.grid, zeroline: true, zerolinecolor: "#3a465c", title: "terminal return  (S_T / S_0 − 1)", tickformat: ".0%" };
    layout.legend = { orientation: "h", y: 1.12, x: 0, font: { size: 10 } };
    Plotly.newPlot("histChart", traces, layout, { displayModeBar: false, responsive: true });
  }

  function pct(x, dp = 2) {
    if (x === undefined || x === null || Number.isNaN(x)) return "–";
    return (Number(x) * 100).toFixed(dp) + "%";
  }

  function renderDistStats(res) {
    const g = res.models.gbm.stats;
    const m = res.models.merton.stats;
    const rows = [
      ["Metric", "GBM", "Merton"],
      ["Mean return", pct(g.mean), pct(m.mean)],
      ["Volatility (σ)", pct(g.std), pct(m.std)],
      ["Skewness", g.skew.toFixed(2), m.skew.toFixed(2)],
      ["Excess kurtosis", g.excess_kurtosis.toFixed(2), m.excess_kurtosis.toFixed(2)],
      ["1% VaR (left tail)", pct(g.var_1pct), pct(m.var_1pct)],
      ["Worst path", pct(g.worst), pct(m.worst)],
    ];
    const w = [22, 14, 14];
    const line = (r) =>
      `<span class="k">${String(r[0]).padEnd(w[0])}</span>` +
      `<span class="v">${String(r[1]).padStart(w[1])}</span>` +
      `<span class="v">${String(r[2]).padStart(w[2])}</span>`;
    const body = rows.map(line).join("\n");
    const note = `<span class="verdict good">Merton's heavier tail (larger |skew| and excess kurtosis, deeper VaR) means more probability mass in extreme moves. An option is convex in the underlying, so that fatter tail raises its expected payoff — hence Merton prices above Black-Scholes for the same σ. Based on ${res.n_paths.toLocaleString()} simulated paths.</span>`;
    $("distStats").innerHTML = body + "\n" + note;
  }

  async function runDistributions() {
    const btn = $("distRunBtn");
    btn.disabled = true;
    const prev = btn.textContent;
    btn.textContent = "… simulating";
    try {
      const resp = await fetch("/api/distributions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readConfig()),
      });
      if (!resp.ok) throw new Error("server " + resp.status);
      const payload = await resp.json();
      distData = payload.result;
      renderPaths(distData);
      renderHist(distData);
      renderDistStats(distData);
    } catch (err) {
      $("distStats").textContent = "Distribution simulation failed: " + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
    }
  }

  // ---- Tabs ------------------------------------------------------------
  function switchTab(name) {
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".tab-panel").forEach((p) =>
      p.classList.toggle("hidden", p.id !== "tab-" + name));
    if (name === "dist") {
      // Lazily run the first time, and always resize charts for the new layout.
      if (!distData) runDistributions();
      else { Plotly.Plots.resize("pathChart"); Plotly.Plots.resize("histChart"); }
    }
  }

  // ---- Presets ---------------------------------------------------------
  function applyPreset(preset) {
    const set = (k, v) => { const el = $(k); if (el) el.value = v; };
    if (preset === "match") {
      set("pricing_model", "black_scholes");
      set("path_model", "gbm");
      set("half_spread", "0.15");
      set("lam", "1.0");
    } else if (preset === "steamroller") {
      // MM naively prices Black-Scholes while the world jumps hard & down.
      set("pricing_model", "black_scholes");
      set("path_model", "merton");
      set("lam", "4.0");
      set("jump_mean", "-0.12");
      set("jump_std", "0.18");
      set("half_spread", "0.15");
    }
    updateMatchHint();
    run();
  }

  // ---- Wire up ---------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    prefill();
    initPriceChart();
    initPnlChart();
    $("runBtn").addEventListener("click", run);
    $("stopBtn").addEventListener("click", () => { stopAnim(); setStatus("PAUSED", ""); });
    $("pricing_model").addEventListener("change", updateMatchHint);
    $("path_model").addEventListener("change", updateMatchHint);
    $("presetMatch").addEventListener("click", () => applyPreset("match"));
    $("presetSteamroller").addEventListener("click", () => applyPreset("steamroller"));

    // Tabs + distributions.
    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => switchTab(t.dataset.tab)));
    $("distRunBtn").addEventListener("click", runDistributions);
  });
})();
