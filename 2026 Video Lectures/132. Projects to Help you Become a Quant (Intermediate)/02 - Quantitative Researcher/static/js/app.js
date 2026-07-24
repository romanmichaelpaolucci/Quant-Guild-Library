/* =========================================================================
   PDE Solver dashboard front-end.
   Gathers inputs, calls /api/price, and renders the FD price surface, the
   value-vs-spot slice with payoff overlay, and the FD/MC/analytic comparison.
   ========================================================================= */

const FRAMEWORKS = JSON.parse(document.getElementById('frameworks-data').textContent);

const COLORS = {
  fd: '#4fd1c5', mc: '#a78bfa', an: '#fbbf24',
  payoff: '#f472b6', grid: '#232b3a', text: '#9aa6bd', spot: '#f87171',
};

let currentFramework = 'vanilla';
let lastSurface = null;      // cache so the 3D/heatmap toggle needn't re-solve
let surfaceView = 'surface';

/* ----------------------------- Layout helpers ---------------------------- */
function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: COLORS.text, family: 'Inter, sans-serif', size: 11 },
    margin: { l: 55, r: 20, t: 20, b: 45 },
    showlegend: false,
  }, extra);
}
const AXIS = { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid };
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/* ------------------------------ Framework UI ----------------------------- */
function selectFramework(fw) {
  currentFramework = fw;
  document.querySelectorAll('.seg-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.framework === fw));

  const meta = FRAMEWORKS[fw];
  document.getElementById('framework-name').textContent = meta.name;
  document.getElementById('result-title').textContent = meta.name;

  const needsBarrier = meta.needs_barrier;
  document.querySelectorAll('.barrier-only').forEach(
    el => el.style.display = needsBarrier ? '' : 'none');
  document.querySelectorAll('.digital-only').forEach(
    el => el.style.display = fw === 'digital' ? '' : 'none');

  renderEquation(meta);
}

function renderEquation(meta) {
  // Use textContent, not innerHTML: LaTeX like the digital-option indicator
  // \mathbf{1}\{S<K\} contains '<', which the browser would otherwise parse as
  // an HTML tag and silently drop the rest of the equation. MathJax still
  // finds the \(...\) / \[...\] delimiters in a plain text node.
  document.getElementById('eq-pde').textContent = `\\[ ${meta.pde} \\]`;
  document.getElementById('eq-terminal').textContent = `\\( ${meta.terminal} \\)`;
  document.getElementById('eq-boundary').textContent = `\\( ${meta.boundary} \\)`;
  document.getElementById('pde-badge').textContent = 'PDE loaded';
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([document.getElementById('eq-pde'),
      document.getElementById('eq-terminal'),
      document.getElementById('eq-boundary')]);
  }
}

/* ------------------------------- Gather form ----------------------------- */
function gatherParams() {
  const val = id => document.getElementById(id).value;
  const p = {
    framework: currentFramework,
    option_type: val('option_type'),
    S0: val('S0'), K: val('K'), r: val('r'), q: val('q'),
    sigma: val('sigma'), T: val('T'),
    scheme: val('scheme'), M: val('M'), N: val('N'),
    n_paths: val('n_paths'), n_steps: val('n_steps'),
  };
  if (currentFramework === 'barrier') {
    p.barrier = val('barrier');
    p.barrier_type = val('barrier_type');
  }
  if (currentFramework === 'digital') p.cash = val('cash');
  return p;
}

/* --------------------------------- Solve --------------------------------- */
async function solve() {
  const btn = document.getElementById('solve-btn');
  btn.disabled = true;
  btn.querySelector('.solve-label').textContent = 'Solving';
  btn.querySelector('.solve-spinner').style.display = 'inline-block';
  document.getElementById('error-banner').style.display = 'none';

  try {
    const resp = await fetch('/api/price', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(gatherParams()),
    });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Solver error');
    render(data);
  } catch (err) {
    const banner = document.getElementById('error-banner');
    banner.style.display = 'block';
    banner.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.querySelector('.solve-label').textContent = 'Solve PDE';
    btn.querySelector('.solve-spinner').style.display = 'none';
  }
}

/* -------------------------------- Rendering ------------------------------- */
function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(d);
}

function render(data) {
  const pr = data.prices;
  document.getElementById('result-sub').textContent =
    `${data.label} · FD (${data.fd_meta.scheme}, ${data.fd_meta.M}×${data.fd_meta.N}) ` +
    `vs MC (${data.mc_meta.n_paths.toLocaleString()} paths)`;

  // Price cards
  document.getElementById('price-fd').textContent = fmt(pr.fd);
  document.getElementById('meta-fd').textContent =
    `${data.fd_meta.scheme}${data.fd_meta.stable ? '' : ' · UNSTABLE'}`;

  document.getElementById('price-mc').textContent = fmt(pr.mc);
  document.getElementById('meta-mc').innerHTML =
    `95% CI [${fmt(pr.mc_ci_low)}, ${fmt(pr.mc_ci_high)}]`;

  document.getElementById('price-an').textContent =
    pr.analytic === null ? 'n/a' : fmt(pr.analytic);
  document.getElementById('meta-an').textContent =
    pr.analytic === null ? 'no closed form' : 'analytic';

  document.getElementById('price-diff').textContent = fmt(pr.diff_fd_mc);
  const inCi = pr.fd_in_ci;
  document.getElementById('meta-diff').innerHTML =
    `rel ${fmt(pr.rel_fd_mc * 100, 3)}% · ` +
    `<span class="${inCi ? 'badge-ok' : 'badge-bad'}">` +
    `${inCi ? 'FD ∈ MC CI ✓' : 'outside CI'}</span>`;

  // Charts
  lastSurface = data.surface;
  drawSurface();
  drawSlice(data.slice);
  drawCompare(pr);
}

function drawSurface() {
  if (!lastSurface) return;
  const { S, t, V } = lastSurface;
  const el = document.getElementById('surface-plot');
  if (surfaceView === 'surface') {
    Plotly.react(el, [{
      type: 'surface', x: S, y: t, z: V,
      colorscale: [[0, '#0d1b2a'], [0.5, '#1f6f6b'], [1, '#4fd1c5']],
      showscale: false, contours: { z: { show: true, usecolormap: true,
        highlightcolor: '#ffffff', project: { z: true } } },
    }], baseLayout({
      scene: {
        xaxis: Object.assign({ title: 'Spot S' }, AXIS),
        yaxis: Object.assign({ title: 'Time t' }, AXIS),
        zaxis: Object.assign({ title: 'Value V' }, AXIS),
        camera: { eye: { x: 1.6, y: -1.5, z: 0.9 } },
        bgcolor: 'rgba(0,0,0,0)',
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
    }), PLOT_CONFIG);
  } else {
    Plotly.react(el, [{
      type: 'heatmap', x: S, y: t, z: V,
      colorscale: [[0, '#0d1b2a'], [0.5, '#1f6f6b'], [1, '#4fd1c5']],
      colorbar: { thickness: 12, outlinewidth: 0, tickfont: { size: 9 } },
    }], baseLayout({
      xaxis: Object.assign({ title: 'Spot S' }, AXIS),
      yaxis: Object.assign({ title: 'Time t' }, AXIS),
    }), PLOT_CONFIG);
  }
}

function drawSlice(slice) {
  const el = document.getElementById('slice-plot');
  const traces = [
    { x: slice.S, y: slice.payoff, mode: 'lines', name: 'Payoff',
      line: { color: COLORS.payoff, width: 1.5, dash: 'dot' } },
    { x: slice.S, y: slice.V, mode: 'lines', name: 'V(S, 0)',
      line: { color: COLORS.fd, width: 2.5 },
      fill: 'tonexty', fillcolor: 'rgba(79,209,197,0.06)' },
  ];
  const maxV = Math.max(...slice.V, ...slice.payoff);
  const layout = baseLayout({
    xaxis: Object.assign({ title: 'Spot S' }, AXIS),
    yaxis: Object.assign({ title: 'Option value' }, AXIS),
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
    shapes: [{
      type: 'line', x0: slice.S0, x1: slice.S0, y0: 0, y1: maxV * 1.02,
      line: { color: COLORS.spot, width: 1.2, dash: 'dash' },
    }],
    annotations: [{
      x: slice.S0, y: maxV * 1.02, text: `S₀=${slice.S0}`, showarrow: false,
      font: { color: COLORS.spot, size: 10 }, yanchor: 'bottom',
    }],
  });
  Plotly.react(el, traces, layout, PLOT_CONFIG);
}

function drawCompare(pr) {
  const el = document.getElementById('compare-plot');
  const labels = ['FD', 'MC', 'Analytic'];
  const vals = [pr.fd, pr.mc, pr.analytic === null ? 0 : pr.analytic];
  const colors = [COLORS.fd, COLORS.mc, COLORS.an];
  const text = [fmt(pr.fd), fmt(pr.mc), pr.analytic === null ? 'n/a' : fmt(pr.analytic)];

  // Error bar on MC = half-width of the 95% CI.
  const mcErr = (pr.mc_ci_high - pr.mc_ci_low) / 2;
  const traces = [{
    type: 'bar', x: labels, y: vals, text, textposition: 'outside',
    textfont: { color: COLORS.text, size: 12 },
    marker: { color: colors, line: { width: 0 } },
    error_y: { type: 'data', array: [0, mcErr, 0], visible: true,
      color: COLORS.mc, thickness: 1.5, width: 6 },
  }];
  const layout = baseLayout({
    xaxis: Object.assign({}, AXIS),
    yaxis: Object.assign({ title: 'Price', rangemode: 'tozero' }, AXIS),
    bargap: 0.5,
  });
  Plotly.react(el, traces, layout, PLOT_CONFIG);
}

/* ------------------------------- Wire events ----------------------------- */
document.getElementById('framework-seg').addEventListener('click', e => {
  const btn = e.target.closest('.seg-btn');
  if (btn) selectFramework(btn.dataset.framework);
});
document.getElementById('solve-btn').addEventListener('click', solve);
document.querySelectorAll('.toggle-btn').forEach(b =>
  b.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    surfaceView = b.dataset.view;
    drawSurface();
  }));

// Solve on Enter from any input.
document.querySelectorAll('.control').forEach(c =>
  c.addEventListener('keydown', e => { if (e.key === 'Enter') solve(); }));

// Initial state.
selectFramework('vanilla');
window.addEventListener('load', () => setTimeout(solve, 400));
