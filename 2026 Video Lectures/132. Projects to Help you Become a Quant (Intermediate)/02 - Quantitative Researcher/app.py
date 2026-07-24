"""
app.py
======

Flask web dashboard for the **PDE Solver** research project.

The app ties together three independent views of an option's value:

* :mod:`fd_solver`  — solves the governing partial differential equation with
  finite differences (Crank-Nicolson / implicit / explicit) and returns the
  full price surface ``V(S, t)``.
* :mod:`mc_pricer`  — prices the *same* instrument by risk-neutral Monte-Carlo
  simulation, with a 95% confidence interval.
* :mod:`analytics`  — supplies the closed-form price where one exists.

The user selects a pricing framework (vanilla, barrier, digital, American),
enters market/contract/grid parameters, and the dashboard shows the FD surface,
the price-vs-spot slice with payoff overlay, and a side-by-side FD/MC/analytic
comparison so the numerical methods can be seen to agree.

Run:  python app.py    then open  http://127.0.0.1:5003
"""

from __future__ import annotations

import traceback

import numpy as np
from flask import Flask, jsonify, render_template, request

import analytics
import fd_solver
import mc_pricer

app = Flask(__name__)

# Cap grid / MC sizes so the interactive dashboard stays responsive and the
# server cannot be asked to allocate an unreasonable amount of memory.
_MAX_M = 600
_MAX_N = 600
_MAX_PATHS = 500_000
_MAX_STEPS = 1000
# Downsample the surface to at most this many points per axis for the browser.
_SURFACE_MAX = 80


@app.route("/")
def index():
    """Render the single-page dashboard with the framework catalogue."""
    return render_template("index.html", frameworks=fd_solver.FRAMEWORKS)


def _clean_params(data: dict) -> dict:
    """Validate and coerce the JSON request body into solver parameters."""
    framework = data.get("framework", "vanilla")
    if framework not in fd_solver.FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework}")

    def fnum(key, default, lo=None, hi=None):
        val = float(data.get(key, default))
        if not np.isfinite(val):
            raise ValueError(f"{key} must be finite")
        if lo is not None and val < lo:
            raise ValueError(f"{key} must be >= {lo}")
        if hi is not None and val > hi:
            raise ValueError(f"{key} must be <= {hi}")
        return val

    p = {
        "framework": framework,
        "S0": fnum("S0", 100.0, lo=1e-6),
        "K": fnum("K", 100.0, lo=1e-6),
        "r": fnum("r", 0.05, lo=-1.0, hi=1.0),
        "q": fnum("q", 0.0, lo=-1.0, hi=1.0),
        "sigma": fnum("sigma", 0.2, lo=1e-4, hi=5.0),
        "T": fnum("T", 1.0, lo=1e-3, hi=50.0),
        "option_type": "put" if str(data.get("option_type", "call")).lower() == "put" else "call",
        "scheme": data.get("scheme", "crank-nicolson"),
        "M": int(max(10, min(_MAX_M, int(data.get("M", 200))))),
        "N": int(max(10, min(_MAX_N, int(data.get("N", 200))))),
        "n_paths": int(max(1000, min(_MAX_PATHS, int(data.get("n_paths", 100000))))),
        "n_steps": int(max(10, min(_MAX_STEPS, int(data.get("n_steps", 252))))),
    }
    if p["scheme"] not in ("explicit", "implicit", "crank-nicolson"):
        p["scheme"] = "crank-nicolson"

    if framework == "barrier":
        p["barrier"] = fnum("barrier", 130.0, lo=1e-6)
        bt = str(data.get("barrier_type", "up-and-out"))
        p["barrier_type"] = bt if bt in ("up-and-out", "down-and-out") else "up-and-out"
    if framework == "digital":
        p["cash"] = fnum("cash", 1.0, lo=0.0)
    return p


def _downsample(arr: np.ndarray, n: int) -> np.ndarray:
    """Return indices selecting at most ``n`` roughly-even samples from ``arr``."""
    if len(arr) <= n:
        return np.arange(len(arr))
    return np.linspace(0, len(arr) - 1, n).round().astype(int)


@app.route("/api/price", methods=["POST"])
def api_price():
    """Solve FD + MC + analytic for the requested instrument and return JSON."""
    try:
        data = request.get_json(force=True) or {}
        p = _clean_params(data)
        framework = p["framework"]

        # --- Finite-difference PDE solve -----------------------------------
        fd = fd_solver.solve(framework, p, M=p["M"], N=p["N"], scheme=p["scheme"])

        # --- Monte-Carlo cross-check ---------------------------------------
        mc = mc_pricer.price(framework, p, n_paths=p["n_paths"],
                             n_steps=p["n_steps"], seed=42)

        # --- Closed form (if available) ------------------------------------
        analytic = analytics.analytic_price(framework, p)

        # --- Surface (downsampled) for the 3D / heatmap plots --------------
        si = _downsample(fd.S, _SURFACE_MAX)
        ti = _downsample(fd.t, _SURFACE_MAX)
        S_ds = fd.S[si]
        t_ds = fd.t[ti]
        V_ds = fd.V[np.ix_(ti, si)]

        # --- Price-vs-spot slice at t=0 (today) with payoff overlay --------
        V_today = fd.V[np.argmin(fd.t)]  # t closest to 0
        # Terminal payoff on the same S grid.
        inst = fd_solver.build_instrument(framework, p, fd.meta["S_max"])
        payoff = np.asarray(inst.terminal(fd.S), dtype=float)

        # --- Comparison numbers -------------------------------------------
        fd_price = fd.price
        mc_price = mc.price
        diff_fd_mc = fd_price - mc_price
        rel_fd_mc = diff_fd_mc / mc_price if abs(mc_price) > 1e-12 else float("nan")
        fd_in_ci = mc.ci_low <= fd_price <= mc.ci_high

        resp = {
            "ok": True,
            "framework": framework,
            "label": fd.meta["label"],
            "prices": {
                "fd": fd_price,
                "mc": mc_price,
                "mc_se": mc.std_error,
                "mc_ci_low": mc.ci_low,
                "mc_ci_high": mc.ci_high,
                "analytic": analytic,
                "diff_fd_mc": diff_fd_mc,
                "rel_fd_mc": rel_fd_mc,
                "fd_in_ci": bool(fd_in_ci),
            },
            "fd_meta": {
                "scheme": fd.scheme, "M": fd.meta["M"], "N": fd.meta["N"],
                "S_min": fd.meta["S_min"], "S_max": fd.meta["S_max"],
                "stable": fd.stable,
            },
            "mc_meta": {"n_paths": mc.n_paths, "n_steps": mc.n_steps},
            "surface": {
                "S": S_ds.tolist(),
                "t": t_ds.tolist(),
                "V": V_ds.tolist(),
            },
            "slice": {
                "S": fd.S.tolist(),
                "V": V_today.tolist(),
                "payoff": payoff.tolist(),
                "S0": p["S0"],
            },
            "params": {k: p[k] for k in p if k != "framework"},
        }
        return jsonify(resp)
    except Exception as exc:  # pragma: no cover - defensive UI error path
        return jsonify({"ok": False, "error": str(exc),
                        "trace": traceback.format_exc()}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5003)
