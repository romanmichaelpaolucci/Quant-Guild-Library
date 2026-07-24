"""
app.py
======

Flask front-end for the Market Making Simulator.

Endpoints
---------
GET  /            -> the single-page trading-desk UI.
POST /api/simulate -> accepts the control-panel parameters as JSON, runs one
                      simulation via `simulator.run_simulation`, and returns the
                      full time series + summary stats as JSON. The browser then
                      "replays" the series tick-by-tick with Plotly so it looks
                      like a live ticking market.

Design note: the heavy numerical work happens once server-side; the animation
is purely a client-side replay of the returned arrays. This keeps the sim
deterministic (given a seed) and the UI buttery-smooth.

Run:
    pip install -r requirements.txt
    python app.py
    # then open http://127.0.0.1:5001
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from simulator import SimConfig, run_simulation, simulate_return_distributions

app = Flask(__name__)


def _config_from_payload(data: dict) -> SimConfig:
    """Build a validated SimConfig from the JSON the front-end posts.

    Every field is coerced to the right type with a fallback to the dataclass
    default, so a malformed or partial payload can never crash the engine.
    """
    defaults = SimConfig()

    def num(key, cast=float):
        try:
            v = data.get(key, getattr(defaults, key))
            return cast(v) if v is not None else getattr(defaults, key)
        except (TypeError, ValueError):
            return getattr(defaults, key)

    # Seed may be blank -> nondeterministic run.
    seed_raw = data.get("seed", defaults.seed)
    if seed_raw in ("", None):
        seed = None
    else:
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = defaults.seed

    pricing_sigma_raw = data.get("pricing_sigma", None)
    if pricing_sigma_raw in ("", None):
        pricing_sigma = None
    else:
        try:
            pricing_sigma = float(pricing_sigma_raw)
        except (TypeError, ValueError):
            pricing_sigma = None

    return SimConfig(
        S0=num("S0"),
        K=num("K"),
        T=num("T"),
        horizon=num("horizon"),
        option_type=str(data.get("option_type", defaults.option_type)),
        path_model=str(data.get("path_model", defaults.path_model)),
        mu=num("mu"),
        sigma=num("sigma"),
        lam=num("lam"),
        jump_mean=num("jump_mean"),
        jump_std=num("jump_std"),
        pricing_model=str(data.get("pricing_model", defaults.pricing_model)),
        r=num("r"),
        pricing_sigma=pricing_sigma,
        half_spread=num("half_spread"),
        quote_size=num("quote_size"),
        fill_intensity=num("fill_intensity"),
        risk_aversion=num("risk_aversion"),
        hedge_steps=num("hedge_steps", int),
        n_steps=num("n_steps", int),
        seed=seed,
    )


@app.route("/")
def index():
    """Serve the single-page app with sensible starting defaults."""
    return render_template("index.html", defaults=SimConfig().__dict__)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run one simulation and return the full result as JSON."""
    data = request.get_json(silent=True) or {}
    cfg = _config_from_payload(data)
    # Guardrails to keep the browser responsive / the pricer well-behaved.
    cfg.n_steps = int(max(10, min(cfg.n_steps, 2000)))
    # Each fill spins up a delta-hedged sub-simulation with `hedge_steps` rebalances,
    # so cap it to keep the per-request compute (and latency) bounded.
    cfg.hedge_steps = int(max(2, min(cfg.hedge_steps, 200)))
    cfg.T = float(max(1e-3, cfg.T))
    cfg.horizon = float(max(1e-3, cfg.horizon))
    result = run_simulation(cfg)
    return jsonify({"config": cfg.__dict__, "result": result.to_dict()})


@app.route("/api/distributions", methods=["POST"])
def distributions():
    """Simulate GBM *and* Merton under the same params and return their
    terminal-return distributions (histograms + sample paths + moments).

    This powers the "Return Distributions" tab, which visualises *why* the two
    models price options differently: Merton's jump component fattens the tail.
    """
    data = request.get_json(silent=True) or {}
    cfg = _config_from_payload(data)
    cfg.n_steps = int(max(10, min(cfg.n_steps, 2000)))
    cfg.T = float(max(1e-3, cfg.T))
    cfg.horizon = float(max(1e-3, cfg.horizon))

    # Optional client override for path count, clamped for responsiveness.
    try:
        n_paths = int(data.get("n_paths", 15000))
    except (TypeError, ValueError):
        n_paths = 15000
    n_paths = max(1000, min(n_paths, 60000))

    payload = simulate_return_distributions(cfg, n_paths=n_paths)
    return jsonify({"config": cfg.__dict__, "result": payload})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
