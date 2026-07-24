"""
Build a standalone interactive dashboard comparing Roman Paolucci's 2025
strategy ("Crisis Alpha") against the SPY price path.

Inputs (same folder):
  - Roman_Paolucci_2025_2025.csv : IBKR PortfolioAnalyst TWR cumulative-return export
  - spy_prices_2025.csv          : Daily SPY OHLCV (from spy_data.py)

Output:
  - crisis_alpha_dashboard.html  : self-contained interactive dashboard (Plotly)

Run:
  python build_dashboard.py
Then open crisis_alpha_dashboard.html in a browser.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PERF_CSV = SCRIPT_DIR / "Roman_Paolucci_2025_2025.csv"
SPY_CSV = SCRIPT_DIR / "spy_prices_2025.csv"
OUT_HTML = SCRIPT_DIR / "crisis_alpha_dashboard.html"

TRADING_DAYS = 252
STRATEGY_NAME = "Crisis Alpha (Roman Paolucci)"
ANN = np.sqrt(TRADING_DAYS)
RV_WINDOW = 21          # ~1 month realized-vol window
EWMA_LAMBDA = 0.94      # RiskMetrics decay


# --------------------------------------------------------------------------- #
# Data loading / cleaning
# --------------------------------------------------------------------------- #
def load_strategy() -> pd.DataFrame:
    """Parse the IBKR PortfolioAnalyst export into a clean date/return frame.

    The export is a multi-section CSV. We only want rows from the
    'Cumulative Performance Statistics' section tagged 'Data', which carry
    columns: [section, 'Data', date(MM/DD/YY), account, cumulative_return_pct].
    """
    rows: list[tuple[pd.Timestamp, float]] = []
    with PERF_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for parts in csv.reader(fh):
            if len(parts) < 5:
                continue
            if parts[0] != "Cumulative Performance Statistics" or parts[1] != "Data":
                continue
            try:
                dt = pd.to_datetime(parts[2], format="%m/%d/%y")
                ret = float(parts[4])
            except (ValueError, TypeError):
                continue
            rows.append((dt, ret))

    df = pd.DataFrame(rows, columns=["date", "cum_return_pct"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # Growth index (Jan 1 == 1.0). cum_return_pct is a percent TWR figure.
    df["growth"] = 1.0 + df["cum_return_pct"] / 100.0
    return df


def load_spy() -> pd.DataFrame:
    df = pd.read_csv(SPY_CSV, parse_dates=["Date"]).rename(columns={"Date": "date"})
    df = df[df["date"].dt.year == 2025].copy()
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["growth"] = df["close"] / df["close"].iloc[0]
    return df


def align(strategy: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on common trading dates and rebase both to 0% at day one."""
    merged = pd.merge(
        strategy[["date", "growth"]].rename(columns={"growth": "strat_growth"}),
        spy[["date", "open", "high", "low", "close", "volume", "growth"]].rename(
            columns={"growth": "spy_growth"}
        ),
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    # Rebase so both start at exactly 0% on the first common date.
    merged["strat_growth"] /= merged["strat_growth"].iloc[0]
    merged["spy_growth"] /= merged["spy_growth"].iloc[0]

    merged["strat_cum_pct"] = (merged["strat_growth"] - 1.0) * 100.0
    merged["spy_cum_pct"] = (merged["spy_growth"] - 1.0) * 100.0

    merged["strat_ret"] = merged["strat_growth"].pct_change().fillna(0.0)
    merged["spy_ret"] = merged["spy_growth"].pct_change().fillna(0.0)

    merged["strat_dd"] = drawdown(merged["strat_growth"])
    merged["spy_dd"] = drawdown(merged["spy_growth"])
    return merged


def drawdown(growth: pd.Series) -> pd.Series:
    peak = growth.cummax()
    return (growth / peak - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Volatility: realized proxies + GARCH conditional forecast
# --------------------------------------------------------------------------- #
def rolling_rv(returns: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Close-to-close realized volatility, rolling & annualized (%)."""
    return returns.rolling(window).std(ddof=0) * ANN * 100.0


def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA volatility, annualized (%). Instantaneous proxy."""
    r = returns.to_numpy(dtype=float)
    var = np.empty_like(r)
    seed = np.var(r[: RV_WINDOW]) if len(r) >= RV_WINDOW else np.var(r)
    prev = seed
    for i, x in enumerate(r):
        prev = lam * prev + (1.0 - lam) * x * x
        var[i] = prev
    return pd.Series(np.sqrt(var) * ANN * 100.0, index=returns.index)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Parkinson high-low range volatility, rolling & annualized (%)."""
    hl = np.log(high / low) ** 2
    daily_var = hl / (4.0 * np.log(2.0))
    return np.sqrt(daily_var.rolling(window).mean() * TRADING_DAYS) * 100.0


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = RV_WINDOW,
) -> pd.Series:
    """Garman-Klass OHLC volatility, rolling & annualized (%)."""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    daily_var = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return np.sqrt(daily_var.rolling(window).mean() * TRADING_DAYS) * 100.0


def garch_conditional_vol(returns: pd.Series) -> tuple[pd.Series, float, dict]:
    """Fit GARCH(1,1) and return in-sample conditional vol (annualized %),
    the 1-step-ahead forecast, and the fitted parameters."""
    from arch import arch_model  # imported lazily so the script still parses

    r = returns.dropna() * 100.0  # arch prefers percent-scale returns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
            disp="off"
        )
    cond = res.conditional_volatility / 100.0 * ANN * 100.0  # daily% -> ann %
    cond = cond.reindex(returns.index)
    fc = res.forecast(horizon=1, reindex=False)
    next_var = float(fc.variance.iloc[-1, 0])  # in percent^2
    next_vol = np.sqrt(next_var) / 100.0 * ANN * 100.0
    params = {k: float(v) for k, v in res.params.items()}
    return cond, float(next_vol), params


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS) * 100.0)


def sharpe(returns: pd.Series) -> float:
    mu = returns.mean() * TRADING_DAYS
    sd = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    return float(mu / sd) if sd > 0 else 0.0


def return_stats(growth: pd.Series) -> dict:
    """Total return, geometric (CAGR), arithmetic, and volatility drag.

    Volatility drag is the gap between the arithmetic annualized return and
    the geometric (compounded) return an investor actually keeps. It grows
    with variance (~= sigma^2 / 2), so a more volatile equity curve leaks
    more of its average return to the path.
    """
    rets = growth.pct_change().dropna()
    steps = int(len(rets))
    total = float(growth.iloc[-1] / growth.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (TRADING_DAYS / steps) - 1.0)
    vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))
    # Volatility drag = sigma^2 / 2 (the return the path costs you). By the
    # standard decomposition, arithmetic ~= geometric (CAGR) + drag.
    drag = (vol ** 2) / 2.0
    arith = cagr + drag
    return {
        "total": total * 100.0,
        "cagr": cagr * 100.0,
        "arith": arith * 100.0,
        "vol": vol * 100.0,
        "drag": drag * 100.0,
    }


def find_crisis_window(df: pd.DataFrame) -> dict:
    """Largest peak-to-trough SPY drawdown episode (the tariff shock)."""
    growth = df["spy_growth"].to_numpy()
    dates = df["date"].to_numpy()
    peak_idx = 0
    trough_idx = 0
    best_dd = 0.0
    running_peak_idx = 0
    for i in range(len(growth)):
        if growth[i] > growth[running_peak_idx]:
            running_peak_idx = i
        dd = growth[i] / growth[running_peak_idx] - 1.0
        if dd < best_dd:
            best_dd = dd
            trough_idx = i
            peak_idx = running_peak_idx

    peak_date = pd.Timestamp(dates[peak_idx])
    trough_date = pd.Timestamp(dates[trough_idx])
    window = df[(df["date"] >= peak_date) & (df["date"] <= trough_date)]
    strat_move = (
        window["strat_growth"].iloc[-1] / window["strat_growth"].iloc[0] - 1.0
    ) * 100.0
    spy_move = (
        window["spy_growth"].iloc[-1] / window["spy_growth"].iloc[0] - 1.0
    ) * 100.0

    # Rebound phase: from the trough to year-end, where the alpha is earned.
    rebound = df[df["date"] >= trough_date]
    strat_rebound = (
        rebound["strat_growth"].iloc[-1] / rebound["strat_growth"].iloc[0] - 1.0
    ) * 100.0
    spy_rebound = (
        rebound["spy_growth"].iloc[-1] / rebound["spy_growth"].iloc[0] - 1.0
    ) * 100.0

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "spy_move": float(spy_move),
        "strat_move": float(strat_move),
        "crisis_alpha": float(strat_move - spy_move),
        "spy_drawdown": float(best_dd * 100.0),
        "strat_rebound": float(strat_rebound),
        "spy_rebound": float(spy_rebound),
        "rebound_alpha": float(strat_rebound - spy_rebound),
    }


def build_metrics(df: pd.DataFrame, crisis: dict) -> dict:
    strat = return_stats(df["strat_growth"])
    spy = return_stats(df["spy_growth"])
    return {
        "start_date": df["date"].iloc[0].strftime("%b %d, %Y"),
        "end_date": df["date"].iloc[-1].strftime("%b %d, %Y"),
        "n_days": int(len(df)),
        # Total return
        "strat_final": strat["total"],
        "spy_final": spy["total"],
        # CAGR (geometric, annualized)
        "strat_cagr": strat["cagr"],
        "spy_cagr": spy["cagr"],
        # Arithmetic annualized return
        "strat_arith": strat["arith"],
        "spy_arith": spy["arith"],
        # Volatility drag (arithmetic - geometric)
        "strat_drag": strat["drag"],
        "spy_drag": spy["drag"],
        # Volatility
        "strat_vol": strat["vol"],
        "spy_vol": spy["vol"],
        # Risk
        "strat_maxdd": float(df["strat_dd"].min()),
        "spy_maxdd": float(df["spy_dd"].min()),
        "strat_sharpe": sharpe(df["strat_ret"]),
        "spy_sharpe": sharpe(df["spy_ret"]),
        # Ending value of $10k
        "strat_end10k": float((1.0 + strat["total"] / 100.0) * 10000.0),
        "spy_end10k": float((1.0 + spy["total"] / 100.0) * 10000.0),
        "crisis": crisis,
    }


def build_leverage(df: pd.DataFrame, betas: list[float]) -> dict:
    """Cumulative return paths for a constant-beta levered SPY exposure,
    i.e. daily return = beta * SPY daily return (financing ignored). Shows
    the net effect of pure leverage across the 2025 series vs the strategy."""
    spy_ret = df["spy_ret"].to_numpy()
    paths: dict[str, list] = {}
    finals: dict[str, float] = {}
    mdds: dict[str, float] = {}
    for b in betas:
        growth = np.cumprod(1.0 + b * spy_ret)
        cum_pct = (growth - 1.0) * 100.0
        dd = (growth / np.maximum.accumulate(growth) - 1.0) * 100.0
        key = f"{b:g}"
        paths[key] = [round(float(v), 4) for v in cum_pct]
        finals[key] = round(float(cum_pct[-1]), 2)
        mdds[key] = round(float(dd.min()), 2)
    return {
        "betas": [f"{b:g}" for b in betas],
        "paths": paths,
        "finals": finals,
        "mdds": mdds,
    }


def _clean(series: pd.Series) -> list:
    """Round and convert NaN -> None so Plotly gaps the line cleanly."""
    return [None if pd.isna(v) else round(float(v), 3) for v in series]


def build_vol(df: pd.DataFrame) -> dict:
    """Realized-vol proxies + GARCH conditional vol for strategy and SPY."""
    strat_r = df["strat_ret"]
    spy_r = df["spy_ret"]

    strat_garch, strat_fc, strat_p = garch_conditional_vol(strat_r)
    spy_garch, spy_fc, spy_p = garch_conditional_vol(spy_r)

    def persistence(p: dict) -> float:
        return p.get("alpha[1]", 0.0) + p.get("beta[1]", 0.0)

    return {
        "series": {
            "strat_rv": _clean(rolling_rv(strat_r)),
            "strat_ewma": _clean(ewma_vol(strat_r)),
            "strat_garch": _clean(strat_garch),
            "spy_rv": _clean(rolling_rv(spy_r)),
            "spy_ewma": _clean(ewma_vol(spy_r)),
            "spy_parkinson": _clean(parkinson_vol(df["high"], df["low"])),
            "spy_gk": _clean(
                garman_klass_vol(df["open"], df["high"], df["low"], df["close"])
            ),
            "spy_garch": _clean(spy_garch),
        },
        "meta": {
            "window": RV_WINDOW,
            "lam": EWMA_LAMBDA,
            "strat_forecast": round(strat_fc, 2),
            "spy_forecast": round(spy_fc, 2),
            "strat_persistence": round(persistence(strat_p), 3),
            "spy_persistence": round(persistence(spy_p), 3),
            "strat_current_garch": round(
                float(strat_garch.dropna().iloc[-1]), 2
            ),
            "spy_current_garch": round(float(spy_garch.dropna().iloc[-1]), 2),
        },
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def render_html(df: pd.DataFrame, m: dict, vol: dict, lev: dict) -> str:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    metrics_safe = {k: v for k, v in m.items() if k != "crisis"}
    payload = {
        "dates": dates,
        "strat_cum": df["strat_cum_pct"].round(4).tolist(),
        "spy_cum": df["spy_cum_pct"].round(4).tolist(),
        "strat_growth10k": (df["strat_growth"] * 10000).round(2).tolist(),
        "spy_growth10k": (df["spy_growth"] * 10000).round(2).tolist(),
        "strat_dd": df["strat_dd"].round(4).tolist(),
        "spy_dd": df["spy_dd"].round(4).tolist(),
        "spy_close": df["close"].round(2).tolist(),
        "spy_open": df["open"].round(2).tolist(),
        "spy_high": df["high"].round(2).tolist(),
        "spy_low": df["low"].round(2).tolist(),
        "vol": vol["series"],
        "vol_meta": vol["meta"],
        "lev": lev,
        "metrics": metrics_safe,
        "strategy_name": STRATEGY_NAME,
        "crisis": {
            "peak_date": m["crisis"]["peak_date"].strftime("%Y-%m-%d"),
            "trough_date": m["crisis"]["trough_date"].strftime("%Y-%m-%d"),
            "peak_label": m["crisis"]["peak_date"].strftime("%b %d"),
            "trough_label": m["crisis"]["trough_date"].strftime("%b %d"),
            "spy_move": m["crisis"]["spy_move"],
            "strat_move": m["crisis"]["strat_move"],
            "crisis_alpha": m["crisis"]["crisis_alpha"],
            "spy_drawdown": m["crisis"]["spy_drawdown"],
            "strat_rebound": m["crisis"]["strat_rebound"],
            "spy_rebound": m["crisis"]["spy_rebound"],
            "rebound_alpha": m["crisis"]["rebound_alpha"],
        },
    }
    data_json = json.dumps(payload)
    return HTML_TEMPLATE.replace("__DATA__", data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Crisis Alpha 2025 vs SPY</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{
    --bg:#0b0f17; --panel:#131a26; --panel2:#0f1520; --line:#233046;
    --txt:#e6edf7; --muted:#8ea0bd; --strat:#4dd6a8; --spy:#7aa2ff;
    --danger:#ff6b6b; --gold:#f6c453; --accent:#4dd6a8;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#16233a 0%,var(--bg) 55%);
    color:var(--txt);font-family:"Segoe UI",Inter,system-ui,-apple-system,sans-serif;}
  .wrap{max-width:1240px;margin:0 auto;padding:32px 22px 60px;}
  header{display:flex;flex-direction:column;gap:6px;margin-bottom:22px;}
  h1{font-size:30px;margin:0;letter-spacing:.3px;}
  h1 .hl{color:var(--strat);}
  .sub{color:var(--muted);font-size:15px;}
  .period{color:var(--muted);font-size:13px;margin-top:2px;}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:22px 0;}
  .kpi{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-radius:14px;padding:16px 18px;position:relative;overflow:hidden;}
  .kpi .lbl{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.6px;}
  .kpi .val{font-size:26px;font-weight:700;margin-top:8px;}
  .kpi .foot{font-size:12px;color:var(--muted);margin-top:6px;}
  .pos{color:var(--strat);} .neg{color:var(--danger);} .blue{color:var(--spy);} .gold{color:var(--gold);}
  .kpi.hero{border-color:rgba(77,214,168,.5);box-shadow:0 0 0 1px rgba(77,214,168,.12),0 8px 30px rgba(77,214,168,.08);}
  .kpi.hero-red{border-color:rgba(255,107,107,.5);box-shadow:0 0 0 1px rgba(255,107,107,.12),0 8px 30px rgba(255,107,107,.08);}
  .card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-radius:16px;padding:16px 16px 6px;margin:18px 0;}
  .card h2{margin:2px 6px 4px;font-size:17px;}
  .card p.note{margin:0 6px 10px;color:var(--muted);font-size:13px;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
  @media(max-width:900px){.grid2{grid-template-columns:1fr;}}
  .plot{width:100%;}
  .banner{background:linear-gradient(90deg,rgba(255,107,107,.10),rgba(246,196,83,.08));
    border:1px solid rgba(246,196,83,.35);border-radius:14px;padding:16px 18px;margin:18px 0;}
  .banner .t{font-size:14px;color:var(--gold);text-transform:uppercase;letter-spacing:.8px;}
  .banner .b{font-size:15px;color:var(--txt);margin-top:8px;line-height:1.5;}
  .banner b.pos{color:var(--strat);} .banner b.neg{color:var(--danger);}
  footer{color:var(--muted);font-size:12px;text-align:center;margin-top:30px;line-height:1.6;}
  .legendkey{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle;}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:0 6px 12px;}
  .seg button{background:var(--panel2);color:var(--muted);border:none;padding:7px 18px;font-size:13px;
    cursor:pointer;font-family:inherit;transition:background .15s,color .15s;}
  .seg button + button{border-left:1px solid var(--line);}
  .seg button:hover{color:var(--txt);}
  .seg button.active{background:#1e2a3d;color:var(--txt);font-weight:600;}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="hl">Crisis Alpha</span> 2025 &nbsp;vs&nbsp; the SPY Price Path</h1>
    <div class="sub" id="subtitle"></div>
    <div class="period" id="period"></div>
  </header>

  <div class="kpis" id="kpis"></div>

  <div class="banner" id="crisisBanner"></div>

  <div class="card">
    <h2>Cumulative Performance</h2>
    <p class="note">Both series rebased to 0% on the first common trading day. Shaded band = the 2025 tariff-shock drawdown in SPY.</p>
    <div id="mainChart" class="plot" style="height:460px;"></div>
  </div>

  <div class="card">
    <h2>Levered SPY Exposure vs the Strategy</h2>
    <p class="note" id="levNote"></p>
    <div id="levChart" class="plot" style="height:460px;"></div>
  </div>

  <div class="card">
    <h2>Return Decomposition — how volatility eats compounding</h2>
    <p class="note">Annualized arithmetic return = the CAGR you actually keep + the volatility drag lost to the path. Drag &asymp; &sigma;&sup2;/2, so the more volatile curve leaks more.</p>
    <div id="dragChart" class="plot" style="height:300px;"></div>
  </div>

  <div class="card">
    <h2>Realized Volatility &amp; GARCH(1,1) Forecast</h2>
    <p class="note" id="volNote"></p>
    <div class="seg" id="volToggle">
      <button data-k="strat" class="active">Strategy</button>
      <button data-k="spy">SPY</button>
    </div>
    <div id="volChart" class="plot" style="height:420px;"></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>SPY Price Path</h2>
      <p class="note">The market backdrop. The tariff shock is highlighted.</p>
      <div id="priceChart" class="plot" style="height:360px;"></div>
    </div>
    <div class="card">
      <h2>Drawdown (Underwater)</h2>
      <p class="note">How far each was below its own running high.</p>
      <div id="ddChart" class="plot" style="height:360px;"></div>
    </div>
  </div>

  <footer>
    Strategy returns are time-weighted (TWR) from an Interactive Brokers PortfolioAnalyst export.
    SPY cumulative return derived from daily closes. Past performance is not indicative of future results.
  </footer>
</div>

<script>
const D = __DATA__;
const C = {strat:'#4dd6a8', spy:'#7aa2ff', danger:'#ff6b6b', gold:'#f6c453',
           grid:'#233046', txt:'#e6edf7', muted:'#8ea0bd', paper:'rgba(0,0,0,0)'};
const m = D.metrics, cr = D.crisis;

function fmtPct(x, dp=2){ return (x>=0?'+':'') + x.toFixed(dp) + '%'; }
function cls(x){ return x>=0 ? 'pos' : 'neg'; }

/* subtitle + period */
document.getElementById('subtitle').innerHTML =
  '<span class="legendkey" style="background:'+C.strat+'"></span>'+D.strategy_name+
  ' &nbsp;&nbsp; <span class="legendkey" style="background:'+C.spy+'"></span>SPY (S&amp;P 500 ETF)';
document.getElementById('period').textContent =
  m.start_date + '  →  ' + m.end_date + '   ·   ' + m.n_days + ' trading days';

/* KPI cards — total return, CAGR, and volatility drag focus */
const money = x => '$'+Math.round(x).toLocaleString('en-US');
const kpis = [
  {lbl:'Strategy Return', val:fmtPct(m.strat_final), c:cls(m.strat_final),
    foot:'CAGR '+fmtPct(m.strat_cagr), hero:true},
  {lbl:'Max Drawdown', val:fmtPct(m.strat_maxdd), c:'neg',
    foot:'SPY: '+fmtPct(m.spy_maxdd), heroRed:true},
  {lbl:'SPY Return', val:fmtPct(m.spy_final), c:'blue', foot:'CAGR '+fmtPct(m.spy_cagr)},
  {lbl:'Volatility Drag — Strategy', val:'−'+m.strat_drag.toFixed(2)+'%', c:'neg',
    foot:'arith '+fmtPct(m.strat_arith)+' → CAGR '+fmtPct(m.strat_cagr)},
  {lbl:'Volatility Drag — SPY', val:'−'+m.spy_drag.toFixed(2)+'%', c:'gold',
    foot:'arith '+fmtPct(m.spy_arith)+' → CAGR '+fmtPct(m.spy_cagr)},
  {lbl:'Volatility (ann.)', val:m.strat_vol.toFixed(1)+'%', c:'gold',
    foot:'SPY: '+m.spy_vol.toFixed(1)+'%'},
];
document.getElementById('kpis').innerHTML = kpis.map(k =>
  '<div class="kpi'+(k.hero?' hero':'')+(k.heroRed?' hero-red':'')+'"><div class="lbl">'+k.lbl+'</div>'+
  '<div class="val '+k.c+'">'+k.val+'</div><div class="foot">'+k.foot+'</div></div>'
).join('');

/* Banner: total return / CAGR net of volatility drag */
document.getElementById('crisisBanner').innerHTML =
  '<div class="t">Compounding through the 2025 tariff shock</div>'+
  '<div class="b">The strategy ran hotter volatility (<b class="gold">'+m.strat_vol.toFixed(1)+
  '%</b> vs SPY <b>'+m.spy_vol.toFixed(1)+'%</b>), so it surrendered more to <b>volatility drag</b> '+
  '(<b class="neg">−'+m.strat_drag.toFixed(2)+'%</b> vs SPY <b>−'+m.spy_drag.toFixed(2)+'%</b> a year). '+
  'Even after that leak it still <b>compounded</b> at a CAGR of <b class="pos">'+fmtPct(m.strat_cagr)+
  '</b> vs SPY\'s <b class="blue">'+fmtPct(m.spy_cagr)+'</b> — turning $10k into '+
  '<b class="pos">'+money(m.strat_end10k)+'</b> against SPY\'s <b class="blue">'+money(m.spy_end10k)+
  '</b>. The shaded band marks the tariff-shock drawdown.</div>';

const baseLayout = {
  paper_bgcolor:C.paper, plot_bgcolor:C.paper, font:{color:C.txt, family:'Segoe UI, sans-serif'},
  margin:{l:56,r:24,t:20,b:40}, hovermode:'x unified',
  legend:{orientation:'h', y:1.12, x:0, bgcolor:'rgba(0,0,0,0)'},
  xaxis:{gridcolor:C.grid, zeroline:false, showspikes:true, spikethickness:1, spikecolor:C.muted, spikemode:'across'},
  yaxis:{gridcolor:C.grid, zeroline:true, zerolinecolor:C.grid},
};
const config = {responsive:true, displayModeBar:true, displaylogo:false,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};

const crisisShape = (yref='paper', y0=0, y1=1) => ({
  type:'rect', xref:'x', yref:yref, x0:cr.peak_date, x1:cr.trough_date,
  y0:y0, y1:y1, fillcolor:'rgba(246,196,83,0.10)', line:{width:0}, layer:'below'
});

/* ---------- Main chart: cumulative return ---------- */
const traceStratPct = {x:D.dates, y:D.strat_cum, name:D.strategy_name, mode:'lines',
  line:{color:C.strat, width:2.6}, fill:'tozeroy', fillcolor:'rgba(77,214,168,0.08)',
  hovertemplate:'%{y:.2f}%<extra>Strategy</extra>'};
const traceSpyPct = {x:D.dates, y:D.spy_cum, name:'SPY', mode:'lines',
  line:{color:C.spy, width:2.2},
  hovertemplate:'%{y:.2f}%<extra>SPY</extra>'};

const troughLine = {
  type:'line', xref:'x', yref:'paper', x0:cr.trough_date, x1:cr.trough_date,
  y0:0, y1:1, line:{color:C.danger, width:1.2, dash:'dot'}, layer:'below'
};
const mainLayout = Object.assign({}, baseLayout, {
  shapes:[crisisShape('paper',0,1), troughLine],
  yaxis:Object.assign({}, baseLayout.yaxis, {title:'Cumulative return (%)', ticksuffix:'%'}),
  annotations:[
    {x:cr.peak_date, y:1, yref:'paper', yanchor:'bottom', xanchor:'left', showarrow:false,
     text:'Tariff shock', font:{color:C.gold, size:12}}
  ],
});
Plotly.newPlot('mainChart', [traceStratPct, traceSpyPct], mainLayout, config);

/* ---------- Levered constant-beta SPY paths vs strategy ---------- */
const L = D.lev;
const levColors = {'0.75':'#8ea0bd','1.25':'#f6c453','1.5':'#ff9f6b','2':'#ff6b6b'};
document.getElementById('levNote').innerHTML =
  'Hypothetical constant-beta exposure to SPY (daily return = β × SPY, financing ignored), rebased to 0%. '+
  'Shows the net effect of pure leverage across 2025 — compounding and the tariff-shock drawdown amplify with β — '+
  'versus the actual strategy. Full-year: '+
  L.betas.map(b=>'<b style="color:'+levColors[b]+'">'+b+'× '+(L.finals[b]>=0?'+':'')+L.finals[b]+'%</b>').join(', ')+
  ', <b class="pos">strategy '+fmtPct(m.strat_final)+'</b>.';

const levTraces = L.betas.map(b=>({
  x:D.dates, y:L.paths[b], name:b+'× SPY  ('+(L.finals[b]>=0?'+':'')+L.finals[b]+'%, MDD '+L.mdds[b]+'%)',
  mode:'lines', line:{color:levColors[b], width:1.8},
  hovertemplate:'%{y:.2f}%<extra>'+b+'× SPY</extra>'
}));
levTraces.push({
  x:D.dates, y:D.strat_cum, name:'Strategy ('+fmtPct(m.strat_final)+')',
  mode:'lines', line:{color:C.strat, width:3.2},
  hovertemplate:'%{y:.2f}%<extra>Strategy</extra>'
});
const levLayout = Object.assign({}, baseLayout, {
  shapes:[crisisShape('paper',0,1)],
  legend:{orientation:'h', y:1.14, x:0, bgcolor:'rgba(0,0,0,0)'},
  yaxis:Object.assign({}, baseLayout.yaxis, {title:'Cumulative return (%)', ticksuffix:'%'}),
  annotations:[{
    x:cr.peak_date, y:1, yref:'paper', yanchor:'bottom', xanchor:'left', showarrow:false,
    text:'Tariff shock', font:{color:C.gold, size:12}
  }],
});
Plotly.newPlot('levChart', levTraces, levLayout, config);

/* ---------- SPY price path ---------- */
const priceLayout = Object.assign({}, baseLayout, {
  showlegend:false, shapes:[crisisShape('paper',0,1)],
  yaxis:Object.assign({}, baseLayout.yaxis, {title:'SPY close ($)', tickprefix:'$', autorange:true}),
});
Plotly.newPlot('priceChart', [{
  x:D.dates, y:D.spy_close, mode:'lines', name:'SPY',
  line:{color:C.spy, width:2},
  hovertemplate:'$%{y:.2f}<extra>SPY</extra>'
}], priceLayout, config);

/* ---------- Drawdown underwater ---------- */
const ddLayout = Object.assign({}, baseLayout, {
  shapes:[crisisShape('paper',0,1)],
  legend:{orientation:'h', y:1.16, x:0, bgcolor:'rgba(0,0,0,0)'},
  yaxis:Object.assign({}, baseLayout.yaxis, {title:'Drawdown (%)', ticksuffix:'%'}),
});
Plotly.newPlot('ddChart', [
  {x:D.dates, y:D.strat_dd, name:D.strategy_name, mode:'lines',
   line:{color:C.strat, width:2}, fill:'tozeroy', fillcolor:'rgba(77,214,168,0.12)',
   hovertemplate:'%{y:.2f}%<extra>Strategy</extra>'},
  {x:D.dates, y:D.spy_dd, name:'SPY', mode:'lines',
   line:{color:C.spy, width:1.8}, fill:'tozeroy', fillcolor:'rgba(122,162,255,0.10)',
   hovertemplate:'%{y:.2f}%<extra>SPY</extra>'},
], ddLayout, config);

/* ---------- Return decomposition: CAGR kept + volatility drag lost ---------- */
const cats = ['SPY', D.strategy_name];
const cagrVals = [m.spy_cagr, m.strat_cagr];
const dragVals = [m.spy_drag, m.strat_drag];
const arithVals = [m.spy_arith, m.strat_arith];
const dragLayout = Object.assign({}, baseLayout, {
  barmode:'stack', hovermode:'closest',
  margin:{l:190,r:80,t:10,b:40},
  legend:{orientation:'h', y:1.18, x:0, bgcolor:'rgba(0,0,0,0)'},
  xaxis:{type:'linear', title:'Annualized return (%)', ticksuffix:'%',
    gridcolor:C.grid, zeroline:true, zerolinecolor:C.grid, showspikes:false},
  yaxis:{type:'category', gridcolor:C.grid, zeroline:false, automargin:true},
  annotations: cats.map((c0,i)=>({
    x:arithVals[i], y:c0, xanchor:'left', yanchor:'middle', showarrow:false,
    text:'  arith '+fmtPct(arithVals[i]), font:{color:C.muted, size:12}
  }))
});
Plotly.newPlot('dragChart', [
  {y:cats, x:cagrVals, name:'CAGR kept', orientation:'h', type:'bar',
   marker:{color:[C.spy, C.strat]},
   text:cagrVals.map(v=>fmtPct(v)), textposition:'inside', insidetextanchor:'middle',
   textfont:{color:'#0b0f17', size:13}, hovertemplate:'CAGR: %{x:.2f}%<extra>%{y}</extra>'},
  {y:cats, x:dragVals, name:'Volatility drag lost', orientation:'h', type:'bar',
   marker:{color:'rgba(255,107,107,0.65)', line:{color:C.danger, width:1}},
   text:dragVals.map(v=>'−'+v.toFixed(2)+'%'), textposition:'inside', insidetextanchor:'middle',
   textfont:{color:'#0b0f17', size:12}, hovertemplate:'Drag: −%{x:.2f}%<extra>%{y}</extra>'},
], dragLayout, config);

/* ---------- Realized volatility proxies + GARCH conditional forecast ---------- */
const V = D.vol, VM = D.vol_meta;
document.getElementById('volNote').innerHTML =
  VM.window+'-day rolling realized vol, RiskMetrics EWMA (λ='+VM.lam+'), high-low range '+
  'estimators (Parkinson &amp; Garman-Klass, SPY only), and a GARCH(1,1)-t conditional '+
  'forecast. GARCH persistence (α+β): strategy '+VM.strat_persistence+', SPY '+VM.spy_persistence+
  '. Next-day GARCH forecast: strategy <b class="pos">'+VM.strat_forecast+'%</b>, SPY '+
  '<b class="blue">'+VM.spy_forecast+'%</b>.';

const proxyCol = {rv:'#f6c453', ewma:'#c792ea', park:'#ff9f6b', gk:'#ff6b9d'};
const volTraces = [
  // Strategy (visible by default)
  {x:D.dates, y:V.strat_rv, name:'Realized vol ('+VM.window+'d)', mode:'lines',
   line:{color:proxyCol.rv, width:1.6}, hovertemplate:'%{y:.1f}%<extra>RV</extra>'},
  {x:D.dates, y:V.strat_ewma, name:'EWMA (λ='+VM.lam+')', mode:'lines',
   line:{color:proxyCol.ewma, width:1.6}, hovertemplate:'%{y:.1f}%<extra>EWMA</extra>'},
  {x:D.dates, y:V.strat_garch, name:'GARCH(1,1) conditional', mode:'lines',
   line:{color:C.strat, width:3}, hovertemplate:'%{y:.1f}%<extra>GARCH</extra>'},
  // SPY (hidden by default)
  {x:D.dates, y:V.spy_rv, name:'Realized vol ('+VM.window+'d)', mode:'lines', visible:false,
   line:{color:proxyCol.rv, width:1.6}, hovertemplate:'%{y:.1f}%<extra>RV</extra>'},
  {x:D.dates, y:V.spy_ewma, name:'EWMA (λ='+VM.lam+')', mode:'lines', visible:false,
   line:{color:proxyCol.ewma, width:1.6}, hovertemplate:'%{y:.1f}%<extra>EWMA</extra>'},
  {x:D.dates, y:V.spy_parkinson, name:'Parkinson (H-L)', mode:'lines', visible:false,
   line:{color:proxyCol.park, width:1.4, dash:'dot'}, hovertemplate:'%{y:.1f}%<extra>Parkinson</extra>'},
  {x:D.dates, y:V.spy_gk, name:'Garman-Klass (OHLC)', mode:'lines', visible:false,
   line:{color:proxyCol.gk, width:1.4, dash:'dot'}, hovertemplate:'%{y:.1f}%<extra>Garman-Klass</extra>'},
  {x:D.dates, y:V.spy_garch, name:'GARCH(1,1) conditional', mode:'lines', visible:false,
   line:{color:C.spy, width:3}, hovertemplate:'%{y:.1f}%<extra>GARCH</extra>'},
];
const volLayout = Object.assign({}, baseLayout, {
  shapes:[crisisShape('paper',0,1)],
  legend:{orientation:'h', y:1.14, x:0, bgcolor:'rgba(0,0,0,0)'},
  yaxis:Object.assign({}, baseLayout.yaxis, {title:'Annualized volatility (%)', ticksuffix:'%', rangemode:'tozero'}),
});
Plotly.newPlot('volChart', volTraces, volLayout, config);

const volVis = {
  strat:[true,true,true, false,false,false,false,false],
  spy:[false,false,false, true,true,true,true,true],
};
document.querySelectorAll('#volToggle button').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('#volToggle button').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    Plotly.restyle('volChart', {visible: volVis[btn.dataset.k]});
  });
});
</script>
</body>
</html>
"""


def main() -> None:
    strategy = load_strategy()
    spy = load_spy()
    df = align(strategy, spy)
    crisis = find_crisis_window(df)
    metrics = build_metrics(df, crisis)
    vol = build_vol(df)
    lev = build_leverage(df, betas=[0.75, 1.25, 1.5, 2.0])

    OUT_HTML.write_text(render_html(df, metrics, vol, lev), encoding="utf-8")

    c = metrics["crisis"]
    print(f"Aligned {metrics['n_days']} trading days "
          f"({metrics['start_date']} -> {metrics['end_date']})")
    print(f"  Total return  strat/SPY : {metrics['strat_final']:+.2f}% / {metrics['spy_final']:+.2f}%")
    print(f"  CAGR          strat/SPY : {metrics['strat_cagr']:+.2f}% / {metrics['spy_cagr']:+.2f}%")
    print(f"  Arithmetic    strat/SPY : {metrics['strat_arith']:+.2f}% / {metrics['spy_arith']:+.2f}%")
    print(f"  Volatility drag         : {metrics['strat_drag']:.2f}% (SPY {metrics['spy_drag']:.2f}%)")
    print(f"  Ann. volatility         : {metrics['strat_vol']:.1f}% (SPY {metrics['spy_vol']:.1f}%)")
    print(f"  $10k grows to           : ${metrics['strat_end10k']:,.0f} (SPY ${metrics['spy_end10k']:,.0f})")
    print(f"  Max drawdown            : {metrics['strat_maxdd']:+.2f}% (SPY {metrics['spy_maxdd']:+.2f}%)")
    print(f"  Sharpe                  : {metrics['strat_sharpe']:.2f} (SPY {metrics['spy_sharpe']:.2f})")
    vm = vol["meta"]
    print(f"  GARCH now  strat/SPY    : {vm['strat_current_garch']:.1f}% / {vm['spy_current_garch']:.1f}%")
    print(f"  GARCH 1d fc strat/SPY   : {vm['strat_forecast']:.1f}% / {vm['spy_forecast']:.1f}%")
    print(f"  GARCH persist strat/SPY : {vm['strat_persistence']:.3f} / {vm['spy_persistence']:.3f}")
    for b in lev["betas"]:
        print(f"  Levered {b}x SPY          : {lev['finals'][b]:+.1f}% final, MDD {lev['mdds'][b]:.1f}%")
    print(f"  Tariff shock            : {c['peak_date'].date()} -> {c['trough_date'].date()}")
    print(f"    Shock  SPY / strat  : {c['spy_move']:+.2f}% / {c['strat_move']:+.2f}%")
    print(f"    Rebound SPY / strat : {c['spy_rebound']:+.2f}% / {c['strat_rebound']:+.2f}%")
    print(f"    Rebound alpha       : {c['rebound_alpha']:+.2f}%")
    print(f"\nWrote {OUT_HTML}")


if __name__ == "__main__":
    main()
