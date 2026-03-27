# ### 📈 Black-Scholes Implied Volatility in 3 Minutes
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)
# 
# - [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)
# 
# - [Central Limit Theorem for Quant Finance](https://youtu.be/q2era-4pnic)
# 
# - [Why Quant Models Break](https://youtu.be/brdG1TmsPlw)
# 
# - [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)
# 
# - [The 5 Papers That Build Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# 
# 
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# ##### [👾 Join the Quant Guild Discord Server](discord.com/invite/MJ4FU2c6c3)
# 
# ---


%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>


# ### 📖 Sections
# 
# #### 1.) 🛠️ Black-Scholes Argument
# 
# - European Option Contracts and Value at Maturity
# 
# - What Should we Pay Today?
# 
# #### 2.) 📈 Market Available Parameters
# 
# - Supply and Demand create Equilibrium Price
# 
# - Backing Out Black-Scholes Volatility
# 
# #### 3.) 📊 Black-Scholes Implied Volatility
# 
# - Implied Volatility
# 
# - Implied Volatility Skew
# 
# - Implied Volatility Surface
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🛠️ Black-Scholes Argument
# 
# ##### European Option Contracts
# 
# European option contracts give the holder the right, not obligation, to buy (sell) at a predetermined price (strike, $K$) at a fixed time in the future (maturity) $T$
# 
# A European option contract $\zeta$ always has the known parameters of the underlying its written on, the strike price, and the maturity.
# 
# The payoff in every state of the world is known at time $T$ in the future
# 
# $$C_T = max(S_T - K, 0) = (S_T - K)^+$$
# 
# **Problem:** We don't know what state of the world we are going to be in.


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors - Neon Theme
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#ff43ff"   # Neon Pink
path_color = "#00f2ff"     # Neon Cyan

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff Diagram", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Base traces
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], line=dict(color=path_color, width=2.5), mode='lines'), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    frame_name = f"f{i}"
    
    # Create Frame
    frames.append(go.Frame(
        data=[
            go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)),
            go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')),
            go.Scatter(x=[current_price, current_price], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'),
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')),
            go.Scatter(x=dates[:i+1], y=path[:i+1], line=dict(color=path_color, width=2.5), mode='lines')
        ],
        name=frame_name
    ))
    
    # Create Slider Step
    step = {
        "args": [
            [frame_name],
            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}
        ],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=150),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.15)', # slightly brighter grid for neon feel
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Payoff ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()


# What should we be willing to pay for this payoff potential?


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Black-Scholes Pricing Argument
# 
# Black-Scholes suggest we can create a portfolio *today* and hold it to maturity such that we can *replicate* every possible payoff of the contract above.
# 
# If our portfolio matches every possible payoff of the option in the future *today*, the portfolio must hold the same value as the option contract by a no arbitrage assumption.
# 
# In other words, if they didn't match, we could just go long one and short there would be arbitrage!
# 
# Turns out, the constructed portfolio that replicates the option payoff, in a Black-Scholes framework, yields a parabolic PDE.
# 
# $$
# \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
# $$
# 
# This Black-Scholes Equation which can be solved for the pricing model or pricing functional giving us the price today under the model assumptions.
# 
# $$
#  C_t(S_0, K, T, r, \sigma) = S_0 N(d_1) - K e^{-rT} N(d_2)
# $$


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

def black_scholes_call(S, K, T, r, sigma):
    # Avoid division by zero as T -> 0
    if T <= 1e-5:
        return np.maximum(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# Generate a Price Path
np.random.seed(42)
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors - Neon Theme
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#ff43ff"   # Neon Pink
bs_color = "#efff14"       # Neon Yellow
path_color = "#00f2ff"     # Neon Cyan

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Pricing vs Payoff", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Initial Black-Scholes Curve (at t=0)
bs_initial = black_scholes_call(S_range, K, T, r, vol)

# Base traces
# 1. Payoff
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, name="Payoff",
                         line=dict(color=payoff_color, width=3)), row=1, col=1)

# 2. Black-Scholes Curve
fig.add_trace(go.Scatter(x=S_range, y=bs_initial, name="BS Price",
                         line=dict(color=bs_color, width=2, dash='dash')), row=1, col=1)

# 3. Strike Line (Hidden from legend to avoid clutter)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], showlegend=False,
                         line=dict(color=strike_color, width=1, dash='dash')), row=1, col=1)

# 4. Current Price Marker (UPDATED: Added Label)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], name="Underlying Price", showlegend=True,
                         line=dict(color=off_white, width=2), mode='lines'), row=1, col=1)

# 5. Strike Line Col 2
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], showlegend=False,
                         line=dict(color=strike_color, width=1, dash='dash')), row=1, col=2)

# 6. Path
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], showlegend=False,
                         line=dict(color=path_color, width=2.5), mode='lines'), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    T_remaining = T - (i * dt) 
    
    bs_current = black_scholes_call(S_range, K, T_remaining, r, vol)
    frame_name = f"f{i}"
    
    frames.append(go.Frame(
        data=[
            # Col 1: Payoff
            go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)),
            # Col 1: BS Curve
            go.Scatter(x=S_range, y=bs_current, line=dict(color=bs_color, width=2, dash='dash')),
            # Col 1: Strike Line
            go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')),
            # Col 1: Current Price (Moving)
            go.Scatter(x=[current_price, current_price], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'),
            
            # Col 2: Strike Line
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')),
            # Col 2: Path
            go.Scatter(x=dates[:i+1], y=path[:i+1], line=dict(color=path_color, width=2.5), mode='lines')
        ],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    # UPDATED: Increased top margin to make room for higher legend
    margin=dict(t=120, b=150), 
    
    # UPDATED: Raised legend position (y=1.12)
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.15)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Option Price ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()


# $
#  C_t(S_0, K, T, r, \sigma) = S_0 N(d_1) - K e^{-rT} N(d_2)
# $
# 
# The 5 Inputs to the Black-Scholes Model
#  
#  1. **Spot Price ($S$):** The current price of the underlying asset.
#  2. **Strike Price ($K$):** The price at which the option can be exercised.
#  3. **Time to Maturity ($T$):** The time remaining until the option expires (in years).
#  4. **Risk-Free Rate ($r$):** The continuously compounded interest rate of a risk-free asset over the life of the option.
#  5. **Volatility ($\sigma$):** The standard deviation of the asset’s returns (annualized).
# 
# This model is later validated by Feynman-Kac showing that the risk-neutral conditional expectation is t


# ---


# #### 2.) 📈 Black-Scholes Implied Volatility
# 
# Where do we get the inputs of our function from?  Well, the law of supply and demand will create them in equilibrium in the market.  Wait, then why do we need a pricing function at all?  The market should produce an equilibrium price by the law of supply and demand.
# 
# Let's revisit our parameters and model equation to see what we have available...
# 
# $
#  C_t(S_0, K, T, r, \sigma) = S_0 N(d_1) - K e^{-rT} N(d_2)
# $
# 
# The 5 Inputs (6 being the output or option price) to the Black-Scholes Model
#  
#  1. **Spot Price ($S$):** ✅
#  2. **Strike Price ($K$):** ✅
#  3. **Time to Maturity ($T$):** ✅
#  4. **Risk-Free Rate ($r$):** ✅
#  5. **Volatility ($\sigma$):** ❌
#  6. **Option Price ($C_t$):** ✅
# 
#  Now our model price looks something like this:
# 
# $
#  C_t(✅, ✅, ✅, ✅, ❌) = ✅
# $
# 
# In other words, we know everything, including the option price!  But what should volatility be?  
# 
# We can solve for it by trying out different volatilities until we can match the market price!  This is a minimization problem from calculus...
# 
#  $$
#  \sigma_{\mathrm{imp}} = \underset{\sigma}{\operatorname{argmin}} \; \left| C_{\text{BS}}(S_0, K, T, r, \sigma) - C_{\text{market}} \right|
#  $$
# Remember, we know everything but the volatility, when we solve for it, we will get the volatility required by the market to reproduce current option prices!
# 
#  $$
#  \sigma_{\mathrm{imp}} = \underset{\sigma}{\operatorname{argmin}} \; \left| C_{\text{BS}}(✅, ✅, ✅, ✅, ❌) - ✅ \right|
#  $$
# 
#  We can solve this minimization technique numerically...


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Market Truth ---
S0 = 100        # Current Spot Price
K = 100         # Strike Price (ATM)
T = 1.0         # Time to maturity
r = 0.05        # Risk-free rate

# The "True" Market parameters
true_vol = 0.25 
# Calculate the fixed "Market Price" target
market_price = (S0 * norm.cdf((np.log(S0/K) + (r + 0.5*true_vol**2)*T)/(true_vol*np.sqrt(T)))) - \
               (K * np.exp(-r*T) * norm.cdf((np.log(S0/K) + (r + 0.5*true_vol**2)*T)/(true_vol*np.sqrt(T)) - true_vol*np.sqrt(T)))

# Solver settings (Slower Animation)
start_vol = 0.65   
n_frames = 100     # INCREASED for smoother slow motion

# --- 2. Helper Functions ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

# --- 3. Data Generation ---

# A. Left Plot Data (Error Function)
vol_range = np.linspace(0.1, 0.8, 100)
model_prices = np.array([black_scholes_call(S0, K, T, r, v) for v in vol_range])
error_values = (model_prices - market_price)**2

# B. Right Plot Data (Fixed BS Curve)
S_range = np.linspace(60, 140, 100)
bs_curve_true = black_scholes_call(S_range, K, T, r, true_vol)

# C. Animation Path (Slower Decay)
steps = np.arange(n_frames)
decay_rate = 0.06 # Slower decay rate
simulated_vols = true_vol + (start_vol - true_vol) * np.exp(-decay_rate * steps)

# --- 4. Visual Styling ---
off_white = "#e0e0e0"
neon_cyan = "#00f2ff"    
neon_pink = "#ff43ff"    
neon_yellow = "#efff14"  
grid_color = "rgba(255,255,255,0.1)"

# --- 5. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Objective Function: Minimize Error</b><br>(Squared Pricing Error vs Vol)", 
        "<b>Price Convergence</b><br>(Candidate Price Line vs Market)"
    ),
    horizontal_spacing=0.12
)

# === INITIAL TRACES ===
# Left: Error Curve (Static)
fig.add_trace(go.Scatter(x=vol_range, y=error_values, name="Error Function",
                         line=dict(color=neon_pink, width=3)), row=1, col=1)

# Left: Solver Ball (Dynamic)
initial_error = (black_scholes_call(S0, K, T, r, start_vol) - market_price)**2
fig.add_trace(go.Scatter(x=[start_vol], y=[initial_error], name="Current Error",
                         mode='markers', marker=dict(color=neon_cyan, size=15, symbol='circle-open', line=dict(width=3))), row=1, col=1)

# Right: Fixed BS Curve (Static)
fig.add_trace(go.Scatter(x=S_range, y=bs_curve_true, name="Market Structure (Fixed)",
                         line=dict(color=neon_yellow, width=2, dash='dash')), row=1, col=2)

# Right: Market Price Star (Static)
fig.add_trace(go.Scatter(x=[S0], y=[market_price], name="Market Price",
                         mode='markers', marker=dict(color=neon_yellow, size=18, symbol='star')), row=1, col=2)

# Right: Candidate Price Line (Dynamic)
initial_model_price = black_scholes_call(S0, K, T, r, start_vol)
fig.add_trace(go.Scatter(x=[S_range[0], S_range[-1]], y=[initial_model_price, initial_model_price], 
                         name="Candidate Model Price",
                         line=dict(color=neon_cyan, width=3)), row=1, col=2)

# --- 6. Animation Frames ---
frames = []
slider_steps = []

for i, vol_i in enumerate(simulated_vols):
    current_price = black_scholes_call(S0, K, T, r, vol_i)
    current_error = (current_price - market_price)**2
    
    frame_name = f"f{i}"
    
    frames.append(go.Frame(
        data=[
            # Update Ball
            go.Scatter(x=[vol_i], y=[current_error], mode='markers', marker=dict(color=neon_cyan, size=15, symbol='circle-open', line=dict(width=3))),
            # Update Line
            go.Scatter(x=[S_range[0], S_range[-1]], y=[current_price, current_price], line=dict(color=neon_cyan, width=3)),
        ],
        name=frame_name,
        traces=[1, 4] 
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": f"{vol_i:.2f}",
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 7. Layout Adjustments ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700, width=1200,
    
    # Increased bottom margin to fit Legend -> Slider -> Buttons
    margin=dict(t=100, b=200),
    
    # 1. LEGEND POSITION (Below charts, above slider)
    # y coordinate is relative to the plot area. Negative values push it down.
    legend=dict(
        orientation="h", 
        y=-0.15, 
        x=0.5, 
        xanchor="center",
        yanchor="top"
    ),
    
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Run Solver",
                "method": "animate",
                # 2. SLOWER ANIMATION: duration=60 (was 20)
                "args": [None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 10},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        # Push buttons down below the legend and slider
        "y": -0.4, 
        "yanchor": "top"
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Vol Guess: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        # Push slider down below the legend
        "pad": {"b": 10, "t": 10},
        "len": 0.85,
        "x": 0.15,
        "y": -0.4, 
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor=grid_color,
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, title_text="Volatility (σ)", row=1, col=1)
fig.update_yaxes(axis_style, title_text="Squared Error (Price - Market)²", row=1, col=1)

fig.update_xaxes(axis_style, range=[60, 140], title_text="Spot Price ($)", row=1, col=2)
fig.update_yaxes(axis_style, range=[0, 40], title_text="Option Price ($)", row=1, col=2)

fig.show()


# Don't get confused!  There is a difference between *realized* volatiltiy and implied volatility.
# 
# Realized volatility *already happened*, and we can't directly observe it, but we can *proxy* for it with statistics like standard deviation and variance.
# 
# Implied volatility is *forward looking*, it's what the market is currently pricing instruments with in anticipation of what's to come - it is not always correct!


# ---


# #### 3.) 📊 Black-Scholes Implied Volatility Skew, Surface, and Applications


# ##### Black-Scholes Implied Volatility Skew
# 
# The Black-Scholes model assumes volatility is constant across strikes and maturities for the same underlying.
# 
# Let's visualize the implied volatility for the same underlying and maturity but at different strike prices.  This is the volatility skew or smile. 


import numpy as np
import plotly.graph_objects as go

# --- Load SPX Option Data ---
spx_data = []
with open("spx.csv") as f:
    for line in f:
        fields = line.strip().split(",")
        if len(fields) == 5:
            strike, ttm, iv, callput, weight = map(float, fields)
            spx_data.append({"K": strike, "T": ttm, "iv": iv, "callput": int(callput)})
        # Skip the first line (market params)

# --- Filter for Front-Month (Shortest T) ---
all_ttms = sorted(set(d['T'] for d in spx_data))
front_ttm = min([t for t in all_ttms if t > 0])
front_month = [d for d in spx_data if np.isclose(d['T'], front_ttm, atol=1e-4)]

strikes = []
ivs = []
for d in front_month:
    strikes.append(d["K"])
    ivs.append(d["iv"])

strikes = np.array(strikes)
ivs = np.array(ivs)
sort_idx = np.argsort(strikes)
strikes = strikes[sort_idx]
ivs = ivs[sort_idx]

# --- Neon Style Settings (consistent with animation above) ---
off_white = "#e0e0e0"
neon_cyan = "#00f2ff"
grid_color = "rgba(255,255,255,0.1)"

# --- Plot: Implied Volatility Skew (Front Month) ---
fig_skew = go.Figure()

fig_skew.add_trace(go.Scatter(
    x=strikes, y=ivs,
    mode="markers+lines",
    line=dict(color=neon_cyan, width=3),
    marker=dict(size=15, color=neon_cyan, line=dict(width=3, color="black")),
    name="Front Month Skew",
    connectgaps=True
))

fig_skew.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=800,
    margin=dict(t=75, b=80),
    title="<b>SPX Implied Volatility Skew - Front Month</b>",
    # Font settings to better match the larger animation chart
    font=dict(color=off_white, size=20, family="Arial"),
    xaxis=dict(
        title=dict(text="Strike Price ($)", font=dict(color=off_white, size=18)),
        tickfont=dict(color=off_white, size=16),
        showgrid=True, gridcolor=grid_color,
        linecolor=off_white,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(text="Implied Volatility (σ)", font=dict(color=off_white, size=18)),
        tickfont=dict(color=off_white, size=16),
        showgrid=True, gridcolor=grid_color,
        linecolor=off_white,
        zeroline=False
    ),
    showlegend=False
)

fig_skew.show()



# Why is implied volatility not flat like the Black-Scholes model would suggest?
# 
# Well, the argument relies on *hedging* to replicate the portfolio.  In reality, not all risks are hedgable, and traders demand compensation in the form of higher prices.
# 
# For example, the risk of a massive gap down is not entirely hedgable (and costly to do so in perpetuity) so traders demand higher volatility (prices) as compensation


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Black-Scholes Implied Volatility Surface
# 
# What is the Black-Scholes Implied Volatility Surface?
# 
# We are simply adding a dimension, the skew or smile is implied volatility across strikes, now we just add maturities...


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load CSV Data (spot, r, q in first line) ---
csv_path = "spx.csv"
with open(csv_path, 'r') as f:
    spot, r, q = map(float, f.readline().strip().split(','))
    df = pd.read_csv(f, header=None)
    df.columns = ['Strike', 'Maturity', 'IV', 'IsCall', 'w']

# --- Process surface for 3D plotting ---
pivot = df.pivot_table(index='Maturity', columns='Strike', values='IV').sort_index().sort_index(axis=1)
pivot = pivot.interpolate('linear', axis=0).bfill().ffill()
X, Y = np.meshgrid(pivot.columns, pivot.index)
Z = pivot.values

# --- Professional, Clean 3D Plot Style ---
# Transparent main background now
axis_params = dict(
    showbackground=True,
    backgroundcolor="rgba(15,19,26,0.74)",   # slightly see-through dark, still readable grid/axes
    showgrid=True,
    gridcolor="rgba(90,115,145,0.34)",    # gentle blue-grey gridlines
    gridwidth=1.05,
    zeroline=False,
    linecolor="rgba(180,255,255,0.39)",  # subtle cyan axis lines
    linewidth=2.0,
    ticks="outside",
    tickcolor="rgba(180,244,243,0.43)",
    showspikes=False,
)

surface = go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[
        [0.0, '#182731'],
        [0.13, '#226aa7'],
        [0.36, '#44f0ff'],
        [0.60, '#a0ddf6'],
        [0.82, '#f1fdff'],
        [1.0, '#ffffff'],
    ],
    opacity=1.0,
    showscale=True,
    colorbar=dict(
        title=dict(
            text="Implied Volatility",
            font=dict(color="#bee9ec", size=14, family="Consolas, monospace"),
        ),
        tickfont=dict(color="#a0e8ff", size=11, family="Consolas, monospace"),
        len=0.74,
        thickness=14,
        bgcolor='#191e26',
        outlinewidth=1.8,
        outlinecolor="#40d2f0",
        x=1.0,
    )
)

fig = go.Figure(data=[surface])

fig.update_layout(
    title=dict(
        text="Black-Scholes Implied Volatility Surface",
        font=dict(color="#dafaff", size=21, family="Consolas, monospace"),
        x=0.55, xanchor="center",
    ),
    autosize=True,
    height=560, width=960,
    margin=dict(l=44, r=35, b=40, t=64),
    font=dict(family="Consolas, monospace", color="#9fd3e8", size=13),
    paper_bgcolor='rgba(0,0,0,0)',         # transparent main background
    plot_bgcolor='rgba(0,0,0,0)',         # transparent plot/scene background
    scene=dict(
        xaxis=dict(
            title=dict(text="Strike", font=dict(color="#c3f8ff", size=17, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=13, family="Consolas, monospace"),
            **axis_params
        ),
        yaxis=dict(
            title=dict(text="Maturity (Years)", font=dict(color="#c3f8ff", size=16, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=13, family="Consolas, monospace"),
            **axis_params
        ),
        zaxis=dict(
            title=dict(text="Implied Volatility", font=dict(color="#c3f8ff", size=16, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=12, family="Consolas, monospace"),
            **axis_params,
        ),
        bgcolor="rgba(0,0,0,0)",  # fully transparent 3d scene background
        camera=dict(eye=dict(x=1.5, y=1.28, z=0.8))
    ),
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Applications to Trading
# 
# The Black-Scholes framework is used everyday by professional market-makers and speculative traders.
# 
# Whether its extrapolating prices or harvesting the volatility risk premium - trading options wouldn't be possible without understanding this structure (see 1987 for more).


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - European options give the holder the right but not obligation to buy (sell) the underlying asset at a price called the strike price at a predetermined time in the future called the maturity
# - We know what the payoff is in every state of the world, but we don't know the *fair* price of these potential payoffs today
# - The Black-Scholes argument produces a replicating portfolio for the contract payoff in every state of the world, by no arbitrage, they must be equal 
# - The subsequent portfolio yields the Black-Scholes equation (parabolic PDE), once solved, it yields the Black-Scholes model or pricing functional
# - Supply and demand give us all the inputs and even the option price except for the input of *volatility*, we can find the volatility that makes the Black-Scholes model match market prices and we will arrive at the so called implied volatility
# - The Black-Scholes model implies a flat surface, but in reality not all risk is hedgable (markets are incomplete) and traders demand compensation in the form of higher prices for assuming that risk generating relatively higher and lower implied volatilities across the skew and term structure
# - Market makers and speculative traders use the volatility surface every day to inform their trading decisions and accumulate expected value over time
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Fama-French / Carhart and Factor Modeling in General
# - Hawkes Processes
# - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
# - Market-Making Models and Simulation (Stoikov-Avellaneda)
# - My First Year as a Quant
# - Kalman Filter for Quant Finance
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Girsanov's Change of Measure
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

