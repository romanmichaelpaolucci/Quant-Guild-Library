# ### 🌌 How Physics Accidentally Proved the Black-Scholes Model
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Ito Integration Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo)
# 
# - [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)
# 
# - [Finite Differences Option Pricing for Quant Finance](https://youtu.be/uzbveN8n34U)
# 
# - [Heston Stochastic Volatility Model and Fast Fourier Transforms](https://youtu.be/2-oAlnZV6hA)
# 
# - [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)
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
# ##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)
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
# #### 1.) 📈 Pricing a European Option Contract
# 
# - Problem setup
# 
# - What should the price be?
# 
# #### 2.) 📈 Black-Scholes Model
# 
# - Black-Scholes equation and solution
# 
# #### 3.) 📈 Fundamental Theorem of Asset Pricing
# 
# - Risk-Neutral Conditional Expectation
# 
# - Which is "Correct"?
# 
# #### 3.) ⚛️ Feynman-Kac Theorem
# 
# - The Bridge between Black-Scholes and Fundamental Theorem of Asset Pricing
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📈 Pricing a European Option Contract
# 
# The payoff of a European call option at expiration $ T $ is:
# 
# $$
# \text{Payoff at } T = \max(S_T - K,\, 0)
# $$
# 
# where $ S_T $ is the stock price at time $ T $, and $ K $ is the strike price.
# 
# 
# 


import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Parameters for the payoff plot
K = 100  # Strike price
r = 0.05  # risk-free rate
sigma = 0.2  # volatility
T = 1      # time to maturity
S = np.linspace(60, 140, 200)
payoff = np.maximum(S - K, 0)

# Black-Scholes European call price as a function
def euro_call(S, K, r, sigma, T):
    S = np.array(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    C = np.where(np.isnan(C), 0, C)
    return C

call_price = euro_call(S, K, r, sigma, T)

# 100th index as the starting slider position
slider_start_idx = 99  # zero-based
slider_x = float(S[slider_start_idx])
slider_y = float(payoff[slider_start_idx])
euro_call_val = euro_call(slider_x, K, r, sigma, T)

# Redefine fig with payoff, vertical line, marker
fig = go.Figure(
    data=[
        go.Scatter(
            x=S, y=payoff, mode='lines',
            name='European Call Payoff', line=dict(color='#39ff14', width=4),
            showlegend=False
        ),
        go.Scatter(
            x=[slider_x, slider_x], y=[0, slider_y], mode='lines',
            name='Final Stock Price', line=dict(color='cyan', width=2, dash='dash'),
            showlegend=False
        ),
        go.Scatter(
            x=[slider_x], y=[slider_y], mode='markers+text',
            marker=dict(color='cyan', size=10, symbol='circle'),
            text=[f"Payoff: {slider_y:.2f}"],
            textposition="top left",
            showlegend=False
        )
    ]
)

# Slider code (1 slider, 1 handle)
steps = []
for i in range(len(S)):
    si = float(S[i])
    yi = float(payoff[i])
    # euro_val_i = euro_call(si, K, r, sigma, T)  # Not needed for title anymore
    step = dict(
        method="update",
        args=[
            {
                "x": [S, [si, si], [si]],
                "y": [payoff, [0, yi], [yi]],
                "text": [None, None, [f"Payoff: {yi:.2f}"]],
            },
            {
                "title": f"European Call Option Payoff at Maturity (T)<br>Stock @ T = {si:.2f} | Payoff = {yi:.2f}"
            }
        ],
        label=f"{si:.2f}"
    )
    steps.append(step)

sliders = [dict(
    active=slider_start_idx,
    currentvalue={"prefix": "Final Stock Price (Stock @ T) = "},
    pad={"t": 30},
    steps=steps,
)]

initial_title = f"European Call Option Payoff at Maturity (T)<br>Stock @ T = {slider_x:.2f} | Payoff = {slider_y:.2f}"

fig.update_layout(
    title=initial_title,
    xaxis_title="Underlying Price (Stock @ T)",
    yaxis_title="Payoff",
    height=400,
    sliders=sliders,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()



# ##### **Problem:**  We don't know what the value of the underlying will be at time $T$ in the future.  What should we pay for this contract today?


# ###### ______________________________________________________________________________________________________________________________________


import plotly.subplots as sp
import warnings
warnings.filterwarnings("ignore")


# Simulation parameters
S0 = K
N_steps = 100
N_paths = 1   # Show a single path in this illustration
dt = T / N_steps

# Preallocate arrays
stock_paths = np.zeros((N_steps + 1, N_paths))
option_path = np.zeros((N_steps + 1, N_paths))
stock_paths[0, :] = S0

# Simulate stock path and corresponding call price at every step
for path in range(N_paths):
    for t in range(1, N_steps + 1):
        Z = np.random.normal()
        stock_paths[t, path] = stock_paths[t-1, path] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    # Calculate Black-Scholes call price at each simulated spot for the *remaining* time
    time_grid = np.linspace(T, 0, N_steps + 1)  # decreasing time to expiry
    for t in range(N_steps + 1):
        option_path[t, path] = euro_call(stock_paths[t, path], K, r, sigma, time_grid[t])

# Setup subplots: (1) Underlying Stock Price path; (2) European Option Price
fig2 = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=("Underlying Stock Price Over Time", "European Option Price Over Time")
)

# Time array for plotting
time_vals = np.linspace(0, T, N_steps + 1)

# Plot simulated underlying stock price
fig2.add_trace(
    go.Scatter(
        x=time_vals,
        y=stock_paths[:, 0],
        mode='lines+markers',
        showlegend=False,
        line=dict(color='#22d3ee')
    ),
    row=1, col=1
)

# Add a vertical dashed line at terminal time T (to first subplot)
fig2.add_shape(
    dict(
        type="line",
        x0=T,
        y0=min(stock_paths[:, 0].min(), K),
        x1=T,
        y1=max(stock_paths[:, 0].max(), K),
        line=dict(color="white", width=2, dash="dash"),
    ),
    row=1, col=1
)

# Add a horizontal line for the strike price K (to first subplot)
fig2.add_shape(
    dict(
        type="line",
        x0=0,
        y0=K,
        x1=T,
        y1=K,
        line=dict(color="yellow", width=2, dash="dash"),
    ),
    row=1, col=1
)

# Plot European option value
fig2.add_trace(
    go.Scatter(
        x=time_vals,
        y=option_path[:, 0],
        mode='lines+markers',
        showlegend=False,
        line=dict(color='#f43f5e')
    ),
    row=1, col=2
)

# Set consistent gridline color as in the previous plot
gridcolor = 'rgba(128,128,128,0.2)'

fig2.update_xaxes(
    title_text="Time (t)", 
    showgrid=True, gridwidth=1, gridcolor=gridcolor, 
    row=1, col=1
)
fig2.update_xaxes(
    title_text="Time (t)", 
    showgrid=True, gridwidth=1, gridcolor=gridcolor, 
    row=1, col=2
)
fig2.update_yaxes(
    title_text="Stock Price", 
    showgrid=True, gridwidth=1, gridcolor=gridcolor, 
    row=1, col=1
)
fig2.update_yaxes(
    title_text="European Option Price", 
    showgrid=True, gridwidth=1, gridcolor=gridcolor, 
    row=1, col=2
)
fig2.update_layout(
    title_text="Underlying Stock Path & European Option Price Evolution",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=400,
    showlegend=False
)

fig2.show()



# ##### **Goal:** Model the function(al) $V$ that gives the Option Price at any Time $t$


# ---


# #### 2.) 📈 Black-Scholes Model
# 
# Using the argument posited by Black-Scholes (a Black-Scholes framework) we get:
# $$
# \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
# $$
# 
# If we solve this partial differential equation we get the function(al) we are looking for $V$!
# 
# where $V(S, t)$ is the option price, $\sigma$ is volatility, and $r$ is the risk-free rate.


import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Parameters for the Black-Scholes plot
K = 100  # Strike price
r = 0.05  # risk-free rate
sigma = 0.2  # volatility
T = 1      # time to maturity
S = np.linspace(60, 140, 200)
t = T  # Make it explicit (at time t, not T)

# Black-Scholes European call price as a function (at fixed time t)
def euro_call(S, K, r, sigma, t):
    S = np.array(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    C = np.where(np.isnan(C), 0, C)
    return C

call_price = euro_call(S, K, r, sigma, t)

# 100th index as the starting slider position
slider_start_idx = 99  # zero-based
slider_x = float(S[slider_start_idx])
slider_y = float(call_price[slider_start_idx])

# Redefine fig with Black-Scholes, vertical line, marker, remove all legends
fig = go.Figure(
    data=[
        go.Scatter(
            x=S, y=call_price, mode='lines',
            line=dict(color='#f43f5e', width=4),
            showlegend=False
        ),
        go.Scatter(
            x=[slider_x, slider_x], y=[0, slider_y], mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            showlegend=False
        ),
        go.Scatter(
            x=[slider_x], y=[slider_y], mode='markers+text',
            marker=dict(color='cyan', size=10, symbol='circle'),
            text=[f"BS Price: {slider_y:.2f}"],
            textposition="top left",
            showlegend=False
        )
    ]
)

# Slider code (1 slider, 1 handle)
steps = []
for i in range(len(S)):
    si = float(S[i])
    yi = float(call_price[i])
    step = dict(
        method="update",
        args=[
            {
                "x": [S, [si, si], [si]],
                "y": [call_price, [0, yi], [yi]],
                "text": [None, None, [f"BS Price: {yi:.2f}"]],
            },
            {
                "title": f"Black-Scholes European Call Price<br>Stock @ t = {si:.2f} | BS Price = {yi:.2f}",
                "showlegend": False
            }
        ],
        label=f"{si:.2f}"
    )
    steps.append(step)

sliders = [dict(
    active=slider_start_idx,
    currentvalue={"prefix": "Stock Price (Stock @ t) = "},
    pad={"t": 30},
    steps=steps,
)]

initial_title = f"Black-Scholes European Call Price<br>Stock @ t = {slider_x:.2f} | BS Price = {slider_y:.2f}"

fig.update_layout(
    title=initial_title,
    xaxis_title="Underlying Price (Stock @ t)",
    yaxis_title="Black-Scholes Option Price",
    height=400,
    sliders=sliders,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False  # remove the legend globally
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()


# ##### **Black-Scholes Model:** The solution to the Black-Scholes Equation gives us the Black-Scholes Model, Function(al) $V$
# 
# The solution $V$ to the Black-Scholes model for a European call option is:
# 
# $$
# V(S, t) = C(S, t) = S \, N(d_1) - K e^{-r(T-t)} N(d_2)
# $$
# 
# where everything required for $V$ is known at time $t$, giving us the option price today!


# ---


# #### 3.) 📈 Fundamental Theorem of Asset Pricing
# 
# The Fundamental Theorem of Asset Pricing (FTAP) underpins all of modern financial mathematics. 
# 
# The FTAP tells us that the current value of a financial derivative, denoted as $V$, is equal to the expected value of its future payoff (under a special "risk-neutral" probability), discounted at the risk-free rate. 
# 
# In other words, $V$ represents the fair present price of a contingent claim when there is no possibility of arbitrage.
# 
# $$
# V(S, t) = e^{-r(T-t)}\mathbb{E}^{\mathbb{Q}}[\text{Payoff at } T \mid S_t = S]
# $$
# 
# In other words, using the Law of Large Numbers (LLN), we can solve for this expectation $V$!


import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp

# --- Parameters ---
S0 = 100         # initial stock price
K = 100          # strike price
r = 0.05         # risk-free rate
sigma = 0.2      # volatility
T = 1.0          # time to maturity (years)
t0 = 0.5         # current intermediate time (so tau = T-t0 as in prior context)

# Left plot: Simulate several GBM paths from S0 to T
n_paths = 15
n_steps = 100
dt = (T - t0) / n_steps

time_grid = np.linspace(t0, T, n_steps+1)
S_paths = np.zeros((n_paths, n_steps+1))
S_paths[:, 0] = S0

for i in range(n_paths):
    for j in range(1, n_steps+1):
        Z = np.random.randn()
        S_paths[i, j] = S_paths[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Right plot: Convergence of Monte Carlo FTAP option price as n_sim increases
n_samples = 40  # moderate granularity on log scale
n_seq = np.unique(np.geomspace(100, 10000, n_samples).astype(int))
tau = T - t0

# Generate a single large batch of random numbers for reproducibility and statistical tightness
Z_big = np.random.randn(n_seq[-1])
ST_big = S0 * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * Z_big)
payoff_big = np.maximum(ST_big - K, 0)

mc_prices = np.empty_like(n_seq, dtype=float)
for idx, n_mc in enumerate(n_seq):
    mc_prices[idx] = np.exp(-r * tau) * payoff_big[:n_mc].mean()

bs_price = None
try:
    bs_price = euro_call(S0, K, r, sigma, tau)
except Exception:
    pass

# Create subplot: left (paths), right (MC convergence)
fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Sample Risk-Neutral GBM Paths (S<sub>T</sub>)",
        "FTAP Convergence to Option Price"
    )
)

# Left: plot ~15 GBM paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=S_paths[i],
            line=dict(width=2, color="#818cf8"),
            opacity=0.5,
            showlegend=False
        ),
        row=1, col=1
    )
# Add strike price K as a horizontal line (hline) on left plot
fig.add_shape(
    dict(
        type="line",
        x0=time_grid[0],
        x1=time_grid[-1],
        y0=K,
        y1=K,
        line=dict(color="crimson", dash="dash", width=2),
    ),
    row=1, col=1
)
# Add terminal time T as a vertical line (vline) on left plot
fig.add_shape(
    dict(
        type="line",
        x0=T,
        x1=T,
        y0=np.min(S_paths),
        y1=np.max(S_paths),
        line=dict(color="orange", dash="dot", width=2),
    ),
    row=1, col=1
)
# Updated: Set gridline color to rgba(128,128,128,0.2) to match other plots
fig.update_xaxes(title_text="Time t", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text="Asset Price S", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

# Right: plot FTAP MC call price convergence (now smooth and tight) & BS price as dashed reference
fig.add_trace(
    go.Scatter(
        x=n_seq,
        y=mc_prices,
        mode='lines+markers',
        line=dict(color="#fbbf24", width=2),
        marker=dict(size=5, color="#fde68a"),
        name="FTAP Value",
        showlegend=False
    ),
    row=1, col=2
)
if bs_price is not None:
    fig.add_trace(
        go.Scatter(
            x=[n_seq[0], n_seq[-1]],
            y=[bs_price, bs_price],
            mode='lines',
            line=dict(color='#818cf8', dash='dash', width=2),
            name="Black-Scholes Price",
            showlegend=False  # hide legend here also
        ),
        row=1, col=2
    )
# Updated: Set gridline color on right plot to match
fig.update_xaxes(title_text="Number of Simulated Paths", row=1, col=2, type="log", showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text="Call Option Value", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_layout(
    title_text="GBM Path Simulation under Risk-Neutral Measure & FTAP Monte Carlo Convergence",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=400,
    showlegend=False
)
fig.show()



# ##### **Remark:** Using Monte Carlo Simulation (LLN) the Fundamental Theorem of Asset Pricing gives us the Function(al) $V$!


# ###### ______________________________________________________________________________________________________________________________________


# ##### **Question:**  Which choice of $V$ is *correct*?
# 
# Under the **Black-Scholes** framework we have:
# 
# $$
# V_{BS}(S, t) = C(S, t) = S \, N(d_1) - K e^{-r(T-t)} N(d_2)
# $$
# 
# Under the **Fundamental Theorem of Asset Pricing** we have:
# 
# $$
# V_{FTAP}(S, t) = e^{-r(T-t)}\mathbb{E}^{\mathbb{Q}}[\text{Payoff at } T \mid S_t = S]
# $$
# 
# Which should we use to price our options today?  Well let's look at the option prices for each approach and see where they differ. . .


# Show—side by side—the Black-Scholes formula price and a FTAP (risk-neutral expectation) price, each evaluated at select asset prices S at a fixed time t0 (not along a single path).
# This compares option prices from both methods over a spread of S at a given t.

# --- Parameter set ---
S0 = 100         # initial stock price
K = 100          # strike price
r = 0.05         # risk-free rate
sigma = 0.2      # volatility
T = 1.0          # time to maturity (years)
t0 = 0.5         # evaluate halfway to expiry

np.random.seed(42)

# Asset grid (10 values around ATM)
S_grid = np.linspace(0.7 * K, 1.3 * K, 10)

# Black-Scholes pricing function (assumes euro_call(S, K, r, sigma, tau) is defined)
def bs_call_price(S, K, r, sigma, tau):
    return euro_call(S, K, r, sigma, tau)

# FTAP (Fundamental Theorem of Asset Pricing) function for European call at given S, t
def ftap_call_price(S, K, r, sigma, tau, n_mc=10000):
    # Simulate n_mc endpoints with risk-neutral drift
    Z = np.random.randn(n_mc)
    ST = S * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * tau) * payoff.mean()

# Calculate both prices for S_grid at t0
tau = T - t0
bs_prices_grid = np.array([bs_call_price(S, K, r, sigma, tau) for S in S_grid])
ftap_prices_grid = np.array([ftap_call_price(S, K, r, sigma, tau) for S in S_grid])

# Plot: x = S, y = price, scatter for each method, REMOVE LEGEND by not setting name and hiding legend
fig_compare = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=("Black-Scholes Price for S @ t", "FTAP Price for S @ t")
)

fig_compare.add_trace(
    go.Scatter(
        x=S_grid,
        y=bs_prices_grid,
        mode='markers+lines',
        marker=dict(color='#818cf8', symbol='circle', size=10),
        line=dict(color='#818cf8'),
        showlegend=False  # Remove legend entry
    ),
    row=1, col=1
)

fig_compare.add_trace(
    go.Scatter(
        x=S_grid,
        y=ftap_prices_grid,
        mode='markers+lines',
        marker=dict(color='#fbbf24', symbol='square', size=10),
        line=dict(color='#fbbf24'),
        showlegend=False  # Remove legend entry
    ),
    row=1, col=2
)

# Set gridline color to match earlier payoff plot section
gridcolor = 'rgba(128,128,128,0.2)'
fig_compare.update_xaxes(
    title_text="Asset Price S",
    row=1,
    col=1,
    showgrid=True,
    gridwidth=1,
    gridcolor=gridcolor
)
fig_compare.update_xaxes(
    title_text="Asset Price S",
    row=1,
    col=2,
    showgrid=True,
    gridwidth=1,
    gridcolor=gridcolor
)
fig_compare.update_yaxes(
    title_text="Call Option Value",
    row=1,
    col=1,
    showgrid=True,
    gridwidth=1,
    gridcolor=gridcolor
)
fig_compare.update_yaxes(
    title_text="Call Option Value",
    row=1,
    col=2,
    showgrid=True,
    gridwidth=1,
    gridcolor=gridcolor
)
fig_compare.update_layout(
    title_text="European Call Option: Black-Scholes and FTAP Pricing for S @ t = %.2f" % t0,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=400,
    showlegend=False
)

fig_compare.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### **Observation:** It Appears Under Both Frameworks we get the Same Prices!
# 
# Is there a way to prove
# 
# $$V(S_t, t) = V_{BS}(S_t, t) = V_{FTAP}(S_t, t)$$
# 
# There sure is, and it's thanks to Physics and the Feyman-Kac theorem.


# ---


# #### 3.) ⚛️ Feynman-Kac Theorem
# 
# Feynman-Kac bridging the gap between Black-Scholes and FTAP step-by-step
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **Step 1. Black-Scholes SDE and Discounted Value Process**
# 
# Recall the Black-Scholes dynamics under the risk-neutral measure $\mathbb{Q}$:
# 
# $
# dS_t = r S_t\,dt + \sigma S_t\,dW_t
# $
# 
# where:
# - $S_t$: asset price
# - $r$: risk-free rate
# - $\sigma$: volatility
# - $W_t$: standard Brownian motion under $\mathbb{Q}$
# 
# Let $V(S_t, t)$ be the price of a European call option (or other contingent claim).
# Define the **discounted option value**:
# 
# $
# Y_t = e^{-r t} V(S_t, t)
# $
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **Step 2. 2D Product Rule (Itô's Lemma on Discounted Value)**
# 
# Apply Itô’s lemma (the multidimensional product rule) to $Y_t$:
# 
# - $f(t) = e^{-r t}$
# - $g(S_t, t) = V(S_t, t)$
# 
# $
# dY_t = d(e^{-r t} V(S_t, t))
# $
# 
# By the product rule:
# 
# $
# dY_t = e^{-rt} dV(S_t, t) + V(S_t, t) d(e^{-rt}) + d\langle e^{-rt}, V(S_t, t)\rangle
# $
# 
# But $e^{-rt}$ has deterministic variation only, so the cross-variation vanishes.
# 
# Now,
# 
# $
# d(e^{-rt}) = -r e^{-rt} dt
# $
# 
# and by Itô's Lemma:
# 
# $
# dV(S_t, t) = \left( \frac{\partial V}{\partial t} + r S_t \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \sigma S_t \frac{\partial V}{\partial S}\, dW_t
# $
# 
# Substitute in:
# 
# $
# dY_t = e^{-rt}\Bigg[
# \left(\frac{\partial V}{\partial t} + r S_t \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} - r V \right)dt + \sigma S_t \frac{\partial V}{\partial S}\, dW_t
# \Bigg]
# $
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **Step 3. Identifying the Martingale**
# 
# For no-arbitrage (risk-neutral pricing), **the discounted price process $Y_t$ should be a martingale**:
# 
# $
# \mathbb{E}^{\mathbb{Q}}[Y_T \mid \mathcal{F}_t] = Y_t
# $
# 
# which requires the **drift term to vanish**:
# 
# $
# \frac{\partial V}{\partial t} + r S_t \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} - r V = 0
# $
# 
# which is precisely the **Black-Scholes PDE**:
# 
# $
# \frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = r V
# $
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **Step 4. Feynman-Kac Connection**
# 
# The **Feynman-Kac theorem** says:
# 
# The unique solution to this PDE with final condition $V(S_T, T) = \Phi(S_T)$ (the option payoff) is
# 
# $
# V(S_t, t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T-t)} \Phi(S_T) \;\big|\; S_t \right]
# $
# 
# That is, the current value equals the **discounted expected payoff** under the risk-neutral measure, where the expectation is over all $S_T$ generated by the SDE above.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# Summary Table
# 
# | Step        | Mathematics                                                       | Financial Insight  |
# |-------------|-------------------------------------------------------------------|--------------------|
# | 1. SDE      | $dS_t = rS_t\,dt + \sigma S_t\,dW_t$                              | Risk-neutral dynamics |
# | 2. Product Rule | Itô’s Lemma for $Y_t = e^{-rt}V(S_t, t)$                      | Discounted value process |
# | 3. Martingale | Drift of $Y_t$ vanishes $\implies$ Black-Scholes PDE            | No-arbitrage pricing |
# | 4. Feynman-Kac | Solution: $V(S_t, t) = \mathbb{E}^{\mathbb{Q}}[e^{-r(T-t)} \Phi(S_T)]$ | Price as risk-neutral discounted expectation |
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **Thus, the Feynman-Kac theorem provides the bridge between the Black-Scholes PDE and the probabilistic (expectational) solution for option prices.**
# 
# 


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Finding the value of any derivative contract at maturity is trivial, it's just the payoff
# - The real question here is what should we pay *today* for that potential payoff?
# - The Black-Scholes framework introduces a portfolio replication argument under no arbitrage assumptions and finds a pricing PDE that yields the function(al) of interest $V$ telling us the price of a European option today
# - The Fundamental Theorem of Asset Pricing suggests that the payoff (in a complete market) is the risk-neutral discounted expectation of the payoff, giving us yet another way to find the value of a European option today, that function(al) $V$
# - Which is correct?  We observed how both solutions yielded identical values for the option contract price, can we prove they are the same?
# - We sure can!  The Feynman-Kac theorem establishes a connection between the stochastic (probabilistic) representation given by the FTAP and the PDE that must hold in the Black-Scholes framework proving they are equivalent!
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Reinforcement Learning for Delta Hedging
# - Approximating Pricing Functionals using Neural Networks
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Neural Network Stochastic Volatility Model Calibration
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

