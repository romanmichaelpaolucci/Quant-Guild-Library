### 📈 Quant Explains Risk-Neutral Option Pricing

##### ▶️ Related Quant Guild Videos:

- [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)

- [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)

- [Central Limit Theorem for Quant Finance](https://youtu.be/q2era-4pnic)

- [Why Quant Models Break](https://youtu.be/brdG1TmsPlw)

- [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)

- [The 5 Papers That Build Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)

###### ______________________________________________________________________________________________________________________________________

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)



##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

##### [👾 Join the Quant Guild Discord Server](discord.com/invite/MJ4FU2c6c3)

---


```python
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
```

### 📖 Sections

#### 1.) 📜 European Option Contracts

- European Option Contract Payoffs

- What is the Price for that Potential Payoff Today

#### 2.) 📈 Pricing Under a Stochastic Model

- If we Assume an SDE, what Should Parameters be? (Mispecification is highly likely)

- Black-Scholes Replication Argument (Mu is hedged away, so itrs irrelevant)

#### 3.) 🌪️ Risk-Neutrality and the $\mathbb{P} \rightarrow \mathbb{Q}$ Change of Measure

- Cox-Ross (1976)

- Harrison and Pliska (1979/1981)

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📜 European Option Contracts

European call (put) options give the holder the right but not obligation to buy (sell) the underlying asset at time $T$ for price $K$

The value of the option at maturity is easily computed as the max between the difference

$$C = max(S_T - K, 0) \quad \quad P = max(K - S_T, 0)$$


```python
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

# Colors
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#00ff88"
path_color = "#00d1ff"

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
            go.Scatter(x=S_range, y=payoff_vals),
            go.Scatter(x=[K, K], y=[0, 60]),
            go.Scatter(x=[current_price, current_price], y=[0, 60]),
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K]),
            go.Scatter(x=dates[:i+1], y=path[:i+1])
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
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Payoff ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

##### What Should we Pay for this Cash Flow Potential?

---

#### 3.) 📈 Pricing Under a Stochastic Model

There are several models we can use

 $$\frac{dS_t}{S_t} = \mu dt + \sigma dW_t \quad\quad W = \{W_t:\geq0\} \text{ is a standard Brownian motion on } (\Omega, \mathcal{F}, \mathbb{P})$$

 Instead of solving this SDE directly, we can use the Euler-Maruyama (EM) method to approximate its evolution numerically:

 $$S_{t+\Delta t} \approx S_t + \mu S_t \Delta t + \sigma S_t \sqrt{\Delta t}\; Z,\quad Z \sim \mathcal{N}(0,1)$$

 This approach is commonly used when analytic solutions aren't available, but can be used for the GBM SDE as well.

###### ______________________________________________________________________________________________________________________________________

##### What Parameters do we Have to Estimate?
 
 The geometric Brownian motion (GBM) SDE has the analytic (closed-form) solution:
 
 $$
 S_t = S_0 \; \exp \left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t \right]
 $$

$S_0$ is spot price at time $t=0$ given by the market, in our model we can only select $\mu, \sigma$


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# --- 1. Parameters ---
S0 = 100        # Initial price
T = 1.0         # Time to maturity
n_steps = 100   # Time resolution
n_paths = 80    # Number of paths (increased slightly for better density)
dt = T / n_steps

# Scenario 1: Low Mean, Low Variance
mu_low = 0.10      # 5% Drift
sigma_low = 0.10   # 10% Volatility

# Scenario 2: High Mean, High Variance
mu_high = 0.20     # 20% Drift
sigma_high = 0.30  # 30% Volatility

# --- 2. Simulation (Optimized) ---
def simulate_gbm_optimized(drift, vol, steps, paths):
    np.random.seed(42) # Consistent seed
    dt_sqrt = np.sqrt(dt)
    drift_term = (drift - 0.5 * vol**2) * dt
    
    # Random Walk
    dW = np.random.normal(0, 1, (steps, paths))
    log_ret = drift_term + vol * dt_sqrt * dW
    
    # Cumulative sum
    log_paths = np.cumsum(log_ret, axis=0)
    log_paths = np.vstack([np.zeros((1, paths)), log_paths]) # Add t=0
    
    return S0 * np.exp(log_paths)

# Generate Data
paths_low = simulate_gbm_optimized(mu_low, sigma_low, n_steps, n_paths)
paths_high = simulate_gbm_optimized(mu_high, sigma_high, n_steps, n_paths)
dates = np.linspace(0, T, n_steps+1)

# Theoretical Expected Value Lines E[S_t] = S0 * exp(mu * t)
exp_low = S0 * np.exp(mu_low * dates)
exp_high = S0 * np.exp(mu_high * dates)

# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f"<b>Scenario A: Low Estimate</b><br>μ={mu_low}, σ={sigma_low}", 
        f"<b>Scenario B: High Estimate</b><br>μ={mu_high}, σ={sigma_high}",
        "Terminal Distribution (Low)", 
        "Terminal Distribution (High)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_low = "#00D2FF"       # Cyan/Blue for Low
c_low_ghost = "rgba(0, 210, 255, 0.15)"
c_high = "#FF5500"      # Orange/Red for High
c_high_ghost = "rgba(255, 85, 0, 0.15)"
c_transparent = "rgba(0,0,0,0)"

# --- Helper: Flatten paths for single-trace optimization ---
def flatten_paths(paths_arr, time_arr):
    n_t, n_p = paths_arr.shape
    x_flat = []
    y_flat = []
    for p in range(n_p):
        x_flat.extend(time_arr)
        x_flat.append(None)
        y_flat.extend(paths_arr[:, p])
        y_flat.append(None)
    return x_flat, y_flat

# --- Initial Static Traces (Drift Lines) [Indices 0, 1] ---
fig.add_trace(go.Scatter(x=dates, y=exp_low, mode='lines', line=dict(color='white', width=2, dash='dash'), name='Exp Low'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=exp_high, mode='lines', line=dict(color='white', width=2, dash='dash'), name='Exp High'), row=1, col=2)

# --- Initial Dynamic Traces (Paths) [Indices 2, 3] ---
x0_low, y0_low = flatten_paths(paths_low[:1, :], dates[:1])
x0_high, y0_high = flatten_paths(paths_high[:1, :], dates[:1])

fig.add_trace(go.Scatter(x=x0_low, y=y0_low, mode='lines', line=dict(color=c_low, width=1), opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=x0_high, y=y0_high, mode='lines', line=dict(color=c_high, width=1), opacity=0.5), row=1, col=2)

# --- Initial KDE Traces [Indices 4, 5, 6, 7] ---
# Widen grid to accommodate high variance
x_grid = np.linspace(20, 250, 300)
y_zero = np.zeros_like(x_grid)

# Left Plot (Low Scenario): Show Ghost High then Main Low
# Trace 4: Ghost High
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_high_ghost, name='Ghost High'), row=2, col=1) 
# Trace 5: Main Low
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_low), name='Main Low'), row=2, col=1) 

# Right Plot (High Scenario): Show Ghost Low then Main High
# Trace 6: Ghost Low
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_low_ghost, name='Ghost Low'), row=2, col=2) 
# Trace 7: Main High
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_high), name='Main High'), row=2, col=2) 

# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps+1, 2) 

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # 1. Prepare Path Data
    curr_dates = dates[:t+1]
    x_L, y_L = flatten_paths(paths_low[:t+1, :], curr_dates)
    x_H, y_H = flatten_paths(paths_high[:t+1, :], curr_dates)
    
    # 2. Prepare KDE Data
    if t > 5:
        # Low KDE
        kde_L = gaussian_kde(paths_low[t, :])
        y_kde_L = kde_L(x_grid)
        # High KDE
        kde_H = gaussian_kde(paths_high[t, :])
        y_kde_H = kde_H(x_grid)
    else:
        y_kde_L = y_zero
        y_kde_H = y_zero

    # 3. Create Frame
    # Updates Traces: 2 (Low Path), 3 (High Path), 4 (Ghost H), 5 (Main L), 6 (Ghost L), 7 (Main H)
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_L, y=y_L),   # Trace 2
            go.Scatter(x=x_H, y=y_H),   # Trace 3
            go.Scatter(y=y_kde_H),      # Trace 4 (Ghost High on Left)
            go.Scatter(y=y_kde_L),      # Trace 5 (Main Low on Left)
            go.Scatter(y=y_kde_L),      # Trace 6 (Ghost Low on Right)
            go.Scatter(y=y_kde_H)       # Trace 7 (Main High on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7], 
        name=frame_name
    ))
    
    # 4. Create Slider Step
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1200,
    margin=dict(t=60, b=100, l=60, r=60),
    showlegend=False,
    
    # Controls: Play/Pause
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.15,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    # Controls: Slider
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Axis Styling
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
price_range = [40, 240] # Widened range for high vol

# Top Row
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

# Bottom Row
fig.update_yaxes(range=[0, 0.09], title="Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.09], title="Density", row=2, col=2, **ax_opts)
fig.update_xaxes(range=price_range, title="Terminal Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=price_range, title="Terminal Price", row=2, col=2, **ax_opts)

fig.show()
```

#### Maximum Likelihood Estimation (MLE) for $\mu$ and $\sigma$ in GBM

 For a time series of prices $S_0, S_1, ..., S_n$ observed at equal time intervals $\Delta t$,
 the GBM model implies that the log returns are normally distributed:

 $$
 X_i = \ln\left(\frac{S_i}{S_{i-1}}\right) \sim \mathcal{N}\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t,\ \sigma^2 \Delta t \right)
 $$

 **Given:** Observed prices $S_0, S_1, ..., S_n$

 1. Compute log returns:
 $$
 x_i = \ln\left(\frac{S_i}{S_{i-1}}\right),\qquad i=1,...,n
 $$

 2. The log-likelihood function of the sample is:
 $$
 \ell(\mu, \sigma) = -\frac{n}{2} \ln(2\pi \sigma^2 \Delta t) - \frac{1}{2 \sigma^2 \Delta t} \sum_{i=1}^{n}\left( x_i - \left[\mu - \frac{1}{2}\sigma^2\right]\Delta t \right)^2
 $$

 3. **MLE Solution:**
 Take partial derivatives of $\ell$ with respect to $\mu$ and $\sigma$, set to zero, and solve the system:

 $$
 \textbf{MLE for}~\mu: \qquad 
 \hat{\mu} = \frac{1}{n \Delta t} \sum_{i=1}^n x_i + \frac{1}{2} \hat{\sigma}^2
 $$

 $$
 \textbf{MLE for}~\sigma^2: \qquad
 \hat{\sigma}^2 = \frac{1}{n\Delta t} \sum_{i=1}^n \left(x_i - \bar{x}\right)^2
 $$
 where $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

 This provides the closed-form solution for the MLEs of the drift $\mu$ and volatility $\sigma$ from historical price data under the GBM.


###### ______________________________________________________________________________________________________________________________________

##### What if we Estimate Parameters During a Bull Run?


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity
r = 0.15        # Drift
vol = 0.15      # Volatility
n_steps = 100   # Number of simulation steps
n_paths = 100   # Number of paths to simulate
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Generate 100 Price Paths
np.random.seed(42)
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), (n_steps, n_paths))
paths = S0 * np.exp(np.cumsum(rets, axis=0))
paths = np.vstack([np.full((1, n_paths), S0), paths]) 

dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')
S_range = np.linspace(50, 180, 200)
payoff_vals = call_payoff(S_range, K)

# --- Colors ---
off_white = "#e0e0e0"
strike_color = "#ff4b4b"     # Red
payoff_color = "#00ff88"     # Green

# Define RGBA colors with opacity for the 100 lines so they blend well
# ITM = In The Money (Green), OTM = Out The Money (Red)

# For vertical lines (Left Chart) - slightly more opaque
itm_line_color = "rgba(0, 255, 136, 0.4)" 
otm_line_color = "rgba(255, 75, 75, 0.4)"

# For paths (Right Chart) - more transparent to handle overlap
itm_path_color = "rgba(0, 255, 136, 0.15)"
otm_path_color = "rgba(255, 75, 75, 0.15)"

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff (Green=ITM, Red=OTM)", "Monte Carlo Simulation"),
    horizontal_spacing=0.12
)

# -- Static Background Traces --
# Left Chart: Payoff Curve
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3), name="Payoff"), row=1, col=1)
# Left Chart: Strike Line
fig.add_trace(go.Scatter(x=[K, K], y=[0, 80], line=dict(color=off_white, width=1, dash='dash'), name="Strike"), row=1, col=1)

# Left Chart: 100 Vertical Lines (Initial State)
# S0=100 is less than K=105, so they start Red (OTM)
for j in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[S0, S0], y=[0, 80],
            mode='lines',
            line=dict(color=otm_line_color, width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1
    )

# Right Chart: Strike Line
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=off_white, width=1, dash='dash'), showlegend=False), row=1, col=2)

# Right Chart: 100 Paths (Initial State)
# Starts at S0 (OTM), so Red
for j in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[dates[0]], y=[S0],
            mode='lines',
            line=dict(color=otm_path_color, width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=2
    )

# --- 3. Animation Logic ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    frame_name = f"f{i}"
    frame_data = []
    
    # 1. Payoff (Static)
    frame_data.append(go.Scatter(x=S_range, y=payoff_vals)) 
    # 2. Strike Left (Static)
    frame_data.append(go.Scatter(x=[K, K], y=[0, 80]))
    
    # 3...102. Update Vertical Lines (Left) with Dynamic Color
    current_prices = paths[i]
    for price in current_prices:
        # Determine color based on current price vs Strike
        c = itm_line_color if price > K else otm_line_color
        frame_data.append(go.Scatter(x=[price, price], y=[0, 80], line=dict(color=c)))
        
    # 103. Strike Right (Static)
    frame_data.append(go.Scatter(x=[dates[0], dates[-1]], y=[K, K]))
    
    # 104...203. Update Paths (Right) with Dynamic Color
    for j in range(n_paths):
        price = paths[i, j]
        # Color the whole path based on where the head is currently
        c = itm_path_color if price > K else otm_path_color
        frame_data.append(go.Scatter(x=dates[:i+1], y=paths[:i+1, j], line=dict(color=c)))

    frames.append(go.Frame(data=frame_data, name=frame_name))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=100),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Run",
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
        "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 14, "color": off_white}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85, "x": 0.15, "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[50, 180], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-5, 80], title_text="Payoff ($)", row=1, col=1)

fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[50, 180], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

We likely would say, if the market will evolve this way we should pay *more* for the option

###### ______________________________________________________________________________________________________________________________________

##### What if we Estimated Parameters from Historic Data During a Period of Distress?


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity
r = -0.15        # Drift
vol = 0.15      # Volatility
n_steps = 100   # Number of simulation steps
n_paths = 100   # Number of paths to simulate
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Generate 100 Price Paths
np.random.seed(42)
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), (n_steps, n_paths))
paths = S0 * np.exp(np.cumsum(rets, axis=0))
paths = np.vstack([np.full((1, n_paths), S0), paths]) 

dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')
S_range = np.linspace(50, 180, 200)
payoff_vals = call_payoff(S_range, K)

# --- Colors ---
off_white = "#e0e0e0"
strike_color = "#ff4b4b"     # Red
payoff_color = "#00ff88"     # Green

# Define RGBA colors with opacity for the 100 lines so they blend well
# ITM = In The Money (Green), OTM = Out The Money (Red)

# For vertical lines (Left Chart) - slightly more opaque
itm_line_color = "rgba(0, 255, 136, 0.4)" 
otm_line_color = "rgba(255, 75, 75, 0.4)"

# For paths (Right Chart) - more transparent to handle overlap
itm_path_color = "rgba(0, 255, 136, 0.15)"
otm_path_color = "rgba(255, 75, 75, 0.15)"

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff (Green=ITM, Red=OTM)", "Monte Carlo Simulation"),
    horizontal_spacing=0.12
)

# -- Static Background Traces --
# Left Chart: Payoff Curve
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3), name="Payoff"), row=1, col=1)
# Left Chart: Strike Line
fig.add_trace(go.Scatter(x=[K, K], y=[0, 80], line=dict(color=off_white, width=1, dash='dash'), name="Strike"), row=1, col=1)

# Left Chart: 100 Vertical Lines (Initial State)
# S0=100 is less than K=105, so they start Red (OTM)
for j in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[S0, S0], y=[0, 80],
            mode='lines',
            line=dict(color=otm_line_color, width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1
    )

# Right Chart: Strike Line
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=off_white, width=1, dash='dash'), showlegend=False), row=1, col=2)

# Right Chart: 100 Paths (Initial State)
# Starts at S0 (OTM), so Red
for j in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[dates[0]], y=[S0],
            mode='lines',
            line=dict(color=otm_path_color, width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=2
    )

# --- 3. Animation Logic ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    frame_name = f"f{i}"
    frame_data = []
    
    # 1. Payoff (Static)
    frame_data.append(go.Scatter(x=S_range, y=payoff_vals)) 
    # 2. Strike Left (Static)
    frame_data.append(go.Scatter(x=[K, K], y=[0, 80]))
    
    # 3...102. Update Vertical Lines (Left) with Dynamic Color
    current_prices = paths[i]
    for price in current_prices:
        # Determine color based on current price vs Strike
        c = itm_line_color if price > K else otm_line_color
        frame_data.append(go.Scatter(x=[price, price], y=[0, 80], line=dict(color=c)))
        
    # 103. Strike Right (Static)
    frame_data.append(go.Scatter(x=[dates[0], dates[-1]], y=[K, K]))
    
    # 104...203. Update Paths (Right) with Dynamic Color
    for j in range(n_paths):
        price = paths[i, j]
        # Color the whole path based on where the head is currently
        c = itm_path_color if price > K else otm_path_color
        frame_data.append(go.Scatter(x=dates[:i+1], y=paths[:i+1, j], line=dict(color=c)))

    frames.append(go.Frame(data=frame_data, name=frame_name))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=100),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Run",
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
        "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 14, "color": off_white}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85, "x": 0.15, "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[50, 180], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-5, 80], title_text="Payoff ($)", row=1, col=1)

fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[50, 180], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

We likely would say, if the market will evolve this way we should pay *less* for the option

###### ______________________________________________________________________________________________________________________________________

##### In Reality, Parameter Estimation (and Misparameterization and Model Mispecification) Break Models

Some break more violently than others

Some break faster than others

Sometimes its not just misparameterization but model mispecification

This applies to literally everything, this generalizes the model development process


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity
r = 0.05        # Risk-free rate
n_steps = 200   # High resolution for smooth GBM
n_paths = 100   # Number of paths
dt = T / n_steps

# -- GBM Parameters --
vol_gbm = 0.15  

# -- Bates Parameters --
kappa = 4.0     
theta = 0.16    
xi = 0.6        
rho = -0.7      
v0 = 0.04       
lambda_j = 6.0  
mu_j = -0.08    
sigma_j = 0.10  

# --- 2. Simulation Functions ---

def simulate_gbm(n_steps, n_paths):
    np.random.seed(50) 
    drift = (r - 0.5 * vol_gbm**2) * dt
    diffusion = vol_gbm * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_paths))
    log_rets = drift + diffusion
    paths = S0 * np.exp(np.cumsum(log_rets, axis=0))
    return np.vstack([np.full((1, n_paths), S0), paths])

def simulate_bates(n_steps, n_paths):
    np.random.seed(60)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    v[0] = v0
    
    Z1 = np.random.normal(0, 1, (n_steps, n_paths)) 
    Z2 = np.random.normal(0, 1, (n_steps, n_paths)) 
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    jump_prob = lambda_j * dt
    is_jump = np.random.rand(n_steps, n_paths) < jump_prob
    jump_sizes = np.random.normal(mu_j, sigma_j, (n_steps, n_paths))
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    
    for t in range(n_steps):
        v_curr = np.maximum(v[t], 0)
        dv = kappa * (theta - v_curr) * dt + xi * np.sqrt(v_curr) * Z2[t] * np.sqrt(dt)
        v[t+1] = np.maximum(v[t] + dv, 1e-6)
        
        drift_term = (r - lambda_j * k - 0.5 * v_curr) * dt
        diff_term = np.sqrt(v_curr) * Z1[t] * np.sqrt(dt)
        jump_term = np.where(is_jump[t], jump_sizes[t], 0)
        
        S[t+1] = S[t] * np.exp(drift_term + diff_term + jump_term)
        
    return S

# Generate Data
gbm_paths = simulate_gbm(n_steps, n_paths)
bates_paths = simulate_bates(n_steps, n_paths)

dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')
S_range = np.linspace(20, 200, 300) 
payoff_vals = np.maximum(S_range - K, 0)

# --- 3. Graphics Setup ---
off_white = "#e0e0e0"

# --- UPDATED OPACITY SETTINGS ---
# Increased from 0.1 to 0.4 for paths
itm_c_path = "rgba(0, 255, 136, 0.4)" 
otm_c_path = "rgba(255, 75, 75, 0.4)"

# Vertical lines remain slightly more solid (0.6)
itm_c_line = "rgba(0, 255, 136, 0.6)"
otm_c_line = "rgba(255, 75, 75, 0.6)"

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "GBM (Smooth Diffusion)", 
        "Bates Model (Jumps + Stochastic Vol)",
        "GBM Payoff (Normal Dist)", 
        "Bates Payoff (Fat Tails/Crash Risk)"
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# --- Static Traces (Backgrounds) ---
for col in [1, 2]:
    # Bottom Row: Payoff Curve
    fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color="#00ff88", width=3), name="Payoff"), row=2, col=col)
    # Bottom Row: Strike Line
    fig.add_trace(go.Scatter(x=[K, K], y=[0, 100], line=dict(color=off_white, width=1, dash='dash'), name="Strike"), row=2, col=col)
    # Top Row: Strike Line
    fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=off_white, width=1, dash='dash'), showlegend=False), row=1, col=col)

# --- Initial Dynamic Traces (Step 0) ---
for j in range(n_paths):
    # Top Left: GBM Path
    fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], mode='lines', line=dict(width=1, color=otm_c_path)), row=1, col=1)
    # Top Right: Bates Path
    fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], mode='lines', line=dict(width=1, color=otm_c_path)), row=1, col=2)
    # Bottom Left: GBM Vert
    fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 100], mode='lines', line=dict(width=1, color=otm_c_line)), row=2, col=1)
    # Bottom Right: Bates Vert
    fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 100], mode='lines', line=dict(width=1, color=otm_c_line)), row=2, col=2)

# --- 4. Animation Frames ---
frames = []
slider_steps = []

anim_step = 2

for i in range(0, n_steps + 1, anim_step):
    frame_name = f"f{i}"
    frame_data = []
    
    # 1. Static Traces Re-add
    frame_data.append(go.Scatter(x=S_range, y=payoff_vals)) 
    frame_data.append(go.Scatter(x=[K, K], y=[0, 100]))      
    frame_data.append(go.Scatter(x=[dates[0], dates[-1]], y=[K, K])) 
    frame_data.append(go.Scatter(x=S_range, y=payoff_vals)) 
    frame_data.append(go.Scatter(x=[K, K], y=[0, 100]))      
    frame_data.append(go.Scatter(x=[dates[0], dates[-1]], y=[K, K])) 

    # 2. Dynamic Traces (Update Data & Color)
    
    # GBM Paths (Top Left)
    for j in range(n_paths):
        p = gbm_paths[i, j]
        c = itm_c_path if p > K else otm_c_path
        frame_data.append(go.Scatter(x=dates[:i+1], y=gbm_paths[:i+1, j], line=dict(color=c)))

    # Bates Paths (Top Right)
    for j in range(n_paths):
        p = bates_paths[i, j]
        c = itm_c_path if p > K else otm_c_path
        frame_data.append(go.Scatter(x=dates[:i+1], y=bates_paths[:i+1, j], line=dict(color=c)))

    # GBM Verticals (Bottom Left)
    for j in range(n_paths):
        p = gbm_paths[i, j]
        c = itm_c_line if p > K else otm_c_line
        frame_data.append(go.Scatter(x=[p, p], y=[0, 100], line=dict(color=c)))

    # Bates Verticals (Bottom Right)
    for j in range(n_paths):
        p = bates_paths[i, j]
        c = itm_c_line if p > K else otm_c_line
        frame_data.append(go.Scatter(x=[p, p], y=[0, 100], line=dict(color=c)))

    frames.append(go.Frame(data=frame_data, name=frame_name))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout with Bottom Controls ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=850, width=1300,
    margin=dict(t=50, b=150, l=50, r=50), 
    showlegend=False,
    
    # Buttons at Bottom Left
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 10},
        "showactive": False,
        "x": 0.0, "xanchor": "left",
        "y": -0.15, "yanchor": "top"
    }],
    
    # Slider at Bottom Center/Right
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 14, "color": off_white}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 0},
        "len": 0.8,
        "x": 0.15, 
        "y": -0.15,
        "steps": slider_steps
    }]
)

# Axis Styling
ax_style = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=off_white), zeroline=False)

# Top Row (Paths)
fig.update_xaxes(ax_style, title="Date", range=[dates[0], dates[-1]], row=1, col=1)
fig.update_xaxes(ax_style, title="Date", range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(ax_style, title="Price ($)", range=[20, 200], row=1, col=1)
fig.update_yaxes(ax_style, title="Price ($)", range=[20, 200], row=1, col=2)

# Bottom Row (Payoff)
fig.update_xaxes(ax_style, title="Asset Price", range=[20, 200], row=2, col=1)
fig.update_xaxes(ax_style, title="Asset Price", range=[20, 200], row=2, col=2)
fig.update_yaxes(ax_style, title="Payoff ($)", range=[-5, 100], row=2, col=1)
fig.update_yaxes(ax_style, title="Payoff ($)", range=[-5, 100], row=2, col=2)

fig.show()
```

##### We are at Risk not only of Misparameterization but also Model Mispecification

We can address the efficacy of our stochastic model another time, clearly a more important question is

**So, how should we choose parameters for our stochastic model?**

as if we can't reconcile this, it doesn't matter what model we use

---

#### 4.) 🌪️ Risk-Neutrality and the $\mathbb{P} \rightarrow \mathbb{Q}$ Change of Measure

When it comes to pricing, we don't want to leave it up to "let's hope or speculate that we are correct"

Instead, we transition from the real-world measure ($\mathbb{P}$) to the risk-neutral measure ($\mathbb{Q}$)

A change of measure is *just a change in distribution* - that's all it is


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# --- 1. Parameters ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity
r = 0.05        # Risk-free rate (Q Drift)
mu = 0.15       # Real World Drift (P Drift)
sigma = 0.20    # Volatility
n_steps = 100   # Time resolution
n_paths = 50    # Number of paths
dt = T / n_steps

# --- 2. Simulation (Optimized) ---
def simulate_gbm_optimized(drift, vol, steps, paths):
    np.random.seed(42) # Consistent seed
    dt_sqrt = np.sqrt(dt)
    drift_term = (drift - 0.5 * vol**2) * dt
    
    # Random Walk
    dW = np.random.normal(0, 1, (steps, paths))
    log_ret = drift_term + vol * dt_sqrt * dW
    
    # Cumulative sum
    log_paths = np.cumsum(log_ret, axis=0)
    log_paths = np.vstack([np.zeros((1, paths)), log_paths]) # Add t=0
    
    return S0 * np.exp(log_paths)

# Generate Data
paths_P = simulate_gbm_optimized(mu, sigma, n_steps, n_paths)
paths_Q = simulate_gbm_optimized(r, sigma, n_steps, n_paths)
dates = np.linspace(0, T, n_steps+1)

# Theoretical Expected Value Lines
exp_P = S0 * np.exp(mu * dates)
exp_Q = S0 * np.exp(r * dates)

# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Real-World (P) Measure</b><br>Drift = μ (15%)", 
        "<b>Risk-Neutral (Q) Measure</b><br>Drift = r (5%)",
        "Terminal Distribution (P)", 
        "Terminal Distribution (Q)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_P = "#0078FF"        # Blue
c_P_ghost = "rgba(0, 120, 255, 0.15)" # Faint Blue
c_Q = "#B400FF"        # Purple
c_Q_ghost = "rgba(180, 0, 255, 0.15)" # Faint Purple
c_transparent = "rgba(0,0,0,0)"       # Valid transparent color

# --- Helper: Flatten paths for single-trace optimization ---
def flatten_paths(paths_arr, time_arr):
    n_t, n_p = paths_arr.shape
    x_flat = []
    y_flat = []
    for p in range(n_p):
        x_flat.extend(time_arr)
        x_flat.append(None)
        y_flat.extend(paths_arr[:, p])
        y_flat.append(None)
    return x_flat, y_flat

# --- Initial Static Traces (Drift Lines) [Indices 0, 1] ---
fig.add_trace(go.Scatter(x=dates, y=exp_P, mode='lines', line=dict(color='cyan', width=3, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=exp_Q, mode='lines', line=dict(color='magenta', width=3, dash='dash')), row=1, col=2)

# --- Initial Dynamic Traces (Paths) [Indices 2, 3] ---
x0, y0_P = flatten_paths(paths_P[:1, :], dates[:1])
x0, y0_Q = flatten_paths(paths_Q[:1, :], dates[:1])

fig.add_trace(go.Scatter(x=x0, y=y0_P, mode='lines', line=dict(color=c_P, width=1), opacity=0.4), row=1, col=1)
fig.add_trace(go.Scatter(x=x0, y=y0_Q, mode='lines', line=dict(color=c_Q, width=1), opacity=0.4), row=1, col=2)

# --- Initial KDE Traces [Indices 4, 5, 6, 7] ---
x_grid = np.linspace(40, 180, 200)
y_zero = np.zeros_like(x_grid)

# Left Plot (P Measure): Show Ghost Q then Main P
# Trace 4: Ghost Q
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_Q_ghost, name='Ghost Q'), row=2, col=1) 
# Trace 5: Main P
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_P), name='Main P'), row=2, col=1) 

# Right Plot (Q Measure): Show Ghost P then Main Q
# Trace 6: Ghost P
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_P_ghost, name='Ghost P'), row=2, col=2) 
# Trace 7: Main Q
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_Q), name='Main Q'), row=2, col=2) 

# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps+1, 2) 

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # 1. Prepare Path Data
    curr_dates = dates[:t+1]
    x_p, y_p = flatten_paths(paths_P[:t+1, :], curr_dates)
    x_q, y_q = flatten_paths(paths_Q[:t+1, :], curr_dates)
    
    # 2. Prepare KDE Data
    if t > 5:
        # P KDE
        kde_p = gaussian_kde(paths_P[t, :])
        y_kde_p = kde_p(x_grid)
        # Q KDE
        kde_q = gaussian_kde(paths_Q[t, :])
        y_kde_q = kde_q(x_grid)
    else:
        y_kde_p = y_zero
        y_kde_q = y_zero

    # 3. Create Frame
    # Updates Traces: 2 (P Path), 3 (Q Path), 4 (Ghost Q), 5 (Main P), 6 (Ghost P), 7 (Main Q)
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_p, y=y_p),   # Trace 2
            go.Scatter(x=x_q, y=y_q),   # Trace 3
            go.Scatter(y=y_kde_q),      # Trace 4 (Ghost Q on Left)
            go.Scatter(y=y_kde_p),      # Trace 5 (Main P on Left)
            go.Scatter(y=y_kde_p),      # Trace 6 (Ghost P on Right)
            go.Scatter(y=y_kde_q)       # Trace 7 (Main Q on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7], 
        name=frame_name
    ))
    
    # 4. Create Slider Step
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1200,
    margin=dict(t=60, b=100, l=60, r=60),
    showlegend=False,
    
    # Controls: Play/Pause
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.15,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",          # Explicit dark grey background
        "font": {"color": "white"}, # Explicit white text
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    # Controls: Slider
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Axis Styling
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

# Top Row
fig.update_yaxes(range=[50, 180], title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[50, 180], title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

# Bottom Row
fig.update_yaxes(range=[0, 0.06], title="Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.06], title="Density", row=2, col=2, **ax_opts)
fig.update_xaxes(range=[50, 180], title="Terminal Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=[50, 180], title="Terminal Price", row=2, col=2, **ax_opts)

fig.show()
```

###### ______________________________________________________________________________________________________________________________________

Remember, Expectations are just Probability Weighted Sums or Integrals

 $$\mathbb{E}[X] = \sum_{i} x_i\,\mathbb{P}(X = x_i) \quad \quad \mathbb{E}[X] = \int_{-\infty}^{\infty} x\, f_X(x)\,dx$$

In a change of measure, we are simply changing the weights (probabilities) effectively transforming the distribution as you saw above

The key to understand here is that in expectation under the risk-neutral measure $\mathbb{Q}$ we evolve at the risk-free rate

 $$
 \mathbb{E}^{\mathbb{Q}}\left[S_T \mid S_0\right] = S_0\, e^{rT}
 $$
##### What is the Justification for this Risk-Neutral Measure?

###### ______________________________________________________________________________________________________________________________________

##### The Black-Scholes Argument

Assuming a GBM
 
 $$
 dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
 $$

Hedge away the randomness, there is no notion of drift anymore, we get the Black-Scholes Equation

 $$\frac{\partial C}{\partial t} + r S \frac{\partial C}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} = r C$$

A key point of confusion for students here: *if $\mu$ cancels out with the replication argument, why do we simulate with a drift equivalent to the risk-free rate*?
- The replicating portfolio is risk-free and must earn the risk-free rate!
- In simulation, we must represent earning this rate by 
    - $dS_t = r S_t\,dt + \sigma S_t\,dW_t$ 
    - **NOT** $dS_t = 0 + \sigma S_t\,dW_t$

###### ______________________________________________________________________________________________________________________________________

Effectively, risk-premium cancels out as we are no longer compensated for bearing that risk, fun fact Black tried to show this idea with CAPM

 Solving the above parabolic PDE gives us
 
 $$
 C_{\text{BS}}(S, t) = S\, N(d_1) - K e^{-r (T-t)} N(d_2)
 $$
 
 What happens if we simulate the GBM with $\mu = r$ and numerically compute the option price?  
 
 Turns out
  
 $$ C_{\text{BS}}(S, t) = e^{-r (T-t)} \, \mathbb{E}^{\mathbb{Q}} \left[ (S_T - K)^+ \mid S_t = S \right] $$



```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters ---
S0 = 100        # Initial price
K = 105         # Strike price
T = .5         # Time to maturity
r = 0.05        # Risk-free rate
sigma = 0.10    # Volatility
n_steps = 252   # Time steps per path
n_sims = 300    # Total number of paths to simulate
dt = T / n_steps

# --- 2. Calculations ---

# A. Theoretical Black-Scholes Price (The "Target")
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# B. Simulation (Vectorized)
np.random.seed(42)
# Generate all random moves at once: [steps, n_sims]
dW = np.random.normal(0, np.sqrt(dt), (n_steps, n_sims))
# Drift (Risk Neutral)
drift = (r - 0.5 * sigma**2) * dt
# Path generation
log_ret = drift + sigma * dW
log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(log_ret, axis=0)])
paths = S0 * np.exp(log_paths)

# Time axis for plotting
time_grid = np.linspace(0, T, n_steps + 1)

# C. Convergence Data Calculation
# 1. Get Terminal Prices
ST = paths[-1, :]
# 2. Calculate Payoff for each path: max(S_T - K, 0)
payoffs = np.maximum(ST - K, 0)
# 3. Discount these payoffs to PV: e^(-rT) * Payoff
discounted_payoffs = np.exp(-r * T) * payoffs
# 4. Calculate Running Mean (Cumulative Sum / Count)
running_mean = np.cumsum(discounted_payoffs) / np.arange(1, n_sims + 1)

# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Risk-Neutral Paths (Q)</b><br>Simulating individual trajectories", 
        f"<b>Monte Carlo Convergence</b><br>Target: Black-Scholes Price (${bs_price:.2f})"
    ),
    horizontal_spacing=0.1
)

# Colors
c_path_cyan = "rgba(0, 229, 255, 0.2)" # Cyan transparent paths
c_bs_cyan = "#00E5FF"                  # Cyan Target Line (Solid/Bright)
c_est_gold = "#B400FF"                 # Purple Estimate Line

# --- Initial Traces ---

# Left Chart: Paths (Initialized empty)
# Trace 0
fig.add_trace(go.Scatter(x=[0], y=[S0], mode='lines', line=dict(color=c_path_cyan, width=1), showlegend=False), row=1, col=1)

# Right Chart: Target Line (BS Price)
# Trace 1
fig.add_trace(go.Scatter(
    x=[0, n_sims], y=[bs_price, bs_price], 
    mode='lines', line=dict(color=c_bs_cyan, width=2, dash='dash'), 
    showlegend=False
), row=1, col=2)

# Right Chart: Convergence Line (The Running Mean)
# Trace 2
fig.add_trace(go.Scatter(
    x=[0], y=[0], 
    mode='lines+markers', 
    marker=dict(size=4, color=c_est_gold),
    line=dict(color=c_est_gold, width=2), 
    showlegend=False
), row=1, col=2)

# --- 4. Animation Logic ---
frames = []
slider_steps = []
batch_size = 2 # Add 2 paths per frame for speed

# Persistent storage for the "accumulating" path lines
flat_x_paths = []
flat_y_paths = []

for n in range(1, n_sims, batch_size):
    # End index for this batch
    limit = min(n + batch_size, n_sims)
    frame_name = f"f{n}"
    
    # 1. Update Left Chart (Append new paths)
    for i in range(n, limit):
        flat_x_paths.extend(time_grid)
        flat_x_paths.append(None) # Break line
        flat_y_paths.extend(paths[:, i])
        flat_y_paths.append(None) # Break line
        
    # 2. Update Right Chart (Extend the convergence line)
    # X axis: 1 to limit
    x_conv = np.arange(1, limit + 1)
    # Y axis: running mean up to limit
    y_conv = running_mean[:limit]
    
    # Current Estimate for Annotation
    curr_est = y_conv[-1]

    # Create Frame
    frames.append(go.Frame(
        data=[
            # Update Trace 0 (Left Paths - Cyan)
            go.Scatter(x=flat_x_paths, y=flat_y_paths),
            # Update Trace 1 (BS Line - constant)
            go.Scatter(x=[0, n_sims], y=[bs_price, bs_price]), 
            # Update Trace 2 (Right Convergence Line - Gold)
            go.Scatter(x=x_conv, y=y_conv)
        ],
        traces=[0, 1, 2],
        layout=go.Layout(
            annotations=[
                # Annotation for Current Value on Right Chart
                dict(
                    x=limit, y=curr_est, xref="x2", yref="y2",
                    text=f"<b>${curr_est:.2f}</b>",
                    showarrow=True, arrowhead=1, ax=0, ay=-30,
                    font=dict(color=c_est_gold, size=12)
                )
            ]
        ),
        name=frame_name
    ))

    # Create Slider Step
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(limit),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600, width=1200,
    margin=dict(t=80, b=100, l=50, r=50),
    showlegend=False,
    
    # Controls: Play Button Only (No Pause)
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.2,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Run Monte Carlo", "method": "animate", "args": [None, {"frame": {"duration": 10, "redraw": True}, "fromcurrent": True}]},
        ]
    }],
    
    # Controls: Slider
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Simulations: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.20, "y": -0.2,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Axis Styling
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

# Left Chart Axes
fig.update_xaxes(title="Time (Years)", range=[0, T], row=1, col=1, **ax_opts)
fig.update_yaxes(title="Asset Price ($)", range=[60, 160], row=1, col=1, **ax_opts)

# Right Chart Axes
y_max_view = max(15, bs_price * 2.5)
fig.update_xaxes(title="Number of Simulations (N)", range=[0, n_sims], row=1, col=2, **ax_opts)
fig.update_yaxes(title="Avg Discounted Payoff ($)", range=[0, y_max_view], row=1, col=2, **ax_opts)

fig.show()
```

the simulation will of course suffer from two sources of error: discretization error and simulation error.

this is of course before Feynman kac, Cox and etc. that formalize the risk-neutral dynamics

But this replication argument is exactly the basis for the no arbitrage price given by FTAP and the fact that the BS framework provides the no arbitrage price gaurenteed by the FTAP via FK

###### ______________________________________________________________________________________________________________________________________

##### Implication: We Don't Care About Speculation ($\mathbb{P}$) We Care About Efficiency & Consistency ($\mathbb{Q}$)

In a complete market (Black-Scholes) we can hedge away all randomness and there is a unique risk-neutral measure (effectively only one distribution or parameterization to use for the evolution of the underlying under $\mathbb{Q}$ as we saw above)

In reality, many models don't operate in this capacity and there are *many* ways to select a parameterization for $\mathbb{Q}$

Ironic isn't it?  Is this not just the model parameterization problem we discussed above?

We don't just arbitrarily select a measure $\mathbb{Q}$ for our framework but rather calibrate our model to liquid instruments to *infer* $\mathbb{Q}$ from the market - this is *forward-looking* as we use a parameterization based on the market's outlook not historical data for calibration


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

# -----------------------------
# 1) Setup & Market Data
# -----------------------------
np.random.seed(42)

S_unique = np.array([80, 90, 100, 110, 120], dtype=float)
T_unique = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=float)
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

Z_market = np.array([
    [0.38, 0.35, 0.24, 0.22, 0.21],
    [0.36, 0.32, 0.23, 0.21, 0.20],
    [0.35, 0.30, 0.22, 0.20, 0.19],
    [0.34, 0.29, 0.21, 0.19, 0.18],
    [0.33, 0.28, 0.20, 0.18, 0.17]
], dtype=float)

Z_model_fit = gaussian_filter(Z_market, sigma=0.8)
Z_flat = np.full_like(Z_market, 0.45, dtype=float)

S_flat = S_mesh.flatten()
T_flat = T_mesh.flatten()
Z_market_flat = Z_market.flatten()

# Lock color scale domain across frames
zmin = float(min(Z_flat.min(), Z_model_fit.min(), Z_market.min()))
zmax = float(max(Z_flat.max(), Z_model_fit.max(), Z_market.max()))

# -----------------------------
# 2) Heston Simulation (price only)
# -----------------------------
n_paths = 15
n_steps = 60
T_max = 2.0
dt = T_max / n_steps

S0 = 100.0
v0 = 0.04

kappa = 2.0
theta = 0.04
xi = 0.3
rho = -0.7

mean = [0, 0]
cov = [[1, rho], [rho, 1]]
dw = np.random.multivariate_normal(mean, cov, (n_steps, n_paths)) * np.sqrt(dt)
dW_S = dw[:, :, 0]
dW_v = dw[:, :, 1]

S_paths = np.zeros((n_steps + 1, n_paths), dtype=float)
v_paths = np.zeros((n_steps + 1, n_paths), dtype=float)
S_paths[0, :] = S0
v_paths[0, :] = v0

for t in range(n_steps):
    v_curr = np.maximum(v_paths[t], 0.0)
    S_curr = S_paths[t]

    dv = kappa * (theta - v_curr) * dt + xi * np.sqrt(v_curr) * dW_v[t]
    v_paths[t + 1] = np.maximum(v_curr + dv, 0.0)

    dS = S_curr * np.sqrt(v_curr) * dW_S[t]
    S_paths[t + 1] = S_curr + dS

time_axis = np.linspace(0, T_max, n_steps + 1)

# -----------------------------
# 3) Figure setup
# -----------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'xy'}]],
    column_widths=[0.55, 0.45],
    subplot_titles=(
        "<b>Calibration:</b> Fitting Heston Surface",
        "<b>Simulation:</b> Price Process"
    )
)

# Capture subplot title annotations created by make_subplots
title_annos = list(fig.layout.annotations)

def get_param_text(progress: float) -> str:
    k = 0.5 + 1.5 * progress
    th = 0.10 - 0.06 * progress
    r = 0.0 - 0.7 * progress
    loss = 0.50 * (1 - progress) ** 2 + 0.002
    status = "Done" if progress >= 0.999999 else "Optimizing..."
    return (
        f"<b>Inference Status"
        f"Loss: {loss:.4f}<br>"
        f"----------------<br>"
        f"κ (Reversion): {k:.2f}<br>"
        f"θ (Long Var): {th:.2f}<br>"
        f"ξ (Vol of Vol): 0.30<br>"
        f"ρ (Corr): {r:.2f}"
    )

def param_anno(text: str):
    return dict(
        text=text,
        x=0, y=1, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        showarrow=False, align="left",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="#444", borderwidth=1
    )

# OPAQUE surface (fixes “missing slices” from WebGL transparency sorting)
def surface_trace(z_values):
    return go.Surface(
        x=S_mesh, y=T_mesh, z=z_values,
        colorscale="Viridis",
        cmin=zmin, cmax=zmax,
        opacity=1.0,          # critical to remove slice artifacts
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.15, roughness=0.9, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=300)
        # NOTE: no flatshading (not supported in your Plotly)
        # NOTE: no contours (they can look like cuts)
    )

market_pts = go.Scatter3d(
    x=S_flat, y=T_flat, z=Z_market_flat,
    mode="markers",
    marker=dict(size=4, color="red", symbol="x"),
    name="Market Quotes"
)

# -----------------------------
# 4) Frames
# -----------------------------
frames = []
n_calib_frames = 25
fixed_path_color = "cyan"

# Phase 1: Calibration morph
for i in range(n_calib_frames + 1):
    alpha = i / n_calib_frames
    Z_curr = (1 - alpha) * Z_flat + alpha * Z_model_fit

    frames.append(
        go.Frame(
            name=f"calib_{i}",
            data=[
                surface_trace(Z_curr),
                market_pts,
                *[go.Scatter(x=[0], y=[S0]) for _ in range(n_paths)]  # placeholders for right panel traces
            ],
            layout=go.Layout(
                annotations=[*title_annos, param_anno(get_param_text(alpha))]
            )
        )
    )

# Phase 2: Simulation (PRICE ONLY), same color, varying opacity
for k in range(1, n_steps + 1):
    path_traces = []
    for p in range(n_paths):
        opacity = 1.0 if p == 0 else 0.18
        width = 3 if p == 0 else 1.5

        path_traces.append(
            go.Scatter(
                x=time_axis[:k + 1],
                y=S_paths[:k + 1, p],
                mode="lines",
                line=dict(color=fixed_path_color, width=width),
                opacity=opacity,
                showlegend=False
            )
        )

    frames.append(
        go.Frame(
            name=f"sim_{k}",
            data=[
                surface_trace(Z_model_fit),
                market_pts,
                *path_traces
            ],
            layout=go.Layout(
                annotations=[*title_annos, param_anno(get_param_text(1.0))]
            )
        )
    )

fig.frames = frames

# -----------------------------
# 5) Initial state
# -----------------------------
fig.add_trace(surface_trace(Z_flat), row=1, col=1)
fig.add_trace(market_pts, row=1, col=1)

for p in range(n_paths):
    init_opacity = 1.0 if p == 0 else 0.18
    init_width = 3 if p == 0 else 1.5
    fig.add_trace(
        go.Scatter(
            x=[0], y=[S0],
            mode="lines",
            line=dict(color=fixed_path_color, width=init_width),
            opacity=init_opacity,
            showlegend=False
        ),
        row=1, col=2
    )

# -----------------------------
# 6) Layout & Styling
# -----------------------------
transparent = "rgba(0,0,0,0)"

fig.update_layout(
    template="plotly_dark",
    title="Heston Model: Calibration & Stochastic Simulation",
    height=600, width=1100,
    font=dict(family="Monospace", size=12),

    # Transparent background
    paper_bgcolor=transparent,
    plot_bgcolor=transparent,

    # Start with titles + param box
    annotations=[*title_annos, param_anno(get_param_text(0.0))],

    # 3D scene
    scene=dict(
        bgcolor=transparent,
        xaxis_title="Strike (K)",
        yaxis_title="Time (T)",
        zaxis_title="Imp Vol",
        xaxis=dict(range=[75, 125], gridcolor="gray", showbackground=False),
        yaxis=dict(range=[0, 2], gridcolor="gray", showbackground=False),
        zaxis=dict(range=[0.15, 0.50], gridcolor="gray", showbackground=False),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.6)
    ),

    # 2D price chart
    xaxis=dict(title="Time (Years)", range=[0, 2], gridcolor="#333"),
    yaxis=dict(title="Asset Price ($)", range=[60, 150], gridcolor="#333"),

    # Animation controls
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.5, "y": -0.15, "xanchor": "center",
        "buttons": [{
            "label": "▶ Calibrate & Simulate",
            "method": "animate",
            "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]
        }]
    }]
)

fig.show()

```

**Key Idea**
- Black-Scholes Replication Offers the Pricing Framework
- In Reality, the Market is Incomplete (sources of risk we can't perfectly hedge)
- We Can't Perfectly Cancel Out the Risk Premium Associated with That Unhedgable Risk
- Instead, We Infer $\mathbb{Q}$ Parameterization for our Models (capable of calibration) Based on the Market's Aggregate Risk Preference for that Unhedgable Risk


```python
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RectBivariateSpline

# --- 1. Input Market Data ---
S_points = np.array([80, 90, 100, 110, 120], dtype=float)
T_points = np.array([0.0833, 0.25, 0.50, 1.00, 2.00], dtype=float)

Z_points = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])

# --- 2. Smooth Interpolation ---
spline = RectBivariateSpline(T_points, S_points, Z_points, kx=3, ky=3)

T_fine = np.linspace(T_points.min(), T_points.max(), 60)
S_fine = np.linspace(S_points.min(), S_points.max(), 60)
T_mesh_fine, S_mesh_fine = np.meshgrid(T_fine, S_fine, indexing='ij')
Z_mesh_fine = spline(T_fine, S_fine)

# --- 3. Plotting ---
fig = go.Figure()

fig.add_trace(go.Surface(
    x=S_mesh_fine,
    y=T_mesh_fine,
    z=Z_mesh_fine,
    colorscale='Viridis',
    opacity=0.95,
    name='Interpolated Surface',
    colorbar=dict(title='Vol', x=0.9, thickness=15, len=0.7),
    
    # --- CHANGED HERE: Removed project_z=True ---
    # We keep show=True to see lines ON the surface, but remove the projection
    contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=False)
))

fig.update_layout(
    title="Implied Volatility Surface",
    template="plotly_dark",
    height=700,
    width=1000,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis_title='Strike (K)',
        yaxis_title='Time to Maturity (T)',
        zaxis_title='Implied Volatility (σ)',
        xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor="gray", showbackground=True),
        yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor="gray", showbackground=True),
        zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor="gray", showbackground=True),
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.7))
    ),
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()
```

This is exactly why volatility surfaces have a skew, "the math says it should be $r$, but I can't hedge that jump risk so I'm charging you more here on the downside" - this is directly relevant to the 1987 crash and the necessity for models capable of fitting a skew

Which is why, ironically again, connecting the idea from the start of this video, this may turn into speculation to a degree again as there isn't one way to derive a price and construct a spread etc...quoted prices can be rejected by the client for being too expensive before hedging

For more on this subject and the connection to market-making see

*On YouTube*
- [Top 5 Papers that Built Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)
- [Quant Explains Algorithmic Market-Making]((https://youtu.be/aVzFKwyzwM0))

*On Quant Guild*
- [Quant Stats](https://quantguild.com/syllabus/quant-stats) Course on Quant Guild
- [Financial Mathematics](https://quantguild.com/syllabus/financial-mathematics) Masterclass on Quant Guild

---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Anytime we create a model we are to suffer from the non-stationarity of reality: model mispecification and misparameterization
- We still need to develop a price for derivative contracts, some may think the price will go to the moon, some may think it'll sell for pennies on the dollar, in any case, we need a consistent framework for pricing that mitigates (not entirely, but in a controlled way) this raw speculation and capacity for mispecification and misparameterization
- Black-Scholes produce a portfolio replication arugment showing in a complete market with randomness that is hedgable the price of maintaining a continuous hedge should be the price of the option
- Clearly, there is no risk, so the portfolio by a no arbitrage assumption must earn the risk-free rate, this is why when we simulate underlying dynamics to derive a price we must include a drift term equivalent to the risk-free rate otherwise we are underpricing the option relative to this hedging framework (e.g. hedging removes the risk premium compensation for bearing the risk but we still earn the risk-free rate)
- This is the notion of a risk-neutral measure, we transform the real world measure $\mathbb{P} \rightarrow \mathbb{Q}$, this is just a change in the underlying probabilities producing the expectation to ensure we are earning the risk-free rate in expectation (there are connections here to martingales and of course the Girsnav change of measure, would love to cover this in the future)
- In complete markets, there is only risk-neutral measure, so its relatively easy to produce a price
- In reality, markets are not compelete, there is not just one risk-neutral measure but infinite to choose from
- Instead of wandering around infinity we *infer* the risk-neutral measure and the premium from unhedgable risk by calibrating models to the market liquid instruments, this is what produces skew and the implied volatility surface
- Ironically, we are back to the mispecification and misparameterization problem only now in the risk-neutral world, we can quote prices too high for our clients, its a framework, not a solution, but the math powers trillions of dollars of transactions in the financial markets
- For more on this topic see my videos on algorithmic market-making, the papers that built modern quant finance, and for formal study of these topics in a classroom setting with connections to the real world and trading desks see the Quant Stats Course and Financial Mathematics Masterclass on Quant Guild


**Future Topics**

Technical Videos and Other Discussions

- Hawkes Processes
- Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
- Market-Making Models and Simulation (Stoikov-Avellaneda)
- Projects that Made me a Quant
- My First Year as a Quant
- Kalman Filter for Quant Finance
- Why Hedge Funds are Actually Secretive
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Top 3 Uses of Linear Algebra for Quant Finance
- Girsanov's Change of Measure
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- How to Backtest a Trading Strategy with Interactive Brokers
- Algorithmic Volatility Trading System

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
