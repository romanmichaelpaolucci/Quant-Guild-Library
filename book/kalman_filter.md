### 📈 Kalman Filter for Quant Finance

##### ▶️ Related Quant Guild Videos:

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)

- [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)

- [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/wZChBKDFFeU)

- [Quant vs. Discretionary Trading](https://youtu.be/3gblERSSHXI)

- [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)

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



### 📖 Sections

#### 1.) 📊 Model Specification and Parameterization

- Model Specification

- Model Parameterization

- When Models Break

#### 2.) 📈 Kalman Filter

- Definition, Intuition, Application to the VIX

- Step-by-Step Example: Measuring the VIX 

#### 3.) 🌪️ Dual Filter Approaches and Real-World Example

- Illiquid Bond Pricing (BVAL \<GO\>)

- Dual Approach to Regime Dynamics

- Extensions to Trending/Reversion, High/Low Vol

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📊 Model Specification and Parameterization

##### Model Specification

If we are trying to make decisions in the face of uncertainty we need to build a model, otherwise, without a crystal ball, we are just guessing.

Typically, rooted in academic or economic theory, our first objective is to *specify a model* $\mathcal{M}$.

Most models we use in practice are *parametric* meaning they have a parameter(s) $\theta$ that we can choose.

Here are some examples of Models and their Parameterizations ($\mathcal{M}(\theta)$):
 - Linear Regression: $y = \beta_0 + \beta_1 x + \epsilon$, $\theta = (\beta_0, \beta_1)$
 - AR(1): $x_t = \phi x_{t-1} + \epsilon_t$, $\theta = \phi$
 - GARCH(1,1): $\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$, $\theta = (\alpha_0, \alpha_1, \beta_1)$
 - Ornstein-Uhlenbeck: $dx_t = \theta_1 (\theta_2 - x_t)dt + \theta_3 dW_t$, $\theta = (\theta_1, \theta_2, \theta_3)$


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Data Generation ---
np.random.seed(42)
n_points = 60
# Generate 2D X data (not a time series)
X = np.linspace(-5, 5, n_points)

# Add some noise to make it realistic
noise_lin = np.random.normal(0, 3, n_points)
noise_quad = np.random.normal(0, 4, n_points)

# True Data Generating Processes
# Left: Linear
Y_lin = 4 * X + 15 + noise_lin
# Right: Quadratic (Non-linear)
Y_quad = 1.5 * X**2 - 2 * X + 10 + noise_quad

# --- Precompute Models ---
# We will animate the addition of points. We need at least 2 points to fit a line.
min_points = 2
lin_fits = []
quad_fits = []

# X coordinates for drawing the regression line across the whole plot
x_grid = np.array([-6, 6])

for t in range(min_points, n_points + 1):
    x_curr = X[:t]
    y_lin_curr = Y_lin[:t]
    y_quad_curr = Y_quad[:t]
    
    # Model 1: Linear fit on Linear Data
    slope_lin, int_lin = np.polyfit(x_curr, y_lin_curr, 1)
    line_lin = int_lin + slope_lin * x_grid
    lin_fits.append(line_lin)
    
    # Model 2: Linear fit on Quadratic Data (Modeling Issue)
    slope_quad, int_quad = np.polyfit(x_curr, y_quad_curr, 1)
    line_quad = int_quad + slope_quad * x_grid
    quad_fits.append(line_quad)

# --- Graphics Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Appropriate Model Selection</b><br>Linear fit captures the dynamics", 
        "<b>Modeling Issue (Underfitting)</b><br>Linear model cannot capture quadratic dynamics"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_lin = "#00FFCC"      # Teal
c_bad = "#FF3366"      # Pinkish red
c_pts = "rgba(255, 255, 255, 0.8)"

# Dynamic Traces: Scatter Points up to t=2 (Initial State)
fig.add_trace(go.Scatter(x=X[:min_points], y=Y_lin[:min_points], mode='markers', 
                         marker=dict(color='white', size=8, line=dict(color='black', width=1)), 
                         name='Linear Data'), row=1, col=1) # Trace 0

fig.add_trace(go.Scatter(x=X[:min_points], y=Y_quad[:min_points], mode='markers', 
                         marker=dict(color='white', size=8, line=dict(color='black', width=1)), 
                         name='Quadratic Data'), row=1, col=2) # Trace 1

# Dynamic Traces: Regression Lines (Initial State)
fig.add_trace(go.Scatter(x=x_grid, y=lin_fits[0], mode='lines', 
                         line=dict(color=c_lin, width=4), name='Valid Fit'), row=1, col=1) # Trace 2

fig.add_trace(go.Scatter(x=x_grid, y=quad_fits[0], mode='lines', 
                         line=dict(color=c_bad, width=4), name='Misspecified Fit'), row=1, col=2) # Trace 3


# --- Animation Logic ---
frames = []
slider_steps = []

for i, t in enumerate(range(min_points, n_points + 1)):
    frame_name = f"f{t}"
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=X[:t], y=Y_lin[:t]),      # 0: Update scatter left
            go.Scatter(x=X[:t], y=Y_quad[:t]),     # 1: Update scatter right
            go.Scatter(x=x_grid, y=lin_fits[i]),   # 2: Update line left
            go.Scatter(x=x_grid, y=quad_fits[i])   # 3: Update line right
        ],
        traces=[0, 1, 2, 3],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": f"n={t}",
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600, width=1100,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.20,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Data Points: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.20,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Fix axis ranges so the plot is stable during the animation
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_xaxes(range=[-6, 6], title="Feature (X)", **ax_opts)

# Set Y-axis ranges based on the generated data
fig.update_yaxes(range=[min(Y_lin)-10, max(Y_lin)+10], title="Target (Y)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[min(Y_quad)-10, max(Y_quad)+10], title="Target (Y)", row=1, col=2, **ax_opts)

fig.show()
```



The efficacy of our model specification is based on how robust it is to new data.  Better models are likely to produce lower forecasting errors.

We can't necessarily *see* data in higher dimensions which demands a theoretical and academic basis for explaining the world around us.

###### ______________________________________________________________________________________________________________________________________

##### Example: Model Specification for the VIX

Model specification isn't that difficult, let's take the VIX for example..


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Load or Generate Data ---
# (Comment this block out and use your real vix.csv)
np.random.seed(42)
n_days = 150
dates = pd.date_range('2024-01-01', periods=n_days)
mock_vix = np.zeros(n_days)
mock_vix[0] = 13.0
for i in range(1, n_days):
    mock_vix[i] = mock_vix[i-1] + 0.08 * (15.0 - mock_vix[i-1]) + np.random.normal(0, 1.5)
    if i == 60: mock_vix[i] += 25.0 # Simulate massive Volatility Spike
pd.DataFrame({'Date': dates, 'SPY': 500, 'VIX': mock_vix}).to_csv('vix.csv', index=False)
# ---------------------------------

df = pd.read_csv('vix.csv')
dates = pd.to_datetime(df['Date'])
vix = df['VIX'].values
n_steps = len(vix)

# --- 2. Rolling Model Setup ---
W = 20  # Lookback window to fit the model parameters
H = 60  # EXTREME Forecast horizon (days ahead) to show terminal behavior

lin_forecasts, lin_stdevs = [], []
mr_forecasts, mr_stdevs = [], []

for t in range(n_steps):
    if t < W:
        # Not enough data to fit, flatline
        lin_forecasts.append(np.ones(H) * vix[t])
        lin_stdevs.append(np.ones(H))
        mr_forecasts.append(np.ones(H) * vix[t])
        mr_stdevs.append(np.ones(H))
        continue
        
    window_vix = vix[t-W:t]
    
    # Model A: Arbitrary Linear Trend (Naive)
    x = np.arange(W)
    slope, intercept = np.polyfit(x, window_vix, 1)
    std_lin = np.std(window_vix - (intercept + slope * x))
    
    # Linear projection straight into the abyss
    f_lin = intercept + slope * (W + np.arange(H))
    s_lin = std_lin * np.sqrt(np.arange(1, H+1)) 
    
    lin_forecasts.append(f_lin)
    lin_stdevs.append(s_lin)
    
    # Model B: Economic Mean Reversion (AR1 / Discrete Ornstein-Uhlenbeck)
    dV = np.diff(window_vix)
    V_lag = window_vix[:-1]
    
    if len(np.unique(V_lag)) > 1:
        b, a = np.polyfit(V_lag, dV, 1)
    else:
        b, a = 0, 0
        
    std_mr = np.std(dV - (a + b * V_lag)) if len(dV) > 0 else 1.0
    
    # Bound parameters to ensure mean reversion mathematically
    if b >= 0:
        b = -0.05
        a = 0.05 * np.mean(window_vix)
        
    f_mr = np.zeros(H)
    s_mr = np.zeros(H)
    curr_v = vix[t-1]
    curr_var = 0
    
    # Project forward curving to terminal theory
    for h in range(H):
        curr_v = a + (1+b) * curr_v
        f_mr[h] = curr_v
        curr_var += std_mr**2 * (1+b)**(2*h)
        s_mr[h] = np.sqrt(curr_var)
        
    mr_forecasts.append(f_mr)
    mr_stdevs.append(s_mr)

# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Linear Model Forecast</b><br>Shoots to Infinity/Negative", 
        "<b>Mean Reversion Forecast</b><br>Roots back to Terminal Theory",
        "Terminal Forecast Distribution (Linear, t+H)", 
        "Terminal Forecast Distribution (Mean Reversion, t+H)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_lin = "#FF3366"      # Pink/Red
c_mr = "#00FFCC"       # Teal
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True VIX
fig.add_trace(go.Scatter(x=dates, y=vix, mode='lines', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=vix, mode='lines', line=dict(color='gray', width=1, dash='dash')), row=1, col=2)

# Dynamic Traces: Historical Line up to t
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color='white', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color='white', width=2)), row=1, col=2)

# Dynamic Traces: Forecast Paths
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color=c_lin, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color=c_mr, width=3)), row=1, col=2)

# Dynamic Traces: Densities at t+H
x_grid = np.linspace(-40, 120, 500) 
y_zero = np.zeros_like(x_grid)

fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_lin)), row=2, col=1)
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_mr)), row=2, col=2)

# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(W, n_steps - 10, 2) 

# Create the future coordinates for the forecast arrays to map to
extended_dates = pd.date_range(dates[0], periods=n_days + H)

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    curr_dates = dates[:t]
    curr_vix = vix[:t]
    forecast_dates = extended_dates[t:t+H]
    
    # Generate Probability Density Functions (PDFs) based on the absolute terminal point
    pdf_lin = norm.pdf(x_grid, lin_forecasts[t][-1], lin_stdevs[t][-1] + 1e-6)
    pdf_mr = norm.pdf(x_grid, mr_forecasts[t][-1], mr_stdevs[t][-1] + 1e-6)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_vix),                 # Historic Line L
            go.Scatter(x=curr_dates, y=curr_vix),                 # Historic Line R
            go.Scatter(x=forecast_dates, y=lin_forecasts[t]),     # Forecast Path L
            go.Scatter(x=forecast_dates, y=mr_forecasts[t]),      # Forecast Path R
            go.Scatter(y=pdf_lin),                                # PDF L
            go.Scatter(y=pdf_mr),                                 # PDF R
        ],
        traces=[2, 3, 4, 5, 6, 7],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": dates[t].strftime('%Y-%m-%d'),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.15, "pad": {"r": 10, "t": 30},
        "bgcolor": "#333", "font": {"color": "white"},
        "bordercolor": "#555", "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Date: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps, "bgcolor": "#333", "font": {"color": "white"}
    }]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

vix_range = [-10, 100] 

fig.update_yaxes(range=vix_range, title="VIX Index", row=1, col=1, **ax_opts)
fig.update_yaxes(range=vix_range, title="VIX Index", row=1, col=2, **ax_opts)

# STRICTLY LOCK X-AXIS to the end of the time series dates
fig.update_xaxes(range=[dates.iloc[0], dates.iloc[-1]], row=1, col=1, **ax_opts)
fig.update_xaxes(range=[dates.iloc[0], dates.iloc[-1]], row=1, col=2, **ax_opts)

# Density Y axes (shrunk scale slightly so flattened curves are visible)
fig.update_yaxes(range=[0, 0.15], title="Probability Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.15], title="Probability Density", row=2, col=2, **ax_opts)

# Broaden density X axes to fit linear extremes
fig.update_xaxes(range=[-40, 120], title="Terminal Expected VIX at t+H", row=2, col=1, **ax_opts)
fig.update_xaxes(range=[-40, 120], title="Terminal Expected VIX at t+H", row=2, col=2, **ax_opts)

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Model Parameterization

How do we choose $\theta$?  What is *"correct"*?  Welcome to the world of mathematical modeling.


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
N_paths = 100
n_steps = 100
T = 5.0  # 5 years
dt = T / n_steps
t_array = np.linspace(0, T, n_steps)

# Common Mean
theta = 20.0 
threshold = 60.0

# Model A (Left): Tight "Normal Market" Parameters
kappa_A = 1.0
sigma_A = 8.0 

# Model B (Right): Fat-Tailed "Crisis-Aware" Parameters
kappa_B = 1.0
sigma_B = 25.0

# Simulate Paths (Euler-Maruyama method for OU process)
paths_A = np.zeros((N_paths, n_steps))
paths_B = np.zeros((N_paths, n_steps))
paths_A[:, 0] = theta
paths_B[:, 0] = theta

for i in range(1, n_steps):
    dW_A = np.random.randn(N_paths) * np.sqrt(dt)
    dW_B = np.random.randn(N_paths) * np.sqrt(dt)
    paths_A[:, i] = paths_A[:, i-1] + kappa_A * (theta - paths_A[:, i-1]) * dt + sigma_A * dW_A
    paths_B[:, i] = paths_B[:, i-1] + kappa_B * (theta - paths_B[:, i-1]) * dt + sigma_B * dW_B

# Helper function to bundle paths for efficient Plotly animation
def get_bundled_paths(paths, t_arr, t_idx):
    curr_t = t_arr[:t_idx+1]
    curr_p = paths[:, :t_idx+1]
    N = paths.shape[0]
    
    t_tiled = np.tile(curr_t, (N, 1))
    nan_col = np.full((N, 1), np.nan)
    
    t_bundled = np.hstack([t_tiled, nan_col]).flatten()
    p_bundled = np.hstack([curr_p, nan_col]).flatten()
    return t_bundled, p_bundled

# Helper to format probability AND Expected Time (Geometric RV)
def format_stats(p, step_size):
    # If probability is mathematically effectively zero
    if p < 1e-16:
        return "Area: ~0%<br>E[VIX > 60]: &infin; yrs"
    
    # E[X] for geometric distribution = 1/p trials. Convert to years.
    expected_years = step_size / p
    
    # Format Area
    if p < 1e-4:
        area_str = f"Area: {p:.2e}"
    else:
        area_str = f"Area: {p*100:.2f}%"
        
    # Format Expected Time
    if expected_years > 1e6:
        time_str = f"E[VIX > 60]: {expected_years:.1e} yrs"
    else:
        time_str = f"E[VIX > 60]: {expected_years:.1f} yrs"
        
    return f"{area_str}<br>{time_str}"

# --- 2. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Model A: Incorrect Parameters</b><br>Path Realizations", 
        "<b>Model B: Correct Parameters</b><br>Path Realizations",
        "Model A: 60 is a <b>7.1-Sigma</b> Event", 
        "Model B: 60 is a <b>2.2-Sigma</b> Event"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_A = "#00D2FF"      
c_B = "#FF5500"      
c_path = "rgba(255, 255, 255, 0.15)"
c_alert = "#FF0044"  

# 0 & 1: Static Thresholds on Top
fig.add_trace(go.Scatter(x=[0, T], y=[threshold, threshold], mode='lines', line=dict(color=c_alert, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=[0, T], y=[threshold, threshold], mode='lines', line=dict(color=c_alert, dash='dash')), row=1, col=2)

# 2 & 3: Animated Paths (Start Empty)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=c_path, width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=c_path, width=1)), row=1, col=2)

# Bottom Left (Model A PDF, Tail, Threshold Line, Text)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', line=dict(color=c_A)), row=2, col=1)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', fillcolor='rgba(255,0,68,0.5)', line=dict(width=0)), row=2, col=1)
fig.add_trace(go.Scatter(x=[threshold, threshold], opacity=0.3, y=[0, 0.15], mode='lines', line=dict(color=c_alert, dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[40], y=[0.06], mode='text', text=[""], textfont=dict(color='white', size=14), textposition='middle right'), row=2, col=1)

# Bottom Right (Model B PDF, Tail, Threshold Line, Text)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', line=dict(color=c_B)), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', fillcolor='rgba(255,0,68,0.5)', line=dict(width=0)), row=2, col=2)
fig.add_trace(go.Scatter(x=[threshold, threshold], opacity=0.3, y=[0, 0.15], mode='lines', line=dict(color=c_alert, dash='dash')), row=2, col=2)
fig.add_trace(go.Scatter(x=[40], y=[0.06], mode='text', text=[""], textfont=dict(color='white', size=14), textposition='middle right'), row=2, col=2)


# --- 3. Animation Logic ---
frames = []
slider_steps = []
x_grid = np.linspace(-20, 100, 400)
steps_to_anim = range(2, n_steps, 2) # Start at step 2 to avoid 0 variance errors

for t_idx in steps_to_anim:
    frame_name = f"f{t_idx}"
    current_time = t_array[t_idx]
    
    # Calculate exact theoretical variance at time t
    var_A = (sigma_A**2 / (2 * kappa_A)) * (1 - np.exp(-2 * kappa_A * current_time))
    var_B = (sigma_B**2 / (2 * kappa_B)) * (1 - np.exp(-2 * kappa_B * current_time))
    std_A = np.sqrt(var_A)
    std_B = np.sqrt(var_B)
    
    # Integrate Tail CDFs
    prob_A = 1 - norm.cdf(threshold, theta, std_A)
    prob_B = 1 - norm.cdf(threshold, theta, std_B)
    
    # Generate PDFs and shaded tails
    pdf_A = norm.pdf(x_grid, theta, std_A)
    pdf_B = norm.pdf(x_grid, theta, std_B)
    
    tail_A_y = np.where(x_grid >= threshold, pdf_A, 0)
    tail_B_y = np.where(x_grid >= threshold, pdf_B, 0)
    
    # Bundle paths for current frame
    t_A, p_A = get_bundled_paths(paths_A, t_array, t_idx)
    t_B, p_B = get_bundled_paths(paths_B, t_array, t_idx)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=t_A, y=p_A),                             
            go.Scatter(x=t_B, y=p_B),                             
            go.Scatter(x=x_grid, y=pdf_A),                        
            go.Scatter(x=x_grid, y=tail_A_y),                     
            go.Scatter(text=[format_stats(prob_A, dt)]),          # Live Stats A          
            go.Scatter(x=x_grid, y=pdf_B),                        
            go.Scatter(x=x_grid, y=tail_B_y),                     
            go.Scatter(text=[format_stats(prob_B, dt)]),          # Live Stats B
        ],
        traces=[2, 3, 4, 5, 7, 8, 9, 11],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        "label": f"{current_time:.1f}y", "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000, margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.15, "pad": {"r": 10, "t": 30},
        "bgcolor": "#333", "font": {"color": "white"}, "bordercolor": "#555", "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Time: ", "visible": True},
        "pad": {"b": 10, "t": 50}, "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps, "bgcolor": "#333", "font": {"color": "white"}
    }]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_yaxes(range=[-20, 100], title="Index Value", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[-20, 100], title="Index Value", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.12], title="Probability Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.12], title="Probability Density", row=2, col=2, **ax_opts)
fig.update_xaxes(range=[-20, 100], title="Forecasted Distribution", row=2, col=1, **ax_opts)
fig.update_xaxes(range=[-20, 100], title="Forecasted Distribution", row=2, col=2, **ax_opts)

fig.show()
```



Some methods of choosing $\theta$ work well in the classroom but poorly in reality,
- Method of Moments
- Maximum Likelihood Estimation
- . . . 

Most have to do with estimatations from data.  

I have videos on my channel dedicated to parameter estimation but if there is interest let me know in the comments and I'd be happy to do a comparison video on the topic.

###### ______________________________________________________________________________________________________________________________________

##### Impact of Model Mispecification

- *Incorrect Parameters:* 65 billion years for an event to occur, wildly incorrect probabilities for that state of the world occuring
- *Correct Parameters:* 4.2 years for an event to occur, still incorrect, but allows us to make more informed decisions

In other words, both parameterizations are wrong, like our models are wrong

However, one choice tells us we have to be weary of a state of the world while another says it effectively can't happen altogether

###### ______________________________________________________________________________________________________________________________________

##### When Models Break

**The key question:**  *is it model specification or parameterization*

The answer is probably *both* but we can live with our specified model being *wrong* if the dynamics it captures are still reasonable.

- It may just need an entirely new parameterization $\theta$

- It may be that we need an entirely new model $\mathcal{M}$

- It may be the case that we need both $\mathcal{M}(\theta)$


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Data Generation (Jump Diffusion) ---
np.random.seed(42)
n_steps = 150
H = 15  # Forecast horizon
W = 20  # Lookback window for the rolling model

# Generate trading days
dates = pd.date_range('2024-01-01', periods=n_steps + H)

# Simulate Jump Diffusion
y = np.zeros(n_steps + H)
y[0] = 100.0
mu = 0.4          # Drift
sigma = 1.5       # Diffusion volatility
jump_idx = 80     # The day the regime changes
jump_size = -45.0 # Massive gap down

for i in range(1, n_steps + H):
    dy = mu + np.random.normal(0, sigma)
    if i == jump_idx:
        dy += jump_size
    y[i] = y[i-1] + dy

# --- Precompute Models & Text Annotations ---
exp_fits_y, exp_fits_x, exp_text = [], [], []
roll_fits_y, roll_fits_x, roll_text = [], [], []

# We start animating from step W
for t in range(W, n_steps + 1):
    # --- Model 1: Expanding Window (Left) ---
    x_exp = np.arange(t)
    y_exp = y[:t]
    slope_exp, int_exp = np.polyfit(x_exp, y_exp, 1)
    
    x_exp_pred = np.arange(0, t + H)
    y_exp_pred = int_exp + slope_exp * x_exp_pred
    
    exp_fits_x.append(dates[0:t+H])
    exp_fits_y.append(y_exp_pred)
    exp_text.append(f"<b>Slope:</b> {slope_exp:.2f}<br><b>Intercept:</b> {int_exp:.2f}")
    
    # --- Model 2: Rolling Window (Right) ---
    x_roll = np.arange(t-W, t)
    y_roll = y[t-W:t]
    slope_roll, int_roll = np.polyfit(x_roll, y_roll, 1)
    
    x_roll_pred = np.arange(t-W, t + H)
    y_roll_pred = int_roll + slope_roll * x_roll_pred
    
    roll_fits_x.append(dates[t-W:t+H])
    roll_fits_y.append(y_roll_pred)
    roll_text.append(f"<b>Slope:</b> {slope_roll:.2f}<br><b>Intercept:</b> {int_roll:.2f}")

# --- Graphics Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Expanding Window Forecast</b><br>Anchored to old regime data", 
        "<b>Rolling Window Forecast</b><br>Scraps old data, adapts to jump"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_bad = "#FF3366"      # Pinkish red (Failing model)
c_good = "#00FFCC"     # Teal (Adapting model)
c_ts = "rgba(255, 255, 255, 0.9)" # White for actual time series

# Pre-calculate Y limits so we can anchor the text perfectly in the upper right
y_min, y_max = min(y) - 10, max(y) + 20
text_x_pos = dates[-5] # Anchor text near the right edge
text_y_pos = y_max - 5 # Anchor text near the top edge

# Dynamic Traces: Actual Time Series
fig.add_trace(go.Scatter(x=dates[:W], y=y[:W], mode='lines', 
                         line=dict(color=c_ts, width=2), name='Actual Data'), row=1, col=1) # Trace 0
fig.add_trace(go.Scatter(x=dates[:W], y=y[:W], mode='lines', 
                         line=dict(color=c_ts, width=2), name='Actual Data'), row=1, col=2) # Trace 1

# Dynamic Traces: Regression Forecasts
fig.add_trace(go.Scatter(x=exp_fits_x[0], y=exp_fits_y[0], mode='lines', 
                         line=dict(color=c_bad, width=3, dash='dot'), name='Expanding Fit'), row=1, col=1) # Trace 2
fig.add_trace(go.Scatter(x=roll_fits_x[0], y=roll_fits_y[0], mode='lines', 
                         line=dict(color=c_good, width=3, dash='dot'), name='Rolling Fit'), row=1, col=2) # Trace 3

# Dynamic Traces: Text Annotations (Pinned to fixed coordinates)
fig.add_trace(go.Scatter(x=[text_x_pos], y=[text_y_pos], mode='text', text=[exp_text[0]], 
                         textposition="bottom left", textfont=dict(color=c_bad, size=14, family="monospace")), 
              row=1, col=1) # Trace 4
fig.add_trace(go.Scatter(x=[text_x_pos], y=[text_y_pos], mode='text', text=[roll_text[0]], 
                         textposition="bottom left", textfont=dict(color=c_good, size=14, family="monospace")), 
              row=1, col=2) # Trace 5

# --- Animation Logic ---
frames = []
slider_steps = []

for i, t in enumerate(range(W, n_steps + 1)):
    frame_name = f"f{t}"
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=dates[:t], y=y[:t]),               # 0: Update TS left
            go.Scatter(x=dates[:t], y=y[:t]),               # 1: Update TS right
            go.Scatter(x=exp_fits_x[i], y=exp_fits_y[i]),   # 2: Update fit left
            go.Scatter(x=roll_fits_x[i], y=roll_fits_y[i]), # 3: Update fit right
            go.Scatter(text=[exp_text[i]]),                 # 4: Update text left
            go.Scatter(text=[roll_text[i]])                 # 5: Update text right
        ],
        traces=[0, 1, 2, 3, 4, 5],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": dates[t].strftime('%Y-%m-%d'),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600, width=1200,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.20,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Date: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.20,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Fix axis ranges to keep the view stable during animation
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_xaxes(range=[dates[0], dates[-1]], title="Date", **ax_opts)
fig.update_yaxes(range=[y_min, y_max], title="Asset Price", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[y_min, y_max], title="Asset Price", row=1, col=2, **ax_opts)

fig.show()
```



Of course, we don't usually have a nice clean cut for a regime change like this.

In any case, the model fails to capture reversion dynmics and suggests price will go negative very steeply.

The parameterization methodology however we can also see creates tremendous impact

If we don't throw out old data we'll be grossly over or underestimating the likelihood of different states of the world.

---

#### 2.) 📈 Kalman Filter

We see that there is an issue when modeling the real world, especially in the context of model parameterization and regime changes

The Kalman Filter is a technique that uses both underlying dynamics (*a specified and parameterized model*) and observations as a filter

Let's discuss this in the context of the VIX above.  

We've established mean reverting dynamics already, but we need to be capable of adapting to changes in parameterization.

The Kalman Filter Relies on the *"Kalman Gain"*

$$K = \frac{\text{Uncertainty in Prediction}}{\text{Uncertainty in Prediction} + \text{Uncertainty in Measurement (R)}}$$

**Note:** Just like any other model, all of the selections are up to us.  We can estimate every of the following in a variety of ways.

**Kalman Filter**
$$x_t = F x_{t-1} + Bu_t + w_t$$

**Hidden State**
- $x_k$ the *true* spot VIX, we have noisy measurements and our goal is to combine our model with data to filter out the noise

**Model Specification Impact**
- $F = e^{-\kappa \Delta t}, \kappa > 0$ the decay factor for speed of mean reversion
- $B = (1-e^{- \kappa \Delta t}), u_t=\theta$ is the long-term mean
- $Q = \frac{\sigma^2}{2 \kappa}(1 - e^{-2 \kappa \Delta t})$ is the process noise variance $w_t \sim N(0, Q)$
- $R$ accounts for "measurement error" of the *true latent VIX process*
    - Think of this as a level (Large R, filter trusts the model more, Small R, filter trusts data more)
    - Larger R $\rightarrow$ Smaller Gain (K), Smaller R $\rightarrow$ Larger Gain (K)

Supposing $y_t$ is the *true* VIX level at time $t$

$\implies y_t = x_t + v_t, v_t \sim N(0, R)$


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
n_steps = 100
T = 1.0
dates = np.linspace(0, T, n_steps)

# True Price: Drifts around 100 consistently (No Regime Change)
true_price = np.ones(n_steps) * 100
true_price += np.cumsum(np.random.normal(0, 0.3, n_steps)) 

# Noisy Quotes (Illiquid Market)
R_true = 10.0 # Lowered slightly so outliers stand out more
quotes = true_price + np.random.normal(0, np.sqrt(R_true), n_steps)

# INJECT MASSIVE OUTLIERS (e.g., flash crash or bad prints)
quotes[50] -= 30
quotes[51] -= 25

# Create a "Market Center" array for the density plot to follow the outliers
market_center = true_price.copy()
market_center[50] = quotes[50]
market_center[51] = quotes[51]

# --- 2. Kalman Filter Implementations ---
# Shared base parameters
R_assumed = 40.0  # High assumed sensor noise
Q_assumed = 0.1   # Low assumed process noise (trusts model)

def run_kalman_filter(quotes, R, Q, inject_uncertainty_at=None):
    x_est = []
    P_est = []
    
    # Initial state
    x = 100.0
    P = 1.0
    
    for i, z in enumerate(quotes):
        # 1. Predict (Prior)
        x_pred = x
        P_pred = P + Q
        
        # Inject massive uncertainty at specified steps (e.g., when it gets "fooled" by large errors)
        if inject_uncertainty_at and i in inject_uncertainty_at:
            P_pred += 150.0 # P is large!
            
        # 2. Update (Posterior)
        K = P_pred / (P_pred + R) # Kalman Gain
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        
        x_est.append(x)
        P_est.append(P)
        
    return np.array(x_est), np.array(P_est)

# Scenario A: Strict Model (ignores outliers, holds strong)
x_low, P_low = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed)

# Scenario B: Dynamic adjustment gets FOOLED by outliers
# It panics at t=50 (bad drop) and has to panic again at t=52 (recovery)
x_high, P_high = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, inject_uncertainty_at=[50, 52])


# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Strict Model (Low K)</b><br>Ignores Bad Prints (Optimal)", 
        "<b>Adaptive Filter (Dynamic K)</b><br>Overreacts to Outliers (Suboptimal)",
        "Belief vs Market (Strict)", 
        "Belief vs Market (Adaptive)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_low = "#00D2FF"       # Cyan for Strict
c_low_ghost = "rgba(0, 210, 255, 0.15)"
c_high = "#FF5500"      # Orange for Adaptive
c_high_ghost = "rgba(255, 85, 0, 0.15)"
c_quote_dist = "rgba(200, 200, 200, 0.2)" # Grey for True Market Noise
c_quote = "rgba(255, 255, 255, 0.6)"
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True Price [Indices 0, 1]
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=2)

# Dynamic Traces Initialization: Quotes (2, 3) and KF Lines (4, 5)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=2)

fig.add_trace(go.Scatter(x=[dates[0]], y=[x_low[0]], mode='lines', line=dict(color=c_low, width=3), name='KF Strict'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[x_high[0]], mode='lines', line=dict(color=c_high, width=3), name='KF Adaptive'), row=1, col=2)

# Dynamic Traces Initialization: Density Plots [Indices 6 through 11]
x_grid = np.linspace(60, 115, 300)
y_zero = np.zeros_like(x_grid)

# Left Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=1) # 6
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_high_ghost, name='Ghost Adaptive'), row=2, col=1) # 7
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_low), name='Main Strict'), row=2, col=1) # 8

# Right Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=2) # 9
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_low_ghost, name='Ghost Strict'), row=2, col=2) # 10
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_high), name='Main Adaptive'), row=2, col=2) # 11


# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps, 2) # Skip every other frame for performance

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # Slice current data
    curr_dates = dates[:t+1]
    curr_quotes = quotes[:t+1]
    curr_x_low = x_low[:t+1]
    curr_x_high = x_high[:t+1]
    
    # Calculate Current PDFs (Market center follows the outliers briefly to show the bad signal)
    pdf_market = norm.pdf(x_grid, market_center[t], np.sqrt(R_true))
    pdf_low = norm.pdf(x_grid, x_low[t], np.sqrt(P_low[t]))
    pdf_high = norm.pdf(x_grid, x_high[t], np.sqrt(P_high[t]))

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_quotes),  # 2
            go.Scatter(x=curr_dates, y=curr_quotes),  # 3
            go.Scatter(x=curr_dates, y=curr_x_low),   # 4
            go.Scatter(x=curr_dates, y=curr_x_high),  # 5
            go.Scatter(y=pdf_market),                 # 6 (Market Noise Left)
            go.Scatter(y=pdf_high),                   # 7 (Ghost High on Left)
            go.Scatter(y=pdf_low),                    # 8 (Main Low on Left)
            go.Scatter(y=pdf_market),                 # 9 (Market Noise Right)
            go.Scatter(y=pdf_low),                    # 10 (Ghost Low on Right)
            go.Scatter(y=pdf_high)                    # 11 (Main High on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

# Add final frame if not included
if n_steps-1 not in steps_to_anim:
    pass

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
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
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
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
price_range = [60, 115] 

fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=2, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=2, **ax_opts)

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

#### Step-by-Step Example: Filtering the VIX
###### ______________________________________________________________________________________________________________________________________

##### Step 1: Offline Model Calibration (Setting the *"Laws of Physics"*)

Our objective is to filter the VIX as an applied example.  

The *Laws of Physics* in this case call for a mean reverting model.  Plenty of well documented financial economics suggests this.

**Model Selection ($\mathcal{M}$):** Mean Reversion (Ornstein-Ullenback)

 **Mean Reverting (OU) Process as AR(1):**
  
  $$
  x_t = \phi x_{t-1} + b + \epsilon_t
  $$
  
  $$
  \phi = e^{-\kappa \Delta t}
  $$
  $$
  b = (1 - e^{-\kappa \Delta t})\theta
  $$

**Parameter Selection ($\Theta$):** Regression

We must estimate $\Theta = (\phi, b)$ from historical data using regression, standard parameter estimation of an AR(1) model


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Load Data
# Assuming vix.csv has columns: Date, SPY, VIX
df = pd.read_csv('vix.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# 2. Prepare AR(1) Variables
# We regress VIX_t (y) on VIX_{t-1} (x)
df['VIX_lag'] = df['VIX'].shift(1)
df = df.dropna()

y = df['VIX']
x = sm.add_constant(df['VIX_lag']) # Adds the intercept (alpha)

# 3. Fit Linear Regression
model = sm.OLS(y, x).fit()
alpha = model.params['const']
beta = model.params['VIX_lag']
resid_std = np.std(model.resid)

# 4. Convert AR(1) to OU Parameters
# Assume daily data, so delta_t = 1/252 (annualized) 
# or delta_t = 1 (step-wise)
dt = 1/252 

kappa = -np.log(beta) / dt
theta = alpha / (1 - beta)
sigma = resid_std * np.sqrt(-2 * np.log(beta) / (dt * (1 - beta**2)))

print(f"--- Calibrated OU Parameters ---")
print(f"Mean Reversion (kappa): {kappa:.4f}")
print(f"Long-term Mean (theta): {theta:.4f}")
print(f"Vol-of-Vol (sigma):     {sigma:.4f}")
print(f"-------------------------------")
print(f"R-squared: {model.rsquared:.4f}")
```

    --- Calibrated OU Parameters ---
    Mean Reversion (kappa): 25.4140
    Long-term Mean (theta): 15.5129
    Vol-of-Vol (sigma):     43.1361
    -------------------------------
    R-squared: 0.8169


Since we observe estimates for the parameters $\Theta$ we can back out the mean reversion parameters easily

This also allows us to specify $Q$ and $R$

###### ______________________________________________________________________________________________________________________________________

##### Step 2: Initialization (Process Inception)

We must start at some value (best guess) at $t = 0$ ($x_0$)

- $x_0$ is usually set to the first available quote or long-term mean $\theta$
- $P_0$ is confidence in $x_0$, the initial covariance, after *burn in* filter convergence occurs quickly regardless


```python
# Assuming results from Step 1 Calibration
# theta was calibrated as your long-term mean
# vix_quotes is your array of VIX prices from the CSV

# 1. Initialize State to first quote
x_hat = quotes[0] 

# 2. Initialize Covariance 
# We start with some uncertainty (e.g., 1.0)
P = 1.0 

# 3. Define the measurement noise (R)
# Unlike Q, which we calibrated, R is your 'trust' in the quotes.
# Let's assume a 1% standard deviation for sensor noise
R = (0.01 * x_hat)**2 

print(f"Filter Initialized:")
print(f"Initial State Estimate: {x_hat}")
print(f"Initial Uncertainty (P): {P}")
```

    Filter Initialized:
    Initial State Estimate: 95.67321896746141
    Initial Uncertainty (P): 1.0


###### ______________________________________________________________________________________________________________________________________

##### Step 3: Recursive Updating (The Live Filter)

Every quote will be run through:

**Step (a): The "*Model*" Guess**

 - Project the state forward using only the **model** (Laws of Physics)
 
   $$\hat{x}_{t|t-1} = e^{-\kappa \Delta t} \hat{x}_{t-1|t-1} + \theta(1 - e^{-\kappa \Delta t})$$  

 - "Predict" the error covariance
 
   $$P_{t|t-1} = (e^{-\kappa \Delta t})^2 P_{t-1|t-1} + Q$$  

**Step (b): The "*Observation*" Correction**

Now we reconcile the noisy observation with the model's guess (effectively combining observed data with our model prediction)

- **Calculate Innovation:**
$$z_t - \hat{x}_{t|t-1}$$
 
*Translation:* "How much did the market surprise my OU model?"

- **Compute the Kalman Gain (K)**
$$K = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

*Translation:*
Who do I trust more? If model uncertainty ($P$) is high, I trust the quote ($K \to 1$). If sensor noise ($R$) is high, I trust the model ($K \to 0$).

- **Update State Estimate**

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K(\text{innovation})$$

- **Update Error Covariance**
$$P_{t|t} = (1 - K)P_{t|t-1}$$

*Translation:* Because I have seen a new data point, my uncertainty about the 'True Spot' has decreased.



```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# 1. Load Data
df = pd.read_csv('vix.csv')
df['Date'] = pd.to_datetime(df['Date'])
n_steps = len(df)
dates = df['Date']
vix_quotes = df['VIX'].values

# 2. Calibration (AR1 Model)
df_lag = df.copy()
df_lag['VIX_prev'] = df_lag['VIX'].shift(1)
df_lag = df_lag.dropna()

y_cal = df_lag['VIX']
x_cal = sm.add_constant(df_lag['VIX_prev'])
res = sm.OLS(y_cal, x_cal).fit()

alpha_cal = res.params['const']
beta_cal = res.params['VIX_prev']
Q_cal = np.var(res.resid) 
R_assumed = 2.0           

# 3. Kalman Filter Loop
x_est, P_est, K_hist = [], [], []
x, P = vix_quotes[0], 5.0 

for z in vix_quotes:
    x_pred = beta_cal * x + alpha_cal
    P_pred = (beta_cal**2) * P + Q_cal
    innovation = z - x_pred
    K = P_pred / (P_pred + R_assumed)
    x = x_pred + K * innovation
    P = (1 - K) * P_pred
    x_est.append(x)
    P_est.append(P)
    K_hist.append(K)

x_est, P_est, K_hist = np.array(x_est), np.array(P_est), np.array(K_hist)

# 4. Graphics Setup
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.65, 0.35],
    subplot_titles=(
        "<b>VIX Kalman Filter: OU Dynamics & 3-Step Forecast</b>", 
        "<b>Filter Dynamics: Gain (K) & Uncertainty (P)</b>"
    ),
    vertical_spacing=0.22 
)

c_quote = "rgba(255, 255, 255, 0.5)"
c_est = "#00D2FF"      
c_conf = "rgba(0, 210, 255, 0.15)"
c_k = "#FF5500"
c_forecast = "#FFD700" 

# --- INITIALIZE WITH FIRST DATA POINT (Fixes the "Blank Chart" issue) ---
t0 = 0
upper0 = [x_est[t0] + 2 * np.sqrt(P_est[t0])]
lower0 = [x_est[t0] - 2 * np.sqrt(P_est[t0])]
band_x0 = [dates[t0], dates[t0]]
band_y0 = upper0 + lower0

# Trace 0: CI
fig.add_trace(go.Scatter(x=band_x0, y=band_y0, fill='toself', fillcolor=c_conf, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
# Trace 1: Quotes
fig.add_trace(go.Scatter(x=[dates[t0]], y=[vix_quotes[t0]], mode='markers', marker=dict(color=c_quote, size=5)), row=1, col=1)
# Trace 2: Estimate
fig.add_trace(go.Scatter(x=[dates[t0]], y=[x_est[t0]], mode='lines', line=dict(color=c_est, width=3)), row=1, col=1)
# Trace 3: Forecast (Initial 1 point)
fig.add_trace(go.Scatter(x=[dates[t0]], y=[x_est[t0]], mode='lines+markers', line=dict(color=c_forecast, width=2, dash='dot')), row=1, col=1)
# Trace 4: K
fig.add_trace(go.Scatter(x=[dates[t0]], y=[K_hist[t0]], mode='lines', line=dict(color=c_k, width=2)), row=2, col=1)
# Trace 5: P
fig.add_trace(go.Scatter(x=[dates[t0]], y=[P_est[t0]], mode='lines', line=dict(color=c_est, width=2, dash='dot')), row=2, col=1)

# 5. Animation Logic
frames = []
steps_to_anim = range(0, n_steps - 3, 1) 

for t in steps_to_anim:
    frame_name = f"f{t}"
    curr_dates = dates[:t+1]
    curr_x = x_est[:t+1]
    curr_P = P_est[:t+1]
    
    upper_bound = curr_x + 2 * np.sqrt(curr_P)
    lower_bound = curr_x - 2 * np.sqrt(curr_P)
    band_x = np.concatenate([curr_dates, curr_dates[::-1]])
    band_y = np.concatenate([upper_bound, lower_bound[::-1]])

    f_dates = dates[t:t+4] 
    f_values = [x_est[t]]
    for k in range(1, 4):
        f_values.append(beta_cal * f_values[-1] + alpha_cal)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=band_x, y=band_y),
            go.Scatter(x=curr_dates, y=vix_quotes[:t+1]),
            go.Scatter(x=curr_dates, y=curr_x),
            go.Scatter(x=f_dates, y=f_values),
            go.Scatter(x=curr_dates, y=K_hist[:t+1]),
            go.Scatter(x=curr_dates, y=curr_P),
        ],
        traces=[0, 1, 2, 3, 4, 5],
        name=frame_name
    ))

fig.frames = frames

# 6. Layout & Styling
slider_steps = [{"args": [[f"f{t}"], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": str(t), "method": "animate"} for t in steps_to_anim]

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    height=850, width=1100, 
    margin=dict(t=100, b=100, l=80, r=80),
    showlegend=False,
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.12, "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 30, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
        ]
    }],
    sliders=[{"steps": slider_steps, "x": 0.15, "y": -0.12, "len": 0.82}]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False)
fig.update_xaxes(range=[dates.min(), dates.max()], **ax_opts)
fig.update_yaxes(range=[vix_quotes.min()-2, vix_quotes.max()+2], title="VIX Level", row=1, col=1, **ax_opts)
fig.update_yaxes(title_text="Gain (K)", color=c_k, range=[0, 1.0], row=2, col=1, **ax_opts)

fig.update_layout(
    yaxis3=dict(
        title="Uncertainty (P)", color=c_est,
        overlaying="y2", side="right",
        range=[0, P_est.max() * 1.1], showgrid=False,
        anchor="x2"
    )
)
fig.data[5].update(yaxis='y3')

fig.show()
```



**Notice:**  We only calibrate parameters for our mean reversion model (the *Laws of Physics*) one time.  This is still a model specification and parameterization problem.  It may be that eventually a mean reversion model is no longer appropriate, or the parameters that we have calibrated are more violently wrong, as we saw in the misparameterization and mispecification examples above.


If this is the case we may choose to use a dual filter approach for model parameters to update with the state.  

This creates many opportunities for extensions to this classical Kalman Filter approach.

---

#### 3.) 🌪️ Dual Filter Approaches and Real-World Example

##### **Real-World Example:** *Bloomberg Terminal Commanod **BVAL \<GO\>***

Illiquid bonds are a great example of an application of the filter.

- The model is simply the PV of the bond based on some risky spread from risk-free treasuries
- Trades occur infrequently but will inform what the risky spread may be across similar instruments that also infrequently trade

Extending to a more generalized framework, the Kalman Filter can be defined as follows...

 **Kalman Filter**  
  $$ \hat{x}_k = \hat{x}_{k-1} + K_k (z_k - H \hat{x}_{k-1}) $$
  where:  
  $\hat{x}_k$ = estimate of the state at step $k$  
  $K_k$ = *Kalman gain* at step $k$  
  $z_k$ = observation (measurement) at step $k$  
  $H$ = measurement sensitivity (observation model)  
  $\hat{x}_{k-1}$ = previous state estimate

**Kalman Gain**
$$ K_k = \frac{P_k H_k^\top}{H_k P_k H_k^\top + R_k} $$


Bloomberg uses a Kalman Filter approach for illiquid bond pricing.

Something we worked on for quite some time, effectively informing the states using comparable issuances 

When a trade hits the tapes we update the states using the Kalman Gains...

Below I give examples of the Dual Kalman Filter approach to updating parameterizations for regime changes and determining the current state

###### ______________________________________________________________________________________________________________________________________

##### Adapting to New Levels

Regime changes so on and so forth


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
n_steps = 100
T = 1.0
dates = np.linspace(0, T, n_steps)

# True Price: Drifts around 100, drops to 75 at t=50 (Regime Change)
true_price = np.ones(n_steps) * 100
true_price += np.cumsum(np.random.normal(0, 0.3, n_steps)) # Slight drift
true_price[50:] -= 25 # The massive regime change

# Noisy Quotes (Illiquid Market)
R_true = 25.0 # True variance of the noise
quotes = true_price + np.random.normal(0, np.sqrt(R_true), n_steps)

# --- 2. Kalman Filter Implementations ---
# Shared base parameters
R_assumed = 40.0  # High assumed sensor noise
Q_assumed = 0.1   # Low assumed process noise (trusts model)

def run_kalman_filter(quotes, R, Q, is_adaptive=False):
    x_est = []
    P_est = []
    
    # Initial state
    x = 100.0
    P = 1.0
    
    for i, z in enumerate(quotes):
        # 1. Predict (Prior)
        x_pred = x
        P_pred = P + Q
        
        # --- THE ADAPTIVE "DUAL" LOGIC ---
        # Calculate the "Innovation" (Residual Error)
        innovation = z - x_pred
        
        # Calculate the expected variance of this innovation
        # (Model Uncertainty + Sensor Uncertainty)
        innovation_variance = P_pred + R
        
        if is_adaptive:
            # Check if the error is mathematically extreme (e.g., > 3 standard deviations)
            # 3-sigma variance threshold is 3^2 = 9
            if innovation**2 > 9 * innovation_variance:
                # Shock detected! The filter realizes it is wrong and artificially inflates P
                P_pred += 150.0 
        
        # 2. Update (Posterior)
        K = P_pred / (P_pred + R) # Kalman Gain
        x = x_pred + K * innovation
        P = (1 - K) * P_pred
        
        x_est.append(x)
        P_est.append(P)
        
    return np.array(x_est), np.array(P_est)

# Scenario A: Strict Model (ignores innovation testing, misses regime change)
x_low, P_low = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, is_adaptive=False)

# Scenario B: Autonomous Dynamic adjustment (detects shock at t=50 automatically)
x_high, P_high = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, is_adaptive=True)


# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Strict Model (Low K)</b><br>High R, Ignores Noise", 
        "<b>Adaptive Filter (Dynamic K)</b><br>Auto-Detects Regime Change",
        "Belief vs Market (Strict)", 
        "Belief vs Market (Adaptive)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_low = "#00D2FF"       # Cyan for Strict
c_low_ghost = "rgba(0, 210, 255, 0.15)"
c_high = "#FF5500"      # Orange for Adaptive
c_high_ghost = "rgba(255, 85, 0, 0.15)"
c_quote_dist = "rgba(200, 200, 200, 0.2)" # Grey for True Market Noise
c_quote = "rgba(255, 255, 255, 0.6)"
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True Price [Indices 0, 1]
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=2)

# Dynamic Traces Initialization: Quotes (2, 3) and KF Lines (4, 5)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=2)

fig.add_trace(go.Scatter(x=[dates[0]], y=[x_low[0]], mode='lines', line=dict(color=c_low, width=3), name='KF Strict'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[x_high[0]], mode='lines', line=dict(color=c_high, width=3), name='KF Adaptive'), row=1, col=2)

# Dynamic Traces Initialization: Density Plots [Indices 6 through 11]
x_grid = np.linspace(60, 115, 300)
y_zero = np.zeros_like(x_grid)

# Left Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=1) # 6
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_high_ghost, name='Ghost Adaptive'), row=2, col=1) # 7
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_low), name='Main Strict'), row=2, col=1) # 8

# Right Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=2) # 9
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_low_ghost, name='Ghost Strict'), row=2, col=2) # 10
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_high), name='Main Adaptive'), row=2, col=2) # 11


# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps, 2) # Skip every other frame for performance

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # Slice current data
    curr_dates = dates[:t+1]
    curr_quotes = quotes[:t+1]
    curr_x_low = x_low[:t+1]
    curr_x_high = x_high[:t+1]
    
    # Calculate Current PDFs
    pdf_market = norm.pdf(x_grid, true_price[t], np.sqrt(R_true))
    pdf_low = norm.pdf(x_grid, x_low[t], np.sqrt(P_low[t]))
    pdf_high = norm.pdf(x_grid, x_high[t], np.sqrt(P_high[t]))

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_quotes),  # 2
            go.Scatter(x=curr_dates, y=curr_quotes),  # 3
            go.Scatter(x=curr_dates, y=curr_x_low),   # 4
            go.Scatter(x=curr_dates, y=curr_x_high),  # 5
            go.Scatter(y=pdf_market),                 # 6 (Market Noise Left)
            go.Scatter(y=pdf_high),                   # 7 (Ghost High on Left)
            go.Scatter(y=pdf_low),                    # 8 (Main Low on Left)
            go.Scatter(y=pdf_market),                 # 9 (Market Noise Right)
            go.Scatter(y=pdf_low),                    # 10 (Ghost Low on Right)
            go.Scatter(y=pdf_high)                    # 11 (Main High on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

# Add final frame if not included
if n_steps-1 not in steps_to_anim:
    pass

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
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
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
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
price_range = [60, 115] 

fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=2, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=2, **ax_opts)

fig.show()
```



This approach can be extended to attempt to discern periods of reversion from trending, high and low volatility, so on and so forth.

If there is interest in applications like these in the future, let me know!  

I'd also like to do a quant build implementing a live KF if there is interest!

---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Any time we are trying to make an informed decision in the face of uncertainty we have a model specification and parameterization problem
- Model selection is rarely the issue, we have a variety of different ways to inform selection, and many are great at capturing dynamics we observe empirically, in any case a model may work, break, work again, so on and so forth
- Model parameterization behaves similarly, it can also make it appear that your specification is incorrect, a parameterization may work, break, work again, so on and so forth
- We'd like to move to a more dynamic modeling space, the Kalman Filter is one such way model to accomplish this
- Effectively, we are combining a model representation that we select and parameterize with data that we actually observe, we are then able to pull different levers to dictate how confident we are in the data (is it noisy?) relative to our model (is it grounded in economic theory?)
- There are a variety of statistical techniques available to us to accomplish the initial model selection, parameterization, and even the filter parameterization itself, herein we've covered the general notion of the filter, a step-by-step example with real-world data (VIX), and a few different examples of extensions in the context of a dual filter approach
- Moreover, there are many extensions that aim to discern different states and inform parameterization and filter states accordingly (trending/reversion, high/low vol, so on and so forth), if there is interest in further videos on the KF I'd certainly be happy to do more deep dives into these applications and even create a quant build implementing these models live

**Future Topics**

Technical Videos and Other Discussions

- Fama-French / Carhart and Factor Modeling in General
- Hawkes Processes
- Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
- Market-Making Models and Simulation (Stoikov-Avellaneda)
- My First Year as a Quant
- Why Hedge Funds are Actually Secretive
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Top 3 Uses of Linear Algebra for Quant Finance
- Girsanov's Change of Measure
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Kalman Filter Model with Interactive Brokers
- How to Backtest a Trading Strategy with Interactive Brokers
- Algorithmic Volatility Trading System with Interactive Brokers

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
