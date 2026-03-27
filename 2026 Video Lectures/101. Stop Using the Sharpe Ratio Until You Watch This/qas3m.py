# ### 📈 Stop Using the Sharpe Ratio Until You Watch This
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [The 5 Papers That Built Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)
# 
# - [I Bet You've Never Found Alpha (and I Can Prove It)](https://youtu.be/UzTJHs3-eT0)
# 
# - [Quant Ranks Retail Trading Mistakes that Blow Up Your Account](https://youtu.be/1mpNxBaBeOw)
# 
# - [Non-Stationarity and Why Market Timing Fails](https://youtu.be/7nvjrgqKjJE)
# 
# - [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)
# 
# - [How a Quant Manages a Portfolio](https://youtu.be/mZLNzqDZHbA)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
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
# #### 1. ) 📈 Sharpe Ratio
# 
# - Expected Returns and Variance
# 
# - Decision Making with the Sharpe Ratio
# 
# #### 2.) 📉 Where the Sharpe Ratio Fails
# 
# - Downside Variation
# 
# - Stability in Forward Returns
# 
# - What Causes Sharpe Instability
# 
# #### 3.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1. ) 📈 Sharpe Ratio
# 
# ##### Expected Returns and Variance
# 
# Whenever we are analyzing a trading strategy or portfolio ($\pi$) we love to discuss statistics, they give us insight into performance
# 
# $$
#  \mathbb{E}[\pi] = \frac{1}{n} \sum_{i=1}^n r_i \quad
#  \quad\quad\quad \mathrm{Var}[\pi] = \mathbb{E}\Big[(\pi - \mathbb{E}[\pi])^2\Big] = \frac{1}{n} \sum_{i=1}^n (r_i - \mathbb{E}[\pi])^2
# $$
# 
# These are properties of the *path* itself, not necessarily the data generating distribution
# 
# Let's observe a walked path relative to other potential outcomes


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# ==========================================
# --- 1. DATA GENERATION & MOMENT MATCHING ---
# ==========================================
np.random.seed(42)

n_years = 3
n_days = 252 * n_years
t_steps = np.linspace(0, n_years, n_days)  # Time from 0 to 3 years
dt = 1 / 252

# Target Parameters
mu_1, var_1 = 0.30, 0.15
sigma_1 = np.sqrt(var_1)

mu_2, var_2 = 0.30, 0.50
sigma_2 = np.sqrt(var_2)

def generate_exact_path(target_mu, target_var, n):
    """Generates a random walk where the empirical sample stats EXACTLY match the targets."""
    raw_shocks = np.random.normal(0, 1, n - 1)
    
    # Moment Matching: Force Mean=0, StdDev=1
    z_scores = (raw_shocks - np.mean(raw_shocks)) / np.std(raw_shocks)
    
    # Scale to daily target moments
    daily_mu = target_mu * dt
    daily_sigma = np.sqrt(target_var * dt)
    exact_increments = z_scores * daily_sigma + daily_mu
    
    path = np.zeros(n)
    path[1:] = np.cumsum(exact_increments)
    
    full_increments = np.concatenate(([0], exact_increments))
    realized_mu = np.mean(exact_increments) * 252
    realized_var = np.var(exact_increments) * 252
    
    return path, realized_mu, realized_var, full_increments, daily_mu, daily_sigma

def generate_random_path(target_mu, target_var, n):
    """Generates a pure random walk (No moment matching). Will naturally drift."""
    daily_mu = target_mu * dt
    daily_sigma = np.sqrt(target_var * dt)
    
    # Pure random draws from the theoretical distribution
    random_increments = np.random.normal(daily_mu, daily_sigma, n - 1)
    
    path = np.zeros(n)
    path[1:] = np.cumsum(random_increments)
    return path

# 1. Generate EXACT primary paths (Preserves your original exact paths based on seed 42)
path_1, emp_mu_1, emp_var_1, inc_1, d_mu_1, d_sig_1 = generate_exact_path(mu_1, var_1, n_days)
path_2, emp_mu_2, emp_var_2, inc_2, d_mu_2, d_sig_2 = generate_exact_path(mu_2, var_2, n_days)

# 2. Generate PURE RANDOM background paths to show the natural spread/fan
bg_paths_1 = [generate_random_path(mu_1, var_1, n_days) for _ in range(9)]
bg_paths_2 = [generate_random_path(mu_2, var_2, n_days) for _ in range(9)]

# Theoretical Expectation & Bounds (Cumulative)
expected_1 = mu_1 * t_steps
bound_1 = 2 * sigma_1 * np.sqrt(t_steps)
upper_1, lower_1 = expected_1 + bound_1, expected_1 - bound_1

expected_2 = mu_2 * t_steps
bound_2 = 2 * sigma_2 * np.sqrt(t_steps)
upper_2, lower_2 = expected_2 + bound_2, expected_2 - bound_2

# Theoretical Probability Density Functions (PDF) for Daily Returns
x_pdf = np.linspace(min(d_mu_2 - 4*d_sig_2, -0.15), max(d_mu_2 + 4*d_sig_2, 0.15), 300)
pdf_1 = stats.norm.pdf(x_pdf, d_mu_1, d_sig_1)
pdf_2 = stats.norm.pdf(x_pdf, d_mu_2, d_sig_2)
max_pdf = max(pdf_1.max(), pdf_2.max())

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = max(1, n_days // 150)
frame_indices = list(range(1, n_days, step_size))
if frame_indices[-1] != n_days - 1:
    frame_indices.append(n_days - 1)

for k in frame_indices:
    frame_data = []
    
    # --- CHART 1 (TOP LEFT): Cumulative Paths ---
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_1[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    # 9 Random Background Paths (0.35 opacity)
    for bg_path in bg_paths_1:
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=bg_path[:k+1], mode='lines', line=dict(color='rgba(0, 255, 255, 0.2)', width=.75)))
    # Primary Moment-Matched Path (solid)
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=path_1[:k+1], mode='lines', line=dict(color='#00ffff', width=3)))

    # --- CHART 3 (BOTTOM LEFT): Distribution & Draw ---
    current_draw_1 = inc_1[k]
    frame_data.append(go.Scatter(x=x_pdf, y=pdf_1, mode='lines', fill='tozeroy', fillcolor='rgba(0,255,255,0.1)', line=dict(color='#00ffff', width=2)))
    frame_data.append(go.Scatter(x=[current_draw_1, current_draw_1], y=[0, max_pdf], mode='lines', line=dict(color='red', width=3)))

    # --- CHART 2 (TOP RIGHT): Cumulative Paths ---
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_2[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    # 9 Random Background Paths (0.35 opacity)
    for bg_path in bg_paths_2:
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=bg_path[:k+1], mode='lines', line=dict(color='rgba(255, 57, 179, 0.2)', width=.75)))
    # Primary Moment-Matched Path (solid)
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=path_2[:k+1], mode='lines', line=dict(color='#ff39b3', width=3)))

    # --- CHART 4 (BOTTOM RIGHT): Distribution & Draw ---
    current_draw_2 = inc_2[k]
    frame_data.append(go.Scatter(x=x_pdf, y=pdf_2, mode='lines', fill='tozeroy', fillcolor='rgba(255,57,179,0.1)', line=dict(color='#ff39b3', width=2)))
    frame_data.append(go.Scatter(x=[current_draw_2, current_draw_2], y=[0, max_pdf], mode='lines', line=dict(color='red', width=3)))

    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    row_heights=[0.75, 0.25],
    vertical_spacing=0.08,
    horizontal_spacing=0.08,
    subplot_titles=[
        "Path 1: Exact Mean vs. Random Drifts", "Path 2: Exact Mean vs. Wild Drifts",
        "Daily Data Generating Distribution", "Daily Data Generating Distribution"
    ]
)

# Add Dummy Traces to match the traces in the frame_data loop exactly
for _ in range(13): fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=1)
for _ in range(2): fig.add_trace(go.Scatter(x=[0], y=[0]), row=2, col=1)
for _ in range(13): fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=2)
for _ in range(2): fig.add_trace(go.Scatter(x=[0], y=[0]), row=2, col=2)

fig.frames = frames

# ==========================================
# --- 4. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step (Years): "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.15,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(round(t_steps[k], 2))
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=800, width=1200,
    title_text="Moment Matched Primary Path vs. Random Realizations",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Combine all paths to find perfect global limits
all_paths_1 = [path_1] + bg_paths_1
all_paths_2 = [path_2] + bg_paths_2

global_y_max = max(upper_1.max(), upper_2.max(), np.max(all_paths_1), np.max(all_paths_2)) * 1.15
global_y_min = min(lower_1.min(), lower_2.min(), np.min(all_paths_1), np.min(all_paths_2)) * 1.15

for col in [1, 2]:
    fig.update_xaxes(title_text='', range=[0, n_years + 0.05], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".2f")
    fig.update_yaxes(title_text='Cumulative Return', range=[global_y_min, global_y_max], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".0%")

x_dist_limit = max(abs(x_pdf.min()), abs(x_pdf.max()))
for col in [1, 2]:
    fig.update_xaxes(title_text='Daily Return (Draw)', range=[-x_dist_limit, x_dist_limit], row=2, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".1%")
    fig.update_yaxes(title_text='Density', range=[0, max_pdf * 1.05], row=2, col=col, gridcolor='rgba(128,128,128,0.2)')

# --- ADD OVERLAY STATISTIC ANNOTATIONS ---
fig.add_annotation(
    text=f"<b>Target E[r]</b> = {emp_mu_1*100:.1f}%<br><b>Target Var</b> = {emp_var_1*100:.1f}%",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=8, row=1, col=1
)

fig.add_annotation(
    text=f"<b>Target E[r]</b> = {emp_mu_2*100:.1f}%<br><b>Target Var</b> = {emp_var_2*100:.1f}%",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#ff39b3", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, borderpad=8, row=1, col=2
)

fig.show()


#  **Volatility Drag:**  
# 
#  Volatility reduces geometric (compounded) wealth growth compared to arithmetic returns.  
# 
#  The geometric growth rate is given by:  
#      $$\text{Geo Mean} \approx \mu - \frac{1}{2}\sigma^2$$  
# 
#  Thus, higher volatility "drags" down long-run wealth—even if arithmetic returns are unchanged.


# Maybe there is merit to the statistics we've observed in our path relative to the data generating distirbution, maybe it is indicative of tradable structure, maybe...


# ###### ______________________________________________________________________________________________________________________________________


# ##### Do You Understand Statistics are a Compression?
# 
# What we observed above was *an outcome of a sample path*, they are *one possible statistic* contained within the cone of variance (68% of outcomes with normality)
# 
# Below is an example of all of the ways in which we could have traveled to produce the same statistic, we do not get to choose which path we walk


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MOMENT MATCHING ---
# ==========================================
np.random.seed(42)

n_years = 3
n_steps = 252 * n_years
dt = 1 / 252
t_steps = np.arange(n_steps + 1) * dt  # Time from 0 to 3 years

# Target Parameters
mu_1, var_1 = 0.30, 0.15
sigma_1 = np.sqrt(var_1)

mu_2, var_2 = 0.30, 0.50
sigma_2 = np.sqrt(var_2)

n_paths = 10
opacities = np.linspace(1.0, 0.25, n_paths)

def generate_exact_path(target_mu, target_var, num_points):
    """Generates a random walk where the empirical sample stats exactly match the targets."""
    n_inc = num_points - 1
    # 1. Generate raw random shocks
    raw_shocks = np.random.normal(0, 1, n_inc)
    
    # 2. Moment Matching: Force the sample to exactly Mean=0, StdDev=1
    z_scores = (raw_shocks - np.mean(raw_shocks)) / np.std(raw_shocks)
    
    # 3. Scale to exact daily target moments
    daily_mu = target_mu * dt
    daily_sigma = np.sqrt(target_var * dt)
    exact_increments = z_scores * daily_sigma + daily_mu
    
    # 4. Build the cumulative path starting at 0
    path = np.zeros(num_points)
    path[1:] = np.cumsum(exact_increments)
    
    return path

# Generate 10 moment-matched paths for each chart
paths_1 = [generate_exact_path(mu_1, var_1, len(t_steps)) for _ in range(n_paths)]
paths_2 = [generate_exact_path(mu_2, var_2, len(t_steps)) for _ in range(n_paths)]

# Theoretical Expectation & Bounds (Cumulative)
expected_1 = mu_1 * t_steps
bound_1 = 2 * sigma_1 * np.sqrt(t_steps)
upper_1, lower_1 = expected_1 + bound_1, expected_1 - bound_1

expected_2 = mu_2 * t_steps
bound_2 = 2 * sigma_2 * np.sqrt(t_steps)
upper_2, lower_2 = expected_2 + bound_2, expected_2 - bound_2

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = max(1, len(t_steps) // 100)
frame_indices = list(range(1, len(t_steps), step_size))
if frame_indices[-1] != len(t_steps) - 1:
    frame_indices.append(len(t_steps) - 1)

for k in frame_indices:
    frame_data = []
    
    # --- CHART 1 TRACES ---
    # 1-3: Theoretical Lines
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_1[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    
    # 4-13: 10 Fading Paths
    for p in range(n_paths):
        alpha = opacities[p]
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=paths_1[p][:k+1], mode='lines', 
                                     line=dict(color=f'rgba(0, 255, 255, {alpha:.2f})', width=1.5)))

    # --- CHART 2 TRACES ---
    # 14-16: Theoretical Lines
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_2[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    
    # 17-26: 10 Fading Paths
    for p in range(n_paths):
        alpha = opacities[p]
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=paths_2[p][:k+1], mode='lines', 
                                     line=dict(color=f'rgba(255, 57, 179, {alpha:.2f})', width=1.5)))

    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=["Path 1: Low Variance", "Path 2: High Variance"],
    horizontal_spacing=0.08
)

# 3 bounds/expected lines + 10 paths = 13 traces per chart
traces_per_chart = 3 + n_paths
for _ in range(traces_per_chart):
    fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=1)
for _ in range(traces_per_chart):
    fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=2)

fig.frames = frames

# ==========================================
# --- 4. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step (Years): "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.15,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(round(t_steps[k], 2))
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=600, width=1200,
    title_text="Moment Matched Bundles: Infinite Paths, Exact Same Realized Statistics",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce strictly static, identical axes limits for objective comparison
all_paths_1 = np.array(paths_1)
all_paths_2 = np.array(paths_2)
max_val = max(upper_1.max(), upper_2.max(), all_paths_1.max(), all_paths_2.max())
min_val = min(lower_1.min(), lower_2.min(), all_paths_1.min(), all_paths_2.min())

global_y_max = max_val + abs(max_val) * 0.15
global_y_min = min_val - abs(min_val) * 0.15 - 0.05 

for col in [1, 2]:
    fig.update_xaxes(title_text='Time (Years)', range=[0, n_years + 0.05], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".2f")
    fig.update_yaxes(title_text='Cumulative Return', range=[global_y_min, global_y_max], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".0%")

# --- ADD OVERLAY STATISTIC ANNOTATIONS ---
fig.add_annotation(
    text=f"<b>Realized E[r]</b> = {mu_1*100:.0f}%<br><b>Realized Var</b> = {var_1*100:.0f}%",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=8, row=1, col=1
)

fig.add_annotation(
    text=f"<b>Realized E[r]</b> = {mu_2*100:.0f}%<br><b>Realized Var</b> = {var_2*100:.0f}%",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#ff39b3", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, borderpad=8, row=1, col=2
)

fig.show()


# We'll return to the efficacy of these statistics later on, until then...
# 
# Which portfolio or strategy would you prefer to allocate capital to should they both have the same expected return?
# 
# Clearly, we would prefer the one with less *volatility*, it gives us more confidence in the strategy or portfolio and can even encourage the use of leverage
# 
# Is there a way to define this visual confidence?  There certainly is, and we call it a measure of risk-adjusted return.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Decision Making with the Sharpe Ratio
# 
# The sharpe ratio takes into account upside and downside deviation from expectation to produce a measure of risk-reward.
# 
# It is defined as the following ratio:
# 
#  $$
#  \text{Sharpe Ratio} = \frac{\mathbb{E}[\pi]}{\sqrt{\mathrm{Var}[\pi]}}
#  $$
#  
#  We typically look at performance metrics like the Sharpe ratio over rolling windows as they are subject to change over time (*more on this in a moment*).


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MOMENT MATCHING ---
# ==========================================
np.random.seed(42)

n_years = 3
n_steps = 252 * n_years
dt = 1 / 252
t_steps = np.arange(n_steps + 1) * dt  # Time from 0 to 3 years
risk_free_rate = 0.0

# Target Parameters
mu_1, var_1 = 0.35, 0.10
sigma_1 = np.sqrt(var_1)
sharpe_1 = (mu_1 - risk_free_rate) / sigma_1

mu_2, var_2 = 0.35, 0.40
sigma_2 = np.sqrt(var_2)
sharpe_2 = (mu_2 - risk_free_rate) / sigma_2

n_paths = 10
opacities = np.linspace(1.0, 0.25, n_paths)

def generate_exact_path(target_mu, target_var, num_points):
    """Generates a random walk where the empirical sample stats exactly match the targets."""
    n_inc = num_points - 1
    # 1. Generate raw random shocks
    raw_shocks = np.random.normal(0, 1, n_inc)
    
    # 2. Moment Matching: Force the sample to exactly Mean=0, StdDev=1
    z_scores = (raw_shocks - np.mean(raw_shocks)) / np.std(raw_shocks)
    
    # 3. Scale to exact daily target moments
    daily_mu = target_mu * dt
    daily_sigma = np.sqrt(target_var * dt)
    exact_increments = z_scores * daily_sigma + daily_mu
    
    # 4. Build the cumulative path starting at 0
    path = np.zeros(num_points)
    path[1:] = np.cumsum(exact_increments)
    
    return path

# Generate 10 moment-matched paths for each chart
paths_1 = [generate_exact_path(mu_1, var_1, len(t_steps)) for _ in range(n_paths)]
paths_2 = [generate_exact_path(mu_2, var_2, len(t_steps)) for _ in range(n_paths)]

# Theoretical Expectation & Bounds (Cumulative)
expected_1 = mu_1 * t_steps
bound_1 = 2 * sigma_1 * np.sqrt(t_steps)
upper_1, lower_1 = expected_1 + bound_1, expected_1 - bound_1

expected_2 = mu_2 * t_steps
bound_2 = 2 * sigma_2 * np.sqrt(t_steps)
upper_2, lower_2 = expected_2 + bound_2, expected_2 - bound_2

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = max(1, len(t_steps) // 100)
frame_indices = list(range(1, len(t_steps), step_size))
if frame_indices[-1] != len(t_steps) - 1:
    frame_indices.append(len(t_steps) - 1)

for k in frame_indices:
    frame_data = []
    
    # --- CHART 1 TRACES ---
    # 1-3: Theoretical Lines
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_1[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_1[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    
    # 4-13: 10 Fading Paths
    for p in range(n_paths):
        alpha = opacities[p]
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=paths_1[p][:k+1], mode='lines', 
                                     line=dict(color=f'rgba(0, 255, 255, {alpha:.2f})', width=1.5)))

    # --- CHART 2 TRACES ---
    # 14-16: Theoretical Lines
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=expected_2[:k+1], mode='lines', line=dict(color='yellow', width=2, dash='dash')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=upper_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    frame_data.append(go.Scatter(x=t_steps[:k+1], y=lower_2[:k+1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot')))
    
    # 17-26: 10 Fading Paths
    for p in range(n_paths):
        alpha = opacities[p]
        frame_data.append(go.Scatter(x=t_steps[:k+1], y=paths_2[p][:k+1], mode='lines', 
                                     line=dict(color=f'rgba(255, 57, 179, {alpha:.2f})', width=1.5)))

    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=[f"Path 1: Sharpe Ratio = {sharpe_1:.2f}", f"Path 2: Sharpe Ratio = {sharpe_2:.2f}"],
    horizontal_spacing=0.08
)

# 3 bounds/expected lines + 10 paths = 13 traces per chart
traces_per_chart = 3 + n_paths
for _ in range(traces_per_chart):
    fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=1)
for _ in range(traces_per_chart):
    fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=2)

fig.frames = frames

# ==========================================
# --- 4. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step (Years): "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.15,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(round(t_steps[k], 2))
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=600, width=1200,
    title_text="Visualizing the Sharpe Ratio via Moment-Matched Path Bundles",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce strictly static, identical axes limits for objective comparison
all_paths_1 = np.array(paths_1)
all_paths_2 = np.array(paths_2)
max_val = max(upper_1.max(), upper_2.max(), all_paths_1.max(), all_paths_2.max())
min_val = min(lower_1.min(), lower_2.min(), all_paths_1.min(), all_paths_2.min())

global_y_max = max_val + abs(max_val) * 0.15
global_y_min = min_val - abs(min_val) * 0.15 - 0.05 

for col in [1, 2]:
    fig.update_xaxes(title_text='Time (Years)', range=[0, n_years + 0.05], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".2f")
    fig.update_yaxes(title_text='Cumulative Return', range=[global_y_min, global_y_max], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".0%")

# --- ADD OVERLAY STATISTIC ANNOTATIONS ---
fig.add_annotation(
    text=f"<b>E[r] (μ)</b> = {mu_1*100:.0f}%<br><b>Vol (σ)</b> = {sigma_1*100:.1f}%<br><b>Realized Sharpe</b> = {sharpe_1:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=8, row=1, col=1
)

fig.add_annotation(
    text=f"<b>E[r] (μ)</b> = {mu_2*100:.0f}%<br><b>Vol (σ)</b> = {sigma_2*100:.1f}%<br><b>Realized Sharpe</b> = {sharpe_2:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#ff39b3", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, borderpad=8, row=1, col=2
)

fig.show()


# Is it really that easy?  Compare strategies or portfolios with different Sharpe ratios and select the highest one?
# 
# No, it doesn't quite work like that, and the selection of strategy and portfolio based entirely on performance metrics can be incredibly misleading.


# ---


# #### 2.) 📉 Where the Sharpe Ratio Fails
# 
# ##### Downside Variation
# 
# Let's assume for the moment that the statistics we are observing *are* indicative of out of sample performance
# 
# The first place the Sharpe ratio fails is in terms of upside deviation.  Recall variance is defined in the sense of absolute (squared) deviations from expectation.
# 
# $$\mathrm{Var}[\pi] = \mathbb{E}\Big[(\pi - \mathbb{E}[\pi])^2\Big] = \frac{1}{n} \sum_{i=1}^n (r_i - \mathbb{E}[\pi])^2$$
# 
# This means the Sharpe ratio is penalizing deviations from expectation on the upside.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulate two strategies ---
np.random.seed(7)
T = 252
days = np.arange(1, T + 1) / 252  # Express days as "years" for axis style matching

# Strategy A: higher expected return, but variance inflated by occasional large positive jumps
rA = 0.00015 + np.random.normal(0, 0.0055, T) + (np.random.rand(T) < 0.02) * 0.03
# Strategy B: lower expected return, smoother return stream
rB = 0.00045 + np.random.normal(0, 0.0030, T)

wealthA = np.cumprod(1 + rA)
wealthB = np.cumprod(1 + rB)

# --- Compute realized 'expected return' and Sharpe for full paths (no risk-free, just ratio of annualized mean/std) ---
def realized_stats(r):
    mean = np.mean(r)
    std = np.std(r)
    exp_return = mean * 252      # annualized expected return
    vol = std * np.sqrt(252)     # annualized standard deviation
    sharpe = exp_return / vol if vol > 0 else np.nan
    return exp_return, sharpe

exp_return_A, sharpe_A = realized_stats(rA)
exp_return_B, sharpe_B = realized_stats(rB)

# === Frame generation, analogous to the attached style ===
frames = []
step_size = max(1, len(days) // 100)
frame_indices = list(range(1, len(days), step_size))
if frame_indices[-1] != len(days) - 1:
    frame_indices.append(len(days) - 1)

for k in frame_indices:
    frame_data = []

    # Chart 1: Strategy A (Cyan style)
    frame_data.append(go.Scatter(x=days[:k+1], y=wealthA[:k+1], mode='lines',
                                 line=dict(color='rgba(0,255,255,1)', width=2)))
    # Chart 2: Strategy B (Magenta style)
    frame_data.append(go.Scatter(x=days[:k+1], y=wealthB[:k+1], mode='lines',
                                 line=dict(color='rgba(255,57,179,1)', width=2)))
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# --- Figure initialization ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Strategy A: 'Jumpy' High Return", "Strategy B: Smooth Slope"],
    horizontal_spacing=0.08
)

# One path per chart: placeholders for animation
fig.add_trace(go.Scatter(x=[days[0]], y=[wealthA[0]]), row=1, col=1)
fig.add_trace(go.Scatter(x=[days[0]], y=[wealthB[0]]), row=1, col=2)

fig.frames = frames

# --- Layout: dark background, overlay slider and play, matching style ---
sliders = [dict(
    active=0, currentvalue={"prefix": "Time (Years): "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.15,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(round(days[k], 2))
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=600, width=1200,
    title_text="Comparing Equity Curves: Two Strategies, Different Risk/Return Profiles",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders,
    margin=dict(t=80, b=80, r=40, l=40),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Axes: strict fixed range, grid styling
min_y = min(wealthA.min(), wealthB.min())
max_y = max(wealthA.max(), wealthB.max())
y_pad = (max_y - min_y) * 0.15

for col in [1, 2]:
    fig.update_xaxes(title_text='Time (Years)', range=[0, 1.02], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".2f")
    fig.update_yaxes(title_text='Equity (Cumulative Product)', range=[min_y - y_pad, max_y + y_pad], row=1, col=col, gridcolor='rgba(128,128,128,0.2)')

# Stat annotation overlays -- use Expected Return and Sharpe based on path
fig.add_annotation(
    text=f"<b>Expected Return</b> (annualized) = {100*exp_return_A:.2f}%<br><b>Sharpe Ratio</b> = {sharpe_A:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=8, row=1, col=1
)
fig.add_annotation(
    text=f"<b>Expected Return</b> (annualized) = {100*exp_return_B:.2f}%<br><b>Sharpe Ratio</b> = {sharpe_B:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#ff39b3", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, borderpad=8, row=1, col=2
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 
# In some cases we do want a measure the overall sensitivity of the path to the expected value
# 
# In others, we really don't care above returns deviating from expectation in a way that's *too positive*, we can remove this penalty using the Sortino ratio


#   $$
#   \text{Sortino Ratio} = \frac{\mathbb{E}[\pi]}{\sqrt{\mathrm{Var}^-[\pi]}}
#   $$
#   where $\mathrm{Var}^-[\pi]$ is the downside semi-variance of returns $\pi$ (i.e., only the variance of returns below the risk-free rate or zero).


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulate two strategies ---
np.random.seed(7)
T = 252
days = np.arange(1, T + 1) / 252  # Express days as "years" for axis style matching

# Strategy A: higher expected return, but variance inflated by occasional large positive jumps
rA = 0.00015 + np.random.normal(0, 0.0055, T) + (np.random.rand(T) < 0.02) * 0.03
# Strategy B: lower expected return, smoother return stream
rB = 0.00045 + np.random.normal(0, 0.0030, T)

wealthA = np.cumprod(1 + rA)
wealthB = np.cumprod(1 + rB)

# --- Compute realized 'expected return' and Sortino for full paths (no risk-free, just ratio of annualized mean/downside semidev) ---
def downside_semideviation(r, threshold=0.0):
    """Computes downside standard deviation (semi-deviation) from threshold (default 0, risk-free rate can be placed here)"""
    downside = r[r < threshold] - threshold
    if len(downside) == 0:
        return 0.0
    return np.sqrt(np.mean(downside ** 2))

def realized_stats_sortino(r, threshold=0.0):
    mean = np.mean(r)
    exp_return = mean * 252  # annualized expected return
    semi_dev = downside_semideviation(r, threshold) * np.sqrt(252)  # annualized downside deviation
    sortino = exp_return / semi_dev if semi_dev > 0 else np.nan
    return exp_return, sortino

exp_return_A, sortino_A = realized_stats_sortino(rA)
exp_return_B, sortino_B = realized_stats_sortino(rB)

# === Frame generation, analogous to the attached style ===
frames = []
step_size = max(1, len(days) // 100)
frame_indices = list(range(1, len(days), step_size))
if frame_indices[-1] != len(days) - 1:
    frame_indices.append(len(days) - 1)

for k in frame_indices:
    frame_data = []

    # Chart 1: Strategy A (Cyan style)
    frame_data.append(go.Scatter(x=days[:k+1], y=wealthA[:k+1], mode='lines',
                                 line=dict(color='rgba(0,255,255,1)', width=2)))
    # Chart 2: Strategy B (Magenta style)
    frame_data.append(go.Scatter(x=days[:k+1], y=wealthB[:k+1], mode='lines',
                                 line=dict(color='rgba(255,57,179,1)', width=2)))
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# --- Figure initialization ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Strategy A: 'Jumpy' High Return", "Strategy B: Smooth Slope"],
    horizontal_spacing=0.08
)

# One path per chart: placeholders for animation
fig.add_trace(go.Scatter(x=[days[0]], y=[wealthA[0]]), row=1, col=1)
fig.add_trace(go.Scatter(x=[days[0]], y=[wealthB[0]]), row=1, col=2)

fig.frames = frames

# --- Layout: dark background, overlay slider and play, matching style ---
sliders = [dict(
    active=0, currentvalue={"prefix": "Time (Years): "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.15,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(round(days[k], 2))
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=600, width=1200,
    title_text="Comparing Equity Curves: Two Strategies, Different Risk/Return Profiles",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders,
    margin=dict(t=80, b=80, r=40, l=40),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Axes: strict fixed range, grid styling
min_y = min(wealthA.min(), wealthB.min())
max_y = max(wealthA.max(), wealthB.max())
y_pad = (max_y - min_y) * 0.15

for col in [1, 2]:
    fig.update_xaxes(title_text='Time (Years)', range=[0, 1.02], row=1, col=col, gridcolor='rgba(128,128,128,0.2)', tickformat=".2f")
    fig.update_yaxes(title_text='Equity (Cumulative Product)', range=[min_y - y_pad, max_y + y_pad], row=1, col=col, gridcolor='rgba(128,128,128,0.2)')

# Stat annotation overlays -- use Expected Return and Sortino based on path
fig.add_annotation(
    text=f"<b>Expected Return</b> (annualized) = {100*exp_return_A:.2f}%<br><b>Sortino Ratio</b> = {sortino_A:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=8, row=1, col=1
)
fig.add_annotation(
    text=f"<b>Expected Return</b> (annualized) = {100*exp_return_B:.2f}%<br><b>Sortino Ratio</b> = {sortino_B:.2f}",
    xref="x domain", yref="y domain", x=0.05, y=0.95, showarrow=False, align="left", font=dict(color="#ff39b3", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, borderpad=8, row=1, col=2
)

fig.show()


# Of course, the statistics **might** tell us about the equity curve we are observing
# 
# This brings us to the most important consideration about these (and any) performance statistics 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Stability in Forward Returns
# 
# Stability is a broader problem in the context of trading strategies and portfolio performance.
# 
# In the classroom, these statistics are assumed to come from a stationary distribution.  In reality, there is no distribution.
# 
# The implications are beyond academic, if you impose these measures on a strategy or portfolio you very well could just be overfitting to noise. 


import numpy as np
import plotly.graph_objects as go

# ==========================================
# --- 1. DATA GENERATION & SETUP ---
# ==========================================
np.random.seed(42)

n_years = 3
n_steps = 252 * n_years
dt = 1 / 252
t_steps = np.arange(n_steps + 1) * dt

live_start_idx = int(1.5 / dt)
crash_start_idx = int(2.25 / dt)

# Regime parameters (annualized)
mu_bt, sig_bt = 0.35, 0.15          # Backtest
mu_flat, sig_flat = 0.02, 0.15      # Live stagnation
mu_crash, sig_crash = -0.45, 0.25   # Live crash

risk_free_rate = 0.0

def generate_path():
    inc = np.zeros(n_steps)
    inc[:live_start_idx] = np.random.normal(
        mu_bt * dt, sig_bt * np.sqrt(dt), live_start_idx
    )
    inc[live_start_idx:crash_start_idx] = np.random.normal(
        mu_flat * dt, sig_flat * np.sqrt(dt), crash_start_idx - live_start_idx
    )
    inc[crash_start_idx:] = np.random.normal(
        mu_crash * dt, sig_crash * np.sqrt(dt), n_steps - crash_start_idx
    )

    path = np.zeros(n_steps + 1)
    path[1:] = np.cumsum(inc)
    return path, inc

path, incs = generate_path()

# Regime-wise expectation line (piecewise drift)
expected = np.zeros_like(t_steps)
for i in range(1, len(t_steps)):
    if i <= live_start_idx:
        expected[i] = expected[i - 1] + mu_bt * dt
    elif i <= crash_start_idx:
        expected[i] = expected[i - 1] + mu_flat * dt
    else:
        expected[i] = expected[i - 1] + mu_crash * dt

# Regime-wise volatility band (piecewise cumulative sqrt-time approximation)
vol_line = np.zeros_like(t_steps)
for i in range(1, len(t_steps)):
    if i <= live_start_idx:
        vol_line[i] = sig_bt * np.sqrt(i * dt)
    elif i <= crash_start_idx:
        bt_var = (sig_bt ** 2) * (live_start_idx * dt)
        live_var = (sig_flat ** 2) * ((i - live_start_idx) * dt)
        vol_line[i] = np.sqrt(bt_var + live_var)
    else:
        bt_var = (sig_bt ** 2) * (live_start_idx * dt)
        flat_var = (sig_flat ** 2) * ((crash_start_idx - live_start_idx) * dt)
        crash_var = (sig_crash ** 2) * ((i - crash_start_idx) * dt)
        vol_line[i] = np.sqrt(bt_var + flat_var + crash_var)

upper = expected + 2 * vol_line
lower = expected - 2 * vol_line

def calc_stats(s):
    if len(s) < 2:
        return 0.0, 0.0, 0.0, 0.0

    mu_ann = np.mean(s) * 252
    vol_ann = np.std(s) * np.sqrt(252)
    sharpe = mu_ann / vol_ann if vol_ann != 0 else 0.0

    downside = np.minimum(0, s - risk_free_rate)
    downside_dev = np.sqrt(np.mean(downside ** 2) * 252)
    sortino = mu_ann / downside_dev if downside_dev != 0 else np.inf

    return mu_ann, vol_ann, sharpe, sortino

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = 12
indices = list(range(1, len(t_steps), step_size))
if indices[-1] != len(t_steps) - 1:
    indices.append(len(t_steps) - 1)

for k in indices:
    frame_data = []

    bt_slice = incs[:min(k, live_start_idx)]
    lv_slice = incs[live_start_idx:k] if k > live_start_idx else np.array([])
    full_slice = incs[:k]

    m_bt, v_bt, s_bt, so_bt = calc_stats(bt_slice)
    m_lv, v_lv, s_lv, so_lv = calc_stats(lv_slice)
    m_full, v_full, s_full, so_full = calc_stats(full_slice)

    if k <= live_start_idx:
        current_color = "#00ffff"  # cyan
        status_text = "BACKTEST (IN-SAMPLE)"
        border_color = "#00ffff"
    elif k <= crash_start_idx:
        current_color = "#ffb347"  # orange-ish
        status_text = "LIVE TRADING (STAGNATION)"
        border_color = "#ffb347"
    else:
        current_color = "#ff3939"  # red
        status_text = "LIVE TRADING (CRASH)"
        border_color = "#ff3939"

    # expectation line
    frame_data.append(go.Scatter(
        x=t_steps[:k+1],
        y=expected[:k+1],
        mode="lines",
        line=dict(color="yellow", width=2, dash="dash"),
        showlegend=False
    ))

    # upper/lower bounds
    frame_data.append(go.Scatter(
        x=t_steps[:k+1],
        y=upper[:k+1],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"),
        showlegend=False
    ))
    frame_data.append(go.Scatter(
        x=t_steps[:k+1],
        y=lower[:k+1],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"),
        showlegend=False
    ))

    # single path
    frame_data.append(go.Scatter(
        x=t_steps[:k+1],
        y=path[:k+1],
        mode="lines",
        line=dict(color=current_color, width=3),
        showlegend=False
    ))

    frames.append(go.Frame(data=frame_data, name=f"f{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = go.Figure()

# Dummy traces must match frame order exactly
for _ in range(4):
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", showlegend=False))

fig.frames = frames

# Static axis limits
max_val = max(path.max(), upper.max())
min_val = min(path.min(), lower.min())
global_y_max = max_val + abs(max_val) * 0.15 + 0.05
global_y_min = min_val - abs(min_val) * 0.15 - 0.05

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step (Years): "},
    pad={"t": 0},
    x=0.15,
    len=0.85,
    y=-0.15,
    steps=[
        dict(
            method="animate",
            args=[[f"f{k}"], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0)
            )],
            label=str(round(t_steps[k], 2))
        )
        for k in indices
    ]
)]

fig.update_layout(
    height=700,
    width=1100,
    title_text="The Fallacy of Performance Metrics: Backtest vs. Live Reality",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=sliders,
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0,
        'y': -0.18,
        'xanchor': 'left',
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 80, 'redraw': True},
                'transition': {'duration': 0},
                'fromcurrent': True,
                'mode': 'immediate'
            }]
        }]
    }]
)

fig.update_xaxes(
    title_text='Time (Years)',
    range=[0, n_years + 0.05],
    gridcolor='rgba(128,128,128,0.2)',
    tickformat=".2f"
)

fig.update_yaxes(
    title_text='Cumulative Return',
    range=[global_y_min, global_y_max],
    gridcolor='rgba(128,128,128,0.2)',
    tickformat=".0%"
)

# ==========================================
# --- 4. STATIC VISUAL MARKERS / ANNOTATIONS ---
# ==========================================
fig.add_vline(
    x=1.5,
    line_width=2,
    line_dash="dash",
    line_color="white",
    opacity=0.7
)

fig.add_annotation(
    x=1.5,
    y=global_y_max - 0.08 * (global_y_max - global_y_min),
    text="LIVE TRADING BEGINS",
    showarrow=False,
    font=dict(color="white", size=11),
    bgcolor="rgba(0,0,0,0.5)"
)

# initial stats box
m_bt0, v_bt0, s_bt0, so_bt0 = calc_stats(incs[:min(indices[0], live_start_idx)])
m_lv0, v_lv0, s_lv0, so_lv0 = calc_stats(
    incs[live_start_idx:indices[0]] if indices[0] > live_start_idx else np.array([])
)
m_f0, v_f0, s_f0, so_f0 = calc_stats(incs[:indices[0]])

fig.add_annotation(
    text=(
        f"<b>BACKTEST (IN-SAMPLE)</b><br>"
        f"<b>Target Sharpe</b> = 2.75<br>"
        f"<b>BT Sharpe</b> = {s_bt0:.2f}<br>"
        f"<b>Live Sharpe</b> = {s_lv0:.2f}"
    ),
    xref="x domain",
    yref="y domain",
    x=0.05,
    y=0.95,
    showarrow=False,
    align="left",
    font=dict(color="#00ffff", size=14),
    bgcolor="rgba(0,0,0,0.6)",
    bordercolor="#00ffff",
    borderwidth=1,
    borderpad=8
)

# ==========================================
# --- 5. DYNAMIC ANNOTATION UPDATE VIA FRAMES ---
# ==========================================
for frame, k in zip(fig.frames, indices):
    bt_slice = incs[:min(k, live_start_idx)]
    lv_slice = incs[live_start_idx:k] if k > live_start_idx else np.array([])
    full_slice = incs[:k]

    _, _, s_bt, so_bt = calc_stats(bt_slice)
    _, _, s_lv, so_lv = calc_stats(lv_slice)
    _, _, s_full, so_full = calc_stats(full_slice)

    if k <= live_start_idx:
        status_text = "BACKTEST (IN-SAMPLE)"
        border_color = "#00ffff"
    elif k <= crash_start_idx:
        status_text = "LIVE TRADING (STAGNATION)"
        border_color = "#ffb347"
    else:
        status_text = "LIVE TRADING (CRASH)"
        border_color = "#ff3939"

    frame.layout = go.Layout(
        annotations=[
            dict(
                x=1.5,
                y=global_y_max - 0.08 * (global_y_max - global_y_min),
                xref="x",
                yref="y",
                text="LIVE TRADING BEGINS",
                showarrow=False,
                font=dict(color="white", size=11),
                bgcolor="rgba(0,0,0,0.5)"
            ),
            dict(
                x=0.05,
                y=0.95,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                align="left",
                font=dict(color=border_color, size=14),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=border_color,
                borderwidth=1,
                borderpad=8,
                text=(
                    f"<b>{status_text}</b><br>"
                    f"<b>Target Sharpe</b> = 2.75<br>"
                    f"<b>BT Sharpe</b> = {s_bt:.2f}<br>"
                    f"<b>Live Sharpe</b> = {s_lv:.2f}"
                )
            )
        ]
    )

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### What Causes Sharpe Instability
# 
# There is either a regime change (macro, crowding, etc...) or (more likely) you've overfit your strategy.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### The Regime Change or Structural Break
# 
# Strategies may work, stop working, and start working again as the space itself changes constantly
# 
# There may be a reason it stops working, maybe it was just noise or luck, sometimes we can answer why other times we can't


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# ==========================================
# --- 1. DATA GENERATION & REGIME SHIFT ---
# ==========================================
np.random.seed(42)

n_years = 3
n_days = 252 * n_years
t_steps = np.linspace(0, n_years, n_days)
dt = 1 / 252
split_idx = n_days // 2 

mu_1, sigma_1 = 0.35, 0.15
d_mu_1, d_sig_1 = mu_1 * dt, sigma_1 * np.sqrt(dt)

mu_2, sigma_2 = -0.15, 0.25
d_mu_2, d_sig_2 = mu_2 * dt, sigma_2 * np.sqrt(dt)

def generate_regime_path():
    shocks_1 = np.random.normal(0, 1, split_idx - 1)
    shocks_1 = (shocks_1 - np.mean(shocks_1)) / np.std(shocks_1)
    inc_1 = shocks_1 * d_sig_1 + d_mu_1
    
    shocks_2 = np.random.normal(0, 1, n_days - split_idx)
    shocks_2 = (shocks_2 - np.mean(shocks_2)) / np.std(shocks_2)
    inc_2 = shocks_2 * d_sig_2 + d_mu_2
        
    full_inc = np.concatenate(([0], inc_1, inc_2))
    path = np.cumsum(full_inc)
    return path, full_inc

path_p, inc_p = generate_regime_path()

x_pdf = np.linspace(-0.08, 0.08, 400)
pdf_1 = stats.norm.pdf(x_pdf, d_mu_1, d_sig_1)
pdf_2 = stats.norm.pdf(x_pdf, d_mu_2, d_sig_2)
max_pdf = max(pdf_1.max(), pdf_2.max())

# ==========================================
# --- 2. ANIMATION FRAMES & SLIDER STEPS ---
# ==========================================
frames = []
step_size = max(1, n_days // 100)
frame_indices = list(range(1, n_days, step_size))
if frame_indices[-1] != n_days - 1:
    frame_indices.append(n_days - 1)

slider_steps = []

for k in frame_indices:
    frame_data = []
    is_live = k >= split_idx
    
    # Equity Curve
    end_bt = min(k + 1, split_idx)
    frame_data.append(go.Scatter(x=t_steps[:end_bt], y=path_p[:end_bt], mode='lines', line=dict(color='#00ffff', width=3)))
    
    if is_live:
        frame_data.append(go.Scatter(x=t_steps[split_idx-1:k+1], y=path_p[split_idx-1:k+1], mode='lines', line=dict(color='#ff39b3', width=3)))
    else:
        frame_data.append(go.Scatter(x=[None], y=[None]))

    # Distribution
    frame_data.append(go.Scatter(x=x_pdf, y=pdf_1, mode='lines', fill='tozeroy', fillcolor='rgba(0,255,255,0.05)', line=dict(color='rgba(0,255,255,0.2)', width=1)))
    
    active_pdf = pdf_2 if is_live else pdf_1
    dist_color = '#ff39b3' if is_live else '#00ffff'
    fill_color = 'rgba(255,57,179,0.15)' if is_live else 'rgba(0,255,255,0.15)'
    
    frame_data.append(go.Scatter(x=x_pdf, y=active_pdf, mode='lines', fill='tozeroy', fillcolor=fill_color, line=dict(color=dist_color, width=2.5)))
    frame_data.append(go.Scatter(x=[inc_p[k], inc_p[k]], y=[0, max_pdf], mode='lines', line=dict(color='red', width=3)))

    frame_name = f"step{k}"
    frames.append(go.Frame(data=frame_data, name=frame_name))
    
    slider_steps.append({
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": f"{t_steps[k]:.2f}",
        "method": "animate"
    })

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], vertical_spacing=0.15, subplot_titles=["<b>Strategy Equity Curve</b>", "<b>Return Distribution Drift</b>"])
fig.add_trace(go.Scatter(x=[0], y=[0]), row=1, col=1)
fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1)
for _ in range(3): fig.add_trace(go.Scatter(x=[0], y=[0]), row=2, col=1)
fig.frames = frames

# ==========================================
# --- 4. LAYOUT: TRANSPARENT BUTTON & SLIDER ---
# ==========================================
# Align play button and slider on y-axis (both at y = -0.10)
fig.update_layout(
    height=850, width=1000,
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 14, "color": "white"}, "prefix": "Time Step (Years): ", "visible": True, "xanchor": "right"},
        "pad": {"b": 10, "t": 60},
        "len": 0.82, "x": 0.18, "y": -0.10,
        "steps": slider_steps
    }],
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.0, "y": -0.10,  # align Y with slider
        "xanchor": "left", "yanchor": "top",
        "direction": "left",
        "bgcolor": "rgba(0,0,0,0)", # Transparent background
        "font": {"color": "white", "size": 12}, 
        "buttons": [{
            "label": "▶ Play Simulation",
            "method": "animate",
            "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
        }],
        "bordercolor": "white", 
        "borderwidth": 1
    }]
)

# Axis & Annotation Config
fig.update_xaxes(title_text='Years', range=[0, n_years + 0.1], row=1, col=1)
fig.update_yaxes(title_text='Cumulative Return', range=[-0.2, 1.2], tickformat=".0%", row=1, col=1)
fig.add_shape(type="line", x0=1.5, y0=-0.2, x1=1.5, y1=1.2, line=dict(color="orange", width=2, dash="dash"), row=1, col=1)

# Corner Metrics Boxes
fig.add_annotation(text="<b>Backtest Metrics</b><br>Exp. Return: 35%<br>Sharpe: 2.3", xref="x domain", yref="y domain", x=0.02, y=0.95, showarrow=False, font=dict(color="#00ffff"), bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, row=1, col=1)
fig.add_annotation(text="<b>Live Metrics</b><br>Exp. Return: -15%<br>Sharpe: -0.6", xref="x domain", yref="y domain", x=0.98, y=0.95, showarrow=False, font=dict(color="#ff39b3"), bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff39b3", borderwidth=1, row=1, col=1)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Overfitting Noise
# 
# Of course, more likely, you're just overfitting noise in a backtest
# 
# What are the implications of this?


import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# --- Data setup ---
np.random.seed(42)
n = 30
X_base = np.linspace(-3, 3, n).reshape(-1, 1)
y_true_f = np.tanh(1.2 * X_base) + 0.2 * np.sin(2.5 * X_base)
y_base = y_true_f + 0.4 * np.random.randn(n, 1)

# --- Models ---
linreg = LinearRegression().fit(X_base, y_base)
pred_lin = lambda x: linreg.predict(x)

poly_over = PolynomialFeatures(degree=18)
overfit = Ridge(alpha=0.0, fit_intercept=False).fit(poly_over.fit_transform(X_base), y_base)
pred_overfit = lambda x: overfit.predict(poly_over.transform(x))

poly_rob = PolynomialFeatures(degree=3)
robust = Ridge(alpha=0.2).fit(poly_rob.fit_transform(X_base), y_base)
pred_robust = lambda x: robust.predict(poly_rob.transform(x))

colors = ['orange', 'red', 'limegreen']

# --- New points ---
np.random.seed(123)
n_new = 10
X_new = np.random.uniform(-3, 3, n_new).reshape(-1, 1)
y_true_new = np.tanh(1.2 * X_new) + 0.2 * np.sin(2.5 * X_new)
y_new = y_true_new + 0.4 * np.random.randn(n_new, 1)

# --- Model predictions ---
xs_r = np.linspace(-3, 3, 250).reshape(-1, 1)
y_pred_lin = pred_lin(xs_r).flatten()
y_pred_overfit = pred_overfit(xs_r).flatten()
y_pred_robust = pred_robust(xs_r).flatten()

ys_lin_new = pred_lin(X_new).flatten()
ys_overfit_new = pred_overfit(X_new).flatten()
ys_robust_new = pred_robust(X_new).flatten()

# --- Figure ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Underfit Model", "Overfit Model", "Robust Model"),
    horizontal_spacing=0.08
)

models = [
    ("Underfit", colors[0], y_pred_lin, ys_lin_new),
    ("Overfit", colors[1], y_pred_overfit, ys_overfit_new),
    ("Robust", colors[2], y_pred_robust, ys_robust_new)
]

# Static: training data + model curve
for i, (name, color, curve, _) in enumerate(models):
    fig.add_trace(
        go.Scatter(
            x=X_base.flatten(),
            y=y_base.flatten(),
            mode='markers',
            marker=dict(size=7, color='skyblue'),
            showlegend=False
        ),
        row=1,
        col=i+1
    )
    fig.add_trace(
        go.Scatter(
            x=xs_r.flatten(),
            y=curve,
            mode='lines',
            line=dict(color=color, width=4, dash='dot' if name == "Overfit" else None),
            name=name,
            showlegend=False
        ),
        row=1,
        col=i+1
    )
    # Dynamic placeholders for OOS points and error bars
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(size=14, color='gold', line=dict(width=2, color='black')),
            showlegend=False
        ),
        row=1,
        col=i+1
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1,
        col=i+1
    )

# Trace index mapping for dynamic content
# (Underfit: 2–3, Overfit: 6–7, Robust: 10–11)
dyn_idx = [(2, 3), (6, 7), (10, 11)]

# --- Frames ---
frames = []
for k in range(1, n_new + 1):
    frame_data = []
    for (name, color, _, ys_pred), (pt_idx, err_idx) in zip(models, dyn_idx):
        # points up to k
        frame_data.append(dict(
            type="scatter",
            x=X_new[:k, 0],
            y=y_new[:k, 0],
            mode="markers",
            marker=dict(size=14, color='gold', line=dict(width=2, color='black')),
        ))
        # error lines up to k
        xerr, yerr = [], []
        for j in range(k):
            xerr += [X_new[j, 0], X_new[j, 0], None]
            yerr += [ys_pred[j], y_new[j, 0], None]
        frame_data.append(dict(
            type="scatter",
            x=xerr, y=yerr,
            mode="lines",
            line=dict(color='red', width=2, dash='dash'),
        ))

    mse_lin = mean_squared_error(y_new[:k], ys_lin_new[:k])
    mse_overfit = mean_squared_error(y_new[:k], ys_overfit_new[:k])
    mse_robust = mean_squared_error(y_new[:k], ys_robust_new[:k])
    title = (f"Out-of-sample Points Added: <b>{k}/{n_new}</b> | "
             f"<span style='color:{colors[0]}'>Underfit MSE: {mse_lin:.3f}</span> &nbsp; "
             f"<span style='color:{colors[1]}'>Overfit MSE: {mse_overfit:.3f}</span> &nbsp; "
             f"<span style='color:{colors[2]}'>Robust MSE: <b>{mse_robust:.3f}</b></span>")

    frames.append(go.Frame(name=f"frame{k}", data=frame_data,
                           traces=[2, 3, 6, 7, 10, 11],
                           layout=go.Layout(title=dict(text=title))))

# --- Layout with updated button style ---
fig.update_layout(
    width=1150,
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title=dict(text="Out-of-sample Points Added: <b>0</b>", y=0.95),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.0,
        "y": -0.25,  # LOWER the button row further than previous (was -0.16)
        "xanchor": "left",
        "yanchor": "top",
        "direction": "left",
        "bgcolor": "rgba(0,0,0,0)",  # Transparent background for match
        "font": {"color": "white", "size": 13},
        "buttons": [
            {
                "label": "▶ Play Simulation",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 700, "redraw": True},
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                    "mode": "immediate"
                }]
            },
            {
                "label": "⟲ Reset",
                "method": "animate",
                "args": [["frame1"], {
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                    "mode": "immediate"
                }]
            }
        ],
        "bordercolor": "white",
        "borderwidth": 1
    }],
    showlegend=False
)

# Make gridline color consistent with previous plot (#323232)
for i in range(1, 4):
    fig.update_xaxes(title_text="Input x", row=1, col=i,
                     showgrid=True, gridcolor="#323232", gridwidth=1)
    fig.update_yaxes(title_text="Target y", row=1, col=i,
                     showgrid=True, gridcolor="#323232", gridwidth=1)

fig.frames = frames
fig.show()



# This is why trading and portfolio construction are about **performance**, not overfitting performance metrics.
# 
# Live strategy performance (*actual decisions made in the face of uncertainty*) are worth 1,000 backtests...
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# Think of this like a batting average, if someone had a good batting average years ago, does that mean they're good in real-time today?
# 
# **Maybe**
# 
# Moreover, what if they had a killer batting average last season or even last week, does that mean they're good in real-time today?
# 
# **Maybe**
# 
# It all comes down to performance, model informed decision making, knowledge and experience.


# ---


# #### 3.) 💭 Closing Thoughts and Future Topics
# 
#  **📑 TL;DW Executive Summary**
# 
# * **Statistics as a Compression:** Realized performance metrics like the Sharpe Ratio are merely a "compression" of a single observed sample path. Infinite potential paths can produce the exact same realized statistics, but we only get to walk one.
# * **The Reality of Volatility Drag:** Variance is not just a risk metric; it is a mathematical "drag" on compounded wealth. Even with identical arithmetic returns, higher volatility erodes long-term geometric growth:
#     $$\text{Geo Mean} \approx \mu - \frac{1}{2}\sigma^2$$
# * **Sharpe vs. Sortino:** The Sharpe Ratio penalizes "good" volatility (upside deviations) from the mean. If your strategy's variance is driven by large positive jumps, the **Sortino Ratio** is a superior diagnostic tool as it only penalizes downside semi-variance $\text{Var}^-[\pi]$.
# * **The Fallacy of Stationarity:** Traditional finance often assumes stationary distributions. In reality, markets undergo **Regime Changes** and **Structural Breaks**, which can turn a "perfect" backtest into a "crash" the moment it goes live.
# * **Overfitting Noise:** Chasing a high Sharpe Ratio in a backtest often results in overfitting to historical noise rather than capturing a repeatable signal. A robust, lower-Sharpe model often outperforms a "perfect" overfit model out-of-sample.
# * **Performance > Metrics:** Live strategy performance—actual decisions made in the face of true uncertainty, is worth 1,000 backtests. Quantitative finance is about robust decision-making and experience, not just optimizing historical ratios.
# 
#  
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
#  - Fama-French / Carhart and Factor Modeling in General
#  - Hawkes Processes
#  - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
#  - Market-Making Models and Simulation (Stoikov-Avellaneda)
#  - My First Year as a Quant
#  - Why Hedge Funds are Actually Secretive
#  - Non-Markovian Models (fractional Brownian motion, Volterra Process)
#  - Top 3 Uses of Linear Algebra for Quant Finance
#  - Girsanov's Change of Measure
#  - Rough Path Theory, Applications of Path Signatures
#  - Sig-Vol Model, Calibration, and Pricing
#  - Trading with Alternative Data Sources
#  - Pairs Trading and Statistical Arbitrage
#  - Data Cleaning & Outlier Handling in Financial Time Series
#  - Practical Issues in Multi-Asset Portfolio Backtesting
#  - Risk Premia Harvesting: Equity, FX, Rates
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How Interactive Broker's API Works (EWrapper/EClient)
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

