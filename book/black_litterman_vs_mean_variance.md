### 📈 Black-Litterman vs. Mean-Variance Portfolio Optimization in Python

##### ▶️ Related Quant Guild Videos:

- [The 5 Papers That Built Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)

- [I Bet You've Never Found Alpha (and I Can Prove It)](https://youtu.be/UzTJHs3-eT0)

- [Quant Ranks Retail Trading Mistakes that Blow Up Your Account](https://youtu.be/1mpNxBaBeOw)

- [Non-Stationarity and Why Market Timing Fails](https://youtu.be/7nvjrgqKjJE)

- [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)

- [How a Quant Manages a Portfolio](https://youtu.be/mZLNzqDZHbA)

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

#### 1. ) 🌊 Mean-Variance Optimization

- Portfolio of Risky Assets

- The Efficient Frontier

- Tangency Portfolio and CAPM

#### 2.) 📉 Downfalls of In-Sample Optimization

- Requirements for MVO

- Reality of Expected Returns and Covariance Matrix

- Frequentist vs. Bayesian Approach

#### 3.) 📈 Black-Litterman

- Market Equilibrium Produces Objective Consensus

- Subjective Tilting for Injecting Views

- Decision Making as Performance and Models as a Filter

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1. ) 🌊 Mean-Variance Optimization

##### Risky Assets and the Efficient Frontier

Modern portfolio theory studies how to combine risky assets to achieve the best balance between expected return and risk. 

Mean-variance optimization (MVO), introduced by Markowitz, answers the question: given your estimates of expected returns and covariances, how should you allocate across $N$ risky assets to maximize expected return for a chosen level of risk, or equivalently, minimize risk for a target return?

By constructing portfolios from only risky assets, MVO traces out the efficient frontier—the set of optimal portfolios that offer the highest expected return for each given level of risk

 
 **Risky assets typically included in the efficient frontier:**
 
 - Equities (e.g., individual stocks, equity indices)
 - Government bonds (short-term and long-term)
 - Corporate bonds
 - Commodities (e.g., gold, oil, agricultural products)
 - Real estate investment trusts (REITs)
 - Currencies (foreign exchange positions)
 - Alternative assets (private equity, hedge funds, etc.)
 
 (The actual set of risky assets depends on the universe considered in your optimization. In many academic and practical examples, only a subset—such as large cap equities or developed market stocks—is used for illustration.)


##### Efficient Frontier of Risky Assets


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA GENERATION & SIMULATION ---
# ==========================================
np.random.seed(42)

# Define our "Universe" of Risky Assets
assets = ['Tech', 'Health', 'Finance', 'Energy', 'Consumer', 'Bonds', 'Gold']
n_assets = len(assets)

# Theoretical Expected Annual Returns and Volatilities
expected_returns = np.array([0.14, 0.10, 0.11, 0.08, 0.09, 0.04, 0.06])
volatilities = np.array([0.22, 0.16, 0.19, 0.25, 0.14, 0.06, 0.15])

# Create a realistic correlation matrix (Bonds/Gold have low correlation to Equities)
corr_matrix = np.array([
    [1.00, 0.60, 0.65, 0.40, 0.70, -0.10, 0.05],
    [0.60, 1.00, 0.55, 0.30, 0.65, -0.05, 0.10],
    [0.65, 0.55, 1.00, 0.50, 0.60, -0.05, 0.05],
    [0.40, 0.30, 0.50, 1.00, 0.40, -0.15, 0.25],
    [0.70, 0.65, 0.60, 0.40, 1.00, 0.00, 0.10],
    [-0.10,-0.05,-0.05,-0.15, 0.00, 1.00, 0.30],
    [0.05, 0.10, 0.05, 0.25, 0.10, 0.30, 1.00]
])

# Covariance Matrix: diag(vol) * corr * diag(vol)
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# Simulate Price Paths (Geometric Brownian Motion)
n_days = 252
dt = 1/252
daily_returns = np.random.multivariate_normal(expected_returns * dt, cov_matrix * dt, n_days)
price_paths = np.vstack([np.ones(n_assets), np.cumprod(1 + daily_returns, axis=0)])
dates_array = pd.date_range(start='2024-01-01', periods=n_days+1, freq='B')

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 3000

# Generate Random Portfolios for the Scatter Cloud
random_weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
port_returns = random_weights @ expected_returns
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

# Sort by Sharpe so higher Sharpe points are drawn last (on top)
sort_idx = np.argsort(port_sharpes)
port_returns = port_returns[sort_idx]
port_vols = port_vols[sort_idx]
port_sharpes = port_sharpes[sort_idx]
random_weights = random_weights[sort_idx]

# Find Tangency Portfolio (Max Sharpe)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret = port_returns[max_sharpe_idx]
opt_vol = port_vols[max_sharpe_idx]
opt_weights = random_weights[max_sharpe_idx]

# Calculate the True Efficient Frontier Line (Upper boundary)
target_returns = np.linspace(port_returns.min(), port_returns.max(), 50)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
    ef_vols.append(res.fun)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = max(1, (n_days + 1) // 100) # Roughly 100 frames for smooth playback
frame_indices = list(range(1, n_days + 1, step_size))
if frame_indices[-1] != n_days:
    frame_indices.append(n_days)

# Colors for the asset paths
path_colors = ['rgba(255,255,255,0.6)', 'rgba(0,255,255,0.6)', 'rgba(255,51,51,0.6)', 
               'rgba(255,255,0,0.6)', 'rgba(153,51,255,0.6)', 'rgba(57,255,20,0.6)', 'rgba(255,153,51,0.6)']

for k in frame_indices:
    t_x = dates_array[:k]
    frame_data = []
    
    # 1. Left Chart: 7 Asset Price Paths
    for i in range(n_assets):
        frame_data.append(go.Scatter(x=t_x, y=price_paths[:k, i], mode='lines', 
                                     line=dict(color=path_colors[i], width=1.5), name=assets[i]))
    
    # Reveal logic for the right side based on time progression
    progress = k / n_days
    num_ports_to_show = int(n_portfolios * progress)
    ef_points_to_show = int(len(target_returns) * progress)
    
    # 2. Right Chart (Top): Random Portfolios Scatter
    frame_data.append(go.Scatter(
        x=port_vols[:num_ports_to_show], y=port_returns[:num_ports_to_show], mode='markers',
        marker=dict(color=port_sharpes[:num_ports_to_show], colorscale='Viridis', 
                    size=4, opacity=0.7, showscale=False), name='Portfolios'
    ))
    
    # 3. Right Chart (Top): True Efficient Frontier Line
    frame_data.append(go.Scatter(
        x=ef_vols[:ef_points_to_show], y=target_returns[:ef_points_to_show], mode='lines',
        line=dict(color='#00ffff', width=3, dash='dash'), name='Efficient Frontier'
    ))
    
    # 4. Right Chart (Top): Tangency Point (Only reveal near the end)
    show_tangency = progress > 0.95
    frame_data.append(go.Scatter(
        x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
        mode='markers', marker=dict(color='#39ff14', size=14, symbol='star', line=dict(color='white', width=1)), 
        name='Tangency Portfolio'
    ))
    
    # 5. Right Chart (Bottom): Tangency Weights Bar Chart (Grow with progress)
    frame_data.append(go.Bar(
        x=assets, y=opt_weights * (progress if not show_tangency else 1.0), 
        marker_color='#39ff14', name='Weights'
    ))

    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        "Simulated Asset Price Paths (Growth of $1)", 
        "Markowitz Bullet: Risk vs. Expected Return",
        "Tangency Portfolio Capital Allocation"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

# Add Dummy Traces (MUST perfectly match the 11 traces in the frame loop)
# 7 Asset lines
for i in range(n_assets):
    fig.add_trace(go.Scatter(x=[dates_array[0]], y=[price_paths[0, i]]), row=1, col=1)
# 3 EF Scatter traces (Portfolios, Frontier Line, Tangency Point)
for _ in range(3):
    fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=2)
# 1 Bar chart for weights
fig.add_trace(go.Bar(x=assets, y=[0]*n_assets), row=2, col=2)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": ""}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(dates_array[k].date())
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=700, width=1100,
    title_text="Modern Portfolio Theory: Building the True Efficient Frontier",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits so charts don't bounce during animation
fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Normalized Price', range=[0.6, price_paths.max()*1.05], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, port_vols.max()*1.1], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return (μ)', range=[0, port_returns.max()*1.1], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Asset Class', row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Weight (%)', range=[0, opt_weights.max()*1.2], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# --- ADD ANNOTATION OVERLAYS ---
max_sharpe_val = port_sharpes.max()
fig.add_annotation(
    text=f"<b>Max Achievable Sharpe: {max_sharpe_val:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=2
)

fig.show()
```

###### ______________________________________________________________________________________________________________________________________

##### Roll's Critique

Just like expected returns (we'll get there), we can't actually observe the market portfolio

The best we can do is proxy for it using different subsets of risky assets

Consider the S&P 500 a common proxy for the U.S. Equity Efficient Frontier relative to a Multi-Asset Efficient Frontier




```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA DEFINITION (The Two Universes) ---
# ==========================================
np.random.seed(42)

# Universe 1: Equities (High Correlation to each other)
eq_assets = ['Tech', 'Health', 'Finance', 'Consumer']
eq_ret = np.array([0.14, 0.10, 0.11, 0.09])
eq_vol = np.array([0.22, 0.16, 0.19, 0.14])

# Universe 2 Additions: Non-Equities (Low/Negative Correlation to Equities)
non_eq_assets = ['Govt Bonds', 'Corp Bonds', 'Gold']
non_eq_ret = np.array([0.04, 0.06, 0.07])
non_eq_vol = np.array([0.06, 0.09, 0.15])

# Combined "Multi-Asset" Universe
multi_assets = eq_assets + non_eq_assets
multi_ret = np.concatenate((eq_ret, non_eq_ret))
multi_vol = np.concatenate((eq_vol, non_eq_vol))
n_eq = len(eq_assets)
n_multi = len(multi_assets)

# Master Correlation Matrix (Equities highly correlated, Bonds/Gold provide diversification)
corr_matrix = np.array([
    #Tech Health Fin   Cons  Govt  Corp  Gold
    [1.00, 0.65, 0.70, 0.75, -0.15, 0.20, 0.05], # Tech
    [0.65, 1.00, 0.60, 0.70, -0.05, 0.25, 0.10], # Health
    [0.70, 0.60, 1.00, 0.65, -0.10, 0.30, 0.05], # Finance
    [0.75, 0.70, 0.65, 1.00,  0.00, 0.25, 0.15], # Consumer
    [-0.15,-0.05,-0.10, 0.00, 1.00, 0.40, 0.25], # Govt Bonds
    [0.20, 0.25, 0.30, 0.25,  0.40, 1.00, 0.15], # Corp Bonds
    [0.05, 0.10, 0.05, 0.15,  0.25, 0.15, 1.00]  # Gold
])

# Covariance Matrices
cov_multi = np.outer(multi_vol, multi_vol) * corr_matrix
cov_eq = cov_multi[:n_eq, :n_eq] # Slice just the equities portion

risk_free_rate = 0.02
n_portfolios = 2500

# ==========================================
# --- 2. MONTE CARLO & OPTIMIZATION ---
# ==========================================

def simulate_universe(n_assets, expected_returns, cov_matrix):
    weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    returns = weights @ expected_returns
    vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
    sharpes = (returns - risk_free_rate) / vols
    
    # Sort by Volatility for smooth left-to-right animation drawing
    sort_idx = np.argsort(vols)
    return weights[sort_idx], returns[sort_idx], vols[sort_idx], sharpes[sort_idx]

# Simulate both universes
eq_w, eq_r, eq_v, eq_s = simulate_universe(n_eq, eq_ret, cov_eq)
multi_w, multi_r, multi_v, multi_s = simulate_universe(n_multi, multi_ret, cov_multi)

# Find Tangency Portfolios
eq_max_idx = np.argmax(eq_s)
eq_tan_w, eq_tan_r, eq_tan_v = eq_w[eq_max_idx], eq_r[eq_max_idx], eq_v[eq_max_idx]

multi_max_idx = np.argmax(multi_s)
multi_tan_w, multi_tan_r, multi_tan_v = multi_w[multi_max_idx], multi_r[multi_max_idx], multi_v[multi_max_idx]

# Calculate True Frontier Lines (Upper Boundaries)
def get_frontier(expected_returns, cov_matrix, n_assets, target_returns):
    vols = []
    for tr in target_returns:
        res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), np.ones(n_assets)/n_assets, 
                           method='SLSQP', bounds=[(0, 1) for _ in range(n_assets)],
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                        {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
        vols.append(res.fun)
    return np.array(vols)

eq_targets = np.linspace(eq_r.min(), eq_r.max(), 40)
eq_frontier_vols = get_frontier(eq_ret, cov_eq, n_eq, eq_targets)

multi_targets = np.linspace(multi_r.min(), multi_r.max(), 40)
multi_frontier_vols = get_frontier(multi_ret, cov_multi, n_multi, multi_targets)

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    
    # Phase timing logic
    eq_progress = min(1.0, progress * 2.5)       # Finishes at frame 40
    multi_progress = min(1.0, max(0, progress - 0.4) * 2.5) # Starts at frame 40, finishes at 80
    
    # Independent reveal logic for the tangency stars
    reveal_eq_star = eq_progress >= 1.0
    reveal_multi_star = multi_progress >= 1.0
    
    eq_pts = int(n_portfolios * eq_progress)
    eq_line_pts = int(len(eq_targets) * eq_progress)
    
    multi_pts = int(n_portfolios * multi_progress)
    multi_line_pts = int(len(multi_targets) * multi_progress)

    frame_data = [
        # 0. Equities Scatter
        go.Scatter(x=eq_v[:eq_pts], y=eq_r[:eq_pts], mode='markers', marker=dict(color='#00E5FF', size=3, opacity=0.4), name='Equity Portfolios'),
        # 1. Equities Frontier Line
        go.Scatter(x=eq_frontier_vols[:eq_line_pts], y=eq_targets[:eq_line_pts], mode='lines', line=dict(color='#00E5FF', width=4), name='Equity EF'),
        # 2. Multi-Asset Scatter
        go.Scatter(x=multi_v[:multi_pts], y=multi_r[:multi_pts], mode='markers', marker=dict(color='#39ff14', size=3, opacity=0.4), name='Multi-Asset Portfolios'),
        # 3. Multi-Asset Frontier Line
        go.Scatter(x=multi_frontier_vols[:multi_line_pts], y=multi_targets[:multi_line_pts], mode='lines', line=dict(color='#39ff14', width=4), name='Multi-Asset EF'),
        
        # 4. Equities Tangency Point (Reveals instantly when eq_progress hits 1.0)
        go.Scatter(x=[eq_tan_v] if reveal_eq_star else [None], y=[eq_tan_r] if reveal_eq_star else [None], 
                   mode='markers', marker=dict(color='#00E5FF', size=16, symbol='star', line=dict(color='white', width=1)), name='Eq Tangency'),
        # 5. Multi-Asset Tangency Point (Reveals instantly when multi_progress hits 1.0)
        go.Scatter(x=[multi_tan_v] if reveal_multi_star else [None], y=[multi_tan_r] if reveal_multi_star else [None], 
                   mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)), name='Multi Tangency'),
        
        # 6. Multi-Asset Bar Chart (Bottom Left) -> Scales up with multi_progress
        go.Bar(x=multi_assets, y=multi_tan_w * multi_progress, marker_color='#39ff14', name='Multi Weights'),
        # 7. Equities Bar Chart (Bottom Right) -> Scales up with eq_progress
        go.Bar(x=eq_assets, y=eq_tan_w * eq_progress, marker_color='#00E5FF', name='Eq Weights')
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION & LAYOUT ---
# ==========================================
# 2 Rows, Top Row spans both columns!
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"colspan": 2}, None], 
           [{}, {}]],
    subplot_titles=[
        "The Expansion of the Efficient Frontier (Equities vs. Multi-Asset)", 
        "Multi-Asset Tangency Portfolio",
        "Equities Tangency Portfolio"
    ],
    vertical_spacing=0.15
)

# Add 8 Dummy Traces to match frame architecture precisely
for _ in range(6): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1)
# Add Multi-Asset Bar to Row 2, Col 1 (Bottom Left)
fig.add_trace(go.Bar(x=multi_assets, y=[0]*n_multi), row=2, col=1)
# Add Equities Bar to Row 2, Col 2 (Bottom Right)
fig.add_trace(go.Bar(x=eq_assets, y=[0]*n_eq), row=2, col=2)

fig.frames = frames

# Animation Sliders & Play Button
sliders = [dict(
    active=0, currentvalue={"prefix": "Simulation Phase: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(i)) for i, f in enumerate(frames)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Portfolio Optimization: The Mathematical Power of Diversification",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Lock Axes to prevent bouncing
max_vol = multi_v.max() * 1.05
max_ret = multi_r.max() * 1.1

# Top Chart (Frontiers)
fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, max_vol], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return (μ)', range=[0, max_ret], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

# Bottom Left Chart (Multi-Asset Bars)
fig.update_yaxes(title_text='Weight (%)', range=[0, 1.0], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

# Bottom Right Chart (Equities Bars)
fig.update_yaxes(title_text='Weight (%)', range=[0, 1.0], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# Sharpe Annotations
fig.add_annotation(
    text=f"<b>Equities Max Sharpe: {eq_s.max():.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#00E5FF", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00E5FF", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.add_annotation(
    text=f"<b>Multi-Asset Max Sharpe: {multi_s.max():.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.88,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.show()
```

In the context of risky assets we have the following set relationship

$$\text{Equities}\subset\text{Multi-Asset}\subset\text{Market Portfolio}$$

The more risky assets we have with diversification benefits (lack of correlation) the further the benefit to our portfolio

Geometrically, as seen above, any subset of risky assets is (theoretically) contained within the true Efficient Frontier

This is a really big deal, we want to optimize across *all risky assets*, right?  

If we can acheive better risk adjusted returns why wouldn't we want to include other assets?

Well, in this context there is a larger issue than the true (unobservable) market portfolio

That is the issue in the context of expected returns...

###### ______________________________________________________________________________________________________________________________________


##### Side Note, Should We Always Diversify?

**CAPM** says the market factor explains 100% of return variation, if this is true we should always diversify

In practice this isn't the case, it only explains roughly 70-80% of variation and the remaining are *other hidden factors*
- Fama-French win a Nobel prize for this, among their other incredible contributions

Modern L/S strategies have net zero market exposure and are just a wrapped alternative beta exposure

**My Personal Take:**
- Diversification for *investing* that is wealth preservation (*you can be passive*)
- Concentration for scaling wealth (*you have to be active and right*)

I like Buffet's take on this *"Diversification is protection against ignorance.  It makes little sense if you know what you are doing."*

That being said: **MANY DO NOT HAVE ANY IDEA WHAT THEY ARE DOING AND SHOULD JUST DIVERSIFY**


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & SIMULATION ---
# ==========================================
np.random.seed(42) # Set seed for reproducible paths

total_years = 10
days_per_year = 252
total_days = total_years * days_per_year
dt = 1 / days_per_year

t_days = np.arange(total_days + 1)
t_years = t_days / days_per_year
n_paths = 10

# --- Path 1: GBM (8% Expected Return, 10% Volatility) ---
mu_1 = 0.08
sigma_1 = 0.10
Z_1 = np.random.normal(0, 1, (total_days, n_paths))
# GBM Formula applied across 10 paths
returns_1 = (mu_1 - 0.5 * sigma_1**2) * dt + sigma_1 * np.sqrt(dt) * Z_1
paths_1 = np.vstack([np.ones(n_paths), np.exp(np.cumsum(returns_1, axis=0))])
# Calculate the cross-sectional mean path
mean_path_1 = np.mean(paths_1, axis=1)

# --- Path 2: GBM (20% Expected Return, 30% Volatility) ---
mu_2 = 0.20
sigma_2 = 0.30
Z_2 = np.random.normal(0, 1, (total_days, n_paths))
# GBM Formula applied across 10 paths
returns_2 = (mu_2 - 0.5 * sigma_2**2) * dt + sigma_2 * np.sqrt(dt) * Z_2
paths_2 = np.vstack([np.ones(n_paths), np.exp(np.cumsum(returns_2, axis=0))])
# Calculate the cross-sectional mean path
mean_path_2 = np.mean(paths_2, axis=1)

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

# Animate in roughly 100 steps (every ~25 trading days) to keep it smooth and fast
step_size = max(1, total_days // 100)
frame_indices = list(range(1, total_days + 1, step_size))
if frame_indices[-1] != total_days:
    frame_indices.append(total_days)

for k in frame_indices:
    current_years = t_years[:k+1]
    frame_data = []
    
    # 1. Left Chart: 10 Faint Paths + 1 Mean Path
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=current_years, y=paths_1[:k+1, p], 
            mode='lines', line=dict(color='#00E5FF', width=1), opacity=0.5,
            showlegend=False, hoverinfo='skip'
        ))
    frame_data.append(go.Scatter(
        x=current_years, y=mean_path_1[:k+1], 
        mode='lines', line=dict(color='#ffffff', width=4),
        name='Mean Path 1'
    ))

    # 2. Right Chart: 10 Faint Paths + 1 Mean Path
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=current_years, y=paths_2[:k+1, p], 
            mode='lines', line=dict(color='#39ff14', width=1), opacity=0.5,
            showlegend=False, hoverinfo='skip'
        ))
    frame_data.append(go.Scatter(
        x=current_years, y=mean_path_2[:k+1], 
        mode='lines', line=dict(color='#ffffff', width=4),
        name='Mean Path 2'
    ))

    frames.append(go.Frame(data=frame_data, name=f"day{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=[
        "Path 1: Expected Return 8%, Volatility 10%", 
        "Path 2: Expected Return 20%, Volatility 30%"
    ],
    horizontal_spacing=0.1
)

# Add Dummy Traces (MUST perfectly match the trace structure in the frame loop: 11 left, 11 right)
for _ in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=1) # Left Mean

for _ in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=2) # Right Mean

fig.frames = frames

# ==========================================
# --- 4. LAYOUT & AESTHETICS ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": ""}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f"{t_years[k]:.1f}"
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=700, width=1100,
    title_text="Geometric Brownian Motion: The Drag of High Volatility (10 Sample Paths + Mean)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits (0 to 10 on both charts)
fig.update_xaxes(title_text='Years', range=[0, total_years], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Growth of $1.00', range=[0, 10], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

fig.update_xaxes(title_text='Years', range=[0, total_years], gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(title_text='Growth of $1.00', range=[0, 10], gridcolor='rgba(128,128,128,0.2)', row=1, col=2)

# --- ADD FINAL VALUE ANNOTATIONS ---
# Calculate the realized geometric compound rate of the MEAN path
geom_1 = ((mean_path_1[-1])**(1/total_years) - 1) * 100
fig.add_annotation(
    text=f"<b>Mean Final Value: ${mean_path_1[-1]:.2f}</b><br>Realized Mean CAGR: {geom_1:.2f}%<br>(Target Return: {mu_1*100:.1f}%)",
    xref="x domain", yref="y domain", x=0.02, y=0.95,
    showarrow=False, align="left", font=dict(color="#ffffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00E5FF", borderwidth=1, borderpad=6,
    row=1, col=1
)

geom_2 = ((mean_path_2[-1])**(1/total_years) - 1) * 100
fig.add_annotation(
    text=f"<b>Mean Final Value: ${mean_path_2[-1]:.2f}</b><br>Realized Mean CAGR: {geom_2:.2f}%<br>(Target Return: {mu_2*100:.1f}%)",
    xref="x domain", yref="y domain", x=0.02, y=0.95,
    showarrow=False, align="left", font=dict(color="#ffffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=2
)

fig.show()
```

As investment professionals our objective is (literally) to estimate expected returns

If you think you've found a severely undervalued company, congrats, you've estimated positive expected returns

If you are good at this, like you see above, why would you diversify?

*Critical Note:* **You are probably not good at this**

**If you are wrong you will eat volatility drag, run into ergodicity problems, and stop out before compounding wealth.**

###### ______________________________________________________________________________________________________________________________________

 Volatility drag: Wealth compounds geometrically  
 For annualized return $r$ and volatility $v$ (in decimals),  
 geometric growth rate $g = r - \frac{1}{2}v^2$
 
 **Example 1: $r=8\%$, $v=10\%$**

 $g = 0.08 - \frac{1}{2}(0.1)^2 = 0.08 - 0.005 = \boxed{0.075} = 7.5\%$
 
 **Example 2: $r=20\%$, $v=30\%$**

 $g = 0.20 - \frac{1}{2}(0.3)^2 = 0.20 - 0.045 = \boxed{0.155} = 15.5\%$
 
 $\rightarrow$ Higher volatility reduces compounding growth, even if the average return is high.

**In other words, if you're wrong and experience mad volatility you are in for a world of hurt...**

###### ______________________________________________________________________________________________________________________________________

In my previous video I refer to the lottery effect and the bewildering negative cross-sectional return effect of holding idiosyncratic risk.  
*- Like I said, most people are not good at this, and the literature corroborates this.*

Given many professionals are in the portfolio management space from a wealth preservation perspective, let's return to the Efficient Frontier to inform our holdings 

###### ______________________________________________________________________________________________________________________________________

##### Tangency Portfolio, CAPM, Mean-Variance Optimization (MVO)

Let's get back to the Efficient Frontier focusing on the Equity Frontier and the bigger structural issue with this approach

We pick the portfolio that maximizes our risk adjusted return, our *Sharpe ratio*

**Steps to Mean-Variance Optimization (the Markowitz MVP):** 
 
 1. **Input:** Annualized expected returns $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.
 
 2. **Portfolio Statistics:**
    - **Expected Portfolio Return:**
      $$
      \mathbb{E}[R_p] = \mathbf{w}^\top \boldsymbol{\mu}
      $$
    - **Portfolio Volatility:**
      $$
      \sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}
      $$
 
 3. **Sharpe Ratio:**
    $$
    \text{Sharpe Ratio} = \frac{\mathbb{E}[R_p] - r_f}{\sigma_p}
    $$
    where $r_f$ is the risk-free rate.
 
 4. **Optimize:** Find weights $\mathbf{w}$ that maximize the Sharpe ratio — this is the tangency portfolio:
    $$
    \mathbf{w}^* = \arg\max_\mathbf{w} \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}
    $$
 
 5. **(Optional):** Vary target return to trace the efficient frontier.


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & PROCESSING ---
# ==========================================
np.random.seed(42)

# Load empirical data
df = pd.read_csv('efficient_frontier_daily_returns.csv')

# Ensure date is a datetime object
df['date'] = pd.to_datetime(df['date'])

# Pivot the table to get a matrix of daily returns: Rows = Dates, Columns = Symbols
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# Annualize historical expected returns and covariance matrix (assuming 252 trading days)
expected_returns = df_ret.mean().values * 252
cov_matrix = df_ret.cov().values * 252

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 6000 # Increased for a dense, beautiful cloud

# THE FIX: Using a Dirichlet alpha of 0.15 instead of 1.0. 
# This forces the random number generator to create more concentrated portfolios,
# allowing the scatter cloud to naturally expand and "hug" the true efficient frontier.
random_weights = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

port_returns = random_weights @ expected_returns
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

# Sort by Sharpe so higher Sharpe points are drawn last (on top)
sort_idx = np.argsort(port_sharpes)
port_returns = port_returns[sort_idx]
port_vols = port_vols[sort_idx]
port_sharpes = port_sharpes[sort_idx]
random_weights = random_weights[sort_idx]

# Find Tangency Portfolio (Max Sharpe)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret = port_returns[max_sharpe_idx]
opt_vol = port_vols[max_sharpe_idx]
opt_weights = random_weights[max_sharpe_idx]

# Calculate the True Efficient Frontier Line (Upper boundary)
target_returns = np.linspace(port_returns.min(), port_returns.max() * 1.05, 60)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
    ef_vols.append(res.fun if res.success else np.nan)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    
    num_ports_to_show = int(n_portfolios * progress)
    ef_points_to_show = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # 1. Top Chart: Random Portfolios Scatter
        go.Scatter(
            x=port_vols[:num_ports_to_show], y=port_returns[:num_ports_to_show], mode='markers',
            marker=dict(color=port_sharpes[:num_ports_to_show], colorscale='Viridis', 
                        size=4, opacity=0.7, showscale=False), name='Portfolios'
        ),
        # 2. Top Chart: True Efficient Frontier Line
        go.Scatter(
            x=ef_vols[:ef_points_to_show], y=target_returns[:ef_points_to_show], mode='lines',
            line=dict(color='#00ffff', width=3, dash='dash'), name='Efficient Frontier'
        ),
        # 3. Top Chart: Tangency Point
        go.Scatter(
            x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
            mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)), 
            name='Tangency Portfolio'
        ),
        # 4. Bottom Chart: Tangency Weights Bar Chart
        go.Bar(
            x=assets, y=opt_weights * (progress if not show_tangency else 1.0), 
            marker_color='#39ff14', name='Weights'
        )
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=[
        "Markowitz Bullet: Risk vs. Expected Return",
        "Tangency Portfolio Capital Allocation"
    ],
    vertical_spacing=0.15,
    row_heights=[0.6, 0.4]
)

# Add Dummy Traces
for _ in range(3):
    fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1)
fig.add_trace(go.Bar(x=assets, y=[0]*n_assets), row=2, col=1)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
# Sliders pushed down and prefix removed
sliders = [dict(
    active=0, currentvalue={"prefix": "", "font": {"color": "white"}}, pad={"t": 10},
    x=0.15, len=0.85, y=-0.15, 
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f"{int(progress * 100)}%"
    ) for idx, progress in enumerate(np.linspace(0, 1, n_frames))]
)]

fig.update_layout(
    height=800, width=1000,
    title_text="Empirical Portfolio Theory: Efficient Frontier from Historical Returns",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=150, r=50, l=50), # Expanded bottom margin (b=150) to fit controls
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, # Buttons pushed further down
        'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, port_vols.max()*1.1], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

ef_y_min = min(0, port_returns.min() * 1.1)
ef_y_max = max(0, target_returns.max() * 1.05)
fig.update_yaxes(title_text='Expected Return (μ)', range=[ef_y_min, ef_y_max], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Asset Class', row=2, col=1, gridcolor='rgba(128,128,128,0.2)', tickangle=45)
fig.update_yaxes(title_text='Weight (%)', range=[0, opt_weights.max()*1.2], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

# --- ADD ANNOTATION OVERLAYS ---
max_sharpe_val = port_sharpes.max()
fig.add_annotation(
    text=f"<b>Max Achievable Sharpe: {max_sharpe_val:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.show()
```

Perfect, every asset manager just needs historical data, one analyst that knows Python or can order Claude around and they have optimized their portfolio

Just hold the optimized selection, rebalance quarterly, and collect your management fees - it can't be that easy, right?

Well, it's not.

---

#### 2.) 📉 Downfalls of In-Sample Optimization

##### Roman's Critique

Historical data isn't indicative of future performance, we hear it all the time

Well, what did we need to generate the Efficient Frontier in any capacity?  Historical expected returns and covariance.

 
 The Efficient Frontier optimization requires two key statistics:
 
 - **Expected returns vector**: $\mu = \mathbb{E}[r] = \begin{bmatrix} \mathbb{E}[r_1] \\ \mathbb{E}[r_2] \\ \vdots \\ \mathbb{E}[r_n] \end{bmatrix}$
 
 - **Covariance matrix**: $\Sigma = \mathbb{E}\left[(r - \mu)(r - \mu)^\top\right] = \begin{bmatrix}
     \operatorname{Cov}(r_1, r_1) & \operatorname{Cov}(r_1, r_2) & \cdots & \operatorname{Cov}(r_1, r_n) \\
     \operatorname{Cov}(r_2, r_1) & \operatorname{Cov}(r_2, r_2) & \cdots & \operatorname{Cov}(r_2, r_n) \\
     \vdots & \vdots & \ddots & \vdots \\
     \operatorname{Cov}(r_n, r_1) & \operatorname{Cov}(r_n, r_2) & \cdots & \operatorname{Cov}(r_n, r_n)
   \end{bmatrix}$
 
 Where $r$ is the vector of asset returns.


These statistics are artifacts of data, there is no signal, no structure, if we purturb them slightly we have wildly different portfolio values


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & PROCESSING ---
# ==========================================
np.random.seed(42)

# Load empirical data
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# Define the subset of assets for the "Horror" demonstration
# Note: Using GOOGL as per your provided CSV sample
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

# --- Original Universe ---
expected_returns = df_ret.mean().values * 252
cov_matrix_orig = df_ret.cov().values * 252

# --- Perturbed Universe (+0.2% Daily Noise) ---
df_centered = df_ret - df_ret.mean()
noise = np.random.normal(0, 0.002, df_ret.shape) 
df_pert = df_centered + noise
cov_matrix_pert = df_pert.cov().values * 252

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 5000 

random_weights_orig = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)
random_weights_pert = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

def compute_frontier_data(weights, expected_ret, cov_mat):
    p_ret = weights @ expected_ret
    p_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_mat, weights))
    p_sharpe = (p_ret - risk_free_rate) / p_vol
    sort_idx = np.argsort(p_sharpe)
    return weights[sort_idx], p_ret[sort_idx], p_vol[sort_idx], p_sharpe[sort_idx]

rw_orig, pr_orig, pv_orig, ps_orig = compute_frontier_data(random_weights_orig, expected_returns, cov_matrix_orig)
rw_pert, pr_pert, pv_pert, ps_pert = compute_frontier_data(random_weights_pert, expected_returns, cov_matrix_pert)

# Find Tangency Portfolios
opt_idx_orig = np.argmax(ps_orig)
opt_w_orig = rw_orig[opt_idx_orig]
opt_v_orig, opt_r_orig = pv_orig[opt_idx_orig], pr_orig[opt_idx_orig]

opt_idx_pert = np.argmax(ps_pert)
opt_w_pert = rw_pert[opt_idx_pert]
opt_v_pert, opt_r_pert = pv_pert[opt_idx_pert], pr_pert[opt_idx_pert]

# Efficient Frontier Lines
target_returns = np.linspace(pr_orig.min(), pr_orig.max() * 1.05, 50)
def optimize_ef_line(cov_mat):
    vols = []
    for tr in target_returns:
        res = sco.minimize(lambda w: np.sqrt(w.T @ cov_mat @ w), 
                           np.ones(n_assets)/n_assets, method='SLSQP',
                           bounds=[(0, 1) for _ in range(n_assets)],
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                        {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
        vols.append(res.fun if res.success else np.nan)
    return np.array(vols)

ef_vols_orig = optimize_ef_line(cov_matrix_orig)
ef_vols_pert = optimize_ef_line(cov_matrix_pert)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    num_ports = int(n_portfolios * progress)
    ef_points = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # --- ORIGINAL (LEFT) ---
        go.Scatter(x=pv_orig[:num_ports], y=pr_orig[:num_ports], mode='markers', marker=dict(color=ps_orig[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        go.Scatter(x=ef_vols_orig[:ef_points], y=target_returns[:ef_points], mode='lines', line=dict(color='#00E5FF', width=3, dash='dash')),
        go.Scatter(x=[opt_v_orig] if show_tangency else [None], y=[opt_r_orig] if show_tangency else [None], mode='markers', marker=dict(color='#00E5FF', size=16, symbol='star', line=dict(color='white', width=1))),
        
        # --- PERTURBED (RIGHT) ---
        go.Scatter(x=pv_pert[:num_ports], y=pr_pert[:num_ports], mode='markers', marker=dict(color=ps_pert[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        go.Scatter(x=ef_vols_pert[:ef_points], y=target_returns[:ef_points], mode='lines', line=dict(color='#39ff14', width=3, dash='dash')),
        go.Scatter(x=[opt_v_pert] if show_tangency else [None], y=[opt_r_pert] if show_tangency else [None], mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1))),
        
        # --- BOTTOM BARS (SUBSET ONLY) ---
        go.Bar(x=subset_symbols, y=opt_w_orig[subset_indices] * (progress if not show_tangency else 1.0), marker_color='#00E5FF'),
        go.Bar(x=subset_symbols, y=opt_w_pert[subset_indices] * (progress if not show_tangency else 1.0), marker_color='#39ff14')
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Original Data: Markowitz Bullet",
        "Perturbed Data (+0.2% Noise): Markowitz Bullet",
        "Original Conviction: AVGO / GOOGL / WELL",
        "Perturbed Conviction: AVGO / GOOGL / WELL"
    ],
    vertical_spacing=0.15, horizontal_spacing=0.08, row_heights=[0.6, 0.4]
)

# Initialize traces
for _ in range(3): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1) 
for _ in range(3): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=2)
fig.add_trace(go.Bar(x=subset_symbols, y=[0]*3), row=2, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=[0]*3), row=2, col=2)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & AESTHETICS ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "", "font": {"color": "white"}}, pad={"t": 10},
    x=0.15, len=0.85, y=-0.15, 
    steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=f"{int(idx)}%") for idx, f in enumerate(frames)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Sensitivity Horror: How Tiny Covariance Shifts Flip Portfolio Conviction",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(t=80, b=150, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Axis Limits
global_max_vol = max(pv_orig.max(), pv_pert.max()) * 1.1
global_max_ret = target_returns.max() * 1.05
global_max_weight = max(opt_w_orig[subset_indices].max(), opt_w_pert[subset_indices].max()) * 1.5

fig.update_xaxes(title_text='Volatility', range=[0, global_max_vol], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return', range=[0, global_max_ret], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_xaxes(title_text='Volatility', range=[0, global_max_vol], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return', range=[0, global_max_ret], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

# Bar Charts
fig.update_yaxes(title_text='Weight (%)', range=[0, global_max_weight], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Weight (%)', range=[0, global_max_weight], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# Annotations
fig.add_annotation(text=f"<b>Orig Sharpe: {ps_orig.max():.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, font=dict(color="#00E5FF"), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
fig.add_annotation(text=f"<b>Pert Sharpe: {ps_pert.max():.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, font=dict(color="#39ff14"), bgcolor="rgba(0,0,0,0.6)", row=1, col=2)

fig.show()
```

This is the most important idea that you will hear today

If you have real structure, purturbations will lead to decay in performance, not performance off a cliff like we see above

This is why performance out-of-sample will be terrible, weights didn't slightly shift, everything was overfit from the start

###### ______________________________________________________________________________________________________________________________________

##### MVO Out of Sample Performance

Clearly, it doesn't "learn" anything in the robust modeling sense as we saw above


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco
import statsmodels.api as sm

# ==========================================
# --- 1. DATA SLICE: THE LATEST YEAR ---
# ==========================================
np.random.seed(42)

df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

# Slice the latest 252 trading days (1 Year)
df_latest_year = df_ret.tail(252)

# 6-Month Split (126 days each)
split_6mo = 126
df_is = df_latest_year.iloc[:split_6mo]   # In-Sample
df_oos = df_latest_year.iloc[split_6mo:]  # Out-of-Sample

assets = df_ret.columns.tolist()
n_assets = len(assets)
risk_free_rate = 0.02

# ==========================================
# --- 2. THE OPTIMIZATION (6-MONTH OVERFIT) ---
# ==========================================
mu_is = df_is.mean().values * 252
cov_is = df_is.cov().values * 252

def get_neg_sharpe(weights, mu, cov):
    p_ret = weights @ mu
    p_vol = np.sqrt(weights.T @ cov @ weights)
    return -(p_ret - risk_free_rate) / p_vol

res = sco.minimize(
    get_neg_sharpe,
    np.ones(n_assets) / n_assets,
    args=(mu_is, cov_is),
    method='SLSQP',
    bounds=[(0, 1) for _ in range(n_assets)],
    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
)

w_opt = res.x
is_vol = np.sqrt(w_opt.T @ cov_is @ w_opt)
is_ret = w_opt @ mu_is
is_sharpe = -res.fun

# --- Out-of-Sample Performance ---
oos_opt_daily = df_oos @ w_opt
oos_mkt_daily = df_oos.mean(axis=1)
oos_opt_cum = (1 + oos_opt_daily).cumprod()
oos_mkt_cum = (1 + oos_mkt_daily).cumprod()

# --- Out-of-Sample Stats ---
oos_opt_sharpe = (oos_opt_daily.mean() * 252 - risk_free_rate) / (oos_opt_daily.std() * np.sqrt(252))
oos_mkt_sharpe = (oos_mkt_daily.mean() * 252 - risk_free_rate) / (oos_mkt_daily.std() * np.sqrt(252))

X = sm.add_constant(oos_mkt_daily)
model = sm.OLS(oos_opt_daily, X).fit()
beta_val = model.params.iloc[1]
alpha_p_value = model.pvalues.iloc[0]

# --- Frontier Line ---
target_rets = np.linspace(df_is.mean().min() * 252, df_is.mean().max() * 252, 50)
ef_vols_is = [
    sco.minimize(
        lambda w: np.sqrt(w.T @ cov_is @ w),
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=[(0, 1) for _ in range(n_assets)],
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w.T @ mu_is - tr}
        ]
    ).fun
    for tr in target_rets
]

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
step_size = max(1, len(df_oos) // n_frames)
oos_indices = list(range(1, len(df_oos) + 1, step_size))

for idx in oos_indices:
    frame_data = [
        # Frame Data must match trace indices 0, 1, 2, 3 in the main fig
        go.Scatter(x=ef_vols_is, y=target_rets),                 # Trace 0
        go.Scatter(x=[is_vol], y=[is_ret]),                      # Trace 1
        go.Scatter(x=df_oos.index[:idx], y=oos_opt_cum[:idx]),  # Trace 2
        go.Scatter(x=df_oos.index[:idx], y=oos_mkt_cum[:idx])   # Trace 3
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{idx}"))

# ==========================================
# --- 4. FIGURE & LAYOUT ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "6-Month In-Sample Optimization: The Star",
        "6-Month Out-of-Sample: Optimized (Green) vs. Market (Cyan)"
    ],
    vertical_spacing=0.15
)

# Initialize 4 Explicit Traces
# Trace 0: IS Frontier Line
fig.add_trace(
    go.Scatter(
        x=ef_vols_is,
        y=target_rets,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.2)', width=2, dash='dash'),
        showlegend=False
    ),
    row=1, col=1
)

# Trace 1: IS Tangency Star
fig.add_trace(
    go.Scatter(
        x=[is_vol],
        y=[is_ret],
        mode='markers',
        marker=dict(color='#39ff14', size=16, symbol='star'),
        showlegend=False
    ),
    row=1, col=1
)

# Trace 2: OOS Optimized Path
fig.add_trace(
    go.Scatter(
        x=[df_oos.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#39ff14', width=4),
        name='Optimized Portfolio'
    ),
    row=2, col=1
)

# Trace 3: OOS Market Path
fig.add_trace(
    go.Scatter(
        x=[df_oos.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#00E5FF', width=3, dash='dot'),
        name='Market Portfolio'
    ),
    row=2, col=1
)

fig.frames = frames

# Play Button and Slider Styling
fig.update_layout(
    height=700, width=1100,
    title_text="The Overfitting Trap: 6-Mo Optimization vs. 6-Mo Forward Reality",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    margin=dict(t=100, b=150, r=50, l=50),
    updatemenus=[{
        'type': 'buttons',
        'x': -0.05,
        'y': -0.18,
        'xanchor': 'left',
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'transition': {'duration': 0},
                'fromcurrent': True,
                'mode': 'immediate'
            }]
        }],
        'font': {'color': 'white', 'size': 14},
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'borderwidth': 1
    }],
    sliders=[dict(
        active=0,
        currentvalue={"prefix": ""},
        pad={"t": 20},
        x=0.15,
        len=0.85,
        y=-0.15,
        steps=[
            dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=f"{int(i / len(frames) * 100)}%"
            )
            for i, f in enumerate(frames)
        ]
    )],
    # Add Static Breakeven Line
    shapes=[dict(
        type="line",
        xref="x2",
        yref="y2",
        x0='2025-11-01',
        x1='2026-04-30',
        y0=1.0,
        y1=1.0,
        line=dict(color="white", width=1, dash="dash"),
        opacity=0.3
    )]
)

# Fixed Axis Configuration
fig.update_xaxes(
    title_text="Volatility",
    range=[0, is_vol * 2.5],
    row=1, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)
fig.update_yaxes(
    title_text="Return",
    row=1, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)

fig.update_xaxes(
    title_text="Date (Out-of-Sample)",
    range=['2025-09-01', '2026-04-30'],
    row=2, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)
fig.update_yaxes(
    title_text="Value ($1 Start)",
    range=[0.9, 1.3],
    row=2, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)

# Annotations
fig.add_annotation(
    text=f"<b>IS Overfitted Sharpe: {is_sharpe:.2f}</b>",
    xref="x domain",
    yref="y domain",
    x=0.02,
    y=0.95,
    showarrow=False,
    font=dict(color="#39ff14", size=14),
    row=1, col=1
)

fig.add_annotation(
    text=(
        f"<b>OOS Optimized Sharpe: {oos_opt_sharpe:.2f}</b><br>"
        f"<b>OOS Market Sharpe: {oos_mkt_sharpe:.2f}</b><br><br>"
        f"Portfolio Beta: {beta_val:.2f}<br>"
        f"Alpha p-value: {alpha_p_value:.4f}"
    ),
    xref="x domain",
    yref="y domain",
    x=0.98,
    y=0.95,
    showarrow=False,
    align="right",
    font=dict(color="#ffffff", size=13),
    bgcolor="rgba(30,30,30,0.8)",
    bordercolor="white",
    borderwidth=1,
    borderpad=8,
    row=2, col=1
)

fig.show()
```

Why is this the case?

When we overfit noise we are chasing in sample metrics

There is no structure there, real structure lies in alternative data, the scientific method, economic interpretations - real trading signals trade structure

###### ______________________________________________________________________________________________________________________________________

##### There Are No Forward Looking Expected Returns

Historical data isn't indicative of future performance because they don't represent the actual expectation in a forward looking sense

You can argue market equilibrium (like implied volatility) is a proxy for this, but more on this in the context of Black-Litterman

In any case, with current technology we do not have good and reliable forward return estimates - the world is non-stationary - there is no distribution!!!!!

**In Other Words:**
- Why is your model suggesting 20% expected returns any different from my analyst saying its overvalued?
- Who is correct?  Welcome to finance.


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA IMPORT & REGIME SPLITS ---
# ==========================================
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

# Use the Equal-Weight Market Portfolio
market_daily_ret = df_ret.mean(axis=1)
n_days = len(market_daily_ret)

# Divide into 3 disjoint "Regimes" (approx 4 months each)
split_points = [0, n_days // 3, (2 * n_days) // 3, n_days]

# Calculate the mean for each disjoint period (Annualized)
regime_means = []
for i in range(3):
    start, end = split_points[i], split_points[i + 1]
    regime_mu = market_daily_ret.iloc[start:end].mean() * 252
    regime_means.append((df_ret.index[start], df_ret.index[end - 1], regime_mu))

# --- LLN Convergence Math ---
running_mean = (market_daily_ret.expanding().mean() * 252).values
market_cum_growth = (1 + market_daily_ret).cumprod().values

# ==========================================
# --- 2. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
indices = np.linspace(1, n_days, n_frames, dtype=int)

for k in indices:
    current_dates = market_daily_ret.index[:k]

    # Logic to show the "Local" regime mean only when we are in that period
    curr_regime_line = np.full(k, np.nan)
    for start_dt, end_dt, mu in regime_means:
        mask = (current_dates >= start_dt) & (current_dates <= end_dt)
        curr_regime_line[mask] = mu

    frame_data = [
        go.Scatter(x=current_dates, y=curr_regime_line),   # Trace 0
        go.Scatter(x=current_dates, y=running_mean[:k]),   # Trace 1
        go.Scatter(x=current_dates, y=market_cum_growth[:k])  # Trace 2
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 3. FIGURE & LAYOUT ---
# ==========================================
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "LLN Convergence: Disjoint Regime Means vs. Running Average",
        "Market Portfolio: Realized Cumulative Growth"
    ],
    vertical_spacing=0.2,
    row_heights=[0.48, 0.52]
)

# Trace 0: Disjoint Local Mean
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[regime_means[0][2]],
        mode='lines',
        line=dict(color='rgba(0,255,255,0.4)', width=4, dash='dot'),
        name='Local Regime Mean'
    ),
    row=1, col=1
)

# Trace 1: Running Average
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[running_mean[0]],
        mode='lines',
        line=dict(color='#ffffff', width=3),
        name='Running Cumulative Mean'
    ),
    row=1, col=1
)

# Trace 2: Realized Equity Curve
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#39ff14', width=4),
        name='Cumulative Growth'
    ),
    row=2, col=1
)

fig.frames = frames

# Play Button Styling
play_button = {
    'type': 'buttons',
    'x': 0.0,
    'y': -0.21,
    'xanchor': 'left',
    'yanchor': 'top',
    'direction': 'left',
    'showactive': False,
    'bgcolor': 'rgba(0,0,0,0)',
    'bordercolor': 'white',
    'borderwidth': 1,
    'font': {'color': 'white'},
    'buttons': [
        {
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 30, 'redraw': True},
                    'fromcurrent': True
                }
            ]
        }
    ]
}

# Fix: Reference frame index and label correctly
slider_steps = []
for i, k in enumerate(indices):
    step = {
        "args": [[f"step{k}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        "label": str(market_daily_ret.index[k - 1].date()),
        "method": "animate"
    }
    slider_steps.append(step)

fig.update_layout(
    height=700,
    width=1100,
    title_text="Regime Shifts & The Law of Large Numbers (Market Portfolio)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.48,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    ),
    margin=dict(t=100, b=160, r=50, l=50),
    updatemenus=[play_button],
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": ""},
            x=0.15,
            len=0.85,
            y=-0.18,
            steps=slider_steps
        )
    ]
)

# Axis Configuration with "modern" grid like in snippet: thin, translucent, mild gray, always shown
fig.update_xaxes(
    title_text="Date",
    row=1,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)
fig.update_yaxes(
    title_text="Annualized Return",
    range=[running_mean.min() - 0.1, running_mean.max() + 0.1],
    row=1,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)

fig.update_xaxes(
    title_text="Date",
    row=2,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)
fig.update_yaxes(
    title_text="Value ($1 Start)",
    range=[0.9, 1.7],
    row=2,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)

# Static Breakeven Line
fig.add_shape(
    type="line",
    xref="x2",
    yref="y2",
    x0=market_daily_ret.index[0],
    x1=market_daily_ret.index[-1],
    y0=1.0,
    y1=1.0,
    line=dict(color="white", width=1, dash="dash"),
    opacity=0.3
)

fig.show()
```

I've long posited model informed decision making, make decisions in the present based on your due diligence and a model informed value

Black-Litterman introduces exactly this idea...

---

#### 3.) 📈 Black-Litterman

##### Quasi-Objective Optimization

Instead of operating only on historical expected returns, Black-Litterman include a qualitative component

This allows for the injection of subjectivity into the model, what is difficult to quantify you can now rank and impact the opotimal portfolio from optimization

In other words, you aren't now just optimizing based on historical data, but there is quantitative representation of forward looking expectations

This quantitative representation need not (but can be) systematic, it can be based on the desk views of a sector, the PM, CIO, etc...

 The classical efficient frontier relies on the historical mean vector $\mu$ and covariance matrix $\Sigma$:
 $$
 \mu = \mathbb{E}[r], \quad \Sigma = \mathbb{E}[(r-\mu)(r-\mu)^\top]
 $$
  The Combined Estimator (The "Posterior")
  The "Magic" of the model happens in the Master Formula, which produces the Posterior Expected Returns ($\mu_{BL}$):
  $$
  \mu_{BL} = \left[ ( \tau \Sigma )^{-1} + P^\top \Omega^{-1} P \right]^{-1} \left[ ( \tau \Sigma )^{-1} \Pi + P^\top \Omega^{-1} Q \right]
  $$

  - Where $P$ encodes which assets each view relates to, $Q$ holds the implied returns of those views, $\Omega$ is the view confidence, and $\tau$ is a scaling hyperparameter.

  - If you have no views, $\mu_{BL}$ equals the Market Equilibrium ($\Pi$, *implied expected return*).

  - If you have high-confidence views, $\mu_{BL}$ tilts toward $Q$.

  - If you have low-confidence views, the model "shrinks" the returns back toward the Market Anchor.

 ##### Market Equilibrium Produces Objective Consensus

Forward expected returns rely only on observable market data and historical covariance estimates

Covariance estimates tend to be more stable than expected return estimates

This is what leads to better optimization performance and stability


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & REVERSE OPTIMIZATION ---
# ==========================================
np.random.seed(42)

df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# --- BLACK-LITTERMAN EQUILIBRIUM MATH ---
risk_free_rate = 0.02
delta = 2.5  # Risk aversion coefficient
cov_matrix = df_ret.cov().values * 252

# Anchor: Assume Equal-Weight is the Market Equilibrium
w_mkt = np.ones(n_assets) / n_assets

# Calculate Implied Equilibrium Returns (Pi)
# Pi = delta * Sigma * w_mkt
pi = delta * cov_matrix @ w_mkt

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
n_portfolios = 6000 
random_weights = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

# Use Equilibrium Returns (Pi)
port_returns = random_weights @ pi
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

sort_idx = np.argsort(port_sharpes)
port_returns, port_vols, port_sharpes = port_returns[sort_idx], port_vols[sort_idx], port_sharpes[sort_idx]

# Tangency Point (Market Equilibrium)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret, opt_vol = port_returns[max_sharpe_idx], port_vols[max_sharpe_idx]

# EF Line Calculation
target_returns = np.linspace(port_returns.min(), port_returns.max() * 1.05, 60)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ pi - tr}])
    ef_vols.append(res.fun if res.success else np.nan)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
for i in range(1, n_frames + 1):
    progress = i / n_frames
    num_ports = int(n_portfolios * progress)
    ef_points = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # 1. Equilibrium Cloud
        go.Scatter(x=port_vols[:num_ports], y=port_returns[:num_ports], mode='markers',
                   marker=dict(color=port_sharpes[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        # 2. Equilibrium EF Line
        go.Scatter(x=ef_vols[:ef_points], y=target_returns[:ef_points], mode='lines',
                   line=dict(color='#00ffff', width=3, dash='dash')),
        # 3. Tangency Star
        go.Scatter(x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
                   mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)))
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================

# Play Button Styling (to match previous code cell, see @file_context_0)
play_button = {
    'type': 'buttons',
    'x': 0.0,
    'y': -0.21,
    'xanchor': 'left',
    'yanchor': 'top',
    'direction': 'left',
    'showactive': False,
    'bgcolor': 'rgba(0,0,0,0)',
    'bordercolor': 'white',
    'borderwidth': 1,
    'font': {'color': 'white'},
    'buttons': [
        {
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 40, 'redraw': True},
                    'fromcurrent': True
                }
            ]
        }
    ]
}

slider_steps = [
    dict(
        method="animate",
        args=[[f"step{i+1}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
        label=f"{i}%"
    ) for i in range(n_frames)
]

fig = go.Figure(
    data=[
        go.Scatter(x=[None], y=[None], mode='markers'),
        go.Scatter(x=[None], y=[None], mode='lines'),
        go.Scatter(x=[None], y=[None], mode='markers')
    ]
)

fig.update_layout(
    title="Black-Litterman Stage 1: The Implied Equilibrium Frontier",
    xaxis=dict(title="Volatility (Risk, σ)", range=[0, port_vols.max()*1.1], gridcolor='rgba(128,128,128,0.2)'),
    yaxis=dict(title="Implied Return (Π)", range=[0, target_returns.max()*1.05], gridcolor='rgba(128,128,128,0.2)'),
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    height=700,
    showlegend=False,
    margin=dict(t=80, b=150, r=50, l=50),
    updatemenus=[play_button],
    sliders=[dict(
        active=0,
        currentvalue={"prefix": ""},
        x=0.15,
        len=0.85,
        y=-0.18,
        steps=slider_steps
    )]
)

fig.frames = frames

# --- ANNOTATION OVERLAY ---
fig.add_annotation(
    text=f"<b>Equilibrium Sharpe: {port_sharpes.max():.2f}</b>",
    xref="paper", yref="paper", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6
)

fig.show()
```

 **Optimality (star portfolio) here literally implies holding the market portfolio, the tangency portfolio**

This is by construction of the framework, we are backing out the equilibrium (implied) returns in a stable capacity
 
This is akin to implied volatility, not necessarily a forecast, but what equilibrium is implying about expected returns

 **To infer equilibrium returns step by step:**
 1. Assume investors collectively hold the market (tangency) portfolio, $w_{mkt}$
 2. The equilibrium (implied) returns, $\Pi$, are reverse-engineered from this portfolio via:
    $$
    \Pi = \delta \Sigma w_{mkt}
    $$
    - $\delta$ = risk aversion coefficient
    - $\Sigma$ = covariance matrix of returns
    - $w_{mkt}$ = market-capitalization weights (or chosen benchmark weights)
 3. $\Pi$ serves as the "market consensus" expected return vector under equilibrium


###### ______________________________________________________________________________________________________________________________________

##### Stability After Perturbation

Recall a small perturbation in the optimization above led us to dramatically different portfolio allocations?

Let's observe the allocations before and after perturbation in classic MVO and Black-Litterman using implied expected returns


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA PREP & SHARED PERTURBATION ---
# ==========================================
np.random.seed(42)
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

# --- THE SHARED PERTURBATION (0.5% Daily Noise) ---
noise_level = 0.005
shared_noise = np.random.normal(0, noise_level, df_ret.shape) 
df_noisy = df_ret + shared_noise

# ==========================================
# --- 2. MODEL CALCULATIONS ---
# ==========================================
risk_free_rate = 0.02
delta = 2.5
w_mkt = np.ones(n_assets) / n_assets

def solve_weights(mu, cov):
    res = sco.minimize(lambda w: -(w @ mu - risk_free_rate) / np.sqrt(w.T @ cov @ w),
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    return res.x if res.success else np.zeros(n_assets)

# --- MVO (Historical) ---
mu_mvo_o, cov_mvo_o = df_ret.mean().values * 252, df_ret.cov().values * 252
mu_mvo_n, cov_mvo_n = df_noisy.mean().values * 252, df_noisy.cov().values * 252
w_mvo_o = solve_weights(mu_mvo_o, cov_mvo_o)
w_mvo_n = solve_weights(mu_mvo_n, cov_mvo_n)

# --- BL (Equilibrium Anchor) ---
pi_o = delta * cov_mvo_o @ w_mkt
pi_n = delta * cov_mvo_n @ w_mkt
w_bl_o = solve_weights(pi_o, cov_mvo_o)
w_bl_n = solve_weights(pi_n, cov_mvo_n)

# ==========================================
# --- 3. PLOTTING WITH SHARED NOISE TITLE ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, vertical_spacing=0.15,
    subplot_titles=[
        "MVO: Original Weights", f"MVO: Shared Noise (+{noise_level*100}%)",
        "BL: Original Weights", f"BL: Shared Noise (+{noise_level*100}%)"
    ]
)

# Colors
c_orig, c_mvo_n, c_bl = '#00E5FF', '#FF3D00', '#39ff14'

fig.add_trace(go.Bar(x=subset_symbols, y=w_mvo_o[subset_indices], marker_color=c_orig), row=1, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_mvo_n[subset_indices], marker_color=c_mvo_n), row=1, col=2)
fig.add_trace(go.Bar(x=subset_symbols, y=w_bl_o[subset_indices], marker_color=c_bl), row=2, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_bl_n[subset_indices], marker_color=c_bl), row=2, col=2)

# Delta Annotations
mvo_delta = np.sum(np.abs(w_mvo_n[subset_indices] - w_mvo_o[subset_indices])) * 100
bl_delta = np.sum(np.abs(w_bl_n[subset_indices] - w_bl_o[subset_indices])) * 100

fig.add_annotation(text=f"<b>Total Subset Shift: {mvo_delta:.1f}%</b>", 
                   xref="x2 domain", yref="y2 domain", x=0.95, y=0.9, showarrow=False, 
                   font=dict(color=c_mvo_n, size=14), bgcolor="rgba(0,0,0,0.7)", row=1, col=2)

fig.add_annotation(text=f"<b>Total Subset Shift: {bl_delta:.1f}%</b>", 
                   xref="x4 domain", yref="y4 domain", x=0.95, y=0.9, showarrow=False, 
                   font=dict(color=c_bl, size=14), bgcolor="rgba(0,0,0,0.7)", row=2, col=2)

fig.update_layout(
    height=700, width=1100, 
    title_text=f"Shared Noise Impact ({noise_level*100}% Perturbation): MVO Overfitting vs. Black-Litterman Stability",
    template="plotly_dark", showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
)

fig.update_yaxes(title_text="Weight (%)", range=[0, 0.7], gridcolor='rgba(255,255,255,0.1)')
fig.show()
```

The equilibrium anchor improves the stability of the optimization routine, as seen above

We can then combine with subjective views to develop our portfolios, the efficacy of our subjectivity however is the chief remaining issue

###### ______________________________________________________________________________________________________________________________________

##### Subjective Tilting for Injecting Views

 **Black-Litterman Subjective Inputs—Quick Reference**
 
 - $P$: Pick matrix. Each row = a "view". Put $+1$ for favored asset, $-1$ for unfavored, $0$ elsewhere.
 - $Q$: View returns. Each entry = subjective excess return for its view.
 - $\Omega$: Diagonal matrix of view uncertainty. Lower values = more confidence.
 
 Typical choice:
 $$
 \Omega_{ii} = P_i\, \tau \Sigma\, P_i^\top
 $$
 where $P_i$ is row $i$ of $P$, $\Sigma$ is asset covariance, and $\tau$ is a small scalar (e.g. $0.025$).




```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA PREP ---
# ==========================================
np.random.seed(42)
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

cov_matrix = df_ret.cov().values * 252

# ==========================================
# --- 2. BLACK-LITTERMAN MATH (THE TILT) ---
# ==========================================
risk_free_rate = 0.02
delta = 2.5
tau = 0.05 # Scaling factor for the uncertainty of the prior
w_mkt = np.ones(n_assets) / n_assets

# --- STAGE 1: The Prior (Equilibrium Returns) ---
pi = delta * cov_matrix @ w_mkt

# --- STAGE 2: The Views (Subjective Expertise) ---
# We express absolute views: AVGO will return 25%, GOOGL will return 20%
K = 2 # Number of views
P = np.zeros((K, n_assets)) # Pick Matrix
P[0, assets.index('AVGO')] = 1.0
P[1, assets.index('GOOGL')] = 1.0

Q = np.array([0.25, 0.20]) # View Expected Returns

# Omega (Uncertainty Matrix). Using the standard proportional formulation.
omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

# --- STAGE 3: The Posterior (Combined Expected Returns) ---
# The BL Master Formula: Combining the Prior (Pi) with our Views (Q)
tau_cov_inv = np.linalg.inv(tau * cov_matrix)
omega_inv = np.linalg.inv(omega)

M1 = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
M2 = tau_cov_inv @ pi + P.T @ omega_inv @ Q
mu_bl = M1 @ M2

# ==========================================
# --- 3. OPTIMIZATION ---
# ==========================================
def solve_weights(mu, cov):
    res = sco.minimize(lambda w: -(w @ mu - risk_free_rate) / np.sqrt(w.T @ cov @ w),
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    return res.x if res.success else np.zeros(n_assets)

# Optimize for both states
w_prior = solve_weights(pi, cov_matrix)
w_posterior = solve_weights(mu_bl, cov_matrix)

# ==========================================
# --- 4. PLOTTING ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=[
        "BL Stage 1: Equilibrium Prior (Neutral)", 
        "BL Stage 2: Posterior (Hard Tech Tilt)"
    ]
)

c_prior = '#39ff14' # Neon Green
c_post = '#00E5FF'  # Cyan

# Bar Charts
fig.add_trace(go.Bar(x=subset_symbols, y=w_prior[subset_indices], marker_color=c_prior), row=1, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_posterior[subset_indices], marker_color=c_post), row=1, col=2)

# Calculate Deltas to show the surgical shift
avgo_shift = (w_posterior[assets.index('AVGO')] - w_prior[assets.index('AVGO')]) * 100
googl_shift = (w_posterior[assets.index('GOOGL')] - w_prior[assets.index('GOOGL')]) * 100
well_shift = (w_posterior[assets.index('WELL')] - w_prior[assets.index('WELL')]) * 100

fig.add_annotation(
    text=f"<b>AVGO Shift: +{avgo_shift:.1f}%</b><br><b>GOOGL Shift: +{googl_shift:.1f}%</b><br><span style='color:#AAAAAA'>WELL Shift: {well_shift:.1f}%</span>", 
    xref="x2 domain", yref="y2 domain", x=0.95, y=0.95, showarrow=False, align="right",
    font=dict(color=c_post, size=14), bgcolor="rgba(0,0,0,0.7)", bordercolor=c_post, borderwidth=1, borderpad=6, row=1, col=2
)

fig.update_layout(
    height=600, width=1100, 
    title_text="Subjective Expertise: Filtering Views Through the Black-Litterman Model",
    template="plotly_dark", showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
)

max_weight = max(w_prior.max(), w_posterior.max()) * 1.2
fig.update_yaxes(title_text="Weight (%)", range=[0, max_weight], gridcolor='rgba(255,255,255,0.1)')

fig.show()
```

###### ______________________________________________________________________________________________________________________________________

##### Decision Making as Performance and Models as a Filter
 
 "Now" only happens once—each portfolio decision is made in a unique, fleeting market environment. We can't replay today, and tomorrow will inherently differ. In reality, the need for pure systematic approaches can be overstated: what we often benefit from is not absolute automation, but effective filtering of our discretionary decisions through ephemeral, adaptive models that respect the singularity of each "now."
 
 By explicitly modeling our current beliefs—however ephemeral—and passing our discretionary choices through a quantitative filter, we benefit from both structure and adaptability. We aren't forced to commit to rigid rules; rather, we use the model to highlight the implications and interactions of our subjective tilts, reducing the risk that our instincts run unchecked by market realities.
 
 The Black-Litterman model offers an elegant way to encode this fusion. By allowing us to incorporate views (Q) and their confidence/uncertainty (Omega) on top of the equilibrium market consensus, we can map beliefs into portfolio weights in a disciplined manner. The Q vector is where we write down the actual opinions—a sector will outperform, a region will underperform, etc.—while Omega quantifies how strongly we trust those views relative to the historical data. Large Omega means the view is vague or low-confidence, giving it little effect; small Omega means high conviction, bending the optimization noticeably.
 
 As discretionary decision-makers, we can use this machinery to test the effect of our judgment, formalize the fuzziness of "I think X will happen," and let the model temper (but not erase) our hunches. In this way, we become comfortable with models not as rigid dictators, but as helpful filters for one-off, in-the-moment choices—embracing the fact that the present is always ephemeral.

---

#### 4.) 💭 Closing Thoughts and Future Topics

 **📑 TL;DW Executive Summary**
 - MVO is an "Error Maximizer": Standard Mean-Variance Optimization treats historical noise as structural truth, causing portfolio weights to flip violently in response to tiny, random data jitters
 - The Equilibrium Anchor ($\Pi$): Black-Litterman replaces noisy historical means with Implied Equilibrium Returns, which are back-calculated from current market weights to create a stable, "neutral" starting point
 - Mathematical Stability: Because BL anchors to the market consensus, the Efficient Frontier remains structurally robust; it requires significant evidence (not just 0.5% noise) to shift the "Star" away from the diversified benchmark
 - Bayesian Prior: The Equilibrium state represents the "Global Consensus" distribution—the portfolio you should hold if you admit you have no unique information that the rest of the market doesn't already have
 - Surgical Subjective Views: Expertise is added by "tilting" this stable anchor; you inject specific views (e.g., "Tech will outperform") to warp the frontier with precision, rather than letting historical noise dictate your conviction
 

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
 - Trading with Alternative Data Sources
 - Pairs Trading and Statistical Arbitrage
 - Data Cleaning & Outlier Handling in Financial Time Series
 - Practical Issues in Multi-Asset Portfolio Backtesting
 - Risk Premia Harvesting: Equity, FX, Rates

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- How Interactive Broker's API Works (EWrapper/EClient)
- How to Backtest a Trading Strategy with Interactive Brokers
- Algorithmic Volatility Trading System

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
