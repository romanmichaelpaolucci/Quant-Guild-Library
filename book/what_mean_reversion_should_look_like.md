### 📈 Quant Strats: What Mean Reversion Should Look Like

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

### 📖 Sections

#### 1.) 📈 Generating Wealth Trading Mean Reversion

#### 2.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📈 Generating Wealth Trading Mean Reversion


 
##### OU Stochastic Differential Equation
 $$
 dX_t = \theta (\mu - X_t)\,dt + \sigma\,dW_t
 $$
 - $X_t$: Price/process value at time $t$
 - $\mu$: Long-run mean
 - $\theta$: Mean reversion speed ($\theta > 0$)
 - $\sigma$: Volatility
 - $dW_t$: Standard Brownian motion increment
 
##### Discretized OU Process (Euler Approximation)
 $$
 X_{t+1} = X_t + \theta (\mu - X_t)\,\Delta t + \sigma \sqrt{\Delta t} \cdot \epsilon_t
 $$
 - $\epsilon_t \sim N(0,1)$ (i.i.d. standard normal)
 - $\Delta t$: Time increment
 
##### Stationary Distribution
 $$
 X_t \sim \mathcal{N}\left(\mu,\ \frac{\sigma^2}{2\theta}\right)
 $$
 
##### Basic Mean Reversion Trading Rules
 
 1. **Estimate Mean:**
    - Compute sample mean $\hat{\mu}$ from a historical window:
      $$
      \hat{\mu} = \frac{1}{N} \sum_{i=1}^N X_i
      $$
 
 2. **Define Trading Bands:**
    - Upper band: $\hat{\mu} + k \cdot \widehat{\sigma}_{\mathrm{stat}}$
    - Lower band: $\hat{\mu} - k \cdot \widehat{\sigma}_{\mathrm{stat}}$
    - Where $k$ is a threshold multiplier, typically $k \sim 1$.
    - $\widehat{\sigma}_{\mathrm{stat}} = \frac{\sigma}{\sqrt{2\theta}}$
 
 3. **Trading Logic:**
    - If $X_t >$ Upper Band: _Enter short_ $(\text{sell})$.
    - If $X_t <$ Lower Band: _Enter long_ $(\text{buy})$.
    - Close positions when $X_t$ returns to $\hat{\mu}$.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_samples = 30     # LOW SAMPLE SIZE: Forces a bad estimate of the mean
n_steps = 400      # EXTENDED HORIZON: Shows the full bleed-out effect
total_steps = n_samples + n_steps

# OU Process Parameters
true_mu = 100.0    # True long-run mean
theta = 0.05       # Mean reversion speed
sigma = 1.2        # Volatility
dt = 1.0

# Calculate theoretical stationary standard deviation
stat_dev = sigma / np.sqrt(2 * theta)

np.random.seed(42) # Yields an initial sample mean significantly above 100

# ==========================================
# --- PHASE 1 & 2: DATA GENERATION ---
# ==========================================
X = np.zeros(total_steps)
X[0] = 106.0  # Start artificially high to skew the small-sample estimate

# Generate the full Ornstein-Uhlenbeck path
for t in range(1, total_steps):
    dW = np.random.randn() * np.sqrt(dt)
    X[t] = X[t-1] + theta * (true_mu - X[t-1]) * dt + sigma * dW

# Phase 1: Estimation
phase1_data = X[:n_samples]
rolling_means = np.cumsum(phase1_data) / np.arange(1, n_samples + 1)
final_estimated_mu = rolling_means[-1]

# Define Trading Bands (1 Stationary Standard Deviation)
band_width = stat_dev * 0.8
upper_band = final_estimated_mu + band_width
lower_band = final_estimated_mu - band_width

# Generate Theoretical Distribution for the Order Book / Profiler
all_prices = np.round(np.arange(true_mu - 4*stat_dev, true_mu + 6*stat_dev, 0.5), 1)
all_vols = np.exp(-0.5 * ((all_prices - true_mu) / stat_dev)**2)

base_colors = []
for p in all_prices:
    if p < true_mu:
        base_colors.append('rgba(0, 150, 0, 0.6)')  # True Undervalued
    elif p > true_mu:
        base_colors.append('rgba(150, 0, 0, 0.6)')  # True Overvalued
    else:
        base_colors.append('rgba(150, 150, 150, 0.6)')

# ==========================================
# --- PHASE 2: TRADING SIMULATION ---
# ==========================================
positions = np.zeros(n_steps)
profits = np.zeros(n_steps)
current_pos = 0  # 1 for Long, -1 for Short, 0 for Flat

trade_markers_x = []
trade_markers_y = []
trade_markers_color = []

for i in range(n_steps):
    t = n_samples + i
    price = X[t]
    prev_price = X[t-1]
    
    # Calculate Mark-to-Market Profit from the previous position
    if i > 0:
        profits[i] = current_pos * (price - prev_price)
    
    # Trading Logic
    if current_pos == 0:
        if price > upper_band:
            current_pos = -1  # Enter Short
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('red')
        elif price < lower_band:
            current_pos = 1   # Enter Long
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('green')
    elif current_pos == 1:
        if price >= final_estimated_mu:
            current_pos = 0   # Close Long
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('gray')
    elif current_pos == -1:
        if price <= final_estimated_mu:
            current_pos = 0   # Close Short
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('gray')
            
    positions[i] = current_pos

cum_profits = np.cumsum(profits)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []

for k in range(total_frames := total_steps):
    if k <= n_samples:
        t1_x = np.arange(k)
        t1_y = X[:k]
        roll_y = rolling_means[:k]
        current_est = rolling_means[k-1] if k > 0 else X[0]
        
        # Keep Phase 2 plots empty during Phase 1
        t2_x, t2_y = [0], [current_est]
        eq_x, eq_y = [0], [0]
        mark_x, mark_y, mark_c = [0], [current_est], ['rgba(0,0,0,0)']
        
        plot_upper, plot_lower = current_est, current_est
        frame_name = f"est{k}"
    else:
        t_trad = k - n_samples
        t1_x = np.arange(n_samples)
        t1_y = phase1_data
        roll_y = rolling_means
        current_est = final_estimated_mu
        
        t2_x = np.arange(t_trad)
        t2_y = X[n_samples : k]
        eq_x = np.arange(t_trad)
        eq_y = cum_profits[:t_trad]
        
        valid_marks = [idx for idx, mx in enumerate(trade_markers_x) if mx < t_trad]
        mark_x = [trade_markers_x[idx] for idx in valid_marks] if valid_marks else [0]
        mark_y = [trade_markers_y[idx] for idx in valid_marks] if valid_marks else [current_est]
        mark_c = [trade_markers_color[idx] for idx in valid_marks] if valid_marks else ['rgba(0,0,0,0)']
        
        plot_upper, plot_lower = upper_band, lower_band
        frame_name = f"trade{t_trad}"

    current_price = X[k-1] if k > 0 else X[0]

    # --- Trace Definitions ---
    # Phase 1
    tr0 = go.Scatter(x=t1_x, y=t1_y, mode='lines', line=dict(color='#00ffff', width=2))
    tr1 = go.Scatter(x=t1_x, y=roll_y, mode='lines', line=dict(color='yellow', width=3))
    
    # PDF
    tr2 = go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors))
    tr3 = go.Scatter(x=[true_mu, true_mu], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash'))
    tr4 = go.Scatter(x=[current_est, current_est], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'))
    tr5 = go.Scatter(x=[plot_upper, plot_upper], y=[0, max(all_vols)], mode='lines', line=dict(color='red', width=2))
    tr6 = go.Scatter(x=[plot_lower, plot_lower], y=[0, max(all_vols)], mode='lines', line=dict(color='green', width=2))
    tr7 = go.Scatter(x=[current_price], y=[max(all_vols)/2], mode='markers', marker=dict(color='white', size=12, symbol='diamond'))
    
    # Phase 2
    tr8 = go.Scatter(x=t2_x, y=t2_y, mode='lines', line=dict(color='#00ffff', width=2))
    tr9 = go.Scatter(x=[0, n_steps], y=[true_mu, true_mu], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash'))
    tr10 = go.Scatter(x=[0, n_steps], y=[current_est, current_est], mode='lines', line=dict(color='yellow', width=2, dash='dash'))
    tr11 = go.Scatter(x=[0, n_steps], y=[plot_upper, plot_upper], mode='lines', line=dict(color='red', width=1, dash='dot'))
    tr12 = go.Scatter(x=[0, n_steps], y=[plot_lower, plot_lower], mode='lines', line=dict(color='green', width=1, dash='dot'))
    tr13 = go.Scatter(x=mark_x, y=mark_y, mode='markers', marker=dict(color=mark_c, size=10))
    
    # Equity
    tr14 = go.Scatter(x=eq_x, y=eq_y, mode='lines', line=dict(color='magenta', width=3, shape='hv'))

    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8, tr9, tr10, tr11, tr12, tr13, tr14], name=frame_name))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Phase 1: 30-Point Sample & Estimated Mean", "Phase 2: Mean Reversion Trading", 
        "True OU PDF vs Estimated Trading Bands", "Phase 2: Equity Curve (PnL)"
    ],
    horizontal_spacing=0.10, vertical_spacing=0.15
)

# Phase 1
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='#00ffff')), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='yellow')), row=1, col=1)

# PDF
fig.add_trace(go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors)), row=2, col=1)
fig.add_trace(go.Scatter(x=[true_mu, true_mu], y=[0, 1], line=dict(color='rgba(255,255,255,0.4)', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='yellow', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color='white')), row=2, col=1)

# Phase 2 (Note the newly added lines)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='#00ffff')), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[true_mu, true_mu], line=dict(color='rgba(255,255,255,0.4)', dash='dash'), name="True Mean"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='yellow', dash='dash'), name="Est Mean"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='red', dash='dot'), name="Upper Band"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='green', dash='dot'), name="Lower Band"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color='rgba(0,0,0,0)')), row=1, col=2)

# Equity
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='magenta', shape='hv')), row=2, col=2)

fig.frames = frames

# ==========================================
# --- SLIDER & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.1,
    steps=[
        dict(
            method="animate",
            args=[[frames[k].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=f"Est {k}" if k <= n_samples else f"Trade {k - n_samples}"
        ) for k in range(total_frames)
    ]
)]

fig.update_layout(
    height=750, width=1100,
    title_text="Mean Reversion Trap: Long-Term Bleed from Structural Bias",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'), showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'fromcurrent': True}]}]
    }]
)

# Axes Styling
fig.update_xaxes(title_text='Sample Step', range=[0, n_samples], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Price', range=[92, 110], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Price', range=[92, 110], row=2, col=1, showgrid=False)
fig.update_yaxes(title_text='PDF Mass', range=[0, max(all_vols) * 1.1], row=2, col=1, showgrid=False)

fig.update_xaxes(title_text='Trade Step', range=[0, n_steps], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Price', range=[92, 110], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

min_pnl, max_pnl = min(-10, np.min(cum_profits) * 1.1), max(5, np.max(cum_profits) * 1.1)
fig.update_xaxes(title_text='Trade Step', range=[0, n_steps], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative PnL', range=[min_pnl, max_pnl], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()
```




    The Kernel crashed while executing code in the current cell or a previous cell. 


    Please review the code in the cell(s) to identify a possible cause of the failure. 


    Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 


    View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.


---

#### 2.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- There is no long-term mean price of a stock, 

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

- Live Kalman Filter with Interactive Brokers
- How to Backtest a Trading Strategy with Interactive Brokers
- Algorithmic Volatility Trading System
