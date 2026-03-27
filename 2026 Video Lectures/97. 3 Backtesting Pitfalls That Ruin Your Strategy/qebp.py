# ### 📈 3 Backtesting Pitfalls That Will Ruin Your Strategy
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
# 
# - [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)
# 
# - [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)
# 
# - [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/wZChBKDFFeU)
# 
# - [Quant vs. Discretionary Trading](https://youtu.be/3gblERSSHXI)
# 
# - [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)
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
# #### 1. ) 👀 Look-Ahead Bias
# 
# - Strategies adapted to a filtration
# 
# - Two-Envelope Analogy
# 
# #### 2.) 🔎 Data Snooping, Overfitting, P-Hacking
# 
# - Pulling Levers for Results
# 
# - Overfit, Underfit, and Robust Models
# 
# #### 3.) ✈️ Survivorship Bias
# 
# - What Isn't the Data Telling You
# 
# - Where It Matters in Finance
# 
# #### 4.) 🎖️ Honorable Mentions
# 
# - Spread
# 
# - Transaction Costs
# 
# - Slippage
# 
# #### 5.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1. ) 👀 Look-Ahead Bias
# 
# The most common pitfall, especially in the age of AI, is look ahead bias
# 
# Accidentally using information not measurable with respect to the filtration $\{\mathcal{F}_t\}_{t \geq 0}$
# 
# 💡 **The "Two-Envelope" Analogy**
# 
# - **The $\mathcal{F}_t$ Envelope:**  
#   This contains everything that has happened up to and including time $t$—all available market data, such as prices, volume, and news.
# 
# - **The $\mathcal{F}_{t+1}$ Envelope:**  
#   This contains future information, for example, tomorrow’s closing price at time $t+1$. At time $t$, this information is not available.
# 
# - **The Bias:**  
#   If your code "peeks" into the $\mathcal{F}_{t+1}$ envelope while making a decision at time $t$—that is, if it uses information from the future—you have broken the causality of the filtration. This is the essence of look-ahead bias in backtesting.
# 
# Depending on the data, and how you engineer your backtest (did you write it or use AI), it may be difficult to catch and performance may be inflated
# 
# In other words, look-ahead bias may explain enough variation to make the strategy appear viable but not enough to make it appear there is a problem


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('spx.csv')

# Ensure Date columns are parsed correctly
df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S')
df = df.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# THE CRYSTAL BALL FEATURE (Engineered Look-Ahead Bias)
# Peeks 35 hours (roughly 1 week) into the FUTURE.
# ---------------------------------------------------------
df['LookAhead_Signal'] = df['Close'].shift(-35) / df['Close'] - 1

# Forward 1-Day Return (7 hours) for the Monotonic Distribution Chart
df['Forward_1D_Return'] = df['Close'].shift(-7) / df['Close'] - 1
df['Hourly_Return'] = df['Close'].pct_change()

# Drop NaNs created by the forward shifts
df = df.dropna().reset_index(drop=True)

# Pre-calculate signal quantiles (5 buckets / quintiles) to build the monotonic distribution chart
df['Signal_Quantile'] = pd.qcut(df['LookAhead_Signal'], 5, labels=False, duplicates='drop')

n_steps = len(df) 

in_position = False
entry_price = 0.0
buy_indices = []  
sell_indices = []
equity_curve = [1.0]

last_entry_idx = -1
trade_duration = 7 # Trade every day or so (hold for 7 hourly bars)

# Simulate the Strategy
for i in range(n_steps):
    current_price = df['Close'].iloc[i]
    strat_ret = 0.0
    
    if in_position:
        strat_ret = df['Hourly_Return'].iloc[i] 
            
        # Exit Conditions: Held for exactly 1 day (7 hours)
        if i - last_entry_idx >= trade_duration:
            in_position = False
            sell_indices.append(i)
            
    else:
        # Entry Condition: Evaluate signal
        signal = df['LookAhead_Signal'].iloc[i]
        
        # If our "predictive" signal is strictly positive, buy
        if signal > 0:
            in_position = True
            entry_price = current_price
            last_entry_idx = i
            buy_indices.append(i)
            
    equity_curve.append(equity_curve[-1] * (1 + strat_ret))

equity_curve = equity_curve[1:]

# ==========================================
# --- 2. CALCULATE METRICS ---
# ==========================================
# Sharpe ratios (annualized, assuming 252 days * 7 hours/day = 1764 periods)
periods_per_year = 252 * 7

strat_series = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(periods_per_year) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0

spx_series = df['Hourly_Return'].dropna()
spx_sharpe = np.sqrt(periods_per_year) * spx_series.mean() / spx_series.std() if spx_series.std() != 0 else 0

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_array = df['Date'].values
closes_array = df['Close'].values
signal_quantile_array = df['Signal_Quantile'].values
fwd_ret_array = df['Forward_1D_Return'].values
buy_idx_arr = np.array(buy_indices)
sell_idx_arr = np.array(sell_indices)
equity_array = np.array(equity_curve)

step_size = max(1, n_steps // 200) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps:
    frame_indices.append(n_steps)

for k in frame_indices:
    t_x = dates_array[:k]
    
    # Left Column: Price & Arrows
    tr_price = go.Scatter(x=t_x, y=closes_array[:k], mode='lines', line=dict(color='white', width=1.5), name='Close')
    
    b_mask = buy_idx_arr < k
    s_mask = sell_idx_arr < k
    tr_buy = go.Scatter(x=dates_array[buy_idx_arr[b_mask]], y=closes_array[buy_idx_arr[b_mask]], mode='markers', marker=dict(color='#39ff14', size=12, symbol='triangle-up'), name='Buy')
    tr_sell = go.Scatter(x=dates_array[sell_idx_arr[s_mask]], y=closes_array[sell_idx_arr[s_mask]], mode='markers', marker=dict(color='#ff3333', size=12, symbol='triangle-down'), name='Sell')
    
    # Right Column (Top): Monotonic Distribution (Quantile vs Mean Fwd Return)
    quantile_means = []
    num_quantiles = 5
    for q in range(num_quantiles):
        mask = (signal_quantile_array[:k] == q)
        if np.any(mask):
            quantile_means.append(np.mean(fwd_ret_array[:k][mask]))
        else:
            quantile_means.append(0)
            
    tr_dist = go.Bar(x=[f"Q{q+1}" for q in range(num_quantiles)], y=quantile_means, marker_color='#00ffff', name='Avg Fwd Ret')

    # Right Column (Bottom): Strategy Equity 
    tr_equity = go.Scatter(x=t_x, y=equity_array[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy')

    # Passing exactly 5 traces per frame
    frames.append(go.Frame(data=[tr_price, tr_buy, tr_sell, tr_dist, tr_equity], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        "SPX Price Path & Trades", 
        "Signal Quantiles vs. Avg Fwd Return",
        "Strategy Equity"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

# Dummy traces to match the exact order of the 5 traces in our frames
for _ in range(3): fig.add_trace(go.Scatter(x=[dates_array[0]], y=[closes_array[0]]), row=1, col=1)
fig.add_trace(go.Bar(x=[f"Q{q+1}" for q in range(5)], y=[0]*5), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[equity_array[0]]), row=2, col=2)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

# Set plots and paper to completely transparent and adjust margins (removed right-side padding)
fig.update_layout(
    height=800, width=1100,
    title_text="Quantitative Strategy: Look-Ahead Feature Distribution & Execution",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), # Greatly reduced right margin 'r'
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
min_price, max_price = closes_array.min() * 0.95, closes_array.max() * 1.05
min_eq = equity_array.min() * 0.95
max_eq = equity_array.max() * 1.05

# Global min/max for the Bar chart Y-axis so it doesn't jump around
global_quantile_means = df.groupby('Signal_Quantile')['Forward_1D_Return'].mean()
min_dist = min(0, global_quantile_means.min() * 1.2)
max_dist = max(0, global_quantile_means.max() * 1.2)

# Row 1, Col 1 (SPX Price)
fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price', range=[min_price, max_price], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

# Row 1, Col 2 (Monotonic Distribution)
fig.update_xaxes(title_text='Signal Quantile (5 buckets)', row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Avg 1D Fwd Return', range=[min_dist, max_dist], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

# Row 2, Col 2 (Equity Curve)
fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# --- ADD SHARPE RATIO OVERLAYS ---

# Overlay SPX Sharpe on the Left Chart
fig.add_annotation(
    text=f"<b>SPX Sharpe: {spx_sharpe:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="white", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ffffff", borderwidth=1, borderpad=6,
    row=1, col=1
)

# Overlay Strategy Sharpe on the Bottom Right Chart
fig.add_annotation(
    text=f"<b>Strategy Sharpe: {sharpe_ratio:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=2, col=2
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('spx.csv')

# Ensure Date columns are parsed correctly
df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S')
df = df.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# REALITY CHECK (Out-of-Sample Feature)
# Replaced look-ahead bias with a trailing 35-hour (1 week) momentum feature.
# ---------------------------------------------------------
df['Trailing_Signal'] = df['Close'] / df['Close'].shift(35) - 1

# Forward 1-Day Return (7 hours) for the Monotonic Distribution Chart
df['Forward_1D_Return'] = df['Close'].shift(-7) / df['Close'] - 1
df['Hourly_Return'] = df['Close'].pct_change()

# Drop NaNs created by the shifts
df = df.dropna().reset_index(drop=True)

# Calculate signal quantiles (5 buckets / quintiles) to build the distribution chart
# Note: For strict walk-forward, thresholds should be rolling, but calculating 
# across the set here will still vividly show the collapse of the edge.
df['Signal_Quantile'] = pd.qcut(df['Trailing_Signal'], 5, labels=False, duplicates='drop')

n_steps = len(df) 

in_position = False
entry_price = 0.0
buy_indices = []  
sell_indices = []
equity_curve = [1.0]

last_entry_idx = -1
trade_duration = 7 # Trade every day or so (hold for 7 hourly bars)

# Simulate the Strategy
for i in range(n_steps):
    current_price = df['Close'].iloc[i]
    strat_ret = 0.0
    
    if in_position:
        strat_ret = df['Hourly_Return'].iloc[i] 
            
        # Exit Conditions: Held for exactly 1 day (7 hours)
        if i - last_entry_idx >= trade_duration:
            in_position = False
            sell_indices.append(i)
            
    else:
        # Entry Condition: Evaluate out-of-sample signal
        signal = df['Trailing_Signal'].iloc[i]
        
        # If past momentum is positive, buy
        if signal > 0:
            in_position = True
            entry_price = current_price
            last_entry_idx = i
            buy_indices.append(i)
            
    equity_curve.append(equity_curve[-1] * (1 + strat_ret))

equity_curve = equity_curve[1:]

# ==========================================
# --- 2. CALCULATE METRICS ---
# ==========================================
# Sharpe ratios (annualized, assuming 252 days * 7 hours/day = 1764 periods)
periods_per_year = 252 * 7

strat_series = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(periods_per_year) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0

spx_series = df['Hourly_Return'].dropna()
spx_sharpe = np.sqrt(periods_per_year) * spx_series.mean() / spx_series.std() if spx_series.std() != 0 else 0

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_array = df['Date'].values
closes_array = df['Close'].values
signal_quantile_array = df['Signal_Quantile'].values
fwd_ret_array = df['Forward_1D_Return'].values
buy_idx_arr = np.array(buy_indices)
sell_idx_arr = np.array(sell_indices)
equity_array = np.array(equity_curve)

step_size = max(1, n_steps // 200) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps:
    frame_indices.append(n_steps)

for k in frame_indices:
    t_x = dates_array[:k]
    
    # Left Column: Price & Arrows
    tr_price = go.Scatter(x=t_x, y=closes_array[:k], mode='lines', line=dict(color='white', width=1.5), name='Close')
    
    b_mask = buy_idx_arr < k
    s_mask = sell_idx_arr < k
    tr_buy = go.Scatter(x=dates_array[buy_idx_arr[b_mask]], y=closes_array[buy_idx_arr[b_mask]], mode='markers', marker=dict(color='#39ff14', size=12, symbol='triangle-up'), name='Buy')
    tr_sell = go.Scatter(x=dates_array[sell_idx_arr[s_mask]], y=closes_array[sell_idx_arr[s_mask]], mode='markers', marker=dict(color='#ff3333', size=12, symbol='triangle-down'), name='Sell')
    
    # Right Column (Top): Distribution (Quantile vs Mean Fwd Return)
    quantile_means = []
    num_quantiles = 5
    for q in range(num_quantiles):
        mask = (signal_quantile_array[:k] == q)
        if np.any(mask):
            quantile_means.append(np.mean(fwd_ret_array[:k][mask]))
        else:
            quantile_means.append(0)
            
    tr_dist = go.Bar(x=[f"Q{q+1}" for q in range(num_quantiles)], y=quantile_means, marker_color='#00ffff', name='Avg Fwd Ret')

    # Right Column (Bottom): Strategy Equity 
    tr_equity = go.Scatter(x=t_x, y=equity_array[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy')

    # Passing exactly 5 traces per frame
    frames.append(go.Frame(data=[tr_price, tr_buy, tr_sell, tr_dist, tr_equity], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        "SPX Price Path & Trades", 
        "Trailing Signal Quantiles vs. Avg Fwd Return",
        "Out-of-Sample Strategy Equity"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

for _ in range(3): fig.add_trace(go.Scatter(x=[dates_array[0]], y=[closes_array[0]]), row=1, col=1)
fig.add_trace(go.Bar(x=[f"Q{q+1}" for q in range(5)], y=[0]*5), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[equity_array[0]]), row=2, col=2)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Quantitative Strategy: Reality Check (Out-of-Sample Trailing Feature)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
min_price, max_price = closes_array.min() * 0.95, closes_array.max() * 1.05
min_eq = equity_array.min() * 0.95
max_eq = equity_array.max() * 1.05

# Global min/max for the Bar chart Y-axis (will likely be much smaller now!)
global_quantile_means = df.groupby('Signal_Quantile')['Forward_1D_Return'].mean()
min_dist = min(0, global_quantile_means.min() * 1.2)
max_dist = max(0, global_quantile_means.max() * 1.2)

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price', range=[min_price, max_price], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Signal Quantile (5 buckets)', row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Avg 1D Fwd Return', range=[min_dist, max_dist], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# --- ADD SHARPE RATIO OVERLAYS ---
fig.add_annotation(
    text=f"<b>SPX Sharpe: {spx_sharpe:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="white", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ffffff", borderwidth=1, borderpad=6,
    row=1, col=1
)

# Notice how this Strategy Sharpe will look vastly different
fig.add_annotation(
    text=f"<b>Strategy Sharpe: {sharpe_ratio:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=2, col=2
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### 📉 Avoiding Look-Ahead Bias
# 
# - At each time t, build all features and signals using only information available up to and including t (never use future data).
# - Do not use future returns or any statistics that "peek ahead" beyond time t.
# - When calculating rolling or trailing features, ensure they only include data at or before t.
# - Shift all target variables (such as forward returns) so they are never visible to the model at the time of prediction.
# - Perform any signal bucketing, normalization, or ranking strictly with past and present data.
# - Strictly separate training and testing data in time order—do not mix information from the future into training or feature engineering steps.


# ---


# #### 2.) 🔎 Data Snooping, Overfitting, P-Hacking
# 
# The classic pitfall, especially in the age of data-driven investing, is data snooping and overfitting
# 
# Repeatedly tweaking or testing your strategy on the same dataset until you find something "good," without correctly validating its forecasting power.
# 
# 💡 **The "Parameter Tuning" Analogy**
# 
# Suppose you try hundreds of moving average combinations—  
# 
# - MA1 = 5, MA2 = 10 → Sharpe: 0.75  
# 
# - MA1 = 7, MA2 = 12 → Sharpe: 1.20  
# 
# - MA1 = 13, MA2 = 30 → Sharpe: 1.35  
# 
# - ...
# 
# ⚠️ **The Bias:**  
# This is data snooping (also known as p-hacking or overfitting): searching for the best results within the same dataset until you get impressive statistics.  
# 
# Without proper validation on new, unseen data, the results are unreliable and are to disappoint in practice.  You haven't actually "learned" anything from your data, your objective is to find a reasonably stable expectation (distribution), spamming different parameter sets won't (necessarily) yield this.


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 0. STRATEGY CONFIGURATION ---
# ==========================================
STRAT1_FAST_MA = 3
STRAT1_SLOW_MA = 5

STRAT2_FAST_MA = 7
STRAT2_SLOW_MA = 12

# Maximum rows to animate (Prevents browser/notebook crashes)
MAX_ROWS = 3000 
TARGET_FRAMES = 80

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('spx.csv')

# 1. Rename and parse dates FIRST to avoid string matching errors
df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S', errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df = df[df['Date'].dt.year == 2025].reset_index(drop=True)

# 2. Slice to the most recent data to prevent Plotly from freezing
if len(df) > MAX_ROWS:
    df = df.tail(MAX_ROWS).reset_index(drop=True)

if len(df) == 0:
    raise ValueError("DataFrame is empty! Check your CSV data.")

df['Hourly_Return'] = df['Close'].pct_change()

# Calculate Moving Averages for both strategies
df['S1_Fast'] = df['Close'].rolling(window=STRAT1_FAST_MA).mean()
df['S1_Slow'] = df['Close'].rolling(window=STRAT1_SLOW_MA).mean()

df['S2_Fast'] = df['Close'].rolling(window=STRAT2_FAST_MA).mean()
df['S2_Slow'] = df['Close'].rolling(window=STRAT2_SLOW_MA).mean()

# Drop NaNs to start clean
df = df.dropna().reset_index(drop=True)

n_steps = len(df) 

# State variables for Strategy 1
in_pos1 = False
buy_idx1 = []  
sell_idx1 = []
eq_curve1 = [1.0]

# State variables for Strategy 2
in_pos2 = False
buy_idx2 = []  
sell_idx2 = []
eq_curve2 = [1.0]

# Simulate Both Strategies simultaneously
for i in range(1, n_steps):
    current_price = df['Close'].iloc[i]
    ret_h = df['Hourly_Return'].iloc[i] 
    
    # ---- Strategy 1 Logic (MA Crossover) ----
    s1_ret = 0.0
    s1_fast_prev, s1_fast_curr = df['S1_Fast'].iloc[i-1], df['S1_Fast'].iloc[i]
    s1_slow_prev, s1_slow_curr = df['S1_Slow'].iloc[i-1], df['S1_Slow'].iloc[i]
    
    if in_pos1:
        s1_ret = ret_h
        # Exit Condition: Fast crosses below Slow
        if s1_fast_curr < s1_slow_curr and s1_fast_prev >= s1_slow_prev:
            in_pos1 = False
            sell_idx1.append(i)
    else:
        # Entry Condition: Fast crosses above Slow
        if s1_fast_curr > s1_slow_curr and s1_fast_prev <= s1_slow_prev:
            in_pos1 = True
            buy_idx1.append(i)
    eq_curve1.append(eq_curve1[-1] * (1 + s1_ret))

    # ---- Strategy 2 Logic (MA Crossover) ----
    s2_ret = 0.0
    s2_fast_prev, s2_fast_curr = df['S2_Fast'].iloc[i-1], df['S2_Fast'].iloc[i]
    s2_slow_prev, s2_slow_curr = df['S2_Slow'].iloc[i-1], df['S2_Slow'].iloc[i]
    
    if in_pos2:
        s2_ret = ret_h
        # Exit Condition: Fast crosses below Slow
        if s2_fast_curr < s2_slow_curr and s2_fast_prev >= s2_slow_prev:
            in_pos2 = False
            sell_idx2.append(i)
    else:
        # Entry Condition: Fast crosses above Slow
        if s2_fast_curr > s2_slow_curr and s2_fast_prev <= s2_slow_prev:
            in_pos2 = True
            buy_idx2.append(i)
    eq_curve2.append(eq_curve2[-1] * (1 + s2_ret))

eq_curve1 = np.array(eq_curve1)
eq_curve2 = np.array(eq_curve2)

# ==========================================
# --- 2. CALCULATE METRICS ---
# ==========================================
periods_per_year = 252 * 7

strat1_series = pd.Series(eq_curve1).pct_change().dropna()
sharpe1 = np.sqrt(periods_per_year) * strat1_series.mean() / strat1_series.std() if strat1_series.std() != 0 else 0

strat2_series = pd.Series(eq_curve2).pct_change().dropna()
sharpe2 = np.sqrt(periods_per_year) * strat2_series.mean() / strat2_series.std() if strat2_series.std() != 0 else 0

spx_series = df['Hourly_Return'].dropna()
spx_sharpe = np.sqrt(periods_per_year) * spx_series.mean() / spx_series.std() if spx_series.std() != 0 else 0

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_arr = df['Date'].values
closes_arr = df['Close'].values

# S1 Arrays
s1_fast_arr = df['S1_Fast'].values
s1_slow_arr = df['S1_Slow'].values
b_idx1_arr = np.array(buy_idx1)
s_idx1_arr = np.array(sell_idx1)

# S2 Arrays
s2_fast_arr = df['S2_Fast'].values
s2_slow_arr = df['S2_Slow'].values
b_idx2_arr = np.array(buy_idx2)
s_idx2_arr = np.array(sell_idx2)

step_size = max(1, n_steps // TARGET_FRAMES) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps - 1:
    frame_indices.append(n_steps - 1)

for k in frame_indices:
    t_x = dates_arr[:k]
    prices = closes_arr[:k]
    
    # STRATEGY 1 TRACES
    tr_p1 = go.Scatter(x=t_x, y=prices, mode='lines', line=dict(color='white', width=1.5), name='Close')
    tr_s1_f = go.Scatter(x=t_x, y=s1_fast_arr[:k], mode='lines', line=dict(color='#00ffff', width=1.5), name=f'{STRAT1_FAST_MA} MA')
    tr_s1_s = go.Scatter(x=t_x, y=s1_slow_arr[:k], mode='lines', line=dict(color='#ff00ff', width=1.5), name=f'{STRAT1_SLOW_MA} MA')
    m_b1, m_s1 = b_idx1_arr < k, s_idx1_arr < k
    tr_b1 = go.Scatter(x=dates_arr[b_idx1_arr[m_b1]], y=closes_arr[b_idx1_arr[m_b1]], mode='markers', marker=dict(color='#39ff14', size=12, symbol='triangle-up'), name='Buy')
    tr_s1 = go.Scatter(x=dates_arr[s_idx1_arr[m_s1]], y=closes_arr[s_idx1_arr[m_s1]], mode='markers', marker=dict(color='#ff3333', size=12, symbol='triangle-down'), name='Sell')
    
    # STRATEGY 2 TRACES 
    tr_p2 = go.Scatter(x=t_x, y=prices, mode='lines', line=dict(color='white', width=1.5), name='Close')
    tr_s2_f = go.Scatter(x=t_x, y=s2_fast_arr[:k], mode='lines', line=dict(color='#00ffff', width=1.5), name=f'{STRAT2_FAST_MA} MA')
    tr_s2_s = go.Scatter(x=t_x, y=s2_slow_arr[:k], mode='lines', line=dict(color='#ff00ff', width=1.5), name=f'{STRAT2_SLOW_MA} MA')
    m_b2, m_s2 = b_idx2_arr < k, s_idx2_arr < k
    tr_b2 = go.Scatter(x=dates_arr[b_idx2_arr[m_b2]], y=closes_arr[b_idx2_arr[m_b2]], mode='markers', marker=dict(color='#39ff14', size=12, symbol='triangle-up'), name='Buy')
    tr_s2 = go.Scatter(x=dates_arr[s_idx2_arr[m_s2]], y=closes_arr[s_idx2_arr[m_s2]], mode='markers', marker=dict(color='#ff3333', size=12, symbol='triangle-down'), name='Sell')

    # EQUITY TRACES 
    tr_eq1 = go.Scatter(x=t_x, y=eq_curve1[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='S1 Equity')
    tr_eq2 = go.Scatter(x=t_x, y=eq_curve2[:k], mode='lines', line=dict(color='#ff00ff', width=2), fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)', name='S2 Equity')

    frames.append(go.Frame(data=[
        tr_p1, tr_s1_f, tr_s1_s, tr_b1, tr_s1, 
        tr_p2, tr_s2_f, tr_s2_s, tr_b2, tr_s2, 
        tr_eq1, tr_eq2
    ], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        f"Strategy 1 ({STRAT1_FAST_MA}/{STRAT1_SLOW_MA} MA Crossover)", 
        f"Strategy 2 ({STRAT2_FAST_MA}/{STRAT2_SLOW_MA} MA Crossover)",
        "Strategy 1 Equity Curve",
        "Strategy 2 Equity Curve"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

for _ in range(5): fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[closes_arr[0]]), row=1, col=1)
for _ in range(5): fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[closes_arr[0]]), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[eq_curve1[0]]), row=2, col=1)
fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[eq_curve2[0]]), row=2, col=2)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=900, width=1100,
    title_text="Moving Average Strategy Comparison (Optimized Performance)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

min_price, max_price = closes_arr.min() * 0.95, closes_arr.max() * 1.05
min_eq1, max_eq1 = eq_curve1.min() * 0.95, eq_curve1.max() * 1.05
min_eq2, max_eq2 = eq_curve2.min() * 0.95, eq_curve2.max() * 1.05

fig.update_xaxes(range=[dates_arr[0], dates_arr[-1]], gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price', range=[min_price, max_price], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Price', range=[min_price, max_price], gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq1, max_eq1], gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq2, max_eq2], gridcolor='rgba(128,128,128,0.2)', row=2, col=2)

fig.add_annotation(text=f"<b>SPX Sharpe: {spx_sharpe:.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="white", size=14), bgcolor="rgba(0,0,0,0.6)", bordercolor="#ffffff", borderwidth=1, borderpad=6, row=1, col=1)
fig.add_annotation(text=f"<b>SPX Sharpe: {spx_sharpe:.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="white", size=14), bgcolor="rgba(0,0,0,0.6)", bordercolor="#ffffff", borderwidth=1, borderpad=6, row=1, col=2)
fig.add_annotation(text=f"<b>S1 Sharpe: {sharpe1:.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="#39ff14", size=14), bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6, row=2, col=1)
fig.add_annotation(text=f"<b>S2 Sharpe: {sharpe2:.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="#ff00ff", size=14), bgcolor="rgba(0,0,0,0.6)", bordercolor="#ff00ff", borderwidth=1, borderpad=6, row=2, col=2)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### If we trade out of sample with this parameter set observe what happens


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 0. STRATEGY CONFIGURATION ---
# ==========================================
STRAT_FAST_MA = 7
STRAT_SLOW_MA = 12

# Set to 6000 rows per your request
MAX_ROWS = 3000 
TARGET_FRAMES = 80

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('spx.csv')

# 1. Rename and parse dates FIRST to avoid string matching errors
df = df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S', errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df = df[df['Date'].dt.year == 2026].reset_index(drop=True)

# 2. Slice to the most recent data
if len(df) > MAX_ROWS:
    df = df.tail(MAX_ROWS).reset_index(drop=True)

if len(df) == 0:
    raise ValueError("DataFrame is empty! Check your CSV data.")

df['Hourly_Return'] = df['Close'].pct_change()

# Calculate Moving Averages 
df['Fast_MA'] = df['Close'].rolling(window=STRAT_FAST_MA).mean()
df['Slow_MA'] = df['Close'].rolling(window=STRAT_SLOW_MA).mean()

# Drop NaNs to start clean
df = df.dropna().reset_index(drop=True)

n_steps = len(df) 

# State variables
in_pos = False
buy_idx = []  
sell_idx = []
eq_curve = [1.0]

# Simulate Strategy
for i in range(1, n_steps):
    current_price = df['Close'].iloc[i]
    ret_h = df['Hourly_Return'].iloc[i] 
    
    strat_ret = 0.0
    fast_prev, fast_curr = df['Fast_MA'].iloc[i-1], df['Fast_MA'].iloc[i]
    slow_prev, slow_curr = df['Slow_MA'].iloc[i-1], df['Slow_MA'].iloc[i]
    
    if in_pos:
        strat_ret = ret_h
        # Exit Condition: Fast crosses below Slow
        if fast_curr < slow_curr and fast_prev >= slow_prev:
            in_pos = False
            sell_idx.append(i)
    else:
        # Entry Condition: Fast crosses above Slow
        if fast_curr > slow_curr and fast_prev <= slow_prev:
            in_pos = True
            buy_idx.append(i)
            
    eq_curve.append(eq_curve[-1] * (1 + strat_ret))

eq_curve = np.array(eq_curve)

# ==========================================
# --- 2. CALCULATE METRICS ---
# ==========================================
periods_per_year = 252 * 7

strat_series = pd.Series(eq_curve).pct_change().dropna()
strat_sharpe = np.sqrt(periods_per_year) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0

spx_series = df['Hourly_Return'].dropna()
spx_sharpe = np.sqrt(periods_per_year) * spx_series.mean() / spx_series.std() if spx_series.std() != 0 else 0

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_arr = df['Date'].values
closes_arr = df['Close'].values
fast_arr = df['Fast_MA'].values
slow_arr = df['Slow_MA'].values
b_idx_arr = np.array(buy_idx)
s_idx_arr = np.array(sell_idx)

step_size = max(1, n_steps // TARGET_FRAMES) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps - 1:
    frame_indices.append(n_steps - 1)

for k in frame_indices:
    t_x = dates_arr[:k]
    prices = closes_arr[:k]
    
    # STRATEGY TRACES
    tr_p = go.Scatter(x=t_x, y=prices, mode='lines', line=dict(color='white', width=1.5), name='Close')
    tr_f = go.Scatter(x=t_x, y=fast_arr[:k], mode='lines', line=dict(color='#00ffff', width=1.5), name=f'{STRAT_FAST_MA} MA')
    tr_s = go.Scatter(x=t_x, y=slow_arr[:k], mode='lines', line=dict(color='#ff00ff', width=1.5), name=f'{STRAT_SLOW_MA} MA')
    
    m_b, m_s = b_idx_arr < k, s_idx_arr < k
    tr_buy = go.Scatter(x=dates_arr[b_idx_arr[m_b]], y=closes_arr[b_idx_arr[m_b]], mode='markers', marker=dict(color='#39ff14', size=12, symbol='triangle-up'), name='Buy')
    tr_sell = go.Scatter(x=dates_arr[s_idx_arr[m_s]], y=closes_arr[s_idx_arr[m_s]], mode='markers', marker=dict(color='#ff3333', size=12, symbol='triangle-down'), name='Sell')
    
    # EQUITY TRACE 
    tr_eq = go.Scatter(x=t_x, y=eq_curve[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy Equity')

    # Total 6 traces
    frames.append(go.Frame(data=[tr_p, tr_f, tr_s, tr_buy, tr_sell, tr_eq], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    row_heights=[0.6, 0.4],
    subplot_titles=[
        f"Out-of-Sample Strategy ({STRAT_FAST_MA}/{STRAT_SLOW_MA} MA Crossover)", 
        "Strategy Equity Curve"
    ],
    vertical_spacing=0.15
)

# Row 1 (5 traces)
for _ in range(5): fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[closes_arr[0]]), row=1, col=1)
# Row 2 (1 trace)
fig.add_trace(go.Scatter(x=[dates_arr[0]], y=[eq_curve[0]]), row=2, col=1)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=800, width=1100, # Adjusted dimensions for 1-column layout
    title_text="OOS Strategy Evaluation (6000 Bars)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

min_price, max_price = closes_arr.min() * 0.95, closes_arr.max() * 1.05
min_eq, max_eq = eq_curve.min() * 0.95, eq_curve.max() * 1.05

fig.update_xaxes(range=[dates_arr[0], dates_arr[-1]], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Price', range=[min_price, max_price], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

fig.update_xaxes(range=[dates_arr[0], dates_arr[-1]], gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], gridcolor='rgba(128,128,128,0.2)', row=2, col=1)

# SPX Sharpe on Top Chart
fig.add_annotation(
    text=f"<b>SPX Sharpe: {spx_sharpe:.2f}</b>", 
    xref="x domain", yref="y domain", x=0.02, y=0.98, 
    showarrow=False, align="left", font=dict(color="white", size=14), 
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#ffffff", borderwidth=1, borderpad=6, 
    row=1, col=1
)

# Strategy Sharpe on Bottom Chart
fig.add_annotation(
    text=f"<b>Strategy Sharpe: {strat_sharpe:.2f}</b>", 
    xref="x domain", yref="y domain", x=0.02, y=0.98, 
    showarrow=False, align="left", font=dict(color="#39ff14", size=14), 
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6, 
    row=2, col=1
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### 🚶‍♂️ Walk-Forward Validation
# 
# - Divide your data into sequential training and testing (out-of-sample) periods, preserving time order.
# - For each period, calibrate (fit) your parameters using only the training set — never touching future data.
# - Evaluate model performance on the immediately following (future) test period to simulate "real world" decision making.
# - After each round, roll forward the entire window and repeat, always forecasting one step into the future.
# - This rolling process limits overfitting, imitates live trading, and gives a more honest out-of-sample metric.


# ---


# #### 3.) ✈️ Survivorship Bias
# 
# - 🛩️ **Survivorship Bias:**  
#     - *Famous Analogy:*  
#         - During WWII, analysts mapped bullet holes on returning planes to determine where to add armor.
#         - Statistician Abraham Wald pointed out: you should reinforce the areas *without* damage on returning planes, because planes hit there *did not* make it back.  
#         - **Lesson:** Only analyzing survivors can lead to false conclusions.
# 
# <div align="center">
#     <img src="Survivorship-bias.svg" alt="WW2 Plane Survivorship Bias Diagram" width="400" height="400"/>
# </div>
#  
# - 📈 *Quant Trading Example:*  
#     - If you only backtest your equity strategy on stocks still active today, you ignore firms that went bankrupt or were delisted.
#     - This skews your historical returns, excluding the "losers" you’d have traded in real time.
# 


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. SYNTHETIC DATA GENERATION ---
# ==========================================
np.random.seed(42)

N_ASSETS = 25
T_STEPS = 500
TARGET_FRAMES = 80

# 1. Market Component (Shared drift and volatility)
market_returns = np.random.randn(T_STEPS, 1) * 0.01 + 0.0003

# 2. Latent Factor (Our predictive signal)
factor_scores = np.random.randn(T_STEPS, N_ASSETS)
# Cross-sectionally demean the factor at each time step
factor_scores = factor_scores - factor_scores.mean(axis=1, keepdims=True)

# 3. Idiosyncratic Noise (Random stock behavior)
idio_noise = np.random.randn(T_STEPS, N_ASSETS) * 0.015

# 4. Generate Properly Aligned Forward Returns
forward_returns = np.zeros((T_STEPS, N_ASSETS))
# The return earned between t and t+1 is driven by the factor observed at t
# Multiplier of 0.00075 precisely targets a ~2.7 Sharpe Ratio
forward_returns[1:] = market_returns[1:] + (factor_scores[:-1] * 0.00075) + idio_noise[1:]

# Generate cumulative sample paths for the assets (starts at 1.0)
asset_paths = np.cumprod(1 + forward_returns, axis=0)

# ==========================================
# --- 2. STRATEGY SIMULATION ---
# ==========================================
TOP_N = 5

strategy_returns = []
quantile_fwd_returns = []

# Loop through time, applying the factor at t to the realized return at t+1
for t in range(T_STEPS - 1):
    signal = factor_scores[t]
    realized_ret = forward_returns[t+1]
    
    # Rank assets by factor
    ranks = np.argsort(signal)
    short_idx = ranks[:TOP_N]       # Bottom 5 
    long_idx = ranks[-TOP_N:]       # Top 5 
    
    # Calculate Market Neutral Portfolio Return
    long_ret = np.mean(realized_ret[long_idx])
    short_ret = np.mean(realized_ret[short_idx])
    
    port_ret = long_ret - short_ret
    strategy_returns.append(port_ret)
    
    # Calculate Monotonic Quantile Distribution 
    q_returns = []
    for q in range(5):
        q_idx = ranks[q*5 : (q+1)*5]
        q_returns.append(np.mean(realized_ret[q_idx]))
    quantile_fwd_returns.append(q_returns)

strategy_returns = np.array(strategy_returns)
quantile_fwd_returns = np.array(quantile_fwd_returns)

# Align lengths back to T_STEPS for clean animation
equity_curve = np.cumprod(1 + strategy_returns)
equity_curve = np.insert(equity_curve, 0, 1.0) 
quantile_fwd_returns = np.insert(quantile_fwd_returns, 0, np.zeros(5), axis=0)

# Calculate Sharpe Ratio 
sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
t_array = np.arange(T_STEPS)

step_size = max(1, T_STEPS // TARGET_FRAMES) 
frame_indices = list(range(1, T_STEPS, step_size))
if frame_indices[-1] != T_STEPS - 1:
    frame_indices.append(T_STEPS - 1)

for k in frame_indices:
    frame_data = []
    
    # LEFT COLUMN: Asset Sample Paths 
    for i in range(N_ASSETS):
        frame_data.append(
            go.Scatter(x=t_array[:k], y=asset_paths[:k, i], mode='lines', 
                       line=dict(width=1, color='rgba(255,255,255,0.3)'), 
                       showlegend=False)
        )
    
    # RIGHT COLUMN TOP: Expanding Quantile Bar Chart
    # Use max(1, k) to avoid mean of empty slice warning on step 0
    expanding_q_mean = np.mean(quantile_fwd_returns[1:max(2, k)], axis=0)
    frame_data.append(
        go.Bar(x=[f"Q{q+1}" for q in range(5)], y=expanding_q_mean, 
               marker_color=['#ff3333', '#ff9933', '#ffff33', '#99ff33', '#39ff14'], 
               name='Signal Quantiles', showlegend=False)
    )
    
    # RIGHT COLUMN BOTTOM: Equity Curve
    frame_data.append(
        go.Scatter(x=t_array[:k], y=equity_curve[:k], mode='lines', 
                   line=dict(color='#00ffff', width=2), fill='tozeroy', 
                   fillcolor='rgba(0, 255, 255, 0.1)', name='L/S Equity', showlegend=False)
    )
    
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        f"Synthetic Panel (N={N_ASSETS}) Sample Paths", 
        "Factor Monotonicity (Expanding Avg Fwd Return)",
        f"Market Neutral Equity Curve (L {TOP_N} / S {TOP_N})"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

for i in range(N_ASSETS):
    fig.add_trace(go.Scatter(x=[0], y=[asset_paths[0, i]]), row=1, col=1)
fig.add_trace(go.Bar(x=[f"Q{q+1}" for q in range(5)], y=[0]*5), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[equity_curve[0]]), row=2, col=2)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Cross-Sectional Factor Strategy (Targeted 2.7 Sharpe)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Global axis bounds 
min_path, max_path = asset_paths.min() * 0.95, asset_paths.max() * 1.05
min_eq, max_eq = equity_curve.min() * 0.95, equity_curve.max() * 1.05

# Bar chart bounds
global_q_means = np.mean(quantile_fwd_returns[1:], axis=0)
max_abs_bar = max(abs(global_q_means.min()), abs(global_q_means.max())) * 1.5

fig.update_xaxes(title_text='Time Step', range=[0, T_STEPS], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Cumulative Return', range=[min_path, max_path], gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=1, row=1, col=1)

fig.update_xaxes(title_text='Factor Quantiles', gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(title_text='Mean 1-Period Fwd Return', range=[-max_abs_bar, max_abs_bar], gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=1, row=1, col=2)

fig.update_xaxes(title_text='Time Step', range=[0, T_STEPS], gridcolor='rgba(128,128,128,0.2)', row=2, col=2)
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], gridcolor='rgba(128,128,128,0.2)', row=2, col=2)

fig.add_annotation(
    text=f"<b>Strategy Sharpe: {sharpe_ratio:.2f}</b>", 
    xref="x domain", yref="y domain", x=0.02, y=0.98, 
    showarrow=False, align="left", font=dict(color="#00ffff", size=16), 
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=6, 
    row=2, col=2
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. SYNTHETIC DATA GENERATION ---
# ==========================================
np.random.seed(101)

N_ASSETS = 25
T_STEPS = 500
TARGET_FRAMES = 80

# 1. Market Component 
market_returns = np.random.randn(T_STEPS, 1) * 0.01 + 0.0005

# 2. Latent Factor 
factor_scores = np.random.randn(T_STEPS, N_ASSETS)

# 3. Idiosyncratic Noise
idio_noise = np.random.randn(T_STEPS, N_ASSETS) * 0.015

# 4. Generate Baseline Forward Returns (Edge significantly reduced to 0.0004)
forward_returns = np.zeros((T_STEPS, N_ASSETS))
forward_returns[1:] = market_returns[1:] + (factor_scores[:-1] * 0.0004) + idio_noise[1:]

# ---------------------------------------------------------
# THE GUARANTEED VALUE TRAPS
# ---------------------------------------------------------
toxic_assets = [10, 15, 20]
decay_start = 100

for asset in toxic_assets:
    # 1. FORCED LONG: Factor score is cranked to 4.0 so they ALWAYS rank in the Top 5
    factor_scores[decay_start:, asset] = 4.0 + np.random.randn(T_STEPS - decay_start) * 0.1
    
    # 2. THE BLEED: They consistently lose money (-0.5% daily avg) with noise
    forward_returns[decay_start:, asset] = -0.005 + np.random.randn(T_STEPS - decay_start) * 0.015

# Cross-sectionally demean the factor at each time step 
factor_scores = factor_scores - factor_scores.mean(axis=1, keepdims=True)

# Shift to align: Factor at t predicts Return at t+1
factor_scores_t = factor_scores[:-1]
actual_returns_t1 = forward_returns[1:]
T_STEPS -= 1

# Generate cumulative sample paths 
asset_paths = np.cumprod(1 + actual_returns_t1, axis=0)

# ==========================================
# --- 2. STRATEGY SIMULATION ---
# ==========================================
TOP_N = 5

strategy_returns = []
quantile_fwd_returns = []

for t in range(T_STEPS):
    signal = factor_scores_t[t]
    realized_ret = actual_returns_t1[t]
    
    # Rank assets by factor
    ranks = np.argsort(signal)
    short_idx = ranks[:TOP_N]       # Bottom 5 
    long_idx = ranks[-TOP_N:]       # Top 5 
    
    # Calculate Market Neutral Portfolio Return
    long_ret = np.mean(realized_ret[long_idx])
    short_ret = np.mean(realized_ret[short_idx])
    
    port_ret = long_ret - short_ret
    strategy_returns.append(port_ret)
    
    # Calculate Monotonic Quantile Distribution 
    q_returns = []
    for q in range(5):
        q_idx = ranks[q*5 : (q+1)*5]
        q_returns.append(np.mean(realized_ret[q_idx]))
    quantile_fwd_returns.append(q_returns)

strategy_returns = np.array(strategy_returns)
quantile_fwd_returns = np.array(quantile_fwd_returns)

# Align lengths back to T_STEPS for clean animation
equity_curve = np.cumprod(1 + strategy_returns)
equity_curve = np.insert(equity_curve, 0, 1.0) 
quantile_fwd_returns = np.insert(quantile_fwd_returns, 0, np.zeros(5), axis=0)

sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
t_array = np.arange(T_STEPS + 1)

step_size = max(1, T_STEPS // TARGET_FRAMES) 
frame_indices = list(range(1, T_STEPS + 1, step_size))
if frame_indices[-1] != T_STEPS:
    frame_indices.append(T_STEPS)

for k in frame_indices:
    frame_data = []
    
    # LEFT COLUMN: Asset Sample Paths 
    for i in range(N_ASSETS):
        color = 'rgba(255,255,255,0.2)'
        width = 1.5
        name = f"Asset {i}"
        
        # Highlight the asymptotic bleeders in red
        if i in toxic_assets:
            color = 'rgba(255, 51, 51, 0.9)'
            width = 2.5
            name = "Forced Value Trap"
            
        frame_data.append(
            go.Scatter(x=t_array[:k], y=asset_paths[:k, i], mode='lines', 
                       line=dict(width=width, color=color), 
                       name=name, showlegend=False)
        )
    
    # RIGHT COLUMN TOP: Expanding Quantile Bar Chart
    expanding_q_mean = np.mean(quantile_fwd_returns[1:max(2, k)], axis=0)
    frame_data.append(
        go.Bar(x=[f"Q{q+1}" for q in range(5)], y=expanding_q_mean, 
               marker_color=['#ff3333', '#ff9933', '#ffff33', '#99ff33', '#39ff14'], 
               name='Signal Quantiles', showlegend=False)
    )
    
    # RIGHT COLUMN BOTTOM: Equity Curve
    frame_data.append(
        go.Scatter(x=t_array[:k], y=equity_curve[:k], mode='lines', 
                   line=dict(color='#00ffff', width=2.5), fill='tozeroy', 
                   fillcolor='rgba(0, 255, 255, 0.1)', name='L/S Equity', showlegend=False)
    )
    
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        f"Synthetic Panel w/ Guaranteed Value Traps", 
        "Factor Monotonicity (Expanding Avg Fwd Return)",
        f"Market Neutral Equity Curve (L {TOP_N} / S {TOP_N})"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

for i in range(N_ASSETS):
    fig.add_trace(go.Scatter(x=[0], y=[asset_paths[0, i]]), row=1, col=1)
fig.add_trace(go.Bar(x=[f"Q{q+1}" for q in range(5)], y=[0]*5), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[equity_curve[0]]), row=2, col=2)

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
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Cross-Sectional Strategy vs. Guaranteed Factor Decay",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Global axis bounds 
min_path, max_path = asset_paths.min() * 0.95, asset_paths.max() * 1.05
min_eq, max_eq = equity_curve.min() * 0.95, equity_curve.max() * 1.05

global_q_means = np.mean(quantile_fwd_returns[1:], axis=0)
max_abs_bar = max(abs(global_q_means.min()), abs(global_q_means.max())) * 1.5

fig.update_xaxes(title_text='Time Step', range=[0, T_STEPS], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Cumulative Return', range=[min_path, max_path], gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=1, row=1, col=1)

fig.update_xaxes(title_text='Factor Quantiles', gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(title_text='Mean 1-Period Fwd Return', range=[-max_abs_bar, max_abs_bar], gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=1, row=1, col=2)

fig.update_xaxes(title_text='Time Step', range=[0, T_STEPS], gridcolor='rgba(128,128,128,0.2)', row=2, col=2)
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], gridcolor='rgba(128,128,128,0.2)', row=2, col=2)

fig.add_annotation(
    text=f"<b>Strategy Sharpe: {sharpe_ratio:.2f}</b>", 
    xref="x domain", yref="y domain", x=0.02, y=0.98, 
    showarrow=False, align="left", font=dict(color="#00ffff", size=16), 
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00ffff", borderwidth=1, borderpad=6, 
    row=2, col=2
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### 🪦 Avoiding Survivorship Bias
# 
# - At each time t, define the universe exactly as it existed at that time—use point-in-time membership, not today's surviving assets.
# - Include assets that later delisted, merged, or "died" (even if their eventual disappearance is known) when you build signals, test strategies, or bucket ranks.
# - If there is no return/price data because an asset became unavailable after t, it should be genuinely missing—not zero or filled with later data.
# - Only use data you could have truly known at that time about which assets existed and were tradeable.
# - Recalculate all cross-sectional rankings, percentiles, and buckets only based on the then-available universe at t.
# - Never retro-fit history using today's list of "long-lived" assets—it leads to unrealistically good backtests!
# - If possible, use a Survivorship-Bias-Free database or reconstruct asset membership/histories carefully.
# - Document universe selection clearly and be transparent about universe construction in research and results.


# ---


# #### 4.) 🎖️ Honorable Mentions: TCs, Spread, Slippage
# 
#  - 💸 **Transaction Costs (TCs):**
#      - Always include realistic estimates for commissions and fees in your backtests.
#      - Ignoring TCs can turn a seemingly profitable strategy into a losing one.
#  
#  - 🔀 **Bid-Ask Spread (usually most important!):**
#      - The spread is the cost you pay when buying at the ask and selling at the bid.
#      - For most liquid markets and retail strategies, the spread eats a bigger chunk of P&L than explicit TCs or slippage.
#      - **Most Insulting:** If your strategy doesn't survive the spread, it definitely won't survive TCs or slippage either.
#  
#  - 🪞 **Slippage:**
#      - Slippage is the difference between your expected trade price and the actual executed price.
#      - Becomes significant when:
#          - Trading with large size relative to market liquidity.
#          - Trading in illiquid or volatile markets.
#      - In such cases, slippage can outweigh TCs and even spreads.
#  
#  - ⚠️ **Key Takeaway:** 
#      - Spread is generally the most important microstructure cost to account for, followed by transaction costs.
#      - For illiquid markets or large orders, slippage can leapfrog TC and even spread as your main performance drag.
#      - **Always model all three realistically in any backtest!**


# ###### ______________________________________________________________________________________________________________________________________


# If your strategy doesn't work *without* transaction costs, it certainly won't work *with* them.  Spread is likely the most important consideration before TCs and Slippage as its the most insulting, once you start getting concerned with size or are dealing with illiquid markets slippage and TCs become more insulting to an equity curve


# ---


# #### 5.) 💭 Closing Thoughts and Future Topics
# 
#  **📑 TL;DW Executive Summary**
#  
#   - 🕵️‍♂️ **Pitfall #1: Look-Ahead Bias**
#       - Using future data in your backtest by accident (e.g., referencing the close price before the bar closes).
#       - Always make sure your simulated trades and logic only use information available at that time.
#   
#   - 🎯 **Pitfall #2: Overfitting**
#       - Creating a strategy that performs well on historical data but fails in live trading due to excessive complexity or curve-fitting.
#       - Favor simple, robust models; always test on out-of-sample data.
#   
#    - 💸 **Pitfall #3: Ignoring Microstructure Costs (TCs, Spread, Slippage)**
#        - Not modeling transaction costs, bid-ask spreads, or slippage can turn profitable strategies into losers in real markets.
#        - Always include realistic estimates of all trading costs in your backtest.
#  
#    -  🏅 *Honorable mentions:*  
#           - **Transaction Costs (TCs):** Commissions, fees, or other costs incurred every time you trade.
#           - **Spread:** The bid-ask spread, i.e., the difference between the price you can buy vs. sell; trading always incurs this cost.
#           - **Slippage:** The difference between your expected execution price and actual price; more relevant in illiquid markets or with large orders.
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
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

