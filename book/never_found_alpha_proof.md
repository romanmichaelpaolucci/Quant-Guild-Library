### 📈 I Bet You've Never Found Alpha (and I Can Prove It)

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


#### 1. ) 💨 Beta is the Wind Blowing

#### 2.) 📈 The "Holy Grail" of Guru Trading

#### 3.) 🌊 It's Probably Just Beta

#### 4.) 🕓 No, We Aren't Timing Beta

#### 5.) 📊 Regime Modeling is Also Probably Just Beta

#### 6.) 📜 The Truth About Alpha and Retail Trading

#### 7.) 💭 Closing Thoughts and Future Topics

---

#### 1. ) 💨 Market Beta is the Wind Blowing

A massive component of returns comes from simple exposure to the market factor

##### Capital Asset Pricing Model

 $$
 r_i = r_f + \beta_i (r_m - r_f)
 $$

Yes, I know this doesn't account for 100% of variation in returns, there are other factors, and the market isn't perfectly efficient.

If you've never heard of the Fama-French, the joint hypothesis problem, or the notion of orthogonal returns you're in rough shape... 

For the purposes of this discussion, the market component is the economy driving returns, we have no control over it

It may contribute positively or negatively over time...


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_datareader.data as web
import warnings

# Suppress the persistent 'date_parser' and GroupBy warnings for a clean console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# --- 1. DATA ACQUISITION (Mkt-RF) ---
# ==========================================
start_date = '2023-01-01'
end_date = '2025-04-08'

# Fetch Fama-French Daily Factors
ff_factors = web.DataReader('F-F_Research_Data_Factors_Daily', 'famafrench', 
                            start=start_date, 
                            end=end_date)[0]

ff_factors = ff_factors / 100.0
ff_factors['Mkt_Cumulative'] = (1 + ff_factors['Mkt-RF']).cumprod()
dates_array = ff_factors.index

# ==========================================
# --- 2. CONTINUOUS REGIME SHADING ---
# ==========================================
ff_factors['Sign'] = np.where(ff_factors['Mkt-RF'] >= 0, 1, -1)
ff_factors['Change'] = ff_factors['Sign'].diff().ne(0).cumsum()

shapes = []
# Robust regime block calculation without .apply()
regime_groups = ff_factors.groupby('Change')
regime_data = []

for _, group in regime_groups:
    regime_data.append({
        'start': group.index[0],
        'end': group.index[-1],
        'sign': group['Sign'].iloc[0]
    })

regimes = pd.DataFrame(regime_data)

# Close the weekend/holiday gaps
for i in range(len(regimes) - 1):
    regimes.loc[i, 'end'] = regimes.loc[i+1, 'start']

for _, row in regimes.iterrows():
    color = 'rgba(57, 255, 20, 0.12)' if row['sign'] == 1 else 'rgba(255, 51, 51, 0.12)'
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=row['start'], x1=row['end'],
        y0=0, y1=1, fillcolor=color,
        layer="below", line_width=0,
    ))

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
indices = np.linspace(0, len(ff_factors)-1, 150, dtype=int)
frames = [go.Frame(
    data=[go.Scatter(x=dates_array[:i], y=ff_factors['Mkt_Cumulative'].iloc[:i], 
                     mode='lines', line=dict(color='#39ff14', width=3))],
    name=f"step{i}"
) for i in indices]

# ==========================================
# --- 4. FIGURE & STYLED CONTROLS ---
# ==========================================
fig = go.Figure(
    data=[go.Scatter(x=[dates_array[0]], y=[ff_factors['Mkt_Cumulative'].iloc[0]], 
                     mode='lines', line=dict(color='#39ff14', width=3))],
    layout=go.Layout(
        title="<b>The Market Wind</b>: Fama-French Mkt-RF (Cumulative)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        shapes=shapes,
        xaxis=dict(range=[dates_array.min(), dates_array.max() + pd.Timedelta(days=20)], 
                   gridcolor='rgba(128,128,128,0.1)', zeroline=False),
        yaxis=dict(title="Cumulative Excess Return", 
                   gridcolor='rgba(128,128,128,0.1)', zeroline=False),
        margin=dict(r=200, t=100, b=150, l=80),
        # Original Quant Guild Playbutton styling
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.12, xanchor="left", yanchor="top",
            buttons=[dict(label="▶ Play", method="animate", 
                          args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, mode="immediate")])]
        )],
        sliders=[dict(
            active=0, pad={"t": 50, "b": 10}, x=0.12, len=0.88,
            currentvalue={"prefix": "Date: ", "font": {"color": "#39ff14", "size": 14}},
            steps=[dict(method="animate", label=str(dates_array[i].date()), 
                        args=[[f"step{i}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) 
                   for i in indices]
        )]
    ),
    frames=frames
)

# Structural Break Line
fig.add_vline(x='2025-01-01', line_width=2, line_dash="dash", line_color="red")

fig.show()
```

When it comes to beta you are a passenger for the ride.

In reality yes, there are other facets of priced risk explaining cross-sectional returns

Though this component dominates arbitrary variance explained, alpha lives in niche structure

I have plenty of strategies developed in this capacity...Roman drop them in a video! No :)

---

#### 2.) 📈 The "Holy Grail" of Guru Trading

Everyone wants to get to the backtests and building strategies, nobody wants to learn the necessary quantitative skills

 The moving average (MA) of a time series $ (x_t) $ at time $ t $ with window size $ k $ is defined as:
 
 $$
 MA_t = \frac{1}{k} \sum_{i=0}^{k-1} x_{t-i}
 $$
 
 where $ MA_t $ is the mean of the latest $ k $ observations up to time $ t $.

I gaurentee nobody who paints this on a chart knows the difference between filtering, smoothing, and forecasting...



```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('AAPL.csv')

# Ensure Date columns are datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Stop the data before 2025-01-01
df = df[df['Date'] < '2025-01-01'].reset_index(drop=True)

# Calculate Daily Return if not in the original CSV
if 'Daily_Return' not in df.columns:
    df['Daily_Return'] = df['Close'].pct_change()

n_steps = len(df) 

# Calculate Moving Averages
df['5_MA'] = df['Close'].rolling(window=5).mean()
df['9_MA'] = df['Close'].rolling(window=9).mean()

take_profit_pct = 0.04  # 4% TP
stop_loss_pct = 0.02    # 2% SL

in_position = False
entry_price = 0.0
trade_pnls = []      
buy_indices = []  
sell_indices = []
equity_curve = [1.0]

# Simulate the Strategy
for i in range(n_steps):
    current_price = df['Close'].iloc[i]
    ma5 = df['5_MA'].iloc[i]
    ma9 = df['9_MA'].iloc[i]
    prev_ma5 = df['5_MA'].iloc[i-1] if i > 0 else np.nan
    prev_ma9 = df['9_MA'].iloc[i-1] if i > 0 else np.nan
    
    daily_strat_ret = 0.0
    
    if in_position:
        daily_strat_ret = df['Daily_Return'].iloc[i] 
            
        unrealized_pnl = (current_price - entry_price) / entry_price
        
        # Exit Conditions
        if unrealized_pnl >= take_profit_pct or unrealized_pnl <= -stop_loss_pct or (ma5 < ma9 and prev_ma5 >= prev_ma9):
            in_position = False
            trade_pnls.append(unrealized_pnl)
            sell_indices.append(i)
            
    else:
        # Entry Condition
        if ma5 > ma9 and prev_ma5 <= prev_ma9:
            in_position = True
            entry_price = current_price
            buy_indices.append(i)
            
    equity_curve.append(equity_curve[-1] * (1 + daily_strat_ret))

equity_curve = equity_curve[1:]

# ==========================================
# --- 2. CALCULATE METRICS ---
# ==========================================
wins = [p for p in trade_pnls if p > 0]
losses = [p for p in trade_pnls if p <= 0]

win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
avg_win = np.mean(wins) if wins else 0
avg_loss = np.mean(losses) if losses else 0

strat_series = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(252) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0

peak = pd.Series(equity_curve).cummax()
mdd = ((pd.Series(equity_curve) - peak) / peak).min()

strat_total_return = equity_curve[-1] - 1

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_array = df['Date'].values
closes_array = df['Close'].values
ma5_array = df['5_MA'].values
ma9_array = df['9_MA'].values
buy_idx_arr = np.array(buy_indices)
sell_idx_arr = np.array(sell_indices)
equity_array = np.array(equity_curve)

step_size = max(1, n_steps // 200) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps:
    frame_indices.append(n_steps)

for k in frame_indices:
    t_x = dates_array[:k]
    
    # Top Chart (Row 1)
    tr_price = go.Scatter(x=t_x, y=closes_array[:k], mode='lines', line=dict(color='white', width=1.5), name='Close')
    tr_ma5 = go.Scatter(x=t_x, y=ma5_array[:k], mode='lines', line=dict(color='#00ffff', width=1.5), name='5 MA')
    tr_ma9 = go.Scatter(x=t_x, y=ma9_array[:k], mode='lines', line=dict(color='#ff00ff', width=1.5), name='9 MA')
    
    b_mask = buy_idx_arr < k
    s_mask = sell_idx_arr < k
    tr_buy = go.Scatter(x=dates_array[buy_idx_arr[b_mask]], y=closes_array[buy_idx_arr[b_mask]], mode='markers', marker=dict(color='#39ff14', size=8, symbol='triangle-up'), name='Buy')
    tr_sell = go.Scatter(x=dates_array[sell_idx_arr[s_mask]], y=closes_array[sell_idx_arr[s_mask]], mode='markers', marker=dict(color='#ff3333', size=8, symbol='triangle-down'), name='Sell')
    
    # Bottom Chart (Row 2): Strategy Equity Only
    tr_equity = go.Scatter(x=t_x, y=equity_array[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy')

    # Note: Passed 6 traces (5 top, 1 bottom) per frame
    frames.append(go.Frame(data=[tr_price, tr_ma5, tr_ma9, tr_buy, tr_sell, tr_equity], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    row_heights=[0.6, 0.4],
    subplot_titles=[
        "Price & Moving Average Crossover", 
        "Strategy Equity"
    ],
    vertical_spacing=0.15 
)

# Dummy traces to match the exact order of the 6 traces in our frames
for _ in range(5): fig.add_trace(go.Scatter(x=[dates_array[0]], y=[closes_array[0]]), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[equity_array[0]]), row=2, col=1)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

metrics_text = (
    f"<b>Strategy Metrics</b><br><br>"
    f"Strat Return: {strat_total_return:.2%}<br><br>"
    f"Total Trades: {len(trade_pnls)}<br>"
    f"Win Prob: {win_rate:.1%}<br>"
    f"Loss Prob: {(1-win_rate):.1%}<br>"
    f"Avg Winner: {avg_win:.2%}<br>"
    f"Avg Loser: {avg_loss:.2%}<br><br>"
    f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
    f"Max Drawdown: {mdd:.2%}"
)

fig.add_annotation(
    text=metrics_text, xref="paper", yref="paper", 
    x=1.35, y=0.85,  
    showarrow=False, align="left", font=dict(color="white", size=13),
    bgcolor="rgba(0,0,0,0)", bordercolor="#00ffff", borderwidth=1, borderpad=10
)

# Set plots and paper to completely transparent
fig.update_layout(
    height=700, width=1800,
    title_text="Quantitative Strategy: 5/9 MA Crossover with Hard TP/SL",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=300), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
min_price, max_price = closes_array.min() * 0.95, closes_array.max() * 1.05
min_eq = equity_array.min() * 0.95
max_eq = equity_array.max() * 1.05

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price', range=[min_price, max_price], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()
```

A great starting point, roughly double the average winner, reasonable Sharpe and MDD; is it though?

###### ______________________________________________________________________________________________________________________________________

#### 3.) 🌊 It's Probably Just Beta

##### Regress Market Exposure on Your Equity Curve

If you don't know what that means, rough

 $$
 \pi_t = \alpha + \beta \cdot MRP_t + \varepsilon_t
 $$
 where:
 - $\pi_t$ = Strategy returns at time $t$
 - $MRP_t$ = Market return proxy (e.g., $S\&P500$) at time $t$
 - $\alpha$ = Intercept (strategy's "alpha": return unexplained by market)
 - $\beta$ = Market sensitivity ("beta": exposure to general market moves)
 - $\varepsilon_t$ = Residual (idiosyncratic noise)


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# ==========================================
# --- 1. DATA GENERATION & BACKTEST ---
# ==========================================
df = pd.read_csv('AAPL.csv')
spx = pd.read_csv('SPY.csv')

# Ensure Date columns are datetime objects for proper alignment
df['Date'] = pd.to_datetime(df['Date'])
spx['Date'] = pd.to_datetime(spx['Date'])

# Merge SPX data onto the strategy dataframe
df = pd.merge(df, spx[['Date', 'Close', 'Daily_Return']], on='Date', how='left', suffixes=('', '_SPX'))

# Forward fill any missing SPX data, then drop any remaining NaNs
df = df.ffill().dropna().reset_index(drop=True)

# Cutoff data before 2025-01-01
df = df[df['Date'] < '2025-01-01'].reset_index(drop=True)

n_steps = len(df) 

# Calculate Moving Averages
df['5_MA'] = df['Close'].rolling(window=5).mean()
df['9_MA'] = df['Close'].rolling(window=9).mean()

take_profit_pct = 0.04  # 4% TP
stop_loss_pct = 0.02    # 2% SL

in_position = False
entry_price = 0.0
trade_pnls = []      
equity_curve = [1.0]

# Simulate the Strategy
for i in range(n_steps):
    current_price = df['Close'].iloc[i]
    ma5 = df['5_MA'].iloc[i]
    ma9 = df['9_MA'].iloc[i]
    prev_ma5 = df['5_MA'].iloc[i-1] if i > 0 else np.nan
    prev_ma9 = df['9_MA'].iloc[i-1] if i > 0 else np.nan
    
    daily_strat_ret = 0.0
    
    if in_position:
        if 'Daily_Return' in df.columns:
            daily_strat_ret = df['Daily_Return'].iloc[i] 
        else:
            daily_strat_ret = (current_price - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            
        unrealized_pnl = (current_price - entry_price) / entry_price
        
        # Exit Conditions
        if unrealized_pnl >= take_profit_pct or unrealized_pnl <= -stop_loss_pct or (ma5 < ma9 and prev_ma5 >= prev_ma9):
            in_position = False
            trade_pnls.append(unrealized_pnl)
            
    else:
        # Entry Condition
        if ma5 > ma9 and prev_ma5 <= prev_ma9:
            in_position = True
            entry_price = current_price
            
    equity_curve.append(equity_curve[-1] * (1 + daily_strat_ret))

equity_curve = equity_curve[1:]

# ==========================================
# --- 2. CALCULATE METRICS & BENCHMARK ---
# ==========================================
# Asset & SPX Buy & Hold
asset_returns = df['Daily_Return'].fillna(0).values if 'Daily_Return' in df.columns else df['Close'].pct_change().fillna(0).values
asset_equity = np.cumprod(1 + asset_returns)

spx_returns = df['Daily_Return_SPX'].fillna(0).values
spx_equity = np.cumprod(1 + spx_returns)

strat_returns = pd.Series(equity_curve).pct_change().fillna(0).values

# Filter out days where strategy had 0 returns (flat/cash) for CAPM
active_mask = strat_returns != 0.0
spx_returns_active = spx_returns[active_mask]
strat_returns_active = strat_returns[active_mask]

# Calculate Beta and Alpha on active days
if np.var(spx_returns_active) > 0:
    X = sm.add_constant(spx_returns_active)
    model = sm.OLS(strat_returns_active, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
else:
    alpha, beta = 0, 0

wins = [p for p in trade_pnls if p > 0]
losses = [p for p in trade_pnls if p <= 0]

win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
avg_win = np.mean(wins) if wins else 0
avg_loss = np.mean(losses) if losses else 0

strat_series = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(252) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0

peak = pd.Series(equity_curve).cummax()
mdd = ((pd.Series(equity_curve) - peak) / peak).min()

strat_total_return = equity_curve[-1] - 1
asset_total_return = asset_equity[-1] - 1
spx_total_return = spx_equity[-1] - 1

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

dates_array = df['Date'].values
equity_array = np.array(equity_curve)

step_size = max(1, n_steps // 200) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps:
    frame_indices.append(n_steps)

# Pre-calculate regression line points for the scatterplot
if len(spx_returns_active) > 0:
    x_range = np.array([spx_returns_active.min(), spx_returns_active.max()])
else:
    x_range = np.array([-0.05, 0.05])
y_range = alpha + beta * x_range

for k in frame_indices:
    t_x = dates_array[:k]
    
    # Top Chart (Row 1): Equity Curves
    tr_asset_eq = go.Scatter(x=t_x, y=asset_equity[:k], mode='lines', line=dict(color='#00ffff', width=1.5), name='Asset B&H')
    tr_spx_eq = go.Scatter(x=t_x, y=spx_equity[:k], mode='lines', line=dict(color='#aaaaaa', width=1.5, dash='dot'), name='SPX B&H')
    tr_strat_eq = go.Scatter(x=t_x, y=equity_array[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy')
    
    # Bottom Chart (Row 2): Scatterplot (Masked to remove flat days)
    t_strat_ret = strat_returns[:k]
    t_spx_ret = spx_returns[:k]
    t_mask = t_strat_ret != 0.0
    
    tr_scatter = go.Scatter(x=t_spx_ret[t_mask], y=t_strat_ret[t_mask], mode='markers', marker=dict(color='rgba(0, 255, 255, 0.5)', size=5), name='Active Returns')
    tr_beta = go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color='#ff00ff', width=2), name=f'Beta: {beta:.2f}')

    # Pass exactly 5 traces per frame
    frames.append(go.Frame(data=[tr_asset_eq, tr_spx_eq, tr_strat_eq, tr_scatter, tr_beta], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    row_heights=[0.6, 0.4],
    subplot_titles=[
        "Strategy Equity vs. Benchmarks (Asset & SPX)", 
        f"Active Return Correlation (Beta: {beta:.2f})"
    ],
    vertical_spacing=0.15 
)

# Dummy traces matching the 5 traces in our frames
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[asset_equity[0]]), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[spx_equity[0]]), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[equity_array[0]]), row=1, col=1)

fig.add_trace(go.Scatter(x=[0], y=[0]), row=2, col=1)
fig.add_trace(go.Scatter(x=x_range, y=y_range), row=2, col=1)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for idx, k in enumerate(frame_indices)]
)]

metrics_text = (
    f"<b>Performance Comparison</b><br><br>"
    f"Asset B&H: <span style='color:#00ffff'>{asset_total_return:.2%}</span><br>"
    f"SPX B&H: <span style='color:#aaaaaa'>{spx_total_return:.2%}</span><br>"
    f"Strategy: <span style='color:#39ff14'>{strat_total_return:.2%}</span><br><br>"
    f"<b>Strategy Metrics</b><br>"
    f"Total Trades: {len(trade_pnls)}<br>"
    f"Win Prob: {win_rate:.1%}<br>"
    f"Avg Winner: {avg_win:.2%}<br>"
    f"Avg Loser: {avg_loss:.2%}<br>"
    f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
    f"Max Drawdown: {mdd:.2%}<br><br>"
    f"<b>CAPM (Active Days)</b><br>"
    f"Beta: {beta:.2f}"
)

fig.add_annotation(
    text=metrics_text, xref="paper", yref="paper", 
    x=1.35, y=0.85,  
    showarrow=False, align="left", font=dict(color="white", size=13),
    bgcolor="rgba(0,0,0,0)", bordercolor="#00ffff", borderwidth=1, borderpad=10
)

# Set plots and paper to completely transparent
fig.update_layout(
    height=700, width=1000,
    title_text="Quantitative Strategy vs Benchmark Performance",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=300), 
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
min_eq = min(equity_array.min(), spx_equity.min(), asset_equity.min()) * 0.95
max_eq = max(equity_array.max(), spx_equity.max(), asset_equity.max()) * 1.05

if len(strat_returns_active) > 0:
    min_ret, max_ret = strat_returns_active.min() * 1.1, strat_returns_active.max() * 1.1
    min_spx, max_spx = spx_returns_active.min() * 1.1, spx_returns_active.max() * 1.1
else:
    min_ret, max_ret = -0.05, 0.05
    min_spx, max_spx = -0.05, 0.05

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Equity Multiplier', range=[min_eq, max_eq], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='SPX Daily Return', range=[min_spx, max_spx], tickformat='.1%', row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Strategy Daily Return', range=[min_ret, max_ret], tickformat='.1%', row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()
```

Beta exposure accounts for your return, no skill, no strategy, no edge...just accumulating a ton of transaction costs for your "*strategy*"

But Roman, it *timed* the correct Beta exposure for us!  

Nah, you're just need a period of negative cross-sectional return for that facet of priced risk to correct that notion...

###### ______________________________________________________________________________________________________________________________________

#### 4.) 🕓 No, We Aren't Timing Beta

Let's use the same strategy that is just Beta exposure and consider the idea that it offers some *timing advantage*

Let's see how this works during market distress...


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# ==========================================
# --- 1. DATA PREP & REGIME SPLIT ---
# ==========================================
df = pd.read_csv('AAPL.csv')
spx = pd.read_csv('SPY.csv')

df['Date'] = pd.to_datetime(df['Date'])
spx['Date'] = pd.to_datetime(spx['Date'])
df = pd.merge(df, spx[['Date', 'Close', 'Daily_Return']], on='Date', how='left', suffixes=('', '_SPX'))
df = df.ffill().dropna().reset_index(drop=True)

# HARD CUTOFF: April 8th, 2025
df = df[df['Date'] <= '2025-04-20'].reset_index(drop=True)
regime_cutoff_idx = df[df['Date'] >= '2025-01-01'].index[0]
regime_date = df['Date'].iloc[regime_cutoff_idx]

# Signal Logic
df['5_MA'] = df['Close'].rolling(window=5).mean()
df['9_MA'] = df['Close'].rolling(window=9).mean()

take_profit_pct, stop_loss_pct = 0.04, 0.02
in_position, entry_price = False, 0.0
equity_curve = [1.0]

for i in range(len(df)):
    curr_p = df['Close'].iloc[i]
    ma5, ma9 = df['5_MA'].iloc[i], df['9_MA'].iloc[i]
    prev_ma5 = df['5_MA'].iloc[i-1] if i > 0 else np.nan
    prev_ma9 = df['9_MA'].iloc[i-1] if i > 0 else np.nan
    
    daily_ret = 0.0
    if in_position:
        daily_ret = df['Daily_Return'].iloc[i]
        unrealized = (curr_p - entry_price) / entry_price
        if unrealized >= take_profit_pct or unrealized <= -stop_loss_pct or (ma5 < ma9 and prev_ma5 >= prev_ma9):
            in_position = False
    else:
        if ma5 > ma9 and prev_ma5 <= prev_ma9:
            in_position, entry_price = True, curr_p
            
    equity_curve.append(equity_curve[-1] * (1 + daily_ret))

df['Strat_Equity'] = equity_curve[1:]
df['Strat_Ret'] = df['Strat_Equity'].pct_change().fillna(0)
df['Asset_Equity'] = np.cumprod(1 + df['Daily_Return'].fillna(0))
df['SPX_Equity'] = np.cumprod(1 + df['Daily_Return_SPX'].fillna(0))

# ==========================================
# --- 2. CALCULATE REGIME METRICS (NO ALPHA) ---
# ==========================================
def get_regime_metrics(data):
    s_ret = (data['Strat_Equity'].iloc[-1] / data['Strat_Equity'].iloc[0]) - 1
    a_ret = (data['Asset_Equity'].iloc[-1] / data['Asset_Equity'].iloc[0]) - 1
    m_ret = (data['SPX_Equity'].iloc[-1] / data['SPX_Equity'].iloc[0]) - 1
    
    active = data[data['Strat_Ret'] != 0]
    if len(active) > 2:
        model = sm.OLS(active['Strat_Ret'], active['Daily_Return_SPX']).fit()
        beta = model.params.iloc[0]
    else:
        beta = 0
    return s_ret, a_ret, m_ret, beta

s1, a1, m1, b1 = get_regime_metrics(df.iloc[:regime_cutoff_idx])
s2, a2, m2, b2 = get_regime_metrics(df.iloc[regime_cutoff_idx:])

# ==========================================
# --- 3. ANIMATION & PLOTTING ---
# ==========================================
frames = []
dates_array = df['Date'].values
step_size = max(1, len(df) // 150)
frame_indices = list(range(1, len(df), step_size))
if frame_indices[-1] != len(df): frame_indices.append(len(df))

for k in frame_indices:
    t_x = dates_array[:k]
    
    tr_asset = go.Scatter(x=t_x, y=df['Asset_Equity'].iloc[:k], line=dict(color='#00ffff', width=1.5), name='Asset')
    tr_spx = go.Scatter(x=t_x, y=df['SPX_Equity'].iloc[:k], line=dict(color='#aaaaaa', width=1.5, dash='dot'), name='SPX')
    tr_strat = go.Scatter(x=t_x, y=df['Strat_Equity'].iloc[:k], line=dict(color='#39ff14', width=2), fill='tozeroy', name='Strat')
    
    curr_scat = df.iloc[:k]
    active_scat = curr_scat[curr_scat['Strat_Ret'] != 0]
    tr_scat = go.Scatter(x=active_scat['Daily_Return_SPX'], y=active_scat['Strat_Ret'], mode='markers', marker=dict(color='rgba(0, 255, 255, 0.4)', size=4))
    
    current_b = b1 if k < regime_cutoff_idx else b2
    x_range = np.array([df['Daily_Return_SPX'].min(), df['Daily_Return_SPX'].max()])
    tr_beta = go.Scatter(x=x_range, y=current_b * x_range, mode='lines', line=dict(color='#ff00ff', width=2))

    frames.append(go.Frame(data=[tr_asset, tr_spx, tr_strat, tr_scat, tr_beta], name=f"step{k}"))

fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.15,
                    subplot_titles=["The Beta Reckoning: Equity Decay", "Unconditional Beta Correlation"])

fig.add_trace(go.Scatter(x=[dates_array[0]], y=[1]), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[1]), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates_array[0]], y=[1]), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0]), row=2, col=1)
fig.add_trace(go.Scatter(x=[0,1], y=[0,1]), row=2, col=1)

fig.add_vline(x=regime_date, line_width=2, line_dash="dash", line_color="red", row=1, col=1)
fig.frames = frames

# ==========================================
# --- 4. CONTROLS & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, pad={"t": 60, "b": 10}, x=0.1, len=0.9,
    currentvalue={"prefix": "Date: ", "font": {"color": "#39ff14"}},
    steps=[dict(method="animate", label=str(df['Date'].iloc[k-1].date()), args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) 
           for k in frame_indices]
)]

# ADD PADDING TO X-AXIS: End range is 30 days after the final date
end_padding = dates_array[-1] + pd.Timedelta(days=30)

fig.update_layout(
    height=850, width=1100, template="plotly_dark", 
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(r=300, t=100, b=150), showlegend=False,
    sliders=sliders,
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.02, y=-0.08,
        buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)])]
    )]
)

metrics_text = (
    f"<b>REGIME 1 (PRE-2025)</b><br>"
    f"SPX Ret: {m1:.2%}<br>Asset Ret: {a1:.2%}<br>Strat Ret: {s1:.2%}<br><b>Beta: {b1:.2f}</b><br><br>"
    f"<b>REGIME 2 (CRASH TO APR 8)</b><br>"
    f"SPX Ret: {m2:.2%}<br>Asset Ret: {a2:.2%}<br>Strat Ret: {s2:.2%}<br><b>Beta: {b2:.2f}</b>"
)
fig.add_annotation(text=metrics_text, xref="paper", yref="paper", x=1.3, y=0.5, showarrow=False, align="left", bordercolor="red", borderwidth=1, borderpad=10)

# Static boundaries with Padding on the X-axis for the Top Chart
fig.update_xaxes(range=[dates_array[0], end_padding], row=1, col=1)
fig.update_yaxes(range=[0.6, df['Asset_Equity'].max()*1.1], row=1, col=1)

# Scatter boundaries
fig.update_xaxes(range=[-0.07, 0.07], row=2, col=1)
fig.update_yaxes(range=[-0.07, 0.07], row=2, col=1)

fig.show()
```

Shocker, it doesn't, there is no alpha, it's just Beta...

Roman, this is why we have regime models, so we don't trade during periods of volatility and market distress.  

That's our *alpha*.

Yeah, no...

---

#### 5.) 📊 Regime Modeling is Also Probably Just Beta
**Regime-Switching Markov Chain for Rolling Volatility:**
 $$ 
 s_t \in \{0, 1, 2\} \\
 P(s_t = j \mid s_{t-1} = i) = p_{ij} \\
 v_t = \hat{\sigma}_{t, 20} = \operatorname{Std}^{20}(r_t) \\
 \text{Assign regime (} s_t \text{)} \quad
 \begin{cases}
 \text{Low Vol} & v_t \leq Q_{33}(v_{t-252:t}) \\
 \text{Med Vol} & Q_{33}(v_{t-252:t}) < v_t \leq Q_{67}(v_{t-252:t}) \\
 \text{High Vol} & v_t > Q_{67}(v_{t-252:t})
 \end{cases}
 $$
 We use MLE as the estimator for regime transition probabilities.


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# ==========================================
# --- 1. DATA GENERATION & VOLATILITY MODEL ---
# ==========================================
df = pd.read_csv('AAPL.csv')
spx = pd.read_csv('SPY.csv')

# Ensure Date columns are datetime objects for proper alignment
df['Date'] = pd.to_datetime(df['Date'])
spx['Date'] = pd.to_datetime(spx['Date'])
df = pd.merge(df, spx[['Date', 'Close', 'Daily_Return']], on='Date', how='left', suffixes=('', '_SPX'))
df = df.ffill().dropna().reset_index(drop=True)

n_steps = len(df) 

# MAs (Calculated on the Close)
df['5_MA'] = df['Close'].rolling(window=5).mean()
df['9_MA'] = df['Close'].rolling(window=9).mean()

# Rolling Volatility Markov Regime Classification
# (Calculated strictly on realized data up to the current Close)
df['Vol_20'] = df['Daily_Return'].rolling(window=20).std()
df['Vol_Q33'] = df['Vol_20'].rolling(window=252, min_periods=20).quantile(0.333)
df['Vol_Q67'] = df['Vol_20'].rolling(window=252, min_periods=20).quantile(0.667)

conditions = [
    (df['Vol_20'] <= df['Vol_Q33']),
    (df['Vol_20'] > df['Vol_Q33']) & (df['Vol_20'] <= df['Vol_Q67']),
    (df['Vol_20'] > df['Vol_Q67'])
]
# 0: Low (Green), 1: Med (Orange), 2: High (Red)
df['Vol_State'] = np.select(conditions, [0, 1, 2], default=1) 

# ==========================================
# --- 2. STRICT NO-LOOK-AHEAD BACKTEST ---
# ==========================================
take_profit_pct = 0.04  
stop_loss_pct = 0.02    

in_position = False
entry_price = 0.0
trade_pnls = []

buy_indices, sell_indices = [], []
equity_curve = [1.0]

for i in range(n_steps):
    current_price = df['Close'].iloc[i]
    ma5 = df['5_MA'].iloc[i]
    ma9 = df['9_MA'].iloc[i]
    prev_ma5 = df['5_MA'].iloc[i-1] if i > 0 else np.nan
    prev_ma9 = df['9_MA'].iloc[i-1] if i > 0 else np.nan
    
    # State known precisely at the Close of Day i
    current_state = df['Vol_State'].iloc[i] 
    
    daily_ret = 0.0
    
    if in_position:
        # We capture return on Day i because we entered at the Close of Day i-1 (or earlier)
        daily_ret = df['Daily_Return'].iloc[i] if 'Daily_Return' in df.columns else (current_price - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
        unrealized_pnl = (current_price - entry_price) / entry_price
        
        # Exit Condition (Evaluated at Close of Day i)
        if unrealized_pnl >= take_profit_pct or unrealized_pnl <= -stop_loss_pct or (ma5 < ma9 and prev_ma5 >= prev_ma9):
            in_position = False
            trade_pnls.append(unrealized_pnl)
            sell_indices.append(i)
            
    else:
        # Entry Condition: Crossover AND we must be in the Low Vol Regime
        if ma5 > ma9 and prev_ma5 <= prev_ma9 and current_state == 0:
            in_position = True
            entry_price = current_price
            buy_indices.append(i)
            
    # If not in_position, daily_ret remains 0.0 (Cash Day)
    equity_curve.append(equity_curve[-1] * (1 + daily_ret))

equity_curve = np.array(equity_curve[1:])

# ==========================================
# --- 3. UNCONDITIONAL CAPM METRICS ---
# ==========================================
spx_returns = df['Daily_Return_SPX'].fillna(0).values
strat_returns = pd.Series(equity_curve).pct_change().fillna(0).values

# Unconditional Beta: Include ALL days (cash days included)
if np.var(spx_returns) > 0:
    X = sm.add_constant(spx_returns)
    model = sm.OLS(strat_returns, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    alpha_pval = model.pvalues[0]
    beta_pval = model.pvalues[1]
else:
    alpha, beta, alpha_pval, beta_pval = 0, 0, 1.0, 1.0

sigA = "Yes" if alpha_pval < 0.05 else "No"
sigB = "Yes" if beta_pval < 0.05 else "No"

# Basic Metrics
wins = [p for p in trade_pnls if p > 0]
losses = [p for p in trade_pnls if p <= 0]
win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
avg_win = np.mean(wins) if wins else 0
avg_loss = np.mean(losses) if losses else 0
spx_equity = np.cumprod(1 + spx_returns)

strat_series = pd.Series(equity_curve).pct_change().dropna()
sharpe_ratio = np.sqrt(252) * strat_series.mean() / strat_series.std() if strat_series.std() != 0 else 0
peak = pd.Series(equity_curve).cummax()
mdd = ((pd.Series(equity_curve) - peak) / peak).min()

# Regression Line coordinates for plot
x_r = np.array([spx_returns.min(), spx_returns.max()])
y_r = alpha + beta * x_r

# ==========================================
# --- 4. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
dates_array = df['Date'].values
closes_array = df['Close'].values
states_array = df['Vol_State'].values
buy_idx_arr = np.array(buy_indices)
sell_idx_arr = np.array(sell_indices)

# Background shading values 
bg_height = closes_array.max() * 2.0 
y_bg_L = np.where(states_array == 0, bg_height, 0)
y_bg_M = np.where(states_array == 1, bg_height, 0)
y_bg_H = np.where(states_array == 2, bg_height, 0)

step_size = max(1, n_steps // 200) 
frame_indices = list(range(1, n_steps, step_size))
if frame_indices[-1] != n_steps: frame_indices.append(n_steps)

for k in frame_indices:
    t_x = dates_array[:k]
    
    # Row 1 (Background Bars)
    tr_bg_L = go.Bar(x=t_x, y=y_bg_L[:k], marker=dict(color='rgba(57, 255, 20, 0.1)', line_width=0), hoverinfo='skip', showlegend=False)
    tr_bg_M = go.Bar(x=t_x, y=y_bg_M[:k], marker=dict(color='rgba(255, 165, 0, 0.1)', line_width=0), hoverinfo='skip', showlegend=False)
    tr_bg_H = go.Bar(x=t_x, y=y_bg_H[:k], marker=dict(color='rgba(255, 51, 51, 0.1)', line_width=0), hoverinfo='skip', showlegend=False)
    
    # Row 1 (Price and MAs on top)
    tr_price = go.Scatter(x=t_x, y=closes_array[:k], mode='lines', line=dict(color='white', width=1.5))
    tr_ma5 = go.Scatter(x=t_x, y=df['5_MA'].values[:k], mode='lines', line=dict(color='#00ffff', width=1.5))
    tr_ma9 = go.Scatter(x=t_x, y=df['9_MA'].values[:k], mode='lines', line=dict(color='#ff00ff', width=1.5))
    
    b_mask, s_mask = buy_idx_arr < k, sell_idx_arr < k
    tr_buy = go.Scatter(x=dates_array[buy_idx_arr[b_mask]], y=closes_array[buy_idx_arr[b_mask]], mode='markers', marker=dict(color='#39ff14', size=8, symbol='triangle-up'))
    tr_sell = go.Scatter(x=dates_array[sell_idx_arr[s_mask]], y=closes_array[sell_idx_arr[s_mask]], mode='markers', marker=dict(color='#ff3333', size=8, symbol='triangle-down'))
    
    # Row 2 (Equity vs SPX)
    tr_spx_eq = go.Scatter(x=t_x, y=spx_equity[:k], mode='lines', line=dict(color='#aaaaaa', width=1.5, dash='dot'), name='SPX B&H')
    tr_eq = go.Scatter(x=t_x, y=equity_curve[:k], mode='lines', line=dict(color='#39ff14', width=2), fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)', name='Strategy (Low Vol Only)')
    
    # Row 3 (Unconditional Scatter - All Days)
    # Highlight cash days vs active days visually
    t_strat_ret = strat_returns[:k]
    t_spx_ret = spx_returns[:k]
    active_mask = t_strat_ret != 0.0
    cash_mask = t_strat_ret == 0.0

    scat_cash = go.Scatter(x=t_spx_ret[cash_mask], y=t_strat_ret[cash_mask], mode='markers', marker=dict(color='rgba(128, 128, 128, 0.3)', size=4), name='Cash Days')
    scat_act = go.Scatter(x=t_spx_ret[active_mask], y=t_strat_ret[active_mask], mode='markers', marker=dict(color='rgba(57, 255, 20, 0.8)', size=5), name='Active Trades')
    
    line_reg = go.Scatter(x=x_r, y=y_r, mode='lines', line=dict(color='#ff00ff', width=2), name='Unconditional CAPM')

    frames.append(go.Frame(data=[tr_bg_L, tr_bg_M, tr_bg_H, tr_price, tr_ma5, tr_ma9, tr_buy, tr_sell, tr_spx_eq, tr_eq, scat_cash, scat_act, line_reg], name=f"step{k}"))

# ==========================================
# --- 5. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=3, cols=1, 
    row_heights=[0.4, 0.3, 0.3],
    subplot_titles=["Price & Signals (Shaded by Volatility Regime)", "Strategy Equity (Trading Low Vol Regime Only)", "Unconditional Return Correlation (Including Cash)"],
    vertical_spacing=0.15 
)

# Initialize Dummies
for _ in range(3): fig.add_trace(go.Bar(x=[dates_array[0]], y=[0]), row=1, col=1)      
for _ in range(5): fig.add_trace(go.Scatter(x=[dates_array[0]], y=[closes_array[0]]), row=1, col=1) 
for _ in range(2): fig.add_trace(go.Scatter(x=[dates_array[0]], y=[1.0]), row=2, col=1) 
for _ in range(3): fig.add_trace(go.Scatter(x=[0], y=[0]), row=3, col=1)               

fig.frames = frames

# ==========================================
# --- 6. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(method="animate", args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(k)) for idx, k in enumerate(frame_indices)]
)]

metrics_text = (
    f"<b>Strategy Metrics (Low Vol Only)</b><br><br>"
    f"Total Trades: {len(trade_pnls)}<br>"
    f"Win Prob: {win_rate:.1%}<br>"
    f"Avg Winner: {avg_win:.2%}<br>"
    f"Avg Loser: {avg_loss:.2%}<br><br>"
    f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
    f"Max Drawdown: {mdd:.2%}<br><br>"
    f"<b>Unconditional CAPM (All Days)</b><br>"
    f"Alpha Sig (p<.05): {sigA}<br>"
    f"Alpha (Ann.): {alpha * 252:.2%}<br>"
    f"Beta Sig (p<.05): {sigB}<br>"
    f"Beta: {beta:.2f}"
)

fig.add_annotation(
    text=metrics_text, xref="paper", yref="paper", 
    x=1.35, y=0.85,  
    showarrow=False, align="left", font=dict(color="white", size=13),
    bgcolor="rgba(0,0,0,0)", bordercolor="#00ffff", borderwidth=1, borderpad=10
)

fig.update_layout(
    height=900, width=1100,
    title_text="Regime Switching: Low Volatility Tactical Trading",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(t=80, b=100, r=250), 
    barmode='overlay', 
    updatemenus=[{'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
                  'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]}]
)

# Axis Boundaries
min_p, max_p = closes_array.min() * 0.95, closes_array.max() * 1.05
min_e = min(equity_curve.min(), spx_equity.min()) * 0.95
max_e = max(equity_curve.max(), spx_equity.max()) * 1.05

fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price', range=[min_p, max_p], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Equity Multiplier', range=[min_e, max_e], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_xaxes(title_text='SPX Daily Return', range=[-0.06, 0.06], tickformat='.1%', row=3, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Strategy Return', range=[-0.06, 0.06], tickformat='.1%', row=3, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()
```

---

#### 6.) 📜 The Truth About Alpha and Retail Trading

- There is an inverse relationship between how easy a signal is to develop and its efficacy

**Ask yourself:**
- did I fetch, clean, preprocess, feature engineer, and develop a model for a massive time series of parquet files containing alt data?
- do I understand the difference between bearing priced risk and being compensated in the cross section and orthogonal return (alpha)
- do I understand the difference between asymptotics in the classroom relative to the non-stationary (uncertain) real world
- am I trading a basket of equities in the cross-section or a composite portfolio of strategies
- do I have the knowledge or experience to carry my lack of the above

If the answer is no to any (or all, which is more likely the case) you should be *studying and learning*, not spamming backtests or trading.

---

#### 7.) 💭 Closing Thoughts and Future Topics

**📑 TL;DW Executive Summary**

- **The "Market Wind":** Most returns are simply the **Fama-French Market Factor**. If your equity curve moves in lockstep with the market's risk premium, you aren't "trading"—you're just a passenger on the economy.
- **The Crossover Illusion:** Standard signals like the **5/9 MA crossover** create the "Holy Grail" backtest by accidentally harvesting positive Beta during trending regimes. Without a risk-adjustment, "skill" is indistinguishable from luck.
- **The Regression Reality:** If your **Intercept ($\alpha$)** is statistically zero when regressed against the SPY, you have no strategy. You are just holding **Beta in a trench coat** and paying fees to do it.
- **The Timing Myth:** Extending the backtest through the **2025 Structural Break** proves that moving averages don't "time" the market; they simply provide exposure. When the market wind stops, the strategy is "merked" by its own Beta.
- **Regime Shifting vs. Edge:** Sophisticated **Markov-Switching Volatility** models are often just expensive barometers. Filtering for "Low Vol" is frequently just another way to isolate a positive market wind, not a source of idiosyncratic Alpha.
- **The Theorist’s Victory:** In a relationship-driven industry where a $200M manager can't define Beta, **Mathematical Precision** is your only protection. Everyone hates the theorist until the regimes shift and the "vibes" stop working.


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

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
