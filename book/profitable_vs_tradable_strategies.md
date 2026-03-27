### 📈 Profitable vs Tradable: Why Most Strategies Fail Live

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

##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)

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

#### 1.) 📉 Entry Signals, Backtests, Edge, P/L Distributions

#### 2.) 📊 P/L Distribution Stability Over Time, Regime Models

#### 3.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📉 Entry Signals, Backtests, Edge, P/L Distributions

What is your *edge*?  What is your *Sharpe* or *Sortino*?  We all hear these questions thrown around all the time but they don't matter, if someone comes to me with a trading signal I'm interested only in *stability*.

The **quality of your entry signal dictates** how much money you'll make, your risk-adjusted return

The **stability in your distributions** (and upstream feature distributions) dictate how usable your strategy is


By the Law of Total Expectation ($\mathbb{E}$):
 
 $$\mathbb{E}[\mathrm{P/L}] = \mathbb{E}[\mathrm{P/L} \mid \text{win}] \cdot P(\text{win}) + \mathbb{E}[\mathrm{P/L} \mid \text{loss}] \cdot P(\text{loss})$$
 
 where:
   - $\mathbb{E}[\mathrm{P/L} \mid \text{win}]$ = average winner
   - $P(\text{win})$ = probability of winning
   - $\mathbb{E}[\mathrm{P/L} \mid \text{loss}]$ = average loser
   - $P(\text{loss})$ = probability of losing

 This formula shows how both average outcomes and their probabilities drive your strategy's edge.


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# --- Load real NVDA data ---
# Assumes 'NVDA_returns.csv' has columns: Date, Close (and possibly others)
nvda_df = pd.read_csv('NVDA_returns.csv')
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])
nvda_df = nvda_df.sort_values('Date').reset_index(drop=True)

# Optionally restrict to a main period for each subplot:
period1 = (nvda_df['Date'] >= '2020-01-01') & (nvda_df['Date'] <= '2023-12-31')
period2 = (nvda_df['Date'] >= '2024-01-01') & (nvda_df['Date'] <= '2026-12-31')

df1 = nvda_df.loc[period1].copy()
df2 = nvda_df.loc[period2].copy()

# Calculate SMAs and signals
def process_df(df):
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=10).mean()
    df['SMA_long'] = df['Close'].rolling(window=20).mean()
    df['Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade_Signal'] = df['Position'].diff().fillna(0)
    # Find trades
    buy_indices = df.index[df['Trade_Signal'] == 1]
    sell_indices = df.index[df['Trade_Signal'] == -1]
    trade_results = []
    for b_idx in buy_indices:
        potential_sells = sell_indices[sell_indices > b_idx]
        if not potential_sells.empty:
            s_idx = potential_sells[0]
            profit = df.loc[s_idx, 'Close'] - df.loc[b_idx, 'Close']
            trade_results.append({'Close_Date': df.loc[s_idx, 'Date'], 'P/L': profit})
    trades_df = pd.DataFrame(trade_results)
    return df, trades_df

df1, trades1 = process_df(df1)
df2, trades2 = process_df(df2)

def get_kde(data, x_range):
    arr = np.asarray(data)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return x_range, np.zeros_like(x_range)
    kde = gaussian_kde(arr)
    return x_range, kde(x_range)

all_pl1 = trades1['P/L'].values if not trades1.empty else np.array([-1, 1])
all_pl1 = all_pl1[np.isfinite(all_pl1)]
kde_x1 = np.linspace(all_pl1.min() - 5, all_pl1.max() + 5, 200)

all_pl2 = trades2['P/L'].values if not trades2.empty else np.array([-1, 1])
all_pl2 = all_pl2[np.isfinite(all_pl2)]
kde_x2 = np.linspace(all_pl2.min() - 5, all_pl2.max() + 5, 200)

fig = make_subplots(
    rows=2, cols=2, 
    vertical_spacing=0.22,
    horizontal_spacing=0.1,
    subplot_titles=("Live Price Action (2020-2023)", "Live Price Action (2024-2026)", 
                    "P/L Distribution (2020-2023)", "P/L Distribution (2024-2026)"),
    row_heights=[0.6, 0.4]
)

def add_base_traces(fig, df, kde_x, col):
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0]], y=[df['Close'].iloc[0]], name=f"Price", line=dict(color='#00bfff'), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0]], y=[df['SMA_short'].iloc[0]], name=f"SMA Short", line=dict(dash='dot', color='#FFD700'), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0]], y=[df['SMA_long'].iloc[0]], name=f"SMA Long", line=dict(dash='dot', color='#ff00cc'), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='lime'), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='red'), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Winners', line=dict(color='lime'), opacity=0.6, showlegend=(col==1)), row=2, col=col)
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Losers', line=dict(color='red'), opacity=0.6, showlegend=(col==1)), row=2, col=col)

add_base_traces(fig, df1, kde_x1, 1)
add_base_traces(fig, df2, kde_x2, 2)

num_frames = 100
frame_indices1 = np.linspace(10, len(df1)-1, num_frames, dtype=int) if len(df1) > 10 else np.array([len(df1)-1])
frame_indices2 = np.linspace(10, len(df2)-1, num_frames, dtype=int) if len(df2) > 10 else np.array([len(df2)-1])

frames = []
for i in range(min(len(frame_indices1), len(frame_indices2))):
    idx1 = frame_indices1[i]
    idx2 = frame_indices2[i]
    current_date1 = df1['Date'].iloc[idx1]
    current_df1 = df1.iloc[:idx1+1]
    past_trades1 = trades1[trades1['Close_Date'] <= current_date1]
    current_date2 = df2['Date'].iloc[idx2]
    current_df2 = df2.iloc[:idx2+1]
    past_trades2 = trades2[trades2['Close_Date'] <= current_date2]
    
    def get_stats_and_kde(trades, kde_x):
        if not trades.empty:
            wins = trades[trades['P/L'] > 0]['P/L']
            wins = wins[np.isfinite(wins)]
            losses = trades[trades['P/L'] <= 0]['P/L']
            losses = losses[np.isfinite(losses)]
            total = len(wins) + len(losses)
            wr = len(wins) / total if total > 0 else 0
            aw = wins.mean() if len(wins) > 0 else 0
            al = losses.mean() if len(losses) > 0 else 0
            ev = np.concatenate([wins, losses]).mean() if total > 0 else 0
            _, win_y = get_kde(wins, kde_x)
            _, loss_y = get_kde(losses, kde_x)
        else:
            wr, aw, al, ev = 0, 0, 0, 0
            win_y = loss_y = np.zeros_like(kde_x)
        stats = (f"Win Rate: {wr:.1%}<br>Avg Win: {aw:.2f} | Avg Loss: {al:.2f}<br><b>EV: {ev:.2f}</b>")
        return stats, win_y, loss_y

    stats1, win_y1, loss_y1 = get_stats_and_kde(past_trades1, kde_x1)
    stats2, win_y2, loss_y2 = get_stats_and_kde(past_trades2, kde_x2)
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=current_df1['Date'], y=current_df1['Close']),
            go.Scatter(x=current_df1['Date'], y=current_df1['SMA_short']),
            go.Scatter(x=current_df1['Date'], y=current_df1['SMA_long']),
            go.Scatter(x=current_df1[current_df1['Trade_Signal'] == 1]['Date'], y=current_df1[current_df1['Trade_Signal'] == 1]['Close']),
            go.Scatter(x=current_df1[current_df1['Trade_Signal'] == -1]['Date'], y=current_df1[current_df1['Trade_Signal'] == -1]['Close']),
            go.Scatter(x=kde_x1, y=win_y1),
            go.Scatter(x=kde_x1, y=loss_y1),
            go.Scatter(x=current_df2['Date'], y=current_df2['Close']),
            go.Scatter(x=current_df2['Date'], y=current_df2['SMA_short']),
            go.Scatter(x=current_df2['Date'], y=current_df2['SMA_long']),
            go.Scatter(x=current_df2[current_df2['Trade_Signal'] == 1]['Date'], y=current_df2[current_df2['Trade_Signal'] == 1]['Close']),
            go.Scatter(x=current_df2[current_df2['Trade_Signal'] == -1]['Date'], y=current_df2[current_df2['Trade_Signal'] == -1]['Close']),
            go.Scatter(x=kde_x2, y=win_y2),
            go.Scatter(x=kde_x2, y=loss_y2)
        ],
        layout=go.Layout(annotations=[
            dict(x=0.225, y=-0.18, xref="paper", yref="paper", text=f"<b>2020-2023 Metrics</b><br>{stats1}", 
                 showarrow=False, align="center", xanchor="center", bgcolor="rgba(30, 30, 30, 0.8)"),
            dict(x=0.775, y=-0.18, xref="paper", yref="paper", text=f"<b>2024-2026 Metrics</b><br>{stats2}", 
                 showarrow=False, align="center", xanchor="center", bgcolor="rgba(30, 30, 30, 0.8)")
        ]),
        name=str(i)
    ))

fig.frames = frames

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000, 
    margin=dict(b=140, t=80, l=50, r=50), 
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.42,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(range=[df1['Date'].min(), df1['Date'].max()], autorange=False),
    yaxis=dict(range=[df1['Close'].min()*0.9, df1['Close'].max()*1.1], autorange=False),
    xaxis3=dict(title="P/L (2020-2023)"),
    xaxis2=dict(range=[df2['Date'].min(), df2['Date'].max()], autorange=False),
    yaxis2=dict(range=[df2['Close'].min()*0.9, df2['Close'].max()*1.1], autorange=False),
    xaxis4=dict(title="P/L (2024-2026)"),
    updatemenus=[{
        'type': 'buttons', 'showactive': False,
        'x': 0.05, 'y': 1.12,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}}]}
        ]
    }]
)

fig.show()
```



---

#### 2.) 📊 P/L Distribution Stability Over Time, Regime Modeling

We'll classify the volatility regime as the quadratic variation (squared returns) of the previous 20 days and break it up into treciles

- $ret^2$ < 33% $\implies$ Low Vol
- 33% < $ret^2$ < 66% $\implies$ Med Vol
- 66% < $ret^2$ < 99% $\implies$ High Vol

This is **NOT** a good way to decide the regime!

###### ______________________________________________________________________________________________________________________________________

Think about it, in periods of high volatility the 33% will contribute higher volatility than normal to the low volatility regime.  This is exactly what is providing instibility in that regime!  The same with the high volatility regime!  We are imposing instability in our own regime distributions!  The mid volatility regime suffers from this problem but on both sides so its likely going to be more stable!

In any case, we can develop regimes in a backtesting capacity with *any* feature

We can trade in this capacity, if we find stability, assuming we have access to the conditioning feature before or concurrently with the entry signal

Our objective is to develop regimes that are economically meaningful, or we can just leave this to a machine learning model (many more modern approaches do this) which we can discuss at the end of this video

###### ______________________________________________________________________________________________________________________________________

##### Backtesting an NVDA Trading Signal 2020 - 2024


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading (Real NVDA Data) ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# --- 2. Quadratic Variation & Terciles ---
df['Returns'] = df['Close'].pct_change()
df['QV'] = (df['Returns']**2).rolling(window=20).sum()
df = df.dropna(subset=['QV']).reset_index(drop=True)
df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])
df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')].reset_index(drop=True)

# --- 3. Strategy Logic ---
df['SMA_short'] = df['Close'].rolling(window=2).mean()
df['SMA_long'] = df['Close'].rolling(window=4).mean()
df['Signal'] = (df['SMA_short'] > df['SMA_long']).astype(int)
df['Position'] = df['Signal'].shift(1).fillna(0)
df['Trade_Signal'] = df['Position'].diff().fillna(0)

buy_indices = df.index[df['Trade_Signal'] == 1]
sell_indices = df.index[df['Trade_Signal'] == -1]
trade_results = []

for b_idx in buy_indices:
    potential_sells = sell_indices[sell_indices > b_idx]
    if not potential_sells.empty:
        s_idx = potential_sells[0]
        profit = df.loc[s_idx, 'Close'] - df.loc[b_idx, 'Close']
        trade_results.append({
            'Close_Date': df.loc[s_idx, 'Date'], 
            'P/L': profit, 
            'QV_Env': df.loc[b_idx, 'QV_Tercile']
        })

trades_df = pd.DataFrame(trade_results)

# --- 4. KDE Helper ---
def get_kde(data, x_range):
    if len(data) < 3: return np.zeros_like(x_range)
    try:
        kde = gaussian_kde(data)
        return kde(x_range)
    except: return np.zeros_like(x_range)

all_pl = trades_df['P/L'].dropna().values
kde_x = np.linspace(all_pl.min() * 1.2, all_pl.max() * 1.2, 100)

# --- 5. Figure Setup ---
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    vertical_spacing=0.15, 
    subplot_titles=("NVDA Live Price Action", "", "", ""),
    row_heights=[0.4, 0.6] 
)

fig.add_trace(go.Scatter(x=[], y=[], name="Price", line=dict(color='#00bfff')), row=1, col=1)

for col in [1, 2, 3]:
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Wins', line=dict(color='lime')), row=2, col=col)
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Losses', line=dict(color='red')), row=2, col=col)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=2, col=col)

# --- 6. Animation Frames ---
frame_indices = np.linspace(40, len(df)-1, 80, dtype=int)
frames = []

for idx in frame_indices:
    current_date = df['Date'].iloc[idx]
    curr_trades = trades_df[trades_df['Close_Date'] <= current_date]

    frame_data = [go.Scatter(x=df['Date'].iloc[:idx], y=df['Close'].iloc[:idx])]
    frame_annotations = []

    for i, regime in enumerate(['Low', 'Mid', 'High'], 1):
        reg_trades = curr_trades[curr_trades['QV_Env'] == regime]
        wins = reg_trades[reg_trades['P/L'] > 0]['P/L']
        losses = reg_trades[reg_trades['P/L'] <= 0]['P/L']

        ev = reg_trades['P/L'].mean() if not reg_trades.empty else 0
        avg_w = wins.mean() if not wins.empty else 0
        avg_l = losses.mean() if not losses.empty else 0

        win_y = get_kde(wins, kde_x)
        loss_y = get_kde(losses, kde_x)

        frame_data.extend([go.Scatter(x=kde_x, y=win_y), go.Scatter(x=kde_x, y=loss_y)])

        stats_text = f"EV: {ev:.2f}<br>AvgW: {avg_w:.2f}<br>AvgL: {avg_l:.2f}"
        y_pos = 0.82 if i == 1 else 0.95

        frame_annotations.append(dict(
            x=0.95, y=y_pos, xref=f"x{i+1} domain", yref=f"y{i+1} domain",
            text=stats_text, showarrow=False, bgcolor="rgba(0,0,0,0.4)", 
            font=dict(size=10, color="white"), align="right", xanchor='right'
        ))

    frames.append(go.Frame(data=frame_data, layout=go.Layout(annotations=frame_annotations), name=str(idx)))

fig.frames = frames

# --- 7. Final Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=450,
    width=1000,
    margin=dict(l=20, r=20, t=50, b=40),
    showlegend=False,
    xaxis1=dict(range=[df['Date'].min(), df['Date'].max()], gridcolor='rgba(255,255,255,0.1)'),
    yaxis1=dict(range=[df['Close'].min()*0.9, df['Close'].max()*1.1], gridcolor='rgba(255,255,255,0.1)'),

    xaxis2=dict(title="<b>LOW QV</b>", title_font=dict(size=10)),
    xaxis3=dict(title="<b>MID QV</b>", title_font=dict(size=10)),
    xaxis4=dict(title="<b>HIGH QV</b>", title_font=dict(size=10)),

    updatemenus=[{
        'type': 'buttons', 
        'direction': 'left',
        # Overlay in upper left of the price chart with relative positioning
        'x': 0.01,  # Near the left edge
        'y': 0.98,  # Near the top edge
        'xanchor': 'left',
        'yanchor': 'top',
        'showactive': False,
        'bgcolor': 'rgba(0,0,0,0)', 
        'bordercolor': 'rgba(0,0,0,0)',
        'font': {'color': 'rgba(255,255,255,0.8)', 'size': 12},
        'buttons': [
            {'label': '▶ PLAY', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]}
        ]
    }]
)

max_density = 0
for regime in ['Low', 'Mid', 'High']:
    r_trades = trades_df[trades_df['QV_Env'] == regime]
    if len(r_trades) > 3:
        dens = get_kde(r_trades['P/L'], kde_x).max()
        max_density = max(max_density, dens)

y_limit = max_density * 1.2 if max_density > 0 else 1.0

for i in [2, 3, 4]:
    fig.update_yaxes(range=[0, y_limit], row=2, col=i-1, gridcolor='rgba(255,255,255,0.05)')

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Backtesting the same NVDA Signal 2024 - Present


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading (Real NVDA Data) ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# --- 2. Quadratic Variation & Terciles ---
df['Returns'] = df['Close'].pct_change()
df['QV'] = (df['Returns']**2).rolling(window=20).sum()
df = df.dropna(subset=['QV']).reset_index(drop=True)
df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])
df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2026-12-31')].reset_index(drop=True)

# --- 3. Strategy Logic ---
df['SMA_short'] = df['Close'].rolling(window=2).mean()
df['SMA_long'] = df['Close'].rolling(window=4).mean()
df['Signal'] = (df['SMA_short'] > df['SMA_long']).astype(int)
df['Position'] = df['Signal'].shift(1).fillna(0)
df['Trade_Signal'] = df['Position'].diff().fillna(0)

buy_indices = df.index[df['Trade_Signal'] == 1]
sell_indices = df.index[df['Trade_Signal'] == -1]
trade_results = []

for b_idx in buy_indices:
    potential_sells = sell_indices[sell_indices > b_idx]
    if not potential_sells.empty:
        s_idx = potential_sells[0]
        profit = df.loc[s_idx, 'Close'] - df.loc[b_idx, 'Close']
        trade_results.append({
            'Close_Date': df.loc[s_idx, 'Date'], 
            'P/L': profit, 
            'QV_Env': df.loc[b_idx, 'QV_Tercile']
        })

trades_df = pd.DataFrame(trade_results)

# --- 4. KDE Helper ---
def get_kde(data, x_range):
    if len(data) < 3: return np.zeros_like(x_range)
    try:
        kde = gaussian_kde(data)
        return kde(x_range)
    except: return np.zeros_like(x_range)

all_pl = trades_df['P/L'].dropna().values
kde_x = np.linspace(all_pl.min() * 1.2, all_pl.max() * 1.2, 100)

# --- 5. Figure Setup ---
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    vertical_spacing=0.15, 
    subplot_titles=("NVDA Live Price Action", "", "", ""),
    row_heights=[0.4, 0.6] 
)

fig.add_trace(go.Scatter(x=[], y=[], name="Price", line=dict(color='#00bfff')), row=1, col=1)

for col in [1, 2, 3]:
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Wins', line=dict(color='lime')), row=2, col=col)
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', name='Losses', line=dict(color='red')), row=2, col=col)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=2, col=col)

# --- 6. Animation Frames ---
frame_indices = np.linspace(40, len(df)-1, 80, dtype=int)
frames = []

for idx in frame_indices:
    current_date = df['Date'].iloc[idx]
    curr_trades = trades_df[trades_df['Close_Date'] <= current_date]

    frame_data = [go.Scatter(x=df['Date'].iloc[:idx], y=df['Close'].iloc[:idx])]
    frame_annotations = []

    for i, regime in enumerate(['Low', 'Mid', 'High'], 1):
        reg_trades = curr_trades[curr_trades['QV_Env'] == regime]
        wins = reg_trades[reg_trades['P/L'] > 0]['P/L']
        losses = reg_trades[reg_trades['P/L'] <= 0]['P/L']

        ev = reg_trades['P/L'].mean() if not reg_trades.empty else 0
        avg_w = wins.mean() if not wins.empty else 0
        avg_l = losses.mean() if not losses.empty else 0

        win_y = get_kde(wins, kde_x)
        loss_y = get_kde(losses, kde_x)

        frame_data.extend([go.Scatter(x=kde_x, y=win_y), go.Scatter(x=kde_x, y=loss_y)])

        stats_text = f"EV: {ev:.2f}<br>AvgW: {avg_w:.2f}<br>AvgL: {avg_l:.2f}"
        y_pos = 0.82 if i == 1 else 0.95

        frame_annotations.append(dict(
            x=0.95, y=y_pos, xref=f"x{i+1} domain", yref=f"y{i+1} domain",
            text=stats_text, showarrow=False, bgcolor="rgba(0,0,0,0.4)", 
            font=dict(size=10, color="white"), align="right", xanchor='right'
        ))

    frames.append(go.Frame(data=frame_data, layout=go.Layout(annotations=frame_annotations), name=str(idx)))

fig.frames = frames

# --- 7. Final Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=450,
    width=1000,
    margin=dict(l=20, r=20, t=50, b=40),
    showlegend=False,
    xaxis1=dict(range=[df['Date'].min(), df['Date'].max()], gridcolor='rgba(255,255,255,0.1)'),
    yaxis1=dict(range=[df['Close'].min()*0.9, df['Close'].max()*1.1], gridcolor='rgba(255,255,255,0.1)'),

    xaxis2=dict(title="<b>LOW QV</b>", title_font=dict(size=10)),
    xaxis3=dict(title="<b>MID QV</b>", title_font=dict(size=10)),
    xaxis4=dict(title="<b>HIGH QV</b>", title_font=dict(size=10)),

    updatemenus=[{
        'type': 'buttons', 
        'direction': 'left',
        # Overlay in upper left of the price chart with relative positioning
        'x': 0.01,  # Near the left edge
        'y': 0.98,  # Near the top edge
        'xanchor': 'left',
        'yanchor': 'top',
        'showactive': False,
        'bgcolor': 'rgba(0,0,0,0)', 
        'bordercolor': 'rgba(0,0,0,0)',
        'font': {'color': 'rgba(255,255,255,0.8)', 'size': 12},
        'buttons': [
            {'label': '▶ PLAY', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]}
        ]
    }]
)

max_density = 0
for regime in ['Low', 'Mid', 'High']:
    r_trades = trades_df[trades_df['QV_Env'] == regime]
    if len(r_trades) > 3:
        dens = get_kde(r_trades['P/L'], kde_x).max()
        max_density = max(max_density, dens)

y_limit = max_density * 1.2 if max_density > 0 else 1.0

for i in [2, 3, 4]:
    fig.update_yaxes(range=[0, y_limit], row=2, col=i-1, gridcolor='rgba(255,255,255,0.05)')

fig.show()
```



###### ______________________________________________________________________________________________________________________________________


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# --- 1. Data Loading & Strategy Execution ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Volatility Context (QV)
df['Returns'] = df['Close'].pct_change()
df['QV'] = (df['Returns']**2).rolling(window=20).sum() * 10000 # Scaling for readability
df = df.dropna(subset=['QV']).reset_index(drop=True)
df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])

# SMA Strategy Logic
df['SMA_short'] = df['Close'].rolling(window=2).mean()
df['SMA_long'] = df['Close'].rolling(window=4).mean()
df['Signal'] = (df['SMA_short'] > df['SMA_long']).astype(int)
df['Position'] = df['Signal'].shift(1).fillna(0)
df['Trade_Signal'] = df['Position'].diff().fillna(0)

# Extract Trades
buy_indices = df.index[df['Trade_Signal'] == 1]
sell_indices = df.index[df['Trade_Signal'] == -1]
trade_results = []

for b_idx in buy_indices:
    potential_sells = sell_indices[sell_indices > b_idx]
    if not potential_sells.empty:
        s_idx = potential_sells[0]
        profit = df.loc[s_idx, 'Close'] - df.loc[b_idx, 'Close']
        trade_results.append({
            'Close_Date': df.loc[s_idx, 'Date'], 
            'P/L': profit, 
            'QV_Env': df.loc[b_idx, 'QV_Tercile']
        })

trades_df = pd.DataFrame(trade_results)

# --- 2. Helper Functions ---
def get_kde(data, x_range):
    if len(data) < 3: return np.zeros_like(x_range)
    try:
        kde = gaussian_kde(data)
        return kde(x_range)
    except: return np.zeros_like(x_range)

def plot_regime_pl(era_df, row_idx, era_label, fig):
    regimes = ['Low', 'Mid', 'High']

    for i, regime in enumerate(regimes):
        subset = era_df[era_df['QV_Env'] == regime]['P/L']
        if subset.empty: continue

        # Local X scaling for the regime
        x_min, x_max = subset.min(), subset.max()
        # If all P/L are identical, avoid zero-length
        if x_min == x_max:
            local_x = np.linspace(x_min - 1, x_max + 1, 200)
        else:
            local_x = np.linspace(x_min, x_max, 200)

        # Split Wins/Losses
        wins = subset[subset > 0]
        losses = subset[subset <= 0]

        win_y = get_kde(wins, local_x)
        loss_y = get_kde(losses, local_x)

        # Add Traces
        col_idx = i + 1
        fig.add_trace(go.Scatter(
            x=local_x, y=win_y, fill='tozeroy', name='Wins', 
            line=dict(color='rgba(0, 255, 150, 0.7)', width=1),
            showlegend=False
        ), row=row_idx, col=col_idx)
        fig.add_trace(go.Scatter(
            x=local_x, y=loss_y, fill='tozeroy', name='Losses', 
            line=dict(color='rgba(255, 50, 50, 0.7)', width=1),
            showlegend=False
        ), row=row_idx, col=col_idx)

        # --- Calculate precise axes domains for annotation positioning ---
        # After make_subplots, the .layout contains xaxis{j} (j=1...N). Find corresponding axis.
        # In plotly, xaxis1 is for (row=1,col=1), xaxis2 is (row=1,col=2), ..., xaxis4 is (row=2,col=1), etc
        # The mapping for subplot (row,col) is axis_number = (row-1)*cols + col
        cols = 3
        axis_number = (row_idx - 1) * cols + col_idx
        xaxis_key = f"xaxis{axis_number}" if axis_number > 1 else "xaxis"
        yaxis_key = f"yaxis{axis_number}" if axis_number > 1 else "yaxis"

        xaxis = fig.layout[xaxis_key]
        yaxis = fig.layout[yaxis_key]

        # Use axis.range in data units, fall back to trace limits if missing
        # If no explicit range, calculate from plotted data
        # x = far right; y = top
        if (getattr(xaxis, "range", None) and xaxis.range is not None and len(xaxis.range) == 2):
            x_annot = xaxis.range[1]  # far right
        else:
            x_annot = local_x.max()
        if (getattr(yaxis, "range", None) and yaxis.range is not None and len(yaxis.range) == 2):
            y_annot = yaxis.range[1]  # top
        else:
            # Use max of the win/loss KDEs
            y_annot = max(win_y.max() if win_y.size else 0, loss_y.max() if loss_y.size else 0)
            # Add a little padding
            y_annot = y_annot * 0.98 + 0.02 * y_annot

        # Some offset so it doesn't overlap the top right
        x_offset = (x_annot - local_x.min()) * 0.03
        y_offset = (y_annot - 0) * 0.05

        # Calculate stat values
        ev = subset.mean()
        wr = len(wins) / len(subset) if len(subset) > 0 else 0

        fig.add_annotation(
            x=x_annot - x_offset,
            y=y_annot - y_offset,
            xanchor='right', yanchor='top', showarrow=False,
            text=(
                f"<b style='font-size:14.5px'>{era_label} {regime}</b><br>"
                f"<span style='font-size:13px'>"
                f"EV: <b>{ev:.2f}</b><br>WR: <b>{wr:.1%}</b>"
                f"</span>"
            ),
            font=dict(size=15, color="white"),
            bordercolor="white",
            borderpad=11,
            borderwidth=2,
            bgcolor="rgba(0,0,0,0.8)",
            opacity=0.92,
            row=row_idx, col=col_idx,
            xref=f"x{axis_number}", yref=f"y{axis_number}"
        )

        # Zero Line
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="white", opacity=0.3, row=row_idx, col=col_idx)

# --- 3. Figure Construction ---
fig = make_subplots(
    rows=2, cols=3,
    vertical_spacing=0.21,
    horizontal_spacing=0.05,
    subplot_titles=[
        "Low QV Regime<br>2020-23", "Mid QV Regime<br>2020-23", "High QV Regime<br>2020-23",
        "Low QV Regime<br>2024-26", "Mid QV Regime<br>2024-26", "High QV Regime<br>2024-26"
    ]
)

era1_trades = trades_df[(trades_df['Close_Date'] >= '2020-01-01') & (trades_df['Close_Date'] <= '2023-12-31')]
era2_trades = trades_df[(trades_df['Close_Date'] >= '2024-01-01') & (trades_df['Close_Date'] <= '2026-12-31')]

plot_regime_pl(era1_trades, 1, "20-23", fig)
plot_regime_pl(era2_trades, 2, "24-26", fig)

# --- 4. Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=560, width=1100,
    showlegend=False,
    margin=dict(l=30, r=30, t=75, b=55),
    font=dict(size=14, family="Arial", color="white")
)

fig.update_xaxes(
    showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.14)',
    title_text="Trade P/L ($)", title_font=dict(size=13), tickfont=dict(size=11)
)
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)',
    showticklabels=False
)

fig.show()
```



---

#### 3.) 📈 Regime Stability

My Markov Chains or Hidden Markov Models "Don't Work"

They Work Fine, The Distribution(s) of Your Feature(s) is Unstable

Developing More Stationary (Stable) P/L Distributions Requires Quantitative Research

##### Quadratic Variation by Volatility Regime 2020 - 2024


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# --- 2. Quadratic Variation & Terciles ---
df['Returns'] = df['Close'].pct_change()
df['QV'] = (df['Returns']**2).rolling(window=20).sum()
df = df.dropna(subset=['QV']).reset_index(drop=True)
df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])
df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')].reset_index(drop=True)

# --- 3. KDE Setup ---
def get_kde(data, x_range):
    if len(data) < 3: return np.zeros_like(x_range)
    try:
        kde = gaussian_kde(data)
        kde_vals = kde(x_range)
        # Scale each distribution to have the same max height, for visibility
        if kde_vals.max() > 0:
            kde_vals = kde_vals / kde_vals.max()
        return kde_vals
    except: 
        return np.zeros_like(x_range)

qv_min, qv_max = df['QV'].min(), df['QV'].max()
kde_x = np.linspace(qv_min * 0.8, qv_max * 1.1, 200)

# --- 4. Figure Setup ---
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    subplot_titles=("NVDA Price Action", "Low Vol Distribution", "Mid Vol Distribution", "High Vol Distribution"),
    row_heights=[0.4, 0.6],
    vertical_spacing=0.15
)

fig.add_trace(go.Scatter(x=[], y=[], name="Price", line=dict(color='#00bfff')), row=1, col=1)

colors = ['#00FFCC', '#FFFF00', '#FF3300']
for i, col in enumerate([1, 2, 3]):
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', line=dict(color=colors[i])), row=2, col=col)

# --- 5. Animation Frames ---

# Scale all KDEs to have their own maximum at 1 (for easy comparison)
def get_normalized_kde(data, x_range):
    k = get_kde(data, x_range)
    return k # Already normalized in get_kde

frame_indices = np.linspace(40, len(df)-1, 60, dtype=int)
frames = [go.Frame(
    data=[go.Scatter(x=df.iloc[:idx]['Date'], y=df.iloc[:idx]['Close'])] + 
         [go.Scatter(x=kde_x, y=get_normalized_kde(df.iloc[:idx][df.iloc[:idx]['QV_Tercile'] == r]['QV'], kde_x)) for r in ['Low', 'Mid', 'High']],
    name=str(idx)
) for idx in frame_indices]
fig.frames = frames

# --- 6. Layout & Gray Play Button ---
fig.update_layout(
    template="plotly_dark",
    height=500, width=1000,
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis1=dict(range=[df['Date'].min(), df['Date'].max()], gridcolor='rgba(255,255,255,0.1)'),
    yaxis1=dict(range=[df['Close'].min()*0.9, df['Close'].max()*1.1], gridcolor='rgba(255,255,255,0.1)'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x': 0.0, 'y': 1.12,
        'xanchor': 'left', 'yanchor': 'top',
        'pad': {'t': 0, 'r': 10},
        'bgcolor': 'rgba(80, 80, 80, 0.8)',
        'bordercolor': '#888',
        'borderwidth': 1,
        'font': {'color': 'white', 'size': 11},
        'buttons': [{
            'label': '▶ PLAY',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

# Force y-axis of KDE plots to always show [0, 1.1]
for i in [1, 2, 3]:
    fig.update_xaxes(row=2, col=i, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(range=[0, 1.1], row=2, col=i, showticklabels=False, gridcolor='rgba(255,255,255,0.1)')

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Quadratic Variation by Volatility Regime 2024 - Present


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# --- 2. Quadratic Variation & Terciles ---
df['Returns'] = df['Close'].pct_change()
df['QV'] = (df['Returns']**2).rolling(window=20).sum()
df = df.dropna(subset=['QV']).reset_index(drop=True)
df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])
df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2026-12-31')].reset_index(drop=True)

# --- 3. KDE Setup ---
def get_kde(data, x_range):
    if len(data) < 3: return np.zeros_like(x_range)
    try:
        kde = gaussian_kde(data)
        kde_vals = kde(x_range)
        # Scale each distribution to have the same max height, for visibility
        if kde_vals.max() > 0:
            kde_vals = kde_vals / kde_vals.max()
        return kde_vals
    except: 
        return np.zeros_like(x_range)

qv_min, qv_max = df['QV'].min(), df['QV'].max()
kde_x = np.linspace(qv_min * 0.8, qv_max * 1.1, 200)

# --- 4. Figure Setup ---
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    subplot_titles=("NVDA Price Action", "Low Vol Distribution", "Mid Vol Distribution", "High Vol Distribution"),
    row_heights=[0.4, 0.6],
    vertical_spacing=0.15
)

fig.add_trace(go.Scatter(x=[], y=[], name="Price", line=dict(color='#00bfff')), row=1, col=1)

colors = ['#00FFCC', '#FFFF00', '#FF3300']
for i, col in enumerate([1, 2, 3]):
    fig.add_trace(go.Scatter(x=kde_x, y=np.zeros_like(kde_x), fill='tozeroy', line=dict(color=colors[i])), row=2, col=col)

# --- 5. Animation Frames ---

# Scale all KDEs to have their own maximum at 1 (for easy comparison)
def get_normalized_kde(data, x_range):
    k = get_kde(data, x_range)
    return k # Already normalized in get_kde

frame_indices = np.linspace(40, len(df)-1, 60, dtype=int)
frames = [go.Frame(
    data=[go.Scatter(x=df.iloc[:idx]['Date'], y=df.iloc[:idx]['Close'])] + 
         [go.Scatter(x=kde_x, y=get_normalized_kde(df.iloc[:idx][df.iloc[:idx]['QV_Tercile'] == r]['QV'], kde_x)) for r in ['Low', 'Mid', 'High']],
    name=str(idx)
) for idx in frame_indices]
fig.frames = frames

# --- 6. Layout & Gray Play Button ---
fig.update_layout(
    template="plotly_dark",
    height=500, width=1000,
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis1=dict(range=[df['Date'].min(), df['Date'].max()], gridcolor='rgba(255,255,255,0.1)'),
    yaxis1=dict(range=[df['Close'].min()*0.9, df['Close'].max()*1.1], gridcolor='rgba(255,255,255,0.1)'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x': 0.0, 'y': 1.12,
        'xanchor': 'left', 'yanchor': 'top',
        'pad': {'t': 0, 'r': 10},
        'bgcolor': 'rgba(80, 80, 80, 0.8)',
        'bordercolor': '#888',
        'borderwidth': 1,
        'font': {'color': 'white', 'size': 11},
        'buttons': [{
            'label': '▶ PLAY',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

# Force y-axis of KDE plots to always show [0, 1.1]
for i in [1, 2, 3]:
    fig.update_xaxes(row=2, col=i, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(range=[0, 1.1], row=2, col=i, showticklabels=False, gridcolor='rgba(255,255,255,0.1)')

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Regime Distribution Stability Dictates Regime Efficacy

We can use distribution distance measures like Kullback-Leibler (KL) Divergence to determine distribution distance

This is a measure of distribution stability, rather than simply observing individual realizations of data points impacting the overall EV of either side of the P/L distribution we have a quantitative measure of drift


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, entropy

# --- 1. Data Loading ---
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Returns'] = df['Close'].pct_change()
# Scale QV by 10,000 to make the X-axis readable (e.g., 0.0001 becomes 1.0)
df['QV'] = (df['Returns']**2).rolling(window=20).sum() * 10000
df = df.dropna(subset=['QV']).reset_index(drop=True)

df['QV_Tercile'] = pd.qcut(df['QV'], 3, labels=['Low', 'Mid', 'High'])

era1 = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')].copy()
era2 = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2026-12-31')].copy()

def get_kde(data, x_range):
    if len(data) < 5: return np.zeros_like(x_range)
    kde = gaussian_kde(data)
    # Use a slightly wider bandwidth if the plot looks too "jagged"
    return kde(x_range)

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=3,
    horizontal_spacing=0.08,
    subplot_titles=("Low Vol", "Mid Vol", "High Vol")
)

colors_era1 = 'rgba(0, 255, 204, 0.3)' # Transparent Cyan
colors_era2 = '#FF3300'               # Solid Orange-Red
regimes = ['Low', 'Mid', 'High']

for i, regime in enumerate(regimes):
    d1 = era1[era1['QV_Tercile'] == regime]['QV'].values
    d2 = era2[era2['QV_Tercile'] == regime]['QV'].values
    
    # Calculate local bounds for this specific subplot
    all_data = np.concatenate([d1, d2])
    x_min, x_max = np.percentile(all_data, [1, 99]) # Clip outliers for better zoom
    local_x = np.linspace(x_min * 0.8, x_max * 1.2, 200)
    
    # Generate KDEs
    kde1 = get_kde(d1, local_x)
    kde2 = get_kde(d2, local_x)
    
    # KL Divergence (normalized)
    p = kde1 / (np.sum(kde1) + 1e-10)
    q = kde2 / (np.sum(kde2) + 1e-10)
    kl_div = entropy(p, q)
    
    # Add Era 1 (History)
    fig.add_trace(go.Scatter(
        x=local_x, y=kde1, fill='tozeroy', 
        name=f"20-23 {regime}",
        line=dict(color=colors_era1, width=1),
    ), row=1, col=i+1)
    
    # Add Era 2 (Recent/Future)
    fig.add_trace(go.Scatter(
        x=local_x, y=kde2, 
        name=f"24-26 {regime}",
        line=dict(color=colors_era2, width=2.5),
    ), row=1, col=i+1)
    
    # Metrics Annotation
    fig.add_annotation(
        xref=f"x{i+1}", yref=f"y{i+1}",
        x=np.median(local_x), y=max(np.max(kde1), np.max(kde2)) * 1.05,
        text=f"KL: {kl_div:.4f}",
        showarrow=False, font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.4)", bordercolor="white", borderwidth=1
    )

    # Force individual scaling for THIS subplot
    fig.update_xaxes(range=[x_min*0.9, x_max*1.1], row=1, col=i+1, title_text="QV (x10k)")
    fig.update_yaxes(range=[0, max(np.max(kde1), np.max(kde2)) * 1.2], row=1, col=i+1)

# --- 3. Clean Transparent Layout ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=350, width=950,
    showlegend=False,
    margin=dict(l=40, r=20, t=60, b=40)
)

fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=False, showticklabels=False)

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

Machine learning aims to learn an non-linear expectation

Non-stationarity makes mispecification likely across time

There is plenty we can do as suggested in the literature to better learn more robust parameters

**My Old Professors @ Columbia (Stochastics & Time Series) Just Put Out a Great Paper on this Matter**

*Capponi, Agostino and Huang, Chengpiao and Huang, Chengpiao and Sidaoui, J. Antonio and Wang, Kaizheng and Zou, Jiacheng, The Nonstationarity-Complexity Tradeoff in Return Prediction (December 28, 2025). Available at SSRN: https://ssrn.com/abstract=5980654 or http://dx.doi.org/10.2139/ssrn.5980654*

Would love to do a video on it if there is interest...

---

#### 3.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**

- Asking if a trader is profitable is like asking if their portfolio did well last year, it isn't asking the right question
- A trader's edge goes one step further but the real traders profitable in the long term understand the time variant nature of their edge, whether in a discretionary or quantitative capacity they are profiting from the same statistical mechanism
- Distributions are not stable out of the box, the space is highly non-stationary which is why machine learning algorithms when naively applied may perform well in sample but extrapolate terribly
- Which is why we look for structure wherever possible (and economical) by different regimes for example: volatility, volume, sentiment, ...
- If we lack stability in our specification of regimes, we won't have any downstream stability either, a difficult problem and a data science problem, this is where most of the work is done, everyone wants to train a model or backtest their strategy but the real work is done feature engineering and building out the trading strategy in an economically (or statistically) meaningful way
- Should we find stability in our P/L distribution in an out of sample capacity (or a walk forward capacity) then we have a *tradable* strategy where we are not placing a trade blindly but rather have an idea of what data generating distribution we are drawing from

**Future Topics**

Technical Videos and Other Discussions

- Projects that Made me a Quant
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Poisson Processes for Quant Finance
- Top 3 Uses of Linear Algebra for Quant Finance
- Quant Roadmap: How I would Study if I Had to Start Over
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Reinforcement Learning for Delta Hedging
- Approximating Pricing Functionals using Neural Networks
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Financial News Sentiment Feed

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
