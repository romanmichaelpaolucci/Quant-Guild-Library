# ### 📈 Non-Stationarity and Why Market Timing Fails
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
# #### 1.) ⏳ The Purpose of Probability and Statistics
# 
# - Random Variables and Expected Value
# 
# - What If we Don't Know the Distribution?
# 
# - Stationarity and Non-Stationarity
# 
# #### 2.) ⚠️ Why Empirical Distributions are Misleading
# 
# - Kurtosis, Excess Kurtosis (Fat Tails), Return Distributions
# 
# - Return Distributions as a Compression
# 
# #### 3.) 📉 Why Market Timing is Difficult
# 
# - Which Distribution Does Data Belong To?
# 
# - Correct Distribution Classification Yields Better Decision Making
# 
# - Future Videos on HMMs, GMMs, etc...
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) ⏳ The Purpose of Probability and Statistics
# 
# The purpose of *"market timing"* is not to call tops and bottoms.
# 
# It's to correctly characterize the current latent distribution which is what we need to generate wealth trading.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Random Variables and Expected Value
# 
# Let the random variable $X$ represent the outcome of a fair six-sided die roll.
# 
# The probability mass function (PMF) and the expectation ($\mathbb{E}$) are given by:
# $$
# P(X = k) =
# \begin{cases}
# \dfrac{1}{6} & \text{if } k \in \{1, 2, 3, 4, 5, 6\} \\
# 0 & \text{otherwise}
# \end{cases}
# 
# \quad \quad
# 
# \mathbb{E}[X] = \sum_{k=1}^{6} k \cdot P(X=k) = \dfrac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5
# 
# $$
# 
# We can never predict the outcome of a dice roll, but we don't have to.
# 
# The expected value is the *best guess* by minimizing the mean squared error (MSE).
# 
# <u>**Two Ways to Generate Wealth:**</u>
# 
# 1. If we trade a spread around it we generate wealth.  
# 
# 2. If we approximate it *correctly* we generate wealth.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_steps = 100
true_val = 3.5
np.random.seed(42)

# --- Generate Order Book (Exponentially decaying from 3.5) ---
prices_bid = np.round(np.arange(1.0, 3.5, 0.1), 1)
prices_ask = np.round(np.arange(3.6, 6.1, 0.1), 1)

decay_rate = 1.5
vol_bid = np.exp(-decay_rate * (true_val - prices_bid))
vol_ask = np.exp(-decay_rate * (prices_ask - true_val))

prob_bid = vol_bid / np.sum(vol_bid)
prob_ask = vol_ask / np.sum(vol_ask)

all_prices = np.concatenate([prices_bid, prices_ask])
all_vols = np.concatenate([vol_bid, vol_ask])

# Base Colors: Opaque Dark Green for Bids, Opaque Dark Red for Asks
base_colors = ['rgba(0, 100, 0, 1.0)'] * len(prices_bid) + ['rgba(100, 0, 0, 1.0)'] * len(prices_ask)

# --- Simulate Trading & Wealth Accumulation ---
sides = np.random.choice(['bid', 'ask'], size=n_steps)
# Dice rolls representing the actual settled value of the asset (1 through 6)
dice_rolls = np.random.randint(1, 7, size=n_steps)

traded_prices = []
profits = []

for i, side in enumerate(sides):
    settlement_val = dice_rolls[i]
    
    if side == 'bid':
        # Market Maker buys at the bid. Profit is Settlement - Purchase Price
        p = np.random.choice(prices_bid, p=prob_bid)
        profit = settlement_val - p 
    else:
        # Market Maker sells at the ask. Profit is Sale Price - Settlement
        p = np.random.choice(prices_ask, p=prob_ask)
        profit = p - settlement_val
        
    traded_prices.append(p)
    profits.append(profit)

# Cumulative Wealth & Time Arrays
cum_profits = np.insert(np.cumsum(profits), 0, 0)
time_grid = np.arange(n_steps + 1)

# Prepend 3.5 to dice rolls just so the plot starts at the theoretical true value at t=0
plot_dice = np.insert(dice_rolls, 0, 3.5)

# Axis boundary buffers
min_profit = min(-5, np.min(cum_profits) * 1.1)
max_profit = max(5, np.max(cum_profits) * 1.1)

# --- Animation Frames ---
frames = []

for t in range(n_steps + 1):
    colors = list(base_colors)
    
    if t > 0:
        p_t = traded_prices[t-1]
        idx = np.where(np.isclose(all_prices, p_t))[0][0]
        
        if sides[t-1] == 'bid':
            colors[idx] = 'rgba(0, 255, 0, 1.0)'  # Bright Neon Green
        else:
            colors[idx] = 'rgba(255, 0, 0, 1.0)'  # Bright Neon Red

    # Trace 1: Order Book Bar (Left)
    bar_frame = go.Bar(
        x=all_prices, y=all_vols, 
        marker=dict(color=colors), hoverinfo='skip'
    )
    
    # Trace 2: Static True Value Line (Left)
    vline_frame = go.Scatter(
        x=[true_val, true_val], y=[0, max(all_vols)], 
        mode='lines', line=dict(color='yellow', width=2, dash='dash'), hoverinfo='skip'
    )
    
    # Trace 3: Dice Roll Stochastic Process (Top Right)
    dice_frame = go.Scatter(
        x=time_grid[:t + 1], y=plot_dice[:t + 1],
        mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'),
        marker=dict(size=6)
    )

    # Trace 4: Cumulative Wealth Path (Bottom Right)
    path_frame = go.Scatter(
        x=time_grid[:t + 1], y=cum_profits[:t + 1],
        mode='lines', line=dict(color='magenta', width=3, shape='hv')
    )

    frames.append(go.Frame(data=[bar_frame, vline_frame, dice_frame, path_frame], name=f"step{t}"))

# --- Initial Setup ---
bar_init = go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors), showlegend=False)
vline_init = go.Scatter(x=[true_val, true_val], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2, dash='dash'), showlegend=False)
dice_init = go.Scatter(x=[0], y=[3.5], mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), showlegend=False)
path_init = go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='magenta', width=3, shape='hv'), showlegend=False)

# --- Figure Structure ---
# Using specs to make the left column span two rows
fig = make_subplots(
    rows=2, cols=2, 
    column_widths=[0.4, 0.6],
    row_heights=[0.4, 0.6],
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=["Order Book Execution", "Asset Settlement (Dice Roll)", "Market-Maker Equity Curve"],
    horizontal_spacing=0.10,
    vertical_spacing=0.15
)

fig.add_trace(bar_init, row=1, col=1)
fig.add_trace(vline_init, row=1, col=1)
fig.add_trace(dice_init, row=1, col=2)
fig.add_trace(path_init, row=2, col=2)

fig.frames = frames

# --- Slider Configuration ---
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Trade No: "},
    pad={"t": 0},
    x=0.25, len=0.75, y=-0.1,
    steps=[dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=600, width=1000,
    title_text="Market Making: Taming Short-Term Chaos with Long-Term Edge",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=sliders,
    margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0, 'y': -0.15, 
        'xanchor': 'left', 'yanchor': 'top',
        'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# --- Axes Styling ---
# Left Plot
fig.update_xaxes(title_text='Price', range=[0.5, 6.5], row=1, col=1, showgrid=False)
fig.update_yaxes(title_text='Volume Mass', range=[0, max(all_vols) * 1.1], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

# Top Right Plot
fig.update_xaxes(range=[0, n_steps], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Settlement (1-6)', range=[0.5, 6.5], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)', dtick=1)

# Bottom Right Plot
fig.update_xaxes(title_text='Number of Trades (Time)', range=[0, n_steps], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative Profit', range=[min_profit, max_profit], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# Clearly, statistics are important if they dictate whether we are capable of generating wealth... 


# ###### ______________________________________________________________________________________________________________________________________


# ##### What If we Don't Know the Distribution?
# 
# In reality, we don't have a well-defined random variable for the returns of a stock, portfolio, or trading strategy.
# 
# So how can we approximate the expected value?
# 
# We can use the Law of Large Numbers (LLN).  It suggests if we have enough data we can approximate the mean.
# 
# Asymptotically it follows...
# 
#  $$
#  \overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{n \to \infty} \mathbb{E}[X]
#  $$
# 
# Let's Observe 50 Dice Rolls and Approximate the Expected Value.  Then we will use that to Trade.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_samples = 50    # LOW SAMPLE SIZE: Forces a bad estimate of the mean
n_steps = 100    # Number of trades in Phase 2
true_val = 3.5
mm_spread = 0.4  # The Market Maker's bid-ask spread
np.random.seed(42) # Yields an initial sample mean of 4.4 (Terrible estimate!)

# ==========================================
# --- PHASE 1: DATA GENERATION ---
# ==========================================
# 1. Generate the Estimation Dice Rolls
est_rolls = np.random.randint(1, 7, size=n_samples)
rolling_means = np.cumsum(est_rolls) / np.arange(1, n_samples + 1)
final_estimated_val = rolling_means[-1]

# 2. Generate Theoretical Order Book (Anchored at true_val = 3.5)
all_prices = np.round(np.arange(1.0, 6.1, 0.1), 1)

# STEEPENED DECAY RATE: Concentrates mass heavily around the true mid
decay_rate = 4.5 
all_vols = np.exp(-decay_rate * np.abs(all_prices - true_val))
all_probs = all_vols / np.sum(all_vols)

# Base Colors: Opaque Green for True Bids, Opaque Red for True Asks
base_colors = []
for p in all_prices:
    if p < true_val:
        base_colors.append('rgba(0, 150, 0, 0.6)')  # True Bids
    elif p > true_val:
        base_colors.append('rgba(150, 0, 0, 0.6)')  # True Asks
    else:
        base_colors.append('rgba(150, 150, 150, 0.6)') # True Mid

# ==========================================
# --- PHASE 2: TRADING SIMULATION ---
# ==========================================
trad_rolls = np.random.randint(1, 7, size=n_steps)
traded_arrival_prices = []
profits = []

mm_half_spread = mm_spread / 2.0
mm_bid = final_estimated_val - mm_half_spread
mm_ask = final_estimated_val + mm_half_spread

for t in range(n_steps):
    trade_executed = False
    
    # Wait for a market participant willing to cross the MM's spread
    while not trade_executed:
        # Market participant arrives with their own fair value 'p'
        p = np.random.choice(all_prices, p=all_probs)
        
        # ADVERSE SELECTION LOGIC
        if p <= mm_bid:
            # Participant values it less than MM's Bid. They SELL to the MM.
            # The MM BUYS at their own quoted Bid price.
            execution_price = mm_bid
            is_buy = True
            trade_executed = True
        elif p >= mm_ask:
            # Participant values it more than MM's Ask. They BUY from the MM.
            # The MM SELLS at their own quoted Ask price.
            execution_price = mm_ask
            is_buy = False
            trade_executed = True
            
    settlement_val = trad_rolls[t]
    
    # Calculate Profit based on the MM's Execution Price, not 'p'
    if is_buy:
        profit = settlement_val - execution_price 
    else:
        profit = execution_price - settlement_val
        
    traded_arrival_prices.append(p)
    profits.append(profit)

cum_profits = np.insert(np.cumsum(profits), 0, 0)
time_grid_trad = np.arange(n_steps + 1)
plot_trad_rolls = np.insert(trad_rolls, 0, final_estimated_val)

min_profit = min(-10, np.min(cum_profits) * 1.1)
max_profit = max(10, np.max(cum_profits) * 1.1)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []
total_frames = n_samples + n_steps + 1
hist_bins = np.arange(0.5, 7.5, 1)

for k in range(total_frames):
    if k <= n_samples:
        # Phase 1: Estimation
        current_est_rolls = est_rolls[:k]
        current_mean = rolling_means[k-1] if k > 0 else 3.5
        
        hist_counts, _ = np.histogram(current_est_rolls, bins=hist_bins)
        book_colors = list(base_colors)
        
        # Trading plots are empty in Phase 1
        t_x, t_y_dice, t_y_eq = [0], [current_mean], [0]
        
        # MM Quotes do not exist yet
        plot_mm_bid, plot_mm_ask = current_mean, current_mean
        frame_name = f"est{k}"
        
    else:
        # Phase 2: Trading
        t_trad = k - n_samples
        current_mean = final_estimated_val
        plot_mm_bid, plot_mm_ask = mm_bid, mm_ask
        
        hist_counts, _ = np.histogram(est_rolls, bins=hist_bins)
        book_colors = list(base_colors)
        
        # Highlight the market arrival price 'p' that hit the MM
        p_t = traded_arrival_prices[t_trad - 1]
        idx = np.where(np.isclose(all_prices, p_t))[0][0]
        
        if p_t <= mm_bid:
            book_colors[idx] = 'rgba(0, 255, 0, 1.0)'  # Bright Green (MM Bought from them)
        else:
            book_colors[idx] = 'rgba(255, 0, 0, 1.0)'  # Bright Red (MM Sold to them)
            
        t_x = time_grid_trad[:t_trad + 1]
        t_y_dice = plot_trad_rolls[:t_trad + 1]
        t_y_eq = cum_profits[:t_trad + 1]
        frame_name = f"trade{t_trad}"

    # Trace 0 & 1: Histogram & Mean Line
    tr0 = go.Bar(x=np.arange(1, 7), y=hist_counts, marker=dict(color='#00ffff'))
    tr1 = go.Scatter(x=[current_mean, current_mean], y=[0, max(1, np.max(hist_counts))], mode='lines', line=dict(color='yellow', width=3))
    
    # Trace 2: Bottom-Left Order Book Arrivals
    tr2 = go.Bar(x=all_prices, y=all_vols, marker=dict(color=book_colors))
    # Trace 3: Bottom-Left TRUE Value Line (Faint White)
    tr3 = go.Scatter(x=[true_val, true_val], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'))
    # Trace 4: Bottom-Left ESTIMATED Mid (Yellow Dashed)
    tr4 = go.Scatter(x=[current_mean, current_mean], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'))
    # Trace 5 & 6: MM Bid and Ask Quotes (Solid Yellow)
    tr5 = go.Scatter(x=[plot_mm_bid, plot_mm_bid], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2))
    tr6 = go.Scatter(x=[plot_mm_ask, plot_mm_ask], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2))

    # Trace 7: Top-Right Trading Dice Rolls
    tr7 = go.Scatter(x=t_x, y=t_y_dice, mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), marker=dict(size=4))
    # Trace 8: Bottom-Right Equity Curve
    tr8 = go.Scatter(x=t_x, y=t_y_eq, mode='lines', line=dict(color='magenta', width=3, shape='hv'))

    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8], name=frame_name))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Phase 1: Sampling & LLN Estimate", "Phase 2: Settlement Volatility", 
        "Phase 1 & 2: Market Arrivals vs MM Quotes", "Phase 2: MM Equity Curve"
    ],
    horizontal_spacing=0.10, vertical_spacing=0.15
)

# Add dummy traces in exact order
fig.add_trace(go.Bar(x=np.arange(1, 7), y=[0]*6, marker=dict(color='#00ffff'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, 1], mode='lines', line=dict(color='yellow', width=3), name='Estimated Mean'), row=1, col=1)

fig.add_trace(go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=[true_val, true_val], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'), name='True Value (3.5)'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'), name='MM Mid'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2), name='MM Bid'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2), name='MM Ask'), row=2, col=1)

fig.add_trace(go.Scatter(x=[0], y=[3.5], mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='magenta', width=3, shape='hv'), showlegend=False), row=2, col=2)

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
    height=650, width=1000,
    title_text=f"Market Making Parameter Risk: Destruction via Adverse Selection",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'), showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]}
        ]
    }]
)

# Axes Styling
fig.update_xaxes(title_text='Dice Roll', range=[0.5, 6.5], dtick=1, row=1, col=1, showgrid=False)
fig.update_yaxes(title_text='Count', row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Price', range=[0.5, 6.5], row=2, col=1, showgrid=False)
fig.update_yaxes(title_text='Volume Mass', range=[0, max(all_vols) * 1.1], row=2, col=1, showgrid=False)

fig.update_xaxes(title_text='Number of Trades', range=[0, n_steps], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Settlement (1-6)', range=[0.5, 6.5], dtick=1, row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Number of Trades', range=[0, n_steps], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative Profit', range=[min_profit, max_profit], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 
# Clearly our Estimate for the Expectation was Insufficient.  Let's just use More Data, the LLN Suggests we'll Eventually Converge to $\mathbb{E}[X]$.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_samples = 200    # LOW SAMPLE SIZE: Forces a bad estimate of the mean
n_steps = 100    # Number of trades in Phase 2
true_val = 3.5
mm_spread = 0.4  # The Market Maker's bid-ask spread
np.random.seed(42) # Yields an initial sample mean of 4.4 (Terrible estimate!)

# ==========================================
# --- PHASE 1: DATA GENERATION ---
# ==========================================
# 1. Generate the Estimation Dice Rolls
est_rolls = np.random.randint(1, 7, size=n_samples)
rolling_means = np.cumsum(est_rolls) / np.arange(1, n_samples + 1)
final_estimated_val = rolling_means[-1]

# 2. Generate Theoretical Order Book (Anchored at true_val = 3.5)
all_prices = np.round(np.arange(1.0, 6.1, 0.1), 1)

# STEEPENED DECAY RATE: Concentrates mass heavily around the true mid
decay_rate = 4.5 
all_vols = np.exp(-decay_rate * np.abs(all_prices - true_val))
all_probs = all_vols / np.sum(all_vols)

# Base Colors: Opaque Green for True Bids, Opaque Red for True Asks
base_colors = []
for p in all_prices:
    if p < true_val:
        base_colors.append('rgba(0, 150, 0, 0.6)')  # True Bids
    elif p > true_val:
        base_colors.append('rgba(150, 0, 0, 0.6)')  # True Asks
    else:
        base_colors.append('rgba(150, 150, 150, 0.6)') # True Mid

# ==========================================
# --- PHASE 2: TRADING SIMULATION ---
# ==========================================
trad_rolls = np.random.randint(1, 7, size=n_steps)
traded_arrival_prices = []
profits = []

mm_half_spread = mm_spread / 2.0
mm_bid = final_estimated_val - mm_half_spread
mm_ask = final_estimated_val + mm_half_spread

for t in range(n_steps):
    trade_executed = False
    
    # Wait for a market participant willing to cross the MM's spread
    while not trade_executed:
        # Market participant arrives with their own fair value 'p'
        p = np.random.choice(all_prices, p=all_probs)
        
        # ADVERSE SELECTION LOGIC
        if p <= mm_bid:
            # Participant values it less than MM's Bid. They SELL to the MM.
            # The MM BUYS at their own quoted Bid price.
            execution_price = mm_bid
            is_buy = True
            trade_executed = True
        elif p >= mm_ask:
            # Participant values it more than MM's Ask. They BUY from the MM.
            # The MM SELLS at their own quoted Ask price.
            execution_price = mm_ask
            is_buy = False
            trade_executed = True
            
    settlement_val = trad_rolls[t]
    
    # Calculate Profit based on the MM's Execution Price, not 'p'
    if is_buy:
        profit = settlement_val - execution_price 
    else:
        profit = execution_price - settlement_val
        
    traded_arrival_prices.append(p)
    profits.append(profit)

cum_profits = np.insert(np.cumsum(profits), 0, 0)
time_grid_trad = np.arange(n_steps + 1)
plot_trad_rolls = np.insert(trad_rolls, 0, final_estimated_val)

min_profit = min(-10, np.min(cum_profits) * 1.1)
max_profit = max(10, np.max(cum_profits) * 1.1)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []
total_frames = n_samples + n_steps + 1
hist_bins = np.arange(0.5, 7.5, 1)

for k in range(total_frames):
    if k <= n_samples:
        # Phase 1: Estimation
        current_est_rolls = est_rolls[:k]
        current_mean = rolling_means[k-1] if k > 0 else 3.5
        
        hist_counts, _ = np.histogram(current_est_rolls, bins=hist_bins)
        book_colors = list(base_colors)
        
        # Trading plots are empty in Phase 1
        t_x, t_y_dice, t_y_eq = [0], [current_mean], [0]
        
        # MM Quotes do not exist yet
        plot_mm_bid, plot_mm_ask = current_mean, current_mean
        frame_name = f"est{k}"
        
    else:
        # Phase 2: Trading
        t_trad = k - n_samples
        current_mean = final_estimated_val
        plot_mm_bid, plot_mm_ask = mm_bid, mm_ask
        
        hist_counts, _ = np.histogram(est_rolls, bins=hist_bins)
        book_colors = list(base_colors)
        
        # Highlight the market arrival price 'p' that hit the MM
        p_t = traded_arrival_prices[t_trad - 1]
        idx = np.where(np.isclose(all_prices, p_t))[0][0]
        
        if p_t <= mm_bid:
            book_colors[idx] = 'rgba(0, 255, 0, 1.0)'  # Bright Green (MM Bought from them)
        else:
            book_colors[idx] = 'rgba(255, 0, 0, 1.0)'  # Bright Red (MM Sold to them)
            
        t_x = time_grid_trad[:t_trad + 1]
        t_y_dice = plot_trad_rolls[:t_trad + 1]
        t_y_eq = cum_profits[:t_trad + 1]
        frame_name = f"trade{t_trad}"

    # Trace 0 & 1: Histogram & Mean Line
    tr0 = go.Bar(x=np.arange(1, 7), y=hist_counts, marker=dict(color='#00ffff'))
    tr1 = go.Scatter(x=[current_mean, current_mean], y=[0, max(1, np.max(hist_counts))], mode='lines', line=dict(color='yellow', width=3))
    
    # Trace 2: Bottom-Left Order Book Arrivals
    tr2 = go.Bar(x=all_prices, y=all_vols, marker=dict(color=book_colors))
    # Trace 3: Bottom-Left TRUE Value Line (Faint White)
    tr3 = go.Scatter(x=[true_val, true_val], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'))
    # Trace 4: Bottom-Left ESTIMATED Mid (Yellow Dashed)
    tr4 = go.Scatter(x=[current_mean, current_mean], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'))
    # Trace 5 & 6: MM Bid and Ask Quotes (Solid Yellow)
    tr5 = go.Scatter(x=[plot_mm_bid, plot_mm_bid], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2))
    tr6 = go.Scatter(x=[plot_mm_ask, plot_mm_ask], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2))

    # Trace 7: Top-Right Trading Dice Rolls
    tr7 = go.Scatter(x=t_x, y=t_y_dice, mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), marker=dict(size=4))
    # Trace 8: Bottom-Right Equity Curve
    tr8 = go.Scatter(x=t_x, y=t_y_eq, mode='lines', line=dict(color='magenta', width=3, shape='hv'))

    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8], name=frame_name))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Phase 1: Sampling & LLN Estimate", "Phase 2: Settlement Volatility", 
        "Phase 1 & 2: Market Arrivals vs MM Quotes", "Phase 2: MM Equity Curve"
    ],
    horizontal_spacing=0.10, vertical_spacing=0.15
)

# Add dummy traces in exact order
fig.add_trace(go.Bar(x=np.arange(1, 7), y=[0]*6, marker=dict(color='#00ffff'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, 1], mode='lines', line=dict(color='yellow', width=3), name='Estimated Mean'), row=1, col=1)

fig.add_trace(go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=[true_val, true_val], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'), name='True Value (3.5)'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'), name='MM Mid'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2), name='MM Bid'), row=2, col=1)
fig.add_trace(go.Scatter(x=[3.5, 3.5], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=2), name='MM Ask'), row=2, col=1)

fig.add_trace(go.Scatter(x=[0], y=[3.5], mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='magenta', width=3, shape='hv'), showlegend=False), row=2, col=2)

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
    height=650, width=1000,
    title_text=f"Market Making Parameter Risk: Destruction via Adverse Selection",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'), showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]}
        ]
    }]
)

# Axes Styling
fig.update_xaxes(title_text='Dice Roll', range=[0.5, 6.5], dtick=1, row=1, col=1, showgrid=False)
fig.update_yaxes(title_text='Count', row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Price', range=[0.5, 6.5], row=2, col=1, showgrid=False)
fig.update_yaxes(title_text='Volume Mass', range=[0, max(all_vols) * 1.1], row=2, col=1, showgrid=False)

fig.update_xaxes(title_text='Number of Trades', range=[0, n_steps], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Settlement (1-6)', range=[0.5, 6.5], dtick=1, row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Number of Trades', range=[0, n_steps], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative Profit', range=[min_profit, max_profit], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# This is pretty straightforward, the LLN seems to make things *too easy* - so what's the problem in reality?


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Stationarity and Non-Stationarity
# 
# I see folks Googling "how to make data stationary" and it drives me bananas, there is a fundamental lack of understanding here.
# 
# What is stationarity and non-stationarity?
# 
# Why can't we *ever* make something non-stationary stationary?
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# Imagine we did the above, approximating the expectation using the LLN, *one time*.  
# 
# We get our expected value and we trade it; we'll generate wealth.  But for how long?
# 
# If the distribution is *stationary* it will not change over time, so we'll generate wealth indefinitely trading that edge.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_samples = 200    
n_steps_p2 = 100   # Phase 2: D6 regime
n_steps_p3 = 100   # Phase 3: D10 regime
total_trades = n_steps_p2 + n_steps_p3

true_val_d6 = 3.5
true_val_d10 = 5.5  
mm_spread = 0.4
decay_rate = 4.5 
np.random.seed(42) 

# ==========================================
# --- ENVIRONMENT GENERATION ---
# ==========================================
all_prices = np.round(np.arange(1.0, 11.1, 0.1), 1)

# Order Book Probabilities
vols_d6 = np.exp(-decay_rate * np.abs(all_prices - true_val_d6))
probs_d6 = vols_d6 / np.sum(vols_d6)

vols_d10 = np.exp(-decay_rate * np.abs(all_prices - true_val_d10))
probs_d10 = vols_d10 / np.sum(vols_d10)

# True Distributions (for Top-Left Chart)
theo_x_d6 = np.arange(1, 7)
theo_y_d6 = np.ones(6) / 6.0

theo_x_d10 = np.arange(1, 11)
theo_y_d10 = np.ones(10) / 10.0

# Base Colors
colors_d6, colors_d10 = [], []
for p in all_prices:
    colors_d6.append('rgba(0, 150, 0, 0.6)' if p < true_val_d6 else 'rgba(150, 0, 0, 0.6)' if p > true_val_d6 else 'rgba(150, 150, 150, 0.6)')
    colors_d10.append('rgba(0, 150, 0, 0.6)' if p < true_val_d10 else 'rgba(150, 0, 0, 0.6)' if p > true_val_d10 else 'rgba(150, 150, 150, 0.6)')

# ==========================================
# --- PHASE 1: DATA GENERATION (D6) ---
# ==========================================
est_rolls = np.random.randint(1, 7, size=n_samples)
rolling_means = np.cumsum(est_rolls) / np.arange(1, n_samples + 1)
final_estimated_val = rolling_means[-1]

mm_half_spread = mm_spread / 2.0
mm_bid = final_estimated_val - mm_half_spread
mm_ask = final_estimated_val + mm_half_spread

# ==========================================
# --- PHASE 2 & 3: TRADING SIMULATION ---
# ==========================================
trad_rolls = np.concatenate([
    np.random.randint(1, 7, size=n_steps_p2),   # D6 Rolls
    np.random.randint(1, 11, size=n_steps_p3)   # D10 Rolls
])

traded_arrival_prices = []
profits = []

for t in range(total_trades):
    current_probs = probs_d6 if t < n_steps_p2 else probs_d10
    trade_executed = False
        
    while not trade_executed:
        p = np.random.choice(all_prices, p=current_probs)
        if p <= mm_bid:
            execution_price, is_buy, trade_executed = mm_bid, True, True
        elif p >= mm_ask:
            execution_price, is_buy, trade_executed = mm_ask, False, True
            
    settlement_val = trad_rolls[t]
    profit = (settlement_val - execution_price) if is_buy else (execution_price - settlement_val)
        
    traded_arrival_prices.append(p)
    profits.append(profit)

cum_profits = np.insert(np.cumsum(profits), 0, 0)
time_grid_trad = np.arange(total_trades + 1)
plot_trad_rolls = np.insert(trad_rolls, 0, final_estimated_val)

min_profit, max_profit = min(-20, np.min(cum_profits) * 1.1), max(20, np.max(cum_profits) * 1.1)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []
total_frames = n_samples + total_trades + 1
hist_bins = np.arange(0.5, 11.5, 1)

for k in range(total_frames):
    t_trad = max(0, k - n_samples)
    
    # Expand the X-axis for the right column dynamically, with a slight padding
    current_x_range = [0, max(10, t_trad + 5)]
    
    if k <= n_samples:
        current_est_rolls = est_rolls[:k]
        current_mean = rolling_means[k-1] if k > 0 else 3.5
        hist_counts, _ = np.histogram(current_est_rolls, bins=hist_bins, density=True)
        
        book_vols, book_colors = vols_d6, list(colors_d6)
        true_val_line = true_val_d6
        theo_x, theo_y = theo_x_d6, theo_y_d6
        theo_color, theo_line = 'rgba(0, 255, 255, 0.2)', '#00ffff'
        
        # Use np.nan for empty charts to prevent Plotly from drawing artifacts
        x_d6, y_d6 = [np.nan], [np.nan]
        x_d10, y_d10 = [np.nan], [np.nan]
        t_x_eq, t_y_eq = [np.nan], [np.nan]
        
        plot_mm_bid, plot_mm_ask = current_mean, current_mean
        frame_name = f"est{k}"
        
    else:
        current_mean = final_estimated_val
        plot_mm_bid, plot_mm_ask = mm_bid, mm_ask
        hist_counts, _ = np.histogram(est_rolls, bins=hist_bins, density=True)
        
        if t_trad <= n_steps_p2:
            book_vols, book_colors = vols_d6, list(colors_d6)
            true_val_line = true_val_d6
            theo_x, theo_y = theo_x_d6, theo_y_d6
            theo_color, theo_line = 'rgba(0, 255, 255, 0.2)', '#00ffff'
            
            x_d6 = time_grid_trad[:t_trad + 1]
            y_d6 = plot_trad_rolls[:t_trad + 1]
            x_d10, y_d10 = [np.nan], [np.nan]
        else:
            book_vols, book_colors = vols_d10, list(colors_d10)
            true_val_line = true_val_d10
            theo_x, theo_y = theo_x_d10, theo_y_d10
            theo_color, theo_line = 'rgba(57, 255, 20, 0.2)', '#39ff14' 
            
            x_d6 = time_grid_trad[:n_steps_p2 + 1]
            y_d6 = plot_trad_rolls[:n_steps_p2 + 1]
            x_d10 = time_grid_trad[n_steps_p2:t_trad + 1]
            y_d10 = plot_trad_rolls[n_steps_p2:t_trad + 1]
        
        p_t = traded_arrival_prices[t_trad - 1]
        idx = np.where(np.isclose(all_prices, p_t))[0][0]
        book_colors[idx] = 'rgba(0, 255, 0, 1.0)' if p_t <= mm_bid else 'rgba(255, 0, 0, 1.0)' 
            
        t_x_eq = time_grid_trad[:t_trad + 1]
        t_y_eq = cum_profits[:t_trad + 1]
        frame_name = f"trade{t_trad}"

    tr0 = go.Bar(x=np.arange(1, 12), y=hist_counts, marker=dict(color='rgba(0,255,255,0.6)'))
    tr1 = go.Bar(x=theo_x, y=theo_y, marker=dict(color=theo_color, line=dict(color=theo_line, width=2)))
    tr2 = go.Scatter(x=[current_mean, current_mean], y=[0, 0.25], mode='lines', line=dict(color='yellow', width=3))
    
    tr3 = go.Bar(x=all_prices, y=book_vols, marker=dict(color=book_colors))
    tr4 = go.Scatter(x=[true_val_line, true_val_line], y=[0, 1], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash'))
    tr5 = go.Scatter(x=[current_mean, current_mean], y=[0, 1], mode='lines', line=dict(color='yellow', width=3, dash='dash'))
    tr6 = go.Scatter(x=[plot_mm_bid, plot_mm_bid], y=[0, 1], mode='lines', line=dict(color='yellow', width=2))
    tr7 = go.Scatter(x=[plot_mm_ask, plot_mm_ask], y=[0, 1], mode='lines', line=dict(color='yellow', width=2))

    tr8 = go.Scatter(x=x_d6, y=y_d6, mode='lines+markers', line=dict(color='#00ffff', width=2, shape='hv'), marker=dict(size=4))
    tr9 = go.Scatter(x=x_d10, y=y_d10, mode='lines+markers', line=dict(color='#39ff14', width=2, shape='hv'), marker=dict(size=4)) 
    tr10 = go.Scatter(x=t_x_eq, y=t_y_eq, mode='lines', line=dict(color='magenta', width=3, shape='hv'))

    # Update the layout in the frame to push the X-axis forward
    frames.append(go.Frame(
        data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8, tr9, tr10], 
        name=frame_name,
        layout=go.Layout(xaxis2=dict(range=current_x_range), xaxis4=dict(range=current_x_range))
    ))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Phase 1: True Distribution vs LLN Sampling", "Phase 2 & 3: Settlement Volatility", 
        "Market Arrivals vs Frozen MM Quotes", "Phase 2 & 3: MM Equity Curve"
    ],
    horizontal_spacing=0.10, vertical_spacing=0.15
)

fig.add_trace(go.Bar(x=[np.nan], y=[np.nan]), row=1, col=1)
fig.add_trace(go.Bar(x=[np.nan], y=[np.nan]), row=1, col=1)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=1, col=1)

fig.add_trace(go.Bar(x=[np.nan], y=[np.nan]), row=2, col=1)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1)

fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=1, col=2)
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=1, col=2)

fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=2)

fig.frames = frames

# ==========================================
# --- SLIDER & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.1,
    steps=[dict(
        method="animate",
        args=[[frames[k].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f"Est {k}" if k <= n_samples else f"Trade {k - n_samples}"
    ) for k in range(total_frames)]
)]

fig.update_layout(
    height=650, width=1000,
    title_text="Regime Shift: The Danger of Non-Stationarity in Market Making",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            # CRITICAL FIX: Set transition duration to 0 and mode to immediate so Plotly doesn't freeze the traces
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes Styling
fig.update_xaxes(title_text='Dice Roll', range=[0.5, 11.5], dtick=1, row=1, col=1, showgrid=False)
fig.update_yaxes(title_text='Probability', range=[0, 0.25], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Price', range=[0.5, 11.5], dtick=1, row=2, col=1, showgrid=False)
fig.update_yaxes(title_text='Volume Mass', range=[0, 1.1], row=2, col=1, showgrid=False)

fig.update_xaxes(title_text='Number of Trades', range=[0, 10], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Settlement (1-10)', range=[0.5, 11.5], dtick=1, row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Number of Trades', range=[0, 10], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative Profit', range=[min_profit, max_profit], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# In reality, distributions are *non-stationary* so we won't generate wealth indefinitely.
# 
# When the distribution changes, we'll be using a stale expected value approximation that no longer represents a *"best guess"*.
# 
# **Two Keys to Understanding This**
# 1. The Two Weighted Dice Example
# 2. The Reality of Modeling and What Non-Stationarity *Actually* Is


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_steps = 200
np.random.seed(42) 

# Generate True Hidden States (Regimes) via a random transition matrix
# 5% chance to switch regimes on any given step
regimes = np.zeros(n_steps, dtype=int)
current_regime = 0
for i in range(n_steps):
    if np.random.rand() < 0.05:  
        current_regime = 1 - current_regime
    regimes[i] = current_regime

# Generate Observations (The Dice Rolls)
rolls = np.zeros(n_steps, dtype=int)
for i in range(n_steps):
    if regimes[i] == 0:
        rolls[i] = np.random.randint(1, 7)   # D6
    else:
        rolls[i] = np.random.randint(1, 11)  # D10

time_grid = np.arange(1, n_steps + 1)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []

for k in range(1, n_steps + 1):
    t_x = time_grid[:k]
    t_y = rolls[:k]
    
    # Split the Y-data based on the true hidden regime for color coding
    t_y_d6 = np.where(regimes[:k] == 0, t_y, np.nan)
    t_y_d10 = np.where(regimes[:k] == 1, t_y, np.nan)

    # Trace 0: Top Chart (Raw Observations - Unknown Regime)
    tr0 = go.Scatter(
        x=t_x, y=t_y, 
        mode='lines+markers', 
        line=dict(color='white', width=2, shape='hv'), 
        marker=dict(color='white', size=6)
    )
    
    # Trace 1: Bottom Chart (Faint connecting line)
    tr1 = go.Scatter(
        x=t_x, y=t_y, 
        mode='lines', 
        line=dict(color='rgba(255, 255, 255, 0.2)', width=2, shape='hv')
    )
    
    # Trace 2: Bottom Chart (D6 Markers - Cyan)
    tr2 = go.Scatter(
        x=t_x, y=t_y_d6, 
        mode='markers', 
        marker=dict(color='#00ffff', size=8)
    )
    
    # Trace 3: Bottom Chart (D10 Markers - Neon Green)
    tr3 = go.Scatter(
        x=t_x, y=t_y_d10, 
        mode='markers', 
        marker=dict(color='#39ff14', size=8)
    )

    # Notice we removed the layout update here so the axes remain completely static
    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3], name=f"step{k}"))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=[
        "Raw Observations (What the Market Maker Sees)", 
        "True Hidden Regimes (The Underlying Reality)"
    ],
    vertical_spacing=0.15
)

# Dummy traces matching the exact order of the frame data
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=1, col=1) # Top Raw
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1) # Bottom Line
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan], name='D6 Regime'), row=2, col=1) # Bottom D6
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan], name='D10 Regime'), row=2, col=1) # Bottom D10

fig.frames = frames

# ==========================================
# --- SLIDER & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Trade No: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.1,
    steps=[dict(
        method="animate",
        args=[[frames[k-1].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(1, n_steps + 1)]
)]

fig.update_layout(
    height=650, width=1000,
    title_text="Structural Breaks: The Illusion of Market Continuity",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Axes Styling (Static Range enforced here!)
fig.update_xaxes(title_text='Time Step', range=[0, n_steps + 5], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Dice Roll Value', range=[0.5, 11.5], dtick=1, row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Time Step', range=[0, n_steps + 5], row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Dice Roll Value', range=[0.5, 11.5], dtick=1, row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# We may find that a distribution we observe in reality appears *"stable"* emulating stationarity sufficiently for wealth generation.
# 
# This is our goal.  It is difficult in practice.  Many if not all statistics are spectacularly misleading. 


# We are treating real life as a random variable - it is not, real decisions and the environment are actually generating the returns, not some theoretical distribution.  We are simply imposing structure to hopefully make more informed decisions rooted in quantitative reasoning rather than "feel etc."


# ---


# #### 2.) ⚠️ Why Empirical Distributions are Misleading
# 
# ##### Kurtosis, Excess Kurtosis (Fat Tails), Return Distributions
# 
# In the context of the above, one of the first things higher education preaches is excess kurtosis.
# 
# This is objectively true, but extremely misleading.
# 
# Let's observe the return distribution for SPX


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, gaussian_kde, kurtosis

# ==========================================
# --- 1. LOAD AND PROCESS DATA ---
# ==========================================
# Assuming 'spx.csv' has columns: Date, SPY, VIX
df = pd.read_csv('spx.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Calculate daily simple returns
df['Returns'] = df['SPY'].pct_change()
returns = df['Returns'].dropna().values

# ==========================================
# --- 2. STATISTICAL FITTING ---
# ==========================================
# Fit Parametric Normal Distribution
mu, std = norm.fit(returns)

# Calculate Excess Kurtosis (Normal Dist = 0)
excess_kurt = kurtosis(returns, fisher=True)

# Create an X-axis grid that stretches slightly beyond the min/max returns
x_grid = np.linspace(returns.min() * 1.2, returns.max() * 1.2, 1000)

# Generate PDF for Normal
pdf_normal = norm.pdf(x_grid, mu, std)

# Fit Empirical Kernel Density Estimate (KDE)
kde = gaussian_kde(returns)
pdf_kde = kde.evaluate(x_grid)

# Calculate empirical vs theoretical probabilities of a > 3 sigma drop
minus_3_sigma = mu - 3 * std
theoretical_prob = norm.cdf(minus_3_sigma, mu, std)
empirical_prob = len(returns[returns < minus_3_sigma]) / len(returns)

# ==========================================
# --- 3. VISUALIZATION ---
# ==========================================
fig = go.Figure()

# Trace 1: Empirical Histogram
fig.add_trace(go.Histogram(
    x=returns,
    histnorm='probability density',
    nbinsx=150,
    name='Empirical Data',
    marker=dict(color='rgba(0, 255, 255, 0.3)', line=dict(color='#00ffff', width=1)),
    hoverinfo='skip'
))

# Trace 2: Parametric Normal Distribution
fig.add_trace(go.Scatter(
    x=x_grid, y=pdf_normal,
    mode='lines',
    name=f'Normal Dist N(μ, σ²)',
    line=dict(color='yellow', width=3, dash='dash')
))

# Trace 3: Empirical KDE
fig.add_trace(go.Scatter(
    x=x_grid, y=pdf_kde,
    mode='lines',
    name='Empirical KDE',
    line=dict(color='#39ff14', width=3)  # Neon Green
))

# Add a shaded region for the -3 Sigma "Fat Tail"
tail_x = x_grid[x_grid <= minus_3_sigma]
tail_y_kde = pdf_kde[x_grid <= minus_3_sigma]

fig.add_trace(go.Scatter(
    x=np.concatenate([tail_x, tail_x[::-1]]),
    y=np.concatenate([tail_y_kde, np.zeros_like(tail_y_kde)]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.4)',
    line=dict(color='rgba(255,0,0,0)'),
    name='Empirical -3σ Tail Risk',
    hoverinfo='skip'
))

# Add Vertical Line for -3 Standard Deviations
fig.add_vline(x=minus_3_sigma, line_width=2, line_dash="dot", line_color="red")
fig.add_annotation(
    x=minus_3_sigma, y=max(pdf_kde) * 0.5, 
    text="-3 Sigma Event", 
    showarrow=False, textangle=-90, 
    xanchor='right', xshift=-5, 
    font=dict(color='red', size=14)
)

# Text Box summarizing the Fat Tail problem
summary_text = (
    f"<b>Distribution Analysis</b><br>"
    f"Excess Kurtosis: {excess_kurt:.2f}<br><br>"
    f"<b>Probability of < -3σ Drop:</b><br>"
    f"Theory (Normal): {theoretical_prob*100:.4f}%<br>"
    f"Reality (Empirical): {empirical_prob*100:.4f}%"
)

fig.add_annotation(
    x=0.98, y=0.95, xref='paper', yref='paper',
    text=summary_text,
    showarrow=False,
    align='left',
    bgcolor='rgba(30, 30, 30, 0.8)',
    bordercolor='#00ffff',
    borderwidth=1,
    font=dict(color='white', size=13)
)

# ==========================================
# --- 4. LAYOUT STYLING ---
# ==========================================
fig.update_layout(
    height=650, width=1000,
    title_text="Empirical Returns vs Parametric Assumption: The 'Fat Tail' Problem",
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=80, b=50, l=50, r=50)
)

# Axes Styling
fig.update_xaxes(
    title_text='Daily Return', 
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=2
)
fig.update_yaxes(
    title_text='Probability Density', 
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    zeroline=False
)

fig.show()


# Relative to a normal (Gaussian) distribution, extremes are far more likely.
# 
# But this is a compression.


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Return Distributions as a Compression
# 
# When we visualize a return distribution (stock, ETF, portfolio, trading strat) we are compressing an unfathomable amount of information.
# 
# We have no idea what the *n-dimensional blob* of a data generating distribution looks like at any given time.
# 
# Remember, there isn't any purely analytical *"mathematical or statistical"* representation for what we are observing.
# 
# We are *building models* to attempt to explain this behavior but they are only *a component* of decision making.
# 


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ==========================================
# --- 1. LOAD AND PROCESS DATA ---
# ==========================================
# Assuming 'spx.csv' has columns: Date, SPY, VIX
try:
    df = pd.read_csv('spx.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
except FileNotFoundError:
    # Fallback dummy data generation just in case the CSV isn't in the exact path
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=3000, freq='B')
    vix_sim = np.random.lognormal(mean=2.8, sigma=0.4, size=3000)
    spy_sim = np.cumprod(1 + np.random.normal(0.0005 - 0.0001*vix_sim, 0.001*vix_sim)) * 100
    df = pd.DataFrame({'Date': dates, 'SPY': spy_sim, 'VIX': vix_sim})

# Calculate daily simple returns and drop missing values
df['Returns'] = df['SPY'].pct_change()
df_clean = df.dropna(subset=['Returns', 'VIX']).copy()

# ==========================================
# --- 2. REGIME CLASSIFICATION & STATS ---
# ==========================================
returns_all = df_clean['Returns'].values
mu_all, std_all = norm.fit(returns_all)

# Define the "Left Tail Event" threshold (e.g., an overall -3 Sigma crash)
tail_threshold = mu_all - 3 * std_all

# Split the market into 3 equal-sized regimes based on VIX levels
q33 = df_clean['VIX'].quantile(0.333)
q67 = df_clean['VIX'].quantile(0.667)

returns_low = df_clean[df_clean['VIX'] <= q33]['Returns'].values
returns_med = df_clean[(df_clean['VIX'] > q33) & (df_clean['VIX'] <= q67)]['Returns'].values
returns_high = df_clean[df_clean['VIX'] > q67]['Returns'].values

# Fit Parametric Normal Distributions for EACH regime
mu_low, std_low = norm.fit(returns_low)
mu_med, std_med = norm.fit(returns_med)
mu_high, std_high = norm.fit(returns_high)

# Calculate conditional probabilities of hitting the Left Tail Event threshold
prob_tail_low = norm.cdf(tail_threshold, mu_low, std_low)
prob_tail_med = norm.cdf(tail_threshold, mu_med, std_med)
prob_tail_high = norm.cdf(tail_threshold, mu_high, std_high)
empirical_prob = len(returns_all[returns_all < tail_threshold]) / len(returns_all)

# Create an X-axis grid spanning the extremes
x_grid = np.linspace(returns_all.min() * 1.1, returns_all.max() * 1.1, 1000)

# Generate PDFs for the 3 Regimes
pdf_low = norm.pdf(x_grid, mu_low, std_low)
pdf_med = norm.pdf(x_grid, mu_med, std_med)
pdf_high = norm.pdf(x_grid, mu_high, std_high)

# ==========================================
# --- 3. VISUALIZATION ---
# ==========================================
fig = go.Figure()

# Trace 1: Overall Empirical Histogram (Background)
fig.add_trace(go.Histogram(
    x=returns_all,
    histnorm='probability density',
    nbinsx=150,
    name='All Empirical Returns',
    marker=dict(color='rgba(0, 255, 255, 0.2)', line=dict(color='rgba(0, 255, 255, 0.5)', width=1)),
    hoverinfo='skip'
))

# Trace 2: Low Vol Regime (Green)
fig.add_trace(go.Scatter(
    x=x_grid, y=pdf_low, mode='lines',
    name=f'Low Vol (VIX < {q33:.1f})',
    line=dict(color='#39ff14', width=3), # Neon Green
    fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)'
))

# Trace 3: Medium Vol Regime (Yellow)
fig.add_trace(go.Scatter(
    x=x_grid, y=pdf_med, mode='lines',
    name=f'Med Vol ({q33:.1f} - {q67:.1f})',
    line=dict(color='yellow', width=3),
    fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.1)'
))

# Trace 4: High Vol Regime (Red)
fig.add_trace(go.Scatter(
    x=x_grid, y=pdf_high, mode='lines',
    name=f'High Vol (VIX > {q67:.1f})',
    line=dict(color='red', width=3),
    fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)'
))

# Add Vertical Line for the Left Tail Event Threshold
fig.add_vline(x=tail_threshold, line_width=2, line_dash="dash", line_color="white")
fig.add_annotation(
    x=tail_threshold, y=max(pdf_med) * 0.8, 
    text=f"Left Tail Event (<-3σ Overall)", 
    showarrow=False, textangle=-90, 
    xanchor='right', xshift=-8, 
    font=dict(color='white', size=14)
)

# Text Box summarizing the Conditional Probabilities (MOVED TO UPPER RIGHT)
summary_text = (
    f"<b>Probability of Left Tail Event (<-3σ Overall):</b><br>"
    f"<i>Empirical Baseline: {empirical_prob*100:.4f}%</i><br><br>"
    f"<b><span style='color:#39ff14'>Low Vol Regime:</span></b> "
    f"{prob_tail_low*100:.6f}% <i>(Virtually Impossible)</i><br><br>"
    f"<b><span style='color:yellow'>Med Vol Regime:</span></b> "
    f"{prob_tail_med*100:.4f}%<br><br>"
    f"<b><span style='color:red'>High Vol Regime:</span></b> "
    f"{prob_tail_high*100:.4f}% <i>(Where Crashes Live)</i>"
)

fig.add_annotation(
    x=0.98, y=0.95, xref='paper', yref='paper',
    xanchor='right', yanchor='top',
    text=summary_text, showarrow=False, align='left',
    bgcolor='rgba(30, 30, 30, 0.85)', bordercolor='white', borderwidth=1,
    font=dict(color='white', size=14)
)

# ==========================================
# --- 4. LAYOUT STYLING ---
# ==========================================
fig.update_layout(
    height=650, width=1000,
    title_text="Conditional Risk: Unmasking the Probability of Left Tail Events by Regime",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    # Legend moved to Upper Left to make room for the annotation box
    legend=dict(x=0.02, y=0.95, xanchor='left', yanchor='top', bgcolor='rgba(30, 30, 30, 0.8)', bordercolor='white', borderwidth=1),
    margin=dict(t=80, b=50, l=50, r=50)
)

# Axes Styling
fig.update_xaxes(
    title_text='Daily Return Magnitude', 
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.8)', zerolinewidth=2
)
fig.update_yaxes(
    title_text='Probability Density', 
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    zeroline=False,
    range=[0, max(pdf_med) * 1.5] # Clip Y-axis to keep red/yellow tails visible
)

fig.show()


# We don't know the current regime explicitly, and we don't know which regime we are going to switch to at any point in time.


# ---


# #### 3.) 📉 Why Market Timing is Difficult
# 
# We are compressing an unfathomable number of dimensions into a simple return series that doesn't tell the entire story.
# 
# Distributions and regimes are subject to change over time, and we can't visualize them.
# 
# The best we can do is develop models and forecast.


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ==========================================
# --- 1. EMPIRICAL PARAMETER FITTING ---
# ==========================================
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', periods=3000, freq='B')
vix_sim = np.random.lognormal(mean=2.8, sigma=0.4, size=3000)
spy_sim = np.cumprod(1 + np.random.normal(0.0005 - 0.0001*vix_sim, 0.001*vix_sim)) * 100
df = pd.DataFrame({'Date': dates, 'SPY': spy_sim, 'VIX': vix_sim})

df['Returns'] = df['SPY'].pct_change() * 100 
df_clean = df.dropna(subset=['Returns', 'VIX']).copy()

# Regimes based on VIX tertiles
q33 = df_clean['VIX'].quantile(0.333)
q67 = df_clean['VIX'].quantile(0.667)

returns_low = df_clean[df_clean['VIX'] <= q33]['Returns'].values
returns_med = df_clean[(df_clean['VIX'] > q33) & (df_clean['VIX'] <= q67)]['Returns'].values
returns_high = df_clean[df_clean['VIX'] > q67]['Returns'].values

mu_low, std_low = norm.fit(returns_low)
mu_med, std_med = norm.fit(returns_med)
mu_high, std_high = norm.fit(returns_high)

mus = [mu_low, mu_med, mu_high]
stds = [std_low, std_med, std_high]

# ==========================================
# --- 2. SIMULATE REGIME-SWITCHING PATH ---
# ==========================================
n_steps = 250
np.random.seed(77) # New seed for a great visual sequence

# Force state transitions to guarantee all 3 regimes appear frequently
regimes = []
current_regime = 0 # Start in Low Vol (Green)

while len(regimes) < n_steps:
    # Stay in the current regime for 15 to 35 days
    chunk_length = np.random.randint(15, 35) 
    regimes.extend([current_regime] * chunk_length)
    
    # Force a transition to one of the OTHER two regimes
    possible_next_regimes = [r for r in [0, 1, 2] if r != current_regime]
    current_regime = np.random.choice(possible_next_regimes)

regimes = np.array(regimes[:n_steps])

# Generate Observable Returns
sim_returns = np.zeros(n_steps)
for i in range(n_steps):
    sim_returns[i] = np.random.normal(mus[regimes[i]], stds[regimes[i]])

time_grid = np.arange(1, n_steps + 1)
x_grid = np.linspace(-max(abs(sim_returns))*1.2, max(abs(sim_returns))*1.2, 500)

# Pre-calculate PDFs
pdf_low = norm.pdf(x_grid, mus[0], stds[0])
pdf_med = norm.pdf(x_grid, mus[1], stds[1])
pdf_high = norm.pdf(x_grid, mus[2], stds[2])

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []

for k in range(1, n_steps + 1):
    t_x = time_grid[:k]
    t_y = sim_returns[:k]
    
    # Split for color coding
    t_y_low = np.where(regimes[:k] == 0, t_y, np.nan)
    t_y_med = np.where(regimes[:k] == 1, t_y, np.nan)
    t_y_high = np.where(regimes[:k] == 2, t_y, np.nan)
    
    # Active Regime Styling for the Bottom Chart
    curr_regime = regimes[k-1]
    
    c_low = 'rgba(57, 255, 20, 1.0)' if curr_regime == 0 else 'rgba(57, 255, 20, 0.15)'
    f_low = 'rgba(57, 255, 20, 0.4)' if curr_regime == 0 else 'rgba(57, 255, 20, 0.0)'
    w_low = 4 if curr_regime == 0 else 1
    
    c_med = 'rgba(255, 255, 0, 1.0)' if curr_regime == 1 else 'rgba(255, 255, 0, 0.15)'
    f_med = 'rgba(255, 255, 0, 0.4)' if curr_regime == 1 else 'rgba(255, 255, 0, 0.0)'
    w_med = 4 if curr_regime == 1 else 1
    
    c_high = 'rgba(255, 0, 0, 1.0)' if curr_regime == 2 else 'rgba(255, 0, 0, 0.15)'
    f_high = 'rgba(255, 0, 0, 0.4)' if curr_regime == 2 else 'rgba(255, 0, 0, 0.0)'
    w_high = 4 if curr_regime == 2 else 1

    # Traces
    tr0 = go.Scatter(x=t_x, y=t_y, mode='lines+markers', line=dict(color='rgba(255, 255, 255, 0.5)', width=1), marker=dict(color='white', size=4))
    
    tr1 = go.Scatter(x=t_x, y=t_y, mode='lines', line=dict(color='rgba(255, 255, 255, 0.2)', width=1))
    tr2 = go.Scatter(x=t_x, y=t_y_low, mode='markers', marker=dict(color='#39ff14', size=8))
    tr3 = go.Scatter(x=t_x, y=t_y_med, mode='markers', marker=dict(color='yellow', size=8))
    tr4 = go.Scatter(x=t_x, y=t_y_high, mode='markers', marker=dict(color='red', size=8))
    
    tr5 = go.Scatter(x=x_grid, y=pdf_low, mode='lines', line=dict(color=c_low, width=w_low), fill='tozeroy', fillcolor=f_low)
    tr6 = go.Scatter(x=x_grid, y=pdf_med, mode='lines', line=dict(color=c_med, width=w_med), fill='tozeroy', fillcolor=f_med)
    tr7 = go.Scatter(x=x_grid, y=pdf_high, mode='lines', line=dict(color=c_high, width=w_high), fill='tozeroy', fillcolor=f_high)

    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7], name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=3, cols=1, 
    row_heights=[0.3, 0.3, 0.4],
    subplot_titles=[
        "Raw Observations (The Illusion of a Single Market)", 
        "Color-Coded Returns (The Hidden States)",
        "Active Data-Generating Distribution (The Mechanism)"
    ],
    vertical_spacing=0.10
)

# Dummy Traces
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=1, col=1) 

fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1) 
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1) 
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1) 
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=2, col=1) 

fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=3, col=1) 
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=3, col=1) 
fig.add_trace(go.Scatter(x=[np.nan], y=[np.nan]), row=3, col=1) 

fig.frames = frames

# ==========================================
# --- 5. SLIDER & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Trade No: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.1,
    steps=[dict(
        method="animate",
        args=[[frames[k-1].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(1, n_steps + 1)]
)]

fig.update_layout(
    height=650, width=1000,
    title_text="Regime Switching: Volatility Clustering & Structural Breaks in Empirical Data",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(b=100, t=80, l=50, r=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.12, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes Styling
grid_style = dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.5)', zerolinewidth=2)
max_y = max(abs(sim_returns)) * 1.1

fig.update_xaxes(title_text='', range=[0, n_steps + 5], **grid_style, row=1, col=1)
fig.update_yaxes(title_text='Daily Return (%)', range=[-max_y, max_y], **grid_style, row=1, col=1)

fig.update_xaxes(title_text='', range=[0, n_steps + 5], **grid_style, row=2, col=1)
fig.update_yaxes(title_text='Daily Return (%)', range=[-max_y, max_y], **grid_style, row=2, col=1)

fig.update_xaxes(title_text='Return Magnitude (%)', range=[-max_y, max_y], **grid_style, row=3, col=1)
fig.update_yaxes(title_text='Probability Density', range=[0, max(pdf_low) * 1.05], **grid_style, row=3, col=1)

fig.show()


# We can't visualize the true data generating distribution.
# 
# There's no gaurentee they will exist again in the future if we do create a model to proxy for them from historic data.
# 
# There is no one man set and forget quant fund.  There is no model that isn't subject to reparameterization over time.
# 
# This is the model selection and parameterization problem I have discussed at length previously.  
# 
# If you'd like a dedicated video on the subject I'd be glad to create one.


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - We can generate wealth without knowing the outcome of a random variable thanks to probability and statistics
# - If we don't know the true underlying mass or density function we can approximate it using the Law of Large Numbers (LLN)
# - However, results from probability and statistics including the LLN and CLT require distributions to be stationary
# - Stationarity in academia is often described as distributions that are invariant across re-sampling, and non-stationarity the opposite
# - In reality, think of non-stationarity as the real world - it is subject to change at any given time from an incredible number of decisions, it doesn't actually follow a distribution 
# - We model real life uncertainty as random, this is why it is non-stationary and why *nothing* can make a non-stationary "distribution" stationary - there is no distribution!
# - This corresponds to the notion of why market timing is difficult, the underlying "distribution" producing returns is changing, we can't observe it, and we don't know what it is going to be next 
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
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Girsanov's Change of Measure
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Kalman Filter with Interactive Brokers
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

