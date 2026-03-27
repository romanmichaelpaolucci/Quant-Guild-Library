# ### Retail and Institutional Trading
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
# - [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)
# 
# - [How to Trade](https://youtu.be/NqOj__PaMec)
# 
# - [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)
# 
# - [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
# 
# - [How to Trade with the Kelly Criterion](https://youtu.be/7tvW3NvRnPk)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
#  
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# ##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)
# 
# ---


# ### 📖 Sections
# 
# #### 1.) 📈 Retail Trading 
# 
# - How to think about retail trading
# 
# - My 2025 YTD P/L
# 
# - Can retailers deploy institutional strategies?
# 
# #### 2.) 🏛️ Institutional Trading - Sell Side
# 
# - Market-Making as a business
# 
# - Risks Market-Makers face
# 
# #### 3.) 💵 Institutional Trading - Buy Side
# 
# - Speculative Trading
# 
# - Quant Hedge Funds
# 
# - Strategies, Horizons, Alpha
# 
# #### 4.) 💰 What Does a Quant do with Money?
# 
# - Understanding the landscape and available strategies
# 
# - Linear combinations and expectations
# 
# - Reality
# 
# #### 5.) 💭 Closing Thoughts and Future Topics


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


# #### 1.) 📈 Retail Trading
# 
# Things to keep in mind as a retail trader
# 
# - The distribution is **highly skewed:** a combination of informed and uninformed traders
# 
# - *The market* doesn't care if you extract hundreds of thousands or even millions of dollars
# 
# - Discretionary and algorithmic trading can be profitable
# 
# - Strategy and information play a key role, 
# 
# - Technology limitations and scalability
# 
# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 💸 My Year-To-Date (YTD) Algorithmic and Discretionary Trading P/L 2025
# 
# <u>This Excludes Crypto and Long-Term Investments</u>
# 
# - I trade in a discretionary and algorithmic capacity with Interactive Brokers
# 
# - I've rebased my portfolio to the notional value at the start of 2025 and put it in terms of $100,000


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read and process data
df = pd.read_csv('2025_YTD_Return.csv')['Stock']

# Rebase to notional 100k to mask actual portfolio size
df = (df / df.iloc[0]) * 100000
df_after_60 = df[60:]

# Calculate metrics for both periods
def calculate_metrics(data):
    returns = data.pct_change().dropna()
    risk_free_rate = 0.05
    daily_rf = (1 + risk_free_rate)**(1/252) - 1
    
    excess_returns = returns - daily_rf
    avg_excess_return = excess_returns.mean()
    std_dev = returns.std()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    sharpe = (avg_excess_return / std_dev) * np.sqrt(252)
    sortino = (avg_excess_return / downside_std) * np.sqrt(252)
    
    return sharpe, sortino

sharpe_full, sortino_full = calculate_metrics(df)
sharpe_after, sortino_after = calculate_metrics(df_after_60)

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=(f'Full Time Series<br>Sharpe: {sharpe_full:.2f} | Sortino: {sortino_full:.2f}',
                                  f'After Day 60<br>Sharpe: {sharpe_after:.2f} | Sortino: {sortino_after:.2f}'))

# Add traces
fig.add_trace(
    go.Scatter(y=df, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=df_after_60, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=2, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=800,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )

fig.show()



# I could rebase this at day 60 and lie but I'm showing you the reality of the situation
# 
# these quantities (ratios, performance metrics) are random variables, they are only *relatively* useful metrics


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 🎯 Can Retailers Trade Institutional Grade Strategies?
# 
# Secrecy is not always indicative of performance or skill...
# 
# Yes and no - do you want a seat at the table or do you want to trade in a retail capacity for your own account?
# 
# - Who is going to build your infrastructure?
# 
# - Who is going to assume the risk in any (or all) trade(s)?
# 
# - Who is going to optimize your strategies? (New strategies to production, retire or optimize strategies with performance degradation)


# <u>What to *watch out for*</u>
# 
# - Platforms that *automate* processes for you, they want subscriptions, tempting but overfitting is only one of the pitfalls you will fall into...
# 
# - News in any format, they want your attention
# 
# - Brokers and apps that make trading *too* easy, they want your commission
# 
# <u>What you *should do*</u>
# 
# - Learn math, probability statistics, finance, economics, ...
# 
# - Make quantitative trading decisions that you believe have edge in an algorithmic or discretionary capacity
# 
# 
# 
# **🚀 Master Your Quantitative Skills to <u>Make Your Own Trading Decisions</u>**


# ---


# ##### 2.) 🏛️ Institutional Trading - Sell Side
# 
# Typically, folks start on the sell-side and work to navigate to the buy-side where speculative trading can be more lucrative.
# 
# 
# Traders on these desks quote a two-way (bid/offer) and aim to collect a spread based on a mid-price.
# 
# Much of this is automated these days, especially in a high-frequency trading capacity.  
# 
# Nevertheless, this is where time series analysis and models that attempt to construct a level (expectation) can be valid.  To be profitable in the long run these models only need to be *correct* on average - similar to the edge that can be constructed playing a coin flip or dice roll game.


# Generate sample market making data with skewed trading and periods of losses
n_steps = 100
t = np.arange(n_steps)

# Generate mid price path with some drift to create challenging periods
mid_price = 100 + np.cumsum(np.random.normal(0.002, 0.1, n_steps))
spread = 0.2
bid = mid_price - spread/2
ask = mid_price + spread/2

# Simulate trades with skewed probabilities and periods of more aggressive trading
trade_probs = np.zeros((n_steps, 3))
for i in range(n_steps):
    if i < 30:  # Period of balanced trading
        trade_probs[i] = [0.3, 0.3, 0.4]
    elif i < 60:  # Period of more selling pressure
        trade_probs[i] = [0.2, 0.5, 0.3]
    else:  # Period of more buying pressure
        trade_probs[i] = [0.5, 0.2, 0.3]
        
trades = [np.random.choice(['buy', 'sell', None], p=probs) for probs in trade_probs]
pnl = np.zeros(n_steps)
cumulative_pnl = np.zeros(n_steps)

# Calculate P&L with occasional losses due to price movement
position = 0
for i in range(n_steps):
    if trades[i] == 'buy':
        pnl[i] = spread/2  # Basic spread capture
        position += 1
    elif trades[i] == 'sell':
        pnl[i] = spread/2
        position -= 1
    
    # Add impact of position and price movement
    if i > 0:
        price_change = mid_price[i] - mid_price[i-1]
        position_impact = -position * price_change  # Negative P&L if price moves against position
        pnl[i] += position_impact
        cumulative_pnl[i] = cumulative_pnl[i-1] + pnl[i]

# Create animation frames
frames = []
for step in range(n_steps):
    # Create figure with two subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Market Prices', 'Market Maker P&L'),
                        horizontal_spacing=0.1)
    
    # Price subplot
    fig.add_trace(
        go.Scatter(x=t[:step+1], y=mid_price[:step+1], 
                  line=dict(color='rgba(60, 179, 113, 0.9)', width=2, dash='dot'),
                  name='Mid Price (Estimated)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[:step+1], y=bid[:step+1],
                  line=dict(color='rgba(178, 34, 34, 0.9)', width=2),
                  name='Bid'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[:step+1], y=ask[:step+1],
                  line=dict(color='rgba(70, 130, 180, 0.9)', width=2),
                  name='Ask'),
        row=1, col=1
    )
    
    # P&L subplot
    fig.add_trace(
        go.Bar(x=t[:step+1], y=pnl[:step+1],
               marker_color=['rgba(178, 34, 34, 0.8)' if x < 0 else 'rgba(31, 119, 180, 0.8)' for x in pnl[:step+1]],
               name='Trade P&L'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=t[:step+1], y=cumulative_pnl[:step+1],
                  line=dict(color='rgba(60, 179, 113, 0.9)', width=2),
                  name='Cumulative P&L'),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)'
        )
    )

    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=1, col=i
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=1, col=i
        )

    frames.append(go.Frame(data=fig.data))

# Create and show animation
fig.frames = frames
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 


# <u> Market-Makers are subject to different types of risk: </u>
# 
# - Inventory Risk (Imbalance in buy/sell pressure, *hit* on bids without *lifting* their ask vice-versa)
# 
# - Adverse Selection (What looks like easy P/L is a trader with *better* information)
# 
# - Technical Errors (Knight Capital Group, -$400m within an hour)
# 
# - High Volatility (Not enough time to hedge)
# 
# - Counterparty Risk (Less algorithmic, think OTC/Exotic products)


# ---


# ##### 3.) 💵 Institutional Trading - Buy Side
# 
# Speculative trading profits can be extremely lucrative, working at some sort of quantitative hedge fund is typically the goal.
# 
# Volatility and mispricings drive profits, bull-markets are boring - *hedge* is literally in the name, out performing the S&P 500 is a useless metric.
# 
# 


# Generate sample data with noisier signal
np.random.seed(42)
n_days = 252
t = np.linspace(0, 1, n_days)
signal = 0.6 * t + 0.5 * np.random.randn(n_days)  # Noisier monotonic signal

# Calculate average returns by signal bucket (5 buckets)
n_buckets = 5
signal_buckets = pd.qcut(signal, n_buckets, labels=False)
avg_returns = np.array([np.mean(signal[signal_buckets == i]) for i in range(n_buckets)])

# Generate different correlated price paths using ABM
dt = 1/252
mu = 0.1  # drift
sigma = 0.2  # volatility

# Generate different random walks for long and short legs
dW_long = np.random.normal(0, np.sqrt(dt), n_days)
dW_short = np.random.normal(0, np.sqrt(dt), n_days)

long_leg = 100000 * np.exp(np.cumsum((mu + 0.002 * signal) * dt + sigma * dW_long))
short_leg = 100000 * np.exp(np.cumsum((mu - 0.002 * signal) * dt + sigma * dW_short))
ls_portfolio = (long_leg + (200000 - short_leg)) / 2

# Create figure with two subplots side by side
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Average Returns by Signal Bucket', 'Portfolio Performance'),
                    horizontal_spacing=0.1)

# Add bar chart of signal buckets
fig.add_trace(
    go.Bar(x=np.arange(n_buckets), y=avg_returns, 
           marker_color='rgba(31, 119, 180, 0.8)',
           name='Avg Return'),
    row=1, col=1
)

# Add portfolio traces
fig.add_trace(
    go.Scatter(y=long_leg, line=dict(color='rgba(70, 130, 180, 0.9)', width=2), 
               name='Long Leg'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=short_leg, line=dict(color='rgba(178, 34, 34, 0.9)', width=2),
               name='Short Leg'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=ls_portfolio, line=dict(color='rgba(60, 179, 113, 0.9)', width=2),
               name='L/S Portfolio'),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(255,255,255,0.2)'
    )
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()


# Strategies in a *signal* capacity are about asset selection, holding periods (daily, weekly, monthly) and turnover are a concern depending on alpha decay of a signal.
# 
# Other strategies (intraday, high-frequency) tend to consider more *statistical arbitrage* approaches (level reversion, implied volatility surface arbitrage)


# ---


# ##### 4.) 💰 What Does a Quant do with their Money?
# 
# <u> Not all quants have the same skills </u>
# 
# - Quant research, development, trading
# 
# - Risk appetite institutionally v.s. personally
# 
# - Breadth of knowledge in finance, economics, technology, etc...
# 
# - Qualitative skills, reading in between the lines, etc.


# ###### ______________________________________________________________________________________________________________________________________
# 


# Returns from some sort of cash or portfolio allocation is a *linear combination*.
# 
# **Remember** we are dealing with randomness so the best we can do is optimize some expected value, I discuss this at length in many videos.
# 
# Diversification can help mitigate risk but also reduces upside, knowing this is why having a quantitative background is quite useful.
# 
# Let's say I have 3 trading strategies with the following annual return profiles...
# 
# - **Discretionary Trading:** 
# 
# $$\mathbb{E}[D] = 25\% \quad \quad \sigma_D = 12\%$$
# 
# - **Algorithmic Trading:** 
# 
# $$\mathbb{E}[A] = 10\% \quad \quad \sigma_A = 5\%$$
# 
# - **Market Trading:** 
# 
# $$\mathbb{E}[M] = 8\% \quad \quad \sigma_M = 15\%$$
# 
# 
# These numbers are *moving targets* estimated from data - they do not converge and change over time.  They do **NOT** function like traditional statistics.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# 
# **My Portfolio**
# 
# Given these options, I can only trade one of the following portfolios based on my total capital $C$
# 
# $$\mathbb{E}[\Pi] = W_D \mathbb{E}[D] + W_A \mathbb{E}[A] + W_M \mathbb{E}[M] \quad \quad \text{ given capital constraint } C$$
# 
# Where I can choose $W_D, W_A, W_M$ freely based on my capital available, $C$


# Generate sample paths using ABM
n_steps = 1000
dt = 1/252  # Daily timesteps

# Convert annual parameters to daily
mu_d, sigma_d = 0.25, 0.12  # Discretionary
mu_a, sigma_a = 0.10, 0.05  # Algorithmic
mu_m, sigma_m = 0.08, 0.15  # Market

# Daily parameters
daily_mu_d = mu_d * dt
daily_mu_a = mu_a * dt
daily_mu_m = mu_m * dt
daily_sigma_d = sigma_d * np.sqrt(dt)
daily_sigma_a = sigma_a * np.sqrt(dt)
daily_sigma_m = sigma_m * np.sqrt(dt)

# Generate paths
np.random.seed(42)
paths = {}
initial_value = 100000

for name, mu, sigma in [('Discretionary', daily_mu_d, daily_sigma_d),
                       ('Algorithmic', daily_mu_a, daily_sigma_a),
                       ('Market', daily_mu_m, daily_sigma_m)]:
    
    dW = np.random.normal(0, 1, n_steps)
    path = np.zeros(n_steps)
    path[0] = initial_value
    
    for t in range(1, n_steps):
        path[t] = path[t-1] * (1 + mu + sigma * dW[t])
    
    paths[name] = path

# Calculate minimum variance portfolio
returns = pd.DataFrame({k: np.diff(v)/v[:-1] for k,v in paths.items()})
cov_matrix = returns.cov()
inv_cov = np.linalg.inv(cov_matrix)
ones = np.ones(len(cov_matrix))
w = inv_cov.dot(ones) / (ones.dot(inv_cov).dot(ones))
min_var_weights = pd.Series(w, index=cov_matrix.index)

# Calculate min var portfolio path
min_var_path = sum(paths[k] * w[i] for i, k in enumerate(paths.keys()))

# Calculate metrics for each path
def calc_metrics(returns):
    rf = 0.05
    daily_rf = (1 + rf)**(1/252) - 1
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    sortino = (excess_returns.mean() / returns[returns < 0].std()) * np.sqrt(252)
    return sharpe, sortino

metrics = {}
for name in paths.keys():
    rets = np.diff(paths[name])/paths[name][:-1]
    metrics[name] = calc_metrics(rets)

min_var_rets = np.diff(min_var_path)/min_var_path[:-1]
metrics['Min Variance'] = calc_metrics(min_var_rets)

# Create figure
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Individual Strategies', 'Minimum Variance Portfolio'))

colors = {'Discretionary': 'rgba(255,65,54,0.6)',
          'Algorithmic': 'rgba(44,160,44,0.6)',
          'Market': 'rgba(31,119,180,0.6)',
          'Min Variance': 'rgba(255,127,14,0.6)'}

# Plot individual paths
for name, path in paths.items():
    sharpe, sortino = metrics[name]
    fig.add_trace(
        go.Scatter(
            y=path,
            name=f'{name}<br>Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}',
            line=dict(color=colors[name])
        ),
        row=1, col=1
    )

# Plot min variance portfolio
sharpe, sortino = metrics['Min Variance']
weights_text = "Min-Var Portfolio <br>" + "<br>".join([f"{k}: {v:.2%}" for k,v in min_var_weights.items()])
fig.add_trace(
    go.Scatter(
        y=min_var_path,
        name=f'Min Variance<br>Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}<br>{weights_text}',
        line=dict(color=colors['Min Variance'])
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=800,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )

fig.show()



# <u> **The problem** </u>
# 
# I can optimize for this and get the *perfect* weights that will maximize my EV
# 
# **but** I don't get to dictate the current market conditions if it better suits me to any particular strategy.
# 
# In other words, the EV of each strategy will vary at different points in time - its not a global optimization problem...
# 
# I've discussed this at length in the past: statistics don't converge, portfolio optimization is overfitting, etc...


# ###### ______________________________________________________________________________________________________________________________________
# 


# <u>What I do with my money</u>
# 
# - Trade in a discretionary capacity when it suits me
# 
# - Turn on my trading systems when it suits me
# 
# - Leave my cash in S&P 500 when it suits me
# 
# Edge changes based on regimes, timing, administration, etc. and will dictate my decisions


# ---


# #### 5.) 💭 Closing Thoughts and Future Topics


# Retail and Institutional Trading are fundamentally different - you can make and lose money in either environment.
# 
# Institutions and hedge blowout *all the time* just like how start-ups go bankrupt *all the time*
# 
# - Know the difference between Citadel and a new Quant Hedge Fund like Microsoft and a new Tech Startup
# 
# **Future Topics**
# 
# - Financial Mathematics, Financial Engineering, Quantitative Finance Content (This will Never Change)
# 
# - How To's: Volatility Trading Dashboard, Algo Systems, etc...
# 
# - Personal Experiences in Quantitative Finance (Gauging Interest)
# 
# - Weekly Market Review (Gauging Interest)
# 
# - Quant Podcast (Interviews with traders and other financial professionals)


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

