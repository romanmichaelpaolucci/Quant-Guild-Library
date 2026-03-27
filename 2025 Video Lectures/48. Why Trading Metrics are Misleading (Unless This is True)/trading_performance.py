# ### 💵 Why Trading Metrics are Misleading (Unless This is True)
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
# - [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)
# 
# - [Why Portfolio Optimization Doesn't Work](https://youtu.be/eZIITtd3UfY)
# 
# - [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)
# 
# - [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
# 
# - [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)
# 
# - [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
#  
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# ##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# ##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)
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
# 
# #### 1.) 📉 *Useless* Investing & Trading Performance Measures
# 
# - Evaluating Investing and Trading Strategies
# 
# - Expected Returns, Win Rate, Cash
# 
# - Example: Investing in **NVDA**
# 
# - Example: Risk-Free Investments (U.S. Treasuries)
# 
# #### 2.) 📊 *Less Useless* Investing & Trading Performance Measures
# 
# - Volatility and Risk-Adjusted Returns
# 
# - Sharpe Ratio, Sortino Ratio, Max Drawdown
# 
# #### 3.) 💥 Why Performance Metrics are Overrated
# 
# - What Really Matters in Your Trading Strategy
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📉 *Useless* Investing & Trading Performance Measures
# 
# #### 📏 Evaluating Investing and Trading Strategies
# 
# 
# When evaluating investing or trading strategies we often proxy performance based on historic data, ratios, and other performance metrics
# 
# There are what I call *useless* performance measures and *less useless* performance measures, even still all are proxied using historic data.
# 
# But I thought past data wasn't indicative of future performance?  What is the missing piece of this puzzle?  


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate sample data for backtest and live trading comparison
np.random.seed(42)
n_backtest = 400
n_live = 400

# Generate dates for both periods
backtest_dates = [datetime(2023,1,1) + timedelta(days=int(x)) for x in range(n_backtest)]
live_dates = [backtest_dates[-1] + timedelta(days=int(x+1)) for x in range(n_live)]
all_dates = backtest_dates + live_dates

# Generate alpha-like strategy returns with steady growth and low volatility
def generate_strategy_returns(n_points, mean=0.0008, std=0.005):
    # Add slight positive drift to create steady upward trend
    drift = np.linspace(0, 0.0002, n_points)
    returns = np.random.normal(mean, std, n_points) + drift
    equity = 100000 * np.exp(np.cumsum(returns))
    return equity, returns

# Generate backtest and live data with very similar characteristics
backtest_equity, backtest_returns = generate_strategy_returns(n_backtest)
live_equity, live_returns = generate_strategy_returns(n_live)

# Ensure smooth transition between backtest and live
live_equity = live_equity * (backtest_equity[-1] / live_equity[0])

# Combine equity curves
full_equity = np.concatenate([backtest_equity, live_equity])

# Calculate performance metrics for both periods
def calculate_metrics(returns):
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    sortino = np.sqrt(252) * np.mean(returns) / np.std(returns[returns < 0])
    cummax = np.maximum.accumulate(returns.cumsum())
    drawdown = cummax - returns.cumsum()
    mdd = np.max(drawdown)
    return sharpe, sortino, mdd

backtest_metrics = calculate_metrics(backtest_returns)
live_metrics = calculate_metrics(live_returns)

# Create the plot
fig = go.Figure()

# Add equity curve
fig.add_trace(
    go.Scatter(
        x=all_dates,
        y=full_equity,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)', width=2),
        name='Strategy Performance'
    )
)

# Add vertical line separating backtest and live
fig.add_shape(
    type="line",
    x0=backtest_dates[-1],
    x1=backtest_dates[-1], 
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash')
)

fig.add_annotation(
    x=backtest_dates[-1],
    y=1,
    yref="paper",
    text="Live Trading Begins",
    showarrow=False,
    yshift=10,
    font=dict(color='white')
)

# Add annotations for metrics
fig.add_annotation(
    x=backtest_dates[len(backtest_dates)//2],
    y=max(full_equity),
    text=f"Backtest Metrics:<br>Sharpe: {backtest_metrics[0]:.2f}<br>Sortino: {backtest_metrics[1]:.2f}<br>MDD: {backtest_metrics[2]:.2%}",
    showarrow=False,
    yshift=10,
    font=dict(color='white')
)

fig.add_annotation(
    x=live_dates[len(live_dates)//2],
    y=max(full_equity),
    text=f"Live Metrics:<br>Sharpe: {live_metrics[0]:.2f}<br>Sortino: {live_metrics[1]:.2f}<br>MDD: {live_metrics[2]:.2%}",
    showarrow=False,
    yshift=10,
    font=dict(color='white')
)

# Update layout
fig.update_layout(
    title="Strategy Performance: Backtest vs Live Trading",
    height=600,
    width=1000,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)"
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### 🎯 Expected Returns, Win Rate, Cash
# 
# **Expected returns** are easy to measure in a backward-looking sense but are, on their own, a useless measure.
# 
# Just take the trading strategy, investment - whatever, and take returns dails/weekly/monthly and compute the average (expectation)
# 
# *Problem:*  This says something about what the performance **was** not what it **will be** or the **amount of risk** we had to take to receive the returns


# ##### 🔎 Example: Investing in *NVDA*
# 
# This example is to illustrate that backward-looking measures are not an indication of forward-looking performance even with "good" performance metrics


# Read NVDA data
import pandas as pd
df = pd.read_csv('nvda_returns_2025_ytd.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Find the major drop point (when price went to 90s)
drop_date = df[df['Close_Price'] < 100].index[0]
drop_date = df.loc[drop_date, 'Date']

# Calculate expected returns for both periods
pre_drop = df[df['Date'] < drop_date]
post_drop = df[df['Date'] >= drop_date]

pre_drop_er = pre_drop['Daily_Return'].mean() * 100
post_drop_er = post_drop['Daily_Return'].mean() * 100
overall_er = df['Daily_Return'].mean() * 100

# Calculate YTD return
ytd_return = ((df['Close_Price'].iloc[-1] / df['Close_Price'].iloc[0]) - 1) * 100

# Create figure
fig = go.Figure()

# Add NVDA price path
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Close_Price'],
        mode='lines',
        line=dict(color='rgba(255, 20, 147, 1)', width=2),
        name='NVDA Price'
    )
)

# Add vertical line at drop date
fig.add_shape(
    type="line",
    x0=drop_date,
    x1=drop_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash')
)

# Add annotations for metrics
fig.add_annotation(
    x=pre_drop['Date'].mean(),
    y=df['Close_Price'].max(),
    text=f"Pre-Drop Expected Return:<br>{pre_drop_er:.2f}% daily",
    showarrow=False,
    font=dict(color='white')
)

fig.add_annotation(
    x=post_drop['Date'].mean(),
    y=df['Close_Price'].max(),
    text=f"Post-Drop Expected Return:<br>{post_drop_er:.2f}% daily",
    showarrow=False,
    font=dict(color='white')
)

fig.add_annotation(
    x=df['Date'].mean(),
    y=df['Close_Price'].min(),
    text=f"Overall Expected Return: {overall_er:.2f}% daily<br>YTD Return: {ytd_return:.2f}%",
    showarrow=False,
    font=dict(color='white')
)

# Update layout
fig.update_layout(
    title="NVDA Price Movement and Expected Returns (2025 YTD)",
    height=600,
    width=1200,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    xaxis_title="Date",
    yaxis_title="NVDA Price ($)"
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# So the YTD return is $22.32%$, it roughly doubled stock price intra-year, why doesn't every portfolio only consist of NVDA?
# 
# It is **NOT** a forward-looking (in the stability sense, we will discuss below) measure of performance!
# 
# **Remark:** We can't just outright consider expected returns, some will say the mean is roughly zero but this absolutely depends on the time period you 
# 
# query returns for as we see above - we might be able to model the price path as a stochastic process but the question still remains what is a reasonable 
# 
# measure of expected return in a forward looking sense?  
# 
# **Is this even a useful measure?**
# 
# - Should we use 1 week of returns?  
# 
# - Should we use 1 year?  
#     - The entire company can change a lot in 1 year, it might not make sense to use that much historic data!
# 
# Clearly we need more pieces of the puzzle to make sense of this. . .


# ###### ______________________________________________________________________________________________________________________________________
# 
# **Win rate** is also, on its own a useless measure, I can give you a strategy right now that has a 100% win rate - it tells us nothing of the performance of
# 
# our strategy in terms of wealth accumulated *or* amount of capital risked
# 
# 
# Effectively, we can trade 1 time and have a $100%$ win rate or trade something so absurd like selling outrageous insurance 
# 
# (earthquake insurance, for example) but who would buy it?  Sure it may payout, then you'll take a massive loss - even with that you'll still have a 99%+ "win rate" 
# 
# Usually, if a strategy has close to a 100% "win rate" we are accumulating barely any wealth and one loss wipes out the entire account of gains


# ##### 🔎 Example: Trading Strategies with Different Win Rates


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate sample data for win rate examples
np.random.seed(42)
n_points = 200
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]

# Strategy A - 100% win rate but tiny gains (flat)
tiny_gains = np.random.uniform(0.0001, 0.0002, n_points)
strategy_a = 100000 * (1 + np.cumsum(tiny_gains))
final_return_a = ((strategy_a[-1] - 100000) / 100000) * 100

# Strategy B - 99% win rate with one catastrophic loss
small_gains = np.random.uniform(0.0005, 0.001, n_points)
strategy_b = 100000 * (1 + np.cumsum(small_gains))
# Add catastrophic loss around day 150
strategy_b[150:] *= 0.3  # 70% loss
final_return_b = ((strategy_b[-1] - 100000) / 100000) * 100

# Strategy C - 40% win rate but overall profitable
wins = np.random.uniform(0.01, 0.03, n_points) * (np.random.random(n_points) < 0.4)  # 40% win rate
losses = np.random.uniform(-0.005, -0.001, n_points) * (np.random.random(n_points) >= 0.4)  # 60% loss rate
combined = wins + losses
strategy_c = 100000 * (1 + np.cumsum(combined))
final_return_c = ((strategy_c[-1] - 100000) / 100000) * 100

# Create figure with subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=(
                        f"100% Win Rate Strategy (Return: {final_return_a:.2f}%)",
                        f"99% Win Rate Strategy (Return: {final_return_b:.2f}%)",
                        f"40% Win Rate Strategy (Return: {final_return_c:.2f}%)"
                    ))

# Add traces for Strategy A
fig.add_trace(
    go.Scatter(x=dates, y=strategy_a, mode='lines',
               line=dict(color='rgba(0, 255, 0, 1)'),
               showlegend=False),
    row=1, col=1
)

# Add traces for Strategy B
fig.add_trace(
    go.Scatter(x=dates, y=strategy_b, mode='lines',
               line=dict(color='rgba(255, 0, 255, 1)'),
               showlegend=False),
    row=2, col=1
)

# Add traces for Strategy C
fig.add_trace(
    go.Scatter(x=dates, y=strategy_c, mode='lines',
               line=dict(color='rgba(0, 255, 255, 1)'),
               showlegend=False),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

# Update y-axes with appropriate titles
for i in range(1, 4):
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title="Portfolio Value ($)",
        row=i, col=1
    )

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# **Cash** is, just like the others above, on its own, a useless measure, it gives us no perception of the capital being allocated or the performance
# 
# relative to portfolio size (for example, \$3,000/mo on a \$1,000,000 portfolio is **NOT** "good" - we'll see why below)
# 
# #### 🔍 Example: Risk-Free Investments (U.S. Treasuries)
# 
# U.S. Treasuries are the closest proxy for a risk-free investment we have in real life - as my old finance professor once said:
# 
# *"If the U.S. is defaulting on your interest payments, you have far more concerns than if it is a valid risk-free rate proxy"*
# 
# Effectively, this shows why **cash** is **not** a useful measure of performance on its own. . .


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate daily dates for 1 year
n_points = 365
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]

# Account A: $1,000,000 at 3.6% annual return (deterministic)
initial_a = 1_000_000
daily_rate_a = (1 + 0.036)**(1/365) - 1
strategy_a = [initial_a * (1 + daily_rate_a)**x for x in range(n_points)]

# Account B: $100,000 with stochastic 36% annual return but more consistent
initial_b = 100_000
target_annual_return = 0.36
daily_vol = 0.005  # Reduced volatility for more consistency
daily_drift = (np.log(1 + target_annual_return) - (daily_vol**2)/2)/365

# Generate stochastic path using GBM
np.random.seed(42)  # For reproducibility
daily_returns = np.random.normal(daily_drift, daily_vol, n_points)
strategy_b = [initial_b]
for ret in daily_returns:
    strategy_b.append(strategy_b[-1] * (1 + ret))
strategy_b = strategy_b[:-1]  # Remove extra point

# Calculate monthly cash generated
months = np.array([date.month for date in dates])
month_ends = np.where(np.diff(months))[0] + 1
month_ends = np.append(month_ends, -1)  # Add final point

# Fixed monthly cash for Account A (3.6% annual return)
monthly_return_a = (1 + 0.036)**(1/12) - 1
fixed_monthly_cash_a = initial_a * monthly_return_a

cash_generated_a = [fixed_monthly_cash_a] * len(month_ends)
cash_generated_b = []
prev_value_b = initial_b

for month_end in month_ends:
    cash_b = strategy_b[month_end] - prev_value_b
    cash_generated_b.append(cash_b)
    prev_value_b = strategy_b[month_end]

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=(f"$1M Account (3.6% Return)", 
                                  f"$100K Account (Stochastic ~36% Return)",
                                  "Monthly Cash Generated - $1M Account",
                                  "Monthly Cash Generated - $100K Account"))

# Add strategy A to first subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_a,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)'),  # Neon green
        name='$1M Account'
    ),
    row=1, col=1
)

# Add strategy B to second subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_b,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)'),  # Neon purple
        name='$100K Account'
    ),
    row=1, col=2
)

# Add monthly cash generated for strategy A
fig.add_trace(
    go.Bar(
        x=[dates[i] for i in month_ends],
        y=cash_generated_a,
        name='Monthly Cash A',
        marker_color='rgba(0, 255, 0, 0.5)'
    ),
    row=2, col=1
)

# Add monthly cash generated for strategy B
fig.add_trace(
    go.Bar(
        x=[dates[i] for i in month_ends],
        y=cash_generated_b,
        name='Monthly Cash B',
        marker_color='rgba(255, 0, 255, 0.5)'
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=1000,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    title=dict(
        text=f'Portfolio Value and Monthly Cash Generation Comparison',
        y=0.95
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title="Value ($)"
)

fig.show()



# **Remark:** Even if we have all of these performance metrics and other better measures that we will talk about in a moment for an investing or trading 
# 
# strategy - it doesn't actually matter what these values are in a backward-looking sense (think backtesting) in the sense that it is a **necessary** but 
# 
# **insufficient** step in the process of evaluating a strategy
# 
# 
# **⚠️ In Other Words**
# 
# $\implies$  "Good" performance metrics for an investing or trading strategy on their own **DO NOT MEAN IT IS A VIABLE STRATEGY TO TRADE!**


# ---


# #### 2.) 📊 *Less Useless* Investing & Trading Performance Measures
# 
# #### 🌊 Volatility and Risk
# 
# So if expected returns, win rate, and cash aren't good metrics what are?
# 
# Well, first we need to understand this idea of **risk** and **volatility**
# 
# If the average of returns represents what is *expected* then we can also measure *expected deviations* from that expected value, this is variance
# 
#  $$\sigma^2 = \mathbb{E}[(X - \mathbb{E}[X])^2]$$
#   
# The square root of variance is *standard deviation* and is typically used to measure volatility or *expected deviations from expected returns*
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate sample time series for two trading strategies
np.random.seed(42)
n_points = 200
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]

# Strategy A - Steady growth with small fluctuations
trend_a = np.linspace(0, 20, n_points) 
noise_a = np.random.normal(0, 1, n_points)
strategy_a = 100000 * (1 + (trend_a + noise_a)/100)

# Strategy B - Same end point but with major drawdown
trend_b = np.linspace(0, 20, n_points)
drawdown = -15 * np.exp(-0.5 * ((np.linspace(0, n_points, n_points) - 100) / 20)**2)  # Major drawdown around day 100
noise_b = np.random.normal(0, 1, n_points)
strategy_b = 100000 * (1 + (trend_b + drawdown + noise_b)/100)

# Calculate metrics
def calculate_metrics(returns):
    total_return = (returns[-1] - returns[0])/returns[0] * 100
    avg_return = np.mean(np.diff(returns)/returns[:-1]) * 100
    rolling_max = np.maximum.accumulate(returns)
    drawdowns = (returns - rolling_max)/rolling_max * 100
    max_drawdown = np.min(drawdowns)
    return total_return, avg_return, max_drawdown

metrics_a = calculate_metrics(strategy_a)
metrics_b = calculate_metrics(strategy_b)

# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Strategy A", "Strategy B"))

# Add strategy A to first subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_a,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)'),  # Neon green
        name='Strategy A'
    ),
    row=1, col=1
)

# Add strategy B to second subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_b,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)'),  # Neon purple
        name='Strategy B'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    title=dict(
        text=f'Strategy Comparison<br>A: Return: {metrics_a[0]:.1f}%, Avg: {metrics_a[1]:.2f}%<br>B: Return: {metrics_b[0]:.1f}%, Avg: {metrics_b[1]:.2f}%',
        y=0.95  # Add some padding below title by moving it up slightly
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title="Portfolio Value ($)"
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 
# #### ⚖️ Sharpe Ratio, Sortino Ratio, Max Drawdown
# 
# First, we adjust for the risk-free rate of return.  If our strategy earns less than the risk-free rate in a frictionless evnironment why would we ever trade it?
# 
#   **Sharpe Ratio**: Risk-adjusted return using total volatility
#   $$ \text{Sharpe Ratio} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p} $$
#   
#   **Sortino Ratio**: Risk-adjusted return using downside volatility
#   $$ \text{Sortino Ratio} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_d} $$
#  
#  **Maximum Drawdown**: Largest peak-to-trough decline
#  $$ \text{Max Drawdown} = \min_t \left(\frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s}\right) $$
# 
#  **Remark:** We may choose to subtract out another benchmark besides the risk-free rate like the S&P500 return over the same period, this can give an
# 
#  idea of relative performance to market exposure without asset pricing theory (Time Series or Fama-MacBeth Regressions) but is largely regime dependent - 
# 
#  everyone wants to cite these metrics or "beat the market" but that isn't at all the point of this analysis. . .again necessary **NOT** sufficient 
# 
#  Before we continue pulling on this thread, let's see some examples of these performance metrics in action. . .


# Calculate Sharpe, Sortino and Max Drawdown
def calculate_performance_metrics(returns):
    # Calculate daily returns
    daily_returns = np.diff(returns)/returns[:-1]
    
    # Assume risk-free rate of 0% for simplicity
    rf = 0
    
    # Sharpe Ratio (annualized)
    excess_returns = daily_returns - rf
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    # Sortino Ratio (annualized)
    downside_returns = excess_returns[excess_returns < 0]
    sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    
    # Max Drawdown
    rolling_max = np.maximum.accumulate(returns)
    drawdowns = (returns - rolling_max)/rolling_max * 100
    max_drawdown = np.min(drawdowns)
    
    return sharpe, sortino, max_drawdown

metrics_a = calculate_performance_metrics(strategy_a)
metrics_b = calculate_performance_metrics(strategy_b)

# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Strategy A", "Strategy B"))

# Add strategy A to first subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_a,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)'),  # Neon green
        name='Strategy A'
    ),
    row=1, col=1
)

# Add strategy B to second subplot
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_b,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)'),  # Neon purple
        name='Strategy B'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    title=dict(
        text=f'Strategy Comparison<br>A: Sharpe: {metrics_a[0]:.2f}, Sortino: {metrics_a[1]:.2f}, MaxDD: {metrics_a[2]:.1f}%<br>B: Sharpe: {metrics_b[0]:.2f}, Sortino: {metrics_b[1]:.2f}, MaxDD: {metrics_b[2]:.1f}%',
        y=0.95
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title="Portfolio Value ($)"
)

fig.show()



# ---


# #### 💥 3.) Why Performance Measures are Overrated


# 
# **Remark:** If expected return isn't viable in a forward-looking sense, doesn't that mean *variance* and *standard deviation* aren't either?  Largely, 
# 
# yes this is the case for your investing or trading strategy which is why performance metrics are necessary but not sufficient for strategy execution
# 
# 
# <br>
# 
# 
# **But I have a 3+ Sharpe Ratio!**  Anyone who cites performance metrics outright is missing the entire point of the space, I can give you a strategy with 
# 
# a Sharpe Ratio of 10 right now, 1000% Expected Returns, 100% Win Rate - these will not be viable to trade anymore!
# 
# 
# <br>
# 
# 
# **⚠️ In Other Words:** 
# 
# These metrics are also *necessary but not sufficient* for trading. 
# 
# We need to consider something else far more important beyond the values of our performance metrics


# Generate multiple random walks
n_paths = 1000
n_steps = len(dates)
paths = np.zeros((n_paths, n_steps))
paths[:,0] = 100  # Starting value

# Generate random walks with 0 expected value
for i in range(n_paths):
    returns = np.random.normal(0, 0.02, n_steps-1)  # Daily volatility of 2%
    for t in range(1, n_steps):
        paths[i,t] = paths[i,t-1] * (1 + returns[t-1])

# Find path with highest final return
final_returns = (paths[:,-1] - paths[:,0]) / paths[:,0]
best_path_idx = np.argmax(final_returns)
best_path = paths[best_path_idx]

# Calculate metrics for best path
metrics = calculate_performance_metrics(best_path)

# Create figure
fig = go.Figure()

# Add all paths with low opacity
for i in range(n_paths):
    if i != best_path_idx:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=paths[i],
                mode='lines',
                line=dict(color='rgba(0, 255, 255, 0.01)'),  # Neon cyan with 0.2 opacity
                showlegend=False
            )
        )

# Add best path with full opacity
fig.add_trace(
    go.Scatter(
        x=dates,
        y=best_path,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),  # Neon cyan with full opacity
        name='Best Performing Path'
    )
)

# Update layout
fig.update_layout(
    height=600,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title=dict(
        text=f'Random Walk Paths (Zero Expected Value)<br>Best Path Metrics - Return: {final_returns[best_path_idx]*100:.1f}%, Sharpe: {metrics[0]:.2f}, Sortino: {metrics[1]:.2f}, MaxDD: {metrics[2]:.1f}%',
        y=0.95
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title="Portfolio Value ($)"
)

fig.show()


# - Maybe you are swinging at pitches, have reasonable economic interpretation of a strategy and believe you have "edge"
# 
# - Maybe you are trading an exposure that is performing particularly well for this time period
# 
# - Maybe you overfit your backtest to noise in historical data
# 
# In any case what **actually matters here** is not these performance metrics but rather if they are *stable* in a forward looking sense. . .


import pandas as pd

# Continue the best path into the future
future_dates = pd.date_range(dates[-1], '2027-12-31', freq='B')
n_future_steps = len(future_dates)

# Generate future returns with same distribution
future_returns = np.random.normal(0, 0.02, n_future_steps)
future_path = np.zeros(n_future_steps)
future_path[0] = best_path[-1]

# Calculate future path values
for t in range(1, n_future_steps):
    future_path[t] = future_path[t-1] * (1 + future_returns[t-1])

# Calculate metrics for original period
original_final_return = (best_path[-1] - best_path[0]) / best_path[0]
original_metrics = metrics

# Calculate metrics for full period including future
full_path = np.concatenate([best_path, future_path[1:]])
full_metrics = calculate_performance_metrics(full_path)
full_final_return = (full_path[-1] - full_path[0]) / full_path[0]

# Create figure
fig = go.Figure()

# Add original best path
fig.add_trace(
    go.Scatter(
        x=dates,
        y=best_path,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Original Best Path'
    )
)

# Add future path
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_path,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 1)'),
        name='Future Path'
    )
)

# Add vertical line at transition point
fig.add_vline(
    x=dates[-1],
    line_width=2,
    line_dash="dash",
    line_color="white",
    opacity=0.5
)
# Update layout
fig.update_layout(
    height=600,  # Increased height
    width=1200,  # Increased width
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    title=dict(
        text=f'Best Path Extended into Future<br>' + 
             f'Original Metrics - Return: {original_final_return*100:.1f}%, Sharpe: {original_metrics[0]:.2f}, Sortino: {original_metrics[1]:.2f}, MaxDD: {original_metrics[2]:.1f}%<br>' +
             f'Full Period Metrics - Return: {full_final_return*100:.1f}%, Sharpe: {full_metrics[0]:.2f}, Sortino: {full_metrics[1]:.2f}, MaxDD: {full_metrics[2]:.1f}%',
        y=0.95
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title="Portfolio Value ($)"
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### 🧮 Backward Looking vs. Forward Looking Performance
# 
# Performance metrics are overrated - what we really care about is **stability**
# 
# Performance metrics are a **necessary** component of a strategy but don't tell us alone if it is viable
# 
# We should see similar performance metrics in a live setting assuming we are actually trading a viable strategy or *statistical mispricing*
# 
# There is a big difference between *bearing exposure to risk* and trading a *statistical mispricing*


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate backtest data (2020-2023)
backtest_dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
n_backtest = len(backtest_dates)

# Generate two strategies with good backtest performance
np.random.seed(42)
backtest_returns1 = np.random.normal(0.001, 0.01, n_backtest)  # Strategy 1 - Stable
backtest_returns2 = np.random.normal(0.001, 0.01, n_backtest)  # Strategy 2 - Will degrade

backtest_path1 = np.zeros(n_backtest)
backtest_path2 = np.zeros(n_backtest)
backtest_path1[0] = backtest_path2[0] = 100

for t in range(1, n_backtest):
    backtest_path1[t] = backtest_path1[t-1] * (1 + backtest_returns1[t-1])
    backtest_path2[t] = backtest_path2[t-1] * (1 + backtest_returns2[t-1])

# Generate live trading data (2024 onwards)
live_dates = pd.date_range('2024-01-01', '2025-12-31', freq='B')
n_live = len(live_dates)

# Strategy 1 maintains performance, Strategy 2 degrades
live_returns1 = np.random.normal(0.001, 0.01, n_live)  # Stable strategy
live_returns2 = np.random.normal(-0.0005, 0.015, n_live)  # Degraded strategy

live_path1 = np.zeros(n_live)
live_path2 = np.zeros(n_live)
live_path1[0] = backtest_path1[-1]
live_path2[0] = backtest_path2[-1]

for t in range(1, n_live):
    live_path1[t] = live_path1[t-1] * (1 + live_returns1[t-1])
    live_path2[t] = live_path2[t-1] * (1 + live_returns2[t-1])

# Create subplots
fig = make_subplots(rows=2, cols=1, subplot_titles=('Strategy 1 - Stable Performance', 
                                                   'Strategy 2 - Performance Degradation'))

# Plot Strategy 1
fig.add_trace(
    go.Scatter(x=backtest_dates, y=backtest_path1, name='Backtest', line=dict(color='rgba(0, 255, 255, 1)')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=live_dates, y=live_path1, name='Live Trading', line=dict(color='rgba(255, 0, 0, 1)')),
    row=1, col=1
)

# Plot Strategy 2
fig.add_trace(
    go.Scatter(x=backtest_dates, y=backtest_path2, name='Backtest', line=dict(color='rgba(0, 255, 255, 1)')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=live_dates, y=live_path2, name='Live Trading', line=dict(color='rgba(255, 0, 0, 1)')),
    row=2, col=1
)

# Add vertical lines at transition points
fig.add_vline(x=backtest_dates[-1], line_width=2, line_dash="dash", line_color="white", opacity=0.5)

# Calculate performance metrics
def sharpe_ratio(returns):
    return np.sqrt(252) * np.mean(returns) / np.std(returns)

backtest_sharpe1 = sharpe_ratio(np.diff(backtest_path1)/backtest_path1[:-1])
live_sharpe1 = sharpe_ratio(np.diff(live_path1)/live_path1[:-1])
backtest_sharpe2 = sharpe_ratio(np.diff(backtest_path2)/backtest_path2[:-1])
live_sharpe2 = sharpe_ratio(np.diff(live_path2)/live_path2[:-1])

# Update layout
fig.update_layout(
    height=800,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title=dict(
        text='Backtest vs Live Performance Comparison<br>' +
             f'Strategy 1 - Backtest Sharpe: {backtest_sharpe1:.2f}, Live Sharpe: {live_sharpe1:.2f}<br>' +
             f'Strategy 2 - Backtest Sharpe: {backtest_sharpe2:.2f}, Live Sharpe: {live_sharpe2:.2f}',
        y=.97
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
        row=i, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title="Portfolio Value ($)",
        row=i, col=1
    )

fig.show()



# **Reasons for Live Performance Degradation**
# 
# - You were just trading an exposure that did well for a period and is now mean reverting
# 
# - You were trading a statistical mispricing (alpha) but that got crowded and is no longer stable
# 
# - You overfit a backtest to noise and didn't ever have a viable strategy
# 
# In any case, the system is dynamic - mispricings may become viable again just as different exposures will perform, outperform, and degrade over different time intervals


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# TL;DW Executive Summary
# - Performance measures like win rate and cash generated are misleading from the perspective of *outcome*, we don't know anything about the performance of a trading strategy *at all*
# 
# - Expected return, risk measures like portfolio variance, volatility, and risk-adjusted returns give a more complete picture of the strategies performance but are leaving out an important piece of the puzzle - these must necessarily perform well but are not sufficient to show a strategy is tradable
# 
# - Stability in performance metrics is the most important component of analysis: Does the trading strategy actually offer a tradable mispricing?  If it does your alpha should persist performance that is relatively stable (**this suggests forward-looking performance**) before crowding, regime change; sometimes stability can decay and come back, so on and so forth
# 
# - Without evidence of stability, you can cite Sharpe ratios over 10+ and it won't matter since you can't trade it
# 
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Is the Market a Random Walk?
# 
# - The Trouble with Stationarity Tests
# 
# - Markov Chains and Hidden Markov Models
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Build an Earnings Event Options Trading Dashboard
# 
# - Live Regime Switching Models
# 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

