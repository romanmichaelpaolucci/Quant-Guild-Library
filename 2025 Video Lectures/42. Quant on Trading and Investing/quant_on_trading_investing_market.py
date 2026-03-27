# ### 🃏 Quant on Trading and Investing
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
# - [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)
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
# #### 1.) 📈 Risk and Return
# 
# - Fundamental Idea of Risk and Return
# 
# - Stability in Statistical Quantities
# 
# - Risks in the Equity Markets
# 
# - Risks in the Option Markets
# 
# #### 2.) 🛤️ Quant's View of the Market
# 
# - Market Exposure
# 
# - Studies on Uncertainty and Fear (Shiller)
# 
# - Implied Probabilities
# 
# - Psychology
# 
# #### 3.) 🃏 How I Trade and Invest
# 
# - Gambling, Poker, and Trading
# 
# - How I Trade
# 
# - How I Invest
# 
# #### 4.) 📊 Performance Metrics Trading and Investing
# 
# - Historic Performance Measures (All Nonsense)
# 
# - Considerations
# 
# #### 5.) 💭 Closing Thoughts and Future Comments


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


# #### 1.) 📈 Risk and Return
# 
# Without *risk* you will not receive a *return* - one of the most fundamental ideas in asset pricing
# 
# If there is no arbitrage we also have a risk-neutral measure where the value of an investment without risk must earn a risk-free rate


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample paths
n_days = 252  # One trading year
n_paths = 20  # Number of sample paths per risk level

# Parameters for low risk vs high risk
low_risk = {'mu': 0.0004, 'sigma': 0.01}  # ~10% annual return, ~16% vol
high_risk = {'mu': 0.0008, 'sigma': 0.02}  # ~20% annual return, ~32% vol

# Generate paths
def generate_paths(mu, sigma, n_paths, n_days):
    returns = np.random.normal(mu, sigma, (n_paths, n_days))
    prices = 100 * np.cumprod(1 + returns, axis=1)
    return prices

low_risk_paths = generate_paths(low_risk['mu'], low_risk['sigma'], n_paths, n_days)
high_risk_paths = generate_paths(high_risk['mu'], high_risk['sigma'], n_paths, n_days)

# Calculate expected paths
low_risk_expected = 100 * np.exp(np.arange(n_days) * (low_risk['mu'] + 0.5 * low_risk['sigma']**2))
high_risk_expected = 100 * np.exp(np.arange(n_days) * (high_risk['mu'] + 0.5 * high_risk['sigma']**2))

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Lower Risk, Lower Expected Return',
                                  'Higher Risk, Higher Expected Return'),
                    horizontal_spacing=0.1)

# Plot low risk paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(y=low_risk_paths[i], line=dict(color='rgba(0, 191, 255, 0.2)', width=1),
                  showlegend=False),
        row=1, col=1
    )

# Plot low risk expected path
fig.add_trace(
    go.Scatter(y=low_risk_expected, name='Expected Path (Low Risk)',
               line=dict(color='rgba(0, 191, 255, 0.8)', width=3, dash='dash')),
    row=1, col=1
)

# Plot high risk paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(y=high_risk_paths[i], line=dict(color='rgba(255, 99, 71, 0.2)', width=1),
                  showlegend=False),
        row=1, col=2
    )

# Plot high risk expected path
fig.add_trace(
    go.Scatter(y=high_risk_expected, name='Expected Path (High Risk)',
               line=dict(color='rgba(255, 99, 71, 0.8)', width=3, dash='dash')),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1,
        xanchor='left',
        yanchor='top'
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


# We are dealing with randomness, and the best we can do is an expectation (average)
# 
# We typically look at expected returns to measure returns and expected deviation from those returns to measure risk
# 
# These quantities are not stable, they don't converge, and they constantly change over time


# **Remark**: Just because we take on MORE risk DOES NOT mean we will have higher expected return
# 
# We can both optimally and sub-optimally allocate capital to different securities. . .
# 
# We can use different types of performance metrics to determine the relative effectiveness of different allocations
# 
# There are several problems with this idea, however, and we will discuss them below


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### ⚖️ Stability in Statistical Quantities
# 
# Whatever statistical technique we are applying to the market (from simple averages to principal component analysis) we need to consider their *stability*
# 
# By *stability* I am referring to how these measures change over time, if we assume a level (average return, standard deviation), or principal component, or whatever - and trade or invest based on that information it may not be correct at the time the decision is made or as time evolves past when that decision was made


# Generate sample data
np.random.seed(42)
n_bins = 20

# Backtest data - create monotonically increasing relationship
backtest_x = np.linspace(-3, 3, n_bins)
backtest_y = 0.3 * backtest_x + np.random.normal(0, 0.1, n_bins)

# Live trading data - create noisy relationship
live_x = np.linspace(-3, 3, n_bins)
live_y = np.random.normal(0, 0.5, n_bins)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Backtest Signal (We Assume is Stable)',
                                  'Live Trading Signal (Is Not Stable)'),
                    horizontal_spacing=0.1)

# Plot backtest data
fig.add_trace(
    go.Bar(x=backtest_x, y=backtest_y, 
           marker_color='rgba(0, 191, 255, 0.6)',
           name='Backtest Signal'),
    row=1, col=1
)

# Plot live trading data
fig.add_trace(
    go.Bar(x=live_x, y=live_y,
           marker_color='rgba(255, 99, 71, 0.6)',
           name='Live Signal'),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1000,
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True
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
        title_text='Signal Strength',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='Return',
        row=1, col=i
    )

fig.show()



# **Remark:** Assume the backtest was constructed correctly with no bias - this is a very real outcome we can face constructing a trading signal
# 
# If we find a statistical inefficiency or alpha and the signal holds we would expect it to remain relatively stable and look like the backtest signal
# 
# There is nothing gaurenteeing this stability, it can be stable then change over time, it can become stable again, etc. . . 


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 💥 Example: Equity Risks
# 
# Generally, we have three types of risk exposures when it comes to equities
# 
# - Idiosyncratic
# 
# - Industry
# 
# - Market
# 
# We can diversify away the first two, but we can't diversify away market risk.
# 
# If we diversify away that risk, we don't get the reward for being exposed to it.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate correlated random walks
np.random.seed(42)
n_days = 252  # One trading year
n_stocks = 15  # Per group

# Create two benchmark returns (one for each industry)
bm1_returns = np.random.normal(0.0007, 0.01, n_days)  # ~18% annual return, ~16% vol
bm2_returns = np.random.normal(0.0007, 0.01, n_days)

# Generate correlated stock returns
def generate_correlated_returns(bm_returns, correlation, n_stocks):
    stock_specific = np.random.normal(0, 0.02, (n_stocks, n_days))
    correlated = np.outer(np.ones(n_stocks), bm_returns)
    returns = correlation * correlated + np.sqrt(1 - correlation**2) * stock_specific
    return returns

# Generate two groups of stocks with different correlations to their benchmarks
group1_returns = generate_correlated_returns(bm1_returns, 0.7, n_stocks)
group2_returns = generate_correlated_returns(bm2_returns, 0.7, n_stocks)

# Convert returns to price series (start at 100)
def returns_to_prices(returns):
    return 100 * np.cumprod(1 + returns, axis=-1)

# Calculate prices
group1_prices = returns_to_prices(group1_returns)
group2_prices = returns_to_prices(group2_returns)

# Calculate portfolios
portfolio1 = np.mean(group1_prices, axis=0)
portfolio2 = np.mean(group2_prices, axis=0)
market_portfolio = np.mean(np.vstack([group1_prices, group2_prices]), axis=0)

# Create figure with subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Individual Stocks (Two Industries)',
                                  'Industry Portfolios',
                                  'Market Portfolio'),
                    vertical_spacing=0.1)

# Plot individual stocks
for i in range(n_stocks):
    fig.add_trace(
        go.Scatter(y=group1_prices[i], line=dict(color='rgba(0, 191, 255, 0.3)', width=1),
                  showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=group2_prices[i], line=dict(color='rgba(255, 99, 71, 0.3)', width=1),
                  showlegend=False),
        row=1, col=1
    )

# Plot industry portfolios
fig.add_trace(
    go.Scatter(y=portfolio1, name='Industry 1', line=dict(color='rgba(0, 191, 255, 0.8)', width=2)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=portfolio2, name='Industry 2', line=dict(color='rgba(255, 99, 71, 0.8)', width=2)),
    row=2, col=1
)

# Plot market portfolio
fig.add_trace(
    go.Scatter(y=market_portfolio, name='Market', line=dict(color='rgba(255, 255, 255, 0.8)', width=2)),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=900,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 4):
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



# Each exposure can be impacted by different things
# 
# - Idiosyncratic: CEO Scandal, New Product Success, New Product Flop, M&A
# 
# - Industry: Regulatory Risks, Seasonality, . . .
# 
# - Market: Interest Rates, Inflation, Government Policies, . . .
# 
# **Remark**: Impact in exposure is nested, if something impacts the market it impacts industries and each individual equity, same with industry to idiosyncratic.
# 
# Using a quantity called $\beta$ (among others) we can measure the impact of each type of risk on a particular equity - this value itself is not stable. . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### 📜 Example: Option Contract Risks
# 
# We have a variety of *different* risk exposures when holding a long or short option position
# 
# - $\rho$: Interest Rate Risk
# 
# - $\sigma$: Volatility Risk
# 
# - $\theta$: Time Decay
# 
# - $\delta$: Directional Risk


# Generate option value paths with decreasing volatility
np.random.seed(42)
n_days = 252  # One trading year
n_paths = 15  # Number of paths

# Initial parameters
S0 = 100  # Initial stock price 
K = 100   # Strike price
r = 0.02  # Risk-free rate
T = 1.0   # Time to maturity

# Generate decreasing volatility path
initial_vol = 0.30
final_vol = 0.15
vol_path = np.linspace(initial_vol, final_vol, n_days)

# Generate correlated stock paths
def generate_option_paths(n_paths):
    dt = T/n_days
    paths = np.zeros((n_paths, n_days))
    paths[:, 0] = S0
    
    for t in range(1, n_days):
        current_vol = vol_path[t]
        z = np.random.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * current_vol**2) * dt + 
                                           current_vol * np.sqrt(dt) * z)
    return paths

stock_paths = generate_option_paths(n_paths)

# Calculate put option values
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    from scipy.stats import norm
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

# Calculate option values along paths
option_values = np.zeros((n_paths, n_days))
time_grid = np.linspace(T, 0, n_days)

for t in range(n_days):
    option_values[:, t] = black_scholes_put(stock_paths[:, t], K, time_grid[t], r, vol_path[t])

# Create figure with subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Individual Option Value Paths',
                                  'Average Option Value',
                                  'Implied Volatility Path'),
                    vertical_spacing=0.1)

# Plot individual option value paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(y=option_values[i], line=dict(color='rgba(0, 191, 255, 0.3)', width=1),
                  showlegend=False),
        row=1, col=1
    )

# Plot average option value
avg_option_value = np.mean(option_values, axis=0)
fig.add_trace(
    go.Scatter(y=avg_option_value, name='Average Option Value', 
               line=dict(color='rgba(255, 255, 255, 0.8)', width=2)),
    row=2, col=1
)

# Plot volatility path
fig.add_trace(
    go.Scatter(y=vol_path * 100, name='Implied Volatility (%)', 
               line=dict(color='rgba(255, 99, 71, 0.8)', width=2)),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=900,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 4):
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



# ---


# #### 2.) 🛤️ Quant's Perspective on the Market


# Market exposure is generally considered to diversify away indiosyncratic and industry exposures.
# 
# Diversifying away that risk means you will not receive the reward for assuming that risk.
# 
# The broader market (and market exposure) is generally impacted by macroeconomic data and decisions made by governments etc . . .


# ##### 😱 Fear Dominates, Volatility Reverts to a Mean Level
# 
# *Goetzmann, William N.; Kim, Dasol; Shiller, Robert J. (2016). “Crash Beliefs From Investor Surveys,” NBER Working Paper No. 22143.*
# 
# **Nobody** knows what is going to happen.  Traders and investors (retailers, institutions) *always* overestimate the probability of a recession.
# 
# The study by Shiller et al. is one example where mean probability of a crash over the next six months is estimated roughly **6-19x** higher than historic data suggests
# 
# Crash defined in terms of roughly a 13% or 23% drop as observed in 1929 or 1987
# 
# **SPX to 2020**


import pandas as pd
import plotly.graph_objects as go

# Read SPX data
spx_data = pd.read_csv('SPX.csv')
spx_data['Date'] = pd.to_datetime(spx_data['Date'])

# Calculate daily returns
spx_data['Returns'] = spx_data['Close'].pct_change()

# Find dates with drops >= 13% and 23%
drops_13 = spx_data[spx_data['Returns'] <= -0.10]['Date']

# Create figure
fig = go.Figure()

# Plot SPX price
fig.add_trace(
    go.Scatter(
        x=spx_data['Date'],
        y=spx_data['Close'],
        name='SPX',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    )
)

# Add vertical lines for 13% drops
for date in drops_13:
    fig.add_shape(
        type='line',
        x0=date,
        x1=date,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='rgba(255, 165, 0, 0.5)', dash='dash')
    )
    fig.add_annotation(
        x=date,
        y=1,
        yref='paper',
        text='10% or Greater Drop',
        showarrow=False,
        yshift=10
    )

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    title='SPX Price History with Major Drops',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    ),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        title='SPX Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()



# 
# **Leverage Effect**: Returns tend to decline during periods of volatility disproportionately to the rise in returns from volatility decreases
# 
# What does this tell us?  If volatility is mean reverting (plenty of evidence there) and most of the time volatility is low and declining we will accumulate returns.


import pandas as pd
import plotly.graph_objects as go

# Read SPX data
spx_data = pd.read_csv('SPX.csv')
spx_data['Date'] = pd.to_datetime(spx_data['Date'])

# Calculate daily returns and volatility
spx_data['Returns'] = spx_data['Close'].pct_change()
spx_data['Volatility'] = spx_data['Returns'].rolling(window=21).std() * (252 ** 0.5)

# Create figure
fig = go.Figure()

# Plot volatility
fig.add_trace(
    go.Scatter(
        x=spx_data['Date'],
        y=spx_data['Volatility'],
        name='Volatility (21-day)',
        line=dict(color='rgba(0, 255, 255, 0.8)', width=2)
    )
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    title='SPX Volatility Over Time',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    ),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        title='Annualized Volatility',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()


# ##### 🎯 Implied Probability
# 
# There is a *massive* difference between *implied probability* and *probability* - they are **NOT** the same.
# 
# Volatility from tariffs during the start of 2025 had the *implied probability* of a recession in 2025 at roughly 66% at its highest
# 
# This is **NOT** the same as a coin flip, it is largely based on where the money lies


# Simulate Law of Large Numbers for coin flips and St. Petersburg Paradox
n_trials = 1000

# Coin flip simulation
coin_flips = np.random.binomial(1, 0.5, n_trials)
cumulative_mean = np.cumsum(coin_flips) / np.arange(1, n_trials + 1)

# St. Petersburg Paradox simulation
def st_petersburg_game():
    flips = 1
    while np.random.binomial(1, 0.5) == 0:
        flips += 1
    return 2 ** flips

st_petersburg_results = np.array([st_petersburg_game() for _ in range(n_trials)])
st_petersburg_mean = np.cumsum(st_petersburg_results) / np.arange(1, n_trials + 1)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Coin Flip LLN Convergence', 'St. Petersburg Paradox (No Convergence)'),
    horizontal_spacing=0.1
)

# Plot coin flip convergence
fig.add_trace(
    go.Scatter(
        y=cumulative_mean,
        name='Sample Mean',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    ),
    row=1, col=1
)
fig.add_hline(
    y=0.5,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=1
)

# Plot St. Petersburg paradox
fig.add_trace(
    go.Scatter(
        y=st_petersburg_mean,
        name='Expected Value',
        line=dict(color='rgba(255, 99, 71, 0.8)', width=2)
    ),
    row=1, col=2
)

# Add horizontal line for empirical mean of St. Petersburg game
empirical_mean = np.mean(st_petersburg_results)
fig.add_hline(
    y=empirical_mean,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Trials',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Sample Mean' if i == 1 else 'Expected Payoff',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()



# ##### 🏈 Example: Sportsbetting American Odds (Moneyline Odds)
# 
# The market and bookmakers construct the odds and implied probability is derived from this value, it is not a probability
# 
# Just like how the *Implied Volatility* from Black-Scholes is not *actually* the forward looking volatility
# 
# - Positive Odds $+X$
# 
# $$\text{Implied Probability} = \frac{100}{X + 100}$$
# 
# - Negative Odds $-X$
# 
# $$\text{Implied Probability} = \frac{X}{X + 100}$$
# 
# *For those who care, if there is a vig, normalize the implied probabilities to find the vig-free odds and implied probabilities*


# **Implied Probabilities are a Proxy for How Priced in Something Is**
# 
# Suppose a rate cut implies bullish price action the following trading day. . .
# 
# - If the implied probability of a rate cut across multiple sources is roughly 90\% - the effect of a rate cut may be marginal
# 
# - If the implied probability of a rate cut across multiple sources is roughly 60\% - the effect of a rate cut may be substatial
# 
# This is where you can gain an edge, if you trade in the *correct direction* **on average** you can generate P/L
# 
# I know many folks that only event trade in this capacity


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample price data
n_points = 100
x = np.arange(n_points)

# 90% priced in scenario
np.random.seed(42)
price_90 = 100 + np.cumsum(np.random.normal(0, 0.2, n_points))
# Small bump after event
price_90[50:] += 0.5

# 60% priced in scenario 
np.random.seed(42)
price_60 = 100 + np.cumsum(np.random.normal(0, 0.2, n_points))
# Larger bump after event
price_60[50:] += 3

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('90% Priced In Rate Cut', '60% Priced In Rate Cut'),
    horizontal_spacing=0.1
)

# Plot 90% priced in scenario
fig.add_trace(
    go.Scatter(
        x=x, y=price_90,
        name='Price Action (90%)',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    ),
    row=1, col=1
)

# Plot 60% priced in scenario
fig.add_trace(
    go.Scatter(
        x=x, y=price_60,
        name='Price Action (60%)',
        line=dict(color='rgba(255, 99, 71, 0.8)', width=2)
    ),
    row=1, col=2
)

# Add vertical lines for rate cut event
fig.add_vline(
    x=50,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=1
)
fig.add_vline(
    x=50,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Time',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()



# ##### 🌐 Example: Polymarket Implied Probabilities
# 
# Implied probabilities are derived similar to the mechanics above but market equilibrium dictates them rather than being backed out from odds
# 
# Each contract has a value and has a binary outcome \$1 if true \$0 if false
# 
# This is similar to the idea of simulating a Bernoulli random variable to generate model probabilities
# 
# **Remark**: *I do not exclusively consider polymarket for implied probabilities* - this is an example


# Simulate convergence of probabilities with different true vs implied probabilities
n_trials = 1000

# Case 1: True prob = 0.1, Implied prob = 0.1 (Correct market assessment)
true_prob_1 = 0.1
outcomes_1 = np.random.binomial(1, true_prob_1, n_trials)
cumulative_mean_1 = np.cumsum(outcomes_1) / np.arange(1, n_trials + 1)

# Case 2: St. Petersburg-like scenario (no convergence)
st_petersburg_results = np.array([st_petersburg_game() for _ in range(n_trials)])
st_petersburg_mean = np.cumsum(st_petersburg_results) / np.arange(1, n_trials + 1)

# Case 3: True prob = 0.01, Implied prob = 0.1 (Market overestimation)
true_prob_2 = 0.01
outcomes_2 = np.random.binomial(1, true_prob_2, n_trials)
cumulative_mean_2 = np.cumsum(outcomes_2) / np.arange(1, n_trials + 1)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Correct Market Assessment', 'Non-Convergent Process', 'Market Overestimation'),
    horizontal_spacing=0.1
)

# Plot Case 1
fig.add_trace(
    go.Scatter(
        y=cumulative_mean_1,
        name='Sample Mean (True=0.1)',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    ),
    row=1, col=1
)
fig.add_hline(
    y=0.1,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=1
)

# Plot St. Petersburg case
fig.add_trace(
    go.Scatter(
        y=st_petersburg_mean,
        name='Non-Convergent Process',
        line=dict(color='rgba(255, 99, 71, 0.8)', width=2)
    ),
    row=1, col=2
)

# Plot Case 3
fig.add_trace(
    go.Scatter(
        y=cumulative_mean_2,
        name='Sample Mean (True=0.01)',
        line=dict(color='rgba(50, 205, 50, 0.8)', width=2)
    ),
    row=1, col=3
)
fig.add_hline(
    y=0.1,
    line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
    row=1, col=3
)

# Update layout
fig.update_layout(
    width=1500,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
for i in range(1, 4):
    fig.update_xaxes(
        title='Number of Trials',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Probability' if i != 2 else 'Expected Value',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()


# ##### 🧠 Psychology
# 
# Algorithmic and quantitative trading does not remove emotion from trading - it automates components of it.  
# 
# Fundamentally, traders still have risk exposures that they manage otherwise they would not be able to generate a return beyond the risk-free rate.
# 
# Yes, arbitrage exists, risks can be managed automatically - still there is someone at the helm operating the system.
# 
# **Making Money** does not feel as good as **Losing Money** feels bad 


# ---


# #### 3.) 🃏 How I Trade and Invest


# ##### 🎰 Gambling, Poker, and Trading
# 
# There is a difference between a game of chance where you are subject to a negative edge and are gaurenteed statistically to lose all of your money
# 
# and games of incomplete information like Poker and Trading.  I've discussed this in the past, and would like to further discuss these ideas in another video soon
# 
# This is all about expected value and the idea of "edge" - I do plenty of deep dives into this topic so I encourage you to check out those videos
# 
# **Games of Chance**
# 
# There is no *optimal* decision to make, you will lose money statistically if you keep playing (for example, Roulette)
# 
# **Games of Incomplete Information**
# 
# Poker and Trading are games of incomplete information, you can act suboptimally with extremely negative expected value or edge which some may say is akin to gambling but it is just poor decision making


# Simulate suboptimal vs optimal betting strategies
n_trials = 1000
bet_size = 1

# Case 1: Suboptimal betting (negative expected value)
poor_edge = -0.03  # -3% edge per trade due to poor decisions
variance = 0.15  # Higher variance due to poor risk management
outcomes_suboptimal = np.random.normal(poor_edge, variance, n_trials) * bet_size
cumulative_suboptimal = np.cumsum(outcomes_suboptimal)

# Case 2: Optimal betting (positive expected value)
good_edge = 0.02  # 2% edge per trade with proper strategy
variance = 0.08  # Lower variance with better risk management
outcomes_optimal = np.random.normal(good_edge, variance, n_trials) * bet_size
cumulative_optimal = np.cumsum(outcomes_optimal)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Suboptimal Trading Strategy', 'Optimal Trading Strategy'),
    horizontal_spacing=0.1
)

# Plot Case 1 - Suboptimal Trading
fig.add_trace(
    go.Scatter(
        y=cumulative_suboptimal,
        name='Poor Strategy P&L',
        line=dict(color='rgba(255, 99, 71, 0.8)', width=2)
    ),
    row=1, col=1
)

# Plot Case 2 - Optimal Trading
fig.add_trace(
    go.Scatter(
        y=cumulative_optimal,
        name='Good Strategy P&L',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Trades',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Cumulative P&L',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()


# ##### 🎲 How I Trade
# 
# **You need strategies PLURAL**
# 
# **They will not always be correct**
# 
# I'll pick some instrument say stock, options, or crypto and determine the current environment
# 
# I'll pick what kind of exposure I want based on what I think is going to happen (sometimes I am wrong)
# 
# - Risk has to come off the books, profit taking activities $\implies$ Directional Exposure 
# 
# - Implied Volatility is Overstated $\implies$ Negative Vega Exposure
# 
# - Unwinding a Long-Term Investment $\implies$ Selling Calls Above Spot
# 
# - Crypto is Mid-Range of a Bull-Cycle $\implies$ Directional Exposure


# Simulate volatility around earnings event
n_points = 100
pre_earnings_vol = np.random.normal(0.15, 0.02, n_points//2)
post_earnings_vol = np.concatenate([
    np.random.normal(0.35, 0.05, 10),  # Spike in volatility
    np.random.normal(0.12, 0.02, n_points//2 - 10)  # Decline afterward
])
volatility = np.concatenate([pre_earnings_vol, post_earnings_vol])

# Create figure
fig = go.Figure()

# Plot volatility line
fig.add_trace(
    go.Scatter(
        y=volatility,
        name='Volatility',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    )
)

# Add vertical line for earnings event
fig.add_vline(
    x=n_points//2, 
    line_width=2,
    line_dash="dash",
    line_color="rgba(255, 99, 71, 0.8)",
    annotation_text="Earnings Event",
    annotation_position="top"
)

# Update layout
fig.update_layout(
    width=1500,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
fig.update_xaxes(
    title='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    title='Volatility',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()


# ##### 📈 How I Invest
# 
# I'll pick some ETFs and incrementally allocate cash . . .
# 
# There is *NO TIMING* it is about *TIME IN*
# 
# I'll determine what to do based on the current macro environment
# 
# - Long-Term Investing $\implies$ Weekly Allocations for Market Exposure
# 
# - VIX Spike and Fear $\implies$ Lump Sum Allocation
# 
# - Generating Cash $\implies$ Daily Trading Swings in ETFs Holding on the Downside
# 
# **Volatility Clustering**: Periods of high volatility tend to follow high volatility until it dissipates


# **Remark**: We don't know where the top of volatility is, we don't know where the bottom of the market is.  We know that volatility will rever to a mean value and we know the market tends upward mechanically. . .we have a variety of techniques available to invest knowing this mentioned above


# Read SPX data


spx_data = pd.read_csv('SPX.csv')[6000:]
spx_data['Date'] = pd.to_datetime(spx_data['Date'])
spx_data.set_index('Date', inplace=True)

# Calculate daily returns and volatility
spx_data['Returns'] = spx_data['Adj Close'].pct_change()
spx_data['Volatility'] = spx_data['Returns'].rolling(window=21).std() * np.sqrt(252) * 100

# Initialize investment tracking
spx_data['Lump_Sum'] = 0
spx_data['Incremental'] = 0
incremental_invested = 0
mid_point = len(spx_data) // 2

# Track investments based on volatility spikes
for i in range(len(spx_data)):
    if i == mid_point:  # Lump sum investment in middle of series
        spx_data['Lump_Sum'].iloc[i] = 5000
    elif i > mid_point and spx_data['Volatility'].iloc[i] > 40 and incremental_invested < 5000:
        spx_data['Incremental'].iloc[i] = 500
        incremental_invested += 500

# Create figure
fig = go.Figure()

# Plot volatility
fig.add_trace(
    go.Scatter(
        x=spx_data.index,
        y=spx_data['Volatility'],
        name='Volatility (%)',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    )
)

# Plot lump sum investments
fig.add_trace(
    go.Scatter(
        x=spx_data[spx_data['Lump_Sum'] > 0].index,
        y=spx_data[spx_data['Lump_Sum'] > 0]['Volatility'],
        mode='markers',
        name='Lump Sum $5000',
        marker=dict(
            color='rgba(255, 99, 71, 0.8)',
            size=12,
            symbol='triangle-up'
        )
    )
)

# Plot incremental investments
fig.add_trace(
    go.Scatter(
        x=spx_data[spx_data['Incremental'] > 0].index,
        y=spx_data[spx_data['Incremental'] > 0]['Volatility'],
        mode='markers',
        name='Incremental $500',
        marker=dict(
            color='rgba(50, 205, 50, 0.8)',
            size=8,
            symbol='circle'
        )
    )
)

# Update layout
fig.update_layout(
    width=1500,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
    )
)

# Update axes
fig.update_xaxes(
    title='Date',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    title='Volatility (%)',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

# Calculate and print total investments and returns
total_lump_sum = spx_data['Lump_Sum'].sum()
total_incremental = spx_data['Incremental'].sum()
total_investment = total_lump_sum + total_incremental

# Calculate returns from investment points
lump_sum_returns = []
incremental_returns = []

for i in range(len(spx_data)):
    if spx_data['Lump_Sum'].iloc[i] > 0:
        initial_price = spx_data['Adj Close'].iloc[i]
        final_price = spx_data['Adj Close'].iloc[-1]
        lump_sum_return = ((final_price - initial_price) / initial_price) * 100
        lump_sum_returns.append(lump_sum_return)
    
    if spx_data['Incremental'].iloc[i] > 0:
        initial_price = spx_data['Adj Close'].iloc[i]
        final_price = spx_data['Adj Close'].iloc[-1]
        incremental_return = ((final_price - initial_price) / initial_price) * 100
        incremental_returns.append(incremental_return)

avg_lump_sum_return = np.mean(lump_sum_returns) if lump_sum_returns else 0
avg_incremental_return = np.mean(incremental_returns) if incremental_returns else 0

print(f"Total Lump Sum Investment: ${total_lump_sum:,.2f}")
print(f"Average Lump Sum Return: {avg_lump_sum_return:.2f}%")
print(f"\nTotal Incremental Investment: ${total_incremental:,.2f}")
print(f"Average Incremental Return: {avg_incremental_return:.2f}%")
print(f"\nTotal Combined Investment: ${total_investment:,.2f}")
print(f"Combined Return: ${total_investment * (1 + (avg_lump_sum_return + avg_incremental_return)/200):,.2f}")


# ---


# #### 4.) 📊 Performance Metrics Trading and Investing
# 
# Yes, we can always look at the Sharpe ratio or Sortino ratio. . .
#  
#   $$\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}$$
#   
#   $$\text{Sortino Ratio} = \frac{E[R_p] - R_f}{\sigma_d}$$
#  
# But are we allowed to use historical data?  Is historical data indicative of future performance?
# 
# Probabilities and expectations don't converge, everything is largely nonsense - all models are wrong, some are useful
# 
# This is *similar* to the idea of an implied probability not actually giving you a probability of an outcome
# 
# 


# Generate two strategies with different risk-adjusted returns
np.random.seed(42)
n_days = 252  # One trading year

# Strategy parameters
initial_capital = 100000
strategy1_ret = 0.15  # Higher return strategy
strategy1_vol = 0.20  # Higher volatility
strategy2_ret = 0.10  # Lower return strategy 
strategy2_vol = 0.12  # Lower volatility
rf_rate = 0.02

# Generate returns
def generate_strategy_returns(mu, sigma, n_days):
    daily_ret = np.random.normal(mu/252, sigma/np.sqrt(252), n_days)
    cumulative_ret = np.cumprod(1 + daily_ret)
    return daily_ret, initial_capital * cumulative_ret

# Generate strategy paths
daily_ret1, strategy1_path = generate_strategy_returns(strategy1_ret, strategy1_vol, n_days)
daily_ret2, strategy2_path = generate_strategy_returns(strategy2_ret, strategy2_vol, n_days)

# Calculate Sharpe and Sortino ratios
def calculate_ratios(returns, rf_rate):
    excess_ret = np.mean(returns) * 252 - rf_rate
    volatility = np.std(returns) * np.sqrt(252)
    downside_vol = np.std(returns[returns < 0]) * np.sqrt(252)
    
    sharpe = excess_ret / volatility
    sortino = excess_ret / downside_vol
    return sharpe, sortino

sharpe1, sortino1 = calculate_ratios(daily_ret1, rf_rate)
sharpe2, sortino2 = calculate_ratios(daily_ret2, rf_rate)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Strategy 1 (Sharpe: {sharpe1:.2f}, Sortino: {sortino1:.2f})',
                                  f'Strategy 2 (Sharpe: {sharpe2:.2f}, Sortino: {sortino2:.2f})'),
                    horizontal_spacing=0.1)

# Plot strategy paths
fig.add_trace(
    go.Scatter(y=strategy1_path, name='Strategy 1', 
               line=dict(color='rgba(0, 191, 255, 0.8)', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=strategy2_path, name='Strategy 2',
               line=dict(color='rgba(255, 99, 71, 0.8)', width=2)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(0,0,0,0)'
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



# **Q**: Roman, are you going to lever up on a system that has a 2+ Sharpe ratio
# 
# **A**: No - how hard is the system to trade, what is the strategy class, whats the max drawdown, . . .


# **Remark**: Without these metrics being time variant quantities, they are subject to change at any time and over time and require careful monitoring. . .


# ###### ______________________________________________________________________________________________________________________________________


# **Considerations**
# 
# Analyzing a strategy on performance metrics alone is a ridiculous notion and leaves out a lot of important information
# 
# - *How* were trading or investing decisions generated, actively, passively, etc. . .
# 
# - *When* do you have access to your capital, not liquidity, *what proportion of the time are you in cash* to make decisions with
# 
# - *Outperforming* the market or buy-and-hold is not **THE** measure of success


# ---


# #### 5.) 💭 Closing Thoughts and Future Comments
# 
# Trading and Investing can be summarized as follows
# 
# - You must assume a type of risk to receive return beyond the risk-free rate
# 
# - We are operating with incomplete information, we can always burn money (act suboptimally), but that doesn't make it gambling
# 
# - Stability of strategies, risk measures, performance measures is never gaurenteed and quite rare even if you find statistical inefficiencies in the market
# 
# - Trading is not a part time job, investing certainly can be - choose whichever you have sufficient *knowledge* and *capacity* for
# 
# **Future Topics**
# 
# - Time Series Analysis, why it *doesn't work*
# 
# - Algorithmic *Investing* Systems (Generate Cash, Hold for Drawdown) using Interactive Brokers
# 
# - Trading Tools, Trading Systems using Interactive Brokers
# 
# - Deriving Option Pricing Partial Differential Equations
# 
# - ARCH and GARCH for Modeling Volatility
# 
# - Simulating Stochastic Processes (Gaussian Cookbook)


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

