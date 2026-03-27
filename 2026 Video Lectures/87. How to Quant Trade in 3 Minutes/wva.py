# ### 📈 How to Quant Trade in 3 Minutes
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
# 
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
# #### 1.) 📉 Developing Trading Models
# 
# #### 2.) 📊 Frequency Based Expectations
# 
# #### 3.) 🎯 Our Jobs as Traders and Investment Professionals
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📉 Developing Trading Models
# 
# We always hear that past performance isn't indicative of future performance - then why do we use historical data at all?
# 
# The truth is, we all have access to the same data, there is no hedge fund or quant fund with a crystal ball, we all operate with information available at time $t$ (now).
# 
# *Note: I am not making a remark about information that is exclusive, insider, or costs a premium*
# 
# Some are just better at making sense of it than others - this is the model specification and parameterization problem, and why modeling is both an art and science.
# 
# Select a model $\mathcal{M}$ and parameters $\Theta$
# 
# Here, I select a linear regression model $\mathcal{M}$ and calibrate the parameters $\Theta$ to historic return and volatility data.


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Setup (Synthetic) ---
np.random.seed(42)
days = 500 
dates = pd.date_range(start='2023-01-01', periods=days, freq='B')

# Generate VIX (Mean reverting Ornstein-Uhlenbeck)
vix_path = np.zeros(days)
vix_path[0] = 20.0
for t in range(1, days):
    shock = np.random.normal(0, 1)
    vix_path[t] = vix_path[t-1] + 0.1 * (18 - vix_path[t-1]) + 1.5 * shock
vix_path = np.maximum(vix_path, 9) 

# Generate SPY returns (negatively correlated to VIX changes)
spy_ret = np.random.normal(0.0005, 0.01, days)
vix_diff = np.diff(vix_path, prepend=vix_path[0])
spy_ret = spy_ret - 0.002 * vix_diff 
spy_price = 400 * np.cumprod(1 + spy_ret)

df = pd.DataFrame({'Date': dates, 'SPY': spy_price, 'VIX': vix_path})

# --- 2. Calculate Volatility Risk Premium (VRP) ---
window = 21 
df['Log_Ret'] = np.log(df['SPY'] / df['SPY'].shift(1))

# Forward Realized Volatility
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
df['Realized_Vol_NextMonth'] = df['Log_Ret'].rolling(window=indexer).std() * np.sqrt(252) * 100

df['VRP_Spread'] = df['VIX'] - df['Realized_Vol_NextMonth']
df['Strategy_PnL'] = df['VRP_Spread'].fillna(0) / 100 
df['Wealth'] = 100 * np.cumprod(1 + df['Strategy_PnL'] * 0.1)

df = df.dropna()
n_frames = len(df)

# --- 3. Visualization Function ---
def make_vrp_frame(step):
    curr_df = df.iloc[:step+1]
    
    x_reg = curr_df['VIX']
    y_reg = curr_df['Realized_Vol_NextMonth']
    
    # Calculate regression equation
    reg_equation = "Calculating..."
    if len(curr_df) > 1:
        m, b = np.polyfit(x_reg, y_reg, 1)
        reg_line_x = np.linspace(x_reg.min(), x_reg.max(), 10)
        reg_line_y = m * reg_line_x + b
        
        # Format the equation string: y = mx + b
        sign = "+" if b >= 0 else "-"
        reg_equation = f"y = {m:.3f}x {sign} {abs(b):.3f}"
    else:
        reg_line_x, reg_line_y = [], []

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Implied (VIX) vs Future Realized Vol", "VRP Strategy Wealth (Short Vol)"),
        horizontal_spacing=0.1
    )

    # LEFT CHART: Scatter VIX vs Realized
    fig.add_trace(go.Scatter(
        x=[10, 50], y=[10, 50], mode='lines',
        line=dict(color='#d400ff', width=2, dash='dash'),
        name="Fair Value (1:1)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=curr_df['VIX'], y=curr_df['Realized_Vol_NextMonth'],
        mode='markers',
        marker=dict(color='#00ffff', size=5, opacity=0.6),
        name="Daily Obs"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=reg_line_x, y=reg_line_y, mode='lines',
        line=dict(color='#FFD700', width=2),
        name="Forward Regression"
    ), row=1, col=1)

    # RIGHT CHART: Wealth
    fig.add_trace(go.Scatter(
        x=curr_df['Date'], y=curr_df['Wealth'], mode='lines',
        line=dict(color='#00ffff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)',
        name="Cumulative Wealth"
    ), row=1, col=2)

    # Styling
    off_white = "#e0e0e0"
    axis_config = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=off_white))
    fig.update_xaxes(axis_config); fig.update_yaxes(axis_config)

    fig.update_xaxes(range=[9, 45], title_text="Implied Vol (VIX)", row=1, col=1)
    fig.update_yaxes(range=[0, 45], title_text="Realized Vol (Next 21 Days)", row=1, col=1)
    fig.update_xaxes(range=[df['Date'].iloc[0], df['Date'].iloc[-1]], row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, width=1000,
        # Updated Title with Regression Equation
        title_text=f"Volatility Risk Premium Analysis (Regression: {reg_equation})",
        font=dict(color=off_white),
        showlegend=False,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    return fig

# --- 4. Animation ---
frames = [
    go.Frame(data=make_vrp_frame(k).data, layout=make_vrp_frame(k).layout, name=str(k))
    for k in range(0, n_frames, 5)
]

fig = make_vrp_frame(0)
fig.frames = frames

fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'buttons': [{
            'label': '▶ Play VRP Regression',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'fromcurrent': True}]
        }],
        'font': {'color': "#143c99", 'size': 16}
    }]
)

fig.show()


# This model gives me an expectation of how volatility will behave in a forward looking sense.
# 
# In any case, models have an *expectation*, statistics tells us in the context of a random variable it is literally our best guess (minimizes the MSE)
# 
# $$
# \text{Let } X \text{ be a random variable and } a \text{ our estimate.}
# $$
# 
# $$
# \text{MSE}(a) = \mathbb{E}\left[(X - a)^2\right]
# $$
# 
# $$
# \frac{d}{da} \mathbb{E}\left[(X - a)^2\right] = 0
# $$
# 
# $$
# \mathbb{E}\left[2(a - X)\right] = 0
# $$
# 
# $$
# 2(a - \mathbb{E}[X]) = 0
# $$
# 
# $$
# a = \mathbb{E}[X]
# $$
# 
# 
# 
# 
# But there is a problem academia rarely tells their students explicitly: *we don't observe convergence in the real world, statistics change over time*.


# ---


# #### 2.) 📊 Frequency Based Expectations
# 
# Let $X$ be a random variable, perhaps the P/L outcome of a trade.
# 
# In the classroom we learn that $\mathbb{E}[X]$ is a gaurentee for a random variable $X$ that has independent and identically distributed draws by the LLN
# 
# Trivially, if we can just keep drawing from this distribution (or observing data) we can find $\mathbb{E}[X]$.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Setup ---
mu, sigma = 0, 1  # Normal parameters
n_trials = 250
np.random.seed(42)
samples = np.random.normal(mu, sigma, size=n_trials)

# x-axis for the theoretical PDF
x_range = np.linspace(-4, 4, 200)
pdf_normal = norm.pdf(x_range, mu, sigma)

# Cumulative averages for LLN
cum_averages = np.cumsum(samples) / np.arange(1, n_trials + 1)

# --- Helper: construct figure for a given step ---
def make_normal_convergence_fig(step):
    current_samples = samples[:step+1]
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Empirical vs. Theoretical", "Cumulative Average (LLN)"),
    )

    # --- Left subplot: Empirical Histogram + Theoretical PDF ---
    # Empirical Histogram
    fig.add_trace(
        go.Histogram(
            x=current_samples,
            histnorm='probability density',
            nbinsx=20,
            marker=dict(color='#00ffff', opacity=0.6),
            name="Empirical PDF",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Theoretical PDF line
    fig.add_trace(
        go.Scatter(
            x=x_range, y=pdf_normal,
            mode='lines',
            line=dict(color='#d400ff', width=3),
            name="Normal PDF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: LLN Cumulative Average ---
    x_lln = np.arange(1, step + 2)
    y_lln = cum_averages[:step + 1]

    # Cumulative average line
    fig.add_trace(
        go.Scatter(
            x=x_lln, y=y_lln,
            mode='lines',
            line=dict(color='#00ffff', width=2),
            name="Cumulative Average",
            showlegend=False
        ),
        row=1, col=2
    )

    # Theoretical Mean Line
    fig.add_trace(
        go.Scatter(
            x=[0, n_trials], y=[mu, mu],
            mode='lines',
            line=dict(color='#d400ff', width=2, dash='dash'),
            name="True Mean (μ)",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Legend-only traces (to match your original UI) ---
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#d400ff', width=4), name="Theoretical"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#00ffff', width=4), name="Empirical/Average"), row=1, col=2)

    # --- Styling & Axes ---
    fig.update_xaxes(title_text="Value", row=1, col=1, range=[-4, 4])
    fig.update_yaxes(title_text="Density", row=1, col=1, range=[0, 0.5])
    fig.update_xaxes(title_text="Trials (n)", row=1, col=2, range=[0, n_trials])
    fig.update_yaxes(title_text="Average", row=1, col=2, range=[mu-1, mu+1])

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    fig.update_layout(
        height=480, width=960,
        title_text=f"Normal Distribution and Statistical Convergence",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(x=0.97, y=0.98, xanchor='right', yanchor='top', bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=14)),
        margin=dict(l=50, r=20, b=80, t=70),
    )

    return fig

# --- Animation frames ---
# Using a step of 2 to keep the animation smooth and the file size manageable
frames = [
    go.Frame(data=make_normal_convergence_fig(step).data, name=str(step))
    for step in range(0, n_trials, 2)
]

# --- Initial figure ---
fig = make_normal_convergence_fig(0)
fig.frames = frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()


# We can then very easily use this to trade, in fact, regardless of your strategy this is the statistical mechanism traders use to make money.
# 
# $$\text{Market Price} > \mathbb{E}[X] \implies \text{ Short}$$
# 
# $$\text{Market Price} < \mathbb{E}[X] \implies \text{ Long}$$
# 
# Any time we observe a deviation in the market from the expectation we can take the long or short position and over time we will accumulate the edge.
# 
# Below, we assume the expectation we have observed in the market from historical data has converged and we can trade mean reversion around it


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (GBM-esque Mean Reversion) ---
np.random.seed(42)
days = 252
dates = pd.date_range(start='2025-01-01', periods=days, freq='B')

# A. Theoretical Drift (The Anchor)
# Annual drift of ~5%
mu_drift = 0.05
dt = 1/252
theoretical_price = 100 * np.exp(mu_drift * np.linspace(0, 1, days))

# B. Realized Price (GBM-like Mean Reversion)
# We model the log-deviation as an Ornstein-Uhlenbeck process
theta = 15.0   # Speed of reversion
sigma = 0.30   # Volatility 
X = np.zeros(days) # Log-deviation

for t in range(1, days):
    dW = np.random.normal(0, np.sqrt(dt))
    X[t] = X[t-1] + theta * (0 - X[t-1]) * dt + sigma * dW

realized_price = theoretical_price * np.exp(X)

# C. Strategy Logic
# Signal: Short if Realized > Theory, Long if Realized < Theory
deviation = theoretical_price - realized_price 
signal = np.sign(deviation) 
shifted_signal = np.roll(signal, 1)
shifted_signal[0] = 0

# Calculate Asset Returns
asset_returns = np.diff(realized_price) / realized_price[:-1]
asset_returns = np.insert(asset_returns, 0, 0)

# Strategy PnL
strategy_returns = shifted_signal * asset_returns
wealth_path = 100 * np.cumprod(1 + strategy_returns)

# Calculate Sharpe Ratio (Annualized)
daily_mean = np.mean(strategy_returns)
daily_std = np.std(strategy_returns)
sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)

# --- 2. Helper: Figure Construction ---
def make_mr_frame(step):
    
    # Slice data
    curr_dates = dates[:step+1]
    curr_theory = theoretical_price[:step+1]
    curr_real = realized_price[:step+1]
    curr_wealth = wealth_path[:step+1]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Asset Price vs. Theoretical Value", "Strategy Wealth (P&L)"),
        horizontal_spacing=0.1
    )

    # --- LEFT PLOT: Price ---
    # Theoretical (Drift)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_theory,
        mode='lines',
        line=dict(color='#d400ff', width=2, dash='dash'),
        name="Theoretical"
    ), row=1, col=1)

    # Realized (GBM Style)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_real,
        mode='lines',
        line=dict(color='#00ffff', width=1.5),
        name="Realized"
    ), row=1, col=1)

    # --- RIGHT PLOT: Wealth ---
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_wealth,
        mode='lines',
        line=dict(color='#00ffff', width=2),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 255, 0.1)',
        name="Wealth"
    ), row=1, col=2)

    # --- Styling & Fixed Ranges ---
    off_white = "#e0e0e0"
    
    # Left Axis: Fixed based on max/min of full series
    p_min = min(np.min(theoretical_price), np.min(realized_price)) * 0.95
    p_max = max(np.max(theoretical_price), np.max(realized_price)) * 1.05
    fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
    fig.update_yaxes(range=[p_min, p_max], title_text="Price", row=1, col=1)
    
    # Right Axis: Fixed based on wealth outcome
    w_min = np.min(wealth_path) * 0.95
    w_max = np.max(wealth_path) * 1.05
    fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
    fig.update_yaxes(range=[w_min, w_max], title_text="Wealth Index", row=1, col=2)
    
    # Grid & Ticks
    axis_config = dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color=off_white), title_font=dict(color=off_white)
    )
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, width=1000,
        # Title with Sharpe Ratio, removed "Day"
        title_text=f"Trading Mean Reversion (Sharpe: {sharpe_ratio:.2f})",
        font=dict(color=off_white),
        showlegend=False,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    return fig

# --- 3. Animation Build ---
frames = [
    go.Frame(data=make_mr_frame(k).data, name=str(k))
    for k in range(0, days, 2)
]

fig = make_mr_frame(0)
fig.frames = frames

# Play Button Config (Restored Dark Blue)
play_darkblue = "#143c99"

fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }],
        'font': {'color': play_darkblue, 'size': 16} 
    }]
)

fig.show()


# The problem: in reality, we aren't drawing from a distribution, these are real companies with real operations and real agents making decisions.
# 
# For this reason, statistics, like the expectation, do not converge in reality as returns don't come from independent and identical distributions.
# 
# We may have a reasonable proxy for an estimation today, maybe for a week or even a month.  
# 
# But what happens when it changes or our model does not effectively estimate it anymore?


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (Structural Break Scenario) ---
np.random.seed(42)
days_per_year = 252
total_days = days_per_year * 2  # 2025 and 2026
dates = pd.date_range(start='2025-01-01', periods=total_days, freq='B')

# A. The TRADER'S Model (Purple Dashed)
# Trader assumes constant +5% drift forever
model_drift_rate = 0.05
dt = 1/252
model_price = 100 * np.exp(model_drift_rate * np.linspace(0, 2, total_days))

# B. The TRUE Regime (Gold Dashed)
# Year 1: Matches Model
# Year 2: Structural Break -> -30% drift (severe downtrend)
true_regime_price = model_price.copy()
start_price_2026 = true_regime_price[days_per_year]
year_2_drift = -0.30
t_year2 = np.linspace(0, 1, days_per_year)
true_regime_price[days_per_year:] = start_price_2026 * np.exp(year_2_drift * t_year2)

# C. Realized Price (Follows TRUE Regime with Noise)
realized_price = np.zeros(total_days)
realized_price[0] = 100
current_dev = 0
theta = 15.0   # Speed of reversion
sigma = 0.30   # Volatility

for t in range(1, total_days):
    # Calculate log-drift step from the TRUE regime curve
    prev_true = true_regime_price[t-1]
    curr_true = true_regime_price[t]
    drift_step = np.log(curr_true / prev_true)
    
    # OU Noise Process
    dW = np.random.normal(0, np.sqrt(dt))
    current_dev = current_dev + theta * (0 - current_dev) * dt + sigma * dW
    
    # Realized price tracks the True Regime + Noise
    realized_price[t] = true_regime_price[t] * np.exp(current_dev)

# D. Strategy Logic
# Signal is based on TRADER'S MODEL (Purple).
deviation = model_price - realized_price
signal = np.sign(deviation) 
shifted_signal = np.roll(signal, 1)
shifted_signal[0] = 0

# Returns
asset_returns = np.diff(realized_price) / realized_price[:-1]
asset_returns = np.insert(asset_returns, 0, 0)

# Leverage increased to 5.0 to fill the 0-600 scale before the crash
leverage = 5.0
strategy_returns = shifted_signal * asset_returns * leverage
wealth_path = 100 * np.cumprod(1 + strategy_returns)

# --- 2. Helper: Figure Construction ---
def make_break_frame(step):
    
    # Slice data
    curr_dates = dates[:step+1]
    curr_model = model_price[:step+1]
    curr_wealth = wealth_path[:step+1]
    curr_real = realized_price[:step+1]
    
    # Gold line logic: hidden in year 1, visible in year 2
    curr_regime = true_regime_price[:step+1].copy()
    if step < days_per_year:
        curr_regime[:] = np.nan 
    else:
        curr_regime[:days_per_year] = np.nan

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Model (Purple) vs Reality (Gold)", "Strategy Wealth (P&L)"),
        horizontal_spacing=0.1
    )

    # --- LEFT PLOT ---
    
    # 1. Trader's Model
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_model, mode='lines',
        line=dict(color='#d400ff', width=2, dash='dash'),
    ), row=1, col=1)

    # 2. True Regime (Gold)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_regime, mode='lines',
        line=dict(color='#FFD700', width=2, dash='dash'),
    ), row=1, col=1)

    # 3. Realized Price (Cyan)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_real, mode='lines',
        line=dict(color='#00ffff', width=1.5),
    ), row=1, col=1)
    
    # 4. Vertical Line
    if step >= days_per_year:
        fig.add_vline(
            x=dates[days_per_year].timestamp() * 1000,
            line_width=1, line_dash="dot", line_color="#e0e0e0", opacity=0.5,
            row=1, col=1
        )
        fig.add_annotation(
            x=dates[days_per_year], y=130,
            text="Structural Break", showarrow=False,
            font=dict(color="#FFD700", size=10),
            row=1, col=1
        )

    # --- RIGHT PLOT ---
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_wealth, mode='lines',
        line=dict(color='#00ffff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)',
    ), row=1, col=2)

    # --- Styling ---
    off_white = "#e0e0e0"
    
    # Axes Ranges
    fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
    fig.update_yaxes(range=[50, 140], title_text="Price", row=1, col=1)
    
    fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
    # Target range 0 to 600
    fig.update_yaxes(range=[0, 2000], title_text="Wealth Index", row=1, col=2)
    
    axis_config = dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color=off_white), title_font=dict(color=off_white)
    )
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, width=1000,
        title_text="Trading Mean Reversion: The Structural Break Trap",
        font=dict(color=off_white),
        showlegend=False, # Legend Removed
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    return fig

# --- 3. Animation Build ---
frames = [
    go.Frame(data=make_break_frame(k).data, name=str(k), layout=make_break_frame(k).layout)
    for k in range(0, total_days, 4)
]

fig = make_break_frame(0)
fig.frames = frames

# Play Button
play_darkblue = "#143c99"
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }],
        'font': {'color': play_darkblue, 'size': 16} 
    }]
)

fig.show()


# This is why modeling isn't necessarily purely quantitative, we have discretionary components
# - When do I respecify my model?
# - When do I reparameterize my model?
# - What do I do if there's variation I can't account for within my model?
# 
# In fact, quantitative research is about the scientific method, and qualitative economic interpretations to market ineffeciencies ($\alpha$)


# ---


# #### 3.) 🎯 Our Jobs as Traders and Investment Professionals
# 
# Theory gives us different structures (models and suggested methodology for parameterization) to produce an expectation - it does not mean it was or is correct.
# 
# We may suffer from model mispecification or model misparameterization, these are unique risks to our decision making process outside of the risks inherit to an investment


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation ---
np.random.seed(42)
days_per_year = 252
total_days = days_per_year * 2  # 2025 and 2026
dates = pd.date_range(start='2025-01-01', periods=total_days, freq='B')

# --- Market Physics ---
dt = 1/252
model_drift_rate = 0.05
# Naive Model (Purple): Continues up forever
naive_model_price = 100 * np.exp(model_drift_rate * np.linspace(0, 2, total_days))

# True Regime (Gold): Structural Break in 2026
true_regime_price = naive_model_price.copy()
start_price_2026 = true_regime_price[days_per_year]
# -30% drift (Crash)
year_2_drift = -0.30
t_year2 = np.linspace(0, 1, days_per_year)
true_regime_price[days_per_year:] = start_price_2026 * np.exp(year_2_drift * t_year2)

# Realized Price (OU Process)
realized_price = np.zeros(total_days)
realized_price[0] = 100
current_dev = 0
theta = 15.0
sigma = 0.30

for t in range(1, total_days):
    prev_true = true_regime_price[t-1]
    curr_true = true_regime_price[t]
    dW = np.random.normal(0, np.sqrt(dt))
    current_dev = current_dev + theta * (0 - current_dev) * dt + sigma * dW
    realized_price[t] = true_regime_price[t] * np.exp(current_dev)

# --- Returns & Strategies ---
asset_returns = np.diff(realized_price) / realized_price[:-1]
asset_returns = np.insert(asset_returns, 0, 0)
leverage = 5.0

# Strategy 1: Naive (Purple Model)
naive_deviation = naive_model_price - realized_price
naive_signal = np.sign(naive_deviation)
naive_signal = np.roll(naive_signal, 1); naive_signal[0] = 0
naive_wealth = 100 * np.cumprod(1 + naive_signal * asset_returns * leverage)

# Strategy 2: Pro (Adjusts Model)
adjustment_lag = 50 
adjustment_day = days_per_year + adjustment_lag

pro_model_price = np.zeros(total_days)
# Phase 1: Uses Purple Model
pro_model_price[:adjustment_day] = naive_model_price[:adjustment_day]
# Phase 2: Uses Gold Model
pro_model_price[adjustment_day:] = true_regime_price[adjustment_day:]

pro_deviation = pro_model_price - realized_price
pro_signal = np.sign(pro_deviation)
pro_signal = np.roll(pro_signal, 1); pro_signal[0] = 0
pro_wealth = 100 * np.cumprod(1 + pro_signal * asset_returns * leverage)

# --- 2. Figure Construction ---
def make_comparison_frame(step):
    
    curr_dates = dates[:step+1]
    curr_real = realized_price[:step+1]
    
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        subplot_titles=(
            "Naive Model (Ignores Break)", "Naive Wealth (Crashes)",
            "Pro Model (Adjusts to Break)", "Pro Wealth (Recovers)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # --- ROW 1: NAIVE TRADER ---
    
    # Trace 0: Naive Model (Purple)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=naive_model_price[:step+1], mode='lines',
        line=dict(color='#d400ff', width=2, dash='dash'), name="Naive Model"
    ), row=1, col=1)
    
    # Trace 1: Realized Price
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_real, mode='lines',
        line=dict(color='#00ffff', width=1.5), showlegend=False
    ), row=1, col=1)

    # Trace 2: Naive Wealth (Red)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=naive_wealth[:step+1], mode='lines',
        line=dict(color='#ff4d4d', width=2), 
        fill='tozeroy', fillcolor='rgba(255, 77, 77, 0.1)', name="Naive Wealth"
    ), row=1, col=2)

    # --- ROW 2: PRO TRADER ---
    
    # Trace 3: Pro Model Part A (Purple - Pre-Adjustment)
    limit_idx = min(step+1, adjustment_day)
    fig.add_trace(go.Scatter(
        x=dates[:limit_idx], y=naive_model_price[:limit_idx], mode='lines',
        line=dict(color='#d400ff', width=2, dash='dash'), showlegend=False
    ), row=2, col=1)
    
    # Trace 4: Pro Model Part B (Gold - Post-Adjustment)
    # IMPORTANT: This trace MUST exist in every frame, even if empty
    x_gold, y_gold = [None], [None]
    if step > adjustment_day:
        x_gold = dates[adjustment_day:step+1]
        y_gold = true_regime_price[adjustment_day:step+1]
        
    fig.add_trace(go.Scatter(
        x=x_gold, y=y_gold, mode='lines',
        line=dict(color='#FFD700', width=2, dash='dash'), name="Adjusted Model"
    ), row=2, col=1)

    # Trace 5: Realized Price (Bottom Row)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=curr_real, mode='lines',
        line=dict(color='#00ffff', width=1.5), showlegend=False
    ), row=2, col=1)

    # Trace 6: Pro Wealth (Green)
    fig.add_trace(go.Scatter(
        x=curr_dates, y=pro_wealth[:step+1], mode='lines',
        line=dict(color='#00ff00', width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)', name="Pro Wealth"
    ), row=2, col=2)

    # --- ANNOTATIONS ---
    
    # 1. Structural Break (Year 2 Start) - On ALL 4 Charts
    if step >= days_per_year:
        for r in [1, 2]:
            for c in [1, 2]: # Now loops over columns 1 and 2
                fig.add_vline(
                    x=dates[days_per_year].timestamp() * 1000, 
                    line_width=1, line_dash="dot", line_color="#e0e0e0", opacity=0.5, 
                    row=r, col=c
                )
        # Text only on left chart
        fig.add_annotation(x=dates[days_per_year], y=135, text="Break", showarrow=False, font=dict(color="white", size=10), row=1, col=1)

    # 2. Adjustment Day (Pro Charts Only)
    if step >= adjustment_day:
        # Bottom Left: Solid Line with Text
        fig.add_vline(x=dates[adjustment_day].timestamp() * 1000, line_width=2, line_color="#00ff00", row=2, col=1)
        
        # Shifted Arrow/Text to the right (ax=60)
        fig.add_annotation(
            x=dates[adjustment_day + 3], y=60, 
            text="Adjustment", 
            showarrow=True, 
            arrowhead=1, 
            ax=50, ay=-30, # Increased ax to shift right
            font=dict(color="#00ff00", size=11, style='italic'), 
            arrowcolor="#00ff00", 
            row=2, col=1
        )
        
        # Bottom Right (Wealth): Solid Line, No Text
        fig.add_vline(
            x=dates[adjustment_day].timestamp() * 1000, 
            line_width=1, 
            line_color="#00ff00", # Made solid to match request/visibility
            opacity=0.6,
            row=2, col=2
        )

    # --- STYLING ---
    off_white = "#e0e0e0"
    axis_config = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=off_white))
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)
    
    # Ranges
    for r in [1, 2]:
        fig.update_xaxes(range=[dates[0], dates[-1]], row=r, col=1)
        fig.update_xaxes(range=[dates[0], dates[-1]], row=r, col=2)
        fig.update_yaxes(range=[40, 150], title_text="Price", row=r, col=1)

    fig.update_yaxes(range=[0, 2000], title_text="Naive P&L", row=1, col=2)
    fig.update_yaxes(range=[0, 3000], title_text="Pro P&L", row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700, width=1100,
        title_text="Regime Change: Naive vs. Adaptive Mean Reversion",
        font=dict(color=off_white),
        showlegend=False,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    return fig

# --- 3. Animation Build ---
frames = [
    go.Frame(data=make_comparison_frame(k).data, layout=make_comparison_frame(k).layout, name=str(k))
    for k in range(0, total_days, 4)
]

fig = make_comparison_frame(0)
fig.frames = frames

play_darkblue = "#143c99"
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1, 'xanchor': 'center',
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }],
        'font': {'color': play_darkblue, 'size': 16} 
    }]
)

fig.show()


# Once you understand this, and the different types of risk you can expose yourself to, your job is to estimate and optimize expected return.
# 
# This idea leads the discussion on the following topics
# 
# - What risks are we being exposed to?
# - What risks are hedgable vs unhedgable?
# - What are different strategies you can trade ($\alpha$, $\beta$, smart $\beta$)?
# - What risks are inherit to a specific strategy?
# - When do I respecify or reparameterize a model?
# 
# These are topics I would like to discuss in the future if there is interest.
# 
# If you can't understand these structures in the first place, you'll have no ability to implement them in practice when everything is moving.
# 
# Master your quantitative skills.


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - If past-performance isn't necessarily indicative of future performance, why do we use historic data to calibrate our models?
# - In reality, nobody has a crystal ball, the efficacy of model specification and parameterization dictates the efficacy of your trading strategy and overall P/L
# - Everyone, from retail traders to institutions, profits from the same statistical mechanisms (expected value, or edge)
# - However, the academic concept in reality is looser than the classroom suggests due to lack of convergence via the Law of Large Numbers (LLN) from non-stationarity
# - Effectively, we are modeling the chaos of the composite decisions of agents as randomness, thus distributions are not necessarily constant or stable
# - Should we develop reasonable models via specification and parameterization we may find an edge and profit from it
# - What separates those that *know what they're doing* from those that don't is the ability to *adapt* and change their model over time
# - Model re-specification and re-parameterization is not trivial, there is often a lag where traders lose money before adapting
# - This non-trivial time variance is what makes trading *"hard"* and effectively a full-time job 
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Hawkes Processes
# - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
# - Market-Making Models and Simulation (Stoikov-Avellaneda)
# - Projects that Made me a Quant
# - My First Year as a Quant
# - Kalman Filter for Quant Finance
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Girsanov's Change of Measure
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

