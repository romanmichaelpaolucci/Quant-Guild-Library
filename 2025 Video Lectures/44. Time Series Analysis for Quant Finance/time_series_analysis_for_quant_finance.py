# ### 🕒 Time Series Analysis for Quant Finance
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
# - [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)
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
# #### 1.) 🕒 Time Series Analysis
# 
# - What is a Time Series?
# 
# - Trend, Seasonality, Shock
# 
# - Filtering, Smoothing, Forecasting
# 
# #### 2.) 📉 Why Forecasts are Wrong
# 
# - Assumptions, Unit Roots, and Stationarity
# 
# - *Example:* Temparature, Bench Press, Stock Prices (guess yesterdays price and mean level)
# 
# - *Example:* 2008 Financial Crisis, The Big Short (information as a predictor)
# 
# - How to Think About Forecasts
# 
# - *Example:* Generating P/L with Forecasts, Dice Markets
# 
# 
# #### 3.) 🎯 Trading and Investing with Time Series Models
# 
# - Qualitative vs. Quantitative Approaches
# 
# - Qualitative vs. Quantitative Edge
# 
# - Alternative Data vs. Time Series Models (more merit in certain types of problems: illiquid instruments, HFT, market making, etc...)
# 
# 
# #### 4.) 💭 Closing Thoughts and Future Comments


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


# #### 1.) 🕒 Time Series Analysis
# 
# 
# ##### 💡 What is a Time Series?
# 
# A time series is a set of values of a certain quantity (typically) obtained at equal intervals between them
# 
# - This isn't always the case (an illiquid product may only trade once in a while yielding a new price)
# 
# - Piecewise linear interpolation is used to connect observations, we don't necessarily know what occurs between observations
# 
# - Statistics are elusive as these systems evolve dynamically over time (these values do not converge and are not constant)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate regular time series data (stock price)
np.random.seed(42)
n_points = 100
dates_regular = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]
price = 100
prices_regular = []
for _ in range(n_points):
    price *= (1 + np.random.normal(0.0002, 0.02))
    prices_regular.append(price)

# Generate irregular time series data (illiquid instrument)
n_irregular = 30
random_days = sorted(np.random.choice(range(n_points), n_irregular, replace=False))
dates_irregular = [datetime(2024,1,1) + timedelta(days=int(x)) for x in random_days]
price = 100
prices_irregular = []
for _ in range(n_irregular):
    price *= (1 + np.random.normal(0.0005, 0.03))
    prices_irregular.append(price)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Regular Time Series', 'Irregular Time Series')
)

# Add regular time series on left
fig.add_trace(
    go.Scatter(
        x=dates_regular,
        y=prices_regular,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'), # Neon cyan
        name='Regular Series'
    ),
    row=1, col=1
)

# Add irregular time series on right
fig.add_trace(
    go.Scatter(
        x=dates_irregular,
        y=prices_irregular,
        mode='markers+lines',
        line=dict(color='rgba(255, 20, 147, 1)'), # Neon pink
        marker=dict(size=8, color='rgba(255, 20, 147, 1)'),
        name='Irregular Series'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1200,
    showlegend=False,
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
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ##### 🛠️ Time Series can be Decomposed into Trend, Shock, and Seasonality
# 
# Oftentimes we are interested in potential underlying structure in a time series
# 
# - Trend indicates long-term underlying directional move (trajectory)
# 
# - Seasonality refers to patterns repeating at fixed intervals
# 
# - Shock or residuals represent random fluctiations or unexpected events in data


# Generate sample time series with trend, seasonality and shock
np.random.seed(42)
n_points = 200
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]

# Create components
trend = np.linspace(0, 20, n_points)  # Linear trend
seasonality = 5 * np.sin(np.linspace(0, 8*np.pi, n_points))  # Seasonal component
shock = np.random.normal(0, 2, n_points)  # Random shock

# Combine components
time_series = trend + seasonality + shock

# Create subplots
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Complete Time Series', 'Trend Component', 'Seasonal Component', 'Random Shock Component'),
    vertical_spacing=0.08,
    row_heights=[0.4, 0.2, 0.2, 0.2]
)

# Add complete time series
fig.add_trace(
    go.Scatter(
        x=dates,
        y=time_series,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)'),  # Neon purple
        name='Time Series'
    ),
    row=1, col=1
)

# Add trend component
fig.add_trace(
    go.Scatter(
        x=dates,
        y=trend,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)'),  # Neon green
        name='Trend'
    ),
    row=2, col=1
)

# Add seasonal component
fig.add_trace(
    go.Scatter(
        x=dates,
        y=seasonality,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),  # Neon cyan
        name='Seasonality'
    ),
    row=3, col=1
)

# Add shock component
fig.add_trace(
    go.Scatter(
        x=dates,
        y=shock,
        mode='lines',
        line=dict(color='rgba(255, 20, 147, 1)'),  # Neon pink
        name='Shock'
    ),
    row=4, col=1
)

# Update layout
fig.update_layout(
    height=1000,
    width=1200,
    showlegend=False,
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
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# A time series may have irregular seasonal components (think more like a regime) due to other time series that have fixed seasonal components
# 
# **Example:** *Economic Expansion and Contraction*
# 
# - Economic expansion and contraction occur during some sort of regular interval (seasonality)
# 
# - Trend and seasonal components of something like a stock price may change subject to these regular intervals
# 
# - It's a difficult space to model since everything continues to change over time and a fixed seasonal interval doesn't necessarily have to continue
# 
# ###### ______________________________________________________________________________________________________________________________________


# ##### 📝 Tasks in Time Series Analysis
# 
#  - *Filtering:* Estimating current state using past and present data to reduce noise
#  
#  - *Smoothing:* Estimating historical states using past, present and future data
#  
#  - *Forecasting:* Predicting future values based on historical patterns (terrible characterization, there is no *prediction* it's an expected value...)


# Generate sample time series data
np.random.seed(42)
n_points = 100
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]
price = 100
prices = []
for _ in range(n_points):
    price *= (1 + np.random.normal(0.0002, 0.02))
    prices.append(price)

# Create figure with subplots
fig = make_subplots(rows=3, cols=1, subplot_titles=('Filtering', 'Smoothing', 'Forecasting'))

# Filtering subplot
# Original Data
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Original Data'
    ),
    row=1, col=1
)

# Filtered Data
window = 10
filtered = np.convolve(prices, np.ones(window)/window, mode='valid')
filtered_dates = dates[window-1:]

# Calculate confidence bands for filtered data
filtered_std = np.std(filtered)
filtered_upper = filtered + 2*filtered_std
filtered_lower = filtered - 2*filtered_std

fig.add_trace(
    go.Scatter(
        x=filtered_dates,
        y=filtered,
        mode='lines',
        line=dict(color='rgba(255, 20, 147, 1)'),
        name='Filtered (Moving Avg)'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=filtered_dates,
        y=filtered_upper,
        mode='lines',
        line=dict(color='rgba(255, 20, 147, 0.2)'),
        name='Filtered Confidence Band',
        showlegend=False
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=filtered_dates,
        y=filtered_lower,
        mode='lines',
        line=dict(color='rgba(255, 20, 147, 0.2)'),
        fill='tonexty',
        showlegend=False
    ),
    row=1, col=1
)

# Smoothing subplot
# Original Data
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Original Data',
        showlegend=False
    ),
    row=2, col=1
)

# Smoothed Data
alpha = 0.1
smoothed = [prices[0]]
for i in range(1, len(prices)):
    smoothed.append(alpha * prices[i] + (1-alpha) * smoothed[i-1])

# Calculate confidence bands for smoothed data
smoothed_std = np.std(smoothed)
smoothed_upper = np.array(smoothed) + 2*smoothed_std
smoothed_lower = np.array(smoothed) - 2*smoothed_std

fig.add_trace(
    go.Scatter(
        x=dates,
        y=smoothed,
        mode='lines',
        line=dict(color='rgba(147, 0, 255, 1)'),
        name='Smoothed (Exp)'
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=dates,
        y=smoothed_upper,
        mode='lines',
        line=dict(color='rgba(147, 0, 255, 0.2)'),
        name='Smoothed Confidence Band',
        showlegend=False
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=dates,
        y=smoothed_lower,
        mode='lines',
        line=dict(color='rgba(147, 0, 255, 0.2)'),
        fill='tonexty',
        showlegend=False
    ),
    row=2, col=1
)

# Forecasting subplot
# Original Data
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Original Data',
        showlegend=False
    ),
    row=3, col=1
)

# Forecast
forecast_days = 20
last_price = prices[-1]
forecast = [last_price]
forecast_std = []
current_std = np.std(prices[-window:])

for _ in range(forecast_days-1):
    forecast.append(forecast[-1] * (1 + np.random.normal(0.0002, 0.01)))
    current_std *= 1.1  # Increasing uncertainty
    forecast_std.append(current_std)

forecast_dates = [dates[-1] + timedelta(days=int(x)) for x in range(forecast_days)]

# Calculate confidence bands for forecast
forecast_upper = np.array(forecast) + 2*np.array([current_std] * forecast_days)
forecast_lower = np.array(forecast) - 2*np.array([current_std] * forecast_days)

fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 1)'),
        name='Forecast'
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_upper,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.2)'),
        name='Forecast Confidence Band',
        showlegend=False
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_lower,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.2)'),
        fill='tonexty',
        showlegend=False
    ),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1200,
    height=1200,
    title='Time Series Analysis: Filtering, Smoothing and Forecasting',
    showlegend=True,
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
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ---


# #### 2.) 📉 Why Forecasts are Wrong
# 
# 
# First and foremost - there is no such thing as a *prediction* without a crystal ball
# 
# We are dealing with an *expectation* which is a far better characterization of the space, if things evolved *as expected* then this is what the future would be *at best*, at worse we are violently wrong
# 
# However, we can barely measure what a *good* expectation would actually be
# 
# We will look at several examples herein why this is the case, and we will also look at how we can make money even when our models are *incorrect*
# 
# ##### 🚩 Assumptions, Unit Roots, and Stationarity
# 
# Tests for stationarity are largely nonsense, its very difficult, unit roots imply nonstationarity... not having a unit root doesnt mean its stationary...
# 
# **Remark:** <u>Some assumptions are more correct than others</u>
# 
# - **No arbitrage $\iff$ Risk-Neutral Measure**, for example, is not a crazy assumption, model prices based on this assumption are reasonable
# 
# - **Constant Volatility**, for example, is a crazy assumption, but for something like the Black-Scholes Model is actually useful (implied volatility) 
# 
# - **Stationarity**, for example, is a crazy assumption and quickly and effortlessly invalidates a model in practice
# 
# 
# Now, the first thing your professor will do in a time series analysis class is assume *stationarity*
# 
# A violently wrong assumption that validates the theoretical world and ensures analysis is *"correct"*
# 
# Very little is done to discuss the imnplications of this violation in practice, it largely turns into a research problem. . .


# Generate sample time series data with an earnings jump
np.random.seed(42)
n_points = 100
dates = [datetime(2024,1,1) + timedelta(days=int(x)) for x in range(n_points)]
price = 100
prices = []

# Add an earnings event around day 60
earnings_day = 60
jump_magnitude = 0.15  # 15% jump

for i in range(n_points):
    if i == earnings_day:
        # Random direction of the earnings jump (up or down)
        direction = np.random.choice([-1, 1])
        price *= (1 + direction * jump_magnitude)
    else:
        price *= (1 + np.random.normal(0.0002, 0.01))
    prices.append(price)

# Create figure for earnings example
fig = make_subplots(rows=1, cols=1, subplot_titles=('Forecast Failure at Earnings Event',))

# Original Data up to earnings
fig.add_trace(
    go.Scatter(
        x=dates[:earnings_day],
        y=prices[:earnings_day],
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Historical Data'
    )
)

# Data after earnings
fig.add_trace(
    go.Scatter(
        x=dates[earnings_day:],
        y=prices[earnings_day:],
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Post-Earnings Data'
    )
)

# Generate pre-earnings forecast
window = 10
last_price = prices[earnings_day-1]
forecast = [last_price]
forecast_days = 20
current_std = np.std(prices[earnings_day-window:earnings_day])

for _ in range(forecast_days-1):
    forecast.append(forecast[-1] * (1 + np.random.normal(0.0002, 0.01)))
    current_std *= 1.1

forecast_dates = [dates[earnings_day-1] + timedelta(days=int(x)) for x in range(forecast_days)]

# Calculate confidence bands
forecast_upper = np.array(forecast) + 2*np.array([current_std] * forecast_days)
forecast_lower = np.array(forecast) - 2*np.array([current_std] * forecast_days)

# Add forecast
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 1)'),
        name='Pre-Earnings Forecast'
    )
)

# Add confidence bands
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_upper,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.2)'),
        name='Forecast Confidence Band',
        showlegend=False
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_lower,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.2)'),
        fill='tonexty',
        showlegend=False
    )
)
# Add vertical line for earnings
# Use add_shape instead of add_vline to avoid type error with datetime
fig.add_shape(
    type="line",
    x0=dates[earnings_day],
    x1=dates[earnings_day], 
    y0=0,
    y1=1,
    yref="paper",
    line=dict(
        color="red",
        width=2,
        dash="dash"
    )
)

# Add earnings annotation
fig.add_annotation(
    x=dates[earnings_day],
    y=1,
    yref="paper",
    text="Earnings Event",
    showarrow=False,
    yshift=10
)

# Update layout
fig.update_layout(
    width=1200,
    height=600,
    title='Time Series Forecast Failure: Earnings Event Example',
    showlegend=True,
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
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ##### 🔎 Example: Temperature, Bench Press


# Generate sample time series data
np.random.seed(42)
n_points = 365 * 3  # 3 years of daily data

# Temperature data with seasonality
dates_temp = [datetime(2021,1,1) + timedelta(days=int(x)) for x in range(n_points)]
temp_base = 60
temps = []
for i in range(n_points):
    # Annual seasonal cycle with some noise
    seasonal = 30 * np.sin(2 * np.pi * i / 365)  # +/- 30 degrees seasonal variation
    noise = np.random.normal(0, 5)  # Daily temperature variation
    temps.append(temp_base + seasonal + noise)

# Bench press data with more variability
n_points_bp = 50
dates_bp = [datetime(2024,1,1) + timedelta(days=int(x*7)) for x in range(n_points_bp)]  # Weekly data
bp_weights = []
weight = 335
for i in range(n_points_bp-1):
    # Add more realistic variability in progress
    if np.random.random() < 0.2:  # 20% chance of a bad week
        weight -= np.random.uniform(5, 10)
    else:
        weight += np.random.normal(1, 2)  # More variable progress
    
    # Add fatigue cycles
    cycle = 5 * np.sin(2 * np.pi * i / 8)  # 8-week fatigue cycle
    weight += cycle
    
    weight = min(355, max(315, weight))  # Keep between 315 and 355
    bp_weights.append(weight)

# Add the dramatic drop
bp_weights.append(45)  # Drop to just the bar

# Create subplots
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Temperature with Seasonality', 'Bench Press Progress'),
                    specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

# Temperature plot with forecast
forecast_days = 60
last_temp = temps[-1]
temp_forecast = [last_temp]
current_std = np.std(temps[-30:])
forecast_dates_temp = [dates_temp[-1] + timedelta(days=int(x)) for x in range(1, forecast_days+1)]

# Simple forecast that doesn't account for seasonality
for _ in range(forecast_days-1):
    temp_forecast.append(temp_forecast[-1] + np.random.normal(0, 2))

# Add temperature traces
fig.add_trace(
    go.Scatter(
        x=dates_temp,
        y=temps,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Historical Temperature'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates_temp,
        y=temp_forecast,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 1)'),
        name='Temperature Forecast'
    ),
    row=1, col=1
)

# Bench press plot with forecast
bp_forecast_weeks = 10
last_bp = bp_weights[-2]  # Use second to last point before drop
bp_forecast = [last_bp]
forecast_dates_bp = [dates_bp[-2] + timedelta(days=int(x*7)) for x in range(1, bp_forecast_weeks+1)]

# Generate forecast (which will fail to predict the drop)
for _ in range(bp_forecast_weeks-1):
    bp_forecast.append(bp_forecast[-1] + np.random.normal(0.5, 1))

# Add bench press traces for heavy weights (primary y-axis)
fig.add_trace(
    go.Scatter(
        x=dates_bp[:-1],
        y=bp_weights[:-1],
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)'),
        name='Heavy Weight Training'
    ),
    row=2, col=1,
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates_bp,
        y=bp_forecast,
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 1)'),
        name='Projected Progress'
    ),
    row=2, col=1,
    secondary_y=False
)

# Recovery path on secondary y-axis (light weights)
recovery_weeks = 12
recovery_dates = [dates_bp[-1] + timedelta(days=int(x*7)) for x in range(recovery_weeks)]
recovery_weights = [45]  # Start with bar weight

# More realistic recovery pattern
for i in range(recovery_weeks-1):
    if i < 4:  # First month - careful progression
        next_weight = recovery_weights[-1] * 1.1
    else:  # Faster progression once form is restored
        next_weight = recovery_weights[-1] * 1.15
        
    # Add some variability
    next_weight *= (1 + np.random.normal(0, 0.05))
    next_weight = min(next_weight, 225)  # Cap at 225
    recovery_weights.append(next_weight)

# Add injury point and recovery path on secondary y-axis
fig.add_trace(
    go.Scatter(
        x=[dates_bp[-1]],
        y=[45],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Injury Event',
        showlegend=True
    ),
    row=2, col=1,
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=recovery_dates,
        y=recovery_weights,
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Recovery Path'
    ),
    row=2, col=1,
    secondary_y=True
)

# Add annotation for the injury
fig.add_annotation(
    x=dates_bp[-1],
    y=45,
    text="Unexpected Injury<br>Return to Bar Weight",
    showarrow=True,
    arrowhead=1,
    arrowcolor='red',
    arrowsize=1,
    arrowwidth=2,
    row=2, col=1
)

# Update layout
fig.update_layout(
    width=1200,
    height=800,
    title='Time Series Examples: Seasonality and Structural Breaks',
    showlegend=True,
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
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

# Update y-axis titles and ranges
fig.update_yaxes(
    title_text="Heavy Weight Training (lbs)", 
    secondary_y=False,
    range=[300, 375],
    row=2, col=1
)
fig.update_yaxes(
    title_text="Recovery Weight (lbs)",
    secondary_y=True, 
    range=[0, 250],
    row=2, col=1
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### 🎯 How to Think About Forecasts
# 
# There is one of two cases based on your provided model forecast
# 
# - On a *good day* you have correctly produced an expected level
# 
# - On a *bad day* you have incorrectly produced an expected level
# 
# There is no *prediction*, we are using an *expectation* as our best guess in a *foreward looking* sense
# 
# I have discussed implied probability previously, and the idea of the expectation being the best guess in well-defined randomness (trading withe edge)
# 
# There is no convergence, *confidence* is relatively arbitrary, outcomes are time dependent random variables - this is the best we got
# 
# External information and qualitative insights (or modeling that qualitative information in a quantitative way) can be far more effective in modeling regime changes and significant deviations


import pandas as pd

# Generate sample dice roll data
np.random.seed(42)
n_rolls = 100
dice_rolls = np.random.randint(1, 7, n_rolls)
dates = pd.date_range(start='2023-01-01', periods=n_rolls)

# Create figure
fig = go.Figure()

# Plot dice rolls
fig.add_trace(
    go.Scatter(
        x=dates,
        y=dice_rolls,
        mode='lines+markers',
        line=dict(color='#FF69B4'),
        name='Dice Roll Outcomes'
    )
)

# Add expected value line
fig.add_hline(y=3.5, line_dash="dash", line_color="lime")

# Update layout
fig.update_layout(
    width=1200,
    height=600,
    title='Time Series of Dice Roll Outcomes',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title="Dice Roll Value"
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
    range=[0.5, 6.5]
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### 🔎 Example: Generating P/L from Forecasts, Dice Markets
# 
# ##### 🎲 Time Invariant Distribution
# 
# Consider the following dice market with a well-defined and well-behaved probability mass function (i.e. the dice doesn't change)
# 
# <u>Consider the following game</u>
# 
# - A market-maker quotes a Bid-Ask/Offer for the outcome of a dice and we can go long at the Ask/Offer or Short at the Bid
# 
# - EV for the Dice is well defined $E[X] = \frac{1+2+3+4+5+6}{6} = 3.5$
# 
# - If we go long below 3.5 and short above 3.5 we can generate P/L even though the outcome is random
# 
# Essentially, we can profit off of this dice market when there is edge on either side of the trade and accumulate the positive expected value over time


n_steps = 100
dates = pd.date_range(start='2023-01-01', periods=n_steps)

# Market making parameters with tighter spread around fair value
bid = 3.2  # Slightly below fair value
ask = 3.8  # Slightly above fair value
lambda_rate = 0.3  # Trading activity rate
fair_value = 3.5

# Initialize arrays
pnl = np.zeros(n_steps)
positions = np.zeros(n_steps)
dice_outcomes = np.zeros(n_steps)
cumulative_pnl = np.zeros(n_steps)

# Create figure for animation
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Dice Outcomes & Market Making Levels',
                                  'Trading Positions',
                                  'Cumulative P&L'),
                    vertical_spacing=0.1)

# Initialize traces
dice_trace = go.Scatter(x=[dates[0]], y=[0], mode='lines+markers',
                       line=dict(color='#FF69B4'), name='Dice Outcomes')
position_trace = go.Bar(x=[dates[0]], y=[0], name='Position',
                       marker_color='#1E90FF')
pnl_trace = go.Scatter(x=[dates[0]], y=[0], mode='lines',
                       line=dict(color='lime'), name='Cumulative P&L')

# Add initial traces
fig.add_trace(dice_trace, row=1, col=1)
fig.add_hline(y=bid, line_dash="dash", line_color="red", name='Bid', row=1, col=1)
fig.add_hline(y=ask, line_dash="dash", line_color="green", name='Ask', row=1, col=1)
fig.add_hline(y=fair_value, line_dash="solid", line_color="yellow", name='Fair Value', row=1, col=1)
fig.add_trace(position_trace, row=2, col=1)
fig.add_trace(pnl_trace, row=3, col=1)

# Update layout
fig.update_layout(
    height=900,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Create frames for animation
frames = []
for i in range(n_steps):
    # Generate trading activity
    buyers = np.random.poisson(lambda_rate)
    sellers = np.random.poisson(lambda_rate)
    dice_outcomes[i] = np.random.randint(1, 7)
    
    # Calculate P&L
    # When we buy (sellers hitting our bid), we profit if outcome < bid
    # When we sell (buyers lifting our ask), we profit if outcome > ask
    buy_pnl = sellers * (bid - dice_outcomes[i])  # We buy at bid
    sell_pnl = buyers * (dice_outcomes[i] - ask)  # We sell at ask
    spread_pnl = (buyers + sellers) * (ask - bid)  # Full spread earned on each round trip
    pnl[i] = buy_pnl + sell_pnl + spread_pnl
    
    # Update positions and cumulative P&L
    positions[i] = sellers - buyers  # Positive when we buy, negative when we sell
    if i > 0:
        cumulative_pnl[i] = cumulative_pnl[i-1] + pnl[i]
    else:
        cumulative_pnl[i] = pnl[i]
        
    frame = go.Frame(
        data=[
            go.Scatter(x=dates[:i+1], y=dice_outcomes[:i+1], mode='lines+markers',
                      line=dict(color='#FF69B4')),
            go.Bar(x=dates[:i+1], y=positions[:i+1],
                  marker_color='#1E90FF'),
            go.Scatter(x=dates[:i+1], y=cumulative_pnl[:i+1], mode='lines',
                      line=dict(color='lime'))
        ]
    )
    frames.append(frame)

# Add frames to figure
fig.frames = frames

# Add animation buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}])
            ]
        )
    ]
)

# Update axes formatting
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

# Set y-axis range for dice outcomes
fig.update_yaxes(range=[0.5, 6.5], row=1, col=1)

fig.show()


# ##### 🃏 Time Variant Distribution
# 
# I often say in my videos we are *wrong* about our forecast, this will be the case *all of the time* but if we are correct *on average* we can generate P/L
# 
# In the previous game we defined the expectation as $3.5$ but what if it (the dice weight, the population distribution) changed *randomly* over time
# 
# A fixed $3.5$ wouldn't be sufficient to trade with an edge, we need a dynamic model - here I use a Kalman Filter to dictate this level and generate P/L
# 
# I will do a video in the future on the Kalman Filter (if there is interest) then another implementing it in Python with Interactive Brokers on live pricing 
# 
# 
# <u>Consider the following game</u>
# 
# MODIFY THE GAME BELOW AND SUGGEST WE DON"T KNOW THE DYNAMICS GIVEN BY THE POPULATION DISTRIBUTION
# 
# - A market-maker quotes a Bid-Ask/Offer for the outcome of a dice and we can go long at the Ask/Offer or Short at the Bid
# 
# - EV for the Dice is well defined $E[X] = \frac{1+2+3+4+5+6}{6} = 3.5$
# 
# - If we go long below 3.5 and short above 3.5 we can generate P/L even though the outcome is random


n_steps = 100
dates = pd.date_range(start='2023-01-01', periods=n_steps)

# Market making parameters
spread = 0.7  # Half spread around moving average
lambda_rate = 0.3  # Trading activity rate
window_size = 10  # Moving average window

# Initialize arrays
pnl = np.zeros(n_steps)
positions = np.zeros(n_steps)
dice_outcomes = np.zeros(n_steps)
cumulative_pnl = np.zeros(n_steps)
moving_avg = np.zeros(n_steps)

# Create figure for animation
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Dice Outcomes & Market Making Levels',
                                  'Trading Positions',
                                  'Cumulative P&L'),
                    vertical_spacing=0.1)

# Initialize traces
dice_trace = go.Scatter(x=[dates[0]], y=[0], mode='lines+markers',
                       line=dict(color='#FF69B4'), name='Dice Outcomes')
ma_trace = go.Scatter(x=[dates[0]], y=[0], mode='lines',
                     line=dict(color='yellow'), name='Moving Average')
position_trace = go.Bar(x=[dates[0]], y=[0], name='Position',
                       marker_color='#1E90FF')
pnl_trace = go.Scatter(x=[dates[0]], y=[0], mode='lines',
                       line=dict(color='lime'), name='Cumulative P&L')

# Add initial traces
fig.add_trace(dice_trace, row=1, col=1)
fig.add_trace(ma_trace, row=1, col=1)
fig.add_trace(position_trace, row=2, col=1)
fig.add_trace(pnl_trace, row=3, col=1)

# Update layout
fig.update_layout(
    height=900,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Create frames for animation
frames = []
for i in range(n_steps):
    # Generate dice outcome with changing weights
    weights = np.random.normal(1, 0.2, 6)  # Random weights that sum to 6
    weights = weights / weights.sum() * 6
    dice_outcomes[i] = np.random.choice(range(1,7), p=weights/6)
    
    # Calculate moving average
    if i < window_size:
        moving_avg[i] = np.mean(dice_outcomes[:i+1])
    else:
        moving_avg[i] = np.mean(dice_outcomes[i-window_size+1:i+1])
    
    # Set bid/ask around moving average
    bid = moving_avg[i] - spread
    ask = moving_avg[i] + spread
    
    # Generate trading activity
    buyers = np.random.poisson(lambda_rate)
    sellers = np.random.poisson(lambda_rate)
    
    # Calculate P&L
    buy_pnl = sellers * (bid - dice_outcomes[i])
    sell_pnl = buyers * (dice_outcomes[i] - ask)
    spread_pnl = (buyers + sellers) * (ask - bid)
    pnl[i] = buy_pnl + sell_pnl + spread_pnl
    
    # Update positions and cumulative P&L
    positions[i] = sellers - buyers
    if i > 0:
        cumulative_pnl[i] = cumulative_pnl[i-1] + pnl[i]
    else:
        cumulative_pnl[i] = pnl[i]
        
    frame = go.Frame(
        data=[
            go.Scatter(x=dates[:i+1], y=dice_outcomes[:i+1], mode='lines+markers',
                      line=dict(color='#FF69B4')),
            go.Scatter(x=dates[:i+1], y=moving_avg[:i+1], mode='lines',
                      line=dict(color='yellow')),
            go.Bar(x=dates[:i+1], y=positions[:i+1],
                  marker_color='#1E90FF'),
            go.Scatter(x=dates[:i+1], y=cumulative_pnl[:i+1], mode='lines',
                      line=dict(color='lime'))
        ]
    )
    frames.append(frame)

# Add frames to figure
fig.frames = frames

# Add animation buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}])
            ]
        )
    ]
)

# Update axes formatting
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

# Set y-axis range for dice outcomes
fig.update_yaxes(range=[0.5, 6.5], row=1, col=1)

fig.show()


# ---


# #### 3.) 🎯 Trading and Investing with Time Series Models


# ##### Qualitative and Quantitative Approaches: Alternative Data
# 
# Trading and Investing is about managing *risk exposure(s)* to generate P/L by accumulating positive expected value over time
# 
# A significant portion of this for traditional portfolio management is still qualitative
# 
# <u>For Example</u>
# 
# - Observing risks in the broader market (geopolitical, inflation, interest rates, . . .)
# 
# - Industry related risks (policy, product development, drug development, . . .)
# 
# - Firm specific risks (New CEO, firm trajectory, . . .)
# 
# We can measure Value at Risk (VaR), Expected Shortfall (C-VaR), and other measures of portfolio risk 
# 
# but in the same sense as the time series example above we are not blanket considering risks that we can hypothesis over qualitatively
# 
# It can be useful to use a *hybrid* approach to assess the likelihood of different types of *qualitative risks* in tandem with *quantitative modeling*
# 
# [Anecdotal Example with 2008 Crisis and Big Short, time series shows no risks but alt data did and was tradable]


# Generate sample CDS spread data for 2008 crisis
np.random.seed(42)
n_points = 250  # Trading days in a year
dates = [datetime(2008,1,1) + timedelta(days=int(x)) for x in range(n_points)]
cds_spread = 100  # Starting spread in bps
spreads = []

# Generate CDS spread data with upward trend before crisis
crisis_start = 180  # Around September 2008
for i in range(n_points):
    if i < crisis_start:
        # Gradually increasing spreads with growing volatility
        drift = 0.003 * (1 + i/crisis_start)  # Increasing drift
        vol = 0.02 * (1 + i/crisis_start)     # Increasing volatility
        cds_spread *= (1 + np.random.normal(drift, vol))
    else:
        if i == crisis_start:
            # Massive drop at default
            cds_spread *= 0.15  # 85% drop
        else:
            # High volatility after default
            cds_spread *= (1 + np.random.normal(0.001, 0.06))
    spreads.append(cds_spread)

# Create figure
fig = make_subplots(rows=1, cols=1, subplot_titles=('CDS Spread During 2008 Financial Crisis: Model Failure',))

# Plot historical data before crisis
fig.add_trace(
    go.Scatter(
        x=dates[:crisis_start],
        y=spreads[:crisis_start],
        mode='lines',
        line=dict(color='cyan'),
        name='Historical CDS Spreads'
    )
)

# Plot data after crisis
fig.add_trace(
    go.Scatter(
        x=dates[crisis_start:],
        y=spreads[crisis_start:],
        mode='lines',
        line=dict(color='cyan'),
        name='Post-Default Spreads'
    )
)

# Generate terrible forecast (linear extrapolation)
forecast_window = 40
forecast_start = crisis_start - 30
slope = (spreads[crisis_start-1] - spreads[forecast_start]) / (crisis_start - forecast_start)
forecast_dates = [dates[crisis_start-1] + timedelta(days=x) for x in range(forecast_window)]
forecast = [spreads[crisis_start-1] + slope * x for x in range(forecast_window)]

# Add forecast line
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        line=dict(color='lime', dash='dash'),
        name='Pre-Crisis Forecast (Linear)'
    )
)

# Add confidence bands (completely wrong)
confidence = 0.2 * np.array(forecast)
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast + confidence,
        mode='lines',
        line=dict(color='rgba(0,255,0,0.2)'),
        name='95% Confidence Band'
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast - confidence,
        mode='lines',
        line=dict(color='rgba(0,255,0,0.2)'),
        fill='tonexty',
        showlegend=False
    )
)

# Add Lehman default line
fig.add_shape(
    type="line",
    x0=dates[crisis_start],
    x1=dates[crisis_start],
    y0=0,
    y1=1,
    yref="paper",
    line=dict(
        color="red",
        width=2,
        dash="dash"
    )
)

# Add default annotation
fig.add_annotation(
    x=dates[crisis_start],
    y=1,
    yref="paper",
    text="Default Event",
    showarrow=False,
    yshift=10
)

# Update layout
fig.update_layout(
    width=1200,
    height=600,
    title='Time Series Forecast Failure Example: CDS Spreads During Crisis',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title="CDS Spread (bps)"
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


# ##### ⚔️ Alternative Data vs. Time Series Models
# 
# The efficacy of time series approaches varies based on the environment - they are *great* at some things. . .
# 
# **Effective Examples** (*Non-Exhaustive*)
# 
# - Pricing an Illiquid Product
# 
# - High-Frequency Trading
# 
# **Remark:** Alternative data, and modeling tangential outcomes with alternative data and information can be more viable than throwing a time series model at the problem
# 
# Generating edge in this capacity, similar to the hybrid approach outlined above for qualitative and quantitative investing approachs is entirely valid


# ##### Kalman Filter Equations
#  
#  State Process:
# 
#  $x_t = x_{t-1} + w_t$ where $w_t \sim N(0,Q)$
#  
#  Measurement Process: 
# 
#  $y_t = x_t + v_t$ where $v_t \sim N(0,R)$
# 
#  Prediction Step:
# 
# $$\mathbb{E}[x_t|y_{1:t-1}] = \mathbb{E}[x_{t-1}|y_{1:t-1}]$$
# $$Var(x_t|y_{1:t-1}) = P_{t-1} + Q$$
# 
#  Update Step:
# 
#  $$\mathbb{E}[x_t|y_{1:t}] = \mathbb{E}[x_t|y_{1:t-1}] + K_t(y_t - \mathbb{E}[x_t|y_{1:t-1}])$$
#  $$Var(x_t|y_{1:t}) = (1-K_t)Var(x_t|y_{1:t-1})$$
#  where $$K_t = \frac{Var(x_t|y_{1:t-1})}{Var(x_t|y_{1:t-1}) + R}$$
# 
# Suppose we had an illiquid instrument that had similar risk profiles to other more liquid or other illiquid instruments.
# 
# We can apply a KF to price the illiquid instruments between observed trades. . .


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='h')  # Changed from 'H' to 'h'
forecast_dates = pd.date_range(start='2023-02-01', end='2023-02-07', freq='h')  # Changed from 'H' to 'h'
all_dates = dates.union(forecast_dates)
n = len(dates)
n_forecast = len(forecast_dates)

# Simulate "true" price process
true_price = 100 + np.cumsum(np.random.normal(0, 0.1, n))

# Generate correlated liquid instruments
liquid_1 = true_price + np.random.normal(0, 0.5, n)
liquid_2 = true_price * 1.1 + np.random.normal(0, 0.6, n)

# Simulate sparse illiquid trades (only 5% of periods have trades)
trade_mask = np.random.random(n) < 0.05
illiquid_trades = np.where(trade_mask, true_price + np.random.normal(0, 0.3, n), np.nan)

class KalmanFilter:
    def __init__(self, R=0.1, Q=0.01):
        self.R = R  # Measurement noise
        self.Q = Q  # Process noise
        self.x = None  # State estimate
        self.P = None  # Error covariance
        
    def initialize(self, initial_value):
        self.x = initial_value
        self.P = 1.0
        
    def predict(self, signal_innovation=0):
        # State prediction
        self.x = self.x + signal_innovation
        # Covariance prediction
        self.P = self.P + self.Q
        return self.x
        
    def update(self, measurement):
        if not np.isnan(measurement):
            # Kalman gain
            K = self.P / (self.P + self.R)
            # State update
            self.x = self.x + K * (measurement - self.x)
            # Covariance update
            self.P = (1 - K) * self.P
        return self.x

# Initialize and run Kalman filter
kf = KalmanFilter()
# Fix: Use first non-NaN value for initialization instead of mean of potentially all NaN values
first_valid_price = next(x for x in illiquid_trades if not np.isnan(x))
kf.initialize(first_valid_price)

estimated_price = []
for t in range(n):
    # Get innovation from liquid signals
    if t > 0:
        signal_innovation = np.mean([
            liquid_1[t] - liquid_1[t-1],
            liquid_2[t] - liquid_2[t-1]
        ])
    else:
        signal_innovation = 0
        
    # Predict and update steps
    pred = kf.predict(signal_innovation)
    est = kf.update(illiquid_trades[t])
    estimated_price.append(est)

# Generate forecasts
forecasted_price = []
last_liquid_1 = liquid_1[-1]
last_liquid_2 = liquid_2[-1]

for _ in range(n_forecast):
    # Simulate future liquid prices
    innovation_1 = np.random.normal(0, 0.5)
    innovation_2 = np.random.normal(0, 0.6)
    last_liquid_1 += innovation_1
    last_liquid_2 += innovation_2
    
    # Average innovation for prediction
    signal_innovation = np.mean([innovation_1, innovation_2])
    
    # Predict only (no updates in forecast window)
    pred = kf.predict(signal_innovation)
    forecasted_price.append(pred)

# Combine historical and forecasted prices
full_estimated_price = np.concatenate([estimated_price, forecasted_price])

# Plot results
fig = go.Figure()

# Plot liquid instruments
fig.add_trace(go.Scatter(x=dates, y=liquid_1, 
                        mode='lines', name='Liquid Instrument 1',
                        line=dict(color='rgba(128,128,128,0.3)')))
fig.add_trace(go.Scatter(x=dates, y=liquid_2, 
                        mode='lines', name='Liquid Instrument 2',
                        line=dict(color='rgba(128,128,128,0.3)')))

# Plot illiquid trades
fig.add_trace(go.Scatter(x=dates[trade_mask], y=illiquid_trades[trade_mask],
                        mode='markers', name='Illiquid Trades',
                        marker=dict(color='red', size=8)))

# Plot Kalman filter estimate
fig.add_trace(go.Scatter(x=dates, y=estimated_price,
                        mode='lines', name='Kalman Estimate',
                        line=dict(color='blue', width=2)))

# Plot forecast
fig.add_trace(go.Scatter(x=forecast_dates, y=forecasted_price,
                        mode='lines', name='Forecast',
                        line=dict(color='green', width=2, dash='dash')))

# Add vertical line separating historical and forecast periods
fig.add_vline(x=dates[-1], line_dash="dash", line_color="white", opacity=0.5)

fig.update_layout(
    title="Kalman Filter Price Estimation and Forecast for Illiquid Instrument",
    template="plotly_dark",
    showlegend=True,
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode='x unified'
)

fig.show()


# ---


# #### 4.) 💭 Closing Thoughts and Future Comments
# 
# Time Series Analysis can be Summarized as Followed...
# 
# - Time Series Models have their Place and are Sufficient (even though they can be incorrect) in a Number of Applications
# 
# - The Space is Heavily Problem Dependent along with Our Choice of Model and Implication of our Assumptions (violations, lackthereof)
# 
# - Forecasts are expectations at best and alternative data can inform decision making before realizations in the price series 
# 
# 
# **Future Topics**
# 
# - AR, MA, ARMA, ARIMA Models (Far more technical than this discussion, Gauging Interest)
# 
# - Market-Making Models (*Avellaneda and Stoikov*)
# 
# - Kalman Filter (Popular Model for Non-Stationarity Systems, I've Deployed Production Systems with This Technique)
# 
# - Algorithmic Trading Systems using the Kalman Filter (A Mini-Kalman-Series?) in Python with Interactive Brokers
# 
# - ARCH and GARCH for Modeling Volatility
# 
# - Volatility Surface Arbitrage (Fit a Model to a Surface and Observe Market Surface Deviations in a High-Frequency)


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

