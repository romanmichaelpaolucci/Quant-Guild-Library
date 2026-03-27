### 📈 Quant Explains Alpha in 3 Minutes

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

#### 1.) 📉 Market vs Portfolio Returns

#### 2.) 📊 Capital Asset Pricing Model (CAPM)

#### 3.) 🎯 Beta, Alpha, and Why It Matters

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📉 Market vs Portfolio Returns

Every year the smartest people in the world compete in an open arena we call the U.S. economy to maximize value to their shareholders

"The Economy" or "The Market" as we often here it is often considered a simple portfolio of the (current) most successful companies

Quantitative hedge funds and traders around the country compete with this simple portfolio which they often consider a benchmark relative to their traidng performance 


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (2025 Simulation) ---
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='B')
n = len(dates)

# Market: Target ~35% annual return
m_mu, m_vol = 0.0012, 0.012
market_rets = np.random.normal(m_mu, m_vol, n)

# Target Sharpe (Annual 2.5 -> Daily 2.5 / sqrt(252))
target_sharpe = 2.5
daily_target = target_sharpe / np.sqrt(252)

# --- Adjusted LEFT PORTFOLIO: Beta 1.5, 
# Formula: R_p = Beta * R_m + Alpha + Noise
beta_left_target = 1.5

# We need: (Mean_p / Std_p) * sqrt(252) = 2.5
# Mean_p = beta * market_mu + alpha
# Std_p = sqrt((beta * market_vol)^2 + sigma_noise^2)

# To hit the target easily, we'll keep noise low and solve for the Alpha needed
noise_vol = 0.002 
expected_p_vol = np.sqrt((beta_left_target * m_vol)**2 + noise_vol**2)
required_p_mu = daily_target * expected_p_vol
alpha_needed = required_p_mu - (beta_left_target * m_mu)

rets_left = (beta_left_target * market_rets) + alpha_needed + np.random.normal(0, noise_vol, n)
# --- RIGHT PORTFOLIO: Beta 0.0, Sharpe 2.5 (Market Neutral Alpha) ---
raw_noise = np.random.normal(0, 1, n)
p2_std = 0.01 
p2_mean = daily_target * p2_std
rets_right = p2_mean + (raw_noise - np.mean(raw_noise)) / np.std(raw_noise) * p2_std

# Helper to calculate realized metrics
def get_realized_sharpe(rets):
    return (np.mean(rets) / np.std(rets)) * np.sqrt(252)

def capm_beta_alpha(port_rets, market_rets):
    # Regression: port_rets = alpha + beta * market_rets + residuals
    X = market_rets
    y = port_rets
    X = np.vstack([np.ones_like(X), X]).T  # add intercept for alpha
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = beta_hat[0]
    beta = beta_hat[1]
    return beta, alpha

s_left = get_realized_sharpe(rets_left)
s_right = get_realized_sharpe(rets_right)
beta_left, alpha_left = capm_beta_alpha(rets_left, market_rets)
beta_right, alpha_right = capm_beta_alpha(rets_right, market_rets)

# Convert alpha (daily) to annualized percent return for reporting
alpha_left_ann_pct = ( (1 + alpha_left) ** 252 - 1 ) * 100
alpha_right_ann_pct = ( (1 + alpha_right) ** 252 - 1 ) * 100

# Cumulative Wealth (Starting at 100)
df = pd.DataFrame({
    'Date': dates,
    'Market': np.cumprod(1 + market_rets) * 100,
    'Port_Left': np.cumprod(1 + rets_left) * 100,
    'Port_Right': np.cumprod(1 + rets_right) * 100
})

# --- 2. Plotly Construction ---
title_l = (
    f"Market vs. Portfolio | Actual Sharpe: {s_left:.2f}"
)
title_r = (
    f"Market vs. Portfolio | Actual Sharpe: {s_right:.2f}"
)

fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(title_l, title_r),
    horizontal_spacing=0.1
)

# Colors and Styling
off_white = "#e0e0e0"
market_color = "#888888"
port_color = "#00d1ff"
play_blue = "#00bfff"
play_darkblue = "#143c99"  # darker blue for play button text

# Start BOTH with line only, no marker, even for initialization
for col_idx, (market_col, port_col) in enumerate([('Market', 'Port_Left'), ('Market', 'Port_Right')], start=1):
    # Only show legend on first subplot
    show_leg = (col_idx == 2)  # put legend on the 2nd subplot instead
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[market_col].iloc[0]],
        line=dict(color=market_color, width=1.5),
        name="Market",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[port_col].iloc[0]],
        line=dict(color=port_color, width=2.5),
        name="Portfolio",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)

# --- 3. Animation Frames ---
num_frames = 60
indices = np.linspace(1, n-1, num_frames, dtype=int)
frames = []
for idx in indices:
    frames.append(go.Frame(
        data=[
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Left'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Right'][:idx], mode='lines')
        ],
        name=str(idx)
    ))
fig.frames = frames

# --- 4. Layout & Transparency ---
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1200,
    margin=dict(t=120, b=120),
    showlegend=True,
    legend=dict(
        orientation='v',
        # Place the legend at the bottom right of the 2nd subplot
        # The 2nd subplot's x-domain is the right half, usually 0.5 to 1.0, so x=1.0 is bottom right
        x=0.995, xanchor='right', y=0.08, yanchor='bottom',
        font=dict(color=off_white, size=15),
        bgcolor='rgba(0,0,0,0.55)',
        bordercolor='rgba(0,0,0,0)'
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.18, 'xanchor': 'center',
        'font': {'color': play_darkblue},
        'buttons': [{'label': '▶ Play 2025 Performance Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)
# Set all axes to off-white
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Set fixed x and y limits per prompt
fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=1
)
fig.update_yaxes(
    range=[90, 215],
    title_text="Wealth Index (Initial=100)", 
    row=1, col=1
)

fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=2
)
fig.update_yaxes(
    range=[90, 155],
    row=1, col=2
)

# Style titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=13)

fig.show()
```



---

#### 2.) 📊 Capital Asset Pricing Model (CAPM)

 **The Capital Asset Pricing Model (CAPM):**
 
 $$
 \mathbb{E}[R_i] = r_f + \beta_i \big(\mathbb{E}[R_m] - r_f\big)
 $$
 
 - $\mathbb{E}[R_i]$ : Expected return of asset $i$
 - $r_f$ : Risk-free rate
 - $\beta_i$ : Sensitivity (“beta”) of $i$ to market returns
 - $\mathbb{E}[R_m]$ : Expected market return
 - $\big(\mathbb{E}[R_m] - r_f\big)$ : Market risk premium
 
 _Reference: Sharpe, W.F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." Journal of Finance._

In practice, we run a regression to see how much our portfolio returns are explained by the market


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Prepare data for scatter/regression ---
def scatter_and_line(mkt, port):
    # Fit regression: y = m*x + b
    X = mkt
    y = port
    # Fit: y = m*X + b
    # m = beta; b = alpha
    m, b = np.polyfit(X, y, deg=1)
    y_fit = m * X + b
    # Calculate R^2 for annotation (optional, for elegance)
    corr = np.corrcoef(X, y)[0,1]
    r2 = corr**2
    # Prepare string for y = mx + b
    line_eq = f"y = {m:.2f}x {'+' if b>=0 else '-'} {abs(b):.5f}"
    # Prepare string for CAPM form: R_p = α + β·R_m
    capm_eq = f"Rₚ = {b:.5f} + {m:.2f}·Rₘ"
    return X, y, y_fit, line_eq, capm_eq, r2

X_left, y_left, yfit_left, eq_left, capm_left, r2_left = scatter_and_line(market_rets, rets_left)
X_right, y_right, yfit_right, eq_right, capm_right, r2_right = scatter_and_line(market_rets, rets_right)

# Style colors
scatter_color = "#00d1ff"
fit_line_color = "#ffaa40"
eq_bgcolor = "rgba(27,27,27,0.92)"
eq_fontcolor = off_white

fig_scatter = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "CAPM Regression: High Beta Portfolio",
        "CAPM Regression: Market-Neutral Alpha Portfolio"
    ),
    horizontal_spacing=0.11
)
# LEFT scatter plot
fig_scatter.add_trace(
    go.Scatter(
        x=X_left, y=y_left, 
        mode="markers", 
        marker=dict(color=scatter_color, size=5, opacity=0.6, line=dict(width=0.3, color='rgba(200,200,255,0.1)')),
        name="Data Points",
        showlegend=False
    ),
    row=1, col=1
)
fig_scatter.add_trace(
    go.Scatter(
        x=np.sort(X_left), 
        y=np.array(yfit_left)[np.argsort(X_left)], 
        mode="lines", 
        name="Line of Best Fit",
        line=dict(color=fit_line_color, width=2.5, dash='solid'),
        showlegend=False
    ),
    row=1, col=1
)
# RIGHT scatter plot
fig_scatter.add_trace(
    go.Scatter(
        x=X_right, y=y_right, 
        mode="markers", 
        marker=dict(color=scatter_color, size=5, opacity=0.6, line=dict(width=0.3, color='rgba(200,200,255,0.1)')),
        name="Data Points",
        showlegend=False
    ),
    row=1, col=2
)
fig_scatter.add_trace(
    go.Scatter(
        x=np.sort(X_right), 
        y=np.array(yfit_right)[np.argsort(X_right)], 
        mode="lines", 
        name="Line of Best Fit",
        line=dict(color=fit_line_color, width=2.5, dash='solid'),
        showlegend=False
    ),
    row=1, col=2
)

# Add annotations: y=mx+b and CAPM
def eq_annotation(equation1, equation2, r2, xpos, ypos, col):
    eq_text = (
        f"<span style='font-family:monospace;font-size:1.02em'>{equation1}</span><br>"
        f"<span style='font-family:monospace;font-size:0.98em'>({equation2})</span><br>"
        f"<span style='color:#cccccc; font-size:0.89em'>R² = {r2:.3f}</span>"
    )
    # Fix: plotly only accepts xref="paper" for per-panel domain annotations!
    return dict(
        xref='paper', yref='paper',
        x=xpos if col == 1 else 0.5 + xpos,
        y=ypos,
        xanchor="left", yanchor="top",
        text=eq_text,
        showarrow=False,
        align='left',
        font=dict(color=eq_fontcolor, size=14),
        bgcolor=eq_bgcolor,
        borderpad=6,
        borderwidth=0,
        opacity=0.98
    )

# Position equations in top left of each subplot using normalized (paper) coordinates
fig_scatter.add_annotation(eq_annotation(eq_left, capm_left, r2_left, xpos=0.02, ypos=0.995, col=1))
fig_scatter.add_annotation(eq_annotation(eq_right, capm_right, r2_right, xpos=0.02, ypos=0.995, col=2))

# Axis styles and layout
for i in range(1,3):
    fig_scatter.update_xaxes(
        title_text="Market Daily Return", 
        zeroline=False,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.13)',
        linecolor=off_white,
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white),
        row=1, col=i
    )
    fig_scatter.update_yaxes(
        title_text="Portfolio Daily Return" if i == 1 else "",
        zeroline=False,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.13)',
        linecolor=off_white,
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white),
        row=1, col=i
    )

fig_scatter.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=100, b=70, l=60, r=30),
    height=540,
    title=dict(
        text="<b>CAPM Regression: Portfolio vs. Market Daily Returns</b>",
        font=dict(size=19, color=off_white), x=0.5
    )
)

# Restyle subplot titles
for ann in fig_scatter['layout']['annotations']:
    if 'CAPM Regression' in ann['text']:
        ann['font'] = dict(color=off_white, size=13)

fig_scatter.show()

```



##### But wait, **where is alpha??**

If CAPM is true, everything is just varying degrees of market exposure - in reality this is not the case!
 
 The standard CAPM model (above) has no "alpha" term—CAPM assumes the market is efficient, so on average, all assets plot exactly on the line: 
 
 $$\mathbb{E}[R_i] = r_f + \beta_i (\mathbb{E}[R_m] - r_f)$$
 
 In real data, we often observe persistent deviations from this line. 
 
 We capture these deviations with an **"alpha"** term by amending the regression equation:
 
 $$
 R_i = \alpha_i + r_f + \beta_i (R_m - r_f) + \varepsilon
 $$
 
 Here, $\alpha_i$ (alpha) measures the excess return of asset $i$ beyond what is predicted by beta and the market, after adjusting for risk-free rate. If $\alpha>0$, the asset or portfolio is "beating CAPM"; if $\alpha<0$, it's underperforming.

 Academics will argue there is "no alpha" and everything is just unidentified priced risk

 Those of us in the real world acknowledge the market is NOT perfectly efficient, and inefficiencies produce ALPHA


```python
# Compute only alpha and beta components for LEFT and RIGHT portfolios, no noise or total

# Custom color themes (reduce to alpha and beta only)
alpha_color = "#00ffae"
beta_color = "#009efb"

# Risk-free rate (set to 0 for simplicity)
rf = 0.0

# Mean market return (for beta * (market - rf) calculation)
mean_market = np.mean(market_rets)

# LEFT PORTFOLIO
beta_left, alpha_left = capm_beta_alpha(rets_left, market_rets)
mean_beta_left = beta_left * (mean_market - rf)
mean_ret_left = np.mean(rets_left)

# RIGHT PORTFOLIO
beta_right, alpha_right = capm_beta_alpha(rets_right, market_rets)
mean_beta_right = beta_right * (mean_market - rf)
mean_ret_right = np.mean(rets_right)

# Aggregate components for each portfolio
components_left = [alpha_left, mean_beta_left]
components_right = [alpha_right, mean_beta_right]

# SET negative components to 0 (cannot plot negative bars in a breakdown)
components_left_nonneg = [max(0, x) for x in components_left]
components_right_nonneg = [max(0, x) for x in components_right]

# To avoid division by zero, check if the total is positive, otherwise assign all zero
sum_left = sum(components_left_nonneg)
sum_right = sum(components_right_nonneg)

if sum_left > 0:
    prop_left = [x / sum_left * 100 for x in components_left_nonneg]
else:
    prop_left = [0, 0]

if sum_right > 0:
    prop_right = [x / sum_right * 100 for x in components_right_nonneg]
else:
    prop_right = [0, 0]

prop_components = prop_left + prop_right  # Normalized, guaranteed nonnegative, out of 100 for each portfolio

labels_left = ['Alpha (Left)', 'Beta × (Market-RF) (Left)']
colors_left = [alpha_color, beta_color]
labels_right = ['Alpha (Right)', 'Beta × (Market-RF) (Right)']
colors_right = [alpha_color, beta_color]

labels = labels_left + labels_right
colors = colors_left + colors_right

import plotly.express as px

fig_bars = go.Figure()
bar_width = 0.28

# Place bars
xs_left = [0, 0.4]
xs_right = [1.5, 1.9]

# Plot LEFT
for idx in range(2):
    fig_bars.add_trace(go.Bar(
        x=[xs_left[idx]],
        y=[prop_left[idx]],
        width=[bar_width],
        name=labels_left[idx],
        marker_color=colors_left[idx],
        showlegend=False  # Remove all legend entries
    ))

# Plot RIGHT
for idx in range(2):
    fig_bars.add_trace(go.Bar(
        x=[xs_right[idx]],
        y=[prop_right[idx]],
        width=[bar_width],
        name=labels_right[idx],
        marker_color=colors_right[idx],
        showlegend=False
    ))

# X axis ticks/labels
x_tics = [0.2, 1.7]
x_labels = ['Left Portfolio', 'Right Portfolio']
fig_bars.update_layout(
    template="plotly_dark",
    height=420,
    width=660,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    barmode="stack",
    title=dict(
        text="<b>2025 Return Breakdown:<br>Alpha and Beta Proportion (Out of 100, %)</b>",
        font=dict(size=17, color=off_white), x=0.5
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=x_tics,
        ticktext=x_labels,
        showgrid=False,
        showticklabels=True,
        linecolor=off_white,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(
            text="Component Weight (Normalized %)",
            font=dict(color=off_white, size=14)
        ),
        range=[0, 110],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.13)",
        zeroline=True,
        zerolinecolor=off_white,
        tickfont=dict(color=off_white),
        linecolor=off_white
    ),
    showlegend=False,  # This disables the legend entirely
    margin=dict(t=70, l=70, r=40, b=70)
)

# Add value labels on bars (show percentages out of 100)
for i, x in enumerate(xs_left + xs_right):
    yval = prop_components[i]
    fig_bars.add_annotation(
        x=x,
        y=yval + 3,
        text=f"{yval:.1f}%",
        showarrow=False,
        font=dict(color=off_white, size=12),
        xanchor='center',
        yanchor='bottom'
    )

fig_bars.show()

```



---

#### 3.) 🎯 Beta, Alpha, and Why It Matters
 
 Suppose you have two portfolios, both with a high Sharpe ratio of 2.5 and similar risk. 
 
 They seem equally attractive at first. But how do we decide which one is better, especially for the future?
 
 The key is this: **choose the portfolio whose returns are *independent* from the market.**

 ##### Let's Look at the Same Two Portfolios from Above but Now in a Bear Market

 *assuming ~stability~* it follows. . .


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (2025 Bear Market Simulation) ---
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='B')
n = len(dates)

# Market: Target ~-30% annual return (Bear Market / Recession)
target_annual_return = -0.30
trading_days = 252
# Calculate daily geometric return for -30% annualized
m_mu = (1 + target_annual_return)**(1/trading_days) - 1
m_vol = 0.012   # keep same vol as before for one year

market_rets = np.random.normal(m_mu, m_vol, n)

# Target Sharpe (Annual 2.5 -> Daily 2.5 / sqrt(252))
target_sharpe = 2.5
daily_target = target_sharpe / np.sqrt(252)

# --- LEFT PORTFOLIO: High Beta (1.5), *NO Alpha* ---
# This portfolio should lose more than the market in a bear market.
beta_left_target = 1.5

# Set alpha = 0, so expected mean is just beta * market_mu
# Mean_p = beta * market_mu (which is deeply negative)
# Std_p = sqrt((beta * market_vol)^2 + sigma_noise^2)
noise_vol = 0.002
rets_left = (beta_left_target * market_rets) + np.random.normal(0, noise_vol, n)

# --- RIGHT PORTFOLIO: Beta 0.0, Sharpe 2.5 (Market Neutral Alpha) ---
raw_noise = np.random.normal(0, 1, n)
p2_std = 0.01 
p2_mean = daily_target * p2_std
rets_right = p2_mean + (raw_noise - np.mean(raw_noise)) / np.std(raw_noise) * p2_std

# Helper to calculate realized metrics
def get_realized_sharpe(rets):
    return (np.mean(rets) / np.std(rets)) * np.sqrt(252)

def capm_beta_alpha(port_rets, market_rets):
    # Regression: port_rets = alpha + beta * market_rets + residuals
    X = market_rets
    y = port_rets
    X = np.vstack([np.ones_like(X), X]).T  # add intercept for alpha
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = beta_hat[0]
    beta = beta_hat[1]
    return beta, alpha

s_left = get_realized_sharpe(rets_left)
s_right = get_realized_sharpe(rets_right)
beta_left, alpha_left = capm_beta_alpha(rets_left, market_rets)
beta_right, alpha_right = capm_beta_alpha(rets_right, market_rets)

# Convert alpha (daily) to annualized percent return for reporting
alpha_left_ann_pct = ( (1 + alpha_left) ** 252 - 1 ) * 100
alpha_right_ann_pct = ( (1 + alpha_right) ** 252 - 1 ) * 100

# Cumulative Wealth (Starting at 100)
df = pd.DataFrame({
    'Date': dates,
    'Market': np.cumprod(1 + market_rets) * 100,
    'Port_Left': np.cumprod(1 + rets_left) * 100,
    'Port_Right': np.cumprod(1 + rets_right) * 100
})

# --- 2. Plotly Construction ---
title_l = (
    f"Bear Market vs. High Beta Portfolio | Previous Sharpe: 2.65"
)
title_r = (
    f"Bear Market vs. Market Neutral Alpha | Previous Sharpe: 2.5"
)

fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(title_l, title_r),
    horizontal_spacing=0.1
)

# Colors and Styling
off_white = "#e0e0e0"
market_color = "#888888"
port_color = "#00d1ff"
play_blue = "#00bfff"
play_darkblue = "#143c99"  # darker blue for play button text

# Plot initialization
for col_idx, (market_col, port_col) in enumerate([('Market', 'Port_Left'), ('Market', 'Port_Right')], start=1):
    show_leg = (col_idx == 2)  # legend on the 2nd subplot
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[market_col].iloc[0]],
        line=dict(color=market_color, width=1.5),
        name="Market",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[port_col].iloc[0]],
        line=dict(color=port_color, width=2.5),
        name="Portfolio",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)

# --- 3. Animation Frames ---
num_frames = 60
indices = np.linspace(1, n-1, num_frames, dtype=int)
frames = []
for idx in indices:
    frames.append(go.Frame(
        data=[
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Left'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Right'][:idx], mode='lines')
        ],
        name=str(idx)
    ))
fig.frames = frames

# --- 4. Layout & Transparency ---
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    margin=dict(t=120, b=120),
    showlegend=True,
    legend=dict(
        orientation='v',
        # Place the legend at the bottom right of the 2nd subplot
        # The 2nd subplot's x-domain is the right half, usually 0.5 to 1.0, so x=1.0 is bottom right
        x=0.995, xanchor='right', y=0.08, yanchor='bottom',
        font=dict(color=off_white, size=15),
        bgcolor='rgba(0,0,0,0.55)',
        bordercolor='rgba(0,0,0,0)'
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.18, 'xanchor': 'center',
        'font': {'color': play_darkblue},
        'buttons': [{'label': '▶ Play 2025 Bear Market Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)
# Set all axes to off-white
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Suitable fixed y-axes for drawdown (bear market)
fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=1
)
fig.update_yaxes(
    range=[50, 120],
    title_text="Wealth Index (Initial=100)", 
    row=1, col=1
)

fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=2
)
fig.update_yaxes(
    range=[65, 155],
    row=1, col=2
)

# Style titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=13)

fig.show()
```



---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**

- Portfolio managers and traders are either subjecting themselves to priced risk like market exposure, other documented facets of risk (Fama-French X factor models) OR are generating alpha through strategic allocations or a viable trading strategy
- CAPM quantifies this exposure by running a simple regression and answering the question: how much of your portfolio is explained by the market?
- If two strategies have the same or similar performance metrics and they are all reasonably stable across time we will opt for the strategy that produces alpha and takes advantage of a structural misprising
- If we are only trading beta(s) we are at the whims of the universe and whenever a factor dominating our strategy return decides to mean revert there is nothing we can do about it but retroactively hedge
- In other words, choose alpha, not beta 

**Future Topics**

Technical Videos and Other Discussions

- Top Quant Research Papers
- Projects that Made me a Quant
- My First Year as a Quant
- Why Hedge Funds are Actually Secretive
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Poisson Processes for Quant Finance
- Top 3 Uses of Linear Algebra for Quant Finance
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Financial News Sentiment Feed

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
