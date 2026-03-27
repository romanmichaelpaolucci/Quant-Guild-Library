### 📈 The 5 Papers That Built Modern Quant Finance

##### ▶️ Related Quant Guild Videos:

- [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4)

- [Heston Stochastic Volatility Model and Fast Fourier Transforms](https://youtu.be/2-oAlnZV6hA)

- [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)

- [My Life as a Quant](https://youtu.be/n1Z90Iwc_co)

- [How Physics Accidentally Proved the Black-Scholes Model](https://youtu.be/IIzGqL3ChEs)

- [Trader Skill or Market Luck?  Quant Explains Alpha in 3 Minutes](https://youtu.be/Ivz58kZLD2U)

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

#### 1.) 📉 Bachelier (1900)

- Théorie de la Spéculation

#### 2.) 📊 Sharpe (1964)

- Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk

#### 3.) 🛡️ Black-Scholes (1973)

- The Pricing of Options and Corporate Liabilities

#### 4.) 🙂 Dupire (1994)

- Pricing with a Smile

#### 5.) 🌊 Carr-Madan (1999)

- Option Valuation Using the Fast Fourier Transform

#### 6.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📉 Bachelier (1900)

- Théorie de la Spéculation

$$
dS_t = \mu\,dt + \sigma\,dW_t
$$

 **Key Result: Bachelier (1900) — "Théorie de la Spéculation"**
 
 Louis Bachelier, who studied at the Sorbonne under the legendary mathematician Henri Poincaré, revolutionized our understanding of uncertainty and randomness in markets. In 1900, he introduced the concept of Brownian motion to model fluctuations in financial prices—well before Einstein’s formalization for physical particles. Bachelier applied ideas from heat dynamics and probability to asset prices, laying the groundwork for what would become stochastic calculus and modern quantitative finance.
 
 His thesis, considered so radical for its time that it was awarded an "honorable" rather than "très honorable" distinction, went largely ignored for decades. Yet, Bachelier is now recognized as the father of financial mathematics, and his insights form the basis of models used throughout finance today.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1
mu = 0.1      # Drift (ABM)
sigma = 1     # Volatility

np.random.seed(42)

# Generate ABM: dW_t = mu*dt + sigma*dZ_t
increments = np.random.normal(mu * dt, sigma * np.sqrt(dt), (n_paths_total, n_steps))
W = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

# --- Precompute Fixed Scales ---
global_min = np.min(W)
global_max = np.max(W)
y_range = [global_min - 2, global_max + 2]

max_pdf = max(norm.pdf(0, 0, np.sqrt(t)) for t in range(1, n_steps + 1)) + 0.05

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    
    # Histogram Calculations
    hist_values = W[:, t]
    hist_fig = np.histogram(hist_values, bins=30, density=True)
    hist_y = 0.5 * (hist_fig[1][1:] + hist_fig[1][:-1])
    hist_x = hist_fig[0]

    # Theoretical Normal Distribution for ABM
    if t == 0:
        theory_x = np.zeros_like(hist_y)
    else:
        theory_x = norm.pdf(hist_y, loc=mu*t, scale=sigma*np.sqrt(t))

    # 1. Background Paths (Uniform Cyan)
    scatter_paths = [
        go.Scatter(
            x=time_show,
            y=W[i, :t + 1],
            mode='lines',
            line=dict(color='#00ffff', width=1), 
            opacity=0.3,
            showlegend=False,
            hoverinfo='none'
        )
        for i in range(n_paths_show)
    ]

    # 2. Histogram (Changed to Cyan to match paths)
    hist = go.Bar(
        y=hist_y,
        x=hist_x,
        width=(hist_y[1] - hist_y[0]) if len(hist_y) > 1 else 0.5,
        orientation='h',
        marker=dict(color='#00ffff', opacity=0.6), # Cyan with same opacity feel
        hoverinfo='skip'
    )

    # 3. Theory Line (Magenta)
    theory_line = go.Scatter(
        x=theory_x,
        y=hist_y,
        mode='lines',
        line=dict(color='magenta', width=3), 
        hoverinfo='skip'
    )

    # 4. Vertical Line (Thicker Magenta)
    vline = go.Scatter(
        x=[t, t],
        y=[y_range[0], y_range[1]], 
        mode='lines',
        line=dict(color='magenta', dash='dash', width=4), # Thicker line
        showlegend=False
    )
    
    frames.append(go.Frame(
        data=[*scatter_paths, vline, hist, theory_line],
        name=f"step{t}"
    ))

# --- Initial Frame Data ---
init_t = 0
time_show = time_grid[:init_t + 1]
hist_y = np.array([0])
hist_x = np.array([0])
theory_x = np.zeros_like(hist_y)

scatter_paths_init = [
    go.Scatter(
        x=time_show,
        y=W[i, :init_t + 1],
        mode='lines',
        line=dict(color='#00ffff', width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='none'
    )
    for i in range(n_paths_show)
]

hist_init = go.Bar(
    y=hist_y, x=hist_x, orientation='h',
    marker=dict(color='#00ffff', opacity=0.6)
)

theory_line_init = go.Scatter(
    x=theory_x, y=hist_y, mode='lines',
    line=dict(color='magenta', width=3)
)

vline_init = go.Scatter(
    x=[init_t, init_t], y=[y_range[0], y_range[1]],
    mode='lines', line=dict(color='magenta', dash='dash', width=4), # Thicker
    showlegend=False
)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=["Arithmetic Brownian Paths (ABM)", "Distribution at t"],
    horizontal_spacing=0.05
)

for s in scatter_paths_init:
    fig.add_trace(s, row=1, col=1)
fig.add_trace(vline_init, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line_init, row=1, col=2)

fig.frames = frames

# --- Create Slider Steps ---
sliders_steps = []
for t in range(n_steps + 1):
    step = dict(
        method="animate",
        args=[
            [f"step{t}"],
            dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0)
            )
        ],
        label=str(t)
    )
    sliders_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=600, width=1200,
    title_text=f"ABM Simulation (Drift mu={mu}, Vol sigma={sigma})",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.1, 'y': -0.15,
        'xanchor': 'right', 'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Time Step: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 0, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': -0.15,
        'steps': sliders_steps,
        'font': {'color': 'white'}
    }]
)

# Axes styling
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

# Fixed Global Ranges
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='W_t', range=y_range, row=1, col=1) 
fig.update_xaxes(title_text='Density', range=[0, max_pdf], row=1, col=2)
fig.update_yaxes(title_text='W_t', range=y_range, row=1, col=2) 

fig.show()
```



Bachelier didn't yet know of

- Brownian motion (formalized by Einstein years later)

- Portfolio Replication in the Context of a Complete Risk-Neutral Market

- Feynman-Kac, the stochastic representation to the closed form price he found

And he developed a remarkably consistent, first of its kind model for what was to come

---

#### 2.) 📊 Sharpe (1964)

- Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk

  $$
  \mathbb{E}[R_i] = r_f + \beta_i \big( \mathbb{E}[R_m] - r_f \big)
  $$

**Key Result: Sharpe (1964) — "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk"**
 
While Harry Markowitz had already formalized the idea of portfolio diversification—showing that risk could be managed and minimized through optimal portfolio selection—there remained a big question: *What actually determines the prices of assets in a market where everyone is optimizing their portfolios?*  
 
William F. Sharpe answered this with the Capital Asset Pricing Model (CAPM). Building on Markowitz’s work, Sharpe asked: “If all investors follow the same rational mean-variance optimization, and markets clear, how should risk be priced?” His 1964 paper introduced the concept that only systematic, or market, risk should be rewarded—because idiosyncratic risk can be diversified away. 
 
The main result, the CAPM equation, links an asset’s expected return to its sensitivity (beta) to the overall market, fundamentally changing how risk and return are understood in finance. Sharpe’s insight underpins much of modern asset pricing and portfolio theory today.



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

# Target Sharpe Metrics
target_sharpe = 2.5
daily_target = target_sharpe / np.sqrt(252)

# --- Adjusted LEFT PORTFOLIO: Beta 1.5 ---
beta_left_target = 1.5
noise_vol = 0.002 
expected_p_vol = np.sqrt((beta_left_target * m_vol)**2 + noise_vol**2)
required_p_mu = daily_target * expected_p_vol
alpha_needed = required_p_mu - (beta_left_target * m_mu)
rets_left = (beta_left_target * market_rets) + alpha_needed + np.random.normal(0, noise_vol, n)

# --- RIGHT PORTFOLIO: Beta 0.0 (Market Neutral) ---
raw_noise = np.random.normal(0, 1, n)
p2_std = 0.01 
p2_mean = daily_target * p2_std
rets_right = p2_mean + (raw_noise - np.mean(raw_noise)) / np.std(raw_noise) * p2_std

# Create DataFrame for Wealth Lines
df = pd.DataFrame({
    'Date': dates,
    'Market': np.cumprod(1 + market_rets) * 100,
    'Port_Left': np.cumprod(1 + rets_left) * 100,
    'Port_Right': np.cumprod(1 + rets_right) * 100
})

# --- 2. Calculate Axis Limits (Locking the View) ---
x_min, x_max = market_rets.min(), market_rets.max()
x_range = [x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1]
y_l_range = [rets_left.min()*1.1, rets_left.max()*1.1]
y_r_range = [rets_right.min()*1.1, rets_right.max()*1.1]

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Wealth Evolution: High Beta Strategy | Sharpe: ~2.5", 
        "Wealth Evolution: Market Neutral Strategy | Sharpe: ~2.5",
        "Live CAPM Regression (High Beta)", 
        "Live CAPM Regression (Market Neutral)"
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

# Colors
off_white = "#e0e0e0"
market_color = "#888888"
port_color = "#00d1ff"
fit_color = "#ffaa40"

# --- Add Initial Traces ---
# Row 1: Time Series
for col, p_col in zip([1, 2], ['Port_Left', 'Port_Right']):
    fig.add_trace(go.Scatter(
        x=[df['Date'][0]], y=[df['Market'][0]],
        mode='lines', line=dict(color=market_color, width=1.5),
        name='Market', showlegend=(col==1)
    ), row=1, col=col)
    
    fig.add_trace(go.Scatter(
        x=[df['Date'][0]], y=[df[p_col][0]],
        mode='lines', line=dict(color=port_color, width=2.5),
        name='Portfolio', showlegend=(col==1)
    ), row=1, col=col)

# Row 2: Scatter & Regression
# Left
fig.add_trace(go.Scatter(x=[market_rets[0]], y=[rets_left[0]], mode='markers', marker=dict(color=port_color, size=5, opacity=0.5), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines', line=dict(color=fit_color, width=3), name='Regression Fit'), row=2, col=1)

# Right
fig.add_trace(go.Scatter(x=[market_rets[0]], y=[rets_right[0]], mode='markers', marker=dict(color=port_color, size=5, opacity=0.5), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines', line=dict(color=fit_color, width=3), showlegend=False), row=2, col=2)

# --- 4. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

num_frames = 60
indices = np.linspace(20, n, num_frames, dtype=int)

for k in indices:
    frame_name = str(k)
    
    # Data Slices
    d_slice = df['Date'][:k]
    m_wealth = df['Market'][:k]
    m_rets_slice = market_rets[:k]
    
    l_wealth = df['Port_Left'][:k]
    l_rets_slice = rets_left[:k]
    
    r_wealth = df['Port_Right'][:k]
    r_rets_slice = rets_right[:k]

    # Fits
    m_l, b_l = np.polyfit(m_rets_slice, l_rets_slice, 1)
    y_fit_l = m_l * np.array(x_range) + b_l
    
    m_r, b_r = np.polyfit(m_rets_slice, r_rets_slice, 1)
    y_fit_r = m_r * np.array(x_range) + b_r
    
    # Create Frame
    frames.append(go.Frame(
        data=[
            go.Scatter(x=d_slice, y=m_wealth), # Trace 0
            go.Scatter(x=d_slice, y=l_wealth), # Trace 1
            go.Scatter(x=d_slice, y=m_wealth), # Trace 2
            go.Scatter(x=d_slice, y=r_wealth), # Trace 3
            go.Scatter(x=m_rets_slice, y=l_rets_slice), # Trace 4
            go.Scatter(x=x_range, y=y_fit_l), # Trace 5
            go.Scatter(x=m_rets_slice, y=r_rets_slice), # Trace 6
            go.Scatter(x=x_range, y=y_fit_r), # Trace 7
        ],
        name=frame_name
    ))

    # Create Slider Step
    slider_steps.append(dict(
        args=[[frame_name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(d_slice.iloc[-1].date()),
        method="animate"
    ))

fig.frames = frames

# --- 5. Layout, Slider & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1200,
    margin=dict(t=80, b=100, l=60, r=40),
    
    # Play Button
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x': 0.05, 'y': -0.15, 'xanchor': 'right', 'yanchor': 'top',
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]
        }, {
            'label': 'II Pause',
            'method': 'animate',
            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
        }]
    }],
    
    # Slider Config
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16, 'color': off_white},
            'prefix': 'Date: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 0, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.07, 'y': -0.15,
        'steps': slider_steps,
        'font': {'color': off_white}
    }]
)

# Lock Axes
# Top Row
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(range=[90, 220], title_text="Wealth", row=1, col=1)
fig.update_yaxes(range=[90, 160], row=1, col=2)

# Bottom Row
fig.update_xaxes(range=x_range, title_text="Market Returns", row=2, col=1)
fig.update_xaxes(range=x_range, title_text="Market Returns", row=2, col=2)
fig.update_yaxes(range=y_l_range, title_text="Portfolio Returns", row=2, col=1)
fig.update_yaxes(range=y_r_range, row=2, col=2)

# Grid styling
fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor=off_white)
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor=off_white)

# Fix annotation colors
for ann in fig['layout']['annotations']:
    ann['font'] = dict(color=off_white, size=14)

fig.show()
```



---

#### 3.) 🛡️ Black-Scholes (1973)

- The Pricing of Options and Corporate Liabilities

 $$
 \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
 $$

 **Key Result: Black-Scholes (1973) — "The Pricing of Options and Corporate Liabilities"**
 
 In 1973, Fischer Black and Myron Scholes—later extended by Robert Merton—introduced a breakthrough formula for pricing European options, forever changing derivatives markets and modern finance. Their key innovation was the *replication argument*: rather than attempt to forecast an option’s future price directly, they constructed a dynamic trading strategy that replicates the option’s payoff by continuously rebalancing a portfolio of the underlying asset and risk-free bonds.
 
 This “replicating portfolio” concept was a revolution: it showed that, under certain assumptions (no arbitrage, continuous trading, and frictionless markets), the price of the option must be equal to the cost of setting up and maintaining this hedge—fully justified by arbitrage logic. No one had formalized this link so precisely until Black-Scholes.
 
 The Black-Scholes equation and resulting closed-form solution provided the first model to fairly price options using observable variables—underpinning the explosion of options markets and much of modern financial engineering. The replication argument is now a foundational principle for all derivatives pricing and risk management.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Simulation Parameters ---
np.random.seed(42)
n_paths = 20      # Increased slightly for better "cloud" effect
n_steps = 60      # Smoother lines
T = 2.0
dt = T / n_steps
S0 = 100

# Market Parameters
r = 0.02          # Risk-free rate (Q drift) - Lowered slightly to increase contrast
mu = 0.18         # Real-world expected return (P drift) - Increased slightly
sigma = 0.20      # Constant Volatility (Flat Surface)

# Time axis
time_axis = np.linspace(0, T, n_steps + 1)

# --- 2. Generate Paths & Expectations ---
# A. Sample Paths (Geometric Brownian Motion)
# Using independent random numbers to show distribution differences clearly
dW_P = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))
dW_Q = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))

S_P = np.zeros((n_steps + 1, n_paths))
S_Q = np.zeros((n_steps + 1, n_paths))
S_P[0] = S0
S_Q[0] = S0

for t in range(n_steps):
    # P-Measure: Drift = mu
    S_P[t+1] = S_P[t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW_P[t])
    # Q-Measure: Drift = r
    S_Q[t+1] = S_Q[t] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW_Q[t])

# B. Theoretical Expected Value Paths (Trend Lines)
# E[S_t] = S0 * exp(drift * t)
E_P = S0 * np.exp(mu * time_axis)
E_Q = S0 * np.exp(r * time_axis)

# --- 3. Surface Data (Flat Constant Vol) ---
S_strike = np.linspace(80, 120, 10)
T_expiry = np.linspace(0.1, 2.0, 10)
S_mesh, T_mesh = np.meshgrid(S_strike, T_expiry)
Z_flat = np.full_like(S_mesh, sigma)

# --- 4. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'xy'}]],
    column_widths=[0.5, 0.5],
    subplot_titles=("Constant Volatility (Flat Surface)", "P-Measure vs Q-Measure Expectations")
)

# Fixed Axis Constraints
fixed_axes = dict(
    xaxis=dict(range=[0, 2], autorange=False),
    # Range needs to accommodate the highest potential P-paths
    yaxis=dict(range=[50, 220], autorange=False) 
)

# --- 5. Build Frames ---
frames = []
# Opacity for sample paths
path_opacity = 0.2

for k in range(1, n_steps + 1):
    frame_data = []
    
    # 1. P-Measure Sample Paths (Transparent Cyan)
    for i in range(n_paths):
        frame_data.append(go.Scatter(
            x=time_axis[:k+1], y=S_P[:k+1, i],
            mode='lines', line=dict(color=f'rgba(0, 255, 255, {path_opacity})', width=1.5),
            legendgroup='P_samples', showlegend=False, name='P Paths'
        ))
        
    # 2. Q-Measure Sample Paths (Transparent Magenta)
    for i in range(n_paths):
        frame_data.append(go.Scatter(
            x=time_axis[:k+1], y=S_Q[:k+1, i],
            mode='lines', line=dict(color=f'rgba(255, 0, 255, {path_opacity})', width=1.5),
            legendgroup='Q_samples', showlegend=False, name='Q Paths'
        ))

    # 3. P-Measure Trend Line (Solid Cyan, Thicker)
    frame_data.append(go.Scatter(
        x=time_axis[:k+1], y=E_P[:k+1],
        mode='lines', line=dict(color='cyan', width=4),
        legendgroup='P_trend', showlegend=False, name="E[S] P-Measure"
    ))

    # 4. Q-Measure Trend Line (Solid Magenta, Thicker)
    frame_data.append(go.Scatter(
        x=time_axis[:k+1], y=E_Q[:k+1],
        mode='lines', line=dict(color='magenta', width=4),
        legendgroup='Q_trend', showlegend=False, name="E[S] Q-Measure"
    ))

    # Calculate indices for the right-hand panel traces. 
    # Trace 0 is the surface on the left. The rest are on the right.
    # Total right traces = 2*n_paths (samples) + 2 (trends)
    num_right_traces = 2 * n_paths + 2
    right_trace_indices = list(range(1, num_right_traces + 1))

    frames.append(go.Frame(
        data=frame_data,
        name=f"step_{k}",
        layout=go.Layout(**fixed_axes),
        traces=right_trace_indices
    ))

# --- 6. Initial Traces (t=0 state) ---
# Trace 0: Left Panel Surface
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_flat,
    colorscale='Blues', opacity=0.8, showscale=False,
), row=1, col=1)

# --- Add legend traces for P and Q measure for the right (xy) panel ---
# We add one invisible trace for each sample family with legend enabled.

# 1. P-Samples (for legend)
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(color=f'rgba(0, 255, 255, {path_opacity})', width=2),
    name="P-Measure Paths",
    legendgroup='P_samples',
    showlegend=True
), row=1, col=2)

# 2. Q-Samples (for legend)
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(color=f'rgba(255, 0, 255, {path_opacity})', width=2),
    name="Q-Measure Paths",
    legendgroup='Q_samples',
    showlegend=True
), row=1, col=2)

# Real initial traces: P-Samples
for i in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[S0], mode='lines', 
        line=dict(color=f'rgba(0, 255, 255, {path_opacity})', width=1.5), showlegend=False, legendgroup='P_samples'), row=1, col=2)
# Q-Samples
for i in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[S0], mode='lines', 
        line=dict(color=f'rgba(255, 0, 255, {path_opacity})', width=1.5), showlegend=False, legendgroup='Q_samples'), row=1, col=2)

# 3. P-Trend (Solid) - Add legend entry here
fig.add_trace(go.Scatter(
    x=[0], y=[S0], mode='lines', 
    line=dict(color='cyan', width=4),
    name=f'E[S] P-Measure (μ={mu:.2f})', legendgroup='P_trend', showlegend=True
), row=1, col=2)

# 4. Q-Trend (Solid) - Add legend entry here
fig.add_trace(go.Scatter(
    x=[0], y=[S0], mode='lines', 
    line=dict(color='magenta', width=4),
    name=f'E[S] Q-Measure (r={r:.2f})', legendgroup='Q_trend', showlegend=True
), row=1, col=2)

fig.frames = frames

# --- 7. Layout Styling ---
fig.update_layout(
    template="plotly_dark",
    title="Measure Change: Real-World (P) vs Risk-Neutral (Q) Expectations",
    height=600, width=1200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    
    legend=dict(x=0.74, y=0.98, bgcolor='rgba(50,50,50,0.8)'),

    scene=dict(
        xaxis_title='Strike (K)', yaxis_title='Time (T)', zaxis_title='Vol (σ)',
        xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
        yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
        zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", range=[0, 0.4]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    ),
    
    xaxis=dict(title='Time (Years)', gridcolor='rgb(60, 60, 60)', **fixed_axes['xaxis']),
    yaxis=dict(title='Price ($)', gridcolor='rgb(60, 60, 60)', **fixed_axes['yaxis']),
    
    annotations=[
        dict(text=f"Vol: {sigma*100:.0f}%", x=0.2, y=0.9, xref="paper", yref="paper", showarrow=False),
    ],

    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1, 'xanchor': 'center',
        'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': '#E0E0E0', 'borderwidth': 1,
        'font': dict(color='#E0E0E0'), 'showactive': False,
        'buttons': [{
            'label': '▶ Simulate Paths & Trends',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}}]
        }]
    }]
)

fig.show()
```



##### The difference in expectation under the $\mathbb{P}$ and $\mathbb{Q}$ measures is denoted:

 $$
     \mathbb{E}^{\mathbb{P}}[S_T] \neq \mathbb{E}^{\mathbb{Q}}[S_T]
 $$
 - $\mathbb{E}^{\mathbb{P}}[\cdot]$ is the **real-world (physical) expectation**.
 - $\mathbb{E}^{\mathbb{Q}}[\cdot]$ is the **risk-neutral expectation** (used for pricing derivatives).

 The risk-free discounting and absence of arbitrage imply:
 
 $$
     S_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[S_T]
 $$

Feynman-Kac show that the arbitrage free price by the Black-Scholes argument is the one gaurenteed by the Fundamental Theorem of Asset Pricing, incredible!

---

#### 4.) 🙂 Dupire (1994)

- Pricing with a Smile
 
  $$
  \frac{\partial C}{\partial T} = \frac{1}{2} \sigma^2(K,T) K^2 \frac{\partial^2 C}{\partial K^2} + [r(T)K - q(T)K] \frac{\partial C}{\partial K} - q(T) C
  $$
 
**Key Result: Dupire (1994) — "Pricing with a Smile"**

 In 1994, Bruno Dupire introduced a groundbreaking framework for modeling and calibrating the volatility surface as observed in option markets. Rather than assuming constant volatility as in Black-Scholes, Dupire developed the *local volatility model*, directly connecting the prices of European call options across strikes and maturities to a “local” volatility function.

 The hallmark of Dupire’s approach is the **Dupire equation** shown above. This forward partial differential equation explicitly links the evolution of option prices for different strikes and maturities to the local volatility function $\sigma(K,T)$. This enables traders and risk managers to infer a volatility surface from observed option prices—*pricing with a smile*, capturing the market’s changing risk views across strikes and expiries.

 Dupire’s result changed derivatives modeling by recognizing that implied volatility is not constant, but rather a dynamic surface. This insight fuels the calibration of more realistic pricing models and risk tools, accommodating volatility smiles and skews witnessed in real markets.


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

# --- 1. Setup Surface Data ---
S_unique = np.array([80, 90, 100, 110, 120], dtype=float)
T_unique = np.array([0.0833, 0.25, 0.50, 1.00, 2.00], dtype=float)
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

# Target Market Surface
Z_market = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])

# Initial Flat Surface
avg_vol = np.mean(Z_market)
Z_flat = np.full_like(Z_market, avg_vol)

# Flatten for Scatter3D
S_flat = S_mesh.flatten()
T_flat = T_mesh.flatten()
Z_market_flat = Z_market.flatten()

# --- 2. Setup Simulation (Local Vol) ---
n_paths = 20
n_steps_sim = 50
T_max = 2.0
dt = T_max / n_steps_sim
S0 = 100

interp_vol = RegularGridInterpolator((T_unique, S_unique), Z_market, 
                                     bounds_error=False, fill_value=None)

def get_vol_from_surface(t, s):
    t_safe = np.clip(t, T_unique.min(), T_unique.max())
    s_safe = np.clip(s, S_unique.min(), S_unique.max())
    return float(interp_vol((t_safe, s_safe)))

# Pre-calculate Paths
paths = np.zeros((n_steps_sim + 1, n_paths))
paths[0, :] = S0
np.random.seed(42)

for t_step in range(n_steps_sim):
    t_curr = t_step * dt
    for p in range(n_paths):
        s_curr = paths[t_step, p]
        sigma = get_vol_from_surface(t_curr, s_curr)
        dW = np.random.normal(0, np.sqrt(dt))
        paths[t_step + 1, p] = s_curr + s_curr * sigma * dW

time_axis = np.linspace(0, T_max, n_steps_sim + 1)

# --- 3. Base Figure & Annotations Setup ---
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'xy'}]],
    column_widths=[0.6, 0.4],
    subplot_titles=("Calibration to Market Data", "Monte Carlo Simulation")
)

static_annotations = list(fig.layout.annotations)

def get_status_annotation(text, color):
    return dict(
        text=text,
        x=0.10, y=0.98,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color=color),
        align="left"
    )

# --- CRITICAL FIX: Axis Constraints ---
fixed_axes = dict(
    xaxis=dict(range=[0, 2], autorange=False),
    yaxis=dict(range=[60, 160], autorange=False)
)

# --- 4. Build Animation Frames ---
frames = []
n_calib_frames = 20

# PHASE 1: Calibration
for k in range(n_calib_frames + 1):
    alpha = k / n_calib_frames
    Z_current = (1 - alpha) * Z_flat + alpha * Z_market
    
    status_text = f"Status: Calibrating Surface... {int(alpha*100)}%"
    frame_layout = go.Layout(
        annotations=static_annotations + [get_status_annotation(status_text, "yellow")],
        **fixed_axes 
    )

    frames.append(go.Frame(
        data=[
            go.Surface(x=S_mesh, y=T_mesh, z=Z_current, colorscale='Viridis', opacity=0.9, showscale=False),
            go.Scatter3d(x=S_flat, y=T_flat, z=Z_market_flat, mode='markers', marker=dict(size=4, color='cyan')),
            *[go.Scatter(x=[0], y=[S0], mode='lines', line=dict(color='cyan', width=1)) for _ in range(n_paths)]
        ],
        layout=frame_layout,
        name=f"calib_{k}"
    ))

# PHASE 2: Simulation
for k in range(1, n_steps_sim + 1):
    path_traces = []
    for p in range(n_paths):
        color = 'cyan' if p == 0 else 'rgba(0, 255, 255, 0.15)'
        width = 3 if p == 0 else 1
        path_traces.append(
            go.Scatter(x=time_axis[:k+1], y=paths[:k+1, p], mode='lines', line=dict(color=color, width=width))
        )

    status_text = "Status: Simulating Prices (Using Local Vol)"
    frame_layout = go.Layout(
        annotations=static_annotations + [get_status_annotation(status_text, "#00ff00")],
        **fixed_axes
    )

    frames.append(go.Frame(
        data=[
            go.Surface(x=S_mesh, y=T_mesh, z=Z_market, colorscale='Viridis', opacity=0.9, showscale=False),
            go.Scatter3d(x=S_flat, y=T_flat, z=Z_market_flat, mode='markers', marker=dict(size=4, color='cyan')),
            *path_traces
        ],
        layout=frame_layout,
        name=f"sim_{k}"
    ))

# --- 5. Initial Plot Setup ---
fig.add_trace(go.Surface(x=S_mesh, y=T_mesh, z=Z_flat, colorscale='Viridis', opacity=0.9, showscale=False), row=1, col=1)
fig.add_trace(go.Scatter3d(x=S_flat, y=T_flat, z=Z_market_flat, mode='markers', marker=dict(size=4, color='cyan'), name='Market Points'), row=1, col=1)

for _ in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[S0], mode='lines', line=dict(color='cyan', width=1), showlegend=False), row=1, col=2)

fig.frames = frames

# --- Layout Styling ---
fig.update_layout(
    # --- TEMPLATE FIX ---
    template="plotly_dark",  # Forces UI elements (like buttons) to handle dark/transparent backgrounds correctly
    
    title="Calibrated Local Volatility & Price Simulation",
    height=600, width=1200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    annotations=static_annotations + [get_status_annotation("Status: Waiting to Calibrate", "white")],
    
    scene=dict(
        xaxis_title='Strike (K)', yaxis_title='Time (T)', zaxis_title='Vol (σ)',
        xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
        yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
        zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, range=[0.15, 0.40]),
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.6))
    ),
    
    # --- FIXED RANGES (Global Layout) ---
    xaxis=dict(
        title='Time (Years)', 
        gridcolor='rgb(60, 60, 60)', 
        range=[0, 2],
        autorange=False
    ),
    yaxis=dict(
        title='Price ($)', 
        gridcolor='rgb(60, 60, 60)', 
        range=[60, 160],
        autorange=False
    ),
    
    # --- Styled Play Button ---
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'xanchor': 'center',
        'bgcolor': 'rgba(0,0,0,0)',       # Transparent Container
        'bordercolor': '#E0E0E0',         # Off-White Border
        'borderwidth': 1,
        'font': dict(color='#E0E0E0'),    # Off-White Text
        'showactive': False,              # Prevents "active" state from turning white
        'buttons': [{
            'label': '▶ Play Sequence',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()
```



$$
dS_t = S_t \, \mu \, dt + S_t \, \sigma_{LV}(S_t, t) \, dW_t
$$

---

#### 5.) 🌊 Carr-Madan (1999)
  
  - Option Valuation Using the Fast Fourier Transform
  
    $$
    C(k) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty \operatorname{Re} \left[ e^{-iu k} \frac{\phi(u-i(\alpha+1))}{\alpha^2 + \alpha - u^2 + i(2\alpha+1)u } \right] du
    $$
  
  **Key Result: Carr-Madan (1999) — "Option Valuation Using the Fast Fourier Transform"**
  
  In 1999, Peter Carr and Dilip Madan introduced a highly efficient framework for option pricing based on the Fast Fourier Transform (FFT). Their core idea was to re-express the price of an option as a Fourier integral leveraging the characteristic function $\phi(u)$ of the underlying’s log-price—a function easily computed in many models, including stochastic volatility models like Heston or jump-diffusion models.
  
  The Carr-Madan approach enables the pricing of entire strips of options (across strikes) in a single computational step. By damping the call price with an exponential factor and then transforming into Fourier space, option prices can be computed rapidly, facilitating model calibration and real-time risk management.
  
  The use of the FFT in option valuation is now standard in the quant toolkit, especially when direct formulas are unavailable. It is foundational for pricing under advanced models and underpins much of the efficiency in modern derivatives analytics.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from scipy.stats import norm

# ==========================================
# 1. Heston FFT Pricing Engine
# ==========================================

@dataclass
class HestonParams:
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Vol of vol
    rho: float    # Correlation
    v0: float     # Initial variance

def heston_cf(u, T, S0, r, q, p: HestonParams):
    """Characteristic function for Heston model."""
    i = 1j
    x0 = np.log(S0)
    a = p.kappa * p.theta
    b = p.kappa - p.rho * p.sigma * i * u
    d = np.sqrt(b*b + (p.sigma**2) * (i*u + u*u))
    g = (b - d) / (b + d)

    eDT = np.exp(-d * T)
    one_minus_g_eDT = 1 - g * eDT
    one_minus_g = 1 - g
    
    # Numerical stability guards
    one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
    one_minus_g = np.where(np.abs(one_minus_g) < 1e-15, 1e-15, one_minus_g)

    C = i*u*(r - q)*T + (a/(p.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
    D = ((b - d)/(p.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
    return np.exp(C + D*p.v0 + i*u*x0)

def _simpson_weights(N: int):
    if N % 2 != 0: raise ValueError("N must be even.")
    w = np.ones(N)
    w[1:N-1:2] = 4
    w[2:N-2:2] = 2
    return w

def heston_fft_calls(S0, T, r, q, p, N=4096, eta=0.25, alpha=1.5):
    """Calculate Call Prices using Carr-Madan FFT."""
    n = np.arange(N)
    v = eta * n
    i = 1j
    
    phi_shift = heston_cf(v - (alpha + 1)*i, T, S0, r, q, p)
    denom = (alpha**2 + alpha - v**2 + i*(2*alpha + 1)*v)
    psi = np.exp(-r*T) * phi_shift / denom
    
    w = _simpson_weights(N) * (eta / 3.0)
    lam = 2.0 * np.pi / (N * eta)
    b = 0.5 * N * lam
    x = psi * np.exp(1j * b * v) * w
    
    F = np.real(np.fft.fft(x))
    k = -b + np.arange(N) * lam
    K = np.exp(k)
    
    calls = np.exp(-alpha * k) / np.pi * F
    return K, np.maximum(calls, 0.0)

# ==========================================
# 2. Black-Scholes Inverse (For Vol Surface)
# ==========================================

def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(price, S, K, T, r):
    """Simple Newton-Raphson to find IV."""
    sigma = 0.3
    for _ in range(10):
        p = bs_call_price(S, K, T, r, sigma)
        diff = price - p
        if abs(diff) < 1e-5: return sigma
        v = bs_vega(S, K, T, r, sigma)
        if v == 0: break
        sigma += diff / v
    return sigma

# ==========================================
# 3. Setup Parameters & Surface Data
# ==========================================

# Define Model Parameters
params = HestonParams(
    kappa=2.0,   # Mean reversion speed
    theta=0.04,  # Long term variance (vol^2 = 0.2^2)
    sigma=0.5,   # Vol of Vol
    rho=-0.7,    # Negative correlation (leverage effect)
    v0=0.04      # Initial variance
)
S0, r, q = 100.0, 0.03, 0.0

# Generate Volatility Surface Data (Right Plot)
T_steps = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
strikes_target = np.linspace(70, 130, 20)
S_mesh, T_mesh = np.meshgrid(strikes_target, T_steps)
IV_surface = np.zeros_like(S_mesh)

print("Calibrating Vol Surface...")
for i, t_exp in enumerate(T_steps):
    K_fft, C_fft = heston_fft_calls(S0, t_exp, r, q, params)
    C_interpolated = np.interp(strikes_target, K_fft, C_fft)
    for j, k_str in enumerate(strikes_target):
        try:
            IV_surface[i, j] = implied_vol(C_interpolated[j], S0, k_str, t_exp, r)
        except:
            IV_surface[i, j] = np.nan

# ==========================================
# 4. Improved Heston Simulation (Log-Euler + Reflection)
# ==========================================

n_paths = 20
n_steps = 100
T_sim = 2.0
dt = T_sim / n_steps

# Generate Correlated Brownian Motions
# Z1, Z2 are independent standard normals
np.random.seed(42)
Z1 = np.random.normal(0, 1, (n_paths, n_steps))
Z2 = np.random.normal(0, 1, (n_paths, n_steps))

# Correlate them: dW_S uses Z1, dW_v uses mix of Z1 and Z2
dW_S = Z1 * np.sqrt(dt)
dW_v = (params.rho * Z1 + np.sqrt(1 - params.rho**2) * Z2) * np.sqrt(dt)

# Initialize Arrays
S = np.zeros((n_paths, n_steps + 1))
v = np.zeros((n_paths, n_steps + 1))
S[:, 0] = S0
v[:, 0] = params.v0

for t in range(n_steps):
    # --- 1. Variance Update (Reflection Scheme) ---
    # We use abs(v) to ensure the square root is real and the process bounces off 0
    v_curr = np.abs(v[:, t])
    sqrt_v = np.sqrt(v_curr)
    
    dv_step = params.kappa * (params.theta - v_curr) * dt + params.sigma * sqrt_v * dW_v[:, t]
    
    # Store absolute value for the next step (Reflection)
    v[:, t+1] = np.abs(v_curr + dv_step)
    
    # --- 2. Stock Update (Log-Euler Scheme) ---
    # d(ln S) = (r - 0.5*v)dt + sqrt(v)dW_S
    # This prevents negative stock prices and corrects for Ito drift
    log_ret = (r - 0.5 * v_curr) * dt + sqrt_v * dW_S[:, t]
    S[:, t+1] = S[:, t] * np.exp(log_ret)

time_grid = np.linspace(0, T_sim, n_steps + 1)

# ==========================================
# 5. Visualization Setup (Twin Axis)
# ==========================================

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"secondary_y": True}, {'type': 'scene'}]],
    column_widths=[0.5, 0.5],
    subplot_titles=("Simulated Processes (Heston Log-Euler)", "Resulting Volatility Surface")
)

# --- Right Panel: Static Surface (Trace Index 0) ---
fig.add_trace(go.Surface(
    x=strikes_target, y=T_steps, z=IV_surface,
    colorscale='Viridis', opacity=0.9, showscale=False
), row=1, col=2)

# --- Left Panel: Initial Traces (t=0) ---
# We add them in pairs: Stock (Index 2i+1), Variance (Index 2i+2)
for i in range(n_paths):
    # Stock Path (Primary Axis) - Cyan
    fig.add_trace(go.Scatter(
        x=[0], y=[S0], mode='lines',
        line=dict(color='cyan', width=1.5), opacity=0.6, showlegend=False
    ), row=1, col=1, secondary_y=False)
    
    # Variance Path (Secondary Axis) - Magenta
    fig.add_trace(go.Scatter(
        x=[0], y=[params.v0], mode='lines',
        line=dict(color='magenta', width=0.5), opacity=0.3, showlegend=False
    ), row=1, col=1, secondary_y=True)

# --- Frames ---
frames = []
# Create a reusable surface object for frames to save memory/processing
surface_trace = go.Surface(
    x=strikes_target, y=T_steps, z=IV_surface,
    colorscale='Viridis', opacity=0.9, showscale=False
)

for k in range(1, n_steps + 1):
    frame_data = []

    # 1. Add Surface (Must match Trace 0)
    frame_data.append(surface_trace)

    # 2. Add Left Panel Paths (Must match order of creation: Stock then Variance for each path)
    for i in range(n_paths):
        # Stock (Primary Axis)
        frame_data.append(go.Scatter(
            x=time_grid[:k+1],
            y=S[i, :k+1], 
            mode='lines',
            line=dict(color='cyan', width=1.5),
            opacity=0.6
        ))

        # Variance (Secondary Axis) - CRITICAL: Must specify yaxis='y2'
        frame_data.append(go.Scatter(
            x=time_grid[:k+1],
            y=v[i, :k+1], 
            mode='lines',
            line=dict(color='magenta', width=0.5),
            opacity=0.3,
            yaxis='y2' 
        ))

    # Update ALL traces: Surface (1) + 2 * n_paths
    indices = list(range(1 + 2 * n_paths))

    frames.append(go.Frame(
        data=frame_data,
        name=f"step{k}",
        traces=indices
    ))

# --- Slider Steps ---
sliders_steps = []
for k in range(1, n_steps + 1):
    sliders_steps.append(dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ))

# --- Layout Styling ---
fig.update_layout(
    template="plotly_dark",
    title="Heston Model: Log-Euler Discretization (Reflection) & FFT Pricing",
    height=600, width=1200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    
    # 3D Scene
    scene=dict(
        xaxis_title='Strike (K)',
        yaxis_title='Maturity (T)',
        zaxis_title='Implied Vol',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
        xaxis=dict(gridcolor='gray', backgroundcolor="rgb(20, 20, 20)", showbackground=True),
        yaxis=dict(gridcolor='gray', backgroundcolor="rgb(20, 20, 20)", showbackground=True),
        zaxis=dict(gridcolor='gray', backgroundcolor="rgb(20, 20, 20)", showbackground=True),
    ),
    
    # Left Panel: Primary Axis (Stock Price)
    xaxis=dict(title='Time (Years)', range=[0, T_sim], gridcolor='rgba(128,128,128,0.3)'),
    
    # Primary Y-Axis (Stock)
    yaxis=dict(
        title=dict(text='Stock Price ($)', font=dict(color="cyan")),
        range=[np.min(S)*0.9, np.max(S)*1.1], 
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(color="cyan")
    ),
    
    # Secondary Y-Axis (Variance)
    yaxis2=dict(
        title=dict(text='Variance process v(t)', font=dict(color="magenta")),
        range=[0, np.max(v)*1.2], 
        overlaying='y', 
        side='right',
        showgrid=False,
        tickfont=dict(color="magenta")
    ),

    # Animation Controls (Customized Play Button)
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'bgcolor': '#232946',            # A subtle dark blue, not fully transparent
        'bordercolor': 'black',        # Black accent border
        'borderwidth': 1.5,
        'font': dict(color='black', family="Arial Black"),    # Black play label text
        'active': 0,
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }],
            # Button color not directly supported, so use border and font colors
        }]
    }],
    
    sliders=[{
        'active': 0,
        'yanchor': 'top', 'xanchor': 'left',
        'currentvalue': {'prefix': 'Step: ', 'visible': True},
        'pad': {'b': 10, 't': 50},
        'len': 0.9, 'x': 0.05, 'y': -0.15,
        'steps': sliders_steps,
        'font': {'color': 'white'}
    }]
)

fig.frames = frames
fig.show()
```

    Calibrating Vol Surface...




 $$
 C_{\text{sim}}(S_0, K, T) = e^{-rT} \ \mathbb{E}^{\mathbb{Q}} \left[ (S_T - K)^+ \ \big| \ S_0, v_0 \right]
 $$
  
  $$
  C_{\text{fourier}}(S_0, K, T)
  = S_0 \left[ \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \operatorname{Re} \left( \frac{e^{-iu\ln K} \ \phi_1(u)}{iu} \right) du \right] 
  - K e^{-rT} \left[ \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \operatorname{Re} \left( \frac{e^{-iu\ln K} \ \phi_2(u)}{iu} \right) du \right]
  $$
 

---

#### 6.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**

- *Bachelier:*  We started with a dream
- *Sharpe:*  There is a skill to this thing, we can measure success
- *Black-Scholes:*  We found a consistent formula
- *Dupire:*  We made the formula usable in practice
- *Carr-Madan:* We found a way to calculate it all instantly

**Future Topics**

Technical Videos and Other Discussions

- Projects that Made me a Quant
- My First Year as a Quant
- Kalman Filter for Quant Finance
- Why Hedge Funds are Actually Secretive
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Poisson Processes for Quant Finance
- Top 3 Uses of Linear Algebra for Quant Finance
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- How to Backtest a Trading Strategy with Interactive Brokers

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
