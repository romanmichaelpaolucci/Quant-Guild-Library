### 📈 Quant Explains Algorithmic Market-Making

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

#### 1.) 🎲 Making a Dice Market

- Finding the mid price

- Quoting a spread

- Making a market

#### 2.) 📈 Making an Options Market

- Finding the mid price

- Quoting a spread (with adjustments)

- Making a market (and hedging)

#### 3.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 🎲 Making a Dice Market

You are a trader at a bank, a client requests a two way (bid, ask/offer) for a dice market

**Market-Making Basics**
 - **Bid**: price client sells at 
 - **Ask/Offer**: price client buys at 

In this market, traders can go long and short the value of a dice roll and once the die is rolled they will realize P/L

**For example,**

1.) Trader goes long 1,000 shares @ $3/share

2.) A dice is rolled and the outcome of 6 is realized

3.) The trader realizes a P/L of $3/share or $3,000

*What price should you quote them?*  

The outcome is *entirely random* can we still make money in a stable way?

##### Remark:  Our Statistical Edge / Trade Dictates the Drift of Our Wealth Process

 
 $$
 \mathbb{E}[\text{Trader P\&L}] > 0 \implies \text{Accumulate Positive Edge}
 $$

 $$
\mathbb{E}[\text{Trader P\&L}] < 0 \implies \text{Accumulate Negative Edge}
 $$



```python
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ---------------------------------------------
### CONFIGURATION LEVER ###
# Change this variable to switch the simulation type: "POSITIVE" or "NEGATIVE"
PLOT_EDGE_TYPE = "POSITIVE" 
# ---------------------------------------------

N_PATHS = 20     # Number of simulation paths to display
n_steps = 120    # Total time steps

# Parameters for edge calculation
liquidity_k = 0.85 
spread = 1
hit_rate = np.exp(-liquidity_k * spread)

# Define the theoretical EV for each scenario
EV_POSITIVE_STRONG = 1.0 # **INCREASED DRIFT: Changed from 0.5 to 1.0**
EV_NEGATIVE = -1.0 

# **Variability Parameter**
VOLATILITY = 0.3 

# Select parameters based on the configuration lever
if PLOT_EDGE_TYPE == "POSITIVE":
    EV_PER_TRADE = EV_POSITIVE_STRONG
    EDGE_LABEL = f"Strong Effective Market-Making (+{EV_PER_TRADE} EV/Trade)"
    PATH_COLOR = '#198754'  # Green
elif PLOT_EDGE_TYPE == "NEGATIVE":
    EV_PER_TRADE = EV_NEGATIVE
    EDGE_LABEL = f"Ineffective Market-Making ({EV_NEGATIVE} EV/Trade)"
    PATH_COLOR = '#DC3545'  # Red
else:
    raise ValueError("PLOT_EDGE_TYPE must be 'POSITIVE' or 'NEGATIVE'")

EV_AVG = EV_PER_TRADE * hit_rate # Average P&L per time step (Expected Value)

def calculate_sharpe(returns, risk_free_rate=0):
    """Calculates the Sharpe Ratio given a series of P&L values."""
    step_returns = np.diff(returns, prepend=0)
    avg_return = np.mean(step_returns)
    std_dev = np.std(step_returns)
    
    if std_dev == 0:
        return 0.0
    
    sharpe = (avg_return - risk_free_rate) / std_dev
    return sharpe

def simulate_edge_multiple(N_paths, n_steps, hit_rate, ev_per_trade, vol):
    """Runs multiple simulations and returns a list of P&L series."""
    pl_series_list = []
    time_steps = np.arange(1, n_steps + 1)
    
    for _ in range(N_PATHS):
        cum_pl = 0
        cum_pl_series = np.zeros_like(time_steps, dtype=float)
        
        for t in range(n_steps):
            if np.random.uniform() < hit_rate:
                trade_pl = ev_per_trade
            else:
                trade_pl = 0
            
            # Add random noise/volatility
            noise = np.random.normal(0, vol)
            
            cum_pl += trade_pl + noise
            cum_pl_series[t] = cum_pl
            
        pl_series_list.append(cum_pl_series)
        
    return time_steps, pl_series_list

# Run the selected simulation
time_steps, pl_paths = simulate_edge_multiple(N_PATHS, n_steps, hit_rate, EV_PER_TRADE, VOLATILITY)


# --- STYLING & COLOR CHOICES ---
AVG_LINE_COLOR = '#6AAED6'  
LINE_WIDTH_INDIVIDUAL = 2   
LINE_WIDTH_AVG = 4          
PATH_OPACITY = 0.3          
GRID_COLOR = '#505050'
FONT_COLOR = 'white' 
TRANSPARENT = 'rgba(0,0,0,0)'

# --- ZERO-JITTER AXIS SETUP (Fixed Ranges) ---
all_pl = np.concatenate(pl_paths)
GLOBAL_Y_MAX = max(np.max(all_pl), 0)
GLOBAL_Y_MIN = min(np.min(all_pl), 0)
GLOBAL_Y_DELTA = GLOBAL_Y_MAX - GLOBAL_Y_MIN
GLOBAL_Y_PAD = max(GLOBAL_Y_DELTA * 0.15, 2) 
STABLE_Y_RANGE = [GLOBAL_Y_MIN - GLOBAL_Y_PAD, GLOBAL_Y_MAX + GLOBAL_Y_PAD]

X_MARGIN = 5
STABLE_X_RANGE = [0, n_steps + X_MARGIN]

# --- CREATE SUBPLOTS FIGURE ---
fig_anim = make_subplots(
    rows=1, cols=1,
    subplot_titles=(EDGE_LABEL,), # Dynamic title
)

# --- CREATE ANIMATION FRAMES ---
frames = []

frame_steps = (
    list(range(1, min(20, n_steps) + 1)) +
    list(range(21, min(60, n_steps) + 1, 3)) +
    list(range(61, n_steps + 1, 5))
)

for i in frame_steps:
    x = time_steps[:i]
    
    # Calculate Sharpe Ratio for the current time step (i)
    current_paths_pl = [path[:i] for path in pl_paths]
    current_sharpe_ratios = [calculate_sharpe(path) for path in current_paths_pl]
    avg_sharpe = np.mean(current_sharpe_ratios) if current_sharpe_ratios else 0.0
    
    frame_data = []

    # Individual Paths
    for k in range(N_PATHS):
        frame_data.append(
            go.Scatter(x=x, y=pl_paths[k][:i], mode="lines", 
                       line=dict(color=PATH_COLOR, width=LINE_WIDTH_INDIVIDUAL), opacity=PATH_OPACITY, 
                       name=f"Path {k}", legendgroup='path', showlegend=False)
        )
    # Theoretical Average Line
    frame_data.append(
        go.Scatter(x=x, y=x * EV_AVG, mode="lines", 
                   line=dict(color=AVG_LINE_COLOR, width=LINE_WIDTH_AVG, dash='dot'), 
                   name="Theoretical Avg", legendgroup='path', showlegend=False)
    )

    frame = go.Frame(
        data=frame_data,
        name=str(i),
        layout=go.Layout(
            # Annotation showing AVERAGE SHARPE - Adjusted position
            annotations=[
                dict(
                    x=STABLE_X_RANGE[0] + (X_MARGIN * 0.5), # Shifted left (near x=0)
                    y=STABLE_Y_RANGE[1] - (GLOBAL_Y_PAD * 2.5), # Shifted down from the top edge
                    xref="x1", yref="y1",
                    text=f"Avg Sharpe (T={i}): {avg_sharpe:.3f}", 
                    showarrow=False,
                    font=dict(color=FONT_COLOR, size=16), 
                    bgcolor="rgba(0,0,0,0.5)", 
                    bordercolor=FONT_COLOR,
                    xanchor='left', 
                    yanchor='top' 
                ),
            ]
        )
    )
    frames.append(frame)

# --- INITIAL TRACES FOR BASE FIGURE ---
initial_i = frame_steps[0]
init_x = time_steps[:initial_i]

# Traces for the legend 
fig_anim.add_trace(go.Scatter(x=[0], y=[0], mode='lines', 
                              line=dict(color=PATH_COLOR, width=LINE_WIDTH_INDIVIDUAL), opacity=PATH_OPACITY,
                              name='Individual Path (20x)', legendgroup='legend', showlegend=True), row=1, col=1)

fig_anim.add_trace(go.Scatter(x=[0], y=[0], mode='lines', 
                              line=dict(color=AVG_LINE_COLOR, width=LINE_WIDTH_AVG, dash='dot'), 
                              name='Theoretical Average', legendgroup='legend', showlegend=True), row=1, col=1)

# Add initial paths
for k in range(N_PATHS):
    fig_anim.add_trace(go.Scatter(x=init_x, y=pl_paths[k][:initial_i], mode='lines', 
                                  line=dict(color=PATH_COLOR, width=LINE_WIDTH_INDIVIDUAL), opacity=PATH_OPACITY,
                                  name=f'Path {k}', legendgroup='path', showlegend=False), row=1, col=1)

# Add initial average line
fig_anim.add_trace(go.Scatter(x=init_x, y=init_x * EV_AVG, mode="lines", 
                              line=dict(color=AVG_LINE_COLOR, width=LINE_WIDTH_AVG, dash='dot'), 
                              name="Theoretical Avg", legendgroup='path', showlegend=False), row=1, col=1)


# --- LAYOUT CONFIGURATION ---
title_text = f"✨ Ergodic Convergence: Accumulation of {PLOT_EDGE_TYPE.capitalize()} Edge" 

fig_anim.update_layout(
    title=dict(
        text=title_text, 
        x=0.5, font=dict(color=FONT_COLOR, size=24, family="Arial, sans-serif")
    ),
    plot_bgcolor=TRANSPARENT, 
    paper_bgcolor=TRANSPARENT, 
    font=dict(color=FONT_COLOR, size=12),
    margin=dict(l=40, r=40, t=80, b=40),
    height=600,
    
    # Configure X and Y axes for JITTER-FREE animation (Fixed Ranges)
    xaxis=dict(
        title="Time Step (N)", showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, 
        range=STABLE_X_RANGE, title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
    ),
    yaxis=dict(
        title="Cumulative P&L", showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, 
        range=STABLE_Y_RANGE, tickformat=".2f", 
        title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
    ),
    
    # Zero line
    shapes=[
        dict(type='line', xref='x', yref='y', x0=0, y0=0, x1=1, y1=0, 
             line=dict(color=GRID_COLOR, width=1.5, dash='dash')),
    ],
    
    legend=dict(x=0, y=1.05, orientation="h", bgcolor='rgba(0,0,0,0)', font=dict(color=FONT_COLOR), tracegroupgap=30),
    
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x':0.5, 'y':-0.1, 'xanchor': 'center', 'yanchor': 'top',
        'buttons':[{
            'label': '▶️ Play Animation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 5, 'easing': 'linear'}, 
                'mode': 'immediate'
            }]
        }]
    }]
)

# --- FINALIZE AND SHOW ---
fig_anim.frames = frames

# Footer annotation with parameters
ev_sign = '+' if EV_PER_TRADE > 0 else ''
fig_anim.add_annotation(
    x=0.99, y=-0.15, xref='paper', yref='paper',
    text=f"EV per Trade: {ev_sign}{EV_PER_TRADE}. Hit Rate={hit_rate:.3f}. Volatility={VOLATILITY}",
    showarrow=False, font=dict(color='#888888', size=10),
    xanchor='right', yanchor='top'
)

fig_anim.show()
```



##### How Do We Make a Market Such That We Accumulate Positive Edge?

###### ______________________________________________________________________________________________________________________________________

##### Finding the Mid Price

If the outcome is random, what is our best guess?

We can choose whatever price we would like for the mid price, let's denote

$$\theta = \text{Mid Price}$$

Let's look at the error function given by the mean squared error

 $$
 \text{MSE}(\theta) = \mathbb{E} \left[(\theta - x)^2\right] = \frac{1}{6} \sum_{x=1}^6 (\theta - x)^2
 $$
 
The optimal $\theta$ is the value that minimizes this MSE; that is,
$$
\theta^* = \underset{\theta}{\arg\min} \, \text{MSE}(\theta)
$$




```python
import numpy as np
import plotly.graph_objs as go

# Define parameter space for mid price (theta)
theta = np.linspace(-3, 9, 400)

# Dice outcomes: 1 through 6
dice_outcomes = np.arange(1, 7)

# Calculate theoretical MSE for each theta
mse = np.array([(1/6) * np.sum((t - dice_outcomes) ** 2) for t in theta])

# Find theta that minimizes MSE (should be mean of dice)
theta_star = dice_outcomes.mean()
min_mse = (1/6) * np.sum((theta_star - dice_outcomes) ** 2)

fig = go.Figure()

# Plot MSE curve
fig.add_trace(
    go.Scatter(
        x=theta,
        y=mse,
        mode='lines',
        line=dict(color='deepskyblue', width=3),
        name='MSE vs Mid Price'
    )
)

# Mark optimal theta
fig.add_trace(
    go.Scatter(
        x=[theta_star],
        y=[min_mse],
        mode='markers',
        marker=dict(size=20, color='orange', symbol='star'),
        name='Optimal Mid Price'
    )
)

# Vertical line at optimal theta
fig.add_trace(
    go.Scatter(
        x=[theta_star, theta_star],
        y=[mse.min() - 0.5, mse.max()],
        mode='lines',
        line=dict(color='orange', dash='dash', width=2),
        showlegend=False
    )
)

fig.update_layout(
    title=f"MSE of Mid Price vs Dice Roll Outcome (Min at θ = {theta_star:.1f})",
    xaxis=dict(
        title="Mid Price (θ)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    yaxis=dict(
        title="Mean Squared Error",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    width=750,
    height=410,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.18,
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    )
)

fig.show()

```



##### Remark:  The value that minimizes the MSE here is given by $3.5$ which is equivalent to the dice roll's expectation!
$$
   \mathbb{E}[X] = \sum_{x} x\,p(x) =\frac{1}{6} \sum_{i=1}^6 x_i = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5
$$


###### ______________________________________________________________________________________________________________________________________

**Why is the MSE minimized at the expectation?**
 
 The Mean Squared Error (MSE) when estimating the dice outcome with price $\theta$ is:
 
 $$
 \mathrm{MSE}(\theta) = \mathbb{E}\left[(X - \theta)^2\right]
 $$
 
 Expanding this:
 
 $$
 \mathrm{MSE}(\theta) = \mathbb{E}[X^2 - 2X\theta + \theta^2] = \mathbb{E}[X^2] - 2\theta \mathbb{E}[X] + \theta^2
 $$
 
 Since $\mathbb{E}[X]$ and $\mathbb{E}[X^2]$ are constants with respect to $\theta$, this is a quadratic in $\theta$:
 
 $$
 \mathrm{MSE}(\theta) = \theta^2 - 2\mathbb{E}[X]\theta + \mathbb{E}[X^2]
 $$
 
 To minimize, take the derivative with respect to $\theta$ and set it to zero:
 
 $$
 \frac{d}{d\theta} \mathrm{MSE}(\theta) = 2\theta - 2\mathbb{E}[X] = 0
 $$
 
 So,
 
 $$
 \theta^\star = \mathbb{E}[X]
 $$
 
 Therefore, the MSE is minimized when $\theta$ equals the expected value of $X$—the mean of the dice roll. 
 
 This is why the optimal mid price is the average outcome!

 ##### We can visualize this numerically and watch convergence to this solution from calculus!


```python
import plotly.graph_objs as go
import numpy as np

# Function for MSE and its gradient
def mse_fn(theta):
    return (1/6) * np.sum((theta - np.arange(1,7))**2)
def grad_mse(theta):
    return (1/3) * np.sum(theta - np.arange(1,7))

# Gradient descent setup
theta_vals = []
tangent_lines = []
mse_vals = []

theta_init = -2.0
theta = theta_init
alpha = 0.15  # Take much smaller steps!
n_steps = 18  # Do more steps to show smaller advancements

for i in range(n_steps):
    theta_vals.append(theta)
    mse_vals.append(mse_fn(theta))
    grad = grad_mse(theta)
    # Tangent line at this theta -- extend very far (well beyond plot axes)
    tangent_x = np.linspace(-20, 20, 2)
    tangent_y = mse_fn(theta) + grad*(tangent_x-theta)
    tangent_lines.append((tangent_x, tangent_y))
    theta -= alpha * grad

theta_star = np.mean(np.arange(1,7))

# Set up frames for Plotly animation
frames = []
theta_range = np.linspace(-3, 9, 400)
all_mses = (1/6) * np.sum((theta_range[:,None] - np.arange(1,7))**2, axis=1)

# Compute tight ylim based on minimum and maximum of the *actual* plotted curve
mse_margin = 0.2
ymin = all_mses.min() - mse_margin - 5
ymax = all_mses.max() + mse_margin

for i in range(n_steps):
    t = theta_vals[i]
    grad = grad_mse(t)
    tangent_x, tangent_y = tangent_lines[i]
    frame_data = [
        go.Scatter(x=theta_range, y=all_mses, mode='lines', line=dict(color='deepskyblue', width=3), name='MSE vs Mid Price'),
        go.Scatter(x=[theta_star], y=[mse_fn(theta_star)], mode='markers',
                   marker=dict(size=20, color='orange', symbol='star'), name='Optimal Mid Price'),
        go.Scatter(x=[theta_star, theta_star],
                   y=[ymin, ymax],
                   mode='lines', line=dict(color='orange', dash='dash', width=2), showlegend=False),
        go.Scatter(x=[t], y=[mse_fn(t)], mode='markers', marker=dict(size=7, color='magenta'), name='Current Guess'),
        go.Scatter(x=tangent_x, y=tangent_y, mode='lines', line=dict(color='magenta', width=2, dash='dot'),
                   name='Tangent (Gradient)')
    ]
    frames.append(go.Frame(data=frame_data, name=f'step{i}', traces=[0,1,2,3,4]))

# Initial figure setup
fig = go.Figure(
    data=frames[0].data,
    frames=frames,
)
fig.update_layout(
    title="Gradient Descent Finds θ Minimizing MSE (Expectation of Dice Roll)",
    xaxis=dict(
        title="Mid Price (θ)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        range=[-3, 9],
    ),
    yaxis=dict(
        title="Mean Squared Error",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        range=[ymin, ymax],
    ),
    width=750,
    height=410,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.38,    # Move legend lower; decreased from -0.15 for more separation
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    ),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "x": 1.08, "y": 1.2,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 800,"redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                },
            ],
        }
    ]
)
# Remove or hide the step slider by not passing 'sliders' to update_layout
# (simply comment out or delete the slider code below)
# sliders = [
#     {
#         "steps": [
#             {
#                 "method": "animate",
#                 "label": f"{i+1}",
#                 "args": [[f"step{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
#             }
#             for i in range(n_steps)
#         ],
#         "active": 0,
#         "x": 0.12,
#         "xanchor": "left",
#         "y": -0.26,
#         "yanchor": "top"
#     }
# ]
# fig.update_layout(sliders=sliders)

fig.show()

```



##### Stability in our Mid Price

In the context of this example, our mid price does not change

The distribution is static along with all of the associated probabilities and statistics


```python
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np

# Dice setup
dice_faces = np.arange(1, 7)
pmf = np.ones_like(dice_faces) / 6
true_mean = dice_faces.mean()  # 3.5

# Simulate dice rolls
n_rolls = 30
rng = np.random.default_rng(seed=1)
rolls = rng.choice(dice_faces, size=n_rolls)

# Compute running mean for LLN visualization
running_means = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

# For yaxis2 range padding (right/empirical mean plot)
min_runmean = running_means.min()
max_runmean = running_means.max()
# Add extra padding for visibility around the displayed text
y2_padding = 1.0
y2_range = [
    min(min_runmean, true_mean) - y2_padding,
    max(max_runmean, true_mean) + y2_padding
]

frames = []
for i in range(n_rolls):
    # PMF left panel (bars: always the same, highlight latest outcome)
    colors = ["deepskyblue"] * 6
    colors[rolls[i] - 1] = "orange"
    pmf_trace = go.Bar(
        x=dice_faces,
        y=pmf,
        marker=dict(color=colors),
        width=0.7,
        name="PMF",
        xaxis="x1", yaxis="y1"
    )
    highlight_dot = go.Scatter(
        x=[rolls[i]],
        y=[pmf[rolls[i] - 1] + 0.02],
        mode="markers+text",
        marker=dict(size=18, color='orange', symbol='star'),
        text=[f"{rolls[i]}"],
        textposition="top center",
        showlegend=False,
        xaxis="x1", yaxis="y1"
    )

    # Running mean up to this step
    mean_trace = go.Scatter(
        x=np.arange(1, i + 2),
        y=running_means[:i + 1],
        mode="lines+markers",
        line=dict(color="magenta", width=3),
        marker=dict(size=10, color="magenta"),
        name="Empirical Mean",
        xaxis="x2", yaxis="y2"
    )
    # Current point marker as a smaller magenta circle (not star) for empirical mean plot
    mean_now = go.Scatter(
        x=[i + 1],
        y=[running_means[i]],
        mode="markers+text",
        marker=dict(size=12, color="magenta", symbol="circle"),
        text=[f"{running_means[i]:.2f}"],
        textposition="bottom center",
        showlegend=False,
        xaxis="x2", yaxis="y2"
    )
    # Theoretical mean line
    true_mean_line = go.Scatter(
        x=[1, n_rolls],
        y=[true_mean, true_mean],
        mode="lines",
        line=dict(color="deepskyblue", dash="dash", width=2),
        name="Theoretical Mean",
        xaxis="x2", yaxis="y2"
    )

    # Use the fixed y2_range for the empirical mean plot (so padding is consistent)
    frames.append(go.Frame(
        data=[pmf_trace, highlight_dot, mean_trace, mean_now, true_mean_line],
        name=f"step{i}",
        layout=go.Layout(
            xaxis2=dict(
                range=[0.5, n_rolls + 0.5],
                title="Number of Rolls",
                showgrid=True,
                gridcolor="#323232",
                gridwidth=1
            ),
            yaxis2=dict(
                range=y2_range,
                title="Empirical Mean of Rolls",
                showgrid=True,
                gridcolor="#323232",
                gridwidth=1
            )
        )
    ))

# Make the figure
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Fixed Distribution PMF", "Empirical Mean Converges (LLN)"))

# Add initial traces
init_traces = [
    go.Bar(
        x=dice_faces, y=pmf, marker=dict(color=["deepskyblue"]*6), width=0.7, name="PMF"
    ),
    go.Scatter(
        x=[rolls[0]], y=[pmf[rolls[0] - 1] + 0.02], mode='markers+text',
        marker=dict(size=18, color='orange', symbol='star'), text=[f"{rolls[0]}"],
        textposition="top center", showlegend=False
    ),
    go.Scatter(
        x=[1], y=[running_means[0]], mode="lines+markers", line=dict(color="magenta", width=3),
        marker=dict(size=10, color="magenta"), name="Empirical Mean"
    ),
    # Right subplot current mean point as smaller magenta circle (not star)
    go.Scatter(
        x=[1], y=[running_means[0]], mode="markers+text",
        marker=dict(size=12, color="magenta", symbol="circle"), text=[f"{running_means[0]:.2f}"],
        textposition="bottom center", showlegend=False
    ),
    go.Scatter(
        x=[1, n_rolls], y=[true_mean, true_mean],
        mode="lines", line=dict(color="deepskyblue", dash="dash", width=2), name="Theoretical Mean"
    )
]

fig.add_trace(init_traces[0], row=1, col=1)
fig.add_trace(init_traces[1], row=1, col=1)
fig.add_trace(init_traces[2], row=1, col=2)
fig.add_trace(init_traces[3], row=1, col=2)
fig.add_trace(init_traces[4], row=1, col=2)

fig.frames = frames

fig.update_layout(
    title="Law of Large Numbers: Empirical Mean Converges, Distribution is Invariant",
    width=1020, height=410,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=1.08,
        xanchor='center', x=0.5,
        font=dict(color="white")
    ),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.5,
            "y": -0.16,
            "xanchor": "center",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 300, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 250, "easing": "linear"}
                        }
                    ]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                },
            ],
        }
    ],
)

# Left subplot (PMF): always the same
fig.update_xaxes(
    dict(title="Dice Face", range=[0.5, 6.5], tickvals=dice_faces, showgrid=True, gridcolor="#323232", gridwidth=1),
    row=1, col=1
)
fig.update_yaxes(
    dict(title="Probability", range=[0, 0.23], showgrid=True, gridcolor="#323232", gridwidth=1),
    row=1, col=1
)

# Right subplot (Empirical mean)
fig.update_xaxes(
    dict(title="Number of Rolls", range=[0.5, n_rolls + 0.5], showgrid=True, gridcolor="#323232", gridwidth=1),
    row=1, col=2
)
fig.update_yaxes(
    dict(
        title="Empirical Mean of Rolls",
        range=y2_range,
        showgrid=True, gridcolor="#323232", gridwidth=1
    ),
    row=1, col=2
)

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### Quoting a Spread

 Let $B$ be the bid price, $A$ the ask price, and $V$ the true (unknown) value.

 Assume that with equal probability, you will either:
 - **Buy** from someone at your quoted bid $B$ (the other party sells to you), **or**
 - **Sell** to someone at your quoted ask $A$ (the other party buys from you).

 By the law of total expectation (averaging over both possible trades and the true value):

 $$
 \mathbb{E}[\text{Profit per trade}] = \frac{1}{2} \mathbb{E}[A - V] + \frac{1}{2} \mathbb{E}[V - B]
 $$

 where:
 - When **selling** at $A$, profit is $A - V$;
 - When **buying** at $B$, profit is $V - B$.

 Assuming you do not know $V$ exactly (your best estimate is $\mathbb{E}[V]$):

 $$
 \mathbb{E}[\text{Profit per trade}] = \frac{1}{2} (A - \mathbb{E}[V]) + \frac{1}{2} (\mathbb{E}[V] - B) = \frac{A - B}{2}
 $$

 So, your **expected profit per trade is half the quoted spread**:
 $$
 \boxed{
 \mathbb{E}[\text{Profit per trade}] = \frac{A - B}{2} = \frac{\text{Spread}}{2}
 }
 $$


##### Let's make a market with an extremely wide spread so we can have the highest EV per trade! 


```python
import numpy as np
import plotly.graph_objs as go

# -- Dice Market Animation: Simulate market-making with a wide quoted spread --
np.random.seed(2025)

true_value = 3.5
bid = 2
ask = 7
spread = ask - bid

n_steps = 120    # Total time steps
liquidity_k = 0.85 
hit_rate = np.exp(-liquidity_k * spread)

pl_history = []
trade_inds = []
trade_profits = []
trade_types = []  

cum_pl = 0
time_steps = np.arange(1, n_steps + 1)
cum_pl_series = np.zeros_like(time_steps, dtype=float)

for t in range(n_steps):
    profit = 0
    if np.random.uniform() < hit_rate:
        if np.random.rand() < 0.5:
            # You sell at ask
            roll = np.random.randint(1, 7)
            profit = ask - roll
            trade_types.append('ask')
        else:
            # You buy at bid
            roll = np.random.randint(1, 7)
            profit = roll - bid
            trade_types.append('bid')
        
        cum_pl += profit
        trade_inds.append(t)
        trade_profits.append(profit)
    
    cum_pl_series[t] = cum_pl


# --- STYLING & COLOR CHOICES ---
LINE_COLOR = '#6AAED6'      
PROFIT_COLOR = '#198754'    
LOSS_COLOR = '#DC3545'      
TRADE_MARKER_COLOR = '#FFC300' 
HIGHLIGHT_SIZE = 10
LINE_WIDTH = 3.5
GRID_COLOR = '#505050'
FONT_COLOR = 'white' 
STAR_SIZE = 18

# --- ZERO-JITTER AXIS SETUP ---

# 1. Calculate the final, stable Y-range based on the entire data set.
GLOBAL_Y_MAX = max(np.max(cum_pl_series), 0)
GLOBAL_Y_MIN = min(np.min(cum_pl_series), 0)
GLOBAL_Y_DELTA = GLOBAL_Y_MAX - GLOBAL_Y_MIN
GLOBAL_Y_PAD = max(GLOBAL_Y_DELTA * 0.1, 1.5)
STABLE_Y_RANGE = [GLOBAL_Y_MIN - GLOBAL_Y_PAD, GLOBAL_Y_MAX + GLOBAL_Y_PAD]

# 2. Calculate the final, stable X-range.
X_MARGIN = 5
STABLE_X_RANGE = [0, n_steps + X_MARGIN]

# --- CREATE ANIMATION FRAMES ---
frames = []

# Strategic stepping
frame_steps = (
    list(range(1, min(20, n_steps) + 1)) +
    list(range(21, min(60, n_steps) + 1, 3)) +
    list(range(61, n_steps + 1, 5))
)

for i in frame_steps:
    x = time_steps[:i]
    y = cum_pl_series[:i]
    current_pl = y[-1]
    current_point_color = PROFIT_COLOR if current_pl >= 0 else LOSS_COLOR
    
    trades_up_to_i = [t_ind for t_ind in trade_inds if t_ind < i]
    trade_x = [time_steps[t_ind] for t_ind in trades_up_to_i]
    trade_y = [cum_pl_series[t_ind] for t_ind in trades_up_to_i]
    
    # NOTE: The frame.layout section is REMOVED to prevent range updates,
    # thereby eliminating jitter. The axes are fixed by the main layout below.
    frame = go.Frame(
        data=[
            # 1. Main Path Trace
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=LINE_COLOR, width=LINE_WIDTH), name="Cumulative P&L"),
            # 2. Trade Event Markers
            go.Scatter(x=trade_x, y=trade_y, mode='markers', marker=dict(size=HIGHLIGHT_SIZE, color=TRADE_MARKER_COLOR, symbol='circle-open', line=dict(width=1.5, color=TRADE_MARKER_COLOR)), name='Trades'),
            # 3. Current Point Highlight
            go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers', marker=dict(size=STAR_SIZE, color=current_point_color, symbol='star', line=dict(width=2, color=FONT_COLOR)), name='Current Step')
        ],
        name=str(i),
        # Layout only contains the annotation for the frame (no axis ranges)
        layout=go.Layout(
            annotations=[
                dict(
                    x=x[-1], y=y[-1], xref="x", yref="y",
                    text=f"Time {i}, P&L {current_pl:.2f}",
                    showarrow=True, arrowhead=1, ax=0, ay=-40,
                    font=dict(color=current_point_color, size=14),
                    bgcolor="rgba(0,0,0,0.5)", bordercolor=current_point_color
                )
            ]
        )
    )
    frames.append(frame)

# --- BASE FIGURE FOR ANIMATION ---
initial_i = frame_steps[0]
initial_x = time_steps[:initial_i]
initial_y = cum_pl_series[:initial_i]
initial_pl = initial_y[-1]
initial_current_point_color = PROFIT_COLOR if initial_pl >= 0 else LOSS_COLOR

trades_init = [t_ind for t_ind in trade_inds if t_ind < initial_i]
trades_x_init = [time_steps[t_ind] for t_ind in trades_init]
trades_y_init = [cum_pl_series[t_ind] for t_ind in trades_init]


fig_anim = go.Figure(
    data = [
        go.Scatter(x=initial_x, y=initial_y, mode='lines', line=dict(color=LINE_COLOR, width=LINE_WIDTH), name='Cumulative P&L'),
        go.Scatter(x=trades_x_init, y=trades_y_init, mode='markers', marker=dict(size=HIGHLIGHT_SIZE, color=TRADE_MARKER_COLOR, symbol='circle-open', line=dict(width=1.5, color=TRADE_MARKER_COLOR)), name='Trades'),
        go.Scatter(x=[initial_x[-1]], y=[initial_y[-1]], mode="markers", marker=dict(size=STAR_SIZE, color=initial_current_point_color, symbol='star', line=dict(width=2, color=FONT_COLOR)), name='Current Step')
    ],
    layout=go.Layout(
        title=dict(
            text="🎲 Dice Market-Making P&L Simulation: Wide Spread (Bid=2, Ask=7)",
            x=0.5, font=dict(color=FONT_COLOR, size=24, family="Arial, sans-serif")
        ),
        xaxis=dict(
            title="Time Step (Order Opportunity)",
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, zeroline=False,
            range=STABLE_X_RANGE, # Fixed range here
            title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
        ),
        yaxis=dict(
            title="Cumulative Profit & Loss (P&L)",
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            range=STABLE_Y_RANGE, # Fixed range here
            tickformat=".2f", title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
        ),
        # Professional zero line
        shapes=[
            dict(
                type='line', xref='paper', yref='y', x0=0, y0=0, x1=1, y1=0,
                line=dict(color=GRID_COLOR, width=1.5, dash='dash')
            )
        ],
        font=dict(color=FONT_COLOR, size=12),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=50, r=40, t=80, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color=FONT_COLOR)),
        
        # Subtle footer annotation for parameters
        annotations=[
            dict(
                x=0.99, y=-0.1, xref='paper', yref='paper',
                text=f"True Value=3.5, Spread={spread}, Hit Rate={hit_rate:.3f}",
                showarrow=False, font=dict(color='#888888', size=10),
                xanchor='right', yanchor='top'
            )
        ],
        
        # Play button styling
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x':0.5, 'y':-0.21, 'xanchor': 'center', 'yanchor': 'top',
            'buttons':[{
                'label': '▶️ Play Animation',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 40, 'redraw': True},
                    'fromcurrent': True,
                    # Added a transition duration to smooth the path growth
                    'transition': {'duration': 5, 'easing': 'linear'}, 
                    'mode': 'immediate'
                }]
            }]
        }]
    ),
    frames=frames
)

fig_anim.show()
```



##### Remark: Theoretical Edge is not the same as Edge in Reality


```python
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# Dice roll values
dice_faces = np.arange(1, 7)
pmf = np.ones_like(dice_faces) / 6
true_value = dice_faces.mean()  # 3.5

# ----- Market: any spread clears (equal-prob buyer & seller) -----
spread_range = np.linspace(0.01, 5, 100)
# In the idealized naive world, each trade is equally likely at bid or ask
# This trader earns (ask-true_value) if selling and (true_value-bid) if buying; both = spread/2
expected_profit_per_trade = spread_range / 2

# ----- More realistic: trade volume decays as spread widens -----
def simple_liquidity_curve(spread, k=0.8):
    """Fraction of max volume as a function of spread, steeper with higher k."""
    # Exponential decay: for k~0.8, at spread=3, volume is ~10% of max
    return np.exp(-k * spread)

liquidity = simple_liquidity_curve(spread_range)
expected_profit_realistic = (spread_range / 2) * liquidity

# Set up subplot
fig2 = make_subplots(
    rows=1, cols=2, subplot_titles=(
        "Naive Model: Wider Spread, Higher Profit",
        "Realistic Model: Too Wide, Profit Shrinks"
    ),
    horizontal_spacing=0.11
)

# --- Naive: profit always grows with spread ---
fig2.add_trace(
    go.Scatter(
        x=spread_range,
        y=expected_profit_per_trade,
        mode='lines',
        line=dict(color='deepskyblue', width=4),
        name='Expected Profit per Trade',
    ),
    row=1, col=1
)

# Show example points: common spreads (e.g. fair, $1, $2) - For Plot 1
for s in [0.5, 1, 2, 3]:
    fig2.add_trace(
        go.Scatter(
            x=[s],
            y=[s/2],
            mode="markers+text",
            marker=dict(size=12, color='orange'),
            text=[f"${s:.1f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=1
    )

# --- Realistic: volume suppression with spread ---
fig2.add_trace(
    go.Scatter(
        x=spread_range,
        y=expected_profit_realistic,
        mode='lines',
        line=dict(color='violet', width=4),
        name='Expected Profit (Volume Adjusted)',
    ),
    row=1, col=2
)
# Envelope for max profit per trade (dashed) - REMOVED AS REQUESTED
# fig2.add_trace(
#     go.Scatter(
#         x=spread_range,
#         y=expected_profit_per_trade,
#         mode='lines',
#         line=dict(color='deepskyblue', width=2, dash='dash'),
#         name='(No Volume Loss)',
#         showlegend=False
#     ),
#     row=1, col=2
# )

# Show effect of decay on a few spreads - For Plot 2
for s in [0.5, 2, 3]:
    y0 = (s/2) * simple_liquidity_curve(s)
    fig2.add_trace(
        go.Scatter(
            x=[s],
            y=[y0],
            mode="markers+text",
            marker=dict(size=12, color='orange'),
            text=[f"${s:.1f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=2
    )
    # Thin dashed vertical lines
    fig2.add_trace(
        go.Scatter(
            x=[s, s],
            y=[0, y0],
            mode='lines',
            line=dict(color="gray", width=1, dash='dot'),
            showlegend=False
        ),
        row=1, col=2
    )

# Shaded area for "sweet spot" in realistic plot (max profit region)
max_idx = np.argmax(expected_profit_realistic)
spread_at_max = spread_range[max_idx]
profit_at_max = expected_profit_realistic[max_idx]
fig2.add_trace(
    go.Scatter(
        x=[spread_at_max],
        y=[profit_at_max],
        mode="markers+text",
        # Marker is size 18 and orange
        marker=dict(size=18, color='orange', symbol='star'),
        text=["Maximum"],
        # Text font is orange
        textfont=dict(color="orange"),
        textposition="bottom center",
        showlegend=False
    ),
    row=1, col=2
)

# Formatting to match the style
fig2.update_layout(
    title="Expected Profit vs. Spread Choice in Dice Market",
    width=780, height=370,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(t=56, r=14, b=48, l=56),
    xaxis1=dict(
        title="Spread ($)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        zeroline=False,
        range=[-0.1, 5.01]
    ),
    yaxis1=dict(
        title="Expected Profit per Trade ($)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        zeroline=False,
        range=[0, expected_profit_per_trade.max() + 0.2]
    ),
    xaxis2=dict(
        title="Spread ($)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        zeroline=False,
        range=[-0.1, 5.01]
    ),
    yaxis2=dict(
        title="Expected Profit per Trade ($)",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        zeroline=False,
        # Y-axis range is [0, 0.4]
        range=[0, 0.35]
    ),
    # Legend is beneath the chart
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.2,
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    )
)
fig2.update_annotations(font=dict(color='white'))
```



##### When we provide a proper spread


```python
import numpy as np
import plotly.graph_objs as go

# -- Dice Market Animation: Simulate market-making with a TIGHTER quoted spread --
np.random.seed(2025)

# --- SIMULATION PARAMETERS ---
true_value = 3.5
bid = 3  # Tighter Bid
ask = 4  # Tighter Ask
spread = ask - bid # Spread = 1

n_steps = 120    # Total time steps
liquidity_k = 0.85  
# The hit rate is calculated based on the new spread=1. 
# It will be much higher than the previous spread=5 (~42% vs ~1.4%)
hit_rate = np.exp(-liquidity_k * spread) 

pl_history = []
trade_inds = []
trade_profits = []
trade_types = []  

cum_pl = 0
time_steps = np.arange(1, n_steps + 1)
cum_pl_series = np.zeros_like(time_steps, dtype=float)

for t in range(n_steps):
    profit = 0
    if np.random.uniform() < hit_rate:
        # Trade occurred (much more frequently now)
        if np.random.rand() < 0.5:
            # You sell at ask
            roll = np.random.randint(1, 7)
            profit = ask - roll
            trade_types.append('ask')
        else:
            # You buy at bid
            roll = np.random.randint(1, 7)
            profit = roll - bid
            trade_types.append('bid')
        
        cum_pl += profit
        trade_inds.append(t)
        trade_profits.append(profit)
    
    cum_pl_series[t] = cum_pl


# --- STYLING & COLOR CHOICES (Kept for consistency) ---
LINE_COLOR = '#6AAED6'      
PROFIT_COLOR = '#198754'    
LOSS_COLOR = '#DC3545'      
TRADE_MARKER_COLOR = '#FFC300' 
HIGHLIGHT_SIZE = 10
LINE_WIDTH = 3.5
GRID_COLOR = '#505050'
FONT_COLOR = 'white' 
STAR_SIZE = 18

# --- ZERO-JITTER AXIS SETUP (Fixed Ranges) ---

# 1. Calculate the final, stable Y-range based on the entire data set.
GLOBAL_Y_MAX = max(np.max(cum_pl_series), 0)
GLOBAL_Y_MIN = min(np.min(cum_pl_series), 0)
GLOBAL_Y_DELTA = GLOBAL_Y_MAX - GLOBAL_Y_MIN
GLOBAL_Y_PAD = max(GLOBAL_Y_DELTA * 0.1, 1.5)
STABLE_Y_RANGE = [GLOBAL_Y_MIN - GLOBAL_Y_PAD, GLOBAL_Y_MAX + GLOBAL_Y_PAD]

# 2. Calculate the final, stable X-range.
X_MARGIN = 5
STABLE_X_RANGE = [0, n_steps + X_MARGIN]

# --- CREATE ANIMATION FRAMES ---
frames = []

# Strategic stepping (Adapted to show the faster growth)
frame_steps = (
    list(range(1, min(20, n_steps) + 1)) +
    list(range(21, min(60, n_steps) + 1, 3)) +
    list(range(61, n_steps + 1, 5))
)

for i in frame_steps:
    x = time_steps[:i]
    y = cum_pl_series[:i]
    current_pl = y[-1]
    current_point_color = PROFIT_COLOR if current_pl >= 0 else LOSS_COLOR
    
    trades_up_to_i = [t_ind for t_ind in trade_inds if t_ind < i]
    trade_x = [time_steps[t_ind] for t_ind in trades_up_to_i]
    trade_y = [cum_pl_series[t_ind] for t_ind in trades_up_to_i]
    
    frame = go.Frame(
        data=[
            # 1. Main Path Trace
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=LINE_COLOR, width=LINE_WIDTH), name="Cumulative P&L"),
            # 2. Trade Event Markers
            go.Scatter(x=trade_x, y=trade_y, mode='markers', marker=dict(size=HIGHLIGHT_SIZE, color=TRADE_MARKER_COLOR, symbol='circle-open', line=dict(width=1.5, color=TRADE_MARKER_COLOR)), name='Trades'),
            # 3. Current Point Highlight
            go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers', marker=dict(size=STAR_SIZE, color=current_point_color, symbol='star', line=dict(width=2, color=FONT_COLOR)), name='Current Step')
        ],
        name=str(i),
        layout=go.Layout(
            annotations=[
                dict(
                    x=x[-1], y=y[-1], xref="x", yref="y",
                    text=f"Time {i}, P&L {current_pl:.2f}",
                    showarrow=True, arrowhead=1, ax=0, ay=-40,
                    font=dict(color=current_point_color, size=14),
                    bgcolor="rgba(0,0,0,0.5)", bordercolor=current_point_color
                )
            ]
        )
    )
    frames.append(frame)

# --- BASE FIGURE FOR ANIMATION ---
initial_i = frame_steps[0]
initial_x = time_steps[:initial_i]
initial_y = cum_pl_series[:initial_i]
initial_pl = initial_y[-1]
initial_current_point_color = PROFIT_COLOR if initial_pl >= 0 else LOSS_COLOR

trades_init = [t_ind for t_ind in trade_inds if t_ind < initial_i]
trades_x_init = [time_steps[t_ind] for t_ind in trades_init]
trades_y_init = [cum_pl_series[t_ind] for t_ind in trades_init]


fig_anim = go.Figure(
    data = [
        go.Scatter(x=initial_x, y=initial_y, mode='lines', line=dict(color=LINE_COLOR, width=LINE_WIDTH), name='Cumulative P&L'),
        go.Scatter(x=trades_x_init, y=trades_y_init, mode='markers', marker=dict(size=HIGHLIGHT_SIZE, color=TRADE_MARKER_COLOR, symbol='circle-open', line=dict(width=1.5, color=TRADE_MARKER_COLOR)), name='Trades'),
        go.Scatter(x=[initial_x[-1]], y=[initial_y[-1]], mode="markers", marker=dict(size=STAR_SIZE, color=initial_current_point_color, symbol='star', line=dict(width=2, color=FONT_COLOR)), name='Current Step')
    ],
    layout=go.Layout(
        title=dict(
            text="🎲 Tighter Spread Simulation: Faster P&L Accumulation (Bid=3, Ask=4)",
            x=0.5, font=dict(color=FONT_COLOR, size=24, family="Arial, sans-serif")
        ),
        xaxis=dict(
            title="Time Step (Order Opportunity)",
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, zeroline=False,
            range=STABLE_X_RANGE, # Fixed range
            title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
        ),
        yaxis=dict(
            title="Cumulative Profit & Loss (P&L)",
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            range=STABLE_Y_RANGE, # Fixed range
            tickformat=".2f", title_font=dict(color=FONT_COLOR), tickfont=dict(color=FONT_COLOR)
        ),
        # Professional zero line
        shapes=[
            dict(
                type='line', xref='paper', yref='y', x0=0, y0=0, x1=1, y1=0,
                line=dict(color=GRID_COLOR, width=1.5, dash='dash')
            )
        ],
        font=dict(color=FONT_COLOR, size=12),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=50, r=40, t=80, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color=FONT_COLOR)),
        
        # Subtle footer annotation for parameters
        annotations=[
            dict(
                x=0.99, y=-0.1, xref='paper', yref='paper',
                text=f"True Value=3.5, Spread={spread}, Hit Rate={hit_rate:.3f} (Significantly Higher)",
                showarrow=False, font=dict(color='#888888', size=10),
                xanchor='right', yanchor='top'
            )
        ],
        
        # Play button styling
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x':0.5, 'y':-0.21, 'xanchor': 'center', 'yanchor': 'top',
            'buttons':[{
                'label': '▶️ Play Animation',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 40, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 5, 'easing': 'linear'}, 
                    'mode': 'immediate'
                }]
            }]
        }]
    ),
    frames=frames
)

fig_anim.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Making a Market

In our dice market, realizations occur immediately, if we quote effectively around our mid price we will accumulate profits

In reality, a derivatives contract isn't realized until maturity so we must *hedge* to accumulate the theoretical spread (discussed below)


```python
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np

# Dice setup
dice_faces = np.arange(1, 7)
pmf = np.ones_like(dice_faces) / 6

# Bid/Ask market parameters
spread = 1.5
true_value = dice_faces.mean() # 3.5
bid = true_value - spread/2
ask = true_value + spread/2

# Simulate dice rolls
n_trades = 30
rng = np.random.default_rng(seed=1)
outcomes = rng.choice(dice_faces, size=n_trades)
pl_list = [0] # initial P&L
for i in range(n_trades):
    # we "buy" the roll at our bid, outcome settled
    pl_list.append(pl_list[-1] + (outcomes[i] - bid))

# Remove initial zero for the animation
pl_vals = pl_list[1:]

frames = []
for i in range(n_trades):
    # --- Dynamic Range Calculation for Right Plot ---
    current_pl = np.array(pl_vals[:i+1])
    
    # xmax: current trade number + 2.5 (increased padding for text)
    current_max_x = i + 2.5 
    
    # ymin: minimum P/L achieved so far, with padding (increased from 2 to 3)
    current_min_y = min(current_pl.min() - 3, 0)
    
    # ymax: maximum P/L achieved so far, with padding (increased from 2 to 3)
    current_max_y = max(current_pl.max() + 3, 0)

    # Ensure min < max for y-axis
    if current_min_y == current_max_y:
         current_min_y -= 1
         current_max_y += 1
         
    # PMF trace: highlight realized outcome
    colors = ["deepskyblue"] * 6
    colors[outcomes[i]-1] = "orange"
    pmf_trace = go.Bar(
        x=dice_faces,
        y=pmf,
        marker=dict(color=colors),
        width=0.7,
        name="PMF",
        xaxis="x1", yaxis="y1" # Assign to 1st subplot axes
    )
    # Draw marker for realized outcome
    outcome_scatter = go.Scatter(
        x=[outcomes[i]],
        y=[pmf[outcomes[i]-1]+0.02], # Small bump above bar
        mode='markers+text',
        marker=dict(size=18, color='orange', symbol='star'),
        text=[f"{outcomes[i]}"],
        textposition="top center",
        showlegend=False,
        xaxis="x1", yaxis="y1" # Assign to 1st subplot axes
    )
    # P&L trace up to this trade
    pl_trace = go.Scatter(
        x=np.arange(1,i+2),
        y=pl_vals[:i+1],
        mode="lines+markers",
        line=dict(color="magenta", width=3),
        marker=dict(size=10, color="magenta"),
        name="Cum. P/L",
        xaxis="x2", yaxis="y2" # Assign to 2nd subplot axes
    )
    # Step marker at current trade (now: magenta circle, size 10, to match the line)
    pl_now = go.Scatter(
        x=[i+1],
        y=[pl_vals[i]],
        mode="markers+text",
        marker=dict(size=10, color="magenta", symbol="circle"),
        text=[f"{pl_vals[i]:+.0f}"],
        textposition="bottom center", # Text position is bottom center
        showlegend=False,
        xaxis="x2", yaxis="y2" # Assign to 2nd subplot axes
    )
    # Reference line for "fair" expected outcome (must be calculated up to current trade for frame)
    fair_line = go.Scatter(
        x=np.arange(1, i+2),
        y=(true_value-bid)*np.arange(1, i+2),
        mode="lines",
        line=dict(color="deepskyblue", dash="dash", width=2),
        name="E[P/L]",
        xaxis="x2", yaxis="y2" # Assign to 2nd subplot axes
    )
    
    frames.append(go.Frame(
        data=[pmf_trace, outcome_scatter, pl_trace, pl_now, fair_line],
        name=f"step{i}",
        # --- Update layout for right plot in the frame ---
        layout=go.Layout(
            xaxis2=dict(
                range=[0.5, current_max_x], 
                title="Trade #", 
                showgrid=True, 
                gridcolor="#323232", 
                gridwidth=1
            ),
            yaxis2=dict(
                range=[current_min_y, current_max_y], 
                title="Our Cumulative P/L",
                showgrid=True, 
                gridcolor="#323232", 
                gridwidth=1
            )
        )
        # -------------------------------------------------
    ))

# Create figure with 1 row and 2 columns
fig = sp.make_subplots(rows=1, cols=2)

# Initial traces for frame 0 (must be explicitly added to the fig)
init_traces = [
    go.Bar(
        x=dice_faces, y=pmf, marker=dict(color=["deepskyblue"]*6), width=0.7, name="PMF"
    ),
    go.Scatter(
        x=[outcomes[0]], y=[pmf[outcomes[0]-1]+0.02], mode='markers+text',
        marker=dict(size=18, color='orange', symbol='star'), text=[f"{outcomes[0]}"],
        textposition="top center", showlegend=False
    ),
    go.Scatter(
        x=[1], y=[pl_vals[0]], mode="lines+markers", line=dict(color="magenta", width=3),
        marker=dict(size=10, color="magenta"), name="Cum. P/L"
    ),
    # Step marker: magenta circle, size 10 to match line
    go.Scatter(
        x=[1], y=[pl_vals[0]], mode="markers+text",
        marker=dict(size=10, color="magenta", symbol="circle"), text=[f"{pl_vals[0]:+.0f}"],
        textposition="bottom center", showlegend=False
    ),
    go.Scatter(
        x=[1], y=(true_value-bid)*np.array([1]),
        mode="lines", line=dict(color="deepskyblue", dash="dash", width=2), name="E[P/L]"
    )
]

# Add initial traces, specifying row/col for correct placement
fig.add_trace(init_traces[0], row=1, col=1)
fig.add_trace(init_traces[1], row=1, col=1)
fig.add_trace(init_traces[2], row=1, col=2)
fig.add_trace(init_traces[3], row=1, col=2)
fig.add_trace(init_traces[4], row=1, col=2)

# Set frames on the new figure
fig.frames = frames

# Update layout
FRAME_DURATION = 300
AXIS_TRANSITION_DURATION = FRAME_DURATION * 0.9 

fig.update_layout(
    title="Dice Market: How Edge Accumulates",
    width=1020, height=410,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=1.08,
        xanchor='center', x=0.5,
        font=dict(color="white")
    ),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.5, 
            "y": -0.18, 
            "xanchor": "center", 
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None, 
                        {
                            "frame": {"duration": FRAME_DURATION, "redraw": True}, 
                            "fromcurrent": True,
                            "transition": {
                                "duration": AXIS_TRANSITION_DURATION, 
                                "easing": "linear"
                            }
                        }
                    ]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                },
            ],
        }
    ],
)
# Left subplot: PMF
fig.update_xaxes(
    dict(title="Dice Face", range=[0.5,6.5], tickvals=dice_faces, showgrid=True, gridcolor="#323232", gridwidth=1), 
    row=1, col=1
)
fig.update_yaxes(
    dict(title="Probability", range=[0,0.23], showgrid=True, gridcolor="#323232", gridwidth=1), 
    row=1, col=1
)

# Right subplot (P/L) axes
initial_x_range = frames[0].layout.xaxis2.range
initial_y_range = frames[0].layout.yaxis2.range

fig.update_xaxes(
    dict(title="Trade #", range=initial_x_range, showgrid=True, gridcolor="#323232", gridwidth=1),
    row=1, col=2
)
fig.update_yaxes(
    dict(title="Our Cumulative P/L", range=initial_y_range, showgrid=True, gridcolor="#323232", gridwidth=1),
    row=1, col=2
)

fig.show()
```



---

#### 2.) 📈 Making an Options Market

This idea of accumulating positive expected value works the same way in making a market for options

**The Problem:** We are transacting at $t = 0$ (spot), unlike the dice market which realizes after the roll we must wait until time $T$ to realize a payoff


```python
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Parameters for option and GBM
K = 100          # Strike price
S0 = 100         # Initial price
T = 1.0          # Time to maturity (years)
r = 0.0          # Risk-free rate
sigma = 0.25     # Volatility
num_paths = 30   # Number of simulated GBM paths
num_steps = 100  # Steps per path

np.random.seed(42)

# 1. Hockey stick diagram: Call option payoff at expiry
S_grid = np.linspace(60, 140, 500)
call_payoff = np.maximum(S_grid - K, 0)

# 2. GBM simulation
dt = T / num_steps
times = np.linspace(0, T, num_steps + 1)

gbm_paths = []
for _ in range(num_paths):
    # Brownian increments
    rnd = np.random.normal(size=num_steps)
    W = np.concatenate(([0], np.cumsum(np.sqrt(dt) * rnd)))
    S_path = S0 * np.exp((r - 0.5 * sigma ** 2) * times + sigma * W)
    gbm_paths.append(S_path)

# 3. Plotting
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=["Call Option Payoff at T", "Possible Underlying Paths to Expiry"],
    horizontal_spacing=0.15
)

# Left plot: Hockey stick diagram (NO area fill)
fig.add_trace(
    go.Scatter(
        x=S_grid, y=call_payoff,
        mode='lines',
        line=dict(color='deepskyblue', width=4),
        name="Payoff at T",
        showlegend=False
    ), row=1, col=1
)
# Add strike line
fig.add_trace(
    go.Scatter(
        x=[K, K], y=[0, max(call_payoff)],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name='Strike Price (K)',
        showlegend=False,
    ), row=1, col=1
)
# Add text annotation for strike price in left plot
fig.add_annotation(
    x=K,
    y=max(call_payoff)*0.95,
    text="Strike<br>K",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color="orange", size=13),
    bgcolor="rgba(50,50,50,0.7)",
    row=1, col=1
)

# Right plot: GBM paths (S vs T)
for i, S_path in enumerate(gbm_paths):
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=S_path,
            mode='lines',
            line=dict(
                color='rgba(90, 140, 200, 0.35)' if i < num_paths - 1 else 'rgba(90, 140, 200, 0.45)', 
                width=2
            ),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )
# Add horizontal strike line to right plot
fig.add_trace(
    go.Scatter(
        x=[0, T], y=[K, K],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name='Strike Price (K)',
        showlegend=False
    ),
    row=1, col=2
)
# Add annotation for strike price in right plot
fig.add_annotation(
    x=0,
    y=K,
    text="Strike<br>K",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(color="orange", size=13),
    bgcolor="rgba(50,50,50,0.7)",
    row=1, col=2
)
# Add vertical line for T in right plot
fig.add_trace(
    go.Scatter(
        x=[T, T],
        y=[min([min(path) for path in gbm_paths + [[K]]]) * 0.98, max([max(path) for path in gbm_paths]) * 1.03],
        mode="lines",
        line=dict(color="lime", width=2, dash="dot"),
        showlegend=False,
        hoverinfo='skip'
    ),
    row=1, col=2
)
# Add annotation for T in right plot (top)
fig.add_annotation(
    x=T,
    y=max([max(path) for path in gbm_paths]) * 1.029,
    text="T",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color="lime", size=13),
    row=1, col=2
)

# Update all axes and layout for aesthetics matching
fig.update_xaxes(
    title_text="Underlying Price at T", 
    row=1, col=1,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[60, 140]
)
fig.update_yaxes(
    title_text="Call Payoff", 
    row=1, col=1, 
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[0, max(call_payoff)*1.1]
)

fig.update_xaxes(
    title_text="Time to Expiry (T)", 
    row=1, col=2,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[0, T]
)
fig.update_yaxes(
    title_text="Underlying Price S(t)",
    row=1, col=2,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[min([min(path) for path in gbm_paths + [[K]]]) * 0.98, max([max(path) for path in gbm_paths]) * 1.03]
)

fig.update_layout(
    width=920, height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    title=dict(text="Call Option Payoff and Possible Underlying Price Paths",x=0.5, font=dict(color='white')),
    font=dict(color='white'),
    margin=dict(l=30, r=30, t=50, b=30),
    showlegend=False
)

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### Finding the Mid Price

Unlike with the dice market, there are many acceptable mid prices depending on the model we select

 $$ 
 \text{GBM:} \qquad dS_t = \mu S_t dt + \sigma S_t dW_t 
 $$
 $$
 \text{Heston SV:} \qquad \begin{cases}
 dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^{(1)} \\
 dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^{(2)}
 \end{cases}
 $$
 $$
 \text{Rough Bergomi:} \qquad \begin{cases}
 dS_t = \sqrt{v_t} S_t dW_t \\
 v_t = \xi_0 \exp\Bigg( \eta \int_0^t (t-s)^{H-1/2} dW_s^{(2)} - \frac{1}{2} \eta^2 t^{2H} \Bigg)
 \end{cases}
 $$

 The "mid price" of a contingent claim, under the Fundamental Theorem of Asset Pricing (FTAP), is the expectation of its payoff under the risk-neutral measure:

 $$
 \text{Mid Price} = \mathbb{E}^{\mathbb{Q}} \left[ e^{-r(T-t)} h(S_T) \mid \mathcal{F}_t \right]
 $$

 where:
  - $ \mathbb{Q} $ is the risk-neutral probability measure
  - $ r $ is the risk-free interest rate
  - $ T $ is the claim's maturity
  - $ h(S_T) $ is the payoff of the claim at maturity
  - $ \mathcal{F}_t $ is the information up to time $ t $


Though we may not be operating in a "complete" market with a unique risk-neutral measure like Black-Scholes

We calibrate to liquid instruments implicitly determining the risk-neutral measure by whatever the market is using


```python
import numpy as np
import plotly.graph_objects as go

# --- Volatility Surface Data ---
S_unique = np.array([80, 90, 100, 110, 120])
T_unique = np.array([0.0833, 0.25, 0.50, 1.00, 2.00])
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

Z_market = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])

# Model initial guess is flat surface, e.g. ATM vol
Z_model_init = np.full_like(Z_market, 0.22)

n_steps = 60

frames = []
for i in range(1, n_steps + 1):
    t = i / n_steps
    smooth_t = 1 - (1 - t) ** 2
    # Partial fit: model comes closer to market data, but doesn't land exactly on it
    alpha = 0.85
    fit_level = alpha * smooth_t
    Zfit_grid = (1 - fit_level) * Z_model_init + fit_level * Z_market

    frames.append(go.Frame(
        data=[
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_market,
                colorscale='Viridis', opacity=0.85,
                showscale=False, name='Market', scene='scene'
            ),
            go.Surface(
                x=S_mesh, y=T_mesh, z=Zfit_grid,
                colorscale='Blues', opacity=0.61,
                showscale=False, name='Model (Fit)', scene='scene'
            ),
        ],
        name=f'calib_{i}'
    ))

# --- Base Figure ---
fig = go.Figure()

# Initial market and model (uncalibrated) surface
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_market,
    colorscale='Viridis', opacity=0.85,
    showscale=False, name='Market', scene='scene'
))

fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_model_init,
    colorscale='Blues', opacity=0.61,
    showscale=False, name='Model (Fit)', scene='scene'
))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    title=dict(
        text=("Calibrating Model Volatility Surface to Market"),
        x=0.5, font=dict(size=18)
    ),
    height=560,
    width=820,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.35, 'y': -0.09,
        'showactive': False,
        'buttons': [{
            'label': 'Play Calibration',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 55, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
    scene=dict(
        domain=dict(x=[0.0, 1.0], y=[0, 1]),
        xaxis_title='Stock Price',
        yaxis_title='Time to Expiry',
        zaxis_title='Implied Volatility',
        xaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        yaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        zaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        bgcolor='rgba(0,0,0,0)'
    )
)

fig.show()

```



Once we have calibrated parameters we can simulate pricing by discretizing the SDE assuming no analytical solution exists

The finer our discretization and the more sample paths we have (by the Law of Large Numbers (LLN)) the more accurate our price!

 $$
 \text{Mid Price} = \mathbb{E}^{\mathbb{Q}} \left[ e^{-rT} \cdot \text{payoff}\left( S_0 \exp\left( \int_{0}^{T} \sqrt{v_t}\, dW_t^S - \frac{1}{2} \int_{0}^{T} v_t\, dt \right) \right) \right]
 $$
 where
 $$
 v_t = \xi_0(t) \exp\left( \eta \int_0^t K_H(t, s)\, dW_s^v - \frac{1}{2} \eta^2 t^{2H} \right)
 $$

##### Example: Monte Carlo Simulation Pricing Convergence


```python
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# === Generic Framework: Stochastic Model + Theoretical Price + Monte Carlo ===
# This structure works for *any* model: simply swap in your SDE, payoff, and theory.

# --- MODEL PARAMETERS (Example: GBM inputs) ---
K = 100         # Strike
S0 = 100        # Initial value
T = 1.0         # Maturity
r = 0.0         # Risk-free rate
sigma = 0.25    # Volatility
num_paths = 100 # Number of simulated paths
num_steps = 100 # Time discretization

np.random.seed(42)
dt = T / num_steps
times = np.linspace(0, T, num_steps + 1)

# --- GENERIC STOCHASTIC MODEL SIMULATOR ---
def simulate_paths(model_func, S0, params, times, num_paths):
    all_paths = []
    for _ in range(num_paths):
        path = model_func(S0, params, times)
        all_paths.append(path)
    return np.array(all_paths)

# Example GBM model - to replace for other models as needed
def gbm_path(S0, params, times):
    sigma, r = params['sigma'], params['r']
    n_steps = len(times) - 1
    W = np.concatenate(([0], np.cumsum(np.sqrt(times[1] - times[0]) * np.random.normal(size=n_steps))))
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * times + sigma * W)
    return S

gbm_params = dict(sigma=sigma, r=r)
all_paths = simulate_paths(gbm_path, S0, gbm_params, times, num_paths)

# --- THEORETICAL PRICE (Provide the appropriate closed-form for your model) ---
from scipy.stats import norm
def bs_call(S, K, T, r, sigma):
    if T == 0:
        return np.maximum(S-K,0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def theoretical_call_price(model, S0, K, T, r, sigma):
    # Swap out logic for your model's analytic price if available
    if model == "GBM":
        return bs_call(S0, K, T, r, sigma)
    else:
        return None # or np.nan

theoretical = theoretical_call_price("GBM", S0, K, T, r, sigma)

# --- GENERIC MONTE CARLO EMPIRICAL PRICE ---
def mc_empirical_prices(all_paths, K, r, T):
    # Payoff: vectorized; replace for other options
    payoffs = np.maximum(all_paths[:, -1] - K, 0)
    discounted = payoffs * np.exp(-r * T)
    means = [np.mean(discounted[:j+1]) for j in range(len(discounted))]
    return means

# --- CONSTANTS FOR PLOT ---
MAX_TRACES_LEFT = num_paths + 1
MAX_TRACES_RIGHT = 3
MAX_X_RANGE_RIGHT = 115

# --- PREPARE ANIMATION FRAMES ---
frames = []

for N in range(1, num_paths + 1):
    # --- Left: Stochastic Paths ---
    traces_left = []
    for i in range(N):
        is_latest = (i == N - 1)
        traces_left.append(go.Scatter(
            x=times, y=all_paths[i],
            line=dict(
                color='rgba(90, 140, 200, 0.38)' if not is_latest else 'rgba(0,255,255,0.9)',
                width=2 if not is_latest else 3
            )
        ))
    for i in range(N, num_paths):
        traces_left.append(go.Scatter(x=[], y=[], mode='lines'))
    # Strike
    traces_left.append(go.Scatter(
        x=[0, T], y=[K, K],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
    ))

    # --- Right: MC Empirical and Theory ---
    emp_means = mc_empirical_prices(all_paths[:N], K, r, T)
    traces_right = []
    # Empirical MC mean
    traces_right.append(go.Scatter(
        x=np.arange(1, N+1), y=emp_means,
        mode='lines+markers',
        line=dict(color='deepskyblue', width=4),
        marker=dict(size=5, color='deepskyblue'),
        name='Empirical MC Price'
    ))
    # Theoretical price (horizontal)
    traces_right.append(go.Scatter(
        x=[1, MAX_X_RANGE_RIGHT], y=[theoretical, theoretical],
        mode='lines',
        line=dict(color='lime', width=3, dash='dash'),
        name="Theoretical Price"
    ))
    # Annotation
    traces_right.append(go.Scatter(
        x=[N], y=[emp_means[-1]],
        mode='markers+text',
        marker=dict(size=10, color='aqua'),
        text=[f"N={N}<br>Price={emp_means[-1]:.2f}"],
        textposition="top left",
        showlegend=False, hoverinfo='skip'
    ))

    frames.append(go.Frame(
        data=traces_left + traces_right,
        name=str(N)
    ))

# --- INITIAL DATA FOR PLOT (N=1) ---
initial_paths = []
initial_paths.append(go.Scatter(
    x=times, y=all_paths[0], mode='lines',
    line=dict(color='rgba(0,255,255,0.9)', width=3),
    hoverinfo='skip', showlegend=False, name=None
))
for i in range(1, num_paths): 
    initial_paths.append(go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(color='rgba(90, 140, 200, 0.38)', width=2),
        hoverinfo='skip', showlegend=False, name=None
    ))
initial_strike = go.Scatter(
    x=[0,T], y=[K,K], mode='lines', line=dict(color='orange', width=2, dash='dash'),
    showlegend=False, hoverinfo='skip'
)

emp_means_n1 = mc_empirical_prices(all_paths[:1], K, r, T)
initial_mean_line = go.Scatter(
    x=[1], y=[emp_means_n1[0]], mode='lines+markers',
    line=dict(color='deepskyblue', width=4),
    marker=dict(size=5, color='deepskyblue'), name='Empirical MC Price'
)
initial_theoretical_line = go.Scatter(
    x=[1, MAX_X_RANGE_RIGHT], y=[theoretical, theoretical], mode='lines',
    line=dict(color='lime', width=3, dash='dash'), name="Theoretical Price"
)
initial_annotation = go.Scatter(
    x=[1], y=[emp_means_n1[0]], mode='markers+text',
    marker=dict(size=10, color='aqua'),
    text=["N=1<br>Price={:.2f}".format(emp_means_n1[0])],
    textposition="top left",
    showlegend=False, hoverinfo='skip'
)

initial_data = initial_paths + [initial_strike] + [initial_mean_line, initial_theoretical_line, initial_annotation]

# --- FIGURE SETUP ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    subplot_titles=[
        "Stochastic Model Simulated Paths", 
        "Empirical Price vs Theoretical Price"
    ],
    horizontal_spacing=0.15
)
for i, trace in enumerate(initial_data[:MAX_TRACES_LEFT]):
    fig.add_trace(trace, row=1, col=1)
for i, trace in enumerate(initial_data[MAX_TRACES_LEFT:]):
    fig.add_trace(trace, row=1, col=2)

# --- AXES/LAYOUT ---
fig.update_xaxes(
    title_text="Time to Expiry (T)", row=1, col=1,
    showgrid=True, gridcolor="#323232", gridwidth=1, range=[0,T]
)
fig.update_yaxes(
    title_text="Model Value $X(t)$", row=1, col=1,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[min([all_paths.min(), K]) * 0.98, all_paths.max() * 1.03]
)
fig.update_xaxes(
    title_text="Number of Simulated Paths (N)", row=1, col=2,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[-5, MAX_X_RANGE_RIGHT] 
)
fig.update_yaxes(
    title_text="Option Price", row=1, col=2,
    showgrid=True, gridcolor="#323232", gridwidth=1,
    range=[0, 20]
)
fig.update_layout(
    width=980, height=380,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    title=dict(
        text="Monte Carlo Framework: Empirical vs. Theoretical Price",
        x=0.5, font=dict(color='white', size=19)
    ),
    font=dict(color='white'),
    margin=dict(l=40, r=40, t=60, b=30),
    showlegend=True,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.38, 'y': -0.34, 
        'showactive': False,
        'buttons': [{
            'label': 'Play MC',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 80, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)
fig.frames = frames

fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### Quoting a Spread

In practice, the quoted price is not just the theoretical price (e.g., as above) but contains adjustments for various risks and requirements:
 $$
  \text{Quoted Price} = \mathbb{E}^{\mathbb{Q}}[\text{Payoff}\ |\ \mathcal{F}_0]
                      + \text{Counterparty Risk Premium} 
                      + \text{Liquidity Premium} 
                      + \text{Profit Margin} 
                      + \cdots
 $$

 Effectively, when we engage in a transaction, we are always *buying low* or *selling high*


```python
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm

# --- SIMULATION AND PRICING PARAMETERS ---
# Keeping the original parameters
sim_sigma = 0.24
sim_r = 0.0
sim_S0 = 100
sim_K = 100
sim_T = 1.0
sim_num_trades = 1000

price_sigma = 0.25
price_r = 0.0
price_S0 = 100
price_K = 100
price_T = 1.0

# --- FINANCIAL FUNCTIONS ---

def simulate_terminal(sim_S0, sim_r, sim_sigma, sim_T, N):
    Z = np.random.normal(size=N)
    ST = sim_S0 * np.exp((sim_r - 0.5 * sim_sigma**2)*sim_T + sim_sigma * np.sqrt(sim_T) * Z)
    return ST

def bs_call(S, K, T, r, sigma):
    if T == 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# --- CALCULATIONS ---

# Set random seed for reproducibility
np.random.seed(2025)

paid_price = bs_call(price_S0, price_K, price_T, price_r, price_sigma)
sim_STs = simulate_terminal(sim_S0, sim_r, sim_sigma, sim_T, sim_num_trades)
payoff = np.maximum(sim_STs - sim_K, 0)
discounted_payoff = payoff * np.exp(-sim_r * sim_T)
PL_per_trade = discounted_payoff - paid_price
cumulative_PL = np.cumsum(PL_per_trade)
num_points = len(cumulative_PL)

# --- COLOR & STYLE CHOICES ---
LINE_COLOR = '#6AAED6'  # Muted blue for the path
PROFIT_COLOR = '#198754'  # Green for positive P&L
LOSS_COLOR = '#DC3545'    # Red for negative P&L
HIGHLIGHT_SIZE = 10
LINE_WIDTH = 3.5
GRID_COLOR = '#505050'
FONT_COLOR = 'white' # Ensure text remains readable on any background

# --- CREATE ANIMATION FRAMES FOR CUMULATIVE P&L ---

frames = []
start_x = 1

# Maintain the original strategic stepping for efficiency
frame_steps = (
    list(range(1, min(30,num_points)+1)) +
    list(range(31, min(200,num_points)+1, 5)) +
    list(range(205, num_points+1, 10))
)

for i in frame_steps:
    x = np.arange(start_x, i+1)
    y = cumulative_PL[:i]
    
    # Dynamic axis range calculation for smooth zooming/panning
    xmargin = max(5, int(0.05 * i))
    x_range = [0, i + xmargin]
    
    # Pad y-axis intelligently
    ymax = max(np.max(y), 0)
    ymin = min(np.min(y), 0)
    ydelta = ymax - ymin
    ypad = max(ydelta * 0.1, 1.5) # Minimum padding of 1.5
    y_range = [ymin - ypad, ymax + ypad]
    
    current_pl = y[-1]
    highlight_color = PROFIT_COLOR if current_pl >= 0 else LOSS_COLOR
    
    # Create the trace for this frame
    frame = go.Frame(
        data=[
            # 1. Main Path Trace
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(color=LINE_COLOR, width=LINE_WIDTH),
                name="Cumulative P&L"
            ),
            # 2. Current Point Highlight Trace
            go.Scatter(
                x=[x[-1]], y=[y[-1]],
                mode="markers",
                marker=dict(size=HIGHLIGHT_SIZE, color=highlight_color, symbol='circle',
                            line=dict(width=2, color=FONT_COLOR)), # White border for emphasis
                name="Current Trade"
            )
        ],
        name=str(i),
        layout=go.Layout(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            # Add an annotation for the current P&L value
            annotations=[
                dict(
                    x=x[-1], y=y[-1],
                    xref="x", yref="y",
                    text=f"Trade {i}: P&L {current_pl:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0, ay=-40,
                    font=dict(color=highlight_color, size=14),
                    bgcolor="rgba(0,0,0,0.5)", bordercolor=highlight_color
                )
            ]
        )
    )
    frames.append(frame)

# --- BASE FIGURE FOR ANIMATION ---
initial_i = frame_steps[0]
initial_x = np.arange(1, initial_i+1)
initial_y = cumulative_PL[:initial_i]
initial_pl = initial_y[-1]
initial_highlight_color = PROFIT_COLOR if initial_pl >= 0 else LOSS_COLOR

# Calculate initial ranges based on initial data
initial_xmargin = max(5, int(0.05 * initial_i))
initial_x_range = [0, initial_i + initial_xmargin]
initial_ymax = max(np.max(initial_y), 0)
initial_ymin = min(np.min(initial_y), 0)
initial_ydelta = initial_ymax - initial_ymin
initial_ypad = max(initial_ydelta * 0.1, 1.5)
initial_y_range = [initial_ymin - initial_ypad, initial_ymax + initial_ypad]


fig_anim = go.Figure(
    data = [
        # 1. Initial Main Path Trace
        go.Scatter(
            x=initial_x,
            y=initial_y,
            mode='lines',
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            name='Cumulative P&L'
        ),
        # 2. Initial Current Point Highlight Trace
        go.Scatter(
            x=[initial_x[-1]],
            y=[initial_y[-1]],
            mode="markers",
            marker=dict(size=HIGHLIGHT_SIZE, color=initial_highlight_color, symbol='circle',
                        line=dict(width=2, color=FONT_COLOR)),
            name="Current Trade"
        )
    ],
    layout=go.Layout(
        title=dict(
            text="📈 Cumulative P&L Simulation: Market-Making Strategy",
            x=0.5, font=dict(color=FONT_COLOR, size=24, family="Arial, sans-serif")
        ),
        xaxis=dict(
            title="Trade Number (N)",
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            zeroline=False,
            range=initial_x_range,
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR)
        ),
        yaxis=dict(
            title="Cumulative Profit & Loss (P&L)", # No LaTeX here
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            range=initial_y_range,
            tickformat=".2f",
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR)
        ),
        # Professional zero line
        shapes=[
            dict(
                type='line',
                xref='paper', yref='y',
                x0=0, y0=0, x1=1, y1=0,
                line=dict(color=GRID_COLOR, width=1.5, dash='dash')
            )
        ],
        font=dict(color=FONT_COLOR, size=12),
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)', # Transparent overall background
        margin=dict(l=50, r=40, t=80, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color=FONT_COLOR)),
        
        # Subtle footer annotation for parameters
        annotations=[
            dict(
                x=0.99, y=-0.1, xref='paper', yref='paper',
                text=f"Sim sigma={sim_sigma}, Pricing sigma={price_sigma}, T={sim_T}",
                showarrow=False, font=dict(color='#888888', size=10),
                xanchor='right', yanchor='top'
            )
        ],
        
        # Play button styling
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x':0.5, 'y':-0.21, 'xanchor': 'center', 'yanchor': 'top',
            'buttons':[{
                'label': '▶️ Play Animation',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 40, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0},
                    'mode': 'immediate'
                }]
            }]
        }]
    ),
    frames=frames
)

fig_anim.show()
```



**Remark:** The simulation above assumes the payoff happens at the transaction of the contract.  This is not true!  In reality, we must hedge our exposure to get as close to the theoretical price as possible to accumulate the spread.

###### ______________________________________________________________________________________________________________________________________

##### Making a Market and Hedging

Unlike with the dice market, the contract isn't realized until a maturity in the future so after we transact we must hedge to collect the theoretical value!

 $$\Delta S + (V - \Delta S) e^{r \Delta t} = V \ e^{r \Delta t}$$



```python
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm
from plotly.subplots import make_subplots

# --- Configuration & Parameters ---

# Simulation parameters (The "real-world" where the option evolves)
SIM_R = 0.05           # Risk-free rate (r)
SIM_T = 1.0            # Time to expiry in years (T)
SIM_S0 = 100.0         # Initial Stock Price (S0)
SIM_K = 100.0          # Strike Price (K)
SIM_SIGMA = 0.30       # Volatility in the simulation

# Model parameters used for pricing and delta (RN valuation)
PRICE_SIGMA = 0.25     # Volatility assumed by the model
PRICE_R = SIM_R
PRICE_K = SIM_K

SIM_STEPS = 252        # Number of hedging steps (daily)
DT = SIM_T / SIM_STEPS # Time step

# Plotting Styles
LINE_COLOR_STOCK = '#6AAED6'   # Muted Blue
LINE_COLOR_DERIV = '#FFA700'   # Vibrant Orange
LINE_COLOR_HEDGED = '#198754'  # Deep Green
FONT_COLOR = 'white'
GRID_COLOR = '#505050'

# Set random seed for reproducibility of the simulation path
np.random.seed(42)

# --- Black-Scholes Functions ---

def bs_call(S, K, T, r, sigma):
    """Black-Scholes Call Option Price"""
    # Handle expiration: value is intrinsic max(S-K, 0)
    if T <= 1e-6:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Black-Scholes Call Option Delta (Hedge Ratio)"""
    # Handle expiration: delta is 1 (in-the-money) or 0 (out-of-the-money)
    if T <= 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- Hedged Portfolio Simulation ---

t_values = np.linspace(0, SIM_T, SIM_STEPS + 1)
S_path = np.zeros(SIM_STEPS + 1)
C_path = np.zeros(SIM_STEPS + 1)
Delta_path = np.zeros(SIM_STEPS + 1)

# Portfolio_Cash: The risk-free bond (cash) component of the replicating portfolio.
Portfolio_Cash = np.zeros(SIM_STEPS + 1)

# 1. Initial State (t=0)
S_path[0] = SIM_S0
T_rem_0 = SIM_T
C_path[0] = bs_call(S_path[0], PRICE_K, T_rem_0, PRICE_R, PRICE_SIGMA)
Delta_path[0] = bs_delta(S_path[0], PRICE_K, T_rem_0, PRICE_R, PRICE_SIGMA)

# The replicating portfolio at t=0 must have a total value equal to the option price C0.
Portfolio_Cash[0] = C_path[0] - Delta_path[0] * S_path[0]

# --- Main Simulation Loop (Discrete Re-hedging) ---
for i in range(SIM_STEPS):
    # 1. Stock Movement (Geometric Brownian Motion)
    Z = np.random.normal()
    S_path[i + 1] = S_path[i] * np.exp((SIM_R - 0.5 * SIM_SIGMA ** 2) * DT + SIM_SIGMA * np.sqrt(DT) * Z)

    t_i_plus_1 = t_values[i + 1]
    T_rem_i_plus_1 = SIM_T - t_i_plus_1

    # 2. Option & Delta Calculation at time t_{i+1}
    C_path[i + 1] = bs_call(S_path[i + 1], PRICE_K, T_rem_i_plus_1, PRICE_R, PRICE_SIGMA)
    Delta_path[i + 1] = bs_delta(S_path[i + 1], PRICE_K, T_rem_i_plus_1, PRICE_R, PRICE_SIGMA)

    # 3. Portfolio Update (Self-Financing Condition)
    prev_cash_value_accrued = Portfolio_Cash[i] * np.exp(PRICE_R * DT)
    delta_change = Delta_path[i + 1] - Delta_path[i]
    rehedge_cost = delta_change * S_path[i + 1]
    Portfolio_Cash[i + 1] = prev_cash_value_accrued - rehedge_cost

# 4. Final P&L Calculation (At expiry T)
Final_Hedge_Value = Delta_path[SIM_STEPS - 1] * S_path[SIM_STEPS] + Portfolio_Cash[SIM_STEPS]
Final_Payoff_Liability = np.maximum(S_path[SIM_STEPS] - PRICE_K, 0)
Replication_Error = Final_Hedge_Value - Final_Payoff_Liability

# --- P&L Path Tracking ---
V_path = Delta_path * S_path + Portfolio_Cash
P_L_Error_path = V_path - C_path
P_L_Error_path[0] = 0.0
P_L_Error_path[-1] = Replication_Error

# --- Calculate Ranges for Subplots ---
Y_PADDING_FACTOR = 0.05
S_range = [S_path.min() * (1 - Y_PADDING_FACTOR), S_path.max() * (1 + Y_PADDING_FACTOR)]
C_range = [0, C_path.max() * (1 + Y_PADDING_FACTOR)]
max_abs_pi = max(abs(P_L_Error_path.min()), abs(P_L_Error_path.max()))
Pi_range = [-max_abs_pi * 1.5, max_abs_pi * 1.5]

# --- Figure Creation ---

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "1. Stock Price Process (S_t)",
        "2. Option Price (Risk-Neutral Valuation C_t)",
        "3. Hedged Portfolio Replication Error (V_t - C_t)"
    ),
    row_heights=[0.35, 0.35, 0.30]
)

# --- Initialize traces: One line and one marker per subplot. ---
# These traces are what the frames will update.

# Subplot 1: Stock Price (Line & Marker)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[S_path[0]], mode='lines', line=dict(color=LINE_COLOR_STOCK, width=3), name='Stock Price Path'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[S_path[0]], mode='markers', marker=dict(size=10, color=LINE_COLOR_STOCK, symbol='circle'), name='Stock Price Marker'),
    row=1, col=1
)

# Subplot 2: Option Price (Line & Marker)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[C_path[0]], mode='lines', line=dict(color=LINE_COLOR_DERIV, width=3), name='Option Price Path'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[C_path[0]], mode='markers', marker=dict(size=10, color=LINE_COLOR_DERIV, symbol='circle'), name='Option Price Marker'),
    row=2, col=1
)

# Subplot 3: Replication Error (Line & Marker)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[P_L_Error_path[0]], mode='lines', line=dict(color=LINE_COLOR_HEDGED, width=3), name='Replication Error Path'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=[t_values[0]], y=[P_L_Error_path[0]], mode='markers', marker=dict(size=10, color=LINE_COLOR_HEDGED, symbol='circle', line=dict(width=2, color=FONT_COLOR)), name='Replication Error Marker'),
    row=3, col=1
)

# Add line at P&L = 0
fig.add_hline(y=0, line_dash="dash", line_color="#DC3545", line_width=1.5, row=3, col=1)

# --- Animation Frames ---
frames = []
# Animate in chunks of 5 days for smoother, faster visual
frame_steps = list(range(0, SIM_STEPS + 1, 5)) + [SIM_STEPS]
frame_steps = sorted(list(set(frame_steps)))

for frame_idx, i in enumerate(frame_steps):
    current_t = t_values[:i + 1]

    # The index 'i' in the path arrays is the data point for the current time step 't'.
    current_S = S_path[i]
    current_C = C_path[i]
    current_pl_error = P_L_Error_path[i]

    # Data for the current frame
    frame_data = [
        # Trace 0 (Subplot 1 Line: Stock Price Path)
        go.Scatter(x=current_t, y=S_path[:i + 1]),
        # Trace 1 (Subplot 1 Marker: Stock Price Marker)
        go.Scatter(x=[current_t[-1]], y=[current_S]),
        # Trace 2 (Subplot 2 Line: Option Price Path)
        go.Scatter(x=current_t, y=C_path[:i + 1]),
        # Trace 3 (Subplot 2 Marker: Option Price Marker)
        go.Scatter(x=[current_t[-1]], y=[current_C]),
        # Trace 4 (Subplot 3 Line: Hedged P&L Path)
        go.Scatter(x=current_t, y=P_L_Error_path[:i + 1]),
        # Trace 5 (Subplot 3 Marker: Hedged P&L Marker)
        go.Scatter(x=[current_t[-1]], y=[current_pl_error]),
    ]

    highlight_color = LINE_COLOR_HEDGED if abs(current_pl_error) < 0.5 else '#DC3545'

    # Annotation to show the replication error
    frame_layout = go.Layout(
        annotations=[
            dict(
                x=current_t[-1],
                y=current_pl_error,
                xref="x3",
                yref="y3",
                text=f"Day {i}: Error {current_pl_error:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color=FONT_COLOR, size=14),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=highlight_color,
                borderwidth=2
            )
        ]
    )

    frames.append(go.Frame(data=frame_data, layout=frame_layout, name=str(i)))

# --- Final Figure Layout & Display ---

# Ensure the initial trace displays the first point correctly before animation starts
fig.data[0].x = [t_values[0]]
fig.data[0].y = [S_path[0]]
fig.data[2].x = [t_values[0]]
fig.data[2].y = [C_path[0]]
fig.data[4].x = [t_values[0]]
fig.data[4].y = [P_L_Error_path[0]]

fig.update_layout(
    title=dict(
        text="🏛️ Dynamic Delta-Hedge Simulation",
        x=0.5,
        font=dict(color=FONT_COLOR, size=24, family="Arial, sans-serif")
    ),
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=850,
    margin=dict(l=50, r=40, t=100, b=100),
    showlegend=False,
    transition={'duration': 0},  # Essential for smooth animation updates

    # X-Axis Settings (Applied to the bottom one)
    xaxis3=dict(
        title="Time (Years)",
        showgrid=True,
        gridcolor=GRID_COLOR,
        range=[t_values[0], t_values[-1]],
        tickfont=dict(color=FONT_COLOR),
        title_font=dict(color=FONT_COLOR, size=18)
    ),

    # Y-Axis 1: Stock Price (S_t)
    yaxis=dict(
        title_text="Stock Price (S_t)",
        gridcolor=GRID_COLOR,
        tickfont=dict(color=FONT_COLOR),
        range=S_range,
        title_font=dict(size=16)
    ),
    # Y-Axis 2: Option Value (C_t)
    yaxis2=dict(
        title_text="Option Value (C_t)",
        gridcolor=GRID_COLOR,
        tickfont=dict(color=FONT_COLOR),
        range=C_range,
        title_font=dict(size=16)
    ),
    # Y-Axis 3: P&L Error (Pi_t)
    yaxis3=dict(
        title_text="Replication Error",
        gridcolor=GRID_COLOR,
        tickformat=".2f",
        tickfont=dict(color=FONT_COLOR),
        range=Pi_range,
        title_font=dict(size=16)
    ),

    # Play button configuration
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'x': 0.5,
        'y': -0.1,
        'xanchor': 'center',
        'yanchor': 'top',
        'buttons': [{
            'label': '▶️ Start Replication',
            'method': 'animate',
            'args': [
                None,  # Animates through all frames sequentially
                {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': False, # Start from the beginning
                    'transition': {'duration': 0}
                }
            ]
        }]
    }]
)

# Crucial step: Assign frames directly to the Figure object
fig.frames = frames

fig.show()
```



Theoretically, if we have a perfect hedge, the hedged portfolio value is the theoretical price of the option in the risk-neutral sense

This is the Black-Scholes portfolio replication argument for a complete market which is precisely the no-arbitrage price (Feynman-Kac)

Since we transacted at that theoretical price at the bid or ask/offer we have effectively then bought low or sold high

In reality, if we then hedge effectively we will accumulate that theoretical spread, realistic hedging however will chip away at our profits!

---

#### 3.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**

- In the face of structured randomness (dice roll market) or uncertainty (the markets) we can apply probability and statistics to systematically make money
- When distributions are fixed, the expected value (mean, average) gives us a best guess in the MSE OOS sense, should we transact in the market around that price (quote a bid, ask/offer) we will accumulate drift even with an entirely random outcome
- In reality, distributions change, the same principles hold but we need to calibrate models to the markets to find the appropriate parameter set to derive our mid price, in other words, to use the relevant non-unique risk-neutral measure displayed in current market prices (vol surface calibration)
- Once a parameter set is selected we can use the law of large numbers to simulate the theoretical (mid) price of our contract and quote a spread around it
- Unlike the dice market, where realizations occur instantly, we will have to hedge throughout the life of our position to collect the spread
- Theoretically, continuous hedging with no transaction costs result in the simulated price of our contract so our position would be net zero if we transacted at mid but we transact at the bid or ask/offer giving us an edge to accumulate
- Therefore, since markets are not frictionless, the efficacy of our hedge dictates how effectively we collect our spread and serve the business function of a market-maker

**Future Topics**

Technical Videos and Other Discussions

- Quant Roadmap
- Projects that Made me a Quant
- Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Reinforcement Learning for Delta Hedging
- Approximating Pricing Functionals using Neural Networks
- Rough Path Theorey, Applications of Path Signatures

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Neural Network Stochastic Volatility Model Calibration
- Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
- Automated Delta-Neutral Trading System

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
