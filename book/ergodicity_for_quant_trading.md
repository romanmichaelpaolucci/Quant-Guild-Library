### 📈 Ergodicity for Quant Trading

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

#### 1.) 📊 Edge in Quant Trading

- Stationary Distributions

- Non-Stationary Distributions

#### 2.) 📉 Bet Sizing and Wealth Paths

- Additive Bets (Ergodic Systems)

- Multiplicative Bets (Non-Ergodic Systems)

#### 3.) 📈 Optimal Bet Sizing

- Optimizing for the Time Average

- Reality of Bet Sizing


#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📊 Edge in Quant Trading

Given a trading strategy (or composite $\alpha$) $\tau$
 
 **Expected Value Decomposition (Law of Total Expectation):**
 
 $$
 \mathbb{E}_{\tau}[\mathrm{trade}] = \mathbb{E}_{\tau}[\mathrm{trade} \mid \mathrm{win}] \cdot P(\mathrm{win}) + \mathbb{E}_{\tau}[\mathrm{trade} \mid \mathrm{lose}] \cdot P(\mathrm{lose})
 $$
 
 For this case:
 - $\mathbb{E}_{\tau}[\mathrm{trade} \mid \mathrm{win}] = +25$
 - $\mathbb{E}_{\tau}[\mathrm{trade} \mid \mathrm{lose}] = -25$
 
 For probability of winning $p$ (e.g., $p = 0.60$ for pos. EV, $p = 0.40$ for neg. EV):
 
 $$
 \mathbb{E}_{\tau}[\mathrm{trade}] = 25 \cdot p + (-25) \cdot (1-p) = 25 \cdot (2p - 1)
 $$
 
 For positive EV ($p = 0.60$):
 
 $$
 \mathbb{E}_{\tau}[\mathrm{trade}] = 25 \cdot 0.60 + (-25) \cdot 0.40 = 15 - 10 = 5
 $$
 
 For negative EV ($p = 0.40$):
 
 $$
 \mathbb{E}_{\tau}[\mathrm{trade}] = 25 \cdot 0.40 + (-25) \cdot 0.60 = 10 - 15 = -5
 $$
 
 This is the expected (average) wealth change per round.


```python
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
n_steps = 100
initial_wealth = 1000
bet_size = 25
n_paths = 10

# Positive EV (60% win rate)
pos_ev_paths = np.zeros((n_paths, n_steps))
pos_ev_paths[:, 0] = initial_wealth
for path in range(n_paths):
    pos_ev_wins = np.random.random(n_steps) < 0.60
    for i in range(1, n_steps):
        pos_ev_paths[path, i] = pos_ev_paths[path, i-1] + (bet_size if pos_ev_wins[i] else -bet_size)

# Negative EV (40% win rate)
neg_ev_paths = np.zeros((n_paths, n_steps))
neg_ev_paths[:, 0] = initial_wealth
for path in range(n_paths):
    neg_ev_wins = np.random.random(n_steps) < 0.40
    for i in range(1, n_steps):
        neg_ev_paths[path, i] = neg_ev_paths[path, i-1] + (bet_size if neg_ev_wins[i] else -bet_size)

# Expected Value lines
pos_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (0.60 - 0.40)
neg_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (0.40 - 0.60)

# --- Frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []

    # Left (Positive EV)
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=pos_ev_paths[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x', yaxis='y',
                showlegend=False
            )
        )
    # Add expected value line
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=pos_ev_line[:i+1],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x', yaxis='y',
            showlegend=False
        )
    )

    # Right (Negative EV)
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=neg_ev_paths[path, :i+1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False
            )
        )
    # Add expected value line
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=neg_ev_line[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x2', yaxis='y2',
            showlegend=False
        )
    )

    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps]),
        )
    ))

# --- Figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Positive EV (edge > 0)', 'Negative EV (edge < 0)'),
    column_widths=[0.5, 0.5]
)

# Initial left-side traces
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Positive EV Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )

# Initial right-side traces
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=1.0 - path * 0.07,
            name='Negative EV Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=2
    )

# Add static expected value lines
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=pos_ev_line, mode='lines',
               line=dict(color='lime', dash='dash', width=3),
               name='Expected Value (Positive EV)'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=neg_ev_line, mode='lines',
               line=dict(color='orange', dash='dash', width=3),
               name='Expected Value (Negative EV)'),
    row=1, col=2
)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=550,
    width=1100,
    title_text="Wealth Paths: Positive vs Negative Expected Value<br><sup>Dashed line = Theoretical EV</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.1,
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

# --- Axes ---
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()

```



##### Having an Edge in Games of Chance (Roulette, Craps, ...) Doesn't Change Over Time and Our Wealth Process Trajectory Above is Accurate

###### ______________________________________________________________________________________________________________________________________

##### In Games of Incomplete Information (Poker, Trading, ...) Edge and Components Change Over Time Impacting Wealth Trajectory


```python
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
n_steps = 100
initial_wealth = 1000
bet_size = 25
n_paths = 10

# --- OU process parameters for "true expected value" ---
dt = 1
# Positive EV process: average is positive, but mean-reverts and fluctuates
ou_mu_pos = 0.18   # mean
ou_theta_pos = 0.1   # speed of mean-reversion
ou_sigma_pos = 0.11  # volatility

# Negative EV process: average is negative (more often negative), mean-reverts and fluctuates
ou_mu_neg = -0.18
ou_theta_neg = 0.11
ou_sigma_neg = 0.10

def ou_process(mu, theta, sigma, n, x0):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        x[i] = x[i-1] + dx
    return x

np.random.seed(42)
true_ev_positive = ou_process(ou_mu_pos, ou_theta_pos, ou_sigma_pos, n_steps, 0.2)
true_ev_negative = ou_process(ou_mu_neg, ou_theta_neg, ou_sigma_neg, n_steps, -0.2)

# --- Wealth path simulation using time-varying "true" EV ---
def simulate_paths(ev_series, n_paths, n_steps, initial_wealth, bet_size):
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = initial_wealth
    for path in range(n_paths):
        for i in range(1, n_steps):
            winrate = 0.5 + ev_series[i] / 2
            win = np.random.random() < winrate
            paths[path, i] = paths[path, i-1] + (bet_size if win else -bet_size)
    return paths

pos_ev_paths = simulate_paths(true_ev_positive, n_paths, n_steps, initial_wealth, bet_size)
neg_ev_paths = simulate_paths(true_ev_negative, n_paths, n_steps, initial_wealth, bet_size)

# --- Dynamic expected value line (integral of EV drift) ---
pos_ev_cum = initial_wealth + bet_size * np.cumsum(true_ev_positive)
neg_ev_cum = initial_wealth + bet_size * np.cumsum(true_ev_negative)

# --- Frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []

    # Upper left: Positive EV paths
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=pos_ev_paths[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x', yaxis='y',
                showlegend=False
            )
        )
    # Expected value line (pos, dynamic)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=pos_ev_cum[:i+1],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x', yaxis='y',
            showlegend=False
        )
    )

    # Upper right: Negative EV paths
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=neg_ev_paths[path, :i+1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False
            )
        )
    # Expected value line (neg, dynamic)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=neg_ev_cum[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x2', yaxis='y2',
            showlegend=False
        )
    )

    # Lower left: True positive EV OU series
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=true_ev_positive[:i+1],
            mode='lines',
            line=dict(color='lime', width=2),
            xaxis='x3', yaxis='y3',
            showlegend=False
        )
    )
    # Lower right: True negative EV OU series
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=true_ev_negative[:i+1],
            mode='lines',
            line=dict(color='orange', width=2),
            xaxis='x4', yaxis='y4',
            showlegend=False
        )
    )

    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps]),
            xaxis3=dict(range=[0, n_steps]),
            xaxis4=dict(range=[0, n_steps]),
        )
    ))

# --- Figure with 2x2 subplots ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Positive Expected Value (Wealth, edge > 0)",
        "Negative Expected Value (Wealth, edge < 0)",
        "True EV Over Time (OU Process, Positive EV)",
        "True EV Over Time (OU Process, Negative EV)"
    ),
    vertical_spacing=0.18,
    horizontal_spacing=0.07
)

# Initially: show NOTHING (hide all initial series by setting x/y empty)
# Add traces with empty data that will be updated during animation

# Upper left: Positive EV paths (empty)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Positive EV Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )
# Upper right: Negative EV paths (empty)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=1.0 - path * 0.07,
            name='Negative EV Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=2
    )

# Add expected value lines as empty initially
fig.add_trace(
    go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='lime', dash='dash', width=3),
        name='Expected Value (Positive EV)'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='Expected Value (Negative EV)'
    ),
    row=1, col=2
)

# Lower left: true positive expected value (OU) line (empty)
fig.add_trace(
    go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='lime', width=2),
        name="True Expected Value (Pos EV, OU)",
        showlegend=True
    ),
    row=2, col=1
)
# Lower right: true negative expected value (OU) line (empty)
fig.add_trace(
    go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='orange', width=2),
        name="True Expected Value (Neg EV, OU)",
        showlegend=True
    ),
    row=2, col=2
)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=755,
    width=1200,
    title_text="Wealth Paths: Positive vs Negative Expected Value<br><sup>Dashed line = Theoretical EV; Lower: True Edge (OU Process)</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation='h', y=-0.16, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.12,
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

# --- Axes ---
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_yaxes(range=[0, 2000], row=1, col=1, title_text="Wealth",
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(range=[0, 2000], row=1, col=2, title_text="Wealth",
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_xaxes(title_text="Round", row=2, col=1)
fig.update_xaxes(title_text="Round", row=2, col=2)
fig.update_yaxes(title_text="True EV", row=2, col=1)
fig.update_yaxes(title_text="True EV", row=2, col=2)

fig.show()

```



##### The Stability of your Edge is Determined by the Strategy Itself and "What" You're Trading (Alpha, Beta, . . .)

---

#### 2.) 📉 Bet Sizing and Wealth Paths

##### Supposing we have a reasonably stable wealth path we have an entirely different problem

Consider a bet where you win \$1 with probability $p=0.6$ and lose \$1 with probability $0.4$:

 **Additive case:**
 $$
 \mathbb{E}[\Delta W] = 0.6 \times 1 + 0.4 \times (-1) = 0.2 > 0
 $$

 **Multiplicative (betting $r = 0.5$ of wealth):**
 $$
 g = 0.6 \log(1.5) + 0.4 \log(0.5) \approx 0.24 - 0.28 = -0.04 < 0
 $$
 Thus, while the additive expectation is positive, the multiplicative (log-growth) expectation is negative.



```python
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
n_steps = 100
initial_wealth = 1000
bet_size = 25      # Only for additive
bet_frac = 0.5     # Only for multiplicative
n_paths = 10

# --- Additive Paths ---
additive_paths = np.zeros((n_paths, n_steps))
additive_paths[:, 0] = initial_wealth
for path in range(n_paths):
    wins = np.random.random(n_steps) < 0.60
    for i in range(1, n_steps):
        additive_paths[path, i] = additive_paths[path, i-1] + (bet_size if wins[i] else -bet_size)

# Theoretical expected value line for additive
additive_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (0.60 - 0.40)

# --- Multiplicative Paths ---
multiplicative_paths = np.zeros((n_paths, n_steps))
multiplicative_paths[:, 0] = initial_wealth
for path in range(n_paths):
    wins = np.random.random(n_steps) < 0.60
    for i in range(1, n_steps):
        prev_wealth = multiplicative_paths[path, i-1]
        bet = bet_frac * prev_wealth
        multiplicative_paths[path, i] = prev_wealth + (bet if wins[i] else -bet)

# Theoretical expected log-growth per step for multiplicative:
# g = 0.6*log(1+r) + 0.4*log(1-r)
r = bet_frac
g = 0.6 * np.log(1 + r) + 0.4 * np.log(1 - r)
multiplicative_ev_line = initial_wealth * np.exp(np.arange(n_steps) * g)

# --- Animation Frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []

    # Left: Additive paths
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=additive_paths[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x', yaxis='y',
                showlegend=False
            )
        )
    # Theoretical EV for additive
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=additive_ev_line[:i+1],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x', yaxis='y',
            showlegend=False
        )
    )

    # Right: Multiplicative paths
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=multiplicative_paths[path, :i+1],
                mode='lines',
                line=dict(color='deepskyblue', width=2),
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False
            )
        )
    # Theoretical EV for multiplicative (log exp line)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=multiplicative_ev_line[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x2', yaxis='y2',
            showlegend=False
        )
    )

    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps]),
        )
    ))

# --- Figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Additive Wealth Paths', 'Multiplicative Wealth Paths (risk r=0.5)'),
    column_widths=[0.5, 0.5]
)

# Initial Additive traces (left panel)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Additive Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )

# Initial Multiplicative traces (right panel)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='deepskyblue', width=2),
            opacity=1.0 - path * 0.07,
            name='Multiplicative Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=2
    )

# Add static expected value lines
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=additive_ev_line, mode='lines',
               line=dict(color='lime', dash='dash', width=3),
               name='Additive Expected Value'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=multiplicative_ev_line, mode='lines',
               line=dict(color='orange', dash='dash', width=3),
               name='Multiplicative Expected Value'),
    row=1, col=2
)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=550,
    width=1100,
    title_text="Additive vs Multiplicative Wealth Paths<br><sup>Dashed line = Theoretical EV</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.1,
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

# --- Axes ---
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### The experience of one is not the experience of many
  
  $$\langle \Delta W \rangle_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^N \Delta W_i$$
  $$\langle \Delta W \rangle_{\text{time}} = \frac{1}{T} \sum_{t=1}^T \Delta W_{t}$$

 **Additive Model:**  
 The *ensemble average* (expectation across many paths) grows over time, and matches what you’d see for a typical individual’s time average.  


 **Multiplicative Model:**  
 The *ensemble average* can increase, but the *time average* for an individual path can decrease—most paths lose wealth even though the average over all paths can rise. Time and ensemble averages differ.

---

#### 3.) 📈 Optimal Bet Sizing

 To optimize time-average wealth, maximize the expected logarithmic growth:
  $$
  \max_f\;\; \mathbb{E}\left[\log(1 + f R)\right]
  $$
 subject to $0 \leq f \leq 1$, where $f$ is the fraction of wealth bet and $R$ is the random return (e.g., $+b$ with probability $p$, $-1$ with probability $q$).

The **Kelly Criterion** gives the optimal fraction of wealth to bet in order to maximize long-term growth (under multiplicative wealth dynamics):
 
 The solution maximizes the expected logarithmic growth rate of wealth.


 $$f^* = \frac{p \cdot b - q}{b}$$

 where $f^*$ is the optimal fraction, $p$ is the probability of winning, $q = 1-p$ is the probability of losing, and $b$ is the odds received per unit bet.




```python
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
n_steps = 100
initial_wealth = 1000
bet_size = 25      # Only for additive
n_paths = 10

# --- Kelly Criterion Parameters ---
p = 0.60            # Probability of win
q = 1 - p           # Probability of loss
b = 1               # Odds received per unit bet (standard even-odds gamble)

# Kelly fraction for this game:
kelly_fraction = (p * b - q) / b  # Kelly formula

# --- Additive Paths ---
additive_paths = np.zeros((n_paths, n_steps))
additive_paths[:, 0] = initial_wealth
for path in range(n_paths):
    wins = np.random.random(n_steps) < p
    for i in range(1, n_steps):
        additive_paths[path, i] = additive_paths[path, i-1] + (bet_size if wins[i] else -bet_size)

# Theoretical expected value line for additive
additive_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (p - q)

# --- Multiplicative Paths (with Kelly Criterion) ---
multiplicative_paths = np.zeros((n_paths, n_steps))
multiplicative_paths[:, 0] = initial_wealth
for path in range(n_paths):
    wins = np.random.random(n_steps) < p
    for i in range(1, n_steps):
        prev_wealth = multiplicative_paths[path, i-1]
        bet = kelly_fraction * prev_wealth
        # Gain b*bet on win, lose bet on loss
        if wins[i]:
            multiplicative_paths[path, i] = prev_wealth + b * bet
        else:
            multiplicative_paths[path, i] = prev_wealth - bet

# Theoretical expected log-growth per step for Kelly bet:
# g = p*log(1 + f*b) + q*log(1 - f)
f = kelly_fraction
g = p * np.log(1 + f * b) + q * np.log(1 - f)
multiplicative_ev_line = initial_wealth * np.exp(np.arange(n_steps) * g)

# --- Animation Frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []

    # Left: Additive paths
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=additive_paths[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x', yaxis='y',
                showlegend=False
            )
        )
    # Theoretical EV for additive
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=additive_ev_line[:i+1],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x', yaxis='y',
            showlegend=False
        )
    )

    # Right: Multiplicative paths (Kelly)
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=multiplicative_paths[path, :i+1],
                mode='lines',
                line=dict(color='deepskyblue', width=2),
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False
            )
        )
    # Theoretical EV for multiplicative (log exp line)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=multiplicative_ev_line[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x2', yaxis='y2',
            showlegend=False
        )
    )

    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps]),
        )
    ))

# --- Figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Additive Wealth Paths',
        f'Multiplicative Wealth Paths (Kelly fraction f*={kelly_fraction:.2f})'
    ),
    column_widths=[0.5, 0.5]
)

# Initial Additive traces (left panel)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Additive Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )

# Initial Multiplicative traces (right panel)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0], y=[initial_wealth],
            mode='lines',
            line=dict(color='deepskyblue', width=2),
            opacity=1.0 - path * 0.07,
            name='Multiplicative Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=2
    )

# Add static expected value lines
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=additive_ev_line, mode='lines',
               line=dict(color='lime', dash='dash', width=3),
               name='Additive Expected Value'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=multiplicative_ev_line, mode='lines',
               line=dict(color='orange', dash='dash', width=3),
               name='Multiplicative Expected Value'),
    row=1, col=2
)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=550,
    width=1100,
    title_text="Additive vs Multiplicative Wealth Paths<br><sup>Dashed line = Theoretical EV</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.1,
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

# --- Axes ---
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### Reality: Your Edge Isn't Fixed, Neither Should Be Your Bet Sizing

Full Kelly betting is rarely used in practice — estimation errors, edge variability, and psychological factors all make the true Kelly fraction riskier and more volatile than it appears. Overbetting amplifies risk of drawdowns or ruin if your edge isn't stable or is overestimated.


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
n_steps = 200       # Increased steps to allow divergence to play out
n_paths = 50        # More paths for a smoother "Ensemble" average
initial_wealth = 1000
rolling_window = 10 # Short window = Higher Estimation Error (Kills Full Kelly)

# --- 1. Generate the "Ground Truth" (Hidden from trader) ---
np.random.seed(42)
dt = 1

# SMOOTHER OU PARAMETERS
# Lower sigma = less jagged line
# Lower theta = trend persists longer (regimes)
# Lower mu = edge is thinner, making overbetting more dangerous
ou_mu = 0.15       
ou_theta = 0.05    # Slow mean reversion
ou_sigma = 0.015   # Very low volatility (Smooth "True Probability" line)

x = np.zeros(n_steps)
x[0] = 0.0
for i in range(1, n_steps):
    dx = ou_theta * (ou_mu - x[i-1]) * dt + ou_sigma * np.sqrt(dt) * np.random.randn()
    x[i] = x[i-1] + dx
true_probs = 0.5 + x

# --- 2. Simulate Ensemble Paths ---
def simulate_ensemble(n_paths, n_steps, true_probs, kelly_fraction):
    wealth = np.zeros((n_paths, n_steps))
    wealth[:, 0] = initial_wealth
    
    avg_bet_sizes = np.zeros(n_steps)
    
    # Pre-generate outcomes based on True Probabilities
    random_draws = np.random.rand(n_paths, n_steps)
    outcomes = (random_draws < true_probs).astype(float)
    
    for t in range(1, n_steps):
        # 1. Estimation (The source of the error)
        start_idx = max(0, t - rolling_window)
        history = outcomes[:, start_idx:t]
        
        if t < 5:
            est_probs = np.ones(n_paths) * 0.5
        else:
            est_probs = np.mean(history, axis=1)
            
        # 2. Kelly Sizing (f = 2p - 1)
        raw_kelly = 2 * est_probs - 1
        raw_kelly = np.maximum(raw_kelly, 0)
        
        # Apply Fraction
        bets = raw_kelly * kelly_fraction
        
        # Safety: Cap at 0.95 to prevent log(0) errors, 
        # but Full Kelly will still get punished heavily near this limit
        bets = np.minimum(bets, 0.95) 
        
        avg_bet_sizes[t] = np.mean(bets)
        
        # 3. Update Wealth
        current_wins = outcomes[:, t]
        growth_factors = np.where(current_wins == 1.0, (1 + bets), (1 - bets))
        wealth[:, t] = wealth[:, t-1] * growth_factors
        
    return wealth, avg_bet_sizes

# Run Simulations
# Full Kelly (1.0) vs Half Kelly (0.5)
full_wealth, full_bets = simulate_ensemble(n_paths, n_steps, true_probs, kelly_fraction=1.0)
half_wealth, half_bets = simulate_ensemble(n_paths, n_steps, true_probs, kelly_fraction=0.5)

# Calculate Geometric Mean (The "Typical" result excluding outliers)
full_geo_mean = np.exp(np.mean(np.log(full_wealth + 1e-9), axis=0))
half_geo_mean = np.exp(np.mean(np.log(half_wealth + 1e-9), axis=0))

# --- 3. Construct Animation ---

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Full Kelly (1.0)</b>: High Volatility & Ruin", 
        "<b>Half Kelly (0.5)</b>: Optimal Compound Growth",
        "<b>True Edge (Hidden)</b>: Smooth Regime Shift",
        "<b>Avg Bet Size</b>: Overreaction vs Prudence"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# --- Static Init Traces ---
# ROW 1 COL 1: Full Kelly
for i in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth], mode='lines', line=dict(color='red', width=1), opacity=0.1, showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[initial_wealth], mode='lines', name='Full Geo Mean', line=dict(color='white', width=3), showlegend=False), row=1, col=1)

# ROW 1 COL 2: Half Kelly
for i in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth], mode='lines', line=dict(color='cyan', width=1), opacity=0.1, showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[initial_wealth], mode='lines', name='Half Geo Mean', line=dict(color='white', width=3), showlegend=False), row=1, col=2)

# ROW 2 COL 1: True Edge
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='lime', width=2), name="True Edge"), row=2, col=1)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], mode='lines', line=dict(color='gray', dash='dot'), showlegend=False), row=2, col=1)

# ROW 2 COL 2: Bet Sizes
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='red', width=2), name="Full Avg Bet"), row=2, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='cyan', width=2), name="Half Avg Bet"), row=2, col=2)

# --- Frames ---
frames = []
for t in range(1, n_steps + 1, 2): # Skip every other frame for speed
    frame_data = []
    x_current = np.arange(t)
    
    # 1. Full Kelly Ensemble
    for p in range(n_paths):
        frame_data.append(go.Scatter(x=x_current, y=full_wealth[p, :t]))
    frame_data.append(go.Scatter(x=x_current, y=full_geo_mean[:t])) # The mean line
    
    # 2. Half Kelly Ensemble
    for p in range(n_paths):
        frame_data.append(go.Scatter(x=x_current, y=half_wealth[p, :t]))
    frame_data.append(go.Scatter(x=x_current, y=half_geo_mean[:t])) # The mean line
    
    # 3. True Edge
    frame_data.append(go.Scatter(x=x_current, y=(true_probs[:t]-0.5)*100))
    frame_data.append(go.Scatter(x=[0, n_steps], y=[0,0]))
    
    # 4. Bet Sizes
    frame_data.append(go.Scatter(x=x_current, y=full_bets[:t]*100))
    frame_data.append(go.Scatter(x=x_current, y=half_bets[:t]*100))
    
    frames.append(go.Frame(name=str(t), data=frame_data))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=700, width=1200,
    title="<b>The Kelly Criterion Trap</b><br><sup>Parameter uncertainty makes Full Kelly sub-optimal long term</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', family="Monospace"),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.1,
        'showactive': False,              # <--- THIS IS THE KEY FIX
        'bgcolor': 'rgba(0,0,0,0)',       # Transparent background
        'bordercolor': 'white',           # White border
        'borderwidth': 1,                 # Thin border
        'font': dict(color='white'),      # White text
        'buttons': [{
            'label': '▶ Run Simulation',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 10, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

# Axes styling
fig.update_xaxes(range=[0, n_steps], showgrid=True, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

# Specific Axes Logic to show the split
fig.update_yaxes(title="Wealth (Log)", type="log", row=1, col=1) 
fig.update_yaxes(title="Wealth (Log)", type="log", row=1, col=2)
fig.update_yaxes(title="True Edge (%)", range=[-5, 15], row=2, col=1)
fig.update_yaxes(title="Avg Bet Size (%)", range=[0, 50], row=2, col=2)

fig.show()
```



---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**

- **Edge**: dictates our wealth path trajectory over time in a closed game of chance and a strategy in a game of incomplete information
- **In reality**: *edge* is not constant and probabilities of winning and losing along with the average winner and losing comprising edge change over time
- **Bet sizing**: dramatically impacts our wealth path trajectory in these games, we don't want to bet fixed amounts as our wealth grows making these systems multiplicative and non-ergodic (the path of some is not the path of many, the ensemble and time average aren't equivalent)
- **Optimizing for Time Average**: Optimizing specifically for multiplicative wealth yields the Kelly Criterion 
- **In reality**: *bet sizing* is a function of our edge components which vary over time and full-kelly is too aggresive leading to higher likelihood of ruin which is why we often modify this quantity such as going half-kelly or other functional variations for optimal sizing

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
