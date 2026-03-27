### 🦄 Quant Busts 3 Trading Myths with Math

##### ▶️ Related Quant Guild Videos:

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)

- [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)

- [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/wZChBKDFFeU)

- [Quant vs. Discretionary Trading](https://youtu.be/3gblERSSHXI)

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

### 📖 Sections


#### 1.) 🃏 Myth 1: Trading is Gambling and the Market is Random

- Accumulating Wealth in Randomness

- Main Problem: The Market Isn't Random

#### 2.) 📈 Myth 2: You Only Need to Be Correct 50.5% of the Time to be Profitable

- Trading Edge in Practice

- Ergodic vs. Non-Ergodic Systems

#### 3.) ⚠️ Myth 3: Trading Strategies are Fixed, Always Work, Nobody Shares Profitable Strategies

- Policy Functions and Time Variance

- Regime Dynamics and Data Generating Distributions

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 🃏 Myth 1: Trading is Gambling and the Market is Random

Trading is **NOT** a game of chance, and the market is **NOT** random - it is uncertain

First off, if a game or system is random it does not mean you can't make money from it

Some casino games are purely random, and they make plenty of money preying on people that don't understand statistics and the concept of edge


##### Equation for Edge: 
$$\mathbb{E}[\text{edge}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$


###### ______________________________________________________________________________________________________________________________________


##### Accumulating Wealth in Randomness: American Roulette
 
This is purely a game of chance, there is nothing the player can do to make money (besides martingaling roulette, which isn't possible due to table limits)

Statistically, (assuming ergodicity) they will lose all of their money if they continue to play, the only optimal decision is to not play

 $$\mathbb{E}[\text{lot}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$
 
 $$\mathbb{E}[\text{lot}] = (+\$100 \times \frac{18}{38}) + (-\$100 \times \frac{20}{38}) = -\$5.26 < 0$$

This negative edge for the player is proven by the expected value equation and convergence gaurenteed by the Law of Large Numbers (LLN)


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_players = 5
n_spins = 100
starting_wealth = 1000
bet_size = 100

# European roulette has 37 numbers (0-36)
p_win = 18/37
payout = 2  # 1:1 payout for red/black bets

# Generate wealth paths for players
player_wealths = np.zeros((n_players, n_spins + 1))
player_wealths[:, 0] = starting_wealth

# Generate casino wealth path
casino_wealth = np.zeros(n_spins + 1)
casino_wealth[0] = 0

# Simulate spins
for i in range(n_spins):
    results = np.random.random(n_players) < p_win
    wins = results * bet_size * (payout - 1)
    losses = ~results * bet_size
    net_results = wins - losses

    player_wealths[:, i+1] = player_wealths[:, i] + net_results
    casino_wealth[i+1] = casino_wealth[i] - sum(net_results)

# Stop player wealth at zero
for i in range(n_players):
    zero_indices = np.where(player_wealths[i] <= 0)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        player_wealths[i, first_zero:] = 0

# Create figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Player Wealth Paths', 'Casino Wealth Path')
)

# Define player colors (varying opacity reds)
player_colors = [f'rgba(255,0,0,{opacity})' for opacity in np.linspace(0.3, 1, n_players)]

# Create frames for animation
frames = []
for step in range(n_spins + 1):
    frame = {"data": [], "name": str(step)}

    # Axis limits
    y_min_players = min(np.min(player_wealths[:, :step+1]), 0)
    y_max_players = max(np.max(player_wealths[:, :step+1]), starting_wealth)
    y_min_casino = min(np.min(casino_wealth[:step+1]), 0)
    y_max_casino = max(np.max(casino_wealth[:step+1]), 0)

    # Player traces
    for i in range(n_players):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=player_wealths[i, :step + 1],
                mode='lines',
                line=dict(color=player_colors[i], width=2),
                name=f'Player {i+1}'
            )
        )

    # Casino trace (green)
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=casino_wealth[:step + 1],
            mode='lines',
            line=dict(color='rgba(0,255,0,1)', width=2),
            name='Casino'
        )
    )

    frame["layout"] = {
        "xaxis": {"range": [0, max(step + 1, 5)]},
        "xaxis2": {"range": [0, max(step + 1, 5)]},
        "yaxis": {"range": [y_min_players * 1.1, y_max_players * 1.1]},
        "yaxis2": {"range": [y_min_casino * 1.1, y_max_casino * 1.1]}
    }

    frames.append(frame)

# Add initial traces
for i in range(n_players):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=player_colors[i], width=2),
            name=f'Player {i+1}'
        ),
        row=1, col=1
    )

# Initial casino trace
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode='lines',
        line=dict(color='rgba(0,255,0,1)', width=2),
        name='Casino'
    ),
    row=1, col=2
)

# Update layout (no slider)
fig.update_layout(
    height=500,
    width=1200,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            },
            {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ]
    }]
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Spins',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Wealth ($)' if i == 1 else 'Casino Profit ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

# Set initial axis ranges
fig.update_xaxes(range=[0, 5], row=1, col=1)
fig.update_xaxes(range=[0, 5], row=1, col=2)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=1)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=2)

# Add frames
fig.frames = frames

fig.show()

```

To make money in a random system you must have an edge and you must make **appropriate** bets to accumulate it over time

Both the player and the casino in the short run can blowout 

The accumulation of wealth is made over hundreds, thousands of reasonably sized bets with a positive edge

###### ______________________________________________________________________________________________________________________________________

##### What Ensure's the Casino's Edge?  The Law of Large Numbers (LLN)

$$\lim_{n \to \infty} P(\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^n X_i = \mu) = 1$$

The Strong Law of Large Numbers (SLLN): Let $X_1, X_2, ...$ be i.i.d. random variables with $E[|X_1|] < \infty$. Then $\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{a.s.} E[X_1]$ as $n \to \infty$







```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Setup ---
outcomes = ['Green', 'Red', 'Black']
theoretical_pmf = np.array([2/38, 18/38, 18/38])

# Simulate spins (Green=0, Red=1, Black=2)
n_spins = 500
np.random.seed(40)
spins = np.random.choice(outcomes, size=n_spins, p=theoretical_pmf)

# Compute running empirical PMF
empirical_pmfs = []
counts = np.zeros(len(outcomes))
for i, outcome in enumerate(spins, start=1):
    counts[outcomes.index(outcome)] += 1
    empirical_pmf = counts / i
    empirical_pmfs.append(empirical_pmf.copy())

# --- Create frames ---
frames = []
for i in range(1, n_spins):
    frames.append(go.Frame(
        data=[
            # Theoretical PMF
            go.Bar(
                x=outcomes,
                y=theoretical_pmf,
                marker_color=['green', 'red', 'black'],
                opacity=0.4,
                name='Theoretical PMF'
            ),
            # Empirical PMF (running)
            go.Bar(
                x=outcomes,
                y=empirical_pmfs[i],
                marker_color=['lime', 'salmon', 'gray'],
                opacity=0.9,
                name='Empirical PMF'
            ),
            # Running total of spins for context - Red
            go.Scatter(
                x=np.arange(1, i+1),
                y=[empirical_pmfs[j][1] for j in range(i)],
                mode='lines',
                line=dict(color='red', width=2),
                name='Empirical P(Red)'
            ),
            # Running total of spins for context - Black
            go.Scatter(
                x=np.arange(1, i+1),
                y=[empirical_pmfs[j][2] for j in range(i)],
                mode='lines',
                line=dict(color='black', width=2),
                name='Empirical P(Black)'
            ),
            # Running total of spins for context - Green
            go.Scatter(
                x=np.arange(1, i+1),
                y=[empirical_pmfs[j][0] for j in range(i)],
                mode='lines',
                line=dict(color='green', width=2),
                name='Empirical P(Green)'
            ),
            # Theoretical lines
            go.Scatter(
                x=np.arange(1, i+1),
                y=[theoretical_pmf[1]] * i,
                mode='lines',
                line=dict(color='white', width=1, dash='dot'),
                name='True P(Red)'
            ),
            go.Scatter(
                x=np.arange(1, i+1),
                y=[theoretical_pmf[2]] * i,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name='True P(Black)'
            ),
            go.Scatter(
                x=np.arange(1, i+1),
                y=[theoretical_pmf[0]] * i,
                mode='lines',
                line=dict(color='lime', width=1, dash='dot'),
                name='True P(Green)'
            )
        ],
        layout=go.Layout(
            xaxis2=dict(range=[0, i + 10])
        ),
        name=f'frame{i}'
    ))

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Empirical vs Theoretical PMF', 'Convergence of Probabilities'),
    column_widths=[0.4, 0.6]
)

# Initial theoretical and empirical PMF
fig.add_trace(
    go.Bar(
        x=outcomes,
        y=theoretical_pmf,
        marker_color=['green', 'red', 'black'],
        opacity=0.4,
        name='Theoretical PMF'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=outcomes,
        y=empirical_pmfs[0],
        marker_color=['lime', 'salmon', 'gray'],
        opacity=0.9,
        name='Empirical PMF'
    ),
    row=1, col=1
)

# Initial convergence lines
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[empirical_pmfs[0][1]],
        mode='lines',
        line=dict(color='red', width=2),
        name='Empirical P(Red)'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[empirical_pmfs[0][2]],
        mode='lines',
        line=dict(color='black', width=2),
        name='Empirical P(Black)'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[empirical_pmfs[0][0]],
        mode='lines',
        line=dict(color='green', width=2),
        name='Empirical P(Green)'
    ),
    row=1, col=2
)

# Theoretical lines
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[theoretical_pmf[1]],
        mode='lines',
        line=dict(color='white', width=1, dash='dot'),
        name='True P(Red)'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[theoretical_pmf[2]],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        name='True P(Black)'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[theoretical_pmf[0]],
        mode='lines',
        line=dict(color='lime', width=1, dash='dot'),
        name='True P(Green)'
    ),
    row=1, col=2
)

# --- Animation & Layout ---
fig.frames = frames

fig.update_layout(
    height=500,
    width=1000,
    title_text="American Roulette: Empirical PMF Converging to Theoretical PMF",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Axes
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', range=[0, 0.6])

fig.show()

```

These probabilities are used directly in the calculation of the player and casino's edge.

Crucially, they **DO NOT** change over time - they converge to theoretical values by the Law of Large Numbers (LLN)

By the Law of Large Numbers (LLN):
$$P(win) = \lim_{n \to \infty} \frac{\text{Number of Wins}}{n}$$
$$P(loss) = \lim_{n \to \infty} \frac{\text{Number of Losses}}{n}$$

 $$\mathbb{E}[\text{lot}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss}) < 0 \text{ for any player}$$

###### ______________________________________________________________________________________________________________________________________

##### We Apply this Same Idea to Trading

$$\mathbb{E}[\text{trade}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$

We accumulate wealth over time if we have positive expected value (EV) or an edge, we lose wealth over time if we act suboptimally


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_traders = 10  # Number of traders per group
n_trades = 200
starting_wealth = 1000
bet_size = 50

# Negative EV traders (like casino players)
p_win_neg = 16/37  # Using roulette odds as example
payout_neg = 2

# Positive EV traders 
p_win_pos = 20/37  # Slightly better odds
payout_pos = 2

# Generate wealth paths for traders
neg_ev_wealths = np.zeros((n_traders, n_trades + 1))
pos_ev_wealths = np.zeros((n_traders, n_trades + 1))
neg_ev_wealths[:, 0] = starting_wealth
pos_ev_wealths[:, 0] = starting_wealth

# Simulate trades
for i in range(n_trades):
    # Negative EV traders
    results_neg = np.random.random(n_traders) < p_win_neg
    wins_neg = results_neg * bet_size * (payout_neg - 1)
    losses_neg = ~results_neg * bet_size
    net_results_neg = wins_neg - losses_neg
    neg_ev_wealths[:, i+1] = neg_ev_wealths[:, i] + net_results_neg

    # Positive EV traders
    results_pos = np.random.random(n_traders) < p_win_pos
    wins_pos = results_pos * bet_size * (payout_pos - 1)
    losses_pos = ~results_pos * bet_size
    net_results_pos = wins_pos - losses_pos
    pos_ev_wealths[:, i+1] = pos_ev_wealths[:, i] + net_results_pos

# Stop wealth at zero
for i in range(n_traders):
    zero_indices = np.where(neg_ev_wealths[i] <= 0)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        neg_ev_wealths[i, first_zero:] = 0
        
    zero_indices = np.where(pos_ev_wealths[i] <= 0)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        pos_ev_wealths[i, first_zero:] = 0

# Create figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Negative EV Traders', 'Positive EV Traders')
)

# Define trader colors (varying opacity reds for neg EV, varying opacity greens for pos EV)
neg_colors = [f'rgba(255,0,0,{opacity})' for opacity in np.linspace(0.3, 1, n_traders)]
pos_colors = [f'rgba(0,255,0,{opacity})' for opacity in np.linspace(0.3, 1, n_traders)]

# Create frames for animation
frames = []
for step in range(n_trades + 1):
    frame = {"data": [], "name": str(step)}

    # Axis limits
    y_min_neg = min(np.min(neg_ev_wealths[:, :step+1]), 0)
    y_max_neg = max(np.max(neg_ev_wealths[:, :step+1]), starting_wealth)
    y_min_pos = min(np.min(pos_ev_wealths[:, :step+1]), 0)
    y_max_pos = max(np.max(pos_ev_wealths[:, :step+1]), starting_wealth)

    # --- LEFT subplot (Negative EV traders) ---
    for i in range(n_traders):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=neg_ev_wealths[i, :step + 1],
                mode='lines',
                line=dict(color=neg_colors[i], width=2),
                name=f'Negative EV Trader {i+1}',
                xaxis='x',
                yaxis='y'
            )
        )

    # --- RIGHT subplot (Positive EV traders) ---
    for i in range(n_traders):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=pos_ev_wealths[i, :step + 1],
                mode='lines',
                line=dict(color=pos_colors[i], width=2),
                name=f'Positive EV Trader {i+1}',
                xaxis='x2',
                yaxis='y2'
            )
        )

    frame["layout"] = {
        "xaxis": {"range": [0, max(step + 1, 5)]},
        "xaxis2": {"range": [0, max(step + 1, 5)]},
        "yaxis": {"range": [y_min_neg * 1.1, y_max_neg * 1.1]},
        "yaxis2": {"range": [y_min_pos * 1.1, y_max_pos * 1.1]},
    }
    frames.append(frame)


# Add initial traces
for i in range(n_traders):
    # Negative EV traders
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=neg_colors[i], width=2),
            name=f'Negative EV Trader {i+1}'
        ),
        row=1, col=1
    )
    
    # Positive EV traders
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=pos_colors[i], width=2),
            name=f'Positive EV Trader {i+1}'
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
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            },
            {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ]
    }]
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
        title='Wealth ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

# Set initial axis ranges
fig.update_xaxes(range=[0, 5], row=1, col=1)
fig.update_xaxes(range=[0, 5], row=1, col=2)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=1)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=2)

# Add frames
fig.frames = frames

fig.show()

```

###### ______________________________________________________________________________________________________________________________________
##### **Problem:** The market **IS NOT** random, which is why our quant models are *even more wrong* than the assumptions imply

The casino is resampling from the same distribution, in other words the roulette wheel won't ever change, but the market sure does

This is confusing because we often model the market as a *random variable* to assign different likelihoods to different outcomes


$$\mathbb{E}[\text{trade}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$

But nobody, especially in the classroom, is willing to discuss if this is appropriate, these likelihoods **do not converge and change over time** 


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Parameters ---
np.random.seed(42)
outcomes = ['Up', 'Down']

# Probabilities
p_market = np.array([0.6, 0.4])   # Market implied
p_true   = np.array([0.45, 0.55]) # True (unknown in real life)

# Simulation
n_events = 10_000
events = np.random.choice(outcomes, size=n_events, p=p_true)

# --- Running empirical probabilities ---
counts = np.zeros(len(outcomes))
empirical = np.zeros((n_events, len(outcomes)))
for i, outcome in enumerate(events, start=1):
    counts[outcomes.index(outcome)] += 1
    empirical[i - 1] = counts / i

# --- Colors ---
colors = {
    "market_up": "rgba(0, 200, 0, 0.4)",
    "market_down": "rgba(255, 0, 0, 0.4)",
    "emp_up": "lime",
    "emp_down": "salmon"
}

# --- Animation frames ---
frames = []
step_size = 50
for i in range(1, n_events, step_size):
    frames.append(go.Frame(
        name=f"frame_{i}",
        data=[
            # LEFT chart – Market vs Empirical
            go.Bar(
                x=outcomes,
                y=p_market,
                marker_color=[colors["market_up"], colors["market_down"]],
                opacity=0.4,
                name="Market Implied",
                showlegend=False
            ),
            go.Bar(
                x=outcomes,
                y=empirical[i],
                marker_color=[colors["emp_up"], colors["emp_down"]],
                opacity=0.9,
                name="Empirical",
                showlegend=False
            ),

            # RIGHT chart – Probability evolution
            go.Scatter(
                x=np.arange(1, i + 1),
                y=empirical[:i, 0],
                mode="lines",
                line=dict(color=colors["emp_up"], width=2),
                name="Empirical P(Up)",
                xaxis="x2", yaxis="y2"
            ),
            go.Scatter(
                x=np.arange(1, i + 1),
                y=[p_market[0]] * i,
                mode="lines",
                line=dict(color="white", dash="dot", width=1),
                name="Market P(Up)",
                xaxis="x2", yaxis="y2"
            ),
            go.Scatter(
                x=np.arange(1, i + 1),
                y=[p_true[0]] * i,
                mode="lines",
                line=dict(color="lime", dash="dot", width=1),
                name="True P(Up)",
                xaxis="x2", yaxis="y2"
            )
        ],
        layout=go.Layout(
            xaxis2=dict(range=[0, i + step_size])  # Dynamic x-axis range update
        )
    ))

# --- Subplot layout ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Market Implied vs Empirical PMF", "Evolution of P(Up)"],
    column_widths=[0.4, 0.6]
)

# --- Initial frame ---
fig.add_trace(go.Bar(
    x=outcomes, y=p_market,
    marker_color=[colors["market_up"], colors["market_down"]],
    opacity=0.4, name="Market Implied"), row=1, col=1)
fig.add_trace(go.Bar(
    x=outcomes, y=empirical[0],
    marker_color=[colors["emp_up"], colors["emp_down"]],
    opacity=0.9, name="Empirical"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=[1], y=[empirical[0, 0]],
    mode="lines", line=dict(color=colors["emp_up"], width=2),
    name="Empirical P(Up)"), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[1], y=[p_market[0]],
    mode="lines", line=dict(color="white", dash="dot"),
    name="Market P(Up)"), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[1], y=[p_true[0]],
    mode="lines", line=dict(color="lime", dash="dot"),
    name="True P(Up)"), row=1, col=2)

# --- Layout ---
fig.update_layout(
    height=500, width=1000,
    title="Earnings Event: Market-Implied vs. Actual Probabilities",
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "x": 0.5, "y": -0.1,
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 40, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0}
            }]
        }]
    }]
)

# --- Axes ---
fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.3)")
fig.update_yaxes(range=[0, 0.8], showgrid=True, gridcolor="rgba(128,128,128,0.3)")

# Attach frames and show
fig.frames = frames
fig.show()

```

Unlike with a random space, probabilities don't converge and you can't plug them into the equation for edge

$$P(win) \neq \lim_{n \to \infty} \frac{\text{Number of Wins}}{n}$$
$$P(loss) \neq \lim_{n \to \infty} \frac{\text{Number of Losses}}{n}$$

The following equation still holds, but estimating the inputs is not a trivial problem

$$\mathbb{E}[\text{trade}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$

Moreover, just because you can act suboptimally easily it does not mean that it is gambling.  

Gambling demands luck to make money in a system that has edge fixed against you - this is not how trading works.

---

#### 2.) 📈 Myth 2: You Only Need to Be Correct 50.5% of the Time to be Profitable

This is **NOT** true in general, and is often a myth propogated by brokers and prop firms to get comission and 'training fees'

The proportion of time you need to be correct depends on your **average winner** and **average loser** 

Both of which depend on your **bet size** and whether or not the system is ergodic

$$\mathbb{E}[\text{trade}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss})$$
$$\text{Strategy A:} \quad \mathbb{E}[\text{trade}] = (0.9x)(0.505) + (-1x)(0.495) = -0.05x \text{ (Negative EV)}$$
$$\text{Strategy B:} \quad \mathbb{E}[\text{trade}] = (3.5x)(0.30) + (-1x)(0.70) = 0.35x \text{ (Positive EV)}$$

Bet size impacts, $\mathbb{E}[\text{winner}|\text{win}]$, $\mathbb{E}[\text{loser}|\text{loss}]$ which will dictate if you have an edge or not



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)

# Simulation parameters
n_trades = 200
starting_wealth = 1000
bet_size = 100

# Strategy A: Negative EV
p_win_A = 0.505
payout_A = 1.9  # Lose 1x, win 0.9x
expected_A = bet_size * (p_win_A * (payout_A - 1) - (1 - p_win_A))

# Strategy B: Positive EV
p_win_B = 0.30
payout_B = 4.5  # Lose 1x, win 3.5x
expected_B = bet_size * (p_win_B * (payout_B - 1) - (1 - p_win_B))

print(f"Expected gain per trade (A): {expected_A:.2f}")
print(f"Expected gain per trade (B): {expected_B:.2f}")

# Simulate single path for each strategy
wealth_A = np.zeros(n_trades + 1)
wealth_B = np.zeros(n_trades + 1)
wealth_A[0] = starting_wealth
wealth_B[0] = starting_wealth

# Run both strategies
for i in range(n_trades):
    # If already at 0, stay there (absorbing state)
    if wealth_A[i] == 0:
        wealth_A[i+1] = 0
    else:
        result_A = np.random.random() < p_win_A
        net_A = bet_size * ((payout_A - 1) if result_A else -1)
        wealth_A[i+1] = max(0, wealth_A[i] + net_A)

    if wealth_B[i] == 0:
        wealth_B[i+1] = 0
    else:
        result_B = np.random.random() < p_win_B
        net_B = bet_size * ((payout_B - 1) if result_B else -1)
        wealth_B[i+1] = max(0, wealth_B[i] + net_B)

# Create figure with 2 subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('50.5% Win Rate Strategy (Negative EV)',
                    '30% Win Rate Strategy (Positive EV)')
)

# Create animation frames
frames = []
for step in range(n_trades + 1):
    frame = {"data": [], "name": str(step)}

    # Axis limits
    y_min_A = min(np.min(wealth_A[:step+1]), 0)
    y_max_A = max(np.max(wealth_A[:step+1]), starting_wealth)
    y_min_B = min(np.min(wealth_B[:step+1]), 0)
    y_max_B = max(np.max(wealth_B[:step+1]), starting_wealth * 5)

    # Left (Strategy A)
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=wealth_A[:step + 1],
            mode='lines',
            line=dict(color='rgba(255,0,0,1)', width=3),
            name='Strategy A',
            showlegend=False
        )
    )

    # Right (Strategy B)
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=wealth_B[:step + 1],
            mode='lines',
            line=dict(color='rgba(0,255,0,1)', width=3),
            name='Strategy B',
            showlegend=False
        )
    )

    frame["layout"] = {
        "xaxis": {"range": [0, max(step + 1, 5)]},
        "xaxis2": {"range": [0, max(step + 1, 5)]},
        "yaxis": {"range": [y_min_A * 1.1, y_max_A * 1.1]},
        "yaxis2": {"range": [y_min_B * 1.1, y_max_B * 1.1]}
    }
    frames.append(frame)

# Initial traces
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[starting_wealth],
        mode='lines',
        line=dict(color='rgba(255,0,0,1)', width=3),
        name='Strategy A'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[0],
        y=[starting_wealth],
        mode='lines',
        line=dict(color='rgba(0,255,0,1)', width=3),
        name='Strategy B'
    ),
    row=1, col=2
)

# Layout and controls
fig.update_layout(
    height=500,
    width=1200,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': 'Play', 'method': 'animate',
             'args': [None, {'frame': {'duration': 50, 'redraw': True},
                             'fromcurrent': True, 'transition': {'duration': 0}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                               'mode': 'immediate', 'transition': {'duration': 0}}]}
        ]
    }]
)

# Axes setup
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
        title='Wealth ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.frames = frames
fig.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Optimal Bet Sizing and Ergodicity

To throw a wrench in this idea of expected value (EV) or edge, a positive edge only means we will accumulate wealth on average **in an ergodic system**

This is not a class on stochastic processes, so I will say effectively, if the system is ergodic the average path will behave as expected when EV is positive

If the system is non-ergodic, the average path will **NOT** behave as expected when EV is positive - in other words there will be a few *lucky* paths dominating the EV

##### In Both Systems Below Traders Operate with Positive Expected Value: $\mathbb{E}[\text{trade}] = \mathbb{E}[\text{winner}|\text{win}]P(\text{win}) + \mathbb{E}[\text{loser}|\text{loss}]P(\text{loss}) > 0$


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)

# Simulation parameters
n_traders = 25
n_trades = 100
starting_wealth = 1000
bet_fraction = 0.2  # 10% of current wealth per trade for multiplicative system
bet_size = 50      # fixed size for additive system

# --- LEFT: Ergodic (Additive) ---
p_win_ergodic = 0.52
payout_ergodic = 2.0  # fair 1:1 payout

# --- RIGHT: Non-Ergodic (Multiplicative) ---
p_win_nonergodic = 0.5
payout_nonergodic = 2.05  

# Calculate expected values per trade
ev_per_trade_ergodic = bet_size * (p_win_ergodic * (payout_ergodic - 1) - (1 - p_win_ergodic))
ev_per_trade_nonergodic = bet_fraction * (p_win_nonergodic * (payout_nonergodic - 1) - (1 - p_win_nonergodic))

# Initialize wealth matrices
wealth_ergodic = np.zeros((n_traders, n_trades + 1))
wealth_nonergodic = np.zeros((n_traders, n_trades + 1))
wealth_ergodic[:, 0] = starting_wealth
wealth_nonergodic[:, 0] = starting_wealth

# Simulate both systems
for i in range(n_trades):
    # Ergodic additive game
    results_e = np.random.random(n_traders) < p_win_ergodic
    net_e = results_e * bet_size * (payout_ergodic - 1) - (~results_e) * bet_size
    wealth_ergodic[:, i+1] = wealth_ergodic[:, i] + net_e

    # Non-ergodic multiplicative game
    results_ne = np.random.random(n_traders) < p_win_nonergodic
    # Percentage of current wealth bet
    delta = bet_fraction * wealth_nonergodic[:, i]
    gains = results_ne * delta * (payout_nonergodic - 1)
    losses = ~results_ne * delta
    wealth_nonergodic[:, i+1] = wealth_nonergodic[:, i] + gains - losses

    # Absorbing zero for non-ergodic
    wealth_nonergodic[:, i+1] = np.where(wealth_nonergodic[:, i+1] <= 0, 0, wealth_nonergodic[:, i+1])

# Create figure with secondary y-axis
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Ergodic System (Additive, Time Avg = Ensemble Avg)',
                   'Non-Ergodic System (Multiplicative, Time Avg ≠ Ensemble Avg)')
)

# Create colors with varying opacity
colors_left = [f'rgba(255,0,0,{0.1 + 0.2*i/n_traders})' for i in range(n_traders)]
colors_right = [f'rgba(0,255,0,{0.1 + 0.2*i/n_traders})' for i in range(n_traders)]

frames = []
for step in range(n_trades + 1):
    frame = {"data": [], "name": str(step)}

    # Axis limits
    y_min_L = min(np.min(wealth_ergodic[:, :step+1]), 0)
    y_max_L = max(np.max(wealth_ergodic[:, :step+1]), starting_wealth * 1.5)
    y_min_R = min(np.min(wealth_nonergodic[:, :step+1]), 0)
    y_max_R = max(np.max(wealth_nonergodic[:, :step+1]), starting_wealth * 10)

    # Left (ergodic)
    for i in range(n_traders):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=wealth_ergodic[i, :step + 1],
                mode='lines',
                line=dict(color=colors_left[i], width=2),
                showlegend=False
            )
        )
    
    # Add theoretical EV line for ergodic
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=starting_wealth + ev_per_trade_ergodic * np.arange(step + 1),
            mode='lines',
            line=dict(color='rgba(255,0,0,1)', width=3),
            showlegend=False
        )
    )

    # Right (non-ergodic)
    for i in range(n_traders):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=wealth_nonergodic[i, :step + 1],
                mode='lines',
                line=dict(color=colors_right[i], width=2),
                showlegend=False
            )
        )
    
    # Add theoretical EV line for non-ergodic
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=starting_wealth * (1 + ev_per_trade_nonergodic) ** np.arange(step + 1),
            mode='lines',
            line=dict(color='rgba(0,255,0,1)', width=3),
            showlegend=False
        )
    )

    frame["layout"] = {
        "xaxis": {"range": [0, n_trades]},
        "xaxis2": {"range": [0, n_trades]},
        "yaxis": {"range": [y_min_L * 1.1, y_max_L * 1.1]},
        "yaxis2": {"range": [y_min_R * 1.1, y_max_R * 1.1]}
    }
    frames.append(frame)

# Add initial traces for line plots
for i in range(n_traders):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=colors_left[i], width=2),
            name='Trader (Ergodic)' if i == 0 else None,
            showlegend=(i == 0)
        ),
        row=1, col=1
    )

# Add initial EV line trace for ergodic
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[starting_wealth],
        mode='lines',
        line=dict(color='rgba(255,0,0,1)', width=3),
        name='Expected Value',
        showlegend=True
    ),
    row=1, col=1
)

for i in range(n_traders):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=colors_right[i], width=2),
            name='Trader (Non-Ergodic)' if i == 0 else None,
            showlegend=(i == 0)
        ),
        row=1, col=2
    )

# Add initial EV line trace for non-ergodic
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[starting_wealth],
        mode='lines',
        line=dict(color='rgba(0,255,0,1)', width=3),
        name='Expected Value',
        showlegend=False
    ),
    row=1, col=2
)

# Layout
fig.update_layout(
    height=600,
    width=1200,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': 'Play', 'method': 'animate',
             'args': [None, {'frame': {'duration': 50, 'redraw': True},
                             'fromcurrent': True, 'transition': {'duration': 0}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                               'mode': 'immediate', 'transition': {'duration': 0}}]}
        ]
    }]
)

# Axes for line plots
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
        title='Wealth ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.frames = frames
fig.show()

```


```python
# Calculate proportion of profitable traders
final_wealth_ergodic = wealth_ergodic[:, -1]
final_wealth_nonergodic = wealth_nonergodic[:, -1]

prop_profitable_ergodic = np.mean(final_wealth_ergodic > starting_wealth)
prop_profitable_nonergodic = np.mean(final_wealth_nonergodic > starting_wealth)

# Create bar chart
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Proportion of Profitable Traders (Ergodic)',
                   'Proportion of Profitable Traders (Non-Ergodic)')
)

# Left bar (ergodic)
fig2.add_trace(
    go.Bar(
        x=['Profitable', 'Unprofitable'],
        y=[prop_profitable_ergodic, 1-prop_profitable_ergodic],
        marker_color=['green', 'red'],
        showlegend=False
    ),
    row=1, col=1
)

# Right bar (non-ergodic) 
fig2.add_trace(
    go.Bar(
        x=['Profitable', 'Unprofitable'],
        y=[prop_profitable_nonergodic, 1-prop_profitable_nonergodic],
        marker_color=['green', 'red'],
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig2.update_layout(
    height=400,
    width=1200,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_range=[0, 1],
    yaxis2_range=[0, 1]
)

# Update axes
for i in range(1, 3):
    fig2.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=1, col=i
    )
    fig2.update_yaxes(
        title='Proportion of Traders',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        tickformat=',.0%',
        row=1, col=i
    )

fig2.show()

```

##### Clearly, there is far more than being "right" 50.5\% of the time to be profitable!

---

#### 3.) ⚠️ Myth 3: Trading Strategies are Fixed, Always Work, Nobody Shares Profitable Strategies

This is not true, trading is quite **literally** a full time job

There are general strategies you can follow but there is no **one man set and forget hedge fund**

Strategies require constant optimization and careful monitoring which is why *understanding* popular academic models can be useful

These models help explain why this is the case, the dynamics of the market are constantly changing

The goal of every trader is to find a set of actions

$$
\pi^* = \underset{\pi}{\operatorname{argmax}} \; \frac{\mathbb{E}_{\pi}[P]}{\sigma_{\pi}(P)}
$$

A policy function is not just a *trading strategy* it is far more dynamic

It is an action from a *collection of actions* we choose to take based on our current environment - this is built from **knowledge and experience** 

**Nobody - no broker, news outlet, or guru trader can gift you a $\pi^*$ function**

###### ______________________________________________________________________________________________________________________________________


##### Effective Policy Function $\pi$

- You retire quant strategies (dead alphas) and trade alpha that is stable 

- Your trading model is robust out of sample to different regimes

- You understand how to trade in different regimes


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

np.random.seed(42)

# ----------------------------
# Generate sample strategy data
# ----------------------------
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

target_annual_return = 0.20   # 20% annual return
target_annual_vol = 0.073     # to get ~2.75 Sharpe
daily_vol = target_annual_vol / np.sqrt(252)
daily_return = target_annual_return / 252

returns = np.random.normal(daily_return, daily_vol, n_days)
returns = returns + 0.1 * returns**2  # add slight positive skew
equity = 100000 * (1 + returns).cumprod()

# Split into backtest and live
backtest_end = pd.Timestamp('2022-01-01')
backtest_mask = dates <= backtest_end
live_mask = dates > backtest_end

backtest_returns = returns[backtest_mask]
live_returns = returns[live_mask]

backtest_sharpe = np.sqrt(252) * np.mean(backtest_returns) / np.std(backtest_returns)
live_sharpe = np.sqrt(252) * np.mean(live_returns) / np.std(live_returns)
ev = np.mean(returns) * 100

# ----------------------------
# Build animation frames
# ----------------------------
frames = []
for i in range(1, n_days + 1):
    partial_dates = dates[:i]
    partial_equity = equity[:i]
    mask_bt = partial_dates <= backtest_end
    mask_live = partial_dates > backtest_end

    frame_data = []

    # Backtest line
    if mask_bt.any():
        frame_data.append(go.Scatter(
            x=partial_dates[mask_bt],
            y=partial_equity[mask_bt],
            mode='lines',
            line=dict(color='rgba(0,255,255,0.9)', width=2),
            name='Pre-Period',
            showlegend=False
        ))

    # Live line
    if mask_live.any():
        frame_data.append(go.Scatter(
            x=partial_dates[mask_live],
            y=partial_equity[mask_live],
            mode='lines',
            line=dict(color='rgba(0,255,0,0.9)', width=2),  # solid green
            name='Post-Period',
            showlegend=False
        ))

    # Keep static split line visible
    frame_data.append(go.Scatter(
        x=[backtest_end, backtest_end],
        y=[np.min(equity), np.max(equity)],
        mode='lines',
        line=dict(color='rgba(150,150,150,0.7)', width=2, dash='dash'),
        name='Backtest/Live/Regime Split',
        showlegend=False
    ))

    frames.append(go.Frame(data=frame_data, name=str(i)))

# ----------------------------
# Base figure setup
# ----------------------------
fig = go.Figure(
    data=[
        # Backtest static trace (for legend)
        go.Scatter(
            x=dates[backtest_mask],
            y=equity[backtest_mask],
            mode='lines',
            line=dict(color='rgba(0,255,255,0.9)', width=2),
            name=f'Backtest (Sharpe: {backtest_sharpe:.2f}, EV: {ev:.3f}%/day)',
            visible=True
        ),
        # Live static trace (for legend)
        go.Scatter(
            x=dates[live_mask],
            y=equity[live_mask],
            mode='lines',
            line=dict(color='rgba(0,255,0,0.9)', width=2),
            name=f'Live Trading (Sharpe: {live_sharpe:.2f})',
            visible=True
        ),
        # Static split line (for legend)
        go.Scatter(
            x=[backtest_end, backtest_end],
            y=[equity.min(), equity.max()],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.7)', width=2, dash='dash'),
            name='Backtest/Live/Regime Split',
            visible=True
        )
    ],
    layout=go.Layout(
        title='Effective Policy Function (2020–2022)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=450,
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # Keep x-axis fixed for full range
        xaxis=dict(range=[dates[0], dates[-1]]),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 10, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 0}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                   'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }]
    ),
    frames=frames
)

# ----------------------------
# Styling and gridlines
# ----------------------------
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

```

###### ______________________________________________________________________________________________________________________________________


##### Ineffective Policy Function $\pi$

**Why this is happening depends on the time of trader you are. . .**

- Your quant strategy's alpha is decaying

- You overfit data to a backtest

- You don't understand how to trade in different regimes


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

np.random.seed(42)

# ----------------------------
# Generate sample "ineffective" strategy data
# ----------------------------
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

# Pre-period (strong performance)
target_annual_return_pre = 0.20   # 20% annual return
target_annual_vol_pre = 0.073     # Sharpe ~2.75
daily_vol_pre = target_annual_vol_pre / np.sqrt(252)
daily_return_pre = target_annual_return_pre / 252

# Post-period (performance degradation)
target_annual_return_post = -0.05   # loses money
target_annual_vol_post = 0.10       # more volatile
daily_vol_post = target_annual_vol_post / np.sqrt(252)
daily_return_post = target_annual_return_post / 252

# Split into pre and post periods
regime_split = pd.Timestamp('2022-01-01')
dates_pre = dates[dates <= regime_split]
dates_post = dates[dates > regime_split]
n_pre = len(dates_pre)
n_post = len(dates_post)

# Generate returns
returns_pre = np.random.normal(daily_return_pre, daily_vol_pre, n_pre)
returns_pre = returns_pre + 0.1 * returns_pre**2  # add slight positive skew
returns_post = np.random.normal(daily_return_post, daily_vol_post, n_post)

# Combine
returns = np.concatenate([returns_pre, returns_post])
equity = 100000 * (1 + returns).cumprod()

# Masks
pre_mask = dates <= regime_split
post_mask = dates > regime_split

# Metrics
pre_sharpe = np.sqrt(252) * np.mean(returns_pre) / np.std(returns_pre)
post_sharpe = np.sqrt(252) * np.mean(returns_post) / np.std(returns_post)
ev = np.mean(returns) * 100  # mean daily return %

# ----------------------------
# Build animation frames
# ----------------------------
frames = []
for i in range(1, n_days + 1):
    partial_dates = dates[:i]
    partial_equity = equity[:i]
    mask_pre = partial_dates <= regime_split
    mask_post = partial_dates > regime_split

    frame_data = []

    # Pre-period line
    if mask_pre.any():
        frame_data.append(go.Scatter(
            x=partial_dates[mask_pre],
            y=partial_equity[mask_pre],
            mode='lines',
            line=dict(color='rgba(0,255,255,0.9)', width=2),
            name='Pre-Period',
            showlegend=False
        ))

    # Post-period line (red for degradation)
    if mask_post.any():
        frame_data.append(go.Scatter(
            x=partial_dates[mask_post],
            y=partial_equity[mask_post],
            mode='lines',
            line=dict(color='rgba(255,100,100,0.9)', width=2),  # solid red
            name='Post-Period',
            showlegend=False
        ))

    # Static regime split line
    frame_data.append(go.Scatter(
        x=[regime_split, regime_split],
        y=[np.min(equity), np.max(equity)],
        mode='lines',
        line=dict(color='rgba(150,150,150,0.7)', width=2, dash='dash'),
        name='Regime Split',
        showlegend=False
    ))

    frames.append(go.Frame(data=frame_data, name=str(i)))

# ----------------------------
# Base figure setup
# ----------------------------
fig = go.Figure(
    data=[
        # Pre-period static trace (legend)
        go.Scatter(
            x=dates[pre_mask],
            y=equity[pre_mask],
            mode='lines',
            line=dict(color='rgba(0,255,255,0.9)', width=2),
            name=f'Pre-Period (Sharpe: {pre_sharpe:.2f}, EV: {ev:.3f}%/day)',
            visible=True
        ),
        # Post-period static trace (legend)
        go.Scatter(
            x=dates[post_mask],
            y=equity[post_mask],
            mode='lines',
            line=dict(color='rgba(255,100,100,0.9)', width=2),
            name=f'Post-Period (Sharpe: {post_sharpe:.2f})',
            visible=True
        ),
        # Static regime split line (legend)
        go.Scatter(
            x=[regime_split, regime_split],
            y=[equity.min(), equity.max()],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.7)', width=2, dash='dash'),
            name='Backtest/Live/Regime Split',
            visible=True
        )
    ],
    layout=go.Layout(
        title='Ineffective Policy Function (2020–2022)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=450,
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # Fixed x-axis for full range
        xaxis=dict(range=[dates[0], dates[-1]]),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 10, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 0}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                   'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }]
    ),
    frames=frames
)

# ----------------------------
# Styling and gridlines
# ----------------------------
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

```

###### ______________________________________________________________________________________________________________________________________


##### Nobody Shares Profitable Trading Strategies

Probably one of the most misunderstood ideas in the entire trading space is the idea of a strategy

**With the capital and strategies you are dealing with you are not going to get your edge or alpha crowded**

An institutional quant strategy, **YES** that can have it's alpha crowed which is why you see this notion of *garden leave* so researchers can't turn around and go to another firm and deploy the strategy there and dry up the statistical inefficiency

This is why nobody can **teach you to trade**, it's like asking someone to **teach you** to be a baseball player

I can tell you to "hit a fastball", that's the strategy - if you can't swing the bat and connect with the ball that's on you

*WHICH IS WHY NOT EVERY IS A BASEBALL PLAYER AND WHY NOT EVERYONE SHOULD BE A TRADER*

This is not a get rich quick scheme, you must have knowledge, experience, AND a tolerance for bearing risk and losses

There are general classes of strategies that are used *everyday*
- mean reversion
- overstated volatility
- momentum and sentiment

But nobody is going to sit there and trade for you, tell you when to enter/exit, why, how to manage risk, when to cut losses. . .

###### ______________________________________________________________________________________________________________________________________

##### Example: Poker with Pocket Aces

Theoretically, we should fold over half the time, if we're flipping - unconditionally, we will loser over half the time against 5 players

What if we could act optimally?  What if we knew our opponent played really tight, could be bluffed out - even with a stronger hand?

I can share the general strategy and theory with you, the raw probabilities, but it's about execution, that's up to you. . .

This is why a *single backtest* can't disprove anything (even technical analysis, yes, I know. . .), it's all about *this one hand* or *this one traded*


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PMF Setup ---
scenarios = ['Win', 'Lose'] 
pmf = np.array([0.46, 0.54])  # theoretical probability of winning with AA vs 5 players

# --- Simulation Parameters ---
n_rounds = 100
initial_wealth = 1000
bet_size = 25
np.random.seed(42)

# Actual probabilities (60/40 due to fold equity)
actual_probs = [0.60, 0.40]

# Calculate payouts (pot odds 5:1 since 5 players)
payouts = {'Win': 5.0, 'Lose': 0}

# --- Simulate Hands ---
wealth_path = np.zeros(n_rounds)
wealth_path[0] = initial_wealth
outcomes_sequence = []

for i in range(1, n_rounds):
    if wealth_path[i-1] <= 0:
        wealth_path[i:] = 0
        outcomes_sequence.extend(['Lose'] * (n_rounds - len(outcomes_sequence)))
        break
        
    # 60/40 real-world result with fold equity
    win = np.random.random() < actual_probs[0]
    
    if win:
        wealth_path[i] = wealth_path[i-1] + bet_size * payouts['Win']
        outcomes_sequence.append('Win')
    else:
        wealth_path[i] = wealth_path[i-1] - bet_size
        outcomes_sequence.append('Lose')

# Pad outcomes if needed        
if len(outcomes_sequence) < n_rounds:
    outcomes_sequence += ['Lose'] * (n_rounds - len(outcomes_sequence))

# Calculate empirical probabilities
win_count = outcomes_sequence.count('Win')
lose_count = outcomes_sequence.count('Lose')
empirical_pmf = np.array([win_count/n_rounds, lose_count/n_rounds])

# --- Animation Frames ---
frames = []
zero_index = np.argmax(wealth_path == 0)
if zero_index == 0: zero_index = n_rounds
extra_frames = 10

for i in range(1, min(zero_index + extra_frames, n_rounds)):
    colors = ['green', 'red']
    line_widths = [0, 0]
    highlight_idx = scenarios.index(outcomes_sequence[i])
    line_widths[highlight_idx] = 4
    
    # Calculate running empirical probabilities
    current_wins = outcomes_sequence[:i+1].count('Win')
    current_loses = len(outcomes_sequence[:i+1]) - current_wins
    current_empirical = [current_wins/(i+1), current_loses/(i+1)]
    
    frames.append(go.Frame(
        data=[
            go.Bar(
                x=scenarios,
                y=pmf,
                name='Theoretical',
                marker_color=colors,
                marker_line_width=line_widths,
                marker_line_color='yellow',
                opacity=0.5
            ),
            go.Bar(
                x=scenarios,
                y=current_empirical,
                name='Empirical',
                marker_color=['lightgreen', 'pink'],
                opacity=0.5
            ),
            go.Scatter(
                x=np.arange(i + 1),
                y=wealth_path[:i+1],
                mode='lines',
                line=dict(color='gold', width=2)
            )
        ],
        layout=go.Layout(
            xaxis2=dict(range=[0, i + 10])
        ),
        name=f'frame{i}'
    ))

# --- Base Figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Pocket Aces vs 5 Players Probabilities', 'Player Wealth Path'),
    column_widths=[0.3, 0.7]
)

# PMF subplot - both theoretical and empirical
fig.add_trace(
    go.Bar(
        x=scenarios,
        y=pmf,
        name='Theoretical',
        marker_color=['green', 'red'],
        marker_line_width=[0, 0],
        marker_line_color='yellow',
        opacity=0.5
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=scenarios,
        y=[0, 0],  # Start with zeros
        name='Empirical',
        marker_color=['lightgreen', 'pink'],
        opacity=0.5
    ),
    row=1, col=1
)

# Wealth subplot
fig.add_trace(
    go.Scatter(
        y=[initial_wealth],
        mode='lines',
        line=dict(color='gold', width=2)
    ),
    row=1, col=2
)

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=500,
    width=1000,
    showlegend=False,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    title_text="Pocket Aces vs 5 Players | Theoretical: 46/54 | Actual with Fold Equity: 60/40",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Axes
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(range=[0, 10000], row=1, col=2)
fig.update_yaxes(range=[0, 1], row=1, col=1)  # Set y-axis range for bar chart

fig.show()

```

I've shared plenty of "profitable" strategies with you on this channel

- Cross-sectional social sentiment
- Selling overvalued volatility
- Mean reversion to a calibration vol surface

People demand my P/L, I've shared this - then they're disatisfied with it (lol)

How many people are *actually* going to setup a paper trading account to train their $\pi^*$ function?  

Exactly, it's work - trading is literally a job, not a get rich quick scheme. . .

---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Myth 1: Trading is Gambling and the Market is Random

        - Trading is not gambling, there is no fixed negative edge against the player, the market is uncertain not random

- Myth 2: You Only Need to Be Correct 50.5% of the Time to be Profitable

        - This is **NOT TRUE** and depends on bet size and ergodicity

- Myth 3: Trading Strategies are Fixed and Always Work, Nobody Shares Profitable Strategies

        - I can tell you exactly how I make money, it does not mean it is going to continue to work, or that you can execute on it

**Future Topics**

Technical Videos and Other Discussions

- Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
- Stochastic Proccesses: Brownian Motion, Arithmetic (additive) Geometric (multiplicative) Brownian Motion
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
- Most Popular Quant Models for Informed Trading
- Buy Side vs. Sell Side Quants

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
- Automated Delta-Neutral Trading System

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
