# ### ⚔️ Quant vs. Discretionary Trading
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
# 
# - [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)
# 
# - [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)
# 
# - [Why Trading Metrics are Misleading (Unless This is True)](https://youtu.be/K-aUu_M02-Y)
# 
# - [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/JuD3KGQhofw)
# 
# - [Quant Proves Trading Can't Be Taught (But You CAN Learn This)](https://youtu.be/uivlsPk0WLQ)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
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
# #### 1.) 📈 Understanding the Trading Space
# 
# - Randomness vs. Uncertainty
# 
# - Trading is About Uncertainty and Assuming Risk
# 
# - The Goal of Every Trader
# 
# #### 2.) 🏛️ Discretionary Trading
# 
# - Experience Based Dynamic Trading
# 
# - Performance Stability and Risks
# 
# #### 3.) 🧮 Quantitative Trading
# 
# - Trading Fixed Statistical Mispricings
# 
# - Performance Stability and Risks
# 
# #### 4.) ⚔️ Quant vs. Discretionary Trading
# 
# - Similarities and Differences
# 
# - Which is Better?  What Should You Do?
# 
# #### 3.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📈 Understanding the Trading Space


# ##### Is Trading Random or Uncertain?
# 
# The trading space is either random or uncertain
# 
# **Random:** Outcomes abide by fixed randomness
# 
# **Uncertain:** A combination of deterministic and dynamic random components
# 
# *Examples of Random:*
# - Powerball Ticket
# - Craps
# - Slots
# - Roulette
# 
# *Examples of Uncertain:*
# - Poker
# - Event Betting
# - Election Betting
# - Trading
# 
# **In either case. . .**
# 
# Whenever we are dealing with an outcome that is unknown we put individual outcomes in terms of expected value (EV) also known as our *edge*
# 
# Law of Total Expectation (LoTE) for our edge:
# 
#  $$\mathbb{E}[Edge] = \mathbb{E}[Edge|Winner] \cdot P(Winner) + \mathbb{E}[Edge|Loser] \cdot P(Loser)$$
# 
# ##### This tells us what happens to our wealth over time if we continue to operate in that random or uncertain system (assuming *ergodicity*)


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



# **⚔️ Randomness vs. Uncertainty**
# 
# A major point of confusion for students and folks first getting involved in the space. . .
# 
# **Similarity** 
# - The outcome of any one event is unknown  (*true* for both!)
# 
# **Difference** 
# - The statistics of any one event is well-defined and stable  (*true* for randomness, *not true* for uncertainty!)
# - Our choices (actions) impact our expected value (*not true* for randomness, *true* for uncertainty!)
# 
# ##### We will observe two intuitive examples: American Roulette (Randomness) and Election Betting (Uncertainty)
# 
# as both a quant and discretionary trader


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example Randomness: American Roulette
# 
# Systems that involve pure randomness have a fixed edge, the expected value **DOES NOT** change
# 
# Law of Total Expectation (LoTE) for American Roulette:
#   
#  $$\mathbb{E}[X] = \mathbb{E}[X|Red] \cdot P(Red) + \mathbb{E}[X|Black] \cdot P(Black) + \mathbb{E}[X|Green] \cdot P(Green)$$
#  $$\mathbb{E}[X] = \frac{18}{38} - \frac{18}{38} - \frac{2}{38} = -\frac{2}{38} \approx -5.26\%$$
#   
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate PMF data for roulette (American roulette)
outcomes = ['Green', 'Red', 'Black']
pmf = np.array([2/38, 18/38, 18/38])

# Generate simulated wealth path
n_spins = 500
initial_wealth = 1000
bet_size = 100
bets = np.random.random(n_spins) < (18/38)

wealth_path = np.zeros(n_spins)
wealth_path[0] = initial_wealth
outcomes_sequence = []

for i in range(1, n_spins):
    if wealth_path[i-1] > 0:
        wealth_path[i] = wealth_path[i-1] + (bet_size if bets[i] else -bet_size)
        if bets[i]:
            outcomes_sequence.append('Red')
        else:
            loss_outcome = np.random.choice(['Black', 'Green'], p=[18/20, 2/20])
            outcomes_sequence.append(loss_outcome)
        if wealth_path[i] <= 0:
            wealth_path[i:] = 0
            outcomes_sequence.extend(['Black'] * (n_spins - len(outcomes_sequence)))
            break
    else:
        outcomes_sequence.append('Black')

# Ensure the same length
if len(outcomes_sequence) < n_spins:
    outcomes_sequence += ['Black'] * (n_spins - len(outcomes_sequence))



# Create frames — stop shortly after player goes bankrupt
frames = []

# Find the index where wealth first hits zero
zero_index = np.argmax(wealth_path == 0)
if zero_index == 0:  # if never hits zero
    zero_index = n_spins

extra_frames = 10  # 🟢 number of extra frames to show after hitting 0

for i in range(1, min(zero_index + extra_frames, n_spins)):
    colors = ['green', 'red', 'black']
    line_widths = [0, 0, 0]
    
    # After bankrupt, highlight 'Black' (loss)
    if i < zero_index:
        highlight_idx = outcomes.index(outcomes_sequence[i])
    else:
        highlight_idx = outcomes.index('Black')
    
    line_widths[highlight_idx] = 4

    frames.append(go.Frame(
        data=[
            go.Bar(
                x=outcomes,
                y=pmf,
                marker_color=colors,
                marker_line_width=line_widths,
                marker_line_color='yellow'
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




# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('American Roulette PMF', 'Player Wealth Path'),
                    column_widths=[0.3, 0.7])

# Initial PMF
fig.add_trace(
    go.Bar(
        x=outcomes,
        y=pmf,
        marker_color=['green', 'red', 'black'],
        marker_line_width=[0, 0, 0],
        marker_line_color='yellow'
    ),
    row=1, col=1
)

# Initial wealth line
fig.add_trace(
    go.Scatter(
        y=[initial_wealth],
        mode='lines',
        line=dict(color='gold', width=2)
    ),
    row=1, col=2
)

# Apply frames properly
fig.frames = frames

# Layout
fig.update_layout(
    height=500,
    width=1000,
    showlegend=False,
    title_text="American Roulette: PMF and Player Wealth Path",
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

# Axes style
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
# Fix y-axis range for the wealth path subplot
fig.update_yaxes(
    range=[0, 1500],  # bottom first, top second (Plotly uses ascending order)
    row=1, col=2
)


fig.show()



# There is **NO** disagreement at this point among quant and discretionary traders
# 
# **Both Quants and Discretionary Traders**
# 
# would agree (beyond martingaling roulette, yes I know) there is **NO** optimal action beyond not playing the game


# ###### ______________________________________________________________________________________________________________________________________


# ##### Randomness is Stable by the Law of Large Numbers
# 
# The Strong Law of Large Numbers (SLLN) states that for i.i.d. random variables $X_1, X_2, ...$:
#  
# $$P(\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^n X_i = \mathbb{E}[X]) = 1$$
#  
# This means that the sample average converges *almost surely* to the expected value.
# 


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
            # Running total of spins for context
            go.Scatter(
                x=np.arange(1, i+1),
                y=[empirical_pmfs[j][1] for j in range(i)],  # track red probability over time
                mode='lines',
                line=dict(color='red', width=2),
                name='Empirical P(Red)'
            ),
            go.Scatter(
                x=np.arange(1, i+1),
                y=[theoretical_pmf[1]] * i,
                mode='lines',
                line=dict(color='white', width=1, dash='dot'),
                name='True P(Red)'
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
    subplot_titles=('Empirical vs Theoretical PMF', 'Convergence of P(Red)'),
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

# Initial convergence line
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
        y=[theoretical_pmf[1]],
        mode='lines',
        line=dict(color='white', width=1, dash='dot'),
        name='True P(Red)'
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



# ###### ______________________________________________________________________________________________________________________________________


# ##### Example Uncertain: Election Betting and Implied Probability
# 
# These events are uncertain, modelled as random
# 
# Folks with insider or *better* information can act optimally and profit
# 
# $$\text{Implied probability} = \frac{1}{\text{decimal odds}} = \frac{\text{stake}}{\text{stake} + \text{payout}}$$
# 
# *Critically there is NO convergence*


import numpy as np
import plotly.graph_objects as go

# Simulation parameters
n_steps = 200
event_step = 150  
np.random.seed(42)

# Base probabilities before the event
pA = np.zeros(n_steps)
pB = np.zeros(n_steps)
pA[0] = 0.8
pB[0] = 0.2

# Generate random fluctuations
for i in range(1, event_step):
    change = np.random.normal(0, 0.01)
    pA[i] = np.clip(pA[i-1] + change, 0, 1)
    pB[i] = 1 - pA[i]

# After the event, collapse instantly
pA[event_step:] = 0
pB[event_step:] = 1

# Create frames for animation
frames = []
for i in range(1, n_steps):
    frames.append(go.Frame(
        data=[
            go.Scatter(
                x=np.arange(i + 1),
                y=pA[:i + 1],
                mode='lines',
                line=dict(color='blue', width=3),
                name='Democrat'
            ),
            go.Scatter(
                x=np.arange(i + 1),
                y=pB[:i + 1],
                mode='lines',
                line=dict(color='red', width=3),
                name='Republican'
            ),
            go.Scatter(
                x=[event_step, event_step],
                y=[0, 1],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dot'),
                name='Event'
            )
        ],
        layout=go.Layout(
            title_text="Election Implied Probabilities",
        ),
        name=f'frame{i}'
    ))

# Base figure setup
fig = go.Figure(
    data=[
        go.Scatter(
            x=[0],
            y=[pA[0]],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Democrat'
        ),
        go.Scatter(
            x=[0],
            y=[pB[0]],
            mode='lines',
            line=dict(color='red', width=3),
            name='Republican'
        ),
        go.Scatter(
            x=[event_step, event_step],
            y=[0, 1],
            mode='lines',
            line=dict(color='yellow', width=2, dash='dot'),
            name='Event'
        )
    ],
    layout=go.Layout(
        title="Election Implied Probabilities",
        xaxis=dict(title='Time', range=[0, n_steps]),
        yaxis=dict(title='Implied Probability', range=[0, 1]),
        height=500,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 60, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }]
    ),
    frames=frames
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### We can profit systematically if we are *more correct* than the probability implies
# 
# In other words, if we *believe* one probability is over/understated and we are correct *on average* we can profit from this edge


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PMF Setup ---
candidates = ['Democrat', 'Republican']
pmf = np.array([0.8, 0.2])  # implied probabilities (odds)

# --- Simulation Parameters ---
n_rounds = 100
initial_wealth = 1000
bet_size = 50
np.random.seed(42)

# Actual probabilities (50/50 reality)
actual_probs = [0.3, 0.7]

# Calculate payouts implied by PMF (1/p - 1)
# Democrat payout: 1/0.8 - 1 = 0.25x, Republican payout: 1/0.2 - 1 = 4x
payouts = { 'Democrat': 0.25, 'Republican': 4.0 }

# --- Simulate Bets ---
wealth_path = np.zeros(n_rounds)
wealth_path[0] = initial_wealth
outcomes_sequence = []

for i in range(1, n_rounds):
    if wealth_path[i-1] <= 0:
        wealth_path[i:] = 0
        outcomes_sequence.extend(['Democrat'] * (n_rounds - len(outcomes_sequence)))
        break

    # 50/50 real-world result
    republican_wins = np.random.random() < 0.5

    if republican_wins:
        # Player bet on Republican
        wealth_path[i] = wealth_path[i-1] + bet_size * payouts['Republican']
        outcomes_sequence.append('Republican')
    else:
        wealth_path[i] = wealth_path[i-1] - bet_size
        outcomes_sequence.append('Democrat')

# Pad outcomes if needed
if len(outcomes_sequence) < n_rounds:
    outcomes_sequence += ['Democrat'] * (n_rounds - len(outcomes_sequence))

# --- Animation Frames ---
frames = []
zero_index = np.argmax(wealth_path == 0)
if zero_index == 0: zero_index = n_rounds
extra_frames = 10

for i in range(1, min(zero_index + extra_frames, n_rounds)):
    colors = ['blue', 'red']
    line_widths = [0, 0]
    highlight_idx = candidates.index(outcomes_sequence[i])
    line_widths[highlight_idx] = 4

    frames.append(go.Frame(
        data=[
            go.Bar(
                x=candidates,
                y=pmf,
                marker_color=colors,
                marker_line_width=line_widths,
                marker_line_color='yellow'
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
    subplot_titles=('Election Implied Probabilities (PMF)', 'Player Wealth Path (Betting on Republican)'),
    column_widths=[0.3, 0.7]
)

# PMF subplot
fig.add_trace(
    go.Bar(
        x=candidates,
        y=pmf,
        marker_color=['blue', 'red'],
        marker_line_width=[0, 0],
        marker_line_color='yellow'
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
    title_text="Election Implied Probabilities and Player Wealth Path | Market: 80/20 | Actual: 70/30",
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

fig.show()



# This is where a quant and discretionary trader will begin to take different perspectives
# 
# **Quants:** 
# - We observe the market systematically overstates the probability of democratic candidates leaving edge on the table for republican trades
# 
# **Discretionary:** 
# - I've seen this before, no way republican is $20\%$ that's way understated and I can profit betting on this outcome
# 
# *They can both be right*, one takes a more rigid/fixed/structured way of going about maximizing EV the other a more dynamic experience based perspective


# ###### ______________________________________________________________________________________________________________________________________


# ##### We Can't Actually Observe the PMF Chart on the Left, Convergence May Occur at Other Levels
# 
# So what use is implied probability without convergence?  Exactly.  


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Setup ---
candidates = ['Democrat', 'Republican']

# Market-implied probabilities (beliefs)
implied_pmf = np.array([0.8, 0.2])

# True (unknown) underlying probabilities
actual_pmf = np.array([0.7, 0.3])  # what reality might be

# Simulate repeated "elections" (hypothetical)
n_trials = 500
np.random.seed(42)
outcomes = np.random.choice(candidates, size=n_trials, p=actual_pmf)

# Running empirical PMF
empirical_pmfs = []
counts = np.zeros(len(candidates))
for i, outcome in enumerate(outcomes, start=1):
    counts[candidates.index(outcome)] += 1
    empirical_pmf = counts / i
    empirical_pmfs.append(empirical_pmf.copy())

# --- Create subplots ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.4, 0.6],
    subplot_titles=(
        "Market-Implied vs Empirical Probabilities",
        "Law of Large Numbers: Convergence Toward True 70/30"
    )
)

# --- Initial traces (frame 0) ---
# Left panel: Bar chart
fig.add_trace(
    go.Bar(
        x=candidates,
        y=implied_pmf,
        marker_color=['blue', 'red'],
        opacity=0.6,
        name='Implied PMF'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=candidates,
        y=empirical_pmfs[0],
        marker_color=['lightblue', 'tomato'],
        opacity=0.9,
        name='Empirical PMF'
    ),
    row=1, col=1
)

# Right panel: LLN convergence chart
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[empirical_pmfs[0][0]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Empirical Democrat',
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[empirical_pmfs[0][1]],
        mode='lines',
        line=dict(color='red', width=2),
        name='Empirical Republican',
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1, n_trials],
        y=[actual_pmf[0]]*2,
        mode='lines',
        line=dict(color='blue', width=1, dash='dot'),
        name='True Democrat (70%)',
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1, n_trials],
        y=[actual_pmf[1]]*2,
        mode='lines',
        line=dict(color='red', width=1, dash='dot'),
        name='True Republican (30%)',
    ),
    row=1, col=2
)

# --- Animation Frames ---
frames = []
for i in range(1, n_trials):
    frames.append(go.Frame(
        data=[
            # Left panel
            go.Bar(x=candidates, y=implied_pmf, marker_color=['blue', 'red'], opacity=0.6),
            go.Bar(x=candidates, y=empirical_pmfs[i], marker_color=['lightblue', 'tomato'], opacity=0.9),
            # Right panel (LLN)
            go.Scatter(x=np.arange(1, i+1), y=[empirical_pmfs[j][0] for j in range(i)], mode='lines', line=dict(color='blue', width=2)),
            go.Scatter(x=np.arange(1, i+1), y=[empirical_pmfs[j][1] for j in range(i)], mode='lines', line=dict(color='red', width=2)),
            go.Scatter(x=[1, n_trials], y=[actual_pmf[0]]*2, mode='lines', line=dict(color='blue', width=1, dash='dot')),
            go.Scatter(x=[1, n_trials], y=[actual_pmf[1]]*2, mode='lines', line=dict(color='red', width=1, dash='dot'))
        ],
        layout=go.Layout(
            annotations=[
                dict(
                    text="We only observe one election — this convergence is hypothetical.",
                    x=0.5, y=1.75, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14, color='lightgray')
                )
            ]
        ),
        name=f'frame{i}'
    ))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=550,
    width=1050,
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
    title_text="Election Implied vs Empirical Probabilities<br>Market: 80/20 | True Process: 70/30 (Unobservable in Reality)",
    title_y=0.93,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05,  # 🟢 moved play button to the left
        'y': -0.15,
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

# --- Axis styling ---
fig.update_xaxes(title_text="Candidate", row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text="Probability", row=1, col=1, range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text="Number of Trials", row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text="Empirical Probability", row=1, col=2, range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()



# Sure it can "predict" the outcome most of the time, basic regressions show evidence of this - *BUT WHAT ABOUT WHEN IT CAN'T*
# 
# **Quants**
# - Aim to profit in a fixed/rigid/structured approach
# 
# **Discretionary**
# - Aim to profit in a dynamic experience based approach
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Explaining Why Technical Analysis is NEVER Disproven
# 
# As a quant I would love to outright disprove technical analysis *BUT THIS IS WHY YOU CAN'T* disprove it
# 
# But the first rule of science is lack of evidence **DOES NOT CONSTITUTE** lack of existence - this goes for edge with technical analysis as well
# - You say there is *plenty of evidence*, look at my backtest of a moving average crossover strategy
# - I say, this is precisely the insufficient evidence I am talking about and why it is subject for discussion


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
np.random.seed(42)
n_steps = 252  # ~ one trading year
initial_wealth = 10000

def simulate_path(mu, sigma, n_steps, initial_wealth):
    """Simulate a single compounding wealth path."""
    returns = np.random.normal(mu / 500, sigma / 200, n_steps)
    wealth = np.zeros(n_steps)
    wealth[0] = initial_wealth
    for t in range(1, n_steps):
        wealth[t] = wealth[t-1] * (1 + returns[t])
    return wealth

# --- Strategy definitions ---
quant_mu, quant_sigma = -0.04, 1.2   # negative EV, moderate vol
disc_mu, disc_sigma = 0.06, 1.5      # positive EV, moderate vol

quant_path = simulate_path(quant_mu, quant_sigma, n_steps, initial_wealth)
disc_path = simulate_path(disc_mu, disc_sigma, n_steps, initial_wealth)

# --- Expected (theoretical drift) lines ---
quant_expected = initial_wealth * (1 + quant_mu / 100) ** np.arange(n_steps)
disc_expected = initial_wealth * (1 + disc_mu / 100) ** np.arange(n_steps)

# --- Animation frames ---
frames = []
for i in range(1, n_steps):
    frame_data = [
        # Left: Quant's Technical Backtest
        go.Scatter(
            x=np.arange(i),
            y=quant_path[:i],
            mode='lines',
            line=dict(color='orange', width=3),
            name="Quant’s Technical Backtest",
            showlegend=False,
            xaxis='x', yaxis='y'
        ),
        go.Scatter(
            x=np.arange(i),
            y=quant_expected[:i],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name="Expected Decay",
            showlegend=False,
            xaxis='x', yaxis='y'
        ),
        # Right: Discretionary Trader
        go.Scatter(
            x=np.arange(i),
            y=disc_path[:i],
            mode='lines',
            line=dict(color='lime', width=3),
            name="Discretionary Trader",
            showlegend=False,
            xaxis='x2', yaxis='y2'
        ),
        go.Scatter(
            x=np.arange(i),
            y=disc_expected[:i],
            mode='lines',
            line=dict(color='gold', dash='dash', width=2),
            name="Expected Growth",
            showlegend=False,
            xaxis='x2', yaxis='y2'
        )
    ]
    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps])
        )
    ))

# --- Figure setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Quant’s Technical Backtest (Illusory Edge)",
        "Discretionary Trader (Adaptive Edge)"
    ],
    column_widths=[0.5, 0.5]
)

# Initial traces (one per side)
fig.add_trace(
    go.Scatter(x=[0], y=[initial_wealth],
               mode='lines', line=dict(color='orange', width=3),
               name="Quant’s Technical Backtest"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[0], y=[initial_wealth],
               mode='lines', line=dict(color='lime', width=3),
               name="Discretionary Trader"),
    row=1, col=2
)

# Expected lines (static backdrop)
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=quant_expected,
               mode='lines', line=dict(color='red', dash='dash', width=2),
               name="Expected Decay"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=disc_expected,
               mode='lines', line=dict(color='gold', dash='dash', width=2),
               name="Expected Growth"),
    row=1, col=2
)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=500,
    width=1000,
    title_text="Quant’s Technical Backtest vs. Discretionary Trader<br><sup>Blind backtests decay — adaptive judgment compounds</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.12,
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

# --- Axes styling ---
for ax in fig.layout:
    if ax.startswith('xaxis') or ax.startswith('yaxis'):
        fig.layout[ax].showgrid = True
        fig.layout[ax].gridcolor = 'rgba(128,128,128,0.3)'
        fig.layout[ax].zeroline = True
        fig.layout[ax].zerolinecolor = 'rgba(128,128,128,0.5)'

# Keep scales comparable
fig.update_yaxes(range=[9000, 11000])
fig.update_xaxes(title_text="Time (Steps)")

# --- Annotation ---
fig.add_annotation(
    text=(
        "Left: Quant’s model shows structure but loses edge out-of-sample.<br>"
        "Right: Discretionary trader embraces uncertainty and adapts — the edge lives in flexibility."
    ),
    x=0.5, y=-0.25, xref='paper', yref='paper',
    showarrow=False, font=dict(size=14, color='lightgray')
)

fig.show()



# **Quant's Remark on Technical Trading**
# - It isn't about *blindly* following a technical strategy, the backtest will look *awful*
# - Rather, from a discretionary point of view, it's about applying it at the *right* time, its arbitrary right? Again, exactly.
# 
# If it helps make an informed decision about the uncertain outcome then its a necessary part of edge (I wish this wasn't the case, but this is how the space operates. . .)


# ###### ______________________________________________________________________________________________________________________________________


# ##### So, How Do We Act with Positive EV?
# 
# **Quants** have one view, **Discretionary Traders** have another which we will Discuss Herein.
# 
# We have to use our information, knowledge and **take a risk** to profit from our edge.
# 
# There is **NO** risk-free strategy, that would necessarily earn a risk-free rate - even market-neutral quant strategies bear significant risks.
# 
# If we continue to take risk with positive edge or expected value (EV) and reasonable bet sizing we will accumulate wealth, otherwise we will blow out


# ###### ______________________________________________________________________________________________________________________________________


# ##### Trading is About Uncertainty and Assuming Risk
# 
# Being an active risk taker is a badge of honor in the industry, it is how you are rewarded with P/L - you don't get someone else's P/L for the risk **they** assume.
# 
# We know now that trading is an uncertain system, so why do we model everything as random variables and apply statistics?
# 
# Firstly, devoid of insider information, which is illegal (but still traded lol) nobody knows what is going to happen there is no
# - News outlet
# - Politician
# - Broker
# - Prediction market
# - Day Trading Guru
# 
# That knows what the outcome of any one trade will be, the space is entirely *uncertain* typically modelled as random
# 
# Everyone is effectively trying to make informed decisions in the face of uncertainty by assigning likelihoods to different outcomes
# 
# We can use statistics based on our informed decision making to determine if we have been operating with an edge


# ###### ______________________________________________________________________________________________________________________________________


# ##### Expected Value (EV, edge) Dictates Trajectory, Volatility Dictates Expected Dispersion (A Risk Measure)
# 
# $$\mathbb{E}[\text{Trade}] = \mathbb{E}[\text{Trade P/L | Win}] \cdot P(\text{Win}) + \mathbb{E}[\text{Trade P/L | Loss}] \cdot P(\text{Loss})$$
# 
# $$\sigma = \sqrt{\text{Var}[\text{Trade}]} = \sqrt{\mathbb{E}[(\text{Trade} - \mathbb{E}[\text{Trade}])^2]}$$
# 
# We typically proxy for strategy or portfolio volatility using $\sigma$, the standard deviation of returns


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation setup ---
np.random.seed(42)
n_steps = 50
n_paths = 8
initial_wealth = 10000

def simulate_paths(mu, sigma, n_paths, n_steps, initial_wealth):
    """Simulate random walks (portfolio paths) starting from initial wealth."""
    returns = np.random.normal(mu / 400, sigma / 100, (n_paths, n_steps))  # % returns
    wealth = np.zeros((n_paths, n_steps))
    wealth[:, 0] = initial_wealth
    for t in range(1, n_steps):
        wealth[:, t] = wealth[:, t-1] * (1 + returns[:, t])
    return wealth

# Strategy dynamics (mu = expected % return per step, sigma = volatility)
strategies = {
    "Positive EV, Low Var": (1.0, 0.4, 'lime'),
    "Negative EV, High Var": (-0.6, 2.2, 'orange'),
    "Positive EV, High Var": (0.6, 2.2, 'gold'),
    "Negative EV, Low Var": (-1.0, 0.4, 'red')
}

# Generate wealth paths
paths = {
    name: simulate_paths(mu, sigma, n_paths, n_steps, initial_wealth)
    for name, (mu, sigma, _) in strategies.items()
}

# --- Create frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []

    # TL: Positive EV, Low Var
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=np.arange(i),
            y=paths["Positive EV, Low Var"][p, :i],
            mode='lines',
            line=dict(color='lime', width=2),
            opacity=0.8,
            xaxis='x', yaxis='y',
            showlegend=False
        ))

    # TR: Negative EV, High Var
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=np.arange(i),
            y=paths["Negative EV, High Var"][p, :i],
            mode='lines',
            line=dict(color='orange', width=2),
            opacity=0.8,
            xaxis='x2', yaxis='y2',
            showlegend=False
        ))

    # BL: Positive EV, High Var
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=np.arange(i),
            y=paths["Positive EV, High Var"][p, :i],
            mode='lines',
            line=dict(color='gold', width=2),
            opacity=0.8,
            xaxis='x3', yaxis='y3',
            showlegend=False
        ))

    # BR: Negative EV, Low Var
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=np.arange(i),
            y=paths["Negative EV, Low Var"][p, :i],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=0.8,
            xaxis='x4', yaxis='y4',
            showlegend=False
        ))

    frames.append(go.Frame(
        name=f"frame{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            xaxis2=dict(range=[0, n_steps]),
            xaxis3=dict(range=[0, n_steps]),
            xaxis4=dict(range=[0, n_steps])
        )
    ))

# --- Create figure ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Positive EV, Low Variance",
        "Negative EV, High Variance",
        "Positive EV, High Variance",
        "Negative EV, Low Variance"
    ],
    column_widths=[0.5, 0.5],
    row_heights=[0.5, 0.5]
)

# Add placeholder traces for structure
for p in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth],
                             mode='lines',
                             line=dict(color='lime', width=2),
                             opacity=0.8, showlegend=False),
                  row=1, col=1)
for p in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth],
                             mode='lines',
                             line=dict(color='orange', width=2),
                             opacity=0.8, showlegend=False),
                  row=1, col=2)
for p in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth],
                             mode='lines',
                             line=dict(color='gold', width=2),
                             opacity=0.8, showlegend=False),
                  row=2, col=1)
for p in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[initial_wealth],
                             mode='lines',
                             line=dict(color='red', width=2),
                             opacity=0.8, showlegend=False),
                  row=2, col=2)

# --- Apply frames ---
fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=700,
    width=1100,
    title_text="The Evolution of Edge and Volatility<br><sup>All portfolios start at $10,000 — differing EV and variance shape outcomes</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.02, 'y': -0.08,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# --- Axes styling ---
for ax in fig.layout:
    if ax.startswith('xaxis') or ax.startswith('yaxis'):
        fig.layout[ax].showgrid = True
        fig.layout[ax].gridcolor = 'rgba(128,128,128,0.3)'
        fig.layout[ax].zeroline = True
        fig.layout[ax].zerolinecolor = 'rgba(128,128,128,0.5)'

fig.update_yaxes(range=[7000, 13000])
fig.update_xaxes(title_text="Time (Steps)")

fig.show()



# We want to maximize EV sure, but we don't get to choose the path we walk, so we could outperform expected or take a massive drawdown
# 
# Which is why we have to take into account our risk and not only maximize EV but also our strategy or portfolio volatility (Sharpe)
# 
# The Sharpe Ratio Measures Risk Adjusted Returns Above the Risk-Free Rate
# $$\text{Sharpe} = \frac{\mathbb{E}[\text{Trade}] - r_f}{\sigma} = \frac{\text{Expected Return} - \text{Risk-Free Rate}}{\text{Standard Deviation}}$$
#   


# ###### ______________________________________________________________________________________________________________________________________
# ##### The Goal of Every Trader: Maximize EV, Minimize Risk, Ensure Stability
# 
# There are many different types of traders, we can continue to diversify by instrument so on and so forth
# 
# Regardless of the type of trader you are or the instruments you trade
# 
#   $$  \begin{array}{ll}
#      \textbf{Types of Traders:} & \textbf{Instruments Traded:} \\
#       \text{- Market Makers} & \text{- Equities} \\
#       \text{- Quant Traders} & \text{- Options} \\
#       \text{- Discretionary Traders} & \text{- Forwards/Futures} \\
#       \text{...} & \text{...}
#       \end{array}
#   $$
# 
# Your objective is to maximize EV and minimize risk.
# 
# Mathematically, we can express this objective as:
# 
# $$\pi^* = \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade}] \approx \underset{\pi}{\text{argmax}} \; \frac{\mathbb{E}_{\pi}[\text{Trade}] - r_f}{\sqrt{\text{Var}_{\pi}[\text{Trade}]}}$$
# 
# where $\pi$ represents your trading strategy/policy and $\pi^*$ is the optimal strategy that maximizes expected value and minimizes risk


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Approximating $\pi^*$ using Machine Learning to Trade NVDA


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Read NVDA returns data
nvda_df = pd.read_csv('NVDA_returns.csv')
nvda_df.drop(columns=['Volume'], inplace=True)

# Remove dividend rows and convert numeric columns
nvda_df = nvda_df[~nvda_df['High'].astype(str).str.contains('Dividend', na=False)]
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close'] 
nvda_df[numeric_cols] = nvda_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Convert Date column to datetime and sort
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])
nvda_df = nvda_df.sort_values('Date')

# Calculate split points
n_samples = len(nvda_df)
train_size = int(0.6 * n_samples)
val_size = int(0.2 * n_samples)

# Split into train/val/test
train_df = nvda_df[:train_size]
val_df = nvda_df[train_size:train_size+val_size]
test_df = nvda_df[train_size+val_size:]

# Create figure
fig = go.Figure()

# Add price data with different colors for each section
fig.add_trace(
    go.Scatter(x=train_df['Date'], y=train_df['Close'],
               mode='lines', name='Training Period',
               line=dict(color='rgba(0,255,0,0.9)', width=2),
               connectgaps=True)
)

fig.add_trace(
    go.Scatter(x=val_df['Date'], y=val_df['Close'],
               mode='lines', name='Validation Period',
               line=dict(color='rgba(147,0,255,0.9)', width=2),
               connectgaps=True)
)

fig.add_trace(
    go.Scatter(x=test_df['Date'], y=test_df['Close'],
               mode='lines', name='Test Period',
               line=dict(color='rgba(0,255,255,0.9)', width=2),
               connectgaps=True)
)

# Add vertical lines at split points
fig.add_vline(x=train_df['Date'].iloc[-1], line_width=1, line_dash="dash", line_color="white", opacity=0.5)
fig.add_vline(x=val_df['Date'].iloc[-1], line_width=1, line_dash="dash", line_color="white", opacity=0.5)

# Update layout
fig.update_layout(
    title='NVDA Price Data Split into Train/Validation/Test Sets',
    xaxis_title='Date',
    yaxis_title='Price',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
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
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Evidence $\pi^*$ is Time Variant
# 
# Here we use machine learning to naively approximate the policy for the data we've observed
# $$\pi^* = \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade | Training Data}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade | Training Data}]$$


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Read NVDA returns data
nvda_df = pd.read_csv('NVDA_returns.csv')
nvda_df.drop(columns=['Volume'], inplace=True)

# Convert Date column to datetime
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])

# Sort by date
nvda_df = nvda_df.sort_values('Date')

# Calculate split points
n_samples = len(nvda_df)
train_size = int(0.6 * n_samples)
val_size = int(0.2 * n_samples)

# Split into train/val/test
train_df = nvda_df[:train_size]
val_df = nvda_df[train_size:train_size+val_size]
test_df = nvda_df[train_size+val_size:]


# Create features (simple example using price differences)
def create_features(df, lookback=5):
    features = pd.DataFrame(index=df.index)
    
    # Price differences over lookback periods
    for i in range(1, lookback+1):
        features[f'price_diff_{i}'] = df['Close'].diff(i)
        features[f'high_diff_{i}'] = df['High'].diff(i)
        features[f'low_diff_{i}'] = df['Low'].diff(i)
    
    # Target: Whether price goes up next day
    features['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return features.dropna()

# Calculate Sharpe ratio
def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

# Prepare data
train_features = create_features(train_df)
val_features = create_features(val_df)
test_features = create_features(test_df)

X_train = train_features.drop('target', axis=1)
y_train = train_features['target']
X_val = val_features.drop('target', axis=1)
y_val = val_features['target']
X_test = test_features.drop('target', axis=1)
y_test = test_features['target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train an overfit random forest (max depth=None, min_samples_leaf=1)
clf = RandomForestClassifier(n_estimators=7, max_depth=2, 
                           min_samples_leaf=1, random_state=42)
clf.fit(X_train_scaled, y_train)

# Get predictions and probabilities
train_probs = clf.predict_proba(X_train_scaled)[:, 1]
val_probs = clf.predict_proba(X_val_scaled)[:, 1]
test_probs = clf.predict_proba(X_test_scaled)[:, 1]

# Calculate returns (simplified: +1% if correct, -1% if wrong)
def get_returns(probs, actual):
    predictions = (probs > 0.5).astype(int)
    returns = np.where(predictions == actual, 0.01, -0.01)
    return returns

train_returns = get_returns(train_probs, y_train)
val_returns = get_returns(val_probs, y_val)
test_returns = get_returns(test_probs, y_test)

# Calculate cumulative returns in percentage
train_cum_returns = np.cumsum(train_returns) * 100
val_cum_returns = np.cumsum(val_returns) * 100
test_cum_returns = np.cumsum(test_returns) * 100

# Calculate metrics
train_sharpe = sharpe_ratio(train_returns)
val_sharpe = sharpe_ratio(val_returns)
test_sharpe = sharpe_ratio(test_returns)

# Calculate average returns per trade
train_ev = np.mean(train_returns)
val_ev = np.mean(val_returns)
test_ev = np.mean(test_returns)

# Create plot
fig = make_subplots(rows=3, cols=1, 
                    subplot_titles=(f'Training (Sharpe: {train_sharpe:.2f}, EV/trade: {train_ev:.4f})',
                                  f'Validation (Sharpe: {val_sharpe:.2f}, EV/trade: {val_ev:.4f})',
                                  f'Test (Sharpe: {test_sharpe:.2f}, EV/trade: {test_ev:.4f})'),
                    vertical_spacing=0.15)

# Training equity curve
fig.add_trace(
    go.Scatter(x=train_df['Date'].iloc[5:], y=train_cum_returns, mode='lines', name='Training Returns',
               line=dict(color='rgba(0,255,127,0.9)', width=2)),
    row=1, col=1
)

# Validation equity curve  
fig.add_trace(
    go.Scatter(x=val_df['Date'].iloc[5:], y=val_cum_returns, mode='lines', name='Validation Returns',
               line=dict(color='rgba(147,112,219,0.9)', width=2)),
    row=2, col=1
)

# Test equity curve
fig.add_trace(
    go.Scatter(x=test_df['Date'].iloc[5:], y=test_cum_returns, mode='lines', name='Test Returns',
               line=dict(color='rgba(0,255,255,0.9)', width=2)),
    row=3, col=1
)

fig.update_layout(
    title_text="Approximate Training Set Optimal Policy for NVDA",
    height=600, width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

for i in [1,2,3]:
    fig.update_xaxes(
        title='Date' if i==3 else '',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        dtick='M2',  # Show every other month
        row=i, col=1
    )
    fig.update_yaxes(
        title='CumRets (%)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )

fig.show()


# ##### If *Actually* Learn a *Good* Policy $\pi^*$ we Should see Relative Stability Reflecting Time Variant Dynamics
# 
# Moreover, there is no economic (logical, intuitive) reason for stability in performance if we overfit a model to training data
# 
# Stability is *necessary* to maximize EV and minimize risk
# 
# Otherwise degradation $\implies$ decreasing EV / increasing risk $\implies \text{our policy } \pi \neq \pi^*$
# 
# - In other words, without stability we aren't maximizing EV or minmizing risk
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Overfitting Noise vs. Learning an Optimal Policy Function
# 
# In practice, quant and discretionary traders each have a reasonable justification for stability which we will discuss


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Parameters ---
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_days = len(dates)
split_idx = int(n_days * 0.5)

# --- Simulate shared backtest (same curve before split) ---
shared_backtest_returns = np.random.normal(0.001, 0.004, size=split_idx)

# --- Divergent live performances ---
stable_live_returns = np.random.normal(0.001, 0.004, size=n_days - split_idx)
degraded_live_returns = np.random.normal(-0.0005, 0.006, size=n_days - split_idx)

# --- Build cumulative series ---
backtest_curve = 100 * (1 + shared_backtest_returns).cumprod()
stable_curve = np.concatenate([backtest_curve, backtest_curve[-1] * (1 + stable_live_returns).cumprod()])
degraded_curve = np.concatenate([backtest_curve, backtest_curve[-1] * (1 + degraded_live_returns).cumprod()])

df = pd.DataFrame({
    'Date': dates,
    'Backtest': np.concatenate([backtest_curve, [np.nan] * (n_days - split_idx)]),
    'Stable': stable_curve,
    'Degraded': degraded_curve
})

deployment_date = df['Date'].iloc[split_idx]

# --- Build frames ---
frames = []
for i in range(split_idx, n_days):
    frame_data = [
        # Always show full backtest line
        go.Scatter(
            x=df['Date'][:split_idx],
            y=df['Backtest'][:split_idx],
            mode='lines',
            line=dict(color='white', width=3),
            name='Shared Backtest',
            showlegend=False
        ),
        # Stable live portion revealed up to i
        go.Scatter(
            x=df['Date'][split_idx:i+1],
            y=df['Stable'][split_idx:i+1],
            mode='lines',
            line=dict(color='cyan', width=3),
            name='Stable Strategy (True Edge)',
            showlegend=False
        ),
        # Degraded live portion revealed up to i
        go.Scatter(
            x=df['Date'][split_idx:i+1],
            y=df['Degraded'][split_idx:i+1],
            mode='lines',
            line=dict(color='red', width=3),
            name='Degraded Strategy (Overfit)',
            showlegend=False
        )
    ]
    frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

# --- Base figure with static traces ---
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(
    x=df['Date'][:split_idx],
    y=df['Backtest'][:split_idx],
    mode='lines',
    name='Shared Backtest',
    line=dict(color='white', width=3)
))
fig.add_trace(go.Scatter(
    x=[], y=[],
    mode='lines',
    name='Stable Strategy (True Edge)',
    line=dict(color='cyan', width=3)
))
fig.add_trace(go.Scatter(
    x=[], y=[],
    mode='lines',
    name='Degraded Strategy (Overfit)',
    line=dict(color='red', width=3)
))

# --- Create Figure with frames ---
fig = go.Figure(data=fig.data, layout=fig.layout, frames=frames)

# --- Vertical deployment marker ---
fig.add_vline(
    x=deployment_date,
    line_width=2,
    line_dash='dash',
    line_color='white'
)
fig.add_annotation(
    x=deployment_date,
    y=max(df['Stable'].max(), df['Degraded'].max()),
    text="Live Deployment →",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(size=14, color="white")
)

# --- Layout tweaks ---
fig.update_layout(
    title=dict(
        text="Optimal Policies are Dynamic<br><sup>Same backtest, very different futures</sup>",
        x=0.5,
        font=dict(size=18)
    ),
    height=600,
    width=1050,
    margin=dict(l=80, r=40, t=120, b=60),
    legend=dict(
        orientation="h",
        y=-0.15,
        x=0.5,
        xanchor="center",
        font=dict(size=12)
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.02, 'y': -0.1,
        'showactive': False,
        'buttons': [dict(
            label='Play',
            method='animate',
            args=[None, {
                'frame': {'duration': 15, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        )]
    }]
)

# --- Fixed axis range ---
fig.update_xaxes(
    title='Date',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.25)',
    tickformat='%b %Y',
    range=[df['Date'].iloc[0], df['Date'].iloc[-1]]  # fix xlim
)
fig.update_yaxes(
    title='Cumulative Return (Indexed to 100)',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.25)',
    zeroline=True,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# The unconditional policy is not equivalent to the conditional policy
# $$\pi^* = \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade}] \neq \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade | Training Data}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade | Training Data}]$$
# 
# In other words, we are showing here that the optimal policy changes over time
# 
# More trivially, past performance isn't indicative of future performance, our model overfits noise, . . .
# 
# Watch what happens if we use machine learning again to naively approximate 
# 
# $$\pi^* \neq \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade | Validation Data}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade | Validation Data}]$$


# ###### ______________________________________________________________________________________________________________________________________


# The second approximation performs worse OOS in the testing period - we wouldn't have this information until *after* the returns were realized.  
# 
# In other words, we can't just *fix* our policy function to maximize expected value, we quite literally need to *strategize* about our *policy* function.


# ###### ______________________________________________________________________________________________________________________________________


# ##### The Only Reasonable Fixed Policy $\pi^*$ is Buy-And-Hold
# 
# Trading is work, it is quite literally a job - if you want to be *lazy*, 
# 
# *(and there is nothing wrong with that, we just need to know why we can't be lazy trading)*
# 
# and operate with a fixed policy the just buy-and-hold, it is the only reasonable fixed policy
# 
# Anecdotally, B&H outperforms many backtests, including the one we looked at above that attempts to extrapolate the optimal policy function


# Read NVDA returns data
nvda_df = pd.read_csv('NVDA_returns.csv')
nvda_df.drop(columns=['Volume'], inplace=True)

# Remove dividend rows and convert numeric columns
nvda_df = nvda_df[~nvda_df['High'].astype(str).str.contains('Dividend', na=False)]
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
nvda_df[numeric_cols] = nvda_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Convert Date column to datetime and sort
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])
nvda_df = nvda_df.sort_values('Date')

# Calculate returns and equity curve
nvda_df['Returns'] = nvda_df['Close'].pct_change()
equity_curve = 100 * (1 + nvda_df['Returns']).cumprod()

# Calculate total return, handling NaN values
total_return = ((equity_curve.iloc[-1] / equity_curve.dropna().iloc[0]) - 1) * 100 if not pd.isna(equity_curve.iloc[-1]) else 0

# Create figure
fig3 = go.Figure()

# Plot equity curve
fig3.add_trace(
    go.Scatter(x=nvda_df['Date'], y=equity_curve,
               mode='lines', 
               name=f'Buy & Hold Returns (+{total_return:.1f}%)',
               line=dict(color='rgba(255,165,0,0.9)', width=2),
               connectgaps=True)
)

# Update layout
fig3.update_layout(
    title='NVDA Buy and Hold Returns (Full Period)',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Update axes
fig3.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig3.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig3.show()


# Yes, its anecdotal, but the same argument holds for holding SPX/NQ or any index or average. . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Why Can't Machines Learn to Trade?  
# 
# We Tried to Use ML to Approximate $\pi^* \neq \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Trade}] - \lambda \cdot \text{Var}_{\pi}[\text{Trade}]$
# 
# Why didn't it work?  Well, we can't just throw ML/AI/RL at all of our problems like VC wants us to think. . .
# 
# We need a theoretical basis for our strategy, think of it like counting cards in Blackjack - card counting is rooted in statistics which pushes the EV for the player into positive territory, we didn't need ML/AI/RL to "discover" it we theorized it as a solution to an otherwise unwinnable game.


# Create data for curves
x = np.linspace(-2, 2, 100)

# Observed curve (concave)
y1 = -4 * x**2 + 5  # Increased curvature from -2 to -4
max_idx1 = np.argmax(y1)

# Unobserved curve (shifted)
y2 = -4 * (x-1)**2 + 3  # Increased curvature from -2 to -4
max_idx2 = np.argmax(y2)

# Global policy curve
y3 = -3.6 * (x-0.5)**2 + 5.5  # Increased curvature from -1.8 to -3.6
max_idx3 = np.argmax(y3)

# Create figure
fig = go.Figure()

# Plot observed curve
fig.add_trace(
    go.Scatter(x=x, y=y1,
               mode='lines',
               name='Observed EV',
               line=dict(color='rgba(255,165,0,0.9)', width=2))
)

# Add star at maximum of observed curve
fig.add_trace(
    go.Scatter(x=[x[max_idx1]], y=[y1[max_idx1]],
               mode='markers',
               name='Local Optimum 1',
               marker=dict(symbol='star',
                          size=15,
                          color='rgba(255,165,0,0.9)'))
)

# Plot unobserved curve  
fig.add_trace(
    go.Scatter(x=x, y=y2,
               mode='lines',
               name='Unobserved EV',
               line=dict(color='rgba(100,149,237,0.9)', width=2))
)

# Add star at maximum of unobserved curve
fig.add_trace(
    go.Scatter(x=[x[max_idx2]], y=[y2[max_idx2]],
               mode='markers', 
               name='Local Optimum 2',
               marker=dict(symbol='star',
                          size=15,
                          color='rgba(100,149,237,0.9)'))
)

# Plot global policy curve
fig.add_trace(
    go.Scatter(x=x, y=y3,
               mode='lines',
               name='Global Policy EV',
               line=dict(color='rgba(50,205,50,0.9)', 
                        width=2,
                        dash='dash'))
)

# Add star at maximum of global policy curve
fig.add_trace(
    go.Scatter(x=[x[max_idx3]], y=[y3[max_idx3]],
               mode='markers',
               name='Global Optimum',
               marker=dict(symbol='star',
                          size=15,
                          color='rgba(50,205,50,0.9)'))
)

# Update layout
fig.update_layout(
    title='Expected Value Under Different Policies',
    xaxis_title='Policy Space',
    yaxis_title='Expected Value',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
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
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### There Doesn't Exist Just One Optimal $\pi^*$, There Are Many Ways to Produce an Optimal Policy 
# 
# If we apply machine learning to Blackjack naively, it won't learn a positive EV strategy - but one exists: card counting (positive EV)
# 
# There is a logical and statistical basis for card counting to work which leads to positive EV - we need to do the same when trading
# 
# Both quants and discretionary traders achieve optimal policies in this capacity, their logic and statistical basis differ but are both justified


# ---


# #### 2.) 🏛️ Discretionary Trading
# 
# The best way to understand discretionary trading is through the lens of poker, which also involved risk taking and uncertainty 
# 
# There is no *one-size-fits-all* set of rules one can follow to *win every hand* but there is a general policy we can *continue* to learn to maximize our EV
# 
# Just like a trader, a player must optimize their expected value according to their policy function
# 
# $$\pi^* = \underset{\pi}{\text{argmax}} \; \mathbb{E}_{\pi}[\text{Hand}]$$
# 
# This optimal policy function is necessarily dynamic and updates with new data just like a trader's


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
    
    frames.append(go.Frame(
        data=[
            go.Bar(
                x=scenarios,
                y=pmf,
                marker_color=colors,
                marker_line_width=line_widths,
                marker_line_color='yellow'
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

# PMF subplot
fig.add_trace(
    go.Bar(
        x=scenarios,
        y=pmf,
        marker_color=['green', 'red'],
        marker_line_width=[0, 0],
        marker_line_color='yellow'
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

fig.show()



# ##### Theoretically, We Should **NOT** Be Accumulating Wealth, the Raw Probability is Against Us
# 
# But, the *actual* probability is more complicated than that, like event betting (election example above) we can skew the actual outcome with our actions
# 
# For example, when we *should have* lost we actual force our opponent to fold and take that win - in actions like this we have better edge and accumulate wealth


# ###### ______________________________________________________________________________________________________________________________________


# ##### Experience Based Dynamic Trading


# ##### Example: Volatility Surface Calibration and Reversion
# 
# Model informed decision making enhances expected value, this doesn't have to be high-touch (low/med/high touch, how many levers can I pull as I operate the system?)
# 
# I'll run algorithmic strategies involving a systematic ruleset but they don't run 24/7 - they run at my discretion
# 
# This is the same idea as placing a trade based on experience and a specific strategy - discretionary trading 


import numpy as np
import plotly.graph_objects as go

# --- Volatility Surface Data ---
S_unique = np.array([80, 90, 100, 110, 120])
T_unique = np.array([0.0833, 0.25, 0.50, 1.00, 2.00])
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

Z_heston = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])

Z_market_base = Z_heston.copy()

# --- Wealth Path Setup ---
np.random.seed(42)
n_steps = 80
returns = np.random.normal(0.0035, 0.005, n_steps)
wealth = 10_000 * np.cumprod(1 + returns)

# --- Mean-Reverting Surface Noise Parameters ---
alpha = 0.85     # mean reversion strength (closer to 1 = smoother)
sigma = 0.005    # noise volatility
noise = np.zeros_like(Z_market_base)

frames = []
for i in range(1, n_steps + 1):
    # AR(1)-style mean-reverting perturbation
    noise = alpha * noise + np.random.normal(0, sigma, Z_market_base.shape)
    Z_market = Z_market_base + noise

    frames.append(go.Frame(
        data=[
            # Market (noisy) surface — left
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_market,
                colorscale='Viridis', opacity=0.85,
                showscale=False, name='Market', scene='scene'
            ),
            # Heston (model) surface — left
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_heston,
                colorscale='Blues', opacity=0.6,
                showscale=False, name='Heston', scene='scene'
            ),
            # Wealth path — right
            go.Scatter(
                x=np.arange(i),
                y=wealth[:i],
                mode='lines',
                line=dict(color='gold', width=3),
                name='Wealth Path',
                xaxis='x2', yaxis='y2'
            )
        ],
        name=f'frame{i}'
    ))

# --- Base Figure ---
fig = go.Figure()

# Initial Market + Heston surfaces
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_market_base,
    colorscale='Viridis', opacity=0.85,
    showscale=False, name='Market', scene='scene'
))
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_heston,
    colorscale='Blues', opacity=0.6,
    showscale=False, name='Heston', scene='scene'
))

# Initial Wealth Path
fig.add_trace(go.Scatter(
    x=[0], y=[wealth[0]],
    mode='lines',
    line=dict(color='gold', width=3),
    name='Wealth Path',
    xaxis='x2', yaxis='y2'
))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    title=dict(
        text="Volatility Surface vs Trader Wealth Path<br><sup>Market noise mean-reverting to model</sup>",
        x=0.5, font=dict(size=18)
    ),
    height=600,
    width=1200,
    paper_bgcolor='rgba(0,0,0,0)',  # transparent figure background
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.45, 'y': -0.08,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 60, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
    # --- Domains to separate panels ---
    scene=dict(
        domain=dict(x=[0.0, 0.5], y=[0, 1]),  # left half
        xaxis_title='Stock Price',
        yaxis_title='Time to Expiry',
        zaxis_title='Implied Volatility',
        xaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        yaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        zaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        bgcolor='rgba(0,0,0,0)'  # transparent overall scene
    ),
    xaxis2=dict(
        domain=[0.55, 1.0],  # right half
        title='Time Step',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[0, n_steps]
    ),
    yaxis2=dict(
        title='Wealth ($)',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[min(wealth)*0.98, max(wealth)*1.02],
        anchor='x2'
    ),
    annotations=[
        dict(
            text="Market vs Model",
            x=0.25, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        ),
        dict(
            text="Trader Wealth Path",
            x=0.78, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        )
    ]
)

fig.show()



# ##### My Policy Function $\pi^*$ Dictates *When* I Trade - At My Discretion, NOT a Fixed Ruleset 
# 
# For example, my vol trading system does very well when vol is high, **but I don't get to choose** what the market considers **high**
# 
# ##### THIS IS EVERYTHING - **HIGH** Volatility is REALATIVE NOT A FIXED QUANTITY OR PERCENTILE
# 
# If I choose to run the system during what **I BELIEVE** the market is interpreting as a high vol episode I will do better than if I leave it on 24/7
# - There is no FIXED metric for this, it is based on opinion - it is running at my discretion


import numpy as np
import plotly.graph_objects as go

# --- Volatility Surface Data ---
S_unique = np.array([80, 90, 100, 110, 120])
T_unique = np.array([0.0833, 0.25, 0.50, 1.00, 2.00])
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

Z_heston = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])
Z_market_base = Z_heston.copy()

# --- Wealth Path Setup ---
np.random.seed(42)
n_steps = 80

# Discretionary Trader: steady positive EV
returns_discretionary = np.random.normal(0.0025, 0.006, n_steps)
wealth_discretionary = 10_000 * np.cumprod(1 + returns_discretionary)

# 24/7 Trader: positive early, then negative EV
returns_auto = np.concatenate([
    np.random.normal(0.0025, 0.006, n_steps // 2),
    np.random.normal(-0.002, 0.006, n_steps // 2)
])
wealth_auto = 10_000 * np.cumprod(1 + returns_auto)

# --- Mean-Reverting Surface Noise Parameters ---
alpha = 0.85
sigma = 0.005
noise = np.zeros_like(Z_market_base)

frames = []
for i in range(1, n_steps + 1):
    # AR(1) mean-reverting noise on surface
    noise = alpha * noise + np.random.normal(0, sigma, Z_market_base.shape)
    Z_market = Z_market_base + noise

    # Determine color transition for 24/7 trader (cyan → red)
    color_auto = 'cyan' if i < n_steps // 2 else 'red'

    frames.append(go.Frame(
        data=[
            # Market (noisy) surface — left
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_market,
                colorscale='Viridis', opacity=0.85,
                showscale=False, name='Market', scene='scene'
            ),
            # Heston (model) surface — left
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_heston,
                colorscale='Blues', opacity=0.6,
                showscale=False, name='Heston', scene='scene'
            ),
            # Discretionary Trader (Gold) — right
            go.Scatter(
                x=np.arange(i),
                y=wealth_discretionary[:i],
                mode='lines',
                line=dict(color='gold', width=3),
                name='Discretionary Trader',
                xaxis='x2', yaxis='y2'
            ),
            # 24/7 Trader (Cyan → Red) — right
            go.Scatter(
                x=np.arange(i),
                y=wealth_auto[:i],
                mode='lines',
                line=dict(color=color_auto, width=3),
                name='24/7 Trader',
                xaxis='x2', yaxis='y2'
            )
        ],
        name=f'frame{i}'
    ))

# --- Base Figure ---
fig = go.Figure()

# Initial Market + Heston surfaces
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_market_base,
    colorscale='Viridis', opacity=0.85,
    showscale=False, name='Market', scene='scene'
))
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_heston,
    colorscale='Blues', opacity=0.6,
    showscale=False, name='Heston', scene='scene'
))

# Initial Wealth Paths
fig.add_trace(go.Scatter(
    x=[0], y=[wealth_discretionary[0]],
    mode='lines', line=dict(color='gold', width=3),
    name='Discretionary Trader', xaxis='x2', yaxis='y2'
))
fig.add_trace(go.Scatter(
    x=[0], y=[wealth_auto[0]],
    mode='lines', line=dict(color='cyan', width=3),
    name='24/7 Trader', xaxis='x2', yaxis='y2'
))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    title=dict(
        text="Volatility Surface vs Trader Wealth Paths<br><sup>Discretionary Consistency vs 24/7 Fatigue</sup>",
        x=0.5, font=dict(size=18)
    ),
    height=600,
    width=1200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.45, 'y': -0.08,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 60, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
    # --- Split Panels ---
    scene=dict(
        domain=dict(x=[0.0, 0.5], y=[0, 1]),  # left half
        xaxis_title='Stock Price',
        yaxis_title='Time to Expiry',
        zaxis_title='Implied Volatility',
        xaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        yaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        zaxis=dict(showgrid=True, gridcolor='darkgray', color='darkgray',
                   backgroundcolor='rgb(20, 20, 25)'),
        bgcolor='rgba(0,0,0,0)'  # transparent overall scene
    ),
    xaxis2=dict(
        domain=[0.55, 1.0],  # right half
        title='Time Step',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[0, n_steps]
    ),
    yaxis2=dict(
        title='Wealth ($)',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[min(wealth_auto.min(), wealth_discretionary.min())*0.98,
               max(wealth_auto.max(), wealth_discretionary.max())*1.02],
        anchor='x2'
    ),
    annotations=[
        dict(
            text="Market vs Model",
            x=0.25, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        ),
        dict(
            text="Trader Wealth Paths",
            x=0.78, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        )
    ]
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# ##### Performance Stability and Risks
# 
# This is what it means to have a dynamic policy, I update my beliefs about the market along with the execution of the strategy
# 
# This is **WHY** backtesting a fixed strategy (time invariant policy) will NOT yield positive results
# 
# Quants want to disprove discretionary traders and technical analysis but the world does not operate in a strict frequentist way - its uncertain
# 
# ##### If My Beliefs Are Correct On Average and I Size my Bets Appropriately I can Operate with Positive EV, Minimize Risk, and Ensure Stability in Performance


import numpy as np
import plotly.graph_objects as go

# --- Volatility Surface Data ---
S_unique = np.array([80, 90, 100, 110, 120])
T_unique = np.array([0.0833, 0.25, 0.50, 1.00, 2.00])
S_mesh, T_mesh = np.meshgrid(S_unique, T_unique)

Z_heston = np.array([
    [0.35, 0.28, 0.22, 0.20, 0.18],
    [0.33, 0.27, 0.21, 0.19, 0.17],
    [0.32, 0.26, 0.205, 0.185, 0.165],
    [0.31, 0.25, 0.20, 0.18, 0.16],
    [0.30, 0.245, 0.195, 0.175, 0.155]
])
Z_market_base = Z_heston.copy()

# --- Parameters ---
np.random.seed(42)
n_steps = 100

# Define volatility regimes (index boundaries)
regime1_end = 35
regime2_end = 65  # low-vol, system off

# --- Wealth Path Setup ---
returns = np.zeros(n_steps)
returns[:regime1_end] = np.random.normal(0.0025, 0.006, regime1_end)        # high-vol profitable
returns[regime1_end:regime2_end] = 0                                        # flat (system off)
returns[regime2_end:] = np.random.normal(0.0025, 0.006, n_steps - regime2_end)  # back on

wealth = 10_000 * np.cumprod(1 + returns)

# --- Mean-Reverting Surface Noise ---
alpha = 0.85
sigma = 0.005
noise = np.zeros_like(Z_market_base)

frames = []
for i in range(1, n_steps + 1):
    noise = alpha * noise + np.random.normal(0, sigma, Z_market_base.shape)
    Z_market = Z_market_base + noise

    # Define vertical lines (appear after regime boundaries)
    vlines = []
    if i >= regime1_end:
        vlines.append(
            dict(type="line", x0=regime1_end, x1=regime1_end,
                 y0=wealth.min()*0.98, y1=wealth.max()*1.02,
                 line=dict(color="white", width=2, dash="dash"), xref="x2", yref="y2")
        )
    if i >= regime2_end:
        vlines.append(
            dict(type="line", x0=regime2_end, x1=regime2_end,
                 y0=wealth.min()*0.98, y1=wealth.max()*1.02,
                 line=dict(color="white", width=2, dash="dash"), xref="x2", yref="y2")
        )

    frames.append(go.Frame(
        data=[
            # Market surface (Left)
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_market,
                colorscale='Viridis', opacity=0.85,
                showscale=False, name='Market', scene='scene'
            ),
            # Heston model (Left)
            go.Surface(
                x=S_mesh, y=T_mesh, z=Z_heston,
                colorscale='Blues', opacity=0.6,
                showscale=False, name='Heston', scene='scene'
            ),
            # Wealth path (Right)
            go.Scatter(
                x=np.arange(i),
                y=wealth[:i],
                mode='lines',
                line=dict(color='gold', width=3),
                name='Discretionary Trader',
                xaxis='x2', yaxis='y2'
            )
        ],
        layout=go.Layout(shapes=vlines),
        name=f'frame{i}'
    ))

# --- Base Figure ---
fig = go.Figure()

# Initial surfaces
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_market_base,
    colorscale='Viridis', opacity=0.85,
    showscale=False, name='Market', scene='scene'
))
fig.add_trace(go.Surface(
    x=S_mesh, y=T_mesh, z=Z_heston,
    colorscale='Blues', opacity=0.6,
    showscale=False, name='Heston', scene='scene'
))

# Initial Wealth Path
fig.add_trace(go.Scatter(
    x=[0], y=[wealth[0]],
    mode='lines', line=dict(color='gold', width=3),
    name='Discretionary Trader', xaxis='x2', yaxis='y2'
))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    title=dict(
        text="Volatility Surface Reversion as a Discretionary Trader<br><sup>Trade high vol, rest during calm</sup>",
        x=0.5, font=dict(size=18)
    ),
    height=600,
    width=1200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.45, 'y': -0.08,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 60, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
    # --- Split Panels ---
    scene=dict(
        domain=dict(x=[0.0, 0.5], y=[0, 1]),
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
    ),
    xaxis2=dict(
        domain=[0.55, 1.0],
        title='Time Step',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[0, n_steps]
    ),
    yaxis2=dict(
        title='Wealth ($)',
        showgrid=True,
        gridcolor='rgba(128,128,128,0.3)',
        color='white',
        range=[min(wealth)*0.98, max(wealth)*1.02],
        anchor='x2'
    ),
    annotations=[
        dict(
            text="Market vs Model",
            x=0.25, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        ),
        dict(
            text="Discretionary Wealth Path",
            x=0.78, y=1.05, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        )
    ]
)

fig.show()



# ##### Justification for Performance in a Forward-Looking Capacity Comes from Experience
# 
# The argument is the same for poker pros, you see the same faces in the tournament scene at final tables - the probability is near zero of this occuring by random chance
# 
# In other words, experience structures stability for discretionary trading


#   $\begin{array}{ll}
#   \textbf{Risk Type} & \textbf{Description} \\
#   \hline
#   \text{Behavioral Risk} & \text{Emotional biases affect decision making} \\
#   \text{Information Risk} & \text{Incomplete/incorrect market information} \\
#   \text{Experience Risk} & \text{Lack of trading experience in conditions} \\
#   \text{Concentration Risk} & \text{Over-exposure to single positions} \\
#   \text{Fatigue Risk} & \text{Mental exhaustion leads to poor choices} \\
#   \text{Analysis Risk} & \text{Misinterpretation of market signals} \\
#   \text{Position Risk} & \text{Inability to exit losing trades} \\
#   \text{Liquidity Risk} & \text{Market impact and transaction costs} \\
#   \end{array}$


# ###### ______________________________________________________________________________________________________________________________________


# ##### People Ask me Why I'm Not Concerned About My Edge Sharing These Videos
# 
# - Nobody can produce the systems that I can, if you can build a low-latency C/C++ stochastic model calibrator and the execution and risk infrastructure, go for it
# 
# - Moreover, the size I am operating with is nowhere near the $10s of millions of dollars that would be required to crowd the strategy especially in a discretionary capacity


# ---


# #### 3.) 🧮 Quantitative Trading
# 
# 
# ##### Trading Fixed Statistical Mispricings
# 
# There are a lot of strategies in the quant space, the ones I will discuss herein produce *alpha* using alternative data
# 
# In other words, they are *statistical or systematic mispricings* that are present in observed data
# 
# In the context of quant research we demand some economic interpretation of these mispricings
# 
# We don't trade **one** firm but a **collection** of firms to capitalize on the mispricing we observe
# 
# In other words, to maximize our EV we are looking to trade these statistical mispricings. . .
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Example: Quantitative Sentiment Trading Strategy
# 
# There is plenty of evidence in the literature of text being able to produce cross-sectional return prediction (alpha)
# 
# I have worked on several large-scale strategies in a variety of capacities and have seen them deployed first-hand
# 
# I actively trade a modified version myself


# Create sample sentiment and return data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

# Generate sentiment scores (0 to 100)
sentiment = np.random.normal(50, 15, n_days)
sentiment = np.clip(sentiment, 0, 100)

# Generate returns correlated with sentiment
noise = np.random.normal(0, 0.01, n_days)
returns = 0.0003 * (sentiment - 50) + noise

# Create sentiment buckets
bucket_edges = np.percentile(sentiment, np.linspace(0, 100, 11))
bucket_labels = [f"{bucket_edges[i]:.0f}-{bucket_edges[i+1]:.0f}" for i in range(len(bucket_edges)-1)]
# Replace first and last bucket labels with "Short" and "Long"
bucket_labels[0] = "Short"
bucket_labels[-1] = "Long"

# Calculate average returns per bucket
avg_returns = pd.DataFrame({'returns': returns, 'bucket': pd.cut(sentiment, bucket_edges, labels=bucket_labels)})
avg_returns = avg_returns.groupby('bucket')['returns'].mean() * 100  # Convert to percentage

# Create figure
fig = go.Figure()

# Add return bars
colors = ['rgba(255,0,0,0.7)' if ret < 0 else 'rgba(0,255,0,0.7)' for ret in avg_returns]
fig.add_trace(
    go.Bar(x=bucket_labels,
           y=avg_returns,
           name='Avg Daily Return',
           marker_color=colors)
)

# Update layout
fig.update_layout(
    height=400,
    width=1000,
    title='Average Daily Returns by Sentiment Bucket',
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
print("9:20AM")
print("Long: AAPL, MSFT, GOOGL, AMZN...")
print("Short: NFLX, META, TSLA, NVDA...")



# ###### ______________________________________________________________________________________________________________________________________
# The strategy is market-neutral (long/short) meaning its theoretically self-funded or that our short positions "fund" our long-position
# 
# Heterogeneity in signals (sentiment vs. images, for example) offer significant diversity in alphas at firms
# 
# Different teams will run different strategies and bear that risk, the approach is scientifc and fixed relative to a discretionary trader
# 
# That is the traded portfolio will reflect the constituents in the signal's long-short bucket


# Create sample quant sentiment strategy data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

# Generate daily returns with target Sharpe of ~2.75
target_annual_return = 0.20  # 20% annual return
target_annual_vol = 0.073    # To achieve ~2.75 Sharpe
daily_vol = target_annual_vol / np.sqrt(252)
daily_return = target_annual_return / 252

# Generate returns for long and short legs
long_returns = np.random.normal(daily_return*1.2, daily_vol*1.1, n_days)  # Slightly higher return and vol
short_returns = np.random.normal(-daily_return*0.8, daily_vol*0.9, n_days)  # Lower magnitude on short side
combined_returns = (long_returns - short_returns) / 2  # Market neutral combination

# Calculate equity curves starting at $100,000
long_equity = 100000 * (1 + long_returns).cumprod()
short_equity = 100000 * (1 + short_returns).cumprod()
combined_equity = 100000 * (1 + combined_returns).cumprod()

# Calculate metrics for combined strategy
sharpe = np.sqrt(252) * np.mean(combined_returns) / np.std(combined_returns)
ev = np.mean(combined_returns) * 100  # Convert to percentage

# Create figure
fig = go.Figure()

# Add equity curves
fig.add_trace(
    go.Scatter(x=dates, y=long_equity,
               mode='lines', 
               name='Long Portfolio',
               line=dict(color='rgba(0,255,0,0.7)', width=2))
)

fig.add_trace(
    go.Scatter(x=dates, y=short_equity,
               mode='lines', 
               name='Short Portfolio',
               line=dict(color='rgba(255,0,0,0.7)', width=2))
)

fig.add_trace(
    go.Scatter(x=dates, y=combined_equity,
               mode='lines', 
               name=f'Combined L/S Strategy (Sharpe: {sharpe:.2f}, EV: {ev:.3f}%/day)',
               line=dict(color='rgba(0,255,255,0.9)', width=2))
)

# Update layout
fig.update_layout(
    title='Quant Sentiment Strategy Components (2020-2022)',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
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
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# ##### Performance Stability and Risks


# The justification of performance in a forward looking sense is critical to our ability to execute the strategy


# Create sample quant sentiment strategy data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

# Generate daily returns with target Sharpe of ~2.75
target_annual_return = 0.20  # 20% annual return
target_annual_vol = 0.073    # To achieve ~2.75 Sharpe
daily_vol = target_annual_vol / np.sqrt(252)
daily_return = target_annual_return / 252

# Generate returns with slight positive skew
returns = np.random.normal(daily_return, daily_vol, n_days)
returns = returns + 0.1 * returns**2  # Add slight positive skew

# Calculate equity curve starting at $100,000
equity = 100000 * (1 + returns).cumprod()

# Split data into backtest and live periods
backtest_end = '2022-01-01'  # Mid-2022 as split point
backtest_mask = dates <= backtest_end
live_mask = dates > backtest_end

# Calculate metrics for both periods
backtest_returns = returns[backtest_mask]
live_returns = returns[live_mask]

backtest_sharpe = np.sqrt(252) * np.mean(backtest_returns) / np.std(backtest_returns)
live_sharpe = np.sqrt(252) * np.mean(live_returns) / np.std(live_returns)
ev = np.mean(returns) * 100  # Convert to percentage
total_return = (equity[-1] - equity[0]) / equity[0] * 100

# Create figure
fig = go.Figure()

# Backtest period
fig.add_trace(
    go.Scatter(x=dates[backtest_mask], y=equity[backtest_mask],
               mode='lines', 
               name=f'Backtest (Sharpe: {backtest_sharpe:.2f}, EV: {ev:.3f}%/day)',
               line=dict(color='rgba(0,255,255,0.9)', width=2))
)

# Live period
fig.add_trace(
    go.Scatter(x=dates[live_mask], y=equity[live_mask],
               mode='lines', 
               name=f'Live Trading (Sharpe: {live_sharpe:.2f})',
               line=dict(color='rgba(0,255,0,0.9)', width=2))
)

# Add vertical line for backtest/live split
fig.add_vline(x=backtest_end, line_dash="dash", line_color="green")

# Update layout
fig.update_layout(
    title='Quant Sentiment Strategy Equity Curve (2020-2022)',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
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
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



# $\begin{array}{ll}
# \textbf{Argument for Forward Looking Stability} \\
# \hline
# \text{Based on market sentiment dynamics and structural mispricing} \\
# \text{Economic interpretable logic reduces curve-fitting risk} \\
# \text{Parameter robustness indicates genuine signal} \\
# \text{Limited competition in sentiment vs technical strategies} \\
# \end{array}$


# Create sample quant sentiment strategy data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)

# Generate daily returns with target Sharpe of ~2.75 for backtest, but degraded for live
target_annual_return = 0.20  # 20% annual return
target_annual_vol = 0.073    # To achieve ~2.75 Sharpe
daily_vol = target_annual_vol / np.sqrt(252)
daily_return = target_annual_return / 252

# Generate returns with slight positive skew for backtest period
returns = np.random.normal(daily_return, daily_vol, n_days)
returns = returns + 0.1 * returns**2  # Add slight positive skew

# Degrade performance for live period
backtest_end = '2022-01-01'  # Mid-2022 as split point
live_start_idx = (dates > backtest_end).argmax()

# Add more volatility and negative drift for live period
returns[live_start_idx:] = np.random.normal(-daily_return*2, daily_vol*1.5, len(returns[live_start_idx:]))
returns[live_start_idx:] = returns[live_start_idx:] - 0.1 * returns[live_start_idx:]**2  # Add negative skew

# Calculate equity curve starting at $100,000
equity = 100000 * (1 + returns).cumprod()

# Split data into backtest and live periods
backtest_mask = dates <= backtest_end
live_mask = dates > backtest_end

# Calculate metrics for both periods
backtest_returns = returns[backtest_mask]
live_returns = returns[live_mask]

backtest_sharpe = np.sqrt(252) * np.mean(backtest_returns) / np.std(backtest_returns)
live_sharpe = np.sqrt(252) * np.mean(live_returns) / np.std(live_returns)
ev = np.mean(returns) * 100  # Convert to percentage
total_return = (equity[-1] - equity[0]) / equity[0] * 100

# Create figure
fig = go.Figure()

# Backtest period
fig.add_trace(
    go.Scatter(x=dates[backtest_mask], y=equity[backtest_mask],
               mode='lines', 
               name=f'Backtest (Sharpe: {backtest_sharpe:.2f}, EV: {ev:.3f}%/day)',
               line=dict(color='rgba(0,255,255,0.9)', width=2))
)

# Live period
fig.add_trace(
    go.Scatter(x=dates[live_mask], y=equity[live_mask],
               mode='lines', 
               name=f'Live Trading (Sharpe: {live_sharpe:.2f})',
               line=dict(color='rgba(255,0,0,0.9)', width=2))
)

# Add vertical line for backtest/live split
fig.add_vline(x=backtest_end, line_dash="dash", line_color="red")

# Update layout
fig.update_layout(
    title='Quant Sentiment Strategy Equity Curve (2020-2022) - Performance Degradation',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    height=400,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
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
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()



#  $\begin{array}{ll}
#  \textbf{Risk Type} & \textbf{Description} \\
#  \hline
#  \text{Alpha Decay} & \text{Strategies lose effectiveness over time} \\
#  \text{Crowding Risk} & \text{Similar strategies reduce returns} \\
#  \text{Model Risk} & \text{Flaws in strategy design/implementation} \\
#  \text{Data Quality Risk} & \text{Poor data leads to incorrect signals} \\
#  \text{Regime Change Risk} & \text{Market conditions shift} \\
#  \text{Technology Risk} & \text{System and infrastructure failures} \\
#  \text{Regulatory Risk} & \text{Rule changes impact viability} \\
#  \text{Liquidity Risk} & \text{Market impact and transaction costs} \\
#  \end{array}$


# ---


# #### 4.) ⚔️ Quant vs. Discretionary Trading
# 
# ##### Similarities and Differences
# 
# **Risk Takers**
# 
# Both quant and discretionary traders are risk takers, they just operate with different policies and assume different risks in forward-looking performance degredation


# Create sample data for quant and discretionary strategies
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
n_days = len(dates)
backtest_end = '2022-01-01'
live_start_idx = (dates > backtest_end).argmax()

# Helper function to generate returns
def generate_returns(daily_return, daily_vol, degrade=False):
    returns = np.random.normal(daily_return, daily_vol, n_days)
    
    if degrade:
        # Degrade live performance
        returns[live_start_idx:] = np.random.normal(-daily_return*2, daily_vol*1.5, len(returns[live_start_idx:]))
        returns[live_start_idx:] = returns[live_start_idx:] - 0.1 * returns[live_start_idx:]**2
    else:
        # Maintain good performance
        returns[live_start_idx:] = np.random.normal(daily_return*0.8, daily_vol*1.2, len(returns[live_start_idx:]))
    
    return returns

# Generate returns for all scenarios
target_annual_return = 0.20
target_annual_vol = 0.073
daily_vol = target_annual_vol / np.sqrt(252)
daily_return = target_annual_return / 252

quant_good = generate_returns(daily_return, daily_vol, degrade=False)
quant_bad = generate_returns(daily_return, daily_vol, degrade=True)
disc_good = generate_returns(daily_return*1.1, daily_vol*1.2, degrade=False)  # Higher return/vol for discretionary
disc_bad = generate_returns(daily_return*1.1, daily_vol*1.2, degrade=True)

# Calculate equity curves
initial_equity = 100000
scenarios = {
    'Quant (Good Live)': quant_good,
    'Quant (Degraded Live)': quant_bad,
    'Discretionary (Good Live)': disc_good,
    'Discretionary (Degraded Live)': disc_bad
}

# Create subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=list(scenarios.keys()))

for idx, (name, returns) in enumerate(scenarios.items()):
    row = (idx // 2) + 1
    col = (idx % 2) + 1
    
    equity = initial_equity * (1 + returns).cumprod()
    backtest_mask = dates <= backtest_end
    live_mask = dates > backtest_end
    
    # Calculate metrics
    backtest_returns = returns[backtest_mask]
    live_returns = returns[live_mask]
    backtest_sharpe = np.sqrt(252) * np.mean(backtest_returns) / np.std(backtest_returns)
    live_sharpe = np.sqrt(252) * np.mean(live_returns) / np.std(live_returns)
    
    # Plot backtest period
    fig.add_trace(
        go.Scatter(x=dates[backtest_mask], y=equity[backtest_mask],
                   mode='lines',
                   name=f'Backtest (Sharpe: {backtest_sharpe:.2f})',
                   line=dict(color='rgba(0,255,255,0.9)', width=2)),
        row=row, col=col
    )
    
    # Plot live period with color based on whether performance degrades
    live_color = 'rgba(255,0,0,0.9)' if 'Degraded' in name else 'rgba(0,255,0,0.9)'
    fig.add_trace(
        go.Scatter(x=dates[live_mask], y=equity[live_mask],
                   mode='lines',
                   name=f'Live (Sharpe: {live_sharpe:.2f})',
                   line=dict(color=live_color, width=2)),
        row=row, col=col
    )
    
    # Add vertical line for backtest/live split with color based on performance
    line_color = "red" if 'Degraded' in name else "green"
    fig.add_vline(x=backtest_end, line_dash="dash", line_color=line_color, row=row, col=col)

# Update layout
fig.update_layout(
    title='Quant vs Discretionary Strategy Performance Comparison (2020-2022)',
    height=600,
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update all axes
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=i, col=j
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            title_text='Portfolio Value ($)' if j == 1 else None,
            row=i, col=j
        )

fig.show()



#  $\begin{array}{ll}
#  \textbf{Quant Risks} & \textbf{Discretionary Risks} \\
#  \hline
#  \text{Model Risk} & \text{Emotional Bias} \\
#  \text{Data Quality Issues} & \text{Information Overload} \\
#  \text{Overfitting} & \text{Inconsistent Process} \\
#  \text{Infrastructure Failures} & \text{Position Sizing Errors} \\
#  \text{Market Regime Changes} & \text{Analysis Paralysis} \\
#  \text{Crowded Trades} & \text{FOMO Trading} \\
#  \end{array}$
# 


# ###### ______________________________________________________________________________________________________________________________________
# ##### Which is Better? What Should You Do?


# **Same Game, Different Policies**
# 
# **High Barrier to Entry:** Quant Trading
# - double edged sword, harder to act, easier to act with positive EV (otherwise you won't act at all)
# 
# **Low Barrier to Entry:** Discretionary Trading
# - double edged sword, easy to act with negative EV


# ---


# #### 5.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Trading is an uncertain system, not a purely random system like craps, slots, or roulette
# - In either system, the outcome of any one event is unknown
# - Expected value dictates the trajectory of your wealth path but this can change over time
# - A trader's policy function dictates how they act and directly influences EV or edge in uncertain systems
# - Quants and discretionary traders are playing the same game with the same goal, their policy functions and overall strategies differ
# - Discretionary traders operate on experience and intuition and use models to inform their decision making, sometimes algorithmic systems
# - Quant traders operate more scientifically and look for structural or statistical mispricings they can capture
# - Each are *risk takers* - **NEITHER** is without risk - the way they go about taking them and managing them greatly differ due to their differing policy functions
# - At the end of the day quant trading has a higher barrier to entry demanding math/stats/coding/data science knowledge and institutional grade data and systems where discretionary trading has a lower barrier to entry but this makes it easier to operate with negative EV and allows those with no business trading to do so
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
# - Stochastic Proccesses: Brownian Motion, Arithmetic (additive) Geometric (multiplicative) Brownian Motion
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
# - Is the Market Random?
# - Most Popular Quant Models for Informed Trading
# - Buy Side vs. Sell Side Quants
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Build an Earnings Event Options Trading Dashboard
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

