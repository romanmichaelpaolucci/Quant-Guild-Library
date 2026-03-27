# ### 🃏 Why Poker Pros Make the Best Traders (It's NOT Luck)
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
# 
# #### 1.) 🃏 Understanding Poker & Trading
# 
# - Is it Luck or Skill?
# 
# - Poker and Trading
# 
# - Professional Players and Traders
# 
# #### 2.) 📈 Objective: Poker & Trading
# 
# - Maximizing Expected Value
# 
# - Model Informed Decision Making (Outs vs Implied Vol/Prob)
# 
# - Taking EV from Other Players
# 
# #### 3.) ⚠️ Risk Management: Poker & Trading
# 
# - Cutting Losses Early
# 
# - Letting Winners Run
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🃏 Poker vs. Trading
# 
# 
# ##### Is it Luck or Skill?
# 
# *Reasonable Question:*  If its luck then why are there *professional* poker players and discretionary traders? 
# 
# There is a big difference between a game of incomplete information and a game of chance
# 
# These games are zero-sum and involve two parties (one has positive P/L the other negative)
# 
# Games of chance involve a fixed edge - no matter what the player does they will *always* lose money if they continue to play, this is gambling in its purest form
# 
# In this context, edge or expected value says something about the expectation of *one outcome* of a game or system - over a series of plays this determines if we make or lose money
# 
# The only *optimal* action the player can make statistically is to **NOT** subject their wealth to these games, to not play them
# 
# **Examples of Games of Chance with Negative EV:**
# - Powerball
# - Scratchoffs
# - Roulette
# - Slots
# - . . .
# 
# **Remark:** Though the decision you make (for example, to pick Black/Red/Green) won't impact the degradation in your wealth path over time, there are some betting 
# 
# strategies that actually *remove* this house edge making it positive for the player - very quickly you will be asked to leave or you will appraoch the table limit and 
# 
# won't be able to run that strategy
# 
# An old colleague of mine is now a mathematician at a casino, his job is to evaluate exposure and probabilities of losses while statistically ensuring players lose and the casino wins


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
# Betting on red/black gives 18/37 chance of winning
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

# Find first zero for each player and set all future values to zero
for i in range(n_players):
    zero_indices = np.where(player_wealths[i] <= 0)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        player_wealths[i, first_zero:] = 0

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Player Wealth Paths', 'Casino Wealth Path')
)

# Define colors
colors = ['rgba(255, 0, 255, 1)', 'rgba(0, 255, 255, 1)', 
         'rgba(0, 255, 0, 1)', 'rgba(255, 165, 0, 1)', 
         'rgba(255, 0, 0, 1)']

# Create frames for animation
frames = []
for step in range(n_spins + 1):
    frame = {"data": [], "name": str(step)}
    
    # Calculate axis limits for current step
    y_min_players = min(np.min(player_wealths[:, :step+1]), 0)
    y_max_players = max(np.max(player_wealths[:, :step+1]), starting_wealth)
    y_min_casino = min(np.min(casino_wealth[:step+1]), 0)
    y_max_casino = max(np.max(casino_wealth[:step+1]), 0)
    
    # Add player traces
    for i in range(n_players):
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=player_wealths[i, :step + 1],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=f'Player {i+1}'
            )
        )
    
    # Add casino trace
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=casino_wealth[:step + 1],
            mode='lines',
            line=dict(color='rgba(255, 215, 0, 1)', width=2),
            name='Casino'
        )
    )
    
    # Update axis ranges for this frame
    frame["layout"] = {
        "xaxis": {"range": [0, max(step + 1, 5)]},
        "xaxis2": {"range": [0, max(step + 1, 5)]},
        "yaxis": {"range": [y_min_players * 1.1, y_max_players * 1.1]},
        "yaxis2": {"range": [y_min_casino * 1.1, y_max_casino * 1.1]}
    }
    
    frames.append(frame)

# Add initial traces (empty)
for i in range(n_players):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[starting_wealth],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=f'Player {i+1}'
        ),
        row=1, col=1
    )

# Add initial casino trace (empty)
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode='lines',
        line=dict(color='rgba(255, 215, 0, 1)', width=2),
        name='Casino'
    ),
    row=1, col=2
)

# Update layout with animation settings
fig.update_layout(
    height=500,
    width=1200,
    showlegend=True,
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
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }, {
            'label': 'Pause',
            'method': 'animate',
            'args': [[None], {
                'frame': {'duration': 0, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }]
        }]
    }],
    sliders=[{
        'currentvalue': {'prefix': 'Spin: '},
        'pad': {'t': 50},
        'len': 0.9,
        'x': 0.1,
        'steps': [{
            'args': [[str(i)], {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }],
            'label': str(i),
            'method': 'animate'
        } for i in range(n_spins + 1)]
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

# Add frames to figure
fig.frames = frames

fig.show()



# **Remark:** The outcome of any one game is *random* by luck any one game or series of games may end up with the player *winning money*
# 
# The statistics suggest something about the tendancy of wealth over time - **NOT** any one outcome, a critical distinction.  
# 
# We can **NEVER** udner any circumstances predict the outcome of a future event or random variable 
# 
# We can only make informed decisions about the likelihood of possible outcomes


# Set random seed for reproducibility 
np.random.seed(42)  # Chosen to demonstrate a "lucky" sequence

# Simulation parameters for lucky player example
n_spins = 50
starting_wealth = 1000
bet_size = 100
p_win = 18/37  # European roulette probability
payout = 2     # 1:1 payout

# Initialize wealth paths
player_wealth = np.zeros(n_spins + 1)
casino_wealth = np.zeros(n_spins + 1)
player_wealth[0] = starting_wealth

# Simulate spins with a "lucky" sequence
for i in range(n_spins):
    result = np.random.random() < p_win
    win = result * bet_size * (payout - 1)
    loss = (not result) * bet_size
    net_result = win - loss
    
    player_wealth[i+1] = player_wealth[i] + net_result
    casino_wealth[i+1] = casino_wealth[i] - net_result

# Create figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Lucky Player Wealth Path', 'Casino Loss Path')
)

# Add player trace
fig.add_trace(
    go.Scatter(
        x=np.arange(n_spins + 1),
        y=player_wealth,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)', width=2),
        name='Lucky Player'
    ),
    row=1, col=1
)

# Add casino trace
fig.add_trace(
    go.Scatter(
        x=np.arange(n_spins + 1),
        y=casino_wealth,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 1)', width=2),
        name='Casino'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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
        title='Wealth ($)' if i == 1 else 'Casino Loss ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()



# More formally, any game with a *fixed probability mass function* and a *negative expected value* (assuming *ergodicity*) is a *game of chance* and is pure *gambling* 
# 
# <u>*Law of Total Expectation:*</u>
# 
#  $$E[X] = E[E[X|Y]] = \sum_{y} E[X|Y=y] \cdot P(Y=y)$$
# 
#  <u>*Probability Mass Function (PMF):*</u>
# 
#  $$P(X = x) \geq 0, \text{ for all } x \quad \sum_{x} P(X = x) = 1$$
#  Defines probability distribution for discrete random variables


#  ##### Example: Roullete
# 
#  Theory from probability and statistics tell us about the general trend of wealth paths over time with this fixed edge
# 
#  More formally, if the system is ergodic then the ensemble average is equivalent to the time average and wealth paths will follow this trend
# 
#  More informally, the trend will solely determine how much money you either make or lose over time or iterations playing the game


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simulation parameters
n_players = 10
n_spins = 100
starting_wealth = 10000
bet_size = 100
p_win = 18/38  # American roulette probability
payout = 1  # 1:1 payout

# Generate wealth paths for players
np.random.seed(42)
player_wealths = np.zeros((n_players, n_spins + 1))
player_wealths[:, 0] = starting_wealth

casino_wealth = np.zeros(n_spins + 1)
casino_wealth[0] = 0

# Simulate spins
for i in range(n_spins):
    results = np.random.random(n_players) < p_win
    wins = results * bet_size * payout
    losses = ~results * bet_size
    net_results = wins - losses
    
    player_wealths[:, i+1] = player_wealths[:, i] + net_results
    casino_wealth[i+1] = casino_wealth[i] - sum(net_results)

# Find first zero for each player and set all future values to zero
for i in range(n_players):
    zero_indices = np.where(player_wealths[i] <= 0)[0]
    if len(zero_indices) > 0:
        first_zero = zero_indices[0]
        player_wealths[i, first_zero:] = 0

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Player Wealth Paths', 'Casino Wealth Path')
)

# Add player traces with neon cyan and varying opacity
x_vals = np.arange(n_spins + 1)
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)  # Varies from 0.3 to 1.0
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=player_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 255, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# Add player wealth trendline
player_mean = np.mean(player_wealths, axis=0)
z = np.polyfit(x_vals, player_mean, 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p(x_vals),
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Player Trend',
        showlegend=False
    ),
    row=1, col=1
)

# Add casino trace
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=casino_wealth,
        mode='lines',
        line=dict(color='rgba(255, 215, 0, 1)', width=2),
        showlegend=False
    ),
    row=1, col=2
)

# Add casino trendline
z_casino = np.polyfit(x_vals, casino_wealth, 1)
p_casino = np.poly1d(z_casino)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_casino(x_vals),
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1100,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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

fig.show()



# ##### We can find the slope of these trendlines using the probability mass function and associated expected value per game


# *PMF for American Roulette (Black/Red Bet):*
# 
#   - $P(X = +1) = 18/38$  # Win (hitting black/red)
#   - $P(X = -1) = 20/38$  # Loss (hitting green or opposite color)
# 
#   *Expected Value for <u>Players</u> (per $1 bet):*
# 
#   $$EV_{player} = (1 × 18/38) + (-1 × 20/38) = -0.0526 \text{ or } -5.26\%$$
# 
#   *Expected Value for <u>Casino</u> (per $1 bet):*
# 
#   $$EV_{casino} = (-1 × 18/38) + (1 × 20/38) = +0.0526 \text{ or } +5.26\%$$
# 
# 
# ##### Crucially, the casino's trendline is scaled linearly by the number of players - wow, they have some very positive EV!


# We can verify this by printing the slope of the line from our simulation above, we should see numbers that roughly match 
# 
# - $-\$5.26$ loss per play for a player
# 
# - $+\$5.26 \times (10 \text{ players})$ win per play from each of the 10 players for the casino
# 
# This is roughly what we observe in our simulation above!  See the slope of the trendlines we print below.


# Print slopes of trendlines
print(f"Player wealth trendline slope: ${z[0]:.2f} per spin")
print(f"Casino wealth trendline slope: ${z_casino[0]:.2f} per spin")


# ##### <u>Summary Games of Chance</u>
# - Generally, pure gambling - player action does not influence wealth path over time (no notion of skill)
# - Edge (expected value) is fixed and against the player
# - We can forecast the wealth path of players and the counterparty (i.e. the casino) using theory from probability and statistics
# - Betting strategies and other caveats may exist but casinos will certainly remove you should you attempt to exploit them
# 
# ##### So how is this different from a *game of incomplete information*?


# Games of incomplete information leave room for optimal player action
# 
# In other words, the game or space is not defined by a fixed probability mass or density function
# 
# There may be *elements* of randomness, but edge or expected value is **NOT** fixed and players can influence its overall value (i.e. positive or negative) based on their actions
# 
# **IN FACT** it may not even need to be theoretically possible for the player's action alone to dictate positive expected value, we will see shortly how even if they play 
# 
# with negative expected value they can *take* the expected value from other players at the table.  I do this all the time playing against **BAD** poker players


# Simulation parameters
n_players = 10
n_spins = 1000
starting_wealth = 10000
bet_size = 100

# Generate wealth paths for two groups of players
np.random.seed(42)

# Group 1: Game of Chance (negative EV)
chance_wealths = np.zeros((n_players, n_spins + 1))
chance_wealths[:, 0] = starting_wealth

# Group 2: Game with Optimal Action (positive EV) 
skilled_wealths = np.zeros((n_players, n_spins + 1))
skilled_wealths[:, 0] = starting_wealth

# Simulate spins
p_win_chance = 18/38  # Negative EV game
p_win_skilled = 0.55  # Positive EV game with optimal action

for i in range(n_spins):
    # Simulate Game of Chance results
    results_chance = np.random.random(n_players) < p_win_chance
    wins_chance = results_chance * bet_size
    losses_chance = ~results_chance * bet_size
    net_results_chance = wins_chance - losses_chance
    chance_wealths[:, i+1] = chance_wealths[:, i] + net_results_chance
    
    # Simulate Game with Optimal Action results
    results_skilled = np.random.random(n_players) < p_win_skilled
    wins_skilled = results_skilled * bet_size
    losses_skilled = ~results_skilled * bet_size
    net_results_skilled = wins_skilled - losses_skilled
    skilled_wealths[:, i+1] = skilled_wealths[:, i] + net_results_skilled

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Players in a Game of Chance', 'Players in a Game of Incomplete Information with Optimal Action')
)

# Add traces for Game of Chance players
x_vals = np.arange(n_spins + 1)
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=chance_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 255, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# Add Game of Chance trendline
chance_mean = np.mean(chance_wealths, axis=0)
z_chance = np.polyfit(x_vals, chance_mean, 1)
p_chance = np.poly1d(z_chance)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_chance(x_vals),
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for Skilled players
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=skilled_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 0, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

# Add Skilled players trendline
skilled_mean = np.mean(skilled_wealths, axis=0)
z_skilled = np.polyfit(x_vals, skilled_mean, 1)
p_skilled = np.poly1d(z_skilled)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_skilled(x_vals),
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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
        title='Wealth ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()

# Print slopes of trendlines
print(f"Game of Chance wealth trendline slope: ${z_chance[0]:.2f} per spin")
print(f"Game with Optimal Action wealth trendline slope: ${z_skilled[0]:.2f} per spin")



# ##### Example: Game of Chance $\mapsto$ Game of Incomplete Information
# 
#  Let's consider a modified roulette where the outcome depends on the previous spin:
#  
#  If it lands on Red:
#  - P(Red|Red) = 0.80  # 80% chance of Red given previous Red
#  - P(Black|Red) = 0.20 # 20% chance of Black given previous Red
#  
#  If it lands on Black:  
#  - P(Red|Black) = 0.20  # 20% chance of Red given previous Black
#  - P(Black|Black) = 0.80 # 80% chance of Black given previous Black
#  
#  This creates a Markov Chain with transition matrix:
#  
#   $$P = \begin{bmatrix} 
#   0.55 & 0.45 \\
#   0.45 & 0.55
#   \end{bmatrix}$$
#   
#   *Expected Value for Players betting on Red (per $1 bet):*
#   
#   If previous was Red:
#   $$EV_{player|Red} = (1 × 0.55) + (-1 × 0.45) = +0.10$$
#   $$EV_{casino|Red} = (1 × 0.55) + (-1 × 0.45) = +0.10$$
#  


# Simulation parameters for modified roulette
n_players = 10
n_spins = 1000
starting_wealth = 1000
bet_size = 10

# Generate wealth paths for two groups of players
np.random.seed(42)

# Group 1: Novice players who bet randomly
novice_wealths = np.zeros((n_players, n_spins + 1))
novice_wealths[:, 0] = starting_wealth

# Group 2: Pro players who bet optimally based on previous outcome
pro_wealths = np.zeros((n_players, n_spins + 1))
pro_wealths[:, 0] = starting_wealth

# Initial state (50-50 chance for first spin)
prev_red = np.random.random() < 0.5

for i in range(n_spins):
    # Determine outcome based on previous state
    p_red = 0.55 if prev_red else 0.45
    is_red = np.random.random() < p_red
    prev_red = is_red
    
    # Novice players bet randomly
    novice_bets = np.random.random(n_players) < 0.5  # True = bet red, False = bet black
    results_novice = novice_bets == is_red  # Win if correctly predicted
    wins_novice = results_novice * bet_size
    losses_novice = ~results_novice * bet_size
    net_results_novice = wins_novice - losses_novice
    novice_wealths[:, i+1] = novice_wealths[:, i] + net_results_novice
    
    # Pro players bet optimally (bet on more likely outcome based on previous)
    # If previous was red, p(red)=0.55 so bet red, if black p(black)=0.55 so bet black
    pro_bets = np.full(n_players, prev_red)  # Bet red if prev was red, black if prev was black
    
    # Each pro player has 55% chance of winning when betting optimally
    results_pro = np.random.random(n_players) < 0.55  # Simulate 55% win rate
    wins_pro = results_pro * bet_size
    losses_pro = ~results_pro * bet_size
    net_results_pro = wins_pro - losses_pro
    pro_wealths[:, i+1] = pro_wealths[:, i] + net_results_pro

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Novice Players (Random Bets)', 'Pro Players (Bet Based on Previous Outcome)')
)

# Add traces for Novice players
x_vals = np.arange(n_spins + 1)
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=novice_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(255, 0, 0, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# Add Novice players trendline
novice_mean = np.mean(novice_wealths, axis=0)
z_novice = np.polyfit(x_vals, novice_mean, 1)
p_novice = np.poly1d(z_novice)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_novice(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for Pro players
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pro_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 0, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

# Add Pro players trendline
pro_mean = np.mean(pro_wealths, axis=0)
z_pro = np.polyfit(x_vals, pro_mean, 1)
p_pro = np.poly1d(z_pro)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_pro(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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
        title='Wealth ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Poker and Trading
# 
# Poker and trading are perfect examples of games of incomplete information with *elements* of randomness
# 
# *Some videos to look forward to on this topic:*
# - Is the Market Random?
# - Quant vs. Discretionary Trading
# 
# In any case, both are about making optimal decisions in the face of uncertainty
# 
# Unlike games of chance whose outcome is solely dictated by the negative fixed edge, players can make optimal decisions in the face of uncertainty to accumulate wealth
# 
# **Key Point:** There is no *fixed* probability mass or density function (even theoretically) governing outcomes in this system - this is a huge problem from a modeling perspective, we can't assume probabilities converge, we can't assume then that expectations converge, so on and so forth.  Moreover, in this context, as in our modified roullete example above, player's actions (optimal or suboptimal) can influence the outcome of their wealth over time
# 
# In other words, the environment is not fixed against the player - they can *learn* to act *optimally* in uncertainty and influence their edge in a positive way to accumulate wealth
# 
#  | 🕒 Dynamic Aspect | ♦️ Poker | 📈 Trading |
#  |---------------|--------|----------|
#  | Player Skill | Players can improve decision-making through study and experience | Traders can develop better analysis and execution skills over time |
#  | Market/Game Evolution | Meta-game evolves as players adapt strategies | Markets evolve with new instruments, technologies, and strategies |
#  | Competition | Player pool changes, requiring constant adaptation | New market participants and changing competitive landscape |
#  | Information Flow | Real-time tells, betting patterns, position dynamics | News, price action, volume, market sentiment |
#  | Risk Management | Stack sizes and pot odds change throughout play | Position sizing and risk parameters adjust with market conditions |
#  | Psychology | Mental game adapts to opponents and situations | Trading psychology evolves with experience and market cycles |
#  | Edge Creation | Players can create advantages through table selection and exploitation | Traders can find edges through research and strategy development |
# 
# **Remark:** This is coming from a quant - *NOT EVERYTHING CAN REDUCED TO A SYSTEMATIC OR ALGORITHMIC STRATEGY*, there is justification for med-high touch systems with seasoned traders acting optimally in the face of uncertainty just as a professional poker player - more on this in the video on *Quant vs. Discretionary Trading . . .*
# 
# 
# In other words, would you bet on a novice poker player winning against a 10y pro?  Would you bet on a new trader being profitable after 1y over a 10y seasoned pro?  


# Simulation parameters for trading strategies
n_traders = 10
n_trades = 1000
starting_capital = 1000
trade_size = 10

# Generate P&L paths for two groups of traders
np.random.seed(42)

# Group 1: Discretionary traders with positive EV
disc_wealths = np.zeros((n_traders, n_trades + 1))
disc_wealths[:, 0] = starting_capital

# Group 2: Quant traders with positive EV
quant_wealths = np.zeros((n_traders, n_trades + 1))
quant_wealths[:, 0] = starting_capital

for i in range(n_trades):
    # Discretionary traders (60% win rate with varying returns)
    results_disc = np.random.random(n_traders) < 0.60
    wins_disc = results_disc * trade_size * (1 + 0.2 * np.random.random(n_traders))
    losses_disc = ~results_disc * trade_size * (0.8 + 0.2 * np.random.random(n_traders))
    net_results_disc = wins_disc - losses_disc
    disc_wealths[:, i+1] = disc_wealths[:, i] + net_results_disc
    
    # Quant traders (58% win rate with more consistent returns)
    results_quant = np.random.random(n_traders) < 0.58
    wins_quant = results_quant * trade_size * 1.1
    losses_quant = ~results_quant * trade_size * 0.9
    net_results_quant = wins_quant - losses_quant
    quant_wealths[:, i+1] = quant_wealths[:, i] + net_results_quant

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Discretionary Traders', 'Quant Traders')
)

# Add traces for Discretionary traders
x_vals = np.arange(n_trades + 1)
for i in range(n_traders):
    opacity = 0.3 + (0.7 * i/n_traders)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=disc_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(255, 0, 255, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# Add Discretionary traders trendline
disc_mean = np.mean(disc_wealths, axis=0)
z_disc = np.polyfit(x_vals, disc_mean, 1)
p_disc = np.poly1d(z_disc)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_disc(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for Quant traders
for i in range(n_traders):
    opacity = 0.3 + (0.7 * i/n_traders)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=quant_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 0, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

# Add Quant traders trendline
quant_mean = np.mean(quant_wealths, axis=0)
z_quant = np.polyfit(x_vals, quant_mean, 1)
p_quant = np.poly1d(z_quant)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_quant(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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
        title='Capital ($)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()



# Notice effective discretionary and quant traders have the same stability in a foward-looking sense - both can accumulate wealth and they are optimizing for the same EV just differently


# ###### ______________________________________________________________________________________________________________________________________


# ##### Professional Players and Traders
# 
# There is no conflicting theory on this - poker and trading are *skills*, I am not talking about your day trading gurus here, I'm talking about *real* academic literature
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# *♥️ Poker (RL can learn optimal play)*
# 
# - Moravčík et al., Science 2017 — “DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker” showed that deep reinforcement learning with lookahead search achieved statistically significant positive EV vs. professional players.
# 
# - Brown et al., NeurIPS 2020 — “ReBeL: A General Reinforcement Learning Algorithm that Uses Search” demonstrated self-play RL with subgame solving can reach superhuman performance in HUNL poker.
# 
# *✂️ Hedging (RL can learn optimal re-hedging)*
# 
# - Buehler et al., Quantitative Finance 2019 — “Deep Hedging” proved neural-network RL agents can optimize hedging under transaction costs and model risk, outperforming classical delta strategies in simulated markets.
# 
# - Kolm & Ritter, SSRN 2019 — “Dynamic Replication and Hedging: A Reinforcement Learning Approach” showed RL-based hedgers can flexibly adapt to discrete trading, costs, and model misspecification to improve P&L risk metrics.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# Something can be learned in games of incomplete information - in other words a player's action can influence their edge or expected value
# 
# Given the expected value tells us something about the outcome of one play, it thus will tell us something about a series of plays dictating a player's wealth path
# 
# A simple argument suggesting the skill of professional poker players can be found in reinforcement learning
# 
# Reinforcement learning is an area of machine learning concerned with experiential learning, improving action in an environment based on a reward function (feedback)
# 
# For a single trade/hand, we want to find the optimal policy:
#  
#   $$\pi^* = \argmax_{\pi} \mathbb{E}[R|\pi]$$
#  
#   where $R$ is the return/payoff and $\pi$ is our policy (trading/betting decisions)
#  
#   More formally, for a sequence of decisions:
# 
#   $$\pi^* = \argmax_{\pi} \mathbb{E}[\sum_{t=0}^{T} \gamma^t R_t|\pi]$$
#  
#   where:
#   - $\pi$ is our policy mapping states to actions  
#   - $R_t$ is the reward at time t
#   - $\gamma$ is a discount factor
#   - $T$ is the time horizon
#  
#   The optimal policy $\pi^*$ gives us the strategy that maximizes expected value across all possible states and actions
# 


# Simulation parameters
n_players = 10
n_periods = 1000
starting_wealth = 1000
bet_size = 10

# Generate wealth paths for two groups
np.random.seed(42)

# Group 1: Novice players/traders (negative EV)
novice_wealths = np.zeros((n_players, n_periods + 1))
novice_wealths[:, 0] = starting_wealth

# Group 2: Pro players/traders (positive EV) 
pro_wealths = np.zeros((n_players, n_periods + 1))
pro_wealths[:, 0] = starting_wealth

for i in range(n_periods):
    # Novice players/traders have 45% win rate
    novice_results = np.random.random(n_players) < 0.45
    novice_wins = novice_results * bet_size
    novice_losses = ~novice_results * bet_size
    novice_net = novice_wins - novice_losses
    novice_wealths[:, i+1] = novice_wealths[:, i] + novice_net
    
    # Pro players/traders have 55% win rate
    pro_results = np.random.random(n_players) < 0.55
    pro_wins = pro_results * bet_size
    pro_losses = ~pro_results * bet_size
    pro_net = pro_wins - pro_losses
    pro_wealths[:, i+1] = pro_wealths[:, i] + pro_net

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Novice Players/Traders (-EV)', 'Pro Players/Traders (+EV)')
)

# Add traces for Novice players/traders
x_vals = np.arange(n_periods + 1)
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=novice_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(255, 0, 255, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

# Add Novice trendline
novice_mean = np.mean(novice_wealths, axis=0)
z_novice = np.polyfit(x_vals, novice_mean, 1)
p_novice = np.poly1d(z_novice)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_novice(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for Pro players/traders
for i in range(n_players):
    opacity = 0.3 + (0.7 * i/n_players)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pro_wealths[i],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 255, {opacity})', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

# Add Pro trendline
pro_mean = np.mean(pro_wealths, axis=0)
z_pro = np.polyfit(x_vals, pro_mean, 1)
p_pro = np.poly1d(z_pro)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=p_pro(x_vals),
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Trend',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Periods',
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

fig.show()



# **Remark:** We see success in the literature deploying reinforcement algorithms for both poker playing and trading (specifically in dynamic hedging) - these agents are only as *smart* as the data they have access to.  They can not experience things continuously (pretty darn close) and they can't consider things as dynamically as a human player or trader.  In terms of pure EV these models can learn to play and trade effectively, even more evidence suggesting a professional poker player or trader is capable of learning such a function along with other intagibles and continuous access to information with dynamic responses.
# 
# **TL;DR:**
# You can make money by learning optimal actions based on your experience in an environment as a machine learning algorithm or a human
# 
# In other words, if you **actually have learned how to play poker or trade** you should see relative stability in your performance over a large series of plays, this is 
# 
# precisely what I argued for in my video on trading metrics and why they are overrated!  Versus a novice who will see tremendous variability and likely blown accounts


# ---


# #### 2.) 📈 General Objective: Poker & Trading
# 
# ##### Maximizing Expected Value
# 
# Regardless of your approach to poker or trading (quant, discretionary, market-making - whatever)
# 
# The objective can be reduced to
# 
# $$\max_{a \in A} \mathbb{E}[R|a]$$
#  
#  where:
#  - $a$ represents possible actions (bet sizes, entry/exit points)
#  - $R$ is the return/payoff
#  - $A$ is the set of all possible actions
# 
# We want to learn a policy $\pi$ that is a function of our environment such that
# 
#  $$\pi(s) \rightarrow a \implies \mathbb{E}[R|\pi(s)]$$
# 
#  where:
#  - $s$ represents the state/environment
#  - $\pi(s)$ maps states to actions
#  - $a$ is the action taken
#  - $R$ is the return/payoff


# Generate sample data for expected value function
a = np.linspace(-5, 5, 100)  # Action space
ev = -0.5 * (a - 2)**2 + 3   # Example EV function with maximum at a=2

# Create figure
fig = go.Figure()

# Add EV function
fig.add_trace(
    go.Scatter(
        x=a,
        y=ev,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 0.8)', width=3),
        name='Expected Value'
    )
)

# Add optimal point
optimal_a = 2
optimal_ev = -0.5 * (optimal_a - 2)**2 + 3
fig.add_trace(
    go.Scatter(
        x=[optimal_a],
        y=[optimal_ev],
        mode='markers',
        marker=dict(color='white', size=12, symbol='star'),
        name='Optimal Action'
    )
)

# Add some "noisy" realizations
np.random.seed(42)
sample_points = 30
random_a = np.random.uniform(-5, 5, sample_points)
noise = np.random.normal(0, 0.5, sample_points)
random_ev = -0.5 * (random_a - 2)**2 + 3 + noise

fig.add_trace(
    go.Scatter(
        x=random_a,
        y=random_ev,
        mode='markers',
        marker=dict(color='rgba(255, 0, 0, 0.3)', size=8),
        name='Individual Outcomes'
    )
)

# Update layout
fig.update_layout(
    title='Expected Value Maximization',
    xaxis_title='Action (a)',
    yaxis_title='Expected Value',
    height=500,
    width=1000,
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



# To maximize EV poker players and traders need to *experience* their environment to learn how to act optimally to
# - decide when to play a hand or enter a trade
# - decide when to raise or increase a position size
# - decide when to cut losses or exit a position
# - . . .


# If this sounds abtract its because it is - there are 10,000,001 strategies and ways to make money in both environments
# 
# Moreover, it is difficult to determine what is *luck* and what is just a *lucky sample path*
# 
# There is a *MASSIVE* difference between a discretionary trader with stable P/L YoY and a guru day trader that throws darts at a dart board
# - We will further discuss this in my video on Quant vs. Discretionary Trading
# 
# This is an *abstraction* but is literally what we are solving for when we optimize for expected value. . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Model Informed Decision Making
# 
# Poker and Trading both operate in extremely complex and dynmic systems
# 
# That being the case, *models will always be wrong* as both poker players and traders operate with incomplete information
# 
# In any case, models and associated values are useful for players and traders to consider as they operate and aim to maximize their expected value
# 
# **Poker Players and Traders form subjective positions based on the accumulation of their experience - if they does this productively they will accumulate wealth**
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Implied Prob/Vol vs. Pot/Out Odds
# 
# These are NOT probabilities in the classical sense - there is NO convergence, the entire system is dynamic 
# 
# If it could be perfectly modeled it would be a game of chance not a game of incomplete information with ability to influence EV
# 
# Though these calculations give a general sense of the environment **it does not tell us what will happen** contrary to what some naive players or traders think
# 
# **AGAIN** the ability for an experienced agent to act optimally off of this information is the key to generating wealth
# ###### ______________________________________________________________________________________________________________________________________
# 
# **The following are surely wrong, but offer information to act off of**
# 
#  *Implied Probability from Odds:*
# 
#  $$P_{implied} = \frac{1}{1 + odds}$$
# 
#  *Pot Odds:*
# 
#  $$P_{pot} = \frac{call\_size}{pot\_size + call\_size}$$
# 
#  *Outs Probability:*
# 
#  $$P_{outs} = \frac{n\_outs}{n\_remaining\_cards}$$
# 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Trading: Implied Probability
# 
# Implied measures are what the market is pricing in a *forward-looking* sense
# 
# This says nothing of *what is going to happen* but gives a sense of the overall pricing methodology
# 
# In any case, no implied measure is *predictive* and it *does not converge*
# 
# An implied probability of *80%* does not mean that out of 100 times you will see that outcome on average *80* times, that is a classical frequentist interpretation
# 
# This does not apply here - we can only realize the outcome one time!


# Simulate fair coin flips converging to 0.5 and implied prob shock scenario
n_flips = 1000
shock_point = 800

# Generate fair coin flip sequence
np.random.seed(42)
flips = np.random.random(n_flips) < 0.5
running_heads = np.cumsum(flips)
running_prob = running_heads / np.arange(1, n_flips + 1)

# Generate implied probability that diverges from reality
implied_prob = np.zeros(n_flips)
implied_prob[:shock_point] = 0.8  # High implied probability
implied_prob[shock_point:] = 0.0  # Shock - drops to zero

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Fair Coin Convergence', 'Implied Probability vs Reality Shock')
)

# Add trace for coin flip convergence
fig.add_trace(
    go.Scatter(
        x=np.arange(1, n_flips + 1),
        y=running_prob,
        mode='lines',
        line=dict(color='#00FFFF', width=2),  # Neon cyan
        name='Running Probability'
    ),
    row=1, col=1
)

# Add horizontal line at 0.5
fig.add_trace(
    go.Scatter(
        x=[1, n_flips],
        y=[0.5, 0.5],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='True Probability'
    ),
    row=1, col=1
)

# Add trace for implied probability and shock
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 10, n_flips),  # Time from 0 to 10
        y=implied_prob,
        mode='lines',
        line=dict(color='#FF1493', width=2),  # Neon pink
        name='Implied Probability'
    ),
    row=1, col=2
)

# Add vertical line at shock point - now extends full height
shock_time = 10 * shock_point/n_flips  # Convert to time scale
fig.add_trace(
    go.Scatter(
        x=[shock_time, shock_time],
        y=[0, 1],  # Full height
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Event Realization'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,  # Show legend
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    title='Number of Flips',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

fig.update_xaxes(
    title='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

for i in range(1, 3):
    fig.update_yaxes(
        title='Probability',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 1],
        row=1, col=i
    )

fig.show()



# Systematically, if we operate on this environment where we believe implied probability is over or under stated *we can profit if we are correct on average*
# 
# This doesn't have to be *static* for a fixed set of rules but at a traders discretion, the response to these events will subsequently be over or understated
# 
# again on average, if the trader is correct they can profit from this mispricing - this is something that can be learned from experience
# 
# *"No way its 80%"*, *"Should be closer to a wash 50%"* and positions can be taken respectively. . .


# Simulate implied probability and price path with shock
n_points = 1000
shock_point = 800

# Generate implied probability sequence
implied_prob = np.zeros(n_points)
implied_prob[:shock_point] = 0.8  # High implied probability
implied_prob[shock_point:] = 0.2  # Drops after event realization

# Generate price path
np.random.seed(42)
price = np.zeros(n_points)
price[0] = 100  # Starting price
volatility = 0.02  # Reduced volatility for more stable path

# Stable price path before event
for i in range(1, shock_point):
    price[i] = price[i-1] * (1 + np.random.normal(0, volatility))

# Sharp drop at event
drop_amount = 50  # 15 point drop
price[shock_point] = price[shock_point-1] - drop_amount

# Continue with lower price level after event
for i in range(shock_point+1, n_points):
    price[i] = price[i-1] * (1 + np.random.normal(0, volatility))

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Implied Probability', 'Price Path')
)

# Add trace for implied probability
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 10, n_points),
        y=implied_prob,
        mode='lines',
        line=dict(color='#00FFFF', width=2),  # Neon cyan
        name='Implied Probability'
    ),
    row=1, col=1
)

# Add trace for price path
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 10, n_points),
        y=price,
        mode='lines',
        line=dict(color='#FF1493', width=2),  # Neon pink
        name='Price'
    ),
    row=1, col=2
)

# Add vertical lines at shock point
shock_time = 10 * shock_point/n_points
for i in range(1, 3):
    fig.add_trace(
        go.Scatter(
            x=[shock_time, shock_time],
            y=[0, max(np.max(implied_prob), 1)] if i==1 else [0, np.max(price)],
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Earnings Announcement' if i==1 else 'Earnings Announcement (Price)',
            showlegend=i==1
        ),
        row=1, col=i
    )

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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
        title='Probability' if i==1 else 'Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 1] if i==1 else None,
        row=1, col=i
    )

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Poker: Pot Odds vs Out Probability
# 
# **Out Probability:** The chance of drawing a card that improves your hand to the winning hand
# 
# **Pot Odds:** The ratio of the amount you need to call versus the total pot size after your call
# 
#  Generally, for positive EV decisions: 
#  
#  $$\text{Out Probability} > \text{Pot Odds}$$
# 
#  Where 
#  
#  $$\text{Pot Odds} = \frac{\text{Call Amount}}{\text{Call Amount} + \text{Pot Size}}$$
# 


# Simulate pot odds vs out probability scenario
n_points = 1000
shock_point = 800

# Generate out probability sequence
out_prob = np.zeros(n_points)
out_prob[:shock_point] = 0.3  # 30% chance of hitting outs
out_prob[shock_point:] = 0.0  # Drops to zero when opponent shows the nuts

# Generate pot odds sequence
pot_odds = np.zeros(n_points)
pot_odds[:shock_point] = 0.2  # Getting 5-to-1 odds
pot_odds[shock_point:] = 0.8  # Odds worsen dramatically after opponent's action

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Out Probability Over Time', 'Pot Odds vs Out Probability')
)

# Add trace for out probability
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 10, n_points),
        y=out_prob,
        mode='lines',
        line=dict(color='#00FFFF', width=2),  # Neon cyan
        name='Out Probability'
    ),
    row=1, col=1
)

# Add trace for pot odds
fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 10, n_points),
        y=pot_odds,
        mode='lines',
        line=dict(color='#FF1493', width=2),  # Neon pink
        name='Pot Odds'
    ),
    row=1, col=1
)

# Add vertical line at shock point
shock_time = 10 * shock_point/n_points
fig.add_trace(
    go.Scatter(
        x=[shock_time, shock_time],
        y=[0, 1],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Opponent Shows'
    ),
    row=1, col=1
)

# Add scatter plot comparing odds
fig.add_trace(
    go.Scatter(
        x=out_prob,
        y=pot_odds,
        mode='markers',
        marker=dict(
            color=np.linspace(0, 10, n_points),
            colorscale='Viridis',
            size=5
        ),
        name='Odds Relationship'
    ),
    row=1, col=2
)

# Add diagonal line for break-even
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Break-even Line'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    title='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

fig.update_xaxes(
    title='Out Probability',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

for i in range(1, 3):
    fig.update_yaxes(
        title='Probability' if i==1 else 'Pot Odds',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 1],
        row=1, col=i
    )

fig.show()




# **Long story short**
# 
# These **models are certainly wrong** but they give the player or trader information to act with
# 
# Their ability to navigate the uncertainty of the space while considering these models and **rejecting them** when necessary is **KEY** to maximizing **EV**


# ###### ______________________________________________________________________________________________________________________________________


# ##### Taking EV from Other Players
# 
# In the context of poker, if you act with intent of maximizing expected value it is easy to smash other players
# 
# In fact, if you act with negative or near zero EV you can make it positive by simply outplaying people who don't understand it as a concept
# 
# ##### Example: Taking EV from Other Players
# 
# You act with $\mathbb{E}[H_{1}] = -\$.10$/hand
# 
# Your opponent acts with $\mathbb{E}[H_{2}] = -\$.20$/hand
# 
# It is a **zero-sum game** so if you outplay them your EV becomes: $ -.10 + (- (-.20)) = \$.10$/hand


# Simulate wealth paths for two players with different EV
n_hands = 1000
np.random.seed(42)

# Player 1 has -$0.10 EV per hand
player1_ev = -0.10
player1_std = 1.0  # Standard deviation of outcomes
player1_outcomes = np.random.normal(player1_ev, player1_std, n_hands)
player1_wealth = np.cumsum(player1_outcomes)

# Player 2 has -$0.20 EV per hand (worse decisions)
player2_ev = -0.20
player2_std = 1.0
player2_outcomes = np.random.normal(player2_ev, player2_std, n_hands)
player2_wealth = np.cumsum(player2_outcomes)

# Player 1's actual wealth including what they gain from Player 2's losses
player1_total_wealth = player1_wealth - player2_wealth

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Individual Wealth Paths', 'Zero-Sum Combined Wealth')
)

# Plot individual wealth paths
fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player1_wealth,
        mode='lines',
        line=dict(color='#00FFFF', width=2),
        name='Player 1 (EV=-$0.10)'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player2_wealth,
        mode='lines',
        line=dict(color='#FF1493', width=2),
        name='Player 2 (EV=-$0.20)'
    ),
    row=1, col=1
)

# Plot zero-sum combined wealth
fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player1_total_wealth,
        mode='lines',
        line=dict(color='#39FF14', width=2),
        name='Player 1 Total Wealth'
    ),
    row=1, col=2
)

# Add horizontal line at 0
fig.add_trace(
    go.Scatter(
        x=[0, n_hands],
        y=[0, 0],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Break Even'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Hands',
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

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Taking EV in Option Contracts
# 
# This is exactly the same for financial derivative contracts like options
# 
# These contracts are zero-sum, one party receives funds the other party pays out funds at experation
# 
# If the contract is overvalued initially due to fear then you can *outplay* your counterparty by selling them an overvalued contract and profit statistically
# 
# *Note:* I am not making a remark about hedging or covered contracts - this is beyond the scope of this video but for those who care to make the point I figured I mention it. . .


# Create sample data for option payoff diagram
strike_price = 100
premium = 5
spot_prices = np.linspace(70, 130, 100)

# Calculate payoffs for long call
long_call_payoffs = np.maximum(spot_prices - strike_price, 0) - premium
short_call_payoffs = -long_call_payoffs

# Create figure
fig = go.Figure()

# Add long call payoff line
fig.add_trace(
    go.Scatter(
        x=spot_prices,
        y=long_call_payoffs,
        mode='lines',
        name='Long Call',
        line=dict(color='#00FFFF', width=2)
    )
)

# Add short call payoff line
fig.add_trace(
    go.Scatter(
        x=spot_prices,
        y=short_call_payoffs,
        mode='lines',
        name='Short Call',
        line=dict(color='#FF1493', width=2)
    )
)

# Add break-even line
fig.add_trace(
    go.Scatter(
        x=[70, 130],
        y=[0, 0],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Break Even'
    )
)

# Update layout
fig.update_layout(
    title='Call Option Payoff Diagram',
    xaxis_title='Spot Price at Expiration',
    yaxis_title='Profit/Loss',
    height=500,
    width=800,
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



# However, unlike in poker we don't actually sit across from an uninformed counterparty who is likely hedging their opposing position
# 
# We are instead playing a game then against the market itself in whether there is a mispricing in the contract or not. . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Being On Tilt and Revenge Trading
# 
# Downside variation **IS NOT** an invitation to blowout your account in either ta Poker or Trading sense
# 
# Unlike in finance, it is extremely easy to profit off of negative EV when playing poker
# 
# For some reason its *cool* to be good at poker you get a bunch of guys together and if you're good you feel like "the man"
# 
# In any case, its crazy how easy it is to take money from people who don't understand EV
# 
# Tilting and Revenge Trading are *suboptimal* and often what lead people to believe the space is *gambling*, along with all the negative connotations (rightfully so)
# 
# Games of chance should carry those, they are literally mathematically suboptimal and that is fixed - this is not true for Poker and Trading
# 
# We can always act suboptimally, that doesn't make it pure gambling 


# Simulate wealth paths showing tilt behavior
n_hands = 100
np.random.seed(2)

# Player 1 has -$0.10 EV per hand (consistent play)
player1_ev = -0.10
player1_std = 1.0
player1_outcomes = np.random.normal(player1_ev, player1_std, n_hands)

# Player 2 starts normal but goes on tilt after a big loss
player2_ev = -0.20
player2_std = 1.0
player2_outcomes = np.random.normal(player2_ev, player2_std, n_hands)

# Find first major drawdown for player 2 (loss > 3 std dev)
# Add check to ensure there is at least one major drawdown
tilt_indices = np.where(player2_outcomes < -3*player2_std)[0]
if len(tilt_indices) > 0:
    tilt_point = tilt_indices[0]
else:
    # If no major drawdown, set tilt point halfway through
    tilt_point = n_hands // 2

# After tilt point, player 2 increases bet size and has worse EV
player2_outcomes[tilt_point:] = np.random.normal(-0.5, 2.0, n_hands-tilt_point)

# Calculate cumulative wealth
player1_wealth = np.cumsum(player1_outcomes)
player2_wealth = np.cumsum(player2_outcomes)
player1_total_wealth = player1_wealth - player2_wealth

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Individual Wealth Paths (With Tilt)', 'Zero-Sum Combined Wealth')
)

# Plot individual wealth paths
fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player1_wealth,
        mode='lines',
        line=dict(color='#00FFFF', width=2),
        name='Player 1 (Consistent EV=-$0.10)'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player2_wealth,
        mode='lines',
        line=dict(color='#FF1493', width=2),
        name='Player 2 (Goes on Tilt)'
    ),
    row=1, col=1
)

# Add vertical line at tilt point
fig.add_vline(
    x=tilt_point,
    line_dash="dash",
    line_color="red",
    annotation_text="Tilt Point",
    row=1, col=1
)

# Plot zero-sum combined wealth
fig.add_trace(
    go.Scatter(
        x=np.arange(n_hands),
        y=player1_total_wealth,
        mode='lines',
        line=dict(color='#39FF14', width=2),
        name='Player 1 Total Wealth'
    ),
    row=1, col=2
)

# Add horizontal line at 0
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="white",
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        title='Number of Hands',
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

fig.show()



# Both players are bad, but here we see Player 1 play consistently bad and they outperform Player 2 who goes on tilt making even worse decisions
# 
# **Manage your risk!**  I literally do not care if I lose a trade or a hand of poker - **if you do then you are not engaging with the space correctly** 


# ---


# #### 3.) ⚠️ Risk Management
# 
# ##### Cutting Losses Early
# 
# Statistically, we have to lose to continue to make money
# 
# The quicker and more efficiently we can lose the more money we can make in the long run 
# 
# This is **NOT** a trivial task and is a function of the strategy you are operating with and experience


# Simulate wealth paths showing loss cutting vs holding losses
n_periods = 200
np.random.seed(42)

# Trader 1 cuts losses early and has good risk management
trader1_outcomes = np.random.normal(0.15, 0.8, n_periods)  # Lower volatility, higher mean
# Add some potential big losses but cut them early
big_loss_points = np.random.randint(0, n_periods, 5)
for point in big_loss_points:
    trader1_outcomes[point] = -0.3  # Smaller losses due to good risk management

# Trader 2 has poor risk management
trader2_outcomes = np.random.normal(0.15, 0.8, n_periods)  # Higher volatility, slightly lower mean
# Add same big loss points but with poor risk management
for point in big_loss_points:
    trader2_outcomes[point] = -10.0  # Larger losses due to poor risk management

# Calculate cumulative wealth
trader1_wealth = np.cumsum(trader1_outcomes)
trader2_wealth = np.cumsum(trader2_outcomes)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Trader With Good Risk Management', 'Trader With Poor Risk Management')
)

# Plot wealth path for trader 1
fig.add_trace(
    go.Scatter(
        x=np.arange(n_periods),
        y=trader1_wealth,
        mode='lines',
        line=dict(color='#00FFFF', width=2),
        name='Good Risk Management'
    ),
    row=1, col=1
)

# Plot wealth path for trader 2  
fig.add_trace(
    go.Scatter(
        x=np.arange(n_periods),
        y=trader2_wealth,
        mode='lines',
        line=dict(color='#FF1493', width=2),
        name='Poor Risk Management'
    ),
    row=1, col=2
)

# Add horizontal lines at 0
fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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

fig.show()



# The amount a novice player will overvalue a pair, two pair, . . . is substantial - **the ability to fold this hands comes from observing a lot of hands**
# 
# The ability to exit a position with a loss early when the trade goes against your initial expectation again **comes from observing a lot of trades**


# ###### ______________________________________________________________________________________________________________________________________


# ##### Letting Winners Run
# 
# We'll talk more about this in the future but letting winners run is a big part of risk management and allows bigger hits without changing the probability of win/loss
# 
# **Law of Total Expectation for Win/Loss:**
#  
#  $$E[X] = P(win)E[X|win] + P(loss)E[X|loss]$$
#  
#  This fundamental principle applies to both domains:
#  - Poker: (Prob of winning hand × Avg winning amount) + (Prob of losing hand × Avg losing amount)
#  - Trading: (Prob of profitable trade × Avg winning trade) + (Prob of unprofitable trade × Avg losing trade)
# 
# In a trading context, we want to get to a "free trade" as quickly as possible
# 
# Then we can let it run with some sort of trailing stop to get bigger hits without impacting overall probability of win/loss
# 
# This is *just one way* to optimize for a lever in that EV equation


# Simulate wealth paths comparing letting winners run vs fixed profit taking
n_periods = 100
np.random.seed(42)

# Base probability of winning for both strategies
win_prob = 0.51
base_outcomes = np.random.choice([1.0, -1.0], n_periods, p=[win_prob, 1-win_prob])

# Trader with trailing stops (letting winners run)
trailing_stop_outcomes = base_outcomes.copy()
# Amplify winning trades with random multipliers
trailing_stop_outcomes[trailing_stop_outcomes > 0] *= np.random.uniform(0.5, 2.5, (base_outcomes > 0).sum())
# Keep losses controlled
trailing_stop_outcomes[trailing_stop_outcomes < 0] *= np.random.uniform(0.3, 0.7, (base_outcomes < 0).sum())

# Add some big winners that keep running
big_win_points = np.random.randint(0, n_periods-5, 3)  # 3 big winning streaks
for point in big_win_points:
    if base_outcomes[point] > 0:  # Only extend actual winning trades
        streak_length = np.random.randint(3, 6)
        for i in range(streak_length):
            if point + i < n_periods:
                trailing_stop_outcomes[point + i] = 1.0 + i * 0.5  # Increasing profits

# Fixed profit/loss trader
fixed_take_outcomes = base_outcomes.copy()
# Apply consistent gains and losses
fixed_take_outcomes[fixed_take_outcomes > 0] *= 0.5  # Fixed profit
fixed_take_outcomes[fixed_take_outcomes < 0] *= 0.4  # Fixed loss

# Calculate cumulative wealth
trailing_stop_wealth = np.cumsum(trailing_stop_outcomes)
fixed_take_wealth = np.cumsum(fixed_take_outcomes)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Letting Winners Run (Trailing Stops)', 'Fixed Profit Taking')
)

# Plot wealth path for trailing stop strategy
fig.add_trace(
    go.Scatter(
        x=np.arange(n_periods),
        y=trailing_stop_wealth,
        mode='lines',
        line=dict(color='#00FF00', width=2),
        name='Trailing Stops'
    ),
    row=1, col=1
)

# Plot wealth path for fixed take profit/loss
fig.add_trace(
    go.Scatter(
        x=np.arange(n_periods),
        y=fixed_take_wealth,
        mode='lines',
        line=dict(color='#FFD700', width=2),
        name='Fixed Takes'
    ),
    row=1, col=2
)

# Add horizontal lines at 0
fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)

# Update layout
fig.update_layout(
    height=500,
    width=1300,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
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

fig.show()



# Calculate win/loss statistics for both strategies
trailing_wins = trailing_stop_outcomes[trailing_stop_outcomes > 0]
trailing_losses = trailing_stop_outcomes[trailing_stop_outcomes < 0]
fixed_wins = fixed_take_outcomes[fixed_take_outcomes > 0]
fixed_losses = fixed_take_outcomes[fixed_take_outcomes < 0]

print("Trailing Stop Strategy:")
print(f"Win Probability: {len(trailing_wins)/len(trailing_stop_outcomes):.2%}")
print(f"Average Win: ${trailing_wins.mean():.2f}")
print(f"Average Loss: ${trailing_losses.mean():.2f}")
print("\nFixed Take Strategy:")
print(f"Win Probability: {len(fixed_wins)/len(fixed_take_outcomes):.2%}")
print(f"Average Win: ${fixed_wins.mean():.2f}")
print(f"Average Loss: ${fixed_losses.mean():.2f}")



# Notice roughly the same probability of winning a trade but letting winners run increases the average winner and overall P/L!
# 
# In Poker, we have to let winners run in this same way but have to avoid forcing other players to fold - if you play too tight nobody is going to call your raise!
# 
# **THIS IS NOT AN INVITATION TO BLOWOUT**
# 
# A good hand or running trade is not an invitation to consider outsized leverage or raises, continue to operate like you have to face 100,000 more hands - that is your goal. . .


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Games of chance (Roullete, Slots, . . .) are pure gambling, no action is optimal and over time you are gaurenteed to statistically lose money, the only optimal action is not playing
# - Games of incomplete information (Poker, Trading) have *random elements* but action or inaction influences edge and wealth over time
# - *Learning* to play games of chance involves learning some optimal policy functions that maximizes expected value, this is how machine learning works and human learning
# - Experience does not mean you *know* what will happen, but assuming a policy is effective, it means you can accumulate wealth over time if $EV>0$ or if you take it from others
# - Poker and Trading demand experience to act optimally in uncertainty, to act (or not) on model values based on informed subjective opinion to generate wealth
# - In addition to optimal policy (action), effective risk management and an understanding theoretically of the space is *necessary* and separetes the good from great
# - Being on tilt or revenge trading shows a lack of theoretical understanding of the space and encourages suboptimal decision making akin to gambling or negative EV
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Quant vs. Discretionary Trading
# 
# - Is the Market Random?
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Build an Earnings Event Options Trading Dashboard
# 
# - Automated Delta-Neutral Trading System (Algorithmically Capitalizing On Volatility Speculation)


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

