# ### 🃏 Quant Explains Why Banks are Casinos
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


# ### 📖 Sections
# 
# 
# #### 1.) 🎰 Casinos
# 
# - Making Money in Randomness
# 
# - Managing Risk
# 
# #### 2.) 🏛️ Banks
# 
# - Making Money Making a Market
# 
# - Managing Risk
# 
# #### 3.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎰 Casinos
# 
# ##### Making Money in Randomness
# 
#  Expected value of a casino game:
# 
#  $$E[X] = E[\text{payoff}|\text{win}] \cdot P(\text{win}) + E[\text{payoff}|\text{loss}] \cdot P(\text{loss})$$
# 
#  We make money if
# 
#  $$E[X] = E[\text{payoff}|\text{win}] \cdot P(\text{win}) + E[\text{payoff}|\text{loss}] \cdot P(\text{loss}) > 0$$
# 
# 
# Popular Casino Games Expected Values:
#  
#  | Game | Expected Value per $1 Bet | House Edge |
#  |------|---------------------------|------------|
#  | **Roulette (Red)** | $$-\$0.053 = (\$1 \cdot \frac{18}{38}) + (-\$1 \cdot \frac{20}{38})$$ | 5.26% |
#  | **Craps (Pass)** | $$-\$0.014 = (\$1 \cdot \frac{244}{495}) + (-\$1 \cdot \frac{251}{495})$$ | 1.4% |
#  | **Slots** | $$(\$\text{jackpot} \cdot p_\text{j}) + (\$\text{med} \cdot p_\text{m}) + (\$\text{small} \cdot p_\text{s}) - \$1$$ | 2-15% |
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_players = 10
n_spins = 100
starting_wealth = 1000
bet_size = 250

# Game probabilities and payouts
# Roulette
p_win_roulette = 18/37
payout_roulette = 2  # 1:1 payout for red/black bets

# Craps (Pass Line)
p_win_craps = 244/495
payout_craps = 2  # 1:1 payout

# Slots (simplified)
p_win_slots = 0.45  # 45% chance of any win
payout_slots = 2  # Average payout when winning

# Generate wealth paths for players in each game
player_wealths_roulette = np.zeros((n_players, n_spins + 1))
player_wealths_craps = np.zeros((n_players, n_spins + 1))
player_wealths_slots = np.zeros((n_players, n_spins + 1))
player_wealths_roulette[:, 0] = starting_wealth
player_wealths_craps[:, 0] = starting_wealth
player_wealths_slots[:, 0] = starting_wealth

# Simulate games
for i in range(n_spins):
    # Roulette
    results_roulette = np.random.random(n_players) < p_win_roulette
    wins_roulette = results_roulette * bet_size * (payout_roulette - 1)
    losses_roulette = ~results_roulette * bet_size
    net_results_roulette = wins_roulette - losses_roulette
    
    # Craps
    results_craps = np.random.random(n_players) < p_win_craps
    wins_craps = results_craps * bet_size * (payout_craps - 1)
    losses_craps = ~results_craps * bet_size
    net_results_craps = wins_craps - losses_craps
    
    # Slots
    results_slots = np.random.random(n_players) < p_win_slots
    wins_slots = results_slots * bet_size * (payout_slots - 1)
    losses_slots = ~results_slots * bet_size
    net_results_slots = wins_slots - losses_slots

    # Update player wealths
    player_wealths_roulette[:, i+1] = player_wealths_roulette[:, i] + net_results_roulette
    player_wealths_craps[:, i+1] = player_wealths_craps[:, i] + net_results_craps
    player_wealths_slots[:, i+1] = player_wealths_slots[:, i] + net_results_slots

# Stop player wealth at zero for all games
for wealth_array in [player_wealths_roulette, player_wealths_craps, player_wealths_slots]:
    for i in range(n_players):
        zero_indices = np.where(wealth_array[i] <= 0)[0]
        if len(zero_indices) > 0:
            first_zero = zero_indices[0]
            wealth_array[i, first_zero:] = 0

# Calculate casino wealth as negative sum of player wealth changes
casino_wealth_roulette = -np.sum(player_wealths_roulette - starting_wealth, axis=0)
casino_wealth_craps = -np.sum(player_wealths_craps - starting_wealth, axis=0)
casino_wealth_slots = -np.sum(player_wealths_slots - starting_wealth, axis=0)
total_casino_wealth = casino_wealth_roulette + casino_wealth_craps + casino_wealth_slots

# Create figure with 3 rows
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Player Wealth Paths', 'Casino Wealth Path', 'Total Casino Wealth', None, 'Bankruptcy Rate', None),
    row_heights=[0.5, 0.2, 0.3],
    specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]]
)

# Define colors for each game
roulette_colors = [f'rgba(0,255,0,{opacity})' for opacity in np.linspace(0.3, 1, n_players)]  # Neon green
craps_colors = [f'rgba(0,255,255,{opacity})' for opacity in np.linspace(0.3, 1, n_players)]   # Neon cyan
slots_colors = [f'rgba(255,0,255,{opacity})' for opacity in np.linspace(0.3, 1, n_players)]   # Neon purple

# Create frames for animation
frames = []
for step in range(n_spins + 1):
    frame = {"data": [], "name": str(step)}

    # Calculate axis limits
    all_player_wealths = np.concatenate([
        player_wealths_roulette[:, :step+1],
        player_wealths_craps[:, :step+1],
        player_wealths_slots[:, :step+1]
    ])
    all_casino_wealths = np.concatenate([
        casino_wealth_roulette[:step+1],
        casino_wealth_craps[:step+1],
        casino_wealth_slots[:step+1]
    ])
    
    y_min_players = min(np.min(all_player_wealths), 0)
    y_max_players = max(np.max(all_player_wealths), starting_wealth)
    y_min_casino = min(np.min(all_casino_wealths), 0)
    y_max_casino = max(np.max(all_casino_wealths), 0)
    y_min_total = min(np.min(total_casino_wealth[:step+1]), 0)
    y_max_total = max(np.max(total_casino_wealth[:step+1]), 0)

    # Calculate bankruptcy rates
    bankrupt_roulette = np.sum(player_wealths_roulette[:, step] <= 0) / n_players
    bankrupt_craps = np.sum(player_wealths_craps[:, step] <= 0) / n_players
    bankrupt_slots = np.sum(player_wealths_slots[:, step] <= 0) / n_players

    # Add traces for each game
    for i in range(n_players):
        # Roulette players
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=player_wealths_roulette[i, :step + 1],
                mode='lines',
                line=dict(color=roulette_colors[i], width=2),
                name='Roulette',
                showlegend=i==0
            )
        )
        # Craps players
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=player_wealths_craps[i, :step + 1],
                mode='lines',
                line=dict(color=craps_colors[i], width=2),
                name='Craps',
                showlegend=i==0
            )
        )
        # Slots players
        frame["data"].append(
            go.Scatter(
                x=np.arange(step + 1),
                y=player_wealths_slots[i, :step + 1],
                mode='lines',
                line=dict(color=slots_colors[i], width=2),
                name='Slots',
                showlegend=i==0
            )
        )

    # Casino traces
    frame["data"].extend([
        go.Scatter(
            x=np.arange(step + 1),
            y=casino_wealth_roulette[:step + 1],
            mode='lines',
            line=dict(color='rgba(0,255,0,1)', width=2),
            showlegend=False
        ),
        go.Scatter(
            x=np.arange(step + 1),
            y=casino_wealth_craps[:step + 1],
            mode='lines',
            line=dict(color='rgba(0,255,255,1)', width=2),
            showlegend=False
        ),
        go.Scatter(
            x=np.arange(step + 1),
            y=casino_wealth_slots[:step + 1],
            mode='lines',
            line=dict(color='rgba(255,0,255,1)', width=2),
            showlegend=False
        )
    ])

    # Total casino wealth trace
    frame["data"].append(
        go.Scatter(
            x=np.arange(step + 1),
            y=total_casino_wealth[:step + 1],
            mode='lines',
            line=dict(color='rgba(255,215,0,1)', width=3),  # Neon gold
            showlegend=False
        )
    )

    # Add bankruptcy rate bars
    frame["data"].extend([
        go.Bar(
            x=['Roulette', 'Craps', 'Slots'],
            y=[bankrupt_roulette, bankrupt_craps, bankrupt_slots],
            marker_color=['rgba(0,255,0,1)', 'rgba(0,255,255,1)', 'rgba(255,0,255,1)'],
            showlegend=False
        )
    ])

    frame["layout"] = {
        "xaxis": {"range": [0, max(step + 1, 5)]},
        "xaxis2": {"range": [0, max(step + 1, 5)]},
        "xaxis3": {"range": [0, max(step + 1, 5)]},
        "xaxis4": {"range": [-0.5, 2.5]},
        "yaxis": {"range": [y_min_players * 1.1, y_max_players * 1.1]},
        "yaxis2": {"range": [y_min_casino * 1.1, y_max_casino * 1.1]},
        "yaxis3": {"range": [y_min_total * 1.1, y_max_total * 1.1]},
        "yaxis4": {"range": [0, 1]}
    }

    frames.append(frame)

# Add initial traces
for i in range(n_players):
    # Roulette
    fig.add_trace(
        go.Scatter(
            x=[0], y=[starting_wealth],
            mode='lines',
            line=dict(color='rgba(0,255,0,1)', width=2),
            name='Roulette',
            showlegend=i==0
        ),
        row=1, col=1
    )
    # Craps
    fig.add_trace(
        go.Scatter(
            x=[0], y=[starting_wealth],
            mode='lines',
            line=dict(color='rgba(0,255,255,1)', width=2),
            name='Craps',
            showlegend=i==0
        ),
        row=1, col=1
    )
    # Slots
    fig.add_trace(
        go.Scatter(
            x=[0], y=[starting_wealth],
            mode='lines',
            line=dict(color='rgba(255,0,255,1)', width=2),
            name='Slots',
            showlegend=i==0
        ),
        row=1, col=1
    )

# Initial casino traces
fig.add_trace(
    go.Scatter(
        x=[0], y=[0],
        mode='lines',
        line=dict(color='rgba(0,255,0,1)', width=2),
        showlegend=False
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[0], y=[0],
        mode='lines',
        line=dict(color='rgba(0,255,255,1)', width=2),
        showlegend=False
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[0], y=[0],
        mode='lines',
        line=dict(color='rgba(255,0,255,1)', width=2),
        showlegend=False
    ),
    row=1, col=2
)

# Initial total casino wealth trace
fig.add_trace(
    go.Scatter(
        x=[0], y=[0],
        mode='lines',
        line=dict(color='rgba(255,215,0,1)', width=3),  # Neon gold
        showlegend=False
    ),
    row=2, col=1
)

# Initial bankruptcy rate bars
fig.add_trace(
    go.Bar(
        x=['Roulette', 'Craps', 'Slots'],
        y=[0, 0, 0],
        marker_color=['rgba(0,255,0,1)', 'rgba(0,255,255,1)', 'rgba(255,0,255,1)'],
        showlegend=False
    ),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=1000,
    width=1200,
    showlegend=True,
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
fig.update_xaxes(
    title='Number of Spins/Rolls/Pulls',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_xaxes(
    title='Number of Spins/Rolls/Pulls',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)
fig.update_xaxes(
    title='Number of Spins/Rolls/Pulls',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=2, col=1
)
fig.update_xaxes(
    title='Game Type',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    row=3, col=1
)

fig.update_yaxes(
    title='Wealth ($)',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_yaxes(
    title='Casino Profit ($)',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)
fig.update_yaxes(
    title='Total Casino Profit ($)',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=2, col=1
)
fig.update_yaxes(
    title='Bankruptcy Rate',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[0, 1],
    row=3, col=1
)

# Set initial axis ranges
fig.update_xaxes(range=[0, 5], row=1, col=1)
fig.update_xaxes(range=[0, 5], row=1, col=2)
fig.update_xaxes(range=[0, 5], row=2, col=1)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=1)
fig.update_yaxes(range=[0, starting_wealth * 1.1], row=1, col=2)
fig.update_yaxes(range=[0, starting_wealth * 3.3], row=2, col=1)

# Add frames
fig.frames = frames

fig.show()



# ##### Managing Risk
# 
#  - **Table Limits:** Capping maximum bets to limit potential losses
#  - **Position Limits:** Restricting total exposure across all tables/games
#  - **Bankroll Management:** Maintaining sufficient capital reserves
#  - **Diversification:** Multiple games/tables spread risk
#  - **Surveillance:** Monitoring for good players, cheating, and fraud
#  - **Insurance:** Coverage for large payouts/jackpots
# 


# ---


# #### 2.) 🏛️ Banks
# 
# ##### Making Money Making a Market
# 
#  Banks act as market makers, quoting prices to clients:
#  
#  $P_{exotic} = P_{base} + \Delta_{liquidity} + \Delta_{credit} + \Delta_{model}$
# 
#  
#  Where:
#  - $P_{base}$ is the theoretical fair value (model price)
# 
# We typically use a pricing model to develop $P_{base}$
# 
#  $$P_{base} = \mathbb{E}^{\mathbb{Q}}\left[\int_0^T e^{-r(T-t)}V_t dt \mid \mathcal{F}_0\right]$$
#  
# 
#  - $\Delta_{liquidity}$ adjusts for market depth/impact
#  - $\Delta_{credit}$ accounts for counterparty risk
#  - $\Delta_{model}$ includes model uncertainty buffer/profit
# 
# The objective is to quote a two-way (bid-ask/offer) then hedge to collect on the premium
# 
# Effectively, we are buying low or selling high and immediately hedging collecting on the difference (we do this a large number of times)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------
# Simulation parameters
# ---------------------------------
np.random.seed(42)
n_trades = 200
starting_wealth = 0

# Define market-making desks
markets = {
    "FI": {"mean_edge": 0.002, "vol": 0.01},        # Fixed Income
    "FX": {"mean_edge": 0.001, "vol": 0.015},       # Foreign Exchange
    "Equities": {"mean_edge": 0.003, "vol": 0.02},  # Equities
    "Options": {"mean_edge": 0.004, "vol": 0.03},   # Options
}

# Simulate P&L paths
bank_wealths = {}
for m, params in markets.items():
    pnl = np.random.normal(params["mean_edge"], params["vol"], n_trades)
    bank_wealths[m] = starting_wealth + np.cumsum(pnl)

# Compute total bank wealth
total_wealth = sum(bank_wealths.values())

# ---------------------------------
# Colors (neon finance palette)
# ---------------------------------
colors = {
    "FI": "rgba(0,255,0,1)",        # Neon green
    "FX": "rgba(0,255,255,1)",      # Neon cyan
    "Equities": "rgba(255,0,255,1)",# Neon magenta
    "Options": "rgba(255,140,0,1)", # Neon orange
    "Total": "rgba(255,215,0,1)",   # Neon gold
}

# ---------------------------------
# Build figure with subplots
# ---------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.15,
    subplot_titles=("Bank Desk Wealth Paths", "Total Bank Wealth Path")
)

# ---------------------------------
# Animation frames with dynamic axis scaling
# ---------------------------------
frames = []
for step in range(1, n_trades + 1):
    frame_data = []

    # Desk wealth lines
    for m, wealth in bank_wealths.items():
        frame_data.append(
            go.Scatter(
                x=np.arange(step),
                y=wealth[:step],
                mode="lines",
                line=dict(color=colors[m], width=3),
                name=m
            )
        )

    # Total wealth line
    frame_data.append(
        go.Scatter(
            x=np.arange(step),
            y=total_wealth[:step],
            mode="lines",
            line=dict(color=colors["Total"], width=4),
            name="Total Bank Wealth"
        )
    )

    # Dynamic axis ranges
    y_min_desk = min(min(w[:step].min() for w in bank_wealths.values()), 0)
    y_max_desk = max(max(w[:step].max() for w in bank_wealths.values()), 0)
    y_min_total = min(total_wealth[:step].min(), 0)
    y_max_total = max(total_wealth[:step].max(), 0)

    frame_layout = {
        "xaxis": {"range": [0, step + 5]},
        "xaxis2": {"range": [0, step + 5]},
        "yaxis": {"range": [y_min_desk * 1.1, y_max_desk * 1.1]},
        "yaxis2": {"range": [y_min_total * 1.1, y_max_total * 1.1]},
    }

    frames.append(go.Frame(data=frame_data, name=str(step), layout=frame_layout))

# ---------------------------------
# Add initial static traces
# ---------------------------------
for m in markets.keys():
    fig.add_trace(
        go.Scatter(
            x=[0], y=[starting_wealth],
            mode="lines",
            line=dict(color=colors[m], width=3),
            name=m
        ),
        row=1, col=1
    )

fig.add_trace(
    go.Scatter(
        x=[0], y=[starting_wealth],
        mode="lines",
        line=dict(color=colors["Total"], width=4),
        name="Total Bank Wealth"
    ),
    row=2, col=1
)

# ---------------------------------
# Layout & Controls
# ---------------------------------
fig.update_layout(
    height=900,
    width=1200,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    showlegend=True,
    title=dict(
        text="Bank Wealth Accumulation by Market Desk",
        y=0.96,
        x=0.5,
        xanchor="center",
        yanchor="top"
    ),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05,       # Upper left corner
        "y": 1.15,       # Above chart, below title
        "xanchor": "left",
        "yanchor": "top",
        "direction": "right",
        "buttons": [
            {
                "label": "▶Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 60, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            },
            {
                "label": "⏸Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }]
            }
        ]
    }]
)

# ---------------------------------
# Axes styling
# ---------------------------------
fig.update_xaxes(
    title="Trade Number",
    showgrid=True,
    gridcolor="rgba(128,128,128,0.2)",
    zeroline=True,
    zerolinecolor="rgba(128,128,128,0.5)"
)
fig.update_yaxes(
    title="Desk P&L",
    showgrid=True,
    gridcolor="rgba(128,128,128,0.2)",
    zeroline=True,
    zerolinecolor="rgba(128,128,128,0.5)",
    row=1, col=1
)
fig.update_yaxes(
    title="Total Bank Wealth",
    showgrid=True,
    gridcolor="rgba(128,128,128,0.2)",
    zeroline=True,
    zerolinecolor="rgba(128,128,128,0.5)",
    row=2, col=1
)

# ---------------------------------
# Initial axis ranges
# ---------------------------------
fig.update_xaxes(range=[0, 10], row=1, col=1)
fig.update_xaxes(range=[0, 10], row=2, col=1)
fig.update_yaxes(range=[-0.05, 0.05], row=1, col=1)
fig.update_yaxes(range=[-0.05, 0.05], row=2, col=1)

# Add frames
fig.frames = frames

fig.show()



# ##### Managing Risk
# 
#   - **Trade Size Limits:** Capping maximum notional value per trade to control risk exposure
#   - **Position Limits:** Restricting total Greeks (delta, gamma, vega) across all books
#   - **Capital Requirements:** Maintaining regulatory capital buffers for market making
#   - **Portfolio Diversification:** Trading multiple products/underlyings to spread risk
#   - **Dynamic Hedging:** Delta-hedging and managing higher-order risks
#   - **Counterparty Limits:** Controlling exposure to individual clients/counterparties
# 


# ---


# #### 3.) 💭 Closing Thoughts and Future Topics
# 
# **Executive Summary**
# 
# Key Similarities
#  - Both earn profits from the spread (bid-ask vs house edge)
#  - High volume of small transactions is key to profitability (no oversized positions)
#  - Both face "tail risk" from extreme events/market moves
# 
# Key Differences
#  - Banks make money in risk-neutrality, counterparty outcome is hedged
#  - Casinos make money in randomness, taking money from their counterparty
#  - Banks face systemic/correlation risk vs casinos operating independent games
# 
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
# - Stochastic Proccesses: Brownian Motion, Arithmetic (additive) Geometric (multiplicative) Brownian Motion
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
# - Most Popular Quant Models for Informed Trading
# - Buy Side vs. Sell Side Quants
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

