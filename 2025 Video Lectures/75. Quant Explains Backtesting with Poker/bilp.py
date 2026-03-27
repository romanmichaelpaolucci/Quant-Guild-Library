# ### 📈 Quant Explains Proper Backtesting with Poker
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
# #### 1.) 🎲 Playing Poker
# 
# - Face Up Poker, Probability, Statistics, and Expected Value
# 
# - Poker, Dynamic Probability, Statistics, and Expected Value
# 
# - Backtests are Real World Poker
# 
# - Dynamic Probability, Statistics, and Expected Value
# 
# #### 2.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎲 Playing Poker
# 
# ##### Backtesting and Finding a Trading Signal
# 
# Pocket aces, (same signal), simply continuation betting each round.
# 
# Regardless of their cards, everyone is staking the same amount every round ($100)
# 
# This is positve EV against 5 players!
# 
# This is equivalent to a backtest - clearly this is positive EV!  Let's just "trade this signal" every time!


import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

def get_positive_normal_bet(mean=100, std=40):
    # Draw from a normal distribution until a positive value is obtained (like conservative version)
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

def simulate_pocket_aces_6max_anim(num_games=200):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    # Deck minus hero's aces
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]

    net_outcomes = []
    cumulative_wealth = [0]
    win_losses = []  # 1 = win, 0 = tie, -1 = loss
    win_counts = [0]
    loss_counts = [0]
    tie_counts = [0]

    games_won = 0
    games_lost = 0
    games_tied = 0

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board = cards_needed[10:15]

        # Simulate three betting rounds, positive gaussian from normal distribution (like context, bilp 1-158)
        hero_total_investment = 0
        total_pot = 0
        for _ in range(3):
            bet = get_positive_normal_bet(mean=100, std=40)  # hero bets $100 ± 40, always positive
            hero_total_investment += bet
            total_pot += bet * 6  # 6 players (hero + 5 opps)

        try:
            hero_score = evaluator.evaluate(board, hero_hand)
            opp_scores = [evaluator.evaluate(board, hand) for hand in opp_hands]
        except KeyError as e:
            print(f"Error: { [Card.int_to_str(c) for c in board + hero_hand] }")
            raise e

        all_scores = [hero_score] + opp_scores
        min_score = min(all_scores)
        winners_count = all_scores.count(min_score)

        if hero_score == min_score:
            winnings = total_pot / winners_count
            if winners_count == 1:
                games_won += 1
                win_losses.append(1)
            else:
                games_tied += 1
                win_losses.append(0)
        else:
            winnings = 0
            games_lost += 1
            win_losses.append(-1)

        net_outcomes.append(winnings - hero_total_investment)
        new_wealth = cumulative_wealth[-1] + (winnings - hero_total_investment)
        cumulative_wealth.append(new_wealth)
        win_counts.append(games_won)
        loss_counts.append(games_lost)
        tie_counts.append(games_tied)

    frames = []
    x_path = np.arange(len(cumulative_wealth))
    num_wins = []
    num_losses = []
    num_ties = []

    for i in range(1, num_games+1):
        wins = win_losses[:i].count(1)
        ties = win_losses[:i].count(0)
        losses = win_losses[:i].count(-1)
        num_wins.append(wins)
        num_ties.append(ties)
        num_losses.append(losses)

    for t in range(1, num_games+1):
        sub_x = x_path[:t+1]
        sub_wealth = cumulative_wealth[:t+1]
        y_vals = [num_wins[t-1], num_ties[t-1], num_losses[t-1]]

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=sub_x, y=sub_wealth,
                    mode="lines",
                    line=dict(color="#00ffff", width=2),
                    name="Cumulative Net EV",
                    showlegend=False
                ),
                go.Bar(
                    x=["Win", "Tie", "Loss"],
                    y=y_vals,
                    marker_color=["#00ff00", "#ffc800", "#ff4060"],
                    name="Game Outcomes",
                    showlegend=False
                )
            ],
            layout=go.Layout(
                xaxis=dict(range=[0, num_games]),
                yaxis=dict(range=[min(cumulative_wealth)-300, max(cumulative_wealth)+300]),
                xaxis2=dict(range=[-0.6,2.6]),
                yaxis2=dict(range=[0, max(max(num_wins), max(num_losses), max(num_ties))+2])
            ),
            name=f"Game {t}"
        ))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cumulative Net Wealth (Hero EV)", "Game Outcomes (W/L/T)"),
        column_widths=[0.6, 0.4]
    )

    fig.add_trace(
        go.Scatter(
            x=[0,1],
            y=cumulative_wealth[:2],
            mode="lines",
            line=dict(color="#00ffff", width=2),
            name="Cumulative Net EV"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=["Win", "Tie", "Loss"],
            y=[num_wins[0], num_ties[0], num_losses[0]],
            marker_color=["#00ff00", "#ffc800", "#ff4060"],
            name="Game Outcomes"
        ),
        row=1, col=2
    )

    fig.frames = frames
    fig.update_layout(
        height=500, width=1000,
        title_text="Poker: Cumulative Backtest Wealth & Game Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        updatemenus=[{
            'type': 'buttons',
            'x': 0.5, 'y': -0.15,
            'xanchor': 'center', 'yanchor': 'top',
            'showactive': False,
            'buttons': [{
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 10, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0, 'easing': 'linear'}
                }]
            }]
        }]
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=2)

    fig.update_xaxes(title_text='Game Number', range=[0, num_games], row=1, col=1)
    fig.update_yaxes(title_text='Net Wealth', range=[min(cumulative_wealth)-300, max(cumulative_wealth)+300], row=1, col=1)
    fig.update_xaxes(title_text='Outcome', range=[-0.6,2.6], row=1, col=2)
    fig.update_yaxes(title_text='Count', range=[0, max(max(num_wins), max(num_losses), max(num_ties))+2], row=1, col=2)

    fig.show()
    return {
        "final_avg_ev": sum(net_outcomes)/num_games,
        "games_won": games_won,
        "games_lost": games_lost,
        "games_tied": games_tied
    }

np.random.seed(2)

if __name__ == "__main__":
    simulate_pocket_aces_6max_anim(200)


# ###### ______________________________________________________________________________________________________________________________________


# ##### Let's Look at our P/L Distribution for Each Game


import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

# Gaussian betting logic from main function (mean=100, std=40, 3 rounds)
def get_positive_normal_bet(mean=100, std=40):
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

def generate_gaussian_net_outcomes(num_games=10000):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    # Deck minus hero's aces
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]

    net_outcomes = []
    game_results = []  # 1 = win, 0 = tie, -1 = loss

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board = cards_needed[10:15]

        # Gaussian betting: sample three rounds per game
        hero_total_investment = 0
        total_pot = 0
        for _ in range(3):
            bet = get_positive_normal_bet(mean=100, std=40)
            hero_total_investment += bet
            total_pot += bet * 6  # 6 players

        try:
            hero_score = evaluator.evaluate(board, hero_hand)
            opp_scores = [evaluator.evaluate(board, hand) for hand in opp_hands]
        except KeyError as e:
            print(f"Error: { [Card.int_to_str(c) for c in board + hero_hand] }")
            raise e

        all_scores = [hero_score] + opp_scores
        min_score = min(all_scores)
        winners_count = all_scores.count(min_score)
        if hero_score == min_score:
            if winners_count == 1:
                game_results.append(1)  # win
            else:
                game_results.append(0)  # tie
            winnings = total_pot / winners_count
        else:
            game_results.append(-1)    # loss
            winnings = 0
        net_outcomes.append(winnings - hero_total_investment)
    return np.array(net_outcomes), np.array(game_results)

np.random.seed(2)
num_games = 10000
net_outcomes, game_results = generate_gaussian_net_outcomes(num_games)

# Separate wins and losses
wins = net_outcomes[game_results == 1]
losses = net_outcomes[game_results == -1]

# compute probabilities and EV
p_win = np.sum(game_results == 1) / num_games
p_loss = np.sum(game_results == -1) / num_games
p_tie = np.sum(game_results == 0) / num_games
EV = np.mean(net_outcomes)
EV_win = np.mean(wins) if wins.size > 0 else np.nan
EV_loss = np.mean(losses) if losses.size > 0 else np.nan

# Subplots: 2x1
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Net Wealth Distribution: Wins", 
        "Net Wealth Distribution: Losses",
        "Probability of Outcome", ""
    ),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "bar", "colspan": 2}, None]],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.13
)

# Histogram for wins (green)
if wins.size > 0:
    fig.add_trace(
        go.Histogram(
            x=wins,
            name='Wins',
            marker_color='rgba(0,255,96,0.85)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=EV_win, line_dash="dash", line_color="green",
        annotation_text=f"Win Mean: {EV_win:.1f}",
        annotation_position="top right", row=1, col=1
    )

# Histogram for losses (red)
if losses.size > 0:
    fig.add_trace(
        go.Histogram(
            x=losses,
            name='Losses',
            marker_color='rgba(255,64,64,0.87)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=2
    )
    fig.add_vline(
        x=EV_loss, line_dash="dash", line_color="red",
        annotation_text=f"Loss Mean: {EV_loss:.1f}",
        annotation_position="top right", row=1, col=2
    )

# (Removed overall/global EV lines! Only win/loss EV vertical lines above.)

# Probability bar chart (win/loss)
fig.add_trace(
    go.Bar(
        x=['Win', 'Loss'],
        y=[p_win, p_loss],
        marker_color=['green', 'red'],
        opacity=0.85,
        showlegend=False,
        text=[f'{p_win*100:.1f}%', f'{p_loss*100:.1f}%'],
        textposition='outside'
    ),
    row=2, col=1,
)

# Set up axes etc
fig.update_layout(
    height=680,
    width=1050,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text="Distributions of Net Wealth per Game (Wins vs Losses)"
)

fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=1)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=2)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=2)
fig.update_xaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1, title_text='Probability', range=[0,1])

fig.update_annotations(font_size=15)

fig.show()

# Print EV and probability values under the bar chart
from IPython.display import display, HTML
display(HTML(f"""
<div style="margin-top: 1.7em; padding:0.5em; background: #282832; color: #70ffff; font-size:1.4em; border-radius:9px;">
    <strong>Expected Value (EV) per Game:</strong> {EV:.2f}<br>
    <span style="color:#3f7;">Win Probability:</span> {p_win:.2%} &nbsp; | &nbsp;
    <span style="color:#f77;">Loss Probability:</span> {p_loss:.2%} &nbsp; | &nbsp;
    <span style="color:#bbb;">Tie Probability:</span> {p_tie:.2%}
</div>
"""))



# ##### We found a Profitable Strategy!  Let's Trade this in the Live Markets!


# ###### ______________________________________________________________________________________________________________________________________


# ##### Trading in the Live Markets with your Backtested Signal
# 
# 


import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

def get_positive_normal_bet(mean=100, std=40):
    # Draw from a normal distribution until a positive value is obtained (like bilp.ipynb 1-158)
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

def simulate_pocket_aces_adverse_anim(num_games=200):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    # All other cards besides hero's aces
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]

    net_outcomes = []
    cumulative_wealth = [0]
    win_losses = []  # 1 = win, 0 = tie, -1 = loss
    win_counts = [0]
    loss_counts = [0]
    tie_counts = [0]

    games_won = 0
    games_lost = 0
    games_tied = 0

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board = cards_needed[10:15]

        flop, turn, river = board[:3], board[:4], board[:5]

        total_street_pot = 0
        hero_total_investment = 0

        # BET-UP USING POSITIVE NORMAL ONLY -- like in file_context_0
        for street in [flop, turn, river]:
            hero_score = evaluator.evaluate(street, hero_hand)
            opp_scores = [evaluator.evaluate(street, hand) for hand in opp_hands]

            # BETTING LOGIC: If behind, "bet up" using a large normal, else use small normal
            if any(o_score < hero_score for o_score in opp_scores):
                street_bet = get_positive_normal_bet(mean=1000, std=400)  # Scale std large for "bet up"
            else:
                street_bet = get_positive_normal_bet(mean=100, std=40)    # Normal/Conservative bet

            hero_total_investment += street_bet
            total_street_pot += (street_bet * 6) # 6 players (hero + 5 opps)

        try:
            hero_score_final = evaluator.evaluate(board, hero_hand)
            opp_scores_final = [evaluator.evaluate(board, hand) for hand in opp_hands]
        except KeyError as e:
            print(f"Error: { [Card.int_to_str(c) for c in board + hero_hand] }")
            raise e

        all_scores = [hero_score_final] + opp_scores_final
        min_score = min(all_scores)
        winners_count = all_scores.count(min_score)

        if hero_score_final == min_score:
            winnings = total_street_pot / winners_count
            if winners_count == 1:
                games_won += 1
                win_losses.append(1)
            else:
                games_tied += 1
                win_losses.append(0)
        else:
            winnings = 0
            games_lost += 1
            win_losses.append(-1)

        net_outcomes.append(winnings - hero_total_investment)
        new_wealth = cumulative_wealth[-1] + (winnings - hero_total_investment)
        cumulative_wealth.append(new_wealth)
        win_counts.append(games_won)
        loss_counts.append(games_lost)
        tie_counts.append(games_tied)

    # Precompute animation values
    frames = []
    x_path = np.arange(len(cumulative_wealth))
    num_wins, num_ties, num_losses = [], [], []
    for i in range(1, num_games+1):
        num_wins.append(win_losses[:i].count(1))
        num_ties.append(win_losses[:i].count(0))
        num_losses.append(win_losses[:i].count(-1))

    wealth_min = min(cumulative_wealth) - 500
    wealth_max = max(cumulative_wealth) + 500
    count_max = max(max(num_wins), max(num_losses), max(num_ties)) + 2

    for t in range(1, num_games+1):
        sub_x = x_path[:t+1]
        sub_wealth = cumulative_wealth[:t+1]
        y_vals = [num_wins[t-1], num_ties[t-1], num_losses[t-1]]
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=sub_x, y=sub_wealth,
                    mode="lines",
                    line=dict(color="#00ffff", width=2),
                    name="Cumulative Net EV",
                    showlegend=False
                ),
                go.Bar(
                    x=["Win", "Tie", "Loss"],
                    y=y_vals,
                    marker_color=["#00ff00", "#ffc800", "#ff4060"],
                    name="Game Outcomes",
                    showlegend=False
                )
            ],
            layout=go.Layout(
                xaxis=dict(range=[0, num_games]),
                yaxis=dict(range=[wealth_min, wealth_max]),
                xaxis2=dict(range=[-0.6, 2.6]),
                yaxis2=dict(range=[0, count_max])
            ),
            name=f"Game {t}"
        ))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cumulative Net Wealth (Hero EV)", "Game Outcomes (W/L/T)"),
        column_widths=[0.6, 0.4]
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=cumulative_wealth[:2],
            mode="lines",
            line=dict(color="#00ffff", width=2),
            name="Cumulative Net EV"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=["Win", "Tie", "Loss"],
            y=[num_wins[0], num_ties[0], num_losses[0]],
            marker_color=["#00ff00", "#ffc800", "#ff4060"],
            name="Game Outcomes"
        ),
        row=1, col=2
    )
    fig.frames = frames
    fig.update_layout(
        height=500, width=1000,
        title_text="Poker: Adverse Dynamic Betting Simulation (AA vs 5 Random)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        updatemenus=[{
            'type': 'buttons',
            'x': 0.5, 'y': -0.15,
            'xanchor': 'center', 'yanchor': 'top',
            'showactive': False,
            'buttons': [{
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 10, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0, 'easing': 'linear'}
                }]
            }]
        }]
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=2)

    fig.update_xaxes(title_text='Game Number', range=[0, num_games], row=1, col=1)
    fig.update_yaxes(title_text='Net Wealth', range=[wealth_min, wealth_max], row=1, col=1)
    fig.update_xaxes(title_text='Outcome', range=[-0.6, 2.6], row=1, col=2)
    fig.update_yaxes(title_text='Count', range=[0, count_max], row=1, col=2)

    fig.show()
    return {
        "final_avg_ev": sum(net_outcomes)/num_games,
        "games_won": games_won,
        "games_lost": games_lost,
        "games_tied": games_tied
    }

np.random.seed(2)

if __name__ == "__main__":
    simulate_pocket_aces_adverse_anim(200)


# ###### ______________________________________________________________________________________________________________________________________


# ##### Let's Look at the Empirical P/L Distributions


import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

# Reuse the dynamic "BET UP" logic from the main function in file_context_0
def get_positive_normal_bet(mean=100, std=40):
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

def generate_betup_net_outcomes(num_games=10000):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    # Deck minus hero's aces
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]

    net_outcomes = []
    game_results = []  # 1 = win, 0 = tie, -1 = loss

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board = cards_needed[10:15]

        flop, turn, river = board[:3], board[:4], board[:5]
        streets = [flop, turn, river]

        hero_total_investment = 0
        total_street_pot = 0

        for street in streets:
            hero_score = evaluator.evaluate(street, hero_hand)
            opp_scores = [evaluator.evaluate(street, hand) for hand in opp_hands]

            # BET-UP LOGIC: If behind to anyone, bet large (mean=1000,std=400). If ahead, bet small (mean=100,std=40)
            if any(o_score < hero_score for o_score in opp_scores):
                street_bet = get_positive_normal_bet(mean=1000, std=400)
            else:
                street_bet = get_positive_normal_bet(mean=100, std=40)

            hero_total_investment += street_bet
            total_street_pot += (street_bet * 6) # 6 players

        try:
            hero_score_final = evaluator.evaluate(board, hero_hand)
            opp_scores_final = [evaluator.evaluate(board, hand) for hand in opp_hands]
        except KeyError as e:
            print(f"Error: { [Card.int_to_str(c) for c in board + hero_hand] }")
            raise e

        all_scores = [hero_score_final] + opp_scores_final
        min_score = min(all_scores)
        winners_count = all_scores.count(min_score)

        if hero_score_final == min_score:
            if winners_count == 1:
                game_results.append(1)  # win
            else:
                game_results.append(0)  # tie
            winnings = total_street_pot / winners_count
        else:
            game_results.append(-1)    # loss
            winnings = 0
        net_outcomes.append(winnings - hero_total_investment)

    return np.array(net_outcomes), np.array(game_results)

np.random.seed(2)
num_games = 10000
net_outcomes, game_results = generate_betup_net_outcomes(num_games)

# Separate wins and losses
wins = net_outcomes[game_results == 1]
losses = net_outcomes[game_results == -1]

# compute probabilities and EV
p_win = np.sum(game_results == 1) / num_games
p_loss = np.sum(game_results == -1) / num_games
p_tie = np.sum(game_results == 0) / num_games
EV = np.mean(net_outcomes)
EV_win = np.mean(wins) if wins.size > 0 else np.nan
EV_loss = np.mean(losses) if losses.size > 0 else np.nan

# Subplots: 2x1
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Net Wealth Distribution: Wins", 
        "Net Wealth Distribution: Losses",
        "Probability of Outcome", ""
    ),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "bar", "colspan": 2}, None]],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.13
)

# Histogram for wins (green)
if wins.size > 0:
    fig.add_trace(
        go.Histogram(
            x=wins,
            name='Wins',
            marker_color='rgba(0,255,96,0.85)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=EV_win, line_dash="dash", line_color="green",
        annotation_text=f"Win Mean: {EV_win:.1f}",
        annotation_position="top right", row=1, col=1
    )

# Histogram for losses (red)
if losses.size > 0:
    fig.add_trace(
        go.Histogram(
            x=losses,
            name='Losses',
            marker_color='rgba(255,64,64,0.87)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=2
    )
    fig.add_vline(
        x=EV_loss, line_dash="dash", line_color="red",
        annotation_text=f"Loss Mean: {EV_loss:.1f}",
        annotation_position="top right", row=1, col=2
    )

# Probability bar chart (win/loss)
fig.add_trace(
    go.Bar(
        x=['Win', 'Loss'],
        y=[p_win, p_loss],
        marker_color=['green', 'red'],
        opacity=0.85,
        showlegend=False,
        text=[f'{p_win*100:.1f}%', f'{p_loss*100:.1f}%'],
        textposition='outside'
    ),
    row=2, col=1,
)

# Set up axes etc
fig.update_layout(
    height=680,
    width=1050,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text="Distributions of Net Wealth per Game (Wins vs Losses, 'Bet Up' Logic)"
)

fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=1)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=2)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=2)
fig.update_xaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1, title_text='Probability', range=[0,1])

fig.update_annotations(font_size=15)

fig.show()

# Print EV and probability values under the bar chart
from IPython.display import display, HTML
display(HTML(f"""
<div style="margin-top: 1.7em; padding:0.5em; background: #282832; color: #70ffff; font-size:1.4em; border-radius:9px;">
    <strong>Expected Value (EV) per Game:</strong> {EV:.2f}<br>
    <span style="color:#3f7;">Win Probability:</span> {p_win:.2%} &nbsp; | &nbsp;
    <span style="color:#f77;">Loss Probability:</span> {p_loss:.2%} &nbsp; | &nbsp;
    <span style="color:#bbb;">Tie Probability:</span> {p_tie:.2%}
</div>
"""))



# WOAH!  You don't quite understand conditional statistics!  
# 
# What happens when someone "bets up" their hand - they're either bluffing or they got the nuts and we're gonna get killed!


# ###### ______________________________________________________________________________________________________________________________________


# So what do we need to do to make money playing poker?
# 
# Clearly information changes the overall distribution!


import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

# Seed for reproducibility
np.random.seed(2)

def get_equity(evaluator, hero_hand, board, full_deck, n_opponents=5, samples=50):
    """Estimate Hero's equity on the current street using Monte Carlo simulation."""
    wins = 0
    available_deck = [c for c in full_deck if c not in board]
    for _ in range(samples):
        random.shuffle(available_deck)
        cards_to_draw = 5 - len(board)
        complete_board = board + available_deck[:cards_to_draw]
        opp_hands = [available_deck[cards_to_draw + i*2 : cards_to_draw + (i+1)*2] for i in range(n_opponents)]
        h_score = evaluator.evaluate(complete_board, hero_hand)
        o_scores = [evaluator.evaluate(complete_board, hand) for hand in opp_hands]
        if h_score <= min(o_scores):
            wins += 1
    return wins / samples

def get_positive_normal_bet(mean=100, std=40):
    # Draw repeatedly until we get a positive value
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

def simulate_pocket_aces_conservative_anim(num_games=200):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]

    net_outcomes, cumulative_wealth, win_losses = [], [0], []
    games_won, games_lost, games_tied = 0, 0, 0

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands_final = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board_final = cards_needed[10:15]

        flop, turn, river = board_final[:3], board_final[:4], board_final[:5]
        total_street_pot, hero_total_investment, folded = 0, 0, False
        
        for street in [flop, turn, river]:
            hero_score = evaluator.evaluate(street, hero_hand)
            opp_scores_now = [evaluator.evaluate(street, hand) for hand in opp_hands_final]
            
            # Use normal distribution for bet amount
            if any(o < hero_score for o in opp_scores_now):
                bet_to_call = get_positive_normal_bet(mean=1000, std=40*10) # scale std for large bets
            else:
                bet_to_call = get_positive_normal_bet(mean=100, std=40)

            pot_odds = bet_to_call / (total_street_pot + bet_to_call * 6)
            equity = get_equity(evaluator, hero_hand, street, full_deck)
            
            # Conservative Strategy
            if equity < (pot_odds + 0.15):
                folded = True
                break
            else:
                hero_total_investment += bet_to_call
                total_street_pot += (bet_to_call * 6)

        if folded:
            winnings = 0
            games_lost += 1
            win_losses.append(-1)
        else:
            final_hero = evaluator.evaluate(board_final, hero_hand)
            final_opps = [evaluator.evaluate(board_final, h) for h in opp_hands_final]
            all_scores = [final_hero] + final_opps
            min_score = min(all_scores)
            winners_count = all_scores.count(min_score)

            if final_hero == min_score:
                winnings = total_street_pot / winners_count
                if winners_count == 1:
                    games_won += 1
                    win_losses.append(1)
                else:
                    games_tied += 1
                    win_losses.append(0)
            else:
                winnings = 0
                games_lost += 1
                win_losses.append(-1)

        net_outcomes.append(winnings - hero_total_investment)
        cumulative_wealth.append(cumulative_wealth[-1] + net_outcomes[-1])

    # Precompute animation frames
    x_path = np.arange(len(cumulative_wealth))
    num_wins = [win_losses[:i].count(1) for i in range(1, num_games+1)]
    num_ties = [win_losses[:i].count(0) for i in range(1, num_games+1)]
    num_losses = [win_losses[:i].count(-1) for i in range(1, num_games+1)]

    wealth_min, wealth_max = min(cumulative_wealth) - 500, max(cumulative_wealth) + 500
    count_max = max(max(num_wins), max(num_losses), max(num_ties)) + 2

    frames = [go.Frame(
        data=[
            go.Scatter(x=x_path[:t+1], y=cumulative_wealth[:t+1], mode="lines", line=dict(color="#00ffff", width=2)),
            go.Bar(x=["Win", "Tie", "Loss/Fold"], y=[num_wins[t-1], num_ties[t-1], num_losses[t-1]], marker_color=["#00ff00", "#ffc800", "#ff4060"])
        ],
        layout=go.Layout(
            yaxis=dict(range=[wealth_min, wealth_max]),
            yaxis2=dict(range=[0, count_max])
        ),
        name=f"Game {t}"
    ) for t in range(1, num_games+1)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cumulative Net Wealth", "Game Outcomes"), column_widths=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=[0, 1], y=cumulative_wealth[:2], mode="lines", line=dict(color="#00ffff", width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=["Win", "Tie", "Loss/Fold"], y=[num_wins[0], num_ties[0], num_losses[0]], marker_color=["#00ff00", "#ffc800", "#ff4060"]), row=1, col=2)

    fig.frames = frames
    fig.update_layout(
        height=500, width=1000, template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        showlegend=False,
        updatemenus=[{
            'type': 'buttons',
            'x': 0.5, 'y': -0.18,
            'xanchor': 'center', 'yanchor': 'top',
            'showactive': False,
            'bgcolor': 'rgba(0,0,0,0)',  # Transparent background
            'font': {'color': 'white'},    # White text
            'borderwidth': 1,
            'bordercolor': 'white',        # White border
            'buttons': [{
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 10, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0, 'easing': 'linear'}
                }]
            }]
        }]
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Game Number', range=[0, num_games], row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', range=[wealth_min, wealth_max], row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Outcome', range=[-0.6, 2.6], row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', range=[0, count_max], row=1, col=2)

    fig.show()

np.random.seed(2)

if __name__ == "__main__":
    simulate_pocket_aces_conservative_anim(200)


import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deuces import Card, Evaluator

# Use the same logic for positive gaussian bet as in the conservative main simulation
def get_positive_normal_bet(mean=100, std=40):
    bet = -1
    while bet <= 0:
        bet = np.random.normal(mean, std)
    return bet

# Estimate Hero's equity on a street by Monte Carlo
def get_equity(evaluator, hero_hand, board, full_deck, n_opponents=5, samples=50):
    wins = 0
    available_deck = [c for c in full_deck if c not in board]
    for _ in range(samples):
        random.shuffle(available_deck)
        cards_to_draw = 5 - len(board)
        complete_board = board + available_deck[:cards_to_draw]
        opp_hands = [available_deck[cards_to_draw + i*2 : cards_to_draw + (i+1)*2] for i in range(n_opponents)]
        h_score = evaluator.evaluate(complete_board, hero_hand)
        o_scores = [evaluator.evaluate(complete_board, hand) for hand in opp_hands]
        if h_score <= min(o_scores):
            wins += 1
    return wins / samples

def generate_conditional_fold_net_outcomes(num_games=10000, fold_buffer=0.15):
    evaluator = Evaluator()
    hero_hand = [Card.new('As'), Card.new('Ah')]
    # Deck minus hero's aces
    full_deck = [Card.new(r+s) for r in Card.STR_RANKS for s in 'shdc' if Card.new(r+s) not in hero_hand]
    net_outcomes = []
    game_results = []  # 1 = win, 0 = tie, -1 = loss/fold

    for step in range(num_games):
        random.shuffle(full_deck)
        cards_needed = full_deck[:15]
        opp_hands_final = [cards_needed[i*2:(i+1)*2] for i in range(5)]
        board_final = cards_needed[10:15]

        flop = board_final[:3]
        turn = board_final[:4]
        river = board_final[:5]
        total_street_pot = 0
        hero_total_investment = 0
        folded = False

        for street in [flop, turn, river]:
            hero_score = evaluator.evaluate(street, hero_hand)
            opp_scores_now = [evaluator.evaluate(street, hand) for hand in opp_hands_final]

            # Use cheaty but similar betting rule as the conservative sim
            if any(o < hero_score for o in opp_scores_now):
                bet_to_call = get_positive_normal_bet(mean=1000, std=400)
            else:
                bet_to_call = get_positive_normal_bet(mean=100, std=40)

            pot_odds = bet_to_call / (total_street_pot + bet_to_call * 6)
            equity = get_equity(evaluator, hero_hand, street, full_deck)
            # Conditional fold logic: fold if edge is not enough
            if equity < (pot_odds + fold_buffer):
                folded = True
                break
            hero_total_investment += bet_to_call
            total_street_pot += bet_to_call * 6

        if folded:
            winnings = 0
            game_results.append(-1)
            net_outcomes.append(-hero_total_investment)
        else:
            final_hero = evaluator.evaluate(board_final, hero_hand)
            final_opps = [evaluator.evaluate(board_final, h) for h in opp_hands_final]
            all_scores = [final_hero] + final_opps
            min_score = min(all_scores)
            winners_count = all_scores.count(min_score)
            if final_hero == min_score:
                winnings = total_street_pot / winners_count
                if winners_count == 1:
                    game_results.append(1)  # win
                else:
                    game_results.append(0)  # tie
            else:
                winnings = 0
                game_results.append(-1) # lost to better hand
            net_outcomes.append(winnings - hero_total_investment)
    return np.array(net_outcomes), np.array(game_results)

np.random.seed(2)
num_games = 10000
net_outcomes, game_results = generate_conditional_fold_net_outcomes(num_games)

# Separate wins, losses (includes folds!), ties
wins = net_outcomes[game_results == 1]
losses = net_outcomes[game_results == -1]
ties = net_outcomes[game_results == 0]

# compute probabilities and EV
p_win = np.sum(game_results == 1) / num_games
p_loss = np.sum(game_results == -1) / num_games
p_tie = np.sum(game_results == 0) / num_games
EV = np.mean(net_outcomes)
EV_win = np.mean(wins) if wins.size > 0 else np.nan
EV_loss = np.mean(losses) if losses.size > 0 else np.nan

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Net Wealth Distribution: Wins",
        "Net Wealth Distribution: Losses/Folds",
        "Probability of Outcome", ""
    ),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "bar", "colspan": 2}, None]],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.13
)

# Histogram for wins (green)
if wins.size > 0:
    fig.add_trace(
        go.Histogram(
            x=wins,
            name='Wins',
            marker_color='rgba(0,255,96,0.85)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=EV_win, line_dash="dash", line_color="green",
        annotation_text=f"Win Mean: {EV_win:.1f}",
        annotation_position="top right", row=1, col=1
    )

# Histogram for losses/folds (red)
if losses.size > 0:
    fig.add_trace(
        go.Histogram(
            x=losses,
            name='Losses/Folds',
            marker_color='rgba(255,64,64,0.87)',
            nbinsx=25,
            showlegend=False,
            opacity=0.88
        ),
        row=1, col=2
    )
    fig.add_vline(
        x=EV_loss, line_dash="dash", line_color="red",
        annotation_text=f"Loss/Fold Mean: {EV_loss:.1f}",
        annotation_position="top right", row=1, col=2
    )

# Probability bar chart (win/loss/fold, tie separated)
fig.add_trace(
    go.Bar(
        x=['Win', 'Tie', 'Loss/Fold'],
        y=[p_win, p_tie, p_loss],
        marker_color=['green', '#ffc800', 'red'],
        opacity=0.88,
        showlegend=False,
        text=[f'{p_win*100:.1f}%', f'{p_tie*100:.1f}%', f'{p_loss*100:.1f}%'],
        textposition='outside'
    ),
    row=2, col=1,
)

# Set up axes, layout
fig.update_layout(
    height=680,
    width=1050,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text="Net Wealth Distribution per Game: Conditional (Equity-based) Folding"
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=1)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Net Wealth', row=1, col=2)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', title_text='Count', row=1, col=2)
fig.update_xaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1, title_text='Probability', range=[0,1])

fig.update_annotations(font_size=15)

fig.show()

# Print EV and probability values under the bar chart
from IPython.display import display, HTML
display(HTML(f"""
<div style="margin-top: 1.7em; padding:0.5em; background: #282832; color: #70ffff; font-size:1.4em; border-radius:9px;">
    <strong>Expected Value (EV) per Game:</strong> {EV:.2f}<br>
    <span style="color:#3f7;">Win Probability:</span> {p_win:.2%} &nbsp; | &nbsp;
    <span style="color:#fc0;">Tie Probability:</span> {p_tie:.2%} &nbsp; | &nbsp;
    <span style="color:#f77;">Loss/Fold Probability:</span> {p_loss:.2%}
</div>
"""))



# ###### ______________________________________________________________________________________________________________________________________


# ##### This is Why your Backtests are Wrong
# 
# You can't treat a single sample path with structure if it doesn't exist.


import numpy as np
import plotly.graph_objects as go

# --- Simulate a single Brownian motion asset path ---
np.random.seed(77)
n_steps = 200
dt = 1 / 252
mu = 0.06
sigma = 0.21
S0 = 100

# --- Extend price series backward and forward for smoother MA/trade examples ---
n_extend_before = 24
n_extend_after = 24
n_sim = n_steps + n_extend_before + n_extend_after

Z = np.random.randn(n_sim)
increments = mu * dt + sigma * np.sqrt(dt) * Z
prices_all = S0 * np.exp(np.cumsum(increments))
times_all = np.arange(n_sim)

# The main plotted region corresponds to indices n_extend_before : n_extend_before + n_steps
main_start = n_extend_before
main_end = main_start + n_steps
prices = prices_all
times = times_all

# --- Moving average crossover strategy ---
fast_window = 6
slow_window = 18

fast_ma = np.convolve(prices, np.ones(fast_window)/fast_window, mode='same')
slow_ma = np.convolve(prices, np.ones(slow_window)/slow_window, mode='same')

# Find crossover signals: fast MA crosses above slow MA (bullish entry)
crosses = (fast_ma[1:] > slow_ma[1:]) & (fast_ma[:-1] <= slow_ma[:-1])
entry_indices = np.where(crosses)[0] + 1

# We'll select two such entries, one where the trade is a win, one a loss.
# For each, define exit as a fixed forward window
exit_offset = 18

def trade_result(idx):
    entry_price = prices[idx]
    exit_idx = min(idx + exit_offset, len(prices)-1)
    exit_price = prices[exit_idx]
    return exit_price - entry_price, idx, exit_idx

# Go through entries to find one win and one loss -- but within the main region
winning_entry = None
losing_entry = None
for idx in entry_indices:
    if idx < main_start or idx >= main_end:
        continue
    pnl, eidx, xidx = trade_result(idx)
    if pnl > 0 and winning_entry is None:
        winning_entry = (eidx, xidx, pnl)
    if pnl < 0 and losing_entry is None:
        losing_entry = (eidx, xidx, pnl)
    if winning_entry and losing_entry:
        break

if not (winning_entry and losing_entry):
    raise RuntimeError("Couldn't find both a win and a loss for the crossover trades.")

winner_idx, winner_exit, winner_pnl = winning_entry
loser_idx, loser_exit, loser_pnl = losing_entry

SLATE_OFF_CYAN = "#6fe7ff"  # Define color to replace SLATE_OFF_CYAN undefined variable

# --- Plot ---
fig_path = go.Figure()

# Price + MAs always shown (on extended region)
fig_path.add_trace(go.Scatter(
    x=times,
    y=prices,
    mode='lines',
    line=dict(color=SLATE_OFF_CYAN, width=3),
    hoverinfo='skip',
    showlegend=False
))
# Fast MA - orange, not dashed
fig_path.add_trace(go.Scatter(
    x=times,
    y=fast_ma,
    mode='lines',
    line=dict(color='#FF9500', width=2),
    showlegend=False,
    hoverinfo='skip'
))
# Slow MA - red, not dashed
fig_path.add_trace(go.Scatter(
    x=times,
    y=slow_ma,
    mode='lines',
    line=dict(color='#FF4136', width=2),
    showlegend=False,
    hoverinfo='skip'
))

# Win entry marker style: green star
win_entry_trace = go.Scatter(
    x=[winner_idx],
    y=[prices[winner_idx]],
    mode='markers+text',
    marker=dict(symbol='star', size=24, color='#44ff77', line=dict(color='white', width=2)),
    text=["Win"],
    textfont=dict(color='#fff', size=17),
    textposition="bottom center",
    showlegend=False,
    name='',  # no legend
)

# Loss entry marker style: big red circle
loss_entry_trace = go.Scatter(
    x=[loser_idx],
    y=[prices[loser_idx]],
    mode='markers+text',
    marker=dict(symbol='circle', size=25, color='#ff435e', line=dict(color='white', width=2)),
    text=["Loss"],
    textfont=dict(color='#fff', size=17),
    textposition="bottom center",
    showlegend=False,
    name='',  # no legend
)

# (Exit markers for clarity--but they are always visible and not toggled)
win_exit_trace = go.Scatter(
    x=[winner_exit],
    y=[prices[winner_exit]],
    mode='markers',
    marker=dict(symbol='circle', size=15, color='#1ded3a', line=dict(color='white', width=2)),
    showlegend=False,
    hoverinfo='skip',
    name='',
)
loss_exit_trace = go.Scatter(
    x=[loser_exit],
    y=[prices[loser_exit]],
    mode='markers',
    marker=dict(symbol='circle', size=15, color='#c23c00', line=dict(color='white', width=2)),
    showlegend=False,
    hoverinfo='skip',
    name='',
)

# Order of traces: price, fast_ma, slow_ma, win_entry, loss_entry, win_exit, loss_exit
fig_path.add_trace(win_entry_trace)
fig_path.add_trace(loss_entry_trace)
fig_path.add_trace(win_exit_trace)
fig_path.add_trace(loss_exit_trace)

# Only the win entry marker shown by default
fig_path.update_traces(visible=True, selector=0) # price
fig_path.update_traces(visible=True, selector=1) # fast_ma
fig_path.update_traces(visible=True, selector=2) # slow_ma
fig_path.update_traces(visible=True, selector=3) # win_entry_trace
fig_path.update_traces(visible=False, selector=4) # loss_entry_trace
fig_path.update_traces(visible=True, selector=5) # win_exit_trace
fig_path.update_traces(visible=True, selector=6) # loss_exit_trace

# Muted style for the selected tab background and text
active_bgcolor = 'rgba(90,150,180,0.19)'    # Muted blue
inactive_bgcolor = 'rgba(50,70,80,0.91)'    # Slightly dark
active_fontcolor = '#edf6ff'
inactive_fontcolor = '#95adc4'

fig_path.update_layout(
    width=900,
    height=440,
    title=dict(
        text="Moving Average Crossover: Example Win and Loss Entry",
        font=dict(color='white')
    ),
    xaxis=dict(
        title="Time Step",
        color='white',
        showgrid=True,
        gridcolor='rgba(120,180,220,0.16)',
        tickfont=dict(color='white'),
        range=[main_start-10, main_end+10]
    ),
    yaxis=dict(
        title="Price",
        color='white',
        showgrid=True,
        gridcolor='rgba(120,180,220,0.10)',
        tickfont=dict(color='white')
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=15),
    margin=dict(l=45, r=30, t=55, b=40)
)

# Add button to toggle between win and loss entry
fig_path.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.50, y=1.08, xanchor="center", yanchor="top",
            showactive=True,
            buttons=[
                dict(
                    label="Show Win Entry",
                    method="update",
                    args=[
                        {'visible': [True, True, True, True, False, True, True]},
                        {"title": "Moving Average Crossover: Example Win Entry"}
                    ],
                    args2=[
                        {'visible': [True, True, True, False, True, True, True]},
                        {"title": "Moving Average Crossover: Example Loss Entry"}
                    ]
                ),
                dict(
                    label="Show Loss Entry",
                    method="update",
                    args=[
                        {'visible': [True, True, True, False, True, True, True]},
                        {"title": "Moving Average Crossover: Example Loss Entry"}
                    ],
                    args2=[
                        {'visible': [True, True, True, True, False, True, True]},
                        {"title": "Moving Average Crossover: Example Win Entry"}
                    ]
                ),
            ],
            bgcolor=inactive_bgcolor,
            # activebgcolor=active_bgcolor,   # <-- Remove this line! Not supported, causes ValueError.
            font=dict(color=inactive_fontcolor, size=14),
            # activefont=dict(color=active_fontcolor, size=14),  # <-- Remove this line! Not supported.
            borderwidth=0,
            pad=dict(r=7, t=8, l=7, b=7)
        )
    ]
)

fig_path.show()



# You can't just bet up pocket aces without considering all the other information that is **currently** right in front of you
# 
# In the markets this means the current
# - Macro climate
# - Volatility regime
# - Volume, spread, . . .
# 
# The list goes on!


# ---


# #### 2.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# 
# - Pocket aces is like an entry trading signal, just because that entry and the way you were betting on it in the past implied positive expected value it doesn't imply this in a forward looking setting as you will be dealing with new information
# - A profitable strategy is not a *fixed set of rules* but a dynamic one that adapts to different climates to acheive an edge, this edge can take advantage of structural inefficiencies on a large scale using alternative data or even at a higher frequency in the microstructure sense
# - Not conditioning on the appropriate information and treating all entries as the same signal producing the same P/L distribution is statistically incorrect!  You are operating on different filtrations at each entry signal, and you need
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - My First Quant Resume
# - Projects that Made me a Quant
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Quant Roadmap: How I would Study if I Had to Start Over
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Reinforcement Learning for Delta Hedging
# - Approximating Pricing Functionals using Neural Networks
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Market Sentiment Trading System ([discourses.io](https://discourses.io))


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

