# ## How to Trade with an Edge
# 
# ##### ▶️ Related Quant Guild Videos:
# 
#  - [Why Quant Traders Care About Pricing](https://youtu.be/s0lVvYMA5OA)
# 
#  - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
#  - [How to Trade](https://youtu.be/NqOj__PaMec)
# 
#  - [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4)
# 
# - [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)
# 
#  
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# ##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)
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


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample data
np.random.seed(42)
n_stocks = 1000
signal = np.random.normal(0, 1, n_stocks)
returns = 0.01 * signal + np.random.normal(0, 0.05, n_stocks)  # Reduced return magnitude

# Calculate quantiles
n_quantiles = 5
df = pd.DataFrame({'signal': signal, 'returns': returns})
df['quantile'] = pd.qcut(df['signal'], n_quantiles, labels=False)

# Calculate mean returns by quantile
quantile_returns = df.groupby('quantile')['returns'].mean()

# Generate daily returns for multiple portfolio simulations
n_days = 50
n_simulations = 100
initial_value = 100
all_portfolio_values = np.zeros((n_simulations, n_days + 1))
all_portfolio_values[:,0] = initial_value

# Get return distributions for top and bottom quantiles
top_quantile_returns = df[df['quantile'] == n_quantiles-1]['returns']
bottom_quantile_returns = df[df['quantile'] == 0]['returns']

# Simulate multiple portfolio paths
for sim in range(n_simulations):
    for day in range(n_days):
        long_return = np.random.choice(top_quantile_returns)
        short_return = np.random.choice(bottom_quantile_returns)
        daily_return = long_return - short_return
        all_portfolio_values[sim, day + 1] = all_portfolio_values[sim, day] * (1 + daily_return)

# Calculate average path
average_portfolio_values = np.mean(all_portfolio_values, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Average Returns by Signal Quantile',
                                  'Long-Short Portfolio Value Paths'))

# Plot quantile returns as bar chart
fig.add_trace(
    go.Bar(x=list(range(n_quantiles)), 
           y=quantile_returns,
           name='Quantile Returns',
           marker_color='#00FFFF'),  # Changed to neon blue
    row=1, col=1
)

# Plot all portfolio paths (semi-transparent)
for sim in range(n_simulations):
    fig.add_trace(
        go.Scatter(x=np.arange(n_days + 1), 
                  y=all_portfolio_values[sim],
                  name='Simulation Path',
                  line=dict(color='rgba(0,255,0,0.1)', width=1),
                  showlegend=False),
        row=1, col=2
    )

# Plot average portfolio path
fig.add_trace(
    go.Scatter(x=np.arange(n_days + 1), 
               y=average_portfolio_values,
               name='Average Portfolio Value',
               line=dict(color='#00FF00', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    height=500,
    title_text='Signal-Based Trading Strategy',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Signal Quantile', row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Average Returns', row=1, col=1)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Days', row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Portfolio Value ($)', row=1, col=2)

fig.show()


# ### Sections
# 
# 1.) 📐 Defining *Edge*
# 
# - Game of chance example
# 
# - Game of optimal decision Making in the face of uncertainty
# 
# 2.) 🧮 Quantitative Edge
# 
# - Trading signals
# 
# 3.) 📖 Qualitative Edge
# 
# - Discretionary decision making in the face of uncertainty
# 
# 4.) 📚 Why learning is the key
# 
# - Anecdotal story about 99% of traders
# 
# - The key to operating in this space
# 
# 5.) 💭 Closing Thoughts and Future Topics


# ---


# ### 1.) 📐 Defining *Edge*
# 
# Trading is about making *optimal decisions in the face of uncertainty*, it is **NOT** a game of chance like roulette.
# 
# There is no such thing as *price prediction* without insider information - the best we can do is predict some level, an *expected value*.
# 
# Why is the expected value so important?  Our actions and inactions directly impact our expected value and P/L.
# 
# **Two Key Ideas:**
# 
# - Expected value of one outcome, one random variable
# 
# - Expected value of our *wealth path*, a collection of random variables representing our wealth of playing a game multiple times
# 
# #### 🎰 Example: Roulette 
# 
# Roulette's outcome is purely random action can outpreform *strategic* betting since its a game of chance.  There is no strategy except don't play that game!
# 
# By the law of total expectation:
# 
#  $EV_{Agent} = P(Black) \cdot Payout_{Black} + P(Red) \cdot Payout_{Red} + P(Green) \cdot Payout_{Green}$
#  
#  $EV_{Agent} = \frac{18}{38} \cdot 1 + \frac{18}{38} \cdot 1 + \frac{2}{38} \cdot 17$
#  
#  $EV_{Agent} = \frac{18}{38} + \frac{18}{38} + \frac{34}{38}$
#  
#  $EV_{Agent} = \frac{70}{38} - 1$
#  
#  $EV_{Agent} = -\frac{2}{38} \approx -5.26\%$ per bet
# 
# 
# **Agent 1:** Bet randomly on Black, Red, or Green with uniform probability (bet randomly)
# 
# **Agent 2:** Bet on Black, then red, then green (bet "strategically")


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Roulette parameters
n_iterations = 1000
n_paths = 100
bet_size = 0.1  # 1% of bankroll
initial_bankroll = 100

# Payout structure
payouts = {
    'black': 1,   # Bet 1, win 1
    'red': 1,     # Bet 1, win 1
    'green': 17   # Bet 1, win 17
}
probabilities = {
    'black': 18/38,  # 18 black numbers
    'red': 18/38,    # 18 red numbers
    'green': 2/38    # 0 and 00
}

def simulate_roulette(strategy='random'):
    paths = np.zeros((n_paths, n_iterations + 1))
    paths[:, 0] = initial_bankroll
    
    for path in range(n_paths):
        bankroll = initial_bankroll
        for i in range(n_iterations):
            bet_amount = bankroll * bet_size
            
            if strategy == 'random':
                bet = np.random.choice(['black', 'red', 'green'])
            else:  # sequential
                bet = ['black', 'red', 'green'][i % 3]
                
            # Simulate spin outcome
            if np.random.random() < probabilities[bet]:
                bankroll += bet_amount * payouts[bet]
            else:
                bankroll -= bet_amount
                
            paths[path, i + 1] = bankroll
            
    return paths

# Generate paths for both strategies
random_paths = simulate_roulette('random')
sequential_paths = simulate_roulette('sequential')

# Calculate mean paths
random_mean = np.mean(random_paths, axis=0)
sequential_mean = np.mean(sequential_paths, axis=0)

# Calculate y-axis range based on mean paths
y_min = min(np.min(random_mean), np.min(sequential_mean)) * 0.9
y_max = max(np.max(random_mean), np.max(sequential_mean)) * 1.1

# Create figure with side by side subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Random Betting Strategy (Uniform Probability)',
                                  'Sequential Betting Strategy (Black→Red→Green)'))

# Plot individual paths for random strategy with lower opacity
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_iterations + 1), y=random_paths[i],
                   line=dict(color='#00FFFF', width=1),
                   opacity=0.05,
                   showlegend=False),
        row=1, col=1
    )

# Plot individual paths for sequential strategy with lower opacity
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_iterations + 1), y=sequential_paths[i],
                   line=dict(color='#FF69B4', width=1),
                   opacity=0.05,
                   showlegend=False),
        row=1, col=2
    )

# Plot mean paths with higher opacity and width
fig.add_trace(
    go.Scatter(x=np.arange(n_iterations + 1), y=random_mean,
               name='Random Mean',
               line=dict(color='#00FFFF', width=4)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(n_iterations + 1), y=sequential_mean,
               name='Sequential Mean',
               line=dict(color='#FF69B4', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=500,
    title_text='Roulette Betting Strategies - Ensemble Average',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes with focused y-range
for i in [1, 2]:
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     title_text='Iterations',
                     row=1, col=i)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     title_text='Bankroll ($)',
                     range=[y_min, y_max],
                     row=1, col=i)

fig.show()


# **Remark:** For folks curious about statements related to ergodicity and ensemble v. time-average stay tuned! I will create a video on this topic in this context at length - *yes* sometimes positive ensemble EV can lead to negative EV in the time average sense.


# ##### 🎲 Example: Dice Roll Game
# 
# We have an action $\pi$ we can choose to transact (play the game) at a given market price.
# 
# $\mathbb{E}[X] = \frac{1+2+3+4+5+6}{6} = 3.5$ be the expected value of a fair dice roll.
#  
# We will receive a quote to play this game and will be able to decide to transact or not.
# 
# **Bid:** The price we can *sell* at
# 
# **Ask/Offer:** The price we can *buy* at
# 
# ##### Edge in a Dice Roll Market:
# 
# - Bid > 3.5 $\implies$ Edge
# 
# - Ask/Offer < 3.5 $\implies$ Edge
# 
# *Quote:* 
# 
# Let $u \sim U(0, 1)$
# 
# **Bid:** $\mathbb{E}[X] - .5 + u$
# 
# **Ask:** $\mathbb{E}[X] + .5 + u$


# Dice game parameters
n_iterations = 1000
n_paths = 100
initial_bankroll = 100
bet_size = 0.01  # 10% of bankroll

def simulate_dice_game(strategy='random'):
    paths = np.zeros((n_paths, n_iterations + 1))
    paths[:, 0] = initial_bankroll
    
    for path in range(n_paths):
        bankroll = initial_bankroll
        for i in range(n_iterations):
            # Generate quote
            u = np.random.uniform(0, 1)
            bid = 3.5 - 0.5 + u  # Price we can sell at
            ask = 3.5 + 0.5 + u  # Price we can buy at
            
            # Determine action based on strategy
            if strategy == 'random':
                # Randomly choose to buy, sell or pass
                action = np.random.choice(['buy', 'sell', 'pass'])
            else:  # smart
                # Only trade when there's edge
                dice_roll = np.random.randint(1, 7)
                expected_value = 3.5
                
                if bid > expected_value:
                    action = 'sell'  # Sell when bid is above EV
                elif ask < expected_value:
                    action = 'buy'   # Buy when ask is below EV
                else:
                    action = 'pass'  # Pass when no edge
            
            # Execute trade
            bet_amount = bankroll * bet_size
            if action == 'buy':
                dice_roll = np.random.randint(1, 7)
                pnl = dice_roll - ask
                bankroll += bet_amount * pnl
            elif action == 'sell':
                dice_roll = np.random.randint(1, 7)
                pnl = bid - dice_roll
                bankroll += bet_amount * pnl
                
            paths[path, i + 1] = bankroll
            
    return paths

# Generate paths for both strategies
random_paths = simulate_dice_game('random')
smart_paths = simulate_dice_game('smart')

# Calculate mean paths
random_mean = np.mean(random_paths, axis=0)
smart_mean = np.mean(smart_paths, axis=0)

# Calculate y-axis range for right plot only
y_min = np.min(smart_mean) * 0.9
y_max = np.max(smart_mean) * 1.1

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Random Trading Strategy',
                                  'Edge-Based Trading Strategy'))

# Plot individual paths for random strategy
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_iterations + 1), y=random_paths[i],
                   line=dict(color='#FF0000', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot individual paths for smart strategy
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_iterations + 1), y=smart_paths[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot mean paths
fig.add_trace(
    go.Scatter(x=np.arange(n_iterations + 1), y=random_mean,
               name='Random Mean',
               line=dict(color='#FF0000', width=4)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(n_iterations + 1), y=smart_mean,
               name='Edge-Based Mean',
               line=dict(color='#00FF00', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=500,
    title_text='Dice Game Trading Strategies - Ensemble Average',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Iterations',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Bankroll ($)',
                 row=1, col=1)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Iterations',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Bankroll ($)',
                 range=[y_min, y_max],
                 row=1, col=2)

fig.show()


# **Remark:** This is quite similar to the notion of algorithmic market-making


# ---


# ### 2.) 🧮 Quantitative Edge
# 
# In real life, we don't have a static level like $3.5$ in the dice roll example above.  
# 
# Instead we have to proxy for some quantity in a statistical capacity and aim to make these optimal decisions relative to this methodology.
# 
# It's important to note
# 
# - Static arbitrage does exist and we can make a risk-free profit if we are fast enough (measures like Sharpe here are silly here and really just proxy for risk of failure or getting stuck in the trade etc.)
# 
# - There are wild systematic inefficiencies that can be exploited in a quantitative capacity with sufficient data - but scaling a strategy will reduce its viability, a common problem in the space for institutions but not retailers
# 
# - Our statistical edge from a systematic inefficiency, or strategy we develop vanishes


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample data
np.random.seed(42)
n_stocks = 1000
signal = np.random.normal(0, 1, n_stocks)
returns = 0.01 * signal + np.random.normal(0, 0.05, n_stocks)  # Reduced return magnitude

# Calculate quantiles
n_quantiles = 5
df = pd.DataFrame({'signal': signal, 'returns': returns})
df['quantile'] = pd.qcut(df['signal'], n_quantiles, labels=False)

# Calculate mean returns by quantile
quantile_returns = df.groupby('quantile')['returns'].mean()

# Generate daily returns for multiple portfolio simulations
n_days = 50
n_simulations = 100
initial_value = 100
all_portfolio_values = np.zeros((n_simulations, n_days + 1))
all_portfolio_values[:,0] = initial_value

# Get return distributions for top and bottom quantiles
top_quantile_returns = df[df['quantile'] == n_quantiles-1]['returns']
bottom_quantile_returns = df[df['quantile'] == 0]['returns']

# Simulate multiple portfolio paths
for sim in range(n_simulations):
    for day in range(n_days):
        long_return = np.random.choice(top_quantile_returns)
        short_return = np.random.choice(bottom_quantile_returns)
        daily_return = long_return - short_return
        all_portfolio_values[sim, day + 1] = all_portfolio_values[sim, day] * (1 + daily_return)

# Calculate average path
average_portfolio_values = np.mean(all_portfolio_values, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Average Returns by Signal Quantile',
                                  'Long-Short Portfolio Value Paths'))

# Plot quantile returns as bar chart
fig.add_trace(
    go.Bar(x=list(range(n_quantiles)), 
           y=quantile_returns,
           name='Quantile Returns',
           marker_color='#00FFFF'),  # Changed to neon blue
    row=1, col=1
)

# Plot all portfolio paths (semi-transparent)
for sim in range(n_simulations):
    fig.add_trace(
        go.Scatter(x=np.arange(n_days + 1), 
                  y=all_portfolio_values[sim],
                  name='Simulation Path',
                  line=dict(color='rgba(0,255,0,0.1)', width=1),
                  showlegend=False),
        row=1, col=2
    )

# Plot average portfolio path
fig.add_trace(
    go.Scatter(x=np.arange(n_days + 1), 
               y=average_portfolio_values,
               name='Average Portfolio Value',
               line=dict(color='#00FF00', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    height=500,
    title_text='Signal-Based Trading Strategy',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Signal Quantile', row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Average Returns', row=1, col=1)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Days', row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Portfolio Value ($)', row=1, col=2)

fig.show()


# ---


# ### 3.) 📖 Qualitative Edge
# 
# The efficacy of qualitative approaches comes under fire quickly. 
# 
# They don't have as much structure as quantitative method and for that reason can lack algorithmic implementation to the same discretionary high-touch capacity.
# 
# **Qualitative Edge comes from Information Advantage:**
# 
# - AI researcher understands and has a perspective on the current limitations of AI and takes a position based on their views
# 
# - An MD understands the current drug-approval landscape and doesn't see a company's product getting approved
# 
# - A volatility trader has seen this boat rocked 10,000 + 1 times and can't see vol higher than it is now
# 
# This is NOT insider information.
# 
# There can be a statistical basis for their decision making, or it can be purely discretionary (think implied probability).
# 
# Maybe there is some underlying statistical structure, maybe there isn't - these sample paths only play out *one* time anyway (think elections, we can bet on the outcome, maybe the implied probability suggests something about where most of the money lies, but we can't repeat this experiment 100,000 times and see if its a reasonable proxy - just like with financial derivatives and implied volatility).
# 
# ##### 📅 Example: Tariffs Early 2025, Increasing Volatility


import pandas as pd

df = pd.read_csv('spy_vix_ytd_data.csv')

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('SPY Price Action',
                                  'VIX Price Action'))

# Plot SPY data
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['SPY'],
               name='SPY',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot VIX data  
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['VIX'],
               name='VIX',
               line=dict(color='#FF0000', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=500,
    title_text='SPY vs VIX YTD',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Date',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Price ($)',
                 row=1, col=1)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Date',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='VIX Level',
                 row=1, col=2)

fig.show()


# *Qualitatively, I know that fear is extremely mispriced*
# 
# **Two Strategies:**
# 
# - 💰 Lump sum dump at vol threshold
# 
# - 📉 Buy the Way Down


# **💰 Lump Sum Strategy:**


# Set VIX threshold for investment
VIX_THRESHOLD = 30

# Create portfolio value series
portfolio = pd.Series(index=df.index, dtype=float)
portfolio.iloc[0] = 100000  # Initial $100k investment

# Find first entry point where VIX crosses threshold
entry_point = df[df['VIX'] >= VIX_THRESHOLD].index[0]

# Calculate returns after entry
spy_returns = df['SPY'].pct_change().astype(float)  # Explicitly cast to float
portfolio_returns = pd.Series(0.0, index=df.index)  # Initialize with float
portfolio_returns.loc[entry_point:] = spy_returns.loc[entry_point:]  # Use .loc for assignment

# Calculate cumulative portfolio value
portfolio = (1 + portfolio_returns).cumprod() * 100000

# Plot portfolio value
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df['Date'], y=portfolio,
               name='Portfolio Value',
               line=dict(color='#00FF00', width=4))
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=500,
    title_text=f'Portfolio Value (Entry at VIX > {VIX_THRESHOLD})',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Date')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Portfolio Value ($)')

fig.show()


# **📉 Buy the Way Down:**


# Set initial cash and portfolio values
initial_cash = 100000
cash = pd.Series(initial_cash, index=df.index)
portfolio_value = pd.Series(0.0, index=df.index)

# Calculate SPY returns
spy_returns = df['SPY'].pct_change().astype(float)

# Iterate through dates after first VIX threshold crossing
for i in range(len(df)):
    if i == 0:
        continue
        
    # Update portfolio value based on previous investments
    if i > 0:
        portfolio_value[i] = portfolio_value[i-1] * (1 + spy_returns[i])
    
    # Check if VIX is above threshold
    if df['VIX'][i] >= VIX_THRESHOLD:
        # Calculate investment amount (10% of remaining cash)
        investment = cash[i-1] * 0.1
        
        # Update cash and portfolio
        cash[i] = cash[i-1] - investment
        portfolio_value[i] += investment
    else:
        cash[i] = cash[i-1]

# Calculate total value (cash + portfolio)
total_value = cash + portfolio_value

# Create subplot figure
fig = make_subplots(rows=1, cols=2, subplot_titles=('Account Equity', 'Portfolio Composition'))

# First subplot - Total Value only
fig.add_trace(
    go.Scatter(x=df['Date'], y=total_value,
               name='Total Value',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Second subplot - Cash and Portfolio breakdown
fig.add_trace(
    go.Scatter(x=df['Date'], y=total_value,
               name='Total Value',
               line=dict(color='#00FF00', width=4)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=df['Date'], y=cash,
               name='Cash',
               line=dict(color='#0000FF', width=2)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=df['Date'], y=portfolio_value,
               name='Portfolio Value',
               line=dict(color='#FF0000', width=2)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=500,
    width=1200,
    title_text=f'Portfolio Analysis (10% Investment at VIX > {VIX_THRESHOLD})',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Date')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Value ($)')

fig.show()


# **Remark:** This is *one* sample path, this strategy will work in the sense that we assume volatility is mean reverting (we can't directly observe volatility), and though the leverage effect implies recovery occurs dispropritiantely to loss, we still assume it will recover when volatility drops because when everything calms down there's no reason to misprice.  
# 
# Yes, other factors are relevant (inflation, interest rates, jobs, policy, etc.) but we can say that about pretty much every strategy $\dots$ something to keep in mind, remember this is qualitative - we can try to impose structure in a quantitative capacity and derive some sort of statistical edge to execute algorithmically but that changes the approach entirely. 


# ---


# ### 4.) 📚 Why Learning is the Key
# 
# **YOU** are the key to this problem, not someone else or some money printing trading system
# 
# - News wants your attention
# 
# - Brokers want your commision
# 
# - Learning is the only way to make optimal decisions in the face of uncertainty
# 
# No good poker player is going to sit next to you at the poker table and tell you what to do
# 
# *What you shouldn't do:*
# 
# - Buy/sell irrationally (every up-tick is to the moon, every down-tick is the apocolypse)
# 
# **📖 An Anecdotal Story:**
# 
# - I train and compete in Brazilian Jiu-Jitsu, when I was in the locker room after training one of my training partners (call him John) walked in with 2 of my friends.  John started lecturing us that he spent the past year learning to trade and lost $2,000 on DOGE coin.  My friends know I'm a quant, John did not, but I was curious what he meant by "learn".  I casually asked
# 
# - What kind of strategies did you try?  
# 
# - What was the macro regime?  
# 
# - What is the *supply* of DOGE?  
# 
# - What is a cryptocurrency? 
# 
# He couldn't answer *any* of these questions.  He was liquidity for more informed traders, and was paying commision to brokers for no reason - he had no edge.
# 
# *Analogous Qestion:* Would you prefer to bank roll a Poker player on the tour or someone just getting into the tournament scene?
# 
# *What you should do*
# 
# - Master your quantitative skills and learn how markets function so you can make your own optimal decisions in the face of uncertainty


# ---


# ### 5.) 💭 Closing Thoughts and Future Topics
# 
# It's important to fully understand the idea of edge and optimal action in the face of uncertainty in a closed (quasi-closed) environment first - if you can't make money in a dice market with perfect information about the EV or even partial information about the EV how can we hope to apply these ideas and make money in a live market?
# 
# Challenges:
# 
# - These mean levels don't exist in the static sense (Video: [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA))
# 
# - It's difficult to gauge if we are on a *bad sample path* or if our strategy (quantitative or qualitative) is bad.  
# 
# I've seem some *guru* traders cite the central limit theorem and that your strategy "if it's good" will converge to positive EV - so incorrect!
# 
# - Moreover, it's difficult to gauge if your strategy *had* edge but now its at best breakeven or its losing money (sample path or time to retire the system)
# 
# Future Topics:
# 
# - Building systems for executing strategies in this capacity (edge hunting, not set and forget)
# 
# - The efficacy of time series approaches for trading and making "predictions"
# 
# - Factor models and classical literature on cross-sectional return predictions
# 
# - The difference between retail and institutional trading (Sell v. Buy Side)


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

