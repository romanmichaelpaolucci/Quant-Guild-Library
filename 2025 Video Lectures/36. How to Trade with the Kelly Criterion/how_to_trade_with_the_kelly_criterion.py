# ## How to Trade with the Kelly Criterion
# 
# #### How Much Should We Bet and Why Bet Sizing is Important
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
# - [How to Trade](https://youtu.be/NqOj__PaMec)
# 
# - [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Kelly system parameters
win_rate = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses
odds = win_pct/loss_pct  # Net odds ratio
kelly_fraction = (odds * win_rate - (1-win_rate)) / odds  # Kelly criterion formula

# Non-ergodic system parameters (multiplicative returns) 
win_rate_pct = 0.51
bet_pct = 0.75  # Percentage of capital bet size

# Simulate Kelly criterion wealth paths
paths_kelly = np.zeros((n_paths, n_trades + 1))
paths_kelly[:, 0] = initial_capital

# Simulate non-ergodic wealth paths
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital

for path in range(n_paths):
    # Kelly system simulation
    capital_kelly = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Kelly system trades
        is_win = np.random.random() < win_rate
        bet_amount = capital_kelly * kelly_fraction
        
        if is_win:
            capital_kelly += bet_amount * win_pct
        else:
            capital_kelly -= bet_amount * loss_pct
        paths_kelly[path, i + 1] = capital_kelly
        
        # Non-ergodic system trades
        is_win_pct = np.random.random() < win_rate_pct
        bet_amount = capital_nonergodic * bet_pct
        
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_kelly = np.mean(paths_kelly, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Kelly Criterion Paths (f*={kelly_fraction:.2%})',
                                  'Non-Ergodic Wealth Paths'))

# Plot Kelly system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_kelly[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot Kelly system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_kelly,
               name='Mean Path (Kelly)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1000,
    height=500,
    title_text='Trading System Analysis: Kelly Criterion vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()



# ### Sections
# 
# ##### 1.) 💻 Expected Value of a Trading System
# 
# - Wealth after a single trade
# 
# - Wealth after a series of trades
# 
# - Ergodicity
# 
# ##### 2.) 🧮 Deriving the Kelly Criterion 
# 
# - Optimizing expected value in the face of uncertainty
# 
# - The solution to maximizing wealth after a series of trades: Kelly Criterion
# 
# ##### 3.) 📈 Trading with the Kelly Criterion
# 
# - Dynamic bet sizes v. static bet sizes 
# 
# - Experienced wealth with Kelly Criterion
# 
# ##### 4.) ⚔️ Challenges and Limitations
# 
# - What is the actual expected value of your trading system: Regime Based Trading
# 
# - Decomposed values change over time and do not converge
# 
# ##### 5.) 💭 Closing Thoughts and Future Topics


# ---


# ### 1.) 💻 Expected Value of a Trading System
# 
# ##### Suppose we have a trading system $\Tau$ that governs each trade defined by a net zero position from entry and exit generating a P/L
# 
# ##### Trades in that system can be denoted as $\tau \in \Tau$ where each $\tau_1, \tau_2, \dots$ corresponds to some P/L denoted $\pi$
# 
# ##### To assess the efficacy of our trading system in generating wealth over time we consider the expected value
# 
# ##### $$\mathbb{E}[\pi] = \mathbb{E}[\pi | W] P(W) + \mathbb{E}[\pi | L]P(L)$$
# 
# or in plain english
# 
# ##### $$\mathbb{E}[\text{P/L}] = \mathbb{E}[\text{P/L} | \text{Winning Trade}] P(\text{Winning Trade}) + \mathbb{E}[\text{P/L} | \text{Losing Trade}]P(\text{Losing Trade})$$
# 
# *Note: These values can be generated via a simple backtest* 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000
win_rate = 0.51
avg_win = 20
avg_loss = 20

# Generate win/loss amounts from normal distributions
win_amounts = np.random.normal(avg_win, avg_win/2, n_trades)
loss_amounts = np.random.normal(avg_loss, avg_loss/2, n_trades)

# Generate sample trades
trades = np.random.random(n_trades) < win_rate
pl_outcomes = np.where(trades, win_amounts, -loss_amounts)

# Simulate multiple wealth paths with small fixed bet size
paths_small = np.zeros((n_paths, n_trades + 1))
paths_small[:, 0] = initial_capital
small_bet = 10  # $10 fixed bet size

# Simulate multiple wealth paths with large fixed bet size
paths_large = np.zeros((n_paths, n_trades + 1))
paths_large[:, 0] = initial_capital
large_bet = 30  # $30 fixed bet size

for path in range(n_paths):
    capital_small = initial_capital
    capital_large = initial_capital
    
    for i in range(n_trades):
        # Generate trade outcomes
        is_win = np.random.random() < win_rate
        trade_amount = np.random.normal(avg_win, avg_win/2) if is_win else np.random.normal(avg_loss, avg_loss/2)
        
        # Small fixed bet size path
        trade_pl_small = small_bet * trade_amount if is_win else -small_bet * trade_amount
        capital_small += trade_pl_small
        paths_small[path, i + 1] = capital_small
        
        # Large fixed bet size path
        trade_pl_large = large_bet * trade_amount if is_win else -large_bet * trade_amount
        capital_large += trade_pl_large
        paths_large[path, i + 1] = capital_large

# Calculate mean paths
mean_path_small = np.mean(paths_small, axis=0)
mean_path_large = np.mean(paths_large, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('P/L Distribution',
                                  'Wealth Paths (Small Bet)',
                                  'Wealth Paths (Large Bet)'))

# Split P/L outcomes into wins and losses
wins = pl_outcomes[pl_outcomes >= 0]
losses = pl_outcomes[pl_outcomes < 0]

# Plot P/L distribution
fig.add_trace(
    go.Histogram(x=losses,
                 nbinsx=25,
                 name='Losses',
                 marker_color='#FF0000', opacity=.75),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(x=wins,
                 nbinsx=25,
                 name='Wins',
                 marker_color='#00FF00', opacity=.75),
    row=1, col=1
)

# Plot small bet size paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_small[i],
                   line=dict(color='#00FFFF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot small bet size mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_small,
               name='Mean Path (Small)',
               line=dict(color='#00FFFF', width=4)),
    row=1, col=2
)

# Plot large bet size paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_large[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=3
    )

# Plot large bet size mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_large,
               name='Mean Path (Large)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=3
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1500,
    height=500,
    title_text='Trading System Analysis: Small vs Large Fixed Bet Size',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='P/L',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=3)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Frequency',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=3)

fig.show()



# Calculate average final values for small and large bet systems
small_bet_final_avg = np.mean(paths_small[:,-1])
large_bet_final_avg = np.mean(paths_large[:,-1])

print(f"After {n_trades} trades:")
print(f"Small Bet System Average Value: ${small_bet_final_avg:.2f}")
print(f"Large Bet System Average Value: ${large_bet_final_avg:.2f}")



# ##### **Remark:** Here, each P/L outcome is a random variable associated with *one* trade, but we are interested in our wealth path over a *series* of trades which is a stochastic process.
# 
# 
# ##### Big Problem: In reality, we don't bet fixed amounts as seen above we bet a percentage of total capital or bankroll - image we had $1mln, we wouldn't bet in increments of $100
# 
# ##### $$\text{In reality, }\mathbb{E}[\pi] \text{ is NOT the same as } \mathbb{E}[W_T]$$


# #### Ergodicity
# 
# Formally, a system is ergodic if the time average is equivalent to its ensemble average 
# 
# ##### $$\lim_{T \to \infty} \frac{1}{T} \int_0^T x(t) dt = \int x P(x) dx$$
#  
# #### Impact of Ergodicity: 
# 
# ##### Both systems below have positive expected value $\mathbb{E}[\pi]$, but the betting dynamics determine the experience of any one sample path
#  
#  Ergodic System (Additive Returns):
#  $$\mathbb{E}[\pi] = 0.51 \cdot 20 - 0.49 \cdot 20 = 0.4\text{ dollars per trade}$$
#  $$\text{Time Value}  \approx \mathbb{E}[\pi]$$
# 
#  Non-Ergodic System (Multiplicative Returns): 
#  $$\mathbb{E}[\pi] = 0.51 \cdot 0.05 - 0.49 \cdot 0.05 = 0.001\text{ or }0.1\%\text{ per trade}$$
#  $$\text{Time Value}  \neq \mathbb{E}[\pi]$$
# 
# 


# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Ergodic system parameters (additive returns)
win_rate = 0.51
avg_win = 20  # Fixed dollar amount
avg_loss = 20  # Fixed dollar amount

# Non-ergodic system parameters (multiplicative returns)
win_rate_pct = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses

# Simulate ergodic wealth paths (additive returns)
paths_ergodic = np.zeros((n_paths, n_trades + 1))
paths_ergodic[:, 0] = initial_capital
bet_size = 20  # Fixed bet size

# Simulate non-ergodic wealth paths (multiplicative returns)
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital
bet_pct = 0.75  # Percentage of capital bet size

for path in range(n_paths):
    # Ergodic system simulation
    capital_ergodic = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Ergodic system trades (additive returns)
        is_win = np.random.random() < win_rate
        trade_amount = np.random.normal(avg_win, avg_win/2) if is_win else np.random.normal(avg_loss, avg_loss/2)
        
        trade_pl = bet_size * trade_amount if is_win else -bet_size * trade_amount
        capital_ergodic += trade_pl
        paths_ergodic[path, i + 1] = capital_ergodic
        
        # Non-ergodic system trades (multiplicative returns)
        is_win_pct = np.random.random() < win_rate_pct
        
        bet_amount = capital_nonergodic * bet_pct
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_ergodic = np.mean(paths_ergodic, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Ergodic Wealth Paths',
                                  'Non-Ergodic Wealth Paths'))

# Plot ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_ergodic[i],
                   line=dict(color='#00FFFF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_ergodic,
               name='Mean Path (Ergodic)',
               line=dict(color='#00FFFF', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Trading System Analysis: Ergodic vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()



# Calculate mean final wealth for both strategies
mean_final_wealth_ergodic = np.mean(paths_ergodic[:,-1])
mean_final_wealth_nonergodic = np.mean(paths_nonergodic[:,-1])

print(f"Mean final wealth after {n_trades} trades:")
print(f"Ergodic strategy: ${mean_final_wealth_ergodic:,.2f}")
print(f"Non-ergodic strategy: ${mean_final_wealth_nonergodic:,.2f}")



# Calculate probability of ending above initial capital
prob_above_initial_ergodic = np.mean(paths_ergodic[:,-1] > initial_capital)
prob_above_initial_nonergodic = np.mean(paths_nonergodic[:,-1] > initial_capital)

print(f"Probability of ending above ${initial_capital:,.0f} after {n_trades} trades:")
print(f"Ergodic strategy: {prob_above_initial_ergodic:.1%}")
print(f"Non-ergodic strategy: {prob_above_initial_nonergodic:.1%}")



# ##### 🌊 Flaw of Averages
# 
# One of the problems with only considering the mean, average, or expectation is we don't have a notion of spread and we might be dragging this estimate in one direction or the other
# 
# ##### ⚠️ Problem: Our trading system $T$ and associated P/L ($\tau \mapsto$ P/L) is not an ergodic system, EV, even if positive, is misleading about the wealth experienced


# ##### ❓ Q: If we operate in a non-ergodic system, what is the optimal bet to ensure the highest probability of experiencing a *good* wealth path?


# ---


# ### 2.) 🧮 Deriving the Kelly Criterion 
# 
# ##### Our goal isn't to maximize our wealth over a single trade $\tau$, but rather our wealth over a number of trades $W_n$ where $n \in \mathbb{N}$ denotes the trade number
# 
# ##### Each wealth path $W_n$ is a random variable, so to maximize this object we need to maximize its expected value
# 
# $$\text{max } \mathbb{E}[log(W_{n+1})]$$
# 
# ##### Maximizing the EV of the log wealth path corresponds to maximizing the geometric mean of returns over the arithmetic mean of return favoring long-term growth
# 
# 
# ##### ❓ Q: What is the solution to the optimization problem maximizing our log wealth? The Kelly Criterion:
# 
# - $W_n$ is current wealth
# 
# - $f$ is the fraction of your bankroll to place on each trade (the value we want to solve for)
# 
# - $b$ net profit per unit risked
# 
# - $p$ is the probability of winning the trade
# 
# - $q$ is the probability of losing the trade
# 
# ### $$\text{Kelly Criterion: }f^* = \frac{bp - q}{b}$$


# ---


# ### 3.) 📈 Trading with the Kelly Criterion
# 
# ##### Expected Value of Non-Ergodic System
# 
# Below is the same non-ergodic trading system as above but we will trade based on the Kelly Criterion and a fixed % of capital 
# 
# **Remark:** The trading system below has the same positive expected value, but different betting strategies


# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Kelly system parameters
win_rate = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses
odds = win_pct/loss_pct  # Net odds ratio
kelly_fraction = (odds * win_rate - (1-win_rate)) / odds  # Kelly criterion formula

# Non-ergodic system parameters (multiplicative returns) 
win_rate_pct = 0.51
bet_pct = 0.75  # Percentage of capital bet size

# Simulate Kelly criterion wealth paths
paths_kelly = np.zeros((n_paths, n_trades + 1))
paths_kelly[:, 0] = initial_capital

# Simulate non-ergodic wealth paths
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital

for path in range(n_paths):
    # Kelly system simulation
    capital_kelly = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Kelly system trades
        is_win = np.random.random() < win_rate
        bet_amount = capital_kelly * kelly_fraction
        
        if is_win:
            capital_kelly += bet_amount * win_pct
        else:
            capital_kelly -= bet_amount * loss_pct
        paths_kelly[path, i + 1] = capital_kelly
        
        # Non-ergodic system trades
        is_win_pct = np.random.random() < win_rate_pct
        bet_amount = capital_nonergodic * bet_pct
        
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_kelly = np.mean(paths_kelly, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Kelly Criterion Paths (f*={kelly_fraction:.2%})',
                                  'Non-Ergodic Wealth Paths'))

# Plot Kelly system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_kelly[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot Kelly system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_kelly,
               name='Mean Path (Kelly)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Trading System Analysis: Kelly Criterion vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()



# ##### Q: How does this impact our expected wealth and wealth experienced?


# Calculate probability of ending above initial capital
kelly_above = np.sum(paths_kelly[:, -1] > initial_capital) / n_paths
kelly_below = np.sum(paths_kelly[:, -1] < initial_capital) / n_paths

nonergodic_above = np.sum(paths_nonergodic[:, -1] > initial_capital) / n_paths
nonergodic_below = np.sum(paths_nonergodic[:, -1] < initial_capital) / n_paths

print("Kelly Criterion System:")
print(f"Probability of ending above initial capital: {kelly_above:.1%}")
print(f"Probability of ending below initial capital: {kelly_below:.1%}")
print("\nNon-Ergodic System:")
print(f"Probability of ending above initial capital: {nonergodic_above:.1%}")
print(f"Probability of ending below initial capital: {nonergodic_below:.1%}")

# Calculate median final wealth
kelly_median = np.median(paths_kelly[:, -1])
nonergodic_median = np.median(paths_nonergodic[:, -1])

print(f"\nMedian final wealth:")
print(f"Kelly Criterion: ${kelly_median:.2f}")
print(f"Non-Ergodic: ${nonergodic_median:.2f}")



# ---


# ### 4.) ⚔️ Challenges and Limitations
# 
# ##### Q: Do you really know your decomposed expected value?
# 
# ##### A: No you don't, you can estimate them but in reality these quantities are also stochastic processes...
# 
# - Likely, you should change your estimates in different regimes (volatile, administration, inflation, interest, etc...)


# Generate OU processes for probabilities and trade sizes
def ou_process(n_steps, mu, theta, sigma, initial):
    dt = 1
    x = np.zeros(n_steps)
    x[0] = initial
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
        x[t] = x[t-1] + dx
    return x

n_steps = 1000
# Parameters for probability OU processes
p_win_mu, p_win_theta, p_win_sigma = 0.6, 0.1, 0.05
p_lose_mu, p_lose_theta, p_lose_sigma = 0.4, 0.1, 0.05

# Parameters for trade size OU processes
win_size_mu, win_size_theta, win_size_sigma = 2.0, 0.1, 0.2
lose_size_mu, lose_size_theta, lose_size_sigma = 1.0, 0.1, 0.1

# Generate processes
p_win = np.clip(ou_process(n_steps, p_win_mu, p_win_theta, p_win_sigma, p_win_mu), 0.4, 0.8)
p_lose = 1 - p_win
win_size = ou_process(n_steps, win_size_mu, win_size_theta, win_size_sigma, win_size_mu)
lose_size = -np.abs(ou_process(n_steps, lose_size_mu, lose_size_theta, lose_size_sigma, lose_size_mu))

# Create figure with two subplots side by side
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("Win/Loss Probabilities", "Average Trade Sizes"))

# Add probability traces to first subplot
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=p_win,
               name="Probability of Winning",
               line=dict(color='#00FF00', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=p_lose,
               name="Probability of Losing",
               line=dict(color='#FF0000', width=2)),
    row=1, col=1
)

# Add trade size traces to second subplot
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=win_size,
               name="Average Winning Trade",
               line=dict(color='#00FFFF', width=2)),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=lose_size,
               name="Average Losing Trade",
               line=dict(color='#FF69B4', width=2)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Evolution of Trading Parameters Over Time",
    title_x=0.5,
    width=1250,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(title_text="Time",
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.update_yaxes(title_text="Probability",
                 range=[0, 1],
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 row=1, col=1)

fig.update_yaxes(title_text="Trade Size ($)",
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 row=1, col=2)

fig.show()



# **Remark:** Above these processes are modelled as mean-reverting, this is not necessarily the case in reality


# ##### Impact of Estimation Error


# Set parameters for correct vs incorrect estimation
true_win_rate = 0.51
true_win_pct = 0.5
true_loss_pct = 0.5
true_odds = true_win_pct/true_loss_pct

# Correct estimation
correct_kelly = (true_odds * true_win_rate - (1-true_win_rate)) / true_odds

# Incorrect estimation (overconfident)
estimated_win_rate = 0.55  # Overestimating win rate
incorrect_kelly = (true_odds * estimated_win_rate - (1-estimated_win_rate)) / true_odds

# Simulate paths
paths_correct = np.zeros((n_paths, n_trades + 1))
paths_incorrect = np.zeros((n_paths, n_trades + 1))
paths_correct[:, 0] = initial_capital
paths_incorrect[:, 0] = initial_capital

for path in range(n_paths):
    capital_correct = initial_capital
    capital_incorrect = initial_capital
    
    for i in range(n_trades):
        # Generate true outcome based on true win rate
        is_win = np.random.random() < true_win_rate
        
        # Correct Kelly system
        bet_amount_correct = capital_correct * correct_kelly
        if is_win:
            capital_correct += bet_amount_correct * true_win_pct
        else:
            capital_correct -= bet_amount_correct * true_loss_pct
        paths_correct[path, i + 1] = capital_correct
        
        # Incorrect Kelly system (using overconfident fraction)
        bet_amount_incorrect = capital_incorrect * incorrect_kelly
        if is_win:
            capital_incorrect += bet_amount_incorrect * true_win_pct
        else:
            capital_incorrect -= bet_amount_incorrect * true_loss_pct
        paths_incorrect[path, i + 1] = capital_incorrect

# Calculate mean paths
mean_path_correct = np.mean(paths_correct, axis=0)
mean_path_incorrect = np.mean(paths_incorrect, axis=0)

# Create figure
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Correct Kelly Estimation (f*={correct_kelly:.2%})',
                                  f'Incorrect Kelly Estimation (f*={incorrect_kelly:.2%})'))

# Plot correct estimation paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_correct[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot correct estimation mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_correct,
               name='Mean Path (Correct)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot incorrect estimation paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_incorrect[i],
                   line=dict(color='#FF0000', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot incorrect estimation mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_incorrect,
               name='Mean Path (Incorrect)',
               line=dict(color='#FF0000', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Impact of Kelly Criterion Estimation Error',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)')

fig.show()

# Print statistics
print("\nFinal Wealth Statistics:")
print(f"Correct Estimation - Median Final Wealth: ${np.median(paths_correct[:, -1]):.2f}")
print(f"Incorrect Estimation - Median Final Wealth: ${np.median(paths_incorrect[:, -1]):.2f}")
print(f"\nProbability of Ending Above Initial Capital:")
print(f"Correct Estimation: {np.sum(paths_correct[:, -1] > initial_capital) / n_paths:.1%}")
print(f"Incorrect Estimation: {np.sum(paths_incorrect[:, -1] > initial_capital) / n_paths:.1%}")



# ---


# ### 5.) 💭 Closing Thoughts and Future Topics
# 
# Challenges in reality:
# 
# - Returns are not ergodic, the very possibility of bankrupcy violates the condition of ergodicity
# 
# - Estimating *correct* values for the Kelly Criterion inputs id *difficult*
# 
# - Even small estimation errors can yield massive discrepencies between the expected value of the strategy and the path experienced
# 
# 
# Future Topics:
# 
# - Stochastic processes and topics in stochastic calculus
# 
# - Estimating these quantities from data
# 
# - Building an adaptive trading bot iteratively estimating the Kelly Criterion with updates to the system's parameters 


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

