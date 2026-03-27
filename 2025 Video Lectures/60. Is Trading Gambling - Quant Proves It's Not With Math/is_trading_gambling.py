# ### 🃏 Is Trading Gambling? Quant Proves It's Not With Math & Logic
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
# #### 1.) 🎲 Random and Uncertain Systems
# 
# - Random Systems
# 
# - Uncertain Systems
# 
# #### 2.) 🎰 Games of Chance and Incomplete Information
# 
# - Games of Chance
# 
# - Games of Incomplete Information
# 
# - Optimal Policy Functions
# 
# #### 3.) 🍅 Groceries Shopping and Trading
# 
# - Grocery Shopping as a Max EV Problem
# 
# - Trading as a Max EV Problem
# 
# - If Trading is Gambling, so is Grocery Shopping
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎲 Random and Uncertain Systems
# 
# ##### Random Variables and Stochastic Processes
# 
# Anytime we are dealing with a random system, probabilities and statistics converge
# 
# This means probabilities *literally* represent the expected proportion of outcomes observed
# 
# This goes for both random variables and stochastic processes (which are just random variables over an index)
# 
# $$X \sim \text{Uniform}\{1,2,3,4,5,6\} \quad\quad P(X = k) = \frac{1}{6},\; k \in \{1,2,3,4,5,6\} \quad\quad F_X(x) = \frac{\lfloor x \rfloor}{6}\;\;\; \text{for } 1 \leq x < 6\;\quad \varphi_X(t) = \frac{1}{6} \sum_{k=1}^6 e^{i k t}$$


# ###### ______________________________________________________________________________________________________________________________________


# ##### Population vs Empirical Distribution


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Setup ---
die_faces = np.arange(1, 7)
pmf_theoretical = np.ones(6) / 6  # true population distribution (uniform)
n_trials = 30
np.random.seed(42)
rolls = np.random.choice(die_faces, size=n_trials, replace=True)

# --- Helper: construct figure for a given step ---
def make_empirical_convergence_fig(step):
    drawn = rolls[step]
    counts = np.bincount(rolls[:step+1], minlength=7)[1:]  # exclude index 0
    empirical_pmf = counts / (step + 1)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Probability Mass Function", "Empirical Mass Function"),
    )

    # --- Left subplot: true PMF ---
    bar_colors = ['#39ff14'] * 6  # neon green
    border_colors = ['rgba(0,0,0,0)'] * 6
    border_widths = [0] * 6
    bar_idx = drawn - 1
    border_colors[bar_idx] = '#FFD700'
    border_widths[bar_idx] = 5

    fig.add_trace(
        go.Bar(
            x=die_faces,
            y=pmf_theoretical,
            width=0.5,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=border_widths)),
            name="Dice Roll Probability",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: empirical PMF over time ---
    fig.add_trace(
        go.Bar(
            x=die_faces,
            y=empirical_pmf,
            width=0.5,
            marker=dict(color='#00ffff', opacity=0.8),
            name="Empirical Distribution",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Overlay theoretical PMF on the right for comparison ---
    fig.add_trace(
        go.Scatter(
            x=die_faces,
            y=pmf_theoretical,
            mode='lines',
            line=dict(color='#ff00ff', width=3, dash='dash'),
            name="Theoretical Distribution",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Legend-only traces ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#39ff14', width=4),
            name="Dice Roll Probability",
            showlegend=True
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#00ffff', width=4),
            name="Empirical Distribution",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Axes ---
    fig.update_xaxes(title_text="Die Face", row=1, col=1, range=[0.5, 6.5], tickvals=die_faces)
    fig.update_yaxes(title_text="True P(X=x)", row=1, col=1, range=[0, 0.25])
    fig.update_xaxes(title_text="Die Face", row=1, col=2, range=[0.5, 6.5], tickvals=die_faces)
    fig.update_yaxes(title_text="Empirical P(X=x)", row=1, col=2, range=[0, 0.25])

    # --- Subtle gridlines ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    # --- Layout ---
    fig.update_layout(
        height=500,
        width=1000,
        title_text="Population Distribution vs Empirical Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.97,
            xanchor='right', yanchor='top',
            orientation='v',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=80, t=70),
    )

    return fig

# --- Animation frames ---
frames = [
    go.Frame(data=make_empirical_convergence_fig(step).data, name=str(step))
    for step in range(n_trials)
]

# --- Initial figure ---
fig = make_empirical_convergence_fig(0)
fig.frames = frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()



# **Remark:** The empirical distribution is just the plot of realizations or draws from the population distribution.  Small samples don't reflect the population distribution well, larger samples do.  However, this is only the case if crucially, the population distribution does not change over time.  In other words, we are iteratively rolling the same die, it will not change over time on an uncertain schedule.  If it changed over time at a random schedule (defined) it would still represent the population distribution well.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Law of Large Numbers (LLN) Statistical Convergence
# 
# All random variables have population statistics that tell us something about the shape and relative location of the distribution
# 
# $$M_X(t) = \mathbb{E}[e^{tX}] = \sum_{k=1}^6 e^{tk} \cdot \frac{1}{6} \quad\quad M_X'(0) = \mathbb{E}[X] = 3.5$$
# 
# 
# $$\text{Mean} = 3.5 \quad\quad \text{Variance} = \frac{35}{12} \quad\quad \text{Skewness} = 0 \quad\quad \text{Kurtosis} = -\frac{6}{5}$$
# 
# The Law of Large Numbers (LLN) ensures that repeated sampling from a fixed distribution


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
die_faces = np.arange(1, 7)
pmf_theoretical = np.ones(6)/6
n_trials = 100
np.random.seed(42)
rolls = np.random.choice(die_faces, size=n_trials, replace=True)

# Prepare means for LLN
means = np.cumsum(rolls) / np.arange(1, n_trials+1)

# Helper: construct figure for a given step
def make_lln_dice_fig(step):
    drawn = rolls[step]
    means_until_now = means[:step+1]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Probability Mass Function", "Law of Large Numbers: Mean of Rolls")
    )

    # PMF Bar Colors and Borders
    bar_colors = ['#39ff14']*6  # neon green
    border_widths = [0]*6
    border_colors = ['rgba(0,0,0,0)']*6
    bar_idx = drawn - 1
    border_widths[bar_idx] = 4
    border_colors[bar_idx] = '#ffd600'  # bright yellow border

    # Left subplot: PMF bars (no legend entry here)
    fig.add_trace(
        go.Bar(
            x=die_faces,
            y=pmf_theoretical,
            width=0.5,
            marker=dict(color=bar_colors, line=dict(width=border_widths, color=border_colors)),
            name="PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # Right subplot: cumulative mean (no legend entry here)
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, step+2),
            y=means_until_now,
            mode='lines',
            line=dict(color='#00ffff', width=4),  # neon cyan
            name="Cumulative Mean (live)",
            showlegend=False
        ),
        row=1, col=2
    )

    # Theoretical mean line (no legend entry)
    fig.add_trace(
        go.Scatter(
            x=[1, n_trials],
            y=[np.mean(die_faces)]*2,
            mode='lines',
            line=dict(color='#FFD600', width=2, dash='dash'),
            name="Theoretical Mean",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Legend-only traces, added to the RIGHT subplot ---
    # Green legend entry: Dice Roll Probability
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],  # legend-only
            mode='lines',
            line=dict(color='#39ff14', width=4),
            name="Dice Roll Probability",
            showlegend=True
        ),
        row=1, col=2
    )

    # Cyan legend entry: Cumulative Mean
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],  # legend-only
            mode='lines',
            line=dict(color='#00ffff', width=4),
            name="Cumulative Mean",
            showlegend=True
        ),
        row=1, col=2
    )

    # Axes
    fig.update_xaxes(title_text="Die Face", row=1, col=1, range=[0.5,6.5], tickmode='array', tickvals=die_faces)
    fig.update_yaxes(title_text="P(X=x)", row=1, col=1, range=[0, 0.22])
    fig.update_xaxes(title_text="Number of Rolls (n)", row=1, col=2, range=[0, n_trials+1])
    fig.update_yaxes(title_text="Sample Mean", row=1, col=2, range=[0.8,6.2])
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')


    # Layout with overlaid, stacked legend on the right subplot
    fig.update_layout(
        height=500,
        width=1000,
        title_text="Law of Large Numbers on a Fair Die",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.97,              # inside the right subplot area
            xanchor='right', yanchor='top',
            orientation='v',              # stacked
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=80, t=70),
    )
    return fig

# -- Animate frames
frames = [go.Frame(data=make_lln_dice_fig(step).data, name=str(step)) for step in range(n_trials)]

# Initial figure
fig = make_lln_dice_fig(0)
fig.frames = frames

fig.update_layout(
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

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Law of Large Numbers (LLN), Probabalistic, and Distribution Convergence
# 
# 
# The Law of Large Numbers (LLN) ensures that for any threshold x, the empirical CDF converges to the true CDF as the sample size increases.
#  
# $$\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{X_i \leq x\} \xrightarrow{n \to \infty} F(x) = P(X \leq x)$$
# 
# This works because for each x, the empirical CDF is just the average of Bernoulli indicators 1{X_i ≤ x}, which by the LLN converges to P(X ≤ x).


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Setup ---
die_faces = np.arange(1, 7)
pmf_theoretical = np.ones(6) / 6  # true population distribution (uniform)
n_trials = 500
np.random.seed(42)
rolls = np.random.choice(die_faces, size=n_trials, replace=True)

# --- Helper: construct figure for a given step ---
def make_empirical_convergence_fig(step):
    drawn = rolls[step]
    counts = np.bincount(rolls[:step+1], minlength=7)[1:]  # exclude index 0
    empirical_pmf = counts / (step + 1)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Probability Mass Function", "Empirical Mass Function"),
    )

    # --- Left subplot: true PMF ---
    bar_colors = ['#39ff14'] * 6  # neon green
    border_colors = ['rgba(0,0,0,0)'] * 6
    border_widths = [0] * 6
    bar_idx = drawn - 1
    border_colors[bar_idx] = '#FFD700'
    border_widths[bar_idx] = 5

    fig.add_trace(
        go.Bar(
            x=die_faces,
            y=pmf_theoretical,
            width=0.5,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=border_widths)),
            name="Dice Roll Probability",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: empirical PMF over time ---
    fig.add_trace(
        go.Bar(
            x=die_faces,
            y=empirical_pmf,
            width=0.5,
            marker=dict(color='#00ffff', opacity=0.8),
            name="Empirical Distribution",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Overlay theoretical PMF on the right for comparison ---
    fig.add_trace(
        go.Scatter(
            x=die_faces,
            y=pmf_theoretical,
            mode='lines',
            line=dict(color='#ff00ff', width=3, dash='dash'),
            name="Theoretical Distribution",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Legend-only traces ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#39ff14', width=4),
            name="Dice Roll Probability",
            showlegend=True
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#00ffff', width=4),
            name="Empirical Distribution",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Axes ---
    fig.update_xaxes(title_text="Die Face", row=1, col=1, range=[0.5, 6.5], tickvals=die_faces)
    fig.update_yaxes(title_text="True P(X=x)", row=1, col=1, range=[0, 0.25])
    fig.update_xaxes(title_text="Die Face", row=1, col=2, range=[0.5, 6.5], tickvals=die_faces)
    fig.update_yaxes(title_text="Empirical P(X=x)", row=1, col=2, range=[0, 0.25])

    # --- Subtle gridlines ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    # --- Layout ---
    fig.update_layout(
        height=500,
        width=1000,
        title_text="Convergence of Empirical Distribution to Population Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.97,
            xanchor='right', yanchor='top',
            orientation='v',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=80, t=70),
    )

    return fig

# --- Animation frames ---
frames = [
    go.Frame(data=make_empirical_convergence_fig(step).data, name=str(step))
    for step in range(n_trials)
]

# --- Initial figure ---
fig = make_empirical_convergence_fig(0)
fig.frames = frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Statistics Inform Decision Making in Randomness
# 
# Probabilities and statistics of stochastic processes also converge by the law of large numbers
# 
# Consider a game where we are trading the outcome of a dice roll, we make the market and set the bid/offer. . .
# 
# $$W_n = \sum_{i=1}^n \left( X_i - P_i \right), \quad \text{where } X_i \sim \text{DieRoll},\; P_i = \begin{cases} 4 & \text{if buy} \\ 2 & \text{if sell} \end{cases}$$
# 
# Assume a 50/50 probability of buying or selling, by the law of total expectation. . .
# 
# $$\mathbb{E}[P/L] = \mathbb{E}[P/L| Long]P(Long) + \mathbb{E}[P/L| Short]P(Short) = (4 - \mathbb{E}[X])(.5) + (\mathbb{E}[X] - 2)(.5) = \$.5 $$
# 
# This is our edge, if we trade 100 times in this random system we will on average accumulate $50 of wealth. . .
# 
# $$\$.5 \text{ per roll} \times 100 \text{ trades} = \$50 $$


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Parameters ---
np.random.seed(42)
n_steps = 100
bid, ask = 3, 4  # fair quotes around expected value (3.5)
die_faces = np.arange(1, 7)
n_samples = 10  # number of extra sample paths

# --- Simulate main sample path (used for animation) ---
directions = np.random.choice([-1, 1], size=n_steps)
executed_prices = np.where(directions == 1, bid, ask)
outcomes = np.random.choice(die_faces, size=n_steps)
trade_pnl = np.where(directions == 1, outcomes - bid, ask - outcomes)
wealth = np.cumsum(trade_pnl)

# --- Theoretical EV line ---
avg_ev_per_trade = 0.5  # theoretical EV per trade
expected_wealth = np.linspace(avg_ev_per_trade, avg_ev_per_trade * n_steps, n_steps)

# --- Generate extra sample paths for comparison ---
sample_paths = []
for i in range(n_samples):
    dirs = np.random.choice([-1, 1], size=n_steps)
    outs = np.random.choice(die_faces, size=n_steps)
    pnl = np.where(dirs == 1, outs - bid, ask - outs)
    sample_paths.append(np.cumsum(pnl))

# --- Opacity gradient for sample paths ---
opacities = np.linspace(0.15, 0.8, n_samples)

# --- Precompute global wealth Y range for right subplot to keep plots steady ---
all_paths = [wealth] + sample_paths + [expected_wealth]
ymin = min([path.min() for path in all_paths])
ymax = max([path.max() for path in all_paths])
y_margin = (ymax - ymin) * 0.05
y_range = [ymin - y_margin, ymax + y_margin]


# --- Helper: build one frame ---
def make_market_fig(step):
    side = "Long @ Bid (3)" if directions[step] == 1 else "Short @ Ask (4)"
    roll = outcomes[step]
    pnl = trade_pnl[step]
    wealth_until_now = wealth[:step + 1]

    # Expanding window for x_range in right subplot
    x_max = step + 2 if step + 2 > 10 else 10
    x_range = [0, x_max]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Market Side & Outcome", "Wealth Paths vs Theoretical EV")
    )

    # --- Left chart: Bid/Ask sides ---
    bar_colors = ['#39ff14', '#ff073a']  # neon green (bid), neon red (ask)
    sides = ["Long @ Bid (3)", "Short @ Ask (4)"]
    border_colors = ['rgba(0,0,0,0)'] * 2
    border_widths = [0, 0]
    idx = sides.index(side)
    border_colors[idx] = '#FFD700'  # highlight current side
    border_widths[idx] = 5

    fig.add_trace(
        go.Bar(
            x=sides,
            y=[0.5, 0.5],
            marker=dict(color=bar_colors,
                        line=dict(color=border_colors, width=border_widths)),
            showlegend=False,
        ),
        row=1, col=1
    )

    # --- Add outcome and P/L annotation ---
    fig.add_annotation(
        text=f"🎲 Outcome = <b>{roll}</b><br>💰 Market Maker P/L = <b>{pnl:+.2f}</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.1,
        showarrow=False,
        font=dict(size=18, color="white"),
        row=1, col=1
    )

    # --- Right chart: sample paths ---
    for i, path in enumerate(sample_paths):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, step + 2),
                y=path[:step + 1],
                mode='lines',
                line=dict(color=f'rgba(0,255,255,{opacities[i]:.2f})', width=1.5),
                showlegend=False
            ),
            row=1, col=2
        )

    # --- Primary observed wealth path ---
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, step + 2),
            y=wealth_until_now,
            mode='lines+markers',
            line=dict(color='#00bfff', width=4),
            marker=dict(size=6, color='#00bfff'),
            name="Observed Path",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Theoretical EV line ---
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, n_steps + 1),
            y=expected_wealth,
            mode='lines',
            line=dict(color='#ff00ff', width=4, dash='dash'),
            name="Theoretical Path",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Axes and layout styling ---
    fig.update_xaxes(title_text="Trade Side", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Trade Number (n)", row=1, col=2, range=x_range)
    fig.update_yaxes(title_text="Cumulative Wealth", row=1, col=2, range=y_range)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    fig.update_layout(
        height=500,
        width=1000,
        title_text="Market Making: Convergence of Multiple Wealth Paths to Expected Value",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.75, y=0.95,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=80, t=70),
    )
    return fig, x_range


# --- Build initial figure ---
fig, _ = make_market_fig(0)

# --- Create animation frames (data + limited layout updates) ---
frames = []
for step in range(n_steps):
    step_fig, x_range = make_market_fig(step)
    frames.append(go.Frame(
        data=step_fig.data,
        name=str(step),
        layout=go.Layout(xaxis2=dict(range=x_range))  # only adjust right subplot xlim
    ))

fig.frames = frames

# --- Play button setup ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': .01}
            }]
        }]
    }]
)

fig.show()



# **Remark:** This isn't a course on stochastic processes but this will only be true if the system is ergodic (i.e. time average = ensemble average), I have a video on this topic in the context of optimal bet sizing (additive vs multiplicative processes) that I will link in the video description below.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Uncertain Systems
# 
# In random systems we observe convergence in probabilities, statistics, and distributions. . .
# 
# In uncertain systems, we observe *do not* observe this behavior - a great source of confusion for both students and folks new to the space
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_steps = 200
initial_wealth = 1000
bet_size = 25
n_paths = 14  # More paths to show more non-stationary cases

# --- LEFT: Positive EV (Random system - convergence to mean) ---
left_p_win = 0.60  # 60% win
left_ev = bet_size * (left_p_win - (1 - left_p_win))
pos_ev_paths = np.zeros((n_paths, n_steps))
pos_ev_paths[:, 0] = initial_wealth
for path in range(n_paths):
    wins = np.random.random(n_steps) < left_p_win
    for i in range(1, n_steps):
        pos_ev_paths[path, i] = pos_ev_paths[path, i-1] + (bet_size if wins[i] else -bet_size)
pos_ev_line = initial_wealth + np.arange(n_steps) * left_ev

# --- RIGHT: Non-stationary system: time-varying EV within each path ---
# For greater variance, increase spread and amplitude of starting/ending p_win and ensure some wild time-variation.
# We'll let starting p_wins range lower/higher, and use larger wave amplitude.

# Broader range for greater variance
right_p_win_start = np.linspace(0.10, 0.85, n_paths)
right_p_win_end   = np.linspace(0.85, 0.10, n_paths)  # Reverse for max spread/high variance

nonstationary_paths = np.zeros((n_paths, n_steps))
nonstationary_paths[:, 0] = initial_wealth

# For one path, make the 'red dashed' highlight (e.g. middle path)
highlight_path_idx = n_paths // 2

# For legend at end, store EV over time for highlight path
highlight_path_evs = []

for idx in range(n_paths):
    # For each path, p_win varies as either wild up, down, or big amplitude sine/cos
    if idx % 3 == 0:
        # Linear variation with even more extreme start & end
        p_win = np.linspace(right_p_win_start[idx], right_p_win_end[idx], n_steps)
    elif idx % 3 == 1:
        # Linear down (opposite)
        p_win = np.linspace(right_p_win_end[idx], right_p_win_start[idx], n_steps)
    else:
        # Very large amplitude sinusoidal oscillation within [0.05,0.95]
        amp = 0.45  # amplitude (nearly full interval)
        mid = 0.5 + np.random.uniform(-0.15, 0.15)
        freq = 1 + np.random.uniform(-0.25, 0.25)  # some randomness to frequency
        p_win = np.clip(
            mid + amp * np.sin(np.linspace(0, np.pi * 2 * freq, n_steps) + np.random.uniform(-2,2)),
            0.05, 0.95
        )
    wins = np.random.random(n_steps) < p_win
    for i in range(1, n_steps):
        # Increase realized variance by adding noise to outcome with some big 'shock' steps
        shock = 0
        if np.random.rand() < 0.03:  # ~3% chance of a wild outcome
            shock = np.random.choice([2, -2]) * bet_size  # Big move up or down!
        nonstationary_paths[idx, i] = nonstationary_paths[idx, i-1] + (bet_size if wins[i] else -bet_size) + shock
    if idx == highlight_path_idx:
        highlight_path_evs = bet_size * (p_win*2 - 1)  # Save EV(t) for annotation if wanted

# --- Animation Frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []
    # Left: random system, positive edge, convergence
    for path in range(n_paths):
        opacity = 1.0 - (path * (0.7/n_paths))
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
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=pos_ev_line[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x', yaxis='y',
            name='EV (Positive)',
            showlegend=False
        )
    )
    # Right: non-stationary, each path's EV changes in time
    for path in range(n_paths):
        opacity = 1.0 - (path * (0.7/n_paths))
        if path == highlight_path_idx:
            # Red dashed highlight
            frame_data.append(
                go.Scatter(
                    x=np.arange(i+1),
                    y=nonstationary_paths[path, :i+1],
                    mode='lines',
                    line=dict(
                        color='red',  # Red
                        width=4,
                        dash='dash'
                    ),
                    opacity=1.0,
                    xaxis='x2', yaxis='y2',
                    name='Non-stationary Example',
                    showlegend=(i == n_steps-1)
                )
            )
        else:
            frame_data.append(
                go.Scatter(
                    x=np.arange(i+1),
                    y=nonstationary_paths[path, :i+1],
                    mode='lines',
                    line=dict(
                        color='#B266FF',  # Neon purple-ish; use a strong neon purple
                        width=2
                    ),
                    opacity=opacity,
                    xaxis='x2', yaxis='y2',
                    name=None,
                    showlegend=False
                )
            )
    frames.append(go.Frame(
        name=f'frame{i}',
        data=frame_data,
        layout=go.Layout(xaxis=dict(range=[0, n_steps]), xaxis2=dict(range=[0, n_steps]))
    ))

# --- Build initial traces for the animated figure ---
fig_nonstat = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Random/Ergodic Process (Positive EV & Convergence)",
        "Non-stationary Process (EV Shifts Over Time)"
    ),
    column_widths=[0.5, 0.5]
)
# LEFT: positive EV sample paths & EV line
for path in range(n_paths):
    fig_nonstat.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * (0.7/n_paths),
            showlegend=False
        ),
        row=1, col=1
    )
fig_nonstat.add_trace(
    go.Scatter(
        x=[0],
        y=[initial_wealth],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='EV (Positive)',
        showlegend=True
    ),
    row=1, col=1
)
# RIGHT: non-stationary, different time-varying EVs per path
for path in range(n_paths):
    if path == highlight_path_idx:
        fig_nonstat.add_trace(
            go.Scatter(
                x=[0],
                y=[initial_wealth],
                mode='lines',
                line=dict(color='red', dash='dash', width=4),
                name='Non-stationary Example',
                showlegend=True
            ),
            row=1, col=2
        )
    else:
        fig_nonstat.add_trace(
            go.Scatter(
                x=[0],
                y=[initial_wealth],
                mode='lines',
                line=dict(color='#B266FF', width=2),  # neon purple lines
                opacity=1.0 - path * (0.7/n_paths),
                showlegend=False
            ),
            row=1, col=2
        )

fig_nonstat.frames = frames

fig_nonstat.update_layout(
    height=550,
    width=1000,
    title_text="Random Convergence (Positive EV) vs Non-stationary Process (No Statistical Convergence!)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.17,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 10, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_nonstat.update_xaxes(title_text="Round", range=[0, n_steps],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig_nonstat.update_yaxes(title_text="Wealth", range=[0, 2000],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig_nonstat.show()



# ##### This is where things get tricky.  Statistics we run on data can be misleading because they *change over time*
# 
# Instead of more samples leading to convergence (like in random systems), more samples may correspond to *different* statistics
# 
# This is why the trading space is so confusing - past performance isn't indicative of future performance
# 
# Nothing converges, no probabilities, no statistics, no distributions, everything changes over time
# 
# We need to change the way we think about these statistics now depending on the context and our goal


# ---


# #### 2.) 🎰 Games of Chance and Incomplete Information
# 
# We can discuss the idea of **edge** and the application to random and uncertain systems
# 
# We never know the outcome of *one* event in either a **random** or **uncertain** system, so how do we make decisions?
# 
# How our wealth evolves over time is dictated by our edge or EV, we can compute it by the Law of Total Expectation (LoTE) below. . .
#  
#  The Law of Total Expectation (for a discrete partition $\{B_i\}$ of the sample space) is given by:
#  
#  $$\mathbb{E}[X] = \sum_{i} \mathbb{E}[X\mid B_i] \cdot \mathbb{P}(B_i)$$
#  
#  where $X$ is a random variable, and $\{B_i\}$ is a partition of the sample space.
# 


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



# ##### Well Then, What Dictates Our Expected Value?
# 
# **The system itself or our actions**


# ###### ______________________________________________________________________________________________________________________________________


# ##### Policy Functions
# 
# Depending on the system we are operating in, we may be able to influence our expected value (EV) or edge with *our own actions*
# 
# The function $\pi$ represents **a collection of actions given the current state of the system** and dictates what action we end up taking
# 
# $$\pi^* = \arg\max_\pi \mathbb{E}[R \mid \pi]$$
# 
# Just because we act optimally, it does not mean we will have positive EV at this point in time
# 
# Moreover, the optimal policy may change over time depending on the system!


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Shared parameters ---
n_policies = 100
policy_vals = np.linspace(0, 1, n_policies)
n_steps = 60
n_paths = 8
init_wealth = 1000
bet_unit = 25
win_prob_optimal = 0.6      # Optimal: positive EV
win_prob_subopt = 0.45      # Suboptimal: negative EV, for demonstration

# ----------- POLICY FUNCTIONS (EV curves) ----------------

# -- Optimal (positive edge) quadratic policy --
a1 = 2.8
h1 = 0.58
k1 = 0.20
evs_opt = -a1*(policy_vals - h1)**2 + k1
opt_idx = np.argmax(evs_opt)
opt_policy = policy_vals[opt_idx]
opt_ev = evs_opt[opt_idx]

# -- Suboptimal (negative edge) quadratic policy --
# Keep quadratic, just shift downward and toward more negative max
a2 = 2.8
h2 = 0.45
k2 = -0.08
evs_subopt = -a2*(policy_vals - h2)**2 + k2
subopt_idx = np.argmax(evs_subopt)
subopt_policy = policy_vals[subopt_idx]
subopt_ev = evs_subopt[subopt_idx]

# ----------- WEALTH PATHS (for optimal/suboptimal) ------------

# -- Sample wealth paths for optimal policy --
paths_opt = np.zeros((n_paths, n_steps))
paths_opt[:, 0] = init_wealth
for path in range(n_paths):
    for i in range(1, n_steps):
        do_bet = np.random.random() < opt_policy
        if do_bet:
            win = np.random.random() < win_prob_optimal
            change = bet_unit if win else -bet_unit
        else:
            change = 0
        paths_opt[path, i] = paths_opt[path, i-1] + change

# -- Sample wealth paths for suboptimal (negative EV) policy --
paths_subopt = np.zeros((n_paths, n_steps))
paths_subopt[:, 0] = init_wealth
for path in range(n_paths):
    for i in range(1, n_steps):
        do_bet = np.random.random() < subopt_policy
        if do_bet:
            win = np.random.random() < win_prob_subopt
            change = bet_unit if win else -bet_unit
        else:
            change = 0
        paths_subopt[path, i] = paths_subopt[path, i-1] + change

# ----------- POLICY & STAR TRACES (LEFT) --------------
policy_curve_trace_opt = go.Scatter(
    x=policy_vals, y=evs_opt,
    mode='lines',
    line=dict(color='royalblue', width=3),
    name='Expected Value by Policy',
    xaxis='x', yaxis='y',
    showlegend=True
)
policy_star_trace_opt = go.Scatter(
    x=[opt_policy], y=[opt_ev],
    mode='markers+text',
    marker=dict(symbol='star', size=24, color='gold', line=dict(color='black', width=2)),
    text=['Optimal Policy'],
    textposition='top center',
    showlegend=False,
    xaxis='x', yaxis='y'
)

policy_curve_trace_sub = go.Scatter(
    x=policy_vals, y=evs_subopt,
    mode='lines',
    line=dict(color='crimson', width=3),
    name='Expected Value by Policy',
    xaxis='x3', yaxis='y3',
    showlegend=True
)
policy_star_trace_sub = go.Scatter(
    x=[subopt_policy], y=[subopt_ev],
    mode='markers+text',
    marker=dict(symbol='star', size=24, color='orange', line=dict(color='black', width=2)),
    text=['Policy Chosen'],
    textposition='top center',
    showlegend=False,
    xaxis='x3', yaxis='y3'
)

# ----------- ANIMATION FRAMES (for both upper and lower subplots) ------------

policy_ylim_pad = 0.2
frames = []
for i in range(1, n_steps):
    frame_data = []
    # --- Top row: optimal policy ---
    frame_data.append(policy_curve_trace_opt)
    frame_data.append(policy_star_trace_opt)
    # Wealth path animations (right subplot 1, row 1 col 2)
    for path in range(n_paths):
        opacity = 1.0 - path * 0.07
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=paths_opt[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False
            )
        )
    frame_data.append(
        go.Scatter(
            x=[0, i],
            y=[init_wealth, init_wealth + i * opt_policy * (2 * win_prob_optimal - 1) * bet_unit],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x2', yaxis='y2',
            showlegend=False
        )
    )

    # --- Lower row: suboptimal policy ---
    frame_data.append(policy_curve_trace_sub)
    frame_data.append(policy_star_trace_sub)
    # Wealth path animations (right subplot 2, row 2 col 2)
    for path in range(n_paths):
        opacity = 1.0 - path * 0.07
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=paths_subopt[path, :i+1],
                mode='lines',
                line=dict(color='firebrick', width=2),
                opacity=opacity,
                xaxis='x4', yaxis='y4',
                showlegend=False
            )
        )
    frame_data.append(
        go.Scatter(
            x=[0, i],
            y=[init_wealth, init_wealth + i * subopt_policy * (2 * win_prob_subopt - 1) * bet_unit],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x4', yaxis='y4',
            showlegend=False
        )
    )
    frames.append(go.Frame(
        name=f"f{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),  # top left
            yaxis=dict(
                range=[evs_opt.min() - 0.05 - policy_ylim_pad, evs_opt.max() + 0.05 + policy_ylim_pad],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)'
            ),
            xaxis2=dict(range=[0, n_steps], showgrid=True, gridcolor='rgba(128,128,128,0.3)'), # top right
            yaxis2=dict(range=[0, 2000], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
            xaxis3=dict(range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),      # bottom left
            yaxis3=dict(
                range=[evs_subopt.min() - 0.05 - policy_ylim_pad, evs_subopt.max() + 0.05 + policy_ylim_pad],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)'
            ),
            xaxis4=dict(range=[0, n_steps], showgrid=True, gridcolor='rgba(128,128,128,0.3)'), # bottom right
            yaxis4=dict(range=[0, 2000], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        )
    ))

# ----------- FIGURE: 2x2 subplots -------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Policy Function (Optimal — Positive EV)', f'Sample Wealth Paths (Optimal Policy = {opt_policy:.2f})',
        'Policy Function (Suboptimal — Negative EV)', f'Sample Wealth Paths (Suboptimal Policy = {subopt_policy:.2f})'
    ],
    horizontal_spacing=0.13,
    vertical_spacing=0.17
)

# --- Top row: optimal (positive edge) ---
fig.add_trace(policy_curve_trace_opt, row=1, col=1)
fig.add_trace(policy_star_trace_opt, row=1, col=1)
fig.update_xaxes(title_text="Policy (P(bet))", range=[0, 1],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=1)
fig.update_yaxes(
    title_text="Expected Value per Play",
    range=[evs_opt.min() - policy_ylim_pad, evs_opt.max() + policy_ylim_pad],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=1)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[init_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Sample Wealth Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=2
    )
fig.add_trace(
    go.Scatter(
        x=[0, n_steps-1],
        y=[init_wealth, init_wealth + (n_steps-1)*opt_policy*(2*win_prob_optimal - 1)*bet_unit],
        mode='lines',
        line=dict(color='lime', dash='dash', width=3),
        name='Expected Value'
    ),
    row=1, col=2
)
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=2)
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=2)

# --- Bottom row: suboptimal (negative edge) ---
fig.add_trace(policy_curve_trace_sub, row=2, col=1)
fig.add_trace(policy_star_trace_sub, row=2, col=1)
fig.update_xaxes(title_text="Policy (P(bet))", range=[0, 1],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=2, col=1)
fig.update_yaxes(
    title_text="Expected Value per Play",
    range=[evs_subopt.min() - policy_ylim_pad, evs_subopt.max() + policy_ylim_pad],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=2, col=1)
for path in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[init_wealth],
            mode='lines',
            line=dict(color='firebrick', width=2),
            opacity=1.0 - path * 0.07,
            name='Sample Wealth Path (Neg EV)' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=2, col=2
    )
fig.add_trace(
    go.Scatter(
        x=[0, n_steps-1],
        y=[init_wealth, init_wealth + (n_steps-1)*subopt_policy*(2*win_prob_subopt - 1)*bet_unit],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='Expected Value (Neg EV)'
    ),
    row=2, col=2
)
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=2, col=2)
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=2, col=2)

# ----------- Layout/Attachment for animation -----------
fig.frames = frames
fig.update_layout(
    height=860, width=1000,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text="Comparison: Positive vs Negative EV Policy Function & Wealth Paths (Animated)",
    showlegend=True,
    legend=dict(orientation='h', y=-0.21, x=0.51, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.07, 'y': -0.13,
        'showactive': False,
        'buttons': [dict(
            label='Play',
            method='animate',
            args=[None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        )]
    }]
)
fig.show()



# **Key Point:** Optimal actions in a policy function guide overall expected value (EV) or edge in a positive direction - but it doesn't make it positive outright.  Positive EV (having an edge) is the accumulation of a set of optimal actions continuously updated over time. 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Games of Chance (Random System, Fixed Negative Edge: Gambling)
# 
# *Gambling* occurs when you **willfully and unnecessarily** engage with a system that has a fixed negative edge against you
# 
# This is a game of chance, and where all the negative connotations of gambling should come from
# 
# No matter what you do, no matter the policy function $\pi$, if you continue to engage with that system you are certain to lose all of your money
# 
#  **Examples:**
#  - Roulette
#  - Slots
#  - Craps
#  - Lottery Tickets
# 
# 
# Effectively, the unconditional edge and edge conditional on your policy funciton are equivalent
# 
# $$\mathbb{E}[X] = \sum_{i} \mathbb{E}[X\mid B_i] \cdot \mathbb{P}(B_i) = \mathbb{E}[X \mid \pi^*]$$
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_steps = 100
initial_wealth = 1000
bet_size = 25
n_paths = 10

# --- Simulate unconditional edge (always bet, policy doesn't matter) ---
neg_ev_paths_uncond = np.zeros((n_paths, n_steps))
neg_ev_paths_uncond[:, 0] = initial_wealth
for path in range(n_paths):
    neg_ev_wins = np.random.random(n_steps) < 0.40  # 40% win probability
    for i in range(1, n_steps):
        neg_ev_paths_uncond[path, i] = neg_ev_paths_uncond[path, i-1] + (bet_size if neg_ev_wins[i] else -bet_size)
neg_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (0.40 - 0.60)

# --- Simulate "policy" edge (skip 50% of rounds at random, still negative EV) ---
neg_ev_paths_cond = np.zeros((n_paths, n_steps))
neg_ev_paths_cond[:, 0] = initial_wealth
all_take_actions = []
for path in range(n_paths):
    neg_ev_wins = np.random.random(n_steps) < 0.40  # 40% win
    take_action = np.random.random(n_steps) < 0.5    # only bet half the time at random
    all_take_actions.append(take_action)
    for i in range(1, n_steps):
        if take_action[i]:
            neg_ev_paths_cond[path, i] = neg_ev_paths_cond[path, i-1] + (bet_size if neg_ev_wins[i] else -bet_size)
        else:
            neg_ev_paths_cond[path, i] = neg_ev_paths_cond[path, i-1]  # sit out

# --- Prepare expected value lines per frame (for animation) ---
# For unconditional, every round is a bet: EV is linear
neg_ev_line_frames = initial_wealth + np.arange(n_steps) * bet_size * (0.40 - 0.60)

# For conditional, use the *first* random skip pattern for the animation (to be repeatable)
take_action_anim = all_take_actions[0]
rounds_participated_anim = np.cumsum(take_action_anim)
neg_ev_line_cond_frames = initial_wealth + rounds_participated_anim * bet_size * (0.40 - 0.60)

# --- Build animation frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []
    # Left: unconditional
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=neg_ev_paths_uncond[path, :i+1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=opacity,
                xaxis='x',
                yaxis='y',
                showlegend=False
            )
        )
    # Left: expected value line
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=neg_ev_line_frames[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x',
            yaxis='y',
            showlegend=False
        )
    )
    # Right: conditional/skipped rounds
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=neg_ev_paths_cond[path, :i+1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=opacity,
                xaxis='x2',
                yaxis='y2',
                showlegend=False
            )
        )
    # Right: expected value line (conditional policy)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=neg_ev_line_cond_frames[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x2',
            yaxis='y2',
            showlegend=False
        )
    )
    frames.append(go.Frame(
        name=f'frame{i}',
        data=frame_data,
        layout=go.Layout(xaxis=dict(range=[0, n_steps]),
                         xaxis2=dict(range=[0, n_steps]))
    ))

# --- Build the initial animated figure ---
fig_ev = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Unconditional Edge (Always Bet)",
        "Conditional Edge (Change Policy, Still Negative EV)"
    ),
    column_widths=[0.5, 0.5]
)

# Initial unconditional traces (left)
for path in range(n_paths):
    fig_ev.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_wealth],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=1.0 - path * 0.07,
            name='Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )
# Initial unconditional expected value line
fig_ev.add_trace(
    go.Scatter(
        x=[0],
        y=[initial_wealth],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='EV (Unconditional)',
        showlegend=True
    ),
    row=1, col=1
)

# Initial conditional traces (right)
for path in range(n_paths):
    fig_ev.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_wealth],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=1.0 - path * 0.07,
            name='Path' if path == 0 else None,
            showlegend=False
        ),
        row=1, col=2
    )
# Initial conditional expected value line
fig_ev.add_trace(
    go.Scatter(
        x=[0],
        y=[initial_wealth],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='EV (Conditional)',
        showlegend=True
    ),
    row=1, col=2
)

fig_ev.frames = frames

# --- Layout ---
fig_ev.update_layout(
    height=550,
    width=1100,
    title_text="Negative Expected Value: Unconditional vs Conditional Edge<br><sup>Your policy function cannot overcome a negative edge</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.17,
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

fig_ev.update_xaxes(title_text="Round", range=[0, n_steps],
                    showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig_ev.update_yaxes(title_text="Wealth", range=[0, 2000],
                    showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig_ev.show()



# There is no optimal set of actions to accumulate wealth in these systems - your optimal action is not engaging in these systems
# 
# *Remark:* This is not a class on stochastic processes but I have to make the remark that this is not a comment on ergodicity.  I have a video discussing this topic in the context of optimal bet sizing (additive vs multiplicative bets) - I will leave a link to it in the description below if you are interested in this topic and how it's connected to this idea of accumulating edge.


# ###### ______________________________________________________________________________________________________________________________________


# #### Games of Incomplete Information
# 
# Unlike gambling and games of chance, we can act with a policy function $\pi$ that dictates the trajectory of our wealth over time
# 
# Crucially, in these systems we don't observe convergence (No LLN) - everything changes over time, including our optimal set of actions ($\pi^*$)!
# 
# **Examples:**
#  - Election Betting
#  - Sports Betting
#  - Poker 
#  - Trading
#  - Grocery Shopping?
# 
#  More specifically, 
#  
#  The unconditional expectation of the system (or conditional on random action) **IS NOT** equal to the expectation conditional on our policy function! 
#  
# $$\mathbb{E}[X] = \sum_{i} \mathbb{E}[X\mid B_i] \cdot \mathbb{P}(B_i) \neq \mathbb{E}[X \mid \pi^*]$$
# 
# **In other words, our actions influence the trajectory of our wealth path!  Unlike in a game of chance or gambling**


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_steps = 100
initial_wealth = 1000
bet_size = 25
n_paths = 10

# --- Simulate negative EV (always bet, policy doesn't matter): left subplot ---
neg_ev_paths_uncond = np.zeros((n_paths, n_steps))
neg_ev_paths_uncond[:, 0] = initial_wealth
for path in range(n_paths):
    neg_ev_wins = np.random.random(n_steps) < 0.40  # 40% win probability (negative edge)
    for i in range(1, n_steps):
        neg_ev_paths_uncond[path, i] = neg_ev_paths_uncond[path, i-1] + (bet_size if neg_ev_wins[i] else -bet_size)
neg_ev_line = initial_wealth + np.arange(n_steps) * bet_size * (0.40 - 0.60)

# --- Simulate positive EV for conditional/policy edge (right subplot): player can accumulate wealth ---
pos_ev_paths_cond = np.zeros((n_paths, n_steps))
pos_ev_paths_cond[:, 0] = initial_wealth
all_take_actions_pos = []
for path in range(n_paths):
    pos_ev_wins = np.random.random(n_steps) < 0.60  # 60% win probability (positive edge)
    take_action = np.random.random(n_steps) < 0.5    # still bet half the time at random
    all_take_actions_pos.append(take_action)
    for i in range(1, n_steps):
        if take_action[i]:
            pos_ev_paths_cond[path, i] = pos_ev_paths_cond[path, i-1] + (bet_size if pos_ev_wins[i] else -bet_size)
        else:
            pos_ev_paths_cond[path, i] = pos_ev_paths_cond[path, i-1]  # sit out

# --- Prepare expected value lines per frame (for animation) ---
# For unconditional (left): negative EV
neg_ev_line_frames = initial_wealth + np.arange(n_steps) * bet_size * (0.40 - 0.60)

# For conditional (right): positive EV, use first skip pattern for animation
take_action_anim_pos = all_take_actions_pos[0]
rounds_participated_anim_pos = np.cumsum(take_action_anim_pos)
pos_ev_line_cond_frames = initial_wealth + rounds_participated_anim_pos * bet_size * (0.60 - 0.40)

# --- Build animation frames ---
frames = []
for i in range(1, n_steps):
    frame_data = []
    # Left: unconditional (always negative EV)
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=neg_ev_paths_uncond[path, :i+1],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=opacity,
                xaxis='x',
                yaxis='y',
                showlegend=False
            )
        )
    # Left: expected value line
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=neg_ev_line_frames[:i+1],
            mode='lines',
            line=dict(color='orange', dash='dash', width=3),
            xaxis='x',
            yaxis='y',
            showlegend=False
        )
    )
    # Right: conditional/policy (POSITIVE EV!)
    for path in range(n_paths):
        opacity = 1.0 - (path * 0.07)
        frame_data.append(
            go.Scatter(
                x=np.arange(i+1),
                y=pos_ev_paths_cond[path, :i+1],
                mode='lines',
                line=dict(color='green', width=2),
                opacity=opacity,
                xaxis='x2',
                yaxis='y2',
                showlegend=False
            )
        )
    # Right: expected value line (conditional positive EV policy)
    frame_data.append(
        go.Scatter(
            x=np.arange(i+1),
            y=pos_ev_line_cond_frames[:i+1],
            mode='lines',
            line=dict(color='lime', dash='dash', width=3),
            xaxis='x2',
            yaxis='y2',
            showlegend=False
        )
    )
    frames.append(go.Frame(
        name=f'frame{i}',
        data=frame_data,
        layout=go.Layout(xaxis=dict(range=[0, n_steps]),
                         xaxis2=dict(range=[0, n_steps]))
    ))

# --- Build the initial animated figure ---
fig_ev = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Ineffective Policy Function (Always Bet, Negative EV)",
        "Effective Policy Function (Policy, Positive EV)"
    ),
    column_widths=[0.5, 0.5]
)

# Initial unconditional traces (left)
for path in range(n_paths):
    fig_ev.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_wealth],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=1.0 - path * 0.07,
            name='Path' if path == 0 else None,
            showlegend=(path == 0)
        ),
        row=1, col=1
    )
# Initial unconditional expected value line
fig_ev.add_trace(
    go.Scatter(
        x=[0],
        y=[initial_wealth],
        mode='lines',
        line=dict(color='orange', dash='dash', width=3),
        name='EV (Unconditional, -EV)',
        showlegend=True
    ),
    row=1, col=1
)

# Initial conditional traces (right, now positive EV)
for path in range(n_paths):
    fig_ev.add_trace(
        go.Scatter(
            x=[0],
            y=[initial_wealth],
            mode='lines',
            line=dict(color='green', width=2),
            opacity=1.0 - path * 0.07,
            name='Path' if path == 0 else None,
            showlegend=False
        ),
        row=1, col=2
    )
# Initial conditional expected value line (positive EV)
fig_ev.add_trace(
    go.Scatter(
        x=[0],
        y=[initial_wealth],
        mode='lines',
        line=dict(color='lime', dash='dash', width=3),
        name='EV (Conditional, +EV)',
        showlegend=True
    ),
    row=1, col=2
)

fig_ev.frames = frames

# --- Layout ---
fig_ev.update_layout(
    height=550,
    width=1100,
    title_text="Ineffective Policy Function vs Effective Policy Function<br><sup>No policy can overcome a negative edge, but with a positive edge, wealth can grow!</sup>",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, 'y': -0.17,
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

fig_ev.update_xaxes(title_text="Round", range=[0, n_steps],
                    showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig_ev.update_yaxes(title_text="Wealth", range=[0, 2000],
                    showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig_ev.show()



# **Remark:** Acting with an optimal policy function may be easy or difficult depending on the system - but the complexity of $\pi$ and its time variance does not make the system or game of incomplete information gambling 


# ---


# #### 3.) 🍅 Grocery Shopping and Trading
# 
# In some systems its easy to act with positive expected value (EV), with an edge (Grocery Shopping)
# 
# In some systems its hard to act with positive expected value (EV), with an edge (Trading)
# 
# ##### Just because it's hard to act with positive expected value (EV), it does **NOT** make the system *gambling* 
# 
# If you consider trading gambling - you must also consider grocery shopping gambling, let's discuss why
# 
# If you consider Grocery prices deterministic you must also consider stock prices deterministic
# 
# We know what the value of the stock prices are on the exchange just like how we know what they are when we go to the store
# 
# In either case, we don't know what **they are going to be in the future**
# 
# In either case, acting with positive EV isn't about randomness or uncertainty - its about your policy function


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Maximizing Expected Value (EV) Grocery Shopping
# 
# Relatively easy, the system is *uncertain* just like trading, we don't know what the price will be before we go into the store
# 
# We are still operating with a policy function to make optimal decisions under uncertainty,
# 
# Acting with positive EV is easy, pick the cheaper eggs, pick the bulk option for value you know you will use over time
# 
#  $$
#  \pi(x) =
#    \begin{cases}
#       x_{\text{min}}, & x \leq x^* \\
#       x_{\text{min}}, & x > x^*
#     \end{cases}
#  $$
#  
#  Where $x_{\text{min}}$ is the point of minimum cost (best price), and $x^*$ is the threshold where buying makes sense.
#  
#  This policy always buys at the minimum cost available (when it's needed).
#  
#  Our Grocery Shopping Edge is now positive.
#  
#  $$
#  \text{EV}_{\text{grocery}} = \mathbb{E}\big[ R \mid \pi \big] > 0
#  $$
#  
#  Where $\pi$ is the policy function (decision rule), and the expected value (EV) of grocery shopping under this policy is positive.
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Parameters ---
np.random.seed(42)
n_steps = 100
n_paths = 7  # number of stochastic sample paths
x_vals = np.linspace(-8, 8, 400)

# --- Piecewise cost function ---
def cost_function(x):
    x = np.array(x, dtype=float)
    flat_val = 7.0
    y = np.full_like(x, flat_val)
    inside = (x > -4) & (x < 4)
    if inside.any():
        # Quadratic between -4 and 4 with min = 5 at x = 4
        a = (7 - 5) / ((-4 - 4)**2)  # 2/64 = 0.03125
        y[inside] = a * (x[inside] - 4)**2 + 5
    y[x <= -4] = flat_val
    y[x >= 4] = flat_val
    return y.item() if np.isscalar(x) else y

# --- Cost baseline ---
y_max = 7.0
x_noise = np.random.normal(0, 1.5, n_steps)
y_costs = cost_function(x_noise)

# --- Generate stochastic cumulative savings sample paths ---
paths = []
for _ in range(n_paths):
    drift = np.random.uniform(0.02, 0.08)
    volatility = np.random.uniform(0.5, 1.2)
    noise = np.random.normal(0, volatility, n_steps)
    increments = drift + noise
    path = np.cumsum(increments)
    paths.append(path)

# Pick one arbitrary path to label as "EV Path"
ev_index = np.random.randint(0, n_paths)

# --- Ranges ---
y_cost_range = [min(y_costs) - 0.5, y_max + 0.5]
wealth_all = np.concatenate(paths)
wealth_range = [wealth_all.min() - 5, wealth_all.max() + 5]
x_range_wealth = [0, n_steps + 2]

# --- Helper to build one frame ---
def make_cost_fig(step):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Grocery Cost Function", "Cumulative Savings Sample Paths")
    )

    # Left: cost curve
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=cost_function(x_vals),
            mode='lines',
            line=dict(color='red', width=4),
            name='Cost Function'
        ),
        row=1, col=1
    )

    # Current stochastic point
    fig.add_trace(
        go.Scatter(
            x=[x_noise[step]],
            y=[y_costs[step]],
            mode='markers',
            marker=dict(color='orange', size=14, line=dict(color='white', width=1.5)),
            name='Current Cost'
        ),
        row=1, col=1
    )

    # Right: sample paths
    for i, path in enumerate(paths):
        color = 'magenta' if i == ev_index else 'orange'
        dash = 'dash' if i == ev_index else 'solid'
        width = 3 if i == ev_index else 2

        fig.add_trace(
            go.Scatter(
                x=np.arange(1, step+2),
                y=path[:step+1],
                mode='lines',
                line=dict(color=color, width=width, dash=dash),
                name="EV Path" if i == ev_index else "Sample Paths",
                opacity=0.7,
                showlegend=(i == ev_index or i == 0)  # only one orange label
            ),
            row=1, col=2
        )

    # Gridlines
    for c in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

    # Axis labels
    fig.update_xaxes(title_text="Grocery Quantity / Index", row=1, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=1, range=y_cost_range)
    fig.update_xaxes(title_text="Time Step", range=x_range_wealth, row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Savings", range=wealth_range, row=1, col=2)

    # Layout
    fig.update_layout(
        height=500,
        width=1000,
        title_text="Cost Minimization and Divergent Stochastic Savings Paths",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.22,
            xanchor="center",
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=110, t=70),
    )
    return fig

# --- Animation frames ---
frames = [
    go.Frame(
        data=make_cost_fig(step).data,
        layout=make_cost_fig(step).layout,
        name=str(step)
    )
    for step in range(n_steps)
]

# --- Initial figure ---
fig = make_cost_fig(0)
fig.frames = frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5,
        'y': -0.1,
        'xanchor': 'center',
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
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Is Grocery Shopping Gambling?
# 
# We can act with negative expected value (EV) or with a negative edge grocery shopping
# 
# Just because we can make suboptimal decisions with our policy function, it doesn't make the space gambling
# 
#  $$\pi(x) =
#   \begin{cases}
#      x_{\text{max}}, & x \geq x^* \\
#      x_{\text{max}}, & x < x^*
#    \end{cases}$$
# 
#  Where $x_{\text{max}}$ is the point of maximum cost, and $x^*$ is the necessary threshold.
# 
#  This policy always buys at maximum cost, even when it's unnecessary (below threshold).
# 
#  Our Grocery Shopping Edge is now negative
# 
#  $$
# \text{EV}_{\text{grocery}} = \mathbb{E}\big[ R \mid \pi \big] < 0
# $$
# 
# where $\pi$ is the policy function (decision rule) and the expected value (EV) of grocery shopping under this policy is negative.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Setup for negative EV scenario ---
neg_ev_n_steps = 100
neg_ev_x_vals = np.linspace(-8, 8, 400)

# Cost is always at its max
neg_ev_x_noise = np.full(neg_ev_n_steps, 7)  # index doesn't matter since cost always at max
neg_ev_y_costs = np.full(neg_ev_n_steps, y_max)

# --- Steep negative drift (~ -10 slope total) ---
# We can set drift so that total over 100 steps ≈ -1000
# so avg per step ≈ -10
neg_ev_drift = -10.0
neg_ev_volatility = 1.5
neg_ev_noise = np.random.normal(0, neg_ev_volatility, neg_ev_n_steps)
neg_ev_increments = neg_ev_drift + neg_ev_noise
neg_ev_path = np.cumsum(neg_ev_increments)

# Ranges for axes
neg_ev_wealth_range = [neg_ev_path.min() - 50, neg_ev_path.max() + 50]
neg_ev_x_range_wealth = [0, neg_ev_n_steps + 2]

# --- Helper to build a frame ---
def make_neg_ev_fig(step):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Grocery Cost Function", "Negative EV Savings Path")
    )

    # Left: cost curve, highlight only cost at max
    fig.add_trace(
        go.Scatter(
            x=neg_ev_x_vals,
            y=cost_function(neg_ev_x_vals),
            mode='lines',
            line=dict(color='gray', width=3, dash='dot'),
            name='Cost Function',
            opacity=0.4
        ),
        row=1, col=1
    )

    # Always mark the max cost point (show only y_max)
    fig.add_trace(
        go.Scatter(
            x=[neg_ev_x_noise[step]],
            y=[y_max],
            mode='markers',
            marker=dict(color='red', size=18, line=dict(color='white', width=2.5)),
            name='Always Max Cost'
        ),
        row=1, col=1
    )

    # Right: Single path with steep negative drift
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, step+2),
            y=neg_ev_path[:step+1],
            mode='lines',
            line=dict(color='crimson', width=5),
            name='Negative EV Path'
        ),
        row=1, col=2
    )

    # Gridlines and axes
    for c in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

    fig.update_xaxes(title_text="Grocery Quantity / Index", row=1, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=1, range=[y_max - 0.5, y_max + 0.5])
    fig.update_xaxes(title_text="Time Step", range=neg_ev_x_range_wealth, row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Savings", range=neg_ev_wealth_range, row=1, col=2)

    # Layout
    fig.update_layout(
        height=500,
        width=1000,
        title_text="Indefinite Negative EV: Always Paying Max Cost, Always Losing Money",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.18, xanchor="center", orientation="h",
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        ),
        margin=dict(l=50, r=20, b=100, t=70),
    )

    return fig

# --- Animation frames ---
neg_ev_frames = [
    go.Frame(
        data=make_neg_ev_fig(step).data,
        layout=make_neg_ev_fig(step).layout,
        name=str(step)
    )
    for step in range(neg_ev_n_steps)
]

# --- Initial figure ---
neg_ev_fig = make_neg_ev_fig(0)
neg_ev_fig.frames = neg_ev_frames

# --- Play button ---
neg_ev_fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'xanchor': 'center',
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

neg_ev_fig.show()


# Just because we can act with *negative EV* it doesn't make the system gambling
# 
# Effectively, we can lose as much money as we want, buying overpriced items, buying things we don't need, wasting, . . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Maximizing EV Trading
# 
# Is there one fixed way to maximize cost savings over time grocery shopping?  
# 
# **No**, acting optimally to accumulate max cost savings grocery shopping, the system changes over time (uncertain)
# 
# Trading oeprates in the same way, there is no one fixed way to accumulate max wealth trading, the system change over time (uncertain)
# 
# It is far more difficult to achieve an optimal policy $\pi$ for trading than it is an optimal or positive edge function for grocery shopping 
# 
# **It is only more difficult to act with positive expected value trading than it is grocery shopping - this does not make the system gambling**
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Shared parameters ---
n_policies = 100
policy_vals = np.linspace(0, 1, n_policies)
n_steps = 60
n_paths = 8
init_wealth = 1000
bet_unit = 25

policy_ylim_pad = 0.2
rng = np.random.default_rng(2025)

# Fixed policy for now (~middle)
fixed_policy_val = 0.6

# Extended list of EV functions, including more random/wiggly cases and various behaviors
def random_ev_fn(seed):
    rngr = np.random.default_rng(seed)
    amps = rngr.uniform(0.05, 0.25, size=5)
    freqs = rngr.choice([2, 4, 6, 8, 10, 12], size=5)
    phases = rngr.uniform(0, 2*np.pi, size=5)
    offset = rngr.uniform(-0.15, 0.15)
    quad = rngr.uniform(-1, 1)
    base = 0
    def f(p):
        val = base
        for a, f_, ph in zip(amps, freqs, phases):
            val += a * np.sin(f_*np.pi*p + ph)
        val += offset
        val += quad * (p-0.5)**2
        return np.clip(val, -0.35, 0.35)
    return f

ev_func_types = [
    dict(name="Optimal (positive edge)",
         fn=lambda p: -2.8 * (p - 0.58) ** 2 + 0.20, color="royalblue"),
    dict(name="Flat (no edge)",
         fn=lambda p: np.zeros_like(p), color="gray"),
    dict(name="Negative edge",
         fn=lambda p: -1.5 * (p - 0.72) ** 2 - 0.10, color="indianred"),
    dict(name="Sharp optimal",
         fn=lambda p: -10.0 * (p - 0.45) ** 2 + 0.25, color="seagreen"),
    dict(name="Wavy/uncertain",
         fn=lambda p: 0.12 * np.sin(12*p) - 0.5*(p-0.5)**2, color="orange"),
    dict(name="Random EV 1",
         fn=random_ev_fn(1001), color="violet"),
    dict(name="Random EV 2",
         fn=random_ev_fn(1234), color="gold"),
    dict(name="Random EV 3",
         fn=random_ev_fn(2025), color="lightblue"),
    dict(name="Random EV 4",
         fn=random_ev_fn(42), color="magenta"),
    dict(name="Deeply Negative",
         fn=lambda p: -0.12 - 0.5 * (p-0.6)**2, color="firebrick"),
    dict(name="Edge Jumps",
         fn=lambda p: np.where(p < 0.4, 0.15, -0.11 + 0.10 * np.sin(5*p)), color="lightgreen"),
    dict(name="Noisy Center Peak",
         fn=lambda p: 0.22 * np.exp(-20*(p-0.5)**2) + 0.08*np.sin(16*p), color="aqua"),
]

# Generate random cycle of EV types to create uncertainty, fixed for reproducibility
cycle_indices = np.arange(len(ev_func_types))
repeat_needed = int(np.ceil(n_steps / len(ev_func_types)))
rng_for_cycle = np.random.default_rng(60060)
all_cycles = []
for _ in range(repeat_needed):
    shuffled = rng_for_cycle.permutation(cycle_indices)
    all_cycles.append(shuffled)
cycle = np.concatenate(all_cycles)[:n_steps]

policy_curves = []
policy_star_traces = []
ev_vals_over_time = []
policy_ev_over_time = []

for step in range(n_steps):
    evf = ev_func_types[cycle[step]]
    evs = evf['fn'](policy_vals)
    policy_ev = evf['fn'](fixed_policy_val)
    ev_vals_over_time.append(evs)
    policy_ev_over_time.append(policy_ev)
    curve = go.Scatter(
        x=policy_vals, y=evs,
        mode='lines',
        line=dict(color=evf['color'], width=3),
        name=None,
        xaxis='x', yaxis='y',
        showlegend=False
    )
    star = go.Scatter(
        x=[fixed_policy_val], y=[policy_ev],
        mode='markers+text',
        marker=dict(symbol='star', size=24, color='gold', line=dict(color='black', width=2)),
        text=[f'Policy {fixed_policy_val:.2f}'],
        textposition='top center',
        showlegend=False,
        xaxis='x', yaxis='y'
    )
    policy_curves.append(curve)
    policy_star_traces.append(star)

# --- Sample Wealth Path Simulation, evolving *one step at a time*, drawing strongly from the left policy EV function ---

# For each path, we will keep the array of wealth across all frames so each frame
# shows their history up to that round:
simulated_path_wealths = np.full((n_paths, n_steps), np.nan)
simulated_path_wealths[:, 0] = init_wealth

for step in range(1, n_steps):
    # Draw current EV function, for computing implied win prob at fixed policy
    evf = ev_func_types[cycle[step]]
    this_ev = evf['fn'](fixed_policy_val)
    if fixed_policy_val > 0.01 and np.abs(bet_unit * fixed_policy_val) > 1e-6:
        win_prob = (this_ev / (bet_unit * fixed_policy_val) + 1) / 2
        win_prob = np.clip(win_prob, 0, 1)
    else:
        win_prob = 0.5
    for pidx in range(n_paths):
        prev_wealth = simulated_path_wealths[pidx, step-1]
        # Make a bet with probability = policy (for now, fixed_policy_val); if bet, use win_prob
        do_bet = rng.random() < fixed_policy_val
        if do_bet:
            win = rng.random() < win_prob
            change = bet_unit if win else -bet_unit
        else:
            change = 0
        simulated_path_wealths[pidx, step] = prev_wealth + change

# ----------- ANIMATION FRAMES (EV function left, sample wealth paths right) --------------
frames = []
min_ev_glob = min(np.min(ev) for ev in ev_vals_over_time) - 0.05 - policy_ylim_pad
max_ev_glob = max(np.max(ev) for ev in ev_vals_over_time) + 0.05 + policy_ylim_pad

for i in range(1, n_steps):
    frame_data = []
    # Left panel: update curve & star
    frame_data.append(policy_curves[i])
    frame_data.append(policy_star_traces[i])

    # Right panel: plot all wealth paths up to i'th round, each path evolves step-by-step
    for pidx in range(n_paths):
        opacity = 1.0 - pidx * 0.07
        frame_data.append(
            go.Scatter(
                x=np.arange(i + 1),
                y=simulated_path_wealths[pidx, :i + 1],
                mode='lines',
                line=dict(color='yellow', width=2),  # Changed from green to yellow
                opacity=opacity,
                xaxis='x2', yaxis='y2',
                showlegend=False,
                name=None
            )
        )
    frames.append(go.Frame(
        name=f"f{i}",
        data=frame_data,
        layout=go.Layout(
            xaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
            yaxis=dict(
                range=[min_ev_glob, max_ev_glob],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)'
            ),
            xaxis2=dict(range=[0, n_steps], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
            yaxis2=dict(range=[0, 2000], showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        )
    ))

# ----------- FIGURE: 1 row x 2 columns (EV function left, sample paths right) ---------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'EV Function (Randomly Changing Edge, Same Policy)',
        f'Sample Wealth Paths (Fixed Policy, EV Function on Left)'
    ],
    horizontal_spacing=0.15
)
# Left: EV function at frame 0
fig.add_trace(policy_curves[0], row=1, col=1)
fig.add_trace(policy_star_traces[0], row=1, col=1)
fig.update_xaxes(title_text="Policy (P(bet))", range=[0, 1],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=1)
fig.update_yaxes(
    title_text="Expected Value per Play",
    range=[min_ev_glob, max_ev_glob],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=1)

# Right: Wealth paths start with only initial wealth at round 0
for pidx in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[init_wealth],
            mode='lines',
            line=dict(color='yellow', width=2),  # Changed from green to yellow
            opacity=1.0 - pidx * 0.07,
            showlegend=False,
            name=None
        ),
        row=1, col=2
    )
fig.update_xaxes(title_text="Round", range=[0, n_steps],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=2)
fig.update_yaxes(title_text="Wealth", range=[0, 2000],
                 showgrid=True, gridcolor='rgba(128,128,128,0.3)',
                 row=1, col=2)

# ----------- Layout/Attachment for animation -----------
fig.frames = frames
fig.update_layout(
    height=600,
    width=1000,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text="How Different Random/Changing Edge Functions Alter Outcomes (Fixed Policy)",
    showlegend=False,  # legend removed
    updatemenus=[{
        'type': 'buttons',
        'x': 0.07, 'y': -0.13,
        'showactive': False,
        'buttons': [dict(
            label='Play',
            method='animate',
            args=[None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        )]
    }]
)
fig.show()



# At one point in time, $\pi$ is the optimal policy function and we can accumulate significant wealth
# 
# But over time the policy space changes, and we don't adjust our policy accumulating at first some positive EV then net zero and negative EV
# 
# **Final Remark:** It is difficult to act with an optimal policy function or even one with positive EV when trading, unlike grocery shopping, that does not make the space gambling - they are both uncertain systems, and our objective is to minimize cost or maximize wealth in the face of uncertainty in either system.  There is not *fixed negative edge* against the agent acting in either system, **neither Trading or Grocery Shopping is gambling**.


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Random variables define a set of possible outcomes with accompanying probabilities and likelihoods
# - Random variable statistics, probabilities, and distributions converge by the Law of Large Numbers (LLN)
# - We tend to model uncertain events (non-stationary events or processes) as random variables *even though* quantities lack convergence, leading to a lot of confusion about the efficacy of statistics in the real world for both students and professionals alike
# - Games of chance and incomplete information are examples systems that impact our wealth over a series of plays or time
# - In either case, the expected value dictates our wealth path trajectory as we **can never predict** the outcome of any one event
# - Depending on the structure of the system, our actions may or may not influence our wealth path trajectory - if they don't and there is fixed negative edge, this is gambling - if they do then we are operating in a game of incomplete information and we influence wealth path trajectory
# - Difficulty of optimizing your optimal policy function does not imply the space itself is gambling, gambling is defined herein as willfully engaging in a system with a fixed negative edge against you, the unconditional edge is equivalent to the conditional one on even an optimal policy function 
# - In other words, no matter what you do you will statistically lose all of your money if you continue to play 
# - If you consider trading gambling due to the difficulting in constructing a productive policy function, then you must necessarily also consider grocery shopping, buying gas for your car, or any other uncertain price or event that enables you to accumulate a wealth path of cost savings "gambling" - and that is outrageous
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
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

