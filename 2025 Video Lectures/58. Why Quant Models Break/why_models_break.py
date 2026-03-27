# ### 🔨 Why Quant Models Break
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
# 
# #### 1.) 🎲 Random and Uncertain Events
# 
# - Random Variables and Uncertain Events
# 
# - Example: Weighted Coin Flip
# 
# - Example: Sports Betting (Giants vs Broncos)
# 
# #### 2.) 🔨 Why Quant Models Break
# 
# - Modeling Stock Returns
# 
# - Why Quant Models Break
# 
# - Black Swan Events
# 
# #### 3.) 📈 Purpose of Modeling
# 
# - Making Optimal Decisions under Uncertainty
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📈 Random Variables and Uncertain Events
# 
# The stock market is not *random* like a coin flip, dice roll, or even casino games like roulette - it is *uncertain*, a major point of confusion for students and folks new to the space
# 
# Examples:
# 
# - Coin flip
# - Dice roll
# - Roulette
# 
# It is a composite of *deterministic* and *random* components, like *random* systems, the outcome of any one event (investment, hedge, trade) is always **unknown**
# 
# - Poker
# - Election betting
# - Sports betting
# - Trading
# 
# Importantly, statistics we use to measure things like returns or volatility **DO NOT CONVERGE** 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Random Variables 
# 
# Random variables are defined in terms of a distribution
# 
# $$X \sim Ber(.7) \quad\quad Y \sim Bin(10,.7) \quad\quad Z \sim N(0,1)$$
# 
# Commonly, these distributions may represent
# 
# - Coin Flips (Bernoulli )
# - Number of coin flips out of 10 that were heads (Binomial)
# - Distribution of average coin flips (roughly, Central Limit Theorem (CLT))


import numpy as np
import plotly.graph_objects as go
from math import comb
from plotly.subplots import make_subplots

# Create figure with 3 subplots side by side
fig = make_subplots(rows=1, cols=3, subplot_titles=('Bernoulli Distribution', 'Binomial Distribution', 'Normal Distribution'))

# Bernoulli PMF (p=0.7)
x_bern = [0, 1]
p = 0.7
pmf_bern = [1-p, p]
fig.add_trace(go.Bar(
    x=x_bern,
    y=pmf_bern,
    name='Bernoulli(p=0.7)',
    width=0.3,
    marker_color='#00ffff'  # neon cyan
), row=1, col=1)

# Binomial PMF (n=10, p=0.7) 
n = 10
x_bin = np.arange(0, n+1)
pmf_bin = [comb(n,k) * (p**k) * ((1-p)**(n-k)) for k in x_bin]
fig.add_trace(go.Bar(
    x=x_bin,
    y=pmf_bin,
    name=f'Binomial(n={n},p=0.7)',
    width=0.3,
    marker_color='#39ff14'  # neon green
), row=1, col=2)

# Normal distribution
x_norm = np.linspace(-4, 4, 100)
pdf_norm = 1/np.sqrt(2*np.pi) * np.exp(-x_norm**2/2)
fig.add_trace(go.Scatter(
    x=x_norm,
    y=pdf_norm,
    name='Normal(0,1)',
    mode='lines',
    line=dict(color='#ff1493', width=3)  # neon pink
), row=1, col=3)

# Update layout
fig.update_layout(
    height=400,
    title_text='Probability Distributions',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(title_text='x', showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Probability', showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

fig.show()



# Random variables represent a set of possible outcomes (either a specific outcome (discrete) or a range of outcomes (continuous)) with probabilities of occuring
# 
# **Critically**, these probabilities **DO NOT CHANGE** and neither does the distribution defining the random variable


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Random Variables and Frequency Interpretations
# 
# Statistics for random variables that draw from the same distribution converge by the Law of Large Numbers (LLN)
# $$\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^n X_i = \mathbb{E}[X] \quad \text{almost surely}$$
# 
# Theoretically,
# 
# If we have a weighted coin that has **heads** appear 70\% of the time, if we flip a coin 10 times, we should see on average 
# 
# $$\mathbb{E}[X] = n p = 10 * 70\% = 7 \text{ heads}$$


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# --- Setup ---
n_trials = 1000  # Number of sets of 10 flips
p_heads = 0.7  # Probability of heads from weighted coin
expected_heads = 10 * p_heads  # Expected number of heads in 10 flips

# Simulate sets of 10 flips
np.random.seed(40)
all_flips = np.random.binomial(n=10, p=p_heads, size=n_trials)

# Compute running mean
running_means = np.cumsum(all_flips) / np.arange(1, n_trials + 1)

# Calculate theoretical PMF for 10 flips
x = np.arange(11)  # 0 to 10 heads possible
theoretical_pmf = np.array([math.comb(10, k) * (p_heads**k) * ((1-p_heads)**(10-k)) for k in x])

# --- Create frames ---
frames = []
for i in range(1, n_trials):
    # Calculate empirical PMF
    values, counts = np.unique(all_flips[:i], return_counts=True)
    empirical_pmf = np.zeros(11)
    empirical_pmf[values] = counts/i
    
    frames.append(go.Frame(
        data=[
            # Theoretical PMF
            go.Bar(
                x=x,
                y=theoretical_pmf,
                marker_color='skyblue',
                opacity=0.4,
                name='Theoretical PMF'
            ),
            # Empirical PMF
            go.Bar(
                x=x,
                y=empirical_pmf,
                marker_color='salmon',
                opacity=0.9,
                name='Empirical PMF'
            ),
            # Running mean convergence
            go.Scatter(
                x=np.arange(1, i+1),
                y=running_means[:i],
                mode='lines',
                line=dict(color='red', width=2),
                name='Empirical Mean'
            ),
            go.Scatter(
                x=np.arange(1, i+1),
                y=[expected_heads] * i,
                mode='lines',
                line=dict(color='white', width=1, dash='dot'),
                name='Expected Value'
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
    subplot_titles=('Distribution of Heads in 10 Flips', 'Convergence to Expected Value'),
    column_widths=[0.4, 0.6]
)

# Initial theoretical and empirical PMF
fig.add_trace(
    go.Bar(
        x=x,
        y=theoretical_pmf,
        marker_color='skyblue',
        opacity=0.4,
        name='Theoretical PMF'
    ),
    row=1, col=1
)

# Calculate initial empirical PMF
initial_empirical = np.zeros(11)
initial_empirical[all_flips[0]] = 1

fig.add_trace(
    go.Bar(
        x=x,
        y=initial_empirical,
        marker_color='salmon',
        opacity=0.9,
        name='Empirical PMF'
    ),
    row=1, col=1
)

# Initial convergence line
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[running_means[0]],
        mode='lines',
        line=dict(color='red', width=2),
        name='Empirical Mean'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[expected_heads],
        mode='lines',
        line=dict(color='white', width=1, dash='dot'),
        name='Expected Value'
    ),
    row=1, col=2
)

# --- Animation & Layout ---
fig.frames = frames

fig.update_layout(
    height=500,
    width=1000,
    title_text="Weighted Coin: Convergence to Expected Number of Heads (10 Flips)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    barmode='overlay',
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
fig.update_yaxes(range=[0, 0.4], row=1, col=1)  # Set y-range for PMF plot
fig.update_yaxes(range=[0, 10], row=1, col=2)  # Set y-range for convergence plot
fig.update_xaxes(title_text='Number of Heads', row=1, col=1)
fig.update_xaxes(title_text='Number of Trials', row=1, col=2)

fig.show()



#  The occurence of any event can be represented as a Bernoulli, by the Law of Large Numbers (LLN) states that the sample proportion converges to the true probability
#  
#  For a $Ber(p)$ distribution where $X_i ~ Ber(p)$:
#  $$\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^n X_i = p \quad \text{almost surely}$$ 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Sports Betting and Belief Interpretations
# 
# Economics tells us that supply and demand create an equilibrium price, so it should reflect the best probabilities of an event occuring. . .right?
# 
# Well, that model also assumes agents act rationally - and in the behavioral finance literature, and structurally in the market this isn't true (vol is overpriced)
# 
# Moreover, this system is *uncertain* if we back out implied probabilities, they just reflect an outlook based on bets - **NOT** actual probabilities
# 
# ##### Anyone see the Giants game the other day?


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Load Giants image
img = Image.open("giants.jpg")

# Simulation parameters
n_steps = 400
event_step = 350
np.random.seed(42)

# Initialize
pG = np.zeros(n_steps)
pB = np.zeros(n_steps)
pG[0] = 0.5
pB[0] = 0.5

# ---  PHASE 1: Giants climb gradually with volatility ---
for i in range(1, 100):
    change = np.random.normal(0.01, 0.02)
    pG[i] = np.clip(pG[i - 1] + change, 0, 1)
    pB[i] = 1 - pG[i]

# ---  PHASE 2: Giants dominate to 99% ---
for i in range(100, 180):
    change = np.random.normal(0.02, 0.015)
    pG[i] = np.clip(pG[i - 1] + change, 0, 1)
    pB[i] = 1 - pG[i]

# ---  PHASE 3: Sudden collapse to near zero ---
for i in range(180, 220):
    change = np.random.normal(-0.05, 0.03)
    pG[i] = np.clip(pG[i - 1] + change, 0, 1)
    pB[i] = 1 - pG[i]

# ---  PHASE 4: Wild rally back to 93% ---
for i in range(220, 300):
    change = np.random.normal(0.03, 0.04)
    pG[i] = np.clip(pG[i - 1] + change, 0, 1)
    pB[i] = 1 - pG[i]

# ---  PHASE 5: Final chaos before event ---
for i in range(300, event_step):
    change = np.random.normal(-0.015, 0.05)
    pG[i] = np.clip(pG[i - 1] + change, 0, 1)
    pB[i] = 1 - pG[i]

# ---  EVENT REALIZATION: final collapse (Giants lose) ---
pG[event_step:] = 0.0
pB[event_step:] = 1.0

# ---  Create animation frames ---
frames = []
for i in range(1, n_steps):
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=np.arange(i + 1),
                    y=pG[:i + 1],
                    mode="lines",
                    line=dict(color="blue", width=3),
                    name="Giants",
                ),
                go.Scatter(
                    x=np.arange(i + 1),
                    y=pB[:i + 1],
                    mode="lines",
                    line=dict(color="orange", width=3),
                    name="Broncos",
                ),
                go.Scatter(
                    x=[event_step, event_step],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="yellow", width=2, dash="dot"),
                    name="Event (Final Whistle)",
                ),
            ],
            name=f"frame{i}",
        )
    )

# ---  Create subplot figure ---
fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.65, 0.35],
    subplot_titles=["Giants vs Broncos — Win Probability", ""],
)

# Add starting traces
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[pG[0]],
        mode="lines",
        line=dict(color="blue", width=3),
        name="Giants",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[pB[0]],
        mode="lines",
        line=dict(color="orange", width=3),
        name="Broncos",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[event_step, event_step],
        y=[0, 1],
        mode="lines",
        line=dict(color="yellow", width=2, dash="dot"),
        name="Event (Final Whistle)",
    ),
    row=1,
    col=1,
)

# ---  Add image to right subplot ---
fig.add_layout_image(
    dict(
        source=img,
        xref="x2",
        yref="y2",
        x=-1,
        y=4,
        sizex=7,
        sizey=7,
        xanchor="left",
        yanchor="top",
        layer="below",
    )
)

# Hide axes on the image subplot
fig.update_xaxes(visible=False, row=1, col=2)
fig.update_yaxes(visible=False, row=1, col=2)

# ---  Style the layout ---
fig.update_layout(
    height=500,
    width=1200,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    xaxis=dict(title="Time", range=[0, n_steps]),
    yaxis=dict(title="4th Qtr Win Probability", range=[-.05, 1.05]),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 30, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                }
            ],
        }
    ],
)

# Assign frames
fig.frames = frames

fig.show()



# It was *never actually* 99.8\%, that is what the money implied, your payout would have been massive with a bet on the Broncos. . .
# 
# **Crucially**, these probabilities reflect relative payouts and market expectation, **NOT** a frequency based likelihood of an outcome
# 
#     - We can trade these contracts and others and if we are correct *on average* we will make money


# ###### ______________________________________________________________________________________________________________________________________


# ##### But Roman, Betting Markets *Predict* Outcomes
# 
# Naively regressing implied probabilities on outcomes (or analyzing relative proportions) suggests this
# 
# I say, **WHAT ABOUT WHEN IT DOESN'T** - the market doesn't know the Giants like I do, based on my expertise as a fan I **KNOW** they are going to lose. . .that's my edge


# ###### ______________________________________________________________________________________________________________________________________


# ##### Simulating the 4th Quarter 10,000 Times
# 
# We can't actually do this, these probabilities are *just a model* to try to assess the relative likelihood of outcomes
# 
# But if we could, we would be able to see the *true frequency* of the outcome and how well the implied probabilities forecast it (we can never do this)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Controls
# -----------------------------
n_total_draws   = 10_000
n_display_paths = 60
n_steps         = 140
true_p_giants   = 0.70   # actual long-run win probability
start_prob      = 0.998   # market implied start-of-4Q
sigma           = 0.06
jump_prob       = 0.07
jump_sigma      = 0.22
mean_revert_k   = 0.08

np.random.seed(42)

# -----------------------------
# Helper: generate one noisy path
# -----------------------------
def generate_path(n_steps, p0, true_p, sigma, jump_prob, jump_sigma, k):
    p = np.empty(n_steps)
    p[0] = p0
    for t in range(1, n_steps):
        drift = k * (true_p - p[t-1])
        noise = np.random.normal(0, sigma)
        jump  = np.random.normal(0, jump_sigma) if np.random.rand() < jump_prob else 0.0
        p[t]  = np.clip(p[t-1] + drift + noise + jump, 0, 1)
    return p

# -----------------------------
# Simulate 10,000 outcomes for actual result bar chart
# -----------------------------
giants_wins = np.random.binomial(1, true_p_giants, size=n_total_draws).sum()
broncos_wins = n_total_draws - giants_wins
giants_pct = giants_wins / n_total_draws
broncos_pct = broncos_wins / n_total_draws

# -----------------------------
# Animated sample paths (Giants + Broncos)
# -----------------------------
paths_G = np.vstack([
    generate_path(n_steps, start_prob, true_p_giants, sigma, jump_prob, jump_sigma, mean_revert_k)
    for _ in range(n_display_paths)
])
paths_B = 1 - paths_G

# Event realization: hard collapse to 0/1 reflecting final outcomes (≈70/30)
visible_outcomes = np.random.binomial(1, true_p_giants, size=n_display_paths)
paths_G[:, -1] = visible_outcomes
paths_B[:, -1] = 1 - visible_outcomes

# -----------------------------
# Figure layout
# -----------------------------
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    specs=[[{"type": "xy"}, {"type": "bar"}]],
    subplot_titles=[
        "Giants vs Broncos — 4th Quarter Sample Paths",
        "Implied vs Actual Win Probability"
    ]
)

# Giants paths
for i in range(n_display_paths):
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="lines",
            line=dict(width=1.5, color="blue"),
            opacity=0.7,
            showlegend=(i == 0),
            name="Giants"
        ),
        row=1, col=1
    )

# Broncos paths
for i in range(n_display_paths):
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="lines",
            line=dict(width=1.5, color="orange"),
            opacity=0.5,
            showlegend=(i == 0),
            name="Broncos"
        ),
        row=1, col=1
    )

# Reference lines
fig.add_trace(
    go.Scatter(
        x=[0, n_steps - 1],
        y=[0.5, 0.5],
        mode="lines",
        line=dict(width=1, dash="dot", color="white"),
        showlegend=False
    ),
    row=1, col=1
)

# Event line (final whistle)
fig.update_layout(
    shapes=[
        dict(
            type="line",
            xref="x1", yref="y1",
            x0=n_steps - 1, x1=n_steps - 1,
            y0=0, y1=1,
            line=dict(color="yellow", width=2, dash="dot")
        )
    ]
)

# -----------------------------
# Frames for animation
# -----------------------------
frames = []
x_full = np.arange(n_steps)

for t in range(1, n_steps):
    frame_traces = []

    # Giants lines
    for i in range(n_display_paths):
        frame_traces.append(
            go.Scatter(
                x=x_full[:t + 1],
                y=paths_G[i, :t + 1],
                mode="lines",
                line=dict(width=1.5, color="blue"),
                opacity=0.7,
                showlegend=False
            )
        )
    # Broncos lines
    for i in range(n_display_paths):
        frame_traces.append(
            go.Scatter(
                x=x_full[:t + 1],
                y=paths_B[i, :t + 1],
                mode="lines",
                line=dict(width=1.5, color="orange"),
                opacity=0.5,
                showlegend=False
            )
        )

    # Add bar chart only at the end
    if t == n_steps - 1:
        frame_traces.append(
            go.Bar(
                x=["Giants (Start Implied)", "Broncos (Start Implied)",
                   "Giants (Actual)", "Broncos (Actual)"],
                y=[start_prob * 100, (1 - start_prob) * 100,
                   giants_pct * 100, broncos_pct * 100],
                marker_color=["blue", "orange", "blue", "orange"],
                text=[
                    f"{start_prob * 100:.1f}%",
                    f"{(1 - start_prob) * 100:.1f}%",
                    f"{giants_pct * 100:.1f}%",
                    f"{broncos_pct * 100:.1f}%"
                ],
                textposition="auto",
                name="Win Probability (%)"
            )
        )

    frames.append(go.Frame(data=frame_traces, name=f"frame{t}"))

# -----------------------------
# Layout polish
# -----------------------------
fig.update_xaxes(title="Time", range=[0, n_steps - 1], row=1, col=1)
fig.update_yaxes(
    title="Implied Win Probability",
    range=[0, 1],
    tickformat=".0%",
    row=1, col=1
)
fig.update_yaxes(
    title="Probability (%)",
    range=[0, 105],  # give headroom for labels
    row=1, col=2
)

fig.update_layout(
    height=520,
    width=1100,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [
                None,
                {"frame": {"duration": 5, "redraw": True},
                 "fromcurrent": True,
                 "transition": {"duration": 0}}
            ]
        }]
    }]
)
fig.frames = frames

fig.add_trace(
    go.Bar(
        x=["Giants (Start Implied)", "Broncos (Start Implied)",
           "Giants (Actual)", "Broncos (Actual)"],
        y=[start_prob * 100, (1 - start_prob) * 100,
           giants_pct * 100, broncos_pct * 100],
        marker_color=["blue", "orange", "blue", "orange"],
        text=[
            f"{start_prob * 100:.1f}%",
            f"{(1 - start_prob) * 100:.1f}%",
            f"{giants_pct * 100:.1f}%",
            f"{broncos_pct * 100:.1f}%"
        ],
        textposition="auto",
        name="Win Probability (%)"
    ),
    row=1, col=2
)


fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Moral of the Story
# 
# We need some way to **assess** the likelihood of certain events occuring (and in a continuous sense, this is not easy!)
# 
# We tend to model uncertain events (like stock returns) as *random variables* to try to accomplish this, even though they aren't *actually* random variables in the frequentist sense
# 
# The market **DOES NOT** price things *fairly* in equilibrium **ALL THE TIME** - otherwise nothing would trade (EMH is nonsense)


# ---


# #### 2.) 🔨 Why Quant Models Break
# 
# ##### Modeling Stock Returns
# 
# Stock returns are an example of an uncertain event (along with trades, investments, etc. . .)
# 
# To assess the relative likelihood of extreme events we can *try* to measure it from historic data
# 
# Similar to how we would look at a team's W/L record or implied probability from the market to try to inform our outcome
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Example: NVDA Stock Returns
# 
# Let's observe the historic distribution of NVDA and see if we can make an optimal decision about investing in the stock


import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load and prepare NVDA returns
df = pd.read_csv("NVDA_returns.csv")
df["Returns"] = df["Close"].pct_change() * 100
returns = df["Returns"].dropna().values

# Calculate percentiles and means
percentile_25 = np.percentile(returns, 25)
percentile_50 = np.percentile(returns, 50)
returns_below_25 = returns[returns <= percentile_25]
returns_below_50 = returns[returns <= percentile_50]
returns_above_50 = returns[returns > percentile_50]
mean_below_25 = np.mean(returns_below_25)
mean_below_50 = np.mean(returns_below_50)
mean_above_50 = np.mean(returns_above_50)

# Create histogram with normalized y-axis
fig = go.Figure()

# Add normalized histogram of returns
fig.add_trace(
    go.Histogram(
        x=returns,
        nbinsx=60,
        marker_color="#00ff00",
        opacity=0.6,
        name="Returns Distribution",
        histnorm='probability density'  # Normalize y-axis
    )
)

fig.add_trace(
    go.Scatter(
        x=[percentile_50, percentile_50],
        y=[0, 0.3],  # Adjusted y range for normalized scale
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='50th Percentile'
    )
)

# Update layout
fig.update_layout(
    title=f"NVDA Return Distribution<br>Mean Return Below 50th Percentile: {mean_below_50:.2f}%<br>Mean Return Above 50th Percentile: {mean_above_50:.2f}%<br><br>",
    yaxis_title="Probability Density",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)", 
    font=dict(color="white"),
    showlegend=True
)

fig.show()



# 50% of the time we will have positive returns and 50% of the time we will have negative returns, and the positive ones outweight the negative!
# 
# $$\mathbb{E}[R] = P(\text{loss}) \times \mathbb{E}[R|\text{loss}] + P(\text{win}) \times \mathbb{E}[R|\text{win}] = 0.5 \times (-1.9\%) + 0.5 \times 3.84\% = 0.97\%$$
# 
# 
# 
# This is positive expected value (EV), there is edge in this trade!  Bet the house!
# 
# **Hold on, hold on, this isn't quite how modeling works**


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### *Problem* Model informed values are not enough!  We need to understand how they work and the goal of our decision making!
# 
# Let's take a look at our model (1999 - 2025) and what happens in the future (2026)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load and prepare NVDA returns ---
df = pd.read_csv("NVDA_returns.csv")
df["Returns"] = df["Close"].pct_change() * 100
returns = df["Returns"].dropna().values

# --- Define distribution regimes ---
mean, std = np.mean(returns), np.std(returns)

middle_mask = (returns > mean - .15 * std) & (returns < mean + 3.25 * std)
middle_returns = returns[middle_mask]

bottom_30_threshold = np.percentile(returns, 80)
bottom_returns = returns[returns < bottom_30_threshold]

# --- Simulation parameters ---
n_steps = 100
n_paths = 5
start_price = 185
dates = pd.date_range(start="2026-01-01", periods=n_steps + 1, freq="D")

# --- Simulate multiple paths and store draws ---
paths, draws = [], []

for p in range(n_paths):
    path = [start_price]
    drawn_returns = []
    for i in range(n_steps):
        if i < 30:  # regime 1: middle
            ret = np.random.choice(middle_returns)
            regime = "middle"
        else:  # regime 2: mostly bottom
            if np.random.rand() < 0.9:
                ret = np.random.choice(bottom_returns)
                regime = "bottom"
            else:
                ret = np.random.choice(returns)
                regime = "random"
        path.append(path[-1] * (1 + ret / 100))
        drawn_returns.append((ret, regime))
    paths.append(path)
    draws.append(drawn_returns)

# --- Compute histogram once (fixed bins) ---
hist_vals, bin_edges = np.histogram(returns, bins=60)

# --- Create figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("NVDA Return Distribution (Model)", "NVDA Sample Paths (Actual, 2026)")
)

# --- Static histogram bars (base layer) ---
fig.add_trace(
    go.Bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=hist_vals,
        marker_color="#00ff00",
        opacity=0.4,
        name="Returns Distribution"
    ),
    row=1, col=1
)

# --- Initialize sample path traces ---
for p in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=[dates[0]],
            y=[paths[p][0]],
            mode="lines",
            line=dict(width=2),
            name=f"Path {p+1}"
        ),
        row=1, col=2
    )

# --- Create frames ---
frames = []
for i in range(1, n_steps + 1):
    # Highlighted bin(s)
    frame_highlights = np.zeros_like(hist_vals)
    for p in range(n_paths):
        ret_val, regime = draws[p][i - 1]
        bin_idx = np.digitize(ret_val, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(hist_vals) - 1)
        frame_highlights[bin_idx] += 1  # mark bin as active

    # Prepare bar colors
    bar_colors = ["#00ff00" if h == 0 else "#ff1493" for h in frame_highlights]

    frame_data = []
    # Histogram highlight update
    frame_data.append(
        go.Bar(
            x=(bin_edges[:-1] + bin_edges[1:]) / 2,
            y=hist_vals,
            marker_color=bar_colors,
            opacity=0.8
        )
    )

    # Add all path updates
    for p in range(n_paths):
        frame_data.append(
            go.Scatter(
                x=dates[:i+1],
                y=paths[p][:i+1],
                mode="lines",
                line=dict(width=2),
                showlegend=False
            )
        )

    frames.append(go.Frame(data=frame_data, name=str(i)))

# --- Layout ---
fig.update_layout(
    height=600,
    showlegend=False,  # remove legend
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [
                None,
                {
                    "frame": {"duration": 80, "redraw": True},
                    "fromcurrent": True,
                    "mode": "immediate"
                }
            ]
        }]
    }],
    xaxis=dict(title="Returns (%)"),
    yaxis=dict(title="Frequency", range=[0, 1000]),  # Set y-axis range to 1000
    xaxis2=dict(title="Date"),
    yaxis2=dict(title="Price", autorange=True)  # dynamic y-axis
)

fig.frames = frames
fig.show()



# But the distribution is relatively symmetric?  We had positive expected value (EV), an edge?  What happened?
# 
# ##### Our Quant Model Broke


# ###### ______________________________________________________________________________________________________________________________________


# ##### Why Quant Models Break
# 
# Our model was not effectively capturing the dynamics, just like the market implied probability from the Giants game
# 
# The likelihood of observing the events we did was **drastically underestimated**
# 
# The entire space is *dynamic*, some processes drive others, each process changes over time - nothing is stable or constant!
# 
# Assumptions can be correct, then incorrect, not violated, then violated - it's quite literally a full time job and why we all still have one!


import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 150_000
n_frames = 60
x_range = np.linspace(-6, 6, 300)

# --- Modified Ornstein–Uhlenbeck process for persistent high variance ---
def ou_process(theta=1.0, mu=3.0, sigma=1.0, dt=0.1, n_steps=n_frames):
    """
    OU process with high mean-reversion level (mu=3) and lower mean reversion speed (theta=1)
    to make variance spike early and remain elevated.
    """
    x = np.zeros(n_steps)
    x[0] = 0.5  # start low, then rise
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx
    return np.maximum(x, 0.1)  # keep variance positive

# Generate variance and mean paths
time_points = np.linspace(0, n_frames/10, n_frames)
variance_path = ou_process()

# --- Modified mean path: stays mostly negative ---
means = np.concatenate([
    np.linspace(3, 0, n_frames//5),    # move quickly to negative
    np.linspace(-2, 0, 3*n_frames//5), # stay negative for most of the time
    np.linspace(-1, 0, n_frames//5)    # end slightly less negative
])

# --- Create Figure with 2-column layout ---
fig = make_subplots(
    rows=3, cols=2,
    column_widths=[0.55, 0.45],  # Left column wider for detail
    row_heights=[0.33, 0.33, 0.34],
    specs=[
        [{"type": "xy"}, {"rowspan": 3, "type": "xy"}],  # sample path spans all rows
        [{"type": "xy"}, None],
        [{"type": "xy"}, None]
    ],
    subplot_titles=(
        'Latent Variance Process',
        'Sample Path',
        'Unobservable Data Generating Distribution',
        'Observed (Empirical) Stock Returns'
    )
)

frames = []
all_samples = []
sample_path = [0]  # Initialize sample path with starting value

# --- Generate frames ---
for i in range(n_frames):
    n_samples_current = int(n_samples * (1 / (1 + np.abs(means[i])**2)))
    X = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current)
    kde_X = stats.gaussian_kde(X)
    samples = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current // 500)
    all_samples.extend(samples)
    
    # Generate next point in sample path
    next_return = np.random.normal(means[i], np.sqrt(variance_path[i]))
    sample_path.append(sample_path[-1] + next_return)
    
    frames.append(
        go.Frame(
            data=[
                # (1) OU variance path
                go.Scatter(
                    x=time_points[:i+1],
                    y=variance_path[:i+1],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 255, 1)', width=2),
                    name='Variance Path'
                ),
                # (2) Theoretical distribution
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Varying Normal'
                ),
                # (3) Histogram of observed samples
                go.Histogram(
                    x=all_samples,
                    nbinsx=50,
                    name='NVDA Distribution',
                    marker_color='rgba(0, 255, 255, 0.6)'
                ),
                # (4) Sample path
                go.Scatter(
                    x=time_points[:i+2],
                    y=sample_path,
                    mode='lines',
                    line=dict(color='rgba(255, 165, 0, 1)', width=2),
                    name='NVDA Price Path'
                )
            ]
        )
    )

# Initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)  # Variance
fig.add_trace(frames[0].data[1], row=2, col=1)  # Distribution
fig.add_trace(frames[0].data[2], row=3, col=1)  # Histogram
fig.add_trace(frames[0].data[3], row=1, col=2)  # Sample Path

# Add frames to figure
fig.frames = frames

# --- Animation Controls ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0},
                'mode': 'immediate',
                'loop': True
            }]
        }]
    }]
)

# --- Layout & Styling ---
fig.update_layout(
    height=600,
    width=1200,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Negative-Biased Normal Distribution with Persistent High Variance (OU Process)'
)

# --- Axes Formatting ---
for (r, c) in [(1,1), (2,1), (3,1), (1,2)]:
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=c
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=c
    )

# --- Specific Ranges ---
fig.update_xaxes(range=[0, n_frames/10], row=1, col=1)
fig.update_yaxes(range=[0, 6], row=1, col=1)
fig.update_xaxes(range=[-6, 6], row=2, col=1)
fig.update_yaxes(range=[0, 1], row=2, col=1)
fig.update_xaxes(range=[-6, 6], row=3, col=1)
fig.update_yaxes(range=[0, 1750], row=3, col=1)
fig.update_xaxes(range=[0, n_frames/10], title="Time", row=1, col=2)
fig.update_yaxes(range=[-20, 20], title="Cumulative Return", row=1, col=2)

fig.show()



# ##### To better understand these dynamics and relative likelihoods of outcomes, we need better models and domain expertise!
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# **We need to understand and be capable of building. . .**
# 
# - Market dynamics
# - Macroeconomics
# - Politics
# - Time variant models
# - . . . the list goes on!
# 
# Our models *will always be wrong*, they are *models* if we knew the dynamics we wouldn't need a *model* (gotta love these comments, they have no idea what they're doing!)
# 
# **Reasonable Question:** *What's the point of building models if they are wrong???*
# - Your professor can explain to you a Hidden Markov Model, but can or will he explain to you the purpose of it in real life when the assumptions are violated?


# ###### ______________________________________________________________________________________________________________________________________


# ##### Black Swan Events (2008)
# 
# The market doesn't always create a *fair* equilibrium price, agents are not perfectly rational like economics suggests
# 
# We've all seen the big short, a massive purchase of insurance against the "strongest" housing market in history
# 
# How did the quants not run risk models and see what was to come?
# 
# Modeling these types of events requires capturing *exogenous factors* - looking into the mortgages themselves uncovering what the market wasn't yet aware of
# 
# See my video on Time Series Analysis for more on this topic, I discuss this there at length. . .


# -----------------------------
# Risk Model and CDS Price Path Simulation
# -----------------------------
n_steps = 140
pre_crash = 80  # Point where crash occurs
cds_start = 10  # Starting CDS price (rebased at 1)
crash_magnitude = 4  # How much CDS spikes during crash (adjusted for rebasing)

# Generate paths
def generate_risk_path(n_steps, pre_crash):
    path = np.empty(n_steps)
    path[0] = 0
    
    # Pre-crash: Mean-reverting random walk
    for t in range(1, pre_crash):
        # Add mean reversion to keep path centered around 0
        mean_reversion = -0.1 * path[t-1]  # Pull back toward 0
        noise = np.random.normal(0, 0.5)
        path[t] = path[t-1] + mean_reversion + noise
        # Bound within smaller range pre-crash
        path[t] = np.clip(path[t], -5, 5)
        
    # Post-crash: Sharp decline
    for t in range(pre_crash, n_steps):
        if t == pre_crash:
            path[t] = path[t-1] - 15  # Sharp drop
        else:
            drift = -0.1  # Continued decline
            noise = np.random.normal(0, 0.3)
            path[t] = path[t-1] + drift + noise
            
    return path

def generate_cds_path(n_steps, pre_crash, start_price, crash_mag):
    path = np.empty(n_steps)
    path[0] = start_price
    
    # Pre-crash: more aggressive downward drift with noise
    for t in range(1, pre_crash):
        drift = -0.003  # Stronger downward drift
        noise = np.random.normal(0, 0.01)
        path[t] = path[t-1] * (1 + drift + noise)
        path[t] = max(path[t], 0.1)  # Floor to prevent negative values
        
    # Crash and post-crash behavior
    for t in range(pre_crash, n_steps):
        if t == pre_crash:
            path[t] = path[t-1] * (1 + crash_mag)
        else:
            drift = 0.02  # Continued upward drift post-crash
            noise = np.random.normal(0, 0.03)
            path[t] = path[t-1] * (1 + drift + noise)
            
    return path

# Generate the paths
np.random.seed(42)
risk_path = generate_risk_path(n_steps, pre_crash)
cds_path = generate_cds_path(n_steps, pre_crash, cds_start, crash_magnitude)

# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2, subplot_titles=("Risk Model Path", "CDS Price Evolution"))

# Add reference lines for crash
for col in [1, 2]:
    fig.add_shape(
        type="line",
        x0=pre_crash, x1=pre_crash,
        y0=-30, y1=1 if col==1 else np.max(cds_path) + 0.5,
        line=dict(color="yellow", width=2, dash="dot"),
        row=1, col=col
    )

# Create animation frames
frames = []
for t in range(1, n_steps):
    frames.append(
        go.Frame(
            data=[
                # Risk path
                go.Scatter(
                    x=np.arange(t+1),
                    y=risk_path[:t+1],
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Risk Path",
                    showlegend=False
                ),
                # CDS path
                go.Scatter(
                    x=np.arange(t+1),
                    y=cds_path[:t+1],
                    mode="lines",
                    line=dict(color="lime", width=2),
                    name="CDS Price",
                    showlegend=False
                )
            ],
            name=f"frame{t}"
        )
    )

# Initial empty traces
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[risk_path[0]],
        mode="lines",
        line=dict(color="red", width=2),
        name="Risk Path"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[0],
        y=[cds_start],
        mode="lines",
        line=dict(color="lime", width=2),
        name="CDS Price"
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1200,
    showlegend=True,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [
                None,
                {"frame": {"duration": 30, "redraw": True},
                 "fromcurrent": True,
                 "transition": {"duration": 0}}
            ]
        }]
    }]
)

# Update axes
fig.update_xaxes(title="Time", showgrid=True, gridcolor="rgba(128,128,128,0.3)", range=[0, n_steps])
fig.update_yaxes(
    title="Standard Deviations",
    showgrid=True,
    gridcolor="rgba(128,128,128,0.3)",
    range=[-30, 5],
    row=1, col=1
)
fig.update_yaxes(
    title="CDS Price",
    showgrid=True,
    gridcolor="rgba(128,128,128,0.3)",
    range=[0, np.max(cds_path) + 0.5],
    row=1, col=2
)

# Add frames and show
fig.frames = frames
fig.show()



# ---


# #### 3.) 🎯 Purpose of Modeling
# 
# ##### Making Optimal Decisions under Uncertainty
# 
# ##### If you are serving a business function, are a trader. . .whatever - you **have to act**
# 
# You don't get to sit there and see how it unfolds, you have to make an assumption take a position and manage your risk
# 
# **So what position do you take?**  Instead of shooting in the dark, we build models to help inform our decision making.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### 🌟 Our goal is to act with an edge
# 
# We aren't trying to *predict* anything, we are trying to be correct *on average* (harder than it sounds, but its how we make money)
# 
# If our edge, give by the law of total expectation is positive, we are accumulating edge (making optimal decisions in the face of uncertainty)
# 
# $$E[X] = \sum_{i=1}^n x_i P(X=x_i) > 0 \quad \quad \text{ or } E[X] = \int_{-\infty}^{\infty} x f(x) dx > 0$$
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



# This concept applies to literally everything, ask yourself how can a model informed decision (or my own optimal actions) give me an edge?
# 
# 
# **To get philosophical on you, this applies to. . .**
# - Goals (edge in discipline, studying when you don't want to. . .)
# - Health (edge in knowledge, understanding optimal workout and diet. . .)
# - Investing (edge in experience, understanding different regimes, drawdowns aren't forever, removing emotion)
# - and so on and so forth!


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - There is a substantial difference between a random and an uncertain system, and our ability to model the unknown outcomes of either system
# - Economics suggests equilibrium prices are *fair*, but this is only true if agents are rational - they **are not** and equilibrium price is far from perfect
# - Implied probabilities are some of the perfect examples of irrationality, mispricings, and lack of convergence misrepresenting *true* probabilities
# - In this vein, quant models break, uncertain events can't be replayed under the same conditions as random events
# - Dynamics in reality are far more complicated, and we have to build better models (constantly, and continue to adjust assumptions) - that is our job as quants
# - All models are wrong, but when we are met with uncertainty we have to take action - we don't get to just "not act" because we don't know what to do
# - Models aid in making informed decisions by tying likelihoods to certain outcomes, makeing optimal decisions with these structures is both a skill and an art
# - If we are making optimal decisions in the face of uncertainty we will accumulate this edge over time in *any system* (health, studies, wealth, . . .)
# 
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

