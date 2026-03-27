# ## Why Monte Carlo Simulation Works
# 
# #### 🎯 Quant Guild Videos Applying Monte Carlo Simulation:
# 
# - [Gambler's Ruin Problem in Quant Trading](https://youtu.be/YNvhjSr_nz0)
# 
# - [Is Quant Trading Gambling - Roulette, Poker, and Trading](https://youtu.be/fI3UHYD389g)
# 
#  
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [🚀 Take Live Classes with Roman on Quant Guild](https://quantguild.com)


# ---


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML
from scipy.stats import norm, gamma

# Set random seed for reproducibility
np.random.seed(42)

# Initialize parameters
n_points = 1000
n_frames = 200
window_size = 500
n_bins = 50  # Increased number of bins for smoother histogram

# Generate initial data with normal distribution
data = np.random.normal(loc=0, scale=1, size=n_points)

# Create figure
fig = go.Figure()

# Create x points for true distributions
x = np.linspace(-6, 10, 200)

# Calculate bin edges across full range
bins = np.linspace(-6, 10, n_bins+1)
hist, bins = np.histogram(data, bins=bins, density=True)
center = (bins[:-1] + bins[1:]) / 2

# Add initial histogram
fig.add_trace(
    go.Scatter(
        x=center,
        y=hist,
        mode='lines',
        line=dict(width=2, color='#88ccee'),
        name='Empirical Distribution'
    )
)

# Add initial true distribution
fig.add_trace(
    go.Scatter(
        x=x,
        y=norm.pdf(x, loc=0, scale=1),
        mode='lines',
        line=dict(width=2, color='#ee8866', dash='dash'),
        name='True Distribution'
    )
)

# Set layout with dark mode
fig.update_layout(
    title='Evolving Distribution Over Time',
    xaxis_range=[-6, 10],
    yaxis_range=[0, 0.5],
    showlegend=True,
    width=1000,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zerolinecolor='rgba(128,128,128,0.2)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zerolinecolor='rgba(128,128,128,0.2)'
    )
)

# Create frames for animation
frames = []
for frame in range(n_frames):
    # Create drift in distribution by mixing with different distributions
    t = frame / n_frames
    # Mix normal with increasingly skewed distribution
    new_data = (1-t) * np.random.normal(loc=0, scale=1, size=window_size) + \
               t * np.random.gamma(2+t*3, 1, size=window_size)
    
    # Update data for the current window
    data[:-window_size] = data[window_size:]
    data[-window_size:] = new_data
    
    # Update histogram using fixed bins across full range
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    # Calculate true mixed distribution
    true_dist = (1-t) * norm.pdf(x, loc=0, scale=1) + \
                t * gamma.pdf(x, a=2+t*3, scale=1)
    
    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=center, y=hist),
                go.Scatter(x=x, y=true_dist)
            ],
            name=f'frame{frame}'
        )
    )

# Add frames to figure
fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

# Display animation
fig.show()



# ### Sections
# 
# 1.) Law of Large Numbers
# 
# 2.) Monte Carlo Simulation
# 
# 3.) Estimating Statistics
# 
# 4.) Estimating Probabilities
# 
# 5.) Estimating Option Prices
# 
# 6.) Challenges in Finance
# 
# 7.) Closing Thoughts and Future Topics


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


# ---


# ### 1.) Law of Large Numbers


# #### <u>Random Variables</u>
# 
# Random variables $X$ are characterized fully by their probability mass or density function, or cumulative distribution function.
# 
# It is often confusing to students as $X$ seems like a traditional variable when in reality it is a *distribution* defined by $D$.
# 
# Different draws from $X$ will yield different values but drawing *a lot* of values and plotting a histogram will yield the distribution $D$ that governs the randomness of $X$.
# 
# *Remark:* Different types of convergence exist in probability theory - convergence in distribution (weakest), convergence in probability (stronger), and almost sure convergence (strongest). Each describes how random variables approach a limit in different ways.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate theoretical normal distribution points
x = np.linspace(-4, 4, 1000)  # Increased points for smoother curve
theoretical_dist = stats.norm.pdf(x)

# Create figure
fig = go.Figure()

# Initialize with empty histogram
fig.add_trace(
    go.Histogram(
        x=[],
        histnorm='probability density',
        name='Empirical Distribution',
        nbinsx=50,  # Increased number of bins
        opacity=0.7
    )
)

# Add theoretical distribution line
fig.add_trace(
    go.Scatter(
        x=x,
        y=theoretical_dist,
        name='Theoretical Distribution',
        line=dict(color='red')
    )
)

# Create frames for animation
frames = []
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]  # Added more samples

for n in sample_sizes:
    # Generate samples
    samples = np.random.normal(0, 1, n)
    
    frame = go.Frame(
        data=[
            go.Histogram(
                x=samples,
                histnorm='probability density',
                name='Empirical Distribution',
                nbinsx=50,  # Increased number of bins
                opacity=0.7
            ),
            go.Scatter(
                x=x,
                y=theoretical_dist,
                name='Theoretical Distribution',
                line=dict(color='red')
            )
        ],
        name=f'n={n}'
    )
    frames.append(frame)

fig.frames = frames

# Add buttons and sliders with dark mode styling
fig.update_layout(
    title='Empirical Convergence',
    xaxis_title='Value',
    yaxis_title='Density',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
    font=dict(color='white'),  # White text
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': 'Play',
             'method': 'animate',
             'args': [None, {'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True}]},
            {'label': 'Pause',
             'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                              'mode': 'immediate',
                              'transition': {'duration': 0}}]}
        ]
    }],
    sliders=[{
        'currentvalue': {'prefix': 'Sample Size: ', 'font': {'color': 'white'}},
        'steps': [
            {'label': str(n),
             'method': 'animate',
             'args': [[f'n={n}'], {'frame': {'duration': 0, 'redraw': True},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}}]}
            for n in sample_sizes
        ]
    }]
)

# Update axes for dark mode
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()



# #### <u>Law of Large Numbers</u>
#  
#  The Law of Large Numbers (LLN) comes in two forms:
# 
#  **Strong Law of Large Numbers (SLLN):**
# 
#  For i.i.d. random variables $X_1, X_2, ..., X_n$ with $E[X_i] = \mu$:
#  
#  $P(\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^n X_i = \mu) = 1$
# 
#  **Weak Law of Large Numbers (WLLN):**
# 
#  For i.i.d. random variables $X_1, X_2, ..., X_n$ with $E[X_i] = \mu$:
# 
#  $\lim_{n \to \infty} P(|\frac{1}{n}\sum_{i=1}^n X_i - \mu| > \epsilon) = 0$ for any $\epsilon > 0$


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate theoretical normal distribution points
x = np.linspace(-4, 4, 1000)
theoretical_dist = stats.norm.pdf(x)

# Create figure
fig = go.Figure()

# Add theoretical distribution line
fig.add_trace(
    go.Scatter(
        x=x,
        y=theoretical_dist,
        name='Theoretical Distribution',
        line=dict(color='#FF69B4')
    )
)

# Add vertical line for theoretical expectation (mean = 0 for standard normal)
fig.add_trace(
    go.Scatter(
        x=[0, 0],
        y=[0, max(theoretical_dist)],
        name='Theoretical Mean',
        line=dict(color='#FF69B4', dash='dash'),  # Neon pink
        showlegend=True
    )
)

# Create frames for animation
frames = []
sample_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 5000]

for n in sample_sizes:
    # Generate samples
    samples = np.random.normal(0, 1, n)
    empirical_mean = np.mean(samples)
    
    frame = go.Frame(
        data=[
            go.Scatter(
                x=x,
                y=theoretical_dist,
                name='Theoretical Distribution',
                line=dict(color='#FF69B4')
            ),
            go.Scatter(
                x=[0, 0],
                y=[0, max(theoretical_dist)],
                name='Theoretical Mean',
                line=dict(color='#FF69B4', dash='dash')
            ),
            go.Scatter(
                x=[empirical_mean, empirical_mean],
                y=[0, max(theoretical_dist)],
                name='Empirical Mean',
                line=dict(color='#00FFFF', width=2),  # Neon cyan line
                showlegend=True
            )
        ],
        name=f'n={n}'
    )
    frames.append(frame)

# Add initial empirical mean line
initial_samples = np.random.normal(0, 1, sample_sizes[0])
initial_mean = np.mean(initial_samples)
fig.add_trace(
    go.Scatter(
        x=[initial_mean, initial_mean],
        y=[0, max(theoretical_dist)],
        name='Empirical Mean',
        line=dict(color='#00FFFF', width=2),  # Match frame styling
        showlegend=True
    )
)

fig.frames = frames

# Add buttons and sliders with dark mode styling
fig.update_layout(
    title='Mean Convergence',
    xaxis_title='Value',
    yaxis_title='Density',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': 'Play',
             'method': 'animate',
             'args': [None, {'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True}]},
            {'label': 'Pause',
             'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                              'mode': 'immediate',
                              'transition': {'duration': 0}}]}
        ]
    }],
    sliders=[{
        'currentvalue': {'prefix': 'Sample Size: ', 'font': {'color': 'white'}},
        'steps': [
            {'label': str(n),
             'method': 'animate',
             'args': [[f'n={n}'], {'frame': {'duration': 0, 'redraw': True},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}}]}
            for n in sample_sizes
        ]
    }]
)

# Update axes for dark mode
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()



# ---


# ### 2.) Monte Carlo Simulation
# 
# Given the law of large numbers (LLN) ensures convergence to the theoretical mean in both the weak and strong case, we can easily sample random variables from their distribution and determine sample statistics as they are based on the mean.
# 
# For large $n$, sampling $X_i$ from distribution $D$:
# 
# $\mathbb{E}[X] \approx \frac{1}{n} \sum_{i=1}^n X_i$
# 
# $\text{Var}[X] \approx \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2$
#  
# $\text{Skewness}[X] \approx \frac{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^3}{(\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2)^{3/2}}$
#  
# $\text{Kurtosis}[X] \approx \frac{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^4}{(\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2)^2}$
# 
# **Remark:** Estimators may exhibit *bias* or deviation from the parameter the estimator is attempting to approximate, in some cases we want to introduce bias, in other cases we don't.  Something to keep in mind as you choose different estimators for parameters.
# 
# ##### 🎲 Example: Dice Roll Mean and Variance 
# 
#  For a fair 6-sided die:
# 
#  $\mathbb{E}[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5$
# 
#  $\text{Var}[X] = \mathbb{E}[(X - \mu)^2] = \frac{(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2}{6}$
# 
#  $\phantom{\text{Var}[X]} = \frac{6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25}{6} = \frac{17.5}{6} \approx 2.917$
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Generate samples from dice rolls (uniform discrete 1-6)
n_samples = 10000
true_mean = 3.5  # (1+2+3+4+5+6)/6
true_var = 2.916667  # Sum((x-mean)^2/6) for x in 1:6
samples = np.random.randint(1, 7, n_samples)

# Calculate cumulative means and variances
cum_means = np.cumsum(samples) / np.arange(1, n_samples + 1)
cum_vars = np.array([np.var(samples[:i+1], ddof=1) for i in range(n_samples)])

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=('Convergence of Sample Mean (Dice Rolls)', 
                                  'Convergence of Sample Variance (Dice Rolls)'))

# Add mean convergence trace
fig.add_trace(
    go.Scatter(x=np.arange(1, n_samples + 1), y=cum_means,
               name='Sample Mean',
               line=dict(color='#00FFFF', width=2)),  # Neon cyan
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[1, n_samples], y=[true_mean, true_mean],
               name='True Mean',
               line=dict(color='#FF69B4', width=2, dash='dash')),  # Neon pink
    row=1, col=1
)

# Add variance convergence trace
fig.add_trace(
    go.Scatter(x=np.arange(1, n_samples + 1), y=cum_vars,
               name='Sample Variance',
               line=dict(color='#00FFFF', width=2)),  # Neon cyan
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=[1, n_samples], y=[true_var, true_var],
               name='True Variance',
               line=dict(color='#FF69B4', width=2, dash='dash')),  # Neon pink
    row=2, col=1
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=800,
    xaxis_type='log',
    xaxis2_type='log',
    title_text='Monte Carlo Convergence for Dice Rolls',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in [1, 2]:
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     row=i, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     row=i, col=1)

fig.show()



# ---


# ### 3.) Estimating Statistics
# 
#  Let's consider a complex game with the following rules:
#  
#  1. Roll a fair 6-sided die
#  2. If you roll an odd number (1,3,5):
#     - Flip a weighted coin (60% heads)
#     - If heads: Win $1000
#     - If tails: Lose $500
#  3. If you roll an even number (2,4,6):
#     - Flip a fair coin (50% heads)
#     - If heads: Lose $500 
#     - If tails: Win $1000
# 
# **What is the fair value of this game? A canonical question in quantitative finance!**
# 
# Though we can solve this analytically, like many stochastic differential equations - what if we coudln't?
# 
# This creates an interesting probability problem that we can solve with Monte Carlo simulation.
# 
# ##### 🎰 Example: Estimating the Fair Value of a Game of Chance 
# 
# You are a casino and you want to ensure you have edge against the players for this new game - what will you charge them to play?


# Calculate theoretical mean
# For odd rolls (probability 1/2):
#   P(win|odd) = 0.6, win $1000
#   P(lose|odd) = 0.4, lose $500
odd_ev = 0.5 * (0.6 * 1000 + 0.4 * (-500))

# For even rolls (probability 1/2):
#   P(win|even) = 0.5, win $1000
#   P(lose|even) = 0.5, lose $500
even_ev = 0.5 * (0.5 * 1000 + 0.5 * (-500))

# Total expected value
theoretical_mean = odd_ev + even_ev

# Create figure
fig = make_subplots(rows=1, cols=1,
                    subplot_titles=('Convergence of Sample Mean (Game Outcomes)'))

# Generate samples
n_samples = 10000
def play_game():
    # Roll die
    roll = np.random.randint(1, 7)
    
    # Odd roll case
    if roll % 2 == 1:
        # Flip weighted coin (60% heads)
        if np.random.random() < 0.6:
            return 1000  # Win on heads
        else:
            return -500  # Lose on tails
    
    # Even roll case
    else:
        # Flip fair coin (50% heads) 
        if np.random.random() < 0.5:
            return -500  # Lose on heads
        else:
            return 1000  # Win on tails

outcomes = np.array([play_game() for _ in range(n_samples)])

# Calculate cumulative means
cum_means = np.cumsum(outcomes) / np.arange(1, n_samples + 1)

# Add mean convergence trace
fig.add_trace(
    go.Scatter(x=np.arange(1, n_samples + 1), y=cum_means,
               name='Sample Mean',
               line=dict(color='#39FF14', width=2)),  # Neon green
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[1, n_samples], y=[theoretical_mean, theoretical_mean],
               name='True Mean',
               line=dict(color='#FF69B4', width=2, dash='dash')),  # Neon pink
    row=1, col=1
)

# Update layout
fig.update_layout(
    showlegend=True,
    height=400,
    xaxis_type='log',
    title_text='Monte Carlo Convergence for Game Outcomes',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

print(f"Theoretical mean: ${theoretical_mean:.2f}")
fig.show()



# ##### Q: What should we charge as the house to give us an edge playing this game?


# Simulate wealth paths with different entry fees
n_plays = 100000

# Initialize arrays for each fee level
wealth_330 = np.zeros(n_plays + 1)  # +1 to include initial wealth
wealth_325 = np.zeros(n_plays + 1)
wealth_320 = np.zeros(n_plays + 1)

# Set initial wealth
wealth_330[0] = wealth_325[0] = wealth_320[0] = 0

for i in range(n_plays):
    # Simulate one game outcome
    die_roll = np.random.randint(1, 7)
    coin_flip = np.random.random()
    
    # Calculate game outcome
    if die_roll % 2 == 1:  # Odd number
        if coin_flip < 0.6:  # Heads on weighted coin
            payout = -1000
        else:  # Tails on weighted coin
            payout = 500
    else:  # Even number
        if coin_flip < 0.5:  # Heads on fair coin
            payout = 500
        else:  # Tails on fair coin
            payout = -1000
    
    # Update wealth for each fee level
    wealth_330[i+1] = wealth_330[i] + 330 + payout
    wealth_325[i+1] = wealth_325[i] + 325 + payout
    wealth_320[i+1] = wealth_320[i] + 320 + payout

# Plot wealth paths
fig = go.Figure()

# Add trace for $330 fee
fig.add_trace(
    go.Scatter(x=np.arange(n_plays + 1), y=wealth_330,
               name='$330 Entry Fee',
               line=dict(color='#00FF00', width=2))  # Neon green
)

# Add trace for $325 fee
fig.add_trace(
    go.Scatter(x=np.arange(n_plays + 1), y=wealth_325,
               name='$325 Entry Fee',
               line=dict(color='#FF00FF', width=2))  # Neon pink
)

# Add trace for $320 fee
fig.add_trace(
    go.Scatter(x=np.arange(n_plays + 1), y=wealth_320,
               name='$320 Entry Fee',
               line=dict(color='#00FFFF', width=2))  # Neon cyan
)

# Add zero line
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

# Update layout
fig.update_layout(
    showlegend=True,
    height=600,
    title_text='House Wealth Paths with Different Entry Fees',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title='Wealth ($)',
    xaxis_title='Number of Games Played'
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()



# ---


# ### 4.) Estimating Probabilities
# 
#  Monte Carlo simulation can be used to estimate probabilities by leveraging the Law of Large Numbers (LLN). When we want to estimate the probability of an event, we can treat each trial as a Bernoulli random variable $X_i \sim \text{Bernoulli}(p)$ - it either occurs $(X_i=1)$ or doesn't $(X_i=0)$. 
# 
#  The LLN states that for independent trials, as $n \to \infty$:
# 
#  $\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mathbb{E}[X] = p$
# 
#  This means our sample mean (proportion of successes) will converge to the true probability $p$. To estimate a probability using Monte Carlo:
# 
#  1. Run $n$ simulations where $X_i = \begin{cases} 1 & \text{if event occurs} \\ 0 & \text{otherwise} \end{cases}$
#  2. Calculate $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$
# 
#  The larger our sample size $n$, the more confident we can be in our probability estimate $\hat{p}$.
# 
# ##### Effectively, we can use Bernoulli random variables to answer interesting probability questions:
# 
# - What is the probability we make money?
# 
# - What is the probability in X plays we make more than Y dollars?
# 
# - What is the probability we go bankrupt before reaching our target wealth? (Gambler's Ruin Problem)


# ##### Q: What is the probability of winning money playing the game?


# Function to simulate one game
def simulate_game():
    # Roll die
    die_roll = np.random.randint(1, 7)
    
    # Odd number case
    if die_roll % 2 == 1:
        # Flip weighted coin (60% heads)
        if np.random.random() < 0.6:
            return 1000  # Win on heads
        else:
            return -500  # Lose on tails
    # Even number case
    else:
        # Flip fair coin (50% heads)
        if np.random.random() < 0.5:
            return -500  # Lose on heads
        else:
            return 1000  # Win on tails

# Number of simulations
n_sims = 1000

# Run simulations and track results
results = np.zeros(n_sims)
win_probability = np.zeros(n_sims)

for i in range(n_sims):
    results[i] = simulate_game()
    win_probability[i] = np.mean(results[:i+1] > 0)

# Create figure
fig = go.Figure()

# Add trace for win probability convergence
fig.add_trace(
    go.Scatter(
        x=np.arange(1, n_sims + 1),
        y=win_probability,
        mode='lines',
        name='Estimated Probability of Winning',
        line=dict(color='#00ff00')
    )
)

# Calculate theoretical probability
# P(Win) = P(Odd)*P(Heads|Odd)*P(Win|Heads) + P(Odd)*P(Tails|Odd)*P(Win|Tails) +
#         P(Even)*P(Heads|Even)*P(Win|Heads) + P(Even)*P(Tails|Even)*P(Win|Tails)
theoretical_prob = (3/6)*(0.6)*(1) + (3/6)*(0.4)*(0) + (3/6)*(0.5)*(0) + (3/6)*(0.5)*(1)
fig.add_hline(y=theoretical_prob, line_dash="dash", line_color="red", 
              annotation_text=f"Theoretical Probability: {theoretical_prob:.3f}",
              annotation_position="bottom right")

# Update layout
fig.update_layout(
    showlegend=True,
    height=600,
    title_text='Convergence of Win Probability Estimate',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title='Probability of Winning',
    xaxis_title='Number of Simulations',
    yaxis_range=[0, 1]
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()



# #### Q: What is the probability of reaching $15,000 starting at $10,000 paying the fair price of $325 per game?


# Simulate multiple wealth paths
n_paths = 1000
starting_wealth = 10000
target_wealth = 15000
max_steps = 100

# Store paths and outcomes
paths = np.zeros((n_paths, max_steps+1))
paths[:,0] = starting_wealth
outcomes = np.zeros(n_paths)

for path in range(n_paths):
    wealth = starting_wealth
    for step in range(max_steps):
        # Generate random number and odd/even
        num = np.random.randint(1,7)
        is_odd = num % 2 == 1
        
        # Flip biased coin if odd, fair coin if even
        if is_odd:
            coin = np.random.choice(['H','T'], p=[0.6, 0.4])
        else:
            coin = np.random.choice(['H','T'], p=[0.5, 0.5])
            
        # Determine if won based on rules:
        # Odd + Heads = Win $1000
        # Odd + Tails = Lose $500
        # Even + Heads = Lose $500
        # Even + Tails = Win $1000
        wealth -= 325 # price to play
        if (is_odd and coin == 'H') or (not is_odd and coin == 'T'):
            wealth += 1000
        else:
            wealth -= 500
            
        paths[path,step+1] = wealth
        
        # Check if reached target or went bankrupt
        if wealth >= target_wealth:
            outcomes[path] = 1
            break
        elif wealth <= 0:
            outcomes[path] = -1
            break
            
    # Fill rest of path with final value
    paths[path,step+2:] = wealth

# Calculate probability of reaching target
prob_target = np.sum(outcomes == 1) / n_paths

# Create figure
fig = go.Figure()

# Add winning paths
winning_paths = paths[outcomes == 1]
for path in winning_paths:
    fig.add_trace(
        go.Scatter(
            x=np.arange(max_steps+1),
            y=path,
            mode='lines',
            line=dict(color='green', width=1),
            showlegend=False,
            opacity=0.3
        )
    )

# Add losing paths
losing_paths = paths[outcomes == -1]
for path in losing_paths:
    fig.add_trace(
        go.Scatter(
            x=np.arange(max_steps+1),
            y=path,
            mode='lines', 
            line=dict(color='red', width=1),
            showlegend=False,
            opacity=0.3
        )
    )

# Add target line
fig.add_hline(y=target_wealth, line_dash="dash", line_color="blue",
              annotation_text="Target Wealth",
              annotation_position="right")

# Update layout
fig.update_layout(
    showlegend=False,
    height=600,
    title_text=f'Wealth Paths (Probability of Reaching Target: {prob_target:.1%})',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis_title='Wealth ($)',
    xaxis_title='Number of Games',
    yaxis_range=[0, target_wealth*1.1]
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()



# ---


# ### 5.) Estimating Financial Derivative Prices
# 
# Knowing that Monte Carlo simulation relies on a fixed distribution, given a stochastic differential equation we can simulate the distribution of stock prices at any time $t$ to help price a financial derivative.
# 
#  The Black-Scholes formula can be written as a risk-neutral expectation:
#  
#  $C(S_0, K, T) = e^{-rT}\mathbb{E}_\mathbb{Q}[\max(S_T - K, 0)]$
#  
#  where:
#  - $S_T$ is the stock price at maturity $T$
#  - $K$ is the strike price
#  - $r$ is the risk-free rate
#  - $\mathbb{Q}$ denotes the risk-neutral measure
#  
#  Under this measure, the stock price follows geometric Brownian motion:
#  
#  $dS_t = rS_tdt + \sigma S_tdW_t$
# 
# We can solve this expectation using Monte Carlo Simulation.
# 
# I have an entire video dedicated to this topic: [Monte Carlo Simulation and Black-Scholes for Pricing Options](https://youtu.be/-1RYvajksjQ)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1     # Time to maturity
n_paths = 1000  # Number of paths
n_steps = 252   # Number of time steps
dt = T/n_steps

# Black-Scholes formula
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# True Black-Scholes price
bs_price = black_scholes_call(S0, K, T, r, sigma)

# Simulate paths
np.random.seed(42)
Z = np.random.standard_normal((n_paths, n_steps))
paths = np.zeros((n_paths, n_steps+1))
paths[:,0] = S0

for t in range(1, n_steps+1):
    paths[:,t] = paths[:,t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t-1])

# Calculate option payoffs
payoffs = np.maximum(paths[:,-1] - K, 0)
mc_prices = np.exp(-r*T) * np.cumsum(payoffs) / np.arange(1, n_paths+1)

# Create subplots
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Stock Price Paths', 'Monte Carlo Price Convergence'),
                    vertical_spacing=0.15)

# Plot 100 random paths with color based on final value
for i in range(100):
    path_color = '#39FF14' if paths[i,-1] > K else '#FF1493'  # neon green if above strike, neon red if below
    fig.add_trace(
        go.Scatter(y=paths[i], mode='lines', line=dict(width=1, color=path_color), 
                  showlegend=False, opacity=0.3),
        row=1, col=1
    )

# Add strike price line
fig.add_hline(y=K, line_dash="dash", line_color="white",
              annotation_text="Strike Price",
              annotation_position="right",
              row=1, col=1)

# Plot convergence
fig.add_trace(
    go.Scatter(y=mc_prices, mode='lines', name='MC Price',
               line=dict(color='#00FFFF', width=2)),  # neon blue
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=[bs_price]*len(mc_prices), mode='lines', name='BS Price',
               line=dict(color='red', width=2, dash='dash')),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    title_text='European Call Option Price: Monte Carlo vs Black-Scholes',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

# Update subplot titles
fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Option Price ($)", row=2, col=1)
fig.update_xaxes(title_text="Time Steps", row=1, col=1)
fig.update_xaxes(title_text="Number of Simulations", row=2, col=1)

fig.show()



# ---


# ### 6.) Challenges in Finance
# 
# In finance, we don't just talk about the distribution of returns - there are a lot of distributions
# 
# - Sentiment
# 
# - Reversion
# 
# - Momentum
# 
# - ...
# 
# #### <u> Two Important Cases: </u>
# 
# 1.) Random variables follow a fixed distribution (this is a tough sell for many examples including stock returns)
# 
# 2.) Random variables follow a distribution that change over time (more likely, and evident in data collected over time)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML
from scipy.stats import norm, gamma

# Set random seed for reproducibility
np.random.seed(42)

# Initialize parameters
n_points = 1000
n_frames = 200
window_size = 500
n_bins = 50  # Increased number of bins for smoother histogram

# Generate initial data with normal distribution
data = np.random.normal(loc=0, scale=1, size=n_points)

# Create figure
fig = go.Figure()

# Create x points for true distributions
x = np.linspace(-6, 10, 200)

# Calculate bin edges across full range
bins = np.linspace(-6, 10, n_bins+1)
hist, bins = np.histogram(data, bins=bins, density=True)
center = (bins[:-1] + bins[1:]) / 2

# Add initial histogram
fig.add_trace(
    go.Scatter(
        x=center,
        y=hist,
        mode='lines',
        line=dict(width=2, color='#88ccee'),
        name='Empirical Distribution'
    )
)

# Add initial true distribution
fig.add_trace(
    go.Scatter(
        x=x,
        y=norm.pdf(x, loc=0, scale=1),
        mode='lines',
        line=dict(width=2, color='#ee8866', dash='dash'),
        name='True Distribution'
    )
)

# Set layout with dark mode
fig.update_layout(
    title='Evolving Distribution Over Time',
    xaxis_range=[-6, 10],
    yaxis_range=[0, 0.5],
    showlegend=True,
    width=1000,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zerolinecolor='rgba(128,128,128,0.2)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zerolinecolor='rgba(128,128,128,0.2)'
    )
)

# Create frames for animation
frames = []
for frame in range(n_frames):
    # Create drift in distribution by mixing with different distributions
    t = frame / n_frames
    # Mix normal with increasingly skewed distribution
    new_data = (1-t) * np.random.normal(loc=0, scale=1, size=window_size) + \
               t * np.random.gamma(2+t*3, 1, size=window_size)
    
    # Update data for the current window
    data[:-window_size] = data[window_size:]
    data[-window_size:] = new_data
    
    # Update histogram using fixed bins across full range
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    # Calculate true mixed distribution
    true_dist = (1-t) * norm.pdf(x, loc=0, scale=1) + \
                t * gamma.pdf(x, a=2+t*3, scale=1)
    
    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=center, y=hist),
                go.Scatter(x=x, y=true_dist)
            ],
            name=f'frame{frame}'
        )
    )

# Add frames to figure
fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

# Display animation
fig.show()



# ### 7.) Closing Thoughts and Future Topics
# 
# Challenges:
# - Real world applications, we are fitting models and models are wrong so estimates are likely wrong...
# 
# - Wrong models and estimates don't preclude us from optimal decision making in uncertainty (so called edge, think back to the casino example if we had no theoretical mean)
# 
# Future Topics:
# 
# - Why VaR and other measures tend to fail
# 
# - Discretization schemes for stochastic processes (some are difficult to simulate!)
# 
# - Rough volatility model family and pricing under such a model
# 
# - Global path v.s. processes adapted to a filtration (especially in the context of the rough model family)


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

