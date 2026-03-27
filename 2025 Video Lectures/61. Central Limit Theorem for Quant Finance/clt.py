# ### 📊 Central Limit Theorem for Quant Finance
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
# #### 1.) 🎲 Random Variables and Statistics
# 
# - Random Variables
# 
# - Statistics and Distributions
# 
# - Law of Large Numbers (LLN)
# 
# - Application and Considerations
# 
# #### 2.) 🎯 Normal Random Variables
# 
# - Definition and Convergence
# 
# - Likelihoods vs Probabilities
# 
# - Statistics and Characteristics
# 
# #### 3.) 📊 Central Limit Theorem (CLT)
# 
# - Rough Proof using the Characteristic Function
# 
# - Example: Poisson Distribution
# 
# - Application to Stock Returns and Trading
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎲 Random Variables and Statistics
# 
# ##### Random Variables, Population and Empirical Distributions
# 
# Random variables define a set of possible outcomes with accompanying probabilities or likelihoods
# 
# They are fully specified by their
# 
# - Probability mass or density function
# 
# - Cumulative distribution function
# 
# - Characteristic function
# 
# There are **a lot** of different types of random variables (both discrete and continuous)
# 
# We can model the total number of events over a specific time interval as a Poisson random variable
# 
# Let $X \sim Pois(\lambda)$ be the number of trades executed on a particular instrument every 100ms
# 
# - $\lambda$ is the expected (average) number of trades over the specified time interval (100ms)
#  
#  The Poisson distribution with parameter $\lambda$ has:
#  
#  $$
#  \text{PMF: } P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad\quad \text{CDF: } F(k) = \sum_{n=0}^k \frac{\lambda^n e^{-\lambda}}{n!} \quad\quad \text{CF: }\varphi(t) = \exp\left(\lambda (e^{it} - 1)\right)
#  $$
# 
#  


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Setup ---
lmbda = 4  # Poisson parameter (mean/events rate)
poisson_support = np.arange(0, 13)
pmf_poisson = poisson.pmf(poisson_support, lmbda)
n_trials = 250
np.random.seed(42)
samples = np.random.poisson(lmbda, size=n_trials)

# --- Helper: construct figure for a given step ---
def make_poisson_empirical_convergence_fig(step):
    drawn = samples[step]
    counts = np.bincount(samples[:step+1], minlength=poisson_support[-1]+1)
    empirical_pmf = counts / (step + 1)
    # For display, align with support
    empirical_display = empirical_pmf[:len(poisson_support)]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Poisson Mass Function", "Empirical Mass Function"),
    )

    # --- Left subplot: true PMF ---
    bar_colors = ['#d400ff'] * len(poisson_support)  # neon purple
    border_colors = ['rgba(0,0,0,0)'] * len(poisson_support)
    border_widths = [0] * len(poisson_support)
    if drawn < len(poisson_support):
        bar_idx = drawn
        border_colors[bar_idx] = '#FFD700'
        border_widths[bar_idx] = 4

    fig.add_trace(
        go.Bar(
            x=poisson_support,
            y=pmf_poisson,
            width=0.5,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=border_widths)),
            name="Poisson PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: empirical PMF over time ---
    fig.add_trace(
        go.Bar(
            x=poisson_support,
            y=empirical_display,
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
            x=poisson_support,
            y=pmf_poisson,
            mode='lines',
            line=dict(color='#d400ff', width=3, dash='dash'),
            name="Theoretical Poisson PMF",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Legend-only traces ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='#d400ff', width=4),
            name="Poisson PMF",
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
    fig.update_xaxes(title_text="k", row=1, col=1, range=[-0.5, poisson_support[-1]+0.5], tickvals=poisson_support)
    fig.update_yaxes(title_text="P(X=k)", row=1, col=1, range=[0, np.max(pmf_poisson)*1.15])
    fig.update_xaxes(title_text="k", row=1, col=2, range=[-0.5, poisson_support[-1]+0.5], tickvals=poisson_support)
    fig.update_yaxes(title_text="Empirical P(X=k)", row=1, col=2, range=[0, np.max(pmf_poisson)*1.15])

    # --- Subtle gridlines ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    # --- Layout ---
    fig.update_layout(
        height=480,
        width=960,
        title_text="Poisson(λ=4) Distribution vs Empirical Distribution (25000ms)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.98,
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
poisson_frames = [
    go.Frame(data=make_poisson_empirical_convergence_fig(step).data, name=str(step))
    for step in range(n_trials)
]

# --- Initial figure ---
fig = make_poisson_empirical_convergence_fig(0)
fig.frames = poisson_frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.11,
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



# **Remarks:**  We can **never** predict a value for $X$, but we can estimate the probabilities of observing specific values and compute statistics of interest.  It turns out, if we are correct *on average* we can use values like the mean to inform trading and generate trading profits (a classic market-making example).


# ###### ______________________________________________________________________________________________________________________________________


# ##### Statistics and Distributions
# 
# Statistics are random variables themselves, they are a function of data
# 
# Suppose trades ($X$) every 100ms followed a poisson distribution with an average of $4$ trades per interval
# 
# $$X \sim Pois(4)$$
# 
# Similar to how we have a population and empirical distribution, we have population and empirical statistics
# 
#  $$
#  \text{Population Mean: }\quad 
#  \mathbb{E}[X] = \sum_{k=0}^\infty k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = \lambda \quad\quad \text{Sample Mean: }\quad 
#  \overline{x} = \frac{1}{n} \sum_{i=1}^n x_i
#  $$
#  
#  where $x_1, ..., x_n$ are observed values.  
#  
#  We are attempting to approximate the true mean of the population distribution given data, *assuming it follows the given distribution*


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Setup ---
lmbda = 4
poisson_support = np.arange(0, 13)
pmf_poisson = poisson.pmf(poisson_support, lmbda)
pmf_poisson /= pmf_poisson.sum()  # normalize
n_samples_per_iter = 10
n_iters = 10
np.random.seed(42)

# --- Simulation of 10 sets of 30 draws ---
means = []
empirical_pmfs = []
for i in range(n_iters):
    sample = np.random.choice(poisson_support, size=n_samples_per_iter, p=pmf_poisson)
    means.append(np.mean(sample))
    counts = np.bincount(sample, minlength=len(poisson_support))
    empirical_pmfs.append(counts / n_samples_per_iter)

# --- Figure generator for each frame ---
def make_sampling_mean_fig(iteration):
    emp_pmf = empirical_pmfs[iteration]
    mean_so_far = means[:iteration + 1]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Empirical Mass Function", "Sample Means by Iteration")
    )

    # Left: empirical PMF
    fig.add_trace(
        go.Bar(
            x=poisson_support,
            y=emp_pmf,
            marker=dict(color='#00ffff', opacity=0.8),
            name="Empirical PMF",
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=poisson_support,
            y=pmf_poisson,
            mode='lines',
            line=dict(color='#d400ff', width=3, dash='dash'),
            name="True Poisson PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # Right: sample means so far
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(mean_so_far)+1)),
            y=mean_so_far,
            marker=dict(color='#ff6fff', opacity=0.8),
            name="Sample Means",
            showlegend=False
        ),
        row=1, col=2
    )

    # True mean line
    fig.add_hline(y=lmbda, line=dict(color='red', dash='dash'), row=1, col=2)

    # Axes
    fig.update_xaxes(title_text="k", row=1, col=1, range=[-0.5, 12.5])
    fig.update_yaxes(title_text="P(X=k)", row=1, col=1, range=[0, max(pmf_poisson)*1.2])
    fig.update_xaxes(title_text="Iteration", row=1, col=2, range=[0.5, n_iters + 0.5])
    fig.update_yaxes(title_text="Sample Mean", row=1, col=2, range=[0, max(means)*1.2])

    # Layout
    fig.update_layout(
        height=480, width=960,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        title_text=f"Iteration {iteration+1}: Sample Mean = {mean_so_far[-1]:.2f}",
        title_x=0.5,
    )
    return fig

# --- Animation frames ---
frames = []
for i in range(n_iters):
    f = make_sampling_mean_fig(i)
    frames.append(go.Frame(
        data=f.data,
        name=str(i),
        layout=go.Layout(
            title_text=f"Iteration {i+1}: Sample Mean = {means[i]:.2f}"
        )
    ))

# --- Initial figure ---
fig = make_sampling_mean_fig(0)
fig.frames = frames

# --- Animation controls ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.12,
        'xanchor': 'center',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 800, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 400, 'easing': 'cubic-in-out'}
            }]
        }]
    }],
    # subtle easing and transition effect
    transition={'duration': 500, 'easing': 'cubic-in-out'}
)

# --- Bar entrance animation tweak ---
for trace in fig.data:
    if isinstance(trace, go.Bar):
        trace.marker.opacity = 0.8

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Law of Large Numbers (LLN)
# 
# When drawing from the *same population distribution*, empirical statistics and distributions are gaurenteed to converge
# 
# This is given by the Law of Large Numbers (LLN)
# 
#  $$
#  \frac{1}{n}\sum_{i=1}^n h(X_i) \longrightarrow \mathbb{E}[h(X)] \qquad \text{and} \qquad \frac{1}{n}\sum_{i=1}^n \mathbf{1}_{\{X_i \leq x\}} \longrightarrow P(X \leq x)
#  $$
# The first equation expresses sample statistics convergence, the second shows distribution convergence by a series of indicators


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# Parameters
lmbda = 4
poisson_support = np.arange(0, 13)
pmf_poisson = poisson.pmf(poisson_support, lmbda)
pmf_poisson = pmf_poisson / pmf_poisson.sum()
n_trials = 300
np.random.seed(3)

samples = np.random.choice(poisson_support, size=n_trials, p=pmf_poisson)
cumulative_means = np.cumsum(samples) / np.arange(1, n_trials + 1)

def make_cumulative_mean_fig(step):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Empirical Mass Function", "Cumulative Sample Mean")
    )

    # --- Left: Empirical PMF ---
    counts = np.bincount(samples[:step + 1], minlength=len(poisson_support))
    empirical_pmf = counts / (step + 1)
    fig.add_trace(
        go.Bar(
            x=poisson_support,
            y=empirical_pmf,
            marker=dict(color='#00ffff', opacity=0.7),
            name="Empirical PMF",
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=poisson_support,
            y=pmf_poisson,
            mode='lines',
            line=dict(color='#d400ff', width=3, dash='dash'),
            name="True Poisson PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right: Cumulative Mean ---
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, step + 2),
            y=cumulative_means[:step + 1],
            mode='lines+markers',
            line=dict(color='#00ffff', width=3),
            marker=dict(size=6, color='#00ffff'),
            name="Cumulative Mean",
            showlegend=False
        ),
        row=1, col=2
    )

    # True mean line
    fig.add_hline(y=lmbda, line=dict(color='#d400ff', dash='dash'), row=1, col=2)

    # --- Axes ---
    fig.update_xaxes(title_text="k", row=1, col=1, range=[-0.5, 12.5])
    fig.update_yaxes(title_text="P(X=k)", row=1, col=1, range=[0, max(pmf_poisson) * 1.2])

    # Right chart: fixed y-range, but expanding x-range smoothly
    fig.update_xaxes(title_text="Number of Draws", row=1, col=2, range=[0, n_trials])
    fig.update_yaxes(title_text="Sample Mean", row=1, col=2, range=[3, 5])

    fig.update_layout(
        height=500, width=950,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        title_text=f"Draws: {step+1}, Cumulative Mean = {cumulative_means[step]:.2f}",
        title_x=0.5,
    )
    return fig


# --- Build frames ---
frames = []
for step in range(n_trials):
    f = make_cumulative_mean_fig(step)
    # xlim expands gradually to the right with each step
    frames.append(go.Frame(
        data=f.data,
        name=str(step),
        layout=go.Layout(
            title_text=f"Draws: {step+1}, Cumulative Mean = {cumulative_means[step]:.2f}",
            xaxis2=dict(range=[0, max(step + 5, 20)]),  # monotonically increasing xlim
            yaxis2=dict(range=[3, 5]),
        ),
    ))

# --- Initialize & animate ---
fig = make_cumulative_mean_fig(0)
fig.frames = frames

fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.12,
        'xanchor': 'center',
        'showactive': False,
        'buttons': [dict(
            label='Play',
            method='animate',
            args=[None, {
                'frame': {'duration': 10, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 40, 'easing': 'cubic-in-out'}
            }]
        )]
    }],
    transition={'duration': 40, 'easing': 'cubic-in-out'}
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Applications and Considerations
# 
# Above we **assumed** trades every 100ms for a particular instrument followed a Poisson distribution ($X \sim Pois(4)$)
# 
# Suppose we wanted to model the distribution of trades every 100ms to look for a tradable statistically abnormal deviation 
# 
# We can't observe the data generating distribution, otherwise we wouldn't need a model
# 
# The best we can do is make assumptions and build a model (quite literally our job as quants)
# 
# #### Example: Calibrated and Out of Sample (OOS) Trades Distribution 100ms
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Setup ---
lmbda = 4  # Poisson parameter (mean/events rate)
poisson_support = np.arange(0, 13)
pmf_poisson = poisson.pmf(poisson_support, lmbda)
n_trials = 100
np.random.seed(2)
samples = np.random.poisson(lmbda, size=n_trials)

# --- Helper: construct figure for a given step ---
def make_poisson_empirical_convergence_fig(step):
    drawn = samples[step]
    counts = np.bincount(samples[:step+1], minlength=poisson_support[-1]+1)
    empirical_pmf = counts / (step + 1)
    # For display, align with support
    empirical_display = empirical_pmf[:len(poisson_support)]

    # --- SHIFT empirical chart to show lack of convergence (artificially offset)
    shift_offset = -2  # <--- shifting right by 2 units

    # Population mean (theoretical)
    population_mean = lmbda
    # Empirical mean from observed data
    realized_mean = np.mean(samples[:step+1])

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Calibrated Poisson PMF", "OOS Empirical PMF"),
    )

    # --- Left subplot: true PMF ---
    bar_colors = ['#d400ff'] * len(poisson_support)
    border_colors = ['rgba(0,0,0,0)'] * len(poisson_support)
    border_widths = [0] * len(poisson_support)
    if drawn < len(poisson_support):
        bar_idx = drawn
        border_colors[bar_idx] = '#FFD700'
        border_widths[bar_idx] = 4

    fig.add_trace(
        go.Bar(
            x=poisson_support,
            y=pmf_poisson,
            width=0.5,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=border_widths)),
            name="Poisson PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Left subplot: population mean vertical line only ---
    fig.add_vline(
        x=population_mean,
        line_dash="dash",
        line_color="#FFD700",
        line_width=3,
        row=1, col=1,
        annotation_text="Population Mean",
        annotation_position="top right",
        annotation_font=dict(color="#FFD700", size=13),
    )

    # --- Right subplot: empirical PMF over time (SHIFTED) ---
    shifted_support = poisson_support + shift_offset
    fig.add_trace(
        go.Bar(
            x=shifted_support,
            y=empirical_display,
            width=0.5,
            marker=dict(color='#00ffff', opacity=0.8),
            name="Empirical Distribution (Shifted)",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Overlay theoretical PMF on the right for comparison (NOT shifted) ---
    fig.add_trace(
        go.Scatter(
            x=poisson_support,
            y=pmf_poisson,
            mode='lines',
            line=dict(color='#d400ff', width=3, dash='dash'),
            name="Theoretical Poisson PMF",
            showlegend=False
        ),
        row=1, col=2
    )

    # --- Right subplot: Empirical mean (dynamic vertical line, SHIFTED) ---
    fig.add_vline(
        x=realized_mean + shift_offset,
        line_dash="dot",
        line_color="red",
        line_width=3,
        row=1, col=2,
        annotation_text="Empirical Mean",
        annotation_position="top left",
        annotation_font=dict(color="red", size=13),
    )

    # --- Axes ---
    fig.update_xaxes(title_text="k", row=1, col=1, range=[-0.5, poisson_support[-1]+0.5], tickvals=poisson_support)
    fig.update_yaxes(title_text="P(X=k)", row=1, col=1, range=[0, np.max(pmf_poisson)*1.15])
    fig.update_xaxes(
        title_text="k",
        row=1, col=2,
        range=[-0.5, poisson_support[-1]+0.5 + shift_offset],
        tickvals=np.concatenate((poisson_support, shifted_support)),
        showgrid=True, gridcolor='rgba(128,128,128,0.3)'
    )
    fig.update_yaxes(title_text="Empirical P(X=k)", row=1, col=2, range=[0, np.max(pmf_poisson)*1.15], showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    # --- Subtle gridlines for left plot still ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)

    # --- Layout ---
    fig.update_layout(
        height=480,
        width=960,
        title_text="Poisson(λ=4) Distribution vs Empirical Distribution (SHIFTED, 10000ms)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=False,
        margin=dict(l=50, r=20, b=80, t=70),
    )

    return fig

# --- Animation frames ---
poisson_frames = [
    go.Frame(data=make_poisson_empirical_convergence_fig(step).data, layout=make_poisson_empirical_convergence_fig(step).layout, name=str(step))
    for step in range(n_trials)
]

# --- Initial figure ---
fig = make_poisson_empirical_convergence_fig(0)
fig.frames = poisson_frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.11,
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


# ##### 💡 Key Questions for Evaluating Model Efficacy
# 
# In reality, events are uncertain, not random variables - but we use these structures to model such outcomes (like # of trades in 100ms)
# 
# - Is a Poisson distribution and accompanying assumptions reasonable to model trades every 100ms?
# - Is the distribution relatively stable or is there severe time variance?
# - Should it persist, is there a reasonable way to model the lack of stability and time variance?
# 
# In any case, statistics (a function of our population or empirical distribution) are necessary for generating trading P/L
# 
# This leads us to a big question (even after considering all of the above)
# 
# **Is there a way to determine if the statistics we observe deviate significantly from a population statistic?** 
# 
# More specifically,
# 
# **If statistics themselves are random variables, what distribution do they follow?**
# 
# First, we must understand normal (Gaussian) random variables. . .


# ---


# #### 2.) 🎯 Normal Random Variables
# 
# Normal (Gaussian) random variables are continuous and are again fully defined by. . .
# 
# $$\text{PDF: }f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \quad\quad 
#  \text{CDF:} \quad F_X(x) = \int_{-\infty}^x f_X(t) dt \quad\quad \text{CF: }\varphi_X(t) = \exp\left(i\mu t - \frac{1}{2}\sigma^2 t^2\right)$$
# 
#  $$\text{CF: }\varphi_X(t) = \exp\left( - \frac{1}{2}t^2\right)$$
# 
# Convergence in both statistics and distribution is also gaurenteed by the Law of Large Numbers (LLN)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Normal Distribution Parameters ---
mu = 0     # mean
sigma = 1  # standard deviation

# x support for standard normal
x_support = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf_norm = norm.pdf(x_support, mu, sigma)

# Set up initial sample size
np.random.seed(0)
n_samples = 300
samples_normal = np.random.normal(mu, sigma, size=n_samples)

def make_normal_empirical_convergence_fig(step):
    drawn = samples_normal[step]
    current_samples = samples_normal[:step+1]
    
    # Empirical histogram (normalize for PDF shape)
    counts, bin_edges = np.histogram(current_samples, bins=16, range=(x_support[0], x_support[-1]), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "Standard Normal Distribution PDF",
            "Normalized Empirical Histogram"
        )
    )

    # --- Left subplot: true normal PDF with marker and vertical line ---
    fig.add_trace(
        go.Scatter(
            x=x_support,
            y=pdf_norm,
            mode='lines',
            line=dict(color='#B026FF', width=3),
            name="Standard Normal PDF",
            showlegend=False
        ),
        row=1, col=1
    )

    # Mark the drawn value (left)
    fig.add_trace(
        go.Scatter(
            x=[drawn],
            y=[norm.pdf(drawn, mu, sigma)],
            mode='markers',
            marker=dict(color='#FFD700', size=14, line=dict(color='black', width=2)),
            name="Latest Sample",
            showlegend=False
        ),
        row=1, col=1
    )

    # Add vertical yellow line at marker (left)
    fig.add_trace(
        go.Scatter(
            x=[drawn, drawn],
            y=[0, norm.pdf(drawn, mu, sigma)],
            mode='lines',
            line=dict(color='#FFD700', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # --- Right subplot: empirical histogram & theoretical curve ---
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.98,
            marker=dict(color='#00ffff', opacity=0.7),
            name="Empirical Distribution",
            showlegend=False
        ),
        row=1, col=2
    )

    # Overlay the standard normal PDF (right)
    fig.add_trace(
        go.Scatter(
            x=x_support,
            y=pdf_norm,
            mode='lines',
            line=dict(color='#B026FF', width=3, dash='dash'),
            name="Standard Normal PDF",
            showlegend=False
        ),
        row=1, col=2
    )

    # Note: legend is removed, so don't add legend-only traces

    # --- Axes ---
    fig.update_xaxes(title_text="x", row=1, col=1, range=[x_support[0], x_support[-1]])
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2, range=[x_support[0], x_support[-1]])
    # Configure right y-axis range to be 0 to .45
    fig.update_yaxes(title_text="Empirical Density", row=1, col=2, range=[0, 0.45])

    # --- Gridlines ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')

    # --- Layout ---
    fig.update_layout(
        height=480,
        width=960,
        title_text="Standard Normal Distribution vs Empirical Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=False,  # Remove legend
        margin=dict(l=50, r=20, b=80, t=70),
    )
    return fig

# --- Animation frames for normal empirical convergence ---
normal_frames = [
    go.Frame(data=make_normal_empirical_convergence_fig(step).data, name=str(step))
    for step in range(n_samples)
]

# --- Initial figure ---
fig_norm = make_normal_empirical_convergence_fig(0)
fig_norm.frames = normal_frames

# --- Play button ---
fig_norm.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.11,
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

fig_norm.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Likelihoods vs Probabilities
# 
# Unlike discrete random variables, probabilities can only be recovered by integrating the density function


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Normal Distribution Parameters ---
mu, sigma = 0, 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf = norm.pdf(x, mu, sigma)

# Initial slider points
k1_initial, k2_initial = 0.5, 1.0
idx_k1, idx_k2 = np.argmin(np.abs(x - k1_initial)), np.argmin(np.abs(x - k2_initial))
k1_val, k2_val = x[idx_k1], x[idx_k2]

y_max_pdf = np.max(pdf)*1.05

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f'Normal PDF (μ={mu}, σ={sigma})',
                    f'PDF with Shaded Area (P(X≤k_2))']
)

# PDF line (left) - use neon purple
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines',
                         line=dict(color='#B026FF', width=3),  # neon purple
                         name='PDF', hovertemplate='x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'),
              row=1, col=1)

# PDF line (right)
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines',
                         line=dict(color='#00ffff', width=3),
                         name='PDF', hovertemplate='x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'),
              row=1, col=2)

# Shaded fill area under PDF (right)
x_fill_initial = x[:idx_k2 + 1]
pdf_fill_initial = pdf[:idx_k2 + 1]
fig.add_trace(go.Scatter(
    x=np.concatenate([x_fill_initial, x_fill_initial[::-1]]),
    y=np.concatenate([pdf_fill_initial, np.zeros_like(pdf_fill_initial)]),
    fill='toself', mode='none',
    fillcolor='rgba(0,255,255,0.3)', name='P(X≤k2)', showlegend=False
), row=1, col=2)

from scipy.integrate import quad

# Compute P(X ≤ k2) for initial fill
prob_k2 = norm.cdf(k2_val, mu, sigma)
initial_title = (
    f'Normal Distribution (μ={mu}, σ={sigma})'
    f'<br>PDF Height at x={k1_val:.2f} is {pdf[idx_k1]:.4f}'
    f' | P(X≤{k2_val:.2f}) = {prob_k2:.4f}'
)

INITIAL_SHAPES = [
    dict(type='line', xref='x1', yref='y1', x0=k1_val, x1=k1_val, y0=0, y1=y_max_pdf,
         line=dict(color='red', width=3, dash='dot')),
    dict(type='line', xref='x2', yref='y2', x0=k2_val, x1=k2_val, y0=0, y1=y_max_pdf,
         line=dict(color='yellow', width=3, dash='dot')),
]

# --- Sliders ---
slider_k1_steps = []
for i, k in enumerate(x):
    pdf_k = pdf[i]
    prob_current = norm.cdf(k2_val, mu, sigma)
    step = dict(
        method="relayout",
        label="",
        args=[{
            "title.text": (
                f'Normal Distribution (μ={mu}, σ={sigma})'
                f'<br>PDF Height at x={k:.2f} is {pdf_k:.4f}'
                f' | P(X≤{k2_val:.2f}) = {prob_current:.4f}'
            ),
            "shapes[0].x0": k,
            "shapes[0].x1": k
        }],
        execute=True
    )
    slider_k1_steps.append(step)

slider_k2_steps = []
for i, k in enumerate(x):
    x_fill = x[:i + 1]
    pdf_fill = pdf[:i + 1]
    prob_k = norm.cdf(k, mu, sigma)
    step = dict(
        method="update",
        label="",
        args=[
            {
                "x": [x, x, np.concatenate([x_fill, x_fill[::-1]])],
                "y": [pdf, pdf, np.concatenate([pdf_fill, np.zeros_like(pdf_fill)])]
            },
            {
                "title.text": (
                    f'Normal Distribution (μ={mu}, σ={sigma})'
                    f'<br>PDF Height at x={k1_val:.2f} is {pdf[idx_k1]:.4f}'
                    f' | P(X≤{k:.2f}) = {prob_k:.4f}'
                ),
                "shapes[1].x0": k,
                "shapes[1].x1": k
            }
        ],
        execute=True
    )
    slider_k2_steps.append(step)

# --- Layout ---
# Move sliders further to the left and right so they clearly sit under their respective plots
fig.update_layout(
    height=550,
    title={'text': initial_title, 'y': 0.97, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=INITIAL_SHAPES,
    sliders=[
        dict(
            steps=slider_k1_steps,
            active=idx_k1,
            pad={"t": 50, "b": 0},
            yanchor='top', y=-0.10,
            x=0.08, len=0.30   # move left slider further left under left plot
        ),
        dict(
            steps=slider_k2_steps,
            active=idx_k2,
            pad={"t": 50, "b": 0},
            yanchor='top', y=-0.10,
            x=0.65, len=0.30   # move right slider further right under right plot
        )
    ]
)

# Axes
fig.update_xaxes(title_text='x', range=[mu - 4*sigma, mu + 4*sigma],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='f(x) (PDF)', range=[0, y_max_pdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_xaxes(range=[mu - 4*sigma, mu + 4*sigma],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(range=[0, y_max_pdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Statistics and Characteristics
# 
# Normal distributions are defined in terms of a mean and variance
# 
# $$X \sim N(\mu, \sigma^2)$$
# 
# Regardless of the shape of the normal distribution, all of the properties above that we saw for a standard normal ($N(0, 1)$) hold


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# Normal distributions: mean, sigma
norm_params = [
    (0, 1),    # Standard Normal
    (2, 0.8),  # Shifted mean, smaller sigma
    (-1, 1.5), # Lower mean, wider sigma
]

n_normals = len(norm_params)
colors = ['#B026FF', '#FF8F00', '#00D4A1']  # purple, orange, greenish

x = np.linspace(-5, 5, 500)

pdfs = []
for mu, sigma in norm_params:
    pdfs.append(norm.pdf(x, mu, sigma))

# We'll allow a slider to select which normal is current to "integrate"
slider_steps = []
shaded_colors = ['rgba(176,38,255,0.18)','rgba(255,143,0,0.18)','rgba(0,212,161,0.18)']

# Set initial shaded area up to some value k2
k2_init = 1.0
k2_idx = np.argmin(np.abs(x - k2_init))

x_fill_init = x[:k2_idx+1]

# Setup Subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'Normal PDFs: Various Mean & Variance',
        'Integrating PDF up to k2 (Area = Probability)',  # Removed LaTeX
    ]
)

# Left: show all PDFs together, with a legend and distinct colors
for i, (mu, sigma) in enumerate(norm_params):
    fig.add_trace(
        go.Scatter(
            x=x, y=pdfs[i],
            mode='lines',
            line=dict(color=colors[i], width=3),
            name=f"N({mu},{sigma}²)",  # No LaTeX
            legendgroup=f"group{i+1}",
        ),
        row=1, col=1
    )

# Match grid lines to right subplot
fig.update_xaxes(title_text='x', range=[-4.5, 4.5], showgrid=True, gridcolor='rgba(128,128,128,0.19)', row=1, col=1)
fig.update_yaxes(title_text='f(x) (PDF)', showgrid=True, gridcolor='rgba(128,128,128,0.19)', row=1, col=1)

# On the right: show one PDF at a time, with the integrated region under the curve up to k2
for i, (mu, sigma) in enumerate(norm_params):
    y_pdf = pdfs[i]
    x_fill = x[:k2_idx+1]
    y_fill = y_pdf[:k2_idx+1]
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pdf,
            mode='lines',
            line=dict(color=colors[i], width=3),
            name=f"N({mu},{sigma}²) (Active)",  # No LaTeX
            legendgroup=f"group{i+1}",
            showlegend=False,
            visible=(i == 0),
        ),
        row=1, col=2
    )
    # Fill under curve
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill='toself', mode='none',
            fillcolor=shaded_colors[i],
            name="P(X ≤ k2)",  # No LaTeX
            showlegend=False,
            visible=(i == 0),
        ),
        row=1, col=2
    )

# Compute the probability for each normal for initial k2
prob_k2_vals = [norm.cdf(k2_init, p[0], p[1]) for p in norm_params]

for i, (mu, sigma) in enumerate(norm_params):
    # Toggle only the selected normal's PDF and fill in the right panel
    step = dict(
        method="update",
        label=f"mu={mu}, sigma={sigma}",  # No LaTeX
        args=[
            {
                "visible": (
                    ([True]*n_normals) +
                    sum(([j==i, j==i] for j in range(n_normals)), [])
                )
            },
            {
                "title.text": (
                    f"<b>Normal PDFs (left): Various Mean & Variance</b>"
                    f"<br><b>Right:</b> N({mu},{sigma}²) | "  # No LaTeX
                    f"P(X ≤ {k2_init:.2f}) = {prob_k2_vals[i]:.4f}"
                )
            }
        ],
    )
    slider_steps.append(step)

# Construct slider to toggle between normals
fig.update_layout(
    height=500,
    title={'text': (
        f"<b>Normal PDFs (left): Various Mean & Variance</b>"
        f"<br><b>Right:</b> N({norm_params[0][0]},{norm_params[0][1]}²) | "
        f"P(X ≤ {k2_init:.2f}) = {prob_k2_vals[0]:.4f}"
    ), 'y':0.97, 'x':0.5, 'xanchor':'center'},
    font=dict(color='white'),
    showlegend=True,
    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)', y=0.96, x=0.04),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    sliders=[
        dict(
            steps=slider_steps,
            active=0,
            pad={"t": 45, "b": 0},
            yanchor='top', y=-0.10,
            x=0.56, len=0.40,
            currentvalue=dict(
                visible=True,
                prefix="Distribution: "
            ),
        ),
    ]
)

fig.update_xaxes(title_text='x', range=[-4.5, 4.5], showgrid=True, gridcolor='rgba(128,128,128,0.19)', row=1, col=2)
fig.update_yaxes(title_text='f(x) (PDF)', showgrid=True, gridcolor='rgba(128,128,128,0.19)', row=1, col=2)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### 🌉 Why do we care about the normal (Gaussian) distribution?
# 
# ##### 🎯 This is our bridge between probability and statistics!


# ---


# #### 3.) 📊 Central Limit Theorem (CLT)
# 
# Amazingly, even if we don't know the population distribution, the CLT gaurentees the empirical (sample) mean is normally distributed
# 
# #####  **Rough Proof using the Characteristic Function**
# 
# We first look at the equation for the standardized sample mean
# 
# $$Z_n = \frac{\sum_{i=1}^{n} X_i - n\mu}{\sigma\sqrt{n}}$$
# 
# Applying the characteristic function, the sum of independent random variables is the product of their CFs
# 
# $$\phi_{Z_n}(t) = \phi_{\sum_{i=1}^n \frac{Y_i}{\sqrt{n}}}(t) = \prod_{i=1}^{n} \phi_{Y_i}\left(\frac{t}{\sqrt{n}}\right) = \left[ \phi_{Y_1}\left(\frac{t}{\sqrt{n}}\right) \right]^n$$
# 
# After substituting in a Taylor series expansion we are left with
# 
# $$\phi_{Z_n}(t) = \left[ 1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right) \right]^n$$
# 
# Take the limit as the sample size goes to infinity and we see convergence to the definition of Euler's number $e$
# 
# $$\lim_{n \to \infty} \phi_{Z_n}(t) = \lim_{n \to \infty} \left[ 1 + \frac{-t^2/2}{n} \right]^n = e^{-t^2/2}$$
# 
# Which is the characteristic function of a standard normal (Gaussian) distribution - fully specifying the random varibiable!
# 
# **By Lévy's Continuity Theorem** 
# 
# Convergence of a characteristic function at $t = 0$ implies distribution convergence
# 
# Therefore, the standardized sample mean converges to a normal (Gaussian) distribution - regardless of the population distribution!
# 
# $$Z_n \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty$$


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Empirical Poisson Means are Normal 
# 
# $$x_i \sim X \sim Pois(\lambda) \quad\quad \frac{\frac{1}{n}\sum_{i = 1}^n x_i - \lambda}{\sqrt{\frac{\lambda}{n}}} \sim N(0, 1)$$


import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PARAMETERS ---
lmbda = 5
n_samples_per_iter = 1000
N_ITER = 500
poisson_support = np.arange(0, 20)

# --- True Poisson PMF ---
pmf_poisson = stats.poisson.pmf(poisson_support, lmbda)

# --- Generate empirical PMFs and sample means ---
empirical_pmfs = []
means = []
for _ in range(N_ITER):
    samples = np.random.poisson(lmbda, n_samples_per_iter)
    counts, _ = np.histogram(samples, bins=np.arange(-0.5, max(poisson_support)+1.5), density=True)
    empirical_pmfs.append(counts)
    means.append(samples.mean())

# --- Function to draw the figure for a given iteration ---
def make_sampling_hist_fig(iteration):
    emp_pmf = empirical_pmfs[iteration]
    means_so_far = np.array(means[:iteration+1])

    # Normalize the sample means (to show CLT convergence)
    normalized_means = (means_so_far - lmbda) / np.sqrt(lmbda / n_samples_per_iter)

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.5, 0.5],
                        subplot_titles=("Empirical Mass Function", "Normalized Sample Means (Z)"))

    # LEFT PANEL: Empirical vs true PMF
    fig.add_trace(go.Bar(x=poisson_support, y=emp_pmf,
                         marker=dict(color='#00ffff', opacity=0.8),
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=poisson_support, y=pmf_poisson,
                             mode='lines', line=dict(color='#d400ff', width=3, dash='dash'),
                             showlegend=False), row=1, col=1)

    # RIGHT PANEL: Histogram of normalized sample means
    n_bins = 80
    hist_y, hist_x = np.histogram(normalized_means, bins=n_bins, density=True)
    bin_centers = (hist_x[:-1] + hist_x[1:]) / 2
    fig.add_trace(go.Bar(x=bin_centers, y=hist_y,
                         marker=dict(color='#7fcfff', opacity=0.8),
                         showlegend=False), row=1, col=2)

    # --- Theoretical Standard Normal Overlay ---
    xx = np.linspace(-4, 4, 300)
    normal_pdf = stats.norm.pdf(xx)
    fig.add_trace(go.Scatter(x=xx, y=normal_pdf,
                             mode='lines', line=dict(color='red', width=3, dash='dash'),
                             showlegend=False), row=1, col=2)

    # --- Axis scaling ---
    fig.update_xaxes(title_text="k", range=[-0.5, 12.5], row=1, col=1)
    fig.update_yaxes(title_text="P(X=k)", range=[0, max(pmf_poisson)*1.2], row=1, col=1)

    fig.update_xaxes(title_text="Z = (mean - λ) / sqrt(λ/n)", range=[-4, 4], row=1, col=2)
    fig.update_yaxes(title_text="Density", range=[0, 0.45], row=1, col=2)

    # --- Styling ---
    fig.update_layout(
        height=420, width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        title_text=f"Iteration {iteration+1}: Normalized Sample Mean Convergence to N(0,1)",
        title_x=0.5,
        margin=dict(l=30, r=30, b=40, t=60)
    )
    return fig

# --- Animation frames ---
hist_frames = []
for i in range(N_ITER):
    f = make_sampling_hist_fig(i)
    hist_frames.append(go.Frame(data=f.data,
                                name=str(i),
                                layout=go.Layout(
                                    title_text=f"Iteration {i+1}: Normalized Sample Mean Convergence to N(0,1)"
                                )))

# --- Initial Figure ---
hist_fig = make_sampling_hist_fig(0)
hist_fig.frames = hist_frames

# --- Controls ---
hist_fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.12,
        'xanchor': 'center',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 60, 'easing': 'cubic-in-out'}
            }]
        }]
    }],
    transition={'duration': 60, 'easing': 'cubic-in-out'}
)

hist_fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### More Samples $\rightarrow$ More Confidence
# 
# The *variance of the sample mean* goes to zero as $n \rightarrow \infty$
# 
# $$
# \mathrm{Var}\left(\bar{X}_n\right) = \frac{\sigma^2}{n}
# $$
# 
# This ensures convergence to the sample mean as observed in the Law of Large Numbers (LLN)


import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- PARAMETERS ---
mu = 5
sigma = 2
sample_sizes = [10, 30, 100, 250, 500]
x = np.linspace(-5, 15, 400)

# --- Function to get sampling distribution PDF ---
def pdf_for_n(n):
    std_err = sigma / np.sqrt(n)
    return stats.norm.pdf(x, mu, std_err)

# --- Color palette for each frame ---
colors = ['#00FFFF', '#7FFF00', '#FFD700', '#FF7F50', '#FF1493']

# --- Initial figure (first n) ---
y0 = pdf_for_n(sample_sizes[0])
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y0,
    mode='lines',
    line=dict(color=colors[0], width=4),
    fill='tozeroy'
))

# --- Animation frames ---
frames = []
for n, c in zip(sample_sizes, colors):
    y = pdf_for_n(n)
    frames.append(go.Frame(
        data=[go.Scatter(x=x, y=y,
                         mode='lines',
                         line=dict(color=c, width=4),
                         fill='tozeroy')],
        name=str(n),
        layout=go.Layout(
            title_text=f"Sampling Distribution Tightening with Increasing n (n = {n})"
        )
    ))

# --- Mean reference line (μ = 5) ---
fig.add_trace(go.Scatter(
    x=[mu, mu],
    y=[0, 8],
    mode='lines',
    line=dict(color='red', width=3, dash='dash')
))

# --- Layout and styling ---
fig.frames = frames
fig.update_layout(
    title=f"Sampling Distribution Tightening Around μ = {mu}",
    xaxis_title="Sample Mean",
    yaxis_title="Density",
    xaxis_range=[3, 7],
    yaxis_range=[0, 5],
    showlegend=False,
    font=dict(size=16, color='white'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    width=900, height=450,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.2,  # lowered play button
        'xanchor': 'center',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 1000, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 700, 'easing': 'cubic-in-out'}
            }]
        }]
    }]
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# #### Statistics $\rightarrow$ Probabilities
# 
# Remember, we don't know the *population distribution* that is generating the uncertain event we are modeling
# 
# However, because we know that the sample mean converges to normality we can **generate probabilities** to assess the likelihood of different population data generating distributions!
# 
# **Theoretical Calibration Procedure (Moment Matching)**
# 
# 1.) Observe a mean return
# 
# 2.) Parameterize (calibrate) a normal distribution by observed mean and variance
# 
# 3.) Find the probability of different states of the world!
# 
# **Remark: Before you take to the comments my keyboard warrior friend, we are in the lab - I understand the notion of a leptokurtic return distribution, patience, we will get there**


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t, norm

# --- Simulate Non-Normal Stock Returns (fat-tailed) ---
np.random.seed(0)
df = 3  # degrees of freedom for t-dist
true_mu, true_sigma = 0.001, 0.02
returns = true_mu + true_sigma * t.rvs(df, size=5000)

# --- Fit a Normal Distribution to Sample ---
mu_hat = np.mean(returns)
sigma_hat = np.std(returns, ddof=1)

# --- X range for plotting PDFs ---
x = np.linspace(returns.min(), returns.max(), 600)

# --- Fixed means to slide over (hypothetical distributions) ---
slider_means = np.array([-0.012, -0.009, -0.003, 0.0, 0.003, 0.009, 0.012])

# --- Initial position ---
mu_initial = 0.0

# --- Calculate the likelihood of observing sample mean if that distribution were true ---
def likelihood_mu(mu, sample_mean, sigma, n):
    # Normal PDF for MLE mean under dist with mean mu
    # The likelihood of observing sample_mean as the sample mean from N(mu, sigma/sqrt(n))
    # For the demonstration, treat sigma known and n large (use sample size)
    return norm.pdf(sample_mean, loc=mu, scale=sigma/np.sqrt(n))

n = len(returns)
sample_mean = mu_hat
likelihoods = [likelihood_mu(mu, sample_mean, sigma_hat, n) for mu in slider_means]

# --- PDF for initial distribution ---
pdf_initial = norm.pdf(x, mu_initial, sigma_hat)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Sample of Stock Returns (Fat-Tailed)",
        f"Hypothetical Normal Distributions"
    ]
)

# --- LEFT: Histogram of returns ---
hist_y, hist_x = np.histogram(returns, bins=60, density=True)
bin_centers = (hist_x[:-1] + hist_x[1:]) / 2
fig.add_trace(go.Bar(
    x=bin_centers, y=hist_y,
    marker=dict(color='#00ffff', opacity=0.6),
    showlegend=False
), row=1, col=1)

# --- Mean line on left chart ---
fig.add_shape(
    type='line', x0=mu_hat, x1=mu_hat, y0=0, y1=max(hist_y)*1.1,
    line=dict(color='red', width=3, dash='dot'),
    xref='x1', yref='y1'
)

# --- RIGHT: Initial hypothetical normal PDF ---
fig.add_trace(go.Scatter(
    x=x, y=pdf_initial, mode='lines',
    line=dict(color='#B026FF', width=3),
    hovertemplate='x=%{x:.4f}<br>PDF=%{y:.4f}<extra></extra>',
    name="Hypothetical Distribution",
    showlegend=False
), row=1, col=2)

# --- Mark the observed sample mean ---
fig.add_trace(go.Scatter(
    x=[sample_mean], y=[norm.pdf(sample_mean, mu_initial, sigma_hat)],
    mode='markers',
    marker=dict(color='yellow', size=12, symbol='x'),
    name="Observed sample mean",
    showlegend=True,
    hovertemplate='Sample Mean: %{x:.4f}<extra></extra>'
), row=1, col=2)

# --- Prepare likelihood at initial mu ---
likelihood_at_initial = likelihood_mu(mu_initial, sample_mean, sigma_hat, n)

# --- Annotate likelihood ---
INITIAL_SHAPE = [
    dict(type='line', xref='x2', yref='y2',
         x0=sample_mean, x1=sample_mean, y0=0, y1=norm.pdf(sample_mean, mu_initial, sigma_hat),
         line=dict(color='yellow', width=3, dash='dot'))
]
INITIAL_ANNOTATION = [
    dict(
        x=sample_mean, y=norm.pdf(sample_mean, mu_initial, sigma_hat),
        xref="x2", yref="y2",
        text=f"Likelihood: {likelihood_at_initial:.3e}",
        showarrow=True, arrowhead=2, ax=60, ay=-40,
        font=dict(color='yellow', size=13),
        bgcolor='rgba(0,0,0,0.7)'
    )
]

# --- Slider steps: for each hypothetical mean, update right PDF and likelihood ---
slider_steps = []
for i, mu in enumerate(slider_means):
    pdf = norm.pdf(x, mu, sigma_hat)
    like = likelihood_mu(mu, sample_mean, sigma_hat, n)
    step = dict(
        method="update",
        label=f"{mu:+.3f}",
        args=[
            {
                "y": [hist_y, pdf, [norm.pdf(sample_mean, mu, sigma_hat)]],
                "x": [bin_centers, x, [sample_mean]]
            },
            {
                "title.text": (
                    f"Hypothetical Normal (μ={mu:.4f}, σ={sigma_hat:.4f})<br>"
                    f"Likelihood of observed sample mean: {like:.3e}"
                ),
                "shapes": [
                    dict(type='line', xref='x2', yref='y2',
                         x0=sample_mean, x1=sample_mean, y0=0,
                         y1=norm.pdf(sample_mean, mu, sigma_hat),
                         line=dict(color='yellow', width=3, dash='dot'))
                ],
                "annotations": [
                    dict(
                        x=sample_mean, y=norm.pdf(sample_mean, mu, sigma_hat),
                        xref="x2", yref="y2",
                        text=f"Likelihood: {like:.3e}",
                        showarrow=True, arrowhead=2, ax=60, ay=-40,
                        font=dict(color='yellow', size=13),
                        bgcolor='rgba(0,0,0,0.7)'
                    )
                ]
            }
        ],
        execute=True
    )
    slider_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=550, width=1000,
    title={'text': f"Hypothetical Normal (μ={mu_initial:+.3f}, σ={sigma_hat:.4f})<br>"
                   f"Likelihood of observed sample mean: {likelihood_at_initial:.3e}",
           'y': 0.96, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=INITIAL_SHAPE,
    annotations=INITIAL_ANNOTATION,
    sliders=[dict(
        steps=slider_steps,
        active=np.where(np.isclose(slider_means, mu_initial))[0][0],
        pad={"t": 50, "b": 0},
        yanchor='top', y=-0.10,
        x=0.65, len=0.30,
        currentvalue={"prefix": "Hypothetical μ = ", "font": {"color": "white", "size": 16}}
    )]
)

# --- Axis formatting ---
fig.update_xaxes(
    title_text='Return',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    row=1, col=1
)
fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    row=1, col=1
)
fig.update_xaxes(
    title_text='Return',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    range=[-0.08, 0.08],
    row=1, col=2
)
fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    range=[0, 13],
    row=1, col=2
)

fig.show()



# Thanks to the Central Limit Theorem (CLT) we don't have to make an assumption about the population distribution of sample mean returns and can find *actual* probabilities of different states of the world for our sample mean return!


# ###### ______________________________________________________________________________________________________________________________________


# ##### Applications of the Central Limit Theorem (CLT)
# 
# - Parameter Calibration
# 
# - Generating Probabilities
# 
# - Confidence Intervals
# 
# - Hypothesis Testing


# ###### ______________________________________________________________________________________________________________________________________


# ##### Application to Stock Returns
# 
# We can observe the mean return of NVDA over disjoint 90-day blocks
# 
# $$\implies \bar{X}_{90} \sim N(\mu, \sigma^2)$$


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Load NVDA close prices and calculate daily returns ---
df = pd.read_csv("NVDA_returns.csv")
df = df.iloc[::-1]  # chronological (oldest to newest)
returns = df['Close'].pct_change().dropna().values

# --- Use disjoint time blocks of 90 days ---
block_size = 90
num_blocks = len(returns) // block_size
block_means = [np.mean(returns[i*block_size:(i+1)*block_size]) for i in range(num_blocks)]
block_means = np.array(block_means)
n = block_size  # Each block is a sample of size block_size

# --- Fit normal distribution to the mean of each block (CLT on block means) ---
sample_mean = block_means.mean()
sample_std = block_means.std(ddof=1)
se_mean = sample_std / np.sqrt(num_blocks)

# --- Probability that block mean return is less than zero (CLT) ---
prob_less_than_zero = norm.cdf(0, loc=sample_mean, scale=se_mean)

# --- Extract year range for plot subtitle ---
df_dates = pd.to_datetime(df['Date'])
start_year = df_dates.min().year
end_year = df_dates.max().year

# --- Histogram of ALL DAILY RETURNS (LEFT) ---
hist_y, hist_x = np.histogram(returns, bins=30, density=True)
bin_centers = (hist_x[:-1] + hist_x[1:]) / 2

# --- Normal PDF for mean sampling distribution (block means, RIGHT) ---
x_mu = np.linspace(sample_mean - 5*se_mean, sample_mean + 5*se_mean, 400)
pdf_mu = norm.pdf(x_mu, loc=sample_mean, scale=se_mean)

# --- Calculate Y axis ranges ---
ylim_hist = max(hist_y) * 1.2
ylim_pdf  = max(pdf_mu) * 1.2

# --- Subplot setup: 1 row x 2 columns ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f"NVDA Daily Return Histogram<br>({start_year}–{end_year})",
        f"Block-mean (n={block_size}) Normal Distribution ({num_blocks} blocks)"
    ]
)

# --- LEFT: Actual returns histogram ---
fig.add_trace(go.Bar(
    x=bin_centers, y=hist_y,
    marker=dict(color="#00FFFF", opacity=0.85),
    name='Returns Histogram',
    showlegend=False,
), row=1, col=1)

# --- Mean line of all daily returns ---
daily_mean = np.mean(returns)
fig.add_trace(go.Scatter(
    x=[daily_mean, daily_mean],
    y=[0, ylim_hist],
    mode='lines',
    line=dict(color='red', width=3, dash='dash'),
    showlegend=False,
), row=1, col=1)

# --- RIGHT: Only normal fit (NO HISTOGRAM!), just the normal distribution ---
fig.add_trace(go.Scatter(
    x=x_mu, y=pdf_mu,
    mode='lines',
    line=dict(color='#B026FF', width=3),
    name='Normal Fit for Block Means',
    hovertemplate="Mean={x:.4f}<br>PDF={y:.4f}<extra></extra>",
    showlegend=False
), row=1, col=2)

# --- Mean line on the normal fit ---
fig.add_trace(go.Scatter(
    x=[sample_mean, sample_mean],
    y=[0, ylim_pdf],
    mode='lines',
    line=dict(color='red', width=3, dash='dash'),
    name='Sample Mean',
    showlegend=False
), row=1, col=2)

# --- Shade area P(mean < 0) on the sampling distribution (normal fit) ---
x_fill = x_mu[x_mu <= 0]
pdf_fill = norm.pdf(x_fill, loc=sample_mean, scale=se_mean)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_fill, x_fill[::-1]]),
    y=np.concatenate([pdf_fill, np.zeros_like(pdf_fill)]),
    fill='toself', mode='none',
    fillcolor='rgba(0,255,255,0.35)',
    showlegend=False
), row=1, col=2)

# --- Styling/annotations to match CLT theme ---
annot_mean = dict(
    x=daily_mean,
    y=ylim_hist * 0.74,
    xref='x1', yref='y1',
    text=f"Daily Mean: {daily_mean*100:.3f}%",
    showarrow=False,
    font=dict(color='red', size=16),
    bgcolor='rgba(255,255,255,0.12)'
)
annot_mu_norm = dict(
    x=sample_mean,
    y=ylim_pdf * 0.89,
    xref='x2', yref='y2',
    text=f"<b>Block Means: μ={sample_mean*100:.3f}%<br>σₘ={se_mean*100:.3f}%</b><br>P(μ < 0) = {prob_less_than_zero:.4f}",
    showarrow=False,
    font=dict(color='#B026FF', size=15),
    bgcolor='rgba(255,255,255,0.08)'
)

fig.update_layout(
    height=500, width=900,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=16, color='white'),
    showlegend=False,  # **REMOVE LEGEND**
    title=dict(
        text=f"NVDA Stock Returns and CLT (90-Day Block Means)",
        font=dict(size=22, color='white'),
        y=0.97, x=0.5, xanchor='center'
    ),
    annotations=[annot_mean, annot_mu_norm],
    margin=dict(l=40, r=30, t=70, b=40),
)

fig.update_xaxes(
    title_text='Return',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    row=1, col=1
)
fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    range=[0, ylim_hist],
    row=1, col=1
)
fig.update_xaxes(
    title_text=f'Block Mean (n={block_size})',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    row=1, col=2
)
fig.update_yaxes(
    title_text='Probability Density',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.2)',
    range=[0, ylim_pdf],
    row=1, col=2
)

fig.show()


# Calculate the probability that a disjoint 90-day block mean is negative
block_mean_neg_prob = prob_less_than_zero

# Avoid division by zero
if block_mean_neg_prob == 0:
    expected_blocks = np.inf
    expected_years = np.inf
else:
    expected_blocks = 1 / block_mean_neg_prob
    # There are trading_days_per_year / 90 non-overlapping 90-day blocks per year
    trading_days_per_year = 252
    blocks_per_year = trading_days_per_year / 90
    expected_years = expected_blocks / blocks_per_year

print(f"Expected years until a negative mean return in a disjoint 90-day block: {expected_years:.2f} years")


# ###### ______________________________________________________________________________________________________________________________________


# Count observed negative 90-day block means
neg_mean_indices = np.where(block_means < 0)[0]
num_neg_blocks = len(neg_mean_indices)

# Build a year vs. negative-mean-block indicator time series
# For plotting, assign the block mean to the last day of its block
block_end_indices = [(i+1)*block_size - 1 for i in range(num_blocks)]
block_end_dates = df_dates.iloc[block_end_indices].reset_index(drop=True)

# Boolean time series for negative block mean
neg_blocks_ts = np.zeros(num_blocks)
neg_blocks_ts[neg_mean_indices] = 1

import plotly.express as px

neg_blocks_plot = go.Figure()

# Main time series: 
# - Positive-block means: green dots (opacity 0.2)
# - Negative-block means: red dots
positive_mask = block_means >= 0
negative_mask = block_means < 0

# Plot positive (green, opacity=0.2) block means
neg_blocks_plot.add_trace(go.Scatter(
    x=block_end_dates[positive_mask],
    y=block_means[positive_mask],
    mode='markers',
    marker=dict(
        color='#00FF00',
        size=12,
        line=dict(color='gray', width=1),
        opacity=0.2
    ),
    name='Positive Block Means',
    hovertemplate="Date=%{x}<br>Mean Return=%{y:.4%}<extra></extra>",
    showlegend=False,
))

# Plot negative (red) block means
neg_blocks_plot.add_trace(go.Scatter(
    x=block_end_dates[negative_mask],
    y=block_means[negative_mask],
    mode='markers',
    marker=dict(
        color='red',
        size=12,
        line=dict(color='gray', width=1),
        opacity=1
    ),
    name='Negative Block Means',
    hovertemplate="Date=%{x}<br>Mean Return=%{y:.4%}<extra></extra>",
    showlegend=False,
))

# Add red vertical lines at each negative block
for idx in neg_mean_indices:
    date = block_end_dates.iloc[idx]
    neg_blocks_plot.add_shape(
        dict(
            type="line",
            x0=date, x1=date, 
            y0=min(block_means)-abs(min(block_means)*0.1), y1=max(block_means)+abs(max(block_means)*0.1),
            line=dict(color="red", width=2, dash="solid"),
            layer="below"
        )
    )

# Add a line at y=0
neg_blocks_plot.add_trace(go.Scatter(
    x=[block_end_dates.min(), block_end_dates.max()],
    y=[0, 0],
    mode='lines',
    line=dict(color='white', width=2, dash='dash'),
    showlegend=False,
    hoverinfo='skip'
))

neg_blocks_plot.update_layout(
    title=f"NVDA: 90-Day Block Mean Returns<br><span style='font-size:17px'>Total negative means: {num_neg_blocks} of {num_blocks} ({100*num_neg_blocks/num_blocks:.1f}%)</span>",
    xaxis_title="Block End Date",
    yaxis_title="90-Day Block Mean Return",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="white", size=17),
    height=430,
    width=850,
    margin=dict(l=40, r=30, t=80, b=40),
)
neg_blocks_plot.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.18)')
neg_blocks_plot.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.18)')

neg_blocks_plot.show()



print(f"Years required to observe the negative blocks we have: {expected_years*20:.2f} years")


# Something in our model appears to be incorrect, we've observed barely over two decades of returns and we are seeing far more negative returns than expected - why is this the case?


# ###### ______________________________________________________________________________________________________________________________________


# ##### Limitations and Considerations
# 
# It is true that the population distribution of sample means is normal, but it changes over time
# 
# This is why we are grossly underestimating the probability of losses


import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed and sample count
np.random.seed(42)
n_samples = 10000

# Parameters for time-varying normal distributions (drifting mean)
n_frames = 60
means = np.concatenate([
    np.linspace(0, 2, n_frames//3),  # Drift positive
    np.linspace(2, -2, n_frames//3), # Drift negative 
    np.linspace(-2, 0, n_frames//3)  # Return to neutral
])

# Parameters for the fixed normal distribution (mean and variance in % units)
fixed_mean = 0.2      
fixed_var = 0.05        
fixed_std = np.sqrt(fixed_var) # std dev

# Plotting range and precompute fixed PDF
x_range = np.linspace(-4, 4, 200)
fixed_pdf = stats.norm.pdf(x_range, loc=fixed_mean, scale=fixed_std)

frames = []
all_samples = []

for mean in means:
    X = np.random.normal(mean, 1, n_samples)
    kde_X = stats.gaussian_kde(X)
    
    # 10 new drifting samples (accumulate for histogram)
    samples = np.random.normal(mean, 1, 10)
    all_samples.extend(samples)

    # Top subplot: drifting (magenta) and fixed (orange) normals
    drifting_curve = go.Scatter(
        x=x_range,
        y=kde_X(x_range),
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)', width=2),
        name='Drifting Normal'
    )
    fixed_curve = go.Scatter(
        x=x_range,
        y=fixed_pdf,
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name='Fixed Normal (μ=0.2%, σ²=0.05%)'
    )
    # Bottom: histogram of all accumulated drifting samples
    histogram = go.Histogram(
        x=all_samples,
        nbinsx=20,
        name='Sample Distribution',
        marker_color='rgba(0, 255, 255, 0.6)'
    )
    frames.append(
        go.Frame(
            data=[drifting_curve, fixed_curve, histogram]
        )
    )

# Build subplot figure
fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])

# Add initial traces: drifting, fixed, histogram
fig.add_trace(frames[0].data[0], row=1, col=1)  # Drifting
fig.add_trace(frames[0].data[1], row=1, col=1)  # Fixed
fig.add_trace(frames[0].data[2], row=2, col=1)  # Histogram

# Animation setup
fig.frames = frames

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
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.update_layout(
    height=600,
    width=900,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Drifting Return Distribution with Accumulated Samples<br>and Fixed Normal Reference'
)

# Top subplot axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-4, 4],
    row=1, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

# Bottom subplot axes and limits
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-4, 4],
    row=2, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[0, 100],
    row=2, col=1
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Central Limit Theorem (CLT) Provides a Point-in-Time Probability Snapshot
# 
# **Trading Signal Example**
# 
# In some cases the *assumption* that the sample reflects the data generating distribution is less violent than others
# 
# For example, if I am trying to assess the sentiment of an equity before *9:20am* and I have a sample of $50$ unique documents, after calibrating a normal distribution I can assess the probability of drawing an *incorrect* signal (observed mean) given a threshold for inclusion in our portfolio.  
# 
# If the probability is too low, I won't include the equity in that bucket and I won't trade it in either the L/S leg.  


# Example: Recentered Normal Distribution at Threshold — Assessing Probability of Observed Mean

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# Suppose we threshold accept for inclusion at a certain sample mean, e.g., 0.05
threshold_mean = 0.05
sample_size = 50
# Observed mean sentiment from sample
observed_mean = 0.15
observed_std = 0.3

# CLT std deviation
clt_std = observed_std / np.sqrt(sample_size)

# Center the normal under the null hypothesis (that true mean is right at the inclusion threshold)
center = threshold_mean

x = np.linspace(-0.1, 0.4, 300)
pdf = norm.pdf(x, loc=center, scale=clt_std)

fig = go.Figure()

# Main normal curve (centered at threshold for inclusion)
fig.add_trace(go.Scatter(
    x=x,
    y=pdf,
    mode="lines",
    line=dict(color='orange', width=3, dash="dash"),
    name="Null Sampling Distribution<br>(mean = threshold)"
))

# Vertical line for actually observed sample mean
fig.add_trace(go.Scatter(
    x=[observed_mean, observed_mean],
    y=[0, norm.pdf(observed_mean, loc=center, scale=clt_std)],
    mode="lines",
    line=dict(color="magenta", width=3),
    name="Observed Mean"
))

# Compute (one-sided) probability (p-value) of seeing a sample mean as large or larger than observed_mean, if true mean is threshold
p_value = 1 - norm.cdf(observed_mean, loc=center, scale=clt_std)

# Fill the right-side tail, i.e., region with mean ≥ observed_mean
fill_x = np.linspace(observed_mean, x[-1], 100)
fill_y = norm.pdf(fill_x, loc=center, scale=clt_std)
fig.add_trace(go.Scatter(
    x=np.concatenate([fill_x, fill_x[::-1]]),
    y=np.concatenate([fill_y, np.zeros_like(fill_y)]),
    fill="toself",
    fillcolor='rgba(0,255,255,0.3)',
    line=dict(color="rgba(0,0,0,0)"),
    name=f"P(≥ observed mean) = {p_value:.4f}"
))

fig.update_layout(
    title="CLT Sampling Distribution under Threshold: P(Observed Mean or Higher)",
    xaxis_title="Sentiment (Sample Mean)",
    yaxis_title="Probability Density",
    height=420,
    width=850,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(128,128,128,0.20)",
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(128,128,128,0.20)",
)

fig.show()

# -- Interpretation text for this plot (to be displayed, not code): --
# "The orange dashed curve is the CLT sampling distribution for mean sentiment if the 'true' mean were exactly at our inclusion threshold.
# The magenta line is the mean we actually observed. The shaded right area is the probability of observing a mean this favorable (or greater) 
# just by chance — if this probability is very low (e.g., less than 5%), we are confident the sample is unusually favorable and may include the equity;
# otherwise, if it's too high, our signal could easily have arisen by luck, so we might skip trading it."


# In this case, the distribution does change over time (we know this)
# 
# But we need to assess the likelihood of this state of the world *now* to make a trading decision
# 
# Even though the distribution changes, it may be relatively stable *now* - quite a useful concept!


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Random variables define a set of outcomes with accompanying probabilities or likelihoods (discrete / continuous)
# - Anytime we draw a random variable we are dealing with an empirical distribution that can generate statistics
# - If draws are from the same population distribution (i.e. it is time invariant) empirical statistics and distributions converge by the LLN, this is not the case in practice as we are battling time variance in distributions and subsequent statistics. . .
# - Given draws from a population distribution are random variables, statistics (being a function of the empirical distribution) are also random variables, what distribution do they follow?
# - The distribution of sample means follows a normal (Gaussian) distribution regardless of the population or data generating distribution, this is quite literally our bridge between theory and practice, statistics and probability
# - If the population or data generating distribution is fixed the distribution of sample means will converge to a normal distribution as the sample size becomes arbitrarily large and probabilities will be precise in the frequentist sense
# - In reality, population or data generating distributions are **NOT** fixed and are time variant leading us to generate incorrect probabilities and draw statistically incorrect conclusions
# - Though distributions in reality change over time, the CLT can offer a **snapshot** which is useful locally for generating probabilities as we saw in the trading signal example where the sentiment distribution producing a trading decision is likely to be more stable in that short region of time (minutes) required to generate a decision than it is to be stable over a series of days (which doesn't matter nearly as much as we will continue to recalibrate our model to new data)
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Reinforcement Learning for Delta Hedging
# - Approximating Pricing Functionals using Neural Networks
# 
# 
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Neural Network Stochastic Volatility Model Calibration
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

