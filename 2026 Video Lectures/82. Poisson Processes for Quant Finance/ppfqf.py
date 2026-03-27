# ### 📈 Poisson Processes for Quant Finance
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)
# 
# - [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)
# 
# - [Central Limit Theorem for Quant Finance](https://youtu.be/q2era-4pnic)
# 
# - [Why Quant Models Break](https://youtu.be/brdG1TmsPlw)
# 
# - [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)
# 
# - [The 5 Papers That Build Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)
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
# ##### [👾 Join the Quant Guild Discord Server](discord.com/invite/MJ4FU2c6c3)
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
# #### 1.) 🎣 Poisson Distribution
# 
# - Stationary Distributions
# 
# - Simulating Poisson Distributions
# 
# - Non-Stationary Distributions
# 
# #### 2.) ⏳ Exponential Distribution
# 
# - Definition, Connection to Poisson Process
# 
# - Memoryless Property
# 
# #### 3.) 📈 Poisson Process
# 
# *Theory*
# 
# - Definition, Simulation, Visualization 
# 
# - Time Homogeneous vs. Time Inhomogeneous
# 
# *Practice*
# 
# - Modeling Trade Arrivals (Practice)
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎣 Poisson Distribution
# 
# The Poisson distribution is used to model "rare" events over a particular time interval 
# 
# $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad P(X \leq k) = \sum_{i=0}^k \frac{\lambda^i e^{-\lambda}}{i!}$$
# 
# **Examples in Finance Where Poisson Distributions**
# 
#  - Modeling the number of trades occurring in a given time window on an exchange.
#  - Counting the number of credit defaults in a fixed period among a portfolio of bonds.
#  - Number of operational risk loss events (e.g. fraud cases) per quarter at a bank.
#  - The count of insurance claims filed per day for a particular policy type.
#  - Rare jump occurrences in market prices (when modeled discretely).


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Poisson Distribution Parameters ---
lam = 4  # mean rate (λ)
k_max = 18
k = np.arange(0, k_max + 1)
pmf = poisson.pmf(k, lam)
cdf = poisson.cdf(k, lam)

# Initial slider values
k1_initial = 4
k2_initial = 8
idx_k1 = int(np.clip(k1_initial, 0, k_max))
idx_k2 = int(np.clip(k2_initial, 0, k_max))
k1_val = k[idx_k1]
k2_val = k[idx_k2]

y_max_pmf = np.max(pmf)*1.08
y_max_cdf = 1.08

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f'Poisson PMF (λ={lam})',
                    f'Poisson CDF (λ={lam})']
)

# PMF (stem plot) and CDF (step plot)
fig.add_trace(go.Bar(x=k, y=pmf, width=0.7,
                     marker_color='#39ff14',
                     name='PMF', hovertemplate='k=%{x}<br>PMF=%{y:.4f}<extra></extra>'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=k, y=cdf, mode='lines+markers',
                         line=dict(color='#00ffff', width=3),
                         marker=dict(color='#00ffff', size=7),
                         name='CDF', hovertemplate='k=%{x}<br>P(X≤k)=%{y:.4f}<extra></extra>'),
              row=1, col=2)

# Shaded region for CDF up to k2
k_fill = k[:idx_k2+1]
cdf_fill = cdf[:idx_k2+1]
fig.add_trace(go.Scatter(
    x=np.concatenate([k_fill, [k_fill[-1]]]),
    y=np.concatenate([cdf_fill, [0]]),
    fill='tozeroy', mode='none',
    fillcolor='rgba(0,255,255,0.3)', name='P(X≤k2)', showlegend=False
), row=1, col=2)

# --- Initial Title & Lines ---
initial_title = (
    f'Poisson Distribution (λ={lam})'
    f'<br>PMF Height at k={k1_val} is {pmf[idx_k1]:.4f}'
    f' | P(X≤{k2_val}) = {cdf[idx_k2]:.4f}'
)

INITIAL_SHAPES = [
    dict(type='line', xref='x1', yref='y1',
         x0=k1_val, x1=k1_val, y0=0, y1=y_max_pmf,
         line=dict(color='red', width=3, dash='dot')),
    dict(type='line', xref='x2', yref='y2',
         x0=k2_val, x1=k2_val, y0=0, y1=y_max_cdf,
         line=dict(color='yellow', width=3, dash='dot'))
]

# --- Sliders ---
slider_k1_steps = []
for idx, kk in enumerate(k):
    pmf_k = pmf[idx]
    step = dict(
        method="relayout",
        label=str(kk),
        args=[{
            "title.text": (
                f'Poisson Distribution (λ={lam})'
                f'<br>PMF Height at k={kk} is {pmf_k:.4f}'
                f' | P(X≤{k2_val}) = {cdf[idx_k2]:.4f}'
            ),
            "shapes[0].x0": kk,
            "shapes[0].x1": kk
        }],
        execute=True
    )
    slider_k1_steps.append(step)

slider_k2_steps = []
for idx, kk in enumerate(k):
    cdf_k = cdf[idx]
    k_fill = k[:idx+1]
    cdf_fill = cdf[:idx+1]
    step = dict(
        method="update",
        label=str(kk),
        args=[
            {
                "x": [k, k, np.concatenate([k_fill, [k_fill[-1]]])],
                "y": [pmf, cdf, np.concatenate([cdf_fill, [0]])]
            },
            {
                "title.text": (
                    f'Poisson Distribution (λ={lam})'
                    f'<br>PMF Height at k={k1_val} is {pmf[idx_k1]:.4f}'
                    f' | P(X≤{kk}) = {cdf_k:.4f}'
                ),
                "shapes[1].x0": kk,
                "shapes[1].x1": kk
            }
        ],
        execute=True
    )
    slider_k2_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=480,
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
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.10, len=0.33,
            currentvalue=dict(prefix="k₁ = ")
        ),
        dict(
            steps=slider_k2_steps,
            active=idx_k2,
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.52, len=0.33,
            currentvalue=dict(prefix="k₂ = ")
        )
    ]
)

# Axes
fig.update_xaxes(title_text='k', range=[-0.5, k_max + 0.5],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=1)
fig.update_yaxes(title_text='P(X=k)', range=[0, y_max_pmf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=1)
fig.update_xaxes(title_text='k', range=[-0.5, k_max + 0.5],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=2)
fig.update_yaxes(title_text='P(X≤k)', range=[0, y_max_cdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=2)

fig.show()



# **Key Assumptions**
# 
# - Independence: Events aren't linked (no covariance structure), one happening doesn't make another more or less likely
# - Stationarity: There is a constant rate $\lambda$ over time interval
# - Simultaneity: Two events can not happen at the same exact infinitesimal instant
# - Discrete Counts: Events counts over the time interval are $\in \mathbb{N}$ (including $0$)


# ###### ______________________________________________________________________________________________________________________________________


# ##### Simulating Poisson Distributions
# 
# Poisson distributions can be simulated by choosing $\lambda$ (in the MLE sense, typically the average number of events over the desired interval)
# 
#  $$\displaystyle \frac{\partial}{\partial \lambda} \log L(\lambda) = 0 \implies \hat{\lambda}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i$$
# 
# For example, measure the *number of jumps in a stock price* over a 5 day period, do this many times and take the average
# 
# The Poisson distribution will then model the number of jumps in a stock price over a 5 day period


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Simulation Parameters ---
np.random.seed(42)  # Fixed seed so the 'diffusion' part of the stock stays constant
S0 = 100
T = 1.0
dt = 1/252
steps = int(T/dt)
time = np.linspace(0, T, steps)

# Market Parameters
mu = 0.10       # Drift
sigma = 0.15    # Volatility (Diffusion)
jump_mean = -0.05 # Average jump size (-5% drop)
jump_std = 0.01   # Jump size volatility

# Pre-calculate the fixed Brownian Motion component
# We do this so the "normal" volatility doesn't change when we move the slider, only the jumps do.
brownian_increments = np.random.normal(0, np.sqrt(dt), steps)
drift_component = (mu - 0.5 * sigma**2) * dt
diffusion_component = sigma * brownian_increments

# --- Helper Functions ---
def generate_path(lam_param):
    """
    Generates a price path with jumps defined by intensity lam_param.
    Returns: price path (array), number of jumps (int)
    """
    # Poisson Jump Times: Prob of jump in dt ~ lambda * dt
    # We use a distinct seed/random call here so jumps vary naturally
    jump_prob = lam_param * dt
    is_jump = np.random.rand(steps) < jump_prob
    num_jumps = np.sum(is_jump)
    
    # Calculate Jumps
    # Jumps are normally distributed around jump_mean
    jump_sizes = np.random.normal(jump_mean, jump_std, steps)
    
    # Combine: If jump, add jump_size log-return, else 0
    total_jump_log_ret = np.zeros(steps)
    total_jump_log_ret[is_jump] = np.log(1 + jump_sizes[is_jump])
    
    # Total Log Returns = Drift + Diffusion + Jumps
    log_returns = drift_component + diffusion_component + total_jump_log_ret
    
    # Construct Price Path
    path = S0 * np.exp(np.cumsum(log_returns))
    return path, num_jumps

def get_calibrated_pmf(k_observed, k_axis):
    """
    Returns the PMF of the Poisson distribution 'calibrated' 
    to the observed count (MLE estimator where lambda_hat = k_observed / T).
    Since T=1, lambda_hat = k_observed.
    """
    if k_observed == 0:
        # Degenerate case: if 0 jumps, probability mass is 1 at k=0
        pmf = np.zeros_like(k_axis, dtype=float)
        pmf[0] = 1.0
        return pmf
    return poisson.pmf(k_axis, k_observed)

# --- Initial State ---
lambda_min = 0.5
lambda_max = 20
lambda_steps_count = 40
lambda_vals = np.linspace(lambda_min, lambda_max, lambda_steps_count)

# Initial values
initial_lam = 5.0
initial_path, initial_k_obs = generate_path(initial_lam)

k_max_plot = 35
k_axis = np.arange(0, k_max_plot)
initial_pmf = get_calibrated_pmf(initial_k_obs, k_axis)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Stock Price Process (Merton Jump Diffusion)", "Calibrated Poisson Distribution"]
)

# Trace 1: Stock Path
path_trace = go.Scatter(
    x=time, y=initial_path,
    mode='lines',
    line=dict(color='#00ffff', width=2),
    name='Stock Price'
)

# Trace 2: Calibrated Distribution
bar_trace = go.Bar(
    x=k_axis, y=initial_pmf,
    marker_color='#39ff14',
    name='Calibrated PMF',
    hovertemplate='k=%{x}<br>Prob=%{y:.4f}<extra></extra>'
)

# Add Traces
fig.add_trace(path_trace, row=1, col=1)
fig.add_trace(bar_trace, row=1, col=2)

# Annotation for the Observed Jumps (Vertical Line on Right Plot)
# We show where the actual observation falls on the distribution
obs_shape = dict(
    type='line', xref='x2', yref='y2',
    x0=initial_k_obs, x1=initial_k_obs, 
    y0=0, y1=max(initial_pmf)*1.1,
    line=dict(color='red', width=3, dash='dot')
)

fig.update_layout(shapes=[obs_shape])

# --- Slider Logic ---
slider_steps = []

for lam in lambda_vals:
    # 1. Simulate Process with True Lambda
    sim_path, k_obs = generate_path(lam)
    
    # 2. Calibrate Distribution (Estimate Lambda based on k_obs)
    # The right plot shows the distribution implied by the specific path we just saw
    calib_pmf = get_calibrated_pmf(k_obs, k_axis)
    
    y_max_pmf = max(calib_pmf) if max(calib_pmf) > 0 else 1.0
    
    step = dict(
        method="update",
        label=f'{lam:.1f}',
        args=[
            {
                "y": [sim_path, calib_pmf],
                "x": [time, k_axis]
            },
            {
                "title.text": (
                    f"True Parameter (Lambda): {lam:.1f}  |  "
                    f"Observed Jumps: {k_obs}<br>"
                    f"<span style='font-size: 10px'>Right plot calibrated to Obs. Jumps (Lambda_est = {k_obs})</span>"
                ),
                # Update the red vertical line to the new observed k
                "shapes[0].x0": k_obs,
                "shapes[0].x1": k_obs,
                "shapes[0].y1": y_max_pmf * 1.1,
                # Dynamic Y-axis scaling for the bar chart
                "yaxis2.range": [0, y_max_pmf * 1.2]
            }
        ]
    )
    slider_steps.append(step)

# --- Final Layout Polish ---
fig.update_layout(
    height=500,
    title={'text': f"True Parameter (Lambda): {initial_lam} | Observed Jumps: {initial_k_obs}", 'x':0.5},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    sliders=[dict(
        active=int(np.searchsorted(lambda_vals, initial_lam)),
        currentvalue={"prefix": "Jump Intensity (Lambda): "},
        pad={"t": 50},
        steps=slider_steps
    )]
)

# Axis Styling
fig.update_xaxes(title_text='Time (Years)', row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Price ($)', row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Number of Jumps (k)', row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Probability', row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### As an Aside, Poisson Distributions and Processes are Helpful for Many Things
# 
# Interestingly, there are many applications of the Poisson distribution even beyond modeling real world phenonenon
# 
# For example, in the market-making games available on https://quantguild, I use a Poisson distribution with a lambda value that is a function of spread size
# 
#  For instance, the lambda is set as a function of the spread size. Suppose we define:
#  
#  $$\lambda_s = \frac{A}{\text{spread}^b}$$
#  
#  where $A$ and $b$ are constants controlling the scale and sensitivity of arrival rates to the spread.
#  As the spread increases, $\lambda_s$ decreases—fewer trades occur.
# 
# If the player widens the spread too much nobody will trade with them (lambda gets much lower) if the spread is reasonable around the mid price more liquidity will be provided to the market 
# 
# Then the actual trade arrivals $T$ are just drawn from 
# 
# $$T \sim Pois(\lambda_s)$$
# 
# for whatever spread size the player chooses
# 
# This will become even more clear in the Poisson processes section below. . .


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Poisson Distribution Parameters ---
A = .8    # Controls overall scale of lambda; tweak as desired
b = 1.1   # Controls rate sensitivity to spread; tweak as desired

spread_min = 0.1
spread_max = 1.0
spread_steps = 25
spread_values = np.round(np.linspace(spread_min, spread_max, spread_steps), 2)

def lam_func(spread):
    return A / (spread ** b)

k_max = 18
k = np.arange(0, k_max + 1)

# Initial slider values
spread_initial_idx = 8  # initial spread index (spread ~ 1.5)
spread_initial = spread_values[spread_initial_idx]
lam_initial = lam_func(spread_initial)

k1_initial = 4
k2_initial = 8
idx_k1 = int(np.clip(k1_initial, 0, k_max))
idx_k2 = int(np.clip(k2_initial, 0, k_max))
k1_val = k[idx_k1]
k2_val = k[idx_k2]

def get_dist_arrays(lam):
    pmf = poisson.pmf(k, lam)
    cdf = poisson.cdf(k, lam)
    return pmf, cdf

pmf, cdf = get_dist_arrays(lam_initial)
y_max_pmf = np.max([poisson.pmf(k, lam_func(s)).max() for s in spread_values]) * 1.1
y_max_cdf = 1.08

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Poisson PMF for lambda(spread)", "Poisson CDF for lambda(spread)"]
)

bar_trace = go.Bar(
    x=k, y=pmf, width=0.7,
    marker_color='#39ff14',
    name='PMF',
    hovertemplate=(
        f'k=%{{x}}<br>PMF=%{{y:.4f}}<br>lambda(spread)={lam_initial:.2f}<br>spread={spread_initial:.2f}<extra></extra>'
    )
)

line_trace = go.Scatter(
    x=k, y=cdf, mode='lines+markers',
    line=dict(color='#00ffff', width=3),
    marker=dict(color='#00ffff', size=7),
    name='CDF',
    hovertemplate=(
        f'k=%{{x}}<br>P(X<=k)=%{{y:.4f}}<br>lambda(spread)={lam_initial:.2f}<br>spread={spread_initial:.2f}<extra></extra>'
    )
)

# Shaded region for CDF up to k2
k_fill = k[:idx_k2+1]
cdf_fill = cdf[:idx_k2+1]
cdf_shade_trace = go.Scatter(
    x=np.concatenate([k_fill, [k_fill[-1]]]),
    y=np.concatenate([cdf_fill, [0]]),
    fill='tozeroy', mode='none',
    fillcolor='rgba(0,255,255,0.3)', name='P(X<=k2)', showlegend=False
)

fig.add_trace(bar_trace, row=1, col=1)
fig.add_trace(line_trace, row=1, col=2)
fig.add_trace(cdf_shade_trace, row=1, col=2)

# --- Initial Title & Lines ---
initial_title = (
    f"Poisson Distribution with lambda=lambda(spread)={lam_initial:.3f}, "
    f"spread={spread_initial:.2f}<br>"
    f"PMF at k={k1_val}: {pmf[idx_k1]:.4f} | "
    f"P(X<={k2_val}): {cdf[idx_k2]:.4f}<br>"
    "lambda_s = A / spread^b    "
    f"A={A}, b={b}"
)

INITIAL_SHAPES = [
    dict(type='line', xref='x1', yref='y1',
         x0=k1_val, x1=k1_val, y0=0, y1=y_max_pmf,
         line=dict(color='red', width=3, dash='dot')),
    dict(type='line', xref='x2', yref='y2',
         x0=k2_val, x1=k2_val, y0=0, y1=y_max_cdf,
         line=dict(color='yellow', width=3, dash='dot'))
]

# --- Sliders ---
# Slider for spread
spread_slider_steps = []
for spread_idx, spread in enumerate(spread_values):
    lam_s = lam_func(spread)
    pmf_s, cdf_s = get_dist_arrays(lam_s)

    y_pmf = pmf_s
    y_cdf = cdf_s
    k_fill_s = k[:idx_k2+1]
    cdf_fill_s = cdf_s[:idx_k2+1]
    # For PMF and CDF, update the y for the first and second trace, and shaded region for third
    step = dict(
        method="update",
        label=f'{spread:.2f}',
        args=[
            {
                "y": [
                    y_pmf,  # bar
                    y_cdf,  # line
                    np.concatenate([cdf_fill_s, [0]])  # shaded area
                ],
                "x": [
                    k, k, np.concatenate([k_fill_s, [k_fill_s[-1]]])
                ]
            },
            {
                "title.text":
                    f"Poisson Distribution with lambda=lambda(spread)={lam_s:.3f}, "
                    f"spread={spread:.2f}<br>"
                    f"PMF at k={k1_val}: {y_pmf[idx_k1]:.4f} | "
                    f"P(X<={k2_val}): {y_cdf[idx_k2]:.4f}<br>"
                    "lambda_s = A / spread^b    "
                    f"A={A}, b={b}"
            }
        ],
        execute=True
    )
    spread_slider_steps.append(step)

# Slider for k2 (adjust CDF vertical line & shaded region)
slider_k2_steps = []
for idx, kk in enumerate(k):
    k_fill = k[:idx+1]
    cdf_fill = cdf[:idx+1]
    step = dict(
        method="relayout",
        label=str(kk),
        args=[{
            # Adjust vertical line on CDF panel for k2
            "shapes[1].x0": kk,
            "shapes[1].x1": kk
        }],
        execute=True
    )
    slider_k2_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=480,
    title={'text': initial_title, 'y': 0.97, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=INITIAL_SHAPES,
    sliders=[
        dict(
            steps=spread_slider_steps,
            active=spread_initial_idx,
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.10, len=0.78,
            currentvalue=dict(prefix="spread = "),
            name="spread"
        ),
        dict(
            steps=slider_k2_steps,
            active=idx_k2,
            pad={"t": 50},
            yanchor='top', y=-0.13,
            x=0.65, len=0.23,
            currentvalue=dict(prefix="k2 = "),
            name='k2',
            visible=False # auxiliary for completeness, but hide (advanced live shading would require update per k2 per spread)
        )
    ]
)

# Axes
fig.update_xaxes(title_text='k', range=[-0.5, k_max + 0.5],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=1)
fig.update_yaxes(title_text='P(X=k)', range=[0, y_max_pmf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=1)
fig.update_xaxes(title_text='k', range=[-0.5, k_max + 0.5],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=2)
fig.update_yaxes(title_text='P(X<=k)', range=[0, y_max_cdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=2)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### A Catch-22 in Modeling the Real World: Non-Stationarity
# 
# Of course this in reality is a bit of a catch-22, distributions are not i.i.d and change over time
# 
# The Poisson distribution attempts to model arrivals *over time* which is inheritely changing
# 
# Some things we model are subject to change more violently over time than others, which will of course impact the efficacy of our probabilities, distributions, and statistics


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Parameters ---
lam_est = 100   # Estimated / Old parameter
lam_act = 250   # Actual / Real parameter
n_draws = 50    # Number of simulation draws

# Generate domain
x_min, x_max = 40, 360
x = np.arange(x_min, x_max)

# Calculate PMFs
pmf_est = poisson.pmf(x, lam_est)
pmf_act = poisson.pmf(x, lam_act)

# Simulate Data
np.random.seed(42)
simulated_draws = poisson.rvs(lam_act, size=n_draws)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'<b>Model Comparison</b>',
        f'<b>Live Simulation (λ={lam_act})</b>'
    ],
    horizontal_spacing=0.1
)

# --- LEFT PLOT: Static Distributions ---
# Trace 0: Estimated PMF
fig.add_trace(go.Scatter(
    x=x, y=pmf_est, mode='lines',
    line=dict(color='#00ffff', width=3), # Cyan
    fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)',
    name=f'Est (λ={lam_est})',
    hovertemplate='x=%{x}<br>Prob=%{y:.4f}<extra></extra>'
), row=1, col=1)

# Trace 1: Actual PMF
fig.add_trace(go.Scatter(
    x=x, y=pmf_act, mode='lines',
    line=dict(color='#39ff14', width=3), # Neon Green
    fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)',
    name=f'Actual (λ={lam_act})',
    hovertemplate='x=%{x}<br>Prob=%{y:.4f}<extra></extra>'
), row=1, col=1)

# --- RIGHT PLOT: Initial State ---
# Trace 2: Histogram (Initialize with first point)
fig.add_trace(go.Histogram(
    x=[simulated_draws[0]], 
    xbins=dict(start=x_min, end=x_max, size=5),
    marker=dict(color='#39ff14', line=dict(color='white', width=1)),
    opacity=0.8,
    name='Samples'
), row=1, col=2)

# --- Animation Frames ---
frames = []

title_left = f'<b>Model Comparison</b>'
title_right = f'<b>Live Simulation (λ={lam_act})</b>'

for i in range(1, n_draws + 1):
    current_data = simulated_draws[:i]
    current_min = np.min(current_data)

    # Calculate Probability
    prob_gt_min = 1 - poisson.cdf(current_min, lam_est)
    
    # --- CHANGE 1: Format as Percent ---
    if prob_gt_min > 0.0001:
        prob_text = f"{prob_gt_min:.2%}" # e.g., 12.50%
    else:
        prob_text = "< 0.01%"

    frames.append(go.Frame(
        # Update ONLY the histogram trace (trace index 2)
        data=[
            go.Histogram(
                x=current_data,
                autobinx=False,
                xbins=dict(start=x_min, end=x_max, size=5),
                marker=dict(color='#39ff14', line=dict(color='white', width=1)),
                opacity=0.8,
                name='Samples'
            )
        ],
        traces=[2],  # <-- critical: apply this frame's data to trace #2 only
        layout=go.Layout(
            shapes=[
                dict(
                    type="line", xref="x2", yref="paper",
                    x0=current_min, x1=current_min, y0=0, y1=1,
                    line=dict(color="red", width=3, dash="dot")
                )
            ],
            annotations=[
                dict(text=title_left,  x=0.225, y=1.1, xref="paper", yref="paper",
                     showarrow=False, font=dict(color='white', size=14)),
                dict(text=title_right, x=0.775, y=1.1, xref="paper", yref="paper",
                     showarrow=False, font=dict(color='white', size=14)),

                dict(
                    x=0.5, 
                    y=-0.15, # --- CHANGE 2: Shifted up from -0.25 to -0.15 ---
                    xref="paper", yref="paper",
                    text=(
                        f'Observed Min Value: <b>{current_min}</b><br>'
                        f'P(X > {current_min} | λ={lam_est}): <b>{prob_text}</b>'
                    ),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="red",
                    borderwidth=1,
                    font=dict(color="red", size=14)
                ),
                dict(
                    text="Min Obs", x=current_min, y=0.8, xref="x2", yref="paper",
                    showarrow=True, arrowhead=2, ax=40, ay=0,
                    font=dict(color="red")
                )
            ]
        ),
        name=f'frame{i}'
    ))

fig.frames = frames


# --- Layout Styling ---
fig.update_layout(
    height=600,
    title={'text': "Impact of Poisson Non-Stationary Visualization", 'y': 0.95, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white', family="Courier New, monospace"),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    
    # Animation Controls
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                "label": "▶ Run",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": "|| Stop",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": "white"},
            "prefix": "Sample Count: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [f'frame{k}'],
                    {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 100}}
                ],
                "label": str(k),
                "method": "animate"
            } for k in range(1, n_draws + 1)
        ]
    }]
)

# Axis Styling
max_y_pmf = max(pmf_est.max(), pmf_act.max()) * 1.1
fig.update_xaxes(title_text='Count', range=[x_min, x_max], showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Probability', range=[0, max_y_pmf], showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

fig.update_xaxes(title_text='Observed Value', range=[x_min, x_max], showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
# Increased y-range slightly for visibility
fig.update_yaxes(title_text='Frequency', range=[0, 20], showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)

fig.show()


# ---


# #### 2.) ⏳ Exponential Distribution
# 
# The Exponential distribution is typically used to model waiting times
# 
#  $$f(x; \lambda) = \lambda e^{-\lambda x} \quad F(x; \lambda) = 1 - e^{-\lambda x}$$
# 
# In the context of our Poisson process, these are the waiting time *in between* draws from our Poisson distribution


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import expon

# --- Exponential Distribution Parameters ---
lam = 1.3  # rate parameter (λ)
x_max = 6
x = np.linspace(0, x_max, 300)
pdf = expon.pdf(x, scale=1/lam)
cdf = expon.cdf(x, scale=1/lam)

# Initial slider values
x1_initial = 1.2
x2_initial = 2.8

# Find nearest indices in x for the initial values
idx_x1 = (np.abs(x - x1_initial)).argmin()
idx_x2 = (np.abs(x - x2_initial)).argmin()
x1_val = x[idx_x1]
x2_val = x[idx_x2]

y_max_pdf = np.max(pdf)*1.08
y_max_cdf = 1.08

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f'Exponential PDF (λ={lam})',
                    f'Exponential CDF (λ={lam})']
)

# PDF and CDF Plots
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines',
                         line=dict(color='#ffff4d', width=3),
                         name='PDF', hovertemplate='x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines',
                         line=dict(color='#00ffff', width=3),
                         name='CDF', hovertemplate='x=%{x:.2f}<br>P(X≤x)=%{y:.4f}<extra></extra>'),
              row=1, col=2)

# Shaded region for CDF up to x2
x_fill = x[:idx_x2+1]
cdf_fill = cdf[:idx_x2+1]
fig.add_trace(go.Scatter(
    x=np.concatenate([x_fill, [x_fill[-1], 0]]),
    y=np.concatenate([cdf_fill, [0, 0]]),
    fill='tozeroy', mode='none',
    fillcolor='rgba(0,255,255,0.3)', name='P(X≤x2)', showlegend=False
), row=1, col=2)

# --- Initial Title & Lines ---
initial_title = (
    f'Exponential Distribution (λ={lam})'
    f'<br>PDF Height at x={x1_val:.2f} is {pdf[idx_x1]:.4f}'
    f' | P(X≤{x2_val:.2f}) = {cdf[idx_x2]:.4f}'
)

INITIAL_SHAPES = [
    dict(type='line', xref='x1', yref='y1',
         x0=x1_val, x1=x1_val, y0=0, y1=y_max_pdf,
         line=dict(color='red', width=3, dash='dot')),
    dict(type='line', xref='x2', yref='y2',
         x0=x2_val, x1=x2_val, y0=0, y1=y_max_cdf,
         line=dict(color='yellow', width=3, dash='dot'))
]

# --- Sliders ---

# Discretize slider points for x1 and x2
slider_x_points = np.round(np.linspace(0, x_max, 41), 2)

slider_x1_steps = []
for x_slider in slider_x_points:
    idx = (np.abs(x - x_slider)).argmin()
    pdf_x = pdf[idx]
    step = dict(
        method="relayout",
        label=f"{x_slider:.2f}",
        args=[{
            "title.text": (
                f'Exponential Distribution (λ={lam})'
                f'<br>PDF Height at x={x_slider:.2f} is {pdf_x:.4f}'
                f' | P(X≤{x2_val:.2f}) = {cdf[idx_x2]:.4f}'
            ),
            "shapes[0].x0": x_slider,
            "shapes[0].x1": x_slider
        }],
        execute=True
    )
    slider_x1_steps.append(step)

slider_x2_steps = []
for x_slider in slider_x_points:
    idx = (np.abs(x - x_slider)).argmin()
    cdf_x = cdf[idx]
    x_fill = x[:idx+1]
    cdf_fill = cdf[:idx+1]
    step = dict(
        method="update",
        label=f"{x_slider:.2f}",
        args=[
            {
                "x": [x, x, np.concatenate([x_fill, [x_fill[-1], 0]])],
                "y": [pdf, cdf, np.concatenate([cdf_fill, [0, 0]])]
            },
            {
                "title.text": (
                    f'Exponential Distribution (λ={lam})'
                    f'<br>PDF Height at x={x1_val:.2f} is {pdf[idx_x1]:.4f}'
                    f' | P(X≤{x_slider:.2f}) = {cdf_x:.4f}'
                ),
                "shapes[1].x0": x_slider,
                "shapes[1].x1": x_slider
            }
        ],
        execute=True
    )
    slider_x2_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=480,
    title={'text': initial_title, 'y': 0.97, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=INITIAL_SHAPES,
    sliders=[
        dict(
            steps=slider_x1_steps,
            active= (np.abs(slider_x_points - x1_val)).argmin(),
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.10, len=0.33,
            currentvalue=dict(prefix="x₁ = ")
        ),
        dict(
            steps=slider_x2_steps,
            active= (np.abs(slider_x_points - x2_val)).argmin(),
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.52, len=0.33,
            currentvalue=dict(prefix="x₂ = ")
        )
    ]
)

# Axes
fig.update_xaxes(title_text='x', range=[0, x_max],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=1)
fig.update_yaxes(title_text='f(x;λ)', range=[0, y_max_pdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=1)
fig.update_xaxes(title_text='x', range=[0, x_max],
                 showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=2)
fig.update_yaxes(title_text='F(x;λ)', range=[0, y_max_cdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=2)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Memoryless Property
# 
# The exponential distribution is memoryless:
#  
#  Let $X \sim \text{Exp}(\lambda)$. Then for $s, t \ge 0$,
#  $$
#  P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
#  $$
#  Thus, the memoryless (Markov) property holds, unlike the Poisson distribution.
# 
# Effectively, given we've already waited $s$, the distribution doesn't change, we are just drawing again from the Exponential parameterized by $\lambda$


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import expon

# --- Parameters ---
lam2 = lam   # use same lambda for clarity
x_max2 = 6
x2 = np.linspace(0, x_max2, 300)
pdf_uncond = expon.pdf(x2, scale=1/lam2)
pdf_cond = expon.pdf(x2, scale=1/lam2)

# Let s be "time already waited"
s = 2.0  # Change for different demos

# Domain for conditional is t ≥ 0 so the second plot is for t
t = np.linspace(0, x_max2, 300)
# Conditional PDF: P(X > s + t | X > s) = exp(-λ t), which is the Exponential(λ) for t≥0
pdf_cond = lam2 * np.exp(-lam2 * t)

# --- Figure Setup ---
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f"Unconditional: Exponential PDF (lambda={lam2})",
        f"Conditional: PDF after waiting s={s:.1f} (Memoryless)"
    ]
)

# Unconditional on the left
fig2.add_trace(go.Scatter(
    x=x2, y=pdf_uncond, mode='lines',
    line=dict(color='#ffff4d', width=3),
    name='Unconditional PDF',
    hovertemplate='x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'
), row=1, col=1)

# Conditional on the right
fig2.add_trace(go.Scatter(
    x=t, y=pdf_cond, mode='lines',
    line=dict(color='#00ffff', width=3),
    name='Conditional PDF',
    hovertemplate='t=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'
), row=1, col=2)

# Add vertical line at s in the left plot for "already waited"
fig2.add_shape(
    dict(type="line", x0=s, x1=s, y0=0, y1=np.max(pdf_uncond)*1.08,
         line=dict(color="red", width=3, dash="dot")),
    row=1, col=1
)

# Title & layout
fig2.update_layout(
    height=380,
    title={'text': 'Exponential Memoryless: Conditional is Same as Unconditional', 'y': 0.95, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

fig2.update_xaxes(
    title_text='x', range=[0, x_max2],
    showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=1
)
fig2.update_yaxes(
    title_text='PDF', range=[0, np.max(pdf_uncond)*1.1],
    showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=1
)
fig2.update_xaxes(
    title_text='t (additional wait)', range=[0, x_max2],
    showgrid=True, gridcolor='rgba(128,128,128,0.22)', row=1, col=2
)
fig2.update_yaxes(
    title_text=f'PDF (conditional, s={s})', range=[0, np.max(pdf_uncond)*1.1],
    showgrid=True, gridcolor='rgba(128,128,128,0.18)', row=1, col=2
)

fig2.show()



# ---


# #### 3.) 📈 Poisson Process
# 
# ##### Definition, Simulation, Visualization
# 
# A **Poisson process** with rate $\lambda > 0$ is a stochastic process $\{N(t): t \geq 0\}$ such that:
# - $N(0) = 0$
# - The process has **independent increments**
# - The number of events in any interval of length $t$, $N(t)$, is Poisson distributed:
#   $$
#   \mathbb{P}[N(t) = k] = \frac{(\lambda t)^k}{k!} e^{-\lambda t}, \quad k = 0,1,2,...
#   $$
# - The process has **stationary increments**: the probability of $k$ arrivals in $[s, s+t)$ is the same as in $[0, t)$.
# 
# Effectively, the distribution of any time increment is given by
# 
# $$\displaystyle N(t+s) - N(s) \sim \mathrm{Poisson}(\lambda t)$$
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1
lam = 0.2

np.random.seed(42)

# Poisson Process Setup
increments = np.random.poisson(lam * dt, (n_paths_total, n_steps))
N = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

# Global Y range
max_y_val = int(np.max(N)) + 5

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    current_vals = N[:, t]
    
    # Histogram
    max_val_t = int(np.max(current_vals))
    bins = np.arange(-0.5, max_val_t + 1.5, 1)
    hist_fig = np.histogram(current_vals, bins=bins, density=True)
    hist_y = 0.5 * (hist_fig[1][1:] + hist_fig[1][:-1])
    hist_x = hist_fig[0]

    # Theoretical PMF
    if t == 0:
        theory_x = np.zeros_like(hist_y)
        if len(theory_x) > 0: theory_x[0] = 1.0 
    else:
        theory_x = poisson.pmf(hist_y, mu=lam * t)

    # Traces
    scatter_paths = [
        go.Scatter(
            x=time_show, y=N[i, :t + 1],
            mode='lines', line=dict(color='#00ffff', width=1, shape='hv'), 
            opacity=0.3, showlegend=False, hoverinfo='none'
        ) for i in range(n_paths_show)
    ]

    scatter_main = go.Scatter(
        x=time_show, y=N[0, :t + 1],
        mode='lines', line=dict(color='magenta', width=3, shape='hv'),
        name="Example Path"
    )

    hist = go.Bar(
        y=hist_y, x=hist_x, orientation='h',
        marker=dict(color='rgba(100,255,100,0.6)'), hoverinfo='skip',
        name='Simulated'
    )

    theory_line = go.Scatter(
        x=theory_x, y=hist_y, mode='lines+markers',
        line=dict(color='magenta', width=2), marker=dict(size=4),
        hoverinfo='skip', name='Theory'
    )

    vline = go.Scatter(
        x=[t, t], y=[-1, max_y_val], mode='lines',
        line=dict(color='yellow', dash='dash'), showlegend=False
    )

    frames.append(go.Frame(
        data=[*scatter_paths, scatter_main, vline, hist, theory_line],
        name=f"step{t}"
    ))

# --- Initial Setup ---
init_t = 0
time_show = time_grid[:init_t + 1]
scatter_paths_init = [
    go.Scatter(x=time_show, y=N[i, :init_t+1], mode='lines', line=dict(color='#00ffff', width=1, shape='hv'), opacity=0.3, showlegend=False)
    for i in range(n_paths_show)
]
scatter_main_init = go.Scatter(x=time_show, y=N[0, :init_t+1], mode='lines', line=dict(color='magenta', width=3, shape='hv'))
vline_init = go.Scatter(x=[init_t, init_t], y=[-1, max_y_val], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)
hist_init = go.Bar(y=[0], x=[1], orientation='h', marker=dict(color='rgba(100,255,100,0.6)'))
theory_line_init = go.Scatter(x=[1], y=[0], mode='lines+markers', line=dict(color='magenta', width=2))

# --- Figure Structure ---
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Poisson Paths", "Distribution of N_t"],
    horizontal_spacing=0.10
)

for s in scatter_paths_init: fig.add_trace(s, row=1, col=1)
fig.add_trace(scatter_main_init, row=1, col=1)
fig.add_trace(vline_init, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line_init, row=1, col=2)

fig.frames = frames

# --- Slider Configuration ---
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 0},  # Removed excessive padding
    x=0.25,        # Start slider at 25% of width (to right of buttons)
    len=0.75,      # Make slider occupy remaining 75%
    y=-0.1,        # Align vertically with buttons
    steps=[dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=650, 
    width=1200,
    title_text=f"Poisson Process (λ={lam})",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=sliders,
    margin=dict(b=100), # Space for controls
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0,       # Buttons anchored to far left
        'y': -0.15,      # Aligned vertically with slider
        'xanchor': 'left', 
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='N_t', range=[-1, max_y_val], row=1, col=1)
fig.update_xaxes(title_text='Prob', range=[0, 0.4], row=1, col=2)
fig.update_yaxes(title_text='N_t', range=[-1, max_y_val], row=1, col=2)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Waiting Times and the Exponential Distribution
# 
#  The waiting time between Poisson events is exponentially distributed: 
#  $$ P(T > t) = e^{-\lambda t}, \quad f_T(t) = \lambda e^{-\lambda t},\ t \geq 0 $$
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import expon

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1
lam = 0.2

np.random.seed(42)

# Poisson Process Simulation (Grid Based)
increments = np.random.poisson(lam * dt, (n_paths_total, n_steps))
N = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

# --- Pre-calculate Waiting Times ---
# Extract time intervals between events from the grid simulation
all_wait_times = []
all_completion_times = []

for i in range(n_paths_total):
    event_indices = np.where(increments[i, :] > 0)[0]
    
    if len(event_indices) > 0:
        # Time to first event (index + 1)
        w_times = [event_indices[0] + 1] 
        # Times between subsequent events
        w_times.extend(np.diff(event_indices))
        
        # Absolute time when these waits completed
        arrival_times = event_indices + 1
        
        all_wait_times.extend(w_times)
        all_completion_times.extend(arrival_times)

all_wait_times = np.array(all_wait_times)
all_completion_times = np.array(all_completion_times)

# Parameters for Right Plot (Waiting Time Distribution)
max_wait_visual = int(10 / lam) 
bins_wait = np.arange(0, max_wait_visual + 2, 1) 
theory_x = np.linspace(0, max_wait_visual, 100)
theory_y = expon.pdf(theory_x, scale=1/lam) 

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    # Data for Left Plot (Paths)
    time_show = time_grid[:t + 1]
    
    # Data for Right Plot (Waiting Times)
    valid_mask = all_completion_times <= t
    current_waits = all_wait_times[valid_mask]
    
    if len(current_waits) > 0:
        hist_counts, _ = np.histogram(current_waits, bins=bins_wait, density=True)
    else:
        hist_counts = np.zeros(len(bins_wait)-1)
    
    hist_x_bar = 0.5 * (bins_wait[1:] + bins_wait[:-1]) 

    # --- Traces ---
    scatter_paths = [
        go.Scatter(
            x=time_show, y=N[i, :t + 1],
            mode='lines', line=dict(color='#00ffff', width=1, shape='hv'), 
            opacity=0.3, showlegend=False, hoverinfo='none'
        ) for i in range(n_paths_show)
    ]

    scatter_main = go.Scatter(
        x=time_show, y=N[0, :t + 1],
        mode='lines', line=dict(color='magenta', width=3, shape='hv'),
        name="Example Path"
    )

    hist_waits = go.Bar(
        x=hist_x_bar, y=hist_counts,
        marker=dict(color='rgba(100,255,100,0.6)'), hoverinfo='skip',
        name='Simulated'
    )

    theory_line = go.Scatter(
        x=theory_x, y=theory_y, mode='lines',
        line=dict(color='magenta', width=2, dash='solid'),
        hoverinfo='skip', name='Theory'
    )

    frames.append(go.Frame(
        data=[*scatter_paths, scatter_main, hist_waits, theory_line],
        name=f"step{t}"
    ))

# --- Initial Setup ---
init_t = 0
scatter_paths_init = [go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='#00ffff', width=1, shape='hv')) for _ in range(n_paths_show)]
scatter_main_init = go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='magenta', width=3, shape='hv'))
hist_init = go.Bar(x=[0], y=[0], marker=dict(color='rgba(100,255,100,0.6)'))
theory_line_init = go.Scatter(x=theory_x, y=theory_y, mode='lines', line=dict(color='magenta', width=2))

# --- Figure Structure ---
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.6, 0.4],
    subplot_titles=["Simulated Poisson Paths N_t", "Waiting Time Distribution (Inter-arrival)"],
    horizontal_spacing=0.10
)

for s in scatter_paths_init: fig.add_trace(s, row=1, col=1)
fig.add_trace(scatter_main_init, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line_init, row=1, col=2)

fig.frames = frames

# --- Slider Configuration ---
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 0},
    x=0.25, len=0.75, y=-0.1,
    steps=[dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=600, width=1200,
    title_text=f"Poisson Process (λ={lam}) & Empirical Waiting Time Convergence",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,  # Legend removed
    sliders=sliders,
    margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0, 
        'y': -0.15, 
        'xanchor': 'left', 
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes styling
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Count N_t', range=[-1, int(np.max(N))+5], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Waiting Time (τ)', range=[0, max_wait_visual], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Density', range=[0, lam * 1.5], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Time Homogeneous vs. Time Inhomogeneous Poisson Processes
# 
#  $$
#  \textbf{Time-Homogeneous:}\qquad \lambda(t) = \lambda 
#  $$
#  $$
#  \mathbb{P}(N_{t+h} - N_t = k) = \frac{(\lambda h)^k}{k!}e^{-\lambda h}
#  $$
#  
#  $$
#  \textbf{Time-Inhomogeneous:}\qquad \lambda = \lambda(t)\ \text{(depends on time)}
#  $$
#  $$
#  \mathbb{P}(N_{t+h} - N_t = k) = \frac{\left(\int_t^{t+h}\lambda(u)\,du\right)^k}{k!}e^{-\int_t^{t+h}\lambda(u)\,du}
#  $$
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Parameters for time-inhomogeneous Poisson process ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1

# Define a time-varying intensity function, e.g., lambda(t) = 0.1 + 0.25*sin(2*pi*t/50)
def lam_t(t):
    return 0.1 + 0.25 * np.sin(2 * np.pi * t / 50)

time_grid = np.arange(n_steps + 1)
lams = lam_t(time_grid)

np.random.seed(42)

# Compute increments for each step using corresponding lambda, eliminating negative/NaN lambdas
increments_inhom = np.zeros((n_paths_total, n_steps), dtype=int)
for k in range(n_steps):
    lam_k = lam_t(k)
    lam_k = max(lam_k, 0)  # Ensure lambda passed to poisson is always >= 0
    if not np.isfinite(lam_k):
        lam_k = 0.0
    increments_inhom[:, k] = np.random.poisson(lam_k * dt, n_paths_total)

# Inhomogeneous Poisson path: sum increments over time, start from zero
N_inhom = np.concatenate(
    [np.zeros((n_paths_total, 1)), np.cumsum(increments_inhom, axis=1)], axis=1
)

# Compute the theoretical mean of increments up to each t for the inhomogeneous process
cum_lam = np.cumsum(lams[:-1]) * dt  # Exclude t=n_steps+1 when integrating up to t

max_y_val_inhom = int(np.max(N_inhom)) + 5

# --- Prepare Animation Frames (for inhomogeneous case) ---
frames_inhom = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    current_vals = N_inhom[:, t]
    
    # Histogram
    max_val_t = int(np.max(current_vals))
    bins = np.arange(-0.5, max_val_t + 1.5, 1)
    hist_fig = np.histogram(current_vals, bins=bins, density=True)
    hist_y = 0.5 * (hist_fig[1][1:] + hist_fig[1][:-1])
    hist_x = hist_fig[0]
    
    # Theoretical PMF: Poisson with parameter mu = integrated lambda up to t
    if t == 0:
        theory_x = np.zeros_like(hist_y)
        if len(theory_x) > 0: 
            theory_x[0] = 1.0
    else:
        mu_t = np.sum(lams[:t]) * dt  # integrate lambda from 0 to t
        theory_x = poisson.pmf(hist_y, mu=mu_t)

    # Traces
    scatter_paths = [
        go.Scatter(
            x=time_show, y=N_inhom[i, :t + 1],
            mode='lines', line=dict(color='#77aae7', width=1, shape='hv'), 
            opacity=0.3, showlegend=False, hoverinfo='none'
        ) for i in range(n_paths_show)
    ]

    scatter_main = go.Scatter(
        x=time_show, y=N_inhom[0, :t + 1],
        mode='lines', line=dict(color='orange', width=3, shape='hv'),
        name="Example Path"
    )

    hist = go.Bar(
        y=hist_y, x=hist_x, orientation='h',
        marker=dict(color='rgba(230,150,100,0.7)'), hoverinfo='skip',
        name='Simulated'
    )

    theory_line = go.Scatter(
        x=theory_x, y=hist_y, mode='lines+markers',
        line=dict(color='orange', width=2), marker=dict(size=4),
        hoverinfo='skip', name='Theory'
    )

    vline = go.Scatter(
        x=[t, t], y=[-1, max_y_val_inhom], mode='lines',
        line=dict(color='yellow', dash='dash'), showlegend=False
    )

    frames_inhom.append(go.Frame(
        data=[*scatter_paths, scatter_main, vline, hist, theory_line],
        name=f"inhom_step{t}"
    ))

# Initial traces for t=0
init_t = 0
time_show = time_grid[:init_t + 1]
scatter_paths_init = [
    go.Scatter(
        x=time_show, y=N_inhom[i, :init_t+1], 
        mode='lines', line=dict(color='#77aae7', width=1, shape='hv'), 
        opacity=0.3, showlegend=False
    ) for i in range(n_paths_show)
]
scatter_main_init = go.Scatter(
    x=time_show, y=N_inhom[0, :init_t+1], 
    mode='lines', line=dict(color='orange', width=3, shape='hv')
)
vline_init = go.Scatter(
    x=[init_t, init_t], y=[-1, max_y_val_inhom], 
    mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False
)
hist_init = go.Bar(y=[0], x=[1], orientation='h', marker=dict(color='rgba(230,150,100,0.7)'))
theory_line_init = go.Scatter(x=[1], y=[0], mode='lines+markers', line=dict(color='orange', width=2))

# --- Figure Structure ---
fig_inhom = make_subplots(
    rows=1, cols=2, column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Inhomogeneous Poisson Paths", "Distribution of N_t (Changing λ(t))"],
    horizontal_spacing=0.10
)

for s in scatter_paths_init: fig_inhom.add_trace(s, row=1, col=1)
fig_inhom.add_trace(scatter_main_init, row=1, col=1)
fig_inhom.add_trace(vline_init, row=1, col=1)
fig_inhom.add_trace(hist_init, row=1, col=2)
fig_inhom.add_trace(theory_line_init, row=1, col=2)

fig_inhom.frames = frames_inhom

# --- Slider Configuration ---
sliders_inhom = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 0},
    x=0.25,
    len=0.75,
    y=-0.1,
    steps=[dict(
        method="animate",
        args=[[f"inhom_step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig_inhom.update_layout(
    height=650,
    width=1200,
    title_text="Time-Inhomogeneous Poisson Process (λ(t) changing over time)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=sliders_inhom,
    margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0,
        'y': -0.15,
        'xanchor': 'left',
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

for c in [1, 2]:
    fig_inhom.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig_inhom.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

fig_inhom.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig_inhom.update_yaxes(title_text='N_t', range=[-1, max_y_val_inhom], row=1, col=1)
fig_inhom.update_xaxes(title_text='Prob', range=[0, 0.4], row=1, col=2)
fig_inhom.update_yaxes(title_text='N_t', range=[-1, max_y_val_inhom], row=1, col=2)

fig_inhom.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Example in Practice: Hawkes Process
# 
#  $$\textbf{Hawkes~Process:}\quad dN_t = \text{Poisson}\left(\lambda_0 + \int_0^t \phi(t-s)\, dN_s\right)$$
#  $$\lambda_t = \lambda_0 + \sum_{t_i < t} \phi(t-t_i)$$
# 
#  - Hawkes processes are time-dependent Poisson processes where each event increases the future intensity, modeling "self-excitation."
#  - The intensity function λₜ = λ₀ + ∑ φ(t - tᵢ) means past arrivals directly impact the likelihood of new events in near future.
#  - Used in finance to capture clustered arrivals of trades, orders, or price jumps (e.g., market impact, contagion).
#  - Unlike homogeneous Poisson, they can reproduce bursts (volatility clustering) often observed in high-frequency market data.
#  - The excitation kernel φ(t) determines how strongly (and for how long) each event influences future arrivals.
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_steps = 200     # Time steps
dt = 0.1          # Time delta
T = n_steps * dt
mu = 0.5          # Baseline intensity (background trading)
alpha = 2.0       # Excitation (panic factor)
beta = 3.0        # Decay (market memory)

np.random.seed(42)

# --- Hawkes Process Simulation ---
# We simulate step-by-step to capture the self-exciting feedback loop
times = np.arange(0, T, dt)
N = np.zeros(len(times))        # Cumulative count
lambda_t = np.zeros(len(times)) # Intensity process
lambda_t[0] = mu

events = [] # Store exact event times
current_N = 0
current_lam = mu

for i in range(1, len(times)):
    # 1. Exponential Decay from previous step
    # lambda(t) decays towards mu
    decay = np.exp(-beta * dt)
    current_lam = mu + (current_lam - mu) * decay
    
    # 2. Bernoulli trial for event occurrence
    prob = current_lam * dt
    if np.random.rand() < prob:
        current_N += 1
        current_lam += alpha # The "Self-Exciting" Jump
        events.append(times[i])
        
    N[i] = current_N
    lambda_t[i] = current_lam

# Global Y ranges for consistent axis scaling
max_N_val = int(np.max(N)) + 2
max_lam_val = np.max(lambda_t) * 1.1

# --- Animation Frames ---
frames = []
# Skip some frames for smoother/faster animation rendering
step_skip = 2 
display_indices = range(0, len(times), step_skip)

for k in display_indices:
    time_show = times[:k+1]
    N_show = N[:k+1]
    lam_show = lambda_t[:k+1]
    
    # 1. Top Plot: Cumulative Trades (The "Step" Function)
    trace_N = go.Scatter(
        x=time_show, y=N_show,
        mode='lines',
        line=dict(color='magenta', width=3, shape='hv'),
        name="Cumulative Trades"
    )

    # 2. Bottom Plot: Intensity (The "Seismograph")
    trace_lam = go.Scatter(
        x=time_show, y=lam_show,
        mode='lines',
        line=dict(color='#00ffff', width=2), # Cyan matching your reference
        fill='tozeroy',                      # Fill to look like a signal
        fillcolor='rgba(0, 255, 255, 0.2)',
        name="Market Intensity"
    )

    # 3. Vertical Line (Current Time Tracker)
    # We add this to both subplots to visually track "Now"
    current_t = times[k]
    
    frames.append(go.Frame(
        data=[trace_N, trace_lam],
        name=f"step{k}"
    ))

# --- Initial Setup ---
init_k = 0
trace_N_init = go.Scatter(x=times[:1], y=N[:1], mode='lines', line=dict(color='magenta', width=3, shape='hv'))
trace_lam_init = go.Scatter(x=times[:1], y=lambda_t[:1], mode='lines', line=dict(color='#00ffff', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.2)')

# --- Figure Structure ---
# Using 2 Rows, 1 Col to align Time Axis
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=["Cumulative Trades (N_t)", "Intensity / Volatility (λ_t)"],
    row_heights=[0.5, 0.5]
)

fig.add_trace(trace_N_init, row=1, col=1)
fig.add_trace(trace_lam_init, row=2, col=1)

fig.frames = frames

# --- Slider Configuration ---
# Matching your reference styling exactly
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 0},
    x=0.25, 
    len=0.75, 
    y=-0.1,
    steps=[dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in display_indices]
)]

# --- Layout ---
fig.update_layout(
    height=650, 
    width=1200,
    title_text="Hawkes Process: Self-Exciting Market Activity",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=sliders,
    margin=dict(b=100), 
    updatemenus=[{
        'type': 'buttons',
        'x': 0.0, 
        'y': -0.15, 
        'xanchor': 'left', 
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes Styling
# Top Plot (Trades)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
fig.update_yaxes(title_text='Count', range=[-1, max_N_val], showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=1)

# Bottom Plot (Intensity)
fig.update_xaxes(title_text='Time (t)', range=[0, T], showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=2, col=1)
fig.update_yaxes(title_text='Intensity λ(t)', range=[0, max_lam_val], showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=2, col=1)

fig.show()


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Poisson distributions can be used to model rare events over a particular time interval and have implications even in game development for modeling and inducing arrivals
# - Of course distributions change over time in the real world and some modelled events violate these assumptions more violently than others, it seems to be a bit of a catch-22 since we are modeling events over an interval while that distribution is subject to change - this is what we deal with in reality
# - The exponential distribution is typically used to model waiting times and in our Poisson process governs the waiting times in between arrivals or events
# - Poisson processes themselves are accumulation or counting processes for these events over a simulated (or analytical) time interval
# - They have independent and stationary increments which makes solving for probabilities analytically tractable and very easy, we also observe nice convergence gaurentees as with our standard random variables (LLN, CLT)
# - Real world applications range from the pricing of financial derivatives to risk-modeling, if there is interest in future videos on the subject I would love to extend this idea (Merton, Hawkes, . . .) since the base process (Poisson process) has now been covered  
# 
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Hawkes Processes
# - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
# - Market-Making Models and Simulation (Stoikov-Avellaneda)
# - Projects that Made me a Quant
# - My First Year as a Quant
# - Kalman Filter for Quant Finance
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Backtest a Trading Strategy with Interactive Brokers


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

