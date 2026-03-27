# ### 🦅 Hawkes Processes for Quant Finance
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
# - [Poisson Processes for Quant Finance](https://youtu.be/oug0vzbwISQ)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
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
# #### 1.) 🐟 Poisson Processes
# 
# - Poisson Random Variables
# 
# - Poisson Processes
# 
# - Inhomogeneous Poisson Processes
# 
# #### 2.) 🦅 Hawkes Processes
# 
# - Definition, Self-Exciting Behavior, Simulation
# 
# - Hawkes Process as a Stochastic Intensity
# 
# #### 3.) 🌊 Modeling Volatility Clusters
# 
# - Standard Jump Diffusion Models
# 
# - Self-Exciting Jump-Diffusion Models
# 
# - Efficacy of Risk Modeling
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🐟 Poisson Processes
# 
# ##### Poisson Random Variables
# 
# Poisson random variables are used to model *rare events* over a particular time interval.
# 
# They are parameterized by $\lambda$: average event frequency over the time interval.
# 
# Interestingly, for this distribution $\mathbb{E}[X] = \lambda = Var(X)$
# 
# $$
# \text{PMF:}\quad P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
# \quad\quad
# \text{CDF:}\quad P(X \leq k) = \sum_{j=0}^k \frac{\lambda^j e^{-\lambda}}{j!}
# \quad\quad
# \text{CF:}\quad \varphi_X(t) = \mathbb{E}[e^{itX}] = \exp\left(\lambda(e^{it}-1)\right)
# $$


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Simulation Parameters ---
n_draws = 1000        # Total number of random draws
frames_step = 10      # Number of draws to add per animation frame
lam = 20              # Fixed lambda

np.random.seed(42)

# Generate all random draws upfront
draws = np.random.poisson(lam, n_draws)

# X-axis range calculations
max_val = int(np.max(draws))
x_grid = np.arange(0, max_val + 5)
theory_pmf = poisson.pmf(x_grid, mu=lam)
max_y_prob = np.max(theory_pmf) * 1.25  # Buffer for y-axis

# Base colors for the charts
base_color = 'rgba(255, 0, 255, 0.3)'        # Dimmed magenta
match_color = 'rgba(100, 255, 100, 0.6)'     # Exact matched green for both sides

# --- Initial Setup ---
# Left: Theoretical PMF (Will be animated to show highlight)
theory_bar = go.Bar(
    x=x_grid, y=theory_pmf,
    marker=dict(color=[base_color] * len(x_grid)),
    name="Theoretical PMF",
    hoverinfo='skip'
)

# Right: Empirical Histogram (Will be animated)
hist_init = go.Bar(
    x=x_grid, y=np.zeros_like(x_grid),
    marker=dict(color=match_color),  
    name='Simulated',
    hoverinfo='skip'
)

# Right: Theoretical Overlay (Redrawn every frame so it doesn't vanish)
theory_line = go.Scatter(
    x=x_grid, y=theory_pmf, mode='lines+markers',
    line=dict(color='magenta', width=2), marker=dict(size=4),
    name='Theory Overlay', hoverinfo='skip'
)

# --- Figure Structure ---
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.5, 0.5],
    subplot_titles=[f"Theoretical Poisson PMF (λ={lam})", "Empirical Convergence"],
    horizontal_spacing=0.10
)

# Trace indices: 0=theory_bar, 1=hist_init, 2=theory_line
fig.add_trace(theory_bar, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line, row=1, col=2)

# --- Animation Frames ---
frames = []
steps_list = list(range(1, n_draws + 1, frames_step))
if steps_list[-1] != n_draws:
    steps_list.append(n_draws)

bins = np.arange(-0.5, max_val + 4.5, 1)
hist_x = 0.5 * (bins[1:] + bins[:-1])

for t in steps_list:
    current_draws = draws[:t]
    latest_draw = current_draws[-1] # The draw to highlight
    
    # 1. Update Left Chart (Light up the current draw with the exact matched green)
    current_colors = [base_color] * len(x_grid)
    if latest_draw in x_grid:
        idx = np.where(x_grid == latest_draw)[0][0]
        current_colors[idx] = match_color
        
    theory_bar_frame = go.Bar(marker=dict(color=current_colors))
    
    # 2. Update Right Chart (Histogram)
    hist_counts, _ = np.histogram(current_draws, bins=bins, density=True)
    hist_frame = go.Bar(x=hist_x, y=hist_counts)
    
    # 3. Keep Theoretical Line intact
    theory_line_frame = go.Scatter(x=x_grid, y=theory_pmf)
    
    # Pass all 3 traces to ensure nothing disappears
    frames.append(go.Frame(
        data=[theory_bar_frame, hist_frame, theory_line_frame], 
        traces=[0, 1, 2],
        name=f"step{t}"
    ))

fig.frames = frames

# --- Slider Configuration ---
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Draws (N): "},
    pad={"t": 0}, 
    x=0.25,        
    len=0.75,      
    y=-0.1,        
    steps=[dict(
        method="animate",
        args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(k)
    ) for k in steps_list]
)]

# --- Layout ---
fig.update_layout(
    height=650, 
    width=1000,
    title_text=f"Drawing from a Poisson Distribution (λ={lam})",
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
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes Formatting
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c, title_text='Value (k)', range=[-0.5, max_val + 3.5])
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c, title_text='Probability', range=[0, max_y_prob])

fig.show()


# What if instead of just realizing the number of events over the time interval (drawing from Poisson),
# 
# we instead watched each of the events arrive over time until the end of the time interval?


# ###### ______________________________________________________________________________________________________________________________________


# ##### Poisson Processes
# 
# Poisson processes are simply a stochastic counting process, draws at each step come from a Poisson distribution.
#  
#  $$N(t) \sim \text{Poisson}(\lambda t) \quad\quad \lambda = \text{event rate per unit time}$$
#   
#  
#  *The Poisson process has independent increments:*
#  
#  $$
#  N(t_2) - N(t_1) \sim \text{Poisson}(\lambda (t_2 - t_1)), \quad \text{for} \quad t_2 > t_1
#  $$
#  
#  *where increments over non-overlapping intervals are independent random variables.*


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
    subplot_titles=["Simulated Poisson Paths", "Distribution of N(t)"],
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
    width=1000,
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


# See something familiar?  Let's do the math!  
# 
# Here we set $\lambda = .2$
# 
# It follows by independent increments...
# 
#  $$
#  N(100) - N(0) \sim \text{Poisson}(20 (100 - 0)) = \text{Poisson}(20)
#  $$
# 
#  Thus, we have the same distribution at $t = 100$ as we did in our static Poisson random variable example above.


# ###### ______________________________________________________________________________________________________________________________________
# 
# Notice lambda is constant, this implies a constant event rate over time.
# 
# Clearly this isn't the case, think about trades at market open relative to a slow afternoon.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Inhomogeneous Poisson Processes
#  
#  Inhomogeneous (nonhomogeneous) Poisson processes generalize Poisson processes to allow for a time-varying event rate.
#  
#  Now, instead of a constant $\lambda$, the intensity function $\lambda(t)$ can change over time:
#  
#  $$
#  N(t) \sim \text{Poisson}\left(\int_0^t \lambda(s) ds \right)
#  $$
#  
#  - $\lambda(t)$: instantaneous event rate at time $t$.
#  
#  *The inhomogeneous Poisson process also has independent increments:*
#  
#  $$
#  N(t_2) - N(t_1) \sim \text{Poisson}\left(\int_{t_1}^{t_2} \lambda(s) ds \right), \quad \text{for} \quad t_2 > t_1
#  $$
#  
#  *where increments over non-overlapping intervals are independent random variables, but the average number of events can now vary in time.*


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import poisson

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1

# Piecewise rates for Time-Inhomogeneous Process
lam1 = 0.2  # Rate for t in (0, 50]
lam2 = 0.8  # Rate for t in (50, 100]

np.random.seed(42)

# --- Piecewise Rate Setup ---
lam_t = np.concatenate([np.full(50, lam1), np.full(50, lam2)])
mu_array = np.concatenate([[0.0], np.cumsum(lam_t * dt)])

# Poisson Process Setup
rates_matrix = np.tile(lam_t * dt, (n_paths_total, 1))
increments = np.random.poisson(rates_matrix)
N = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

# Global Y range
max_y_val = int(np.max(N)) + 5

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    current_vals = N[:, t]
    
    # Switch right-side distribution colors at t=50
    if t <= 50:
        hist_color = 'rgba(100,255,100,0.6)'  # Green
        theory_color = 'magenta'              # Purple/Magenta
    else:
        hist_color = 'rgba(255,50,50,0.6)'    # Red
        theory_color = 'orange'               # Orange
    
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
        theory_x = poisson.pmf(hist_y, mu=mu_array[t])

    # Traces (Split Paths)
    scatter_paths_cyan = []
    scatter_paths_pink = []
    
    for i in range(n_paths_show):
        # Cyan segment (t=0 to min(t, 50))
        x_cyan = time_grid[:min(t + 1, 51)]
        y_cyan = N[i, :min(t + 1, 51)]
        scatter_paths_cyan.append(
            go.Scatter(x=x_cyan, y=y_cyan, mode='lines', line=dict(color='#00ffff', width=1, shape='hv'), opacity=0.3, showlegend=False, hoverinfo='none')
        )
        
        # Pink segment (t=50 to t)
        if t >= 50:
            x_pink = time_grid[50:t + 1]
            y_pink = N[i, 50:t + 1]
        else:
            x_pink = [None]
            y_pink = [None]
            
        scatter_paths_pink.append(
            go.Scatter(x=x_pink, y=y_pink, mode='lines', line=dict(color='pink', width=1, shape='hv'), opacity=0.3, showlegend=False, hoverinfo='none')
        )

    hist = go.Bar(
        y=hist_y, x=hist_x, orientation='h',
        marker=dict(color=hist_color), hoverinfo='skip',
        name='Simulated'
    )

    theory_line = go.Scatter(
        x=theory_x, y=hist_y, mode='lines+markers',
        line=dict(color=theory_color, width=2), marker=dict(size=4),
        hoverinfo='skip', name='Theory'
    )

    vline = go.Scatter(
        x=[t, t], y=[-1, max_y_val], mode='lines',
        line=dict(color='yellow', dash='dash'), showlegend=False
    )

    # Note: Order must perfectly match the initial setup below
    frames.append(go.Frame(
        data=[*scatter_paths_cyan, *scatter_paths_pink, vline, hist, theory_line],
        name=f"step{t}"
    ))

# --- Initial Setup ---
init_t = 0
scatter_cyan_init = [
    go.Scatter(x=time_grid[:1], y=N[i, :1], mode='lines', line=dict(color='#00ffff', width=1, shape='hv'), opacity=0.3, showlegend=False)
    for i in range(n_paths_show)
]
scatter_pink_init = [
    go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='pink', width=1, shape='hv'), opacity=0.3, showlegend=False)
    for i in range(n_paths_show)
]
vline_init = go.Scatter(x=[init_t, init_t], y=[-1, max_y_val], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)
hist_init = go.Bar(y=[0], x=[1], orientation='h', marker=dict(color='rgba(100,255,100,0.6)'))
theory_line_init = go.Scatter(x=[1], y=[0], mode='lines+markers', line=dict(color='magenta', width=2))

# --- Figure Structure ---
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Poisson Paths", "Distribution of N(t)"],
    horizontal_spacing=0.10
)

# Add traces in the exact same order as the frames
for s in scatter_cyan_init: fig.add_trace(s, row=1, col=1)
for s in scatter_pink_init: fig.add_trace(s, row=1, col=1)
fig.add_trace(vline_init, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line_init, row=1, col=2)

fig.frames = frames

# --- Static Marker for Change in Rate ---
fig.add_vline(x=50, line_width=2, line_dash="dash", line_color="white", row=1, col=1, opacity=0.7)

# --- Slider Configuration ---
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
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=650, 
    width=1000,
    title_text=f"Time-Inhomogeneous Poisson Process (λ₁={lam1}, λ₂={lam2})",
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


# Here, we model lambda(t) as a piecewise (time-dependent) function:
# 
#  $$
#  \lambda(t) = 
#  \begin{cases}
#  0.2 & 0 < t \leq 50 \\
#  0.8 & 50 < t \leq 100 \\
#  \end{cases}
#  $$
# 
# At time $t$ throughout the life of the process we are still dealing with a Poisson distribution.


import numpy as np
import plotly.graph_objects as go

# --- Parameters ---
n_steps = 100
lam1 = 0.2  # Rate for t in (0, 50]
lam2 = 0.8  # Rate for t in (50, 100]

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    
    # Logic to split the line precisely at t=50
    if t <= 50:
        x_cyan = [0, t]
        y_cyan = [lam1, lam1]
        x_pink = [None]
        y_pink = [None]
    else:
        x_cyan = [0, 50]
        y_cyan = [lam1, lam1]
        # The pink line starts at t=50, jumps from lam1 to lam2, then goes to current t
        x_pink = [50, 50, t]
        y_pink = [lam1, lam2, lam2]

    scatter_cyan = go.Scatter(
        x=x_cyan, y=y_cyan, mode='lines', 
        line=dict(color='#00ffff', width=4), 
        showlegend=False, hoverinfo='none'
    )
    
    scatter_pink = go.Scatter(
        x=x_pink, y=y_pink, mode='lines', 
        line=dict(color='pink', width=4), 
        showlegend=False, hoverinfo='none'
    )

    # Moving yellow time indicator
    vline = go.Scatter(
        x=[t, t], y=[0, 1], mode='lines',
        line=dict(color='yellow', dash='dash'), showlegend=False, hoverinfo='none'
    )

    # Note: Data list order must perfectly match the initial setup below
    frames.append(go.Frame(
        data=[scatter_cyan, scatter_pink, vline],
        name=f"step{t}"
    ))

# --- Initial Setup ---
init_t = 0
scatter_cyan_init = go.Scatter(x=[0, 0], y=[lam1, lam1], mode='lines', line=dict(color='#00ffff', width=4), showlegend=False)
scatter_pink_init = go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='pink', width=4), showlegend=False)
vline_init = go.Scatter(x=[init_t, init_t], y=[0, 1], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)

# --- Figure Structure ---
fig = go.Figure(
    data=[scatter_cyan_init, scatter_pink_init, vline_init],
    frames=frames
)

# --- Static Marker for Change in Rate ---
fig.add_vline(x=50, line_width=2, line_dash="dash", line_color="white", opacity=0.7)

# --- Slider Configuration ---
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
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=500, 
    width=1000,
    title_text="Rate Parameter λ(t) over Time",
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
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Rate λ(t)', range=[0, 1.0], showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# However, the rate function is an integral accumulating $\lambda(s)$ up to time $t$.
# 
# $$N(t) - N(0) \sim \text{Poisson}(\mu(t)) \quad \quad \mu(t) = \int_0^t \lambda(s)ds$$
# 
# Effectively, we are dealing with a Poisson distribution at anytime $t$,
# 
# where the rate is accumulated (integrated) in our intensity function $\lambda(s)$ over time.


# ###### ______________________________________________________________________________________________________________________________________
# 
# We are getting closer, but time varying intensity clearly doesn't capture the full story.
# 
# Think of a market crash: big gap down, panic, more selling; trades aren't just a function of time but also recent events.


# ---


# #### 2.) 🦅 Hawkes Processes
#   
#   Hawkes processes generalize Poisson processes further to include *self-exciting* (event-dependent) intensities.
#   
#   The rate of future events can increase after each event capturing clustering and contagion effects.
#   
#   Now, the intensity function $\lambda(t)$ depends both on time and the *history* of previous events:
#   
#   $$
#   \lambda(t) = \mu + \sum_{t_i < t} \phi(t - t_i)
#   $$
#   
#   - $\mu$: baseline (exogenous) event rate.
#   - $\phi(t - t_i)$: triggering kernel measuring the influence of past event at $t_i$ on the current intensity.
#   
#   The Hawkes process is **non-Markovian** (the future depends on the *entire* past, not just the present).


import numpy as np
import plotly.graph_objects as go

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 3     # Showing exactly 3 paths
n_steps = 100
dt = 1

# Hawkes Process Parameters
mu_base = 0.2  # Baseline intensity
alpha = 0.4    # Jump size (how much intensity increases per event)
beta = 0.8     # Decay rate (how fast intensity returns to baseline)

np.random.seed(42)

# --- Hawkes Process Simulation ---
lambdas_history = np.zeros((n_paths_total, n_steps + 1))
lambdas = np.full(n_paths_total, mu_base)
lambdas_history[:, 0] = lambdas

for t in range(1, n_steps + 1):
    # 1. Generate events for this time step
    increments = np.random.poisson(lambdas * dt)
    
    # 2. Update intensities for the next step 
    lambdas = mu_base + (lambdas - mu_base) * np.exp(-beta * dt) + alpha * increments
    lambdas_history[:, t] = lambdas

time_grid = np.arange(n_steps + 1)

# Global Y range for the lambda plot
max_y_val = np.max(lambdas_history[:n_paths_show, :]) * 1.1

# --- Styling Parameters ---
# Neon Pink, Neon Green, Neon Cyan
neon_colors = ['#ff00ff', '#39ff14', '#00ffff'] 
path_width = 1.2  # Slightly thinner than before

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    
    scatter_paths = [
        go.Scatter(
            x=time_show, y=lambdas_history[i, :t + 1], 
            mode='lines', line=dict(color=neon_colors[i], width=path_width), 
            opacity=0.9, showlegend=False, hoverinfo='none'
        ) for i in range(n_paths_show)
    ]

    vline = go.Scatter(
        x=[t, t], y=[0, max_y_val], mode='lines',
        line=dict(color='yellow', dash='dash'), showlegend=False
    )

    frames.append(go.Frame(
        data=[*scatter_paths, vline],
        name=f"step{t}"
    ))

# --- Initial Setup ---
init_t = 0
time_show = time_grid[:init_t + 1]
scatter_paths_init = [
    go.Scatter(x=time_show, y=lambdas_history[i, :init_t+1], mode='lines', line=dict(color=neon_colors[i], width=path_width), opacity=0.9, showlegend=False)
    for i in range(n_paths_show)
]
vline_init = go.Scatter(x=[init_t, init_t], y=[0, max_y_val], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)

# --- Figure Structure ---
fig = go.Figure(data=[*scatter_paths_init, vline_init])

fig.frames = frames

# --- Static Marker for Baseline mu ---
fig.add_hline(
    y=mu_base, line_width=1.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.6)", 
    annotation_text="μ (Base Rate)", annotation_position="top left", annotation_font_color="white"
)

# --- Slider Configuration ---
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
    ) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=500, 
    width=1000,
    title_text=f"Hawkes Process Rate Parameter λ(t) (μ={mu_base}, α={alpha}, β={beta})",
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
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Rate λ(t)', range=[0, max_y_val], showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Hawkes Process as a Stochastic Intensity
# 
# The Hawkes process generalizes the Poisson process by letting the intensity (rate) be self-exciting and history-dependent:
# 
# $$
# \lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}
# $$
# 
# - $\mu$: Baseline intensity (constant background rate)
# - $\alpha$: Excitation added per event
# - $\beta$: Decay rate of excitation
# 
# The counting process $N(t)$ then follows:
# 
# $$
# N(t) \sim \text{Hawkes}\left(\lambda(s) \text{ as above}\right)
# $$
# 
# *Like the inhomogeneous Poisson process, increments are independent given the path of $\lambda(t)$, but here $\lambda(t)$ depends on the process history.*


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulation Parameters ---
n_paths = 3
n_steps = 100
dt = 1

# Hawkes Process Parameters
mu_base = 0.2  # Baseline intensity
alpha = 0.4    # Jump size (self-excitation)
beta = 0.8     # Decay rate

np.random.seed(42)

# --- Hawkes Process Simulation ---
N = np.zeros((n_paths, n_steps + 1))
lambdas = np.zeros((n_paths, n_steps + 1))
lambdas[:, 0] = mu_base

for t in range(1, n_steps + 1):
    increments = np.random.poisson(lambdas[:, t-1] * dt)
    N[:, t] = N[:, t - 1] + increments
    lambdas[:, t] = mu_base + (lambdas[:, t-1] - mu_base) * np.exp(-beta * dt) + alpha * increments

time_grid = np.arange(n_steps + 1)

# Global Y ranges
max_y_N = int(np.max(N)) + 5
max_y_L = np.max(lambdas) * 1.1

# --- Styling Parameters ---
neon_colors = ['#ff00ff', '#39ff14', '#00ffff']  # Pink, Green, Cyan
path_width = 1.8
opacity_val = .9

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    
    # Top Row: N(t) Traces
    traces_N = [
        go.Scatter(
            x=time_show, y=N[i, :t+1], 
            mode='lines', line=dict(color=neon_colors[i], width=path_width, shape='hv'),
            opacity=opacity_val,
            name=f"Path {i+1}", legendgroup=f"group{i}"
        ) for i in range(n_paths)
    ]
    vline_N = go.Scatter(x=[t, t], y=[-1, max_y_N], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)
    
    # Bottom Row: Lambda(t) Traces
    traces_L = [
        go.Scatter(
            x=time_show, y=lambdas[i, :t+1], 
            mode='lines', line=dict(color=neon_colors[i], width=path_width), 
            opacity=opacity_val,
            showlegend=False, legendgroup=f"group{i}"
        ) for i in range(n_paths)
    ]
    vline_L = go.Scatter(x=[t, t], y=[0, max_y_L], mode='lines', line=dict(color='yellow', dash='dash'), showlegend=False)

    # Frame creation
    frames.append(go.Frame(
        data=[*traces_N, vline_N, *traces_L, vline_L],
        name=f"step{t}"
    ))

# --- Figure Structure ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=[
        "Cumulative Events N(t)", 
        "Underlying Self-Exciting Rate λ(t)"
    ]
)

# Populate initial traces
init_data = frames[0].data

# Top Row Traces (3 paths + 1 vline)
for i in range(n_paths): fig.add_trace(init_data[i], row=1, col=1)
fig.add_trace(init_data[n_paths], row=1, col=1)

# Bottom Row Traces (3 paths + 1 vline)
offset = n_paths + 1
for i in range(n_paths): fig.add_trace(init_data[offset + i], row=2, col=1)
fig.add_trace(init_data[offset + n_paths], row=2, col=1)

fig.frames = frames

# --- Static Marker for Baseline mu (Applied to Bottom Row) ---
fig.add_hline(
    y=mu_base, row=2, col=1,
    line_width=1.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.6)", 
    annotation_text="μ (Base Rate)", annotation_position="top left", annotation_font_color="white"
)

# --- Slider Configuration ---
sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step: "}, pad={"t": 0}, x=0.25, len=0.75, y=-0.05,
    steps=[dict(method="animate", args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(k)) for k in range(n_steps + 1)]
)]

# --- Layout ---
fig.update_layout(
    height=800, width=1000,
    title_text=f"Hawkes Process Dynamics (μ={mu_base}, α={alpha}, β={beta})",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    sliders=sliders, margin=dict(b=80, t=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]
    }]
)

# Axes formatting
for r in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=r, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=r, col=1)

fig.update_yaxes(title_text='Events N(t)', range=[-1, max_y_N], row=1, col=1)
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=2, col=1)
fig.update_yaxes(title_text='Rate λ(t)', range=[0, max_y_L], row=2, col=1)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# **How can the Hawkes process improve our decision making under uncertainty?**
# 
# *Many Applications*
# - Pricing
# - Risk Modeling
# - Volatility Forecasting
# - . . .
# 
# Let's observe the impact on our risk models and the ability to generate more accurate probabilities and statistics for forecasts.


# ---


# #### 3.) 🌊 Modeling Volatility Clusters
#  
#  **Jump Diffusions: Two SDEs**
# 
# *Standard (Merton) Jump Diffusion:*
#    $$ dS_t = \mu S_t dt + \sigma S_t dW_t + S_{t-} (J - 1) dN_t $$
#    where $N_t$ is a Poisson process with intensity $\lambda$.
# 
#  *Hawkes Jump Diffusion:*
#    $$ dS_t = \mu S_t dt + \sigma S_t dW_t + S_{t-} (J - 1) dN_t $$
#    but now $\lambda_t$ is *itself stochastic* ("self-exciting"; e.g. $\lambda_t = \mu + \sum_{t_i < t} \alpha e^{-\beta (t-t_i)}$, depends on jump history).
# 
#  - Both: $J$ = jump size, $dW_t$ = Brownian, $dN_t$ = jumps, $\mu$ = drift, $\sigma$ = volatility
#  - Standard: jumps independent/constant-rate. Hawkes: bursts/clustering of jumps.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis

# --- Realistic Financial Simulation Parameters ---
n_steps = 200
dt = 1

# Continuous Diffusion (Market Noise: ~1% daily vol)
mu_drift = 0.0
sigma_diff = 0.01

# Jump Shock Parameters (Market Shocks: ~4% vol)
mu_J = 0.0
sigma_J = 0.04

# Comparable Rates (Both average exactly 0.15 jumps per day)
lam_fixed = 0.15  # Standard JD Constant Rate
mu_base = 0.06    # Hawkes Baseline Rate
alpha = 0.3       # Hawkes Excitation (Jump size in rate)
beta = 0.5        # Hawkes Decay
# Theoretical Hawkes Mean = mu_base / (1 - alpha/beta) = 0.06 / (1 - 0.6) = 0.15

np.random.seed(42)

# --- Process Simulation ---
time_grid = np.arange(1, n_steps + 1)

R_std = np.zeros(n_steps)
R_hwk = np.zeros(n_steps)
lam_hwk = np.zeros(n_steps + 1)
lam_hwk[0] = mu_base

# Shared underlying diffusion (ensures differences are strictly due to jump distribution)
dW_shared = np.random.normal(mu_drift * dt, sigma_diff * np.sqrt(dt), n_steps)

for t in range(n_steps):
    # Standard JD
    dN_std = np.random.poisson(lam_fixed * dt)
    jump_std = np.sum(np.random.normal(mu_J, sigma_J, dN_std)) if dN_std > 0 else 0
    R_std[t] = dW_shared[t] + jump_std
    
    # Hawkes JD
    dN_hwk = np.random.poisson(lam_hwk[t] * dt)
    jump_hwk = np.sum(np.random.normal(mu_J, sigma_J, dN_hwk)) if dN_hwk > 0 else 0
    R_hwk[t] = dW_shared[t] + jump_hwk
    
    # Update Stochastic Rate
    lam_hwk[t+1] = mu_base + (lam_hwk[t] - mu_base)*np.exp(-beta*dt) + alpha*dN_hwk

# --- Strict Axis Scaling ---
# Symmetrical Returns X-Axis
max_abs_R = max(np.max(np.abs(R_std)), np.max(np.abs(R_hwk))) * 1.15
bins = np.linspace(-max_abs_R, max_abs_R, 60) # 60 fixed bins for smooth histograms
hist_x = 0.5 * (bins[1:] + bins[:-1])

# --- Styling Parameters ---
c_cyan = '#00ffff'
c_pink = '#ff00ff'
path_width = 1.2
opacity_val = 0.9

def get_kurtosis_text(arr):
    # Fisher's Kurtosis (Normal Dist = 0.0)
    if len(arr) > 10 and np.var(arr) > 1e-8:
        return f"Excess Kurtosis: {kurtosis(arr):.2f}"
    return "Excess Kurtosis: ---"

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t]
    
    if t == 0:
        hist_y_std = np.zeros_like(hist_x)
        hist_y_hwk = np.zeros_like(hist_x)
        k_std, k_hwk = "Excess Kurtosis: ---", "Excess Kurtosis: ---"
    else:
        hist_y_std, _ = np.histogram(R_std[:t], bins=bins, density=True)
        hist_y_hwk, _ = np.histogram(R_hwk[:t], bins=bins, density=True)
        k_std = get_kurtosis_text(R_std[:t])
        k_hwk = get_kurtosis_text(R_hwk[:t])

    # Traces Standard
    tr_std_ts = go.Scatter(x=time_show, y=R_std[:t], mode='lines', line=dict(color=c_cyan, width=path_width), opacity=opacity_val, hoverinfo='none')
    vl_std = go.Scatter(x=[t, t], y=[-max_abs_R, max_abs_R], mode='lines', line=dict(color='yellow', dash='dash'))
    tr_std_hist = go.Bar(x=hist_x, y=hist_y_std, marker=dict(color='rgba(0, 255, 255, 0.5)', line=dict(color=c_cyan, width=0.5)))
    
    # Upper right anchor for Standard Density Plot (Y-max is 50)
    txt_std = go.Scatter(
        x=[max_abs_R * 0.95], y=[45], mode='text', text=[k_std], 
        textfont=dict(color=c_cyan, size=14), textposition='middle left', hoverinfo='skip'
    )

    # Traces Hawkes
    tr_hwk_ts = go.Scatter(x=time_show, y=R_hwk[:t], mode='lines', line=dict(color=c_pink, width=path_width), opacity=opacity_val, hoverinfo='none')
    vl_hwk = go.Scatter(x=[t, t], y=[-max_abs_R, max_abs_R], mode='lines', line=dict(color='yellow', dash='dash'))
    tr_hwk_hist = go.Bar(x=hist_x, y=hist_y_hwk, marker=dict(color='rgba(255, 0, 255, 0.5)', line=dict(color=c_pink, width=0.5)))
    
    # Upper right anchor for Hawkes Density Plot (Y-max is 45)
    txt_hwk = go.Scatter(
        x=[max_abs_R * 0.95], y=[40.5], mode='text', text=[k_hwk], 
        textfont=dict(color=c_pink, size=14), textposition='middle left', hoverinfo='skip'
    )

    frames.append(go.Frame(
        data=[tr_std_ts, vl_std, tr_std_hist, txt_std, tr_hwk_ts, vl_hwk, tr_hwk_hist, txt_hwk],
        name=f"step{t}"
    ))

# --- Figure Structure ---
fig = make_subplots(
    rows=2, cols=2, column_widths=[0.7, 0.3], vertical_spacing=0.12,
    subplot_titles=["Standard Jump-Diffusion (Random Shocks)", "Distribution (Standard)", 
                    "Hawkes Jump-Diffusion (Clustered Shocks)", "Distribution (Hawkes)"]
)

# Initial population
init_data = frames[0].data
for i, (r, c) in enumerate([(1,1), (1,1), (1,2), (1,2), (2,1), (2,1), (2,2), (2,2)]):
    fig.add_trace(init_data[i], row=r, col=c)

fig.frames = frames

# --- Static Stylings ---
for r in [1, 2]:
    fig.add_hline(y=0, row=r, col=1, line_width=1, line_dash="solid", line_color="rgba(255, 255, 255, 0.2)")

# --- Controls ---
sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step: "}, pad={"t": 0}, x=0.1, len=0.8, y=-0.05,
    steps=[dict(method="animate", args=[[f"step{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(k)) for k in range(n_steps + 1)]
)]

fig.update_layout(
    height=750, width=1000,
    title_text=f"Volatility Clustering & Excess Kurtosis (Shared Average Jump Rate λ={lam_fixed})",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(b=80, t=80, l=40, r=40), bargap=0.05,
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [
           {'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}]}            
           ]
    }]
)

# --- Axis Formatting ---
for r in [1, 2]:
    for c in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', row=r, col=c)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', row=r, col=c)

# Left Column (Time Series)
for r in [1, 2]:
    fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=r, col=1)
    fig.update_yaxes(title_text='Return', range=[-max_abs_R, max_abs_R], row=r, col=1)

# Right Column (Upright Histograms) - HARDCODED Y-LIMITS
fig.update_xaxes(title_text='Return', range=[-max_abs_R, max_abs_R], row=1, col=2)
fig.update_yaxes(title_text='Density', range=[0, 50], row=1, col=2)

fig.update_xaxes(title_text='Return', range=[-max_abs_R, max_abs_R], row=2, col=2)
fig.update_yaxes(title_text='Density', range=[0, 45], row=2, col=2)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### What is the Impact on our Probabilities and Statistics?
# 
# If we fail to capture facets of risk or dynamics we observe in the real world we will misrepresent the likelihoods of different states of the world.
# 
# Models better capable of capturing real-world dynamics lead to arbitrarily better probabilities and statistics.
# 
# When it comes to optimal decision making under uncertainty, models are a component of the process, better models, better decisions.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, norm

# --- Simulation Parameters ---
n_paths = 1000  
n_steps = 200
dt = 1

# Continuous Diffusion (~1% daily vol)
sigma_diff = 0.01

# Jump Shock Parameters (~4% shock)
sigma_J = 0.04

# Rates (Shared mean of 0.15)
lam_fixed = 0.15
mu_base = 0.06
alpha = 0.3
beta = 0.5

np.random.seed(42)

# --- Process Simulation ---
all_R_std = []
all_R_hwk = []

# Standard JD
for p in range(n_paths):
    dW = np.random.normal(0, sigma_diff, n_steps)
    dN = np.random.poisson(lam_fixed * dt, n_steps)
    jumps = np.array([np.sum(np.random.normal(0, sigma_J, n)) if n > 0 else 0 for n in dN])
    all_R_std.extend(dW + jumps)

# Hawkes JD
for p in range(n_paths):
    dW = np.random.normal(0, sigma_diff, n_steps)
    lam = mu_base
    for t in range(n_steps):
        dN = np.random.poisson(lam * dt)
        jump = np.sum(np.random.normal(0, sigma_J, dN)) if dN > 0 else 0
        all_R_hwk.append(dW[t] + jump)
        lam = mu_base + (lam - mu_base) * np.exp(-beta * dt) + alpha * dN

all_R_std = np.array(all_R_std)
all_R_hwk = np.array(all_R_hwk)

# --- Analysis ---
def get_daily_stats(data):
    mu, std = np.mean(data), np.std(data)
    ek = kurtosis(data)
    thresh = 3 * std
    p_emp = np.mean(np.abs(data - mu) > thresh)
    p_norm = 2 * (1 - norm.cdf(3))
    return mu, std, ek, thresh, p_emp, p_norm

# --- Visualization ---
fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.1,
    subplot_titles=["Daily Returns: Standard Jump-Diffusion", "Daily Returns: Hawkes Jump-Diffusion"]
)

max_r = 0.22 
bins = np.linspace(-max_r, max_r, 100)
models = [("Standard", all_R_std, '#00ffff'), ("Hawkes", all_R_hwk, '#ff00ff')]

for i, (name, data, color) in enumerate(models):
    mu, std, ek, thresh, p_emp, p_norm = get_daily_stats(data)
    
    # Histogram
    counts, edges = np.histogram(data, bins=bins, density=True)
    x_hist = 0.5 * (edges[1:] + edges[:-1])
    fig.add_trace(go.Bar(x=x_hist, y=counts, marker_color=color, opacity=0.5, name='Empirical'), row=1, col=i+1)
    
    # Normal Fit
    x_norm = np.linspace(-max_r, max_r, 400)
    y_norm = norm.pdf(x_norm, mu, std)
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, line=dict(color='white', width=2, dash='dot'), name='Normal Fit'), row=1, col=i+1)

    # 3-Sigma Visual Aids
    fig.add_vline(x=mu+thresh, line_dash="dash", line_color="yellow", row=1, col=i+1)
    fig.add_vline(x=mu-thresh, line_dash="dash", line_color="yellow", row=1, col=i+1)

    stats_text = (
        f"<b>Daily Excess Kurtosis: {ek:.2f}</b><br>"
        f"----------------------------<br>"
        f"<b>Normal Model (3σ):</b><br>"
        f"Prob: {p_norm*100:.3f}%<br>"
        f"Wait: {1/p_norm:.0f} days<br>"
        f"----------------------------<br>"
        f"<b>Empirical Model (3σ):</b><br>"
        f"Prob: {p_emp*100:.3f}%<br>"
        f"Wait: {1/p_emp:.1f} days"
    )
    
    # Use explicit subplot axes for annotation to avoid the xref error
    fig.add_annotation(
        x=max_r * 0.95, y=55, 
        xref=f"x{i+1}", yref=f"y{i+1}",
        text=stats_text, showarrow=False, align='right',
        bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=1,
        font=dict(color="white", size=11)
    )

fig.update_layout(
    height=600, width=1200, template="plotly_dark",
    title_text="Standard vs. Hawkes: Daily Return Non-Normality",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False
)

fig.update_xaxes(title_text="Daily Return", range=[-max_r, max_r])
fig.update_yaxes(title_text="Density", range=[0, 65])

fig.show()


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Poisson random variables are used to model the number of *rare events* over a particular time interval
# - Poisson processes let us *walk the path* of realizing these events over that time interval, at the end of the process (assuming comparable parameterization), we'll have the same Poisson distribution
# - Poisson processes are parameterized by a fixed average event frequency over a time interval, clearly in practice frequencies vary, think about quantity of trades at market open relative to a slow afternoon
# - Inhomogeneous Poisson processes make the rate parameter a function of time allowing us to change the frequency, or intensity of arrivals, over time however we choose to define it
# - In reality, the intensity isn't just a function of time but also a *function of itself*, think volatility clusters
# - Hawkes processes are a stochastic processes that are *self-exciting* meaning past values impact future values and more *exciting* or extreme values can imply *even more* exciting or extreme values
# - We can parameterize the rate or intensity parameter in our Poisson processes using this Hawkes process, this makes arrivals *self-exciting*
# - It's then possible to include this self-exciting Poisson process in a standard diffusion to get a Hawkes Jump-Diffusion Model which can better capture the empirical dynamics of volatility clustering and excess kurtosis
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Fama-French / Carhart and Factor Modeling in General
# - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
# - Market-Making Models and Simulation (Stoikov-Avellaneda)
# - My First Year as a Quant
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Girsanov's Change of Measure
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Kalman Filter with Interactive Brokers
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

