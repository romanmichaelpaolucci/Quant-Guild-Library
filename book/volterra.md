### 📈 Quant Derives Volterra Process Discretization and Simulation

##### ▶️ Related Quant Guild Videos:

- [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)

- [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)

- [Central Limit Theorem for Quant Finance](https://youtu.be/q2era-4pnic)

- [Why Quant Models Break](https://youtu.be/brdG1TmsPlw)

- [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)

- [The 5 Papers That Build Modern Quant Finance](https://youtu.be/ZwS1gMGegrM)

###### ______________________________________________________________________________________________________________________________________

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)



##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

##### [👾 Join the Quant Guild Discord Server](discord.com/invite/MJ4FU2c6c3)

---


```python
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
```


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



### 📖 Sections

#### 1.) 📜 Volterra Process

- Definition

- Covariance

- Causality

#### 2.) ⚛️ Brownian Volterra Kernel

#### 3.) 🌊 Mean-Reverting Volterra Kernel

#### 4.) 🧮 Fractional Volterra Kernel

#### 5.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📜 Volterra Process

##### Generalization

The Volterra process is defined by the integral

$$X_t = \int_0^s K(s, t) dW_s$$

Where $K(s, t)$ is the Volterra kernel dictating the (auto)covariance structure of the process

$$Cov(X_s, X_t) = \gamma(s, t) = \mathbb{E}[(X_s - \mathbb{E}[X_s])((X_t - \mathbb{E}[X_t])]$$

We know $\mathbb{E}[X_s] = \mathbb{E}[X_t] = 0$ as $W_t$ is standard Brownian motion

$$\implies \gamma(s, t) = \mathbb{E}[X_s X_t] = \mathbb{E}[\int_0^u K(u, s) dW_u \int_0^v K(v, t) dW_v]$$

Ito Isometry

  \begin{align*}
  \mathbb{E}\left[ \int_0^T f(u)\,dW_u \int_0^T g(v)\,dW_v \right]
  &= \mathbb{E}\left[ \int_0^T \int_0^T f(u)g(v)\,dW_u\,dW_v \right] \\
  &\text{Note: } \mathbb{E}[dW_u\,dW_v] = 
     \begin{cases}
       du, & u = v \\
       0, & u \ne v
     \end{cases} \\
  &\implies \int_0^T \int_0^T f(u)g(v)\,\mathbb{E}[dW_u\,dW_v] = \int_0^T f(u)g(u)\,du
  \end{align*}

Implication in our covariance function

$$\implies \gamma(s, t) = \mathbb{E}[X_s X_t] = \mathbb{E}[\int_0^u K(u, s) dW_u \int_0^v K(v, t) dW_v] = \int_0^T K(u, s) K(u, t) du$$

In terms of Causality

- $K(u, s)$ is only non-zero when $u \leq s$

- $K(u, t)$ is only non-zero when $u \leq t$

- Therefore, the product $K(u, s) K(u, t)$ is only non-zero where $u \leq min(s, t)$

$$\gamma(s, t) = \mathbb{E}[X_s X_t] = \int_0^T K(u, s) K(u, t) du\implies \int_0^{min(s, t)} K(u, s) K(u, t) du$$

##### Discretization

Using EM discretization

Consider a partition of $n$ equally spaced points on the time domain $t \in [0, T]$

$$\Delta t = \frac{T}{n}, \quad\quad \Delta t_i = \frac{iT}{n}, \quad\quad 0 = \Delta t_0, \Delta t_1 = \frac{T}{n}, \dots, \Delta t_n = \frac{nT}{n} = \Delta t_T = T$$

It follows

$$X_{\Delta t_i} \approx X_{\Delta t_{i - 1}} + \sum_{j = 0}^{\Delta t_i} K(\Delta t_j, \Delta t_i) \Delta W_{\Delta t} \quad \text{for } i = \{0, 1, \dots, T\}$$

---

#### 2.) ⚛️ Brownian Volterra Kernel

Let the Volterra kernel be $K(s, t) = 1$

$$\implies X_{\Delta t_i} \approx X_{\Delta t_{i - 1}} + \sum_{j = 0}^{\Delta t_i} K(\Delta t_j, \Delta t_i) \Delta W_{\Delta t} = X_{\Delta t_{i - 1}} + \sum_{j = 0}^{\Delta t_i} \Delta W_{\Delta t}$$

Wait, given this Volterra kernel what is the increment?

$$\implies X_{\Delta t_i} - X_{\Delta t_{i-1}} =  X_{\Delta t_{i - 1}} + \sum_{j = 0}^{\Delta t_i} \Delta W_{\Delta t} - X_{\Delta t_{i - 1}} = \sum_{j = 0}^{\Delta t_i} \Delta W_{\Delta t}$$

But this is just the sum of independent increments, hence we are left with standard Brownian motion

Let's simulate Brownian motion as a Volterra process and as a standard draw from a Guassian with variance of the total time step


```python
import numpy as np

def K(s, t):
    # Volterra kernel K(s, t) = 1 regardless of s, t
    return 1

def volterra_process(n, T):
    """
    Simulate Brownian motion using the Volterra representation
    with kernel K(s, t) = 1.
    """
    dt = T / n
    t_grid = np.linspace(0, T, n)
    dW = np.random.normal(0, np.sqrt(dt), n)

    X = np.zeros(n)
    for i in range(n):
        current_t = t_grid[i]
        # By definition, using the Volterra representation:
        # X[t_i] = sum_{j=0}^i K(t_j, t_i) * dW_j
        X[i] = np.sum([K(t_grid[j], current_t) * dW[j] for j in range(i + 1)])
    return X
```


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
n_steps = 252
T = 1
n_paths = 1000

# Simulate single sample paths
X_volterra_path = volterra_process(n_steps, T)
X_gaussian_path = np.cumsum(np.random.normal(0, np.sqrt(T / n_steps), n_steps))

# Simulate many terminal values
volterra_terminals = np.array([volterra_process(n_steps, T)[-1] for _ in range(n_paths)])
gaussian_matrix = np.random.normal(0, np.sqrt(T / n_steps), (n_paths, n_steps))
gaussian_terminals = np.cumsum(gaussian_matrix, axis=1)[:, -1]

# Set up plotly figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Single Simulated Paths", "Terminal Distribution (1000 paths)"],
    horizontal_spacing=0.20
)

# --- Left: Show one Brownian Volterra and one standard BM path ---
fig.add_trace(
    go.Scatter(
        y=X_volterra_path,
        mode="lines",
        name="Volterra Path",
        line=dict(color="#39ff14", width=3),
        showlegend=True
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        y=X_gaussian_path,
        mode="lines",
        name="Gaussian Path",
        line=dict(color="#00ffff", width=3, dash="dash"),
        showlegend=True
    ),
    row=1, col=1
)
fig.update_xaxes(
    title_text="Time Step", row=1, col=1,
    showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)
fig.update_yaxes(
    title_text="Value", row=1, col=1,
    showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)

# --- Right: Terminal distributions ---
hist_colors = dict(volterra="rgba(57,255,20,0.44)", gaussian="rgba(0,255,255,0.34)")
binrange = [
    min(volterra_terminals.min(), gaussian_terminals.min()), 
    max(volterra_terminals.max(), gaussian_terminals.max())
]
fig.add_trace(
    go.Histogram(
        x=volterra_terminals,
        nbinsx=30,
        name="Volterra Terminal",
        opacity=0.88,
        marker_color=hist_colors['volterra'],
        showlegend=True
    ),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(
        x=gaussian_terminals,
        nbinsx=30,
        name="Gaussian Terminal",
        opacity=0.74,
        marker_color=hist_colors['gaussian'],
        showlegend=True
    ),
    row=1, col=2
)
fig.update_xaxes(
    title_text="Terminal Value", row=1, col=2,
    showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)
fig.update_yaxes(
    title_text="Frequency", row=1, col=2,
    showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)

# --- Layout/theme ---
fig.update_layout(
    height=480,
    width=950,
    title={'text': "Brownian Path as Volterra Process", 
           'y': 0.98, 'x': 0.5, 'xanchor': 'center'},
    barmode='overlay',
    font=dict(color='white'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

fig.show()
```



We would never use summation or the Volterra representation for Brownian motion as when $K(s, t) = 1$

$$X_{\Delta t_{i}} = X_{\Delta t_{i - 1}} + \sum_{j = 0}^{\Delta t_i} K(\Delta t_j, \Delta t_i) \Delta W_{\Delta t} \sim N(0, \Delta t) \quad \text{for }i=\{0, 1, \dots, T\}$$

Just draw from the distribution $N(0, \Delta t)$ directly, it is *significantly* faster

When there is a long-range dependence (covariance structure demands path memory in the increment) we can't do this

---

#### 3.) 🌊 Mean-Reverting Volterra Kernel

 Let's derive the Volterra representation for the Ornstein-Uhlenbeck (OU) process, which incorporates mean reversion.

 The OU process solves the SDE:
 $$
 dX_t = -\theta X_t\,dt + dW_t
 $$
 Its solution can be written as:
 $$
 X_t = \int_{0}^t e^{-\theta(t-s)}\,dW_s
 $$
 Here, the Volterra kernel is $K(s, t) = e^{-\theta(t-s)}$.

  Discretizing this, we have for grid points $0 = t_0 < t_1 < ... < t_n = T$:
   $$
   X_{t_i} = \sum_{j=0}^i K(t_j, t_i)\, \Delta W_{t_j}
   $$

 So, the OU process can be simulated pathwise as a Volterra process with this exponentially decaying kernel, encoding its mean-reverting behaviour.



```python
import numpy as np

global theta 
theta = 2

def K(s, t):
    return np.exp(-theta * (t-s))

def volterra_process(n, T):
    dt = T / n
    t_grid = np.linspace(0, T, n)
    # Pre-generate shocks (dW)
    dW = np.random.normal(0, np.sqrt(dt), n)
    
    X = np.zeros(n)
    for i in range(n):
        current_t = t_grid[i]
        # Sum all shocks up to AND INCLUDING the current time i
        # This matches your notes: sum_{j=0}^i K(t_j, t_i) dW_j
        s = 0
        for j in range(i + 1):
            s += K(t_grid[j], current_t) * dW[j]
        X[i] = s
    return X
```


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
n_steps = 252
T = 1.0
n_paths = 1000

# Simulate volterra process paths for empirical covariance
volterra_matrix = np.array([volterra_process(n_steps, T) for _ in range(n_paths)])  # shape (n_paths, n_steps)
t_grid = np.linspace(0, T, n_steps)
theta_local = theta  # from global set earlier

# For sample paths plot: plot a smaller number of paths (for visualization)
n_show = 12
sample_paths = volterra_matrix[:n_show]

# Empirical covariance (sample covariance across paths)
empirical_cov = np.cov(volterra_matrix, rowvar=False)

# Theoretical covariance for OU process (Volterra kernel, see cov formula above)
theoretical_cov = np.zeros((n_steps, n_steps))
for i in range(n_steps):
    for j in range(n_steps):
        ti, tj = t_grid[i], t_grid[j]
        tau = min(ti, tj)
        theoretical_cov[i, j] = (
            np.exp(-theta_local * (ti + tj)) * (np.exp(2 * theta_local * tau) - 1) / (2 * theta_local)
        )

# --- Create composite figure: rows=2, cols=2, sample paths spanning both columns on top ---
fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.42, 0.58],  # top row for paths
    shared_xaxes=False,
    vertical_spacing=0.08,
    horizontal_spacing=0.20,
    specs=[
        [{"colspan": 2}, None],  # row 1: paths across both columns
        [{}, {}]                 # row 2: empirical and theoretical heatmaps
    ],
    subplot_titles=(
        [  # subplot_titles will refer to lower plots, so pad with empty string for upper
            "",
            "Empirical Covariance (n=1000 paths)",
            "Theoretical Covariance (as n→infinity)"
        ]
    ),
)

# --- Add sample path lines to top row (row=1, col=1 in the layout, across both cols) ---
# Make paths cyan of varying opacity (from 0.28 to 0.95)
opacities = np.linspace(0.28, 0.95, n_show)
for idx in range(n_show):
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=sample_paths[idx],
            mode="lines",
            line=dict(width=1.15, color=f'rgba(0, 255, 255, {opacities[idx]:.3f})'),  # cyan with variable alpha
            name=f"Path {idx+1}",
            showlegend=False
        ),
        row=1, col=1
    )
# Add a horizontal line: mean-reverting level (Ornstein-Uhlenbeck is mean 0)
fig.add_hline(
    y=0.0,
    line=dict(color="red", width=2.5, dash="dash"),
    row=1, col=1,
    annotation_text="",
    annotation_position="top left",
    annotation_font=dict(color="red", size=13)
)

# --- Empirical Covariance Heatmap (row=2, col=1) ---
fig.add_trace(
    go.Heatmap(
        z=empirical_cov,
        x=np.arange(n_steps),
        y=np.arange(n_steps),
        coloraxis="coloraxis",
        showscale=False,
    ),
    row=2, col=1
)

# --- Theoretical Covariance Heatmap (row=2, col=2) ---
fig.add_trace(
    go.Heatmap(
        z=theoretical_cov,
        x=np.arange(n_steps),
        y=np.arange(n_steps),
        coloraxis="coloraxis",
        zmin=min(empirical_cov.min(), theoretical_cov.min()),
        zmax=max(empirical_cov.max(), theoretical_cov.max()),
    ),
    row=2, col=2
)

fig.update_layout(
    coloraxis=dict(
        colorscale="Viridis",
        colorbar=dict(title="Covariance", len=0.51, y=0.31, x=1.15),
    ),
    # Space for the full combined plot
    height=800,
    width=1000,
    title={
        'text': "Ornstein-Uhlenbeck (Volterra) Process: Sample Paths and Covariance Structure",
        'y': 0.96, 'x': 0.5, 'xanchor': 'center'
    },
    font=dict(color='white'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=60, l=60, b=35, r=35)
)
# Remove latex/math formatting from axis titles for browser display
fig.update_xaxes(
    title_text="time step (t_j)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)
fig.update_xaxes(
    title_text="time step (t_j)", row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)

fig.update_yaxes(
    title_text="time step (t_i)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="time step (t_i)", row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)
# Add label for the sample paths plot
fig.update_yaxes(
    title_text="OU Process value", row=1, col=1
)
fig.update_xaxes(
    title_text="time", row=1, col=1
)
# Show colorful legend only for covariance, not sample paths
fig.show()
```



---

#### 4.) 🧮 Fractional Volterra 

Let's now consider a *fractional* Volterra kernel, which gives us access to processes like fractional Brownian motion (fBm) and more general Gaussian Volterra processes. In this case, the kernel is no longer exponential, but of a power-law type.

A commonly used fractional kernel is:
$$
K(s, t) = (t - s)^{H - \frac{1}{2}}, \quad 0 \leq s < t \leq T
$$
where $H \in (0, 1)$ is the Hurst parameter. This kernel gives rise to the Mandelbrot–van Ness representation of fBm for $H > 0.5$ (and a related but technically different one for $H < 0.5$). 

The Volterra process is then:
$$
X_t = \int_0^t (t-s)^{H-1/2} dW_s
$$

For discretization, grid the interval $[0, T]$ as $0 = t_0 < t_1 < ... < t_n = T$, and approximate:
$$
X_{t_i} \approx \sum_{j=0}^{i-1} (t_i - t_j)^{H - 1/2} \Delta W_{t_j}
$$

This kernel decays more slowly than the exponential, so past history has a more persistent effect. For $H > 0.5$, increments of the process are positively correlated (long memory); for $H < 0.5$, increments are negatively correlated (anti-persistence).

**Fractional Volterra kernels enable the simulation of a rich class of non-Markovian, self-similar processes.**



```python
import numpy as np

# Hurst parameter for fractional kernel
H = 0.7  # can be any value in (0,1)

def K_frac(s, t):
    """Fractional Volterra kernel: (t-s)^{H - 1/2}, defined for s < t."""
    # For s == t, the kernel is zero or undefined; set to 0 for discretization
    diff = t - s
    return (diff ** (H - 0.5)) if diff > 0 else 0.0

def volterra_fractional_process(n, T):
    dt = T / n
    t_grid = np.linspace(0, T, n)
    dW = np.random.normal(0, np.sqrt(dt), n)

    X = np.zeros(n)
    for i in range(n):
        current_t = t_grid[i]
        s = 0
        for j in range(i):
            s += K_frac(t_grid[j], current_t) * dW[j]
        X[i] = s
    return X
```


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use the function from the notebook to simulate the Volterra fractional process
def K_frac(s, t, H):
    diff = t - s
    return (diff ** (H - 0.5)) if diff > 0 else 0.0

def volterra_fractional_process(n, T, H):
    dt = T / n
    t_grid = np.linspace(0, T, n)
    dW = np.random.normal(0, np.sqrt(dt), n)
    X = np.zeros(n)
    for i in range(n):
        current_t = t_grid[i]
        s = 0
        for j in range(i):
            s += K_frac(t_grid[j], current_t, H) * dW[j]
        X[i] = s
    return X

# Simulation parameters
H = 0.1
n_steps = 252
T = 1.0
n_paths = 10000
dt = T / n_steps
t_grid = np.linspace(0, T, n_steps)

# Simulate n_paths of the Volterra (fractional) process, shape (n_paths, n_steps)
volterra_frac_matrix = np.zeros((n_paths, n_steps))
for k in range(n_paths):
    volterra_frac_matrix[k, :] = volterra_fractional_process(n_steps, T, H)

# Empirical covariance from simulated paths
empirical_cov_frac = np.cov(volterra_frac_matrix, rowvar=False)

# --- Build theoretical covariance using vectorized kernel construction (as before) ---
t_grid_kernel = np.linspace(dt, T, n_steps)  # to avoid the singularity at 0 for the kernel definition
t_i, t_j = np.meshgrid(t_grid_kernel, t_grid_kernel, indexing='ij')
diff_matrix = t_i - t_j
K_matrix = np.zeros_like(diff_matrix)
mask = diff_matrix > 0
K_matrix[mask] = diff_matrix[mask]**(H - 0.5)
L = K_matrix * np.sqrt(dt)
theoretical_cov_frac = L @ L.T

# For path plotting, pick a few paths
n_show = 12
sample_frac_paths = volterra_frac_matrix[:n_show, :]

# --- Create composite figure: rows=2, cols=2, sample paths spanning both columns on top ---
fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.42, 0.58],  # top row for paths
    shared_xaxes=False,
    vertical_spacing=0.08,
    horizontal_spacing=0.20,
    specs=[
        [{"colspan": 2}, None],
        [{}, {}]
    ],
    subplot_titles=(
        [
            "",
            "Empirical Covariance (n=10000 paths)",
            f"Theoretical Covariance (H={H:.2f})"
        ]
    )
)

# Plot sample paths
opacities = np.linspace(0.28, 0.95, n_show)
for idx in range(n_show):
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=sample_frac_paths[idx],
            mode="lines",
            line=dict(width=1.15, color=f'rgba(245, 180, 106, {opacities[idx]:.3f})'),  # orange-like
            name=f"Frac Path {idx+1}",
            showlegend=False
        ),
        row=1, col=1
    )

fig.add_trace(
    go.Heatmap(
        z=empirical_cov_frac,
        x=np.arange(n_steps),
        y=np.arange(n_steps),
        coloraxis="coloraxis",
        showscale=False,
    ),
    row=2, col=1
)

fig.add_trace(
    go.Heatmap(
        z=theoretical_cov_frac,
        x=np.arange(n_steps),
        y=np.arange(n_steps),
        coloraxis="coloraxis",
        zmin=min(empirical_cov_frac.min(), theoretical_cov_frac.min()),
        zmax=max(empirical_cov_frac.max(), theoretical_cov_frac.max()),
    ),
    row=2, col=2
)

fig.update_layout(
    coloraxis=dict(
        colorscale="Plasma",
        colorbar=dict(title="Covariance", len=0.51, y=0.31, x=1.15),
    ),
    height=800,
    width=1000,
    title={
        'text': f"Fractional Volterra (fBM Kernel, H={H:.2f}): Sample Paths and Covariance Structure",
        'y': 0.965, 'x': 0.5, 'xanchor': 'center'
    },
    font=dict(color='white'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=60, l=60, b=35, r=35)
)

fig.update_xaxes(
    title_text="time step (t_j)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)
fig.update_xaxes(
    title_text="time step (t_j)", row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.22)'
)
fig.update_yaxes(
    title_text="time step (t_i)", row=2, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="time step (t_i)", row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="Fractional Process value", row=1, col=1
)
fig.update_xaxes(
    title_text="time", row=1, col=1
)
fig.show()
```



---

#### 5.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Volterra Processes are a subset of Gausian Processes emitting a causal representation capable of simulating stochastic processes with a desired covariance structure
- The Voltera Kernel ($K(s, t)$) dictates the covariance function, we select kernels to emit a desired covariance structure that is by construction herein: *causal*
- This process allows us to simulate rough dynamics from the financial maths literature in the context of pricing path-dependent options, something that a Davies-Harte scheme is incapable of due to global path knowledge and spectral methods (as seen in my previous video)
- Herein we saw examples of selections of a Volterra Kernel that yielded: Brownian motion, Mean-Reverting process (OU), Fractional process (fBm) but we can construct many more processes in this capacity
- In future videos, I would like to discuss implications and applications in more modern models like the quadratic rough Heston and qrH+ (old Bloomberg colleague just wrote a great note on this) for the pricing and hedging of 0DTE options


**Future Topics**

Technical Videos and Other Discussions

- Hawkes Processes
- Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
- Market-Making Models and Simulation (Stoikov-Avellaneda)
- Projects that Made me a Quant
- My First Year as a Quant
- Kalman Filter for Quant Finance
- Why Hedge Funds are Actually Secretive
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Top 3 Uses of Linear Algebra for Quant Finance
- Girsanov's Change of Measure
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- How to Backtest a Trading Strategy with Interactive Brokers
- Algorithmic Volatility Trading System

---

####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
