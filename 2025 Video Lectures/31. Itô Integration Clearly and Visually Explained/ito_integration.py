# ## Itô Integration Clearly and Visually Explained
# 
# Recommended Prerequisite Quant Guild Lectures:
#  - [Itô's Lemma Clearly and Visually Explained](https://youtu.be/YH0tWpBaKGs) (Differentiating a Time-Dependent Function of a Stochastic Process)
#  
#  - [Why is the Definition of a Derivative Useful](https://youtu.be/CD8XYP4lq4g) (Numerical Simulation of Calculus)
# 
#  Lectures Applying Itô's Calculus:
#   - [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4) (Application of the Black-Scholes Model)
#  
#   - [How to Find the Black-Scholes-Merton Partial Differential Equation](https://youtu.be/2iClLEfXuqA?si=AgjS1PE1-uyuf5sh) (Applying Stochastic Calculus)


import numpy as np
import plotly.graph_objects as go

# Generate time points
t = np.linspace(0, 1, 100)

# Simulate Brownian motion
np.random.seed(42)
dW = np.random.normal(0, np.sqrt(1/len(t)), len(t))
W = np.cumsum(dW)

# Create the weight surface
T, B = np.meshgrid(t, np.linspace(min(W)-0.5, max(W)+0.5, 100))
Z = T  # Weight surface is linear in time

# Create the figure
fig = go.Figure()

# Add the weight surface
fig.add_trace(go.Surface(
    x=T,
    y=B,
    z=Z,
    colorscale='Viridis',
    opacity=0.7,
    showscale=False,
    name='Weighting Function f(t,W) = t'
))

# Add the lifted path (Brownian path scaled by weight surface)
fig.add_trace(go.Scatter3d(
    x=t,
    y=W,
    z=t,  # The z-coordinate should match the surface height at (t,W)
    mode='lines',
    line=dict(color='rgb(255,50,50)', width=4),
    name='Lifted Path'
))

# Add the path on the Brownian motion/time plane (z=0)
fig.add_trace(go.Scatter3d(
    x=t,
    y=W,
    z=np.zeros_like(t),
    mode='lines',
    line=dict(color='red', width=4),
    name='Brownian Path'
))

# Add vertical projection lines at several points
projection_points = np.linspace(0, 1, 6)  # 6 evenly spaced points
for tp in projection_points:
    idx = np.abs(t - tp).argmin()
    fig.add_trace(go.Scatter3d(
        x=[t[idx], t[idx]],
        y=[W[idx], W[idx]],
        z=[0, t[idx]],  # Project to surface height
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    ))

    ito_integral = np.cumsum(t * dW)  # Approximate Itô integral
fig.add_trace(go.Scatter3d(
    x=t,
    y=np.zeros_like(t),
    z=ito_integral,
    mode='lines',
    line=dict(color='orange', width=4),
    name='Itô Integral Value'
))


# Update layout
fig.update_layout(
    title='Itô Integration as a Lifted Path onto a Surface Weight',
    scene=dict(
        xaxis_title='Time',
        yaxis_title='Brownian Motion',
        zaxis_title='Weight',
        camera=dict(
            eye=dict(x=1, y=1.5, z=0.4)
        ),
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    width=1000,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',  # Black background
    font=dict(color='white')  # White text
)

fig.show()



# ## Sections
# 1.) Traditional Integration
# 
# 2.) Random Variables and Functions of Random Variables
# 
# 3.) Itô's Lemma
# 
# 4.) Integrating Over a Random Variable
# 
# 5.) Visualizing the Stochastic Integral
# 
# 6.) "Solving" Stochastic Integrals (Stochastic Differential Equations)
# 
# 7.) Statistics, Distributions, and Martingale Property
# 
# 8.) Closing Thoughts and Future Topics


# ---


# ## 1.) Traditional (Riemann) Integration


# #### <u>Analytical Integration</u>
# 
# Let $f(x) = x^2$,
# 
# $$\int_0^2 f(x) dx = \int_0^2 x^2 dx = \frac{x^3}{3} |_0^2 = \frac{2^3}{3} - \frac{0^3}{3} = \frac{8}{3} \approx 2.667$$


# #### <u>Numerical Integration</u>
# 
# **Riemann Integration:**
# 
# Similar to numerical differentiation with the definition of a derivative:
# 
# $$f'(x) = lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h} \approx \frac{f(x+h) - f(x)}{h} \text{: when h is small}$$
# 
# We can use the definition of the integral and instead of a limit just use a small change in $x$!
# 
# $$\int_a^b f(x)dx = lim_{\Delta x \rightarrow 0} \sum_{k = 1}^n f(x_k^*) \Delta x \approx \sum_{k = 1}^n f(x_k^*) \Delta x \text{: when } \Delta x \text{ is small}$$
# 
# *Note:* $x_k^*$ simply dictates, based on the partition, if we are using the left/mid/right point of the rectangle for approximation - may have major implications in pricing and whether a more complicated random funciton is a martingale so we have to consider this! 
# 
# **Pros:**
# - Simple to implement, much like the approximate derivative
# - Error dercreases at $O(\frac{1}{n})$ 
# 
# **Cons:**
# - Becomes quite expensive in higher dimensions


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
from ipywidgets import interact
import ipywidgets as widgets

def f(x):
    return x**2

def plot_riemann_sum(n):
    # Create points for smooth curve
    x_curve = np.linspace(0, 2, 200)
    y_curve = f(x_curve)
    
    # Create Riemann sum with n rectangles
    x = np.linspace(0, 2, n+1)
    dx = x[1] - x[0]
    
    # Left endpoints
    x_left = x[:-1]
    y_left = f(x_left)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the curve
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        name='f(x) = x²',
        line=dict(color='#4A9EFF', width=3)  # Brighter blue
    ))
    
    # Add the rectangles
    for i in range(len(x_left)):
        fig.add_trace(go.Scatter(
            x=[x_left[i], x_left[i], x_left[i]+dx, x_left[i]+dx, x_left[i]],
            y=[0, y_left[i], y_left[i], 0, 0],
            fill="toself",
            fillcolor='rgba(255,99,71,0.4)',  # Slightly more opaque
            line=dict(color='rgba(255,99,71,0.6)'),
            showlegend=False
        ))
    
    # Calculate actual integral and approximation
    actual = (2**3)/3  # Analytical solution of ∫x² dx from 0 to 2
    approx = np.sum(y_left * dx)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Riemann Sum Approximation of ∫x² dx from 0 to 2',
            font=dict(size=24, color='#F0F0F0')  # Brighter text
        ),
        xaxis_title=dict(text='x', font=dict(size=18, color='#F0F0F0')),
        yaxis_title=dict(text='y', font=dict(size=18, color='#F0F0F0')),
        annotations=[
            dict(
                x=0.02,
                y=2.8,
                xref='paper',
                yref='y',
                text=f'Actual integral: {actual:.3f}',
                showarrow=False,
                font=dict(size=16, color='#F0F0F0')
            ),
            dict(
                x=0.02,
                y=2.4,
                xref='paper',
                yref='y',
                text=f'Approximation: {approx:.3f}',
                showarrow=False,
                font=dict(size=16, color='#F0F0F0')
            ),
            dict(
                x=0.02,
                y=2.0,
                xref='paper',
                yref='y',
                text=f'Number of rectangles: {n}',
                showarrow=False,
                font=dict(size=16, color='#F0F0F0')
            )
        ],
        showlegend=True,
        width=900,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
        legend=dict(
            font=dict(size=16, color='#F0F0F0'),
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        ),
        margin=dict(t=100, l=80, r=50, b=80)
    )
    
    # Add grid
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.15)',  # Subtle grid
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(128,128,128,0.4)',
        tickfont=dict(size=14, color='#F0F0F0')
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.15)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(128,128,128,0.4)',
        tickfont=dict(size=14, color='#F0F0F0')
    )
    
    fig.show()

# Create interactive slider with improved styling
style = {'description_width': 'initial'}
layout = widgets.Layout(width='400px')
slider = widgets.IntSlider(
    min=2,
    max=50,
    step=1,
    value=6,
    description='Number of rectangles:',
    style=style,
    layout=layout,
    continuous_update=True  # Enable live updates
)
# Apply dark theme to widgets
slider.style.handle_color = '#4A9EFF'  # Match curve color
slider.style.progress_color = '#4A9EFF'

interact(plot_riemann_sum, n=slider)



# 
# There are alternative schemes (i.e. Monte Carlo Integration) that offer difference error gaurentees and computational cost. 
# 
# Certainly important as many integrals (especially stochastic integrals for SDEs) will require simulation!  A video for another day!


# ---


# ## 2.) Random Variables and Functions of Random Variables
# 
# #### <u>Random Variables</u>
# 
# Random Variables that are *well-defined* are governed by a probability mass or density function - they follow a *distribution* of possible outcomes.
# 
# $$X \sim N(0, 1) \implies f_X(s) = \frac{1}{\sqrt{2\pi}} e^{-\frac{s^2}{2}}$$
# 
# Moreover, functions of random variables are also themselves random variables!
# 
# Suppose $X \sim N(0, 1), f(X) = X^2$, since $X$ is uncertain the function $f$ of random variable $X$ must also be uncertain and therefore a random variable itself!


import numpy as np
import plotly.graph_objects as go

# Generate x values
x = np.linspace(-4, 4, 200)

# Calculate standard normal PDF
pdf = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)

# Create figure
fig = go.Figure()

# Add PDF curve
fig.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        name='Standard Normal PDF',
        line=dict(color='#4A9EFF', width=3)
    )
)

# Update layout for dark theme
fig.update_layout(
    template='plotly_dark',
    title=dict(
        text='Standard Normal Probability Density Function',
        x=0.5,
        font=dict(size=20)
    ),
    xaxis_title=dict(text='x', font=dict(size=16)),
    yaxis_title=dict(text='f(x)', font=dict(size=16)),
    showlegend=False,
    hovermode='x',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.15)',
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='rgba(128,128,128,0.4)',
    tickfont=dict(size=14, color='#F0F0F0')
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.15)',
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='rgba(128,128,128,0.4)',
    tickfont=dict(size=14, color='#F0F0F0')
)

fig.show()



# #### <u>Stochastic Processes: Brownian Motion</u>
# 
# Stochastic processes are simply a collection of random variables indexed by time, typically, with properties governing their behavior over time.
#  
#  The key characteristics of Brownian motion ($W_t$) are:
#  
#  - $W_t$ is almost surely continuous
#  
#  - $W_t$ has independent increments
# 
#  - $W_0 = 0$
# 
#  - $W_t - W_s \sim N(0, t - s)$ 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Generate Brownian motion paths
def generate_brownian_path(n_steps=1000, T=1.0):
    dt = T/n_steps
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # Add W₀ = 0
    t = np.linspace(0, T, n_steps + 1)
    return t, W, dW

# Generate multiple paths
n_paths = 5
paths_data = [generate_brownian_path() for _ in range(n_paths)]

# Create subplots
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Sample Paths of Brownian Motion', 'Increments (dW)'),
                    vertical_spacing=0.15,
                    row_heights=[0.7, 0.3])

# Colors for paths
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']

# Add each path and its increments
for i, (t, W, dW) in enumerate(paths_data):
    # Plot Brownian path
    fig.add_trace(
        go.Scatter(
            x=t,
            y=W,
            mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'Sample Path {i+1}'
        ),
        row=1, col=1
    )
    
    # Plot increments
    t_increments = t[:-1]  # Remove last point since we have n-1 increments
    fig.add_trace(
        go.Scatter(
            x=t_increments,
            y=dW,
            mode='lines',
            line=dict(color=colors[i], width=1),
            name=f'Increments {i+1}',
            showlegend=False
        ),
        row=2, col=1
    )

# Update layout
fig.update_layout(
    template='plotly_dark',
    title=dict(
        text='Brownian Motion and Brownian Increments',
        x=0.5,
        font=dict(size=24)
    ),
    showlegend=True,
    hovermode='x',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=800  # Make plot taller to accommodate subplot
)

# Update axes
for row in [1, 2]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(128,128,128,0.5)',
        tickfont=dict(size=16),
        row=row, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(128,128,128,0.5)',
        tickfont=dict(size=16),
        row=row, col=1
    )

# Update axis labels
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="W(t)", row=1, col=1)
fig.update_yaxes(title_text="dW", row=2, col=1)

fig.show()



# The key takeaway here is any time we are dealing with a random variable or a stochastic process (collection of random variables indexed by time) we may be able to say something with certainty about their distribution or statistics (mean, variance) but there will not be one value they can take on.
# 
# Think like the integral example above - there is one solution.  If we have a random variable in our integration there will be *multiple* solutions based on the random path we generate.
# 
# However! We can say something deterministically (typically, under certain circumstances) about the statistics of that integral! Its mean, variance, perhaps its distribution, etc.


# ---


# ## 3.) Itô's Lemma
# 
# The differential for a time-dependent function of a stochastic process can be decomposed into a drift and shock component by the stochastic chain rule (Itô's Lemma).
# 
# $$df(t, W_t) = (\frac{\delta f}{\delta t} + \frac{1}{2}\frac{\delta^2 f}{\delta W_t^2})dt+ \frac{\delta f}{\delta W_t} dW_t$$
# 
# I've discussed Itô's Lemma in a previous video focusing on the Itô correction term and how the lemma will not *predict* the differential for any one sample path, but rather will match the appropriate expectation and variance associated with its distribution.
# 
# Moreover, any one differential of a time dependent function of a stochastic process will itself be random; but we can precisely decompose it into its drift and shock components.


# Example: $f(t, W_t) = W_t^2$
# 
# $$\implies df(t, W_t) = dt + 2W_tdW_t$$
# 
# Any one path's differential will be random, but the differential can be decomposed into drift and shock components!
# 
# $$f(t, W_t) - f(s, W_s) \approx (t-s) + 2 W_t (W_t - W_s)$$


# Simulate W_t^2 and show differential convergence
import numpy as np
import plotly.graph_objects as go

# Simulation parameters
n_paths = 1
n_steps = 1000
dt = 1/n_steps
t = np.linspace(0, 1, n_steps+1)

# Generate Brownian paths
dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
W = np.cumsum(dW, axis=1)
W = np.insert(W, 0, 0, axis=1)  # Add W_0 = 0

# Calculate W_t^2
W_squared = W**2

# Calculate empirical differential
dW_squared = np.diff(W_squared, axis=1)

# Calculate theoretical differential components
dt_component = np.ones_like(W[:,:-1])  # dt term
dW_component = 2 * W[:,:-1] * dW  # 2W_t dW_t term

# Calculate average empirical differential
avg_empirical = np.mean(dW_squared, axis=0)
theoretical = dt + np.mean(2 * W[:,:-1] * dW, axis=0)

# Create plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t[1:],
    y=avg_empirical,
    name='Empirical Average',
    line=dict(color='#00FF00', width=2)
))

fig.add_trace(go.Scatter(
    x=t[1:],
    y=theoretical,
    name='Theoretical (Itô\'s Lemma)',
    line=dict(color='#FF00FF', width=2, dash='dash')
))

# Update layout
fig.update_layout(
    title='Convergence of W_t² Differential to Itô\'s Lemma',
    xaxis_title='t',
    yaxis_title='Differential',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#F0F0F0'),
    showlegend=True,
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(128,128,128,0.4)',
        borderwidth=1
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.15)',
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='rgba(128,128,128,0.4)',
    tickfont=dict(size=14, color='#F0F0F0')
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.15)',
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='rgba(128,128,128,0.4)',
    tickfont=dict(size=14, color='#F0F0F0')
)

fig.show()



# Fundamentally, we can consider the statistics of the distribution of the change 
# 
# $$\mathbb{E}[df(t, W_t)] = dt, Var(df(t, W_t)) = 4W_t^2 dt$$
# 
# or the decomposition of the change itself into drift and shock components give by Itô's Lemma.
# 
# Though the expectation of the differential in this case is just $dt$, we need to accumulate the variation in the distribution at terminal time $T$ which is particularly important in pricing as this terminal distribution is the distribution of interest!
# 
# More on this when we discuss SDEs!


# ---


# ## 4.) Integration Over a Random Variable
# 
# We can use Itô's Lemma to bridge the gap between the differential of a time-dependent function of a stochastic process and a stochastic (Itô) integral!
# 
# **Itô's Lemma:**
# $$df(t, W_t) = (\frac{\delta f}{\delta t} + \frac{1}{2}\frac{\delta^2 f}{\delta W_t^2})dt+ \frac{\delta f}{\delta W_t} dW_t$$
# 
# **Itô Integrals:**
# $$\int_0^t f(s, W_s) dW_s$$
# 
# Effectively, this is the **cumulative effect of weighted randomness**.
# 
# **Components of the Stochastic Integral:**
# 
# $$\int_0^t f(s, W_s) dW_s$$
# 
# - $f(t, W_t)$ the function weighting the Brownian increments
# 
# - $dW_t$ the Brownian increment
# 
# - $\int_0^t$ in the discretized sense is simply a cumulative sum
# 
# - *Lifting* is simply the weight $f(t, W_t)$ relative to $(t, W_t)$ being used to scale the Brownian increment $dW_t$ (i.e. used for $f(t, W_t) \times dW_t$)
# 
# It's pretty easy to compute numerically in the discrete sense as:
# 
# $$\int_0^t f(s, W_s) dW_s \approx \sum_{i=0}^{N-1}f(t_i, W_{t_i})(W_{t_i} - W_{t_i})$$


# ---


# ## 5.) Visualizing the Stochastic Integral
# 
# There are different ways to think about the Itô integral - I quite enjoy the following visual to illustrate the weighting of Brownian increments!


# #### Example:
# 
# $$\int_0^t s dW_s$$
# 
# Remember, this integral isn't *one value* its a random variable!
# 
# **Components of the Stochastic Integral:**
# 
# - $f(t, W_t) = t$ the function weighting the Brownian increments
# 
# - $dW_t$ the Brownian increment
# 
# - $\int_0^t$ in the discretized sense is simply a cumulative sum
# 
# - *Lifting* is simply the weight $f(t, W_t) = t$ relative to $(t, W_t)$ being used to scale the Brownian increment $dW_t$ (i.e. $t \times dW_t$)


import numpy as np
import plotly.graph_objects as go

# Generate time points
t = np.linspace(0, 1, 100)

# Simulate Brownian motion
dW = np.random.normal(0, np.sqrt(1/len(t)), len(t))
W = np.cumsum(dW)

# Create the weight surface
T, B = np.meshgrid(t, np.linspace(min(W)-0.5, max(W)+0.5, 100))
Z = T  # Weight surface is linear in time

# Create the figure
fig = go.Figure()

# Add the weight surface
fig.add_trace(go.Surface(
    x=T,
    y=B,
    z=Z,
    colorscale='Viridis',
    opacity=0.7,
    showscale=False,
    name='Weighting Function f(t,W) = t'
))

# Add the lifted path (Brownian path scaled by weight surface)
fig.add_trace(go.Scatter3d(
    x=t,
    y=W,
    z=t,  # The z-coordinate should match the surface height at (t,W)
    mode='lines',
    line=dict(color='rgb(255,50,50)', width=4),
    name='Lifted Path'
))

# Add the path on the Brownian motion/time plane (z=0)
fig.add_trace(go.Scatter3d(
    x=t,
    y=W,
    z=np.zeros_like(t),
    mode='lines',
    line=dict(color='red', width=4),
    name='Brownian Path'
))

# Add vertical projection lines at several points
projection_points = np.linspace(0, 1, 6)  # 6 evenly spaced points
for tp in projection_points:
    idx = np.abs(t - tp).argmin()
    fig.add_trace(go.Scatter3d(
        x=[t[idx], t[idx]],
        y=[W[idx], W[idx]],
        z=[0, t[idx]],  # Project to surface height
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    ))

    ito_integral = np.cumsum(t * dW)  # Approximate Itô integral
fig.add_trace(go.Scatter3d(
    x=t,
    y=np.zeros_like(t),
    z=ito_integral,
    mode='lines',
    line=dict(color='orange', width=4),
    name='Itô Integral Value'
))


# Update layout
fig.update_layout(
    title='Itô Integration as a Lifted Path onto a Surface Weight',
    scene=dict(
        xaxis_title='Time',
        yaxis_title='Brownian Motion',
        zaxis_title='Weight',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8)
        ),
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    width=900,
    height=700,
    paper_bgcolor='rgba(0,0,0,0)',  # Black background
    font=dict(color='white')  # White text
)

fig.show()



# ---


# ## 6.) "Solving" Stochastic Integrals (Stochastic Differential Equations)
# 
# - Ito Isometry, Expectation, Variance to Characterize the Solution (its a Distribution)
# - Analytical Solution (SDEs)
# - Numerical Solutions


# #### Example: 
# 
# $$\int_0^t W_sdW_s$$
# 
# Remember, this integral isn't *one value* its a random variable!


# #### <u>Analytical Solutions</u>


# **General Recipe for Solving SDEs:**
# 
# 1.) Generate a candidate function
# 
# 2.) Apply Itô's Lemma
# 
# 3.) Rearrange and integrate for a solution


# #### Solving the Stochastic Integral
# 
# **1.) Guess $W_t^2$**
# 
# **2.) Apply Itô's Lemma**
# 
# $$f(t, W_t) = W_t^2 \implies df(t, W_t) = dW_t$$
# 
# $$dW_t^2 = \frac{\partial W_t^2}{\partial t}dt + \frac{\partial W_t^2}{\partial W_t}dW_t + \frac{1}{2}\frac{\partial^2 W_t^2}{\partial W_t^2}dt$$
# 
# Partially differentiating $f(t, W_t) = W_t^2$:
# 
#  $\frac{\partial W_t^2}{\partial t} = 0$
# 
#  $\frac{\partial W_t^2}{\partial W_t} = 2W_t$
# 
#  $\frac{\partial^2 W_t^2}{\partial W_t^2} = 2$
# 
# 
# $$dW_t^2 = 0 \cdot dt + 2W_t dW_t + \frac{1}{2}(2)dt$$
# 
# **3.) Rearrange and integrate for a solution:**
#  
# $$dW_t^2 = 2W_t dW_t + dt$$
# 
# Integrating both sides:
#  
# $$W_t^2 = 2\int_0^t W_s dW_s + t$$
#  
# $$\int_0^t W_s dW_s = \frac{1}{2}W_t^2 - \frac{1}{2}t$$


# #### <u>Numerical Solutions</u>
# 
# Let's compare two ways of computing the stochastic integral $\int_0^t W_s dW_s$ numerically:
# 
# 1. Using the analytical solution: $\int_0^t W_s dW_s = \frac{1}{2}W_t^2 - \frac{1}{2}t$
#    - We can directly compute this using our simulated Brownian paths
# 
# 2. Using Euler-Maruyama discretization: $\sum_i W_i \Delta W_i$
#    - We approximate the integral by summing the product of $W$ at each point 
#      with the corresponding Brownian increment
# 
# These two approaches should converge to the same values as our time step $dt \to 0$.
# This is because the Euler-Maruyama scheme is a consistent discretization of the 
# stochastic integral.
# 
# In the visualization below, we'll plot multiple paths using both methods to show
# their equivalence. The orange paths use the analytical solution while the blue
# paths use Euler-Maruyama discretization.
# 


# #### <u>Visualizing Both Numerical Solutions</u>


# Simulate paths with different discretization levels to show convergence
n_paths = 100
discretization_levels = [10, 50, 200, 1000]  # Different numbers of time steps
T = 1.0

# Create subplots - 3 for paths and 1 for error
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'dt = {T/n:.3f}' for n in discretization_levels[:3]] + ['Error vs Time Steps'],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Colors for different methods
analytical_color = 'rgba(255, 127, 14, 0.1)'  # Orange with transparency
euler_color = 'rgba(31, 119, 180, 0.1)'       # Blue with transparency
mean_analytical_color = 'orange'
mean_euler_color = 'blue'
std_analytical_color = 'rgba(255, 127, 14, 0.2)'
std_euler_color = 'rgba(31, 119, 180, 0.2)'

errors = []  # Store errors for each discretization level

for idx, n_steps in enumerate(discretization_levels):
    dt = T/n_steps
    t = np.linspace(0, T, n_steps)
    
    # Generate Brownian paths and increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps-1))
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])
    
    # Analytical solution
    analytical_integral = 0.5 * W**2 - 0.5 * t
    
    # Euler-Maruyama discretization
    euler_integral = np.zeros((n_paths, n_steps))
    for i in range(n_steps-1):
        euler_integral[:, i+1] = euler_integral[:, i] + W[:, i] * dW[:, i]
    
    # Calculate means and standard deviations
    mean_analytical = np.mean(analytical_integral, axis=0)
    mean_euler = np.mean(euler_integral, axis=0)
    std_analytical = np.std(analytical_integral, axis=0)
    std_euler = np.std(euler_integral, axis=0)
    
    # Calculate error
    error = np.mean(np.abs(analytical_integral - euler_integral))
    errors.append(error)
    
    # Only plot paths for first 3 subplots
    if idx < 3:
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Plot individual paths (only 20 for clarity)
        for i in range(20):
            fig.add_trace(
                go.Scatter(x=t, y=analytical_integral[i], mode='lines',
                          line=dict(color=analytical_color, width=1),
                          showlegend=False),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=t, y=euler_integral[i], mode='lines',
                          line=dict(color=euler_color, width=1),
                          showlegend=False),
                row=row, col=col
            )
        
        # Plot mean paths
        fig.add_trace(
            go.Scatter(x=t, y=mean_analytical, mode='lines',
                      line=dict(color=mean_analytical_color, width=3),
                      showlegend=False),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=t, y=mean_euler, mode='lines',
                      line=dict(color=mean_euler_color, width=3),
                      showlegend=False),
            row=row, col=col
        )
        
        # Add confidence bands (±1 standard deviation)
        fig.add_trace(
            go.Scatter(x=t, y=mean_analytical + std_analytical,
                      mode='lines', line=dict(width=0),
                      fillcolor=std_analytical_color, fill='tonexty',
                      showlegend=False),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=t, y=mean_analytical - std_analytical,
                      mode='lines', line=dict(width=0),
                      fillcolor=std_analytical_color, fill='tonexty',
                      showlegend=False),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=t, y=mean_euler + std_euler,
                      mode='lines', line=dict(width=0),
                      fillcolor=std_euler_color, fill='tonexty',
                      showlegend=False),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=t, y=mean_euler - std_euler,
                      mode='lines', line=dict(width=0),
                      fillcolor=std_euler_color, fill='tonexty',
                      showlegend=False),
            row=row, col=col
        )

# Plot error in the last subplot
fig.add_trace(
    go.Scatter(x=discretization_levels, y=errors,
               mode='lines+markers',
               line=dict(color='white', width=2),
               name='Mean Absolute Error'),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title='Convergence of Analytical and Euler-Maruyama Solutions with Different Time Steps',
    showlegend=False,
    width=1000,
    height=800,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes labels
for i in range(1, 4):
    fig.update_xaxes(title_text='Time', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    fig.update_yaxes(title_text='Integral Value', row=(i-1)//2 + 1, col=(i-1)%2 + 1)

# Update error plot axes
fig.update_xaxes(title_text='Number of Time Steps', row=2, col=2, type='log')
fig.update_yaxes(title_text='Mean Absolute Error', row=2, col=2, type='log')

fig.show()



# ---


# ## 7.) Statistics, Distribution of Solutions, Martingale Property
# 
# We may analyze the expected value and variance of these distributions along with the distribution that characterizes the integral itself (in this case it is a scaled and shifted chi-squared distribution)!
# 
# Let's examine why both the Euler-Maruyama discretization and the analytical solution follow the same distribution for our stochastic integral.
# 
# For a Brownian motion $W_t$, we know that:
# 
# $\int_0^t W_s dW_s = \frac{1}{2}(W_t^2 - t)$
# 
# The Euler-Maruyama discretization approximates this as:
# 
# $\sum_{i=0}^{n-1} W_{t_i}(W_{t_{i+1}} - W_{t_i})$
# 
# The key insight is that both methods preserve the martingale property and the quadratic variation of the underlying Brownian motion, ensuring they converge to the same distribution as $\Delta t \to 0$.
# 


# ---


# ## 8.) Closing Thoughts and Future Topics
# 
# Stochastic integration is tricky! The integral itself is a random variable, and many of the functions are themselves random variables weighting the Brownian increments.  Herein we've broken down the stochastic integral into its components, visualized each component, and acknowledged its place in the overall weighting of the Brownian increments.
# 
# In future videos we will look to explore more on:
# 
# - Stochastic Differential Equations
# 
# - Martingales and Pricing
# 
# - Quadrature (numerical integral simulation)


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

