# ## Finite Differences Option Pricing for Quant Finance
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# *Necessary Stochastic Calculus:*
# -  [Itô's Lemma Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo)
# 
# -  [Itô Integration Clearly and Visually Explained](https://youtu.be/dUvZ8m3QpeI)
# 
# - [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)
# 
# *Why the Expectation is Sufficient for Pricing and Problems in Practice:*
# - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
# - [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)
# 
# - [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
# 
# *Applications in Market-Making and Trading:*
# 
# - [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4)
# 
# - [Trading with the Black-Scholes Implied Volatility Surface](https://youtu.be/YH0tWpBaKGs)
# 
# - [How to Price Exotic Options](https://youtu.be/hsot26myYYM)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
#  
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# ##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)
# 
# ---


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x values for the parabola
x = np.linspace(-2, 5, 200)

# Create the parabola function
def f(x):
    return x**2

# Calculate derivative approximation
h = 0.0001
def approx_derivative(x0):
    return (f(x0 + h) - f(x0))/h

# Create figure
fig = go.Figure()

# Plot the actual parabola
fig.add_trace(
    go.Scatter(
        x=x,
        y=f(x),
        mode='lines',
        line=dict(color='rgb(57, 255, 20)', width=3),
        name='Original Function f(x) = x²'
    )
)

# Add tangent lines at many points from x=0 to x=5
num_points = 20  # Increase number of points for smoother transition
points = np.linspace(0, 5, num_points)
base_opacity = 1
opacity_step = 0.9 / len(points)

# Add first tangent line that will appear in legend
x0 = points[0]
slope = approx_derivative(x0)
tangent = slope * (x - x0) + f(x0)
fig.add_trace(
    go.Scatter(
        x=x,
        y=tangent,
        mode='lines',
        line=dict(color='rgb(0, 191, 255)', width=2),
        name='Approximating Tangent Lines'
    )
)

# Add remaining tangent lines with decreasing opacity
for i, x0 in enumerate(points[1:]):
    slope = approx_derivative(x0)
    tangent = slope * (x - x0) + f(x0)
    opacity = base_opacity - (i * opacity_step)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=tangent,
            mode='lines',
            line=dict(
                color=f'rgba(0, 191, 255, {opacity})',
                width=2
            ),
            showlegend=False
        )
    )

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='Reconstructing a Function from its Derivatives',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Update axes with fixed ranges to show full range
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='x',
    range=[-2, 5]  # Extended range to show full parabola
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='f(x)',
    range=[-2, 25]  # Extended range to show full parabola height
)

fig.show()



# ---


# ### 📖 Sections
# 
# #### 1.) 🎯 Differential Equations in Quant Finance
# 
# - Pricing Differential Equations
# 
# - Analytical v. Numerical Solutions
# 
# #### 2.) 🌊 Approximating Ordinary Differential Equations 
# 
# - Definition of a derivative
# 
# - Approximating Derivatives using the Definition
# 
# - Euler's Method and Finite Differences
# 
# - Numerically Solving an Ordinary Differential Equation
# 
# - Code Walkthrough and Visualization
# 
# #### 3.) 🔥 Approximating Partial Differential Equations
# 
# - Visualizing a Partial Differential Equation Solution
# 
# - Numerically Solving a Partial Differential Equation
# 
# - Code Walkthrough and Visualization
# 
# #### 4.) 📝 Approximating the Black-Scholes Partial Differential Equation
# 
# - Visualizing a Finite Differences Approximation in a Black-Scholes Framework
# 
# #### 5.) 💭 Closing Thoughts and Future Topics


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


# ##### 1.) 🎯 Differential Equations in Quant Finance
# 
# Where do differential equations fit into quantitative finance?  
# 
# Well, option pricing arguments can be constructed in terms of continuous hedging which yields a differential equation
# 
# Where does the randomness *go*?  Well, randomness from the model's framework is cancelled out by hedging arguments!
# 
# **The Black-Scholes partial differential equation:**
#  
# $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$
#  
# 
# **The Heston partial differential equation:**
#  
# $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + \rho\sigma\nu S\frac{\partial^2 V}{\partial S\partial \nu} + \frac{1}{2}\nu\frac{\partial^2 V}{\partial \nu^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-\nu)\frac{\partial V}{\partial \nu} - rV = 0$$
#  
# 
# When that differential equation is solved we get the function that gives an option price in that model framework
# 
# These arguments can be constructed for vanilla and exotic options - we will focus on European options herein as they are not path dependent
# 
# ##### 🔍 Example: Analytical and Numerical Solutions to Pricing Differential Equations
# 
# Let's begin by focusing on the differential equation and what it's solution gives us 
# 
# Suppose we've constructed an option pricing argument (*Black-Scholes, Heston,* ...) that gives us the following differential equation
#  
#  $$\frac{dy}{dx} = y' = f'(x) \quad \quad f'(x) = \frac{1}{2}x^{-1/2} \quad \quad f(0) = 1$$
# 
#  Here the *model framework* only requires stock price for the corresponding option price, real-world models demand more inputs...
# 
#  The function we are after $f(x) = ?$ will give us the price of the option by plugging in the current stock price $S_t$ for $x$
# 
#  In this case, we can *analytically* solve the equation and get a function that perfectly maps inputs 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x values (using positive values since we're taking square root)
x = np.linspace(0.01, 4, 200)  # Start from small positive number to avoid division by zero

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Pricing Differential Equation: f\'(x) = 1/(2√x)', 'Option Pricing Function: f(x) = √x + C'))

# Plot derivative on left subplot
derivative = 1/(2*np.sqrt(x))
fig.add_trace(
    go.Scatter(
        x=x,
        y=derivative,
        mode='lines',
        line=dict(color='rgb(255, 165, 0)', width=2),
        showlegend=False
    ),
    row=1, col=1
)

# Plot different solutions for various values of C on right subplot
C_values = [-3, -2, -1, 0, 1, 2, 3]
base_opacity = 1
opacity_step = 0.8 / len(C_values)

# First add the general solution trace that will appear in legend
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.sqrt(x) + C_values[0],
        mode='lines',
        line=dict(color='rgb(0, 191, 255)', width=2),
        name='General Solution'
    ),
    row=1, col=2
)

# Then add all other general solutions without showing in legend
for i, C in enumerate(C_values[1:]):
    if C != 1:  # Skip C=1 as it will be the particular solution
        opacity = base_opacity - (i * opacity_step)
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=np.sqrt(x) + C,
                mode='lines',
                line=dict(
                    color='rgba(0, 191, 255, {})'.format(opacity),
                    width=2
                ),
                showlegend=False
            ),
            row=1, col=2
        )

# Add particular solution last so it appears on top
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.sqrt(x) + 1,
        mode='lines',
        line=dict(
            color='rgb(57, 255, 20)',  # Neon green
            width=3
        ),
        name='Particular Solution'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,  # Increased width for two subplots
    height=500,
    title_text='Option Pricing Differential Equation and Its Solutions',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Update axes
for i in [1, 2]:
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='x - Stock Price',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='f(x) - Cost of Option in Dollars',
        row=1, col=i
    )

fig.show()



# Solving for the equation analytically yields
# 
# $$\int f'(x) = \int \frac{1}{2}x^{- \frac{1}{2}} \implies f(x) = x^{\frac{1}{2}} + C$$
# $$f(0) = 1 \implies f(0) = 0^{\frac{1}{2}} + C = 1 \implies C = 1$$
# 
# Thus, the option pricing function is given by $f(x) = x^{\frac{1}{2}} + 1$ which gives the price of our option given a stock price $x$
# 
# Under the assumed model framework that produces the differential equation, this function gives the *correct* price *everywhere*
# 
# Any stock price I give the pricing function will return the valid model price because this is the *analytical* or *closed-form* solution


# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'xy'}, {'type': 'surface'}]],
    subplot_titles=('Option Pricing Function', 'Market Implied Volatility Surface')
)

# Generate x values for pricing function
x = np.linspace(0.01, 4, 200)

# Add the pricing function to first subplot
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.sqrt(x) + 1,
        mode='lines',
        line=dict(
            color='rgb(57, 255, 20)',  # Neon green
            width=3
        ),
        name='Option Pricing Function',
        showlegend=False
    ),
    row=1, col=1
)

strikes = [90, 95, 100, 105, 110]
maturities = [1/12, 3/12, 6/12, 1, 2]    # in years
maturity_labels = ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years']

# Market volatility values (in %)
market_vols = np.array([
    [28.0, 24.5, 22.0, 20.5, 19.5],  # 1 month
    [27.5, 24.0, 21.8, 20.3, 19.3],  # 3 months
    [27.0, 23.5, 21.5, 20.0, 19.0],  # 6 months
    [26.5, 23.0, 21.2, 19.8, 18.8],  # 1 year
    [26.0, 22.5, 21.0, 19.5, 18.5]   # 2 years
])

# Create meshgrid for 3D surface
X, Y = np.meshgrid(strikes, maturities)

# Add market volatility surface to second subplot
fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=market_vols,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True,
        name='Market Volatility Surface',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1000,
    height=600,
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for pricing function plot
fig.update_xaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True, 
    zerolinewidth=1, 
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='x - Stock Price',
    row=1, col=1
)
fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True, 
    zerolinewidth=1, 
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='f(x) - Cost of Option in Dollars',
    row=1, col=1
)

# Update 3D scene for volatility surface
fig.update_scenes(
    xaxis_title='Strike Price',
    yaxis_title='Time to Maturity (Years)',
    zaxis_title='Implied Volatility (%)',
    xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
    yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
    zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
    bgcolor='rgba(0,0,0,0)',
    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
)

fig.show()



# Analytical solutions are preferred but not always possible.
# 
# Why do we use other models if they don't have analytical solutions?  
# 
# Some models capture necessary dynamics of the market other models don't have the ability to capture 
# 
# For example, constant v.s. stochastic volatility and the dynamics of the implied volatility surface
# 
# So what can we do if there is no analytical solution available? 
# 
# We have a variety of techniques that are sufficient to approximate solutions to the pricing partial differential equations!


# ---


# ##### 2.) 🌊 Approximating Differential Equations 


# When analytical solutions are unavailable we can approximate differential equations numerically
# 
# Computers are particularly effective for this!  Let's understand *why* approximations work by looking at a method for ordinary differential equations
# 
# ##### 📐 Definition of a Derivative
# 
# Assuming a function is differentiable, the derivative is defined as
# 
# $$f'(x) = lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$$
# 
# We learn it as the slope of the tangent line, and we take for granted the definition - in fact many students have a distaste for it


import numpy as np
import plotly.graph_objects as go

# Create a simple function and its derivative
x = np.linspace(-2, 2, 100)
f = x**2  # Example function: f(x) = x^2
f_prime = 2*x  # True derivative: f'(x) = 2x

# Point where we want to show the tangent line
x0 = 1
y0 = x0**2
slope = 2*x0

# Create points for tangent line with extended domain
x_tangent = np.array([x0-2, x0+2])  # Extended from [-0.5,0.5] to [-2,2]
y_tangent = slope*(x_tangent - x0) + y0

# Create the plot
fig = go.Figure()

# Plot the original function
fig.add_trace(go.Scatter(
    x=x, y=f,
    mode='lines',
    name='f(x) = x²',
    line=dict(color='rgb(0, 191, 255)', width=2),
    showlegend=True
))

# Plot the tangent line
fig.add_trace(go.Scatter(
    x=x_tangent, y=y_tangent,
    mode='lines',
    name=f'Tangent at x={x0}',
    line=dict(color='red', width=2, dash='dot'),
    showlegend=True
))

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='Function and its Tangent Line',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='x'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='y'
)

fig.show()



# Math professors and academics will have calculus students to apply this limit definition to find derivatives and prove rules...
# 
# **Proving the Power Rule:**
# 
#  $$\begin{align}
#  \frac{d}{dx}x^n &= \lim_{h \to 0} \frac{(x+h)^n - x^n}{h} \\
#  &= \lim_{h \to 0} \frac{\sum_{k=0}^n \binom{n}{k}x^{n-k}h^k - x^n}{h} \\
#  &= \lim_{h \to 0} \frac{x^n + nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + ... - x^n}{h} \\
#  &= \lim_{h \to 0} \left(nx^{n-1} + \frac{n(n-1)}{2}x^{n-2}h + ...\right) \\
#  &= nx^{n-1}
#  \end{align}$$
# 
# **Using the Power Rule:**
# 
# $f(x) = x^2 \implies f'(x) = 2x$
# 
# 
# How does this get students stoked about the definition of a derivative? This is a terrible approach!  
# 
# First we should appreciate the limit definition and how *easy* it is to use!
# 
# **Approximating Derivatives:**
# 
# Assuming f(x) is differentiable, we can approximate derivatives and therefore diffrential equations
# 
# Instead of infinitely small we can just pick a *very smamll* value for *h*
# 
# $$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \quad \text{ when } h \text{ is small}$$
# 
# This is **so** easy, in the following code I'll take the derivative of a function approximately (numerically) and analytically


# Function
def f(x):
    return x**2

# Actual Derivative
def f_prime(x):
    return 2*x

h = .0000001

print("Approximation:", (f(1 + h) - f(1))/h)
print("Actual:", f_prime(1))


# #####  💭 Big Idea
# 
# **What if we rearrange the approximate equation?**
# 
# $$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$
# 
# $$\implies hf'(x) \approx f(x+h) - f(x)$$
# 
# $$\implies f(x+h) \approx f(x) + hf'(x)$$
# 
# We have now expressed the *original* function *approximately* in terms of its derivative!
# 
# That means we can solve for the original function iteratively if we are given a differential equation!
# 
# We just derived *Euler's method* and *Finite Differences* for approximating ODEs!


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x values
x = np.linspace(-2, 5, 200)

# Create parabola function and its derivative
def f(x):
    return x**2

def approx_derivative(x0, h=0.0001):
    return (f(x0 + h) - f(x0))/h

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Plot original parabola with reduced opacity
fig.add_trace(
    go.Scatter(
        x=x,
        y=f(x),
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.3)', width=3),
        name='Target Function f(x) = x²'
    )
)

# Initialize reconstruction at x=0
x_recon = [0]
y_recon = [0]
steps = 50
x_max = 5
dx = x_max/steps

# Create frames for animation
frames = []
for i in range(steps + 1):
    x_current = i * dx
    
    # Get current derivative and point
    slope = approx_derivative(x_current)
    y_current = f(x_current)
    
    # Add to reconstruction arrays
    x_recon.append(x_current)
    y_recon.append(y_current)
    
    # Create tangent line points with extended domain
    x_tangent = np.linspace(x_current-10, x_current+10, 100)  # Extended domain
    y_tangent = slope*(x_tangent - x_current) + y_current
    
    # Create frame
    frame = go.Frame(
        data=[
            # Original function (stays constant)
            go.Scatter(
                x=x,
                y=f(x),
                mode='lines',
                line=dict(color='rgba(57, 255, 20, 0.3)', width=3),
                name='Target Function'
            ),
            # Current reconstruction points
            go.Scatter(
                x=x_recon,
                y=y_recon,
                mode='lines+markers',
                line=dict(color='rgb(0, 191, 255)', width=3),
                marker=dict(size=8),
                name='Reconstruction'
            ),
            # Current tangent line
            go.Scatter(
                x=x_tangent,
                y=y_tangent,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                name='Current Tangent'
            )
        ]
    )
    frames.append(frame)

# Add initial empty reconstruction trace
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode='lines+markers',
        line=dict(color='rgb(0, 191, 255)', width=3),
        marker=dict(size=8),
        name='Reconstruction'
    )
)

# Add initial tangent line with extended domain
x_tangent = np.linspace(-10, 10, 100)  # Extended domain
y_tangent = approx_derivative(0)*(x_tangent - 0) + f(0)
fig.add_trace(
    go.Scatter(
        x=x_tangent,
        y=y_tangent,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
        name='Current Tangent'
    )
)

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='Reconstructing a Function from its Derivatives',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True},
                              'fromcurrent': True}],
                'label': '▶️ Play',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 1.2,  # Changed from 1.1 to 1.2 to move button higher
        'yanchor': 'top'
    }]
)

# Set axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='x',
    range=[-2, 5]
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='f(x)',
    range=[-2, 25]
)

# Add frames to figure
fig.frames = frames

fig.show()



# **Remark:** One of the consequences of using numerical methods is that approximations for values outside our *explored* space are poor - we need to iteratively approximate the space of interest to get accurate approximations!


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### Finite Differences:
# 
# This is *all* the finite differences method for approximating solutions to differential equations is!
# 
# We are approximating the derivative, rearranging the equation, and iteratively solving for the original function!
# 
#  **Forward Difference:**
# $$\frac{\partial f}{\partial x} \approx \frac{f(x + \Delta x) - f(x)}{\Delta x} + O(\Delta x)$$
# 
#  **Backward Difference:**
#  $$\frac{\partial f}{\partial x} \approx \frac{f(x) - f(x - \Delta x)}{\Delta x} + O(\Delta x)$$
# 
#  **Central Difference:**
#  $$\frac{\partial f}{\partial x} \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2\Delta x} + O(\Delta x^2)$$
# 
#  **Second Derivative (Central):**
#  $$\frac{\partial^2 f}{\partial x^2} \approx \frac{f(x + \Delta x) - 2f(x) + f(x - \Delta x)}{\Delta x^2} + O(\Delta x^2)$$
# 
#  Where $O(\Delta x)$ and $O(\Delta x^2)$ represent the order of accuracy of the approximation.


# **Remark:** This is why we should be stoked to see derivatives or differential equations in any capacity - they are very easily solved even if we have to resort to techniques to approximate the solution!


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 🔎 Example Euler's Method (Forward Finite-Difference Scheme)
#  
# Consider the pricing differential equation from the start of this notebook.
# 
#  Let's solve the following differential equation using finite differences:
#  
#   $$\frac{dy}{dx} = y' = f'(x) \quad \quad f'(x) = \frac{1}{2}x^{-1/2} \quad \quad f(0) = 1$$
#  
# Let's solve this differential equation step by step using Euler's method:
# 
# **Step 1:** 
# 
# Start with our differential equation
# $$\frac{dy}{dx} = \frac{1}{2}x^{-1/2}$$
# 
# First, we discretize space:
# - Space: $x_n = nh$ for $n = 0,1,...,N$ 
# - Let $y_n$ represent $y(x_n)$
# 
# **Step 2:** 
# 
# Recall finite difference approximation
# $$\frac{dy}{dx} \approx \frac{y_{n+1} - y_n}{h}$$
# 
# **Step 3:** 
# 
# Plug our approximation into the differential equation
# $$\frac{y_{n+1} - y_n}{h} = \frac{1}{2}x_n^{-1/2}$$
# 
# **Step 4:** 
# 
# Solve for the next step $y_{n+1}$
# $$y_{n+1} = y_n + h(\frac{1}{2}x_n^{-1/2})$$
# 
# **Step 5:** 
# 
# Implementation steps:
# 1. Choose initial condition $y_0 = 1$
# 2. Select small step size $h$
# 3. For each step n:
#     - Calculate $x_n = nh$
#     - Calculate $y_{n+1}$ using our formula
#     - Move to next step


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 💻 Coding the Solution: Finite Differences Ordinary Differential Equation


x = np.linspace(0.0001, 5, 2000)

f_prime = lambda x: .5 * x **(-.5)

x_d = [0]
y_hat = [1]

for i in range(1, len(x)):
    x_d.append(x[i])
    y_hat.append(y_hat[-1] + f_prime(x[i]) * (x[i] - x[i-1]))


y_hat


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 🖼️ Visualizing the Numerical and Analytical Solution


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x values (avoid x=0 since derivative is undefined there)
x = np.linspace(0.01, 5, 200)

# Create function and its derivative
def f(x):
    return np.sqrt(x) + 1

def f_prime(x):
    return 0.5 * x**(-0.5)

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Plot original function with reduced opacity
fig.add_trace(
    go.Scatter(
        x=x,
        y=f(x),
        mode='lines',
        line=dict(color='rgba(57, 255, 20, 0.3)', width=3),
        name='Target Function f(x) = √x + 1'
    )
)

# Initialize reconstruction at x=0.01 (to avoid division by zero)
x_recon = [0.01]
y_recon = [f(0.01)]
steps = 50
x_max = 5
dx = x_max/steps

# Create frames for animation
frames = []
for i in range(steps + 1):
    x_current = 0.01 + i * dx
    
    # Get current derivative and point
    slope = f_prime(x_current)
    y_current = f(x_current)
    
    # Add to reconstruction arrays
    x_recon.append(x_current)
    y_recon.append(y_current)
    
    # Create tangent line points
    x_tangent = np.linspace(max(0.01, x_current-2), x_current+2, 100)
    y_tangent = slope*(x_tangent - x_current) + y_current
    
    # Create frame
    frame = go.Frame(
        data=[
            # Original function (stays constant)
            go.Scatter(
                x=x,
                y=f(x),
                mode='lines',
                line=dict(color='rgba(57, 255, 20, 0.3)', width=3),
                name='Target Function'
            ),
            # Current reconstruction points
            go.Scatter(
                x=x_recon,
                y=y_recon,
                mode='lines+markers',
                line=dict(color='rgb(0, 191, 255)', width=3),
                marker=dict(size=8),
                name='Reconstruction'
            ),
            # Current tangent line
            go.Scatter(
                x=x_tangent,
                y=y_tangent,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                name='Current Tangent'
            )
        ]
    )
    frames.append(frame)

# Add initial empty reconstruction trace
fig.add_trace(
    go.Scatter(
        x=[0.01],
        y=[f(0.01)],
        mode='lines+markers',
        line=dict(color='rgb(0, 191, 255)', width=3),
        marker=dict(size=8),
        name='Reconstruction'
    )
)

# Add initial tangent line
x_tangent = np.linspace(0.01, 2, 100)
y_tangent = f_prime(0.01)*(x_tangent - 0.01) + f(0.01)
fig.add_trace(
    go.Scatter(
        x=x_tangent,
        y=y_tangent,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
        name='Current Tangent'
    )
)

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='Reconstructing f(x) = √x + 1 from its Derivative',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True},
                              'fromcurrent': True}],
                'label': '▶️ Play',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 1.2,
        'yanchor': 'top'
    }]
)

# Set axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='x',
    range=[0, 5]
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text='f(x)',
    range=[0, 4]
)

# Add frames to figure
fig.frames = frames

fig.show()


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x values (using positive values since we're taking square root)
x = np.linspace(0.01, 4, 200)  # Start from small positive number to avoid division by zero

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Pricing Differential Equation: f\'(x) = 1/(2√x)', 'Option Pricing Function: f(x) = √x + C'))

# Plot derivative on left subplot
derivative = 1/(2*np.sqrt(x))
fig.add_trace(
    go.Scatter(
        x=x,
        y=derivative,
        mode='lines',
        line=dict(color='rgb(255, 165, 0)', width=2),
        showlegend=False
    ),
    row=1, col=1
)

# Plot different solutions for various values of C on right subplot
C_values = [-3, -2, -1, 0, 1, 2, 3]
base_opacity = 1
opacity_step = 0.8 / len(C_values)

# First add the general solution trace that will appear in legend
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.sqrt(x) + C_values[0],
        mode='lines',
        line=dict(color='rgb(0, 191, 255)', width=2),
        name='General Solution'
    ),
    row=1, col=2
)

# Then add all other general solutions without showing in legend
for i, C in enumerate(C_values[1:]):
    if C != 1:  # Skip C=1 as it will be the particular solution
        opacity = base_opacity - (i * opacity_step)
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=np.sqrt(x) + C,
                mode='lines',
                line=dict(
                    color='rgba(0, 191, 255, {})'.format(opacity),
                    width=2
                ),
                showlegend=False
            ),
            row=1, col=2
        )

# Add particular solution last so it appears on top
fig.add_trace(
    go.Scatter(
        x=x,
        y=np.sqrt(x) + 1,
        mode='lines',
        line=dict(
            color='rgb(57, 255, 20)',  # Neon green
            width=3
        ),
        name='Particular Solution'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,  # Increased width for two subplots
    height=500,
    title_text='Option Pricing Differential Equation and Its Solutions',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Update axes
for i in [1, 2]:
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='x - Stock Price',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='f(x) - Cost of Option in Dollars',
        row=1, col=i
    )

fig.show()



# ---


# ##### 3.) 🔥 Approximating Partial Differential Equations


# We can apply this same methodology to partial differential equations
# 
#  For example, consider the 1-D heat equation:
# 
# *Why care about a heat equation?*
# 
# The Black-Scholes PDE is often referred to as a second-order parabolic PDE likened to a heat equation from physics!*
# 
#  $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
# 
#  where:
#  - $u(x,t)$ is the temperature at position $x$ and time $t$
#  - $\alpha$ is the thermal diffusivity coefficient 
#  - $\frac{\partial u}{\partial t}$ represents the rate of change of temperature with respect to time
#  - $\frac{\partial^2 u}{\partial x^2}$ represents the second spatial derivative (curvature) of temperature
# 
#  This PDE describes how heat diffuses through a one-dimensional medium over time.
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x and t values for 3D surface
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 200)  # Increased number of points while keeping same interval
X, T = np.meshgrid(x, t)

# Calculate u(x,t) = e^(-π²t) * sin(πx)
Z = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'surface'}]]
)

# Add surface plot
fig.add_trace(
    go.Surface(
        x=X,
        y=T,
        z=Z,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ),
    row=1, col=1
)

# Update layout
fig.update_layout(
    title='Heat Equation Solution Surface Plot',
    width=1000,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    scene=dict(
        xaxis_title='x - Position',
        yaxis_title='t - Time',
        zaxis_title='u(x,t) - Temperature',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    )
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 🔎 Example: Step-by-Step Explicit Finite Differences for Heat Equation
# 
# Let's solve the heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ using explicit finite differences.
# 
# **Step 1:** 
# 
# Start with our heat equation
# $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
# 
# First, we discretize space and time:
# - Space: $x_i = i\Delta x$ for $i = 0,1,...,N$
# - Time: $t_j = j\Delta t$ for $j = 0,1,...,M$
# - Let $u^j_i$ represent $u(x_i,t_j)$
# 
# **Step 2:** 
# 
# Recall finite difference approximations
# $$\frac{\partial u}{\partial t} \approx \frac{u^{j+1}_i - u^j_i}{\Delta t}$$
# $$\frac{\partial^2 u}{\partial x^2} \approx \frac{u^j_{i+1} - 2u^j_i + u^j_{i-1}}{(\Delta x)^2}$$
# 
# **Step 3:** 
# 
# Plug our approximations into the heat equation
# $$\frac{u^{j+1}_i - u^j_i}{\Delta t} = \alpha \frac{u^j_{i+1} - 2u^j_i + u^j_{i-1}}{(\Delta x)^2}$$
# 
# **Step 4:** 
# 
# Solve for the next time step $u^{j+1}_i$
# $$u^{j+1}_i = u^j_i + \alpha\frac{\Delta t}{(\Delta x)^2}(u^j_{i+1} - 2u^j_i + u^j_{i-1})$$
# 
# **Step 5:** 
# 
# Implementation steps:
# 1. Choose initial condition $u(x,0)$
# 2. Select small step sizes $\Delta x$ and $\Delta t$ that satisfy $\alpha\frac{\Delta t}{(\Delta x)^2} \leq \frac{1}{2}$
# 3. For each time step j:
# - Calculate $u^{j+1}_i$ using our formula for all interior points i
# - Apply boundary conditions
# - Move to next time step
# 
# **Key Idea:** *Stability* refers to if errors grow in time (as we iterate through our solution) - so long as we satisfy the stability requirement $\alpha\frac{\Delta t}{(\Delta x)^2} \leq \frac{1}{2}$ of our discretization we will be ok!


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 💻 Coding the Solution: Finite Differences Partial Differential Equation


alpha = 1.0
L = 1.0
T = 1.0
dx = 0.1
dt = .005

N = int(L / dx) + 1
M = int(T / dt) + 1

if alpha * dt / dx ** 2 > .5:
    print("Stability condition violated")

x = np.linspace(0, L, N)
t = np.linspace(0, T, M)
X, T = np.meshgrid(x, t)

u = np.zeros((M, N))

u[0,:] = np.sin(np.pi * x)
u[:,0] = u[:,-1] = 0

for j in range(0, M-1):
    for i in range(1, N-1):
        update = (alpha * dt / dx ** 2) * (u[j,i+1] - 2*u[j,i] + u[j,i-1])
        if np.isfinite(update):
            u[j+1, i] = u[j,i] + update
        else:
            u[j+1, i] = u[j,i]


u


# ###### ______________________________________________________________________________________________________________________________________
# 


# ##### 🖼️ Visualizing the Numerical and Analytical Solution


# Parameters
alpha = 1.0  # Diffusion coefficient
L = 1.0      # Length of domain
T = 1.0      # Total time
N = int(L/0.1) + 1   # Number of spatial points based on dx=0.1
M = int(T/0.005) + 1 # Number of time points based on dt=0.005

# Grid spacing
dx = 0.1   # Finer dx for better spatial resolution
dt = 0.005 # Smaller dt to maintain stability

# Stability check
r = alpha * dt / (dx**2)
if r > 0.5:
    print(f"Warning: Scheme may be unstable. r = {r:.3f} > 0.5")

# Initialize grid
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)
X, T = np.meshgrid(x, t)

# Initialize solution array
u_numeric = np.zeros((M, N))

# Initial condition: u(x,0) = sin(πx)
u_numeric[0,:] = np.sin(np.pi * x)

# Boundary conditions: u(0,t) = u(1,t) = 0
u_numeric[:,0] = u_numeric[:,-1] = 0

# Finite difference scheme with type checking to prevent overflow
for j in range(0, M-1):
    for i in range(1, N-1):
        # Calculate the update term separately with bounds checking
        update = r * (u_numeric[j,i+1] - 2*u_numeric[j,i] + u_numeric[j,i-1])
        if np.isfinite(update):  # Only update if result is finite
            u_numeric[j+1,i] = u_numeric[j,i] + update
        else:
            u_numeric[j+1,i] = u_numeric[j,i]  # Maintain previous value if overflow occurs

# Calculate analytical solution using same grid
u_analytic = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)

# Create figure with two subplots side by side
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=('Numerical Solution', 'Analytical Solution')
)

# Add numerical solution surface plot
fig.add_trace(
    go.Surface(
        x=X,
        y=T,
        z=u_numeric,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ),
    row=1, col=1
)

# Add analytical solution surface plot
fig.add_trace(
    go.Surface(
        x=X,
        y=T,
        z=u_analytic,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='Heat Equation Solutions: Numerical vs Analytical',
    width=1500,  # Increased width to accommodate two plots
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    scene=dict(
        xaxis_title='x - Position',
        yaxis_title='t - Time',
        zaxis_title='u(x,t) - Temperature',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    scene2=dict(
        xaxis_title='x - Position',
        yaxis_title='t - Time', 
        zaxis_title='u(x,t) - Temperature',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    )
)



# Create 3D animation showing how the numerical solution evolves over time
frames = []

# Create base figure with 3D axes
fig = go.Figure()

# Add frames for each time step
for i in range(0, M, 10):  # Sample every 10th time step
    # Create meshgrid for current time slice
    X_slice, T_slice = np.meshgrid(x, t[:i+1])
    
    frames.append(
        go.Frame(
            data=[
                go.Surface(
                    x=X_slice,
                    y=T_slice,
                    z=u_numeric[:i+1,:],
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,
                    name='Numerical Solution'
                ),
                go.Surface(
                    x=X_slice,
                    y=T_slice,
                    z=u_analytic[:i+1,:],
                    colorscale='Plasma',
                    opacity=0.4,
                    showscale=False,
                    name='Analytical Solution'
                )
            ],
            name=f't={t[i]:.3f}'
        )
    )

# Add initial surface
fig.add_trace(
    go.Surface(
        x=np.meshgrid(x, t[:1])[0],
        y=np.meshgrid(x, t[:1])[1],
        z=u_numeric[:1,:],
        colorscale='Viridis',
        opacity=0.8,
        showscale=True,
        name='Numerical Solution'
    )
)

fig.add_trace(
    go.Surface(
        x=np.meshgrid(x, t[:1])[0],
        y=np.meshgrid(x, t[:1])[1],
        z=u_analytic[:1,:],
        colorscale='Plasma',
        opacity=0.4,
        showscale=False,
        name='Analytical Solution'
    )
)

# Update layout
fig.update_layout(
    title='Evolution of Heat Equation Solution in 3D',
    width=1000,
    height=800,
    scene=dict(
        xaxis_title='Position (x)',
        yaxis_title='Time (t)',
        zaxis_title='Temperature u(x,t)',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            up=dict(x=0, y=0, z=1)
        )
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Add frames and animation settings
fig.frames = frames
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=100, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate',
                        transition=dict(duration=0)
                    )]
                )
            ],
            x=0.1,
            y=0
        )
    ]
)

fig.show()



# ---


# ##### 4.) 📝 Approximating the Black-Scholes Partial Differential Equation
# 
# Now instead of pretending we are using a model framework like earlier, the Black-Scholes model framework actually gives us a PDE...
# 
#  The Black-Scholes PDE:
#  
#  $\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$
#  
#  where:
#  - $V$ is the option value
#  - $t$ is time
#  - $S$ is the stock price
#  - $\sigma$ is volatility
#  - $r$ is the risk-free rate
# 
#  
#  **Step 1:**
#  
#  First, we discretize space and time:
#  - Stock price: $S_i = i\Delta S$ for $i = 0,1,...,M$ where $\Delta S = S_{max}/M$
#  - Time: $t_j = j\Delta t$ for $j = 0,1,...,N$ where $\Delta t = T/N$
#  - Let $V^j_i$ represent $V(S_i,t_j)$
#  
#  **Step 2:**
#  
#  Recall finite difference approximations:
#  $$\frac{\partial V}{\partial t} \approx \frac{V^{j+1}_i - V^j_i}{\Delta t}$$
#  $$\frac{\partial V}{\partial S} \approx \frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S}$$
#  $$\frac{\partial^2 V}{\partial S^2} \approx \frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2}$$
#  
#  **Step 3:**
#  
#  Plug our approximations into the Black-Scholes PDE:
#  $$\frac{V^{j+1}_i - V^j_i}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2} + rS_i\frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S} - rV^j_i = 0$$
#  
#  **Step 4:**
#  
#  Solve for the next time step $V^{j+1}_i$:
#  $$V^{j+1}_i = V^j_i + \Delta t(\frac{1}{2}\sigma^2S_i^2\frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2} + rS_i\frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S} - rV^j_i)$$
#  
#  **Step 5:**
#  
#  Implementation steps:
#  1. Set terminal condition $V(S,T)$ (option payoff at expiry)
#  2. Select small step sizes $\Delta S$ and $\Delta t$ for stability
#  3. For each time step j (working backwards):
#     - Calculate $V^{j}_i$ using our formula for all interior points i
#     - Apply boundary conditions at $S = 0$ and $S = S_{max}$
#     - Move to previous time step


# Parameters
S_max_1 = 150  # First max stock price
S_max_2 = 2000  # Second max stock price
K = 100  # Strike price
T = 1.0  # Time to maturity
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
M = 100  # Number of stock price steps
N = 1000  # Number of time steps

def finite_difference_bs(S_max):
    # Grid parameters
    dt = T/N
    dS = S_max/M
    
    # Initialize grid
    S = np.linspace(0, S_max, M+1)
    t = np.linspace(0, T, N+1)
    V = np.zeros((M+1, N+1))
    
    # Terminal condition (payoff at expiry)
    V[:,-1] = np.maximum(S - K, 0)
    
    # Coefficients for explicit scheme
    a = 0.5*dt*(sigma**2*S**2/dS**2 - r*S/dS)
    b = 1 - dt*(sigma**2*S**2/dS**2 + r)
    c = 0.5*dt*(sigma**2*S**2/dS**2 + r*S/dS)
    
    # Solve backwards in time
    for j in range(N-1, -1, -1):
        for i in range(1, M):
            V[i,j] = a[i]*V[i-1,j+1] + b[i]*V[i,j+1] + c[i]*V[i+1,j+1]
    
    return V[:,0], S

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Calculate solutions
from scipy.stats import norm
V1, S1 = finite_difference_bs(S_max_1)
V2, S2 = finite_difference_bs(S_max_2)
S_analytical = np.linspace(0, S_max_2, 1000)
V_analytical = black_scholes(S_analytical, K, T, r, sigma)

# Create subplots
fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=('Finite Differences (S_max=150)', 
                                  'Finite Differences (S_max=2000)',
                                  'Analytical Solution'))

# Plot finite differences solution (S_max=150)
fig.add_trace(
    go.Scatter(
        x=S1,
        y=V1,
        mode='lines',
        line=dict(color='rgb(255, 165, 0)', width=2),
        name='FD (S_max=150)'
    ),
    row=1, col=1
)

# Plot finite differences solution (S_max=2000)
fig.add_trace(
    go.Scatter(
        x=S2,
        y=V2,
        mode='lines',
        line=dict(color='rgb(0, 191, 255)', width=2),
        name='FD (S_max=2000)'
    ),
    row=1, col=2
)

# Plot analytical solution
fig.add_trace(
    go.Scatter(
        x=S_analytical,
        y=V_analytical,
        mode='lines',
        line=dict(color='rgb(57, 255, 20)', width=2),
        name='Analytical'
    ),
    row=1, col=3
)

# Update layout
fig.update_layout(
    width=1500,
    height=500,
    title_text='Black-Scholes Option Pricing: Finite Differences vs Analytical',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Update axes
for i in [1, 2, 3]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='Stock Price',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text='Option Value',
        row=1, col=i
    )
    
    # Set x-axis range for second and third plots
    if i in [2, 3]:
        fig.update_xaxes(range=[70, 150], row=1, col=i)
        fig.update_yaxes(range=[0, 100], row=1, col=i)

fig.show()



# ---


# ##### 5.) 💭 Closing Thoughts and Future Topics
# 
# 
# In quantitative finance we construct pricing arguments under a model framework (*Black-Scholes, Heston, ...*) that yields a pricing partial differential equation.  
# 
# The definition of a derivative is extremely useful from approximating solutions (the pricing functions given a models framework) to these differential equations.
# 
# Different payoff structures will change the boundary conditions of the solutions to the pricing differential equations!
# 
# It should be noted that there are other ways to go about solving for these pricing functionals via different arguments and techniques (i.e. Monte Carlo Simulation) a topic that I have discussed in great detail on this channel!  
# 
# Future Topics
# 
# - Risk-Neutral Pricing, Change of Measure, Binomial Trees
# 
# - Deriving the Black-Scholes Equation (Delta Hedging)
# 
# - Deriving the Heston Equation (Vega, then Delta Hedging)
# 
# - Coding more Advanced PDE Approximations (*Black-Scholes, Heston, ...*)
# 
# - Schemes, Conditions, Errors (Less Exciting, Extremely Necessary)

