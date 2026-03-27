## Finite Differences Option Pricing for Quant Finance

##### ▶️ Related Quant Guild Videos:

*Necessary Stochastic Calculus:*
-  [Itô's Lemma Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo)

-  [Itô Integration Clearly and Visually Explained](https://youtu.be/dUvZ8m3QpeI)

- [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)

*Why the Expectation is Sufficient for Pricing and Problems in Practice:*
- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)

- [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)

*Applications in Market-Making and Trading:*

- [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4)

- [Trading with the Black-Scholes Implied Volatility Surface](https://youtu.be/YH0tWpBaKGs)

- [How to Price Exotic Options](https://youtu.be/hsot26myYYM)

###### ______________________________________________________________________________________________________________________________________

 
##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)

##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)

---


```python
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

```



---

### 📖 Sections

#### 1.) 🎯 Differential Equations in Quant Finance

- Pricing Differential Equations

- Analytical v. Numerical Solutions

#### 2.) 🌊 Approximating Ordinary Differential Equations 

- Definition of a derivative

- Approximating Derivatives using the Definition

- Euler's Method and Finite Differences

- Numerically Solving an Ordinary Differential Equation

- Code Walkthrough and Visualization

#### 3.) 🔥 Approximating Partial Differential Equations

- Visualizing a Partial Differential Equation Solution

- Numerically Solving a Partial Differential Equation

- Code Walkthrough and Visualization

#### 4.) 📝 Approximating the Black-Scholes Partial Differential Equation

- Visualizing a Finite Differences Approximation in a Black-Scholes Framework

#### 5.) 💭 Closing Thoughts and Future Topics

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



##### 1.) 🎯 Differential Equations in Quant Finance

Where do differential equations fit into quantitative finance?  

Well, option pricing arguments can be constructed in terms of continuous hedging which yields a differential equation

Where does the randomness *go*?  Well, randomness from the model's framework is cancelled out by hedging arguments!

**The Black-Scholes partial differential equation:**
 
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$
 

**The Heston partial differential equation:**
 
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + \rho\sigma\nu S\frac{\partial^2 V}{\partial S\partial \nu} + \frac{1}{2}\nu\frac{\partial^2 V}{\partial \nu^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-\nu)\frac{\partial V}{\partial \nu} - rV = 0$$
 

When that differential equation is solved we get the function that gives an option price in that model framework

These arguments can be constructed for vanilla and exotic options - we will focus on European options herein as they are not path dependent

##### 🔍 Example: Analytical and Numerical Solutions to Pricing Differential Equations

Let's begin by focusing on the differential equation and what it's solution gives us 

Suppose we've constructed an option pricing argument (*Black-Scholes, Heston,* ...) that gives us the following differential equation
 
 $$\frac{dy}{dx} = y' = f'(x) \quad \quad f'(x) = \frac{1}{2}x^{-1/2} \quad \quad f(0) = 1$$

 Here the *model framework* only requires stock price for the corresponding option price, real-world models demand more inputs...

 The function we are after $f(x) = ?$ will give us the price of the option by plugging in the current stock price $S_t$ for $x$

 In this case, we can *analytically* solve the equation and get a function that perfectly maps inputs 


```python
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

```



Solving for the equation analytically yields

$$\int f'(x) = \int \frac{1}{2}x^{- \frac{1}{2}} \implies f(x) = x^{\frac{1}{2}} + C$$
$$f(0) = 1 \implies f(0) = 0^{\frac{1}{2}} + C = 1 \implies C = 1$$

Thus, the option pricing function is given by $f(x) = x^{\frac{1}{2}} + 1$ which gives the price of our option given a stock price $x$

Under the assumed model framework that produces the differential equation, this function gives the *correct* price *everywhere*

Any stock price I give the pricing function will return the valid model price because this is the *analytical* or *closed-form* solution


```python
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

```



Analytical solutions are preferred but not always possible.

Why do we use other models if they don't have analytical solutions?  

Some models capture necessary dynamics of the market other models don't have the ability to capture 

For example, constant v.s. stochastic volatility and the dynamics of the implied volatility surface

So what can we do if there is no analytical solution available? 

We have a variety of techniques that are sufficient to approximate solutions to the pricing partial differential equations!

---

##### 2.) 🌊 Approximating Differential Equations 

When analytical solutions are unavailable we can approximate differential equations numerically

Computers are particularly effective for this!  Let's understand *why* approximations work by looking at a method for ordinary differential equations

##### 📐 Definition of a Derivative

Assuming a function is differentiable, the derivative is defined as

$$f'(x) = lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$$

We learn it as the slope of the tangent line, and we take for granted the definition - in fact many students have a distaste for it


```python
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

```



Math professors and academics will have calculus students to apply this limit definition to find derivatives and prove rules...

**Proving the Power Rule:**

 $$\begin{align}
 \frac{d}{dx}x^n &= \lim_{h \to 0} \frac{(x+h)^n - x^n}{h} \\
 &= \lim_{h \to 0} \frac{\sum_{k=0}^n \binom{n}{k}x^{n-k}h^k - x^n}{h} \\
 &= \lim_{h \to 0} \frac{x^n + nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + ... - x^n}{h} \\
 &= \lim_{h \to 0} \left(nx^{n-1} + \frac{n(n-1)}{2}x^{n-2}h + ...\right) \\
 &= nx^{n-1}
 \end{align}$$

**Using the Power Rule:**

$f(x) = x^2 \implies f'(x) = 2x$


How does this get students stoked about the definition of a derivative? This is a terrible approach!  

First we should appreciate the limit definition and how *easy* it is to use!

**Approximating Derivatives:**

Assuming f(x) is differentiable, we can approximate derivatives and therefore diffrential equations

Instead of infinitely small we can just pick a *very smamll* value for *h*

$$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \quad \text{ when } h \text{ is small}$$

This is **so** easy, in the following code I'll take the derivative of a function approximately (numerically) and analytically


```python
# Function
def f(x):
    return x**2

# Actual Derivative
def f_prime(x):
    return 2*x

h = .0000001

print("Approximation:", (f(1 + h) - f(1))/h)
print("Actual:", f_prime(1))
```

    Approximation: 2.0000001010878066
    Actual: 2


#####  💭 Big Idea

**What if we rearrange the approximate equation?**

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

$$\implies hf'(x) \approx f(x+h) - f(x)$$

$$\implies f(x+h) \approx f(x) + hf'(x)$$

We have now expressed the *original* function *approximately* in terms of its derivative!

That means we can solve for the original function iteratively if we are given a differential equation!

We just derived *Euler's method* and *Finite Differences* for approximating ODEs!


```python
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

```



**Remark:** One of the consequences of using numerical methods is that approximations for values outside our *explored* space are poor - we need to iteratively approximate the space of interest to get accurate approximations!

###### ______________________________________________________________________________________________________________________________________


##### Finite Differences:

This is *all* the finite differences method for approximating solutions to differential equations is!

We are approximating the derivative, rearranging the equation, and iteratively solving for the original function!

 **Forward Difference:**
$$\frac{\partial f}{\partial x} \approx \frac{f(x + \Delta x) - f(x)}{\Delta x} + O(\Delta x)$$

 **Backward Difference:**
 $$\frac{\partial f}{\partial x} \approx \frac{f(x) - f(x - \Delta x)}{\Delta x} + O(\Delta x)$$

 **Central Difference:**
 $$\frac{\partial f}{\partial x} \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2\Delta x} + O(\Delta x^2)$$

 **Second Derivative (Central):**
 $$\frac{\partial^2 f}{\partial x^2} \approx \frac{f(x + \Delta x) - 2f(x) + f(x - \Delta x)}{\Delta x^2} + O(\Delta x^2)$$

 Where $O(\Delta x)$ and $O(\Delta x^2)$ represent the order of accuracy of the approximation.

**Remark:** This is why we should be stoked to see derivatives or differential equations in any capacity - they are very easily solved even if we have to resort to techniques to approximate the solution!

###### ______________________________________________________________________________________________________________________________________


##### 🔎 Example Euler's Method (Forward Finite-Difference Scheme)
 
Consider the pricing differential equation from the start of this notebook.

 Let's solve the following differential equation using finite differences:
 
  $$\frac{dy}{dx} = y' = f'(x) \quad \quad f'(x) = \frac{1}{2}x^{-1/2} \quad \quad f(0) = 1$$
 
Let's solve this differential equation step by step using Euler's method:

**Step 1:** 

Start with our differential equation
$$\frac{dy}{dx} = \frac{1}{2}x^{-1/2}$$

First, we discretize space:
- Space: $x_n = nh$ for $n = 0,1,...,N$ 
- Let $y_n$ represent $y(x_n)$

**Step 2:** 

Recall finite difference approximation
$$\frac{dy}{dx} \approx \frac{y_{n+1} - y_n}{h}$$

**Step 3:** 

Plug our approximation into the differential equation
$$\frac{y_{n+1} - y_n}{h} = \frac{1}{2}x_n^{-1/2}$$

**Step 4:** 

Solve for the next step $y_{n+1}$
$$y_{n+1} = y_n + h(\frac{1}{2}x_n^{-1/2})$$

**Step 5:** 

Implementation steps:
1. Choose initial condition $y_0 = 1$
2. Select small step size $h$
3. For each step n:
    - Calculate $x_n = nh$
    - Calculate $y_{n+1}$ using our formula
    - Move to next step

###### ______________________________________________________________________________________________________________________________________


##### 💻 Coding the Solution: Finite Differences Ordinary Differential Equation


```python
x = np.linspace(0.0001, 5, 2000)

f_prime = lambda x: .5 * x **(-.5)

x_d = [0]
y_hat = [1]

for i in range(1, len(x)):
    x_d.append(x[i])
    y_hat.append(y_hat[-1] + f_prime(x[i]) * (x[i] - x[i-1]))
```


```python
y_hat
```




    [1,
     np.float64(1.0245206289628248),
     np.float64(1.042028414841836),
     np.float64(1.0563703860443832),
     np.float64(1.068811366731748),
     np.float64(1.079949946656656),
     np.float64(1.0901247608570048),
     np.float64(1.0995492654216739),
     np.float64(1.1083682129914876),
     np.float64(1.1166850944681765),
     np.float64(1.1245769263561),
     np.float64(1.1321028551926628),
     np.float64(1.1393094709888347),
     np.float64(1.1462342479601837),
     np.float64(1.1529078608222554),
     np.float64(1.1593557954471847),
     np.float64(1.1655995000001629),
     np.float64(1.171657227259242),
     np.float64(1.177544663691017),
     np.float64(1.1832754077574348),
     np.float64(1.1888613393891763),
     np.float64(1.1943129094385287),
     np.float64(1.1996393693229874),
     np.float64(1.2048489553022241),
     np.float64(1.2099490378824387),
     np.float64(1.2149462440889247),
     np.float64(1.2198465583956544),
     np.float64(1.2246554066954631),
     np.float64(1.2293777266686343),
     np.float64(1.2340180271492556),
     np.float64(1.2385804385213313),
     np.float64(1.2430687557475046),
     np.float64(1.2474864753054),
     np.float64(1.251836827053769),
     np.float64(1.2561228018539286),
     np.float64(1.2603471756177103),
     np.float64(1.2645125303311957),
     np.float64(1.2686212725064514),
     np.float64(1.2726756494356706),
     np.float64(1.276677763559377),
     np.float64(1.280629585209424),
     np.float64(1.2845329639459668),
     np.float64(1.2883896386734832),
     np.float64(1.2922012466928054),
     np.float64(1.2959693318228185),
     np.float64(1.2996953517060823),
     np.float64(1.3033806843964117),
     np.float64(1.3070266343128318),
     np.float64(1.3106344376328363),
     np.float64(1.314205267188165),
     np.float64(1.3177402369180597),
     np.float64(1.3212404059279246),
     np.float64(1.3247067821953),
     np.float64(1.3281403259598996),
     np.float64(1.3315419528300174),
     np.float64(1.3349125366337802),
     np.float64(1.3382529120404008),
     np.float64(1.3415638769737102),
     np.float64(1.3448461948377384),
     np.float64(1.3481005965719297),
     np.float64(1.3513277825516647),
     np.float64(1.3545284243480875),
     np.float64(1.3577031663597594),
     np.float64(1.3608526273273713),
     np.float64(1.3639774017415944),
     np.float64(1.3670780611531435),
     np.float64(1.3701551553932272),
     np.float64(1.3732092137117669),
     np.float64(1.3762407458400527),
     np.float64(1.3792502429838833),
     np.float64(1.382238178752664),
     np.float64(1.3852050100294417),
     np.float64(1.3881511777864008),
     np.float64(1.3910771078499413),
     np.float64(1.3939832116190982),
     np.float64(1.396869886740732),
     np.float64(1.3997375177446287),
     np.float64(1.4025864766413807),
     np.float64(1.4054171234856798),
     np.float64(1.408229806907438),
     np.float64(1.4110248646129515),
     np.float64(1.41380262385815),
     np.float64(1.4165634018958075),
     np.float64(1.4193075063984437),
     np.float64(1.4220352358585122),
     np.float64(1.4247468799673506),
     np.float64(1.4274427199742508),
     np.float64(1.4301230290269127),
     np.float64(1.4327880724944448),
     np.float64(1.4354381082739958),
     np.float64(1.438073387082016),
     np.float64(1.4406941527310844),
     np.float64(1.443300642393163),
     np.float64(1.4458930868500826),
     np.float64(1.4484717107320124),
     np.float64(1.4510367327446068),
     np.float64(1.4535883658854805),
     np.float64(1.456126817650621),
     np.float64(1.458652290231301),
     np.float64(1.4611649807020204),
     np.float64(1.4636650811999745),
     np.float64(1.4661527790965083),
     np.float64(1.468628257160991),
     np.float64(1.4710916937175156),
     np.float64(1.4735432627948057),
     np.float64(1.4759831342696834),
     np.float64(1.4784114740044354),
     np.float64(1.4808284439783894),
     np.float64(1.4832342024139988),
     np.float64(1.48562890389771),
     np.float64(1.4880126994958776),
     np.float64(1.4903857368659699),
     np.float64(1.4927481603632988),
     np.float64(1.495100111143491),
     np.float64(1.4974417272609055),
     np.float64(1.499773143763196),
     np.float64(1.5020944927821946),
     np.float64(1.5044059036212964),
     np.float64(1.506707502839504),
     np.float64(1.508999414332289),
     np.float64(1.5112817594094148),
     np.float64(1.5135546568698623),
     np.float64(1.5158182230739845),
     np.float64(1.5180725720130184),
     np.float64(1.520317815376069),
     np.float64(1.522554062614679),
     np.float64(1.5247814210050872),
     np.float64(1.5269999957082783),
     np.float64(1.5292098898279172),
     np.float64(1.5314112044662596),
     np.float64(1.533604038778124),
     np.float64(1.5357884900230068),
     np.float64(1.5379646536154168),
     np.float64(1.5401326231735057),
     np.float64(1.5422924905660618),
     np.float64(1.5444443459579347),
     np.float64(1.5465882778539557),
     np.float64(1.5487243731414115),
     np.float64(1.5508527171311326),
     np.float64(1.5529733935972485),
     np.float64(1.5550864848156634),
     np.float64(1.5571920716013015),
     np.float64(1.5592902333441716),
     np.float64(1.5613810480442936),
     np.float64(1.563464592345533),
     np.float64(1.5655409415683832),
     np.float64(1.5676101697417362),
     np.float64(1.5696723496336795),
     np.float64(1.5717275527813546),
     np.float64(1.5737758495199135),
     np.float64(1.575817309010604),
     np.float64(1.5778519992680184),
     np.float64(1.5798799871865326),
     np.float64(1.581901338565968),
     np.float64(1.5839161181365013),
     np.float64(1.5859243895828516),
     np.float64(1.5879262155677674),
     np.float64(1.58992165775484),
     np.float64(1.5919107768306668),
     np.float64(1.5938936325263857),
     np.float64(1.5958702836386034),
     np.float64(1.5978407880497383),
     np.float64(1.599805202747797),
     np.float64(1.6017635838456041),
     np.float64(1.6037159865995048),
     np.float64(1.605662465427555),
     np.float64(1.6076030739272196),
     np.float64(1.6095378648925907),
     np.float64(1.6114668903311473),
     np.float64(1.6133902014800658),
     np.float64(1.6153078488220984),
     np.float64(1.6172198821010344),
     np.float64(1.6191263503367554),
     np.float64(1.6210273018398984),
     np.float64(1.6229227842261402),
     np.float64(1.6248128444301126),
     np.float64(1.6266975287189633),
     np.float64(1.6285768827055693),
     np.float64(1.6304509513614176),
     np.float64(1.6323197790291613),
     np.float64(1.6341834094348608),
     np.float64(1.6360418856999217),
     np.float64(1.6378952503527349),
     np.float64(1.6397435453400326),
     np.float64(1.641586812037965),
     np.float64(1.6434250912629071),
     np.float64(1.6452584232820042),
     np.float64(1.6470868478234633),
     np.float64(1.6489104040865974),
     np.float64(1.6507291307516316),
     np.float64(1.6525430659892753),
     np.float64(1.654352247470069),
     np.float64(1.6561567123735124),
     np.float64(1.6579564973969791),
     np.float64(1.659751638764424),
     np.float64(1.6615421722348898),
     np.float64(1.6633281331108176),
     np.float64(1.6651095562461686),
     np.float64(1.6668864760543598),
     np.float64(1.6686589265160203),
     np.float64(1.6704269411865733),
     np.float64(1.6721905532036478),
     np.float64(1.6739497952943254),
     np.float64(1.6757046997822262),
     np.float64(1.6774552985944375),
     np.float64(1.6792016232682916),
     np.float64(1.6809437049579938),
     np.float64(1.682681574441108),
     np.float64(1.6844152621248998),
     np.float64(1.6861447980525446),
     np.float64(1.687870211909202),
     np.float64(1.6895915330279596),
     np.float64(1.6913087903956525),
     np.float64(1.6930220126585571),
     np.float64(1.6947312281279674),
     np.float64(1.6964364647856536),
     np.float64(1.6981377502892065),
     np.float64(1.6998351119772714),
     np.float64(1.7015285768746737),
     np.float64(1.703218171697439),
     np.float64(1.7049039228577112),
     np.float64(1.7065858564685683),
     np.float64(1.7082639983487435),
     np.float64(1.7099383740272485),
     np.float64(1.7116090087479043),
     np.float64(1.713275927473783),
     np.float64(1.7149391548915578),
     np.float64(1.7165987154157705),
     np.float64(1.7182546331930117),
     np.float64(1.7199069321060205),
     np.float64(1.7215556357777024),
     np.float64(1.7232007675750711),
     np.float64(1.7248423506131114),
     np.float64(1.7264804077585687),
     np.float64(1.7281149616336657),
     np.float64(1.729746034619747),
     np.float64(1.7313736488608553),
     np.float64(1.732997826267238),
     np.float64(1.7346185885187895),
     np.float64(1.7362359570684267),
     np.float64(1.7378499531454024),
     np.float64(1.7394605977585562),
     np.float64(1.7410679116995054),
     np.float64(1.742671915545776),
     np.float64(1.7442726296638766),
     np.float64(1.7458700742123157),
     np.float64(1.747464269144563),
     np.float64(1.7490552342119579),
     np.float64(1.750642988966564),
     np.float64(1.7522275527639735),
     np.float64(1.7538089447660588),
     np.float64(1.755387183943677),
     np.float64(1.7569622890793248),
     np.float64(1.7585342787697469),
     np.float64(1.7601031714284965),
     np.float64(1.761668985288453),
     np.float64(1.7632317384042941),
     np.float64(1.7647914486549237),
     np.float64(1.766348133745859),
     np.float64(1.767901811211576),
     np.float64(1.7694524984178128),
     np.float64(1.7710002125638349),
     np.float64(1.7725449706846603),
     np.float64(1.7740867896532482),
     np.float64(1.7756256861826478),
     np.float64(1.7771616768281129),
     np.float64(1.7786947779891793),
     np.float64(1.7802250059117082),
     np.float64(1.7817523766898942),
     np.float64(1.7832769062682405),
     np.float64(1.7847986104435007),
     np.float64(1.786317504866589),
     np.float64(1.787833605044457),
     np.float64(1.7893469263419426),
     np.float64(1.7908574839835851),
     np.float64(1.7923652930554133),
     np.float64(1.7938703685067023),
     np.float64(1.795372725151704),
     np.float64(1.7968723776713469),
     np.float64(1.7983693406149113),
     np.float64(1.799863628401675),
     np.float64(1.8013552553225343),
     np.float64(1.8028442355415988),
     np.float64(1.8043305830977598),
     np.float64(1.8058143119062358),
     np.float64(1.8072954357600908),
     np.float64(1.808773968331732),
     np.float64(1.8102499231743816),
     np.float64(1.8117233137235256),
     np.float64(1.8131941532983418),
     np.float64(1.814662455103104),
     np.float64(1.8161282322285652),
     np.float64(1.817591497653319),
     np.float64(1.8190522642451405),
     np.float64(1.820510544762307),
     np.float64(1.8219663518548965),
     np.float64(1.8234196980660704),
     np.float64(1.8248705958333324),
     np.float64(1.826319057489771),
     np.float64(1.8277650952652824),
     np.float64(1.8292087212877757),
     np.float64(1.8306499475843594),
     np.float64(1.8320887860825101),
     np.float64(1.833525248611225),
     np.float64(1.8349593469021552),
     np.float64(1.836391092590725),
     np.float64(1.8378204972172323),
     np.float64(1.839247572227934),
     np.float64(1.8406723289761158),
     np.float64(1.8420947787231456),
     np.float64(1.8435149326395124),
     np.float64(1.8449328018058488),
     np.float64(1.8463483972139412),
     np.float64(1.8477617297677222),
     np.float64(1.8491728102842522),
     np.float64(1.8505816494946838),
     np.float64(1.8519882580452154),
     np.float64(1.8533926464980284),
     np.float64(1.8547948253322135),
     np.float64(1.8561948049446828),
     np.float64(1.8575925956510686),
     np.float64(1.858988207686611),
     np.float64(1.8603816512070321),
     np.float64(1.861772936289398),
     np.float64(1.8631620729329685),
     np.float64(1.8645490710600372),
     np.float64(1.8659339405167565),
     np.float64(1.8673166910739545),
     np.float64(1.8686973324279383),
     np.float64(1.870075874201288),
     np.float64(1.8714523259436384),
     np.float64(1.8728266971324519),
     np.float64(1.8741989971737785),
     np.float64(1.8755692354030082),
     np.float64(1.876937421085611),
     np.float64(1.8783035634178677),
     np.float64(1.8796676715275924),
     np.float64(1.8810297544748422),
     np.float64(1.882389821252621),
     np.float64(1.883747880787571),
     np.float64(1.8851039419406566),
     np.float64(1.886458013507839),
     np.float64(1.8878101042207422),
     np.float64(1.8891602227473094),
     np.float64(1.8905083776924512),
     np.float64(1.891854577598686),
     np.float64(1.8931988309467709),
     np.float64(1.894541146156326),
     np.float64(1.8958815315864486),
     np.float64(1.8972199955363207),
     np.float64(1.8985565462458092),
     np.float64(1.8998911918960573),
     np.float64(1.9012239406100684),
     np.float64(1.9025548004532835),
     np.float64(1.90388377943415),
     np.float64(1.9052108855046852),
     np.float64(1.9065361265610294),
     np.float64(1.9078595104439962),
     np.float64(1.9091810449396118),
     np.float64(1.9105007377796515),
     np.float64(1.911818596642165),
     np.float64(1.9131346291520002),
     np.float64(1.9144488428813153),
     np.float64(1.9157612453500894),
     np.float64(1.9170718440266232),
     np.float64(1.918380646328035),
     np.float64(1.9196876596207508),
     np.float64(1.9209928912209875),
     np.float64(1.9222963483952313),
     np.float64(1.9235980383607094),
     np.float64(1.924897968285856),
     np.float64(1.926196145290773),
     np.float64(1.9274925764476853),
     np.float64(1.92878726878139),
     np.float64(1.9300802292697006),
     np.float64(1.9313714648438856),
     np.float64(1.9326609823891021),
     np.float64(1.9339487887448246),
     np.float64(1.9352348907052674),
     np.float64(1.9365192950198031),
     np.float64(1.9378020083933762),
     np.float64(1.9390830374869112),
     np.float64(1.940362388917716),
     np.float64(1.9416400692598805),
     np.float64(1.9429160850446718),
     np.float64(1.9441904427609227),
     np.float64(1.9454631488554166),
     np.float64(1.946734209733269),
     np.float64(1.948003631758302),
     np.float64(1.9492714212534181),
     np.float64(1.9505375845009658),
     np.float64(1.9518021277431035),
     np.float64(1.9530650571821588),
     np.float64(1.954326378980983),
     np.float64(1.9555860992633023),
     np.float64(1.9568442241140642),
     np.float64(1.9581007595797808),
     np.float64(1.959355711668868),
     np.float64(1.96060908635198),
     np.float64(1.9618608895623417),
     np.float64(1.9631111271960753),
     np.float64(1.9643598051125248),
     np.float64(1.9656069291345772),
     np.float64(1.9668525050489776),
     np.float64(1.9680965386066436),
     np.float64(1.9693390355229754),
     np.float64(1.9705800014781611),
     np.float64(1.9718194421174802),
     np.float64(1.9730573630516033),
     np.float64(1.9742937698568879),
     np.float64(1.9755286680756727),
     np.float64(1.976762063216566),
     np.float64(1.9779939607547328),
     np.float64(1.9792243661321793),
     np.float64(1.980453284758032),
     np.float64(1.9816807220088162),
     np.float64(1.9829066832287296),
     np.float64(1.9841311737299145),
     np.float64(1.985354198792726),
     np.float64(1.986575763665997),
     np.float64(1.987795873567303),
     np.float64(1.9890145336832197),
     np.float64(1.9902317491695816),
     np.float64(1.9914475251517363),
     np.float64(1.9926618667247962),
     np.float64(1.9938747789538875),
     np.float64(1.9950862668743972),
     np.float64(1.9962963354922159),
     np.float64(1.9975049897839798),
     np.float64(1.9987122346973105),
     np.float64(1.9999180751510492),
     np.float64(2.001122516035492),
     np.float64(2.0023255622126217),
     np.float64(2.003527218516334),
     np.float64(2.004727489752668),
     np.float64(2.0059263807000276),
     np.float64(2.0071238961094044),
     np.float64(2.008320040704598),
     np.float64(2.009514819182431),
     np.float64(2.010708236212968),
     np.float64(2.0119002964397255),
     np.float64(2.0130910044798833),
     np.float64(2.0142803649244945),
     np.float64(2.01546838233869),
     np.float64(2.016655061261884),
     np.float64(2.017840406207977),
     np.float64(2.0190244216655553),
     np.float64(2.020207112098089),
     np.float64(2.021388481944129),
     np.float64(2.0225685356175007),
     np.float64(2.0237472775074976),
     np.float64(2.02492471197907),
     np.float64(2.0261008433730154),
     np.float64(2.027275676006164),
     np.float64(2.0284492141715633),
     np.float64(2.0296214621386626),
     np.float64(2.0307924241534927),
     np.float64(2.031962104438847),
     np.float64(2.0331305071944565),
     np.float64(2.034297636597169),
     np.float64(2.03546349680112),
     np.float64(2.0366280919379083),
     np.float64(2.0377914261167644),
     np.float64(2.0389535034247217),
     np.float64(2.040114327926782),
     np.float64(2.041273903666083),
     np.float64(2.0424322346640613),
     np.float64(2.043589324920616),
     np.float64(2.0447451784142703),
     np.float64(2.045899799102329),
     np.float64(2.0470531909210394),
     np.float64(2.0482053577857458),
     np.float64(2.049356303591045),
     np.float64(2.050506032210941),
     np.float64(2.0516545474989947),
     np.float64(2.0528018532884773),
     np.float64(2.0539479533925182),
     np.float64(2.0550928516042526),
     np.float64(2.056236551696969),
     np.float64(2.0573790574242534),
     np.float64(2.0585203725201326),
     np.float64(2.0596605006992186),
     np.float64(2.0607994456568473),
     np.float64(2.0619372110692202),
     np.float64(2.063073800593541),
     np.float64(2.0642092178681537),
     np.float64(2.065343466512679),
     np.float64(2.066476550128148),
     np.float64(2.0676084722971364),
     np.float64(2.0687392365838955),
     np.float64(2.069868846534484),
     np.float64(2.0709973056768978),
     np.float64(2.0721246175211983),
     np.float64(2.07325078555964),
     np.float64(2.074375813266796),
     np.float64(2.075499704099684),
     np.float64(2.0766224614978896),
     np.float64(2.0777440888836893),
     np.float64(2.0788645896621727),
     np.float64(2.079983967221362),
     np.float64(2.0811022249323328),
     np.float64(2.082219366149331),
     np.float64(2.0833353942098927),
     np.float64(2.084450312434957),
     np.float64(2.085564124128984),
     np.float64(2.086676832580069),
     np.float64(2.087788441060054),
     np.float64(2.0888989528246413),
     np.float64(2.0900083711135045),
     np.float64(2.0911166991503984),
     np.float64(2.092223940143269),
     np.float64(2.0933300972843614),
     np.float64(2.0944351737503264),
     np.float64(2.0955391727023285),
     np.float64(2.0966420972861504),
     np.float64(2.0977439506322977),
     np.float64(2.098844735856102),
     np.float64(2.0999444560578255),
     np.float64(2.1010431143227613),
     np.float64(2.102140713721334),
     np.float64(2.103237257309201),
     np.float64(2.104332748127352),
     np.float64(2.1054271892022056),
     np.float64(2.106520583545709),
     np.float64(2.1076129341554326),
     np.float64(2.108704244014668),
     np.float64(2.109794516092521),
     np.float64(2.1108837533440075),
     np.float64(2.111971958710146),
     np.float64(2.1130591351180503),
     np.float64(2.1141452854810225),
     np.float64(2.1152304126986423),
     np.float64(2.116314519656858),
     np.float64(2.117397609228077),
     np.float64(2.118479684271253),
     np.float64(2.1195607476319758),
     np.float64(2.1206408021425567),
     np.float64(2.1217198506221164),
     np.float64(2.122797895876671),
     np.float64(2.1238749406992152),
     np.float64(2.1249509878698096),
     np.float64(2.1260260401556628),
     np.float64(2.1271001003112135),
     np.float64(2.1281731710782146),
     np.float64(2.1292452551858143),
     np.float64(2.130316355350637),
     np.float64(2.131386474276863),
     np.float64(2.132455614656309),
     np.float64(2.133523779168507),
     np.float64(2.134590970480782),
     np.float64(2.1356571912483306),
     np.float64(2.1367224441142976),
     np.float64(2.137786731709852),
     np.float64(2.1388500566542636),
     np.float64(2.1399124215549783),
     np.float64(2.1409738290076916),
     np.float64(2.1420342815964237),
     np.float64(2.1430937818935925),
     np.float64(2.1441523324600857),
     np.float64(2.145209935845335),
     np.float64(2.1462665945873853),
     np.float64(2.147322311212968),
     np.float64(2.148377088237569),
     np.float64(2.1494309281655006),
     np.float64(2.1504838334899707),
     np.float64(2.151535806693151),
     np.float64(2.1525868502462453),
     np.float64(2.1536369666095574),
     np.float64(2.1546861582325585),
     np.float64(2.155734427553953),
     np.float64(2.1567817770017466),
     np.float64(2.157828208993309),
     np.float64(2.1588737259354422),
     np.float64(2.159918330224442),
     np.float64(2.160962024246164),
     np.float64(2.162004810376087),
     np.float64(2.1630466909793764),
     np.float64(2.1640876684109447),
     np.float64(2.165127745015516),
     np.float64(2.1661669231276863),
     np.float64(2.1672052050719857),
     np.float64(2.168242593162938),
     np.float64(2.169279089705121),
     np.float64(2.170314696993227),
     np.float64(2.171349417312121),
     np.float64(2.172383252936901),
     np.float64(2.173416206132955),
     np.float64(2.174448279156019),
     np.float64(2.1754794742522354),
     np.float64(2.176509793658209),
     np.float64(2.177539239601065),
     np.float64(2.1785678142985034),
     np.float64(2.1795955199588555),
     np.float64(2.1806223587811395),
     np.float64(2.1816483329551155),
     np.float64(2.1826734446613396),
     np.float64(2.183697696071218),
     np.float64(2.18472108934706),
     np.float64(2.1857436266421333),
     np.float64(2.1867653101007147),
     np.float64(2.187786141858143),
     np.float64(2.1888061240408723),
     np.float64(2.1898252587665223),
     np.float64(2.1908435481439303),
     np.float64(2.191860994273202),
     np.float64(2.192877599245762),
     np.float64(2.1938933651444037),
     np.float64(2.1949082940433393),
     np.float64(2.1959223880082495),
     np.float64(2.1969356490963317),
     np.float64(2.1979480793563497),
     np.float64(2.198959680828682),
     np.float64(2.1999704555453685),
     np.float64(2.200980405530159),
     np.float64(2.201989532798562),
     np.float64(2.2029978393578875),
     np.float64(2.204005327207298),
     np.float64(2.2050119983378527),
     np.float64(2.206017854732553),
     np.float64(2.207022898366389),
     np.float64(2.2080271312063853),
     np.float64(2.209030555211643),
     np.float64(2.2100331723333886),
     np.float64(2.2110349845150146),
     np.float64(2.212035993692125),
     np.float64(2.213036201792579),
     np.float64(2.2140356107365338),
     np.float64(2.215034222436489),
     np.float64(2.216032038797327),
     np.float64(2.2170290617163566),
     np.float64(2.2180252930833557),
     np.float64(2.2190207347806123),
     np.float64(2.220015388682966),
     np.float64(2.221009256657849),
     np.float64(2.2220023405653286),
     np.float64(2.2229946422581452),
     np.float64(2.223986163581755),
     np.float64(2.224976906374368),
     np.float64(2.22596687246699),
     np.float64(2.22695606368346),
     np.float64(2.2279444818404914),
     np.float64(2.228932128747708),
     np.float64(2.2299190062076857),
     np.float64(2.2309051160159896),
     np.float64(2.231890459961212),
     np.float64(2.232875039825011),
     np.float64(2.233858857382146),
     np.float64(2.2348419144005174),
     np.float64(2.2358242126412033),
     np.float64(2.236805753858495),
     np.float64(2.2377865397999352),
     np.float64(2.2387665722063526),
     np.float64(2.2397458528118985),
     np.float64(2.240724383344084),
     np.float64(2.241702165523814),
     np.float64(2.242679201065422),
     np.float64(2.2436554916767077),
     np.float64(2.24463103905897),
     np.float64(2.24560584490704),
     np.float64(2.24657991090932),
     np.float64(2.2475532387478125),
     np.float64(2.248525830098157),
     np.float64(2.249497686629664),
     np.float64(2.2504688100053465),
     np.float64(2.2514392018819542),
     np.float64(2.2524088639100066),
     np.float64(2.253377797733826),
     np.float64(2.2543460049915693),
     np.float64(2.2553134873152603),
     np.float64(2.2562802463308227),
     np.float64(2.257246283658111),
     np.float64(2.258211600910943),
     np.float64(2.2591761996971305),
     np.float64(2.260140081618511),
     np.float64(2.2611032482709787),
     np.float64(2.262065701244515),
     np.float64(2.26302744212322),
     np.float64(2.263988472485341),
     np.float64(2.2649487939033053),
     np.float64(2.265908407943748),
     np.float64(2.2668673161675437),
     np.float64(2.267825520129834),
     np.float64(2.268783021380059),
     np.float64(2.269739821461985),
     np.float64(2.2706959219137346),
     np.float64(2.2716513242678147),
     np.float64(2.2726060300511457),
     np.float64(2.27356004078509),
     np.float64(2.2745133579854806),
     np.float64(2.275465983162647),
     np.float64(2.2764179178214463),
     np.float64(2.277369163461289),
     np.float64(2.2783197215761675),
     np.float64(2.279269593654682),
     np.float64(2.280218781180069),
     np.float64(2.281167285630229),
     np.float64(2.2821151084777513),
     np.float64(2.2830622511899414),
     np.float64(2.284008715228849),
     np.float64(2.284954502051292),
     np.float64(2.285899613108885),
     np.float64(2.2868440498480633),
     np.float64(2.2877878137101093),
     np.float64(2.288730906131179),
     np.float64(2.289673328542327),
     np.float64(2.2906150823695315),
     np.float64(2.2915561690337194),
     np.float64(2.2924965899507916),
     np.float64(2.2934363465316485),
     np.float64(2.2943754401822134),
     np.float64(2.2953138723034576),
     np.float64(2.296251644291426),
     np.float64(2.2971887575372585),
     np.float64(2.298125213427218),
     np.float64(2.2990610133427096),
     np.float64(2.299996158660309),
     np.float64(2.300930650751783),
     np.float64(2.301864490984114),
     np.float64(2.3027976807195234),
     np.float64(2.3037302213154947),
     np.float64(2.3046621141247967),
     np.float64(2.3055933604955063),
     np.float64(2.3065239617710307),
     np.float64(2.307453919290131),
     np.float64(2.3083832343869437),
     np.float64(2.3093119083910043),
     np.float64(2.310239942627268),
     np.float64(2.311167338416132),
     np.float64(2.312094097073459),
     np.float64(2.3130202199105976),
     np.float64(2.313945708234403),
     np.float64(2.3148705633472613),
     np.float64(2.3157947865471082),
     np.float64(2.3167183791274515),
     np.float64(2.3176413423773927),
     np.float64(2.318563677581647),
     np.float64(2.3194853860205638),
     np.float64(2.320406468970149),
     np.float64(2.321326927702085),
     np.float64(2.32224676348375),
     np.float64(2.3231659775782396),
     np.float64(2.3240845712443874),
     np.float64(2.3250025457367838),
     np.float64(2.3259199023057966),
     np.float64(2.326836642197591),
     np.float64(2.3277527666541493),
     np.float64(2.328668276913291),
     np.float64(2.3295831742086905),
     np.float64(2.3304974597698993),
     np.float64(2.331411134822363),
     np.float64(2.3323242005874416),
     np.float64(2.333236658282427),
     np.float64(2.3341485091205647),
     np.float64(2.3350597543110703),
     np.float64(2.335970395059149),
     np.float64(2.3368804325660144),
     np.float64(2.337789868028906),
     np.float64(2.338698702641109),
     np.float64(2.3396069375919715),
     np.float64(2.3405145740669235),
     np.float64(2.341421613247494),
     np.float64(2.34232805631133),
     np.float64(2.343233904432213),
     np.float64(2.3441391587800773),
     np.float64(2.3450438205210293),
     np.float64(2.3459478908173623),
     np.float64(2.3468513708275753),
     np.float64(2.347754261706391),
     np.float64(2.348656564604771),
     np.float64(2.349558280669935),
     np.float64(2.3504594110453763),
     np.float64(2.3513599568708803),
     np.float64(2.3522599192825395),
     np.float64(2.3531592994127717),
     np.float64(2.354058098390335),
     np.float64(2.3549563173403474),
     np.float64(2.3558539573842987),
     np.float64(2.356751019640071),
     np.float64(2.3576475052219528),
     np.float64(2.3585434152406557),
     np.float64(2.35943875080333),
     np.float64(2.3603335130135816),
     np.float64(2.3612277029714868),
     np.float64(2.362121321773609),
     np.float64(2.363014370513013),
     np.float64(2.363906850279283),
     np.float64(2.364798762158536),
     np.float64(2.3656901072334375),
     np.float64(2.366580886583218),
     np.float64(2.367471101283687),
     np.float64(2.3683607524072485),
     np.float64(2.3692498410229166),
     np.float64(2.3701383681963297),
     np.float64(2.3710263349897662),
     np.float64(2.371913742462159),
     np.float64(2.3728005916691095),
     np.float64(2.373686883662903),
     np.float64(2.374572619492524),
     np.float64(2.375457800203669),
     np.float64(2.376342426838762),
     np.float64(2.3772265004369686),
     np.float64(2.37811002203421),
     np.float64(2.3789929926631777),
     np.float64(2.3798754133533473),
     np.float64(2.380757285130992),
     np.float64(2.3816386090191974),
     np.float64(2.382519386037875),
     np.float64(2.383399617203775),
     np.float64(2.384279303530502),
     np.float64(2.385158446028527),
     np.float64(2.386037045705202),
     np.float64(2.3869151035647715),
     np.float64(2.3877926206083893),
     np.float64(2.388669597834128),
     np.float64(2.3895460362369954),
     np.float64(2.3904219368089454),
     np.float64(2.3912973005388927),
     np.float64(2.392172128412725),
     np.float64(2.393046421413316),
     np.float64(2.393920180520538),
     np.float64(2.394793406711276),
     np.float64(2.3956661009594393),
     np.float64(2.396538264235974),
     np.float64(2.3974098975088762),
     np.float64(2.3982810017432046),
     np.float64(2.3991515779010926),
     np.float64(2.4000216269417605),
     np.float64(2.4008911498215286),
     np.float64(2.4017601474938295),
     np.float64(2.4026286209092187),
     np.float64(2.4034965710153884),
     np.float64(2.404363998757179),
     np.float64(2.4052309050765914),
     np.float64(2.4060972909127982),
     np.float64(2.406963157202156),
     np.float64(2.4078285048782178),
     np.float64(2.4086933348717436),
     np.float64(2.409557648110713),
     np.float64(2.4104214455203365),
     np.float64(2.4112847280230665),
     np.float64(2.41214749653861),
     np.float64(2.4130097519839384),
     np.float64(2.413871495273301),
     np.float64(2.4147327273182344),
     np.float64(2.415593449027575),
     np.float64(2.416453661307469),
     np.float64(2.4173133650613847),
     np.float64(2.4181725611901226),
     np.float64(2.419031250591827),
     np.float64(2.419889434161998),
     np.float64(2.420747112793499),
     np.float64(2.4216042873765717),
     np.float64(2.422460958798844),
     np.float64(2.423317127945342),
     np.float64(2.4241727956985),
     np.float64(2.425027962938172),
     np.float64(2.425882630541641),
     np.float64(2.4267367993836313),
     np.float64(2.427590470336317),
     np.float64(2.4284436442693327),
     np.float64(2.4292963220497854),
     np.float64(2.4301485045422635),
     np.float64(2.431000192608847),
     np.float64(2.4318513871091185),
     np.float64(2.432702088900172),
     np.float64(2.4335522988366245),
     np.float64(2.4344020177706245),
     np.float64(2.435251246551864),
     np.float64(2.4360999860275854),
     np.float64(2.436948237042594),
     np.float64(2.437796000439268),
     np.float64(2.438643277057565),
     np.float64(2.439490067735035),
     np.float64(2.4403363733068293),
     np.float64(2.4411821946057093),
     np.float64(2.4420275324620566),
     np.float64(2.4428723877038823),
     np.float64(2.4437167611568364),
     np.float64(2.444560653644218),
     np.float64(2.4454040659869833),
     np.float64(2.446246999003757),
     np.float64(2.4470894535108383),
     np.float64(2.447931430322214),
     np.float64(2.4487729302495644),
     np.float64(2.4496139541022743),
     np.float64(2.4504545026874416),
     np.float64(2.451294576809886),
     np.float64(2.4521341772721588),
     np.float64(2.4529733048745506),
     np.float64(2.4538119604151016),
     np.float64(2.45465014468961),
     np.float64(2.4554878584916393),
     np.float64(2.4563251026125297),
     np.float64(2.457161877841405),
     np.float64(2.4579981849651817),
     np.float64(2.4588340247685774),
     np.float64(2.45966939803412),
     np.float64(2.4605043055421567),
     np.float64(2.4613387480708604),
     np.float64(2.46217272639624),
     np.float64(2.4630062412921485),
     np.float64(2.463839293530291),
     np.float64(2.4646718838802335),
     np.float64(2.4655040131094106),
     np.float64(2.4663356819831344),
     np.float64(2.467166891264602),
     np.float64(2.4679976417149043),
     np.float64(2.4688279340930337),
     np.float64(2.4696577691558925),
     np.float64(2.4704871476583015),
     np.float64(2.471316070353006),
     np.float64(2.472144537990686),
     np.float64(2.472972551319963),
     np.float64(2.4738001110874084),
     np.float64(2.4746272180375506),
     np.float64(2.4754538729128837),
     np.float64(2.476280076453875),
     np.float64(2.4771058293989716),
     np.float64(2.4779311324846103),
     np.float64(2.4787559864452238),
     np.float64(2.4795803920132484),
     np.float64(2.480404349919132),
     np.float64(2.481227860891341),
     np.float64(2.4820509256563685),
     np.float64(2.482873544938742),
     np.float64(2.483695719461029),
     np.float64(2.484517449943848),
     np.float64(2.4853387371058715),
     np.float64(2.4861595816638364),
     np.float64(2.486979984332551),
     np.float64(2.4877999458249),
     np.float64(2.4886194668518553),
     np.float64(2.4894385481224797),
     np.float64(2.4902571903439363),
     np.float64(2.491075394221495),
     np.float64(2.4918931604585395),
     np.float64(2.4927104897565737),
     np.float64(2.4935273828152305),
     np.float64(2.494343840332277),
     np.float64(2.4951598630036225),
     np.float64(2.4959754515233246),
     np.float64(2.4967906065835965),
     np.float64(2.497605328874815),
     np.float64(2.498419619085525),
     np.float64(2.4992334779024477),
     np.float64(2.500046906010488),
     np.float64(2.50085990409274),
     np.float64(2.501672472830493),
     np.float64(2.502484612903241),
     np.float64(2.5032963249886864),
     np.float64(2.504107609762748),
     np.float64(2.504918467899568),
     np.float64(2.505728900071517),
     np.float64(2.506538906949201),
     np.float64(2.50734848920147),
     np.float64(2.5081576474954215),
     np.float64(2.5089663824964084),
     np.float64(2.5097746948680446),
     np.float64(2.510582585272213),
     np.float64(2.511390054369069),
     np.float64(2.5121971028170496),
     np.float64(2.513003731272879),
     np.float64(2.5138099403915732),
     np.float64(2.514615730826448),
     np.float64(2.515421103229124),
     np.float64(2.516226058249534),
     np.float64(2.5170305965359283),
     np.float64(2.5178347187348806),
     np.float64(2.5186384254912944),
     np.float64(2.519441717448409),
     np.float64(2.520244595247806),
     np.float64(2.521047059529414),
     np.float64(2.521849110931516),
     np.float64(2.522650750090754),
     np.float64(2.5234519776421362),
     np.float64(2.524252794219042),
     np.float64(2.5250532004532276),
     np.float64(2.525853196974833),
     np.float64(2.5266527844123874),
     np.float64(2.527451963392813),
     np.float64(2.528250734541434),
     np.float64(2.5290490984819796),
     np.float64(2.5298470558365915),
     np.float64(2.5306446072258284),
     np.float64(2.531441753268672),
     np.float64(2.5322384945825327),
     np.float64(2.5330348317832554),
     np.float64(2.5338307654851246),
     np.float64(2.5346262963008703),
     np.float64(2.535421424841673),
     np.float64(2.536216151717169),
     np.float64(2.5370104775354574),
     np.float64(2.537804402903104),
     np.float64(2.538597928425147),
     np.float64(2.5393910547051033),
     np.float64(2.5401837823449718),
     np.float64(2.5409761119452408),
     np.float64(2.541768044104893),
     np.float64(2.5425595794214093),
     np.float64(2.543350718490776),
     ...]



###### ______________________________________________________________________________________________________________________________________


##### 🖼️ Visualizing the Numerical and Analytical Solution


```python
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
```




```python
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

```



---

##### 3.) 🔥 Approximating Partial Differential Equations

We can apply this same methodology to partial differential equations

 For example, consider the 1-D heat equation:

*Why care about a heat equation?*

The Black-Scholes PDE is often referred to as a second-order parabolic PDE likened to a heat equation from physics!*

 $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

 where:
 - $u(x,t)$ is the temperature at position $x$ and time $t$
 - $\alpha$ is the thermal diffusivity coefficient 
 - $\frac{\partial u}{\partial t}$ represents the rate of change of temperature with respect to time
 - $\frac{\partial^2 u}{\partial x^2}$ represents the second spatial derivative (curvature) of temperature

 This PDE describes how heat diffuses through a one-dimensional medium over time.




```python
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

```



###### ______________________________________________________________________________________________________________________________________


##### 🔎 Example: Step-by-Step Explicit Finite Differences for Heat Equation

Let's solve the heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ using explicit finite differences.

**Step 1:** 

Start with our heat equation
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

First, we discretize space and time:
- Space: $x_i = i\Delta x$ for $i = 0,1,...,N$
- Time: $t_j = j\Delta t$ for $j = 0,1,...,M$
- Let $u^j_i$ represent $u(x_i,t_j)$

**Step 2:** 

Recall finite difference approximations
$$\frac{\partial u}{\partial t} \approx \frac{u^{j+1}_i - u^j_i}{\Delta t}$$
$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u^j_{i+1} - 2u^j_i + u^j_{i-1}}{(\Delta x)^2}$$

**Step 3:** 

Plug our approximations into the heat equation
$$\frac{u^{j+1}_i - u^j_i}{\Delta t} = \alpha \frac{u^j_{i+1} - 2u^j_i + u^j_{i-1}}{(\Delta x)^2}$$

**Step 4:** 

Solve for the next time step $u^{j+1}_i$
$$u^{j+1}_i = u^j_i + \alpha\frac{\Delta t}{(\Delta x)^2}(u^j_{i+1} - 2u^j_i + u^j_{i-1})$$

**Step 5:** 

Implementation steps:
1. Choose initial condition $u(x,0)$
2. Select small step sizes $\Delta x$ and $\Delta t$ that satisfy $\alpha\frac{\Delta t}{(\Delta x)^2} \leq \frac{1}{2}$
3. For each time step j:
- Calculate $u^{j+1}_i$ using our formula for all interior points i
- Apply boundary conditions
- Move to next time step

**Key Idea:** *Stability* refers to if errors grow in time (as we iterate through our solution) - so long as we satisfy the stability requirement $\alpha\frac{\Delta t}{(\Delta x)^2} \leq \frac{1}{2}$ of our discretization we will be ok!

###### ______________________________________________________________________________________________________________________________________


##### 💻 Coding the Solution: Finite Differences Partial Differential Equation


```python
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
```


```python
u
```




    array([[0.00000000e+00, 3.09016994e-01, 5.87785252e-01, ...,
            5.87785252e-01, 3.09016994e-01, 0.00000000e+00],
           [0.00000000e+00, 2.93892626e-01, 5.59016994e-01, ...,
            5.59016994e-01, 2.93892626e-01, 0.00000000e+00],
           [0.00000000e+00, 2.79508497e-01, 5.31656755e-01, ...,
            5.31656755e-01, 2.79508497e-01, 0.00000000e+00],
           ...,
           [0.00000000e+00, 1.49566669e-05, 2.84492711e-05, ...,
            2.84492711e-05, 1.49566669e-05, 0.00000000e+00],
           [0.00000000e+00, 1.42246355e-05, 2.70568646e-05, ...,
            2.70568646e-05, 1.42246355e-05, 0.00000000e+00],
           [0.00000000e+00, 1.35284323e-05, 2.57326074e-05, ...,
            2.57326074e-05, 1.35284323e-05, 0.00000000e+00]])



###### ______________________________________________________________________________________________________________________________________


##### 🖼️ Visualizing the Numerical and Analytical Solution


```python
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

```




```python
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

```



---

##### 4.) 📝 Approximating the Black-Scholes Partial Differential Equation

Now instead of pretending we are using a model framework like earlier, the Black-Scholes model framework actually gives us a PDE...

 The Black-Scholes PDE:
 
 $\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$
 
 where:
 - $V$ is the option value
 - $t$ is time
 - $S$ is the stock price
 - $\sigma$ is volatility
 - $r$ is the risk-free rate

 
 **Step 1:**
 
 First, we discretize space and time:
 - Stock price: $S_i = i\Delta S$ for $i = 0,1,...,M$ where $\Delta S = S_{max}/M$
 - Time: $t_j = j\Delta t$ for $j = 0,1,...,N$ where $\Delta t = T/N$
 - Let $V^j_i$ represent $V(S_i,t_j)$
 
 **Step 2:**
 
 Recall finite difference approximations:
 $$\frac{\partial V}{\partial t} \approx \frac{V^{j+1}_i - V^j_i}{\Delta t}$$
 $$\frac{\partial V}{\partial S} \approx \frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S}$$
 $$\frac{\partial^2 V}{\partial S^2} \approx \frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2}$$
 
 **Step 3:**
 
 Plug our approximations into the Black-Scholes PDE:
 $$\frac{V^{j+1}_i - V^j_i}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2} + rS_i\frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S} - rV^j_i = 0$$
 
 **Step 4:**
 
 Solve for the next time step $V^{j+1}_i$:
 $$V^{j+1}_i = V^j_i + \Delta t(\frac{1}{2}\sigma^2S_i^2\frac{V^j_{i+1} - 2V^j_i + V^j_{i-1}}{(\Delta S)^2} + rS_i\frac{V^j_{i+1} - V^j_{i-1}}{2\Delta S} - rV^j_i)$$
 
 **Step 5:**
 
 Implementation steps:
 1. Set terminal condition $V(S,T)$ (option payoff at expiry)
 2. Select small step sizes $\Delta S$ and $\Delta t$ for stability
 3. For each time step j (working backwards):
    - Calculate $V^{j}_i$ using our formula for all interior points i
    - Apply boundary conditions at $S = 0$ and $S = S_{max}$
    - Move to previous time step


```python
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

```

    C:\Users\Roman\AppData\Local\Temp\ipykernel_13464\72657281.py:37: RuntimeWarning:
    
    divide by zero encountered in log
    




---

##### 5.) 💭 Closing Thoughts and Future Topics


In quantitative finance we construct pricing arguments under a model framework (*Black-Scholes, Heston, ...*) that yields a pricing partial differential equation.  

The definition of a derivative is extremely useful from approximating solutions (the pricing functions given a models framework) to these differential equations.

Different payoff structures will change the boundary conditions of the solutions to the pricing differential equations!

It should be noted that there are other ways to go about solving for these pricing functionals via different arguments and techniques (i.e. Monte Carlo Simulation) a topic that I have discussed in great detail on this channel!  

Future Topics

- Risk-Neutral Pricing, Change of Measure, Binomial Trees

- Deriving the Black-Scholes Equation (Delta Hedging)

- Deriving the Heston Equation (Vega, then Delta Hedging)

- Coding more Advanced PDE Approximations (*Black-Scholes, Heston, ...*)

- Schemes, Conditions, Errors (Less Exciting, Extremely Necessary)
