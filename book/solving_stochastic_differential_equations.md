## Stochastic Differential Equations for Quant Finance



##### ▶️ Related Quant Guild Videos:

*Necessary Stochastic Calculus:*
-  [Itô's Lemma Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo)

-  [Itô Integration Clearly and Visually Explained](https://youtu.be/dUvZ8m3QpeI)

*Why the Expectation is Sufficient for Pricing and Problems in Practice:*
- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)

- [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
###### ______________________________________________________________________________________________________________________________________

 
##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)

##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)

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

##### 1.) 🧮 Differential Equations

- Visualizing differential equations

- How to think about differential equations

##### 2.) 🔥 Partial Differential Equations

- The Black-Scholes equation

- Solutions to partial differential equations (exact, approximate)

##### 3.) 🎲 Stochastic Differential Equations

- How to think about stochastic differential equations

- Solving geometric Brownian motion

##### 4.) 📝 Analytical Solutions

- Discretizing exact solutions

##### 5.) 💻 Numerical Solutions

- Euler-Maruyama Scheme

##### 6.) ⚔️ Analytical v. Numerical Solutions

- Pros and cons to analytical and numerical techniques for solving stochastic differential equations

##### 7.) 💭 Closing Thoughts and Future Topics

---

##### 1.) 🧮 Differential Equations
 
 A differential equation is simply an equation involving the derivative of a function
 
 Typically, this is the function we are solving for, and because the derivative of a constant is zero we aren't too sure which function is the *right one*
 
 There are several classes of *parent* functions and their general forms come in similar structure, to name a few...
 - Polynomial Functions: $$ f(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_1x + a_0 $$
 - Rational Functions: $$ f(x) = \frac{a_nx^n + a_{n-1}x^{n-1} + ... + a_1x + a_0}{b_mx^m + b_{m-1}x^{m-1} + ... + b_1x + b_0} $$
 - Cyclical or Trigonometric Functions: $$ f(x) = a\sin(bx + c) + d \text{ or } f(x) = a\cos(bx + c) + d $$


Boundary conditions will help us determine the *correct*

#### 🔎 Example: Understanding the Differential Equation Problem

  Consider a basic parabola - the general form of this quadratic is $f(x) = ax^2 + bx + c$

  Let's assume $a = 2, b = 0, \text{ and } c = 1$, this means the function is $f(x) = 2x^2 + 1$ 

  Suppose for the sake of this example that input ($x$) is stock price and output ($f(x) = y$) is cost of an option in dollars


```python
import numpy as np
import plotly.graph_objects as go

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate y values for f(x) = 2x^2 + 1
y = 2*x**2 + 1

# Create figure
fig = go.Figure()

# Add trace for the quadratic function
fig.add_trace(
    go.Scatter(x=x, y=y,
               name='f(x) = 2x² + 1',
               line=dict(color='#00FF00', width=3))
)

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='Quadratic Function: f(x) = 2x² + 1',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='x - Stock Price')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='f(x) - Cost of Option in Dollars')

fig.show()

```



In many cases, we *know* the derivative - meaning we model the *change* of something and would like to recover the original function

 For example:
 - Population Growth Rate $\rightarrow$ Total population over time
 - Water Flow Rate $\rightarrow$ Total volume of water over time 
 - Dynamics of the Change in a Stock's Price $\rightarrow$ Terminal Stock Price Distribution (this introduces randomness)

Keep in mind, these are models making assumptions only as necessary

We want to state as little as possible about the *true* distribution while capturing as much variability in the system as possible...

Suppose in the example above we modelled the change in option contract value $f(x) = y$ as a differential equation, 

assume now we don't know the function $f(x) = 2x^2 + 1$

$f'(x) = 4x \iff \frac{dy}{dx} = 4x \iff y' = 4x$

Let's integrate to solve this

$\int f'(x) = \int 4x \implies f(x) = 2x^2 + C$


```python
import numpy as np
import plotly.graph_objects as go

# Generate x values
x = np.linspace(-2, 2, 200)

# Create figure
fig = go.Figure()

# Plot different solutions for various values of C
C_values = [-3, -2, -1, 0, 1, 2, 3]
base_opacity = 1
opacity_step = 0.8 / len(C_values)

for i, C in enumerate(C_values):
    y = 2*x**2 + C
    opacity = base_opacity - (i * opacity_step)
    fig.add_trace(go.Scatter(
        x=x, 
        y=y,
        mode='lines',
        line=dict(
            color='rgba(0, 191, 255, {})'.format(opacity),
            width=2
        ),
        name=f'C = {C}'
    ))

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='General Solution: f(x) = 2x² + C',
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
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='x - Stock Price')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='f(x) - Cost of Option in Dollars')

fig.show()

```



So the solution to this differential equation is $f(x) = 2x^2 + C$, but what is $C$

- General Solution: includes $C$ and other arbitrary coefficients as necessary

- Particular Solution: determines what these arbitrary coefficients are based on some initial or boundary conditions

Suppose we knew if the stock price was \$0 then the contract would be worth precisely \$1 due to its intrinsic value

$\implies f(0) = 1 \text{(initial condition)} \implies f(0) = 2(0) + C = 1 \implies C = 1$

Now we have recovered the particular solution for the option contract as defined above: $f(x) = 2x^2 + 1$


```python
import numpy as np
import plotly.graph_objects as go

# Generate x values
x = np.linspace(-2, 2, 200)

# Create figure
fig = go.Figure()

# Plot different solutions for various values of C
C_values = [-3, -2, -1, 0, 1, 2, 3]
base_opacity = 1
opacity_step = 0.8 / len(C_values)

for i, C in enumerate(C_values):
    y = 2*x**2 + C
    if C == 1:  # Highlight particular solution
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(
                color='rgb(57, 255, 20)',  # Neon green
                width=3
            ),
            name=f'C = {C} (Particular Solution)'
        ))
    else:
        opacity = base_opacity - (i * opacity_step)
        fig.add_trace(go.Scatter(
            x=x, 
            y=y,
            mode='lines',
            line=dict(
                color='rgba(0, 191, 255, {})'.format(opacity),
                width=2
            ),
            name=f'C = {C}'
        ))

# Update layout
fig.update_layout(
    width=800,
    height=500,
    title_text='General Solution: f(x) = 2x² + C',
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
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='x - Stock Price')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='f(x) - Cost of Option in Dollars')

fig.show()

```



##### 💡 **Thinking About Differential Equations:**

- Students can get hung up on differential equations because courses throw so many different forms and techniques for solving them out so quickly, similar to how students may find a distaste for matrices and vectors - these are all incredibly useful tools and we should be absolutely stoked to see equations or solutions involving them (they make our lives easier not harder!)

- Differential equations offer a way to represent some underlying function as an equation of that function's changes - that's all.  General and particular solutions exist depending on the intitial and boundary conditions of the desired function.

##### 💬 **Remarks:**

- There are a variety of forms of differential equations (first-order, second-order, higher-order, ...) and techniques for solving them

- Where analytical techniques are not possible we can use numerical techniques and computers to approximate solutions

- The efficacy of approximated solutions can be judged relative to efficiency, accuracy, and impact of error

---

##### 2.) 🔥 Partial Differential Equations

In practice, functions typically model outputs as a function of more than one input

For example, $f(x, y) = x^2 + y^2 = z$ could model option price in dollars where $x = \text{Stock Price}, y = \text{Volatility}$

Notice we can still *visualize* this function as it is in 3-D, the same ideas herein hold in higher dimensions but we lose visual intuition and interpretation...


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate x and y values for 3D surface
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # f(x,y) = x^2 + y^2

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'surface'}]],
    subplot_titles=('Paraboloid Surface: f(x,y) = x² + y²')
)

# Add surface plot
fig.add_trace(
    go.Surface(
        x=X,
        y=Y, 
        z=Z,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ),
    row=1, col=1
)

# Update layout
fig.update_layout(
    title='Paraboloid Surface Plot',
    width=1000,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    scene=dict(
        xaxis_title='x - Stock Price',
        yaxis_title='y - Volatility',
        zaxis_title='z = x² + y² - Cost of Option in Dollars',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    )
)

fig.show()

```



Further extending this idea, let's look at an example partial differential equation

##### 🔎 Example: Black-Scholes Equation

 Contrary to what some folks think, the Black-Scholes equation is not the same as the Black-Scholes Model.  The model, that is the European option price within the Black-Scholes framework, is the solution to the Black-Scholes equation.

 After applying the necessary assumptions and hedge portfolio argument, the Black-Scholes equation is a partial differential equation as the randomness from the original stochastic differential equation is hedged away, this PDE describes how the value of a European-style option changes over time and must earn the risk-free rate in a no arbitrage risk-neutral world:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

 where:
 - $V$ is the option value
 - $t$ is time
 - $S$ is the stock price
 - $r$ is the risk-free interest rate
 - $\sigma$ is the volatility of the stock

 Notice $V(\theta) \mapsto \mathbb{R} \text{ where } \theta = (t, S, r, K, \sigma) \in \mathbb{R}^5$ - there is no visualizing this function but in the same sense as the multivariable function above we are looking to recover it from the PDE.

 ##### Solution Methodologies

 - **Analytical Solution** 

 - **Numerical Solution:** Finite-Differences (Direct PDE Method), Monte Carlo Simulation (Expectation of Derivative Contract) 

Remember, with the Black-Scholes model we aren't *predicting* anything - its a risk-neutral expectation in a specific framework to assess the fair value of the derivative given possible forward looking states of the world while attempting to say *as little as possible* about the true empirical distribution...

  $$V(t,S) = e^{-r(T-t)}\mathbb{E}_\mathbb{Q}\left[h(S_T)|S_t=S\right]$$

The Black-Scholes equation can be solved analytically, below I deploy a finite-differences approach via a Crank-Nicolson scheme for time stepping and compare the approach with a small grid and stock price upper bound and a larger grid and stock price upper bound along with the analytical solution...


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# Parameters
S0 = 100  # Initial stock price
K = 50   # Strike price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1.0   # Time to maturity

# Grid parameters for small grid
M_small = 20  # Time steps
N_small = 20  # Stock price steps
S_max_small = 150  # Maximum stock price for small grid

# Grid parameters for large grid
M_large = 100  # Time steps 
N_large = 100  # Stock price steps
S_max_large = 1000  # Maximum stock price for large grid

def crank_nicolson_bs(S_max, M, N):
    # Grid setup
    dt = T/M
    ds = S_max/N
    
    # Grid points
    s = np.linspace(0, S_max, N+1)
    t = np.linspace(0, T, M+1)
    
    # Initialize solution grid
    V = np.zeros((N+1, M+1))
    
    # Terminal condition (payoff)
    V[:,-1] = np.maximum(s - K, 0)
    
    # Boundary conditions
    V[0,:] = 0
    V[-1,:] = S_max - K*np.exp(-r*(T-t))
    
    # Setup tridiagonal coefficients
    alpha = 0.25*dt*((sigma**2)*(s[1:-1]**2)/(ds**2) - r*s[1:-1]/ds)
    beta = -dt*0.5*((sigma**2)*(s[1:-1]**2)/(ds**2) + r)
    gamma = 0.25*dt*((sigma**2)*(s[1:-1]**2)/(ds**2) + r*s[1:-1]/ds)
    
    # Setup matrices - fix the size mismatch by using consistent slicing
    A = np.diag(1 - beta) + np.diag(-gamma[:-1], k=1) + np.diag(-alpha[1:], k=-1)
    B = np.diag(1 + beta) + np.diag(gamma[:-1], k=1) + np.diag(alpha[1:], k=-1)
    
    # Solve backwards in time
    for j in range(M-1, -1, -1):
        rhs = np.dot(B, V[1:-1,j+1])
        V[1:-1,j] = np.linalg.solve(A, rhs)
    
    return s, V[:,0]

def black_scholes(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Generate solutions
s_small, v_small = crank_nicolson_bs(S_max_small, M_small, N_small)
s_large, v_large = crank_nicolson_bs(S_max_large, M_large, N_large)
s_analytical = np.linspace(0, S_max_large, 200)
v_analytical = black_scholes(s_analytical, K, r, sigma, T)

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Small Grid (Poor Approximation)', 
                   'Large Grid (Better Approximation)',
                   'Analytical Solution')
)

# Add traces
fig.add_trace(
    go.Scatter(x=s_small, y=v_small, mode='lines', name='Small Grid',
               line=dict(color='rgba(0, 191, 255, 0.8)', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=s_large, y=v_large, mode='lines', name='Large Grid',
               line=dict(color='rgba(0, 191, 255, 0.8)', width=2)),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=s_analytical, y=v_analytical, mode='lines', name='Analytical',
               line=dict(color='rgba(0, 191, 255, 0.8)', width=2)),
    row=1, col=3
)

# Update layout
fig.update_layout(
    width=1250,
    height=400,
    title_text='Black-Scholes Option Pricing: Numerical vs Analytical Solutions',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
for i in range(1, 4):
    fig.update_xaxes(
        title_text='Stock Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 120],
        row=1, col=i
    )
    fig.update_yaxes(
        title_text='Option Price' if i == 1 else '',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 100],
        row=1, col=i
    )

fig.show()

```

    C:\Users\Roman\AppData\Local\Temp\ipykernel_22868\2308557047.py:59: RuntimeWarning:
    
    divide by zero encountered in log
    




##### ⚠️ Problems with Numerical Solutions

- *Finite-Differences*: just as with a neural network, approximation techniques including finite-differences out-of-the-box suffer from an incapacity to extrapolate meaning if we don't aim to price within the predefined grid, we will produce poor price approximations

- *Monte Carlo Methods* (Specifically for Option Pricing): are often impractical when other techniques are unavailable due to the frequency and quantity of prices required to quote a grid (specifically referring to rough model family - reusable paths and other techniques are available in other senses)

- *Efficacy*: when quoting prices we aim to do so efficiently and accurately - the efficacy of a numerical solution can be attributed to its speed, and tendancy for error (over/under estimation in different parameters)

---

##### 3.) 🎲 Stochastic Differential Equations

Stochastic differential equations are dealing with random variables - an *analytical* solution - that is a precise solution, and I use that term loosely, in this capacity refers to a *distribution* not a function that is predicting any sort of randomness.

Remember, anytime we are dealing with randomness or random variables we are dealing with sets of possible outcomes or a distribution - not a number in the classical sense so these quantities whether they are...

- random variables $X$

- stochastic integrals $\int dW_t$

- stochsatic differential equations $\frac{dS_t}{S_t} = \sigma dW_t$

we are operating with and trying to discern *distribution* functions that prescribe randomness of the output **NOT** functions that provide a deterministic output like the cases (ODE/PDE) above.

**Wait...**

But Black-Scholes relies on a stochastic differential equation doesn't it? The Black-Scholes equation gives a precise price not a distribution, right? Exactly, because again we are dealing with the EXPECTATION of the randomness, not the randomness itself - any one value for the European option contract may be net positive or negative but in expectation there is only one value in this framework: $V(t,S) = e^{-r(T-t)}\mathbb{E}_\mathbb{Q}\left[h(S_T)|S_t=S\right]$

##### 🔎 See the Multiplicative Dynamics v. Arithmetic Dynamics in the Chart Below

Use the legend to show and hide each line to see the relative impact on the price path

- **ABM**: Linear SDE, easily solvable

- **GBM**: Multiplicative SDE (Linear in the Logarithm), solvable with a log transformation


```python
# Parameters
T = 10  # Time horizon (10 years)
N = 10000 * T  # Number of time steps (daily)
dt = T/N  # Time step size
t = np.linspace(0, T, N)  # Time grid

# Initial conditions
S0 = 100  # Initial stock price
mu = 0.05  # Drift
sigma = 0.2  # Volatility

# Generate Brownian motion increments
np.random.seed(42)  # For reproducibility
dW = np.random.normal(0, np.sqrt(dt), N)  # Generate one set of increments
W = np.cumsum(dW)  # Cumulative sum for Brownian motion

# Simulate Arithmetic Brownian Motion using same Brownian path
S_abm = S0 + mu * t + sigma * W

# Simulate Geometric Brownian Motion using same Brownian path
S_gbm = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Create figure
fig = go.Figure()

# Add ABM trace
fig.add_trace(go.Scatter(
    x=t,
    y=S_abm,
    name='Arithmetic BM',
    line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
))

# Add GBM trace
fig.add_trace(go.Scatter(
    x=t,
    y=S_gbm,
    name='Geometric BM',
    line=dict(color='rgba(255, 127, 80, 0.8)', width=2)
))

fig.update_layout(
    width=1000,
    height=400,
    title_text='Arithmetic vs Geometric Brownian Motion',
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

fig.update_xaxes(
    title_text='Time (years)',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```



##### 💬 **Remark:**

There are a variety of different forms of stochastic processes, techniques for generating solutions, and of course, means of simulation.  We will look at a specific example of a solution to a stochastic differential equation for the case of the Black-Scholes framework.  That is, the underlying asset dynamics are assumed to follow a Geometric Brownian motion, so we will look to generate a solution to this equation.

##### 🔎 Example: Geometric Brownian Motion

The dynamics of the stock price process are governed by a Geometric Brownian motion in a Black-Scholes framework

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Rather than operate under Physical measure $\mathbb{P}$ we operate under risk-neutral measure $\mathbb{Q}$ from the no-arbitrage assumptions in the Black-Scholes framework - I will create a video in the near future deriving the model from scratch and discuss all assumptions in this capacity.

The Bachelier (father of financial mathematics) model assumes an arithmetic Brownian motion which is a *linear* SDE and quite easy to solve, 

$$dS_t = \mu dt + \sigma dW_t$$

the geometric Brownian motion is a multiplicative process which can be made *linear* via a log transformation (recall the relationship between log and exponential functions).  A trick that will help us solve this stochastic differential equation.  The process being multiplicative *may be more reasonable*, non-negative asset price paths and exponential behavior...

Remember, we are interested in the random time evolution of the stock price process $S_t$

Let's transform this process via a logarthim and define a new process $X_t = ln(S_t)$ and differentiate it using Ito's Lemma

 - (1) [Itô's Lemma Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo)

 - (2) [Itô Integration Clearly and Visually Explained](https://youtu.be/dUvZ8m3QpeI)

I assume the viewer or reader is familiar with these concepts here, it is likely I will do a video lecture on exclusively Brownian motion in the near future.

Here $X_t = f(S_t, t) \implies dX_t = \frac{\delta f}{\delta t} dt + \frac{\delta f}{\delta S_t} dS_t + \frac{1}{2} \frac{\delta^2f}{\delta S_t^2} dS_t^2 + \frac{1}{2} \frac{\delta^2f}{\delta t^2} dt^2 + \frac{1}{2} \frac{\delta^2f}{\delta t \delta S_t} dS_tdt$

From stochastic calculus we know

$$dt^2 = 0 = dW_tdt \quad dW_t^2 = dt, \text{ see (1) for a discussion on quadratic variation}$$

So our differential reduces as the dynamics given by Geometric Brownian motion cancel many terms out

$$dS_tdt = (\mu S_t dt + \sigma S_t dW_t)dt = 0$$

$$dS_t^2 = (\sigma^2 S_t^2 dW_t^2) = (\sigma^2 S_t^2 dt)$$

Plugging this back into the differential from Itô's lemma

$$X_t = f(S_t, t) \implies dX_t = \frac{\delta f}{\delta t} dt + \frac{\delta f}{\delta S_t} dS_t + \frac{1}{2} \frac{\delta^2f}{\delta S_t^2} dS_t^2 + \frac{1}{2} \frac{\delta^2f}{\delta t^2}(0) + \frac{1}{2} \frac{\delta^2f}{\delta t \delta S_t} (0)$$
$$= \frac{\delta f}{\delta t} dt + \frac{\delta f}{\delta S_t} dS_t + \frac{1}{2} \frac{\delta^2f}{\delta S_t^2} dS_t^2$$

Let's now find the rest of these derivatives of the function $f(S_t, t)$

$$\frac{\delta f}{\delta t} = 0 \text{ since } f(S_t,t) = ln(S_t) \text{ has no explicit time dependence}$$

$$\frac{\delta f}{\delta S_t} = \frac{1}{S_t}$$

$$\frac{\delta^2 f}{\delta S_t^2} = - \frac{1}{S_t^2}$$

Ok so the differential is 

$$dX_t = \frac{1}{S_t}(dS_t) - \frac{1}{2}\frac{1}{S_t^2} (dS_t^2)$$

$$\implies dX_t = \frac{1}{S_t}(\mu S_t dt + \sigma S_t dW_t) - \frac{1}{2}\frac{1}{S_t^2} (\sigma^2 S_t^2 dt)$$

$$\implies dX_t = (\mu - \frac{1}{2}\sigma^2)dt + \sigma dW_t$$

Now we can integrate to solve this equation, remember, even though we are dealing with $X_t$ we are still in terms of the original SDE: geometric Brownian motion - just under a log transformation. Moreover, we use dummy variables in the integrand to avoid confusion with the variable indexing time.

$$\implies \int_0^t dX_s = \int_0^t(\mu - \frac{1}{2}\sigma^2)dt + \int_0^t \sigma dW_s$$

Let's recall $\int_0^t \sigma dW_s$ is a random variable - see (2) for a discussion on Itô integration

Effectively $\int_0^t \sigma dW_s$ is the cumulative weighted sum of Brownian increments up to time $t$ where we initially have $W_0 = 0$

So in reality, we want the cumulative randomness from time $T = 0 \rightarrow T = t \implies \int_0^t \sigma dW_s = \sigma W_t$ 

This is due to the independent Brownian increment $W_t - W_0$ accounting for the cumulative randomness where we are left with weight $\sigma$ scaling it

$$\int_0^t dX_s = \int_0^t(\mu - \frac{1}{2}\sigma^2)dt + \int_0^t \sigma dW_s \implies X_t - X_0 = (\mu - \frac{1}{2}\sigma^2)(t - 0) + \sigma (W_t - W_0) \implies X_t = X_0 + (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t$$

Remember, $X_t = f(S_t, t) = ln(S_t) \implies e^{X_t} = S_t$ so let's exponentiate both sides of this equation

$$\implies e^{ln(S_t)} = e^{ln(S_0) + (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}$$

$$\implies e^{ln(S_t)} = e^{ln(S_0)} e^{(\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}$$

$$\implies S_t = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}$$

The solution to geometric Brownian motion!

**Wait there is still randomness?**
Yes, the solution is a distribution, specifically the dynamics imply a lognormal distribution.

##### 🔎 Lognormal Distribution of Stock Returns

**Log Normal Distribution:** If there is a normal random variable $Y$, $X$ is said to follow a lognormal distribution if

$$X = e^Y, X \sim Lognormal(\mu, \sigma^2)$$

Notice, $S_t \sim Lognormal(log(S_0) + (\mu - \frac{1}{2}\sigma^2)t, \sigma^2t^2)$

since

$S_t = e^{ln(S_0) + (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}$ where the only random variable is in fact Normal...


```python
# Parameters
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
t = 1  # Time

# Generate points for the lognormal distribution
x = np.linspace(0, 300, 1000)
mean = np.log(S0) + (mu - 0.5 * sigma**2) * t
std = sigma * np.sqrt(t)
pdf = (1 / (x * std * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mean)**2 / (2 * std**2))

# Create the plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        name='Lognormal Distribution',
        line=dict(color='rgba(0, 191, 255, 0.8)', width=2)
    )
)

fig.update_layout(
    width=1250,
    height=400,
    title_text='Lognormal Distribution of Stock Price at t=1',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Probability Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```

    C:\Users\Roman\AppData\Local\Temp\ipykernel_22868\754027290.py:11: RuntimeWarning:
    
    divide by zero encountered in divide
    
    C:\Users\Roman\AppData\Local\Temp\ipykernel_22868\754027290.py:11: RuntimeWarning:
    
    divide by zero encountered in log
    
    C:\Users\Roman\AppData\Local\Temp\ipykernel_22868\754027290.py:11: RuntimeWarning:
    
    invalid value encountered in multiply
    




---

##### 4.) 📝 Analytical Solutions

In any case, even if we have an analytical solution to a stochastic differential equation, if we want to find statistics for instruments with more complicated payoffs we have to discretize (*simulate*, I use this loosely here but its similar to numerical solutionsin this capacity) the analytical solution.  The more granular our discretization the more *accurate* it will be.

Let's look at an example finding the probability of breaching a barrier option - *sigh* yes we can't use a Black-Scholes framework to price exotics, we need to ensure consistent pricing - for a video on this matter please see [How to Price Exotic Options](https://youtu.be/hsot26myYYM)

##### 🔎 Discretizing Solution to Geometric Brownian Motion

$$S_t = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}$$

 For a discretization with n steps, where $\Delta t = \frac{t}{n}$ and $t_i = i\Delta t$:
 
 $$S_{t_i} = S_{t_{i-1}} e^{(\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t}Z_i}$$
 
 where $Z_i \sim N(0,1)$ are independent standard normal random variables.

Typical path dependent statistics demand discretization even with a continuous solution to the SDE...


```python
# Parameters
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
T = 1.0  # Time horizon
n = 1000  # Number of time steps
dt = T/n  # Time step size
n_paths = 500  # Number of simulation paths

# Generate time points
t = np.linspace(0, T, n+1)

# Generate random normal variables
Z = np.random.normal(0, 1, (n_paths, n))

# Initialize array for stock prices
S = np.zeros((n_paths, n+1))
S[:, 0] = S0

# Simulate paths
for i in range(1, n+1):
    S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i-1])

# Create the plot
fig = go.Figure()

# Add individual paths with low opacity
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=S[i, :],
            mode='lines',
            line=dict(color='rgba(0, 191, 255, 0.1)', width=1),
            showlegend=False
        )
    )

# Add mean path
mean_path = np.mean(S, axis=0)
fig.add_trace(
    go.Scatter(
        x=t,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='white', width=2)
    )
)

fig.update_layout(
    width=1000,
    height=400,
    title_text='Simulated Stock Price Paths (GBM)',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    title_text='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```



##### 🔎 Distribution of Maximum Prices: Path Dependent Probabilities

We need to *correctly* discretize GBM and simulate each path to ensure these values are correct.

The CI ensures with 95% confidence the parameter is contained within that interval.


```python
# Calculate breach probability for barrier at 125
barrier = 125
breaches = np.any(S >= barrier, axis=1)
breach_prob = np.mean(breaches)

# Calculate 95% confidence interval using normal approximation
z = 1.96  # 95% confidence level
std_error = np.sqrt((breach_prob * (1 - breach_prob)) / n_paths)
ci_lower = breach_prob - z * std_error
ci_upper = breach_prob + z * std_error

print(f"Probability of breaching {barrier}: {breach_prob:.2%}")
print(f"95% Confidence Interval: ({ci_lower:.2%}, {ci_upper:.2%})")

# Plot histogram of maximum prices reached
max_prices = np.max(S, axis=1)

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=max_prices,
    nbinsx=50,
    name='Maximum Price Distribution',
    marker_color='rgba(0, 191, 255, 0.6)'
))

# Add vertical line for barrier
fig.add_vline(
    x=barrier,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Barrier: {barrier}",
    annotation_position="top right"
)

fig.update_layout(
    width=1000,
    height=400,
    title_text='Distribution of Maximum Stock Prices Reached',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    bargap=0.1
)

fig.update_xaxes(
    title_text='Maximum Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Count',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```

    Probability of breaching 125: 34.00%
    95% Confidence Interval: (29.85%, 38.15%)




---

##### 5.) 💻 Numerical Solutions

Let's assume we don't have any solution to GBM

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

 We can use Euler-Maruyama discretization:

 $$S_{t+\Delta t} - S_t = \mu S_t \Delta t + \sigma S_t \Delta W_t$$

 where $\Delta W_t \sim \mathcal{N}(0,\sqrt{\Delta t})$

 This gives us:

 $$S_{t+\Delta t} = S_t + \mu S_t \Delta t + \sigma S_t \Delta W_t$$

 We can implement this numerically to approximate the solution




```python
# Parameters
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
T = 1.0  # Time horizon
n = 1000  # Number of time steps
dt = T/n  # Time step size
n_paths = 500  # Number of simulation paths

# Generate time points
t = np.linspace(0, T, n+1)

# Generate random normal variables
Z = np.random.normal(0, np.sqrt(dt), (n_paths, n))

# Initialize array for stock prices
S = np.zeros((n_paths, n+1))
S[:, 0] = S0

# Simulate paths using Euler-Maruyama
for i in range(1, n+1):
    S[:, i] = S[:, i-1] + mu*S[:, i-1]*dt + sigma*S[:, i-1]*Z[:, i-1]

# Create the plot
fig = go.Figure()

# Add individual paths with low opacity
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=S[i, :],
            mode='lines',
            line=dict(color='rgba(0, 191, 255, 0.1)', width=1),
            showlegend=False
        )
    )

# Add mean path
mean_path = np.mean(S, axis=0)
fig.add_trace(
    go.Scatter(
        x=t,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='white', width=2)
    )
)

fig.update_layout(
    width=1000,
    height=400,
    title_text='Simulated Stock Price Paths (Euler-Maruyama)',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    title_text='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```




```python
# Calculate breach probability for barrier at 125 using EM paths
barrier = 125
breaches = np.any(S >= barrier, axis=1)
breach_prob = np.mean(breaches)

# Calculate 95% confidence interval using normal approximation
z = 1.96  # 95% confidence level
std_error = np.sqrt((breach_prob * (1 - breach_prob)) / n_paths)
ci_lower = breach_prob - z * std_error
ci_upper = breach_prob + z * std_error

print(f"Probability of breaching {barrier} using Euler-Maruyama: {breach_prob:.2%}")
print(f"95% Confidence Interval: ({ci_lower:.2%}, {ci_upper:.2%})")

# Plot histogram of maximum prices reached
max_prices = np.max(S, axis=1)

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=max_prices,
    nbinsx=50,
    name='Maximum Price Distribution',
    marker_color='rgba(0, 191, 255, 0.6)'
))

# Add vertical line for barrier
fig.add_vline(
    x=barrier,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Barrier: {barrier}",
    annotation_position="top right"
)

fig.update_layout(
    width=1000,
    height=400,
    title_text='Distribution of Maximum Stock Prices (Euler-Maruyama)',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    bargap=0.1
)

fig.update_xaxes(
    title_text='Maximum Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Count',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```

    Probability of breaching 125 using Euler-Maruyama: 33.20%
    95% Confidence Interval: (29.07%, 37.33%)




---

##### 6.) ⚔️ Analytical v. Numerical Solutions

🔎 **Key Observations:**

For a derivative whose payoff only depends on the terminal stock price distribution we don't have to simulate paths if the terminal distribution is directly available for Monte Carlo simulation of derivative prices.  We can simply sample from the terminal distribution for the model's framework (i.e. Black-Scholes framework means sampling from GBM Lognormal terminal price distribution), computing a discounted payoff, and averaging those discounted payoffs to determine the risk-neutral price.

**1.) 🎯 Accuracy:**
   - The analytical solution is exact (up to floating-point precision)
   - Euler-Maruyama introduces discretization error that depends on step size
   - The error becomes more pronounced for larger time steps or higher volatility

**2.) ⏳ Computational Efficiency:**
   - Analytical solution is generally faster as it requires fewer calculations
   - Euler-Maruyama requires step-by-step calculations which can be slower
   - However, Euler-Maruyama is more versatile for complex SDEs

**3.) 📋 Implementation:**
   - Analytical solution requires knowing the exact solution form
   - Euler-Maruyama can be applied to any SDE, even when no analytical solution exists
   - Euler-Maruyama is easier to modify for different drift and diffusion terms

**4.) 🌊 Stability:**
   - Analytical solution is inherently stable
   - Euler-Maruyama can become unstable for large time steps
   - Both methods preserve the positivity of stock prices


```python
import numpy as np
import plotly.graph_objects as go

# Parameters for lognormal distribution
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
T = 1.0  # Time horizon
n = 1000  # Number of time steps
dt = T/n  # Time step size
n_paths = 10000  # Increased number of paths for better distribution estimation

# Generate EM paths
Z = np.random.normal(0, np.sqrt(dt), (n_paths, n))
S_em = np.zeros((n_paths, n+1))
S_em[:, 0] = S0

for i in range(1, n+1):
    S_em[:, i] = S_em[:, i-1] + mu*S_em[:, i-1]*dt + sigma*S_em[:, i-1]*Z[:, i-1]

# Final prices from EM
final_prices_em = S_em[:, -1]

# Generate analytical lognormal distribution points
x = np.linspace(50, 200, 1000)
mu_ln = np.log(S0) + (mu - 0.5*sigma**2)*T
sigma_ln = sigma*np.sqrt(T)
pdf_analytical = np.exp(-(np.log(x) - mu_ln)**2 / (2*sigma_ln**2)) / (x*sigma_ln*np.sqrt(2*np.pi))

# Create histogram of EM final prices and overlay analytical PDF
fig = go.Figure()

# Add histogram of EM prices
fig.add_trace(go.Histogram(
    x=final_prices_em,
    nbinsx=50,
    name='EM Final Prices',
    histnorm='probability density',
    marker_color='rgba(0, 191, 255, 0.6)'
))

# Add analytical PDF
fig.add_trace(go.Scatter(
    x=x,
    y=pdf_analytical,
    mode='lines',
    name='Analytical Lognormal',
    line=dict(color='red', width=2)
))

fig.update_layout(
    width=1000,
    height=400,
    title_text='Distribution of Final Stock Prices: EM vs Analytical',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    bargap=0.1
)

fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

# Calculate KL divergence
# First, create histograms with same bins for both distributions
bins = np.linspace(min(final_prices_em), max(final_prices_em), 100)
hist_em, _ = np.histogram(final_prices_em, bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Calculate analytical PDF at bin centers
pdf_analytical_bins = np.exp(-(np.log(bin_centers) - mu_ln)**2 / (2*sigma_ln**2)) / (bin_centers*sigma_ln*np.sqrt(2*np.pi))

# Add small constant to avoid division by zero
epsilon = 1e-10
hist_em = hist_em + epsilon
pdf_analytical_bins = pdf_analytical_bins + epsilon

# Calculate KL divergence
kl_divergence = np.sum(hist_em * np.log(hist_em / pdf_analytical_bins)) * (bins[1] - bins[0])

print(f"KL Divergence between EM and analytical distributions: {kl_divergence:.6f}")

```



    KL Divergence between EM and analytical distributions: 0.005572



```python
from scipy.stats import norm

# Parameters for option pricing
S0 = 100  # Initial stock price
K = 110   # Strike price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1.0    # Time to maturity
n_samples = 100000  # Number of Monte Carlo samples

# Black-Scholes analytical price
d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_call = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Monte Carlo price using terminal distribution
Z = np.random.normal(0, 1, n_samples)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoffs = np.maximum(ST - K, 0)
mc_call = np.exp(-r*T) * np.mean(payoffs)

# Calculate 95% confidence interval for MC
std_error = np.std(payoffs) / np.sqrt(n_samples)
ci_lower = np.exp(-r*T) * (np.mean(payoffs) - 1.96 * std_error)
ci_upper = np.exp(-r*T) * (np.mean(payoffs) + 1.96 * std_error)

print(f"Black-Scholes Call Price: ${bs_call:.2f}")
print(f"Monte Carlo Call Price: ${mc_call:.2f}")
print(f"95% CI: (${ci_lower:.2f}, ${ci_upper:.2f})")

# Create histogram of terminal prices
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=ST,
    nbinsx=50,
    name='Terminal Price Distribution',
    marker_color='rgba(0, 191, 255, 0.6)',
    histnorm='probability density'
))

# Add vertical lines for important prices
fig.add_vline(
    x=K,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Strike: {K}",
    annotation_position="top right"
)

fig.add_vline(
    x=S0,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Initial Price: {S0}",
    annotation_position="top left"
)

# Add theoretical lognormal density
x = np.linspace(max(0, min(ST)), max(ST), 1000)
pdf = np.exp(-(np.log(x) - (np.log(S0) + (r - 0.5*sigma**2)*T))**2 / (2*sigma**2*T)) / (x*sigma*np.sqrt(2*np.pi*T))

fig.add_trace(go.Scatter(
    x=x,
    y=pdf,
    mode='lines',
    name='Theoretical Density',
    line=dict(color='white', width=2)
))

fig.update_layout(
    width=1000,
    height=400,
    title_text='Terminal Stock Price Distribution',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    bargap=0.1
)

fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```

    Black-Scholes Call Price: $6.04
    Monte Carlo Call Price: $6.03
    95% CI: ($5.96, $6.10)





```python
# Parameters
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
T = 1.0  # Time horizon
n = 1000  # Number of time steps
dt = T/n  # Time step size
n_paths = 500  # Number of simulation paths

# Generate time points
t = np.linspace(0, T, n+1)

# Generate random normal variables for both methods
Z = np.random.normal(0, np.sqrt(dt), (n_paths, n))

# Initialize arrays for both methods
S_euler = np.zeros((n_paths, n+1))
S_analytical = np.zeros((n_paths, n+1))

S_euler[:, 0] = S0
S_analytical[:, 0] = S0

# Simulate paths
for i in range(1, n+1):
    # Euler-Maruyama method
    S_euler[:, i] = S_euler[:, i-1] + mu*S_euler[:, i-1]*dt + sigma*S_euler[:, i-1]*Z[:, i-1]
    
    # Analytical solution (using the exact solution of geometric Brownian motion)
    S_analytical[:, i] = S0 * np.exp((mu - 0.5*sigma**2)*t[i] + sigma*np.cumsum(Z, axis=1)[:, i-1])

# Calculate covariance matrices
cov_euler = np.cov(S_euler)
cov_analytical = np.cov(S_analytical)

# Create comparison plot
fig = go.Figure()

# Plot a few sample paths for each method
for i in range(5):  # Plotting just 5 paths for clarity
    fig.add_trace(
        go.Scatter(
            x=t,
            y=S_euler[i, :],
            mode='lines',
            line=dict(color='rgba(0, 191, 255, 0.3)', width=1),
            name='Euler-Maruyama' if i == 0 else None,
            showlegend=(i == 0)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=t,
            y=S_analytical[i, :],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.3)', width=1),
            name='Analytical' if i == 0 else None,
            showlegend=(i == 0)
        )
    )

# Add mean paths
mean_euler = np.mean(S_euler, axis=0)
mean_analytical = np.mean(S_analytical, axis=0)

fig.add_trace(
    go.Scatter(
        x=t,
        y=mean_euler,
        mode='lines',
        name='Mean (Euler)',
        line=dict(color='white', width=2, dash='dash')
    )
)

fig.add_trace(
    go.Scatter(
        x=t,
        y=mean_analytical,
        mode='lines',
        name='Mean (Analytical)',
        line=dict(color='yellow', width=2, dash='dash')
    )
)

fig.update_layout(
    width=1000,
    height=400,
    title_text='Comparison: Euler-Maruyama vs Analytical Solution',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    title_text='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

# Calculate and print error metrics
mse = np.mean((S_euler - S_analytical)**2)
max_error = np.max(np.abs(S_euler - S_analytical))

print("\nComparison Metrics:")
print(f"Mean Squared Error between methods: {mse:.6f}")
print(f"Maximum Absolute Error: {max_error:.6f}")
print("\nCovariance Matrix Difference:")
print(f"Frobenius norm of difference: {np.linalg.norm(cov_euler - cov_analytical):.6f}")


fig.show()

```

    
    Comparison Metrics:
    Mean Squared Error between methods: 0.004595
    Maximum Absolute Error: 0.350697
    
    Covariance Matrix Difference:
    Frobenius norm of difference: 181.256137




##### 📉 Error Analysis of Euler-Maruyama Method
 
It is highly likely I will do a video exclusively on this methodology in the future, but some relevant information

Remember, we are dealing with randomness so we are talking about convergence and error in terms of the expected value of the process...

 The Euler-Maruyama (EM) method exhibits two types of convergence with different error rates:
 
 1. **💪 Strong Convergence:**
    - Order of convergence: $\frac{1}{2}$
    - Error bound: $\mathbb{E}[|X(T) - X_N(T)|] \leq C\sqrt{\Delta t}$
    - Halving step size: $\Delta t \rightarrow \frac{\Delta t}{2}$ reduces error by factor $\frac{1}{\sqrt{2}}$
 
 2. **🦾 Weak Convergence (a function of the process, typically sufficient for mathematical finance needs):**
    - Order of convergence: $1$
    - Error bound: $|\mathbb{E}[f(X(T))] - \mathbb{E}[f(X_N(T))]| \leq C\Delta t$
    - Halving step size: $\Delta t \rightarrow \frac{\Delta t}{2}$ reduces error by factor $\frac{1}{2}$
 
 where:
 - $X(T)$ is the true solution
 - $X_N(T)$ is the numerical solution
 - $C$ is a constant
 - $\Delta t$ is the time step
 - $f$ is a suitable test function

---

##### 7.) 💭 Closing Thoughts and Future Topics


This video covered a lot of topics:

- **Ordinary Differential Equations** (the idea of recovering a function)

- **Partial Differential Equations** (the idea of recovering a function or approximating it via numerical solutions e.g. finite-differences)

- **Stochastic Differential Equations** (recovering the function governing a process' distribution analytically and numerically)

In any case, whenever we are dealing with randomness - think a set of possible outcomes, there is nothing to predict, like a dice roll it is random.

Whenever we need a quantity from this randomness (set of outcomes, like set of potential stock prices) we need the expectation - this is *literally* the best we can do in the framework of these mathematical models for producing a fair, efficient, and accurate price.

See my video [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ) and [How to Trade with Edge](https://youtu.be/NlqpDB2BhxE) for discussions on why this is the *best* we can do when randomness is prescribed and *is well defined* by some theoretical distribution that is *known* not some unseen unmeasured theoretical distribution.

Future Topics

- Martingales and Girsanov's Theorem (No Arbitrage Assumption, Change of Measure)

- Finite-Differences for Approximating Solutions to PDEs

- Euler-Maruyama Method for Simulating Stochastic Processes (Rough Volatility Model Family, Volterra Processes, etc...)

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
