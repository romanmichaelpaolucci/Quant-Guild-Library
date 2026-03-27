# ## How Trading Desks Price Exotic Options
# 
# Recommended Prerequisite Quant Guild Lectures:
#  - [Trading with the Black-Scholes Implied Volatility Surface](https://youtu.be/YH0tWpBaKGs) (Why Black-Scholes is insufficient)
#  
#  - [Monte Carlo Simulation and Black-Scholes for Pricing Options](https://youtu.be/-1RYvajksjQ) (Why we can simulate prices)
# 
#  - [Why Quant Traders Care About Pricing](https://youtu.be/s0lVvYMA5OA) (Why we care about pricing)
# 
# Quant Guild Lectures Discussing Pricing Model Theory:
# - [Ito's Lemma Clearly and Visually Explained](https://youtu.be/TgBzqdN24fo) (Basis for stochastic differentiation)
# 
# - [Ito Integration Clearly and Visually Explained](https://youtu.be/dUvZ8m3QpeI) (Basis for solving stochastic differential equations)
# 
# - [How to Find the Black-Scholes-Merton Partial Differential Equation](https://youtu.be/2iClLEfXuqA?si=AgjS1PE1-uyuf5sh) (Applying stochastic calculus to pricing)
# 
# Lectures Applying Pricing Models:
#   - [How to Trade with the Black-Scholes Model](https://youtu.be/0x-Pc-Z3wu4) (Applying the Black-Scholes model as a market-maker)
# 
#   - [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ) (Implications in pricing model parameters and statistical trading)
# 
#   - [How to Trade](https://youtu.be/NqOj__PaMec) (Why we still use and trade incorrect models)
#  
# 


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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display, HTML

def simulate_heston_paths(S0=100, v0=0.04, kappa=2, theta=0.04, sigma=0.3, rho=-0.7, 
                         r=0.02, T=1, N=252, paths=50):
    dt = T/N
    # Initialize arrays
    S = np.zeros((paths, N+1))
    v = np.zeros((paths, N+1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated random numbers
    z1 = np.random.standard_normal((paths, N))
    z2 = rho * z1 + np.sqrt(1-rho**2) * np.random.standard_normal((paths, N))
    
    # Simulate paths
    for t in range(N):
        S[:, t+1] = S[:, t] * np.exp((r - 0.5*v[:, t])*dt + np.sqrt(v[:, t]*dt)*z1[:, t])
        v[:, t+1] = np.maximum(v[:, t] + kappa*(theta - v[:, t])*dt + sigma*np.sqrt(v[:, t]*dt)*z2[:, t], 0)
    
    return S, v

def price_up_and_in_barrier(S, K=100, B=120, r=0.02, T=1):
    # Check if barrier was hit and calculate payoff
    barrier_hit = np.max(S, axis=1) >= B
    payoff = np.maximum(S[:, -1] - K, 0) * barrier_hit
    price = np.mean(payoff) * np.exp(-r*T)
    return price
def plot_heston_barrier(n_paths=50, barrier=120, strike=100):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Make background transparent
    fig.patch.set_alpha(0)
    ax1.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    
    # Simulate paths
    S, v = simulate_heston_paths(paths=n_paths)
    price = price_up_and_in_barrier(S, K=strike, B=barrier)
    
    # Plot price paths
    t = np.linspace(0, 1, S.shape[1])
    for i in range(n_paths):
        color = 'lime' if np.max(S[i]) >= barrier else 'gray'
        ax1.plot(t, S[i], alpha=0.3, color=color, linewidth=1)
    
    # Plot barrier
    ax1.axhline(y=barrier, color='red', linestyle='--', label='Barrier')
    ax1.axhline(y=strike, color='yellow', linestyle=':', label='Strike')
    
    # Plot volatility paths
    for i in range(n_paths):
        ax2.plot(t, np.sqrt(v[i])*100, alpha=0.3, color='cyan', linewidth=1)
    
    # Formatting
    ax1.set_title(f'Heston Model: Up-and-In Barrier Option\nPrice = {price:.2f}', pad=20)
    ax1.set_ylabel('Stock Price')
    ax1.grid(True, alpha=0.2)
    ax1.legend(facecolor='none')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility (%)')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

# Create interactive widgets
paths_slider = widgets.IntSlider(value=50, min=10, max=200, step=10, 
                               description='Paths:', layout=widgets.Layout(width='400px'))
barrier_slider = widgets.IntSlider(value=120, min=105, max=150, step=5,
                                 description='Barrier:', layout=widgets.Layout(width='400px'))
strike_slider = widgets.IntSlider(value=100, min=80, max=120, step=5,
                                description='Strike:', layout=widgets.Layout(width='400px'))

# Display interactive plot
widgets.interactive(plot_heston_barrier, 
                   n_paths=paths_slider,
                   barrier=barrier_slider, 
                   strike=strike_slider)



# ## Sections
# 1.) General Recipe for Pricing Exotics
# 
# 2.) Model Selection: Beyond Black-Scholes
# 
# 3.) Approximating Vanilla Pricing Functionals
# 
# 4.) Model Calibration to an Implied Volatility Surface
# 
# 5.) Approximating Exotic Pricing Functionals
# 
# 6.) Closing Thoughts and Future Topics


# ---


# ### 1.) General Recipe for Pricing Exotics
# 
# Informally, exotics are contracts with *interesting* payoff structures:
# 
# - Barrier Options (Knock-Out, Knock-In)
# 
# - Cliquet Options (Ratchets, Series of Forward-Start Options)
# 
# - Asian Options (Mean Price)
# 
# #### <u>What does the payoff of an exotic look like?</u>
# 
# e.x.) A knock-out barrier option requires the price path to *not* hit a certain price threshold otherwise the payoff is zero.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simulate_heston_path(S0, v0, kappa, theta, xi, rho, r, T, N):
    dt = T/N
    S = np.zeros(N+1)
    v = np.zeros(N+1)
    S[0] = S0
    v[0] = v0
    
    for i in range(N):
        z1 = np.random.normal()
        z2 = rho * z1 + np.sqrt(1-rho**2) * np.random.normal()
        
        v[i+1] = v[i] + kappa*(theta-v[i])*dt + xi*np.sqrt(v[i]*dt)*z2
        v[i+1] = max(0, v[i+1])  # Ensure variance stays positive
        
        S[i+1] = S[i] * np.exp((r - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*z1)
    
    return S, v

# Parameters
S0 = 100  # Initial stock price
v0 = 0.04  # Initial variance
kappa = 2.0  # Mean reversion speed
theta = 0.04  # Long-term variance
xi = 0.4  # Volatility of variance
rho = -0.5  # Correlation
r = 0.02  # Risk-free rate
T = 1.0  # Time horizon
N = 252  # Number of steps
K = 100  # Strike price
B = 120  # Barrier level

# Simulate path
S, v = simulate_heston_path(S0, v0, kappa, theta, xi, rho, r, T, N)
t = np.linspace(0, T, N+1)

# Check if barrier was hit
barrier_hit = np.any(S >= B)
if barrier_hit:
    path_color = 'red'
else:
    # Check if in the money at expiry
    if S[-1] > K:
        path_color = 'green'  # In the money
    else:
        path_color = 'yellow'   # Out of the money

# Create payoff
payoff = max(S[-1] - K, 0) if not barrier_hit else 0

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add price path
fig.add_trace(
    go.Scatter(x=t, y=S, name="Stock Price", line=dict(color=path_color)),
    secondary_y=False
)

# Add barrier line
fig.add_trace(
    go.Scatter(x=t, y=[B]*len(t), name="Barrier", line=dict(color='red', dash='dash')),
    secondary_y=False
)

# Add strike line
fig.add_trace(
    go.Scatter(x=t, y=[K]*len(t), name="Strike", line=dict(color='yellow', dash='dash')),
    secondary_y=False
)

# Update layout
fig.update_layout(
    title='Up-and-Out Call Option: Price Path and Barrier',
    xaxis_title='Time (Years)',
    yaxis_title='Stock Price ($)',
    width=900,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.95
    ),
    font=dict(color='white')
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')

# Add annotation for payoff
fig.add_annotation(
    text=f"Payoff: ${payoff:.2f}",
    xref="paper", yref="paper",
    x=0.02, y=0.94,
    showarrow=False,
    font=dict(size=14, color='white'),
    bgcolor="rgba(0,0,0,0.5)"
)

fig.show()



# #### <u>General Recipe</u>
# 
# 1.) Model Selection ($\mathcal{M}$, $\zeta$)
# 
# 2.) Determine Vanilla Pricing Scheme (Closed-Form, Quasi-Closed-Form, PDE Methods, Simulation)
# 
# 3.) Calibrate Model to Market Volatility Surface (Consistent Prices)
# 
# 4.) Determine Exotic Option Pricing Scheme Given Parameters in Step (3)
# 
# 5.) Apply Price Adjustments (Counterparty Risk, Liquidity, Profit)
# 
# 6.) Send Quote


# ---


# ### 2.) Model Selection: Beyond Black-Scholes
# 
# #### <u>The Black-Scholes Model</u>
# 
# Solving the Black-Scholes Equation Yields:
# 
# $$C = S_t \Phi(d_1) - Ke^{-rt} \Phi(d_2)$$
# 
# $$\Phi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}}e^{\frac{-s^2}{2}}ds$$
# 
# $$d_1 = \frac{ln(\frac{S_t}{K})+(r+\frac{\sigma^2}{2})t}{\sigma \sqrt{t}}$$
# 
# $$d_2 = d_1 - \sigma \sqrt{t}$$
# 
# Mathematically, the Black-Scholes formula is a functional, a map from $\mathbb{R}^5 \rightarrow \mathbb{R}$:
# 
# $BS: (S, K, r, \sigma, T) \mapsto C$
# 
# Where:
# - $S \in \mathbb{R}^+ $ (spot price)
# - $K \in \mathbb{R}^+$ (strike price) 
# - $r \in \mathbb{R}$ (risk-free rate)
# - $\sigma \in \mathbb{R}^+$ (volatility)
# - $T \in \mathbb{R}^+$ (time to expiry)
# $\mapsto C \in \mathbb{R}^+$ (call option price)
# 
# Realistically, the market or contract selection determines $(S, K, r, T)$
# 
# $$\implies BS: (\sigma | S, K, r, T) \mapsto C$$
# 
# That is, given the market and contract parameters $(S, K, r, T)$ we choose a volatility to price the options according to the assumptions of the Black-Scholes model.
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interactive, FloatSlider, Layout

# Data for the volatility surface - precompute meshgrid
strikes = [90, 95, 100, 105, 110]
maturities = [1/12, 3/12, 6/12, 1, 2]  # in years
maturity_labels = ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years']

# Market volatility values (in %)
market_vols = np.array([
    [28.0, 24.5, 22.0, 20.5, 19.5],  # 1 month
    [27.5, 24.0, 21.8, 20.3, 19.3],  # 3 months
    [27.0, 23.5, 21.5, 20.0, 19.0],  # 6 months
    [26.5, 23.0, 21.2, 19.8, 18.8],  # 1 year
    [26.0, 22.5, 21.0, 19.5, 18.5]   # 2 years
])

# Create meshgrid for 3D surface once
X, Y = np.meshgrid(strikes, maturities)

# Set fixed previous volatility to 20%
previous_vol = 20.0

# Pre-compute the previous vol surface
previous_vol_surface = np.full_like(market_vols, previous_vol)

def update_plot(bs_vol):
    # Black-Scholes constant volatility surface
    current_vol_surface = np.full_like(market_vols, bs_vol)
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Market Implied Volatility Surface', 'Black-Scholes Volatility Surface')
    )

    # Add market volatility surface
    fig.add_trace(
        go.Surface(x=X, y=Y, z=market_vols, colorscale='Viridis', opacity=0.7, showscale=True),
        row=1, col=1
    )

    # Add previous Black-Scholes volatility surface (transparent)
    fig.add_trace(
        go.Surface(x=X, y=Y, z=previous_vol_surface, colorscale='Reds', 
                  opacity=0.3, showscale=False),
        row=1, col=2
    )

    # Add current Black-Scholes volatility surface
    fig.add_trace(
        go.Surface(x=X, y=Y, z=current_vol_surface, colorscale='Blues', 
                  opacity=0.7, showscale=False),
        row=1, col=2
    )

    # Add vector arrows showing the shift - only at key points to reduce computation
    if abs(bs_vol - previous_vol) > 0.1:
        arrow_x = np.array([95, 100, 105])
        arrow_y = np.array([0.5, 1.0, 1.5])
        X_arrows, Y_arrows = np.meshgrid(arrow_x, arrow_y)
        
        arrow_color = 'red' if bs_vol < previous_vol else 'green'
        for i in range(len(arrow_x)):
            for j in range(len(arrow_y)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[X_arrows[j,i], X_arrows[j,i]],
                        y=[Y_arrows[j,i], Y_arrows[j,i]],
                        z=[previous_vol_surface[j,i], current_vol_surface[j,i]],
                        mode='lines+markers',
                        line=dict(color=arrow_color, width=2),
                        marker=dict(size=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )

    # Update layout
    fig.update_layout(
        title='Market vs Black-Scholes Volatility Surfaces',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity (Years)', 
            zaxis_title='Implied Volatility (%)',
            xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
            yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
            zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene2=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity (Years)',
            zaxis_title='Implied Volatility (%)',
            xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
            yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
            zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)', range=[np.min(market_vols)-2, np.max(market_vols)+2]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1000,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    fig.show()

# Create interactive slider with improved styling
interactive_plot = interactive(
    update_plot,
    bs_vol=FloatSlider(
        min=15.0,
        max=30.0,
        step=0.5,
        value=np.mean(market_vols),
        description='BS Volatility (%):',
        style={'description_width': '120px'},
        layout=Layout(width='500px')
    )
)

interactive_plot



# #### <u>Problem with Black-Scholes (Consistency): Statistical and Static Arbitrage</u>
# 
# The simplest example of why we need to demand consistent pricing can be seen above.
# 
# - Volatility increases $\rightarrow$ Option Price increase
# - Volatility decreases $\rightarrow$ Option Price decrease
# 
# I am constantly quoting prices against the market that are over and undervalued. Imagine if I were selling jewels all for the same price, and others knew the market price differed for diamonds and gold they could sell me what I overvalue and buy what is undervalued and I'll get killed.
# 
# We need a way to **consistently and efficiently** quote prices. 


# Calculate skew relative to flat 20% BS vol
flat_bs_vol = 20.0
spot = 100

# Get market volatilities for each strike at each maturity
market_skew = market_vols[0]  # Take first maturity for simplicity
bs_skew = [flat_bs_vol] * len(strikes)

# Create figure
fig = go.Figure()

# Add market skew line
fig.add_trace(
    go.Scatter(
        x=strikes,
        y=market_skew,
        name='Market Volatility Skew',
        line=dict(color='#00b3ff', width=3)
    )
)

# Add flat BS vol line
fig.add_trace(
    go.Scatter(
        x=strikes,
        y=bs_skew,
        name='Black-Scholes Volatility',
        line=dict(color='#ff9100', width=3, dash='dash')
    )
)

# Update layout
fig.update_layout(
    title='Market, Black-Scholes Skew',
    xaxis_title='Strike Price',
    yaxis_title='Implied Volatility (%)',
    width=900,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.95
    ),
    font=dict(color='white')
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')

fig.show()



# #### <u>Extensions and Alternative Models</u>
# 
# The Black-Scholes model assumes a geometric Brownian motion stochastic differential equation for the underlying leading to the constant volatility assumption
# $$\frac{dS_t}{S_t} = \mu dt + \sigma dW_t$$
# 
# To better match market prices and account for the volatility smile/skew, we can use:
# 
#  - **Local Volatility Models - Make volatility a deterministic function of spot and time:**
#     $$\frac{dS_t}{S_t} = \mu dt + \sigma(S_t,t) dW_t$$
#     - Dupire's formula provides a way to compute local volatility from market prices
#     - Ensures consistent pricing with vanilla options
# 
#  - **Stochastic Volatility Models - Make volatility follow its own stochastic process:**
#     $$\begin{align}
#     \frac{dS_t}{S_t} &= \mu dt + \sqrt{v_t} dW^1_t \\
#     dv_t &= \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW^2_t
#     \end{align}$$
#     - Heston model is a popular example
#     - Adds correlation between spot and vol ($\rho dt = dW^1_t dW^2_t$)
#     - Better captures dynamics of implied volatility surface
# 
# **Functional Framework - The Heston model as a map:**
#       $$\begin{align}
#       \mathcal{H}: \mathbb{R}^7 &\rightarrow \mathbb{R} \\
#       (\kappa, \theta, \xi, \rho, v_0, K, T) &\mapsto \text{Price}
#       \end{align}$$
#       where $K$ is the strike price and $T$ is time to maturity
#   
# 
# 
#  These models help trading desks price exotic options more accurately and manage risk better.


from scipy.stats import norm
# Heston model parameters
kappa = 2.0  # Mean reversion speed
theta = .08  # Long-run variance
xi = 1.4  # Vol of vol
rho = -0.7  # Correlation
v0 = (flat_bs_vol/100)**2  # Initial variance

# Risk-free rate and time to maturity
r = 0.02
T = 1.0

# Simulation parameters
n_paths = 10000
n_steps = 252  # Daily steps
dt = T/n_steps

# Generate correlated Brownian motions
dW1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

# Simulate Heston paths
S = np.zeros((n_paths, n_steps + 1))
v = np.zeros((n_paths, n_steps + 1))
S[:, 0] = spot
v[:, 0] = v0

for t in range(n_steps):
    v[:, t+1] = np.maximum(v[:, t] + kappa*(theta - v[:, t])*dt + xi*np.sqrt(v[:, t])*dW2[:, t], 0)
    S[:, t+1] = S[:, t] * np.exp((r - 0.5*v[:, t])*dt + np.sqrt(v[:, t])*dW1[:, t])

# Calculate Heston implied volatilities
heston_vols = []
for K in strikes:
    # Calculate call option prices
    payoffs = np.maximum(S[:, -1] - K, 0)
    heston_price = np.exp(-r*T) * np.mean(payoffs)
    
    # Find implied vol using Newton-Raphson
    def bs_price(sigma):
        d1 = (np.log(spot/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return spot*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def vega(sigma):
        d1 = (np.log(spot/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return spot*np.sqrt(T)*norm.pdf(d1)
    
    sigma = flat_bs_vol/100  # Initial guess
    for _ in range(50):
        diff = bs_price(sigma) - heston_price
        if abs(diff) < 1e-5:
            break
        sigma = sigma - diff/vega(sigma)
    
    heston_vols.append(sigma*100)

# Create new figure
fig = go.Figure()

# Add market skew line
fig.add_trace(
    go.Scatter(
        x=strikes,
        y=market_skew,
        name='Market Volatility Skew',
        line=dict(color='#00b3ff', width=3)
    )
)

# Add flat BS skew line
fig.add_trace(
    go.Scatter(
        x=strikes,
        y=[flat_bs_vol]*len(strikes),
        name='Black-Scholes Volatility',
        line=dict(color='#ff9100', width=3, dash='dash')
    )
)

# Add Heston skew line
fig.add_trace(
    go.Scatter(
        x=strikes,
        y=heston_vols,
        name='Heston Model Skew',
        line=dict(color='#00ff00', width=3, dash='dot')
    )
)

# Update layout
fig.update_layout(
    title='Market, Black-Scholes, Heston Skew',
    xaxis_title='Strike Price',
    yaxis_title='Implied Volatility (%)',
    width=900,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.95
    ),
    font=dict(color='white')
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='darkgray')

fig.show()



# ---


# ### 3.) Solutions to Pricing Functionals
# 
# 
# Establishing Pricing PDEs:
# - [How to Find the Black-Scholes-Merton Partial Differential Equation](https://youtu.be/2iClLEfXuqA?si=AgjS1PE1-uyuf5sh)
# 
#   **Pricing Function:** For a European call option with strike K and maturity T:
#   $$C(S_t,v_t,t) = \mathbb{E}_t\left[e^{-r(T-t)}\max(S_T - K, 0)\right]$$
#   where $S_T$ follows the Heston dynamics above. The expectation can be computed:
#   1. Using Monte Carlo simulation of paths (as shown in code below)
#   2. Using semi-analytical Fourier methods with characteristic function
#   3. Using finite difference methods to solve the PDE:
#   $$\begin{align}
#   \frac{\partial C}{\partial t} + \frac{1}{2}vS^2\frac{\partial^2 C}{\partial S^2} &+ \rho\xi vS\frac{\partial^2 C}{\partial S\partial v} + \frac{1}{2}\xi^2v\frac{\partial^2 C}{\partial v^2} \\
#   &+ rS\frac{\partial C}{\partial S} + \kappa(\theta-v)\frac{\partial C}{\partial v} - rC = 0
#   \end{align}$$
# 
#  **Remark:** Desks don't just use the price generated by these models, nor do they blindly hedge with the model greeks - we know what assumptions are associated with which models and adjust our decision making in uncertainty accordingly in a discretionary capacity usually based on desk heuristics!
# 
# 
# Pricing Functional Solutions:
# 
# - Closed-Form Solutions
# 
# - Tricks (i.e. FFT)
# 
# - Partial Differential Equation Schemes (Finite-Differences)
# 
# - Stochastic Process Simulation 
# 
# #### <u>Pricing Notation from Horvath et al. 2019</u>
# 
# The market dictates the price for a set of contracts $\zeta$ a vector of instruments to price $\mathbb{R}^m, m \in \mathbb{N}$
# 
# Given a parameter set $\theta \in \Theta \subset \mathbb{R}^n$ and model framework $\mathcal{M}$ we can produce a price for the given framework by the pricing map $P : \mathcal{M}(\theta, \zeta) \mapsto \mathbb{R}^m$
# 
# $$P(\mathcal{M}(\theta, \zeta)) \mapsto \mathbb{R}^m$$
# 
# For our Heston model:
# 
# $$\theta = (\kappa, \theta, \xi, \rho, v_0) \quad \mathcal{M} = \mathcal{H} \text{ (Heston)}$$
# $$P(\mathcal{H}(\theta, \zeta)) \mapsto \mathbb{R}^m$$
# 
# However, if we use simulation or some sort of numerical PDE solver we have
# 
# $$\tilde{P}(\mathcal{H}(\theta, \zeta)) \approx P(\mathcal{H}(\theta, \zeta))$$


# ---


# ### 4.) Model Calibration to an Implied Volatility Surface
# 
# Once we determine *how* we will produce a price (Closed-Form, Quasi-Closed-Form, Simulation, Numerical PDE)
# 
# We can go about finding the values of the parameters to use $\theta$ by calibrating our model to a market volatility surface
# 
# This is equivalent to solving the following minimization problem which yields the parameters to use
# 
# $$\theta^* = argmin_{\theta \in \Theta} \delta (\tilde{P}(\mathcal{H}(\theta, \zeta)), \mathcal{P}^{MKT}(\zeta))$$
# 
# Where $\delta$ is some measure of distance between the model price for instruments $\zeta$ given by parameters $\theta$ and the corresponding market prices.
# 
# **Don't get lost in the notation!**
# 
# $\theta^*$ just tells us the values of the Heston parameters $(\kappa, \theta, \xi, \rho, v_0, K, T)$ to use based on market prices!
# 
# #### <u>Efficiency Concerns</u>
# 
# - Simulation MC schemes can be SLOW - fitting is itterative meaning we have to iteratively simulate prices...
# 
# - Price/IVol space - we can fit in the price space and conver to BS Ivols one to one
# 
# - Some schemes require full MC simulation (for example, a Volterra processes)
# 
# - We may approximate these functionals offline with neural networks
# 
# There is a lot to this process - I will have to punt on this topic and save it for another video!  
# 
# For the rest of the notebook herein, we can assume that we've found the parameters $\theta^*$ as $\hat{\theta}$ to use.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import plotly.io as pio # Import plotly.io to save the plot

# Data for the volatility surface - precompute meshgrid
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

def heston_paths(S0, v0, kappa, theta, xi, rho, r, T, N, M):
    """
    Simulate Heston paths using Euler discretization
    S0: initial stock price
    v0: initial variance
    kappa: mean reversion speed
    theta: long-run variance
    xi: volatility of variance
    rho: correlation
    r: risk-free rate
    T: time horizon
    N: number of time steps
    M: number of paths
    """
    dt = T/N
    
    # Initialize arrays
    S = np.zeros((N+1, M))
    v = np.zeros((N+1, M))
    S[0] = S0
    v[0] = v0
    
    # Generate correlated random numbers
    Z1 = np.random.standard_normal((N, M))
    Z2 = rho * Z1 + np.sqrt(1-rho**2) * np.random.standard_normal((N, M))
    
    # Simulate paths
    for i in range(N):
        # Ensure variance remains non-negative
        v[i+1] = np.maximum(v[i] + kappa*(theta - v[i])*dt + xi*np.sqrt(v[i]*dt)*Z1[i], 0)
        S[i+1] = S[i] * np.exp((r - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*Z2[i])
    
    return S, v

def bs_implied_vol(S0, K, T, r, price, call=True):
    """Calculate BS implied volatility using Newton-Raphson"""
    def bs_price(sigma):
        # Black-Scholes price formula
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if call:
            return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    
    sigma = 0.3  # Initial guess for volatility
    for _ in range(100): # Max 100 iterations for convergence
        price_diff = bs_price(sigma) - price
        if abs(price_diff) < 1e-5: # Convergence criterion
            return sigma
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        vega = S0*np.sqrt(T)*norm.pdf(d1) # Vega for Newton-Raphson
        if vega == 0: # Avoid division by zero
            return sigma
        sigma = sigma - price_diff/vega
        if sigma <= 0: # Ensure sigma remains positive
            sigma = 0.01
    return sigma # Return the best estimate if not converged

# Fixed parameters for plot
S0 = 100
v0 = 5.208/100  # Initial variance (converted from percentage)
kappa = 2.0
theta = 0.04
xi = 0.3
rho = -0.7
r = 0.02

# Calculate Heston implied volatility surface
heston_vols = np.zeros_like(market_vols)
N_paths = 100000 # Number of simulation paths

for i, T in enumerate(maturities):
    N_steps = int(T * 252)  # Daily steps (assuming 252 trading days in a year)
    if N_steps == 0: # Ensure at least one step for very short maturities
        N_steps = 1
    for j, K in enumerate(strikes):
        # Simulate paths
        S, v = heston_paths(S0, v0, kappa, theta, xi, rho, r, T, N_steps, N_paths)
        
        # Calculate option price (for a call option)
        payoffs = np.maximum(S[-1] - K, 0)
        price = np.exp(-r*T) * np.mean(payoffs)
        
        # Convert to implied vol
        try:
            impl_vol = bs_implied_vol(S0, K, T, r, price) * 100  # Convert to percentage
            heston_vols[i,j] = impl_vol
        except Exception as e:
            print(f"Could not calculate implied volatility for T={T}, K={K}: {e}")
            heston_vols[i,j] = np.nan # Assign NaN if calculation fails
# Create figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=('Heston Implied Volatility Surface', 'Market Implied Volatility Surface')
)

# Add Heston surface
fig.add_trace(
    go.Surface(x=X, y=Y, z=heston_vols, colorscale='Viridis', opacity=0.7, showscale=True),
    row=1, col=1
)

# Add Market surface
fig.add_trace(
    go.Surface(x=X, y=Y, z=market_vols, colorscale='Viridis', opacity=0.7, showscale=True),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='Heston vs Market Implied Volatility Surfaces',
    scene=dict(
        xaxis_title='Strike Price',
        yaxis_title='Time to Maturity (Years)', 
        zaxis_title='Implied Volatility (%)',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    scene2=dict(
        xaxis_title='Strike Price',
        yaxis_title='Time to Maturity (Years)',
        zaxis_title='Implied Volatility (%)',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)', range=[np.min(market_vols)-2, np.max(market_vols)+2]),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    width=1000,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.show()



# ---


# ### 5.) Exotic Pricing Functional Approximation
# 
# Let $B$ be the barrier level and $\tau$ be the first hitting time of $S_t$ to $B$
#   
#  For an up-and-out barrier option with maturity $T$:
#   
#  $V(S_0, v_0; \hat{\theta}) = \mathbb{E}^\mathbb{Q}\left[e^{-rT}(S_T - K)^+\mathbf{1}_{\{\tau > T\}} \mid S_0, v_0\right]$
# 
#  We can approximate the price of this price using monte carlo simulation.
# 
# How do we *know* simulation works? Stay tuned for a video where we will discuss *why* simulation methods work to approximate these values!


def simulate_barrier_option(num_paths=1000, S0=100, K=100, B=120, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04, sigma=0.3, rho=-0.7):
    """
    Simulate up-and-out barrier option price using Heston model
    """
    dt = 1/252  # Daily steps
    N = int(T/dt)
    
    # Initialize arrays
    S = np.zeros((num_paths, N+1))
    v = np.zeros((num_paths, N+1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated random numbers
    Z1 = np.random.normal(0, 1, (num_paths, N))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (num_paths, N))
    
    # Simulate paths
    for t in range(N):
        S[:, t+1] = S[:, t] * np.exp((r - 0.5*v[:, t])*dt + np.sqrt(v[:, t]*dt)*Z1[:, t])
        v[:, t+1] = np.maximum(v[:, t] + kappa*(theta - v[:, t])*dt + sigma*np.sqrt(v[:, t]*dt)*Z2[:, t], 0)
    
    # Calculate payoff (up-and-out barrier option)
    max_prices = np.maximum.accumulate(S, axis=1)
    barrier_not_hit = (max_prices[:, -1] < B)
    payoff = np.maximum(S[:, -1] - K, 0) * barrier_not_hit
    
    # Discount payoff
    price = np.exp(-r*T) * np.mean(payoff)
    
    return price, S

def plot_barrier_paths(S, B, K, price, num_display=100):
    """
    Plot sample paths with barrier level
    """
    plt.figure(figsize=(12, 6))
    
    # Get final values and barrier hits
    final_values = S[:, -1]
    max_values = np.maximum.accumulate(S, axis=1)[:, -1]
    barrier_hit = max_values >= B
    itm = final_values > K
    
    # Plot paths with different colors based on conditions
    for i in range(min(len(S), num_display)):
        if barrier_hit[i]:
            plt.plot(S[i].T, color='red', alpha=0.1)
        elif itm[i]:
            plt.plot(S[i].T, color='green', alpha=1.0)
        else:
            plt.plot(S[i].T, color='yellow', alpha=0.1)
    
    plt.axhline(y=B, color='r', linestyle='--', label='Barrier')
    plt.axhline(y=K, color='blue', linestyle='--', label='Strike')
    plt.grid(True, alpha=0.3)
    plt.title(f'Sample Price Paths (Number of Paths: {len(S)}, Option Price: ${price:.4f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    
    # Make legend background transparent
    legend = plt.legend(facecolor='none', edgecolor='none')
    plt.setp(legend.get_texts(), color='white')
    
    # Set transparent background
    plt.gca().set_facecolor('none')
    plt.gcf().patch.set_alpha(0.0)
    
    plt.show()

def barrier_option_widget():
    """
    Create interactive widget for barrier option simulation
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='400px')
    
    paths_slider = widgets.IntSlider(
        value=1000,
        min=10,
        max=10000,
        step=10,
        description='Number of Paths:',
        style=style,
        layout=layout
    )
    
    def update(paths):
        price, S = simulate_barrier_option(num_paths=paths)
        plot_barrier_paths(S, B=120, K=100, price=price)
    
    return widgets.interactive(update, paths=paths_slider)

# Display the interactive widget
display(barrier_option_widget())



def simulate_vanilla_call(S0=100, K=100, r=0.05, sigma=0.2, T=1, num_paths=1000, num_steps=252):
    """
    Simulate vanilla call option price using Monte Carlo
    """
    dt = T/num_steps
    S = np.zeros((num_paths, num_steps+1))
    S[:, 0] = S0
    
    # Generate paths
    for t in range(num_steps):
        Z = np.random.standard_normal(num_paths)
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Calculate payoffs
    payoffs = np.maximum(S[:, -1] - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, S

def plot_vanilla_paths(S, K=100, price=None):
    """
    Plot simulated price paths for vanilla call option
    """
    plt.figure(figsize=(12, 6))
    
    # Plot paths
    for i in range(len(S)):
        if S[i, -1] > K:  # ITM paths
            plt.plot(S[i].T, color='green', alpha=0.1)
        else:
            plt.plot(S[i].T, color='yellow', alpha=0.1)
    
    plt.axhline(y=K, color='yellow', linestyle='--', label='Strike')
    plt.grid(True, alpha=0.3)
    plt.title(f'Sample Price Paths (Number of Paths: {len(S)}, Option Price: ${price:.4f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    
    # Make legend background transparent
    legend = plt.legend(facecolor='none', edgecolor='none')
    plt.setp(legend.get_texts(), color='white')
    
    # Set transparent background
    plt.gca().set_facecolor('none')
    plt.gcf().patch.set_alpha(0.0)
    
    plt.show()

def vanilla_call_widget():
    """
    Create interactive widget for vanilla call option simulation
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='400px')
    
    paths_slider = widgets.IntSlider(
        value=1000,
        min=10,
        max=10000,
        step=10,
        description='Number of Paths:',
        style=style,
        layout=layout
    )
    
    def update(paths):
        price, S = simulate_vanilla_call(num_paths=paths)
        plot_vanilla_paths(S, K=100, price=price)
    
    return widgets.interactive(update, paths=paths_slider)

# Display the interactive widget
display(vanilla_call_widget())



# #### <u>Price Adjustments</u>
# 
# After calculating the base exotic option price using our calibrated model, several adjustments are typically made:
# 
#  Counterparty Risk Adjustment
# - Credit Value Adjustment (CVA) accounts for counterparty default risk
# - Debt Value Adjustment (DVA) accounts for own default risk 
# - These adjustments reduce the price when default risk is higher
# 
# Liquidity Adjustment  
# - Exotic options are less liquid than vanilla options
# - Wider bid-ask spreads reflect higher hedging costs
# - Size of adjustment depends on:
#   - Option complexity
#   - Market conditions
#   - Position size
# 
# Profit Margin
# - Trading desks add profit margin to cover:
#   - Operating costs
#   - Capital charges
#   - Return on equity requirements
# - Typical margins range from 2-10% depending on:
#   - Client relationship
#   - Competition
#   - Market conditions
# 
# The final quote sent to clients incorporates all these adjustments on top of the theoretical price.
#  Let $Q$ be the final quote sent to clients. Then:
# 
#  $Q = P + CVA + DVA + L + M$
# 
#  where:
#  - $P$ is the theoretical price from the calibrated model
#  - $CVA$ is the credit value adjustment
#  - $DVA$ is the debt value adjustment  
#  - $L$ is the liquidity adjustment
#  - $M$ is the profit margin
# 
# **You're then ready to fire off that quote on IB**


# ---


# ### 6.) Closing Thoughts and Future Topics
# 
# #### <u>General Recipe used Herein</u>
# 
#  1.) Model Selection ($\mathcal{M}$, $\zeta$) $\rightarrow$ Heston
#  
#  2.) Determine Vanilla Pricing Scheme $\rightarrow$ Simulation
#  
#  3.) Calibrate Model to Market Volatility Surface $\rightarrow$ Assume we have $\hat{\theta}$
#  
#  4.) Determine Exotic Option Pricing Scheme $\rightarrow$ Simulation
#  
#  5.) Apply Price Adjustments $\rightarrow$ Counterparty Risk, Liquidity, Profit
#  
#  6.) Send Quote $\rightarrow$ IB (Instant Bloomberg)
# 
# ---
# 
# <u>Pricing Theory</u>
# 
# Diving deeper into stochastic calculus and different model frameworks, we can consider
# 
# - Different forms of stochastic differential equations
# - Different discretization schemes for analytically intractable equations
# - Fancy tricks for approximate or quasi-closed-form solutions (i.e. Heston FFT)
# 
# <u>Efficiency</u>
# 
# Calibration can be **terribly** slow depending on our methodology for developing a price, we can consider
# 
# - Models with approximate or asymptotic solutions (for example, SABR)
# - Pricing functional approximation using neural networks
# 
# <u>Simulation</u>
# - *Why* does simulation work? (LLN)
# 
# <u>Book Management</u>
# - How do we go about making adjustments to our simulated (model) price?
# - How can we dynamically hedge our exposures if our model is wrong?
# - How does our opinion on exposures (for example, volatility) impact our hedging?


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

