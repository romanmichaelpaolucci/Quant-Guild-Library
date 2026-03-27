# ### Beyond Black-Scholes: Heston Stochastic Volatility Model and Fast Fourier Transforms
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
# *Applications Sitting on a Trading Desk:*
# - [Trading with the Black-Scholes Implied Volatility Surface](https://youtu.be/YH0tWpBaKGs)
# 
# - [How to Price Exotic Options](https://youtu.be/hsot26myYYM)
# 
# *Statistic Basis for Pricing:*
# - [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)
# 
# - [Monte Carlo Simulation and Black-Scholes for Pricing Options](https://youtu.be/-1RYvajksjQ)
# 
# *Why the Expectation is Sufficient for Pricing and Problems in Practice:*
# 
# - [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)
# 
# - [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)
# 
# - [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
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

# Generate grid points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Calculate bivariate normal PDF
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])
inv_sigma = np.linalg.inv(sigma)
det_sigma = np.linalg.det(sigma)

Z = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        point = np.array([X[i,j], Y[i,j]])
        Z[i,j] = (1/(2*np.pi*np.sqrt(det_sigma))) * np.exp(-0.5 * point.dot(inv_sigma).dot(point))

# Calculate characteristic function
u = np.linspace(-2, 2, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)

# Characteristic function for bivariate normal
CF_real = np.zeros_like(U)
CF_imag = np.zeros_like(U)
for i in range(len(u)):
    for j in range(len(v)):
        t = np.array([U[i,j], V[i,j]])
        CF = np.exp(1j * t.dot(mu) - 0.5 * t.dot(sigma).dot(t))
        CF_real[i,j] = CF.real
        CF_imag[i,j] = CF.imag

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=('Bivariate Normal PDF', 'Characteristic Function (Real Part)')
)

# Add PDF surface plot
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

# Add characteristic function surface plot
fig.add_trace(
    go.Surface(
        x=U,
        y=V,
        z=CF_real,
        colorscale='RdBu',
        opacity=0.7,
        showscale=True
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='Bivariate Normal Distribution and its Characteristic Function',
    width=1600,
    height=800,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='PDF',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
    ),
    scene2=dict(
        xaxis_title='u',
        yaxis_title='v',
        zaxis_title='Re(CF)',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
    )
)

fig.show()



# ---


# ### Sections
# 
# #### 1.) 🏷️ Option Pricing
# 
# - Risk-Neutral Discounted Expected Payoffs
# 
# - Black-Scholes Model
# 
# - Heston Model
# 
# #### 2.) 🌊 Fourier Transforms
# 
# - Discrete Fourier Transform
# 
# - Fast Fourier Transform
# 
# - Fourier Inversion Theorem
# 
# #### 3.) 🃏 Characteristic Functions
# 
# - Characteristic Functions
# 
# - Fourier Transform of Random Variable $\mapsto$ Characteristic Function
# 
# - Inverse Fourier Transform Characteristic Function $\mapsto$ Random Variable
# 
# #### 4.) 🎯 Carr-Madan
# 
# - Damping Trick
# 
# - Fourier Transform
# 
# - Characteristic Function
# 
# - Fourier Inversion Theorem
# 
# - Key Equation (Undamp and Correct)
# 
# #### 5.) 📜 General Recipe
# 
# - Discretize Frequency Domain
# 
# - Construct Complex-Valued Vector
# 
# - Apply Inverse FFT
# 
# - Undamp Correct
# 
# #### 6.) 💭 Closing Thoughts and Future Topics


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


# ##### 1.) 🏷️ Option Pricing
# 
# Option contracts give the holder the right but not obligation to buy (or sell) an underlying asset at some point in the future at a predetermined price.  These contracts themselves have a value at expiration that is easily determined, but what should we pay *today* for this contract?
# 
#  We model stock prices as a random variable, option prices are functions of this underlying randomness so they are also random
#  
#  
#  In the face of randomness the best we can do to model uncertainty in the outcome of said random variable is to use the *expectation* 
# 
#  $$C(K,T) = e^{-rT}\mathbb{E}^{\mathbb{Q}}\left[\max(S_T - K, 0)\right]$$
#  
#  where $\mathbb{Q}$ is the risk-neutral measure. We use the risk-neutral measure because under no-arbitrage, 
#  the discounted price process must be a martingale under $\mathbb{Q}$, ensuring the price today equals 
#  the expected discounted future payoff.
#  
# Applying the definition of the *expectation* we get the following equation
#  
#  $$C(K,T) = e^{-rT}\int_{K}^{\infty} (S_T - K)f_{\mathbb{Q}}(S_T)dS_T$$
#  
# How the heck are we going to solve this analytically, particularly for the case of more complicated model frameworks?
# 
# We can solve this analytically for a Black-Scholes model, but what if we wanted to price options consistent with a market surface?
# 
# Think of something like the Heston model - numerical techniques (simulating paths) can be *slow*
# 
# What if there were a different way to represent the random variable (option price)?


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1.0   # Time to maturity
N = 1000  # Number of time steps

# Generate time points
t = np.linspace(0, T, N)
dt = T/N

# Generate one sample path
dW = np.random.normal(0, np.sqrt(dt), N)
S = np.zeros(N)
S[0] = S0

for i in range(1, N):
    S[i] = S[i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW[i-1])

# Get final stock price and option payoff
final_S = S[-1]
final_payoff = max(final_S - K, 0)

# Create stock price points for payoff diagram
S_range = np.linspace(0.5*S0, 1.5*S0, 200)
payoff = np.maximum(S_range - K, 0)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('GBM Sample Path', f'European Call Option Payoff (Value at T: {final_payoff:.2f})')
)

# Plot GBM path
fig.add_trace(
    go.Scatter(x=t, y=S, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)

# Plot option payoff
fig.add_trace(
    go.Scatter(x=S_range, y=payoff, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=2
)

# Add vertical line at final stock price
fig.add_vline(x=final_S, line_width=1, line_dash="dash", line_color="white", row=1, col=2)

# Update layout
fig.update_layout(
    width=1000,
    height=400,
    showlegend=False,
    title_text='GBM Path and European Call Option Payoff',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update x-axes
fig.update_xaxes(
    title_text='Time',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

# Update y-axes
fig.update_yaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_yaxes(
    title_text='Option Payoff',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

fig.show()


# #### Model Framework
# 
# Random variables are entirely defined by their distributions.  For pricing European options, options that only depend on terminal stock price at time $T$ it is sufficient to know the terminal stock price distribution for simulating prices rather than simulate sample paths.
# 
# **Black-Scholes**
# 
#  $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
#  
# Assumes a solvable stochastic differential equation for the underlying: *Geometric Brownian Motion*
# 
# The Black-Scholes framework is solvable
# 
# - Analytically (Closed-Form Solution Available to the Black-Scholes Equation)
# 
# - Numerically (PDE Methods applied to the Black-Scholes Equation)
# 
# - Numerically (Simulation Methods, Law of Large Numbers)
# 
# *Problem:* Can't quote consistent prices based on a market volatility surface from a constant volatility assumption


# Calculate lognormal distribution parameters
mu = (r - 0.5*sigma**2)*T  # Drift term
std = sigma*np.sqrt(T)      # Standard deviation

# Generate price points for distribution
S_dist = np.linspace(0.5*S0, 1.5*S0, 200)

# Calculate lognormal PDF
log_returns = np.log(S_dist/S0)
pdf = np.exp(-(log_returns - mu)**2 / (2*std**2)) / (S_dist*std*np.sqrt(2*np.pi))

# Generate one random draw from the distribution
final_S = S0 * np.exp(np.random.normal(mu, std))
payoff = np.maximum(S_dist - K, 0)
final_payoff = max(final_S - K, 0)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('GBM Terminal Distribution', f'European Call Option Payoff (Value at T: {final_payoff:.2f})')
)

# Plot distribution
fig.add_trace(
    go.Scatter(x=S_dist, y=pdf, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)

# Plot option payoff
fig.add_trace(
    go.Scatter(x=S_dist, y=payoff, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=2
)

# Add vertical line at final stock price
fig.add_vline(x=final_S, line_width=1, line_dash="dash", line_color="white", row=1, col=2)

# Update layout
fig.update_layout(
    width=1000,
    height=400,
    showlegend=False,
    title_text='GBM Terminal Distribution and Option Payoff',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update x-axes
fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_xaxes(
    title_text='Stock Price',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

# Update y-axes
fig.update_yaxes(
    title_text='Probability Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_yaxes(
    title_text='Option Payoff',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=2
)

fig.show()



# **Heston (Stochastic Volatility)**
# 
#  
#  $$
#   dS_t = rS_t dt + \sqrt{v_t}S_t dW^1_t \\
#   dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW^2_t \\
#   dW^1_t dW^2_t = \rho dt
#  $$
#  
# 
# 
# Assumes volatility is not constant but rather follows a mean-reverting process
# 
# Captures more consistent pricing dynamics observed empirically (skew, leverage effect, etc.)
# 
# *Problem:* 
# - We don't have the terminal distribution of the stock price for this more complicated model (simulation means we need sample paths)
# 
# - We can't explicitly solve for the option price, unlike Black-Scholes


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
r = 0.05  # Risk-free rate
v0 = 0.04  # Initial variance
kappa = 2.0  # Mean reversion speed
theta = 0.04  # Long-run variance
xi = 0.3  # Volatility of variance
rho = -0.7  # Correlation between stock and variance
T = 1.0   # Time to maturity
N = 1000  # Number of time steps

# Generate time points
t = np.linspace(0, T, N)
dt = T/N

# Generate correlated Brownian motions
dW1 = np.random.normal(0, np.sqrt(dt), N)
dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), N)

# Initialize arrays
S = np.zeros(N)
v = np.zeros(N)
S[0] = S0
v[0] = v0

# Generate paths
for i in range(1, N):
    v[i] = v[i-1] + kappa * (theta - v[i-1]) * dt + xi * np.sqrt(v[i-1]) * dW2[i-1]
    v[i] = max(v[i], 0)  # Ensure variance stays positive
    S[i] = S[i-1] * np.exp((r - 0.5*v[i-1])*dt + np.sqrt(v[i-1])*dW1[i-1])

# Get final values
final_S = S[-1]
final_payoff = max(final_S - K, 0)

# Create stock price points for payoff diagram
S_range = np.linspace(0.5*S0, 1.5*S0, 200)
payoff = np.maximum(S_range - K, 0)

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Heston Stock Price Path', 'Variance Process', 'Option Payoff'),
    specs=[[{}, {}], [{"colspan": 2}, None]]
)

# Plot stock price path
fig.add_trace(
    go.Scatter(x=t, y=S, name='Stock Price', line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)

# Plot variance process
fig.add_trace(
    go.Scatter(x=t, y=v, name='Variance', line=dict(color='rgba(255, 165, 0, 0.6)')),
    row=1, col=2
)

# Plot option payoff
fig.add_trace(
    go.Scatter(x=S_range, y=payoff, name='Payoff', line=dict(color='rgba(0, 255, 0, 0.6)')),
    row=2, col=1
)

# Add vertical line at final stock price
fig.add_vline(x=final_S, line_width=1, line_dash="dash", line_color="white", row=2, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    title_text='Heston Model Simulation',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for row in [1, 2]:
    for col in [1, 2]:
        if row == 2 and col == 2:
            continue
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=row, col=col
        )

fig.show()


# What if there were a *different* way to represent the random variable that is an option's price under the Heston model?
# 
# Perhaps we could then we could solve for this value in (quasi)-closed-form?


# ---


# ##### 2.) 🌊 Fourier Transforms
# 
# ##### The Fourier Transform maps a function from the time domain to the frequency domain
#  
#  $$F(\omega) = \int_{-\infty}^{\infty} f(t)e^{-i\omega t}dt$$
#  
#  - $f(t)$ is the input signal in the *time domain*
#  
#  - $F(\omega)$ is the output signal in the *frequency domain*
#  
#  - $e^{-i\omega t}$ is the complex exponential, the basis function of the transform
# 
# Great for formal proof, numerically, however, we can't operate in continuous space...
# 
# ##### Discrete Fourier Transform
# 
# $$X_k = \sum_{n=0}^{N-1} x_n e^{\frac{-i2 \pi k n}{N}}$$
# 
# The Discrete Fourier Transform (DFT) maps an input signal from the time domain to an output signal in the frequency domain in $O(N^2)$
# 
# - $x_n$ is the input signal in the *time domain*
# 
# - $X_k$ is the output signal in the *frequency domain*
# 
# - $e^{\frac{-i2 \pi k n}{N}}$ is the complex exponential, the basis function of the transform


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate time points
t = np.linspace(0, 1, 1000)

# Create signals with different frequencies but same amplitude
signal1 = np.sin(2 * np.pi * 1 * t)  # 1 Hz
signal2 = np.sin(2 * np.pi * 2 * t)  # 2 Hz
signal3 = np.sin(2 * np.pi * 3 * t)  # 3 Hz

# Compute DFT for each signal
freq1 = np.fft.fft(signal1)
freq2 = np.fft.fft(signal2)
freq3 = np.fft.fft(signal3)

# Get frequency axis
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# Create subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('1 Hz Signal', '1 Hz Frequency Spectrum',
                   '2 Hz Signal', '2 Hz Frequency Spectrum',
                   '3 Hz Signal', '3 Hz Frequency Spectrum')
)

# Plot time domain signals and their frequency spectrums
fig.add_trace(
    go.Scatter(x=t, y=signal1, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(freq1)[:len(freqs)//2]/len(t),
               line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=t, y=signal2, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(freq2)[:len(freqs)//2]/len(t),
               line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=t, y=signal3, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(freq3)[:len(freqs)//2]/len(t),
               line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=3, col=2
)

# Update layout
fig.update_layout(
    width=1000,
    height=800,
    showlegend=False,
    title_text='Discrete Fourier Transform Example',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update all xaxes
for i in range(1, 4):
    fig.update_xaxes(
        title_text='Time (s)' if i == 3 else None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )
    fig.update_xaxes(
        title_text='Frequency (Hz)' if i == 3 else None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 4],  # Limit x-axis range to better show peaks
        row=i, col=2
    )

# Update all yaxes
for i in range(1, 4):
    fig.update_yaxes(
        title_text='Amplitude' if i == 2 else None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[-1.1, 1.1],  # Fix y-axis range for time domain
        row=i, col=1
    )
    fig.update_yaxes(
        title_text='Magnitude' if i == 2 else None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 0.6],  # Fix y-axis range for frequency domain
        row=i, col=2
    )

fig.show()



# ##### Fast Fourier Transform
#  
#  $$X_k = \sum_{n=0}^{N-1} x_n e^{\frac{-i2 \pi k n}{N}}$$
#  
#  The Fast Fourier Transform (FFT) is an efficient algorithm for computing the DFT that reduces complexity from $O(N^2)$ to $O(N \log N)$ by:
# 
#  - Recursively dividing the transform into smaller sub-transforms
#  - Exploiting symmetries in the complex exponentials
#  - Using a divide-and-conquer approach
# 
#  Key properties:
#  - Input size must be a power of 2 ($N = 2^m$)
#  - Widely used in signal processing, numerical computing, and financial applications
#  - Most common variant is the Cooley-Tukey FFT algorithm


# ##### Inverse Fast Fourier Transform
#   
#   $$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{\frac{i2 \pi k n}{N}}$$
#   
#   The Inverse Fast Fourier Transform (IFFT) reverses the FFT operation, transforming from frequency domain back to time domain:
# 
#   - Uses same efficient $O(N \log N)$ algorithm as FFT
#   - Only difference is positive exponent and $\frac{1}{N}$ scaling factor
#   - Preserves all properties of FFT including power of 2 input size requirement
# 
#   The IFFT allows us to recover the original signal after performing frequency domain operations
# 


# Create sample signal
t = np.linspace(0, 1, 1024)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t)

# Compute FFT
fft_result = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), t[1]-t[0])

# Compute IFFT to recover original signal
recovered_signal = np.fft.ifft(fft_result)

# Create subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Original Signal',
                                  'Frequency Domain (FFT)', 
                                  'Recovered Signal (IFFT)'))

# Plot original signal
fig.add_trace(
    go.Scatter(x=t, y=signal, line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=1, col=1
)

# Plot frequency domain
fig.add_trace(
    go.Scatter(x=freq[:len(freq)//2],
               y=np.abs(fft_result)[:len(freq)//2]/len(t),
               line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=2, col=1
)

# Plot recovered signal
fig.add_trace(
    go.Scatter(x=t, y=np.real(recovered_signal),
               line=dict(color='rgba(0, 191, 255, 0.6)')),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=800,
    showlegend=False,
    title_text='FFT Analysis Example',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update all xaxes
for i in range(1, 4):
    fig.update_xaxes(
        title_text='Time (s)' if i in [1,3] else 'Frequency (Hz)',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[0, 15] if i == 2 else None,
        row=i, col=1
    )

# Update all yaxes
for i in range(1, 4):
    fig.update_yaxes(
        title_text='Amplitude' if i == 2 else None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[-1.6, 1.6] if i in [1,3] else [0, 0.6],
        row=i, col=1
    )

fig.show()



# ---


# ##### 3.) 🃏 Characteristic Functions
# 
# 
# #### Fourier Transform: Characteristic Function of a Random Variable
# 
#  The characteristic function $\phi_X(t) = \mathbb{E}[e^{itX}]$ transforms a probability distribution into the frequency domain:
#  
#   $$\phi_X(t) = \mathbb{E}[e^{itX}] = \int_{-\infty}^{\infty} e^{itx} f_X(x) dx$$
# 
# There is a sign difference, however these mathematical objects (characteristic function, Fourier transform) are roughly interchangable
# 
#  Key properties of characteristic functions:
# 
#  - Uniquely determines the probability distribution of a random variable
#  - Always exists for any random variable (unlike moment generating functions)
#  - Provides one-to-one correspondence with PDF via Fourier transform
#  - Enables derivation of all distribution moments
#  - Useful for distributions lacking closed-form density functions
#  - Simplifies calculations through properties like sum of independent variables
# 
#  The characteristic function serves as a bridge between probability theory and frequency analysis
# 
# 


# Create t values for characteristic function
t = np.linspace(-5, 5, 1000)

# Normal distribution characteristic function
# For N(μ,σ²), φ(t) = exp(iμt - σ²t²/2)
mu = 2
sigma = 1
cf = np.exp(1j * mu * t - (sigma**2 * t**2) / 2)

# Calculate normal distribution PDF
normal_pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(t - mu)**2 / (2*sigma**2))

# Create subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Normal Distribution PDF',
                                  'Real Part of Characteristic Function',
                                  'Imaginary Part of Characteristic Function'))

# Plot normal distribution
fig.add_trace(
    go.Scatter(x=t, y=normal_pdf, line=dict(color='rgba(255, 255, 255, 0.6)')),
    row=1, col=1
)

# Plot real part
fig.add_trace(
    go.Scatter(x=t, y=np.real(cf), line=dict(color='rgba(57, 255, 20, 1)')),
    row=2, col=1
)

# Plot imaginary part
fig.add_trace(
    go.Scatter(x=t, y=np.imag(cf), line=dict(color='rgba(187, 41, 255, 1)')),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=900,
    showlegend=False,
    title_text='Normal Distribution Characteristic Function (μ=2, σ²=1)',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update all xaxes
for i in range(1, 4):
    fig.update_xaxes(
        title_text='t',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )

# Update all yaxes
for i in range(1, 4):
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[-1.2, 1.2] if i in [2,3] else [0, 0.5],
        row=i, col=1
    )

fig.show()



# ##### Fourier Inversion: Recovering the Random Variable from the Characteristic Function
# 
# $$f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \phi(t) e^{-itx} dt$$
# 
# The Fourier inversion formula allows us to recover the probability density function (PDF) from the characteristic function:
# 
# - Uses numerical integration to compute the inverse transform
# - Requires characteristic function $\phi(t)$ as input
# - Returns original PDF $f(x)$ as output
# - Integration performed over frequency domain
# 
# This inversion process is essential for recovering probability distributions from their characteristic functions


# Fourier inversion to recover PDF
x = np.linspace(-2, 6, 1000)  # Range covering our normal distribution
recovered_pdf = np.zeros_like(x, dtype=np.float64)

# Numerical integration for each x point
for i, x_val in enumerate(x):
    integrand = np.exp(-1j * x_val * t) * cf
    recovered_pdf[i] = np.real(np.trapz(integrand, t)) / (2 * np.pi)

# Create subplots
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Real Part of Characteristic Function',
                                  'Imaginary Part of Characteristic Function',
                                  'Recovered PDF vs Original PDF'))

# Plot real part of characteristic function
fig.add_trace(
    go.Scatter(x=t, y=np.real(cf), line=dict(color='rgba(57, 255, 20, 1)')),
    row=1, col=1
)

# Plot imaginary part of characteristic function
fig.add_trace(
    go.Scatter(x=t, y=np.imag(cf), line=dict(color='rgba(187, 41, 255, 1)')),
    row=2, col=1
)

# Plot recovered PDF
fig.add_trace(
    go.Scatter(x=x, y=recovered_pdf, 
               name='Recovered PDF',
               line=dict(color='rgba(0, 255, 255, 1)')),  # Changed to neon cyan
    row=3, col=1
)

# Plot original PDF for comparison
original_pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
fig.add_trace(
    go.Scatter(x=x, y=original_pdf, 
               name='Original PDF',
               line=dict(color='rgba(255, 255, 255, 0.6)', dash='dash')),
    row=3, col=1
)

# Update layout
fig.update_layout(
    width=1000,
    height=900,
    showlegend=True,
    title_text='Recovered Normal Distribution PDF via Fourier Inversion',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update all xaxes
for i in range(1, 4):
    fig.update_xaxes(
        title_text='t' if i < 3 else 'x',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )

# Update all yaxes
for i in range(1, 4):
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[-1.2, 1.2] if i in [1,2] else [0, 0.5],
        row=i, col=1
    )

fig.show()



# ---


# ##### 4.) 🎯 Carr-Madan
# 
# Revisiting the option pricing problem now in the Heston framework, we have the following pricing problem
# 
#  
#  $$C(K,T) = e^{-rT}\mathbb{E}^{\mathbb{Q}}\left[\max(S_T - K, 0)\right]$$
#  
#  where the expectation is defined as
#  
# $$\mathbb{E}^{\mathbb{Q}}\left[\max(S_T - K, 0)\right] = \int_{-\infty}^{\infty} \max(S_T - K, 0) f_{S_T}(x) dx = \int_{K}^{\infty} (S_T - K) f_{S_T}(x) dx$$
#  
# What if we were to apply a Fourier transform to this call option value?
# 
# Well we run into an issue - the standard call option pricing function is NOT *square-integrable* which is necessary for the Fourier transform 
# 
# This is due to the log-strike approaching a constant at $-\infty$ which is not zero
# 
# #### Damping Factor $e^{\alpha k}$
# 
# By introducing a damping factor, Carr-Madan enable square-integrability of the pricing function and thus the Fourier transform applies
# 
#  Let $k = \ln(K)$ and $s = \ln(S_0)$, then the damped call price is:
# 
#  $$c_T(k) = e^{\alpha k}C_T(k)$$
# 
#  where $\alpha > 0$ is the damping parameter. The modified call price $c_T(k)$ is now square-integrable.
# 
#  The Fourier transform of the damped call price is:
# 
#  $$\psi_T(v) = \int_{-\infty}^{\infty} e^{ivk}c_T(k)dk$$
# 
#  And the inverse transform gives us the original call price:
# 
#  $$C_T(k) = \frac{e^{-\alpha k}}{2\pi}\int_{-\infty}^{\infty} e^{-ivk}\psi_T(v)dv$$
# 
# #### Key Equation
# 
#  The key equation for the Fourier transform of the damped call price in terms of the characteristic function is:
# 
#   Starting with the Fourier transform of the damped call price:
#   $$\psi_T(v) = \int_{-\infty}^{\infty} e^{ivk}c_T(k)dk = \int_{-\infty}^{\infty} e^{ivk}e^{\alpha k}C_T(k)dk$$
# 
#   Substituting the call price formula:
#   $$\psi_T(v) = \int_{-\infty}^{\infty} e^{ivk}e^{\alpha k}e^{-rT}\int_k^{\infty} (e^y - e^k)f_{S_T}(y)dy\,dk$$
# 
#   Changing order of integration:
#   $$\psi_T(v) = e^{-rT}\int_{-\infty}^{\infty}\int_{-\infty}^y e^{ivk}e^{\alpha k}(e^y - e^k)dk\,f_{S_T}(y)dy$$
# 
#   Evaluating the inner integral:
#   $$\psi_T(v) = e^{-rT}\int_{-\infty}^{\infty}\left[\frac{e^y e^{(\alpha+iv)k}}{(\alpha+iv)} - \frac{e^{(\alpha+iv+1)k}}{(\alpha+iv+1)}\right]_{-\infty}^y f_{S_T}(y)dy$$
# 
#   Simplifying:
#   $$\psi_T(v) = e^{-rT}\int_{-\infty}^{\infty} e^y\left(\frac{e^{(\alpha+iv)y}}{(\alpha+iv)} - \frac{e^{(\alpha+iv+1)y}}{(\alpha+iv+1)}\right)f_{S_T}(y)dy$$
# 
#   Recognizing the characteristic function:
#   $$\psi_T(v) = \frac{e^{-rT}}{(\alpha+iv)(\alpha+iv+1)}\phi_T(v-(i\alpha+1))S_0^{iv+\alpha+1}$$
# 
#  where $\phi_T(u)$ is the characteristic function of the log-price process.
# 
# Fortunately, the characteristic equation is known for the Heston model
# 
#  The characteristic function for the Heston model (log-stock prices) is:
# 
#  $$\phi_T(u) = \exp\left(iu(s + rT) + \frac{v_0}{\sigma^2}\left(\frac{1-e^{-dT}}{1-ge^{-dT}}\right)(a-\rho\sigma iu-d)\right) \times \exp\left(\frac{\theta\kappa}{\sigma^2}\left(2\ln\left(\frac{1-ge^{-dT}}{1-g}\right) + (a-\rho\sigma iu-d)T\right)\right)$$
# 
#  where:
#  - $d = \sqrt{(\rho\sigma iu - a)^2 + \sigma^2(iu + u^2)}$
#  - $g = \frac{a-\rho\sigma iu-d}{a-\rho\sigma iu+d}$
#  - $a = \kappa - \rho\sigma iu$
# 
#  Parameters:
#  - $\kappa$: mean reversion rate
#  - $\theta$: long-run variance
#  - $\sigma$: volatility of variance
#  - $\rho$: correlation between asset and variance processes
#  - $v_0$: initial variance
#  - $s$: log spot price
#  - $r$: risk-free rate
#  - $T$: time to maturity
# 
#  To obtain the option price, we need to invert the Fourier transform:
# 
#  $$C_T(k) = \frac{e^{-\alpha k}}{2\pi}\int_{-\infty}^{\infty} e^{-ivk}\psi_T(v)dv$$
# 
#  Substituting the expression for $\psi_T(v)$:
# 
#  $$C_T(k) = \frac{e^{-\alpha k}}{2\pi}\int_{-\infty}^{\infty} e^{-ivk}\frac{e^{-rT}}{(\alpha+iv)(\alpha+iv+1)}\phi_T(v-(i\alpha+1))S_0^{iv+\alpha+1}dv$$
# 
#  This integral can be evaluated numerically using the Fast Fourier Transform (FFT).
# 
#  The key steps are:
#  1. Choose a suitable damping parameter $\alpha > 0$ to ensure integrability
#  2. Discretize the integral using a grid of points
#  3. Apply FFT to compute the integral efficiently
#  4. Undamp the result by multiplying by $e^{-\alpha k}$
# 
#  The resulting $C_T(k)$ gives us the call option price for log-strike $k$. 
#  For actual strikes $K$, we use the relationship $k = \ln(K)$.
# 
# 
# #### Putting the Pieces Together
# 
# - Carr-Madan introduce a damping factor to enable the Fourier transform of option prices
# 
# - They identify a relationship between the characteristic function of a pricing framework and the Fourier transform of the damped price
# 
# - Applying the IFT and undamping and correcting yields the price of options for then a *set* of strikes


# ---


# ##### 5.) 📜 General Recipe
# 
#  
# The Carr-Madan FFT algorithm for Heston option pricing follows these steps:
# 
#  1. Formulate the Heston Characteristic Function
#     The characteristic function $\phi(u)$ is the Fourier transform of the risk-neutral probability density:
#     $$\phi_T(u) = \exp\left(iu(s + rT) + \frac{v_0}{\sigma^2}\left(\frac{1-e^{-dT}}{1-ge^{-dT}}\right)(a-\rho\sigma iu-d)\right) \times \exp\left(\frac{\theta\kappa}{\sigma^2}\left(2\ln\left(\frac{1-ge^{-dT}}{1-g}\right) + (a-\rho\sigma iu-d)T\right)\right)$$
#     where $d = \sqrt{(\rho\sigma iu - a)^2 + \sigma^2(iu + u^2)}$, $g = \frac{a-\rho\sigma iu-d}{a-\rho\sigma iu+d}$, and $a = \kappa - \rho\sigma iu$
# 
#  2. Damping and the Carr-Madan Formula
#     Introduce damping factor $e^{\alpha k}$ and express the Fourier transform of damped prices:
#     $$\hat{c}(v) = \frac{e^{-rT}\phi_T(v-i(\alpha+1))}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$
# 
#  3. Discretize for the FFT
#     Set up discrete grids:
#     - Frequencies: $v_j = j\Delta_v$ for $j = 0,\ldots,N-1$
#     - Log-strikes: $k_m = k_{min} + m\Delta_k$ for $m = 0,\ldots,N-1$
#     where $\Delta_k\Delta_v = \frac{2\pi}{N}$
# 
#  4. Apply the Inverse FFT
#     Transform frequency domain to log-strike domain:
#     $$c(k_m) = \frac{1}{2\pi}\sum_{j=0}^{N-1} e^{-iv_jk_m}\hat{c}(v_j)\Delta_v w_j$$
#     where $w_j$ are quadrature weights
# 
#  5. Undamp and Correct
#     Recover actual call prices:
#     $$C(k_m) = e^{-\alpha k_m}c(k_m)$$
#     where $K_m = e^{k_m}$ gives option prices at strikes $K_m$
# 
# 


import numpy as np
from dataclasses import dataclass

# =========================
# Heston + Carr–Madan (FFT)
# =========================

@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

def heston_cf(u, T, S0, r, q, p: HestonParams):
    """
    Characteristic function φ(u) = E[exp(i u ln S_T)] under Q.
    Uses the Little Heston Trap parametrization.
    u can be scalar or numpy array.
    """
    i = 1j
    x0 = np.log(S0)
    a  = p.kappa * p.theta
    b  = p.kappa - p.rho * p.sigma * i * u
    d  = np.sqrt(b*b + (p.sigma**2) * (i*u + u*u))
    g  = (b - d) / (b + d)

    eDT = np.exp(-d * T)
    one_minus_g_eDT = 1 - g * eDT
    one_minus_g     = 1 - g
    # small guards
    one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
    one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

    C = i*u*(r - q)*T + (a/(p.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
    D = ((b - d)/(p.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
    return np.exp(C + D*p.v0 + i*u*x0)

def _simpson_weights(N: int):
    """Simpson weights on an N-point uniform grid (N must be even)."""
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson weights.")
    w = np.ones(N)
    w[1:N-1:2] = 4
    w[2:N-2:2] = 2
    return w

def heston_fft_calls(
    S0: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    N: int = 4096,      # power-of-two recommended; must be even
    eta: float = 0.25,  # frequency step Δv
    alpha: float = 1.5  # damping (>0)
):
    """
    Carr–Madan FFT for call prices across a K-grid.
    IMPORTANT: k = ln K (log-STRIKE), not ln(K/S0).

    Returns
    -------
    K : (N,) ascending strikes
    C : (N,) call prices for these K
    """
    n = np.arange(N)
    v = eta * n  # frequency grid

    i = 1j
    # ψ(v) = e^{-rT} φ(v - i(α+1)) / [(α + iv)(α + iv + 1)]
    phi_shift = heston_cf(v - (alpha + 1)*i, T, S0, r, q, p)
    denom = (alpha**2 + alpha - v**2 + i*(2*alpha + 1)*v)  # (α+iv)(α+iv+1)
    psi = np.exp(-r*T) * phi_shift / denom

    # Simpson weights for the v-integral
    w = _simpson_weights(N) * (eta / 3.0)

    # FFT coupling
    lam = 2.0 * np.pi / (N * eta)   # Δk (log-strike step)
    b   = 0.5 * N * lam             # half-width in k
    x   = psi * np.exp(1j * b * v) * w

    F   = np.fft.fft(x)
    F   = np.real(F)

    j = np.arange(N)
    k = -b + j * lam                 # k = ln K
    K = np.exp(k)

    calls = np.exp(-alpha * k) / np.pi * F
    order = np.argsort(K)
    return K[order], np.maximum(calls[order], 0.0)

def heston_fft_call_price(
    S0: float, K: float, T: float, r: float, q: float, p: HestonParams,
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
):
    """Price a single call via FFT + linear interpolation on the K-grid."""
    K_grid, C_grid = heston_fft_calls(S0, T, r, q, p, N=N, eta=eta, alpha=alpha)
    if K <= K_grid[0]:
        return C_grid[0]
    if K >= K_grid[-1]:
        return C_grid[-1]
    idx = np.searchsorted(K_grid, K)
    x0, x1 = K_grid[idx-1], K_grid[idx]
    y0, y1 = C_grid[idx-1], C_grid[idx]
    return y0 + (y1 - y0) * (K - x0) / (x1 - x0)

def heston_fft_put_price(
    S0: float, K: float, T: float, r: float, q: float, p: HestonParams,
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
):
    """Put via put–call parity."""
    C = heston_fft_call_price(S0, K, T, r, q, p, N=N, eta=eta, alpha=alpha)
    return C - S0*np.exp(-q*T) + K*np.exp(-r*T)


# parameters
S0, r, q, T = 100.0, 0.01, 0.00, 1.0
hp = HestonParams(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)

# full grid of calls
K, C = heston_fft_calls(S0, T, r, q, hp, N=4096, eta=0.25, alpha=1.5)

# single strikes
call_100 = heston_fft_call_price(S0, 100, T, r, q, hp)

call_100


# Simulate Heston price with given parameters
S0 = 100          # Initial stock price
K = 100           # Strike price  
T = 1             # Time to maturity
r = 0.01          # Risk-free rate
v0 = 0.04         # Initial variance
kappa = 2         # Mean reversion speed
theta = 0.04      # Long-run variance
sigma = 0.3       # Volatility of variance
rho = -0.7        # Correlation
N = 16384         # Number of grid points
alpha = 1.5       # Damping factor

# Simulation parameters
n_paths = 10000   # Number of paths
n_steps = 252     # Number of time steps (daily)
dt = T/n_steps    # Time step size

# Initialize arrays for paths
S = np.zeros((n_paths, n_steps + 1))
v = np.zeros((n_paths, n_steps + 1))

# Set initial values
S[:, 0] = S0
v[:, 0] = v0

# Generate correlated random numbers
rng = np.random.default_rng()
dW1 = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))
dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

# Euler-Maruyama simulation
for t in range(n_steps):
    # Ensure variance stays positive
    v[:, t] = np.maximum(v[:, t], 0)
    
    # Update stock price
    S[:, t+1] = S[:, t] * np.exp((r - 0.5*v[:, t])*dt + np.sqrt(v[:, t])*dW1[:, t])
    
    # Update variance
    v[:, t+1] = v[:, t] + kappa*(theta - v[:, t])*dt + sigma*np.sqrt(v[:, t])*dW2[:, t]

# Calculate call option prices
payoffs = np.maximum(S[:, -1] - K, 0)
discounted_payoffs = np.exp(-r*T) * payoffs
price = np.mean(discounted_payoffs)

# Calculate 95% confidence interval
std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
ci_lower = price - 1.96 * std_error
ci_upper = price + 1.96 * std_error

print(f"Heston call option price: {price:.8f}")
print(f"95% Confidence Interval: [{ci_lower:.8f}, {ci_upper:.8f}]")


# Generate strike grid centered around S0
strikes = np.arange(S0-20, S0+25, 5)

# Get FFT prices for all strikes
K_fft, C_fft = heston_fft_calls(S0, T, r, q, hp, N=4096, eta=0.25, alpha=1.5)

# Interpolate FFT prices to our strike grid
fft_prices = np.interp(strikes, K_fft, C_fft)

# Calculate MC prices for all strikes
mc_prices = []
mc_ci_lower = []
mc_ci_upper = []

for K in strikes:
    payoffs = np.maximum(S[:, -1] - K, 0)
    discounted_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(discounted_payoffs)
    
    # Calculate 95% confidence interval
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    mc_prices.append(price)
    mc_ci_lower.append(price - 1.96 * std_error)
    mc_ci_upper.append(price + 1.96 * std_error)

# Print comparison table
print("\nPrice Comparison:")
print("Strike   FFT Price   MC Price   MC 95% CI")
print("-" * 45)
for i in range(len(strikes)):
    print(f"{strikes[i]:6.1f}   {fft_prices[i]:9.4f}   {mc_prices[i]:8.4f}   [{mc_ci_lower[i]:8.4f}, {mc_ci_upper[i]:8.4f}]")



# ---


# ##### 6.) 💭 Closing Thoughts and Future Topics
# 
# This notebook shows if we know the characteristic function of stock prices given by a model framework, by Carr-Madan a relationship between the characteristic function and European option prices by the Fourier transform.
# 
# Since the Heston model characteristic function is known, we can apply the inverse Fourier transform to the key equation relating the Fourier transform directly to the relative grid of option prices.
# 
# **Future Topics:**
# 
# - Calibrating a Heston Model to a Market Implied Volatility Surface
# 
# - Alternative Means of Solving for Exotic Prices (PDE Methods)
# 
# - Approximating the Pricing Functional Given by Heston using Neural Networks (Calibration)
# 
# - Applications of Machine Learning in Approximating Pricing Functionals


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

