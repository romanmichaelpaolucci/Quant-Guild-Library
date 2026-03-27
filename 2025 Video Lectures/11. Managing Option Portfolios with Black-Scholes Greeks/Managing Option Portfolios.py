# # Managing Option Portfolios
# 
# In this lesson, we will explore how market makers manage option portfolios using the Black-Scholes model. We’ll cover two major parts:
# 
# 1. **Linear Approximations:**  
#    We start by revisiting the idea of a linear approximation using the simple function $ f(x)=x^2 $. Using an interactive slider (ipywidgets), we’ll see how a linear (tangent line) approximation is accurate close to the point of tangency but deteriorates as we move farther away. This illustrates why sensitivities (or “Greeks”)—which are local derivatives—must be recalculated frequently in practice.
# 
# 2. **Black-Scholes Model & Option Greeks:**  
#    Next, we’ll introduce the Black-Scholes pricing model for options. We’ll define the key sensitivities (Greeks) such as Delta, Theta, Vega, and Rho. Then we’ll discuss how a market maker hedges these exposures. (For instance, delta exposure can be hedged with the underlying asset, while theta, vega, and rho exposures require taking positions in other options.)
# 
# Let’s get started!


# ## Part 1: Linear Approximation Using $f(x)=x^2$
# 
# The linear approximation (or tangent line approximation) of a function $ f(x) $ at a point $ x_0 $ is given by:
# $$
# f(x) \approx f(x_0) + f'(x_0) (x - x_0)
# $$
# For  $ f(x)=x^2 $:
# - $ f(x_0)=x_0^2 $
# - $ f'(x)=2x $ so $ f'(x_0)=2x_0 $
# 
# Thus, the linear approximation becomes:
# $$
# f(x) \approx x_0^2 + 2x_0 (x - x_0)
# $$
# Below, use the slider to choose $ x_0 $ and see how the tangent line approximates $ x^2 $.
# 
# Notice that the approximation is very good near $ x_0 $but diverges as you move farther away.


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

def linear_approximation(x0, x):
    return x0**2 + 2*x0*(x - x0)

def plot_approximation(x0):
    x = np.linspace(x0 - 10, x0 + 10, 400)
    y = x**2
    y_lin = linear_approximation(x0, x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'$f(x)=x^2$', color='blue')
    plt.plot(x, y_lin, label=f'Linear Approximation at x={x0}', linestyle='--', color='red')
    plt.title(r'Linear Approximation of $x^2$ at $x_0$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

interact(plot_approximation, x0=(-10, 10, 1));


# ### Computing the Actual Change v.s. Approximate Change


center = 10
change = .1

# Actual Function
def f(x):
    return x ** 2

# Taylor series approximation
def af(x, x0):
    return f(x0) + 2*x0*(x - x0)

# Actual Change
print('Actual Function Change:', f(center + change) - f(center))

# Approx Change
print('Approx Function Change:', af(center + change, center)- f(center))


# ### Observations:
# - The red dashed line (tangent) is a very good approximation for $ f(x)=x^2 $ close to $ x_0 $.
# - As you move away from $ x_0 $, the linear approximation (tangent line) diverges from the actual curve.
# - This behavior is analogous to the sensitivities (Greeks) in option pricing, which are “local” measures and must be recomputed as market conditions (the underlying variables) change.


# ## Part 2: Black-Scholes Model and Option Greeks
# 
# Market makers use the Black-Scholes model to determine the theoretical price of options. For a European call option, the price is given by:
# $$
# \text{Call Price} = S \, N(d_1) - K \, e^{-rT} \, N(d_2)
# $$
# where:
# $$
# d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
# $$
# and $ N(\cdot) $ is the cumulative distribution function of the standard normal distribution.
# 


# ### The Greeks:
# - **Delta:** Sensitivity of the option price to changes in the underlying asset’s price.
# - **Theta:** Sensitivity to the passage of time (time decay).
# - **Vega:** Sensitivity to volatility.
# - **Rho:** Sensitivity to the interest rate.
# - **Gamma:** (Second order) Sensitivity of delta to changes in the underlying price.
# 
# Market makers hedge these risks:
# - **Delta hedging:** Adjust the position in the underlying asset.
# - **Hedging other Greeks:** Typically involves taking positions in other options to offset exposures (since you cannot directly hedge theta, vega, or rho with the underlying asset).
# 
# Below, we define functions to compute the call price and its first order Greeks.


import scipy.stats as si

def black_scholes_call(S, K, T, r, sigma):
    """Calculate European call option price using Black-Scholes model."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price, d1, d2

def call_delta(S, K, T, r, sigma):
    """Delta: sensitivity to underlying price changes."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1)

def call_theta(S, K, T, r, sigma):
    """Theta: time decay of the option price."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r*T) * si.norm.cdf(d2)
    return term1 - term2

def call_vega(S, K, T, r, sigma):
    """Vega: sensitivity to volatility."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

def call_rho(S, K, T, r, sigma):
    """Rho: sensitivity to interest rate."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * T * np.exp(-r*T) * si.norm.cdf(d2)


# Example: Compute the call price and Greeks for a sample set of parameters
S = 101       # Underlying price
K = 100       # Strike price
T = 1         # Time to maturity (years)
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility

price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
delta = call_delta(S, K, T, r, sigma)
theta = call_theta(S, K, T, r, sigma)
vega = call_vega(S, K, T, r, sigma)
rho = call_rho(S, K, T, r, sigma)

print(f"Call Option Price: {price:.2f}")
print(f"Delta: {delta:.2f}")
print(f"Theta: {theta:.2f}")
print(f"Vega: {vega:.2f}")
print(f"Rho: {rho:.2f}")


# ### Interactive Exploration of Option Greeks
# 
# The following interactive widget allows you to adjust the parameters $ S $, $ K $, $ T $, $ r $, and $ \sigma $ to see how the option price and its sensitivities (Greeks) change. Use this tool to understand:
# - **Delta Hedging:** Market makers hedge their delta exposure using the underlying asset.
# - **Other Greeks:** Theta, Vega, and Rho exposures are hedged by taking positions in other options since they cannot be directly hedged with the underlying asset.
# 
# Try changing the parameters and observe the impact on the call option price and each Greek.


def option_greeks_interactive(S=100, K=100, T=1, r=0.05, sigma=0.2):
    price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
    delta = call_delta(S, K, T, r, sigma)
    theta = call_theta(S, K, T, r, sigma)
    vega = call_vega(S, K, T, r, sigma)
    rho = call_rho(S, K, T, r, sigma)
    
    print(f"Call Option Price: {price:.2f}")
    print(f"Delta (sensitivity to underlying): {delta:.2f}")
    print(f"Theta (time decay): {theta:.2f}")
    print(f"Vega (sensitivity to volatility): {vega:.2f}")
    print(f"Rho (sensitivity to interest rate): {rho:.2f}")

interact(option_greeks_interactive,
         S=(50, 150, 1),
         K=(50, 150, 1),
         T=(0.1, 2, 0.1),
         r=(0.0, 0.1, 0.005),
         sigma=(0.1, 0.5, 0.01));


def theta_approximation_chart(S=100, K=100, T=1, r=0.05, sigma=0.2, dt=0.01):
    # Current price and theta at time T
    price, _, _ = black_scholes_call(S, K, T, r, sigma)
    theta_val = call_theta(S, K, T, r, sigma)
    
    # New time to maturity is T - dt (ensure positive time)
    T_new = T - dt if T - dt > 0 else 0.001
    
    # Compute the actual new price at T_new
    new_price, _, _ = black_scholes_call(S, K, T_new, r, sigma)
    
    # Approximate new price using the theta sensitivity (note: theta is typically negative for calls)
    approx_price = price + theta_val * (-dt)
    
    # Plot the initial price, the actual new price, and the theta approximation
    labels = ['Initial Price', 'Actual Price (T-dt)', 'Theta Approximation']
    values = [price, new_price, approx_price]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['green', 'blue', 'orange'])
    plt.title("Theta Effect: Actual vs. Approximated Option Price Change")
    plt.ylabel("Option Price")
    plt.ylim([min(values)*0.95, max(values)*1.05])
    plt.show()
    
    error = new_price - approx_price
    print(f"Initial Price: {price:.2f}")
    print(f"New Price (with T-dt): {new_price:.2f}")
    print(f"Approximated Price (using Theta): {approx_price:.2f}")
    print(f"Error (Actual - Approximation): {error:.4f}")

from ipywidgets import interact
interact(theta_approximation_chart, 
         S=(50, 150, 1), 
         K=(50, 150, 1), 
         T=(0.1, 2, 0.1), 
         r=(0.0, 0.1, 0.005), 
         sigma=(0.1, 0.5, 0.01),
         dt=(0.001, 0.2, 0.001));



# ## Part 3: Managing Option Portfolios and Hedging Strategies
# 
# A market maker’s job is to provide liquidity by quoting bid and ask prices for options. Using the Black-Scholes model, the market maker:
# - Computes the theoretical price of options.
# - Calculates the **Greeks** (sensitivities) that estimate how the option price changes with respect to underlying factors.
# 
# ### Hedging Strategies:
# - **Delta Hedging:**  
#   Since delta measures the sensitivity to the underlying asset’s price, market makers can hedge delta risk by taking an offsetting position in the underlying stock.
#   
# - **Hedging Theta, Vega, and Rho:**  
#   These sensitivities cannot be hedged using the underlying asset alone. To manage these risks, market makers use other options:
#   - **Theta:** Although time decay is inevitable, using options with different maturities can help manage the overall time decay.
#   - **Vega:** Options with different volatilities or strikes can be used to offset volatility risk.
#   - **Rho:** Taking positions in options with varying sensitivities to interest rates helps manage rho risk.
#   
# Since the Greeks are based on a linear (tangent) approximation, they are accurate only for small changes in the underlying parameters. This is why market makers must continuously recalculate the Greeks and adjust their hedging positions dynamically.
# 


# ## Conclusion
# 
# In this lesson, we:
# - Explored the concept of linear approximation using $ f(x)=x^2 $ and saw how the approximation worsens as we move away from the point of tangency.
# - Reviewed the Black-Scholes model for pricing options and computed the first order sensitivities (Greeks) such as Delta, Theta, Vega, and Rho.
# - Discussed how market makers use these Greeks to hedge their option portfolios, using the underlying asset for delta and other options for managing additional risks.
# 
# This hands-on approach helps illustrate why continuous recalibration of sensitivities is necessary in managing an option portfolio. Happy exploring and trading!
# 

