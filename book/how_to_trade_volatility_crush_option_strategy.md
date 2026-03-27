### 💻 How to Trade the Volatility Crush Option Strategy

##### ▶️ Related Quant Guild Videos:

- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)

- [How to Trade](https://youtu.be/NqOj__PaMec)

- [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)

- [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)

- [How to Trade with the Kelly Criterion](https://youtu.be/7tvW3NvRnPk)

- [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)

- [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)

###### ______________________________________________________________________________________________________________________________________

 
##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)

##### [📅 Take Live Classes with Roman on Quant Guild](https://quantguild.com/live-classes)

---

### 📖 Sections

#### 1.) 💥 Volatility Crush Option Trading Strategy

- What is the Option Trading Strategy?

- Why is Option Pricing Important?

#### 2.) 📉 Trading Volatility Crush in Interactive Brokers

Earnings Events: https://www.nasdaq.com/market-activity/earnings

- Earnings Events

- Macro Events

- Examples: *My Trades Last Week* NVDA, DELL

- Example: ZS

#### 3.) 🛠️ Volatility Crush Trading Tools

- Theoretical Option Pricing and Empirical Pricing

- Volatility Crush Trading Tool

- Earnings Event Volatility Crush Trading Tool

#### 5.) 💭 Closing Thoughts and Future Comments

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



#### 1.) 💥 Volatility Crush

##### 📝 What is the Option Trading Strategy?

Fear is often mispriced in the market (statistically overpriced), volatility crush or *implied* volatility crush is a phenomenon in the markets consequent of this mispricing.

Before important events that may create a price movement in either direction option prices tend to be inflated relative to the subsequent move.

Specifically, *implied volatility* tends to be higher before this event then as uncertainty dissapates there is a rapid decline in option prices as this *implied volatility* declines.

Should the underlying asset not move as much as priced in by the options one can effectively trade this effect and do their best to hedge out other factors impacting price to collect the theoretical mispricing.

**Remark:**  Collecting the pure theoretical mispricing in volatility is *not* possible, but we can do our by constructing different types of option strategies


```python
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import norm

# Black-Scholes formula for option pricing
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

# Fixed parameters
S = 100  # Stock price
K = 100  # Strike price
T = 30/365  # Time to expiry (30 days)
r = 0.05  # Risk-free rate

# Different implied volatilities
implied_vols = [0.2, 0.3, 0.4, 0.5, 0.6]  # 20% to 60% volatility

# Calculate straddle prices (call + put) for each volatility
straddle_prices = []
for vol in implied_vols:
    call = S * 0.1 * vol  # Simplified pricing for demonstration
    put = S * 0.1 * vol   # Simplified pricing for demonstration
    straddle_prices.append(call + put)

# Create bar chart
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[f"{vol*100}% Vol" for vol in implied_vols],
        y=straddle_prices,
        marker_color='rgba(0, 191, 255, 0.6)',
        name='Straddle Price'
    )
)

# Update layout
fig.update_layout(
    title='Straddle Option Prices at Different Implied Volatilities',
    xaxis_title='Implied Volatility',
    yaxis_title='Straddle Price ($)',
    width=800,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()


```



##### 📜 Why is Option Pricing Important?

Theoretical pricing tells us what we can expect out P/L to be relative to what actually happens in the market.

There are two ways to understand the impact in your option P/L

##### 📈 Being a *Trader* and trading *a lot*, both have pros and cons

**✔️ Pros:** 
- More inline with what the market expects as you aren't making assumptions
- Experience and P/L impact is a strong motivator for intuition

**❌ Cons:**
- Lack of experience can lead to questions like *why did my option contract lose or increase in value?*
- Acquiring this experience can be quite costly if you constantly don't know what you're doing 

##### 🧮 Understanding *theoretical* option pricing

**✔️ Pros:** 
- General understanding of where P/L comes from and increases/decreases in position value
- Understanding of the different exposures non-linearly, for example, when volatility crush can't save you from delta exposure

**❌ Cons:**
- Lack of experience can lead to questions like *why didn't my option's value change how the greeks implied it?*
- Assumptions create a model but explicitly and only relying on model greeks isn't what professional traders do, they have to be modified its a model


**Remark:** Change in option price by different exposures are linearly approximated by greeks, as different exposures move their relative impact in your option price changes in a *non-linear* way, your greek is going to change and changing rates!



```python
# Fixed parameters
S = 100  # Initial stock price
K = 100  # Strike price
T = 30/365  # Time to expiry (30 days)
r = 0.05  # Risk-free rate

# Different implied volatilities and corresponding spot prices
implied_vols = [0.2, 0.3, 0.4, 0.5, 0.6]  # 20% to 60% volatility
spot_prices = [S - (i * 15) for i in range(len(implied_vols))]  # Decrease spot by 10 each time

# Calculate straddle prices (call + put) for each volatility and spot price
straddle_prices = []
for vol, spot in zip(implied_vols, spot_prices):
    call = spot * 0.1 * vol  # Simplified pricing for demonstration
    put = spot * 0.1 * vol   # Simplified pricing for demonstration
    straddle_prices.append(call + put)

# Create bar chart
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[f"{vol*100}% Vol (S={spot})" for vol, spot in zip(implied_vols, spot_prices)],
        y=straddle_prices,
        marker_color='rgba(0, 191, 255, 0.6)',
        name='Straddle Price'
    )
)

# Update layout
fig.update_layout(
    title='Straddle Option Prices at Different Implied Volatilities and Spot Prices',
    xaxis_title='Implied Volatility (Spot Price)',
    yaxis_title='Straddle Price ($)',
    width=800,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```



---

#### 2.) 📉 Trading Volatility Crush in Interactive Brokers

We'll head on over to my live Interactive Brokers account and discuss some examples of trading this volatility crush strategy.

- Example: NVDA

- Example: DELL (Earnings & PCE Combined Event)

- Trading Tools Example

---

#### 3.) 🛠️ Volatility Crush Trading Tools

Stay tuned we will be going over relevant theory and building both volatility crush trading tools from scratch!

1.) Deriving and Understanding the Black-Scholes Equation for Trading Options

2.) Volatility Crush Pricing Trading Tool

3.) Earnings Event Pricing Trading Tool

---

#### 5.) 💭 Closing Thoughts and Future Comments

If you want to capitalize on volatility crush you are an insurance seller - you are being paid to assume tail risk!

If you are worried about such an event occuring then you are an insurance BUYER not SELLER.

Statistically, (historically) we observe this mispricing and volatility crush - it is not indicative of performance on any one trade going forward.

Remember, our goal is to accumulate edge (positive EV) over a series of trades and optimize our breakdown of EV

 $\text{E [P/L]} = \text{Probability of Win} \times \text{Average Winner} - \text{Probability of Loss} \times \text{Average Loser}$
 
**Future Topics**

- How to Build the Volatility Crush Trade Analyzer

- How to Build the Earnings Trading Dashboard

- Different Option Strategies: Straddles, Strangles, Iron Condors, . . .

- How to Derive and Solve the Black-Scholes Equation

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
