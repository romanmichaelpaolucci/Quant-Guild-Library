### 🎢 ARCH and GARCH Models in Quant Finance 🌊

##### ▶️ Related Quant Guild Videos:

- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)

- [Why Portfolio Optimization Doesn't Work](https://youtu.be/eZIITtd3UfY)

- [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [How to Build a Volatility Trading Dashboard in Python with Interactive Brokers](https://youtu.be/19-rFVgJVkg)

###### ______________________________________________________________________________________________________________________________________

 
##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

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

#### 1.) 🎢 Volatility

- Realized Volatility

- Implied Volatility

- Relevant Statistics, Value at Risk, Stylized Facts

#### 2.) 🌊 ARCH Models (Engle 1982)

- ARCH(q) Model

- Previous Models for Volatility

- Example: EWMA vs. ARCH(1)

- Why the ARCH(q) Model was Innovative

#### 3.) 🧮 GARCH Models (Bollerslev 1986)

- GARCH as an Infinite Order ARCH Model

- GARCH vs. ARCH Model

#### 4.) 🎯 Application of ARCH/GARCH Models 

- Portfolio Management (Value-at-Risk)

#### 5.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 🎢 Volatility

In finance, volatility refers to a measure of deviation from an expectation, typically in the context of the return space - not the price path space

Since volatility requires deviation from some sort of expectation, a mean level is required for reference

In other words, we think of this expectation as a benchmark "this is volatility relative to some expectation"

  Expected returns (forward looking): 
  
  $$\mathbb{E}[R_t] = \mu_t$$
  
  Variance of returns (forward looking):
  $$\text{Var}(R_t) = \mathbb{E}[(R_t - \mu_t)^2] = \sigma_t^2$$
  
  Note: $\mu_t$ is not directly observable and likely changes over time. This makes it difficult to get accurate 
  estimates of expected returns. Since variance depends on deviations from $\mu_t$, this uncertainty in the mean
  also impacts our ability to measure volatility accurately.

  **Remark:** Here we are talking about a return based on some risk exposure, not some sort of statistical mispricing which would change this analysis entirely.
  
Neither is **directly** observable

- we can proxy forward looking returns using historical returns

- we can proxy forward looking volatility using realized/historic volatility or implied volatility

But the spot estimates in a forward looking sense depend on a time interval (forward looking returns) and current market state (forward implied vol)

Nothing says the market has to play out according to these spot estimations. . .

##### Realized or Historic Volatility





```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the volatility data
df = pd.read_csv('aapl_volatility_data_2023.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate annualized squared deviations from rolling means
df['sq_dev_7d'] = (df['returns'] - df['returns'].rolling(7).mean())**2 * 252
df['sq_dev_30d'] = (df['returns'] - df['returns'].rolling(30).mean())**2 * 252

# Cut off first 30 days
df = df[30:]

# Create subplots - 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'AAPL Returns', 
        'Volatility Measures'
    ),
    horizontal_spacing=0.1
)

# Plot returns
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['returns'],
        mode='lines',
        name='Returns',
        line=dict(color='#FF00FF', width=1.5),  # Neon pink
        showlegend=True
    ),
    row=1, col=1
)

# Plot volatility measures
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['ib_realized_vol_30d'],
        mode='lines',
        name='30-day Realized Vol',
        line=dict(color='#39FF14', width=1.5),  # Neon green
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=np.sqrt(df['sq_dev_7d'].rolling(7).mean()),
        mode='lines',
        name='7-day Rolling Vol',
        line=dict(color='#FF3131', width=1.5),  # Neon red
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=np.sqrt(df['sq_dev_30d'].rolling(30).mean()),
        mode='lines',
        name='30-day Rolling Vol',
        line=dict(color='#00FFFF', width=1.5),  # Neon cyan
        showlegend=True
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for col in [1,2]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Date",
        row=1, col=col
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Returns" if col==1 else "Volatility",
        row=1, col=col
    )

fig.show()
```



###### ______________________________________________________________________________________________________________________________________


##### Implied Volatility

The Black-Scholes model gives us a forward looking measure of volatility called implied volatility

Essentially, volatility is an input into an option's price and we can use market prices (generated by supply and demand) to produce this value

 The implied volatility $\sigma_{IV}$ is found by solving:
  The Black-Scholes PDE:
  
  $$\frac{\partial C}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 C}{\partial S^2} + rS\frac{\partial C}{\partial S} - rC = 0$$
  
  With solution for a European call option:
  
  $$C_{BS}(S,K,r,T,\sigma) = SN(d_1) - Ke^{-rT}N(d_2)$$
  
  where $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$ and $d_2 = d_1 - \sigma\sqrt{T}$
  
  The implied volatility $\sigma_{IV}$ is then found by solving:
  
  $$\sigma_{IV} = \underset{\sigma}{\text{argmin}} \left| C_{market} - C_{BS}(S, K, r, T, \sigma) \right|$$
 
 where $C_{market}$ is the market price of the option and $C_{BS}$ is the Black-Scholes price


This gives us a sense of what level of volatility traders are pricing options expiring in the future at

It is effectively a best guess at the volatility in a forward looking sense with *money on the line* 


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the volatility data
df = pd.read_csv('aapl_volatility_data_2023.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate annualized squared deviations from rolling means
df['sq_dev_7d'] = (df['returns'] - df['returns'].rolling(7).mean())**2 * 252
df['sq_dev_30d'] = (df['returns'] - df['returns'].rolling(30).mean())**2 * 252

# Calculate rolling 30-day correlation
rolling_corr = df['ib_realized_vol_30d'].rolling(30).corr(df['implied_vol'])

# Remove first 30 days of data
df = df[30:]

# Create subplots - 2 rows
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'AAPL Returns', 
        'Volatility Measures',
        'Rolling 30-day Correlation between Realized and Implied Volatility',
        ''
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
    row_heights=[0.7, 0.3],
    specs=[[{}, {}], [{"colspan": 2}, None]]
)

# Plot returns
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['returns'],
        mode='lines',
        name='Returns',
        line=dict(color='#B026FF', width=1.5), # Neon purple
        showlegend=True
    ),
    row=1, col=1
)

# Plot volatility measures
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['ib_realized_vol_30d'],
        mode='lines',
        name='30-day Realized Vol',
        line=dict(color='#39FF14', width=1.5), # Neon green
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['implied_vol'],
        mode='lines',
        name='Implied Vol',
        line=dict(color='#00FFFF', width=1.5), # Neon cyan
        showlegend=True
    ),
    row=1, col=2
)

# Plot rolling correlation
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=rolling_corr[60:],
        mode='lines',
        name='Rolling Correlation',
        line=dict(color='#FF4500', width=1.5), # Neon orange
        showlegend=True
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    width=1200,
    height=800,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for top row
for col in [1,2]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Date",
        row=1, col=col
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Returns" if col==1 else "Volatility",
        row=1, col=col
    )

# Update axes for bottom row
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date",
    row=2, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Correlation",
    range=[-1, 1],
    row=2, col=1
)

fig.show()
```



**Note:** We can always apply time series techniques (smoothing, filtering, forecasting) to any measure of volatility (historic or implied) but the efficacy of such an approach is problem dependent!  As observed above the correlation between historic (30-day realized vol) and implied volatility varies significantly over time - there are several reasons why this is the case which we will look at. . .

Primarily, implied volatility is a *forward* looking measure and realized is a *backward* looking measure, correlations between these structures depend on time and the regime of volatility and if it is increasing or decreasing which will then cause these measures to become more or less correlated. . .

**Remark:** Implied volatility changes based on the moneyness of the option and the time to expiration (so called skew or term structure) - we are looking at an *aggregate* measure of implied volatility which will suffice for our analysis as it is largely correlated with the relative option implied volatilities for a particular underlying (herein we look at AAPL).

A well documented phenomenon is the notion of volatility realizing *lower* than implied volatility would suggest

Many volatility trading strategies involve this idea, for example, holding net negative vega exposure while delta hedging directional risk


```python
# Create lagged implied vol and forward implied vol
df['ivol_lag'] = df['implied_vol']
# Calculate 5-day forward average implied vol
n = 20
df['ivol_fwd'] = df['implied_vol'].rolling(window=n, min_periods=1).mean().shift(-n)

# Remove NaN values
df_reg = df.dropna(subset=['ivol_lag', 'ivol_fwd'])

# Create subplots
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

# First subplot - Regression with y=x line
fig.add_trace(
    go.Scatter(
        x=df_reg['ivol_lag'],
        y=df_reg['ivol_fwd'],
        mode='markers',
        name='Data Points',
        marker=dict(
            color='#39FF14',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=1
)

# Calculate and add regression line
z = np.polyfit(df_reg['ivol_lag'], df_reg['ivol_fwd'], 1)
p = np.poly1d(z)
x_reg = np.linspace(df_reg['ivol_lag'].min(), df_reg['ivol_lag'].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_reg,
        y=p(x_reg),
        mode='lines',
        name=f'Regression Line (slope={z[0]:.3f})',
        line=dict(color='#00FFFF', width=2)
    ),
    row=1, col=1
)

# Add y=x baseline to first subplot
fig.add_trace(
    go.Scatter(
        x=x_reg,
        y=x_reg,
        mode='lines',
        name='y=x (Perfect Realization)',
        line=dict(
            color='red',
            width=2,
            dash='dot'
        )
    ),
    row=1, col=1
)

# Find intersection point of regression line and y=x
intersection_x = -z[1]/(z[0]-1)
intersection_y = p(intersection_x)

# Split data into low and high vol regimes
df_low = df_reg[df_reg['ivol_lag'] <= intersection_x]
df_high = df_reg[df_reg['ivol_lag'] > intersection_x]

# Second subplot - regime specific regressions
fig.add_trace(
    go.Scatter(
        x=df_low['ivol_lag'],
        y=df_low['ivol_fwd'],
        mode='markers',
        name='Low Vol Regime',
        marker=dict(
            color='#39FF14',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df_high['ivol_lag'],
        y=df_high['ivol_fwd'],
        mode='markers',
        name='High Vol Regime',
        marker=dict(
            color='#FF10F0',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=2
)

# Add regime specific regression lines
z_low = np.polyfit(df_low['ivol_lag'], df_low['ivol_fwd'], 1)
p_low = np.poly1d(z_low)
x_low = np.linspace(df_low['ivol_lag'].min(), df_low['ivol_lag'].max(), 100)

z_high = np.polyfit(df_high['ivol_lag'], df_high['ivol_fwd'], 1)
p_high = np.poly1d(z_high)
x_high = np.linspace(df_high['ivol_lag'].min(), df_high['ivol_lag'].max(), 100)

fig.add_trace(
    go.Scatter(
        x=x_low,
        y=p_low(x_low),
        mode='lines',
        name=f'Low Vol Regression (slope={z_low[0]:.3f})',
        line=dict(color='#00FFFF', width=2)
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=x_high,
        y=p_high(x_high),
        mode='lines',
        name=f'High Vol Regression (slope={z_high[0]:.3f})',
        line=dict(color='#FF00FF', width=2)
    ),
    row=1, col=2
)

# Add vertical line at regime split point
fig.add_vline(
    x=intersection_x,
    line_dash="dash",
    line_color="red",
    row=1,
    col=2
)

# Update layout
fig.update_layout(
    width=1600,
    height=600,
    title='Implied Vol Analysis with Regime Split',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for both subplots
for i in [1, 2]:
    fig.update_xaxes(
        title_text='Current Implied Volatility',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title_text='5-Day Forward Average Implied Volatility',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### Relevant Statistics

A feature we observe in returns is this idea of *excess kurtosis* or "fat tails" which is defined as leptokurtosis or a leptokurtic return distribution.

Formally, Kurtosis is the 4th statistical moment
 For a random variable X, kurtosis can be derived from the moment generating function (MGF):
 
 $$M_X(t) = E[e^{tX}]$$
 
 The kurtosis is then:
 
 $$Kurt[X] = \frac{M_X^{(4)}(0)}{(M_X^{(2)}(0))^2}$$
 
 For a Normal distribution with mean $\mu$ and variance $\sigma^2$, the MGF is:
 
 $$M_X(t) = e^{\mu t + \frac{\sigma^2t^2}{2}}$$
 
 Taking derivatives and evaluating at t=0:
 
 $$M_X^{(4)}(0) = 3\sigma^4$$
 $$M_X^{(2)}(0) = \sigma^2$$
 
 Therefore:
 
 $$Kurt[X] = \frac{3\sigma^4}{(\sigma^2)^2} = 3$$
 
 This shows that a Normal distribution always has kurtosis = 3, while empirical returns typically have kurtosis > 3 (leptokurtic).



```python
# Import necessary packages
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# Calculate returns distribution statistics
returns_mean = df['returns'].mean()
returns_std = df['returns'].std()
returns_kurt = df['returns'].kurtosis() + 3  # Convert from excess kurtosis to regular kurtosis

# Generate normal distribution with same mean and std
normal_dist = np.random.normal(returns_mean, returns_std, 100000)
normal_kurt = 3  # Normal distribution has kurtosis of 3

# Create KDE plots
kde_returns = gaussian_kde(df['returns'])
kde_normal = gaussian_kde(normal_dist)

# Create evaluation points
x_range = np.linspace(min(df['returns'].min(), normal_dist.min()), 
                      max(df['returns'].max(), normal_dist.max()), 
                      1000)

# Create figure
fig = go.Figure()

# Add returns KDE
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_returns(x_range),
        name='Returns Distribution',
        line=dict(color='#39FF14', width=2),
        mode='lines'
    )
)

# Add normal KDE
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_normal(x_range),
        name='Normal Distribution',
        line=dict(color='#FF10F0', width=2),
        mode='lines'
    )
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    title=f'Returns Distribution vs Normal<br>Returns Kurtosis: {returns_kurt:.2f}, Normal Kurtosis: {normal_kurt:.2f}',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    title_text='Returns',
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



**Remark:** Should we model returns via a normal distribution with a parametric or moment-matching approach (take the average and variance of historic returns and assume that returns follow a normal distribution parameterized with those values) we will violently underestimate tail risk!  

Typically, news outlets will cite events as "X-sigma" events referencing how many standard deviations away from some mean quantity the event in discussion is and that is crazy unlikely - ever wonder why 6-$\sigma$ events happen "so often"?  They aren't actually 6-$\sigma$ events, we are underestimating tail risk!


```python
# Calculate empirical and normal 0.05% VaR
empirical_var = np.percentile(df['returns'], 0.05)
normal_var = np.percentile(normal_dist, 0.05)

# Calculate average loss beyond VaR (Expected Shortfall)
empirical_es = df['returns'][df['returns'] <= empirical_var].mean()
normal_es = normal_dist[normal_dist <= normal_var].mean()

# Calculate percent differences
var_diff_pct = (normal_var - empirical_var) / abs(empirical_var) * 100
es_diff_pct = (normal_es - empirical_es) / abs(empirical_es) * 100

# Create figure
fig = go.Figure()

# Add returns distribution with gradient fill
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_returns(x_range),
        line=dict(color='#39FF14', width=2.5),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(57, 255, 20, 0.2)'
    )
)

# Add normal distribution with gradient fill
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_normal(x_range),
        line=dict(color='#FF10F0', width=2.5),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 16, 240, 0.1)'
    )
)

# Add VaR lines with values
fig.add_vline(x=empirical_var, line_dash="dash", line_color="#39FF14",
              annotation=dict(
                  text=f"{empirical_var:.3f}<br>{empirical_es:.3f}",
                  x=empirical_var,
                  y=1.15,
                  yref="paper",
                  showarrow=False,
                  font=dict(size=12)
              ))
fig.add_vline(x=normal_var, line_dash="dash", line_color="#FF10F0",
              annotation=dict(
                  text=f"{normal_var:.3f}<br>{normal_es:.3f}",
                  x=normal_var,
                  y=0.85,
                  yref="paper",
                  showarrow=False,
                  font=dict(size=12)
              ))

# Update layout with enhanced styling
fig.update_layout(
    width=900,
    height=600,
    title=dict(
        text=f'Returns vs Normal Distribution<br>VaR Difference: {var_diff_pct:.1f}% | ES Difference: {es_diff_pct:.1f}%',
        x=0.5,
        y=0.95
    ),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12)
)

# Update axes with enhanced styling
fig.update_xaxes(
    title_text='Returns',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_font=dict(size=14)
)

fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_font=dict(size=14)
)

fig.show()

```



##### Stylized Facts of Volatility

1. Volatility Clustering: Periods of high volatility tend to cluster together, and periods of low volatility tend to cluster together. This means volatility shows persistence and autocorrelation.

2. Mean Reversion: While volatility clusters, it tends to revert back to a long-run average level over time. Extremely high or low volatility periods don't persist indefinitely.

3. Leverage Effect: Volatility tends to increase more after negative returns compared to positive returns of the same magnitude. This creates an asymmetric response.

4. Heavy Tails: Returns distributions show excess kurtosis (fat tails) compared to normal distribution, indicating more extreme events than expected.

5. Long Memory: Volatility shows long-range dependence, meaning past volatility can influence future volatility even after significant time lags. (some evidence that rough volatility models and fractional Brownian motions can capture this, other literature suggests the evidence is ill founded. . .)


```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Create subplots to show stylized facts
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Volatility Mean Reversion',
                                  'Leverage Effect'),
                    horizontal_spacing=0.1)

# Plot 1: Volatility Mean Reversion
rolling_vol = df['returns'].rolling(window=20).std() * np.sqrt(252)
long_term_vol = rolling_vol.mean()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=rolling_vol,
        mode='lines',
        name='20-day Rolling Volatility',
        line=dict(color='#00FFFF', width=1.5)
    ),
    row=1, col=1
)

fig.add_hline(
    y=long_term_vol,
    line_dash="dash",
    line_color="red",
    annotation_text="Long-term Average",
    row=1, col=1
)

# Plot 2: Leverage Effect
returns_lag = df['returns'].shift(1)
vol_change = rolling_vol.diff()

neg_returns = returns_lag[returns_lag < 0]
pos_returns = returns_lag[returns_lag > 0]
neg_vol_change = vol_change[returns_lag < 0]
pos_vol_change = vol_change[returns_lag > 0]

# Calculate average vol changes
avg_neg_change = neg_vol_change.mean()
avg_pos_change = pos_vol_change.mean()

fig.add_trace(
    go.Scatter(
        x=neg_returns,
        y=neg_vol_change,
        mode='markers',
        name=f'After Negative Returns (avg: {avg_neg_change:.4f})',
        marker=dict(color='#FF10F0', size=6, opacity=0.6)
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=pos_returns,
        y=pos_vol_change,
        mode='markers',
        name=f'After Positive Returns (avg: {avg_pos_change:.4f})',
        marker=dict(color='#39FF14', size=6, opacity=0.6)
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1600,
    height=600,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12)
)

# Update axes styling for all subplots
for i in range(1, 2):
    for j in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=i, col=j
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=i, col=j
        )

# Add specific axis labels
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Previous Return", row=1, col=2)

fig.update_yaxes(title_text="Volatility", row=1, col=1)
fig.update_yaxes(title_text="Volatility Change", row=1, col=2)

fig.show()

```



**Remark:** Clearly, all of these dynamics are important to capture in our model otherwise our estimates may be extremely far from reality.  Prior to Engle and ARCH, models were unable to capture these stylized facts of volatility, heteroskedasticity, and excess kurtosis in one parsimonious model - sure there were ways to capture it, but there is always a complexity/efficiency tradeoff. . .

---

#### 2.) 🌊 ARCH Models (Engle 1982)

##### *"Autoregressive Conditionally Heteroskedastic Models"*

##### ARCH(q) Models

$$y_t = \mu + \epsilon_t \quad \epsilon_t = \sigma_t z_t \quad z_t \sim i.i.d (0, 1)$$

- $y_t$ is the observed time series (e.g., returns at time t; we don't use price here as $y_t$ is defined as stationary in expectation in terms of $\mu$)
- $\mu$ is the constant mean of the process $y_t$, something like the mean return (typically *about zero*)
- $\epsilon_t$ is the rror term (shock, innovation, residual) at time t
- $z_t$ is an i.i.d random variable with mean 0 and variance 1 (often assumed $\sim N(0, 1)$ but other heavy tailed distributions are also relevant)

$$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2 + . . . + \alpha_q \epsilon_{t-q}^2$$
- $\sigma^t_t$ is the conditional variance of $\epsilon_t$ or the *volatility* given past information dictated by the order of the ARCH process
- $\alpha_0$ is the constant term in the variance equation and must be positive to ensure nonzero variance
- $\alpha_i$ is the coefficient on lagged squared residuals ($\epsilon_{t-i}^2$) and must also be nonnegative to ensure nonnegative variance


**Autoregressive :** Uses a linear combination of its own past values to estimate future states

**Heteroskedastic vs. Homoskedastic:** If a model is homoskedastic it proposes constant variance (volatility) which is not the case as observed in data! where if a model is heteroskedastic it propses variance (volatility) is not constant - more inline with what we observe in data!

**Conditionally Heteroskedastic:** Variance of errors in a model changes over time and is dependent on past information - quite useful here for volatility!

###### ______________________________________________________________________________________________________________________________________

##### Previous Models Failed to Capture the Dynamics Discussed Above

 *Historical volatility:* Simple rolling window standard deviation of returns

 *Exponential weighted moving average (EWMA):* Gives more weight to recent observations

 *Implied volatility:* Derived from option prices using Black-Scholes model

 *Stochastic volatility models:* Allow volatility to follow its own random process
 
 These models had limitations:
   - Could not capture volatility clustering well
   - Did not model the relationship between returns and volatility
   - Often assumed constant parameters over time

How can we compare the efficacy of these models?  We can benchmark against realized volatility in a forward looking sense!

We give the model data up to time $t-1$ and use it to try to forecast volatility at time $t$ and see which does better on average!

###### ______________________________________________________________________________________________________________________________________

##### Example: EWMA vs. ARCH(1)

**Remark:** There is a slight bias, even out of sample, through grid search with window size as this is acting on global path knowledge, but sufficient for our analysis as we apply the same methodology to produce OOS statistics for each technique. . .


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

# -----------------------
# Load data and prepare returns
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[30:]

# Realized Variance (Daily, annualized)
realized_var = (df['returns']**2) * 252  

# -----------------------
# Train/Test Split
# -----------------------
train_end_idx = int(len(df) * 0.7)  # 70% training
train_data = df.iloc[:train_end_idx]
test_data = df.iloc[train_end_idx:]
train_end = train_data.index[-1]

# -----------------------
# EWMA Forecast Function
# -----------------------
def ewma_forecast(returns, lam):
    ewma_var = np.zeros(len(returns))
    ewma_var[0] = returns.var()  # initialize with unconditional variance
    for t in range(1, len(returns)):
        ewma_var[t] = (
            lam * ewma_var[t-1] + (1-lam) * returns.iloc[t-1]**2
        )
    return ewma_var * 252  # annualize

# -----------------------
# Grid Search for Optimal λ (train set only)
# -----------------------
lambda_grid = np.linspace(0.01, 0.90, 100)
best_rmse = float('inf')
best_lambda = None

for lam in lambda_grid:
    ewma_train = ewma_forecast(train_data['returns'], lam)
    rmse = np.sqrt(((ewma_train - realized_var[train_data.index])**2).mean())
    if rmse < best_rmse:
        best_rmse = rmse
        best_lambda = lam

print(f"Best λ found: {best_lambda:.3f} (Train RMSE={best_rmse:.6f})")

# -----------------------
# Out-of-Sample Forecast using Best λ
# -----------------------
ewma_all = ewma_forecast(df['returns'], best_lambda)
ewma_forecast_var = pd.Series(ewma_all, index=df.index)

aligned = pd.DataFrame({
    'realized_var': realized_var,
    'ewma_var': ewma_forecast_var
})

# OOS performance
oos_data = aligned[train_end:]
mse = ((oos_data['ewma_var'] - oos_data['realized_var'])**2).mean()
rmse = np.sqrt(mse)
mae = (oos_data['ewma_var'] - oos_data['realized_var']).abs().mean()

X = sm.add_constant(oos_data['ewma_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

# Realized variance
fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance (Daily)',
        line=dict(color='#BC13FE', width=1.5)
    )
)

# EWMA variance forecast
fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['ewma_var'],
        mode='lines',
        name=f"EWMA Forecast (λ={best_lambda:.3f})",
        line=dict(color='#00FFFF', width=1.5)
    )
)

# Train/Test split marker
fig.add_shape(
    type='line',
    x0=train_end,
    x1=train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="EWMA Forecast vs Realized Variance (Daily, Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)
fig.show()

print("\nOut-of-Sample Performance Metrics:")
print(f"RMSE (Variance units): {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.2%}")

```

    Best λ found: 0.900 (Train RMSE=0.103561)




    
    Out-of-Sample Performance Metrics:
    RMSE (Variance units): 0.420223
    MAE: 0.146483
    R-squared: 5.12%



```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import product

# -----------------------
# Load data
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Realized variance (annualized)
realized_var = (df['returns']**2) * 252

# -----------------------
# ARCH(1) Forecast Function
# -----------------------
def arch1_forecast(returns, omega, alpha):
    """Generate ARCH(1) conditional variance forecasts (annualized)."""
    var = np.zeros(len(returns))
    var[0] = np.var(returns)  # unconditional variance init
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns.iloc[t-1]**2
    return var * 252

# -----------------------
# Candidate grids
# -----------------------
window_candidates = [50, 100, 250, 500, 750]  # you can add more
omega_grid = np.linspace(0.00001, 0.001, 10)
alpha_grid = np.linspace(0.05, 0.95, 10)

best_rmse = float("inf")
best_params = None
best_forecast = None
best_train_end = None

# -----------------------
# Grid Search over (window, omega, alpha)
# -----------------------
for window in window_candidates:
    if window >= len(df):
        continue  # skip if window > data
    
    # define train/test split based on window size
    train_data = df.iloc[:window]
    test_data = df.iloc[window:]
    train_end = train_data.index[-1]

    for omega, alpha in product(omega_grid, alpha_grid):
        # Forecast on training data
        var_train = arch1_forecast(train_data['returns'], omega, alpha)

        # Out-of-sample forecast
        var_all = pd.Series(index=df.index, dtype=float)
        var_all.iloc[:window] = var_train
        for t in range(window, len(df)):
            var_all.iloc[t] = (omega + alpha * df['returns'].iloc[t-1]**2) * 252

        # Compute OOS RMSE
        oos_data = pd.DataFrame({
            'realized': realized_var[train_end:],
            'arch_var': var_all[train_end:]
        }).dropna()
        rmse = np.sqrt(((oos_data['arch_var'] - oos_data['realized'])**2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (window, omega, alpha)
            best_forecast = var_all
            best_train_end = train_end

# -----------------------
# Report Best Params
# -----------------------
print(f"Best parameters: window={best_params[0]}, omega={best_params[1]:.6f}, alpha={best_params[2]:.6f}")
print(f"Out-of-sample RMSE: {best_rmse:.6f}")

# -----------------------
# Regression (OOS only)
# -----------------------
aligned = pd.DataFrame({
    'realized_var': realized_var,
    'arch_var': best_forecast
})

oos_data = aligned[best_train_end:]
X = sm.add_constant(oos_data['arch_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

print(f"Out-of-sample R²: {r2:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance',
        line=dict(color='#BC13FE', width=1.5)
    )
)

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['arch_var'],
        mode='lines',
        name=f'ARCH(1) Forecast (window={best_params[0]})',
        line=dict(color='#00FF7F', width=1.5)
    )
)

# Add vertical line for train/test split
fig.add_shape(
    type='line',
    x0=best_train_end,
    x1=best_train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=best_train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="ARCH(1) Forecast vs Realized Variance (Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)

fig.show()

```

    Best parameters: window=50, omega=0.000230, alpha=0.250000
    Out-of-sample RMSE: 0.248797
    Out-of-sample R²: 5.42%




###### ______________________________________________________________________________________________________________________________________

##### Why the ARCH(q) Model was Innovative?

<u> **Conditional and Unconditional Expectation** </u>

Given a filtration $\mathcal{F}_{t-1}$, for an ARCH(q) process $\epsilon_t = \sigma_t z_t$ where $z_t \sim i.i.d(0,1)$:

$E[\epsilon_t|\mathcal{F}_{t-1}] = E[\sigma_t z_t|\mathcal{F}_{t-1}] = \sigma_t E[z_t|\mathcal{F}_{t-1}] = 0$

$E[\epsilon_t] = E[E[\epsilon_t|\mathcal{F}_{t-1}]] = 0$

<u> **Conditional Variance** </u>

The conditional variance follows directly from the ARCH(q) specification:

$Var(\epsilon_t|\mathcal{F}_{t-1}) = \sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2$

<u> **Unconditional Variance** </u>

The unconditional variance for an ARCH(q) process is:

$Var(\epsilon_t) = E[\epsilon_t^2] = \frac{\alpha_0}{1-\sum_{i=1}^q \alpha_i}$

This exists when $\sum_{i=1}^q \alpha_i < 1$

The distribution itself is leptokurtic as it is a combination of a series of Gaussian innovations with different variances, big modelling innovation proven below!

<u> **Kurtosis and Fat Tails** </u>

For an ARCH(q) process, assuming $z_t$ is normally distributed:

$Kurt(\epsilon_t) = \frac{E[\epsilon_t^4]}{(E[\epsilon_t^2])^2} = 3\frac{1-(\sum_{i=1}^q \alpha_i)^2}{1-3\sum_{i=1}^q \alpha_i^2}$

This exists when $3\sum_{i=1}^q \alpha_i^2 < 1$ and is greater than 3 (the normal distribution's kurtosis), indicating fat tails.


###### ______________________________________________________________________________________________________________________________________

**TL;DR : Given an ARCH(1) Process**
- Zero conditional and unconditional mean
- Conditional variance: $\alpha_0 + \alpha_1 \epsilon_{t-1}^2$ which is time-varying and depends on past shocks better capturing what we observed in data
- Unconditional variance: $\frac{\alpha_0}{1 - \alpha_1}$ if $\alpha_1 < 1$ which is dictated by the model fit and parameters estimated 
- Kurtosis: Captures kurtosis in excess to a normal distribution assuming $3\alpha_1^2 < 1$ which is what we observe in data
- Volatility clustering is captured naturally! 

---

#### 3.) 🧮 GARCH Models (Bollerslev 1986)

##### *"Generalized Autoregressive Conditionally Heteroskedastic Models"*

##### GARCH(p, q)
 
 $$y_t = \mu + \epsilon_t \quad \epsilon_t = \sigma_t z_t \quad z_t \sim i.i.d(0, 1)$$
 
 - $y_t$ is the observed time series (e.g., returns at time t)
 - $\mu$ is the constant mean of the process, typically close to zero
 - $\epsilon_t$ is the error term (shock/innovation) at time t
 - $z_t$ is an i.i.d random variable with mean 0 and variance 1
 
 $$\sigma_t^2 = \alpha_0 + \sum_{i=0}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=0}^p \beta_j \sigma_{t-j}^2$$
 
 - $\sigma_t^2$ is the conditional variance or volatility at time t
 - $\alpha_0$ is the constant term (must be positive)
 - $\alpha_i$ are coefficients on lagged squared residuals (must be nonnegative)
 - $\beta_1$ are coefficients on lagged conditional variances (must be nonnegative)
 
 The most widely used case is a GARCH(1, 1)
 
 $$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

##### Why use GARCH over ARCH?

GARCH is a more parsimonious ARCH model, in fact, an ARCH($\infty$) is equal to a GARCH(1, 1)

 Consider an ARCH(∞) process:
 $$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2 + ...$$
 
 We can rewrite this as:
 $$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2 + ...$$
 $$\sigma_{t-1}^2 = \alpha_0 + \alpha_1 \epsilon_{t-2}^2 + \alpha_2 \epsilon_{t-3}^2 + ...$$
 
 Multiply the second equation by β:
 $$\beta\sigma_{t-1}^2 = \beta\alpha_0 + \beta\alpha_1 \epsilon_{t-2}^2 + \beta\alpha_2 \epsilon_{t-3}^2 + ...$$
 
 Subtracting this from the first equation:
 $$\sigma_t^2 - \beta\sigma_{t-1}^2 = \alpha_0(1-\beta) + \alpha_1 \epsilon_{t-1}^2 + (\alpha_2-\beta\alpha_1)\epsilon_{t-2}^2 + ...$$
 
 Rearranging:
 $$\sigma_t^2 = \alpha_0(1-\beta) + \alpha_1 \epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$
 
 This is exactly the form of a GARCH(1,1) process with parameters:
 - $\omega = \alpha_0(1-\beta)$
 - $\alpha = \alpha_1$
 - $\beta = \beta$
 
 Therefore, any ARCH(∞) process can be represented as a more parsimonious GARCH(1,1) process.

We can capture the same dynamics in the infinite ARCH process as a simple GARCH(1,1), wild!

##### Example: GARCH Model


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import product

# -----------------------
# Load data
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[30:]

# Realized variance (annualized)
realized_var = (df['returns']**2) * 252

# -----------------------
# GARCH(1,1) Forecast Function
# -----------------------
def garch11_forecast(returns, omega, alpha, beta):
    """Generate GARCH(1,1) conditional variance forecasts (annualized)."""
    var = np.zeros(len(returns))
    var[0] = np.var(returns)  # unconditional variance init
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns.iloc[t-1]**2 + beta * var[t-1]
    return var * 252

# -----------------------
# Candidate grids
# -----------------------
window_candidates = [50, 100, 250, 500, 750]
omega_grid = np.linspace(0.00001, 0.001, 5)
alpha_grid = np.linspace(0.05, 0.95, 5)
beta_grid = np.linspace(0.05, 0.95, 5)

best_rmse = float("inf")
best_params = None
best_forecast = None
best_train_end = None

# -----------------------
# Grid Search over (window, omega, alpha, beta)
# -----------------------
for window in window_candidates:
    if window >= len(df):
        continue
    
    train_data = df.iloc[:window]
    test_data = df.iloc[window:]
    train_end = train_data.index[-1]

    for omega, alpha, beta in product(omega_grid, alpha_grid, beta_grid):
        if alpha + beta >= 1:  
            continue  # ensure stationarity

        # Forecast on training data
        var_train = garch11_forecast(train_data['returns'], omega, alpha, beta)

        # Out-of-sample forecast
        var_all = pd.Series(index=df.index, dtype=float)
        var_all.iloc[:window] = var_train
        for t in range(window, len(df)):
            var_all.iloc[t] = (omega +
                               alpha * df['returns'].iloc[t-1]**2 +
                               beta * var_all.iloc[t-1]/252) * 252

        # Compute OOS RMSE
        oos_data = pd.DataFrame({
            'realized': realized_var[train_end:],
            'garch_var': var_all[train_end:]
        }).dropna()
        rmse = np.sqrt(((oos_data['garch_var'] - oos_data['realized'])**2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (window, omega, alpha, beta)
            best_forecast = var_all
            best_train_end = train_end

# -----------------------
# Report Best Params
# -----------------------
print(f"Best parameters: window={best_params[0]}, "
      f"omega={best_params[1]:.6f}, alpha={best_params[2]:.6f}, beta={best_params[3]:.6f}")
print(f"Out-of-sample RMSE: {best_rmse:.6f}")

# -----------------------
# Regression (OOS only)
# -----------------------
aligned = pd.DataFrame({
    'realized_var': realized_var,
    'garch_var': best_forecast
})

oos_data = aligned[best_train_end:]
X = sm.add_constant(oos_data['garch_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

print(f"Out-of-sample R²: {r2:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance',
        line=dict(color='#BC13FE', width=1.5)
    )
)

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['garch_var'],
        mode='lines',
        name=f'GARCH(1,1) Forecast (window={best_params[0]})',
        line=dict(color='#00FF7F', width=1.5)
    )
)

fig.add_shape(
    type='line',
    x0=best_train_end,
    x1=best_train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=best_train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="GARCH(1,1) Forecast vs Realized Variance (Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)

fig.show()

```

    Best parameters: window=50, omega=0.000010, alpha=0.275000, beta=0.500000
    Out-of-sample RMSE: 0.253176
    Out-of-sample R²: 8.18%




###### ______________________________________________________________________________________________________________________________________

**TL;DR: GARCH(1,1) Process Compared to ARCH(∞)**

Given a GARCH(1,1) Process:
- Zero conditional and unconditional mean
- Conditional variance: $\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$
- Can be rewritten as an ARCH(∞) process by recursive substitution:
  $\sigma_t^2 = \frac{\omega}{1-\beta} + \alpha\sum_{i=1}^{\infty}\beta^{i-1}\epsilon_{t-i}^2$
- Unconditional variance: $\frac{\omega}{1-\alpha-\beta}$ if $\alpha + \beta < 1$
- More parsimonious than ARCH with similar benefits:
  - Captures volatility clustering
  - Models excess kurtosis
  - Time-varying conditional variance
- Advantage over ARCH: Requires fewer parameters while capturing long-memory dependence through the $\beta$ term


**Remark:** It is well documented daily squared returns tend to have low explanatory power ($R^2 < 10\%$) as we have observed herein), however, intraday returns appear to better represent the latent volatility process and ARCH/GARCH can explain up to $40-60\%$ of such variation - impressive!

---

#### 4.) 🎯 Applications of ARCH/GARCH Models 

##### Portfolio Risk Management

 Value at Risk (VaR) is a critical risk measure in portfolio management, defined as:
 
 $$VaR_{\alpha} = -\sigma \cdot z_{\alpha}$$
 
 where $z_{\alpha}$ is the $\alpha$-quantile of the standard normal distribution and $\sigma$ is the volatility.
 
 The traditional parametric VaR approach assumes homoskedasticity (constant variance), using:
 
 $$\sigma_{global} = \sqrt{\frac{1}{T-1}\sum_{t=1}^T (r_t - \bar{r})^2}$$
 
 However, this fails to capture the dynamic nature of volatility in financial markets. GARCH models provide superior VaR estimates by modeling time-varying volatility:
 
 $$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$
 
 This allows the VaR estimate to adapt to changing market conditions, providing more accurate risk forecasts during both calm and volatile periods.



```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# -----------------------
# Parameters
# -----------------------
alpha = 0.05   # 5% VaR
z_alpha = norm.ppf(alpha)  # ≈ -1.645

# -----------------------
# Naïve (parametric) VaR
# -----------------------
# Use unconditional volatility (std of returns)
sigma_naive = df['returns'].std()
sigma_naive_daily = sigma_naive / np.sqrt(252)   # convert annual to daily if needed

VaR_naive = z_alpha * sigma_naive_daily

# -----------------------
# GARCH(1,1) VaR
# -----------------------
sigma_daily = np.sqrt(best_forecast / 252)
VaR_garch = z_alpha * sigma_daily

# -----------------------
# Create DataFrame
# -----------------------
var_df = pd.DataFrame({
    'returns': df['returns'],
    'VaR_Naive': VaR_naive,
    'VaR_GARCH': VaR_garch
}).dropna()

# -----------------------
# Exceedances
# -----------------------
exceed_naive = var_df[var_df['returns'] < var_df['VaR_Naive']]
exceed_garch = var_df[var_df['returns'] < var_df['VaR_GARCH']]

prop_exceed_naive = len(exceed_naive) / len(var_df)
prop_exceed_garch = len(exceed_garch) / len(var_df)

print(f"Proportion of exceedances (Naïve): {prop_exceed_naive:.2%}")
print(f"Proportion of exceedances (GARCH): {prop_exceed_garch:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

# Portfolio returns
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['returns'],
        mode='lines',
        name='Portfolio Returns',
        line=dict(color='white', width=1)
    )
)

# Naïve VaR
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['VaR_Naive'],
        mode='lines',
        name=f'5% Naïve VaR (Exceed: {prop_exceed_naive:.2%})',
        line=dict(color='orange', width=2, dash='dot')
    )
)

# GARCH VaR
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['VaR_GARCH'],
        mode='lines',
        name=f'5% GARCH(1,1) VaR (Exceed: {prop_exceed_garch:.2%})',
        line=dict(color='#FF3131', width=2, dash='dash')
    )
)

# Highlight exceedances (Naïve)
fig.add_trace(
    go.Scatter(
        x=exceed_naive.index,
        y=exceed_naive['returns'],
        mode='markers',
        name='Naïve Exceedances',
        marker=dict(color='orange', size=7, symbol='x')
    )
)

# Highlight exceedances (GARCH)
fig.add_trace(
    go.Scatter(
        x=exceed_garch.index,
        y=exceed_garch['returns'],
        mode='markers',
        name='GARCH Exceedances',
        marker=dict(color='red', size=7, symbol='cross')
    )
)

# Layout
fig.update_layout(
    title="1-Day 5% VaR: Naïve vs GARCH(1,1)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Return / VaR"
)

fig.show()

```

    Proportion of exceedances (Naïve): 40.65%
    Proportion of exceedances (GARCH): 9.74%




**Remark:** Typically with Value at Risk (VaR) we expect for example 1-Day 5% VaR to imply something of the effect that "1 out of 20 trading days, portfolio losses will exceed this threshold" - in the naive approach we see we underestimate this risk, we talked about this earlier with the parametric approach to modeling returns!  The GARCH(1, 1) model is far more capable at capturing this tail risk as its exceedences occur 9.74% of the time rather than the 40.65% (ouch!) offered by the naive methodology.  No wonder these models are implemented in practice!

---

#### 5.) 💭 Closing Thoughts and Future Topics

TL;DW Executive Summary
- Volatility is an unobservable measure of variability in the return space
- We can proxy for volatility in a backward looking sense (historic or realized volatility) or in a forward looking sense (implied volatility)
- There are many stylized facts about volatility including the leverage effect, volatility clustering, excess kurtosis (fat tails, leptokurtic return distributions)
- Naive parametric models fail to capture these dynamics and severly underestimate tail risk - a big problem!
- Engle proposed ARCH, an autoregressive conditionally heteroskedastic model capable of modeling these dynamics improving forecasts!
- Bollerslev proposed a generalized ARCH model (GARCH) which is an infinite order ARCH model, thus a more parsimonious version
- GARCH can capture richer dynamics with fewer lags, impressive!
- These volatility models outperform other models that do not account for dynamics especially in the context of risk modelling as we saw in our VaR example


**Future Topics**

Technical Videos and Other Discussions

- Extensions to ARCH/GARCH Models and Applications

- Kalman Filters and Other Time Series Analysis Techniques

- Deriving and solving the Black-Scholes equation (understanding option exposures and where they come from theoretically)

- Calibrating a stochastic model to a market implies volatility surface (Heston example)

- Neural networks and machine learning approaches to approximate pricing functionals

- Methodology for simulating stochastic processes

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Extensions and Implementations of GARCH in a Live System

- Kalman Filter and Other Time Series Techniques Applied to Live Pricing

- How to Build an Earnings Event Options Trading Dashboard

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
