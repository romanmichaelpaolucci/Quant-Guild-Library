### 📈 How to Price Options with Monte Carlo Simulation

##### ▶️ Related Quant Guild Videos:

- [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)

- [Finite Differences Option Pricing for Quant Finance](https://youtu.be/uzbveN8n34U)

- [Brownian Motion for Quant Finance](https://youtu.be/jiAdz9W4aDI)

- [How Physics Accidentally Proved the Black-Scholes Model](https://youtu.be/IIzGqL3ChEs)

- [Quant Explains Algorithmic Market-Making](https://youtu.be/aVzFKwyzwM0)

###### ______________________________________________________________________________________________________________________________________

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)



##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)

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

### 📖 Sections

#### 1.) 🎲 Black-Scholes Model and Fundamental Theorem of Asset Pricing

- Black-Scholes Equation and Replication Argument

- No Arbitrage Price Implies by the Fundamental Theorem of Asset Pricing

- Feynman-Kac Theorem Connecting Parabolic PDEs to Conditional Expectation

#### 2.) 🎯 Law of Large Numbers (LLN)

- Analytical Expectation of a Random Variable

- Empirical Expectation of a Random Variable

- Analytical Expectation of a Stochastic Process

- Empirical Expectation of a Stochastic Process

#### 3.) 📈 Simulating Option Prices

- Selecting an Options Contract

- Simulating the Underlying Asset Dynamics

- Finding the Expectation and Discounting it to Present

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 🎲 Black-Scholes Model and Fundamental Theorem of Asset Pricing

The Black-Scholes Portfolio Replication Argument Implies the Black-Scholes Equation:

$$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

The Fundamental Theorem of Asset Pricing (FTAP) Implies the Arbitrage Free Price:
 
$$
V_0 = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}[ \text{Payoff}(S_T) ]
$$

Feynman-Kac Theorem Suggests that if $V$ is a Solution to the Black-Scholes PDE, it **MUST BE** the Arbitrage Free Price Implied by the FTAP:

 $$
 \boxed{
   V(t, S) = \mathbb{E}^\mathbb{Q}\left[ e^{-r(T-t)} \text{Payoff}(S_T) \mid S_t = S \right]
 }
 $$

In other words,

To find the risk-neutral price of our option we can either solve a partial differential equation or find the risk-neutral expectation!


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Black-Scholes Analytical Call Value as function of Spot ---
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """Black-Scholes price for a European Call option."""
    S = np.asarray(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# --- Parameters ---
S0 = 100     # Initial spot
K = 105      # Strike
T = 1.0      # Time to maturity (years)
r = 0.04     # Risk-free rate
sigma = 0.24 # Volatility

# --- Spot Range for Black-Scholes Plot ---
S_range = np.linspace(60, 140, 100)
bs_values = black_scholes_call_price(S_range, K, T, r, sigma)

# --- Simulate Sample Paths of Geometric Brownian Motion (risk-neutral under Q) ---

N_paths = 8    # Number of paths to plot
N_steps = 120  # Time steps
dt = T / N_steps

np.random.seed(1)
dW = np.random.normal(0, np.sqrt(dt), size=(N_paths, N_steps))
W = np.cumsum(dW, axis=1)
t_grid = np.linspace(0, T, N_steps+1)
S_paths = np.zeros((N_paths, N_steps+1))
S_paths[:,0] = S0
for i in range(N_paths):
    X = (r - 0.5*sigma**2) * t_grid[1:] + sigma*W[i]
    S_paths[i,1:] = S0 * np.exp(X)

# --- Make Figure: Left = Black-Scholes vs Spot, Right = Brownian Sample Paths ---
fig = make_subplots(
    rows=1, cols=2, 
    column_widths=[0.53, 0.47],
    subplot_titles=("", "") # remove left/right titles
)

# --- Left Panel: Black-Scholes Curve ---
fig.add_trace(
    go.Scatter(
        x=S_range,
        y=bs_values,
        mode="lines",
        line=dict(color="#23AFFF", width=4),
        name="BS Value"
    ),
    row=1, col=1
)

# Highlight S0 vertical
fig.add_trace(
    go.Scatter(
        x=[S0, S0],
        y=[0, black_scholes_call_price(S0, K, T, r, sigma)],
        mode="lines",
        line=dict(color="#FFC300", dash="dot", width=3),
        showlegend=False
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[S0],
        y=[black_scholes_call_price(S0, K, T, r, sigma)],
        mode="markers+text",
        marker=dict(size=10, color="#FFC300"),
        text=[f"${black_scholes_call_price(S0, K, T, r, sigma):.2f}$"],
        textposition="top right",
        showlegend=False
    ),
    row=1, col=1
)

fig.update_xaxes(
    title_text="Spot Price S0",
    row=1, col=1,
    range=[58, 142],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="Option Value C(S0, 0)",
    row=1, col=1,
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)

# --- Right Panel: Sample Paths w/ Strike and Maturity ---

for i in range(N_paths):
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=S_paths[i],
            mode="lines",
            line=dict(width=2, color="#9B59B6"),
            opacity=0.74,
            showlegend=False
        ),
        row=1, col=2
    )

# Strike: Horizontal
fig.add_trace(
    go.Scatter(
        x=[0, T],
        y=[K, K],
        mode="lines",
        line=dict(color="#FFD700", width=3, dash="dash"),
        name="Strike K"
    ),
    row=1, col=2
)

# Maturity: Vertical
fig.add_trace(
    go.Scatter(
        x=[T, T],
        y=[min(S_paths.min(), S0*0.4), max(S_paths.max(), S0*1.6)],
        mode="lines",
        line=dict(color="#12C95A", width=3, dash="dot"),
        name="Maturity T"
    ),
    row=1, col=2
)

fig.update_xaxes(
    title_text="Time t",
    row=1, col=2,
    range=[-0.01, T + 0.02],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="Spot Price St",
    row=1, col=2,
    range=[min(50, S_paths.min()-5), max(150, S_paths.max()+5)],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)

# --- General Style ---
fig.update_layout(
    height=500,
    width=1150,  # wider chart
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FAF7E0', size=16),
    title=dict(
        text="<br>Black-Scholes Value and Monte Carlo Simulations<br>"
             "<span style='font-size:18px;color:#b7ead8'>Analytical Value & Simulated Paths (Risk-Neutral)</span><br>",
        y=0.98,
        x=0.5,
        font=dict(color="#FAFAFA", size=25)
    ),
    margin=dict(l=44, r=26, t=95, b=62),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.18,
        xanchor="center",
        x=0.5,
        font=dict(size=16)
    )
)

fig.show()
```

This gives us two ways of solving for the option price.

If the PDE doesn't have an analytical solution, we can use Finite-Differences to approximate the solution (check out my video)

If we prefer solving for the expectation using Monte Carlo Simulation (it can be easier in some (many) cases) why can we do this?  We know the values are equal by Feynman-Kac but why does simulation give us the correct expectation?

---

#### 2.) 🎯 Law of Large Numbers (LLN)

##### The Expectation of a Discrete Random Variable is Given By

$$ \mathbb{E}[X] = \sum_{i} x_i \, \mathbb{P}(X = x_i) $$

##### Example: Dice Roll
 
 For a fair 6-sided die, the probability mass function (PMF) assigns equal probability to each face:
 
 $$
 \mathbb{P}(X = x) = 
 \begin{cases}
     \dfrac{1}{6} & \text{if } x \in \{1, 2, 3, 4, 5, 6\} \\
     0 & \text{otherwise}
 \end{cases}
 $$
 
 
 Calculate the expectation (mean) using:
 
 $$
 \begin{align*}
 \mathbb{E}[X] &= \sum_{i=1}^{6} x_i \, \mathbb{P}(X = x_i) \\
               &= 1\times\dfrac{1}{6} + 2\times\dfrac{1}{6} + 3\times\dfrac{1}{6} + 4\times\dfrac{1}{6} + 5\times\dfrac{1}{6} + 6\times\dfrac{1}{6} \\
               &= \dfrac{1+2+3+4+5+6}{6} \\
               &= \dfrac{21}{6} \\
               &= 3.5
 \end{align*}
 $$



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Dice PMF Setup ---
dice_faces = np.arange(1, 7)
pmf = np.full(6, 1/6)  # Fair dice, uniform PMF
products = dice_faces * pmf
expectation = products.sum()  # Should be 3.5 for a fair 6-sided die

# --- Helper to Generate Animation Frame at Given Step ---
def make_pmf_expectation_fig(step):
    value = dice_faces[step]
    prob = pmf[step]
    product = products[step]
    partial_sum = products[:step+1].sum()
    highlight_sum = partial_sum
    # For right bar, show sum up to current step
    right_bar_val = partial_sum

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=(
            # Add extra line breaks before and after to increase vertical padding
            "<br>Dice Probability (blue) and Face Value (gold)<br>",
            "<br>Accumulating Expectation (Sum up to current step)<br>"
        ),
        specs=[[{"secondary_y": True}, {}]]
    )

    # --- Left: Twin Bar Chart with Two y-axes (dots/labels removed) ---
    pmf_bar_colors = ["#23AFFF"]*6
    pmf_bar_colors[step] = "#104AF9"
    face_bar_colors = ["#D8A040"]*6
    face_bar_colors[step] = "#FFE066"

    # ADD HIGHLIGHT (thicker border) for current step
    # Set highlight border color to green for both
    highlight_border_color = "#12C95A"  # vibrant green
    pmf_bar_line_colors = ["rgba(0,0,0,0)"] * 6
    pmf_bar_line_colors[step] = highlight_border_color
    pmf_bar_line_widths = [0]*6
    pmf_bar_line_widths[step] = 4

    face_bar_line_colors = ["rgba(0,0,0,0)"] * 6
    face_bar_line_colors[step] = highlight_border_color
    face_bar_line_widths = [0]*6
    face_bar_line_widths[step] = 4

    # Probability bar
    fig.add_trace(
        go.Bar(
            x=dice_faces,
            y=pmf,
            marker=dict(
                color=pmf_bar_colors,
                line=dict(color=pmf_bar_line_colors, width=pmf_bar_line_widths)
            ),
            width=0.38,
            name="Probability",
            showlegend=False,
            offset=-0.16
        ),
        row=1, col=1, secondary_y=False
    )
    # Face value bar
    fig.add_trace(
        go.Bar(
            x=dice_faces,
            y=dice_faces,
            marker=dict(
                color=face_bar_colors,
                line=dict(color=face_bar_line_colors, width=face_bar_line_widths)
            ),
            width=0.34,
            name="Face Value",
            showlegend=False,
            offset=0.16
        ),
        row=1, col=1, secondary_y=True
    )

    # Dots and labels on left chart have been removed

    fig.update_xaxes(
        title_text="Dice Face Value",
        tickvals=dice_faces,
        range=[0.18, 7.1],  # expand left margin and give more space on xlim (was [0.5, 6.5])
        showgrid=False, row=1, col=1
    )
    fig.update_yaxes(
        title_text="Probability", range=[0, 0.22], row=1, col=1,
        secondary_y=False, showgrid=False,
        title_font=dict(color="#23AFFF"), tickfont=dict(color="#23AFFF")
    )
    fig.update_yaxes(
        title_text="Face Value", range=[0.5, 6.5], row=1, col=1,
        secondary_y=True, title_font=dict(color="#D8A040"), tickfont=dict(color="#D8A040"),
        showgrid=False
    )

    # --- Right: Accumulating green bar for partial sum ---
    # Highlight the right bar if this is the last step
    right_bar_marker = dict(color="#14D68B")
    if step == len(dice_faces)-1:
        right_bar_marker["line"] = dict(color="#12C95A", width=4)  # Use green for highlight
    else:
        right_bar_marker["line"] = dict(color="rgba(0,0,0,0)", width=0)

    fig.add_trace(
        go.Bar(
            x=["Sum"],
            y=[right_bar_val],
            marker=right_bar_marker,
            width=0.44,
            name="Partial Sum",
            showlegend=False
        ),
        row=1, col=2
    )

    # Draw final expectation as a faint 'target' horizontal dashed line for illustration
    fig.add_trace(
        go.Scatter(
            x=["Sum","Sum"],
            y=[expectation, expectation],
            mode="lines",
            line=dict(color="#2CFAE6", dash="dash", width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )
    # Also show value numerically atop the bar
    fig.add_trace(
        go.Scatter(
            x=["Sum"],
            y=[right_bar_val],
            mode="text",
            text=[f"{right_bar_val:.2f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(
        title_text="", row=1, col=2, tickvals=["Sum"]
    )
    fig.update_yaxes(
        title_text="Accumulated Value",
        row=1, col=2,
        range=[0, 4],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.18)'
    )

    # --- Figure Layout ---
    fig.update_layout(
        height=470,                                 # Taller for more title-to-chart space (was 420)
        width=1200,
        plot_bgcolor='rgba(0,0,0,0)',   # Fully transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Fully transparent
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            # Add line breaks before and after to give more vertical padding
            text=(
                "<br>"  # top vertical padding
                f"Expectations using a Die<br>Accumulated sum: {right_bar_val:.2f} → Expectation (total) = 3.5"
                "<br><br>"  # bottom vertical padding
            ),
            y= .99,                    # Lower the charts slightly from subtitle (was 0.81)
            font=dict(color="#FAFAFA", size=25, family=None),  # Larger size, off-white
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        # More top margin (t) and more bottom margin (b) for visually loose spacing
        margin=dict(l=42, r=24, b=85, t=160),  # More space top/bottom (was t=125, b=60)
        barmode='overlay'
    )

    # Also add more vertical space between subplots and any annotations by customizing the subplot_row_heights and spacing/annotations if needed (not strictly necessary for this case).

    return fig

# --- Create Animation Frames for Each Dice Face ---
pmf_frames = [
    go.Frame(
        data=make_pmf_expectation_fig(step).data,
        name=str(step),
        layout=make_pmf_expectation_fig(step).layout
    )
    for step in range(len(dice_faces))
]

fig_pmf = make_pmf_expectation_fig(0)
fig_pmf.frames = pmf_frames

# --- Play Button ---
fig_pmf.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.32,  # Lower play button for more margin (was y=-0.27)
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 700, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_pmf.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Empirically, We Observe Convergence to the Same Value by the Law of Large Numbers (LLN)


 **Weak Law of Large Numbers:**  
 $$
 \forall \varepsilon > 0,\ \lim_{n\to\infty} \Pr\left(|\overline{X}_n-\mu| > \varepsilon\right) = 0
 $$
 *(The sample mean converges in probability to $\mu$, but not necessarily for every sequence.)*

 **Strong Law of Large Numbers:**  
 $$
 \Pr\left(\lim_{n\to\infty} \overline{X}_n = \mu\right) = 1
 $$
 *(The sample mean almost surely converges to $\mu$: this is a much stronger guarantee.)*
 
In other words, we can estimate the population mean by randomly drawing samples from the same distribution and calculating their average. 

As we collect more samples, the average of these values will get closer and closer to the true mean of the original population.


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Empirical Proportion + Empirical Mean LLN Animation Setup ---

dice_faces = np.arange(1, 7)
pmf = np.full(6, 1/6)
expectation = 3.5

np.random.seed(42)
N = 250  # Number of samples ("rolls") to simulate

samples = np.random.choice(dice_faces, size=N, replace=True)
empirical_counts = np.zeros((N, len(dice_faces)))
empirical_probs = np.zeros((N, len(dice_faces)))
empirical_means = np.zeros(N)
for i in range(N):
    rolls_so_far = samples[:i+1]
    for k, face in enumerate(dice_faces):
        empirical_counts[i, k] = np.sum(rolls_so_far == face)
        empirical_probs[i, k] = empirical_counts[i, k] / (i+1)
    empirical_means[i] = np.mean(rolls_so_far)

def make_lln_empirical_fig(step):
    # step is 0-based (number of data points so far is step+1)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.54, 0.46],
        subplot_titles=(
            "<br>Empirical Proportions vs Theoretical PMF<br>(as Number of Rolls Increases)",
            "<br>Empirical Mean vs Theoretical Mean<br>(Law of Large Numbers in Action)"
        )
    )

    # --- Left: Bar chart, theoretical PMF, and empirical freq ---
    # Theoretical PMF bars (static)
    fig.add_trace(
        go.Bar(
            x=dice_faces - 0.12,  # shift left a touch
            y=pmf,
            width=0.24,
            name='Theoretical $P(X=x)$',
            marker_color='#23AFFF',
            opacity=0.45,
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )

    # Empirical frequency bars so far
    fig.add_trace(
        go.Bar(
            x=dice_faces + 0.12,  # shift right a touch
            y=empirical_probs[step],
            width=0.24,
            name='Empirical Proportion',
            marker_color='#FFE066',
            opacity=1.0,
            showlegend=True if step==0 else False,
            text=[f"{v:.2f}" for v in empirical_probs[step]],
            textposition="outside"
        ),
        row=1, col=1
    )

    # -- Removed roll count annotation here --

    fig.update_xaxes(
        title_text="Dice Face Value",
        tickvals=dice_faces,
        range=[0.7, 6.3],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Probability",
        range=[0, 0.44],
        row=1, col=1,
        tickformat=".2f"
    )

    # --- Right: Bar chart for mean --- 
    # Means displayed as two bars side by side
    means_x = ["Theoretical Mean", "Empirical Mean"]
    means_y = [expectation, empirical_means[step]]

    fig.add_trace(
        go.Bar(
            x=means_x,
            y=means_y,
            width=0.44,
            name="Mean Values",
            marker_color=["#23AFFF", "#FFE066"],
            opacity=0.9,
            showlegend=False,
            text=[f"{expectation:.3f}", f"{empirical_means[step]:.3f}"],
            textposition="outside"
        ),
        row=1, col=2
    )

    # Optionally show convergence path (scatter curve for empirical mean)
    if step > 1:
        fig.add_trace(
            go.Scatter(
                x=["Empirical Mean"]*step,
                y=empirical_means[:step],
                mode="lines+markers",
                line=dict(color='#BB9900', width=2, dash='dot'),
                marker=dict(size=3, color='#FFD700'),
                name="Empirical Mean\n(trace)",
                showlegend=False,
                hoverinfo="y+x"
            ),
            row=1, col=2
        )

    # Horizontal target band for expectation, for visibility
    fig.add_shape(
        type="rect",
        xref="x2", yref="y2",
        x0=-0.5, x1=1.5, y0=expectation-0.03, y1=expectation+0.03,
        fillcolor="rgba(30,250,240,0.09)",
        line_width=0, row=1, col=2
    )

    fig.update_xaxes(
        title_text="",
        row=1, col=2,
        tickvals=means_x
    )
    fig.update_yaxes(
        title_text="Sample Mean Value",
        range=[1, 6],
        row=1, col=2
    )

    # --- Layout ---
    fig.update_layout(
        height=450, width=1000,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text="<br>Law of Large Numbers: Empirical Frequencies & Mean Converge to Theoretical Values<br>",
            y= .99, font=dict(color="#FAFAFA", size=23),
            x=0.5, xanchor="center", yanchor="top"
        ),
        margin=dict(l=45, r=22, b=80, t=98),
        barmode='group'   # group bars for means
    )
    return fig

# --- Create Animation Frames ---
lln_frames = [
    go.Frame(
        data=make_lln_empirical_fig(n).data,
        name=str(n),
        layout=make_lln_empirical_fig(n).layout
    )
    for n in range(N)
]

fig_lln = make_lln_empirical_fig(0)
fig_lln.frames = lln_frames

# --- Animation play button (no slider) ---
fig_lln.update_layout(
    updatemenus=[
        {
            'type': 'buttons',
            'x': 0.48, 'y': -0.22,
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }
    ]
)

fig_lln.show()

```

###### ______________________________________________________________________________________________________________________________________

##### The Expectation of a Continuous Random Variable is Given By

$$ \mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f_X(x)\, dx $$

 ##### Example: Standard Normal Distribution
 
 For a standard normal random variable $X \sim \mathcal{N}(0, 1)$, the probability density function (PDF) is given by:
 
 $$
 f_X(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}, \quad \text{for } x \in (-\infty, \infty)
 $$
 
 The expectation (mean) is defined as:
 
 $$
 \mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f_X(x)\, dx
 $$
 
 Plugging in the standard normal PDF:
 
 $$
 \begin{align*}
 \mathbb{E}[X] &= \int_{-\infty}^\infty x \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\, dx = 0
 \end{align*}
 $$
 
 This result holds because the function $x \cdot f_X(x)$ is an odd function, so its integral over $(-\infty, \infty)$ is zero. 
 
 Thus, the expected value of a standard normal random variable is $0$.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from numpy import trapezoid

def simps(y, x):
    return trapezoid(y, x)

# --- Normal Distribution Setup ---
mu = 0
sigma = 1
x = np.linspace(-4, 4, 400)  # Higher resolution for smoothness
pdf = norm.pdf(x, mu, sigma)
dx = x[1] - x[0]
norm_expectation = simps(x * pdf, x)  # Should be 0 for standard normal

# --- Animation sweep setup ---
n_steps = 120
step_indices = np.linspace(0, len(x)-1, n_steps, dtype=int)

def make_normal_integration_fig(step_idx):
    cur_idx = step_indices[step_idx]
    x_sweep = x[:cur_idx+1]
    pdf_sweep = pdf[:cur_idx+1]
    partial_sum = simps(x_sweep * pdf_sweep, x_sweep)

    # --- Color/style palette (match discrete die style) ---
    left_primary = "#23AFFF"    # vivid blue for PDF
    left_secondary = "#14D68B"  # fill/area (greenish-teal)
    left_accent = "#FFE066"     # highlight sweep line (gold)
    left_density = "#D8A040"    # x*p(x) curve (gold/brown)
    right_bar_main = "#14D68B"  # green for accumulating bar
    right_bar_highlight = "#12C95A"   # border highlight
    faint_target = "#2CFAE6"    # blue/turq target
    text_color = "#FAF7E0"

    # --- Figure and subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=(
            "Normal PDF (blue) and x*p(x) (gold)",
            "Accumulated Expectation (integral partial sum)"
        ),
        specs=[[{"secondary_y": False}, {}]]
    )

    # --- Left: PDF, x*p(x) (gold), swept area, sweep line ---
    # PDF curve
    fig.add_trace(
        go.Scatter(
            x=x, y=pdf,
            mode='lines',
            line=dict(color=left_primary, width=2.3),
            name="Normal PDF",
            showlegend=False
        ),
        row=1, col=1
    )
    # x*p(x) curve (gold/brown dotted)
    fig.add_trace(
        go.Scatter(
            x=x, y=x*pdf,
            mode='lines',
            line=dict(color=left_density, width=2.0, dash='dot'),
            name="x*p(x)",
            showlegend=False,
            opacity=0.78
        ),
        row=1, col=1
    )
    # Area fill under x*p(x)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_sweep, x_sweep[::-1]]),
            y=np.concatenate([x_sweep*pdf_sweep, np.zeros_like(x_sweep)]),
            fill='toself',
            fillcolor='rgba(20, 214, 139, 0.22)',  # match right_bar_main, translucent
            line=dict(color='rgba(20,214,139,0.42)', width=0.9),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    # Sweep vertical line
    fig.add_trace(
        go.Scatter(
            x=[x[cur_idx], x[cur_idx]],
            y=[min(np.min(pdf), np.min(x*pdf)), max(pdf) * 0.99],
            mode="lines",
            line=dict(color=left_accent, width=4, dash='dash'),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1
    )
    # Marker at (x[cur_idx], pdf[cur_idx])
    fig.add_trace(
        go.Scatter(
            x=[x[cur_idx]], y=[pdf[cur_idx]],
            mode="markers",
            marker=dict(size=15, color=left_accent, line=dict(width=2, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1
    )

    # Axes config to resemble discrete version
    y_min = np.min(x*pdf) * 1.14
    y_max = max(pdf)*1.11
    fig.update_xaxes(
        title_text="x",
        tickvals=[-4, -2, 0, 2, 4],
        range=[-4.2, 4.2],
        row=1, col=1,
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="p(x), x·p(x)",
        range=[y_min, y_max],
        row=1, col=1,
        showgrid=False,
        title_font=dict(color="#23AFFF"), tickfont=dict(color="#23AFFF")
    )

    # --- Right: Accumulating bar for the current integral value ---
    right_bar_marker = dict(color=right_bar_main)
    if step_idx == n_steps-1:
        right_bar_marker["line"] = dict(color=right_bar_highlight, width=4)
    else:
        right_bar_marker["line"] = dict(color="rgba(0,0,0,0)", width=0)

    fig.add_trace(
        go.Bar(
            x=["Sum"],
            y=[partial_sum],
            marker=right_bar_marker,
            width=0.48,
            name="Partial Sum",
            showlegend=False
        ),
        row=1, col=2
    )

    # Faint target dashed line for true expectation
    fig.add_trace(
        go.Scatter(
            x=["Sum","Sum"],
            y=[norm_expectation, norm_expectation],
            mode="lines",
            line=dict(color=faint_target, dash="dash", width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )
    # Numeric annotation on bar
    fig.add_trace(
        go.Scatter(
            x=["Sum"],
            y=[partial_sum],
            mode="text",
            text=[f"{partial_sum:.4f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(
        title_text="", row=1, col=2, tickvals=["Sum"]
    )
    fig.update_yaxes(
        title_text="Accumulated Value",
        row=1, col=2,
        range=[-0.45, 0.45],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.18)'
    )

    # --- Figure Layout (discrete style, transparent, padding) ---
    fig.update_layout(
        height=470,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text=(
                "Normal Expectation Accumulation<br>"
                f"Accumulated sum: {partial_sum:.4f}  (Final: {norm_expectation:.4f})"
            ),
            y=0.93,  # shifted main title down a bit (was .99)
            font=dict(color="#FAFAFA", size=22, family=None),
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        margin=dict(l=42, r=24, b=85, t=160),
        barmode='overlay'
    )

    return fig

# --- Build frames for each sweep position ---
integral_frames = [
    go.Frame(
        data=make_normal_integration_fig(step).data,
        name=str(step),
        layout=make_normal_integration_fig(step).layout
    )
    for step in range(n_steps)
]

fig_integral = make_normal_integration_fig(0)
fig_integral.frames = integral_frames

# --- Play Button ---
fig_integral.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.32,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_integral.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Empirically, We Observe Convergence to the Same Value by the Law of Large Numbers (LLN)

 **Weak Law of Large Numbers:**  
 $$
 \forall \varepsilon > 0,\ \lim_{n\to\infty} \Pr\left(|\overline{X}_n-\mu| > \varepsilon\right) = 0
 $$
 *(The sample mean converges in probability to $\mu$, but not necessarily for every sequence.)*

 **Strong Law of Large Numbers:**  
 $$
 \Pr\left(\lim_{n\to\infty} \overline{X}_n = \mu\right) = 1
 $$
 *(The sample mean almost surely converges to $\mu$: this is a much stronger guarantee.)*

In other words, we can estimate the population mean by randomly drawing samples from the same distribution and calculating their average. 

As we collect more samples, the average of these values will get closer and closer to the true mean of the original population.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- LLN for the Standard Normal! ---

np.random.seed(42)
N = 250  # Number of standard normal samples

# Generate samples from standard normal
samples = np.random.randn(N)
empirical_means = np.array([np.mean(samples[:i+1]) for i in range(N)])
# empirical_vars = np.array([np.var(samples[:i+1], ddof=0) for i in range(N)])

true_mean = 0.0
# true_var = 1.0

def make_lln_normal_fig(step):
    # step is 0-based (number of data points so far is step+1)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=(
            "<br>Histogram of Standard Normal Samples<br> (as Number of Draws Increases)",
            "<br>Empirical Mean Converges to Theoretical Mean"
        )
    )

    data_so_far = samples[:step+1]
    
    # --- Left: Histogram and theoretical density ---
    hist_vals, bins = np.histogram(data_so_far, bins=15, range=[-4, 4], density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_vals,
            width=(bins[1]-bins[0]),
            marker_color="#FFE066",
            name="Empirical Histogram",
            opacity=0.77,
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )
    # Theoretical normal curve
    x_grid = np.linspace(-4, 4, 100)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=norm.pdf(x_grid),
            mode="lines",
            line=dict(width=2, color="#23AFFF"),
            name="Standard Normal PDF",
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )

    fig.update_xaxes(
        title_text="Value",
        range=[-4, 4],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Density",
        range=[0, np.max(hist_vals)*1.18 if len(hist_vals)>0 else 0.42],
        row=1, col=1
    )

    # --- Right: Bar chart for mean only --- 
    metrics_x = ["Theoretical Mean", "Empirical Mean"]
    metrics_y = [true_mean, empirical_means[step]]

    fig.add_trace(
        go.Bar(
            x=metrics_x,
            y=metrics_y,
            width=0.43,
            marker_color=["#23AFFF","#FFE066"],
            text=[f"{true_mean:.2f}", f"{empirical_means[step]:.2f}"],
            textposition="outside",
            opacity=0.92,
            showlegend=False
        ),
        row=1, col=2
    )

    # Draw convergence path for empirical mean only
    if step > 1:
        fig.add_trace(
            go.Scatter(
                x=["Empirical Mean"]*step,
                y=empirical_means[:step],
                mode="lines+markers",
                line=dict(color='#BB9900', width=2, dash='dot'),
                marker=dict(size=3, color='#FFD700'),
                name="Empirical Mean (trace)",
                showlegend=False,
                hoverinfo="y+x"
            ),
            row=1, col=2
        )

    # Highlight target band for the mean (for visibility)
    fig.add_shape(
        type="rect",
        xref="x2", yref="y2",
        x0=-0.4, x1=1.4, y0=true_mean-0.07, y1=true_mean+0.07,
        fillcolor="rgba(30,250,240,0.09)",
        line_width=0, row=1, col=2
    )

    fig.update_xaxes(
        title_text="",
        row=1, col=2,
        tickvals=metrics_x
    )
    fig.update_yaxes(
        title_text="Value",
        range=[-1.25, 3],
        row=1, col=2
    )

    # --- Layout ---
    fig.update_layout(
        height=450, width=900,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text="<br>Law of Large Numbers (LLN) for Standard Normal: Empirical Mean Converges<br>",
            y= .99, font=dict(color="#FAFAFA", size=23),
            x=0.5, xanchor="center", yanchor="top"
        ),
        margin=dict(l=45, r=22, b=80, t=98),
        barmode='group'
    )
    return fig

# --- Create Animation Frames ---
lln_normal_frames = [
    go.Frame(
        data=make_lln_normal_fig(n).data,
        name=str(n),
        layout=make_lln_normal_fig(n).layout
    )
    for n in range(N)
]

fig_lln_normal = make_lln_normal_fig(0)
fig_lln_normal.frames = lln_normal_frames

# --- Animation play button (no slider) ---
fig_lln_normal.update_layout(
    updatemenus=[
        {
            'type': 'buttons',
            'x': 0.48, 'y': -0.22,
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }
    ]
)

fig_lln_normal.show()

```

###### ______________________________________________________________________________________________________________________________________

##### This Same Idea Applies to Stochastic Processes

Each time index is a distribution, and if we continue to simulate (discretize) the process we will observe convergence like in the examples above

Here we observe convergence to the theoretical distribution of Brownian motion at time $T$


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100  # Number of paths to be animated one by one
n_steps = 100
dt = 1
mu = 0
sigma = 1

np.random.seed(42)
# Brownian Motion W_t = sum(increments)
increments = np.random.normal(mu, sigma * np.sqrt(dt), (n_paths_total, n_steps))
W = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

T = n_steps  # final time

# --- Precompute theoretical normal distribution for W_T ~ N(0, T) ---
hist_bins = 30
w_T_all = W[:, T]
# Use a wide range for histogram bins to cover all paths
hist_range = [w_T_all.min() - 1, w_T_all.max() + 1]
bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)
hist_y_bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

theory_pdf = norm.pdf(hist_y_bins, 0, np.sqrt(T))

# Determine maximum density value for fixed x-axis on histogram
max_empirical_density = 0
n_paths_to_check = min(n_paths_show, n_paths_total)
for k in range(1, n_paths_to_check + 1):
    w_T_slice = W[:k, T]
    hist_k = np.histogram(w_T_slice, bins=bin_edges, density=True)
    max_empirical_density = max(max_empirical_density, max(hist_k[0]))

# Hardcode max for right xlim as requested (regardless of pdf/hist maximum)
max_pdf = 0.045  # <-- UPDATED from 0.04 to 0.045 as requested

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Brownian Paths", f"Distribution of $W_T$ (T={T})"],
    horizontal_spacing=0.10
)

# --- Initial Traces (Placeholders for the first frame's data) ---

# Left Chart: All historical paths (muted) and the single new path (highlighted)
# Initial state: no paths shown (empty lists)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines', line=dict(color='#00ffff', width=1),
    showlegend=False, name="Historical Paths (Muted)"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines', line=dict(color='magenta', width=3),
    showlegend=False, name="New Path (Highlighted)"
), row=1, col=1)

# Right Chart: Histogram (empty) and Theoretical PDF (full)
hist_init = go.Bar(
    y=hist_y_bins,
    x=np.zeros_like(hist_y_bins),
    width=(hist_y_bins[1] - hist_y_bins[0]) if len(hist_y_bins) > 1 else 0.5,
    orientation='h',
    marker=dict(color='rgba(100,255,100,0.6)'),
    name='Empirical'
)
fig.add_trace(hist_init, row=1, col=2)

theory_line_init = go.Scatter(
    x=theory_pdf,
    y=hist_y_bins,
    mode='lines',
    line=dict(color='magenta', width=3),
    name='Theoretical'
)
fig.add_trace(theory_line_init, row=1, col=2)

# --- Animation Frames ---
frames = []
path_indices = np.arange(n_paths_show)
for k in range(1, n_paths_show + 1):  # animate adding 1, 2, ..., n_paths_show paths

    # --- Left: Paths (Scatter) ---
    # 1. Historical Paths (1 to k-1) - Muted color

    if k > 1:
        # Get paths 1 to k-1
        paths_muted = W[path_indices[:k-1]]

        # Initialize x_muted and y_muted as float arrays from the beginning
        x_muted = np.tile(time_grid, k - 1).astype(float)
        y_muted = paths_muted.flatten().astype(float)

        # Indices where NaNs should be inserted (after every path)
        insert_indices = np.arange(n_steps + 1, len(x_muted), n_steps + 1)

        # Insert np.nan (float) into the float array
        x_muted = np.insert(x_muted, insert_indices, np.nan)
        y_muted = np.insert(y_muted, insert_indices, np.nan)
    else:
        # For k=1, no historical paths yet. Use sentinel NaNs initialized as floats.
        x_muted, y_muted = np.array([np.nan]), np.array([np.nan])

    scatter_muted = go.Scatter(
        x=x_muted,
        y=y_muted,
        mode='lines',
        line=dict(color='#00ffff', width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='none',
    )

    # 2. New Path (k) - Highlighted color
    scatter_new = go.Scatter(
        x=time_grid.astype(float),
        y=W[k-1].astype(float),  # W[k-1] is the k-th path
        mode='lines',
        line=dict(color='magenta', width=3),
        name="New Path",
        showlegend=False,
        hoverinfo='none',
    )

    # --- Right: Distribution (Bar/Scatter) ---
    # 3. Histogram of the current k path endpoints at T
    w_T_slice = W[path_indices[:k], T]
    hist_k = np.histogram(w_T_slice, bins=bin_edges, density=True)
    hist_k_x = hist_k[0]  # Density values

    hist_trace = go.Bar(
        y=hist_y_bins,
        x=hist_k_x,
        orientation='h',
        marker=dict(color='rgba(100,255,100,0.6)'),
        hoverinfo='skip',
        name='Empirical',
    )

    # 4. Theoretical normal distribution for W_T ~ N(0, T)
    theory_line = go.Scatter(
        x=theory_pdf,
        y=hist_y_bins,
        mode='lines',
        line=dict(color='magenta', width=3),
        hoverinfo='skip',
        name='Theoretical',
    )

    # Combine traces for the frame data. The order must match the initial figure setup order.
    frames.append(go.Frame(
        data=[scatter_muted, scatter_new, hist_trace, theory_line],
        name=f"paths{k}",
        # Keep the title dynamically updated per frame
        layout=go.Layout(title_text=f"Convergence of Sample Paths to Theoretical Distribution at Final Time T ({k} paths shown)")
    ))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=600, width=1000,
    title_text=f"Convergence of Sample Paths to Theoretical Distribution at Final Time T (0 paths shown)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.1)'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.10,
        'xanchor': 'center', 'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},  # Smoother animation
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Axes styling and limits
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

# --- Axes (fixed globally) ---
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='$W_t$', range=[-20, 20], row=1, col=1)
fig.update_xaxes(title_text='Density', range=[0, max_pdf], row=1, col=2)  # <-- FIXED TO 0.045!
fig.update_yaxes(title_text='$W_T$', range=hist_range, row=1, col=2)  # Use calculated range

fig.show()
```

###### ______________________________________________________________________________________________________________________________________

##### We can combine the Feynman-Kac Theorem and Law of Large Numbers to Produce Option Prices without Solving a PDE

 For a general derivative with a possibly path-dependent or exotic payoff function $\Psi$, the risk-neutral pricing PDE may take the form:
 
 $$
 \frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V + \mathcal{G}[V] = 0
 $$
 
 where $\mathcal{G}[V]$ reflects the dependence on additional features such as the running maximum, average, or other path-dependent quantities (for example, barrier or Asian options).
  
  By the Feynman-Kac Theorem, the solution $V(t, S, A)$ to this PDE is also equivalent to the conditional expectation (under the risk-neutral measure $\mathbb{Q}$):
  $$
  V(t, S, A) = \mathbb{E}^{\mathbb{Q}}\left[
      e^{-r(T-t)}\, \Psi(S_T, A_T)\ \bigg|\ S_t = S,\, A_t = A
  \right]
  $$
  That is, given the state $(S, A)$ at time $t$, the value function is the discounted expected payoff over all risk-neutral paths starting from this state.

  Given we can approximate the expectation, we just need to approximate appropriate values for $S_T$ and $A_T$ to observe convergence to the correct model price!

---

#### 3.) 📈 Simulating Option Prices

##### Why can we Simulate Option Prices?

**1.) Black-Scholes Argument Produces the "Correct" Arbitrage Free Price Implied by the Fundamental Theorem of Asset Pricing (Feynman-Kac)**

$$ V_0 = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}\left[\,\Phi(S_T)\,\right] $$

**2.) The Law of Large Numbers (LLN) Ensures Empirical Statistics Converge to Theoretical Ones if the Underlying Distribution is Stable**

 $$
 \frac{1}{N} \sum_{i=1}^N X_i \xrightarrow[]{N \to \infty} \mathbb{E}[X] \quad \text{(Law of Large Numbers)}
 $$


**3.) Option Price, Represented as a Conditional Expectation from (1), Can be Approximated via Euler-Maruyama Discretization Satisfying Draws for the LLN to Apply**

 The Euler–Maruyama discretization for the SDE $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ is:
 
 $$
 S_{t+\Delta t} = S_t + \mu S_t\, \Delta t + \sigma S_t\, \Delta W_t
 $$

 where $\Delta W_t \sim N(0, \Delta t)$.


###### ______________________________________________________________________________________________________________________________________

**Following from the above we can:**

1.) Select a stochastic model for the underlying (ABM/GBM, Heston, ...)

2.) Discretize and simulate the process to terminal time $T$

3.) Compute the payoff of that sample path $\Phi(S_T)$

4.) Discount it back to the present  $\text{Price} = e^{-rT} \cdot \Phi(S_T)$ and add the value to a list

5.) Repeat at (1) for many iterations and observe convergence by the Law of Large Numbers (LLN)

###### ______________________________________________________________________________________________________________________________________

##### Example: 🌈 Rainbow Option

1.) I will use two Heston models to simulate the underlying assets

2.) I will discretize each with a risk-neutral parameter set that I receive from market calibration

3.) I will compute the Best-of Call payoff at simulated time $T$

4.) I will discount this value back to the present and store it in a list

5.) I will repeat at (1) for many iterations and observe convergence by the Law of Large Numbers (LLN)!


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Risk-Neutral Heston Model Parameters (Illustrative Example) ---
r = 0.02    # risk-free rate
kappa = 1.5 # mean reversion speed
theta = 0.04 # long run variance
xi = 0.3    # volatility of variance ("vol of vol")
rho = -0.7  # correlation between Brownian motions
v0 = 0.04   # initial variance
S0_1 = 100  # initial price Asset 1
S0_2 = 100  # initial price Asset 2

# Simulation settings
T = 1.0                  # time horizon
n_steps = 100
dt = T / n_steps
n_paths_total = 3000
n_paths_show = 100        # Animate showing up to n_paths_show paths

np.random.seed(42)

# Simulate two Heston processes (Assets 1 and 2), correlated via rho
def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths):
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # Pre-generate correlated normals
    Z1_raw = np.random.normal(size=(n_paths, n_steps))
    Z2_raw = np.random.normal(size=(n_paths, n_steps))
    Z1 = Z1_raw
    Z2 = rho * Z1_raw + np.sqrt(1 - rho ** 2) * Z2_raw

    for t in range(n_steps):
        dW1 = np.sqrt(dt) * Z1[:, t]
        dW2 = np.sqrt(dt) * Z2[:, t]
        v_prev = np.maximum(v[:, t], 0)  # Full truncation for variance
        v_sqrt = np.sqrt(v_prev)

        # Euler-Maruyama for volatility
        v[:, t + 1] = v_prev + kappa * (theta - v_prev) * dt + xi * v_sqrt * dW2
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)

        # Euler-Maruyama for price dynamics
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * v_prev) * dt + v_sqrt * dW1)
    return S, v

# Precompute Heston price paths (all at once for efficiency)
S1, v1 = simulate_heston_paths(S0_1, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths_total)
S2, v2 = simulate_heston_paths(S0_2, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths_total)
time_grid = np.linspace(0, T, n_steps + 1)

# --- Payoff: Best-of Rainbow Call Option ---
K = 100
payoff = lambda ST1, ST2: np.maximum(np.maximum(ST1, ST2) - K, 0)
discount = np.exp(-r * T)
payoff_all = payoff(S1[:, -1], S2[:, -1])
price_all = discount * payoff_all

# --- Compute Cumulative Monte Carlo Means ---
cum_prices = np.cumsum(price_all[:n_paths_show]) / np.arange(1, n_paths_show + 1)
true_mean_price = np.mean(price_all)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=[
        "Simulated Heston Asset Paths (Rainbow Option)",
        "Convergence of MC Price Estimate"
    ],
    horizontal_spacing=0.10
)

# --- Initial Traces (Placeholders for frame data) ---

# 1. Historical Asset 1 Paths (Muted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='#00ffff', width=1), name='Asset 1 Paths (Historical)', showlegend=True, opacity=0.12
), row=1, col=1)

# 2. Historical Asset 2 Paths (Muted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='magenta', width=1), name='Asset 2 Paths (Historical)', showlegend=True, opacity=0.12
), row=1, col=1)

# 3. Most Recent Asset 1 Path (Highlighted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='#00ffff', width=3), name='Asset 1 Path (New)', showlegend=False
), row=1, col=1)

# 4. Most Recent Asset 2 Path (Highlighted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='magenta', width=3), name='Asset 2 Path (New)', showlegend=False
), row=1, col=1)

# 5. Running MC Price Estimate (Right Chart)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan],
    mode='lines+markers',
    marker=dict(size=7, color='white'),
    line=dict(color='white', width=3),
    name='MC Mean Price', showlegend=False
), row=1, col=2)

# 6. Theoretical Mean (Reference Line on Right Chart)
fig.add_trace(go.Scatter(
    x=[0, n_paths_show], y=[true_mean_price, true_mean_price],
    mode='lines',
    line=dict(color='yellow', width=2, dash='dash'),
    name='MC True Mean', showlegend=True
), row=1, col=2)

# --- Animation Frames ---
frames = []

for k in range(1, n_paths_show + 1):

    # ------- LEFT CHART: Plot Heston Asset Price Paths --------

    # 1. All previous (historical) Asset 1 paths up to k-1, faded
    if k > 1:
        S1_hist = S1[:k-1]
        time_hist = np.tile(time_grid, k - 1)
        S1h_flat = S1_hist.T.ravel()
        x1_hist = np.insert(time_hist, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
        y1_hist = np.insert(S1h_flat, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
    else:
        x1_hist, y1_hist = np.array([np.nan]), np.array([np.nan])

    trace1_hist = go.Scatter(
        x=x1_hist,
        y=y1_hist,
        mode='lines',
        line=dict(color='#00ffff', width=1),
        name='Asset 1 Paths (Historical)',
        opacity=0.12,
        showlegend=False,
        hoverinfo='none'
    )

    # 2. All previous (historical) Asset 2 paths up to k-1, faded
    if k > 1:
        S2_hist = S2[:k-1]
        time_hist = np.tile(time_grid, k - 1)
        S2h_flat = S2_hist.T.ravel()
        x2_hist = np.insert(time_hist, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
        y2_hist = np.insert(S2h_flat, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
    else:
        x2_hist, y2_hist = np.array([np.nan]), np.array([np.nan])

    trace2_hist = go.Scatter(
        x=x2_hist,
        y=y2_hist,
        mode='lines',
        line=dict(color='magenta', width=1),
        name='Asset 2 Paths (Historical)',
        opacity=0.12,
        showlegend=False,
        hoverinfo='none'
    )

    # 3. The latest Asset 1 Heston price path, highlighted
    trace3_new_1 = go.Scatter(
        x=time_grid,
        y=S1[k-1],
        mode='lines',
        line=dict(color='#00ffff', width=3),
        name='Asset 1 Path (New)',
        opacity=1.0,
        showlegend=False,
        hoverinfo='none'
    )

    # 4. The latest Asset 2 Heston price path, highlighted
    trace4_new_2 = go.Scatter(
        x=time_grid,
        y=S2[k-1],
        mode='lines',
        line=dict(color='magenta', width=3),
        name='Asset 2 Path (New)',
        opacity=1.0,
        showlegend=False,
        hoverinfo='none'
    )

    # ------- RIGHT CHART: MC Estimate --------

    # 5. Running MC price estimate (discounted payoff)
    trace5_mc_mean = go.Scatter(
        x=np.arange(1, k + 1),
        y=cum_prices[:k],
        mode='lines+markers',
        marker=dict(size=7, color='white'),
        line=dict(color='white', width=3),
        name='MC Mean',
        hoverinfo='x+y',
        showlegend=False
    )

    frames.append(go.Frame(
        data=[
            trace1_hist,
            trace2_hist,
            trace3_new_1,
            trace4_new_2,
            trace5_mc_mean
        ],
        name=f"step{k}",
        layout=go.Layout(
            title_text=f"Rainbow MC: {k} Sample Path{'s' if k > 1 else ''} (Right: Price Estimate Convergence)"
        )
    ))

fig.frames = frames

# Set axes labels and ranges
fig.update_layout(
    height=650,
    width=1000,
    title_text=f"Rainbow MC: 0 Sample Paths",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.1)', itemsizing='constant'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.10,
        'xanchor': 'center', 'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 250, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Grid and axes for both subplots
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

# Left chart (Heston price paths) axis scaling
all_prices = np.concatenate([S1[:, :], S2[:, :]])
fig.update_xaxes(title_text='Time $t$', range=[0, T], row=1, col=1)
fig.update_yaxes(
    title_text='Asset Price',
    range=[np.nanmin(all_prices) * 0.8, np.nanmax(all_prices) * 1.2],
    row=1, col=1
)

# Right chart (MC mean estimate) axis scaling
fig.update_xaxes(title_text='MC Sample Size', range=[0, n_paths_show + 1], row=1, col=2)
fig.update_yaxes(
    title_text='Discounted Payoff Estimate',
    range=[
        np.min(cum_prices) * 0.95,
        np.max(cum_prices) * 1.05
    ],
    row=1, col=2
)

fig.show()
```

$$\text{Rainbow option price:} \quad V = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]$$



---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- The Black-Scholes portfolio replication argument is precisely the arbitrage free price implied by the Fundamental Theorem of Asset Pricing proven by the Feynman-Kac Theorem
- We thus have two ways to solve for an option price: solving (analytically or numerically) a pricing PDE or by solve for (analytically or numerically) an expectation
- Since we are dealing with stable underlying distributions we can approxoimate the expectation via Monte Carlo simulation and the Law of Large Numbers (LLN) to produce the necessary expectation corresponding to the option price
- After discounting the simulated expectation back to the present we will find the price of the option consistent with the replication argument
- This recipe is how the fair (mid) price is approximated by large institutions every day before adjustments, quoting a two-way, and hedging as I discuss in my video on Algorithmic Market-Making


**Future Topics**

Technical Videos and Other Discussions

- Projects that Made me a Quant
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Quant Roadmap (I am currently making the roadmap!)
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Reinforcement Learning for Delta Hedging
- Approximating Pricing Functionals using Neural Networks
- Rough Path Theory, Applications of Path Signatures
- Sig-Vol Model, Calibration, and Pricing

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Market Sentiment Filter
- Delta-Neutral Algorithmic Trading System (Automatically Trading Gamma/Theta Positions)

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
