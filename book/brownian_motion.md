### 📈 Brownian Motion for Quant Finance

##### ▶️ Related Quant Guild Videos:

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [Stochastic Differential Equations for Quant Finance](https://youtu.be/qDAeSC40ZJE)

- [Finite Differences for Quant Finance](https://youtu.be/uzbveN8n34U)

- [Master Volatility with ARCH & GARCH Models](https://youtu.be/iImtlBRcczA)

- [Markov Chains for Quant Finance](https://youtu.be/k8oQfd6M5sA)

- [Hidden Markov Models for Quant Finance](https://youtu.be/Bru4Mkr601Q)

###### ______________________________________________________________________________________________________________________________________

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)



##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)

---

### 📖 Sections


#### 1.) 🎲 Random Variables and Stochastic Processes

- Random Variables

- Statistics and Simulating Random Variables

- Stochastic Processes

- Statistics and Simulating Stochastic Processes

#### 2.) 🔨 Brownian Motion (Theory)

- Definition, Increments, Distributions

- Monte Carlo Simulation

#### 3.) 📈 Pricing Options (Practice)

- Arithmetic Brownian Motion

- Pricing Options with Monte Carlo Simulation

#### 4.) 💭 Closing Thoughts and Future Topics

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

#### 1.) 🎲 Random Variables and Stochastic Processes

##### Random Variables

Random variables are fully specified by their. . .

- Cumulative Distribution Function

$$F_X(a) = p(X \leq a) = \int_{-\infty}^a f(x) dx$$

- Probability Mass (Discrete) or Density (Continuous) Function

$$f_X(a) = p(X = a) \text{ (discrete) or } f_X(a) = \frac{d}{da}P(X \leq a) \text{ (continuous)}$$

- Characteristic Function (Fourier Transform)
$$\varphi_X(t) = \mathbb{E}\left[e^{itX}\right] = \int_{-\infty}^{\infty} e^{itx} f_X(x) dx$$

###### ______________________________________________________________________________________________________________________________________

##### Example: Outcomes of Random Variables

Random variables represent possible outcomes with associated probabilities

Importantly, **we can never predict the outcome of a random variable**

Instead, we use them as models to analyze the likelihoods of different states of the world

See below a data generating normal (Gaussian) distribution and the draws from it, the empirical distribution


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# Parameters for the normal distribution
mu, sigma = 0, 1
n_samples = 500
np.random.seed(42)
sample = np.random.normal(mu, sigma, n_samples)

# Data for the theoretical population distribution
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf = norm.pdf(x, mu, sigma)

# Data for the empirical (sample) distribution: histogram (density=True normalizes)
hist_y, hist_x = np.histogram(sample, bins=50, range=(x.min(), x.max()), density=True)
hist_x_mid = (hist_x[:-1] + hist_x[1:]) / 2

# Setup subplots
fig_sample = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Theoretical Normal Distribution<br>(Population: μ=0, σ=1)",
        f"Empirical Histogram<br>(n={n_samples} samples)"
    ]
)

# --- Theoretical (Population) PDF ---
fig_sample.add_trace(go.Scatter(
    x=x, y=pdf,
    mode='lines',
    line=dict(color='#39ff14', width=3),
    name='Theoretical Normal PDF',
    showlegend=False,
    hovertemplate='x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>'
), row=1, col=1)

# --- Empirical (Sample) Histogram ---
fig_sample.add_trace(go.Bar(
    x=hist_x_mid, y=hist_y,
    width=hist_x[1] - hist_x[0],
    marker_color='#00ffff',
    name='Sample Histogram',
    opacity=0.65,
    showlegend=False,
    hovertemplate='x=%{x:.2f}<br>density=%{y:.4f}<extra></extra>'
), row=1, col=2)

# Also overlay the true pdf on the sample histogram for comparison
fig_sample.add_trace(go.Scatter(
    x=x, y=pdf,
    mode='lines',
    line=dict(color='#ff00ff', width=2, dash='dash'),
    name='True PDF',
    showlegend=False,
    hovertemplate='x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>'
), row=1, col=2)

# --- Axes & Layout Styling ---
for c in [1, 2]:
    fig_sample.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=False,
        row=1, col=c
    )
    fig_sample.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=False,
        row=1, col=c
    )

fig_sample.update_xaxes(title_text='x', row=1, col=1)
fig_sample.update_yaxes(title_text='PDF f(x)', row=1, col=1)
fig_sample.update_xaxes(title_text='x', row=1, col=2)
fig_sample.update_yaxes(title_text='Density (normalized)', row=1, col=2)

fig_sample.update_layout(
    height=410,
    title_text='Theoretical Population vs Empirical Sample: Normal RV',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig_sample.show()
```

###### ______________________________________________________________________________________________________________________________________

##### Example: Normal Random Variable

When we say a normal (Gaussian) random variable $X \sim N(\mu, \sigma)$ we are suggesting $X$ has the following characteristics. . .

 $$\text{PDF: }f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \quad\quad 
 \text{CDF:} \quad F_X(x) = \int_{-\infty}^x f_X(t) dt$$

In terms of likelihoods and probabilities we can visualize the *probability density function* and the *cumulative distribution function*


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Normal Distribution Parameters ---
mu, sigma = 0, 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf, cdf = norm.pdf(x, mu, sigma), norm.cdf(x, mu, sigma)

# Initial slider points
k1_initial, k2_initial = 0.5, 1.0
idx_k1, idx_k2 = np.argmin(np.abs(x - k1_initial)), np.argmin(np.abs(x - k2_initial))
k1_val, k2_val = x[idx_k1], x[idx_k2]

y_max_pdf, y_max_cdf = np.max(pdf)*1.05, 1.05

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f'Normal PDF (μ={mu}, σ={sigma})',
                    f'Normal CDF (μ={mu}, σ={sigma})']
)

# PDF and CDF lines
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines',
                         line=dict(color='#39ff14', width=3),
                         name='PDF', hovertemplate='x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines',
                         line=dict(color='#00ffff', width=3),
                         name='CDF', hovertemplate='x=%{x:.2f}<br>P(X≤x)=%{y:.4f}<extra></extra>'),
              row=1, col=2)

# Shaded fill area for CDF
x_fill_initial, cdf_fill_initial = x[:idx_k2 + 1], cdf[:idx_k2 + 1]
fig.add_trace(go.Scatter(
    x=x_fill_initial, y=cdf_fill_initial, fill='tozeroy', mode='none',
    fillcolor='rgba(0,255,255,0.3)', name='P(X≤k2)', showlegend=False
), row=1, col=2)

# --- Initial Title & Lines ---
initial_title = (
    f'Normal Distribution (μ={mu}, σ={sigma})'
    f'<br>PDF Height at x={k1_val:.2f} is {pdf[idx_k1]:.4f}'
    f' | P(X≤{k2_val:.2f}) = {cdf[idx_k2]:.4f}'
)

INITIAL_SHAPES = [
    dict(type='line', xref='x1', yref='y1', x0=k1_val, x1=k1_val, y0=0, y1=y_max_pdf,
         line=dict(color='red', width=3, dash='dot')),
    dict(type='line', xref='x2', yref='y2', x0=k2_val, x1=k2_val, y0=0, y1=y_max_cdf,
         line=dict(color='yellow', width=3, dash='dot'))
]

# --- Sliders ---
slider_k1_steps = []
for i, k in enumerate(x):
    pdf_k = pdf[i]
    step = dict(
        method="relayout",
        label="",
        args=[{
            "title.text": (
                f'Normal Distribution (μ={mu}, σ={sigma})'
                f'<br>PDF Height at x={k:.2f} is {pdf_k:.4f}'
                f' | P(X≤{k2_val:.2f}) = {cdf[idx_k2]:.4f}'
            ),
            "shapes[0].x0": k,
            "shapes[0].x1": k
        }],
        execute=True
    )
    slider_k1_steps.append(step)

slider_k2_steps = []
for i, k in enumerate(x):
    cdf_k = cdf[i]
    x_fill, cdf_fill = x[:i + 1], cdf[:i + 1]
    step = dict(
        method="update",
        label="",
        args=[
            {"x": [x, x, x_fill],
             "y": [pdf, cdf, cdf_fill]},  # preserve all traces
            {"title.text": (
                f'Normal Distribution (μ={mu}, σ={sigma})'
                f'<br>PDF Height at x={k1_val:.2f} is {pdf[idx_k1]:.4f}'
                f' | P(X≤{k:.2f}) = {cdf_k:.4f}'
            ),
                "shapes[1].x0": k,
                "shapes[1].x1": k}
        ],
        execute=True
    )
    slider_k2_steps.append(step)

# --- Layout ---
fig.update_layout(
    height=550,
    title={'text': initial_title, 'y': 0.97, 'x': 0.5, 'xanchor': 'center'},
    font=dict(color='white'),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=INITIAL_SHAPES,
    sliders=[
        dict(
            steps=slider_k1_steps,
            active=idx_k1,
            pad={"t": 50},  # more top padding
            yanchor='top', y=-0.05,
            x=0.08, len=0.35
        ),
        dict(
            steps=slider_k2_steps,
            active=idx_k2,
            pad={"t": 50},
            yanchor='top', y=-0.05,
            x=0.55, len=0.35
        )
    ]
)

# Axes
fig.update_xaxes(title_text='x', range=[mu - 4*sigma, mu + 4*sigma],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='f(x) (PDF)', range=[0, y_max_pdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_xaxes(range=[mu - 4*sigma, mu + 4*sigma],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(range=[0, y_max_cdf],
                 showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=2)

fig.show()

```

By the Central Limit Theorem (CLT) the distribution of all sample averages (regardless of the initial distribution) is normal (Gaussian)

We can then bridge the gap between statistics and probability and attempt to assess the probability of different states of the world

###### ______________________________________________________________________________________________________________________________________

$X$ is also uniquely defined by its characteristic function (Fourier transform)

$$\text{CF: }\varphi_X(t) = \exp\left(i\mu t - \frac{1}{2}\sigma^2 t^2\right)$$

Effectively, the Fourier transform doesn't *change the function* it just brings it to a different space

 
 $$f(x)\ \xrightarrow{\quad\mathcal{F}\quad}\ \varphi(t)\ \xrightarrow{\quad\mathcal{F}^{-1}\quad}\ f(x)$$
 
where $\mathcal{F}$ denotes the Fourier transform, taking the PDF $f(x)$ to its characteristic function $\varphi(t)$ (frequency domain), and $\mathcal{F}^{-1}$ is the inverse, bringing us back to $f(x)$.



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ----------------------------------------------------------------------
# ⚙️ Define Normal Distribution Parameters
# ----------------------------------------------------------------------
mu, sigma = 0, 1
x_min, x_max = mu - 4*sigma, mu + 4*sigma
N = 512  # number of sample points (power of 2 for efficient FFT)
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# PDF as our "signal"
pdf = norm.pdf(x, mu, sigma)

# ----------------------------------------------------------------------
# ⚡ Forward FFT → Characteristic Function (frequency domain)
# ----------------------------------------------------------------------
# Characteristic Function φ(t) = E[e^{i t X}] ≈ FFT(pdf(x)) with shift and scaling
fft_result = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pdf))) * dx
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi  # frequency → t (radians)

char_func_real = np.real(fft_result)
char_func_imag = np.imag(fft_result)

# ----------------------------------------------------------------------
# 🔁 Inverse FFT → Reconstruct PDF from Characteristic Function
# ----------------------------------------------------------------------
reconstructed_pdf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(fft_result))) / dx
reconstructed_pdf = np.real(reconstructed_pdf)  # small imaginary residuals

# ----------------------------------------------------------------------
# 📊 Visualization
# ----------------------------------------------------------------------
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Original PDF (Normal RV)',
        'Characteristic Function: Re[φ(t)]',
        'Characteristic Function: Im[φ(t)]',
        'Reconstructed PDF (Inverse FFT)'
    ]
)

# Original PDF
fig2.add_trace(go.Scatter(
    x=x, y=pdf,
    name='PDF',
    mode='lines',
    line=dict(color='#39ff14', width=3)
), row=1, col=1)

# Real part of characteristic function
fig2.add_trace(go.Scatter(
    x=freqs, y=char_func_real,
    name='Re[φ(t)]',
    mode='lines',
    line=dict(color='#00ffff', width=3)
), row=1, col=2)

# Imag part of characteristic function
fig2.add_trace(go.Scatter(
    x=freqs, y=char_func_imag,
    name='Im[φ(t)]',
    mode='lines',
    line=dict(color='#ff00ff', width=3)
), row=2, col=1)

# Reconstructed PDF
fig2.add_trace(go.Scatter(
    x=x, y=reconstructed_pdf,
    name='Reconstructed PDF',
    mode='lines',
    line=dict(color='#ffff00', width=3)
), row=2, col=2)

# ----------------------------------------------------------------------
# 🎨 Layout
# ----------------------------------------------------------------------
fig2.update_layout(
    height=700,
    title='FFT Round-Trip for Normal RV: PDF ↔ Characteristic Function ↔ Reconstructed PDF',
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# --- Axes Styling (light grid like previous plot) ---
for r in [1, 2]:
    for c in [1, 2]:
        fig2.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            row=r, col=c
        )
        fig2.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            row=r, col=c
        )

# Axis titles
fig2.update_xaxes(title_text='x (domain)', row=1, col=1)
fig2.update_yaxes(title_text='f(x)', row=1, col=1)
fig2.update_xaxes(title_text='t (frequency domain)', row=1, col=2)
fig2.update_yaxes(title_text='Re[φ(t)]', row=1, col=2)
fig2.update_xaxes(title_text='t (frequency domain)', row=2, col=1)
fig2.update_yaxes(title_text='Im[φ(t)]', row=2, col=1)
fig2.update_xaxes(title_text='x (domain)', row=2, col=2)
fig2.update_yaxes(title_text='f̂(x)', row=2, col=2)

fig2.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Statistics and Simulating Random Variables

The mean (expected value) of a random variable $X$ is defined as
$$\mathbb{E}[X] = \sum_x x P(X=x) \quad \text{(discrete)} \qquad \text{or} \qquad \mathbb{E}[X] = \int_{-\infty}^\infty x f(x)\, dx \quad \text{(continuous)}.$$

The variance of a random variable $X$ is 
$$\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.$$

For a normal random variable $X \sim N(\mu, \sigma^2)$:
- The mean is $\mathbb{E}[X] = \mu$
- The variance is $\mathrm{Var}(X) = \sigma^2$






```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ----------------------------------------------------------------------
# ⚙️ Setup Parameters
# ----------------------------------------------------------------------
x = np.linspace(-6, 10, 500)

# Different normal distributions (mean, std, color, label)
distributions = [
    (0, 1, '#00ffff', 'μ=0, σ=1'),
    (2, 0.5, '#39ff14', 'μ=2, σ=0.5'),
    (4, 1.5, '#ff00ff', 'μ=4, σ=1.5'),
    (6, 1, '#ffaa00', 'μ=6, σ=1')
]

# ----------------------------------------------------------------------
# 🎨 Figure Setup
# ----------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=['Different Normal Distributions']
)

# ----------------------------------------------------------------------
# 📈 Add Curves
# ----------------------------------------------------------------------
for mu, sigma, color, label in distributions:
    y = norm.pdf(x, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color=color, width=3),
        name=label
    ))

# ----------------------------------------------------------------------
# 🧩 Layout & Styling
# ----------------------------------------------------------------------
fig.update_layout(
    height=500,
    width=900,
    title_text='Comparison of Normal Distributions with Different Means and Variances',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        x=0.02, y=0.98,
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(255,255,255,0.1)'
    )
)

# Axes styling
fig.update_xaxes(
    title_text='Value',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.3)',
    zeroline=False
)
fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.3)',
    zeroline=False
)

# ----------------------------------------------------------------------
# 🖥️ Show Figure
# ----------------------------------------------------------------------
fig.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Law of Large Numbers (LLN) and Statistical Convergence

$$\displaystyle \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{\text{a.s.}} \mathbb{E}[X] \qquad \text{as}\ n \to \infty$$

The Law of Large Numbers (LLN) states that as the number of independent samples increases, the sample mean and other moments (e.g., variance) converge to their true values, as long as those moments exist.



```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# ⚙️ Simulation Setup
# ----------------------------------------------------------------------
mu, sigma = 0, 1
n_samples = 1000
np.random.seed(42)
samples = np.random.normal(mu, sigma, n_samples)

# Compute running statistics
running_means = np.cumsum(samples) / np.arange(1, n_samples + 1)
running_vars = [np.var(samples[:i], ddof=1) for i in range(1, n_samples + 1)]

# ----------------------------------------------------------------------
# 🎞️ Create Frames
# ----------------------------------------------------------------------
frames = []
for i in range(10, n_samples):
    frames.append(go.Frame(
        data=[
            # Left: Mean convergence
            go.Scatter(
                x=np.arange(1, i + 1),
                y=running_means[:i],
                mode='lines',
                line=dict(color='#39ff14', width=3),
                name='Sample Mean'
            ),
            # Theoretical Mean Line
            go.Scatter(
                x=np.arange(1, i + 1),
                y=[mu] * i,
                mode='lines',
                line=dict(color='yellow', width=2, dash='dot'),
                name='Theoretical μ'
            ),
            # Right: Variance convergence
            go.Scatter(
                x=np.arange(1, i + 1),
                y=running_vars[:i],
                mode='lines',
                line=dict(color='#00ffff', width=3),
                name='Sample Var'
            ),
            # Theoretical Variance Line
            go.Scatter(
                x=np.arange(1, i + 1),
                y=[sigma**2] * i,
                mode='lines',
                line=dict(color='magenta', width=2, dash='dot'),
                name='Theoretical σ²'
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, i + 20]),
            xaxis2=dict(range=[0, i + 20])
        ),
        name=f'frame{i}'
    ))

# ----------------------------------------------------------------------
# 📊 Initial Figure Setup
# ----------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Convergence of Sample Mean', 'Convergence of Sample Variance'),
    column_widths=[0.5, 0.5]
)

# Left plot (mean)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[running_means[0]],
        mode='lines',
        line=dict(color='#39ff14', width=3),
        name='Sample Mean'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[mu],
        mode='lines',
        line=dict(color='yellow', width=2, dash='dot'),
        name='Theoretical μ'
    ),
    row=1, col=1
)

# Right plot (variance)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[running_vars[0]],
        mode='lines',
        line=dict(color='#00ffff', width=3),
        name='Sample Var'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[sigma**2],
        mode='lines',
        line=dict(color='magenta', width=2, dash='dot'),
        name='Theoretical σ²'
    ),
    row=1, col=2
)

# ----------------------------------------------------------------------
# 🧩 Animation & Layout
# ----------------------------------------------------------------------
fig.frames = frames

fig.update_layout(
    height=500,
    width=1000,
    title_text="Law of Large Numbers: Convergence of Sample Mean and Variance",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.1,
        'showactive': False,
        'xanchor': 'center',
        'yanchor': 'top',
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 20, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Axes styling
for c in [1, 2]:
    fig.update_xaxes(
        title_text='Sample Size (n)',
        showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c
    )

fig.update_yaxes(title_text='Sample Mean', row=1, col=1)
fig.update_yaxes(title_text='Sample Variance', row=1, col=2)

fig.show()
```

Effectively, since every statistic is really just a function of the mean, so long as it converges the others will as well. . .

Assuming finite or whatever. . .

The indicator $\mathbf{1}\{X_i \leq x\}$ is a Bernoulli random variable whose average gives the empirical probability, converging to the true probability $\Phi(x)$ by the Law of Large Numbers.




The empirical distribution function $F_n(x)$, constructed from i.i.d. samples $X_1, \dots, X_n$ from a normal distribution, converges to the true CDF $\Phi(x)$ as $n$ increases. For each fixed $x$, the indicator $\mathbf{1}\{X_i \leq x\}$ is a Bernoulli random variable with mean $\Phi(x)$, so by the Law of Large Numbers (LLN):

$$
F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{X_i \leq x\} \xrightarrow{a.s.} \Phi(x) \quad \text{as}~n \to \infty
$$


```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ----------------------------------------------------------------------
# ⚙️ Setup Parameters
# ----------------------------------------------------------------------
mu, sigma = 0, 1
N_total = 2000        # total samples
frames_count = 100    # smoother animation
bins = 30

np.random.seed(42)
samples = np.random.normal(mu, sigma, N_total)

# Theoretical normal PDF
x = np.linspace(-4, 4, 200)
theoretical_pdf = norm.pdf(x, mu, sigma)

# ----------------------------------------------------------------------
# 🧮 Compute histogram snapshots
# ----------------------------------------------------------------------
sample_counts = np.linspace(1, N_total, frames_count, dtype=int)
histograms = []
for n in sample_counts:
    hist_y, bin_edges = np.histogram(samples[:n], bins=bins, range=(-4, 4), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    histograms.append((bin_centers, hist_y))

# ----------------------------------------------------------------------
# 🎞️ Build Frames
# ----------------------------------------------------------------------
frames = []
for i, n in enumerate(sample_counts):
    # smooth “falling in” animation: interpolate bars from previous frame
    if i == 0:
        prev_y = np.zeros_like(histograms[i][1])
    else:
        prev_y = histograms[i-1][1]
    curr_y = histograms[i][1]
    intermediate = [
        prev_y + (curr_y - prev_y) * t for t in np.linspace(0, 1, 4)
    ]

    # make micro-frames for bar rise
    for step, yvals in enumerate(intermediate):
        frames.append(go.Frame(
            data=[
                go.Bar(
                    x=histograms[i][0],
                    y=yvals,
                    width=0.25,
                    marker_color='cyan',
                    opacity=0.7,
                    name='Empirical Histogram'
                ),
                go.Scatter(
                    x=x,
                    y=theoretical_pdf,
                    mode='lines',
                    line=dict(color='magenta', width=3),
                    name='Normal PDF'
                )
            ],
            layout=go.Layout(
                xaxis=dict(range=[-4.2, 4.2]),
                yaxis=dict(range=[0, 0.5])
            ),
            name=f"n={n}_step{step}"
        ))

# ----------------------------------------------------------------------
# 🎨 Initial Figure (empty start)
# ----------------------------------------------------------------------
fig = go.Figure(
    data=[
        go.Bar(
            x=np.linspace(-4, 4, bins),
            y=[0]*bins,
            width=0.25,
            marker_color='cyan',
            opacity=0.7,
            name='Empirical Histogram'
        ),
        go.Scatter(
            x=x,
            y=theoretical_pdf,
            mode='lines',
            line=dict(color='magenta', width=3),
            name='Normal PDF'
        )
    ],
    layout=go.Layout(
        title_text='Sampling from a Normal Distribution — Histogram Converging to True PDF',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title='Value',
            range=[-4.2, 4.2],
            showgrid=True, gridcolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title='Density',
            range=[0, 0.5],
            showgrid=True, gridcolor='rgba(128,128,128,0.3)'
        ),
        updatemenus=[{
            'type': 'buttons',
            'x': 0.5, 'y': -0.25,
            'xanchor': 'center',
            'yanchor': 'top',
            'showactive': False,
            'buttons': [{
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0, 'easing': 'linear'}
                }]
            }]
        }]
    ),
    frames=frames
)

fig.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Random vs Uncertain

In reality, these distributions aren't fixed, and they are unobservable, all we can observe are the samples from the distribution

This is a big problem with quant models - we face this non-stationarity in practice

In other words, we don't get statistical convergence or convergence in distribution!

**The likelihood of different events (the center and spread of the distribution) is certainly going to change over time**


```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 150_000
n_frames = 60
x_range = np.linspace(-6, 6, 300)

# Variance and mean paths (as before)
def ou_process(theta=1.0, mu=3.0, sigma=1.0, dt=0.1, n_steps=n_frames):
    x = np.zeros(n_steps)
    x[0] = 0.5
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx
    return np.maximum(x, 0.1)

time_points = np.linspace(0, n_frames/10, n_frames)
variance_path = ou_process()
means = np.concatenate([
    np.linspace(3, 0, n_frames//5),
    np.linspace(-2, 0, 3*n_frames//5),
    np.linspace(-1, 0, n_frames//5)
])

# Create Figure: just two rows (top: theoretical dist, bottom: sample histogram)
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.5, 0.5],
    specs=[[{"type": "xy"}],
           [{"type": "xy"}]],
    subplot_titles=(
        'Unobservable Data Generating Distribution',
        'Observed (Empirical) Samples'
    )
)

frames = []
all_samples = []

for i in range(n_frames):
    n_samples_current = int(n_samples * (1 / (1 + np.abs(means[i])**2)))
    X = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current)
    kde_X = stats.gaussian_kde(X)
    samples = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current // 500)
    all_samples.extend(samples)

    frames.append(
        go.Frame(
            data=[
                # (1) Theoretical distribution (density)
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Varying Normal'
                ),
                # (2) Histogram of observed samples
                go.Histogram(
                    x=all_samples,
                    nbinsx=50,
                    name='Samples',
                    marker_color='rgba(0, 255, 255, 0.6)'
                )
            ]
        )
    )

# Initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)  # Distribution (density)
fig.add_trace(frames[0].data[1], row=2, col=1)  # Histogram

# Add frames to figure
fig.frames = frames

# Animation Controls
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0},
                'mode': 'immediate',
                'loop': True
            }]
        }]
    }]
)

# Layout & Styling
fig.update_layout(
    height=600,
    width=900,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Evolving Data Generating Distribution and Observed Samples'
)

# Axes Formatting
for r in [1, 2]:
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=1
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=1
    )

fig.update_xaxes(range=[-6, 6], row=1, col=1)
fig.update_yaxes(range=[0, 1], row=1, col=1)
fig.update_xaxes(range=[-6, 6], row=2, col=1)
fig.update_yaxes(range=[0, 1750], row=2, col=1)

fig.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Stochastic Processes

Stochastic processes are simply random variables over an index (draws, time)

We can have both discrete time stochastic processes (like Markov Chains) and continuous time ones (Brownian motion)

These processes can be defined in a variety of ways, sometimes there is memory where path history matters, sometimes they are memoryless

##### Example: White Noise Process

 
 $$
 X_t = \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,\,\sigma^2)
 $$
 
 where $\epsilon_t$ are i.i.d. standard normal random variables, i.e., "white noise".



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ----------------------------------------------------------------------
# ⚙️ Simulation Setup
# ----------------------------------------------------------------------
mu, sigma = 0, 1
n_samples = 300
samples = np.random.normal(mu, sigma, n_samples)
x_vals = np.arange(1, n_samples + 1)

# Theoretical normal PDF (for right subplot)
y_pdf = np.linspace(-4, 4, 300)
pdf = norm.pdf(y_pdf, mu, sigma)

# ----------------------------------------------------------------------
# 🎞️ Create Frames
# ----------------------------------------------------------------------
frames = []
for i in range(1, n_samples):
    y_current = samples[i]
    pdf_x_current = norm.pdf(y_current, mu, sigma)
    
    frames.append(go.Frame(
        data=[
            # --- Left: white noise process path ---
            go.Scatter(
                x=x_vals[:i],
                y=samples[:i],
                mode='lines+markers',
                line=dict(color='#00ffff', width=2),
                marker=dict(size=6, color='#39ff14'),
                name='White Noise Path'
            ),
            
            # --- Right: rotated Normal PDF ---
            go.Scatter(
                x=pdf,
                y=y_pdf,
                mode='lines',
                line=dict(color='magenta', width=3),
                name='Normal PDF (rotated)'
            ),
            
            # Horizontal connector line
            go.Scatter(
                x=[0, pdf_x_current],
                y=[y_current, y_current],
                mode='lines',
                line=dict(color='#39ff14', width=2),
                name='Connection'
            )
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, n_samples]),
            yaxis=dict(range=[-4, 4]),
            xaxis2=dict(range=[0, 0.5]),
            yaxis2=dict(range=[-4, 4])
        ),
        name=f'frame{i}'
    ))

# ----------------------------------------------------------------------
# 📊 Initial Figure Setup
# ----------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('White Noise Process', 'Normal Distribution'),
    column_widths=[0.6, 0.4]
)

# Initial path
fig.add_trace(
    go.Scatter(
        x=[1],
        y=[samples[0]],
        mode='lines+markers',
        line=dict(color='#00ffff', width=2),
        marker=dict(size=6, color='#39ff14'),
        name='White Noise Path'
    ),
    row=1, col=1
)

# Rotated normal PDF
fig.add_trace(
    go.Scatter(
        x=pdf,
        y=y_pdf,
        mode='lines',
        line=dict(color='magenta', width=3),
        name='Normal PDF (rotated)'
    ),
    row=1, col=2
)

# Connection line
fig.add_trace(
    go.Scatter(
        x=[0, norm.pdf(samples[0], mu, sigma)],
        y=[samples[0], samples[0]],
        mode='lines',
        line=dict(color='#39ff14', width=2),
        name='Connection'
    ),
    row=1, col=2
)

# ----------------------------------------------------------------------
# 🎬 Layout & Animation
# ----------------------------------------------------------------------
fig.frames = frames

fig.update_layout(
    height=500,
    width=1000,
    title_text="White Noise Process — Live Sampling from a Normal Distribution",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25,
        'xanchor': 'center',
        'yanchor': 'top',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0, 'easing': 'linear'}
            }]
        }]
    }]
)

# Axes styling
fig.update_xaxes(
    title_text='Time (t)',
    range=[0, n_samples],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=1
)
fig.update_yaxes(
    title_text='Noise Value',
    range=[-4, 4],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=1
)

# Rotated distribution
fig.update_xaxes(
    title_text='Density',
    range=[0, 0.5],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=2
)
fig.update_yaxes(
    title_text='Value',
    range=[-4, 4],
    showgrid=True, gridcolor='rgba(128,128,128,0.3)',
    row=1, col=2
)

fig.show()

```

###### ______________________________________________________________________________________________________________________________________

##### Statistics and Simulating Stochastic Processes

Fortunately, stochastic processes are random *not* uncertain and exhibit the same statistical convergence we saw in the sampling above!

 $$
 \frac{1}{N} \sum_{i=1}^N X_i(t) \xrightarrow{N \to \infty} \mathbb{E}[X(t)]
 $$

 
 and the increment distribution converging:
 
 $$
 \frac{1}{N} \sum_{i=1}^N \left( X_i(t+\Delta t) - X_i(t) \right) \xrightarrow{N \to \infty} \mathbb{E}[X(t+\Delta t) - X(t)]
 $$



```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ----------------------------------------------------------------------
# ⚙️ Simulation Setup
# ----------------------------------------------------------------------
np.random.seed(42)
mu, sigma = 0, 1
n_paths = 100
n_steps = 200
t = np.arange(n_steps)

# Simulate 100 white-noise paths
paths = np.random.normal(mu, sigma, (n_paths, n_steps))

# Theoretical normal PDF
x_pdf = np.linspace(-4, 4, 300)
pdf = norm.pdf(x_pdf, mu, sigma)

# ----------------------------------------------------------------------
# 🎞️ Build Frames — each uses actual cross-section values
# ----------------------------------------------------------------------
# 🎞️ Build Frames — each uses actual cross-section values
frames = []
for i in range(0, n_steps, 4):  # skip steps for smoother playback
    cross_section = paths[:, i]
    hist_y, bin_edges = np.histogram(cross_section, bins=25, range=(-4, 4), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    frames.append(go.Frame(
        name=f"t={i}",
        data=[
            # Left subplot (time marker)
            go.Scatter(
                x=[i, i],
                y=[-4, 4],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dot'),
                name='Time Marker',
                xaxis='x1',  # 👈 explicitly target left subplot
                yaxis='y1'
            ),
            # Right subplot (empirical histogram)
            go.Bar(
                x=bin_centers, y=hist_y,
                marker_color='cyan', opacity=0.7,
                name='Empirical Distribution',
                xaxis='x2',  # 👈 target right subplot
                yaxis='y2'
            ),
            # Right subplot (theoretical PDF)
            go.Scatter(
                x=x_pdf, y=pdf,
                mode='lines',
                line=dict(color='magenta', width=3),
                name='Normal PDF',
                xaxis='x2',
                yaxis='y2'
            )
        ]
    ))


# ----------------------------------------------------------------------
# 🧩 Figure Setup
# ----------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('White Noise Ensemble (100 Paths)', 'Empirical Cross-Section'),
    column_widths=[0.65, 0.35]
)

# Left: white-noise paths
for k in range(n_paths):
    alpha = 0.15 + 0.8 * (k / n_paths) ** 0.9
    color = f'rgba(0,255,255,{alpha:.3f})'
    fig.add_trace(
        go.Scatter(
            x=t, y=paths[k],
            mode='lines',
            line=dict(width=1.3, color=color),
            showlegend=False
        ),
        row=1, col=1
    )

# Add initial vertical line
fig.add_trace(
    go.Scatter(
        x=[0, 0], y=[-4, 4],
        mode='lines',
        line=dict(color='yellow', width=2, dash='dot'),
        name='Time Marker'
    ),
    row=1, col=1
)

# Initial histogram
initial_cross_section = paths[:, 0]
hist_y, bin_edges = np.histogram(initial_cross_section, bins=25, range=(-4, 4), density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fig.add_trace(
    go.Bar(
        x=bin_centers, y=hist_y,
        marker_color='cyan', opacity=0.7,
        name='Empirical Distribution'
    ),
    row=1, col=2
)

# Theoretical PDF overlay
fig.add_trace(
    go.Scatter(
        x=x_pdf, y=pdf,
        mode='lines',
        line=dict(color='magenta', width=3),
        name='Normal PDF'
    ),
    row=1, col=2
)

# ----------------------------------------------------------------------
# 🎬 Layout & Slider
# ----------------------------------------------------------------------
fig.frames = frames

fig.update_layout(
    height=500,
    width=1100,
    title_text="White Noise Ensemble — Cross-Sectional Empirical Distribution (100 Paths)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    sliders=[{
        "pad": {"t": 40},
        "len": 0.8,
        "x": 0.1, "y": -0.05,
        "xanchor": "left", "yanchor": "top",
        "steps": [
            {
                "method": "animate",
                "label": f"t={i}",
                "args": [[f"t={i}"],
                         {"frame": {"duration": 0, "redraw": True},
                          "mode": "immediate",
                          "transition": {"duration": 0}}],
            }
            for i in range(0, n_steps, 4)
        ]
    }]
)

# Axis styling
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='Value', range=[-4, 4], row=1, col=1)
fig.update_xaxes(title_text='Value', range=[-4, 4], row=1, col=2)
fig.update_yaxes(title_text='Density', range=[0, 0.5], row=1, col=2)

fig.show()

```

**Remark:**  There is an important note to make here about ergodicity and the ensemble vs time average.  It is beyond the scope of this video, but dictates whether the experience of some paths reflect the experience of the many - see my video on How to Trade with the Kelly Criterion for more on this topic applied to optimal bet sizing.

---

#### 2.) 🔨 Brownian Motion (Theory)

**Definition.** *Standard Brownian Motion* (or *Wiener process*) $\{W_t\}_{t \geq 0}$ is a stochastic process with the following properties:
 
 1. $W_0 = 0$.
 2. (Independent increments) For $0 \leq s < t$, the increment $W_t - W_s$ is independent of the process history up to time $s$.
 3. (Stationary increments) For $0 \leq s < t$, the increment $W_t - W_s$ is distributed as $\mathcal{N}(0, t-s)$, i.e.,
    $$
    W_t - W_s \sim \mathcal{N}(0, t-s).
    $$
 4. Almost all sample paths $t \mapsto W_t$ are continuous functions of $t$ (but nowhere differentiable).

 That is, Brownian motion is a continuous-time stochastic process that starts at zero, has independent and stationary Gaussian increments, and has continuous paths.

 We can retrieve the distribution of the stochastic process at any point by the sum of disjoint increments or a single large increment. . .

 $$
 W_t = W_0 + (W_{t_1} - W_0) + (W_{t_2} - W_{t_1}) + \cdots + (W_t - W_{t_{n-1}})
 = \sum_{k=1}^n (W_{t_k} - W_{t_{k-1}})
 = W_t - W_0
 $$

  Essentially, we can sum the changes, or we can just draw from the distribution defining the increment at a specific step




```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Brownian Motion Path and Distribution Animation ---

# Simulation parameters
n_steps = 200
dt = 1
mu = 0
sigma = 1

np.random.seed(123)
increments = np.random.normal(mu, np.sqrt(dt), n_steps)
W = np.concatenate([[0], np.cumsum(increments)])  # W_0 = 0

# Animation frames
frames = []
for t in range(1, n_steps + 1):
    W_path = W[:t+1]
    x_path = np.arange(t+1)
    
    current_var = t * dt
    y_pdf = np.linspace(-4 * np.sqrt(current_var), 4 * np.sqrt(current_var), 300)
    main_pdf = norm.pdf(y_pdf, mu, np.sqrt(current_var))

    current_point_y = W[t]
    current_point_x = norm.pdf(current_point_y, mu, np.sqrt(current_var))

    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_path, y=W_path, mode='lines',
                       line=dict(color='#00ffff', width=2),
                       name="Brownian Path"),
            go.Scatter(x=main_pdf, y=y_pdf, mode='lines',
                       line=dict(color='magenta', width=3),
                       name="W_t ~ N(0, t Δt) PDF"),
            go.Scatter(x=[current_point_x], y=[current_point_y], mode='markers',
                       marker=dict(size=10, color='#00ffff', symbol='circle'),
                       name='W_t Sample'),
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, n_steps]),
            yaxis=dict(range=[-20, 20]),
            xaxis2=dict(range=[0, 0.25]),
            yaxis2=dict(range=[-20, 20])
        ),
        name=f"step{t}"
    ))

# Initial figure setup
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Brownian Motion W_t (Path)",
                    "Distribution of W_t (N(0, t dt))"),
    column_widths=[0.6, 0.4]
)

# Initial Brownian path
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, W[1]],
        mode='lines',
        line=dict(color='#00ffff', width=2),
        name="Brownian Path"
    ),
    row=1, col=1
)

# Initial PDF
current_var = dt
y_pdf = np.linspace(-4 * np.sqrt(current_var), 4 * np.sqrt(current_var), 300)
main_pdf = norm.pdf(y_pdf, mu, np.sqrt(current_var))
fig.add_trace(
    go.Scatter(
        x=main_pdf,
        y=y_pdf,
        mode='lines',
        line=dict(color='magenta', width=3),
        name="W_t PDF"
    ),
    row=1, col=2
)

# Current sample point
fig.add_trace(
    go.Scatter(
        x=[norm.pdf(W[1], mu, np.sqrt(current_var))],
        y=[W[1]],
        mode='markers',
        marker=dict(size=10, color='#00ffff', symbol='circle'),
        name='W_1 Sample'
    ),
    row=1, col=2
)

# Animation configuration
fig.frames = frames
fig.update_layout(
    height=500, width=1000,
    title_text="Brownian Motion: Path and Distribution Evolution",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.22,
        'xanchor': 'center', 'yanchor': 'top',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0, 'easing': 'linear'}
            }]
        }]
    }]
)

# Axes styling and limits
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='W_t', range=[-20, 20], row=1, col=1)
fig.update_xaxes(title_text='Density', range=[0, 0.25], row=1, col=2)
fig.update_yaxes(title_text='W_t Value', range=[-20, 20], row=1, col=2)

fig.show()

```

###### ______________________________________________________________________________________________________________________________________

Just as above with the white noise process each step will have a distribution, and in this case they are all defined in terms of Gaussians!

However, we can see here the Gaussians will change over time - a key component of stochastic processes is capturing these time variant dynamics!


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100
n_steps = 100
dt = 1
mu = 0
sigma = 1

np.random.seed(42)
increments = np.random.normal(mu, sigma * np.sqrt(dt), (n_paths_total, n_steps))
W = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

# --- Precompute global density range for right plot (stable layout) ---
max_pdf = max(norm.pdf(0, 0, np.sqrt(t)) for t in range(1, n_steps + 1)) + 0.1

# --- Animation Frames ---
frames = []
for t in range(n_steps + 1):
    time_show = time_grid[:t + 1]
    hist_values = W[:, t]
    hist_fig = np.histogram(hist_values, bins=30, density=True)
    hist_y = 0.5 * (hist_fig[1][1:] + hist_fig[1][:-1])
    hist_x = hist_fig[0]

    # Theoretical normal distribution for W_t ~ N(0, t)
    if t == 0:
        theory_x = np.zeros_like(hist_y)
    else:
        theory_x = norm.pdf(hist_y, 0, np.sqrt(t))

    # Brownian paths (separate traces)
    scatter_paths = [
        go.Scatter(
            x=time_show,
            y=W[i, :t + 1],
            mode='lines',
            line=dict(color='#00ffff', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='none'
        )
        for i in range(n_paths_show)
    ]

    scatter_main = go.Scatter(
        x=time_show,
        y=W[0, :t + 1],
        mode='lines',
        line=dict(color='magenta', width=3),
        name="Example Path"
    )

    hist = go.Bar(
        y=hist_y,
        x=hist_x,
        width=(hist_y[1] - hist_y[0]) if len(hist_y) > 1 else 0.5,
        orientation='h',
        marker=dict(color='rgba(100,255,100,0.6)'),
        hoverinfo='skip'
    )

    theory_line = go.Scatter(
        x=theory_x,
        y=hist_y,
        mode='lines',
        line=dict(color='magenta', width=3),
        hoverinfo='skip'
    )

    vline = go.Scatter(
        x=[t, t],
        y=[-20, 20],
        mode='lines',
        line=dict(color='yellow', dash='dash'),
        showlegend=False
    )

    # Add frame without per-frame layout changes
    frames.append(go.Frame(
        data=[*scatter_paths, scatter_main, vline, hist, theory_line],
        name=f"step{t}"
    ))

# --- Initial Frame ---
init_t = 0
time_show = time_grid[:init_t + 1]
hist_values = W[:, init_t]
hist_fig = np.histogram(hist_values, bins=30, density=True)
hist_y = 0.5 * (hist_fig[1][1:] + hist_fig[1][:-1])
hist_x = hist_fig[0]
theory_x = np.zeros_like(hist_y)

scatter_paths_init = [
    go.Scatter(
        x=time_show,
        y=W[i, :init_t + 1],
        mode='lines',
        line=dict(color='#00ffff', width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='none'
    )
    for i in range(n_paths_show)
]

scatter_main_init = go.Scatter(
    x=time_show,
    y=W[0, :init_t + 1],
    mode='lines',
    line=dict(color='magenta', width=3),
)
hist_init = go.Bar(
    y=hist_y,
    x=hist_x,
    width=(hist_y[1] - hist_y[0]) if len(hist_y) > 1 else 0.5,
    orientation='h',
    marker=dict(color='rgba(100,255,100,0.6)')
)
theory_line_init = go.Scatter(
    x=theory_x,
    y=hist_y,
    mode='lines',
    line=dict(color='magenta', width=3)
)
vline_init = go.Scatter(
    x=[init_t, init_t],
    y=[-20, 20],
    mode='lines',
    line=dict(color='yellow', dash='dash'),
    showlegend=False
)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Brownian Paths", "Distribution of W_t"],
    horizontal_spacing=0.10
)
for s in scatter_paths_init:
    fig.add_trace(s, row=1, col=1)
fig.add_trace(scatter_main_init, row=1, col=1)
fig.add_trace(vline_init, row=1, col=1)
fig.add_trace(hist_init, row=1, col=2)
fig.add_trace(theory_line_init, row=1, col=2)

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=600, width=1200,
    title_text="Brownian Motion: Multiple Paths and Theoretical Distribution Evolution",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
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
                'frame': {'duration': 40, 'redraw': True},
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

# --- Axes (fixed globally, not per-frame) ---
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='W_t', range=[-20, 20], row=1, col=1)
fig.update_xaxes(title_text='Density', range=[0, max_pdf], row=1, col=2)
fig.update_yaxes(title_text='W_t', range=[-20, 20], row=1, col=2)

fig.show()
```

---

#### 3.) 📈 Pricing Options (Practice)

##### European Call Options

We know at expiration what the payoff of the option is depending on where the price ends up, its given by the max function below


```python
import plotly.graph_objects as go
import numpy as np

# Call option parameters
K = 100  # strike price

# Range of terminal prices for the payoff plot
S_T_range = np.linspace(80, 120, 200)
payoff = np.maximum(S_T_range - K, 0)

fig_payoff = go.Figure(
    data=[
        go.Scatter(
            x=S_T_range,
            y=payoff,
            mode='lines',
            line=dict(color="royalblue", width=4),
            fill='tozeroy',
            fillcolor='rgba(255,200,0,0.5)',
            name='Call Payoff'
        )
    ],
    layout=go.Layout(
        title_text="European Call Option Payoff at Expiry",
        xaxis=dict(
            title="Underlying Price $S_T$",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="Payoff $(S_T - K)^+$",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        width=800,
    )
)

fig_payoff.add_shape(
    type='line',
    x0=K, x1=K, y0=0, y1=max(payoff),
    line=dict(color="orange", width=2, dash="dash"),
    name="Strike"
)

fig_payoff.update_layout(
    showlegend=False
)
fig_payoff.show()
```

##### Problem: We have no idea where the stock price is going to end up!  We need to model that somehow. . .let's use Brownian motion!

###### ______________________________________________________________________________________________________________________________________

Let's assume our stock price follows an arithmetic Brownian motion
$$dS_t = \mu dt + \sigma dW_t$$

How do we find the distribution of $S_T$?  

We can simulate sample paths!  The terminal distribution will converge by the LLN

By the Law of Large Numbers, the empirical distribution of the simulated $S_T$ converges to the theoretical distribution: 
$$\frac{1}{n} \sum_{i=1}^n f(S_T^{(i)}) \xrightarrow{n \to \infty} \mathbb{E}[f(S_T)] \text{ for any (integrable) function } f$$





```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Parameters for arithmetic Brownian motion
mu = 0.2
sigma = 1.0
T = 1.0
S0 = 100

n_paths = 1000
n_steps = 100
dt = T / n_steps

np.random.seed(42)
# Simulate Brownian increments and path terminal values
increments = np.random.normal(mu * dt, sigma * np.sqrt(dt), (n_paths, n_steps))
S_paths = S0 + np.cumsum(increments, axis=1)
S_T = S_paths[:, -1]

# Histogram of S_T - S0
diffs = S_T - S0
hist_y, bin_edges = np.histogram(diffs, bins=30, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Theoretical terminal distribution: N(mu T, sigma^2 T)
x = np.linspace(diffs.min()-0.5, diffs.max()+0.5, 200)
theory_pdf = norm.pdf(x, loc=mu * T, scale=sigma * np.sqrt(T))

# Create subplots: Left for sample paths, right for terminal distribution
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(
        "Sample Paths of ABM", 
        "Terminal Distribution S_T - S_0"
    ),
    column_widths=[0.65, 0.35],
    horizontal_spacing=0.17
)

# ---- LEFT: Sample Paths of ABM ----
time_grid = np.linspace(0, T, n_steps)
for i in range(min(50, n_paths)):
    # Color: faded green for many, highlight if wanted
    fig.add_trace(
        go.Scatter(
            x=time_grid, y=S_paths[i] - S0, 
            mode='lines',
            line=dict(width=1, color='rgba(46, 204, 113,0.12)'),
            showlegend=False
        ),
        row=1, col=1
    )

# Draw a vertical dashed line at t=T
fig.add_trace(
    go.Scatter(
        x=[T, T], y=[(S_paths - S0).min() - 5, (S_paths - S0).max() + 5],
        mode="lines",
        line=dict(color="magenta", dash="dash", width=2),
        name="t = T", 
        showlegend=False
    ),
    row=1, col=1
)

# (Removed: Histogram and PDF on LEFT subplot)

fig.update_xaxes(
    title_text='Time (t)', row=1, col=1,
    range=[0, T*1.1], showgrid=True, gridcolor='rgba(128,128,128,0.25)'
)
fig.update_yaxes(
    title_text='S_T - S_0', row=1, col=1,
    range=[min((S_paths - S0).min(), bin_centers.min())-5, max((S_paths - S0).max(), bin_centers.max())+5],
    showgrid=True, gridcolor='rgba(128,128,128,0.25)'
)

# ---- RIGHT: Histogram + Theory PDF at Terminal ----
# Histogram (empirical)
fig.add_trace(go.Bar(
    x=bin_centers,
    y=hist_y,
    width=(bin_edges[1] - bin_edges[0]),
    marker=dict(color='rgba(100,255,100,0.6)'),
    name="Simulated Distribution",
    showlegend=False
), row=1, col=2)

# Theoretical curve
fig.add_trace(go.Scatter(
    x=x, y=theory_pdf,
    mode='lines',
    line=dict(color='magenta', width=3),
    name="Theoretical Normal",
    showlegend=False
), row=1, col=2)

fig.update_xaxes(
    title_text='$S_T - S_0$', row=1, col=2,
    range=[diffs.min()-1, diffs.max()+1], showgrid=True, gridcolor='rgba(128,128,128,0.3)'
)
fig.update_yaxes(
    title_text='Density', row=1, col=2,
    showgrid=True, gridcolor='rgba(128,128,128,0.3)'
)

fig.update_layout(
    title="Arithmetic Brownian Motion: Paths and Terminal Distribution",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=450,
    width=900,
    showlegend=True,
    legend=dict(x=0.65, y=1.0, bordercolor="grey", borderwidth=0.5, font=dict(color='white'))
)

fig.show()

```

In this case, we don't actually need to simulate sample paths - we have the terminal distribution by independent increments!

We can simply use the distribution of the increment scaled by the drift and volatility term!

$$ S_T - S_0 = \mu T + \sigma (W_T - W_0) \sim \mathcal{N}(\mu T, \sigma^2 T) $$

$$ S_T - S_0 \sim \mathcal{N}(\mu T, \sigma^2 T) $$


```python
# --- Animate empirical histogram of S_T - S_0 converging to N(mu T, sigma^2 T) normal ---
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Parameters for our arithmetic Brownian motion
mu = 0.2         # drift
sigma = 1.0      # volatility
T = 1.0          # time horizon
S0 = 100         # initial stock price

N_total = 2000   # total samples (trajectories)
frames_count = 100
bins = 30

np.random.seed(123)

# Simulate increments: S_T - S_0 = mu T + sigma * sqrt(T) * Z,  Z ~ N(0,1)
Z = np.random.randn(N_total)
increments = mu * T + sigma * np.sqrt(T) * Z

# Theoretical normal PDF for increments
x = np.linspace(
    increments.min() - 0.5, increments.max() + 0.5, 200
)
theoretical_pdf = norm.pdf(x, loc=mu * T, scale=sigma * np.sqrt(T))

# Compute empirical histograms for increasing n
sample_counts = np.linspace(10, N_total, frames_count, dtype=int)
histograms = []
for n in sample_counts:
    hist_y, bin_edges = np.histogram(increments[:n], bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    histograms.append((bin_centers, hist_y))

# Build animation frames
frames = []
for i, n in enumerate(sample_counts):
    prev_y = np.zeros_like(histograms[i][1]) if i == 0 else histograms[i-1][1]
    curr_y = histograms[i][1]
    intermediates = [
        prev_y + (curr_y - prev_y) * t for t in np.linspace(0, 1, 4)
    ]
    for step, yvals in enumerate(intermediates):
        frames.append(go.Frame(
            data=[
                go.Bar(
                    x=histograms[i][0],
                    y=yvals,
                    width=0.9 * (histograms[i][0][1] - histograms[i][0][0]),
                    marker_color="orange",
                    opacity=0.7,
                    name="Empirical Histogram"
                ),
                go.Scatter(
                    x=x,
                    y=theoretical_pdf,
                    mode="lines",
                    line=dict(color="royalblue", width=3),
                    name="Theoretical Normal"
                )
            ],
            layout=go.Layout(
                xaxis=dict(range=[increments.min()-1, increments.max()+1]),
                yaxis=dict(range=[0, 0.6])
            ),
            name=f"n={n}_step{step}"
        ))

# Initial (empty) figure
fig_abm = go.Figure(
    data=[
        go.Bar(
            x=np.linspace(increments.min(), increments.max(), bins),
            y=[0]*bins,
            width=0.9 * (increments.max()-increments.min()) / bins,
            marker_color="orange",
            opacity=0.7,
            name="Empirical Histogram"
        ),
        go.Scatter(
            x=x,
            y=theoretical_pdf,
            mode="lines",
            line=dict(color="royalblue", width=3),
            name="Theoretical Normal"
        )
    ],
    layout=go.Layout(
        title_text="Convergence of ABM Increment S_T - S_0 to Normal Distribution",
        xaxis=dict(
            title="S_T - S_0",
            range=[increments.min()-1, increments.max()+1],
            showgrid=True, gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(
            title="Density",
            range=[0, 0.6],
            showgrid=True, gridcolor="rgba(128,128,128,0.3)",
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        updatemenus=[{
            "type": "buttons",
            "x": 0.5, "y": -0.19,   # lower the button further down
            "xanchor": "center", "yanchor": "top",
            "showactive": False,
            "buttons": [{
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 50, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0, "easing": "linear"}
                }]
            }]
        }]
    ),
    frames=frames
)

fig_abm.show()
```

###### ______________________________________________________________________________________________________________________________________

 The value of a European call option can be expressed as the expected discounted payoff:
 
 $$
 C = \mathbb{E} \left[ e^{-rT} \; (S_T - K)^+ \right]
 $$
 
 where $C$ is the option price, $r$ is the risk-free rate, $T$ is the maturity, $S_T$ is the price at maturity, $K$ is the strike price, and $(x)^+ = \max(x,0)$.  **Since we can never predict the outcome of a random variable the expectation is the best we can do to come up with a price for this option.**  
 
 In reality, we are dealing with uncertainty, not randomness, so non-linear conditional expectations can certainly outpreform - but I digress for the sake of brevity.



Using the definition of an expectation for a function of a random variable
$$
C = \mathbb{E}\left[ e^{-rT} (S_T - K)^+ \right] = \int_{-\infty}^{\infty} e^{-rT} \, (S_0 + \mu T + \sigma \sqrt{T} z - K)^+ \; \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz
$$

  By the Law of Large Numbers (LLN), if we simulate many samples of $S_T$, the sample mean of the discounted payoff converges to the true expectation as $N \to \infty$:
  
 $$
  \frac{1}{N} \sum_{i=1}^N e^{-rT} \max(S_T^{(i)} - K, 0) \xrightarrow{N \to \infty} \mathbb{E}\left[ e^{-rT} (S_T - K)^+ \right] = C
 $$


```python
import numpy as np
import plotly.graph_objects as go

# Parameters
S0 = 100      # initial price
mu = 0.05     # drift
sigma = 0.2   # volatility
T = 1.0       # time (in years)
n_steps = 200
K = 105       # strike
dt = T / n_steps

increments = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
ln_S = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(increments)])
S = np.exp(ln_S)

frames = []
for t in range(1, n_steps+1):
    expired = (t == n_steps)
    if expired:
        color = 'green' if S[t] > K else 'red'
    else:
        color = 'royalblue'
    frames.append(go.Frame(
        data=[
            go.Scatter(x=np.arange(t+1), y=S[:t+1],
                       mode='lines',
                       line=dict(width=3, color=color),
                       name="Sample Path"),
            go.Scatter(x=[0, n_steps], y=[K, K], mode='lines',
                       line=dict(dash='dash', width=2, color='orange'),
                       name='Strike', showlegend=(t==1)),
        ],
        name=f"step{t}"
    ))

# Match the grid color to the one used above:
grid_color = "rgba(128,128,128,0.3)"

# Initial plot
fig_sample = go.Figure(
    data=[
        go.Scatter(x=[0, 1], y=[S[0], S[1]], mode='lines',
                   line=dict(width=3, color='royalblue'), name='Sample Path'),
        go.Scatter(x=[0, n_steps], y=[K, K], mode='lines',
                   line=dict(dash='dash', width=2, color='orange'), name='Strike')
    ],
    layout=go.Layout(
        title="Sample Path of Option Underlying and Strike",
        xaxis=dict(
            title="Step", 
            range=[0, n_steps],
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
        ),
        yaxis=dict(
            title="Underlying Price", 
            range=[min(S0, np.min(S))-5, max(S0, np.max(S))+5],
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        updatemenus=[{
            "type": "buttons",
            "x": 0.5, "y": -0.20,  # Lowered the play button
            "xanchor": "center", "yanchor": "top",
            "showactive": False,
            "buttons": [{
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 10, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0}}]
            }]
        }]
    ),
    frames=frames
)

fig_sample.show()

```

Effectively, we are simulating the distribution of European call option payoffs

$$C = e^{-rT} \mathbb{E}[(S_0 + \mu T + \sigma W_T - K)^+]$$

and taking the average value (then discounting it back to the present, time value of money) to get the option price!


```python
import plotly.subplots as sp

# Parameters for simulation
n_paths = 100  # number of sample paths to visualize LLN
r = 0.05       # risk-free rate for discounting

# Simulate n_paths terminal values
rand_increments = np.random.normal((mu - 0.5 * sigma**2) * T, sigma * np.sqrt(T), n_paths)
S_T = S0 * np.exp(rand_increments)
payoffs = np.maximum(S_T - K, 0)
discounted_payoffs = np.exp(-r * T) * payoffs
cum_avg = np.cumsum(discounted_payoffs) / np.arange(1, n_paths + 1)

# x for "hockey stick"
S_T_fine = np.linspace(0, max(np.max(S_T), K+30), 400)
hockey_payoff = np.maximum(S_T_fine - K, 0)
discounted_hockey_payoff = np.exp(-r * T) * hockey_payoff

# Prepare frames for animation
frames = []
for t in range(1, n_paths + 1):
    # Left: Hockey stick diagram with points up to t
    left_scatter = [
        go.Scatter(
            x=S_T_fine, 
            y=discounted_hockey_payoff,
            mode='lines',
            line=dict(color='orange', dash='dash', width=2),
            name='Payoff Function',
            showlegend=(t==1)
        ),
        go.Scatter(
            x=S_T[:t], 
            y=discounted_payoffs[:t],
            mode='markers',
            marker=dict(size=10, color='royalblue', line=dict(width=2, color='white')),
            name="Discounted Payoff",
            showlegend=(t==1)
        )
    ]
    
    # Right: LLN convergence (cumulative average and final option value)
    right_scatter = [
        go.Scatter(
            x=np.arange(1, t+1), 
            y=cum_avg[:t],
            mode='lines+markers',
            marker=dict(color='limegreen', size=7),
            line=dict(color='limegreen', width=3),
            name='Cumulative Price',
            showlegend=(t==1)
        ),
        go.Scatter(
            x=[1, t], 
            y=[cum_avg[-1], cum_avg[-1]],
            mode='lines',
            line=dict(color='orange', dash='dash', width=2),
            name='Final Est.',
            showlegend=(t==1)
        )
    ]
    # Dynamic xlim
    right_xlim = [1, max(10, t + 3)]

    frames.append(go.Frame(
        data=left_scatter + right_scatter,
        name=f"step{t}",
        layout=go.Layout(
            xaxis2=dict(range=right_xlim)
        )
    ))

# Build subplot figure  
specs = [[{"type": "xy"}, {"type": "xy"}]]
subplot_fig = sp.make_subplots(
    rows=1, cols=2, 
    subplot_titles=(
        "Discounted Payoff vs. Terminal Price (Hockey-Stick)", 
        "Law of Large Numbers and Option Price Convergence"
    )
)

# Initial frame data
# Left
subplot_fig.add_trace(
    go.Scatter(
        x=S_T_fine, 
        y=discounted_hockey_payoff, 
        mode='lines',
        line=dict(color='orange', dash='dash', width=2),
        name='Payoff Function'
    ),
    row=1, col=1
)
subplot_fig.add_trace(
    go.Scatter(
        x=[S_T[0]], 
        y=[discounted_payoffs[0]], 
        mode='markers',
        marker=dict(size=10, color='royalblue', line=dict(width=2, color='white')),
        name="Discounted Payoff"
    ),
    row=1, col=1
)

# Right
subplot_fig.add_trace(
    go.Scatter(
        x=[1], 
        y=[cum_avg[0]],
        mode='lines+markers',
        marker=dict(color='limegreen', size=7),
        line=dict(color='limegreen', width=3),
        name='Cumulative Price'
    ),
    row=1, col=2
)
subplot_fig.add_trace(
    go.Scatter(
        x=[1, 1],
        y=[cum_avg[-1]]*2,
        mode='lines',
        line=dict(color='orange', dash='dash', width=2),
        name='Final Est.'
    ),
    row=1, col=2
)

subplot_fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title="Visualizing Option Monte Carlo: Payoff Distribution vs. Price Convergence",
    showlegend=False,
    width=1000,
    height=420,
    xaxis=dict(
        title="Terminal Price $S_T$",
        showgrid=True, gridcolor=grid_color, gridwidth=1
    ),
    yaxis=dict(
        title="Discounted Payoff",
        showgrid=True, gridcolor=grid_color, gridwidth=1,
        range=[-1, np.max(discounted_hockey_payoff)+5]
    ),
    xaxis2=dict(
        title="Number of Simulations",
        showgrid=True, gridcolor=grid_color, gridwidth=1,
        range=[1, max(10, n_paths//15)]
    ),
    yaxis2=dict(
        title="Option Price Estimate",
        showgrid=True, gridcolor=grid_color, gridwidth=1,
        range=[np.min(cum_avg)-1, np.max(cum_avg)+1]
    ),
    updatemenus=[{
        "type": "buttons",
        "x": 0.5, "y": -0.18,
        "xanchor": "center", "yanchor": "top",
        "showactive": False,
        "buttons": [{
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 20, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0}
            }]
        }]
    }]
)
subplot_fig.frames = frames

subplot_fig.show()
```

---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Random variables represent a set of possible outomces with associated probabilities, we can **NEVER** predict the outcome of any one event
- Random variables have statistical and distribution convergence but in reality we are modeling uncertainty which lacks convergence by non-stationarity
- Normal random variables are useful as sample averages by the CLT follow a normal distribution and are useful for bridging probability and statistics
- Brownian motion is also defined in terms of a normal (Gaussian) distribution, quite an important distribution!
- Stochastic processes are a collection of random variables indexed by a draw or even time, we can **NEVER** predict them either
- Brownian motion is defined by a zero mean, independent and stationary Gaussian increments, and is a.s. continuous
- We observe how uncertainty increases in the increment the larger the time step capture time varying dynamics, the core use of stochastic processes
- In practice, we may choose to model a stock price process with a stochastic process using Brownian motion to model the uncertainty
- We see even in stochastic processes we observe the *nice* properties of convergence allowing us to easily simulate option prices
- In reality, more sophisticated modeling is required (Black-Scholes, and beyond for modeling skew/term structure of volatility to extrapolate prices)

**Future Topics**

Technical Videos and Other Discussions

- Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
- Stochastic Proccesses: Brownian Motion, Arithmetic (additive) Geometric (multiplicative) Brownian Motion
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
- Most Popular Quant Models for Informed Trading
- Buy Side vs. Sell Side Quants
- Quant Roadmap (How I would start learning if I had to start over)

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
- Automated Delta-Neutral Trading System

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
