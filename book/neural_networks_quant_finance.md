###  🧠 Neural Networks for Quant Finance

##### ▶️ Related Quant Guild Videos:

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)

- [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)

- [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/wZChBKDFFeU)

- [Quant vs. Discretionary Trading](https://youtu.be/3gblERSSHXI)

- [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)

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

#### 1.) 📊 How Machines Learn

- Data and Mean Squared Error

- Overfitting vs Learning (Robustness OOS)

- What Does AI Actually Learn?

#### 2.) 🎯 Neural Networks

- Visualization and Mathematics

- Neural Networks as Universal Function Approximators

- Nearly Solved vs Unsolved Problems

- Applications in Quant Finance

#### 3.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 📊 How Machines Learn

Machines learn from data, the primary objective is to minimize distance of a *prediction* to a target

We pick some model $\mathcal{M}$ and that model will produce a prediction $\hat{y}_i$ for a corresponding target $y_i$

Choices for $\mathcal{M}$ include but are not limited to. . .
- Linear or Logistic Regression
- Random Forests
- Support Vector Machines
- Neural Networks
- . . .

Each model requires parameters $\Theta$, and available data $X_i$, effectively $\mathcal{M}(\hat{\Theta}, X_i) \rightarrow \hat{y}_i$

We call any of these *machine learning* models, neural networks are a type of machine learning model that facilitate *deep learning* or "artificial intelligence" when there are multiple layers (as we will see in section 2 of this notebook)

###### ______________________________________________________________________________________________________________________________________

##### Example: Max Football Throwing Distance

I will **NOT** use predicting stock prices or returns here to explain how models learn as it is **NOT** how these learning models are applied in practice, we will discuss this later on - for now let's focus on understanding how it is models learn from data. . .

We ask people to throw a football as far as possible, our goal is to build a model to predict their distance before they throw it

<div align="center">
    <img src="football.png" alt="bazaar_items" width="700"/>
</div>

We pick some model $\mathcal{M}(\Theta)$ and train it on data, let's see what this entails. . .

We observe hundres of people throw a football and record their **height, weight, max bench press, age**

These are just a few examples of *features* that may help inform our prediction - other features may be more useful

- **Input Data**, $X_i:$ (6'1, 200, 350, 30)
- **Target**, $y_i:$ (observed distance) 50yards
- **Prediction**, $\hat{y}_i:$ $\mathcal{M}(\hat{\Theta})$, our model learns this from data

<div align="center">
    <img src="football_2.png" alt="bazaar_items" width="700"/>
</div>

Models effectively *learn* from a data set of features we select (player stats in this case) and target outcomes


```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 100

# Generate features
heights = np.random.normal(70, 3, n_samples)              # Height in inches, ~avg adult male
weights = np.random.normal(200, 20, n_samples)            # Weight in lbs
bench_press = np.random.normal(250, 40, n_samples)        # Max bench press in lbs
ages = np.random.normal(30, 5, n_samples)                 # Age in years

# Distance is most correlated to height & weight, weakly to age & bench press
# coefficient for each (height: strong, weight: strong, bench_press: weak, age: weak/neg)
noise = np.random.normal(0, 4, n_samples)
distance = (
    1.2 * heights +          # strong
    0.9 * weights +          # strong
    0.2 * bench_press +      # weak
    -0.2 * ages +            # weak/neg
    noise
)

# You may scale distance so it's realistic (yards)
distance = distance / 10    # Bring to typical football throw scale

# Create DataFrame
df = pd.DataFrame({
    'height': heights,
    'weight': weights,
    'max_bench_press': bench_press,
    'age': ages,
    'distance': distance
})

# Display the first few rows
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>height</th>
      <th>weight</th>
      <th>max_bench_press</th>
      <th>age</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71.490142</td>
      <td>171.692585</td>
      <td>264.311494</td>
      <td>25.855025</td>
      <td>28.162508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>69.585207</td>
      <td>191.587094</td>
      <td>272.431381</td>
      <td>27.199095</td>
      <td>30.257959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71.943066</td>
      <td>193.145710</td>
      <td>293.322050</td>
      <td>33.736468</td>
      <td>31.210091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.569090</td>
      <td>183.954455</td>
      <td>292.152082</td>
      <td>33.051851</td>
      <td>30.704989</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69.297540</td>
      <td>196.774286</td>
      <td>194.893225</td>
      <td>29.895492</td>
      <td>29.145319</td>
    </tr>
  </tbody>
</table>
</div>



###### ______________________________________________________________________________________________________________________________________

Functionally, we can represent a model's error as the distance between its prediction and actual observed value (target)

 This is typically called a loss function

Sometimes (depending on the model) we can graph it, in any case, we can graph an abstraction to understand what's happening

 The mean squared error (MSE) is a common loss function for regression problems:

 $$
 \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^N (y_i - \mathcal{M}(\Theta))^2
 $$

- $y_i$ is the target (observed distance, for example) value
- $\hat{y}_i = \mathcal{M}(\Theta)$  is the model predicted value (we need to learn $\Theta$)
- $N$ is the number of observations

To minimize the error we find

 $$
 \min_{\Theta}~ \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
 $$

The parameters for our model

 $$
 \Theta^* = \argmin_{\Theta}~ \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
 $$


```python
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp

# x values: model parameter space
theta = np.linspace(-6, 6, 400)

# Convex: parabola
convex_error = 0.5*(theta - 2)**2 + 1.5
theta_star = 2
min_error = 0.5*(theta_star - 2)**2 + 1.5

# Non-convex: sum of parabola (same min) + bumps
nonconvex_error = 0.09*(theta - 2)**2 + 2 + 1.3*np.sin(1.2*theta) + 0.7*np.sin(2.5*theta)
nonconvex_min_ix = np.argmin(nonconvex_error)
theta_star_ncvx = theta[nonconvex_min_ix]
min_error_ncvx = nonconvex_error[nonconvex_min_ix]

fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Convex Loss: Unique Minimum",
        "Non-Convex Loss: Local Extrema"
    ),
    horizontal_spacing=0.13
)

# Left plot: convex
fig.add_trace(
    go.Scatter(
        x=theta, y=convex_error,
        mode='lines',
        line=dict(color='deepskyblue', width=3),
        name='Convex Error',
        showlegend=False
    ),
    row=1, col=1
)
# Mark theta star
fig.add_trace(
    go.Scatter(
        x=[theta_star], y=[min_error],
        mode='markers',
        marker=dict(size=20, color='orange', symbol='star'),
        name='optimal convex',
        showlegend=True
    ),
    row=1, col=1
)
# Draw vertical line at theta star
fig.add_trace(
    go.Scatter(
        x=[theta_star, theta_star], y=[convex_error.min()-0.5, convex_error.max()],
        mode='lines',
        line=dict(color='orange', dash='dash', width=2),
        showlegend=False
    ),
    row=1, col=1
)

# Right plot: non-convex
fig.add_trace(
    go.Scatter(
        x=theta, y=nonconvex_error,
        mode='lines',
        line=dict(color='mediumvioletred', width=3),
        name='Non-Convex Error',
        showlegend=False
    ),
    row=1, col=2
)
# Find and show local minima
from scipy.signal import argrelextrema
local_min_ix = argrelextrema(nonconvex_error, np.less, order=10)[0]
fig.add_trace(
    go.Scatter(
        x=theta[local_min_ix], y=nonconvex_error[local_min_ix],
        mode='markers',
        marker=dict(size=13, color='orange', symbol='diamond'),
        name='Local Minima',
        showlegend=True
    ),
    row=1, col=2
)
# Global min as a star
fig.add_trace(
    go.Scatter(
        x=[theta_star_ncvx], y=[min_error_ncvx],
        mode='markers',
        marker=dict(size=20, color='deepskyblue', symbol='star'),
        name='optimal non convex',
        showlegend=True
    ),
    row=1, col=2
)
# Vertical line at global non-convex min
fig.add_trace(
    go.Scatter(
        x=[theta_star_ncvx, theta_star_ncvx], y=[nonconvex_error.min()-0.5, nonconvex_error.max()],
        mode='lines',
        line=dict(color='deepskyblue', dash='dash', width=2),
        showlegend=False
    ),
    row=1, col=2
)

fig.update_layout(
    width=950,
    height=410,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        title="Model parameters",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    yaxis=dict(
        title="Error",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    xaxis2=dict(
        title="Model parameters",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    yaxis2=dict(
        title="Error",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    title="Convex vs Non-Convex Optimization Landscapes",
    legend=dict(
        orientation='h',     # Horizontal legend
        yanchor='top',
        y=-0.23,             # Position legend below the plot
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    )
)
fig.show()

```



**How are these minimization problems sovled?**

- These models are *far too complex* to solve with traditional minimization (set derivative to zero)

- We typically deploy numerical schemes (like gradient descent and variations) to find minima

- *Problem:* we can't visualize the parameter space (millions of parameters, sometimes more) so we can get stuck at local extrema

**Remark:** Non-convex optimization is a favorite topic of mine, if you'd like to see a video on different optimization schemes let me know in the comments below

###### ______________________________________________________________________________________________________________________________________

##### Learning Model Parameters

Any model we choose ($\mathcal{M}$) has a parameter set $\Theta \in \mathbb{R}^{\xi}$ of arbitrary dimension $\xi$

When we *fit* a model to data, we get one possible estimation for $\Theta$, we can call it $\hat{\Theta}$


```python
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# --- Data setup ---
np.random.seed(42)
n = 30
X_base = np.linspace(-3, 3, n).reshape(-1, 1)
y_true_f = np.tanh(1.2 * X_base) + 0.2 * np.sin(2.5 * X_base)
y_base = y_true_f + 0.4 * np.random.randn(n, 1)

# --- Models ---
linreg = LinearRegression().fit(X_base, y_base)
pred_lin = lambda x: linreg.predict(x)

poly_over = PolynomialFeatures(degree=18)
overfit = Ridge(alpha=0.0, fit_intercept=False).fit(poly_over.fit_transform(X_base), y_base)
pred_overfit = lambda x: overfit.predict(poly_over.transform(x))

poly_rob = PolynomialFeatures(degree=3)
robust = Ridge(alpha=0.2).fit(poly_rob.fit_transform(X_base), y_base)
pred_robust = lambda x: robust.predict(poly_rob.transform(x))

colors = ['orange', 'red', 'limegreen']

# --- New points ---
np.random.seed(123)
n_new = 10
X_new = np.random.uniform(-3, 3, n_new).reshape(-1, 1)
y_true_new = np.tanh(1.2 * X_new) + 0.2 * np.sin(2.5 * X_new)
y_new = y_true_new + 0.4 * np.random.randn(n_new, 1)

# --- Model predictions ---
xs_r = np.linspace(-3, 3, 250).reshape(-1, 1)
y_pred_lin = pred_lin(xs_r).flatten()
y_pred_overfit = pred_overfit(xs_r).flatten()
y_pred_robust = pred_robust(xs_r).flatten()

ys_lin_new = pred_lin(X_new).flatten()
ys_overfit_new = pred_overfit(X_new).flatten()
ys_robust_new = pred_robust(X_new).flatten()

# --- Figure ---
fig = make_subplots(rows=1, cols=3,
    subplot_titles=("Underfit Model", "Overfit Model", "Robust Model"),
    horizontal_spacing=0.08)

models = [
    ("Underfit", colors[0], y_pred_lin, ys_lin_new),
    ("Overfit", colors[1], y_pred_overfit, ys_overfit_new),
    ("Robust", colors[2], y_pred_robust, ys_robust_new)
]

# Static: training data + model curve
for i, (name, color, curve, _) in enumerate(models):
    fig.add_trace(go.Scatter(x=X_base.flatten(), y=y_base.flatten(),
                             mode='markers', marker=dict(size=7, color='skyblue'),
                             showlegend=False),
                  row=1, col=i+1)
    fig.add_trace(go.Scatter(x=xs_r.flatten(), y=curve,
                             mode='lines', line=dict(color=color, width=4, dash='dot' if name=="Overfit" else None),
                             name=name, showlegend=False),
                  row=1, col=i+1)
    # Dynamic placeholders for OOS points and error bars
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                             marker=dict(size=14, color='gold', line=dict(width=2, color='black')),
                             showlegend=False),
                  row=1, col=i+1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                             line=dict(color='red', width=2, dash='dash'),
                             showlegend=False, hoverinfo='skip'),
                  row=1, col=i+1)

# Trace index mapping for dynamic content
# (Underfit: 2–3, Overfit: 6–7, Robust: 10–11)
dyn_idx = [(2,3), (6,7), (10,11)]

# --- Frames ---
frames = []
for k in range(1, n_new+1):
    frame_data = []
    for (name, color, _, ys_pred), (pt_idx, err_idx) in zip(models, dyn_idx):
        # points up to k
        frame_data.append(dict(
            type="scatter",
            x=X_new[:k,0],
            y=y_new[:k,0],
            mode="markers",
            marker=dict(size=14, color='gold', line=dict(width=2, color='black')),
        ))
        # error lines up to k
        xerr, yerr = [], []
        for j in range(k):
            xerr += [X_new[j,0], X_new[j,0], None]
            yerr += [ys_pred[j], y_new[j,0], None]
        frame_data.append(dict(
            type="scatter",
            x=xerr, y=yerr,
            mode="lines",
            line=dict(color='red', width=2, dash='dash'),
        ))

    mse_lin = mean_squared_error(y_new[:k], ys_lin_new[:k])
    mse_overfit = mean_squared_error(y_new[:k], ys_overfit_new[:k])
    mse_robust = mean_squared_error(y_new[:k], ys_robust_new[:k])
    title = (f"Out-of-sample Points Added: <b>{k}/{n_new}</b> | "
             f"<span style='color:{colors[0]}'>Underfit MSE: {mse_lin:.3f}</span> &nbsp; "
             f"<span style='color:{colors[1]}'>Overfit MSE: {mse_overfit:.3f}</span> &nbsp; "
             f"<span style='color:{colors[2]}'>Robust MSE: <b>{mse_robust:.3f}</b></span>")

    frames.append(go.Frame(name=f"frame{k}", data=frame_data,
                           traces=[2,3,6,7,10,11],
                           layout=go.Layout(title=dict(text=title))))

# --- Layout ---
fig.update_layout(
    width=1150, height=400,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title=dict(text="Out-of-sample Points Added: <b>0</b>", y=0.95),
    updatemenus=[dict(
        type="buttons", x=0.5, xanchor="center", y=-0.23, yanchor="top",
        direction="right",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=700, redraw=True),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode="immediate"
                )],
                # Add visual styling for the button
                # For dark background with white border and white text
                args2=None,  # ignore if unsupported in plotly
                # Plotly allows design via updatemenus, but you can set extra styling hints here for clarity
            ),
            dict(
                label="Reset",
                method="animate",
                args=[["frame1"], dict(
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                    mode="immediate"
                )]
            )
        ],
        bgcolor="rgba(0,0,0,0.85)",
bordercolor="white",
borderwidth=1.5,
font=dict(color="white", size=13)
  # ensure text remains visible
    )],
    showlegend=False
)

# Make gridline color consistent with previous plot (#323232)
for i in range(1,4):
    fig.update_xaxes(title_text="Input x", row=1, col=i,
                     showgrid=True, gridcolor="#323232", gridwidth=1)
    fig.update_yaxes(title_text="Target y", row=1, col=i,
                     showgrid=True, gridcolor="#323232", gridwidth=1)

fig.frames = frames
fig.show()

```

    c:\Users\Roman\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\linear_model\_ridge.py:215: LinAlgWarning:
    
    Ill-conditioned matrix (rcond=1.76539e-21): result may not be accurate.
    




###### ______________________________________________________________________________________________________________________________________

##### How Can we Avoid Overfitting Models?

We split up our data into training, validation, and testing sets

Use the training and validation data to fit the model parameters $\Theta$ and ensure we don't *overfit*

Then test the model on the *testing set* which resembles out-of-sample (OOS, unseen) data

<div align="center">
    <img src="trainvaltest.png" alt="bazaar_items" width="700"/>
</div>

If the data is sampled in the same capacity, and the population distribution is stable, we will observe similar performance on new data


```python
import plotly.graph_objs as go
import numpy as np

# Simulate epochs
epochs = np.arange(1, 51)
# Model training loss (decays rapidly then asymptotes)
train_loss = 0.85 * np.exp(-0.15*epochs) + 0.12 + 0.01*np.random.randn(len(epochs))
# Validation loss (decays, then bottoms and increases due to overfitting)
val_loss = 0.95 * np.exp(-0.14*epochs) + 0.13 + 0.22 * (epochs > 33) * (epochs-33)/17 + 0.01*np.random.randn(len(epochs))

# Pick epoch where validation loss is minimum (early stopping)
early_stop_epoch = np.argmin(val_loss) + 1  # epochs are 1-indexed

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=epochs, y=train_loss, mode='lines+markers',
    name='Training Loss',
    line=dict(color='deepskyblue', width=3)
))

fig.add_trace(go.Scatter(
    x=epochs, y=val_loss, mode='lines+markers',
    name='Validation Loss',
    line=dict(color='orange', width=3)
))

# Add vertical line at early_stopping point
fig.add_trace(go.Scatter(
    x=[early_stop_epoch, early_stop_epoch],
    y=[min(train_loss.min(), val_loss.min())-0.02, max(train_loss.max(), val_loss.max())+0.02],
    mode='lines',
    line=dict(color='crimson', width=4, dash='dash'),
    name='Stop Training',
    showlegend=True
))

# Annotate 'Stop Training'
fig.add_annotation(
    x=early_stop_epoch,
    y=val_loss[early_stop_epoch-1],
    text="Stop Training",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=-40,
    font=dict(color='crimson', size=14),
    bgcolor="rgba(255,255,255,0.94)",
    bordercolor="crimson"
)

fig.update_layout(
    width=770, height=400,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        title="Epoch",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        tickmode='linear', dtick=5
    ),
    yaxis=dict(
        title="Loss",
        showgrid=True, gridcolor="#323232", gridwidth=1,
        rangemode='tozero'
    ),
    title="Training vs Validation Loss (Early Stopping Shown)",
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.18,
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    )
)
fig.show()
```



###### ______________________________________________________________________________________________________________________________________

##### What Do Models Learn?

This is the difference between a random variable and an uncertain event. . .
 
 Models, especially those trained using loss functions like mean squared error (MSE), learn to predict the *conditional expectation* of the target variable given the input features.
 
 $$
 P(Y = k) = 
 \begin{cases}
 \frac{1}{6} & \text{for } k \in \{1,2,3,4,5,6\} \\
 0 & \text{otherwise}
 \end{cases}
 \qquad\qquad
 \mathbb{E}[Y] = \sum_{k=1}^6 k \cdot \frac{1}{6} = 3.5
 $$
 
 
Let's train a model to predict the output of a dice roll. . .


```python
import numpy as np
from tensorflow import keras

# Simulate 10,000 dice rolls (fair 6-sided die, faces 1-6)
np.random.seed(42)
n_samples = 10000
X_dice = np.zeros((n_samples, 1))  # No features, so just use a constant input
y_dice = np.random.randint(1, 7, size=(n_samples, 1))

# Build a very simple neural network (one input, one output, no hidden layers)
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1, activation=None)
])
model.compile(optimizer='adam', loss='mse')

# Train (fit) the neural network with verbose output
history = model.fit(X_dice, y_dice, epochs=30, verbose=1)

# Predict the learned value
y_pred_nn = model.predict(np.zeros((1, 1)))[0,0]

# Theoretical dice expectation
expected_value = (1+2+3+4+5+6)/6

print(f"Theoretical expectation of a dice roll: {expected_value:.3f}")
print(f"Neural network predicted value for dice roll: {y_pred_nn:.3f}")
```

    Epoch 1/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 240us/step - loss: 14.1188
    Epoch 2/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 302us/step - loss: 12.2033
    Epoch 3/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228us/step - loss: 10.5418
    Epoch 4/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step - loss: 9.1067
    Epoch 5/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230us/step - loss: 7.8741
    Epoch 6/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 321us/step - loss: 6.8253
    Epoch 7/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225us/step - loss: 5.9436
    Epoch 8/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 227us/step - loss: 5.2116
    Epoch 9/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 241us/step - loss: 4.6143
    Epoch 10/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 243us/step - loss: 4.1363
    Epoch 11/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 243us/step - loss: 3.7640
    Epoch 12/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230us/step - loss: 3.4820
    Epoch 13/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 232us/step - loss: 3.2767
    Epoch 14/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step - loss: 3.1335
    Epoch 15/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 240us/step - loss: 3.0397
    Epoch 16/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 223us/step - loss: 2.9819
    Epoch 17/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221us/step - loss: 2.9487
    Epoch 18/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 2.9317
    Epoch 19/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step - loss: 2.9240
    Epoch 20/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225us/step - loss: 2.9211
    Epoch 21/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 243us/step - loss: 2.9201
    Epoch 22/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 239us/step - loss: 2.9197
    Epoch 23/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 234us/step - loss: 2.9196
    Epoch 24/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 2.9197
    Epoch 25/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 224us/step - loss: 2.9196
    Epoch 26/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 2.9195
    Epoch 27/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 223us/step - loss: 2.9196
    Epoch 28/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 2.9196
    Epoch 29/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 2.9196
    Epoch 30/30
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222us/step - loss: 2.9197
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 14ms/step
    Theoretical expectation of a dice roll: 3.500
    Neural network predicted value for dice roll: 3.504


**Formally:**  
 Given input features $X$ and target $Y$, the model trained with MSE minimizes:
 $$
 \mathbb{E}\left[(Y - \hat{f}(X))^2\right]
 $$
 The function $\hat{f}(x)$ that minimizes this expected loss is:
 $$
 \hat{f}(x) = \mathbb{E}[Y|X=x]
 $$
 This means:  
 > The best the model can do (on average) is to learn to output the expected value of $Y$ given $X$.

###### ______________________________________________________________________________________________________________________________________

##### Models are Learning Expectations - There is **NO SUCH THING AS A PREDICTION**

As we have seen on this channel, producing a reasonable expectation is sufficient for generating P/L from both a sell side and buy side perspective.  This is equivalent to the idea of quoting prices around mid and hedging and trading with an edge.

###### ______________________________________________________________________________________________________________________________________

I largely follow Wolfram's notion of computational irreducibility on the limitations of "prediction" in the context of modeling, that is we must simulate a near perfect step-by-step evolution of the system to provide a "prediction" which is unreasonable as by the time we have this the event of interest has already occurred.  If there is interest I would like to do a video on this in the future especially considering Markovian and non-Markovian models (where entire path simulation is required)

###### ______________________________________________________________________________________________________________________________________

##### Why Can't Models Learn Stock Prices or Returns?

It should also be clear why we can't apply neural networks outright to stock prices or returns

The entire space is non-stationary; probabilities, distributions, and statistics do not converge

If we try to use a nueral network to approximate an expectation function in the context of stock returns we will be overfitting


```python
import numpy as np
import plotly.graph_objs as go

# Simulate ABM-like stock price path
np.random.seed(42)
T = 200
t = np.arange(T)

price = np.zeros(T)
price[0] = 100

for i in range(1, T):
    drift = 0.05
    shock = np.random.randn() * 0.6
    mean_revert = 0.03 * (100 - price[i-1])
    price[i] = price[i-1] + drift + shock + mean_revert

# Split train/test
train_cut = 130
t_train, t_test = t[:train_cut], t[train_cut:]
p_train, p_test = price[:train_cut], price[train_cut:]

# Polynomial fit & predict
deg = 6
coef = np.polyfit(t_train, p_train, deg)
poly_fit = np.poly1d(coef)
p_pred_train = poly_fit(t_train)
p_pred_test = poly_fit(t_test)
t_future = np.arange(T, T + 20)
p_pred_future = poly_fit(t_future)

# Compose plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=price, mode='lines+markers',
    line=dict(color='skyblue', width=2),
    name='ABM True Path'
))
fig.add_trace(go.Scatter(
    x=t_train, y=p_pred_train,
    mode='lines',
    line=dict(color='limegreen', width=3),
    name='Polynomial Fit (Train)'
))
fig.add_trace(go.Scatter(
    x=t_test, y=p_pred_test,
    mode='lines',
    line=dict(color='crimson', width=3, dash='dot'),
    name='Polynomial Fit (OOS)'
))
fig.add_trace(go.Scatter(
    x=t_future, y=p_pred_future,
    mode='lines',
    line=dict(color='crimson', width=2, dash='dot'),
    opacity=0.6,
    name='Polynomial Extrapolation (Future)'
))
fig.add_trace(go.Scatter(
    x=[train_cut, train_cut],
    y=[min(price)-5, max(price)+5],
    mode='lines',
    line=dict(color='white', width=2, dash='dash'),
    showlegend=False
))

fig.update_xaxes(showgrid=True, gridcolor="#323232", gridwidth=1, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#323232", gridwidth=1, zeroline=False)

fig.update_xaxes(range=[110, 145])
fig.update_yaxes(range=[95, 110])

fig.update_layout(
    width=650,
    height=450,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=10, r=10, t=70, b=10),
    title=dict(
        text="Stock Path: In-sample Fit vs Out-of-sample Reality",
        y=0.97
    ),
    showlegend=False
)

fig.show()

```



##### Data Generating Distributions Change Over Time

If we fit a model to a non-stationary distribution (uncertain events rather than random) we will effectively be using *incorrect* data or data from another distribution to inform our model and subsequent new realizations


```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seed for reproducibility
np.random.seed(42)

# Parameters for dynamic (left) side
n_samples = 150_000
n_frames = 60
x_range = np.linspace(-6, 6, 300)

# --- Modified Ornstein–Uhlenbeck process for persistent high variance ---
def ou_process(theta=1.0, mu=3.0, sigma=1.0, dt=0.1, n_steps=n_frames):
    """
    OU process with high mean-reversion level (mu=3) and lower mean reversion speed (theta=1)
    to make variance spike early and remain elevated.
    """
    x = np.zeros(n_steps)
    x[0] = 0.5  # start low, then rise
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx
    return np.maximum(x, 0.1)  # keep variance positive

# Generate variance and mean paths
time_points = np.linspace(0, n_frames/10, n_frames)
variance_path = ou_process()

# --- Modified mean path: stays mostly negative ---
means = np.concatenate([
    np.linspace(3, 0, n_frames//5),    # move quickly to negative
    np.linspace(-2, 0, 3*n_frames//5), # stay negative for most of the time
    np.linspace(-1, 0, n_frames//5)    # end slightly less negative
])

# --- Create Figure with 2-column layout ---
fig = make_subplots(
    rows=3, cols=2,
    column_widths=[0.55, 0.45],  # Left column wider for detail
    row_heights=[0.33, 0.33, 0.34],
    specs=[
        [{"type": "xy"}, {"rowspan": 3, "type": "xy"}],  # Right: dice (static), left: animated
        [{"type": "xy"}, None],
        [{"type": "xy"}, None]
    ],
    subplot_titles=(
        'Latent Variance Process',
        'Uniform Dice Distribution',
        'Unobservable Data Generating Distribution',
        'Observed (Empirical) Stock Returns'
    )
)

frames = []
all_samples = []

# --- Generate frames for dynamic (left) side ---
for i in range(n_frames):
    n_samples_current = int(n_samples * (1 / (1 + np.abs(means[i])**2)))
    X = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current)
    kde_X = stats.gaussian_kde(X)
    samples = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current // 500)
    all_samples.extend(samples)
    
    frames.append(
        go.Frame(
            data=[
                # (1) OU variance path
                go.Scatter(
                    x=time_points[:i+1],
                    y=variance_path[:i+1],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 255, 1)', width=2),
                    name='Variance Path'
                ),
                # (2) Theoretical distribution
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Varying Normal'
                ),
                # (3) Histogram of observed samples
                go.Histogram(
                    x=all_samples,
                    nbinsx=50,
                    name='NVDA Distribution',
                    marker_color='rgba(0, 255, 255, 0.6)'
                ),
                # (4) DICE HISTOGRAM (STATIC, right column, always the same)
                go.Bar(
                    x=np.arange(1, 7),
                    y=np.repeat(1/6, 6),
                    marker_color='rgba(50, 200, 50, 0.8)',
                    width=0.8,
                    name='Dice Distribution',
                )
            ]
        )
    )

# Initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)  # Variance (dynamic, left)
fig.add_trace(frames[0].data[1], row=2, col=1)  # Distribution (dynamic, left)
fig.add_trace(frames[0].data[2], row=3, col=1)  # Histogram (dynamic, left)
fig.add_trace(frames[0].data[3], row=1, col=2)  # Dice Distribution (STATIC, right)

# Add frames to figure (dynamic, left panel only)
fig.frames = frames

# --- Animation Controls ---
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

# --- Layout & Styling ---
fig.update_layout(
    height=600,
    width=1200,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Dynamic Stock Return Distribution (left) vs Fixed Dice Roll Uniform (right)'
)

# --- Axes Formatting ---
for (r, c) in [(1,1), (2,1), (3,1)]:
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=c
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
        row=r, col=c
    )

# Format DICE panel separately (right)
fig.update_xaxes(
    tickmode='array',
    tickvals=np.arange(1, 7),
    range=[0.5, 6.5],
    row=1,
    col=2,
    title="Dice Number"
)
fig.update_yaxes(
    range=[0, 0.3],
    title="Probability",
    row=1,
    col=2
)

# --- Specific Ranges (left column) ---
fig.update_xaxes(range=[0, n_frames/10], row=1, col=1)
fig.update_yaxes(range=[0, 6], row=1, col=1)
fig.update_xaxes(range=[-6, 6], row=2, col=1)
fig.update_yaxes(range=[0, 1], row=2, col=1)
fig.update_xaxes(range=[-6, 6], row=3, col=1)
fig.update_yaxes(range=[0, 1750], row=3, col=1)

fig.show()

```



---

#### 2.) 🎯 Neural Networks

##### Visualization and Mathematics

A neural network is just a composite function, a collection of weights that transform the input into an output

<div align="center">
    <img src="nn.png" alt="bazaar_items" width="700"/>
</div>

 The explicit composite function for a feedforward neural network with 2 hidden layers (3 neurons each) and 1 output neuron, in one line, is:
 $$
 \hat{y}(x) = W^{[3]} \, \sigma\big( W^{[2]} \, \sigma(W^{[1]} x + b^{[1]}) + b^{[2]} \big) + b^{[3]}
 $$

 where $\sigma$ is the activation function applied elementwise, $W^{[l]}$ and $b^{[l]}$ are the weights and biases of each layer.

 The composite function above is *literally* the neural network, however parameters for the model are **weight matrices** and products are **matrix products**

<div align="center">
    <img src="mm.png" alt="bazaar_items" width="700"/>
</div>


 **Remark:** We should see at this point that both basic probability and statistics along with linear algebra is crucial for all of machine learning.  Moreover, there are *a lot* of different structures we can build in this context to tackle different problems:

 - Image Recognition : Convolutional Nueral Networks
 - Image Generation : VAEs / GANs
 - Language Modeling : Transformers

 Not all that finance related but I would be stoked to do a video on these topics if there is interest .  . .

###### ______________________________________________________________________________________________________________________________________

##### Neural Networks as a Universal Function Approximator

Let $f(x)$ be a continuous function on a compact subset $K\subset\mathbb{R}^n$.
 Consider a single-hidden-layer neural network:
 $$
 F(x) = \sum_{i=1}^N \alpha_i \, \sigma(w_i^\top x + b_i)
 $$
 where $\sigma$ is a nonconstant, bounded, and continuous activation (e.g., sigmoid or ReLU).
 
**Theorem (Universal Approximation, Cybenko 1989)**

 For any $\varepsilon > 0$, there exist $N$, $(\alpha_i, w_i, b_i)$ such that
 $$
 \sup_{x\in K} |f(x) - F(x)| < \varepsilon.
 $$
 - The set of sums of the form $\sum_i \alpha_i \sigma(w_i^\top x + b_i)$ is dense in $C(K)$ (the set of continuous functions on $K$).
 - This follows from the Hahn-Banach theorem and properties of $\sigma$.
 
 Thus, neural networks can approximate any continuous function to arbitrary accuracy on compact sets, making them universal function approximators.


```python
import warnings
warnings.filterwarnings("ignore")  # Hide warnings

import numpy as np
import plotly.graph_objs as go

# Synthetic data as before
np.random.seed(42)
n = 100
X = np.linspace(-3, 3, n).reshape(-1, 1)
y_true_f = np.tanh(1.2 * X) + 0.2 * np.sin(2.5 * X)
y = y_true_f + 0.2 * np.random.randn(n, 1)

# Helper for ReLU
def relu(z):
    return np.maximum(0, z)

# Set up a series of piecewise linear (ReLU) approximations, each adding more segments (i.e., more breakpoints/knots)
n_epochs = 20
snapshots = []
for ep in range(1, n_epochs+1):
    # Choose 'ep+2' breakpoints in x for increasing approximation power
    num_knots = min(ep+2, 18)  # Capped to avoid excessive complexity
    knots = np.linspace(-3, 3, num_knots)
    # Fit a continuous, piecewise linear function (sum of shifted ReLUs)
    # Construct basis: phi_i(x) = relu(x - c_i)
    Phi = np.concatenate([relu(X - c) for c in knots], axis=1)
    # Least squares fit to noisy y in initial stages, to clean y_true_f in late stages
    target = y if ep < int(0.75*n_epochs) else y_true_f
    # Add mild noise in convergence phase for "finishing" animation
    if ep > int(0.85*n_epochs):
        target = target + 0.04*np.random.randn(*target.shape)
    # Linear fit
    coefs, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)
    pred = Phi @ coefs
    snapshots.append(pred.flatten())

# --- 
# Keep legend in upper left and ensure it stays throughout animation
# So: showlegend=True always, vertical legend top left

# Build plotly animation frames
frames = []
for ep, pred in enumerate(snapshots):
    frame = go.Frame(
        data=[
            go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers',
                       marker=dict(color='deepskyblue', size=6, opacity=0.42),
                       name='Data', showlegend=True),
            go.Scatter(x=X.flatten(), y=y_true_f.flatten(), mode='lines',
                       line=dict(color='orange', width=4, dash='dash'),
                       name='True Function', showlegend=True),
            go.Scatter(x=X.flatten(), y=pred, mode='lines',
                       line=dict(color='mediumvioletred', width=4),
                       name='Neural Network', showlegend=True)
        ],
        name=f'Epoch {ep+1}'
    )
    frames.append(frame)

# Static traces for epoch 0
trace_data = go.Scatter(
    x=X.flatten(), y=y.flatten(), mode='markers',
    marker=dict(color='deepskyblue', size=6, opacity=0.42), name="Data", showlegend=True
)
trace_true = go.Scatter(
    x=X.flatten(), y=y_true_f.flatten(), mode='lines',
    line=dict(color='orange', width=4, dash='dash'), name="True Function", showlegend=True
)
trace_relu = go.Scatter(
    x=X.flatten(), y=snapshots[0], mode='lines',
    line=dict(color='mediumvioletred', width=4), name="Neural Network", showlegend=True
)

fig = go.Figure(
    data=[trace_data, trace_true, trace_relu],
    frames=frames
)

fig.update_layout(
    width=700,
    height=490,
    title=dict(
        text="ReLU Piecewise Linear Approximation Converging to Conditional Expectation",
        y=1.0  # <-- Valid value for y is in [0, 1]
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=30, r=20, t=100, b=30),  # Increase top margin for spacing
    xaxis=dict(
        title="x",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    yaxis=dict(
        title="y",
        showgrid=True, gridcolor="#323232", gridwidth=1,
    ),
    legend=dict(
        orientation='v',
        xanchor='left',
        x=0.01,
        yanchor='top',
        y=0.99,
        font=dict(color='white'),
        bgcolor='rgba(30,30,30,0.92)'
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        y=1.13,
        x=0.5,
        xanchor="center",
        yanchor="top",
        direction="left",
        buttons=[
            dict(label="Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=110, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate", transition=dict(duration=0))]),
        ])]
)

# Removed slider for progress display

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### Nearly Solved vs Unsolved Problems

In this space we generally consider two classes of problems

- *solved* or *nearly-solved* problems 
- *unsolved* problems

For example, we don't **worry** that our phone can't open with FaceId - sometimes it *doesn't work* but that is intentional for security, the problem is *nearly-solved* it has a significant degree of confidence its you, so much so its safe enough to lock you phone with it along with your passwords and wallet.  The distribution is relatively stable, our face does not change daily producing new samples for the model to predict making it a (relatively) easy modeling problem (even though it is also non-stationary it is far more stable)

In contrast to this, stock returns (as we've seen above) are non-stationary to a different degree, we effectively fit a model to data from an entirely different distribution incorrectly informing our models and parameters with a naiive model training process.


```python
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression

# --- Figure setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Facial Recognition Latent Space", "Stock Path: In-sample Fit vs Out-of-sample Reality"),
    horizontal_spacing=0.08
)

# ===============================
# LEFT PANEL: Latent-space scatter
# ===============================
np.random.seed(10)
n = 120
x_pos = np.random.randn(n//2, 2) * 0.6 + np.array([2, 2])
x_neg = np.random.randn(n//2, 2) * 0.6 + np.array([-2, -2])
X = np.vstack([x_pos, x_neg])
y = np.hstack([np.ones(n//2), np.zeros(n//2)])

clf = LogisticRegression().fit(X, y)
xx, yy = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-4, 4, 120))
zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

contour = go.Contour(
    x=np.linspace(-4, 4, 120),
    y=np.linspace(-4, 4, 120),
    z=zz,
    showscale=False,
    contours=dict(
        start=0, end=0, size=1,
        showlines=True, coloring='none'
    ),
    line=dict(color='white', width=3)
)

pos_scatter = go.Scatter(
    x=x_pos[:, 0], y=x_pos[:, 1],
    mode='markers',
    marker=dict(color='limegreen', size=10, line=dict(width=1, color='white')),
    name='Recognized (Yes)'
)
neg_scatter = go.Scatter(
    x=x_neg[:, 0], y=x_neg[:, 1],
    mode='markers',
    marker=dict(color='crimson', size=10, line=dict(width=1, color='white')),
    name='Not Recognized (No)'
)

fig.add_trace(pos_scatter, row=1, col=1)
fig.add_trace(neg_scatter, row=1, col=1)
fig.add_trace(contour, row=1, col=1)

# ===============================
# RIGHT PANEL: ABM simulation & poor polynomial extrapolation
# ===============================
np.random.seed(42)
T = 200
t = np.arange(T)
# ABM-like price path: drift + volatility + mean-reverting noise
price = np.zeros(T)
price[0] = 100
for i in range(1, T):
    drift = 0.05
    shock = np.random.randn() * 0.6
    mean_revert = 0.03 * (100 - price[i-1])
    price[i] = price[i-1] + drift + shock + mean_revert

# Split into train/test (in-sample vs OOS)
train_cut = 130
t_train, t_test = t[:train_cut], t[train_cut:]
p_train, p_test = price[:train_cut], price[train_cut:]

# Polynomial fit to training
deg = 6
coef = np.polyfit(t_train, p_train, deg)
poly_fit = np.poly1d(coef)

p_pred_train = poly_fit(t_train)
p_pred_test = poly_fit(t_test)

# Add one extra "future" region to show divergence
t_future = np.arange(T, T + 20)
p_pred_future = poly_fit(t_future)

# --- Traces ---
true_line = go.Scatter(
    x=t, y=price, mode='lines+markers',
    line=dict(color='skyblue', width=2),
    name='ABM True Path'
)
fit_train = go.Scatter(
    x=t_train, y=p_pred_train,
    mode='lines',
    line=dict(color='limegreen', width=3),
    name='Polynomial Fit (Train)'
)
fit_test = go.Scatter(
    x=t_test, y=p_pred_test,
    mode='lines',
    line=dict(color='crimson', width=3, dash='dot'),
    name='Polynomial Fit (OOS)'
)
fit_future = go.Scatter(
    x=t_future, y=p_pred_future,
    mode='lines',
    line=dict(color='crimson', width=2, dash='dot'),
    opacity=0.6,
    name='Polynomial Extrapolation (Future)'
)
split_line = go.Scatter(
    x=[train_cut, train_cut],
    y=[min(price)-5, max(price)+5],
    mode='lines',
    line=dict(color='white', width=2, dash='dash'),
    showlegend=False
)

fig.add_trace(true_line, row=1, col=2)
fig.add_trace(fit_train, row=1, col=2)
fig.add_trace(fit_test, row=1, col=2)
fig.add_trace(fit_future, row=1, col=2)
fig.add_trace(split_line, row=1, col=2)

# ===============================
# Styling
# ===============================
for i in range(1, 3):
    fig.update_xaxes(showgrid=True, gridcolor="#323232", gridwidth=1, zeroline=False, row=1, col=i)
    fig.update_yaxes(showgrid=True, gridcolor="#323232", gridwidth=1, zeroline=False, row=1, col=i)

# Limit x and y axes on the right (second) subplot
fig.update_xaxes(range=[110, 145], row=1, col=2)
fig.update_yaxes(range=[95, 110], row=1, col=2)

fig.update_layout(
    width=1150,
    height=450,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=10, r=10, t=70, b=10),
    title=dict(
        text="Model Comparison: Nearly Solved vs Unsolved Problems",
        y=0.97
    ),
    showlegend=False
)

fig.show()

```



**Remark:** Our models are only as good as our economic interpretations and the stability of the underlying distribution.  In the case of facial recognition, that distribution is stable - if we project down into a latent space we can observe a high degree of separability even for out of sample images (where the model trained on a face is being asked "hey, is this that face we trained you on?").  

In contrast, the stock price or stock returns model does not have stability, the underlying data generating distribution constantly changes and we are overfitting effectively to noise (no signal or expectation) during that period.  This is **NOT** how AI is applied in quant trading - I will do a video in the future on this topic. . .

###### ______________________________________________________________________________________________________________________________________

##### Applications in Quant Finance

- Quant Trading Signal Generation (**NOT** directly to the price path or returns)

- Efficient Approximation of Option Pricing Functionals for Calibration and Exotics Pricing

- Experimental Portfolio Risk Measures (VAEs, for example)

###### ______________________________________________________________________________________________________________________________________

##### Example: Approximating the Black-Scholes Pricing Functional

If a neural network is a universal function approximator, and neural networks learn expectation functions, recall the Black-Scholes model (result of replicating portfolio argument) is equivalent to the risk-neutral discounted expectation - we can thus use a neural network to learn the Black-Scholes model

$$ C(S_t, K, T, \sigma, r) = e^{-rT}\mathbb{E}^{\mathbb{Q}} \left[ (S_T - K)^+ \right] = S_t \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

**Procedure:**

- Generate parameters for the Black-Scholes model $\xi = (S_t, K, T, \sigma, r) \in \Xi \subset \mathbb{R}^5$ (our $X_i$'s)

- Produce the Black-Scholes price for $\xi$

- Train a model $\mathcal{M}(\Theta, \xi) \rightarrow \hat{y}_i \approx y_i$ 

**Result:**

- $\mathcal{M}(\Theta^*, \xi)$ can produce a price for the Black-Scholes framework

Not useful in a Black-Scholes framework (analytical form available), but when pricing is slow (beyond a Black-Scholes framework) we can do this all offline and implement the model for efficient pricing and calibration to imnplied volatility surfaces


```python
import numpy as np
import pandas as pd
from scipy.stats import norm

# Black-Scholes Pricing Formula (for European Call)
def black_scholes_call(S, K, T, sigma, r):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan  # Handle degenerate cases safely
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

# Number of samples to generate
N = 1000

# Generate synthetic (random) Black-Scholes parameter sets
np.random.seed(42)
S = np.random.uniform(50, 150, N)        # Spot price between $50 and $150
K = np.random.uniform(50, 150, N)        # Strike price between $50 and $150
T = np.random.uniform(0.1, 2.0, N)       # Maturity between 0.1 and 2 years
sigma = np.random.uniform(0.1, 0.5, N)   # Volatility between 10% and 50%
r = np.random.uniform(0.0, 0.1, N)       # Risk-free rate between 0% and 10%

# Compute Black-Scholes price for each set of parameters
prices = [
    black_scholes_call(S[i], K[i], T[i], sigma[i], r[i]) for i in range(N)
]

# Create DataFrame
df_bs = pd.DataFrame({
    'spot': S,
    'strike': K,
    'maturity': T,
    'volatility': sigma,
    'risk_free_rate': r,
    'call_price': prices
})

# Display first few rows
df_bs.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spot</th>
      <th>strike</th>
      <th>maturity</th>
      <th>volatility</th>
      <th>risk_free_rate</th>
      <th>call_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>87.454012</td>
      <td>68.513293</td>
      <td>0.597241</td>
      <td>0.369081</td>
      <td>0.057200</td>
      <td>23.123900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>145.071431</td>
      <td>104.190095</td>
      <td>0.569260</td>
      <td>0.418673</td>
      <td>0.080543</td>
      <td>47.692614</td>
    </tr>
    <tr>
      <th>2</th>
      <td>123.199394</td>
      <td>137.294584</td>
      <td>1.821884</td>
      <td>0.200187</td>
      <td>0.076016</td>
      <td>14.954783</td>
    </tr>
    <tr>
      <th>3</th>
      <td>109.865848</td>
      <td>123.222489</td>
      <td>0.574138</td>
      <td>0.349950</td>
      <td>0.015390</td>
      <td>7.055293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.601864</td>
      <td>130.656115</td>
      <td>0.616704</td>
      <td>0.328698</td>
      <td>0.014925</td>
      <td>0.031074</td>
    </tr>
  </tbody>
</table>
</div>




```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Prepare input features (X) and target (y)
X = df_bs[['spot', 'strike', 'maturity', 'volatility', 'risk_free_rate']].values
y = df_bs['call_price'].values

# Build a simple feedforward neural network model
model = keras.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output: call price
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
loss, mae = model.evaluate(X, y, verbose=1)
print(f"Train MAE: {mae:.4f}")

# Example: Use the trained model to predict Black-Scholes price
predicted = model.predict(X[:5])
print("Neural Net predictions vs Actual:")
print(np.hstack([predicted, y[:5, None]]))

```

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 362us/step - loss: 10.9231 - mae: 2.3527
    Train MAE: 2.3527
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    Neural Net predictions vs Actual:
    [[ 2.31309376e+01  2.31239000e+01]
     [ 4.40762596e+01  4.76926142e+01]
     [ 1.24728165e+01  1.49547826e+01]
     [ 5.03093624e+00  7.05529301e+00]
     [-1.56799865e+00  3.10742288e-02]]



```python
# Plot the analytical Black-Scholes and neural network prices in the same plot style

import plotly.graph_objs as go

# Define a grid over S, K, T, sigma, r
S_grid = np.linspace(df_bs['spot'].min(), df_bs['spot'].max(), 100)
K_fixed = df_bs['strike'].median()
T_fixed = df_bs['maturity'].median()
sigma_fixed = df_bs['volatility'].median()
r_fixed = df_bs['risk_free_rate'].median()

# Create input grid for S
X_plot = np.column_stack([
    S_grid,
    np.full_like(S_grid, K_fixed),
    np.full_like(S_grid, T_fixed),
    np.full_like(S_grid, sigma_fixed),
    np.full_like(S_grid, r_fixed),
])

# Analytical Black-Scholes values for the line
def black_scholes_call(spot, strike, maturity, volatility, risk_free_rate):
    # vectorized Black-Scholes formula
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    from scipy.stats import norm
    call = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)
    return call

bs_analytical = black_scholes_call(S_grid, K_fixed, T_fixed, sigma_fixed, r_fixed)

# Neural net predictions
nn_pred = model.predict(X_plot).flatten()

# Make plot like above (dark bg, colored lines, etc.)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=S_grid, y=bs_analytical,
    mode='lines',
    line=dict(color='deepskyblue', width=3),
    name='Analytical Black-Scholes',
))
fig.add_trace(go.Scatter(
    x=S_grid, y=nn_pred,
    mode='lines',
    line=dict(color='mediumvioletred', width=3, dash='dash'),
    name='Neural Network Approx',
))
fig.update_layout(
    title="Black-Scholes vs Neural Network Price Functional<br><sup>Strike, T, Vol, r: fixed at median dataset values</sup>",
    xaxis_title="Spot Price (S)",
    yaxis_title="Call Price",
    width=840,
    height=420,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(showgrid=True, gridcolor="#323232", gridwidth=1),
    yaxis=dict(showgrid=True, gridcolor="#323232", gridwidth=1),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.18,
        xanchor='center',
        x=0.5,
        font=dict(color='white')
    )
)
fig.show()

```

    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step 




---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- Data is the oil of the 21st century, models require data to learn *anything*
- All models learn by minimizing the distance between some prediction and some observed target of interest, other features available before the event we are trying to predict are used to help inform the model's prediction
- We can always overfit or underfit a model, we use training/validation/testing splits to ensure our model is learning in a robust manner while we can also emulate out-of-sample performance metrics on the unseen testing set
- Our models are only as effective as the stability of the space, we are effectively learning expectation functions, in spaces that are random (fixed population distribution) or reasonably stable (think like facial recognition) we can train a model once and it can be sufficient for a good while, when population distributions are not stable and exhibit severe non-stationarity like stock prices or returns we can't apply these models here outright
- Neural networks are a popular model to effectively learn these expectations as they are a universal function approximator
- Neural networks are *just functions* that are parameterized by weight matrices, the optimal matrices are learned through the training process
- There are a variety of models in this context from generative models to learn some target, even time variant, n-dimensional distribution, to non-linear dimensionality reduction, to image classification and language modeling
- In quantitative finance we implement these models for a variety of regression and classification tasks, we saw a simple example above learning the Black-Scholes functional but we can apply this methodology to more complicated models to enhance calibration efficiency, manage portfolio risk, and even generate quant trading signals

**Future Topics**

Technical Videos and Other Discussions

- Alternate Structures (VAEs/GANs, CNNs, Transformers, . . .)
- Artificial Intelligence for Quant Trading
- Approximating Pricing Functionals Nonmarkovian/markovian/rough models
- Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
- Non-Markovian Models (fractional Brownian motion, Volterra Process)
- Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
- Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
- Risk-Neutral Measures (Complete vs Incomplete Markets)
- Reinforcement Learning for Delta Hedging
- Approximating Pricing Functionals using Neural Networks

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- Live Neural Network Stochastic Volatility Model Calibration
- Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
- Automated Delta-Neutral Trading System

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
