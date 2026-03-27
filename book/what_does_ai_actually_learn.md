## What Does AI Actually Learn

##### ▶️ Related Quant Guild Videos:

- [Why Monte Carlo Simulation Works](https://youtu.be/-4sf43SLL3A)

- [Analyzing Stock Returns with Principal Component Analysis in Python](https://youtu.be/oKJ5Rb3PI-o)

- [Can AI Learn Black-Scholes?](https://youtu.be/aRr3chiwkrI)

- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [How to Trade](https://youtu.be/NqOj__PaMec)

- [How to Trade with an Edge](https://youtu.be/NlqpDB2BhxE)
 
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




```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate random 3D data points
np.random.seed(42)
n_points = 100

x = np.random.normal(0, 1, n_points)
y = np.random.normal(0, 1, n_points)
z = 0.3*x - 0.2*y + np.random.normal(0, 0.5, n_points)

# Create figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('Raw Data Points', 'Points Colored by Return with Trading Signal')
)

# Add traces for first subplot - all points in blue
fig.add_trace(
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.7
        ),
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for second subplot - points colored by return
fig.add_trace(
    go.Scatter3d(
        x=x[z >= 0], y=y[z >= 0], z=z[z >= 0],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
            opacity=0.7
        ),
        name='Positive Return'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=x[z < 0], y=y[z < 0], z=z[z < 0],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.7
        ),
        name='Negative Return'
    ),
    row=1, col=2
)

# Add hyperplane
xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
# z = 0 plane represents the decision boundary
zz = np.zeros_like(xx)

fig.add_trace(
    go.Surface(
        x=xx,
        y=yy,
        z=zz,
        opacity=0.3,
        showscale=False,
        name='Trading Signal',
        colorscale=[[0, 'gray'], [1, 'gray']]
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='3D Feature Space: Raw vs Return-Colored Points with Trading Signal',
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Return',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    scene2=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Return',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    width=900,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True
)

fig.show()

```



### Sections

##### 1.) 🎲 Outcomes as Random Variables 

##### 2.) 🎯 Predictions as Expectations

##### 3.) 🦾 What AI Actually Learns

##### 4.) 💻 Stock Price Prediction v. Facial Recognition

##### 5.) 💭 Closing Thoughts and Future Topics

---

### 1.) 🎲 Outcomes as Random Variables

Random variables are mathematical objects used to model randomness

We can say that $Y$ is the outcome of a dice roll, and $X$ is the average value of a trading signal

In each of these cases, both $Y$ and $X$ are not *one outcome* in the sense of classical variables like $y$ and $x$ that represent numbers, instead these variables ($Y$, $X$) represent *distributions* - a set of possible outcomes according to a certain likelihood that is either predefined or observed.


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2, subplot_titles=("Random Variable Y - Dice Roll", "Random Variable X - Trading Signal"))

# Left plot - Theoretical dice roll distribution (uniform probability)
dice_values = np.arange(1, 7)
probabilities = np.ones(6) / 6  # Equal probability of 1/6 for each outcome

fig.add_trace(
    go.Bar(x=dice_values, y=probabilities,
           name='Dice Roll Distribution',
           marker_color='#00FF00'),  # Neon green
    row=1, col=1
)

# Right plot - Normal distribution
x = np.linspace(-6, 2, 1000)
y = 1/(2*np.pi)**0.5 * np.exp(-0.5*(x - (-2))**2)

fig.add_trace(
    go.Scatter(x=x, y=y,
               name='Normal Distribution',
               line=dict(color='#FF00FF', width=2)),  # Neon pink
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=600,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Random Variable Examples',
    title_x=0.5,
    title_font_size=20
)

# Update axes
for i in [1, 2]:
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     row=1, col=i)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                     row=1, col=i)

# Update subplot titles
fig.update_xaxes(title_text="Dice Value", row=1, col=1)
fig.update_xaxes(title_text="X", row=1, col=2)
fig.update_yaxes(title_text="Probability", row=1, col=1)
fig.update_yaxes(title_text="Density", row=1, col=2)

fig.show()

```



##### <u>Two Important Distinctions:</u>

- We are modeling something *actually* random (for example, a coin flip, dice roll, roulette,)

- We are modeling something deterministic as random (for example, the number of chips in a bag)

In the first case, we can't predict the outcome - in the second, if we had sufficient information (we were at the factory and SAW the chips go into that specifc bag) it wouldn't be a random variable and we would know the count.

More often than not, we are interacting with the second method - there are too many observable or unobservable factors that go into producing a quantity or outcome so we can model it as random.

##### <u>Empirical v. Theoretical Distributions</u>

In machine learning and artificial intelligence, we rarely have the theoretical distribution.  That is, when we have random variables say $Y$ a dice roll, $X$ a trading signal, and $Z$ the number of chips in a bag - what governs the randomness of $Y$ is clear, but not $X$ (inflation, interest rates, etc.) and $Z$ (factory scales, factory dispensor, etc.).

So typically we use the empirical distribution, which is just a histogram of observed outcomes.


```python
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(42)

# Create subplots
fig = sp.make_subplots(rows=2, cols=2, 
                       subplot_titles=('5 Rolls', '1000 Rolls', '10000 Rolls', 'Theoretical'),
                       vertical_spacing=0.12,
                       horizontal_spacing=0.1)

# Theoretical probability (uniform distribution)
theoretical = np.ones(6) / 6
x_vals = np.arange(1, 7)

# Generate different numbers of rolls
rolls = [5, 1000, 10000]
positions = [(1,1), (1,2), (2,1)]

for rolls_count, pos in zip(rolls, positions):
    # Generate random rolls
    empirical = np.random.randint(1, 7, size=rolls_count)
    
    # Calculate empirical probabilities
    counts = np.bincount(empirical)[1:]
    empirical_prob = counts / rolls_count
    
    # Add bars for empirical distribution
    fig.add_trace(
        go.Bar(x=x_vals, y=empirical_prob, name=f'Empirical ({rolls_count} rolls)',
               marker_color='rgba(58, 71, 80, 0.6)'),
        row=pos[0], col=pos[1]
    )
    
    # Add line for theoretical distribution
    fig.add_trace(
        go.Scatter(x=x_vals, y=theoretical, name='Theoretical',
                  line=dict(color='rgba(255, 0, 0, 0.8)', width=2)),
        row=pos[0], col=pos[1]
    )

# Add theoretical distribution plot
fig.add_trace(
    go.Bar(x=x_vals, y=theoretical, name='Theoretical',
           marker_color='rgba(255, 0, 0, 0.8)'),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=700,
    showlegend=False,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Empirical vs Theoretical Dice Roll Distributions',
    title_x=0.5,
    title_font_size=20
)

# Update axes
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text='Dice Value', row=i, col=j,
                        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
        fig.update_yaxes(title_text='Probability', row=i, col=j,
                        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.show()

```



**Remark:** Though here we can see the distributions will converge, in reality this will only be the case under specific circumstances where there is stable well defined structure that is relatively time invariant.  More on this below...

---

### 2.) 🎯 Predictions as Expectations

When we are dealing with *actual* randomness, the best we can do is the expectation.  There is no machine learning model or artificial intelligence model on this planet that can predict the outcome of my dice that I roll.  

If you want to get theoretical and argue that its not *true* randomness, substitute this for JPM's *true* quantum randomness.

##### <u>Predicting a Dice Roll using the Expectation and Artificial Intelligence</u>


```python
import torch
import torch.nn as nn
import numpy as np

# Generate data
X = torch.ones((10000, 1))  # Input: vector of 1s
y = torch.tensor(np.random.randint(1, 7, size=10000)).float()  # Random dice rolls

# Define simple neural network
class DicePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize model, loss function, and optimizer
model = DicePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y.unsqueeze(1))
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions
with torch.no_grad():
    test_input = torch.ones((1000, 1))
    predictions = model(test_input)
    
print("\nAverage prediction:", predictions.mean().item())
print("Expected value of a fair die:", 3.5)

```

    Epoch [100/1000], Loss: 3.3479
    Epoch [200/1000], Loss: 2.9152
    Epoch [300/1000], Loss: 2.9152
    Epoch [400/1000], Loss: 2.9152
    Epoch [500/1000], Loss: 2.9152
    Epoch [600/1000], Loss: 2.9152
    Epoch [700/1000], Loss: 2.9152
    Epoch [800/1000], Loss: 2.9152
    Epoch [900/1000], Loss: 2.9152
    Epoch [1000/1000], Loss: 2.9152
    
    Average prediction: 3.5019001960754395
    Expected value of a fair die: 3.5



```python
# Generate 100,000 dice rolls
rolls = np.random.randint(1, 7, size=100000)

# Calculate MSE for constant prediction of 3.5
mse_constant = np.mean((rolls - 3.5) ** 2)

# Get neural network predictions for 100,000 inputs
with torch.no_grad():
    test_input = torch.ones((100000, 1))
    nn_predictions = model(test_input).numpy().flatten()
    mse_nn = np.mean((rolls - nn_predictions) ** 2)

# Create figure
fig = go.Figure()

# Add bars for MSE comparison
fig.add_trace(
    go.Bar(
        x=['Expected Value (3.5)', 'Neural Network'],
        y=[mse_constant, mse_nn],
        marker_color=['rgba(58, 71, 80, 0.6)', 'rgba(58, 71, 80, 0.6)']
    )
)

# Update layout
fig.update_layout(
    height=400,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Mean Squared Error Comparison',
    title_x=0.5,
    title_font_size=20,
    xaxis_title='Prediction Method',
    yaxis_title='Mean Squared Error',
    yaxis=dict(
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        showgrid=True
    )
)

fig.show()
print(f"MSE Constant: {mse_constant:.6f}")
print(f"MSE Neural Network: {mse_nn:.6f}")
print(f"MSE Difference: {mse_nn - mse_constant:.6f}")
```



    MSE Constant: 2.922240
    MSE Neural Network: 2.922230
    MSE Difference: -0.000010


---

### 3.) 🦾 What AI Actually Learns

Any machine learning or artificial intelligence algorithm is learning the expectation, more specifically its learning a non-linear conditional expectation. 

There is no such thing as a prediction, its producing an expectation level for decision making.  

Most thinks without an infinite number of inputs are not *one-to-one*, think about age and height - is everyone who is the same age the same height? No, but if we condition on enough data we can model the differences.  Too much conditioning and we miss the plot and overfit - not enough and we are just left with a terrible level or (*sigh* prediction).


```python
import numpy as np
import plotly.graph_objects as go

# Create sample data
np.random.seed(42)
ages = [25, 25, 30, 35, 40, 45, 50]  # Two people are age 25
heights = [68, 72, 70, 69, 71, 68, 67]  # Different heights for age 25

# Create scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=ages,
    y=heights,
    mode='markers',
    marker=dict(
        size=12,
        color='#00b4ff',
        opacity=0.6
    ),
    name='Age-Height Data'
))

# Highlight the two points at age 25
fig.add_trace(go.Scatter(
    x=[25, 25],
    y=[68, 72],
    mode='markers',
    marker=dict(
        size=15,
        color='#ff4b00',
        opacity=0.8,
        line=dict(color='white', width=2)
    ),
    name='Same Age, Different Heights'
))

# Update layout
fig.update_layout(
    height=400,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Age vs Height Example',
    title_x=0.5,
    title_font_size=20,
    xaxis_title='Age (years)',
    yaxis_title='Height (inches)',
    yaxis=dict(
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        showgrid=True
    )
)

fig.show()

```



**Market-Making Example:**

Let $X$ be a dice roll with $\mathbb{E}[X] = 3.5$

Let $u \sim U(0, 1)$

**Bid:** $\mathbb{E}[X] - .5 + u$

**Ask:** $\mathbb{E}[X] + .5 + u$


```python
import numpy as np
import plotly.graph_objects as go

# Generate sample data
np.random.seed(42)
n_samples = 10000
E_X = 3.5
u = np.random.uniform(0, 1, n_samples)

# Calculate bid, ask and mid prices
bids = E_X - 0.5 + u
asks = E_X + 0.5 + u
mids = (bids + asks) / 2

# Create histogram
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=mids,
    nbinsx=50,
    name='Mid Price Distribution',
    marker_color='#00b4ff'
))

# Update layout
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Distribution of Mid Prices',
    title_x=0.5,
    title_font_size=20,
    xaxis_title='Mid Price',
    yaxis_title='Frequency',
    showlegend=False,
    yaxis=dict(
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        showgrid=True
    )
)

fig.show()

```



**Goal:** We want to figure out what the midpoint price is to quote a more competitive spread.  

Let's observe 10,000 quotes then come up with a mean level using the expectation and a neural network.

$\mathbb{E}[Mid] = \frac{\mathbb{E}[Bid + Ask]}{2} = \frac{2\mathbb{E}[X] + 2\mathbb{E}[u]}{2} = 4$


```python
import numpy as np
import torch
import torch.nn as nn

# Generate the data
np.random.seed(42)
n_samples = 10000

# Expected value of dice roll is 3.5
E_X = 3.5

# Generate uniform random numbers for spread adjustment
u = np.random.uniform(0, 1, n_samples)

# Calculate bid and ask
bids = E_X - 0.5 + u
asks = E_X + 0.5 + u

# Calculate empirical midpoints
mids = (bids + asks) / 2

# Create input tensor (vector of 1s)
X = torch.ones((n_samples, 1))
y = torch.tensor(mids, dtype=torch.float32).reshape(-1, 1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
n_epochs = 1000
for epoch in range(n_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')

print(f"Neural network prediction: {model(torch.ones(1)).item():.6f}")
```

    Epoch [100/1000], Loss: 0.083348
    Epoch [200/1000], Loss: 0.082723
    Epoch [300/1000], Loss: 0.082723
    Epoch [400/1000], Loss: 0.082723
    Epoch [500/1000], Loss: 0.082723
    Epoch [600/1000], Loss: 0.082723
    Epoch [700/1000], Loss: 0.082723
    Epoch [800/1000], Loss: 0.082723
    Epoch [900/1000], Loss: 0.082723
    Epoch [1000/1000], Loss: 0.082723
    Neural network prediction: 3.994159


##### <u>Probabilities as Expectations</u>

Bernoulli random variables can be used as indicators for if a desired event occurs or not, if we take the average we will get the probability of that event occuring...

**Example:** We roll 10,000 dice, what is the probability of rolling an even number?

We can record a 1 everywhere we see an even number and 0 everytime we see an odd number.  By the Law of Large Numbers (LLN) this will converge to the true probability of that event occuring. 

$P(Even) = \frac{3}{6} = .5$


```python
# Create subplots
fig = sp.make_subplots(rows=2, cols=2, 
                       subplot_titles=('5 Rolls', '1000 Rolls', '10000 Rolls', 'Theoretical'),
                       vertical_spacing=0.12,
                       horizontal_spacing=0.1)

# Theoretical probability (0.5 for even)
theoretical = np.array([0.5, 0.5])
x_vals = ['Odd', 'Even']

# Generate different numbers of rolls
rolls = [5, 1000, 10000]
positions = [(1,1), (1,2), (2,1)]

for rolls_count, pos in zip(rolls, positions):
    # Generate random rolls
    dice_rolls = np.random.randint(1, 7, size=rolls_count)
    
    # Calculate empirical probabilities
    even_count = np.sum(dice_rolls % 2 == 0)
    empirical_prob = np.array([1 - even_count/rolls_count, even_count/rolls_count])
    
    # Add bars for empirical distribution
    fig.add_trace(
        go.Bar(x=x_vals, y=empirical_prob, name=f'Empirical ({rolls_count} rolls)',
               marker_color='rgba(58, 71, 80, 0.6)'),
        row=pos[0], col=pos[1]
    )
    
    # Add line for theoretical distribution
    fig.add_trace(
        go.Scatter(x=x_vals, y=theoretical, name='Theoretical',
                  line=dict(color='rgba(255, 0, 0, 0.8)', width=2)),
        row=pos[0], col=pos[1]
    )

# Add theoretical distribution plot
fig.add_trace(
    go.Bar(x=x_vals, y=theoretical, name='Theoretical',
           marker_color='rgba(255, 0, 0, 0.8)'),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=700,
    showlegend=False,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title_text='Empirical vs Theoretical Probability of Even Dice Rolls',
    title_x=0.5,
    title_font_size=20
)

# Update axes
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text='Outcome', row=i, col=j,
                        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')
        fig.update_yaxes(title_text='Probability', row=i, col=j,
                        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                        range=[0, 1])

fig.show()

```



##### <u>Artificial Intelligence Learns Probabilities this Way</u>

Let's generate 10,000 dice rolls, record 1 when we see a dice roll that's even and 0 when we see one that's not.


```python
# Generate 10,000 dice rolls
n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)

# Create indicator variable (1 for even, 0 for odd)
even_indicators = (rolls % 2 == 0).astype(int)

# Define simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Initialize model, loss and optimizer
model = SimpleNet()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Convert data to tensor
X = torch.ones((n_rolls, 1))
y = torch.FloatTensor(even_indicators).reshape(-1, 1)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')

# Make prediction
prediction = model(torch.ones(1))
print(f"\nNeural network prediction (probability of rolling even): {prediction.item():.4f}")
print(f"True probability: 0.5000")
```

    Epoch [200/1000], Loss: 0.734251
    Epoch [400/1000], Loss: 0.698658
    Epoch [600/1000], Loss: 0.693714
    Epoch [800/1000], Loss: 0.693047
    Epoch [1000/1000], Loss: 0.692957
    
    Neural network prediction (probability of rolling even): 0.4925
    True probability: 0.5000


**Remark:** The advanced artificial intelligence model can't be a simple average - still think statistics isn't important?

---

### 4.) 💻 Stock Price Prediction v. Facial Recognition

There are fundamentally two problems we can consider in this space

##### <u>Two Types of Problems</u>

- Regression (*predicting* a level or producing an expectation)

- Classification (*predicting* a level or producing an expected probability)

In any case, the ability to produce an extremely effective level (*sigh*, prediction) is to have stability and separability in an N-dimensional space. 

##### Trading Signal Example


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate random 3D data points
np.random.seed(42)
n_points = 100

x = np.random.normal(0, 1, n_points)
y = np.random.normal(0, 1, n_points)
z = 0.3*x - 0.2*y + np.random.normal(0, 0.5, n_points)

# Create figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('Raw Data Points', 'Points Colored by Return with Trading Signal')
)

# Add traces for first subplot - all points in blue
fig.add_trace(
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.7
        ),
        showlegend=False
    ),
    row=1, col=1
)

# Add traces for second subplot - points colored by return
fig.add_trace(
    go.Scatter3d(
        x=x[z >= 0], y=y[z >= 0], z=z[z >= 0],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
            opacity=0.7
        ),
        name='Positive Return'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=x[z < 0], y=y[z < 0], z=z[z < 0],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.7
        ),
        name='Negative Return'
    ),
    row=1, col=2
)

# Add hyperplane
xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
# z = 0 plane represents the decision boundary
zz = np.zeros_like(xx)

fig.add_trace(
    go.Surface(
        x=xx,
        y=yy,
        z=zz,
        opacity=0.3,
        showscale=False,
        name='Trading Signal',
        colorscale=[[0, 'gray'], [1, 'gray']]
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='3D Feature Space: Raw vs Return-Colored Points with Trading Signal',
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Return',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    scene2=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Return',
        xaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        yaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        zaxis=dict(gridcolor='darkgray', showgrid=True, color='darkgray', backgroundcolor='rgb(30, 30, 35)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    width=900,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True
)

fig.show()

```



---

### 5.) 💭 Closing Thoughts and Future Topics

**TL;DW:** If the expectation is not a good proxy due to instability and lack of structure (especially w.r.t time) of the population distribution, a neural network approximation won't be any better.

AI is far from a cure-all and modeling certain events as simple random variables can easily and consistently outpreform.

More complicated models will try to learn a distribution, moreover, with sufficient data where the outcome of interest is heavily dependent on a number of factors it is possible to produce levels that outpreform expectations.

Some problems demand artificial intelligence, the efficacy of this approach will depend on the stability and separability of the problem space in latent dimensions in the ability to produce *good* levels.

**What AI is Amazing At:**

- Stable problem spaces with time invariant distributions (Language Modeling, Face ID, ...)

**What AI is NOT Amazing At:**

- Providing *predictions* or levels in a random environment (either *actually* random or where there is insufficient data to provide reasonable levels)

Future Topics:

- Goodness of Fit Tests (are these valid? can we use these to find an edge?)

- Fundamentals of Neural Networks (Math, Structures, Problems, ...)

- Analyzing Latent Spaces (PCA, Autoencoders, VAEs, ...)

- Machine Learning and Artificial Intelligence in Finance (Signals, Pricing, ...)

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
