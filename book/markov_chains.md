### ⛓️ Markov Chains for Quant Finance

##### ▶️ Related Quant Guild Videos:

- [Expected Stock Returns Don't Exist](https://youtu.be/iXNSBn5xqrA)

- [What Does AI Actually Learn](https://youtu.be/tX7b2KT63WQ)

- [Why Portfolio Optimization Doesn't Work](https://youtu.be/eZIITtd3UfY)

- [How to Trade Option Implied Volatility](https://youtu.be/kQPCTXxdptQ)

- [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)

- [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)

- [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)

###### ______________________________________________________________________________________________________________________________________

##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)

##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)

##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)

---

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


#### 1.) 🎲 Random Variables

- Modeling Randomness

- Stochastic Processes and Violated Assumptions

#### 2.) ⛓️ Markov Chains

- Definition, Transition Matrix, Markov Property

- Probabilities and Expectations

- Key Properties

#### 3.) 📈 Applying Markov Chains

- Defining States and Estimating Probabilities

- Maximum Likelihood Estimation (MLE)

#### 4.) 💭 Closing Thoughts and Future Topics

---

#### 1.) 🎲 Random Variables

##### <u>Modeling Randomness</u>

When the outcome of an event is uncertain we need to consider the various possible states of the world

Random variables are a way to mathematically define randomness based on the likelihood of outcomes

These variables are denoted by letters and associated distributions that govern the probability of each state of the world

$$X \sim N(\mu, \sigma) \quad Y \sim Bin(n, p) \quad Z \sim Ber(p)$$

These variables $X, Y, Z$ are **NOT** variables in a classical sense - they literally represent a distribution of possible outcomes


```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample data
np.random.seed(42)
n_samples = 10000

# Normal distribution
X = np.random.normal(0, 1, n_samples)
x_range = np.linspace(-4, 4, 200)
kde_X = stats.gaussian_kde(X)

# Binomial distribution
n, p = 20, 0.3
Y = np.random.binomial(n, p, n_samples)
y_range = np.arange(0, n+1)
pmf_Y = stats.binom.pmf(y_range, n, p)

# Bernoulli distribution
p = 0.7
Z = np.random.binomial(1, p, n_samples)
z_range = np.array([0, 1])
pmf_Z = np.array([1-p, p])

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('X ~ N(0,1)', f'Y ~ Bin({n},{p})', f'Z ~ Ber({p})')
)

# Add traces with rotating neon colors
colors = ['rgba(255, 0, 255, 1)', 'rgba(0, 255, 255, 1)', 'rgba(0, 255, 0, 1)']

fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_X(x_range),
        mode='lines',
        line=dict(color=colors[0], width=2),
        name='Normal'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=y_range,
        y=pmf_Y,
        marker_color=colors[1],
        name='Binomial'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Bar(
        x=z_range,
        y=pmf_Z,
        marker_color=colors[2],
        name='Bernoulli'
    ),
    row=1, col=3
)

# Update layout
fig.update_layout(
    height=400,
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 4):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
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



These distributions *literally* are the random variables $X, Y, Z$ which is why we don't have just one value for them - they are distributions

The mass (amount of space under the curve, or the height of the bar) tells us the likelihood of an outcome *near* or *at* that value (continous vs. discrete)

When we model events with these distributions we typically assume independent and identically distributed draws (i.i.d)

In other words, we are drawing from the same distribution (same random variable) every time and seeing different outcomes or different states of the world governed by the defined randomness

**This is very rarely true in practice, especially in finance** and violating this assumption has a variety of consequences from underestimating risk to violently wrong forecasts. . .


```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample data
np.random.seed(42)
n_samples = 10000

# Create time-varying normal distributions
n_frames = 60
means = np.concatenate([
    np.linspace(0, 2, n_frames//3),  # Drift positive
    np.linspace(2, -2, n_frames//3), # Drift negative 
    np.linspace(-2, 0, n_frames//3)  # Return to neutral
])

x_range = np.linspace(-4, 4, 200)
frames = []
all_samples = []  # Keep track of all samples

for mean in means:
    X = np.random.normal(mean, 1, n_samples)
    kde_X = stats.gaussian_kde(X)
    
    # Generate 10 samples from current distribution
    samples = np.random.normal(mean, 1, 10)
    all_samples.extend(samples)  # Add new samples to running list
    
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Drifting Normal'
                ),
                go.Histogram(
                    x=all_samples,  # Use all accumulated samples
                    nbinsx=20,
                    name='Sample Distribution',
                    marker_color='rgba(0, 255, 255, 0.6)'
                )
            ]
        )
    )

# Create base figure with subplots
fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])

# Add initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)
fig.add_trace(frames[0].data[1], row=2, col=1)

# Add frames
fig.frames = frames

# Add animation buttons
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
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Update layout
fig.update_layout(
    height=800,  # Increased height to accommodate both plots
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Drifting Return Distribution with Accumulated Samples'
)

# Update axes for both subplots
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-4, 4],
    row=1, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

# Update axes for bottom subplot with increased ylim
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-4, 4],
    row=2, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[0, 100],  # Increased ylim for bottom distribution
    row=2, col=1
)

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### <u>Stochastic Processes and Violated Assumptions</u>

##### A Dice Roll Stock as a Stochastic Process (Simplified Example)

We now consider a *stochastic process* which is a series of random variable outcomes indexed by *time* or a *step*

Suppose we had a stock portfolio whose value was dictated by a dice roll, the value of our portfolio every day is simply the outcome of a dice roll

The randomness is well defined, time invariant, and independent meaning no matter what information I give you it won't impact the randomness of the outcome

In other words, the path draws i.i.d from a dice roll outcome and it perfectly models the portfolio value as a function of this randomness


```python
# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2, subplot_titles=('Dice Roll PMF', 'Sequence of 100 Dice Rolls'))

# Generate PMF data for fair dice
dice_outcomes = np.arange(1, 7)
dice_pmf = np.ones(6) / 6

# Generate sequence of 100 dice rolls
np.random.seed(42)
dice_sequence = np.random.randint(1, 7, size=100)

# Plot PMF
fig.add_trace(
    go.Bar(
        x=dice_outcomes,
        y=dice_pmf,
        name='Probability',
        marker_color='rgba(255, 0, 255, .6)'  # Changed to neon purple
    ),
    row=1, col=1
)

# Plot sequence
fig.add_trace(
    go.Scatter(
        x=np.arange(100),
        y=dice_sequence,
        mode='lines+markers',
        name='Dice Rolls',
        line=dict(color='rgba(255, 0, 255, 1)', width=2),  # Changed to neon purple
        marker=dict(size=4)
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=400,
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in range(1, 3):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

# Update specific axis labels
fig.update_xaxes(title_text="Outcome", row=1, col=1)
fig.update_xaxes(title_text="Roll Number", row=1, col=2)
fig.update_yaxes(title_text="Probability", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)

fig.show()

```



###### ______________________________________________________________________________________________________________________________________

##### <u>A Portfolio of Loans (Real World Example)</u>

Suppose instead now we had a portfolio of loans and wanted to model deliquencies (missed payments)

**Each loan can be one of either**

- Current
- 30-59 days late
- 60-89 days late
- 90+ days late

We are effectively measuring (based on the status of each loan) the overall proportion of loans in default in this portfolio


```python
# Create figure for initial loan portfolio state
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]])

# Initial portfolio state data
labels = ['Current', '30-59 Days', '60-89 Days', '90+ Days']
values = [50, 20, 20, 10]  # Percentages

# Add pie chart
fig.add_trace(
    go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=['rgba(0, 255, 0, 0.8)', 
                   'rgba(255, 255, 0, 0.8)',
                   'rgba(255, 165, 0, 0.8)', 
                   'rgba(255, 0, 0, 0.8)']
        ),
        textinfo='label+percent',
        hoverinfo='label+percent'
    ),
    row=1, col=1
)

# Update layout
fig.update_layout(
    height=400,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation="h",
        yanchor="bottom", 
        y=1.15,
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text="Example Loan Portfolio Delinquency State",
        y=0.95
    )
)

fig.show()

```



**We need to model this thorugh time!  What if we take the approach of the dice roll above?**

###### ______________________________________________________________________________________________________________________________________

We want to measure the overall proportion of loans in our portfolio that are deliquent in any of those states

One way to do this is use data on historic deliquencies and their proportions to simulate outcomes in a forward-looking sense
 
 $$\hat{p} = \frac{\text{number of delinquent loans}}{\text{total number of loans}} = \frac{\sum_{i=1}^n X_i}{n}$$


This is mathematically justified in terms of Maximum Likelihood Estimation (MLE) with the accompanying assumptions. . .

*What's the problem* with assuming these are independent?  Well we can't just go from no loans deliquent to 90+ days deliquent!

This is one example of how if we wanted to model the randomness of our loan portfolio the assumption of independence doesn't work!


```python
# Number of time steps and loans to simulate
n_steps = 3
n_loans = 100

# Create figure for loan delinquency simulation
fig = make_subplots(rows=1, cols=n_steps, subplot_titles=[f'Month {t}' for t in range(n_steps)])

# Initialize arrays to store proportions
np.random.seed(50)
current = np.zeros(n_steps)
late_30_59 = np.zeros(n_steps)
late_60_89 = np.zeros(n_steps)
late_90_plus = np.zeros(n_steps)

# Start with all loans current
current[0] = 1.0

# Simulate independent transitions
for t in range(1, n_steps):
    # At each step, independently determine state of loans
    random_states = np.random.choice(
        ['current', '30-59', '60-89', '90+'],
        size=n_loans,
        p=[0.85, 0.05, 0.05, 0.05]  # Probabilities for each state
    )
    
    # Calculate proportions
    current[t] = np.sum(random_states == 'current') / n_loans
    late_30_59[t] = np.sum(random_states == '30-59') / n_loans
    late_60_89[t] = np.sum(random_states == '60-89') / n_loans
    late_90_plus[t] = np.sum(random_states == '90+') / n_loans

# Plot bars for each time step separately
for t in range(n_steps):
    fig.add_trace(
        go.Bar(
            x=['Current'],
            y=[current[t]],
            name='Current',
            marker_color='rgba(0, 255, 0, 0.8)',
            showlegend=True if t==0 else False
        ),
        row=1, col=t+1
    )

    fig.add_trace(
        go.Bar(
            x=['30-59 Days'],
            y=[late_30_59[t]],
            name='30-59 Days Late',
            marker_color='rgba(255, 255, 0, 0.8)',
            showlegend=True if t==0 else False
        ),
        row=1, col=t+1
    )

    fig.add_trace(
        go.Bar(
            x=['60-89 Days'],
            y=[late_60_89[t]],
            name='60-89 Days Late',
            marker_color='rgba(255, 165, 0, 0.8)',
            showlegend=True if t==0 else False
        ),
        row=1, col=t+1
    )

    fig.add_trace(
        go.Bar(
            x=['90+ Days'],
            y=[late_90_plus[t]],
            name='90+ Days Late',
            marker_color='rgba(255, 0, 0, 0.8)',
            showlegend=True if t==0 else False
        ),
        row=1, col=t+1
    )

# Update layout
fig.update_layout(
    height=400,
    width=1200,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.15,  # Increased from 1.1 to 1.15 to add more padding below legend
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text="Independent Model Shows Impossible Transitions",
        y=0.95  # Add more space between title and plot
    ),
    barmode='group'  # Changed from 'stack' to 'group' for side-by-side bars
)

# Update axes
fig.update_xaxes(
    title_text="Delinquency Status",
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text="Proportion of Loans",
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()

```



Clearly, the naive independence assumption does not match reality

We can't jump from a portfolio of **all current loans** to a portfolio with some proportion of loans that are **60-89 days deliquent**

Why don't we just restrict the model to the case where we *can't* *transition* to **60-89 days deliquent** unless we already have some that are **30-59 days deliquent**?  

This is effectively what Markov Chains are doing but implicitly in their structure! 

---

#### 2.) ⛓️ Markov Chains

A Markov Chain is a stochastic model that describes a sequence of possible events where the probability of each event depends only on the state in the previous event.
 
 Key Properties:
 * **Memoryless**: $P(X_{t+1}|X_t,X_{t-1},...,X_1) = P(X_{t+1}|X_t)$
 * **Time-homogeneous**: Transition probabilities are constant over time
 * **Finite state space**: System can only be in a finite number of states
 
 Transition Matrix $P$:
 * $P_{ij} = P(X_{t+1}=j|X_t=i)$ represents probability of moving from state $i$ to state $j$
 * Each row sums to 1: $\sum_j P_{ij} = 1$ 
 * All entries non-negative: $P_{ij} \geq 0$

  For example, if $X_t$ represents the state at time $t$:
  
  $$P(X_{t+1} = \text{60-89} | X_t = \text{30-59}, X_{t-1} = \text{Current}) = P(X_{t+1} = \text{60-89} | X_t = \text{30-59})$$
  
  The Markov Property implies this idea of memorylessness, the next state only depends on the current state - a tremendous compramise of a simplifying assumption!  We are essentially reducing long-range dependence to a local conditional dependence
  
  In reality, long range dependencies can exist, this does not account for *all* transition variation and can lead to incorrect estimates or models - but it dramatically improves on the naive methodology above!

  ###### ______________________________________________________________________________________________________________________________________

##### <u>Example: Modeling a Portfolio of Loans: Diagram and Interpretation</u>

# ![Markov Chains 1](chain_11.png)

  Transition Matrix $P$ for Loan Delinquency States:
  
$$
P = \begin{array}{c|cccc}
& \text{Current} & \text{30-59} & \text{60-89} & \text{90+} \\
\hline
\text{Current} & P_{11} & P_{12} & 0 & 0 \\
\text{30-59} & P_{21} & P_{22} & P_{23} & 0 \\
\text{60-89} & P_{31} & 0 & P_{33} & P_{34} \\
\text{90+} & P_{41} & 0 & 0 & P_{44}
\end{array}
$$
  
  where states are:
  1. Current
  2. 30-59 Days Delinquent  
  3. 60-89 Days Delinquent
  4. 90+ Days Delinquent
  
  For example, $P_{12}$ represents the probability of transitioning from "Current" to "30-59 Days Delinquent"


We can add probabilities to the diagram too and fill in our transition matrix!  

These probabilities tell us the likelihood of a transition from one portfolio state to another

# ![Markov Chains 1](chain_22.png)

No arrows to another state imply a zero probability of transition.  

The following transition matrix comes from the above diagram of probabilities

  Transition Matrix $P$ for Loan Delinquency States:
  
$$
P = \begin{array}{c|cccc}
& \text{Current} & \text{30-59} & \text{60-89} & \text{90+} \\
\hline
\text{Current} & .99 & .01 & 0 & 0 \\
\text{30-59} & .25 & .70 & .05 & 0 \\
\text{60-89} & .15 & 0 & .80 & .05 \\
\text{90+} & .01 & 0 & 0 & .99
\end{array}
$$
  
  where states are:
  1. Current
  2. 30-59 Days Delinquent  
  3. 60-89 Days Delinquent
  4. 90+ Days Delinquent
  
  For example, $P_{12}$ represents the probability of transitioning from "Current" to "30-59 Days Delinquent"


We can add probabilities to the diagram too and fill in our transition matrix!  

These probabilities tell us the likelihood of a transition from one portfolio state to another


```python
import numpy as np

P = np.array([
    [0.99, 0.01, 0.00, 0.00],
    [0.25, 0.70, 0.05, 0.00],
    [0.15, 0.00, 0.80, 0.05],
    [0.01, 0.00, 0.00, 0.99]
])

P
```




    array([[0.99, 0.01, 0.  , 0.  ],
           [0.25, 0.7 , 0.05, 0.  ],
           [0.15, 0.  , 0.8 , 0.05],
           [0.01, 0.  , 0.  , 0.99]])



###### ______________________________________________________________________________________________________________________________________

##### <u>Probability and Expectations</u>

 Chapman-Kolmogorov Equation:
 
 $$P(X_{n+m} = j | X_n = i) = \sum_k P(X_{n+m} = j | X_{n+k} = k)P(X_{n+k} = k | X_n = i)$$
 
 Matrix Form:
 $$P^{(m+n)} = P^{(m)}P^{(n)}$$

Using these equations we can answer some really cool questions, for example. . .

- What is the probability of going from current to delinquent in 12 months?

- What is the expected number of delinquent 90+ days loans in our portfolio after 12 months?

Extremely useful information!

###### ______________________________________________________________________________________________________________________________________

##### <u>Example: Probability of 90+ Days Delinquent after 12 Months</u>

Here we are considering a single loan's transition, this is the magic of the Chapman-Kolmogorv equations

We don't need to even assume an initial state - the final matrix enables us to determine the transition probability


```python
import numpy as np

# Define transition matrix
P = np.array([
    [0.99, 0.01, 0.00, 0.00],
    [0.25, 0.70, 0.05, 0.00],
    [0.15, 0.00, 0.80, 0.05],
    [0.01, 0.00, 0.00, 0.99]
])

# Calculate 12-month transition probabilities using matrix power
P_12 = np.linalg.matrix_power(P, 12)

# Print probability of transitioning from current (state 0) to 90+ days delinquent (state 3) in 12 months
print(f"Probability of going from current to 90+ days delinquent in 12 months: {P_12[0,3]*100:.4f}%")
```

    Probability of going from current to 90+ days delinquent in 12 months: 0.1826%



```python
# Print 12-month transition probabilities for each starting state
states = ['Current', '30-60 Days', '60-90 Days', '90+ Days']

print("12-Month Transition Probabilities:")
print("---------------------------------")
for i, start_state in enumerate(states):
    print(f"\nStarting from {start_state}:")
    for j, end_state in enumerate(states):
        prob = P_12[i,j] * 100
        print(f"  → {end_state}: {prob:.2f}%")
```

    12-Month Transition Probabilities:
    ---------------------------------
    
    Starting from Current:
      → Current: 95.98%
      → 30-60 Days: 3.17%
      → 60-90 Days: 0.66%
      → 90+ Days: 0.18%
    
    Starting from 30-60 Days:
      → Current: 89.36%
      → 30-60 Days: 4.11%
      → 60-90 Days: 3.21%
      → 90+ Days: 3.32%
    
    Starting from 60-90 Days:
      → Current: 69.19%
      → 30-60 Days: 2.03%
      → 60-90 Days: 7.20%
      → 90+ Days: 21.58%
    
    Starting from 90+ Days:
      → Current: 11.05%
      → 30-60 Days: 0.27%
      → 60-90 Days: 0.04%
      → 90+ Days: 88.64%


###### ______________________________________________________________________________________________________________________________________

##### <u>Example: Expected Proportion of Loans in our Portfolio being Delinquent (90+ days) after 12 Months</u>
 
 Here we consider an initial distribution of loan proportions in our portfolio.  
 
 If we want to know the *expected* proportion of loan states in our portfolio then this will depend on the *inital* state distribution

 **Intuition:** 
 
 If there are more loans in deliquency in the initial portfolio then there will be a higher expectation of deliquent loans relative to a portfolio that starts out with all current loans - this is why we must consider the *initial state distribution*.

###### ______________________________________________________________________________________________________________________________________

##### <u>Understanding $\pi(t)$ - The State Distribution Vector</u>
 
  The vector $\pi(t)$ represents the probability distribution across all states at time t.
  Each element $\pi_i(t)$ gives the probability of being in state i at time t.
 
  For example, in our loan portfolio case:
  - $\pi_1(t)$ = Proportion of current loans
  - $\pi_2(t)$ = Proportion of 30-day delinquent loans  
  - $\pi_3(t)$ = Proportion of 60-day delinquent loans
  - $\pi_4(t)$ = Proportion of 90+ day delinquent loans
 
  The elements must sum to 1 since they represent probabilities:
  $\sum_{i=1}^n \pi_i(t) = 1$

  ###### ______________________________________________________________________________________________________________________________________

 Using the Chapman-Kolmogorov equation and law of total expectation, we can easily calculate this as:
 
 $$\pi(t) = \pi(0)P^t$$
 
 where $\pi(0)$ is our initial distribution vector and $P^t$ is our transition matrix raised to power $t$
 
 Fortunately, we just defind a vector of our proportions and use matrix multiplication - very easy!


```python
import numpy as np

# Define transition matrix
P = np.array([
    [0.99, 0.01, 0.00, 0.00],
    [0.25, 0.70, 0.05, 0.00],
    [0.15, 0.00, 0.80, 0.05],
    [0.01, 0.00, 0.00, 0.99]
])

# Initial distribution (assume all loans start as current)
initial_dist = np.array([.5, .2, 0, .3])

# Calculate 12-month transition probabilities using matrix power
P_12 = np.linalg.matrix_power(P, 10000)

# Calculate expected distribution after 12 months
final_dist = initial_dist @ P_12

# Print expected proportion of loans that will be 90+ days delinquent after 12 months
print(f"Expected proportion of loans that will be 90+ days delinquent after 12 months: {final_dist[3]*100:.4f}%")
```

    Expected proportion of loans that will be 90+ days delinquent after 12 months: 3.8462%



  ###### ______________________________________________________________________________________________________________________________________

##### <u>Key Properties of Markov Chains</u>

 1. *Recurrence*: A state is recurrent if the chain returns to it with probability 1

 2. *Irreducibility*: A chain is irreducible if every state can be reached from every other state

 3. *Steady State*: The long-term probabilities that stabilize as time approaches infinity

 4. *Absorbing States*: States that once entered, cannot be left (probability of staying = 1)

 5. *Periodicity*: The number of steps needed to return to a state (aperiodic if random)

 6. *Ergodicity*: When a chain is both irreducible and aperiodic, leading to unique steady state

 7. *Transience*: States that have a non-zero probability of never being revisited

 These properties help analyze long-term behavior and stability of Markov processes

   ###### ______________________________________________________________________________________________________________________________________

##### <u>Example: Long-Run Proportion of Portfolio Loans in Each State</u>

In an arbitrarily large amount of time (50 months), the proportion of loans in our portfolio at each state may converge given some assumptions

This shows the evolution of loans over time to a steady-state.

However, if the assumptions are violated (which they are) this steady state will *not* be true!


```python
# Calculate steady state distribution by iterating transition matrix
n_steps = 50
states = np.zeros((n_steps, 4))
states[0] = initial_dist

for t in range(1, n_steps):
    states[t] = states[t-1] @ P

# Create figure for convergence plot
fig = go.Figure()

# Plot convergence for each state
fig.add_trace(go.Scatter(
    x=list(range(n_steps)),
    y=states[:, 0],
    name='Current',
    line=dict(color='rgba(0, 255, 0, 0.8)', width=2)
))

fig.add_trace(go.Scatter(
    x=list(range(n_steps)),
    y=states[:, 1],
    name='30-59 Days Late',
    line=dict(color='rgba(255, 255, 0, 0.8)', width=2)
))

fig.add_trace(go.Scatter(
    x=list(range(n_steps)),
    y=states[:, 2],
    name='60-89 Days Late',
    line=dict(color='rgba(255, 165, 0, 0.8)', width=2)
))

fig.add_trace(go.Scatter(
    x=list(range(n_steps)),
    y=states[:, 3],
    name='90+ Days Late',
    line=dict(color='rgba(255, 0, 0, 0.8)', width=2)
))

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    title='Convergence to Steady State Distribution',
    xaxis_title='Time Steps',
    yaxis_title='Proportion of Loans',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
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

print("\nImportant Notes on Steady State Analysis:")
print("1. This steady state analysis assumes temporal homogeneity (transition probabilities")
print("   remain constant over time), which may not hold in reality due to:")
print("   - Economic cycles")
print("   - Seasonal effects")
print("   - Policy changes")
print("   - External shocks")
print("\n2. The convergence shown here assumes the Markov chain is:")
print("   - Irreducible (all states can be reached from all other states)")
print("   - Aperiodic (returns to states occur at irregular intervals)")
print("   Without these properties, a unique steady state may not exist.")

```



    
    Important Notes on Steady State Analysis:
    1. This steady state analysis assumes temporal homogeneity (transition probabilities
       remain constant over time), which may not hold in reality due to:
       - Economic cycles
       - Seasonal effects
       - Policy changes
       - External shocks
    
    2. The convergence shown here assumes the Markov chain is:
       - Irreducible (all states can be reached from all other states)
       - Aperiodic (returns to states occur at irregular intervals)
       Without these properties, a unique steady state may not exist.


---

#### 3.) 📈 Applying Markov Chains

The natural next questions are:

- What do we define as states for our Markov chain?
- How do we find the probabilities for our transition matrix?

Defining states isn't too difficult and is largely problem dependent.  

**Some examples:**
- Volatility (Temperature): Low, Med, High
- Market (Trend): Bullish, Bearish, Sideways
- Liquidity (Spread): Low, Med, High

We can come up with various thresholds for each of these states using data from the market
- VIX
- SPX
- Spreads

Ok, then how do we get probabilities?

###### ______________________________________________________________________________________________________________________________________

##### <u>Maximum Likelihood Estimation (MLE)</u>

The method of maximum likelihood estimation is extremely intuitive

What is the data generating distribution that has *the highest likelihood* of producing the data we've seen historically?

There are several assumptions (both for Markov Chains & MLE) that enable this functionality, we will discuss them after observing the technique itself


```python
# Generate synthetic data from a normal distribution
np.random.seed(42)
n_samples = 1000
true_mean = 0
true_std = 1
data = np.random.normal(true_mean, true_std, n_samples)

# Calculate MLE parameters at different sample sizes
window_sizes = np.arange(10, n_samples, 10)
mle_means = np.zeros(len(window_sizes))
mle_stds = np.zeros(len(window_sizes))

for i, window in enumerate(window_sizes):
    sample = data[:window]
    mle_means[i] = np.mean(sample)
    mle_stds[i] = np.std(sample)

# Create figure for MLE convergence
fig = go.Figure()

# Plot histogram of full dataset
fig.add_trace(go.Histogram(
    x=data,
    name='Observed Data',
    histnorm='probability density',
    nbinsx=30,
    opacity=0.7,
    marker_color='rgba(128, 128, 128, 0.6)'
))

# Plot true distribution
x = np.linspace(-4, 4, 100)
true_pdf = 1/(true_std * np.sqrt(2*np.pi)) * np.exp(-(x-true_mean)**2/(2*true_std**2))
fig.add_trace(go.Scatter(
    x=x,
    y=true_pdf,
    name='True Distribution',
    line=dict(color='rgba(255, 0, 0, 0.8)', width=2, dash='dash')
))

# Plot final MLE fit
final_pdf = 1/(mle_stds[-1] * np.sqrt(2*np.pi)) * np.exp(-(x-mle_means[-1])**2/(2*mle_stds[-1]**2))
fig.add_trace(go.Scatter(
    x=x,
    y=final_pdf,
    name='MLE Fit',
    line=dict(color='rgba(0, 255, 0, 0.8)', width=2)
))

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    title='Maximum Likelihood Estimation - Fitting Normal Distribution',
    xaxis_title='Value',
    yaxis_title='Probability Density',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
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

print("\nKey Points about Maximum Likelihood Estimation for Normal Distribution:")
print("1. The histogram shows the actual distribution of observed data")
print("2. The red dashed line shows the true underlying distribution")
print("3. The green line shows our MLE fit, which closely matches the true distribution")
print("4. MLE finds the parameters (mean, std) that maximize the likelihood of observing our data")

```



    
    Key Points about Maximum Likelihood Estimation for Normal Distribution:
    1. The histogram shows the actual distribution of observed data
    2. The red dashed line shows the true underlying distribution
    3. The green line shows our MLE fit, which closely matches the true distribution
    4. MLE finds the parameters (mean, std) that maximize the likelihood of observing our data


##### This Exact Same Methodology Works for Markov Chains but Instead of a Distribution its a Transition Matrix of States

The diagram below shows the exact same methodology (MLE) as the above with a Gaussian (normal) distribution but applied to Markov Chains


```python
# Generate synthetic data to demonstrate MLE
np.random.seed(42)
n_samples = 1000
true_probs = np.array([0.7, 0.2, 0.1])  # True probabilities for 3 states
states = np.random.choice(3, size=n_samples, p=true_probs)

# Calculate empirical probabilities over time
window_sizes = np.arange(10, n_samples, 10)
empirical_probs = np.zeros((len(window_sizes), 3))

for i, window in enumerate(window_sizes):
    counts = np.bincount(states[:window], minlength=3)
    empirical_probs[i] = counts / window

# Create figure for MLE convergence
fig = go.Figure()

# Plot convergence for each state probability
state_names = ['State A', 'State B', 'State C']
colors = ['rgba(0, 255, 0, 0.8)', 'rgba(255, 165, 0, 0.8)', 'rgba(255, 0, 0, 0.8)']

for i in range(3):
    # Plot empirical probabilities
    fig.add_trace(go.Scatter(
        x=window_sizes,
        y=empirical_probs[:, i],
        name=f'{state_names[i]} (Empirical)',
        line=dict(color=colors[i], width=2)
    ))
    
    # Plot true probabilities
    fig.add_trace(go.Scatter(
        x=[window_sizes[0], window_sizes[-1]],
        y=[true_probs[i], true_probs[i]],
        name=f'{state_names[i]} (True)',
        line=dict(color=colors[i], width=2, dash='dash')
    ))

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    title='Maximum Likelihood Estimation - Convergence to True Probabilities',
    xaxis_title='Number of Samples',
    yaxis_title='Probability',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
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
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[0, 1]
)

fig.show()

print("\nKey Points about Maximum Likelihood Estimation:")
print("1. As we collect more samples, our empirical probabilities (solid lines)")
print("   converge to the true probabilities (dashed lines)")
print("2. MLE provides consistent estimates - more data leads to better estimates")
print("3. The convergence rate follows the Law of Large Numbers")
print("4. Early estimates can be volatile due to small sample sizes")
```



    
    Key Points about Maximum Likelihood Estimation:
    1. As we collect more samples, our empirical probabilities (solid lines)
       converge to the true probabilities (dashed lines)
    2. MLE provides consistent estimates - more data leads to better estimates
    3. The convergence rate follows the Law of Large Numbers
    4. Early estimates can be volatile due to small sample sizes


##### <u>Deriving the MLE for a Markov Chain</u>

Let's derive the Maximum Likelihood Estimate (MLE) for a Markov chain:

Given a sequence of states $X_1, X_2, ..., X_T$, the likelihood function is:

$$L(\mathbf{P}) = P(X_1) \prod_{t=1}^{T-1} P(X_{t+1}|X_t)$$

Taking the log:

$$\log L(\mathbf{P}) = \log P(X_1) + \sum_{t=1}^{T-1} \log P(X_{t+1}|X_t)$$

$$= \log P(X_1) + \sum_{i=1}^m \sum_{j=1}^m n_{ij} \log p_{ij}$$

Subject to constraints:
$$\sum_{j=1}^m p_{ij} = 1 \quad \text{for all } i$$
$$p_{ij} \geq 0 \quad \text{for all } i,j$$

Using Lagrange multipliers and solving:

$$p_{ij} = \frac{n_{ij}}{\sum_k n_{ik}}$$


 For a Markov chain, the Maximum Likelihood Estimate (MLE) of transition probabilities is:

 $p_{ij} = \frac{n_{ij}}{\sum_k n_{ik}}$

 where:
 - $p_{ij}$ is the probability of transitioning from state $i$ to state $j$
 - $n_{ij}$ is the number of observed transitions from state $i$ to state $j$
 - $\sum_k n_{ik}$ is the total number of transitions from state $i$ to any state


###### ______________________________________________________________________________________________________________________________________

##### <u>Example: MLE of Transition Probabilities for Loan Portfolio</u>

Remember!  We have a portfolio of loans and we are modeling the transition matrix for *each loan* **NOT** the entire portfolio at once


```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define states
states = ['Current', '30-59 Days', '60-89 Days', '90+ Days']

# Generate synthetic loan transitions
n_loans = 1000
n_months = 12

# Initialize with mostly current loans, but some in each delinquency bucket
initial_distribution = [0.85, 0.08, 0.05, 0.02]
loan_states = np.random.choice(states, size=n_loans, p=initial_distribution)

# Create transition matrix (based on typical loan behavior)
transition_matrix = np.array([
    [0.99, 0.01, 0.00, 0.00],
    [0.25, 0.70, 0.05, 0.00],
    [0.15, 0.00, 0.80, 0.05],
    [0.01, 0.00, 0.00, 0.99]
])


# Generate transitions for each loan over time
loan_history = []
for month in range(n_months):
    next_states = []
    for loan in loan_states:
        current_state_idx = states.index(loan)
        next_state = np.random.choice(states, p=transition_matrix[current_state_idx])
        next_states.append(next_state)
    loan_states = next_states
    loan_history.append(loan_states.copy())

# Convert to DataFrame for easier analysis
loan_df = pd.DataFrame(loan_history, columns=[f'Loan_{i}' for i in range(n_loans)])
loan_df.index.name = 'Month'

print("Sample of synthetic loan transition data:")
print(loan_df.iloc[:5, :6])  # Show first 5 months for first 6 loans

# Count transitions between states
print("\nDistribution of loans in month 0:")
print(pd.Series(loan_history[0]).value_counts())

```

    Sample of synthetic loan transition data:
            Loan_0      Loan_1   Loan_2   Loan_3   Loan_4   Loan_5
    Month                                                         
    0      Current  60-89 Days  Current  Current  Current  Current
    1      Current  60-89 Days  Current  Current  Current  Current
    2      Current  60-89 Days  Current  Current  Current  Current
    3      Current  60-89 Days  Current  Current  Current  Current
    4      Current  60-89 Days  Current  Current  Current  Current
    
    Distribution of loans in month 0:
    Current       865
    30-59 Days     67
    60-89 Days     46
    90+ Days       22
    Name: count, dtype: int64


Give our dataset of loans we can apply the MLE of transition probabilities derived above:

 $$p_{ij} = \frac{n_{ij}}{\sum_k n_{ik}}$$

 where:
 - $p_{ij}$ is the probability of transitioning from state $i$ to state $j$
 - $n_{ij}$ is the number of observed transitions from state $i$ to state $j$
 - $\sum_k n_{ik}$ is the total number of transitions from state $i$ to any state


```python
# Calculate MLE transition matrix from the data
transitions = np.zeros((len(states), len(states)))

# Count transitions between states
for t in range(len(loan_history)-1):
    current_states = loan_history[t]
    next_states = loan_history[t+1]
    
    for curr, next_state in zip(current_states, next_states):
        i = states.index(curr)
        j = states.index(next_state)
        transitions[i,j] += 1

# Convert counts to probabilities (MLE estimate)
mle_transition_matrix = transitions / transitions.sum(axis=1)[:,np.newaxis]

print("Maximum Likelihood Estimate of Transition Matrix:")
print(mle_transition_matrix)

# Compare with original transition matrix
print("\nOriginal Transition Matrix:")
print(transition_matrix)

# Calculate absolute difference
print("\nAbsolute Difference between MLE and Original:")
print(np.abs(mle_transition_matrix - transition_matrix))

```

    Maximum Likelihood Estimate of Transition Matrix:
    [[0.99197754 0.00802246 0.         0.        ]
     [0.23325062 0.71712159 0.04962779 0.        ]
     [0.15503876 0.         0.77131783 0.07364341]
     [0.00544959 0.         0.         0.99455041]]
    
    Original Transition Matrix:
    [[0.99 0.01 0.   0.  ]
     [0.25 0.7  0.05 0.  ]
     [0.15 0.   0.8  0.05]
     [0.01 0.   0.   0.99]]
    
    Absolute Difference between MLE and Original:
    [[0.00197754 0.00197754 0.         0.        ]
     [0.01674938 0.01712159 0.00037221 0.        ]
     [0.00503876 0.         0.02868217 0.02364341]
     [0.00455041 0.         0.         0.00455041]]


###### ______________________________________________________________________________________________________________________________________

##### <u>Assumptions and Properties of Maximum Likelihood Estimation (MLE) for Markov Chains</u>

##### Key Assumptions:
 1. Markov Property: The future state depends only on the current state, not past states
 2. Time Homogeneity: Transition probabilities remain constant over time
 3. Ergodicity: Unique stationary distribution
 4. Sufficient Data: Enough transitions are observed to estimate all probabilities
 5. Independence: Each loan's transitions are independent of other loans

 Likely 2 & 5 are our biggest concerns. . .

#####  Properties of MLE for Markov Chains:
 1. Consistency: As sample size increases, MLE converges to true parameters
 2. Asymptotic Efficiency: Achieves minimum variance among consistent estimators
 3. Asymptotic Normality: Distribution approaches normal as sample size grows
 4. Invariance: MLE of a function of parameters equals function of MLE
 5. Maximum Entropy: Provides least biased estimate given available information


---

#### 4.) 💭 Closing Thoughts and Future Topics

**TL;DW Executive Summary**
- When modeling something as a random variable *independence* is way too strong of an assumption that can lead to extremely inaccurate models
- A major correction can be applied by considering the simplifying assumption of local conditional dependence rather than full independence
- Markov chains effectively model these dynamics and enable us to estimate probabilities and expectations from an initial state given a local dependency structure
- The transition matrices and corresponding probabilities can be easily estimated from data using the result from MLE
- Though Markov chains aid in the modeling process offering more *accurate* or *reasonable* estimates there are many assumptions that are still violated in practice
- Typically, Markov chains are a first step in the modeling process (regime switching or HMM models, for example) - understanding them is the first step toward more comprehensive applications  

**Future Topics**

Technical Videos and Other Discussions

- Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)

- Hidden Markov Models and Other Applications of Such Models

- Citadel, Jane Street, . . . Interview Questions

[Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)

- How to Build an Earnings Event Options Trading Dashboard

- Live Kalman Filter Model with Regime Dynamics (HMMs) 

- Automated Delta-Neutral Trading System (Algorithmically Capitalizing On Volatility Speculation)
    - We can actually use Markov chains and other TS models here to aid in producing these speculative positions 

---

####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$
