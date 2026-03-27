# ### 📈 Markov Property for Quant Trading
# 
# ##### ▶️ Related Quant Guild Videos:
# 
# - [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
# 
# - [Quant Trader on Retail vs Institutional Trading](https://youtu.be/j1XAcdEHzbU)
# 
# - [Quant on Trading and Investing](https://youtu.be/CKXp_sMwPuY)
# 
# - [Why Poker Pros Make the Best Traders (It's NOT Luck)](https://youtu.be/wZChBKDFFeU)
# 
# - [Quant vs. Discretionary Trading](https://youtu.be/3gblERSSHXI)
# 
# - [Quant Busts 3 Trading Myths with Math](https://youtu.be/wJfIk3VnubE)
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### [🚀 Master your Quantitative Skills with Quant Guild](https://quantguild.com)
# 
# 
# 
# ##### [📚 Visit the Quant Guild Library for more Jupyter Notebooks](https://github.com/romanmichaelpaolucci/Quant-Guild-Library)
# 
# ##### [📈 Interactive Brokers for Algorithmic Trading](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# ##### [👾 Quant Guild Discord](discord.com/invite/MJ4FU2c6c3)
# 
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


# ### 📖 Sections
# 
# #### 1.) 🎲 Unconditional and Conditional Statistics
# 
# - Unconditional distributions, probabilities, and statistics
# 
# - Conditional distributions, probabilities and statistics
# 
# #### 2.) 📊 Markov (Memoryless) Property
# 
# - Definition of the Markov (Memoryless) Property
# 
# - Example: Cumulative Dice Roll Sum
# 
# #### 3.) 📈 Trading Implications
# 
# - Trading Strategy P/L Distribution
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 🎲 Unconditional and Conditional Statistics
# 
# We rarely need more beyond basic probability and statistics to evaluate the efficacy of a trading strategy
# 
# Most don't understand the difference between conditional and unconditional statistics (a big problem!)
# 
# The (discrete) probability mass function (PMF) for the sum $S$ of two fair six-sided dice, $S = X_1 + X_2$, is piecewise:
# 
#  $$
#  P(S = k) =
#  \begin{cases}
#  \frac{k-1}{36}, & 2 \leq k \leq 7 \\
#  \frac{13-k}{36}, & 8 \leq k \leq 12 \\
#  0, & \text{otherwise}
#  \end{cases}
#  $$
# 
# ##### Empirical Unconditional Distribution Convergence


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Dice distribution setup ---
dice_faces = np.arange(1, 7)
sums = np.arange(2, 13)

# Theoretical PMF for sum of two dice
theoretical_counts = np.zeros(11)
for i in dice_faces:
    for j in dice_faces:
        theoretical_counts[i + j - 2] += 1
theoretical_pmf = theoretical_counts / theoretical_counts.sum()

# --- Simulate sampled rolls ---
n_trials = 400  # More samples for smoother convergence
np.random.seed(42)
samples_1 = np.random.choice(dice_faces, size=n_trials)
samples_2 = np.random.choice(dice_faces, size=n_trials)
sum_samples = samples_1 + samples_2

# --- Plotting helper, for a given step ---
def make_dice_sum_empirical_convergence_fig(step):
    drawn = sum_samples[step]
    counts = np.bincount(sum_samples[:step+1], minlength=13)[2:13]
    empirical_pmf = counts / (step + 1)
    empirical_display = empirical_pmf[:len(sums)]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Theoretical Dice Sum Distribution", "Theoretical vs. Empirical")
    )

    # Theoretical PMF: highlight current drawn sum with gold border (left only)
    bar_colors = ['#00bfff'] * len(sums)
    border_colors = ['rgba(0,0,0,0)'] * len(sums)
    border_widths = [0] * len(sums)
    if 2 <= drawn <= 12:
        idx = drawn - 2
        border_colors[idx] = '#FFD700'
        border_widths[idx] = 4

    # --- Theoretical PMF (first subplot, solid, left) ---
    fig.add_trace(
        go.Bar(
            x=sums,
            y=theoretical_pmf,
            width=0.65,
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths)
            ),
            name="Theoretical PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: perfectly stacked bars, theory behind, empirical on top, bars align ---
    # First add the theoretical PMF as semi-transparent background
    fig.add_trace(
        go.Bar(
            x=sums,
            y=theoretical_pmf,
            width=0.65,
            marker=dict(color='#00bfff', opacity=0.26),
            name="Theoretical PMF",
            showlegend=False,
            offset=0
        ),
        row=1, col=2
    )
    # On top, add empirical PMF, same x, same width
    fig.add_trace(
        go.Bar(
            x=sums,
            y=empirical_display,
            width=0.65,
            marker=dict(color='#ff00cc', opacity=0.80),
            name="Empirical",
            showlegend=False,
            offset=0
        ),
        row=1, col=2
    )

    # --- Invisible traces for legend, these show the correct colors ---
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color='#00bfff'),
            name="Theoretical PMF",
            showlegend=True
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color='#ff00cc'),
            name="Empirical",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Axes formatting: Both y axes fixed to [0,0.19] for 2-12 dice sums ---
    y_range = [0, 0.19]
    fig.update_xaxes(title_text="Sum", range=[1.5, 12.5], tickvals=sums, row=1, col=1)
    fig.update_yaxes(title_text="P(Sum=k)", range=y_range, row=1, col=1)
    fig.update_xaxes(title_text="Sum", range=[1.5, 12.5], tickvals=sums, row=1, col=2)
    fig.update_yaxes(title_text="P(Sum=k)", range=y_range, row=1, col=2)

    # --- Subtle gridlines and layout
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_layout(
        height=470,
        width=960,
        title_text=f"Dice Sum: Theoretical vs. Empirical (LLN, {n_trials} Trials)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.98,
            xanchor='right', yanchor='top',
            orientation='v',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=15)
        ),
        margin=dict(l=50, r=20, b=70, t=55),
        barmode='overlay'  # makes bars stack perfectly
    )

    return fig

# --- Animation frames ---
dice_sum_frames = [
    go.Frame(data=make_dice_sum_empirical_convergence_fig(step).data, name=str(step))
    for step in range(n_trials)
]

fig = make_dice_sum_empirical_convergence_fig(0)
fig.frames = dice_sum_frames

# --- Play button (much faster animation) ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.11,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 25, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()



# Convergence is gaurenteed by the Law of Large Numbers (LLN)
#  $$
#  \frac{1}{n} \sum_{i=1}^n \mathbb{I}\{ X_i \leq x \} \xrightarrow{n \to \infty} \mathbb{P}(X \leq x)
#  $$
#  
#  where $\mathbb{I}\{\cdot\}$ is the indicator function. 
#  
#  This shows that the empirical CDF converges to the theoretical CDF at every $x$ by the Law of Large Numbers.
# 
# In other words, all distributions, probabilities, and statistics in this context will converge to their theoretical values
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Empirical Unconditional Probability Convergence
# 
# This is a big deal, let's roll 10,000 dice and observe the probability of seeing an 11. . .
# 
#  Probability of rolling an 11 with two dice:
#  $$
#  \mathbb{P}(X = 11) = \frac{\text{Number of favorable outcomes}}{\text{Total outcomes}} = \frac{2}{36} = \frac{1}{18} \approx 0.0556
#  $$
# 


import numpy as np
import plotly.graph_objects as go

# Define a vibrant purple/magenta color
HOT_PURPLE = '#ff00ff' 
THEORETICAL_PROB_COLOR = '#ffaa00' # Keeping the theoretical line orange for better contrast/visibility

# --- Setup for Dice Rolls ---
dice_faces = np.array([1, 2, 3, 4, 5, 6])
n_trials_anim = 1000   # Max rolls for the animation

# Theoretical Probability of Sum=11 (2/36)
theoretical_pmf = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36.0
theoretical_prob_11 = theoretical_pmf[11 - 2] 

# Generate 1000 rolls
np.random.seed(42)
sample_1 = np.random.choice(dice_faces, size=n_trials_anim)
sample_2 = np.random.choice(dice_faces, size=n_trials_anim)
sum_rolls = sample_1 + sample_2

# Indicator: 1 for a hit (sum=11), 0 for a miss
is_eleven = (sum_rolls == 11).astype(int)

# Calculate the rolling empirical probability
rolling_prob = np.cumsum(is_eleven) / (np.arange(1, n_trials_anim + 1))

# Assign colors: Bright Green for Hit, Bright Red for Miss
indicator_colors = np.where(is_eleven == 1, '#33ff00', '#ff3300') 


# --- 1. Initial Figure Setup (Frame 0) ---

# CHANGES FOR SPEED:
step_size = 10
frame_duration = 20

steps = np.arange(1, n_trials_anim + 1, step_size)
if steps[-1] != n_trials_anim:
    steps = np.append(steps, n_trials_anim) # Ensure the final frame is included

initial_rolls = 1
initial_x = np.arange(1, initial_rolls + 1)
initial_y_ind = is_eleven[:initial_rolls]
initial_y_prob = rolling_prob[:initial_rolls]
initial_colors = indicator_colors[:initial_rolls]

fig = go.Figure(
    data=[
        # Trace 1: Indicator Dots (Left Y-Axis)
        go.Scatter(
            x=initial_x,
            y=initial_y_ind,
            mode='markers',
            # CHANGES: Opacity raised to 0.9, Line width set to 0 to remove border
            marker=dict(color=initial_colors, size=7, opacity=0.9, line=dict(width=0)), 
            name="Hit (Green) / Miss (Red)",
            yaxis='y1',
            showlegend=True
        ),
        # Trace 2: Empirical Probability Line (Right Y-Axis) - HOT PURPLE
        go.Scatter(
            x=initial_x,
            y=initial_y_prob,
            mode='lines',
            line=dict(color=HOT_PURPLE, width=3),
            name='Empirical P(Sum=11)',
            yaxis='y2'
        ),
        # Trace 3: Theoretical Probability Line (Right Y-Axis) - HOT PURPLE (if requested)
        go.Scatter(
            x=[1, n_trials_anim],
            y=[theoretical_prob_11, theoretical_prob_11],
            mode='lines',
            line=dict(color=HOT_PURPLE, width=2), # Changed to HOT_PURPLE
            name='Theoretical P(Sum=11)',
            yaxis='y2'
        )
    ],
    layout=go.Layout(
        # General Layout Settings
        width=1000, height=500,
        
        title=dict(
            text="Animation: Convergence of P(Sum=11) to Theoretical Probability",
            font=dict(color='white') 
        ),
        
        xaxis=dict(
            title="Number of Rolls", 
            range=[1, n_trials_anim],
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.24)', 
            color='white', 
            tickfont=dict(color='white')
        ),
        
        # Left Y-axis (Indicator) - No horizontal grid lines
        yaxis=dict(
            title="Indicator: Sum=11",
            range=[-0.1, 1.1],
            color="#33ff00",
            side='left',
            tickvals=[0, 1],
            showgrid=False, 
            tickfont=dict(color='white')
        ),
        
        # Right Y-axis (Probability) - No horizontal grid lines
        yaxis2=dict(
            title="Probability",
            range=[0, 0.12],
            overlaying='y',
            side='right',
            color=HOT_PURPLE, # Axis label color
            showgrid=False, 
            tickfont=dict(color='white')
        ),
        
        legend=dict(
            x=0.5, y=-0.15,
            xanchor='center',
            yanchor='top',
            orientation='h', 
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        ),
        
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14), 
        
        # PLAY BUTTON: Raised slightly to y=-0.28
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                bgcolor='rgba(0,0,0,0)', 
                bordercolor='white', 
                x=0, 
                y=-0.28, # Raised from -0.32 to -0.28
                xanchor='left',
                yanchor='top',
                font=dict(color='white'),
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": frame_duration, "redraw": True}, 
                                     "fromcurrent": True,
                                     "transition": {"duration": 0}}
                                    ]
                    )
                ]
            )
        ]
    )
)

# --- 2. Create Frames ---
frames = []
for k in steps:
    current_x = np.arange(1, k + 1)
    
    frames.append(go.Frame(
        data=[
            # Indicator Points (Trace 1)
            go.Scatter(
                x=current_x, y=is_eleven[:k],
                # CHANGES: Opacity raised to 0.9, Line width set to 0 to remove border
                marker=dict(color=indicator_colors[:k], size=7, opacity=0.9, line=dict(width=0))
            ),
            # Empirical Probability Line (Trace 2) - HOT PURPLE
            go.Scatter(
                x=current_x, y=rolling_prob[:k],
                line=dict(color=HOT_PURPLE, width=3), mode='lines'
            ),
            # Theoretical Line (Trace 3) - HOT PURPLE
            go.Scatter(
                x=[1, n_trials_anim], y=[theoretical_prob_11, theoretical_prob_11],
                line=dict(color=HOT_PURPLE, width=2, dash='dash'), mode='lines'
            )
        ],
        name=str(k),
        layout=go.Layout(
            title=f"Convergence: Probability of Rolling an 11 (Rolls: {k}/{n_trials_anim})"
        )
    ))

# --- 3. Add Frames to Figure and Show ---
fig.frames = frames

# Set the initial frame title
fig.update_layout(
    title=f"Convergence: Probability of Rolling an 11 (Rolls: {initial_rolls}/{n_trials_anim})"
)

fig.show()


# By the Law of Large Numbers (LLN) the unconditional probability will converge
# 
# $$
# \lim_{n\to \infty} \frac{1}{n} \sum_{i=1}^n \mathbb{I}\{X_i = 11\} \rightarrow \mathbb{P}(X = 11)
# $$
# 
# This is literally what the animation above is showing


# ###### ______________________________________________________________________________________________________________________________________


# ##### Empirical Unconditional Statistics Convergence
# 
# Statistics function in the same way, and actually imply the probability convergence we say above (they are Bernoulli RVs)
# 
# If we continue to roll a dice we will observe convergence to the theoretical values
#  $$
#  \lim_{n\to\infty} \frac{1}{n} \sum_{i=1}^n X_i = \mathbb{E}[X]
#  $$
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Dice distribution setup ---
dice_faces = np.arange(1, 7)
sums = np.arange(2, 13)

# Theoretical PMF for sum of two dice
theoretical_counts = np.zeros(11)
for i in dice_faces:
    for j in dice_faces:
        theoretical_counts[i + j - 2] += 1
theoretical_pmf = theoretical_counts / theoretical_counts.sum()

# Theoretical Mean (Expected Value, mu) for sum of two dice
# E[X] = sum(k * P(X=k)) for all k
theoretical_mean = np.sum(sums * theoretical_pmf)

# --- Simulate sampled rolls ---
n_trials = 200  # Number of trials
np.random.seed(42)
samples_1 = np.random.choice(dice_faces, size=n_trials)
samples_2 = np.random.choice(dice_faces, size=n_trials)
sum_samples = samples_1 + samples_2

# --- Calculate the running empirical mean for LLN plot ---
cumulative_sum = np.cumsum(sum_samples)
trial_indices = np.arange(1, n_trials + 1)
empirical_means = cumulative_sum / trial_indices

# --- Plotting helper, for a given step ---
def make_lln_convergence_fig(step):
    drawn = sum_samples[step]
    current_empirical_mean = empirical_means[step]

    # Use specs to enable a secondary y-axis on the right subplot (column 2)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Theoretical Dice Sum Distribution", "Empirical Mean Convergence (LLN)"),
        specs=[[{}, {"secondary_y": True}]]
    )

    # Theoretical PMF: highlight current drawn sum with gold border (left only)
    bar_colors = ['#00bfff'] * len(sums)
    border_colors = ['rgba(0,0,0,0)'] * len(sums)
    border_widths = [0] * len(sums)
    if 2 <= drawn <= 12:
        idx = drawn - 2
        border_colors[idx] = '#FFD700'
        bar_colors[idx] = '#1E90FF' # Slightly darker for drawn bar
        border_widths[idx] = 4

    # --- Theoretical PMF (first subplot, solid, left) ---
    fig.add_trace(
        go.Bar(
            x=sums,
            y=theoretical_pmf,
            width=0.65,
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths)
            ),
            name="Theoretical PMF",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right subplot: LLN Convergence (Primary Y-axis: Mean, Pinkish) ---
    
    # 1. Theoretical Mean Line (Expected Value)
    fig.add_trace(
        go.Scatter(
            x=trial_indices,
            y=[theoretical_mean] * n_trials,
            mode='lines',
            line=dict(color='#00bfff', width=2, dash='dash'),
            name=f"Theoretical Mean (E[X]) = {theoretical_mean:.2f}",
            showlegend=True
        ),
        row=1, col=2, secondary_y=False
    )

    # 2. Empirical Mean Line (up to current step)
    fig.add_trace(
        go.Scatter(
            x=trial_indices[:step+1],
            y=empirical_means[:step+1],
            mode='lines+markers',
            line=dict(color='#ff00cc', width=3),
            marker=dict(size=4, color='#ff00cc'),
            name=f"Empirical Mean: {current_empirical_mean:.3f}",
            showlegend=True
        ),
        row=1, col=2, secondary_y=False
    )
    
    # 3. Highlight current empirical mean point
    fig.add_trace(
        go.Scatter(
            x=[trial_indices[step]],
            y=[current_empirical_mean],
            mode='markers',
            marker=dict(size=12, color='#ff00cc', line=dict(width=2, color='black')),
            name="Current Empirical Mean",
            showlegend=False
        ),
        row=1, col=2, secondary_y=False
    )
    
    # --- Right subplot: Individual Roll Outcomes (Secondary Y-axis: Roll Outcome, Orange) ---
    ORANGE_COLOR = '#FFA000' # Bright Orange for Roll Outcomes

    # 4. Individual Roll Outcome Scatter Points
    fig.add_trace(
        go.Scatter(
            x=trial_indices[:step+1],
            y=sum_samples[:step+1],
            mode='markers',
            marker=dict(size=6, color=ORANGE_COLOR, opacity=0.5, symbol='circle-open'),
            name="Individual Roll Outcome",
            showlegend=True,
        ),
        row=1, col=2, secondary_y=True # Link to secondary Y-axis
    )
    
    # 5. Highlight the current point on the outcome scatter
    fig.add_trace(
        go.Scatter(
            x=[trial_indices[step]],
            y=[drawn],
            mode='markers',
            marker=dict(size=10, color=ORANGE_COLOR, line=dict(width=2, color='black')),
            name="Current Roll Outcome",
            showlegend=False,
        ),
        row=1, col=2, secondary_y=True
    )

    # --- Axes formatting ---
    
    # Left Plot (PMF) - Removed LaTeX, disabled Y-gridlines
    y_pmf_range = [0, 0.19]
    fig.update_xaxes(title_text="Sum (k)", range=[1.5, 12.5], tickvals=sums, row=1, col=1)
    fig.update_yaxes(title_text="P(Sum=k)", range=y_pmf_range, row=1, col=1, showgrid=False) # Removed gridlines here
    
    # Right Plot (LLN)
    
    # Primary Y-axis (Mean Value) - Pinkish title color
    y_lln_range = [4.5, 9.5] # Mean is 7, range around it
    fig.update_xaxes(title_text="Number of Trials (N)", range=[0, n_trials], row=1, col=2)
    fig.update_yaxes(
        title_text="Mean Value", 
        title_font=dict(color='#ff00cc'), 
        range=y_lln_range, row=1, col=2, secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.3)'
    )
    
    # Secondary Y-axis (Roll Outcome) - Orange title color
    y_outcome_range = [1.5, 12.5] # Roll outcomes 2 through 12
    fig.update_yaxes(
        title_text="Roll Outcome", 
        title_font=dict(color=ORANGE_COLOR), 
        range=y_outcome_range, row=1, col=2, secondary_y=True, showgrid=False
    )
    
    # --- Subtle gridlines and layout
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_layout(
        height=470,
        width=960,
        title_text=f"Law of Large Numbers (LLN) Demonstration: Sum of Two Dice (N={step+1})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        # --- Legend moved to bottom and horizontal ---
        legend=dict(
            x=0.5, y=-0.25, # Moved to bottom
            xanchor='center', yanchor='top',
            orientation='h', # Horizontal
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=15)
        ),
        margin=dict(l=50, r=20, b=110, t=55), # Increased bottom margin for the legend
        barmode='overlay' 
    )

    return fig

# --- Animation frames ---
lln_frames = [
    go.Frame(data=make_lln_convergence_fig(step).data, name=str(step), layout=make_lln_convergence_fig(step).layout)
    for step in range(n_trials)
]

fig = make_lln_convergence_fig(0)
fig.frames = lln_frames

# --- Play button (much faster animation) ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.4, # Moved play button down to account for horizontal legend
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 25, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Empirical Conditional Distribution Convergence
# 
# Conditioning on information may or may not entirely change the overall distribution
# 
# In the case of our dice roll, if we condition on the first die outcome being known already, the overall distribution changes entirely
# 
#  The conditional probability mass function (PMF) for the sum of two dice,
#  given the first die is 5, is:
#  
#  $$
#  \mathbb{P}(X_1 + X_2 = k \mid X_1 = 5) = 
#  \begin{cases}
#  \frac{1}{6}, & k = 6, 7, 8, 9, 10, 11 \\
#  0, & \text{otherwise}
#  \end{cases}
#  $$
#  
#  That is, given the first die is 5, the possible sums are 6 through 11,
#  each occurring with probability $1/6$.
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Dice distribution setup ---
dice_faces = np.arange(1, 7)
sums = np.arange(2, 13)

# Theoretical PMF for sum of two dice (unconditional)
uncond_pmf = np.zeros(11)
for i in dice_faces:
    for j in dice_faces:
        uncond_pmf[i + j - 2] += 1
uncond_pmf = uncond_pmf / uncond_pmf.sum()

# Theoretical PMF for sum of two dice GIVEN first dice is 5 (conditional)
fixed_dice = 5
cond_pmf = np.zeros(11)
for j in dice_faces:
    cond_pmf[fixed_dice + j - 2] += 1
cond_pmf = cond_pmf / cond_pmf.sum()

# --- Simulate sampled rolls: only those where first dice is 5 ---
n_trials = 10000  # Significantly more samples for stronger convergence
np.random.seed(42)
samples_1 = np.random.choice(dice_faces, size=n_trials)
samples_2 = np.random.choice(dice_faces, size=n_trials)
mask = (samples_1 == fixed_dice)
filtered_sum_samples = samples_1[mask] + samples_2[mask]
filtered_len = len(filtered_sum_samples)
frames_n = min(filtered_len, n_trials)  # Limit to available samples

# --- Helper for plotting the conditional sum convergence vs unconditional for *sum of two dice* ---
def make_conditional_vs_unconditional_sum_convergence_fig(step):
    step = min(step, filtered_len - 1)
    drawn = filtered_sum_samples[step]
    counts = np.bincount(filtered_sum_samples[:step+1], minlength=13)[2:13]
    empirical_pmf = np.zeros_like(cond_pmf)
    if step >= 0:
        empirical_pmf = counts / (step + 1)
    empirical_display = empirical_pmf[:len(sums)]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "Conditional vs Unconditional Dice Sum Distribution (First die = 5)",
            "Empirical Convergence (Conditional, First die = 5)"
        )
    )

    # Highlight currently drawn sum on conditional plot (left)
    bar_colors = ['#00bfff'] * len(sums)
    border_colors = ['rgba(0,0,0,0)'] * len(sums)
    border_widths = [0] * len(sums)
    if 2 <= drawn <= 12:
        idx = drawn - 2
        border_colors[idx] = '#FFD700'
        border_widths[idx] = 4

    # --- Left subplot: unconditional (gray, background) and conditional (blue, overlay) ---
    # Unconditional distribution (gray, background)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=uncond_pmf,
            width=0.65,
            marker=dict(color='#aaaaaa', opacity=0.3),
            name="Unconditional PMF (sum of two dice)",
            showlegend=False,
        ),
        row=1, col=1
    )
    # Conditional distribution (colored, for first die=5), with gold border if just drawn
    fig.add_trace(
        go.Bar(
            x=sums,
            y=cond_pmf,
            width=0.65,
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths)
            ),
            name="Conditional PMF (first die=5)",
            showlegend=False,
        ),
        row=1, col=1
    )

    # --- Right subplot: Empirical (from simulation, conditional only) over conditional/theoretical ---
    # Unconditional distribution (gray, background for comparison)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=uncond_pmf,
            width=0.65,
            marker=dict(color='#aaaaaa', opacity=0.22),
            name="Unconditional PMF (sum of two dice)",
            showlegend=False,
            offset=0
        ),
        row=1, col=2
    )
    # Theoretical conditional (faint blue background)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=cond_pmf,
            width=0.65,
            marker=dict(color='#00bfff', opacity=0.25),
            name="Conditional PMF (first die=5)",
            showlegend=False,
            offset=0
        ),
        row=1, col=2
    )
    # Empirical conditional (main)
    empirical_slate_cyan = "#50c4d3"
    fig.add_trace(
        go.Bar(
            x=sums,
            y=empirical_display,
            width=0.65,
            marker=dict(color=empirical_slate_cyan, opacity=0.82),
            name="Empirical (first die=5)",
            showlegend=False,
            offset=0
        ),
        row=1, col=2
    )

    # --- Invisible traces so legend shows correct color coding
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color='#aaaaaa', opacity=0.55),
            name="Unconditional PMF (sum of two dice)",
            showlegend=True
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color='#00bfff', opacity=0.9),
            name="Conditional PMF (first die=5)",
            showlegend=True
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color=empirical_slate_cyan, opacity=0.85),
            name="Empirical (first die=5)",
            showlegend=True
        ),
        row=1, col=2
    )

    # --- Axes formatting
    y_range = [0, 0.22]
    fig.update_xaxes(title_text="Sum of Two Dice", range=[1.5, 12.5], tickvals=sums, row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=y_range, row=1, col=1)
    fig.update_xaxes(title_text="Sum of Two Dice", range=[1.5, 12.5], tickvals=sums, row=1, col=2)
    fig.update_yaxes(title_text="Probability", range=y_range, row=1, col=2)

    # --- Grid/background/layout ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_layout(
        height=470,
        width=960,
        title_text=f"Conditional vs Unconditional Sum Distribution: Two Dice (First die = 5, {frames_n} Trials)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5, y=-0.23,
            xanchor='center', yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=15)
        ),
        margin=dict(l=50, r=20, b=110, t=55),
        barmode='overlay'
    )
    return fig

# --- Animation frames (only as many as valid trials with first die=5) ---
dice_sum_frames = [
    go.Frame(data=make_conditional_vs_unconditional_sum_convergence_fig(step).data, name=str(step))
    for step in range(frames_n)
]

fig = make_conditional_vs_unconditional_sum_convergence_fig(0)
fig.frames = dice_sum_frames

# --- Play button (faster animation) ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.11,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 25, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()



# Formally, 
# $$\frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_i \leq x\,|\,A\}} \xrightarrow{a.s.} P(X \leq x \mid A)$$
# 
# Which is the same convergence implied by the Law of Large Numbers (LLN) but in the conditional sense


# ###### ______________________________________________________________________________________________________________________________________


# ##### Empirical Conditional Probability Convergence
# 
# Since the entire distribution changes when conditioned on new information, probabilities will also change
# 
# Here we observe the conditional probability of the sum being $11$ given the first dice roll is a $5$
# 


import numpy as np
import plotly.graph_objects as go

# Define the colors
SLATE_OFF_CYAN = '#53dee1'  # A cyan-ish slate color

# --- Setup for Conditional Dice Rolls (First Die is 5) ---

dice_faces = np.array([1, 2, 3, 4, 5, 6])
n_trials_anim = 1000

np.random.seed(43)
# Fix die 1 as 5, so sum is 5 + die_2
first_die_val = 5
sample_2 = np.random.choice(dice_faces, size=n_trials_anim)
sum_conditional = first_die_val + sample_2

# Indicator: 1 for a hit (sum = 11), 0 for a miss
is_eleven_cond = (sum_conditional == 11).astype(int)
rolling_prob_cond = np.cumsum(is_eleven_cond) / (np.arange(1, n_trials_anim + 1))

# Theoretical conditional probability: P(sum=11 | first die=5) = P(second die=6) = 1/6
theoretical_prob_11_cond = 1/6  # ~0.1667

# Unconditional probability (for hline)
theoretical_pmf = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36.0
theoretical_prob_11_uncond = theoretical_pmf[11 - 2]  # P(sum = 11) = 2/36 ≈ 0.0556

step_size = 10
frame_duration = 20

steps = np.arange(1, n_trials_anim + 1, step_size)
if steps[-1] != n_trials_anim:
    steps = np.append(steps, n_trials_anim)

# Assign colors: Hit (Bright Green), Miss (Bright Red)
indicator_colors_cond = np.where(is_eleven_cond == 1, '#33ff00', '#ff3300')

initial_rolls = 1
initial_x = np.arange(1, initial_rolls + 1)
initial_y_ind = is_eleven_cond[:initial_rolls]
initial_y_prob = rolling_prob_cond[:initial_rolls]
initial_colors = indicator_colors_cond[:initial_rolls]


fig_cond = go.Figure(
    data=[
        # Trace 1: Indicator Dots (Left Y-Axis)
        go.Scatter(
            x=initial_x,
            y=initial_y_ind,
            mode='markers',
            marker=dict(color=initial_colors, size=7, opacity=0.9, line=dict(width=0)), 
            name="Hit (Green) / Miss (Red)",
            yaxis='y1',
            showlegend=True
        ),
        # Trace 2: Empirical Conditional Probability Line (Right Y-Axis) - SLATE_OFF_CYAN
        go.Scatter(
            x=initial_x,
            y=initial_y_prob,
            mode='lines',
            line=dict(color=SLATE_OFF_CYAN, width=3),
            name='Empirical P(Sum=11 | Die1=5)',
            yaxis='y2'
        ),
        # Trace 3: Theoretical Unconditional Probability Line (Dashed Orange)
        go.Scatter(
            x=[1, n_trials_anim],
            y=[theoretical_prob_11_uncond, theoretical_prob_11_uncond],
            mode='lines',
            line=dict(color='#ffaa00', width=2, dash='dash'),
            name='Unconditional P(Sum=11) ≈ 0.056',
            yaxis='y2'
        ),
        # Trace 4: Theoretical Conditional Probability Line (Solid SLATE_OFF_CYAN)
        go.Scatter(
            x=[1, n_trials_anim],
            y=[theoretical_prob_11_cond, theoretical_prob_11_cond],
            mode='lines',
            line=dict(color=SLATE_OFF_CYAN, width=2, dash='solid'),
            name='Theoretical P(Sum=11 | Die1=5) = 1/6',
            yaxis='y2'
        ),
    ],
    layout=go.Layout(
        width=1000, height=500,
        title=dict(
            text="Convergence: Conditional Probability of Rolling 11 (Given First Die = 5)",
            font=dict(color='white')
        ),
        xaxis=dict(
            title="Number of Rolls (First Die is 5 Only)",
            range=[1, n_trials_anim],
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.24)', 
            color='white',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title="Indicator: Sum=11",
            range=[-0.1, 1.1],
            color="#33ff00",
            side='left',
            tickvals=[0, 1],
            showgrid=False, 
            tickfont=dict(color='white')
        ),
        yaxis2=dict(
            title="Probability",
            range=[0, 0.25],
            overlaying='y',
            side='right',
            color=SLATE_OFF_CYAN, 
            showgrid=False,
            tickfont=dict(color='white')
        ),
        # Legend position across the bottom
        legend=dict(
            x=0.5, y=-0.15, 
            xanchor='center',
            yanchor='top',
            orientation='h', # Horizontal legend
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        # Adjusted margin to make space for the legend and button
        margin=dict(l=50, r=20, b=70, t=55) 
    )
)

# --- 2. Create Animation Frames ---
frames_cond = []
for k in steps:
    current_x = np.arange(1, k + 1)
    frames_cond.append(go.Frame(
        data=[
            go.Scatter(
                x=current_x, y=is_eleven_cond[:k],
                marker=dict(color=indicator_colors_cond[:k], size=7, opacity=0.9, line=dict(width=0))
            ),
            go.Scatter(
                x=current_x, y=rolling_prob_cond[:k],
                line=dict(color=SLATE_OFF_CYAN, width=3), mode='lines'
            ),
            # Unconditional line (Trace 3)
            go.Scatter(
                x=[1, n_trials_anim], y=[theoretical_prob_11_uncond, theoretical_prob_11_uncond],
                line=dict(color='#ffaa00', width=2, dash='dash'), mode='lines'
            ),
            # Conditional line (Trace 4)
            go.Scatter(
                x=[1, n_trials_anim], y=[theoretical_prob_11_cond, theoretical_prob_11_cond],
                line=dict(color=SLATE_OFF_CYAN, width=2, dash='solid'), mode='lines'
            )
        ],
        name=str(k),
        layout=go.Layout(
            title=f"Conditional: Probability of Rolling an 11, Given First Die=5 (Rolls: {k}/{n_trials_anim})"
        )
    ))

fig_cond.frames = frames_cond

# --- Play Button (Slightly Raised) ---
# The y position (-0.21) raises it a bit above the previous -0.28 position
fig_cond.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0, 'y': -0.21,  # Raised slightly compared to previous -0.28
        'xanchor': 'left',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'font': dict(color='white'),
        'showactive': False,
        'buttons': [{
            'label': '▶️ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_cond.show()


# Formally,
# 
# $$\frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_1 + X_2 = 11\,|\,X_1 = 5\}} \xrightarrow{a.s.} P(X_1 + X_2 = 11 \mid X_1 = 5)$$
#  
#  That is, the empirical frequency of the event that the sum is 11 given $X_1=5$ converges almost surely to its true conditional probability by the Law of Large Numbers.


# 
# In other words, the probability of rolling an $11$ increases as we already observed a higher than average value for the first dice roll


# ###### ______________________________________________________________________________________________________________________________________


# ##### Empirical Convergence of Statistics
# 
# Given the probability result above implied by the Law of Large Numbers (LLN) it necessarily precedes that result that all statistics would converge as well


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration and Setup ---
dice_faces = np.arange(1, 7)
sums = np.arange(2, 13)
conditional_value = 5 # Condition: first die is a 5

# --- UNCONDITIONAL Distributions and Means (Sum of two dice) ---
theoretical_counts_unconditional = np.zeros(11)
for i in dice_faces:
    for j in dice_faces:
        theoretical_counts_unconditional[i + j - 2] += 1
theoretical_pmf_unconditional = theoretical_counts_unconditional / theoretical_counts_unconditional.sum()
theoretical_mean_unconditional = np.sum(sums * theoretical_pmf_unconditional) # E[X] = 7.0

# --- CONDITIONAL Distributions and Means (Sum of two dice | First die = 5) ---
theoretical_pmf_conditional = np.zeros(11)
# Rolls are 5+1, 5+2, ..., 5+6 (Sums 6 to 11)
for j in dice_faces:
    theoretical_pmf_conditional[conditional_value + j - 2] += 1/6
theoretical_mean_conditional = 5 + (1+2+3+4+5+6)/6 # E[X | X_1=5] = 8.5

# --- Simulate sampled rolls (for filtering) ---
n_total_trials = 2000 # Total rolls simulated
np.random.seed(42)
samples_1 = np.random.choice(dice_faces, size=n_total_trials)
samples_2 = np.random.choice(dice_faces, size=n_total_trials)
sum_samples = samples_1 + samples_2

# --- Filter for Conditional Rolls (X_1 = 5) ---
conditional_mask = (samples_1 == conditional_value)
conditional_sum_samples = sum_samples[conditional_mask]
n_conditional_trials = len(conditional_sum_samples)

# --- Calculate the running CONDITIONAL empirical mean for LLN plot (Right Chart) ---
cumulative_sum_conditional = np.cumsum(conditional_sum_samples)
conditional_trial_indices = np.arange(1, n_conditional_trials + 1)
empirical_means_conditional = cumulative_sum_conditional / conditional_trial_indices

# --- Plotting helper function ---
def make_conditional_lln_convergence_fig(step_index):
    # Ensure step_index is valid for the conditional data
    step_index = min(step_index, n_conditional_trials - 1)
    
    # Data for LLN (Right)
    current_empirical_mean_conditional = empirical_means_conditional[step_index]
    drawn_conditional = conditional_sum_samples[step_index]
    
    # Data for PMF convergence (Left)
    counts = np.bincount(conditional_sum_samples[:step_index+1], minlength=13)[2:13]
    empirical_pmf_conditional = np.zeros_like(theoretical_pmf_conditional)
    if step_index >= 0:
        empirical_pmf_conditional = counts / (step_index + 1)

    # --- Colors and Constants ---
    ORANGE_DASH = '#ff7f0e' # Unconditional Mean (Reference)
    BLUE_SOLID = '#00bfff' # Conditional Mean (Target)
    PINK_LINE = '#ff00cc' # Conditional Empirical Mean Convergence
    GREEN_OUTCOME = '#2ca02c' # Individual Conditional Roll Outcome
    EMPIRICAL_PMF_COLOR = "#50c4d3" # Conditional Empirical PMF Bars

    # --- Subplot setup ---
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(
            f"Conditional PMF Convergence (First die = {conditional_value}, N={step_index+1})", 
            "Conditional Mean Convergence (LLN)"
        ),
        specs=[[{}, {"secondary_y": True}]]
    )
    
    # #################################################################################
    # --- Left Subplot: PMF Convergence (3 Bar traces: Uncond, Theor Cond, Empir Cond) ---
    # #################################################################################

    # 1. Unconditional Theoretical PMF (Gray, Background)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=theoretical_pmf_unconditional,
            width=0.65,
            marker=dict(color='#aaaaaa', opacity=0.3),
            name="Unconditional PMF P(X)", # Removed LaTeX
            showlegend=True,
            legendgroup='PMF',
        ),
        row=1, col=1
    )
    
    # 2. Conditional Theoretical PMF (Blue, Target Outline)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=theoretical_pmf_conditional,
            width=0.65,
            marker=dict(
                color=BLUE_SOLID, 
                opacity=0.35,
                line=dict(width=1.5, color=BLUE_SOLID)
            ),
            name="Theoretical Conditional PMF P(X | X1=5)", # Removed LaTeX
            showlegend=True,
            legendgroup='PMF',
        ),
        row=1, col=1
    )
    
    # 3. Conditional Empirical PMF (Slate Cyan, Foreground/Convergence)
    fig.add_trace(
        go.Bar(
            x=sums,
            y=empirical_pmf_conditional,
            width=0.65,
            marker=dict(
                color=EMPIRICAL_PMF_COLOR, 
                opacity=0.9,
                line=dict(
                    color=['#FFD700' if i == drawn_conditional else 'rgba(0,0,0,0)' for i in sums], 
                    width=4
                )
            ),
            name="Empirical Conditional PMF",
            showlegend=True,
            legendgroup='PMF',
        ),
        row=1, col=1
    )
    
    # #################################################################################
    # --- Right Subplot: Mean Convergence (LLN) ---
    # #################################################################################
    
    # 4. UNCONDITIONAL Theoretical Mean Line (Reference, Orange Dashed)
    fig.add_trace(
        go.Scatter(
            x=conditional_trial_indices,
            y=[theoretical_mean_unconditional] * n_conditional_trials,
            mode='lines',
            line=dict(color=ORANGE_DASH, width=2, dash='dash'),
            name=f"Unconditional Mean E[X] = {theoretical_mean_unconditional:.2f}", # Removed LaTeX
            showlegend=True,
            legendgroup='Mean'
        ),
        row=1, col=2, secondary_y=False
    )

    # 5. CONDITIONAL Theoretical Mean Line (New Target, Blue Solid)
    fig.add_trace(
        go.Scatter(
            x=conditional_trial_indices,
            y=[theoretical_mean_conditional] * n_conditional_trials,
            mode='lines',
            line=dict(color=BLUE_SOLID, width=3),
            name=f"Conditional Mean E[X | X1=5] = {theoretical_mean_conditional:.2f}", # Removed LaTeX
            showlegend=True,
            legendgroup='Mean'
        ),
        row=1, col=2, secondary_y=False
    )
    
    # 6. CONDITIONAL Empirical Mean Line (Convergence, Pink)
    fig.add_trace(
        go.Scatter(
            x=conditional_trial_indices[:step_index+1],
            y=empirical_means_conditional[:step_index+1],
            mode='lines+markers',
            line=dict(color=PINK_LINE, width=3),
            marker=dict(size=4, color=PINK_LINE),
            name=f"Empirical Conditional Mean ({step_index+1} rolls)",
            showlegend=True,
            legendgroup='Mean'
        ),
        row=1, col=2, secondary_y=False
    )
    
    # 7. Highlight current empirical mean point (No change)
    fig.add_trace(
        go.Scatter(
            x=[conditional_trial_indices[step_index]],
            y=[current_empirical_mean_conditional],
            mode='markers',
            marker=dict(size=12, color=PINK_LINE, line=dict(width=2, color='black')),
            name=f"Current Mean: {current_empirical_mean_conditional:.3f}",
            showlegend=False
        ),
        row=1, col=2, secondary_y=False
    )
    
    # 8. Individual Conditional Roll Outcome Scatter Points (Secondary Y-axis) (No change)
    fig.add_trace(
        go.Scatter(
            x=conditional_trial_indices[:step_index+1],
            y=conditional_sum_samples[:step_index+1],
            mode='markers',
            marker=dict(size=6, color=GREEN_OUTCOME, opacity=0.6, symbol='circle-open'),
            name="Individual Conditional Roll Outcome",
            showlegend=True,
            legendgroup='Outcome'
        ),
        row=1, col=2, secondary_y=True
    )
    
    # 9. Highlight the current point on the outcome scatter (No change)
    fig.add_trace(
        go.Scatter(
            x=[conditional_trial_indices[step_index]],
            y=[drawn_conditional],
            mode='markers',
            marker=dict(size=10, color=GREEN_OUTCOME, line=dict(width=2, color='black')),
            name="Current Roll Outcome",
            showlegend=False,
        ),
        row=1, col=2, secondary_y=True
    )

    # --- Axes formatting ---
    
    y_pmf_range = [0, 0.22]
    # Removed LaTeX: Sum ($k$) -> Sum (k)
    fig.update_xaxes(title_text="Sum (k)", range=[1.5, 12.5], tickvals=sums, row=1, col=1)
    # Removed LaTeX: Probability ($P$) -> Probability (P)
    fig.update_yaxes(title_text="Probability (P)", range=y_pmf_range, row=1, col=1)
    
    y_lln_range = [4.5, 9.5] 
    # Removed LaTeX: Number of Conditional Trials ($N$ | $X_1=5$) -> Number of Conditional Trials (N | X1=5)
    fig.update_xaxes(title_text=f"Number of Conditional Trials (N | X1=5)", range=[0, n_conditional_trials], row=1, col=2)
    fig.update_yaxes(
        title_text="Mean Value", 
        title_font=dict(color=PINK_LINE), 
        range=y_lln_range, row=1, col=2, secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.3)'
    )
    
    y_outcome_range = [1.5, 12.5]
    fig.update_yaxes(
        title_text="Roll Outcome", 
        title_font=dict(color=GREEN_OUTCOME), 
        range=y_outcome_range, row=1, col=2, secondary_y=True, showgrid=False
    )
    
    # --- Layout ---
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    fig.update_layout(
        height=500,
        width=1050,
        title_text=f"Law of Large Numbers (LLN) Demonstration: Conditional Distributions (N={step_index+1})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=16),
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.25, 
            xanchor='center', yanchor='top',
            orientation='h',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color='white', size=15)
        ),
        margin=dict(l=50, r=20, b=110, t=55),
        barmode='overlay' 
    )

    return fig

# --- Animation frames ---
lln_frames = [
    go.Frame(
        data=make_conditional_lln_convergence_fig(step).data, 
        name=str(step), 
        layout=make_conditional_lln_convergence_fig(step).layout
    )
    for step in range(n_conditional_trials)
]

fig = make_conditional_lln_convergence_fig(0)
fig.frames = lln_frames

# --- Play button ---
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.05, # Moved from 0.5 to 0.05 (left corner)
        'y': -0.4,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 25, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()


# Formally,
# 
# $$
# \frac{1}{n}\sum_{i=1}^n X_i \mid X_1 = 5 \xrightarrow{n \to \infty} \mathbb{E}[X\,|\,X_1 = 5]
# $$
# 
# That is, the sample mean must converge to a higher value given the already observed higher value of the first dice.


# ---


# #### 2.) 📊 Markov (Memoryless) Property
# 
# Generally speaking, we typically analyze a *sequence of random variables* or a *stochastic process*, not just one outcome as we have above
# 
# The Markov (memoryless) property is simple: 
# 
# **does *previous* information, before the *current state*, impact the distribution of *new* information**
# 
# $$
# P(X_{n+1} \mid X_1, X_2, \ldots, X_n) = P(X_{n+1} \mid X_n)
# $$
# 
# A *state* can be defined in any way we would like, for the purposes of the analysis herein, we will define the spot value of a process as the current state


# ###### ______________________________________________________________________________________________________________________________________


# ##### Example: Dice Roll Process
# 
# Let now consider a stochastic process, we will roll a dice and sum up the values we realize along the way
# 
#  Let $S_n$ denote the cumulative sum of dice rolls up to the $n^{th}$ roll:
#  
#  $$
#  S_n = X_1 + X_2 + \ldots + X_n \qquad \text{where each } X_i\sim \mathrm{Uniform}\{1,2,3,4,5,6\}
#  $$
#  
#  Here, $\{X_n\}_{n=1}^\infty$ is a sequence of independent, identically distributed random variables representing the outcome of each dice roll.
#  
#  The process $\{S_n\}_{n=1}^\infty$ is a **stochastic process** showing the running total of dice values as we continue to roll.
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the color
SLATE_OFF_CYAN = '#53dee1'

# --- Simulation Setup ---

dice_faces = np.array([1, 2, 3, 4, 5, 6])
n_trials_anim = 10 # Number of rolls

# Simulate 10 rolls of a single die
roll_outcomes = np.random.choice(dice_faces, size=n_trials_anim)

# Calculate the cumulative sum of the outcomes
cumulative_sum = np.cumsum(roll_outcomes)
final_sum = cumulative_sum[-1] # The final sum to highlight

# --- Theoretical Expectation (for comparison) ---
# Expected cumulative sum after k rolls E[S_k] = 3.5 * k
expected_cumulative_sum = 3.5 * np.arange(1, n_trials_anim + 1)

# --- Theoretical Distribution Calculation (S_10) ---
N = n_trials_anim # 10 rolls
# Un-normalized distribution for a single die (sums 1 to 6)
p_single_die = np.array([1, 1, 1, 1, 1, 1]) 

# Compute the distribution for the sum of N dice rolls via repeated convolution
# This is the 10-fold convolution of the single-die distribution
ways_to_sum = p_single_die
for _ in range(N - 1):
    ways_to_sum = np.convolve(ways_to_sum, p_single_die)

# The possible sums range from N to 6*N
possible_sums = np.arange(N, 6 * N + 1)
# Normalize to get probabilities
total_outcomes = 6**N
probabilities = ways_to_sum / total_outcomes

# The index in the probabilities array that corresponds to final_sum
final_sum_index = final_sum - N # Since the array starts at sum N

# --- Animation Configuration ---
step_size = 1
frame_duration = 500

steps = np.arange(1, n_trials_anim + 1, step_size)
if steps[-1] != n_trials_anim:
    steps = np.append(steps, n_trials_anim)

initial_rolls = 1
initial_x = np.arange(1, initial_rolls + 1)
initial_y_sum = cumulative_sum[:initial_rolls]
initial_y_expected = expected_cumulative_sum[:initial_rolls]


# --- 1. Create Initial Figure with Subplots ---
fig_sum = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35], # More space for the cumulative plot
    subplot_titles=("Cumulative Sum of Dice Rolls", "Theoretical Distribution of Sum of 10 Rolls"),
    specs=[[{"type": "scatter"}, {"type": "bar"}]],
    horizontal_spacing=0.08
)

# --- 2a. Left Subplot (Cumulative Sum Plot) ---
# Trace 1: Cumulative Sum (Empirical Data)
fig_sum.add_trace(
    go.Scatter(
        x=initial_x,
        y=initial_y_sum,
        mode='lines+markers',
        line=dict(color=SLATE_OFF_CYAN, width=4),
        marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white')),
        name='Cumulative Sum',
    ),
    row=1, col=1
)

# Trace 2: Expected Cumulative Sum (Theoretical Mean)
fig_sum.add_trace(
    go.Scatter(
        x=[1, n_trials_anim],
        y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Expected Sum (3.5 * N)',
    ),
    row=1, col=1
)

# --- 2b. Right Subplot (Theoretical Distribution) ---
# Bar colors: All grey/white initially (or just for non-highlighted bars)
bar_colors_initial = ['rgba(255, 255, 255, 0.5)'] * len(possible_sums)

# Trace 3: Bar Chart
fig_sum.add_trace(
    go.Bar(
        x=possible_sums,
        y=probabilities,
        marker=dict(color=bar_colors_initial, line=dict(width=1, color='white')),
        name='P(Sum=k)',
        showlegend=False
    ),
    row=1, col=2
)


# --- 3. Update Layout ---
fig_sum.update_layout(
    width=1200, height=600,
    title=dict(
        text=f"Cumulative Sum of Dice Rolls ({n_trials_anim} Rolls) vs. Theoretical Distribution",
        font=dict(color='white'),
        yanchor="top",
        y=0.96  # Lowered from 0.98 to 0.96 (down by 0.02) for 10px lower title
    ),
    # Common Layout
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=14),
    legend=dict(
        x=0.25, y=-0.21,   # moved up from -0.22 to -0.21 (10px higher)
        xanchor='left',
        yanchor='top',
        orientation='h',
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13)
    ),
    margin=dict(l=50, r=20, b=100, t=100)  # t increased from 80 to 100 for 20px padding
)

# Left Subplot (Cumulative Sum) Axis
fig_sum.update_xaxes(
    title_text="Number of Rolls (N)",
    range=[0.5, n_trials_anim + 0.5],
    showgrid=True, 
    gridcolor='rgba(128,128,128,0.24)', 
    color='white',
    tickfont=dict(color='white'),
    tickmode='linear',
    dtick=1,
    row=1, col=1
)

fig_sum.update_yaxes(
    title_text="Cumulative Sum",
    range=[0, n_trials_anim * 6],
    color=SLATE_OFF_CYAN,
    side='left',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=1
)

# Right Subplot (Theoretical Distribution) Axis
fig_sum.update_xaxes(
    title_text="Possible Sum (k)",
    range=[N - 0.5, 6 * N + 0.5],
    showgrid=False, 
    color='white',
    tickfont=dict(color='white'),
    row=1, col=2
)

fig_sum.update_yaxes(
    title_text="Probability P(Sum=k)",
    range=[0, np.max(probabilities) * 1.05], 
    color='white',
    side='right',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=2
)

# Adjust subplot titles font color
fig_sum.layout.annotations[0].update(font=dict(color='white', size=16))
fig_sum.layout.annotations[1].update(font=dict(color='white', size=16))


# --- 4. Create Animation Frames ---
frames_sum = []
for k in steps:
    current_x = np.arange(1, k + 1)
    
    # Check if this is the final frame
    is_final_frame = (k == n_trials_anim)
    
    # Define bar colors for the right subplot
    bar_colors = ['rgba(255, 255, 255, 0.5)'] * len(possible_sums)
    if is_final_frame:
        # Highlight the final sum in the final frame
        bar_colors[final_sum_index] = SLATE_OFF_CYAN # Highlight color
        
    
    frames_sum.append(go.Frame(
        data=[
            # Frame Data for Subplot 1 (Cumulative Sum) - Trace 1
            go.Scatter(
                x=current_x, 
                y=cumulative_sum[:k],
                marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white'))
            ),
            # Frame Data for Subplot 1 (Expected Sum) - Trace 2
            go.Scatter(
                x=[1, n_trials_anim], 
                y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
            ),
            
            # Frame Data for Subplot 2 (Theoretical Distribution) - Trace 3
            go.Bar(
                x=possible_sums,
                y=probabilities,
                marker=dict(color=bar_colors, line=dict(width=1, color='white')),
            )
        ],
        name=str(k),
        # Update the title of the overall figure
        layout=go.Layout(
            title_text=f"Cumulative Sum of Dice Rolls (Rolls: {k}/{n_trials_anim})<br>Last Roll: {roll_outcomes[k-1]} | Current Sum: {cumulative_sum[k-1]}",
        )
    ))

fig_sum.frames = frames_sum

# --- 5. Play Button ---
fig_sum.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0, 'y': -0.15,
        'xanchor': 'left',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'font': dict(color='white'),
        'showactive': False,
        'buttons': [{
            'label': '▶️ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)


# Given we know no information, the unconditional distribution encodes all possible states of the world starting with no dice rolled.
# 
# However, we know that if we have some information and condition on it, it may dramatically change the outcome distribution of interest.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Terminal Distribution Conditional on the Path
# 
# Here, we condition on the first 5 dice rolls and we are interested in the distribution of the path at step 10.
# 
#  $$
#  \mathbb{P}\left(X_{10} = x \;\middle|\; X_1, X_2, X_3, X_4, X_5\right)
#  $$
# 
# Let's see how this changes the terminal distribution. . .


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Define Colors and Constants ---
SLATE_OFF_CYAN = '#53dee1'
HOT_PURPLE = '#FF1493' # Deep Pink/Hot Pink for the conditioned path
N_CONDITION = 5
N_TOTAL = 10

np.random.seed(42)

# --- Simulation Setup ---
dice_faces = np.array([1, 2, 3, 4, 5, 6])
n_trials_anim = N_TOTAL

# Simulate 10 rolls of a single die
roll_outcomes = np.random.choice(dice_faces, size=n_trials_anim)

# Calculate the cumulative sum of the outcomes
cumulative_sum = np.cumsum(roll_outcomes)
S_5 = cumulative_sum[N_CONDITION - 1] # Cumulative sum after 5 rolls
final_sum = cumulative_sum[-1]

# --- Theoretical Expectation (for comparison) ---
expected_cumulative_sum = 3.5 * np.arange(1, n_trials_anim + 1)

# --- Conditional Distribution Calculation (S_10 | S_5=s) ---
N_rem = n_trials_anim - N_CONDITION # 5
p_single_die = np.array([1, 1, 1, 1, 1, 1]) 

# Calculate the distribution for the sum of N_rem=5 dice rolls (Y)
ways_to_Y = p_single_die
for _ in range(N_rem - 1):
    ways_to_Y = np.convolve(ways_to_Y, p_single_die)

# Conditional possible sums for S_10: S_5 + (5 to 30)
possible_Y = np.arange(N_rem, 6 * N_rem + 1)
conditional_possible_sums = S_5 + possible_Y
conditional_probabilities = ways_to_Y / (6**N_rem)

# Final simulated sum (S_10) index in the conditional distribution
final_conditional_index = final_sum - conditional_possible_sums[0] 

# --- Animation Configuration ---
step_size = 1
frame_duration = 500

# Start animation steps from N=5 (k=5)
# This means the "steps" array starts at 5.
steps = np.arange(N_CONDITION, n_trials_anim + 1, step_size)


# --- Initial Plot Data (k=5) ---
initial_rolls = N_CONDITION
initial_x_1 = np.arange(1, initial_rolls + 1)
initial_y_sum_1 = cumulative_sum[:initial_rolls]

# Trace 3 (N>5) starts at N=5
initial_x_3 = [N_CONDITION]
initial_y_sum_3 = [S_5]


# --- 1. Create Initial Figure with Subplots ---
fig_sum = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    subplot_titles=(f"Cumulative Sum of Dice Rolls (Conditioned at N={N_CONDITION})", 
                    f"Conditional Terminal Distribution P(S_10 | S_{N_CONDITION}={S_5})"),
    specs=[[{"type": "scatter"}, {"type": "bar"}]],
    horizontal_spacing=0.08
)

# --- 2a. Left Subplot (Cumulative Sum Plot) ---

# Trace 1: Cumulative Sum (Path 1-5) - HOT_PURPLE (Static in this setup)
fig_sum.add_trace(
    go.Scatter(
        x=initial_x_1,
        y=initial_y_sum_1,
        mode='lines+markers',
        line=dict(color=HOT_PURPLE, width=4),
        marker=dict(size=10, color=HOT_PURPLE, line=dict(width=1, color='white')),
        name='Simulated Path (N≤5)',
    ),
    row=1, col=1
)

# Trace 2: Expected Cumulative Sum (Theoretical Mean) - Dashed White
fig_sum.add_trace(
    go.Scatter(
        x=[1, n_trials_anim],
        y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Expected Sum (3.5 * N)',
    ),
    row=1, col=1
)

# Trace 3: Conditional Cumulative Sum (Path 5-10, starting point) - SLATE_OFF_CYAN
fig_sum.add_trace(
    go.Scatter(
        x=initial_x_3,
        y=initial_y_sum_3,
        mode='markers', # Start with only a marker at N=5
        marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white')),
        name='Simulated Path (N>5)',
        showlegend=True 
    ),
    row=1, col=1
)

# --- 2b. Right Subplot (Conditional Theoretical Distribution) ---

# Trace 4: Bar Chart
bar_colors_initial = ['rgba(255, 255, 255, 0.5)'] * len(conditional_possible_sums)
fig_sum.add_trace(
    go.Bar(
        x=conditional_possible_sums,
        y=conditional_probabilities,
        marker=dict(color=bar_colors_initial, line=dict(width=1, color='white')),
        name='P(S_10|S_5)',
        showlegend=False
    ),
    row=1, col=2
)


# --- 3. Update Layout (Removed LaTeX) ---
fig_sum.update_layout(
    width=1200, height=600,
    title=dict(
        text=f"Conditional Cumulative Sum: S_10 given S_{N_CONDITION}={S_5}",
        font=dict(color='white')
    ),
    # Add the hot purple vertical line (V-line)
    shapes=[
        dict(
            type="line",
            x0=N_CONDITION, y0=0, x1=N_CONDITION, y1=N_TOTAL * 6,
            line=dict(color=HOT_PURPLE, width=3, dash="dot"),
            xref="x1", yref="y1"
        )
    ],
    # Common Layout
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=14),
    legend=dict(
        x=0.07, y=-0.15,   # Move legend 0.02 further right on the x-axis (20px in plot width)
        xanchor='left',
        yanchor='top',
        orientation='h',
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13)
    ),
    margin=dict(l=50, r=20, b=100, t=100) # <-- increased t from 80 to 100 for 20px of padding between title and plots
)

# Left Subplot (Cumulative Sum) Axis
fig_sum.update_xaxes(
    title_text="Number of Rolls (N)",
    range=[0.5, n_trials_anim + 0.5],
    showgrid=True, 
    gridcolor='rgba(128,128,128,0.24)', 
    color='white',
    tickfont=dict(color='white'),
    tickmode='linear',
    dtick=1,
    row=1, col=1
)

fig_sum.update_yaxes(
    title_text="Cumulative Sum",
    range=[0, n_trials_anim * 6],
    color=SLATE_OFF_CYAN,
    side='left',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=1
)

# Right Subplot (Conditional Distribution) Axis
fig_sum.update_xaxes(
    title_text="Possible Terminal Sum (k)",
    range=[conditional_possible_sums.min() - 0.5, conditional_possible_sums.max() + 0.5],
    showgrid=False, 
    color='white',
    tickfont=dict(color='white'),
    row=1, col=2
)

fig_sum.update_yaxes(
    title_text="Conditional Probability P(S_10=k)",
    range=[0, np.max(conditional_probabilities) * 1.05], 
    color='white',
    side='right',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=2
)

# Adjust subplot titles font color (Removed LaTeX)
fig_sum.layout.annotations[0].update(text=f"Cumulative Sum of Dice Rolls (Conditioned at N={N_CONDITION})", font=dict(color='white', size=16))
fig_sum.layout.annotations[1].update(text=f"Conditional Terminal Distribution P(S_10 | S_{N_CONDITION}={S_5})", font=dict(color='white', size=16))


# --- 4. Create Animation Frames (Starting from k=5) ---
frames_sum = []
for k in steps:
    
    # Trace 1: Path 1-5 (Hot Purple) - Static for animation since k >= 5
    x_trace1 = np.arange(1, N_CONDITION + 1)
    y_trace1 = cumulative_sum[:N_CONDITION]
    
    # Trace 3: Path 5-10 (Slate Off Cyan)
    # Path starts at N=5 (S_5) and goes up to k
    x_trace3 = np.arange(N_CONDITION, k + 1)
    y_trace3 = cumulative_sum[N_CONDITION - 1:k]
    mode_trace3 = 'lines+markers' if k > N_CONDITION else 'markers'

    # Trace 4: Conditional Distribution (Highlight Final Sum)
    bar_colors = ['rgba(255, 255, 255, 0.5)'] * len(conditional_possible_sums)
    if k == N_TOTAL:
        # Highlight the final sum in the final frame
        bar_colors[final_conditional_index] = HOT_PURPLE

    
    frames_sum.append(go.Frame(
        data=[
            # Trace 1: Path 1-5
            go.Scatter(
                x=x_trace1, 
                y=y_trace1,
                line=dict(color=HOT_PURPLE, width=4),
                marker=dict(size=10, color=HOT_PURPLE, line=dict(width=1, color='white'))
            ),
            # Trace 2: Expected Sum (Static)
            go.Scatter(
                x=[1, n_trials_anim], 
                y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
            ),
            # Trace 3: Path 5-10
            go.Scatter(
                x=x_trace3,
                y=y_trace3,
                mode=mode_trace3,
                line=dict(color=SLATE_OFF_CYAN, width=4),
                marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white'))
            ),
            # Trace 4: Conditional Distribution
            go.Bar(
                x=conditional_possible_sums,
                y=conditional_probabilities,
                marker=dict(color=bar_colors, line=dict(width=1, color='white')),
            )
        ],
        name=str(k),
        layout=go.Layout(
            # Removed LaTeX: "S_{10}" -> "S_10", etc.
            title_text=f"Rolls: {k}/{N_TOTAL} | Last Roll: {roll_outcomes[k-1]} | Current Sum: {cumulative_sum[k-1]}<br>Conditional Distribution: S_10 given S_{N_CONDITION}={S_5}",
        )
    ))

fig_sum.frames = frames_sum

# --- 5. Play Button ---
fig_sum.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0, 'y': -0.15,
        'xanchor': 'left',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'font': dict(color='white'),
        'showactive': False,
        'buttons': [{
            'label': '▶️ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_sum.show()


# ##### Terminal Distribution Conditional on the Current State
# 
# 
#  Now, we condition only on the current state $X_5$.
#  
#   $$
#   \mathbb{P}\left(X_{10} = x \;\middle|\; X_5\right)
#   $$
#  
#  Let's see how this affects the terminal distribution...


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Define Colors and Constants ---
SLATE_OFF_CYAN = '#53dee1'
HOT_PURPLE = '#FF1493'  # Deep Pink/Hot Pink for the conditioned path
N_CONDITION = 5
N_TOTAL = 10

np.random.seed(42)

# --- Simulation Setup ---
dice_faces = np.array([1, 2, 3, 4, 5, 6])
n_trials_anim = N_TOTAL

# Simulate 10 rolls of a single die
roll_outcomes = np.random.choice(dice_faces, size=n_trials_anim)

# Calculate the cumulative sum of the outcomes
cumulative_sum = np.cumsum(roll_outcomes)
S_5 = cumulative_sum[N_CONDITION - 1]  # Cumulative sum after 5 rolls
final_sum = cumulative_sum[-1]

# --- Theoretical Expectation (for comparison) ---
expected_cumulative_sum = 3.5 * np.arange(1, n_trials_anim + 1)

# --- Conditional Distribution Calculation (S_10 | S_5=s) ---
N_rem = n_trials_anim - N_CONDITION  # 5
p_single_die = np.array([1, 1, 1, 1, 1, 1])
ways_to_Y = p_single_die
for _ in range(N_rem - 1):
    ways_to_Y = np.convolve(ways_to_Y, p_single_die)
possible_Y = np.arange(N_rem, 6 * N_rem + 1)
conditional_possible_sums = S_5 + possible_Y
conditional_probabilities = ways_to_Y / (6 ** N_rem)
final_conditional_index = final_sum - conditional_possible_sums[0]
max_conditional_probability = np.max(conditional_probabilities)

# --- Animation Configuration ---
step_size = 1
frame_duration = 500
steps = np.arange(N_CONDITION, n_trials_anim + 1, step_size)  # Start animation from k=5

# --- Initial Plot Data (k=5) ---
initial_x_1 = [N_CONDITION]
initial_y_sum_1 = [S_5]
initial_x_3 = [N_CONDITION]
initial_y_sum_3 = [S_5]

# --- 1. Create Initial Figure with Subplots ---
fig_sum = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    subplot_titles=(
        f"Cumulative Sum of Dice Rolls (Conditioned at N={N_CONDITION})",
        f"Conditional Terminal Distribution P(S_10 | S_{N_CONDITION}={S_5})"
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}]],
    horizontal_spacing=0.08
)

# --- 2a. Left Subplot (Cumulative Sum Plot) ---
# Trace 1: Cumulative Sum (Path 1-5, but only the marker at N=5) - HOT_PURPLE
fig_sum.add_trace(
    go.Scatter(
        x=initial_x_1,
        y=initial_y_sum_1,
        mode='markers',
        marker=dict(size=15, color=HOT_PURPLE, line=dict(width=2, color='white')),
        name=f'Start Point S_{N_CONDITION}={S_5}',
    ),
    row=1, col=1
)

# Trace 2: Expected Cumulative Sum (Theoretical Mean) - Dashed White
fig_sum.add_trace(
    go.Scatter(
        x=[1, n_trials_anim],
        y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Expected Sum (3.5 * N)',
    ),
    row=1, col=1
)

# Trace 3: Conditional Cumulative Sum (Path 5-10, starting point) - SLATE_OFF_CYAN
fig_sum.add_trace(
    go.Scatter(
        x=initial_x_3,
        y=initial_y_sum_3,
        mode='markers',
        marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white')),
        name='Simulated Path (N>5)',
        showlegend=True
    ),
    row=1, col=1
)

# --- 2b. Right Subplot (Conditional Theoretical Distribution) ---
# Make sure right subplot matches style and logic of context snippet!
bar_colors_conditional = ['rgba(255, 255, 255, 0.5)'] * len(conditional_possible_sums)
# In original @file_context_0, color only changes/highlights at terminal step in animation

fig_sum.add_trace(
    go.Bar(
        x=conditional_possible_sums,
        y=conditional_probabilities,
        marker=dict(
            color=bar_colors_conditional,
            line=dict(width=1, color='white')
        ),
        name=f'P(S_10 | S_{N_CONDITION}={S_5})',
        showlegend=False
    ),
    row=1, col=2
)

# --- 3. Update Layout ---
fig_sum.update_layout(
    width=1200, height=600,
    title=dict(
        text=f"Demonstration of Markov Property: S_10 given S_{N_CONDITION}={S_5}",
        font=dict(color='white'),
        y=0.96  # Move the title up, so plots are shifted 20px down, increasing padding
    ),
    shapes=[
        dict(
            type="line",
            x0=N_CONDITION, y0=0, x1=N_CONDITION, y1=N_TOTAL * 6,
            line=dict(color=HOT_PURPLE, width=3, dash="dot"),
            xref="x1", yref="y1"
        )
    ],
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=14),
    legend=dict(
        x=0.05+0.0258,  # Move right by 20px for width=1200px
        y=-0.15,
        xanchor='left',
        yanchor='top',
        orientation='h',
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13)
    ),
    margin=dict(l=50, r=20, b=100, t=100)  # t=100 vs t=80 adds 20px between title and top of plot
)

# Left Subplot Axis
fig_sum.update_xaxes(
    title_text="Number of Rolls (N)",
    range=[0.5, n_trials_anim + 0.5],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    color='white',
    tickfont=dict(color='white'),
    tickmode='linear',
    dtick=1,
    row=1, col=1
)
fig_sum.update_yaxes(
    title_text="Cumulative Sum",
    range=[0, n_trials_anim * 6],
    color=SLATE_OFF_CYAN,
    side='left',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=1
)

# Right Subplot Axis - as in the reference (no hot purple or shifted coloring unless in animation)
fig_sum.update_xaxes(
    title_text="Possible Terminal Sum (k)",
    range=[conditional_possible_sums.min() - 0.5, conditional_possible_sums.max() + 0.5],
    showgrid=False,
    color='white',
    tickfont=dict(color='white'),
    row=1, col=2
)
fig_sum.update_yaxes(
    title_text="Conditional Probability P(S_10=k)",
    range=[0, max_conditional_probability * 1.05],
    color='white',
    side='right',
    showgrid=True,
    gridcolor='rgba(128,128,128,0.24)',
    tickfont=dict(color='white'),
    row=1, col=2
)

fig_sum.layout.annotations[0].update(
    text=f"Cumulative Sum of Dice Rolls (Conditioned at N={N_CONDITION})",
    font=dict(color='white', size=16)
)
fig_sum.layout.annotations[1].update(
    text=f"Conditional Terminal Distribution P(S_10 | S_{N_CONDITION}={S_5})",
    font=dict(color='white', size=16)
)

# --- 4. Create Animation Frames (Starting from k=5) ---
frames_sum = []
for k in steps:
    # Trace 1: Path 1-5 (Hot Purple marker only) - Static
    x_trace1 = [N_CONDITION]
    y_trace1 = [S_5]

    # Trace 3: Path 5-10 (Slate Off Cyan)
    x_trace3 = np.arange(N_CONDITION, k + 1)
    y_trace3 = cumulative_sum[N_CONDITION - 1:k]
    mode_trace3 = 'lines+markers' if k > N_CONDITION else 'markers'

    # Trace 4 (Conditional Distribution) - As in reference: all bars white except last frame (highlight final)
    bar_colors = ['rgba(255, 255, 255, 0.5)'] * len(conditional_possible_sums)
    if k == N_TOTAL:
        bar_colors[final_conditional_index] = HOT_PURPLE

    frames_sum.append(go.Frame(
        data=[
            # Trace 1: Hot Purple start marker
            go.Scatter(
                x=x_trace1,
                y=y_trace1,
                marker=dict(size=15, color=HOT_PURPLE, line=dict(width=2, color='white'))
            ),
            # Trace 2: Expected Sum (Static)
            go.Scatter(
                x=[1, n_trials_anim],
                y=[expected_cumulative_sum[0], expected_cumulative_sum[-1]],
            ),
            # Trace 3: Path 5-10
            go.Scatter(
                x=x_trace3,
                y=y_trace3,
                mode=mode_trace3,
                line=dict(color=SLATE_OFF_CYAN, width=4),
                marker=dict(size=10, color=SLATE_OFF_CYAN, line=dict(width=1, color='white'))
            ),
            # Trace 4: Conditional Distribution - bars white (final highlighted on last frame)
            go.Bar(
                x=conditional_possible_sums,
                y=conditional_probabilities,
                marker=dict(color=bar_colors, line=dict(width=1, color='white')),
            ),
        ],
        name=str(k),
        layout=go.Layout(
            title_text=f"Rolls: {k}/{N_TOTAL} | Last Roll: {roll_outcomes[k-1]} | Current Sum: {cumulative_sum[k-1]}<br>Conditional Distribution: S_10 given S_{N_CONDITION}={S_5}",
        )
    ))

fig_sum.frames = frames_sum

# --- 5. Play Button ---
fig_sum.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0, 'y': -0.15,
        'xanchor': 'left',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'font': dict(color='white'),
        'showactive': False,
        'buttons': [{
            'label': '▶️ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': frame_duration, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_sum.show()


# We get the same terminal distribution - this is the idea of the Markov property
# 
# $$
#  \mathbb{P}\left(X_{10} = x \;\middle|\; X_1, X_2, X_3, X_4, X_5\right) = \mathbb{P}\left(X_{10} = x \;\middle|\; X_5\right)
#  $$
# 
# In other words, if we condition on the entire path or the current state value (the dice roll sum at step 5) we get the same conditional distribution
# 
# *In more other words*, we don't need the entire path!
# 
# All the information necessary to determine the conditional distribution is decided by the one value at the current time step!


# ---


# #### 3.) 📈 Trading Implications
# 
# **Are markets Markovian?  Plenty of evidence suggests not. . .**
# 
# $$
# P(X_{n+1} \mid X_1, X_2, \ldots, X_n) \neq P(X_{n+1} \mid X_n)
# $$
# 
# For you Ph.D.s watching or reading:
# 
# **Remark:** We can make a process "Markovian" by expanding the state space, but if we need to expand to the entire path of a process (or processes) we are not taking advantage of the Markov property anymore, it's just path dependent.
# 
# ###### ______________________________________________________________________________________________________________________________________
# *Jacqueline, A. and Lekeufack, R. (2020). Path-dependent volatility models.*
# - Proves the existence and uniqueness of solutions for Path-Dependent Volatility (PDV) models, arguing they provide accurate fits to market data where Markovian models fail.
# 
# *Guyon, J. (2023). Volatility is (mostly) path-dependent.*
# - Synthesizes empirical evidence demonstrating that realized volatility depends on the historical price path, reinforcing that path-dependence is a primary characteristic of volatility.
# 
# *Foschi, G. and Pascucci, A. (2025). Linking Path-Dependent and Stochastic Volatility Models.*
# - Shows that the statistical properties of a process produced by virtually any Stochastic Volatility (SV) model can be exactly reproduced by a Path-Dependent Volatility (PDV) model.
# 
# *Baldovin, F., Bovina, D., Camana, F., and Stella, A. L. (2009). Modeling the non-Markovian, non-stationary scaling dynamics of financial markets.*
# - Concludes that the dynamics driving high-frequency exchange rates are best described by a non-Markovian, self-similar process due to the empirical presence of memory in price movements.
# 
# *Wang, S., Wang, R., and Chen, B. (2015). How Volatilities Nonlocal in Time Affect the Price Dynamics in Complex Financial Systems.*
# - Provides empirical evidence that past volatilities "nonlocal in time" (lasting over two weeks) affect future returns, explicitly violating the Markovian assumption.
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Example: Trading System Statistics are Misleading by Implicit Dependence
# 
# Why does this matter?  It impacts our backtests and interpretation of them, plenty of naive traders fall into this when they first get started. . .


import numpy as np
import plotly.graph_objects as go

# --- Simulate a single Brownian motion asset path ---
np.random.seed(7)
n_steps = 200
dt = 1 / 252
mu = 0.08    # drift
sigma = 0.23 # volatility
S0 = 100

# Standard Brownian increments
Z = np.random.randn(n_steps)
increments = mu * dt + sigma * np.sqrt(dt) * Z
prices = S0 * np.exp(np.cumsum(increments))
times = np.arange(n_steps)

# Choose two entry points (arbitrary signals, for demonstration)
entry_idx1, entry_idx2 = 60, 120

fig_path = go.Figure(
    data=[
        # Asset price path
        go.Scatter(
            x=times, y=prices,
            mode='lines',
            line=dict(color=SLATE_OFF_CYAN, width=3),
            name="Asset Price"
        ),
        # Entry signal 1 (vertical line)
        go.Scatter(
            x=[entry_idx1, entry_idx1],
            y=[prices.min()*0.99, prices.max()*1.01],
            mode='lines',
            line=dict(color='#ffaa00', width=2, dash='dash'),
            name='Entry Signal 1',
            showlegend=True
        ),
        # Entry signal 2 (vertical line)
        go.Scatter(
            x=[entry_idx2, entry_idx2],
            y=[prices.min()*0.99, prices.max()*1.01],
            mode='lines',
            line=dict(color='#ff3300', width=2, dash='dash'),
            name='Entry Signal 2',
            showlegend=True
        ),
    ],
    layout=go.Layout(
        width=900,
        height=420,
        title=dict(
            text="Sample Asset Price Path",
            font=dict(color='white')
        ),
        xaxis=dict(
            title="Time Step",
            color='white',
            showgrid=True,
            gridcolor='rgba(120,180,220,0.16)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title="Price",
            color='white',
            showgrid=True,
            gridcolor='rgba(120,180,220,0.10)',
            tickfont=dict(color='white')
        ),
        legend=dict(
            x=0.03, y=0.97,
            xanchor='left',
            yanchor='top',
            orientation='v',
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=15),
        margin=dict(l=45, r=30, t=55, b=40)
    )
)

fig_path.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Empirical Trading Strategy P/L Distribution
# 
# There is negative expectancy, so we shouldn't trade it . . . *is this right?*


import plotly.graph_objects as go
import numpy as np

# --- Simulate two different-shaped theoretical P/L distributions ---

# Settings
sim_n = 10_000

# Distribution 1: Skewed Normal, positive expected P/L (right skew, positive mean)
from scipy.stats import skewnorm
skew_1 = 7
mean_1 = 4
std_1 = 6
pl1 = skewnorm.rvs(a=skew_1, loc=mean_1, scale=std_1, size=sim_n)
exp_pl1 = np.mean(pl1)

# Distribution 2: Bimodal, negative expected P/L
centers_2 = [-16, 3]
weights_2 = [0.65, 0.35]
stds_2 = [5, 2]
choices = np.random.choice([0,1], p=weights_2, size=sim_n)
pl2 = np.random.normal(
    loc=np.take(centers_2, choices),
    scale=np.take(stds_2, choices)
)
exp_pl2 = np.mean(pl2)

# Combine both distributions into one array
pl_all = np.concatenate([pl1, pl2])
exp_pl_all = np.mean(pl_all)

# Plot a single histogram in blue
hist_fig = go.Figure()

hist_fig.add_trace(
    go.Histogram(
        x=pl_all,
        nbinsx=60,
        marker_color="#3388ff",
        name='Combined P/L',
        opacity=0.92,
        showlegend=False,
    )
)

# Add vertical line for expected P/L (mean)
hist_fig.add_shape(
    dict(
        type='line',
        x0=exp_pl_all, x1=exp_pl_all,
        y0=0, y1=len(pl_all)//8,
        line=dict(color='white', width=3, dash="dash"),
    )
)

hist_fig.add_annotation(
    dict(
        x=exp_pl_all,
        y=len(pl_all)//7,
        xref='x',
        yref='y',
        text=f"Expected P/L: {exp_pl_all:,.2f}",
        showarrow=True,
        arrowcolor="#3388ff",
        font=dict(color="white", size=13),
        bgcolor="rgba(20,20,20,0.63)",
        ax=80 if exp_pl_all > 0 else -80,
        ay=-30
    )
)

hist_fig.update_xaxes(
    title_text="Simulated P/L ($)",
    color='white',
    showgrid=True,
    gridcolor='rgba(120,180,220,0.13)',
    tickfont=dict(color='white')
)
hist_fig.update_yaxes(
    title_text="Frequency",
    color='white',
    showgrid=True,
    gridcolor='rgba(120,180,220,0.11)',
    tickfont=dict(color='white')
)

hist_fig.update_layout(
    width=650,
    height=420,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=15),
    margin=dict(l=40, r=35, t=80, b=45),
    title=dict(
        text="Trading Strategy P/L Distribution",
        font=dict(color='white', size=20),
        y=0.97,
        x=0.5,
        xanchor='center'
    ),
    bargap=0.02
)

hist_fig.show()



# ###### ______________________________________________________________________________________________________________________________________


#  **The analysis above is incorrect,**
# 
#  (*A little*) more formally,
#  
#  $$ 
#  \text{Let PL}_1 = S_T - S_{t_1},\quad \text{PL}_2 = S_T - S_{t_2}
#  $$
#  
#  $$ 
#  \mathbb{P}(\text{PL}_1 \leq x \mid \mathcal{F}_{t_1}) \neq \mathbb{P}(\text{PL}_2 \leq x \mid \mathcal{F}_{t_2})
#  $$
#  
#  The distribution of P/L depends on the path up to each signal, so trading signals at $t_1$ and $t_2$ do not yield the same outcome distribution.
# 
# 


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Simulate a single Brownian motion asset path ---
np.random.seed(7)
n_steps = 200
dt = 1 / 252
mu = 0.08    # drift
sigma = 0.23 # volatility
S0 = 100

# Standard Brownian increments
Z = np.random.randn(n_steps)
increments = mu * dt + sigma * np.sqrt(dt) * Z
prices = S0 * np.exp(np.cumsum(increments))
times = np.arange(n_steps)

# Choose two entry points (arbitrary signals, for demonstration)
entry_idx1, entry_idx2 = 60, 120

# Define the colors (use SLATE_OFF_CYAN if you defined it previously, otherwise set a similar color)
try:
    cyan_col = SLATE_OFF_CYAN
except NameError:
    cyan_col = "#3dd1cc" # fallback

# --- Create both charts stacked vertically ---
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=(
        "Asset Price Path (Orange up to Entry 1)",
        "Asset Price Path (Red up to Entry 2)"
    )
)

# -- Top chart: orange up to entry_idx1, blue after --
fig.add_trace(
    go.Scatter(
        x=times[:entry_idx1+1],
        y=prices[:entry_idx1+1],
        mode='lines',
        line=dict(color='#ffaa00', width=3),
        name="Asset Path (Pre Entry 1)"
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=times[entry_idx1:],
        y=prices[entry_idx1:],
        mode='lines',
        line=dict(color=cyan_col, width=3),
        showlegend=False,
        name="Asset Path (Post Entry 1)"
    ),
    row=1, col=1
)
# Entry signal 1 vertical
fig.add_trace(
    go.Scatter(
        x=[entry_idx1, entry_idx1],
        y=[prices.min()*0.99, prices.max()*1.01],
        mode='lines',
        line=dict(color='#ffaa00', width=2, dash='dash'),
        name='Entry Signal 1'
    ),
    row=1, col=1
)

# -- Bottom chart: red up to entry_idx2, blue after --
fig.add_trace(
    go.Scatter(
        x=times[:entry_idx2+1],
        y=prices[:entry_idx2+1],
        mode='lines',
        line=dict(color='#ff3300', width=3),
        name="Asset Path (Pre Entry 2)"
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=times[entry_idx2:],
        y=prices[entry_idx2:],
        mode='lines',
        line=dict(color=cyan_col, width=3),
        showlegend=False,
        name="Asset Path (Post Entry 2)"
    ),
    row=2, col=1
)
# Entry signal 2 vertical
fig.add_trace(
    go.Scatter(
        x=[entry_idx2, entry_idx2],
        y=[prices.min()*0.99, prices.max()*1.01],
        mode='lines',
        line=dict(color='#ff3300', width=2, dash='dash'),
        name='Entry Signal 2'
    ),
    row=2, col=1
)

# Style both axes and background
fig.update_layout(
    width=900,
    height=750,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=15),
    margin=dict(l=45, r=30, t=90, b=45),
    title=dict(
        text="Sample Asset Price Paths with Entry Signals",
        font=dict(color='white')
    ),
    legend=dict(
        x=0.02, y=0.99,
        xanchor='left',
        yanchor='top',
        orientation='v',
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13)
    ),
)

fig.update_xaxes(
    title_text="Time Step",
    color='white',
    showgrid=True,
    gridcolor='rgba(120,180,220,0.16)',
    tickfont=dict(color='white')
)
fig.update_yaxes(
    title_text="Price",
    color='white',
    showgrid=True,
    gridcolor='rgba(120,180,220,0.10)',
    tickfont=dict(color='white')
)

fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### Theoretically Simulating Each Trade P/L and Producing the Conditional Distribution Yields Different Distributions


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Simulate two different-shaped theoretical P/L distributions ---

# Settings
sim_n = 10_000

# Distribution 1: Skewed Normal, positive expected P/L (right skew, positive mean)
from scipy.stats import skewnorm
skew_1 = 7
mean_1 = 4
std_1 = 6
pl1 = skewnorm.rvs(a=skew_1, loc=mean_1, scale=std_1, size=sim_n)
exp_pl1 = np.mean(pl1)

# Distribution 2: Bimodal, negative expected P/L
centers_2 = [-16, 3]
weights_2 = [0.65, 0.35]
stds_2 = [5, 2]
choices = np.random.choice([0,1], p=weights_2, size=sim_n)
pl2 = np.random.normal(
    loc=np.take(centers_2, choices),
    scale=np.take(stds_2, choices)
)
exp_pl2 = np.mean(pl2)

# --- Plot P/L Distributions Side by Side ---

hist_fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.13,
    subplot_titles=(
        "P/L Distribution: Entry 1 (orange, positive mean)",
        "P/L Distribution: Entry 2 (red, negative mean)"
    )
)

# Entry 1: orange, positive mean/expectation
hist_fig.add_trace(
    go.Histogram(
        x=pl1,
        nbinsx=60,
        marker_color="#ffaa00",
        name='Entry 1',
        opacity=0.97,
        showlegend=False,
    ),
    row=1, col=1
)
hist_fig.add_shape(
    dict(
        type='line',
        x0=exp_pl1, x1=exp_pl1,
        y0=0, y1=sim_n//12,
        line=dict(color='white', width=3, dash="dash"),
    ),
    row=1, col=1
)
hist_fig.add_annotation(
    dict(
        x=exp_pl1,
        y=sim_n//11,
        xref='x1',
        yref='y1',
        text=f"Expected P/L: {exp_pl1:,.2f}",
        showarrow=True,
        arrowcolor="#ffaa00",
        font=dict(color="white", size=13),
        bgcolor="rgba(20,20,20,0.63)",
        ax=60,
        ay=-30
    )
)

# Entry 2: red, negative mean/expectation
hist_fig.add_trace(
    go.Histogram(
        x=pl2,
        nbinsx=60,
        marker_color="#ff3300",
        name='Entry 2',
        opacity=0.97,
        showlegend=False,
    ),
    row=1, col=2
)
hist_fig.add_shape(
    dict(
        type='line',
        x0=exp_pl2, x1=exp_pl2,
        y0=0, y1=sim_n//12,
        line=dict(color='white', width=3, dash="dash"),
    ),
    row=1, col=2
)
hist_fig.add_annotation(
    dict(
        x=exp_pl2,
        y=sim_n//11,
        xref='x2',
        yref='y2',
        text=f"Expected P/L: {exp_pl2:,.2f}",
        showarrow=True,
        arrowcolor="#ff3300",
        font=dict(color="white", size=13),
        bgcolor="rgba(20,20,20,0.63)",
        ax=-50,
        ay=-30
    )
)

for i in [1, 2]:
    hist_fig.update_xaxes(
        row=1, col=i,
        title_text="Simulated P/L ($)",
        color='white',
        showgrid=True,
        gridcolor='rgba(120,180,220,0.13)',
        tickfont=dict(color='white')
    )
    hist_fig.update_yaxes(
        row=1, col=i,
        title_text="Frequency" if i==1 else "",
        color='white',
        showgrid=True,
        gridcolor='rgba(120,180,220,0.11)',
        tickfont=dict(color='white')
    )

hist_fig.update_layout(
    width=1000,
    height=420,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=15),
    margin=dict(l=40, r=35, t=80, b=45),
    title=dict(
        text="Example: Simulated P/L Distributions with Different Shapes & Expected Values",
        font=dict(color='white', size=20),
        y=0.98,
        x=0.5,
        xanchor='center'
    ),
    bargap=0.02
)

hist_fig.show()



# ###### ______________________________________________________________________________________________________________________________________


# **In practice, how do we handle this?**
# 
# We can't just treat each signal entry as producing the same P/L distribution (sometimes we can if the structural inefficiency we are trading is quite stable, a topic for another day)
# 
# In reality we must
# 
#     A.) Compress the path representation into a reasonable state variable to proxy the path dependence
# 
#     B.) Expand the state space to capture more path dependent dynamics
# 
#     C.) Develop more intricate models to capture the explicit path dependent dynamics
# 
# Of course, the problem here is we aren't just implicitly conditioning on the information of the asset price path, we are implicitly conditioning on all of the information available in our filtration: volume dynamics, volatility dynamics, liquidity, so on and so forth making modeling path dependency explicitly likely computationally intractible implying we instead need better models to capture this variation so we have reasonably stable P/L distributions.


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# 
# - Unconditional statistics are vastly different than conditional statistics impacting the overall distribution, probabilities, or statistics of the random variable or process of interest
# - The Markov (memoryless) property suggests that the information necessary to construct the (correct) conditional distribution is available in the current state of a random process
# - If the Markov property is true, all information before time the current time *t* wouldn't be useful at all - for example, the information of historic pricing data would be compressed into the price of the asset *right now* making it redundant if we were to include it in any model as the conditional distribution is fully decided by the current price
# - In practice, processes are *not Markovian* and the current state alone doesn't perfectly model the conditional distribution directly - we need to include other information to develop a more accurately model
# - If we neglect this idea of path dependency (non-Markovian processes) we will be fooling ourselves or misinforming ourselves via backtests and other statistical procedures as everything we do is implicitly filtration and path-dependent as we saw in our trading strategy P/L distributions example
# - In reality, we have to account for these path-dependent dynamics by building more intricate models capable of compressing the state space into reasonable representation (for example, low-mid-high volatility, gaining-losing momentum, increased-decreased social attention, so on and so forth) to produce better approximations of the conditional distributions we ultimately trade
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - My First Quant Resume
# - Projects that Made me a Quant
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Quant Roadmap: How I would Study if I Had to Start Over
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Reinforcement Learning for Delta Hedging
# - Approximating Pricing Functionals using Neural Networks
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Volatility Regime Classifier


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

