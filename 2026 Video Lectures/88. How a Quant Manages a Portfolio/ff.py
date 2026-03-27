# ### 📈 How a Quant Manages a Portfolio
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
# ##### [👾 Join the Quant Guild Discord Server](discord.com/invite/MJ4FU2c6c3)
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
# #### 1.) 📉 Quantitative Portfolio Theory
# 
# - Market, Industry, Idiosyncratic
# 
# - Covariance and Correlation Structures (Statistics Change Over Time)
# 
# - Diversification and Systematic Risk (Correlations Increase During Distress)
# 
# #### 2.) 📊 Spectral Decomposition of Select Assets
# 
# - Linear Compression of Common Variation
# 
# - Loadings (Eigenvectors) and Interpretations
# 
# #### 3.) 🎯 Capital Asset Pricing Model (CAPM)
# 
# - Capturing and Explaining Common Variation
# 
# - CAPM: Alpha and Beta
# 
# - The Reality of the Model (Extensions)
# 
# #### 4.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 📉 Different Facets of Risk
# 
# ##### Market, Industry, and Idiosyncratic Risk
# 
# Consider a portfolio comprised of the following stocks with equal weight...
# 
#  | **Tech** | **Healthcare** | **Consumer Staples** |
#  |----------|---------------|----------------------|
#  | AAPL     | UNH           | WMT                  |
#  | MSFT     | JNJ           | COST                 |
#  | AVGO     | AMGN          | PG                   |
# 
# ##### Let's begin by visualizing each sector...


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Colors
sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Line styles (cycling)
line_styles = ['solid', 'dash', 'dot']

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        # Load and standardize
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Grab close
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found. Check dates/files.")

dates = df_all.index

# Normalize to 100
df_norm = (df_all / df_all.iloc[0]) * 100

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=list(sector_map.keys()),
    horizontal_spacing=0.05
)

trace_configs = []

# --- 4. Add Initial Traces ---
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    color = sector_colors[sector]
    
    for i, ticker in enumerate(tickers):
        style = line_styles[i % len(line_styles)]
        
        fig.add_trace(go.Scatter(
            x=[dates[0]], 
            y=[df_norm[ticker].iloc[0]],
            mode='lines',
            name=ticker,
            line=dict(color=color, width=2.5, dash=style),
            legendgroup=sector,
            showlegend=True
        ), row=1, col=col_idx)
        
        trace_configs.append({'ticker': ticker, 'col': col_idx})

# --- 5. Animation Frames ---
num_frames = 60
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    for conf in trace_configs:
        ticker = conf['ticker']
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[ticker][:idx],
            mode='lines'
        ))
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700, # Increased height to accommodate the larger bottom margin
    width=1000,
    
    # Large bottom margin to fit Legend AND Button without overlap
    margin=dict(t=100, b=250), 
    
    # --- Legend: Positioned just below the X-axis ---
    showlegend=True,
    legend=dict(
        orientation='h',         # Horizontal
        x=0.5, xanchor='center', # Centered
        y=-0.15, yanchor='top',  # Just below the plots
        font=dict(color=off_white, size=12),
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)'
    ),
    
    # --- Play Button: Pushed significantly further down ---
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.35, 'xanchor': 'center', # Moves button way down
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play 2024-Present', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- 7. Y-Axis Limits (As Requested) ---

# Tech (Auto/Data Driven)
tech_min = df_norm[sector_map['Tech']].min().min() * 0.95
tech_max = df_norm[sector_map['Tech']].max().max() * 1.05
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
fig.update_yaxes(range=[tech_min, tech_max], title_text="Wealth Index", row=1, col=1)

# Healthcare (40 to 160)
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(range=[40, 160], row=1, col=2)

# Staples (90 to 250)
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=3)
fig.update_yaxes(range=[90, 250], row=1, col=3)

# Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=16)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Idiosyncratic (Firm Specific) Risk


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
tickers = ['UNH', 'JNJ', 'AMGN']
sector_color = '#2ca02c' # Green
line_styles = ['solid', 'dash', 'dot']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# Normalize to 100
df_norm = (df_all / df_all.iloc[0]) * 100
# Calculate Returns
df_pct = ((df_all / df_all.iloc[0]) - 1) * 100

# --- 3. Plotly Construction ---
fig = go.Figure()

# --- 4. Add Initial Traces ---
for i, ticker in enumerate(tickers):
    style = line_styles[i % len(line_styles)]
    
    # Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[ticker].iloc[0]],
        mode='lines',
        name=ticker,
        line=dict(color=sector_color, width=3, dash=style),
        legendgroup=ticker
    ))
    
    # Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[ticker].iloc[0]],
        mode='markers+text',
        name=ticker,
        marker=dict(color=sector_color, size=10),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=ticker
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for ticker in tickers:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[ticker][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_norm[ticker].iloc[idx]
        current_ret = df_pct[ticker].iloc[idx]
        label_text = f" {ticker}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    title=dict(
        text="<b>Healthcare Sector Idiosyncratic Risk Analysis</b><br><sup>Divergence of Returns (2024-Present)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Risk Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- KEY FIX: Extend X-Axis ---
# We calculate a buffer date 180 days (approx 6 months) past the last data point
end_date_buffer = dates[-1] + pd.Timedelta(days=180)

fig.update_xaxes(
    range=[dates[0], end_date_buffer], # Force the range to include empty space
    title_text="Date"
)

fig.update_yaxes(
    range=[40, 160], 
    title_text="Wealth Index (Start=100)"
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Industry (Sector) Risk


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

line_styles = ['solid', 'dash', 'dot']
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# Normalize to 100 (Wealth Index)
df_norm = (df_all / df_all.iloc[0]) * 100

# Calculate Sector Composites (Equally Weighted)
for sector, tickers in sector_map.items():
    df_norm[f"{sector}_Composite"] = df_norm[tickers].mean(axis=1)

# --- 3. Generate Titles with Performance ---
titles = []
for sector in sector_map.keys():
    # Calculate Total Return: (End Value - 100)
    final_val = df_norm[f"{sector}_Composite"].iloc[-1]
    total_ret = final_val - 100
    titles.append(f"{sector} | {total_ret:+.1f}%")

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=titles,
    horizontal_spacing=0.05
)

trace_configs = []

# --- 5. Add Initial Traces ---
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    base_color = sector_colors[sector]
    
    # A. Individual Stocks (Faded, Thin)
    for i, ticker in enumerate(tickers):
        style = line_styles[i % len(line_styles)]
        
        fig.add_trace(go.Scatter(
            x=[dates[0]], 
            y=[df_norm[ticker].iloc[0]],
            mode='lines',
            name=ticker,
            opacity=0.4,  # Faded
            line=dict(color=base_color, width=1, dash=style), # Thin
            showlegend=False
        ), row=1, col=col_idx)
        
        trace_configs.append({'name': ticker, 'col': col_idx, 'type': 'stock'})

    # B. Composite Index (Solid, Medium Thickness)
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[f"{sector}_Composite"].iloc[0]],
        mode='lines',
        name=f"{sector} Index",
        opacity=1.0,
        line=dict(color=base_color, width=3), # Thinner than before (was 5)
        showlegend=False
    ), row=1, col=col_idx)
    
    trace_configs.append({'name': f"{sector}_Composite", 'col': col_idx, 'type': 'composite'})

# --- 6. Animation Frames ---
num_frames = 60
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    
    for conf in trace_configs:
        name = conf['name']
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[name][:idx],
            mode='lines'
        ))
            
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 7. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=100, b=100), # Reduced bottom margin since legend is gone
    
    showlegend=False, # Global disable
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Sector Correlation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- 8. Y-Axis Limits ---
# Tech
tech_min = df_norm[sector_map['Tech']].min().min() * 0.95
tech_max = df_norm[sector_map['Tech']].max().max() * 1.05
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
fig.update_yaxes(range=[tech_min, tech_max], title_text="Wealth Index", row=1, col=1)

# Healthcare
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(range=[40, 160], row=1, col=2)

# Staples
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=3)
fig.update_yaxes(range=[90, 250], row=1, col=3)

# Style Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=16)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Market Risk


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728', # Red
    'Composite Portfolio': '#ffffff' # White (Replaces SPY)
}

# Load only the sector stocks (Removed SPY)
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Create Portfolios ---
df_norm = (df_all / df_all.iloc[0]) * 100
df_plot = pd.DataFrame(index=dates)

# A. Calculate Sector Portfolios
for sector, tickers in sector_map.items():
    df_plot[sector] = df_norm[tickers].mean(axis=1)

# B. Calculate Composite Portfolio (Average of the 3 Sector Indices)
# This creates an equally weighted benchmark of your sectors
df_plot['Composite Portfolio'] = df_plot[list(sector_map.keys())].mean(axis=1)

# Calculate Returns for Labels
df_pct = ((df_plot / 100) - 1) * 100

# --- 4. Plotly Construction ---
fig = go.Figure()

# Helper for style
def get_style(name):
    color = sector_colors.get(name, '#ffffff')
    # Make Composite line slightly thicker for emphasis
    width = 4 if name == 'Composite Portfolio' else 2.5
    return dict(color=color, width=width, dash='solid')

# Add Initial Traces (Static)
for name in df_plot.columns:
    # 1. Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='lines',
        name=name,
        line=get_style(name),
        legendgroup=name
    ))
    
    # 2. Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='markers+text',
        name=name,
        marker=dict(color=get_style(name)['color'], size=8),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=name
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for name in df_plot.columns:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_plot[name][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_plot[name].iloc[idx]
        current_ret = df_pct[name].iloc[idx]
        label_text = f" {name}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Annotations ---
off_white = "#e0e0e0"
off_black = "#222222"

# Define the two vertical red lines (Dates from your snippet)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-03-01", y0=0, x1="2025-03-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-20", y0=0, x1="2025-04-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Systematic Risk Analysis</b><br><sup>Sector Performance vs. Composite Portfolio (Equal Weighted)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    # Add the red lines here
    shapes=drawdown_lines,
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Risk Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis
end_date_buffer = dates[-1] + pd.Timedelta(days=180)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-axis
y_min = df_plot.min().min() * 0.95
y_max = df_plot.max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title_text="Wealth Index (Start=100)")

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Covariance and Correlation Structures
# 
# Before we talk about diversification to reduce or eliminate different facets of risk we must discuss covariance and correlation.
# 
# These are statistics that tell us what *tends to happen* to one variable when another increases (it tells us how they tend to move together).
# 
# Covariance is defined as:
# 
# $$\mathrm{Cov}(X,Y) = \mathbb{E}\left[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\right]$$
# 
# Correlation is defined as normalized covariance:
# 
#  $$
#  \mathrm{Corr}(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [0, 1]
#  $$
# 
# Let's analyze the correlation of stocks in each sector...


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Map sectors to Plotly built-in sequential colorscales
# Tech -> Blue, Healthcare -> Green, Staples -> Red
color_scales = {
    'Tech': 'Blues',
    'Healthcare': 'Greens',
    'Consumer Staples': 'Reds'
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns for Correlation
df_returns = df_all.pct_change().dropna()

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=list(sector_map.keys()),
    horizontal_spacing=0.10
)

for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    # 1. Slice returns for this sector
    valid_tickers = [t for t in tickers if t in df_returns.columns]
    sector_rets = df_returns[valid_tickers]
    
    # 2. Calculate Correlation Matrix
    corr_matrix = sector_rets.corr()
    
    # 3. Add Heatmap
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=color_scales[sector],
        zmin=0, zmax=1,
        showscale=False, # Hide individual colorbars
        text=corr_matrix.values,
        texttemplate="%{z:.2f}", # Show values with 2 decimals
        # Removed the problematic 'textfont' argument to use default black/readable text
    ), row=1, col=col_idx)

# --- 4. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Intra-Sector Correlation Matrix</b><br><sup>(2024-Present)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1100,
    margin=dict(t=120, b=50),
)

# Force square aspect ratio
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.show()


# ##### Let's See What Happens with Stocks (AVGO, AMGN, WMT) from Different Sectors


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
tickers = ['AVGO', 'AMGN', 'WMT']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_returns = df_all.pct_change().dropna()

# --- 3. Calculate Correlation ---
corr_matrix = df_returns.corr()

# --- 4. Plotly Construction ---
fig = go.Figure()

fig.add_trace(go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='Viridis', # A good scale for mixed correlations
    zmin=-1, zmax=1,      # Range from -1 to 1 for full correlation spectrum
    text=corr_matrix.values,
    texttemplate="%{z:.2f}",
    showscale=True,
    colorbar=dict(title="Correlation")
))

# --- 5. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Cross-Sector Correlation Matrix</b><br><sup>AVGO (Tech) vs. AMGN (Healthcare) vs. WMT (Staples)</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=700,
    margin=dict(t=120, b=50),
)

# Force square aspect ratio
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### The Elephant in the Room
# 
# Statistics in the real world do not converge, we can not apply the Law of Large Numbers (LLN) or the Central Limit Theorem (CLT).
# 
# This means all statistics (mean, variance, skewness, kurtosis, covariance, correlation, . . .) will all change over time.


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# 2 Weeks = approx 10 trading days
window_size = 30 
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2025-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change()

# --- 3. Build Indices ---
sector_rets = pd.DataFrame(index=df_rets.index)

# A. Create Sector Indices (Equal Weight of their stocks)
for sector, tickers in sector_map.items():
    sector_rets[sector] = df_rets[tickers].mean(axis=1)

# B. Create Grand Market Index (Equal Weight of the 3 Sectors)
# This serves as the "All-Sector" benchmark to test correlation against
sector_rets['Grand_Index'] = sector_rets[list(sector_map.keys())].mean(axis=1)

# --- 4. Calculate Rolling Correlation ---
df_corr = pd.DataFrame(index=sector_rets.index)

for sector in sector_map.keys():
    # Rolling correlation of Sector vs. Grand Index
    df_corr[sector] = sector_rets[sector].rolling(window=window_size).corr(sector_rets['Grand_Index'])

# Drop NaNs from the start
df_corr.dropna(inplace=True)
dates = df_corr.index

# --- 5. Plotly Construction ---
fig = go.Figure()

# Add Initial Traces (Static)
for sector in sector_map.keys():
    color = sector_colors[sector]
    
    # Line (Thinner width=1.5)
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_corr[sector].iloc[0]],
        mode='lines',
        name=sector,
        line=dict(color=color, width=1.5),
        legendgroup=sector
    ))
    
    # Floating Label
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_corr[sector].iloc[0]],
        mode='markers+text',
        name=sector,
        marker=dict(color=color, size=6),
        text=[f"{df_corr[sector].iloc[0]:.2f}"],
        textposition="middle right",
        textfont=dict(color="white", size=13, family="monospace"),
        showlegend=False,
        legendgroup=sector
    ))

# --- 6. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for sector in sector_map.keys():
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_corr[sector][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_corr[sector].iloc[idx]
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[f" {sector}: {current_val:.2f}"]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 7. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

# Define the Vertical Red Lines (Drawdown Markers)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-04-20", y0=0, x1="2025-04-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-01", y0=0, x1="2025-04-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Sector Correlation Dynamics</b><br><sup>Rolling 2-Week Correlation: Sector vs. Aggregate Portfolio</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1200,
    margin=dict(t=100, b=150, r=80),
    
    # Add vertical lines
    shapes=drawdown_lines,
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.15, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Sector Correlations', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis to see the future lines and label text
end_date_buffer = dates[-1] + pd.Timedelta(days=90)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-Axis (-1 to 1, though usually positive for sectors)
fig.update_yaxes(range=[-0.5, 1.1], title_text="Correlation to Aggregate Portfolio")

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Diversification and Systematic Risk
# 
# You can diversify away industry and idiosyncratic risk, that means you won't be compensated for it if a sector or stock goes up!
# 
# This *does not mean* you will be compensated in the cross-section (if you don't know what this means, think *on average*) for bearing this risk (lottery ticket effect)


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# Window for rolling correlation (approx 3 months)
window_size = 63 

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Calculations ---
# A. Daily Returns
df_rets = df_all.pct_change()

# B. Construct Portfolio Return (Equal Weight of 9 stocks)
stock_cols = [c for c in df_rets.columns if c != 'SPY']
df_rets['Portfolio'] = df_rets[stock_cols].mean(axis=1)

# C. Rolling Correlation (Portfolio vs SPY)
rolling_corr = df_rets['Portfolio'].rolling(window=window_size).corr(df_rets['SPY'])
rolling_corr.fillna(0, inplace=True)

# D. Wealth Indices (Start=100)
df_norm = pd.DataFrame(index=dates)
df_norm['Portfolio'] = (1 + df_rets['Portfolio']).cumprod() * 100
df_norm['SPY'] = (1 + df_rets['SPY']).cumprod() * 100

# Align all data
df_plot = df_norm.join(rolling_corr.rename('Corr'))
df_plot = df_plot.iloc[window_size:] 
dates_plot = df_plot.index

# --- Calculate Final Performance for Legend ---
spy_ret = (df_plot['SPY'].iloc[-1] / df_plot['SPY'].iloc[0] - 1) * 100
port_ret = (df_plot['Portfolio'].iloc[-1] / df_plot['Portfolio'].iloc[0] - 1) * 100

spy_name = f"SPY Benchmark ({spy_ret:+.1f}%)"
port_name = f"Eq. Wgt Portfolio ({port_ret:+.1f}%)"

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3], 
    vertical_spacing=0.1,
    subplot_titles=("<b>Performance: Portfolio vs. SPY</b>", "<b>Rolling Correlation (60-Day)</b>")
)

# --- Initial Traces (Static) ---
# Top Chart: SPY (Gray)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['SPY'].iloc[0]],
    mode='lines', name=spy_name,
    line=dict(color='#888888', width=2),
    legendgroup='perf'
), row=1, col=1)

# Top Chart: Portfolio (White)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['Portfolio'].iloc[0]],
    mode='lines', name=port_name,
    line=dict(color='white', width=3),
    legendgroup='perf'
), row=1, col=1)

# Bottom Chart: Rolling Correlation (Yellow)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['Corr'].iloc[0]],
    mode='lines', name='Correlation',
    line=dict(color='#FFD700', width=2), # Gold
    fill='tozeroy', 
    fillcolor='rgba(255, 215, 0, 0.1)',
    showlegend=False
), row=2, col=1)

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates_plot) // num_frames)
indices = list(range(1, len(dates_plot), step))
if indices[-1] != len(dates_plot)-1:
    indices.append(len(dates_plot)-1)

frames = []
for idx in indices:
    frame_data = [
        # 1. SPY Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['SPY'][:idx]),
        # 2. Portfolio Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['Portfolio'][:idx]),
        # 3. Correlation Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['Corr'][:idx]),
    ]
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1100, 
    
    # Legend Position: Outside Right
    showlegend=True,
    legend=dict(
        orientation='v', 
        y=0.5, yanchor='middle',
        x=1.02, xanchor='left',
        font=dict(color=off_white, size=12)
    ),
    
    margin=dict(t=80, b=100, r=150),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

# Styling Axes
axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), title_font=dict(color=off_white)
)
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Set Ranges
# Top Chart
y_min = df_plot[['Portfolio', 'SPY']].min().min() * 0.95
y_max = df_plot[['Portfolio', 'SPY']].max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title="Wealth Index", row=1, col=1)
fig.update_xaxes(range=[dates_plot[0], dates_plot[-1]], row=1, col=1)

# Bottom Chart
fig.update_yaxes(range=[-0.5, 1.1], title="Correlation", row=2, col=1)
fig.update_xaxes(range=[dates_plot[0], dates_plot[-1]], title="Date", row=2, col=1)

# Style Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=14)

fig.show()


# ##### This is exactly what it means to diversify, effectively we are just left with market risk as we can see above
# 
# **We beat the market above and generated excess return**, however, **THIS IS NOT ALPHA** as we will discuss...
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Trivially, we are in academia, told that we "can never diversify away market risk"
# 
# This is true to an extent, we saw above that correlations rise during distress, even if something once was not highly correlated with the market it may begin to exhibit this feature
# 
# However, there are quantitative means of reducing overall market exposure.  This is where quantitative portfolio management and models like CAPM, Fama-French, and extensions come in.


# ---


# #### 2.) 📊 Spectral Decomposition of Select Assets
# 
# ##### Linear Compression of Common Variation
# 
# Let's talk about some more advanced quantitative techniques for expressing this market factor...
# 
# **Eigendecomposition of the Covariance Matrix (or SVD of Data Matrix)**
# 
# - $X$ is our $n \times p$ data matrix of $n$ observations and $p$ assets.
# 
# - $X$ is mean-centered before computation.
# 
# - $\Sigma = \frac{1}{n-1} X^\top X$ is the sample covariance matrix.
# 
# - $\Sigma \mathbf{v} = \lambda \mathbf{v}$ gives us eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$.
# 
# - Eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_p$ are the principal directions.
# 
# - Eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$ indicate variance explained by each principal component.
# 
# - Projecting original data onto first $k$ eigenvectors gives us first $k$ principal components.
# 


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# --- 3. Spectral Decomposition (PCA) ---
# Calculate Daily Log Returns (Standard for PCA)
df_rets = np.log(df_all / df_all.shift(1)).dropna()

# Compute Correlation Matrix
corr_matrix = df_rets.corr()

# Eigendecomposition
# eigh is for symmetric matrices (like correlation matrices), returns sorted eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

# Sort descending (eigh returns ascending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate Explained Variance
total_var = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# --- 4. Plotly Construction ---
fig = go.Figure()

# A. Scree Plot (Bar Chart of Eigenvalues)
fig.add_trace(go.Bar(
    x=[f"PC{i+1}" for i in range(len(eigenvalues))],
    y=eigenvalues,
    name='Eigenvalue',
    marker_color='#00d1ff', # Cyan
    opacity=0.7
))

# B. Cumulative Variance (Line Chart)
# Map to secondary Y-axis? Or just show % on hover.
# Standard scree plot plots Eigenvalues. 
# Sometimes people plot Explained Variance % instead.
# Let's plot Explained Variance % (Scree) and Cumulative % (Line).
# The user asked for "Spectral Decay", which usually refers to the Eigenvalues themselves.
# I will stick to Eigenvalues on the left, but maybe add text for %.

# Let's do a Dual Axis: Left=Eigenvalue, Right=Cumulative %
fig.add_trace(go.Scatter(
    x=[f"PC{i+1}" for i in range(len(eigenvalues))],
    y=cumulative_variance_ratio,
    name='Cumulative Variance',
    mode='lines+markers',
    marker=dict(color='white', size=8),
    line=dict(color='white', width=2),
    yaxis='y2'
))

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Spectral Decomposition of Portfolio Returns</b><br><sup>Eigenvalue Decay (Scree Plot)</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=100, b=50, r=50),
    showlegend=True,
    legend=dict(
        orientation='h',
        y=1.02, x=0.5, xanchor='center'
    ),
    
    # Axes
    xaxis=dict(
        title="Principal Component (Mode)",
        showgrid=False,
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white)
    ),
    yaxis=dict(
        title="Eigenvalue Magnitude",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white)
    ),
    yaxis2=dict(
        title="Cumulative Explained Variance",
        overlaying='y',
        side='right',
        range=[0, 1.1],
        tickformat='.0%',
        showgrid=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )
)

# Add Marchenko-Pastur Threshold (Optional, but cool for "Spectral Decay")
# Only if requested, but standard spectral decay is just the scree plot.
# I'll keep it simple.

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### Let's Analyze the Loadings (Eigenvectors) on Each of the First Three Principal Components 


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Color Mapping for Bars
color_map = {
    'AAPL': '#1f77b4', 'MSFT': '#1f77b4', 'AVGO': '#1f77b4', # Blue
    'UNH': '#2ca02c', 'JNJ': '#2ca02c', 'AMGN': '#2ca02c',   # Green
    'WMT': '#d62728', 'COST': '#d62728', 'PG': '#d62728'     # Red
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Run PCA ---
# Standardize returns (Mean=0, Std=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_rets)

# Fit PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)

# Extract Loadings (Eigenvectors)
loadings = pd.DataFrame(
    pca.components_.T, 
    index=df_rets.columns, 
    columns=['PC1', 'PC2', 'PC3']
)

# Explained Variance
exp_var = pca.explained_variance_ratio_ * 100

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=(
        f"<b>PC1 (Market)</b><br>Var Explained: {exp_var[0]:.1f}%", 
        f"<b>PC2 (Sector Factor A)</b><br>Var Explained: {exp_var[1]:.1f}%", 
        f"<b>PC3 (Sector Factor B)</b><br>Var Explained: {exp_var[2]:.1f}%"
    ),
    horizontal_spacing=0.08
)

# Helper to add bars
def add_pc_bars(col_name, col_idx):
    # Sort values for cleaner visualization in the specific PC
    # (Optional: remove sort_values if you want fixed order)
    # subset = loadings[col_name].sort_values(ascending=False)
    subset = loadings[col_name] # Keep fixed order to compare across plots easily
    
    colors = [color_map[t] for t in subset.index]
    
    fig.add_trace(go.Bar(
        x=subset.index,
        y=subset.values,
        marker_color=colors,
        text=subset.values,
        texttemplate="%{text:.2f}",
        textposition="auto",
        showlegend=False
    ), row=1, col=col_idx)

# Add traces for PC1, PC2, PC3
add_pc_bars('PC1', 1)
add_pc_bars('PC2', 2)
add_pc_bars('PC3', 3)

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Principal Component Loadings</b><br><sup>Decomposing Risk Factors: Market vs. Sector Exposure</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1000,
    margin=dict(t=120, b=50),
)

# Y-Axis Labels
fig.update_yaxes(title="Loading Value", row=1, col=1)

# Style Annotations
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=14)

# Add a dummy legend for sectors manually (since we disabled auto-legend)
# This is just a visual trick to show the color key
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#1f77b4', size=10), name='Tech'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#2ca02c', size=10), name='Healthcare'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#d62728', size=10), name='Staples'))

fig.update_layout(
    showlegend=True,
    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### What About the Remaining Variance
# 
# That's typically considered *everything else* including the idiosyncratic components of firm specific returns.
# 
# Let's observe how much variation is left unexplained for each individual stock...


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Color Mapping
sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Calculate Idiosyncratic Risk via PCA ---
# Standardize returns so variance = 1.0 for all stocks
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_rets)

# Fit PCA with 3 Components (Market + Sector Factors)
# We assume the first 3 factors capture the "Systematic" risk
pca = PCA(n_components=3)
components = pca.fit_transform(scaled_data)

# Reconstruct the data using ONLY the systematic factors
reconstructed_data = pca.inverse_transform(components)

# The "Error" between original and reconstructed is the Idiosyncratic part
residuals = scaled_data - reconstructed_data

# Calculate Variance of the residuals
# Since original variance was standardized to 1.0, this result is directly the % Idio
idio_variance = np.var(residuals, axis=0)

# Create DataFrame for Plotting
df_idio = pd.DataFrame({
    'Ticker': df_rets.columns,
    'Idio_Pct': idio_variance * 100,
    'Sector': [next(k for k, v in sector_map.items() if t in v) for t in df_rets.columns]
})

# Sort for better visuals (optional)
df_idio = df_idio.sort_values('Idio_Pct', ascending=False)

# --- 4. Plotly Construction ---
fig = go.Figure()

for sector, color in sector_colors.items():
    subset = df_idio[df_idio['Sector'] == sector]
    
    fig.add_trace(go.Bar(
        x=subset['Ticker'],
        y=subset['Idio_Pct'],
        name=sector,
        marker_color=color,
        text=subset['Idio_Pct'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        textfont=dict(color='white', size=14, weight='bold')
    ))

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Idiosyncratic Risk Analysis</b><br><sup>% of Stock Variance Unexplained by Market & Sector Factors</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=120, b=50),
    
    xaxis=dict(
        title="Stock Ticker",
        title_font=dict(size=16),
        tickfont=dict(size=14)
    ),
    
    yaxis=dict(
        title="Idiosyncratic Variance (%)",
        title_font=dict(size=16),
        tickfont=dict(size=14),
        range=[0, 100] # Full percentage scale
    ),
    
    legend=dict(
        orientation="h", 
        y=-0.2, x=0.5, xanchor="center",
        font=dict(size=14)
    ),
    
    # Add an annotation explaining the metric
    annotations=[dict(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text="Higher % = More 'Unique' Movement (Earnings, News, Product Launches)",
        showarrow=False,
        font=dict(color="gray", size=12)
    )]
)

fig.show()


# ##### Clearly, by the Spectral Analysis Above, there is Common Variation in a **Linear** Capacity
# 
# Is there now a way that we can develop a model (or a family of models) knowing this information?
# 
# Let's now discuss the Capital Asset Pricing Model (CAPM).


# ---


# #### 3.) 🎯 Capital Asset Pricing Model (CAPM)
# 
# $R_i = R_f + \beta_i (R_m - R_f) + \varepsilon_i$
#  Where:
#  - $R_i$ : Expected return of asset $i$
#  - $R_f$ : Risk-free rate
#  - $\beta_i$ : Sensitivity of asset $i$ to the market
#  - $R_m$ : Expected return of the market
#  - $\varepsilon_i$ : Idiosyncratic error term (asset-specific shock)
# 
# 
# 
# ##### Capturing and Explaining Common Variation
# 
# We can easily see if our portfolio is trading the components explaining variation or not by running a regression in the CAPM framework.
# 
# CAPM is just one of many frameworks, in practice we use more intricate extensions...


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# --- 1. Data Setup ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Combine lists including SPY
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Tech", "Healthcare", "Consumer Staples"),
    horizontal_spacing=0.08,
    x_title="Market Return (SPY)",
    y_title="Sector Return"
)

# Iterate through sectors to Calculate & Plot CAPM
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    color = sector_colors[sector]
    
    # Construct Sector Portfolio Returns (Equal Weight)
    valid_tickers = [t for t in tickers if t in df_rets.columns]
    if not valid_tickers: continue
        
    sector_series = df_rets[valid_tickers].mean(axis=1)
    market_series = df_rets['SPY']
    
    # Linear Regression (scipy.stats)
    slope, intercept, r_value, p_value, std_err = stats.linregress(market_series, sector_series)
    beta = slope
    alpha = intercept
    
    # Generate Regression Line Points (for plotting the line)
    x_range = np.linspace(market_series.min(), market_series.max(), 100)
    y_pred = alpha + beta * x_range
    
    # 1. Scatter Plot (Actual Data)
    fig.add_trace(go.Scatter(
        x=market_series,
        y=sector_series,
        mode='markers',
        name=f"{sector} Data",
        marker=dict(color=color, opacity=0.4, size=5),
        showlegend=False
    ), row=1, col=col_idx)
    
    # 2. Regression Line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f"{sector} Fit",
        line=dict(color='white', width=2, dash='solid'),
        showlegend=False
    ), row=1, col=col_idx)
    
    # 3. Annotation (The Equation)
    # Determine correct axis reference names for annotation positioning
    # (Plotly names them x, x2, x3... y, y2, y3...)
    axis_suffix = "" if col_idx == 1 else str(col_idx)
    xref = f"x{axis_suffix} domain"
    yref = f"y{axis_suffix} domain"
    
    equation_text = (
        f"<b>{sector}</b><br>"
        f"y = {alpha:.5f} + {beta:.2f}x<br>"
        f"----------------<br>"
        f"<b>Beta: {beta:.2f}</b><br>"
        f"Alpha: {alpha:.5f}"
    )
    
    # Place annotation inside the plot (top-left corner)
    fig.add_annotation(
        x=0.05, y=0.95,
        xref=xref, yref=yref,
        text=equation_text,
        showarrow=False,
        align="left",
        font=dict(color="white", size=12, family="monospace"),
        bgcolor="rgba(0,0,0,0.6)", # Semi-transparent background
        bordercolor=color,
        borderwidth=1
    )

# --- 4. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>CAPM Regression Analysis</b><br><sup>Sector Returns vs. SPY (2024-Present) | Isolating Alpha & Beta</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1200,
    margin=dict(t=100, b=50, l=60),
    showlegend=False
)

# Update axis styling (Zero lines help visualize Alpha/Beta)
fig.update_xaxes(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'
)
fig.update_yaxes(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'
)

fig.show()


# ##### Notice, the $\beta$ Coefficient Dictates How Aggresively it Depends on the Market


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728', # Red
    'Market (SPY)': '#ffffff' # White
}

# Add SPY to the loading list
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Create Portfolios ---
df_norm = (df_all / df_all.iloc[0]) * 100
df_plot = pd.DataFrame(index=dates)

# Sector Portfolios
for sector, tickers in sector_map.items():
    df_plot[sector] = df_norm[tickers].mean(axis=1)

# Market Benchmark
if 'SPY' in df_norm.columns:
    df_plot['Market (SPY)'] = df_norm['SPY']

# Calculate Returns for Labels
df_pct = ((df_plot / 100) - 1) * 100

# --- 4. Plotly Construction ---
fig = go.Figure()

# Helper for style
def get_style(name):
    color = sector_colors.get(name, '#ffffff')
    # Reduced thickness to 2.5, all solid lines
    return dict(color=color, width=2.5, dash='solid')

# Add Initial Traces (Static)
for name in df_plot.columns:
    # 1. Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='lines',
        name=name,
        line=get_style(name),
        legendgroup=name
    ))
    
    # 2. Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='markers+text',
        name=name,
        marker=dict(color=get_style(name)['color'], size=8),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=name
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for name in df_plot.columns:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_plot[name][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_plot[name].iloc[idx]
        current_ret = df_pct[name].iloc[idx]
        label_text = f" {name}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Annotations ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    title=dict(
        text="<b>Systematic Risk Analysis</b><br><sup>Sector Performance vs. Broad Market (SPY)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Market Risk Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis
end_date_buffer = dates[-1] + pd.Timedelta(days=180)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-axis
y_min = df_plot.min().min() * 0.95
y_max = df_plot.max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title_text="Wealth Index (Start=100)")

fig.show()


# ##### There is Literally **No Alpha**! Even if a Sector Outpreforms the Market, we Needed the Market to Preform Otherwise we Couldn't Outpreform
# 
# Effectively, if we are trading *$\beta$* we need that facet of risk (in the case of CAPM, the market) to preform otherwise we will lose money.
# 
# Generating excess return is not *$\alpha$*, the number of people I have heard state *"excess return is alpha"* is tremendously concerning...
# 
# $\alpha$ is literally the orthogonal component of our returns not generated by the market or other priced risk factors.  
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Most Important Takeaway in the Context of "Trading Alpha" 
# 
# In the sense of our sprectral decomposition above, we would be trading the *leftover* variation, not the compressed variation!


# ###### ______________________________________________________________________________________________________________________________________


# ##### Trading Alpha vs. Beta
# 
# If we are trading an "alpha" we are trading a *mispricing* or a structural inefficiency.
# 
# One example, and I know many hedge funds that do this, is selling volatility.  This is just premium capture on the VRP.
# 
# In essence, if you are trading alpha, you don't care what happens to the broader market, it won't impact your trading strategy/portfolio value.


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (2025 Bear Market Simulation) ---
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='B')
n = len(dates)

# Market: Target ~-30% annual return (Bear Market / Recession)
target_annual_return = -0.30
trading_days = 252
# Calculate daily geometric return for -30% annualized
m_mu = (1 + target_annual_return)**(1/trading_days) - 1
m_vol = 0.012   # keep same vol as before for one year

market_rets = np.random.normal(m_mu, m_vol, n)

# Target Sharpe (Annual 2.5 -> Daily 2.5 / sqrt(252))
target_sharpe = 2.5
daily_target = target_sharpe / np.sqrt(252)

# --- LEFT PORTFOLIO: High Beta (1.5), *NO Alpha* ---
# This portfolio should lose more than the market in a bear market.
beta_left_target = 1.5

# Set alpha = 0, so expected mean is just beta * market_mu
# Mean_p = beta * market_mu (which is deeply negative)
# Std_p = sqrt((beta * market_vol)^2 + sigma_noise^2)
noise_vol = 0.002
rets_left = (beta_left_target * market_rets) + np.random.normal(0, noise_vol, n)

# --- RIGHT PORTFOLIO: Beta 0.0, Sharpe 2.5 (Market Neutral Alpha) ---
raw_noise = np.random.normal(0, 1, n)
p2_std = 0.01 
p2_mean = daily_target * p2_std
rets_right = p2_mean + (raw_noise - np.mean(raw_noise)) / np.std(raw_noise) * p2_std

# Helper to calculate realized metrics
def get_realized_sharpe(rets):
    return (np.mean(rets) / np.std(rets)) * np.sqrt(252)

def capm_beta_alpha(port_rets, market_rets):
    # Regression: port_rets = alpha + beta * market_rets + residuals
    X = market_rets
    y = port_rets
    X = np.vstack([np.ones_like(X), X]).T  # add intercept for alpha
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = beta_hat[0]
    beta = beta_hat[1]
    return beta, alpha

s_left = get_realized_sharpe(rets_left)
s_right = get_realized_sharpe(rets_right)
beta_left, alpha_left = capm_beta_alpha(rets_left, market_rets)
beta_right, alpha_right = capm_beta_alpha(rets_right, market_rets)

# Convert alpha (daily) to annualized percent return for reporting
alpha_left_ann_pct = ( (1 + alpha_left) ** 252 - 1 ) * 100
alpha_right_ann_pct = ( (1 + alpha_right) ** 252 - 1 ) * 100

# Cumulative Wealth (Starting at 100)
df = pd.DataFrame({
    'Date': dates,
    'Market': np.cumprod(1 + market_rets) * 100,
    'Port_Left': np.cumprod(1 + rets_left) * 100,
    'Port_Right': np.cumprod(1 + rets_right) * 100
})

# --- 2. Plotly Construction ---
title_l = (
    f"Bear Market vs. High Beta Portfolio | Previous Sharpe: 2.65"
)
title_r = (
    f"Bear Market vs. Market Neutral Alpha | Previous Sharpe: 2.5"
)

fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(title_l, title_r),
    horizontal_spacing=0.1
)

# Colors and Styling
off_white = "#e0e0e0"
market_color = "#888888"
port_color = "#00d1ff"
play_blue = "#00bfff"
play_darkblue = "#143c99"  # darker blue for play button text

# Plot initialization
for col_idx, (market_col, port_col) in enumerate([('Market', 'Port_Left'), ('Market', 'Port_Right')], start=1):
    show_leg = (col_idx == 2)  # legend on the 2nd subplot
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[market_col].iloc[0]],
        line=dict(color=market_color, width=1.5),
        name="Market",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[port_col].iloc[0]],
        line=dict(color=port_color, width=2.5),
        name="Portfolio",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)

# --- 3. Animation Frames ---
num_frames = 60
indices = np.linspace(1, n-1, num_frames, dtype=int)
frames = []
for idx in indices:
    frames.append(go.Frame(
        data=[
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Left'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Right'][:idx], mode='lines')
        ],
        name=str(idx)
    ))
fig.frames = frames

# --- 4. Layout & Transparency ---
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    margin=dict(t=120, b=120),
    showlegend=True,
    legend=dict(
        orientation='v',
        # Place the legend at the bottom right of the 2nd subplot
        # The 2nd subplot's x-domain is the right half, usually 0.5 to 1.0, so x=1.0 is bottom right
        x=0.995, xanchor='right', y=0.08, yanchor='bottom',
        font=dict(color=off_white, size=15),
        bgcolor='rgba(0,0,0,0.55)',
        bordercolor='rgba(0,0,0,0)'
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.18, 'xanchor': 'center',
        'font': {'color': play_darkblue},
        'buttons': [{'label': '▶ Play 2025 Bear Market Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)
# Set all axes to off-white
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Suitable fixed y-axes for drawdown (bear market)
fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=1
)
fig.update_yaxes(
    range=[50, 120],
    title_text="Wealth Index (Initial=100)", 
    row=1, col=1
)

fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=2
)
fig.update_yaxes(
    range=[65, 155],
    row=1, col=2
)

# Style titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=13)

fig.show()


# ###### ______________________________________________________________________________________________________________________________________


# ##### The Elephant in the Room (Again)
# 
# $\beta$ in our regression is a *statistic*, we know statistics don't converge in reality.
# 
# Effectively, the regression we ran above to show the betas of each portfolio was wrong in so many ways.  
# 
# There isn't just one beta, it changes over time.  In fact, beta is likely to be exacerbated (increase) during distress like we saw with correlations.


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Data Setup ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Combine lists including SPY
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change()

# --- 3. Build Indices & Calculate Rolling Beta ---
sector_rets = pd.DataFrame(index=df_rets.index)
window_size = 30 # 30 Days

# A. Create Sector Indices (Equal Weight)
for sector, ticks in sector_map.items():
    valid_ticks = [t for t in ticks if t in df_rets.columns]
    sector_rets[sector] = df_rets[valid_ticks].mean(axis=1)

# B. Market Index
market_rets = df_rets['SPY']

# C. Calculate Rolling Beta
# Beta = Cov(Sector, Market) / Var(Market)
df_beta = pd.DataFrame(index=sector_rets.index)
rolling_market_var = market_rets.rolling(window=window_size).var()

for sector in sector_map.keys():
    rolling_cov = sector_rets[sector].rolling(window=window_size).cov(market_rets)
    df_beta[sector] = rolling_cov / rolling_market_var

# Drop initial NaNs from window
df_beta.dropna(inplace=True)
dates_plot = df_beta.index

# --- 4. Plotly Construction ---
fig = go.Figure()

# Add Initial Traces (Static)
for sector in sector_map.keys():
    color = sector_colors[sector]
    
    # Line
    fig.add_trace(go.Scatter(
        x=[dates_plot[0]], 
        y=[df_beta[sector].iloc[0]],
        mode='lines',
        name=sector,
        line=dict(color=color, width=2),
        legendgroup=sector
    ))
    
    # Floating Label
    fig.add_trace(go.Scatter(
        x=[dates_plot[0]], 
        y=[df_beta[sector].iloc[0]],
        mode='markers+text',
        name=sector,
        marker=dict(color=color, size=6),
        text=[f"{df_beta[sector].iloc[0]:.2f}"],
        textposition="middle right",
        textfont=dict(color="white", size=13, family="monospace"),
        showlegend=False,
        legendgroup=sector
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates_plot) // num_frames)
indices = list(range(1, len(dates_plot), step))
if indices[-1] != len(dates_plot)-1:
    indices.append(len(dates_plot)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates_plot[idx]
    
    for sector in sector_map.keys():
        # Update Line
        frame_data.append(go.Scatter(
            x=dates_plot[:idx], 
            y=df_beta[sector][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_beta[sector].iloc[idx]
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[f" {sector}: {current_val:.2f}"]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

# Vertical Red Lines (Drawdown Markers)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-03-20", y0=0, x1="2025-03-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-01", y0=0, x1="2025-04-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Rolling Sector Beta (30-Day)</b><br><sup>Sensitivity to SPY Benchmark Over Time</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1200,
    margin=dict(t=100, b=150, r=80),
    shapes=drawdown_lines,
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.15, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Beta Dynamics', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

# Axis Styling
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- KEY UPDATE: Extend X-axis to Aug 2026 ---
target_end_date = pd.Timestamp('2026-08-31')
fig.update_xaxes(range=[dates_plot[0], target_end_date], title_text="Date")

# Y-Axis
fig.update_yaxes(title_text="Beta (Sensitivity to SPY)")

fig.show()


# We can see in the time of distress, correlations become increasingly negative for all sectors and the betas decrease in value emphasizing the diversification issue.
# 
# We can't just select stocks with no market exposure, we wouldn't accumulate any return alongside the market.
# 
# However, there are optimal ways to construct portfolios to maintain reasonably low market exposure and even exceed the SPX smoothing returns and reducing drawdowns.
# 
# This is how I manage my portfolio, and how I will for my firm (more to come info on this soon)...


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - Basic equity risks we are exposed to include market, industry, and idiosyncratic risk
# - We can diversify away industry and idiosyncratic risk but we are left with market risk as the statistical diversification mechanism breaks down
# - In the context of quantitative frameworks, we can use unsupervised ML (PCA/Linear Dimensionality reducation) as we've observed these risks qualitatively (economically) in a linear way via covariance and correlation
# - The variance explained by the first few components comprises of the major facets of risk: market and sector, but what about the rest?
# - That's where other priced risk factors and alpha live, the excess variation not explained by these market factors
# - Beyond PCA we can model these specific exposures in pricing frameworks like CAPM, Fama-French 3/5, Carhart so on and so forth
# - Portfolio contstruction is about developing target allocations to these different exposures (explained or unexplained which would be considered manager skill) to meet an investment goal, but of course these change over time and the efficacy of constructions and techniques certainly will too
# - This is how I manage my portfolio and how I will for my fund
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - Fama-French / Carhart and Factor Modeling in General
# - Hawkes Processes
# - Merton Jump Diffusion Model (and Characteristic Function Pricing, Carr-Madan 1999)
# - Market-Making Models and Simulation (Stoikov-Avellaneda)
# - My First Year as a Quant
# - Kalman Filter for Quant Finance
# - Why Hedge Funds are Actually Secretive
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Top 3 Uses of Linear Algebra for Quant Finance
# - Girsanov's Change of Measure
# - Rough Path Theory, Applications of Path Signatures
# - Sig-Vol Model, Calibration, and Pricing
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - How to Backtest a Trading Strategy with Interactive Brokers
# - Algorithmic Volatility Trading System


# ---


# ####  $\text{Copyright © 2026 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

