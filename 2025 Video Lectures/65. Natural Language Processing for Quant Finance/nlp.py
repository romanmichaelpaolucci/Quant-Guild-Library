# ### 💬 Natural Language Processing (NLP) for Quant Trading
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
# #### 1.) 💬 Natural Language Processing (NLP)
# 
# - Corpora, Documents, Tokens
# 
# - Tokenization and Vector Representation
# 
# - Cosine (Document) Similarity
# 
# - Common Language Tasks in Finance
# 
# #### 2.) 📈 Extracting Signal from Text
# 
# - Modeling Financial Sentiment
# 
# - Named Entity Recognition (NER)
# 
# - Cross Sectional Equity Trading
# 
# #### 3.) 💭 Closing Thoughts and Future Topics


# ---


# #### 1.) 💬 Natural Language Processing (NLP)
# 
# Generally speaking, there is an inverse relationship between data availability and alpha to extract.  The harder a relationship in terms of some structural inefficiency is to find, the more opportunity there is to profit from it. 
# 
# ##### Does Text Data Impact Stock Prices?
# 
# ##### **Long Answer:**


# Causality is difficult to discern, correlation doesn't imply causation, causality tests are weak at best, due to the nature of the space we can't isolate experiments to a control and treatment group - still I hypothesize there is a systematic edge to harvest from large text datasets.


# ##### **Short Answer:**


# <img src="doge_coin.png" alt="Doge Coin" width="500"/>
# <br>
# <img src="doge_effect.png" alt="Doge NLP Meme" width="500"/> 
# <br>
# <a href='https://fabiandablander.com/r/Causal-Doge.html'> Source </a>
# 
# We can find a significant amount of signal in alternative data, but we can also find a significant amount of noise.
# 
# We need to be careful if we are making trading decisions, especially in the context of large data sets (where often we never observe individual samples) that we aren't confounding information.  For example, DOGE could mean the coin, or it could mean the organization.
# 
# <img src="doge_gov.png" alt="Doge Org" width="500"/>
# 
# Trading algorithms have (and still do) confuse signal with noise.  For example, a well known article discussed that trading algorithms consuming alternative data (like text) for trading decisions would buy up Berkshire-Hathaway ($BRK.A) when the entity in question was actually Anne Hathaway.  These problems are *mostly* solved in language processing (especially with transformers) but there is still a non-zero probability of error, we are hopeful this noise is mitigated in the cross section on a firm level in a trading strategy. . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Corpora, Documents, Tokens
#  
# - 📚 **Corpus (plural: Corpora):**
#    - A large, structured collection of natural language texts.
#    - Used as the foundational dataset for NLP tasks.
#    - Example: A corpus could be all articles from a financial news website, a decade of SEC filings, or a collection of research papers.
#  
# - 📄 **Document:**
#    - An individual piece of text within a corpus.
#    - Can range from a single sentence or email, to an entire news article, report, or book.
#    - In finance, a document might be an earnings report, news item, or analyst note.
#  
# - 🧩 **Token:**
#    - The smallest unit of text with meaning, often a word, symbol, or punctuation mark.
#    - Tokenization is the process of splitting documents into tokens.
#    - Example: Splitting the sentence "Stocks rallied today!" yields tokens ["Stocks", "rallied", "today", "!"].
#  
#  Understanding these basic ideas ensures clear data handling during preprocessing, feature engineering, and modeling in NLP pipelines.


# <img src="doge_coin.png" alt="Doge Coin" width="500"/>


import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt

financial_tickers = ['$AAPL', '$TSLA', '$GOOG', '$AMZN', '$MSFT', '$NVDA', '$META', '$BTC', '$ETH', '$NFLX']
tweet_templates = [
    "Big news for {ticker} today! 🚀",
    "Thinking of buying more {ticker}. What's your take?",
    "{ticker} just broke resistance—bullish!",
    "Sell-off incoming for {ticker}? Or just a dip?",
    "Is {ticker} undervalued right now?",
    "Earnings coming up for {ticker}, any predictions?",
    "Watching {ticker} closely after that announcement.",
    "{ticker}: To the moon or nah?",
    "Bearish on {ticker} this week. Prove me wrong.",
    "Another wild day for {ticker}.",
    "Huge volume spike in {ticker}!",
    "{ticker} is trending—FOMO kicking in...",
    "Should I hodl my {ticker} position?",
    "What a run for {ticker}!",
    "Anyone else watching {ticker} options activity?",
    "Loaded up on {ticker}, let's go!",
    "{ticker} with a strong open today.",
    "Is now the time to short {ticker}?",
    "News moving {ticker} right now! 🚨",
    "Long term bullish on {ticker}."
]

def generate_financial_tweet_corpus(n=20):
    corpus = []
    for _ in range(n):
        ticker = random.choice(financial_tickers)
        template = random.choice(tweet_templates)
        tweet = template.format(ticker=ticker)
        corpus.append(tweet)
    return corpus

# Generate a sample corpus of 20 synthetic financial tweets
synthetic_corpus = generate_financial_tweet_corpus(10)

# Display the tweets visually in a table (text grid)
import pandas as pd
df = pd.DataFrame({'Corpus (Tweets)': synthetic_corpus})
display(df.style.set_properties(**{'font-size': '14pt'}))


# ##### We need a quantitative way to analyze this qualitative data to extract signal for trading . . .


# ###### ______________________________________________________________________________________________________________________________________


# ##### Tokenization and Vector Representation
# 
# Tokenization transforms documents into a list of tokens
# 
# <img src="doge_coin.png" alt="Doge Coin" width="400"/>
# 
# **Tokenized Document:** [*"Dogecoin", "is", "the", "people's", "crypto"*]
# 
# Much like the sample space of a random variable, we still need a quantitative way to understand this document
# 
# Let's create a table, the columns will be the words we observe and the rows will be documents, each entry is the number of times we see the word in a document
# 
# **This is called a Document-Term Matrix**
#  
#   Suppose we have tokenized documents
# - doc 1: [*"Dogecoin", "is", "the", "people's", "crypto"*]
# - doc 2: [*"NVDA", "stock", "and", "DOGE", "crypto", "are", "good", "buy"*]
# - doc 3: [*"AAPL", "stock", "is", "good"*]
#  
# If we only consider the tokens "buy", "stock", and "crypto", our document-term matrix is:
# 
# |           | buy | stock | crypto |
# |-----------|:---:|:-----:|:------:|
# | doc 1     |  0  |   0   |   1    |
# | doc 2     |  1  |   1   |   1    |
# | doc 3     |  0  |   1   |   0    |
# 
#  In this matrix, rows represent documents and columns are the tokens from the documents.
# 
#  Typically Doc-Term matrices have *many* columns (words) and rows (documents).  The matrix is typically *very sparse* leading to different efficient ways to analyze it and compress information in terms of dimensionality reduction prior to downstream analysis and modeling, for now though, I will digress.
# 
#  ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Let's Visualize the Documents in 3D: (crypto, stock, buy)


import numpy as np
import plotly.graph_objects as go

# Raw vectors
raw_vectors = np.array([
    [1, 0, 0],   # doc 1
    [1, 1, 1],   # doc 2
    [0, 1, 0],   # doc 3
])

# Normalize for display
norms = np.linalg.norm(raw_vectors, axis=1)
vectors = np.array([v / norms.max() for v in raw_vectors])

labels = [
    "doc 1: (1, 0, 0)",
    "doc 2: (1, 1, 1)",
    "doc 3: (0, 1, 0)"
]

colors = ['#636EFA', '#EF553B', '#00CC96']

fig = go.Figure()

# Arrowhead size
cone_len = 0.15

for v, label, color in zip(vectors, labels, colors):
    # Normalize direction
    direction = v / np.linalg.norm(v)

    # Tip and base of arrowhead
    tip = v
    cone_base = tip - direction * cone_len

    # Shaft (line)
    fig.add_trace(go.Scatter3d(
        x=[0, cone_base[0]],
        y=[0, cone_base[1]],
        z=[0, cone_base[2]],
        mode="lines",
        line=dict(color=color, width=8),
        showlegend=False
    ))

    # Arrowhead (Cone)
    fig.add_trace(go.Cone(
        x=[cone_base[0]],
        y=[cone_base[1]],
        z=[cone_base[2]],
        u=[direction[0]],
        v=[direction[1]],
        w=[direction[2]],
        sizemode="absolute",
        sizeref=cone_len,
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo="text",
        hovertext=[label],
        opacity=1.0
    ))

    # Tip marker + label
    fig.add_trace(go.Scatter3d(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        mode="markers+text",
        marker=dict(size=8, color=color),
        text=[label],
        textposition="middle right",
        showlegend=False
    ))

# Camera zoom-out
fig.update_layout(
    title="Documents Visualized in 3D Word Space",
    width=1000, height=700,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=18),
    scene=dict(
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Crypto",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False  # remove tick values, keep gridlines
            # No tickvals specified!
        ),
        yaxis=dict(
            title="Stock",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False  # remove tick values, keep gridlines
            # No tickvals specified!
        ),
        zaxis=dict(
            title="Buy",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False  # remove tick values, keep gridlines
            # No tickvals specified!
        ),
        xaxis_range=[-0.2, 1.3],
        yaxis_range=[-0.2, 1.3],
        zaxis_range=[-0.2, 1.3],
        camera=dict(
            eye=dict(x=2, y=1, z=1.2)  # zoomed out
        )
    ),
    margin=dict(l=10, r=10, b=10, t=60)
)

fig.show()



# We can see now in 3D that similar documents will be closer together, this idea holds for any number of dimensions!  Documents in the Doc-Term sense are effectively n-dimensional clouds of points.
# 
# Each entry shows the count of the word in that document.  There are a variety of other ways to develop the entries here including methods like TF-IDF and trucnation of words that don't appear often enough and proper preprocessing and the removal of stop words (and, the, . . .)
# 
# In any case, we now have a quantitative framework to analyze our corpus, this table is a *matrix*


#  ###### ______________________________________________________________________________________________________________________________________
# 


# ##### How Can We Measure Document Similarity?
# 
#  Let's consider the n-dimensional Euclidean distance
#  
#  $$
#  d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_{i=1}^n (x_i - y_i)^2 }
#  $$
# 
# 
# Distance isn't going to cut it here, we are operating across different dimensions and we want to keep the relative position in mind in some sort of calculation.  


import numpy as np
import plotly.graph_objects as go

# ------------------------------------
# HELPER FUNCTIONS
# ------------------------------------

def euclidean_distance(u, v):
    """Return Euclidean distance between vectors u and v."""
    return np.linalg.norm(u - v)


# ------------------------------------
# ORIGINAL VECTORS
# ------------------------------------
raw_vectors = np.array([
    [1, 0, 0],   # doc 1
    [1, 1, 1],   # doc 2
    [0, 1, 0],   # doc 3
])

vectors = raw_vectors.copy()

labels = [
    "doc 1: (1, 0, 0)",
    "doc 2: (1, 1, 1)",
    "doc 3: (0, 1, 0)"
]

colors = ['#636EFA', '#EF553B', '#00CC96']


# ------------------------------------
# MAIN FIGURE
# ------------------------------------
fig = go.Figure()
cone_len = 0.15

for v, label, color in zip(vectors, labels, colors):
    direction = v / np.linalg.norm(v)
    tip = v
    cone_base = tip - direction * cone_len

    # Arrow shaft
    fig.add_trace(go.Scatter3d(
        x=[0, cone_base[0]], y=[0, cone_base[1]], z=[0, cone_base[2]],
        mode="lines",
        line=dict(color=color, width=8),
        showlegend=False
    ))

    # Arrowhead
    fig.add_trace(go.Cone(
        x=[cone_base[0]], y=[cone_base[1]], z=[cone_base[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=cone_len,
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hovertext=[label]
    ))

    # Tip marker and label
    fig.add_trace(go.Scatter3d(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        mode="markers+text",
        marker=dict(size=8, color=color),
        text=[label], textposition="middle right",
        showlegend=False
    ))


# ------------------------------------
# STRAIGHT DISTANCE LINES
# ------------------------------------
pairs = [
    (0, 1, "#AB63FA"),   # purple
    (0, 2, "#FFA15A"),   # orange
    (1, 2, "#19D3F3")    # light blue
]

for i, j, line_color in pairs:
    u = vectors[i]
    v = vectors[j]

    dist = euclidean_distance(u, v)

    # Straight dashed line between tips
    fig.add_trace(go.Scatter3d(
        x=[u[0], v[0]],
        y=[u[1], v[1]],
        z=[u[2], v[2]],
        mode="lines",
        line=dict(color=line_color, width=6, dash="dash"),
        showlegend=False
    ))

    # Midpoint label
    mid = (u + v) / 2
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="text",
        text=[f"{dist:.2f}"],
        textfont=dict(size=22, color=line_color),
        showlegend=False
    ))


# ------------------------------------
# ORIGINAL AXIS TITLES + GRIDLINES
# ------------------------------------
fig.update_layout(
    title="Documents Visualized in 3D Word Space (Euclidean Distances)",
    width=1000, height=700,
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(color="white", size=18),
    scene=dict(
        bgcolor="rgba(0, 0, 0, 0)",

        xaxis=dict(
            title="Crypto",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),
        yaxis=dict(
            title="Stock",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),
        zaxis=dict(
            title="Buy",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),

        xaxis_range=[-0.2, 1.3],
        yaxis_range=[-0.2, 1.3],
        zaxis_range=[-0.2, 1.3],

        camera=dict(eye=dict(x=2, y=1, z=1.2))
    ),
    margin=dict(l=10, r=10, b=10, t=60)
)

fig.show()



# In other words, all of the documents as points may have the same distance between each other but they may each represent something similar or completely different! 


#  ###### ______________________________________________________________________________________________________________________________________


# ##### Cosine (Document) Similarity
# 
# Let's recall from precalculus
# 
# $$\cos(90^\circ) = 0 \quad\quad \cos(0) = 1$$


from IPython.display import HTML

HTML("""
<style>

    /* Remove notebook cell background and borders */
    div.output_area,
    div.output_subarea,
    div.output_html,
    div.jp-OutputArea,
    .jp-OutputArea-output, 
    .jp-RenderedHTMLCommon {
        background-color: #282c34 !important;
        border: none !important;
    }

    /* Widget container area */
    .widget-area,
    .widget-area .widget-subarea,
    .widget-inline-hbox,
    .jupyter-widgets,
    .widget-container,
    .widget-box,
    .widget-vbox,
    .widget-hbox {
        background-color: #282c34 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Slider label */
    .widget-label {
        color: #dbedf7 !important;
        background-color: #282c34 !important;
        border: none !important;
    }

    /* Slider readout box */
    .widget-readout {
        background-color: #282c34 !important;
        color: #dbedf7 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Slider track */
    .noUi-target {
        background: #3a3f4b !important;
        border: 1px solid #49505f !important;
        box-shadow: none !important;
    }

    /* Slider filled track */
    .noUi-connect {
        background: #61dafb !important;
    }

    /* Slider handle */
    .noUi-handle {
        background: #a9f784 !important;
        border: 2px solid #49505f !important;
        box-shadow: none !important;
    }

    /* Remove any residual padding from widget wrapper */
    .jp-Widget,
    .widget-container,
    .widget-hslider {
        padding: 0px !important;
        margin: 0px !important;
        background-color: #282c34 !important;
    }

</style>
""")



import numpy as np
import plotly.graph_objs as go
from ipywidgets import interact, FloatSlider

# Dark theme setup
BACKGROUND = "#282c34"
GRIDCOLOR = "#49505f"
TITLE_COLOR = "#ffdf6b"
TEXT_COLOR = "#dbedf7"

# Unit circle data
t = np.linspace(0, 2 * np.pi, 200)
x_circle = np.cos(t)
y_circle = np.sin(t)

def plot_doc_vectors(theta_deg):
    theta_rad = np.deg2rad(theta_deg)

    # --- Document vectors ---
    doc1 = np.array([0, 1])                        # fixed
    doc2 = np.array([np.cos(theta_rad), np.sin(theta_rad)])  # slider

    # Compute angle & cosine between the docs
    dot = np.dot(doc1, doc2)
    cos_angle = dot / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    fig = go.Figure()

    # Draw unit circle
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color="#6cc0ff", width=3)
    ))

    # Y-axis (reference)
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1.15],
        mode='lines',
        line=dict(color="#fd8585", dash='dash', width=2)
    ))

    # X-axis (reference), extended to x=1.75
    fig.add_trace(go.Scatter(
        x=[0, 1.75], y=[0, 0],
        mode='lines',
        line=dict(color="#abbacf", dash='dash', width=2)
    ))

    # --- doc 1 vector ---
    fig.add_trace(go.Scatter(
        x=[0, doc1[0]], y=[0, doc1[1]],
        mode='lines+markers+text',
        line=dict(color="#6effa6", width=6),
        marker=dict(size=14, color="#6effa6"),
        text=[None, f"<span style='color:{TEXT_COLOR};font-size:19px'><b>doc 1</b></span>"],
        textposition='top right'
    ))

    # --- doc 2 vector (NO coordinates; angle + cosine only) ---
    fig.add_trace(go.Scatter(
        x=[0, doc2[0]], y=[0, doc2[1]],
        mode='lines+markers+text',
        line=dict(color="#a9f784", width=6),
        marker=dict(size=14, color="#a9f784"),
        text=[None,
              f"<span style='color:{TEXT_COLOR};font-size:19px'><b>doc 2</b><br>"
              f"θ = {angle_deg:.1f}°<br>cos θ = {cos_angle:.3f}</span>"],
        textposition='middle right'
    ))

    # Fill the angle sector
    start = np.pi/2
    end = theta_rad
    arc_t = np.linspace(start, end, 50)
    arc_x = np.cos(arc_t)
    arc_y = np.sin(arc_t)

    if len(arc_x) > 1:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], arc_x]),
            y=np.concatenate([[0], arc_y]),
            mode='lines',
            fill='toself',
            fillcolor='rgba(103,232,192,0.14)',
            line=dict(color="#00CC96", width=0),
            hoverinfo='skip'
        ))

    # Layout
    fig.update_layout(
        title=(
            f"<span style='color:{TITLE_COLOR};font-size:26px'><b>Doc Vectors on Unit Circle</b></span>"
            ),
        width=600, height=600,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR, size=19),
        showlegend=False,
        margin=dict(l=40, r=40, t=75, b=40),

        # 🔥 Updated axis ranges
        xaxis=dict(
            title="<b>X</b>",
            range=[-1.2, 2],   # EXTENDED RIGHTWARD to 1.75
            zeroline=True,
            showgrid=True, gridcolor=GRIDCOLOR,
            color=TEXT_COLOR, zerolinecolor=GRIDCOLOR
        ),
        yaxis=dict(
            title="<b>Y</b>",
            range=[-0.25, 1.25],
            zeroline=True,
            scaleanchor="x", scaleratio=1,
            showgrid=True, gridcolor=GRIDCOLOR,
            color=TEXT_COLOR, zerolinecolor=GRIDCOLOR
        ),
    )

    fig.update_xaxes(visible=True, mirror=True, linewidth=2, color='#9fffc7', showline=True)
    fig.update_yaxes(visible=True, mirror=True, linewidth=2, color='#9fffc7', showline=True)

    fig.show()


# 🔥 Slider NOW restricted to 0 → 90 degrees
from ipywidgets import Layout

interact(
    plot_doc_vectors,
    theta_deg=FloatSlider(
        value=45,
        min=0, max=90, step=1,
        description='Angle θ° (doc 2 direction)',
        style={'description_width': 'initial'},
        continuous_update=True,
        readout=True,
        layout=Layout(width='500px', height='60px')  # Make the slider larger
    )
)



#  ###### ______________________________________________________________________________________________________________________________________


# ##### How similar are our documents?


# Recall our documents
#    - doc 1: [*"Dogecoin", "is", "the", "people's", "crypto"*]
#    - doc 2: [*"NVDA", "stock", "and", "DOGE", "crypto", "are", "good", "buy"*]
#    - doc 3: [*"AAPL", "stock", "is", "good"*]
# 
# I would suggest that doc 1 and doc 3 are about crypto, and doc 2 and doc 3 are about stock
# 
# Cosine similarity captures this quantitatively!


import numpy as np
import plotly.graph_objects as go

# ------------------------------------
# HELPER FUNCTIONS
# ------------------------------------

def angle_between(u, v):
    """Return angle (radians) between vectors u and v."""
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def arc_points(u, v, scale=0.55, steps=60):
    """
    Generate 3D arc points between vectors u and v.
    scale < 1 shrinks the arc toward the origin.
    """
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    theta = angle_between(u, v)
    axis = np.cross(u, v)
    axis = axis / np.linalg.norm(axis)

    pts = []
    for t in np.linspace(0, 1, steps):
        ang = t * theta
        u_rot = (
            u * np.cos(ang)
            + np.cross(axis, u) * np.sin(ang)
            + axis * np.dot(axis, u) * (1 - np.cos(ang))
        )
        pts.append(u_rot * scale)

    return np.array(pts)


# ------------------------------------
# ORIGINAL VECTORS (scaled upward a bit)
# ------------------------------------
raw_vectors = np.array([
    [1, 0, 0],   # doc 1
    [1, 1, 1],   # doc 2
    [0, 1, 0],   # doc 3
])

labels = [
    "doc 1: (1, 0, 0)",
    "doc 2: (1, 1, 1)",
    "doc 3: (0, 1, 0)"
]

colors = ['#636EFA', '#EF553B', '#00CC96']


# ------------------------------------
# MAIN FIGURE
# ------------------------------------
fig = go.Figure()
cone_len = 0.15

for v, label, color in zip(vectors, labels, colors):
    direction = v / np.linalg.norm(v)
    tip = v
    cone_base = tip - direction * cone_len

    # Arrow shaft
    fig.add_trace(go.Scatter3d(
        x=[0, cone_base[0]], y=[0, cone_base[1]], z=[0, cone_base[2]],
        mode="lines",
        line=dict(color=color, width=8),
        showlegend=False
    ))

    # Arrowhead
    fig.add_trace(go.Cone(
        x=[cone_base[0]], y=[cone_base[1]], z=[cone_base[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=cone_len,
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hovertext=[label]
    ))

    # Tip marker and label
    fig.add_trace(go.Scatter3d(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        mode="markers+text",
        marker=dict(size=8, color=color),
        text=[label], textposition="middle right",
        showlegend=False
    ))


# ------------------------------------
# ANGLE ARCS (between each vector pair)
# ------------------------------------
pairs = [
    (0, 1, "#AB63FA"),   # purple arc
    (0, 2, "#FFA15A"),   # orange arc
    (1, 2, "#19D3F3")    # light blue arc
]

for i, j, arc_color in pairs:
    u = vectors[i]
    v = vectors[j]

    pts = arc_points(u, v, scale=0.55)

    # Draw the arc
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(color=arc_color, width=6),
        showlegend=False
    ))

    # Midpoint label for theta
    mid = pts[len(pts) // 2]
    theta_deg = np.degrees(angle_between(u, v))
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="text",
        text=[f"{theta_deg:.1f}°"],
        textfont=dict(size=22, color=arc_color),
        showlegend=False
    ))


# ------------------------------------
# ORIGINAL AXIS TITLES + GRIDLINES RESTORED
# ------------------------------------
fig.update_layout(
    title="Documents Visualized in 3D Word Space (With Angles)",
    width=1000, height=700,
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(color="white", size=18),
    scene=dict(
        bgcolor="rgba(0, 0, 0, 0)",

        # 🔥 ORIGINAL AXIS TITLES + GRID LINES
        xaxis=dict(
            title="Crypto",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),
        yaxis=dict(
            title="Stock",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),
        zaxis=dict(
            title="Buy",
            showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray",
            zerolinecolor="darkgray",
            showticklabels=False
        ),

        # Keep your original ranges exactly
        xaxis_range=[-0.2, 1.3],
        yaxis_range=[-0.2, 1.3],
        zaxis_range=[-0.2, 1.3],

        camera=dict(eye=dict(x=2, y=1, z=1.2))
    ),
    margin=dict(l=10, r=10, b=10, t=60)
)

fig.show()



# Compute (and print elegantly) cosine similarity between all doc pairs
from itertools import combinations

print("Pairwise Cosine Similarities Between Documents:")
for i, j in combinations(range(len(raw_vectors)), 2):
    u = raw_vectors[i]
    v = raw_vectors[j]
    cos_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    print(f"  Cosine(doc {i+1}, doc {j+1}) = {cos_sim:.3f}")



# - doc 1: [*"Dogecoin", "is", "the", "people's", "crypto"*]
# - doc 2: [*"NVDA", "stock", "and", "DOGE", "crypto", "are", "good", "buy"*]
# - doc 3: [*"AAPL", "stock", "is", "good"*]


# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Stock Signal Representations
# 
# Knowing we can measure document similarity can be useful for clustering tasks (grouping documents by categories, so on and so forth) but what if we wanted to begin modeling some sort of signal?
# 
# A simple example of this may be constructing a unit signal vector in a dimension of interest to then rank documents against.
# 
# Consider the following buy signal vector relative to our documents. . .


import numpy as np
import plotly.graph_objects as go

# ------------------------------------
# HELPER FUNCTIONS
# ------------------------------------

def angle_between(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def arc_points(u, v, scale=0.55, steps=80):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    theta = angle_between(u, v)
    axis = np.cross(u, v)

    if np.linalg.norm(axis) < 1e-6:   # parallel fallback
        axis = np.array([0, 0, 1])
    axis = axis / np.linalg.norm(axis)

    pts = []
    for t in np.linspace(0, 1, steps):
        ang = t * theta
        u_rot = (
            u * np.cos(ang) +
            np.cross(axis, u) * np.sin(ang) +
            axis * np.dot(axis, u) * (1 - np.cos(ang))
        )
        pts.append(u_rot * scale)
    return np.array(pts)


# ------------------------------------
# ORIGINAL VECTORS
# ------------------------------------
raw_vectors = np.array([
    [1, 0, 0],   # doc 1
    [1, 1, 1],   # doc 2
    [0, 1, 0],   # doc 3
])

signal_vector = np.array([0, 0, 1])  # <-- NEW: SIGNAL VECTOR

vectors = raw_vectors.copy()
labels = [
    "doc 1: (1, 0, 0)",
    "doc 2: (1, 1, 1)",
    "doc 3: (0, 1, 0)"
]
colors = ['#636EFA', '#EF553B', '#00CC96']


# ------------------------------------
# MAIN FIGURE
# ------------------------------------
fig = go.Figure()
cone_len = 0.15

# Plot document vectors
for v, label, color in zip(vectors, labels, colors):
    direction = v / np.linalg.norm(v)
    tip = v
    cone_base = tip - direction * cone_len

    fig.add_trace(go.Scatter3d(
        x=[0, cone_base[0]], y=[0, cone_base[1]], z=[0, cone_base[2]],
        mode="lines", line=dict(color=color, width=8), showlegend=False
    ))
    fig.add_trace(go.Cone(
        x=[cone_base[0]], y=[cone_base[1]], z=[cone_base[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=cone_len,
        anchor="tail", colorscale=[[0, color], [1, color]],
        showscale=False, hovertext=[label]
    ))
    fig.add_trace(go.Scatter3d(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        mode="markers+text", marker=dict(size=8, color=color),
        text=[label], textposition="middle right", showlegend=False
    ))

# ------------------------------------
# ADD SIGNAL VECTOR (Z axis)
# ------------------------------------
signal_color = "#FFFFFF"
sv = signal_vector
sv_dir = sv / np.linalg.norm(sv)

fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, 1],
    mode="lines",
    line=dict(color=signal_color, width=8),
    showlegend=False
))
fig.add_trace(go.Cone(
    x=[0], y=[0], z=[1 - cone_len],
    u=[0], v=[0], w=[1],
    sizemode="absolute", sizeref=cone_len,
    anchor="tail",
    colorscale=[[0, signal_color], [1, signal_color]],
    showscale=False,
    hovertext="Signal Vector (0,0,1)"
))
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[1],
    mode="markers+text",
    marker=dict(size=8, color=signal_color),
    text=["Signal Vector (0,0,1)"],
    textposition="top center",
    showlegend=False
))


# ------------------------------------
# ANGLES BETWEEN DOCS AND SIGNAL VECTOR (NO FLATTENING)
# ------------------------------------
for (v, color, label) in zip(vectors, colors, labels):
    u = v
    w = signal_vector  # (0,0,1)

    theta = angle_between(u, w)
    theta_deg = np.degrees(theta)

    # Always use the 3D Rodrigues arc
    pts = arc_points(u, w, scale=0.55)

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(color=color, width=6),
        showlegend=False
    ))

    # Label at midpoint
    mid = pts[len(pts)//2]
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="text",
        text=[f"{theta_deg:.1f}°"],
        textfont=dict(size=22, color=color),
        showlegend=False
    ))




# ------------------------------------
# AXIS STYLING
# ------------------------------------
fig.update_layout(
    title="Document Vectors and Angle to Signal Vector (0,0,1)",
    width=1000, height=700,
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(color="white", size=18),
    scene=dict(
        bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(
            title="Crypto", showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray", zerolinecolor="darkgray",
            showticklabels=False
        ),
        yaxis=dict(
            title="Stock", showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray", zerolinecolor="darkgray",
            showticklabels=False
        ),
        zaxis=dict(
            title="Buy", showbackground=True,
            backgroundcolor="rgb(30,30,35)",
            gridcolor="darkgray", zerolinecolor="darkgray",
            showticklabels=False
        ),
        xaxis_range=[-0.2, 1.3],
        yaxis_range=[-0.2, 1.3],
        zaxis_range=[-0.2, 1.3],
        camera=dict(eye=dict(x=2, y=1, z=1.2))
    ),
    margin=dict(l=10, r=10, b=10, t=60)
)

fig.show()



# Recall our documents:
# - doc 1: [*"Dogecoin", "is", "the", "people's", "crypto"*]
# - doc 2: [*"NVDA", "stock", "and", "DOGE", "crypto", "are", "good", "buy"*]
# - doc 3: [*"AAPL", "stock", "is", "a", "good"*]


for i, (v, label) in enumerate(zip(vectors, labels), 1):
    dot = np.dot(v, signal_vector)
    norm_v = np.linalg.norm(v)
    norm_signal = np.linalg.norm(signal_vector)
    cosine = dot / (norm_v * norm_signal)
    print(f"{label} cosine with buy vector (0,0,1): {cosine:.4f}")


# We can see however that this model is naive, if the statement was "**DON'T** buy AAPL stock" we would have an entirely different meaning.  Moreover, "BUY" and "buy" in this framework are treated as different tokens.  In reality, we need a more sophisticated language processing pipeline for document acquisition, tokenization, and of course signal modeling.
# 
# In any case, we can produce a trading signal using this methodology, but we will need some way to match each document to then a firm.
# 
# This turns into a named entity recognition problem and a larger cross-sectional equity trading strategy which we will discuss toward the end of this video.


# ###### ______________________________________________________________________________________________________________________________________


# ##### Common Language Tasks in Finance
# 
# Common Language Processing Tasks in Quant Finance:
#  
#   - **Language Modeling**: Predicting word sequences for tasks like next-word prediction and text generation in finance.
#   - **Sentiment Analysis**: Classifying financial text as positive, negative, or neutral to assess market sentiment.
#   - **Topic Modeling (e.g., LDA)**: Uncovering main topics in financial text using methods like LDA.
#   - **Named Entity Recognition (NER)**: Identifying key entities (companies, instruments, people, dates) in unstructured financial text.
# 
# Given this brief survey of language processing, we'll focus on one aspect of it for extracting signal: *sentiment analysis*


# ---


# #### 2.) 📈 Extracting Signal from Text
# 
# 


# We've seen how langauge can be modeled in a quantitative way above via a matrix and saw a naive model for extracting signal from text
# 
# We'll now discuss some alternate methods of extracting signal from text as explored by both the academic literature and is practiced in industry
# 
# ##### Modeling Financial Sentiment
# 
# Before other machine learning approaches (taking advantage of a Doc-Term structure).
# 
# The finance literature has implemented dictionary methods with success in the past.
# 
# The dictionary-based sentiment scoring can be represented as a piecewise function:
# 
# $$
# S(w) = 
# \begin{cases}
#   2, & \text{if } w \in \{\text{"excellent"},\, \text{"fantastic"},\, \text{"amazing"}\} \\
#   1, & \text{if } w \in \{\text{"good"},\, \text{"nice"},\, \text{"pleasant"},\, \text{"positive"}\} \\
#   0, & \text{if } w \in \{\text{"neutral"},\, \text{"average"},\, \text{"ok"}\} \\
#   -1, & \text{if } w \in \{\text{"bad"},\, \text{"poor"},\, \text{"unpleasant"},\, \text{"negative"}\} \\
#   -2, & \text{if } w \in \{\text{"terrible"},\, \text{"awful"},\, \text{"horrible"}\} \\
#   0, & \text{otherwise}
# \end{cases}
# $$
# 
# **Example finance tweets:**
# 
# - *"Earnings beat expectations; guidance was raised for the next quarter."* 📈💼
# - *"Dividend increased by 10% — strong signal for long term investors."* 💰📢
# - *"Operational cash flow improved considerably, reducing leverage."* 💵🔻
# - *"M&A rumors driving volatility; watch asset repricing."* 🔄🔥
# - *"Stock downgraded by several brokers despite revenue growth."* 📉🔽
# - *"Management announced a buyback; float will shrink by Q3."* ♻️📊
# - *"Balance sheet weak, high debt-to-equity makes this risky."* 📉⚠️
# - *"Margin compression expected due to rising input costs."* 📉💸
# - *"Company secures major contract extension."* ✍️🤝
# - *"Sector rotation into defensives changing capital flows."* 🔄🔒
# 
# 
# ###### ______________________________________________________________________________________________________________________________________
# 
# ##### Let's Apply this Model to the Text Data and Observe the Scores


import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use the same "plain" sentiment dictionary (no finance words)
sentiment_dict = {
    2: {"fantastic", "amazing"},
    1: {"good", "nice", "pleasant"},
    0: {"neutral", "average", "ok"},
    -1: {"bad", "poor", "unpleasant"},
    -2: {"terrible", "awful", "horrible"},
}

def word_score(w):
    lw = w.lower()
    for score, words in sentiment_dict.items():
        if lw in words:
            return score
    return 0

tweets = [
    "Earnings and cash flow are excellent, but debt remains high.",
    "Dividend increased, but balance sheet is weak. Positive guidance overall."
]

def get_token_scores(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    scores = [word_score(tok) for tok in tokens]
    return tokens, scores

ngram_bar_colors = {
    "pos": "#6cc0ff",
    "neg": "#fd8585",
    "neu": "#AAAAAA"
}

fig_ngram = make_subplots(
    rows=1, cols=2, 
    column_widths=[0.53, 0.47],
    subplot_titles=[
        "<span style='color:#ffdf6b;font-size:18px'><b>Document 1</b></span>",
        "<span style='color:#ffdf6b;font-size:18px'><b>Document 2</b></span>"
    ],
    shared_yaxes=True,
    horizontal_spacing=0.13
)

def aggregate(scores):
    pos = sum(s for s in scores if s > 0)
    neg = sum(s for s in scores if s < 0)
    tot = sum(scores)
    return pos, neg, tot

for i, tweet in enumerate(tweets):
    tokens, scores = get_token_scores(tweet)
    colors = [
        ngram_bar_colors["pos"] if s > 0 else ngram_bar_colors["neg"] if s < 0 else ngram_bar_colors["neu"]
        for s in scores
    ]
    fig_ngram.add_trace(
        go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            marker_line_color="#45474d",
            marker_line_width=1.7,
            text=[str(s) if s != 0 else "" for s in scores],
            textposition='auto',
            opacity=0.94,
            showlegend=False
        ), row=1, col=i+1
    )
    pos, neg, total = aggregate(scores)
    fig_ngram.layout.annotations[i]['text'] += f"<br><span style='color:#a9f784;font-size:14px'>Pos={pos} Neg={neg} Sum={total}</span>"

for idx in range(1, 3):
    fig_ngram.update_xaxes(
        title="<b>Token</b>" if idx == 1 else None,
        row=1, col=idx,
        color='#dbedf7',
        showline=True, linecolor="#45474d",
        tickangle=45,
        gridcolor="#45474d",
        tickfont=dict(size=15)
    )
fig_ngram.update_yaxes(
    title="<b>Sentiment Score</b>",
    row=1, col=1,
    color='#dbedf7',
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d"
)
fig_ngram.update_yaxes(
    color="#dbedf7",
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d",
    row=1, col=2
)

fig_ngram.update_layout(
    width=950, height=360,
    plot_bgcolor="#282c34",
    paper_bgcolor="#282c34",
    font=dict(size=16, color="#dbedf7"),
    margin=dict(l=44, r=24, t=145, b=40),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.62,
        font=dict(color="#dbedf7", size=15),
        bgcolor='rgba(0,0,0,0)'
    ),
    title=dict(
        text=(
            "<b>Word-by-word Sentiment Scores (Non-Financial Dictionary)</b><br>"
            "<span style='font-size:15px; color:#dbedf7'>Naive sentiment scoring without finance lexicon</span>"
        ),
        x=0.5, y=0.875,
        font=dict(size=20, color="#ffdf6b")
    ),
    hovermode='x'
)

fig_ngram.show()



# ###### ______________________________________________________________________________________________________________________________________


# ##### What Happened?  We Didn't Capture Financial Lexicon
# 
#  
# A lexicon is a collection of words or vocabulary, often with associated meanings, used for language analysis (jargon).
# 
# Words like **buy** or **sell** don't have generic sentiment, but they mean something in a financial context.
# 
# Let's modify our model to include financial lexicon. . .
# 
# $$
# S(w) = 
# \begin{cases}
#   2, & \text{if } w \in \{\text{"excellent"},\, \text{"fantastic"},\, \text{"amazing"},\, \text{"earnings"},\, \text{"guidance"},\, \text{"dividend"},\, \text{"improved"},\, \text{"cash"},\, \text{"flow"},\, \text{"beat"},\, \text{"raised"},\, \text{"increased"}\} \\
#   1, & \text{if } w \in \{\text{"good"},\, \text{"nice"},\, \text{"pleasant"},\, \text{"positive"},\, \text{"strong"},\, \text{"signal"},\, \text{"reducing"},\, \text{"leverage"},\, \text{"buyback"},\, \text{"float"},\, \text{"shrink"},\, \text{"will"}\} \\
#   0, & \text{if } w \in \{\text{"neutral"},\, \text{"average"},\, \text{"ok"}\} \\
#   -1, & \text{if } w \in \{\text{"bad"},\, \text{"poor"},\, \text{"unpleasant"},\, \text{"negative"},\, \text{"high"},\, \text{"debt"},\, \text{"to"},\, \text{"equity"}\} \\
#   -2, & \text{if } w \in \{\text{"terrible"},\, \text{"awful"},\, \text{"horrible"},\, \text{"downgraded"},\, \text{"weak"},\, \text{"balance"},\, \text{"sheet"}\} \\
#   0, & \text{otherwise}
# \end{cases}
# $$


import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sentiment dictionary matching the LaTeX model above (single-token, not phrase-based)
sentiment_dict = {
    2: {
        "excellent", "fantastic", "amazing", "earnings", "guidance", "dividend",
        "improved", "cash", "flow", "beat", "raised", "increased"
    },
    1: {
        "good", "nice", "pleasant", "positive", "strong", "signal", "reducing",
        "leverage", "buyback", "float", "shrink", "will"
    },
    0: {
        "neutral", "average", "ok"
    },
    -1: {
        "bad", "poor", "unpleasant", "negative", "high", "debt", "to", "equity"
    },
    -2: {
        "terrible", "awful", "horrible", "downgraded", "weak", "balance", "sheet"
    }
}

def word_score(w):
    lw = w.lower()
    for score, words in sentiment_dict.items():
        if lw in words:
            return score
    return 0

tweets = [
    "Earnings and cash flow are excellent, but debt remains high.",
    "Dividend increased, but balance sheet is weak. Positive guidance overall."
]

def get_token_scores(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    scores = [word_score(tok) for tok in tokens]
    return tokens, scores

def aggregate(scores):
    total_positive = sum(s for s in scores if s > 0)
    total_negative = sum(s for s in scores if s < 0)
    total = sum(scores)
    return total_positive, total_negative, total

tweet_indices = [0, 1]
example_sentences_finance = [tweets[0], tweets[1]]

fig_finance = make_subplots(
    rows=1, cols=2, 
    column_widths=[0.53, 0.47],
    subplot_titles=[
        "<span style='color:#ffdf6b;font-size:18px'><b>Document 1</b></span>",
        "<span style='color:#ffdf6b;font-size:18px'><b>Document 2</b></span>"
    ],
    shared_yaxes=True,
    horizontal_spacing=0.13
)

finance_bar_colors = {
    "pos": "#6cc0ff",
    "neg": "#fd8585",
    "neu": "#AAAAAA"
}

for i, sent in enumerate(example_sentences_finance):
    tokens, scores = get_token_scores(sent)
    colors = [
        finance_bar_colors["pos"] if s > 0 else finance_bar_colors["neg"] if s < 0 else finance_bar_colors["neu"]
        for s in scores
    ]
    fig_finance.add_trace(
        go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            marker_line_color="#45474d",
            marker_line_width=1.7,
            text=[str(s) if s != 0 else "" for s in scores],
            textposition='auto',
            opacity=0.94,
            showlegend=False
        ), row=1, col=i+1
    )
    pos, neg, total = aggregate(scores)
    fig_finance.layout.annotations[i]['text'] += f"<br><span style='color:#a9f784;font-size:14px'>Pos={pos} Neg={neg} Sum={total}</span>"

for idx in range(1, 3):
    fig_finance.update_xaxes(
        title="<b>Token</b>" if idx == 1 else None,
        row=1, col=idx,
        color='#dbedf7',
        showline=True, linecolor="#45474d",
        tickangle=45,
        gridcolor="#45474d",
        tickfont=dict(size=15)
    )
fig_finance.update_yaxes(
    title="<b>Sentiment Score</b>",
    row=1, col=1,
    color='#dbedf7',
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d"
)
fig_finance.update_yaxes(
    color="#dbedf7",
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d",
    row=1, col=2
)

fig_finance.update_layout(
    width=950, height=360,
    plot_bgcolor="#282c34",
    paper_bgcolor="#282c34",
    font=dict(size=16, color="#dbedf7"),
    margin=dict(l=44, r=24, t=145, b=40),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.62,
        font=dict(color="#dbedf7", size=15),
        bgcolor='rgba(0,0,0,0)'
    ),
    title=dict(
        text=(
            "<b>Word-by-word Sentiment Scores for Example Finance Tweets</b><br>"
            "<span style='font-size:15px; color:#dbedf7'>Scores use a Financial (Non-Generic) Sentiment Lexicon</span>"
        ),
        x=0.5, y=0.875,
        font=dict(size=20, color="#ffdf6b")
    ),
    hovermode='x'
)

fig_finance.show()



# ###### ______________________________________________________________________________________________________________________________________


# **Remark:**  Dictionary based models do not capture *dependencies* in a sentence.  There is a significant amount of structure to language to study aiding in our model construction efforts.
# 
# - The document **["good"]** is certainly positive
# - The document **["not", "good"]** is certainly negative
# 
# But dictionary based models and even the naive vector similarity model above does not account for this.
# 
# To combat this, we typically consider n-grams in our models. . .


example_sentences = [
    "AAPL is a good buy",
    "AAPL is not a good buy"
]

fig_simple = make_subplots(
    rows=1, cols=2,
    column_widths=[0.53, 0.47],
    subplot_titles=[
        "<span style='color:#ffdf6b;font-size:18px'><b>{}</b></span>".format(example_sentences[0]),
        "<span style='color:#ffdf6b;font-size:18px'><b>{}</b></span>".format(example_sentences[1])
    ],
    shared_yaxes=True,
    horizontal_spacing=0.13
)

bar_colors = {
    "pos": "#6cc0ff",
    "neg": "#fd8585",
    "neu": "#AAAAAA"
}

for i, sent in enumerate(example_sentences):
    tokens, scores = get_token_scores(sent)
    colors = [
        bar_colors["pos"] if s > 0 else bar_colors["neg"] if s < 0 else bar_colors["neu"]
        for s in scores
    ]
    fig_simple.add_trace(
        go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            marker_line_color="#45474d",
            marker_line_width=1.7,
            text=[str(s) if s != 0 else "" for s in scores],
            textposition='auto',
            opacity=0.94,
            showlegend=False
        ), row=1, col=i+1
    )
    pos, neg, total = aggregate(scores)
    fig_simple.layout.annotations[i]['text'] += f"<br><span style='color:#a9f784;font-size:14px'>Pos={pos} Neg={neg} Sum={total}</span>"

for idx in range(1, 3):
    fig_simple.update_xaxes(
        title="<b>Token</b>" if idx == 1 else None,
        row=1, col=idx,
        color='#dbedf7',
        showline=True, linecolor="#45474d",
        tickangle=45,
        gridcolor="#45474d",
        tickfont=dict(size=15)
    )
fig_simple.update_yaxes(
    title="<b>Sentiment Score</b>",
    row=1, col=1,
    color='#dbedf7',
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d"
)
fig_simple.update_yaxes(
    color="#dbedf7",
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d",
    row=1, col=2
)

fig_simple.update_layout(
    width=950, height=360,
    plot_bgcolor="#282c34",
    paper_bgcolor="#282c34",
    font=dict(size=16, color="#dbedf7"),
    margin=dict(l=44, r=24, t=145, b=40),   # more top margin for head room
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.62,
        font=dict(color="#dbedf7", size=15),
        bgcolor='rgba(0,0,0,0)'
    ),
    title=dict(
        text="<b>Dictionary-based Sentiment Scores (ignores 'not')</b><br><span style='font-size:15px; color:#dbedf7'>No n-gram negation considered</span>",
        x=0.5, y=0.875,
        font=dict(size=20, color="#ffdf6b")
    ),
    hovermode='x'
)
fig_simple.show()



# 
#  ###### ______________________________________________________________________________________________________________________________________
#  
#  **Definition:**  
#  An *n-gram* is a contiguous sequence of *n* items (typically words or tokens) from a given text or speech sample.  
#  
#  - When *n = 1*, it is called a **unigram** (single word).  
#  - When *n = 2*, it is called a **bigram** (two-word sequence).  
#  - When *n = 3*, it is called a **trigram**, and so on.
#  
# We can then construct rules around certain combinations of sequences in this framework. . .


def get_token_scores_with_not_ngram(text, n_negate=2):
    """
    Returns tokens and sentiment scores, flipping polarity when 'not' or similar negations occur,
    negating the polarity of the next one or two tokens as a crude bigram/trigram approximation.
    Assumes get_token_scores returns (tokens, scores) where scores align with tokens.

    Args:
        text: str
        n_negate: int, number of tokens to flip after negation cue ("not", "never", etc.)
    """
    # Negate words following these cues
    negation_cues = {"not", "never", "no", "none", "n't"}
    tokens, scores = get_token_scores(text)
    new_scores = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() in negation_cues:
            new_scores.append(0)  # cue itself gets neutral score
            for j in range(1, n_negate+1):
                if i + j < len(scores):
                    new_scores.append(-scores[i + j])
                else:
                    break
            i += 1 + n_negate
        else:
            new_scores.append(scores[i])
            i += 1
    if len(new_scores) > len(tokens):
        tokens += [""] * (len(new_scores) - len(tokens))
    elif len(new_scores) < len(tokens):
        tokens = tokens[:len(new_scores)]
    return tokens, new_scores

example_sentences_ngram = [
    "AAPL is a good buy",
    "AAPL is not a good buy"
]

fig_ngram = make_subplots(
    rows=1, cols=2, 
    column_widths=[0.53, 0.47],
    subplot_titles=[
        "<span style='color:#ffdf6b;font-size:18px'><b>{}</b></span>".format(example_sentences_ngram[0]),
        "<span style='color:#ffdf6b;font-size:18px'><b>{}</b></span>".format(example_sentences_ngram[1])
    ],
    shared_yaxes=True,
    horizontal_spacing=0.13
)

ngram_bar_colors = {
    "pos": "#6cc0ff",
    "neg": "#fd8585",
    "neu": "#AAAAAA"
}

for i, sent in enumerate(example_sentences_ngram):
    tokens, scores = get_token_scores_with_not_ngram(sent, n_negate=2)
    colors = [
        ngram_bar_colors["pos"] if s > 0 else ngram_bar_colors["neg"] if s < 0 else ngram_bar_colors["neu"]
        for s in scores
    ]
    fig_ngram.add_trace(
        go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            marker_line_color="#45474d",
            marker_line_width=1.7,
            text=[str(s) if s != 0 else "" for s in scores],
            textposition='auto',
            opacity=0.94,
            showlegend=False
        ), row=1, col=i+1
    )
    pos, neg, total = aggregate(scores)
    fig_ngram.layout.annotations[i]['text'] += f"<br><span style='color:#a9f784;font-size:14px'>Pos={pos} Neg={neg} Sum={total}</span>"

for idx in range(1, 3):
    fig_ngram.update_xaxes(
        title="<b>Token</b>" if idx == 1 else None,
        row=1, col=idx,
        color='#dbedf7',
        showline=True, linecolor="#45474d",
        tickangle=45,
        gridcolor="#45474d",
        tickfont=dict(size=15)
    )
fig_ngram.update_yaxes(
    title="<b>Sentiment Score</b>",
    row=1, col=1,
    color='#dbedf7',
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d"
)
fig_ngram.update_yaxes(
    color="#dbedf7",
    showline=True, linecolor="#45474d",
    gridcolor="#45474d",
    zeroline=True, zerolinecolor="#45474d",
    row=1, col=2
)

# Updated: Give even *more* head room for the main title (t=145) and drop the y fraction close so title sits higher
fig_ngram.update_layout(
    width=950, height=360,
    plot_bgcolor="#282c34",
    paper_bgcolor="#282c34",
    font=dict(size=16, color="#dbedf7"),
    margin=dict(l=44, r=24, t=145, b=40),   # much more top margin for max head room
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.62,
        font=dict(color="#dbedf7", size=15),
        bgcolor='rgba(0,0,0,0)'
    ),
    title=dict(
        text=(
            "<b>Sentiment Scores with N-gram Negation</b><br>"
            "<span style='font-size:15px; color:#dbedf7'>Negation cues flip polarity of next 2 tokens</span>"
        ),
        x=0.5, y=0.875,  # y lowered for more visual spacing above title
        font=dict(size=20, color="#ffdf6b")
    ),
    hovermode='x'
)

fig_ngram.show()



# ###### ______________________________________________________________________________________________________________________________________
#  
# ##### More Advanced Dictionary Based Models
# 
# **VADER** is a prepackaged sentiment model in NLTK designed for social text [Hutto & Gilbert, 2014].  
# 
# This model does not include financial lexicon out of the box but it can be added directly to the model.
# 
# ##### Documents:
# 
# **Doc 1:** *Just read Apple's earnings call: despite CEO's optimism, the subtle language around supply chain delays and the lukewarm guidance for Q2 give me serious pause 🤔📉. Not as bullish as the headline suggests.*
# 
# **Doc 2:** *Wow! AAPL breaks through resistance and the CFO drops hints about bigger buybacks, while social sentiment explodes 🚀💰. This could be the setup for another legendary run!*


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import re

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()

tweets = [
    "Just read Apple's earnings call: despite CEO's optimism, the subtle language around supply chain delays and the lukewarm guidance for Q2 give me serious pause 🤔📉. Not as bullish as the headline suggests.",
    "Wow! AAPL breaks through resistance and the CFO drops hints about bigger buybacks, while social sentiment explodes 🚀💰. This could be the setup for another legendary run!"
]


for tweet in tweets:
    print(f"Tweet: {tweet}")
    scores = vader.polarity_scores(tweet)
    print("VADER Sentiment:", scores)
    print('\n')



# VADER takes steps towards more productive modeling by outsourcing the scoring procedure of individual tokens and considering n-grams in the overall sentiment score of the document.


# ###### ______________________________________________________________________________________________________________________________________


# ##### 🤖 Machine Learning Approaches
# 
# ##### In reality, sentiment analysis (and other NLP tasks) are a data science problem
# 
# There are a variety of approaches to sentiment analysis, we may also consider clustering documents by their information type, referenced instruments, so on and so forth to further enhance signal value.
# 
# In terms of strict positivity and negativity (long or short) that a document implies it is a near solved problem with LLMs, but of course, is still probabilistic.  The question is if there is signal stability in the cross section depending on your construction methodology.  We will discuss this later on.  
# 
# If you are interested in progressing toward more advanced sentiment models I have a full article available on Medium with code that develops a neural network for this task in PyTorch trained in GPU.  More advanced extensions include sequences and recurrent structures in the analysis. https://medium.com/quant-guild/how-to-build-a-neural-network-for-nlp-tasks-with-pytorch-and-gpu-da542591dbe1 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Named Entity Recognition (NER)
# 
# In any case with alternate data, if we are trying to build a strategy signal extraction is just one component.
# 
# We also need to know *what* to buy based on each signal. . .
# 
# For example, if **we have 1,000,000 documents** we can extract a signal sure, but we need to link them to an entity that is tradable (and produce a composite score for each entity) otherwise our signal is not very useful.
# 
# The mapping of an object to some noun is typically referred to as a named entity recognition problem. . .
# 
# ##### $\text{Example: "Buy appl stock"}$
# 
# <img src="aavs.png" alt="aavs" height="250"/>


# We can use **spacy** which has a pretrained NER model


!python -m spacy download en_core_web_sm


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("APPLE announced new products today.")

for ent in doc.ents:
    print(ent.text, ent.label_)


# This is model is however subject to the noise we discussed earlier


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("That APPLE was delicious")

for ent in doc.ents:
    print(ent.text, ent.label_)


# Roman, just use ChatGPT or another LLM this task is *easy*. . .
# 
# Transformer models can reason through this with relative ease and sure outputs can be structured
# 
# **However**, relative to neural networks and other vectorized models they are extremely inefficient especially if speed in a live setting is a concern to develop a basket of tradable assets **now**
# 
# There are more effective ways than this out of the box model to produce an entity to conduct subsequent signal analysis on (in the cross section for trading), but that is a deep sea so I will digress for now since we have all of the building blocks required to discuss the notion of trading these strategies 


# ###### ______________________________________________________________________________________________________________________________________


# ##### Cross-Sectional Equity Trading
# 
# If we have the ability to do the following
# 
# - Produce a signal for an alternate data source
# - Identify an equity associated with that signal
# - Combine into a set of data for a particular equity universe
# 
# We are able to construct a cross-sectional equity trading strategy


import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

np.random.seed(42)
n = 150

signal = np.linspace(-2, 2, n)
future_return = 0.12 * signal + 0.08 * np.random.randn(n)

n_bins = 5
quantile_bins = pd.qcut(signal, n_bins, labels=False, duplicates='drop')
df = pd.DataFrame({
    'signal': signal,
    'future_return': future_return,
    'bin': quantile_bins
})

bin_stats = df.groupby('bin').agg(
    signal_mean=('signal', 'mean'),
    future_return_mean=('future_return', 'mean'),
    count=('signal', 'count')
).reset_index()

bin_labels = [f"{i+1}" for i in bin_stats['bin']]
bar_colors = ["#6cc0ff"] * n_bins

dt = 1.0
T = n

# L/S Portfolio Path (zero-drift for demonstration)
mu = 0.25
sigma = 0.08
np.random.seed(99)
rets = np.random.normal(loc=mu * dt / 252, scale=sigma * np.sqrt(dt / 252), size=T)
cum_rets = np.cumsum(rets)
portfolio_path = 100 * np.exp(cum_rets)

# Long Leg (positive drift)
mu_long = 0.28  # slightly higher drift
np.random.seed(123)
rets_long = np.random.normal(loc=mu_long * dt / 252, scale=sigma * np.sqrt(dt / 252), size=T)
cum_rets_long = np.cumsum(rets_long)
portfolio_long = 100 * np.exp(cum_rets_long)

# Short Leg (negative drift)
mu_short = -0.11  # negative drift
np.random.seed(456)
rets_short = np.random.normal(loc=mu_short * dt / 252, scale=sigma * np.sqrt(dt / 252), size=T)
cum_rets_short = np.cumsum(rets_short)
portfolio_short = 100 * np.exp(cum_rets_short)

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.53, 0.47],
    subplot_titles=[
        "<span style='color:#ffdf6b;font-size:19px'><b>Mean Future Return by Signal Quantile</b></span>",
        "<span style='color:#ffdf6b;font-size:19px'><b>L/S Portfolio & Legs</b></span>"
    ],
    horizontal_spacing=0.13
)

# Left: signal buckets barplot
fig.add_trace(
    go.Bar(
        x=bin_labels,
        y=bin_stats["future_return_mean"],
        marker_color=bar_colors,
        name="Mean Future Return",
        showlegend=False
    ), row=1, col=1
)
fig.update_xaxes(
    title="<b>Signal Quantile Bucket</b>", row=1, col=1, color='#dbedf7',
    showline=True, linecolor="#45474d", tickmode='array',
    tickvals=bin_labels, tickfont=dict(size=15),
    gridcolor="#45474d"
)
fig.update_yaxes(
    title="<b>Mean<br>Future Return</b>", row=1, col=1, color='#dbedf7',
    showline=True, linecolor="#45474d", gridcolor="#45474d"
)

# Right: L/S Portfolio (yellow)
fig.add_trace(
    go.Scatter(
        x=np.arange(T),
        y=portfolio_path,
        mode="lines",
        line=dict(color='#ffdf6b', width=3),
        name="L/S Portfolio"
    ), row=1, col=2
)

# Right: Long Leg (blue)
fig.add_trace(
    go.Scatter(
        x=np.arange(T),
        y=portfolio_long,
        mode="lines",
        line=dict(color='#6cc0ff', width=2, dash='dash'),
        name="Long Leg"
    ), row=1, col=2
)

# Right: Short Leg (red)
fig.add_trace(
    go.Scatter(
        x=np.arange(T),
        y=portfolio_short,
        mode="lines",
        line=dict(color='#e46c6c', width=2, dash='dot'),
        name="Short Leg"
    ), row=1, col=2
)

fig.update_xaxes(title="<b>Time (from first trade)</b>", row=1, col=2, color='#dbedf7', showline=True, linecolor="#45474d", gridcolor="#45474d")
fig.update_yaxes(title="<b>Equity (Gross)</b>", row=1, col=2, color='#dbedf7', showline=True, linecolor="#45474d", gridcolor="#45474d")

fig.update_layout(
    width=950, height=440,
    plot_bgcolor="#282c34",
    paper_bgcolor="#282c34",
    font=dict(size=17, color="#dbedf7"),
    margin=dict(l=44, r=24, t=65, b=40),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.62,
        font=dict(color="#dbedf7", size=15),
        bgcolor='rgba(0,0,0,0)'
    ),
    title=dict(
        text=None, x=0.5, y=0.94,
        font=dict(size=23, color="#ffdf6b")),
    hovermode='x'
)


# ---


# #### 4.) 💭 Closing Thoughts and Future Topics
# 
# **TL;DW Executive Summary**
# - There is often an inverse relationship between data accessibility and opportunity in subsequent alphas
# - Alternative data (like text) can be rich of signal but is not without noise (i.e. DOGE vs DOGE)
# - Other quant fields like natural language processing (NLP) are concerned in many cases with extracting signal from noise
# - Doc-Term Matrices are just one quantitative representation of the qualitative information we see in text
# - Similarly scores and machine learning techniques can be applied directly to the Doc-Term matrix but largely become data science problems
# - Other techniques for signal extraction exist, the finance literature has long considered dictionary based approaches to produce sentiment scores for example 
# - Once we have a signal, we still need to know what to trade, this then becomes a named entity recognition problem to determine what to buy and sell
# - Again, this named entity recognition problem is not without noise and can impact our trading decisions
# - Once we have established means of producing a signal and instrument we can observe the signal in the cross-section of a basket of equities (not just one document but many documents, not just one firm but many firms) and go long the highest signal quantile and short the lowest signal quantile to produce a market-neutral portfolio aiming to profit from a structural inefficiency we observed in this alternate data
# 
# **Future Topics**
# 
# Technical Videos and Other Discussions
# 
# - How to Research Quant Trading Strategies
# - Cross-Sectional Equity Backtesting and Trading
# - AI for Quantitative Trading (Images and other Alt Data)
# - Advanced NLP Topics (Linguistics, Language Modeling, Transformers)
# - Advanced Markov Chains (Absorbing States, Communication Classes, Ergodicity and Stationary Distributions, . . .)
# - Non-Markovian Models (fractional Brownian motion, Volterra Process)
# - Deriving the Black-Scholes Equation: PDE, Analytical/Numerical Solutions
# - Kalman Filters and Non-Stationary (A Big Problem in Quant Modeling)
# - Risk-Neutral Measures (Complete vs Incomplete Markets)
# - Reinforcement Learning for Delta Hedging
# 
# [Ideas for Interactive Brokers Apps and Tutorials](https://www.interactivebrokers.com/mkt/?src=quantguildY&url=%2Fen%2Fwhyib%2Foverview.php)
# 
# - Live Neural Network Stochastic Volatility Model Calibration
# - Live Kalman Filter Model with Regime Dynamics (MCs/HMMs) 
# - Automated Delta-Neutral Trading System


# ---


# ####  $\text{Copyright © 2025 Quant Guild} \quad \quad \quad \quad \text{Author: Roman Paolucci}$

