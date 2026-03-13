import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────
# Regime Map (SPY + shaded regimes)
# ─────────────────────────────────────────────

def plot_regime_map(df):

    import plotly.graph_objects as go

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="SPY",
            line=dict(color="white", width=2)
        )
    )

    # Detect regime change points
    regime_change = df["regime"] != df["regime"].shift()
    regime_starts = df.index[regime_change]

    for i in range(len(regime_starts)):

        start = regime_starts[i]

        if i < len(regime_starts) - 1:
            end = regime_starts[i + 1]
        else:
            end = df.index[-1]

        regime = df.loc[start, "regime"]
        color = config.REGIME_COLORS.get(regime, "#888")

        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.15,
            line_width=0
        )

    fig.update_layout(
        title="Market Regimes",
        xaxis_title="Date",
        yaxis_title="Price",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ─────────────────────────────────────────────
# Model Comparison (KMeans vs HMM)
# ─────────────────────────────────────────────

def plot_model_comparison(df_kmeans: pd.DataFrame, df_hmm: pd.DataFrame):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_kmeans.index,
            y=[1]*len(df_kmeans),
            mode="markers",
            marker=dict(
                size=6,
                color=df_kmeans["regime"].map(config.REGIME_COLORS)
            ),
            name="KMeans"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_hmm.index,
            y=[0]*len(df_hmm),
            mode="markers",
            marker=dict(
                size=6,
                color=df_hmm["regime"].map(config.REGIME_COLORS)
            ),
            name="HMM"
        )
    )

    fig.update_layout(
        title="Model Regime Comparison",
        yaxis=dict(
            tickvals=[0,1],
            ticktext=["HMM","KMeans"]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ─────────────────────────────────────────────
# Feature Distributions
# ─────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame):

    features = [
        "rolling_volatility",
        "sma_ratio",
        "rsi",
        "log_return"
    ]

    df_long = df.melt(
        id_vars="regime",
        value_vars=features,
        var_name="feature",
        value_name="value"
    )

    fig = px.box(
        df_long,
        x="regime",
        y="value",
        color="regime",
        facet_col="feature",
        color_discrete_map=config.REGIME_COLORS,
        title="Feature Distribution by Regime"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ─────────────────────────────────────────────
# Regime Statistics
# ─────────────────────────────────────────────

def compute_regime_statistics(df: pd.DataFrame):

    stats = df.groupby("regime").agg(
        mean_return=("log_return", "mean"),
        volatility=("rolling_volatility", "mean"),
        count=("regime", "count")
    )

    stats["pct_time"] = stats["count"] / stats["count"].sum()
    stats["sharpe"] = stats["mean_return"] / stats["volatility"]

    return stats


def plot_regime_frequency(stats: pd.DataFrame):

    fig = px.bar(
        stats,
        x=stats.index,
        y="pct_time",
        color=stats.index,
        color_discrete_map=config.REGIME_COLORS,
        title="Regime Frequency"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

# ─────────────────────────────────────────────
# Model Agreement %
# ─────────────────────────────────────────────

def compute_model_agreement(df_kmeans, df_hmm):

    agreement = (df_kmeans["regime"] == df_hmm["regime"]).mean()

    return agreement