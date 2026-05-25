import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_loader import SENSOR_COLS, SENSOR_LABELS, CONDITION_LABELS, CONDITION_COLORS

_EMPTY = go.Figure().update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def condition_donut(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _EMPTY
    counts = df["engine_condition"].value_counts().sort_index(ascending=False)
    labels = [CONDITION_LABELS.get(i, str(i)) for i in counts.index]
    colors = [CONDITION_COLORS.get(i, "#95a5a6") for i in counts.index]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} readings<extra></extra>",
    ))
    fig.update_layout(
        title="Engine Condition Distribution",
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
        height=350,
    )
    return fig


def sensor_means_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _EMPTY
    col_mins = df[SENSOR_COLS].min()
    col_maxs = df[SENSOR_COLS].max()

    fig = go.Figure()
    for cond in sorted(df["engine_condition"].unique(), reverse=True):
        subset = df[df["engine_condition"] == cond]
        means = subset[SENSOR_COLS].mean()
        normalized = (means - col_mins) / (col_maxs - col_mins + 1e-9)
        fig.add_trace(go.Bar(
            name=CONDITION_LABELS.get(cond, str(cond)),
            x=[SENSOR_LABELS[c] for c in SENSOR_COLS],
            y=normalized.values,
            marker_color=CONDITION_COLORS.get(cond, "#95a5a6"),
            hovertemplate="<b>%{x}</b><br>Normalized mean: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="Normalized Sensor Means by Condition",
        barmode="group",
        yaxis_title="Normalized Value (0–1)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(t=60, b=80, l=50, r=10),
        height=350,
    )
    return fig


def sensor_boxplots(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _EMPTY
    subplot_titles = [SENSOR_LABELS[c] for c in SENSOR_COLS]
    fig = make_subplots(rows=2, cols=3, subplot_titles=subplot_titles)
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for col, (row, col_pos) in zip(SENSOR_COLS, positions):
        for cond in sorted(df["engine_condition"].unique(), reverse=True):
            subset = df[df["engine_condition"] == cond][col]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=CONDITION_LABELS.get(cond, str(cond)),
                    marker_color=CONDITION_COLORS.get(cond, "#95a5a6"),
                    showlegend=(col == SENSOR_COLS[0]),
                    legendgroup=CONDITION_LABELS.get(cond, str(cond)),
                    boxpoints=False,
                ),
                row=row, col=col_pos,
            )

    fig.update_layout(
        title="Sensor Distributions by Engine Condition",
        height=500,
        margin=dict(t=60, b=20, l=50, r=10),
        boxmode="group",
        legend=dict(orientation="h", y=1.5, x=0.5, xanchor="center"),

    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or len(df) < 2:
        return _EMPTY
    all_cols = SENSOR_COLS + ["engine_condition"]
    labels = [SENSOR_LABELS.get(c, c) for c in SENSOR_COLS] + ["Engine Condition"]
    corr = df[all_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlation Matrix",
        margin=dict(t=50, b=100, l=130, r=10),
        height=420,
    )
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 2000) -> go.Figure:
    if df.empty:
        return _EMPTY
    sample_df = df.sample(min(max_points, len(df)), random_state=42)
    fig = go.Figure()
    for cond in sorted(df["engine_condition"].unique(), reverse=True):
        subset = sample_df[sample_df["engine_condition"] == cond]
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            name=CONDITION_LABELS.get(cond, str(cond)),
            marker=dict(color=CONDITION_COLORS.get(cond, "#95a5a6"), size=4, opacity=0.6),
            hovertemplate=(
                f"{SENSOR_LABELS.get(x_col, x_col)}: %{{x:.2f}}<br>"
                f"{SENSOR_LABELS.get(y_col, y_col)}: %{{y:.2f}}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title=SENSOR_LABELS.get(x_col, x_col),
        yaxis_title=SENSOR_LABELS.get(y_col, y_col),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(t=20, b=50, l=60, r=10),
        height=370,
    )
    return fig


def line_chart(df: pd.DataFrame, sensor_col: str, max_points: int = 3000) -> go.Figure:
    if df.empty:
        return _EMPTY
    step = max(1, len(df) // max_points)
    sampled = df.iloc[::step]
    fig = go.Figure()
    for cond in sorted(df["engine_condition"].unique(), reverse=True):
        subset = sampled[sampled["engine_condition"] == cond]
        fig.add_trace(go.Scatter(
            x=subset.index,
            y=subset[sensor_col],
            mode="markers",
            name=CONDITION_LABELS.get(cond, str(cond)),
            marker=dict(color=CONDITION_COLORS.get(cond, "#95a5a6"), size=3, opacity=0.7),
        ))
    fig.update_layout(
        xaxis_title="Reading Index",
        yaxis_title=SENSOR_LABELS.get(sensor_col, sensor_col),
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center"),
        margin=dict(t=20, b=50, l=60, r=10),
        height=350,
    )
    return fig
