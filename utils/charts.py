"""
Automated EDA — deterministic chart generation using Plotly.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any


PALETTE = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg": "#0f1117",
    "surface": "#1e2130",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}


def _base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=14, family="Inter, sans-serif")),
        height=height,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif", size=11),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#2d3748", linecolor="#2d3748", tickfont=dict(color=PALETTE["muted"])),
        yaxis=dict(gridcolor="#2d3748", linecolor="#2d3748", tickfont=dict(color=PALETTE["muted"])),
    )


def plot_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    series = df[col].dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])

    fig.add_trace(go.Histogram(
        x=series, nbinsx=30, name=col,
        marker_color=PALETTE["primary"], opacity=0.8
    ), row=1, col=1)

    fig.add_trace(go.Box(
        y=series, name=col, boxpoints="outliers",
        marker_color=PALETTE["accent"], line_color=PALETTE["accent"]
    ), row=1, col=2)

    layout = _base_layout(f"Distribution of {col}", height=350)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    for i in [1, 2]:
        fig.update_xaxes(gridcolor="#2d3748", linecolor="#2d3748", row=1, col=i)
        fig.update_yaxes(gridcolor="#2d3748", linecolor="#2d3748", row=1, col=i)
    return fig


def plot_categorical(df: pd.DataFrame, col: str, max_cats: int = 15) -> go.Figure:
    vc = df[col].value_counts().head(max_cats)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["success"], PALETTE["warning"]] * 5

    fig = go.Figure(go.Bar(
        x=vc.index.astype(str),
        y=vc.values,
        marker_color=colors[:len(vc)],
        text=vc.values,
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=10),
    ))
    fig.update_layout(**_base_layout(f"Value Counts: {col}", height=350))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    cols = numeric_df.columns.tolist()[:20]
    corr = numeric_df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0, "#ef4444"], [0.5, PALETTE["surface"]], [1, PALETTE["primary"]]
        ],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(tickfont=dict(color=PALETTE["muted"])),
    ))
    fig.update_layout(**_base_layout("Correlation Heatmap", height=500))
    return fig


def plot_missing_values(profile: Dict[str, Any]) -> go.Figure:
    cols_with_missing = {
        col: info["missing_pct"]
        for col, info in profile["columns"].items()
        if info["missing_count"] > 0
    }
    if not cols_with_missing:
        return None

    sorted_items = sorted(cols_with_missing.items(), key=lambda x: -x[1])
    names, pcts = zip(*sorted_items)

    colors = [PALETTE["danger"] if p > 30 else PALETTE["warning"] if p > 10 else PALETTE["success"] for p in pcts]

    fig = go.Figure(go.Bar(
        y=list(names), x=list(pcts), orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=10),
    ))
    layout = _base_layout("Missing Values by Column", height=max(300, len(names) * 28))
    layout["xaxis"]["title"] = "% Missing"
    fig.update_layout(**layout)
    return fig


def plot_health_breakdown(health: Dict[str, Any]) -> go.Figure:
    breakdown = health["breakdown"]
    categories = list(breakdown.keys())
    scores = [100 - breakdown[c]["deduction"] for c in categories]

    color_map = {"Excellent": PALETTE["success"], "Good": PALETTE["accent"],
                 "Moderate": PALETTE["warning"], "Poor": PALETTE["danger"]}
    colors = [color_map.get(breakdown[c]["label"], PALETTE["primary"]) for c in categories]

    fig = go.Figure(go.Bar(
        x=categories, y=scores,
        marker_color=colors,
        text=[f"{s}/100" for s in scores],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=11),
    ))
    layout = _base_layout("Dataset Health Breakdown", height=350)
    layout["yaxis"]["range"] = [0, 115]
    fig.update_layout(**layout)
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    vc = df[target_col].value_counts()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 15

    if is_numeric:
        fig = go.Figure(go.Histogram(
            x=df[target_col].dropna(), nbinsx=40,
            marker_color=PALETTE["primary"], opacity=0.8,
        ))
    else:
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str), y=vc.values,
            marker_color=[PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"],
                          PALETTE["success"], PALETTE["warning"]][:len(vc)],
            text=vc.values, textposition="outside",
            textfont=dict(color=PALETTE["text"]),
        ))
    fig.update_layout(**_base_layout(f"Target Distribution: {target_col}", height=350))
    return fig


def plot_feature_importance(features: List[Dict[str, Any]]) -> go.Figure:
    if not features:
        return None
    top = features[:12]
    names = [f["feature"] for f in top]
    scores = [f["combined_score"] for f in top]

    fig = go.Figure(go.Bar(
        y=names[::-1], x=scores[::-1], orientation="h",
        marker_color=PALETTE["primary"],
        marker=dict(
            color=scores[::-1],
            colorscale=[[0, PALETTE["secondary"]], [1, PALETTE["accent"]]],
            showscale=False,
        ),
        text=[f"{s:.3f}" for s in scores[::-1]],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=10),
    ))
    layout = _base_layout("Feature Importance (MI + Correlation)", height=max(350, len(top) * 32))
    layout["xaxis"]["title"] = "Combined Score"
    fig.update_layout(**layout)
    return fig


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str, color_col: Optional[str] = None) -> go.Figure:
    sample = df.sample(min(2000, len(df)), random_state=42)

    if color_col and color_col in df.columns and df[color_col].nunique() <= 10:
        fig = px.scatter(sample, x=col_x, y=col_y, color=color_col,
                         color_discrete_sequence=px.colors.qualitative.Vivid)
    else:
        fig = go.Figure(go.Scatter(
            x=sample[col_x], y=sample[col_y], mode="markers",
            marker=dict(color=PALETTE["primary"], opacity=0.5, size=4),
        ))
    fig.update_layout(**_base_layout(f"{col_x} vs {col_y}", height=400))
    return fig


def plot_ml_readiness(readiness: Dict[str, Any]) -> go.Figure:
    items = {k: v for k, v in readiness.items() if k != "overall"}
    categories = list(items.keys())
    scores = [v["score"] for v in items.values()]

    # Radar chart
    categories_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=scores_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(99,102,241,0.2)",
        line=dict(color=PALETTE["primary"], width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=PALETTE["muted"]),
                            gridcolor="#2d3748"),
            angularaxis=dict(tickfont=dict(color=PALETTE["text"]), gridcolor="#2d3748"),
            bgcolor=PALETTE["surface"],
        ),
        paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(text="ML Readiness Radar", font=dict(color=PALETTE["text"])),
        height=380,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
