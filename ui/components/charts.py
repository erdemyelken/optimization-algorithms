"""
Reusable chart functions for the Streamlit UI.
All functions accept matplotlib Figure or return Plotly figures.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.result import AggregatedResult


# ── Colour palette ───────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Plotly
_ALGO_COLORS: Dict[str, str] = {}


def _color(algo: str) -> str:
    if algo not in _ALGO_COLORS:
        _ALGO_COLORS[algo] = PALETTE[len(_ALGO_COLORS) % len(PALETTE)]
    return _ALGO_COLORS[algo]


# ── Convergence curves ───────────────────────────────────────────────────────

def convergence_plot(
    agg_results: List[AggregatedResult],
    log_scale: bool = True,
    show_std: bool = True,
    title: str = "Convergence Curves",
) -> go.Figure:
    """Mean ± std convergence curve for each algorithm."""
    fig = go.Figure()

    for agg in agg_results:
        x = list(range(1, len(agg.convergence_mean) + 1))
        y = agg.convergence_mean
        std = agg.convergence_std if show_std else [0] * len(y)
        col = _color(agg.algorithm_name)

        if show_std and agg.convergence_std:
            y_upper = [m + s for m, s in zip(y, std)]
            y_lower = [max(m - s, 1e-15) for m, s in zip(y, std)]
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=col,
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=agg.algorithm_name,
                line=dict(color=col, width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Best Fitness (log scale)" if log_scale else "Best Fitness",
        yaxis_type="log" if log_scale else "linear",
        template="plotly_dark",
        legend=dict(x=0.75, y=0.95),
        height=450,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ── Runtime comparison bar ────────────────────────────────────────────────────

def runtime_bar(
    agg_results: List[AggregatedResult],
    title: str = "Runtime Comparison",
) -> go.Figure:
    names = [a.algorithm_name for a in agg_results]
    means = [a.mean_runtime for a in agg_results]
    stds = [a.std_runtime for a in agg_results]
    colors = [_color(n) for n in names]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Algorithm",
        yaxis_title="Runtime (s)",
        template="plotly_dark",
        height=380,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ── Boxplot ───────────────────────────────────────────────────────────────────

def fitness_boxplot(
    runs_by_algo: Dict[str, List[float]],
    title: str = "Fitness Distribution",
    log_scale: bool = False,
) -> go.Figure:
    """Box plot of best fitness values per algorithm over multiple runs."""
    fig = go.Figure()
    for algo, vals in runs_by_algo.items():
        fig.add_trace(
            go.Box(
                y=vals,
                name=algo,
                marker_color=_color(algo),
                boxmean="sd",
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title="Best Fitness",
        yaxis_type="log" if log_scale else "linear",
        template="plotly_dark",
        height=400,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────────

def heatmap_figure(
    matrix: np.ndarray,
    x_labels: List[Any],
    y_labels: List[Any],
    x_title: str = "Parameter 1",
    y_title: str = "Parameter 2",
    metric_title: str = "Mean Fitness",
    colorscale: str = "Viridis",
    reverse_scale: bool = False,
) -> go.Figure:
    """Interactive heatmap for 2D parameter sweeps."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(v) for v in x_labels],
            y=[str(v) for v in y_labels],
            colorscale=colorscale + "_r" if reverse_scale else colorscale,
            colorbar=dict(title=metric_title),
            text=np.round(matrix, 4).astype(str),
            hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>{metric_title}: %{{z:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_dark",
        height=450,
        margin=dict(t=50, b=60, l=80, r=20),
    )
    return fig


# ── Dimension scaling line plot ───────────────────────────────────────────────

def dimension_scaling_plot(
    records: List[Any],
    y: str = "mean_runtime",
    y_label: str = "Mean Runtime (s)",
    title: str = "Dimension Scaling",
) -> go.Figure:
    """Line plot of a metric vs. dimension for multiple algorithms."""
    from collections import defaultdict
    by_algo: Dict[str, List] = defaultdict(list)
    for r in records:
        by_algo[r.algorithm_name].append(r)

    fig = go.Figure()
    for algo, recs in by_algo.items():
        recs = sorted(recs, key=lambda r: r.dimension)
        dims = [r.dimension for r in recs]
        vals = [getattr(r, y) for r in recs]
        fig.add_trace(
            go.Scatter(
                x=dims,
                y=vals,
                mode="lines+markers",
                name=algo,
                line=dict(color=_color(algo), width=2),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension",
        yaxis_title=y_label,
        template="plotly_dark",
        height=420,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ── Performance vs runtime scatter ────────────────────────────────────────────

def performance_runtime_scatter(
    agg_results: List[AggregatedResult],
    title: str = "Performance vs. Runtime Trade-off",
) -> go.Figure:
    fig = go.Figure()
    for agg in agg_results:
        fig.add_trace(
            go.Scatter(
                x=[agg.mean_runtime],
                y=[agg.mean_fitness],
                mode="markers+text",
                name=agg.algorithm_name,
                text=[agg.algorithm_name],
                textposition="top center",
                marker=dict(size=16, color=_color(agg.algorithm_name)),
                error_x=dict(type="data", array=[agg.std_runtime], visible=True),
                error_y=dict(type="data", array=[agg.std_fitness], visible=True),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Mean Runtime (s)",
        yaxis_title="Mean Fitness",
        template="plotly_dark",
        height=420,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ── Landscape plot (2D contour) ───────────────────────────────────────────────

def landscape_plot(func: Any, resolution: int = 80) -> go.Figure:
    """Interactive 2D landscape of a benchmark function."""
    lb, ub = func.bounds
    x = np.linspace(lb, ub, resolution)
    y = np.linspace(lb, ub, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func.evaluate(np.array([X[i, j], Y[i, j]])) for j in range(resolution)] for i in range(resolution)])

    fig = go.Figure(
        go.Contour(
            x=x, y=y, z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=False),
            colorbar=dict(title="f(x)"),
        )
    )
    fig.update_layout(
        title=f"{func.name} — 2D Landscape",
        xaxis_title="x₁",
        yaxis_title="x₂",
        template="plotly_dark",
        height=420,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig
