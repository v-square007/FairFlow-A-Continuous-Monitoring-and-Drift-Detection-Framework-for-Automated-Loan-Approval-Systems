"""
Dashboard Chart Components
==========================
Reusable Plotly chart components for the monitoring dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from dashboard_config import COLORS, CHART_CONFIG


def create_metric_card_figure(value: float, 
                              label: str,
                              baseline: float = None,
                              format_type: str = "percent") -> go.Figure:
    """
    Create a metric card visualization.
    
    Args:
        value: Current metric value
        label: Metric label
        baseline: Baseline value for comparison
        format_type: "percent", "decimal", or "integer"
    
    Returns:
        Plotly figure
    """
    # Format value
    if format_type == "percent":
        display_value = f"{value*100:.1f}%"
    elif format_type == "decimal":
        display_value = f"{value:.4f}"
    else:
        display_value = f"{int(value)}"
    
    # Calculate change
    change_text = ""
    if baseline is not None:
        change = value - baseline
        change_pct = (change / baseline * 100) if baseline != 0 else 0
        arrow = "▲" if change > 0 else "▼" if change < 0 else "●"
        color = COLORS["success"] if change > 0 else COLORS["danger"] if change < 0 else COLORS["info"]
        change_text = f"{arrow} {abs(change_pct):.1f}% from baseline"
    
    fig = go.Figure()
    
    fig.add_annotation(
        text=display_value,
        x=0.5, y=0.6,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=48, color=COLORS["primary"], family=CHART_CONFIG["font"]["family"]),
        align="center"
    )
    
    fig.add_annotation(
        text=label,
        x=0.5, y=0.3,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color=COLORS["text_secondary"]),
        align="center"
    )
    
    if change_text:
        fig.add_annotation(
            text=change_text,
            x=0.5, y=0.1,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color=color),
            align="center"
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def create_performance_trend_chart(df: pd.DataFrame, 
                                   baseline: Dict = None) -> go.Figure:
    """
    Create performance metrics trend chart.
    
    Args:
        df: DataFrame with columns: batch_num, auc_roc, accuracy, bad_recall
        baseline: Baseline metrics dict
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # AUC-ROC trend
    fig.add_trace(go.Scatter(
        x=df["batch_num"],
        y=df["auc_roc"],
        name="AUC-ROC",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8)
    ))
    
    # Accuracy trend
    fig.add_trace(go.Scatter(
        x=df["batch_num"],
        y=df["accuracy"],
        name="Accuracy",
        mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=3),
        marker=dict(size=8)
    ))
    
    # Bad recall trend
    fig.add_trace(go.Scatter(
        x=df["batch_num"],
        y=df["bad_recall"],
        name="Bad Recall",
        mode="lines+markers",
        line=dict(color=COLORS["warning"], width=3),
        marker=dict(size=8)
    ))
    
    # Add baseline reference lines
    if baseline:
        fig.add_hline(
            y=baseline.get("auc_roc", 0),
            line_dash="dash",
            line_color=COLORS["text_secondary"],
            annotation_text="Baseline AUC",
            annotation_position="right"
        )
    
    fig.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Batch Number",
        yaxis_title="Score",
        **CHART_CONFIG,
        hovermode="x unified"
    )
    
    return fig


def create_fairness_trend_chart(df: pd.DataFrame,
                                baseline: Dict = None) -> go.Figure:
    """
    Create fairness metrics trend chart.
    
    Args:
        df: DataFrame with columns: batch_num, demographic_parity, disparate_impact
        baseline: Baseline fairness metrics
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Demographic Parity", "Disparate Impact"),
        vertical_spacing=0.15
    )
    
    # Demographic Parity
    fig.add_trace(
        go.Scatter(
            x=df["batch_num"],
            y=df["demographic_parity"],
            name="Demographic Parity",
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_hline(
        y=0.1,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="Threshold (0.1)",
        annotation_position="right",
        row=1, col=1
    )
    
    # Disparate Impact
    fig.add_trace(
        go.Scatter(
            x=df["batch_num"],
            y=df["disparate_impact"],
            name="Disparate Impact",
            mode="lines+markers",
            line=dict(color=COLORS["secondary"], width=3),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="Threshold (0.8)",
        annotation_position="right",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Batch Number", row=2, col=1)
    fig.update_yaxes(title_text="Difference", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    
    fig.update_layout(
        title="Fairness Metrics Over Time",
        height=600,
        showlegend=False,
        **{k: v for k, v in CHART_CONFIG.items() if k != "height"}
    )
    
    return fig


def create_drift_heatmap(batch_ids: List[str],
                        features: List[str],
                        drift_matrix: np.ndarray) -> go.Figure:
    """
    Create feature drift heatmap.
    
    Args:
        batch_ids: List of batch IDs
        features: List of feature names
        drift_matrix: Binary matrix (1=drift, 0=no drift)
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=drift_matrix.T,
        x=batch_ids,
        y=features,
        colorscale=[
            [0, COLORS["success"]],
            [1, COLORS["danger"]]
        ],
        showscale=False,
        hovertemplate="Batch: %{x}<br>Feature: %{y}<br>Drift: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Feature Drift Detection Heatmap",
        xaxis_title="Batch",
        yaxis_title="Feature",
        **CHART_CONFIG
    )
    
    return fig


def create_confusion_matrix_chart(cm: Dict) -> go.Figure:
    """
    Create confusion matrix visualization.
    
    Args:
        cm: Confusion matrix dict with keys: true_negative, false_positive, 
            false_negative, true_positive
    
    Returns:
        Plotly figure
    """
    z = [
        [cm.get("true_negative", 0), cm.get("false_positive", 0)],
        [cm.get("false_negative", 0), cm.get("true_positive", 0)]
    ]
    
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(z[i][j]),
                    showarrow=False,
                    font=dict(size=20, color="white")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Bad", "Predicted Good"],
        y=["Actual Bad", "Actual Good"],
        colorscale="Blues",
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        annotations=annotations,
        **CHART_CONFIG,
        height=400
    )
    
    return fig


def create_approval_rate_gauge(approval_rate: float) -> go.Figure:
    """
    Create gauge chart for approval rate.
    
    Args:
        approval_rate: Approval rate (0-1)
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=approval_rate * 100,
        title={"text": "Approval Rate"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 50], "color": COLORS["danger"]},
                {"range": [50, 75], "color": COLORS["warning"]},
                {"range": [75, 100], "color": COLORS["success"]}
            ],
            "threshold": {
                "line": {"color": COLORS["text_primary"], "width": 4},
                "thickness": 0.75,
                "value": 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_alert_timeline(alerts: List[Dict]) -> go.Figure:
    """
    Create alert timeline chart.
    
    Args:
        alerts: List of alert dictionaries
    
    Returns:
        Plotly figure
    """
    if not alerts:
        fig = go.Figure()
        fig.add_annotation(
            text="No alerts to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )
        fig.update_layout(height=300)
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(alerts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    # Color map
    color_map = {
        "critical": COLORS["danger"],
        "warning": COLORS["warning"],
        "info": COLORS["info"]
    }
    
    df["color"] = df["severity"].map(color_map)
    
    fig = go.Figure()
    
    for severity in ["critical", "warning", "info"]:
        df_severity = df[df["severity"] == severity]
        if not df_severity.empty:
            fig.add_trace(go.Scatter(
                x=df_severity["timestamp"],
                y=df_severity.index,
                mode="markers",
                name=severity.capitalize(),
                marker=dict(
                    size=12,
                    color=color_map[severity],
                    symbol="diamond"
                ),
                text=df_severity["message"],
                hovertemplate="%{text}<br>%{x}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Alert Timeline",
        xaxis_title="Time",
        yaxis_title="Alert Index",
        **CHART_CONFIG
    )
    
    return fig


def create_batch_comparison_chart(df: pd.DataFrame,
                                  selected_batches: List[str]) -> go.Figure:
    """
    Create batch comparison bar chart.
    
    Args:
        df: Performance trend DataFrame
        selected_batches: List of batch IDs to compare
    
    Returns:
        Plotly figure
    """
    df_selected = df[df["batch_id"].isin(selected_batches)]
    
    fig = go.Figure()
    
    metrics = ["auc_roc", "accuracy", "bad_recall", "good_recall"]
    metric_names = ["AUC-ROC", "Accuracy", "Bad Recall", "Good Recall"]
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        fig.add_trace(go.Bar(
            name=name,
            x=df_selected["batch_id"],
            y=df_selected[metric],
            marker_color=COLORS["chart_colors"][i]
        ))
    
    fig.update_layout(
        title="Batch Performance Comparison",
        xaxis_title="Batch",
        yaxis_title="Score",
        barmode="group",
        **CHART_CONFIG
    )
    
    return fig


def create_drift_percentage_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create drift percentage over time chart.
    
    Args:
        df: Drift summary DataFrame
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["batch_num"],
        y=df["drift_percentage"],
        marker_color=COLORS["warning"],
        name="Drift %",
        text=df["drift_percentage"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Feature Drift Percentage by Batch",
        xaxis_title="Batch Number",
        yaxis_title="Drift Percentage (%)",
        **CHART_CONFIG
    )
    
    return fig


def create_fairness_violation_chart(fairness_reports: List[Dict]) -> go.Figure:
    """
    Create stacked bar chart of fairness violations.
    
    Args:
        fairness_reports: List of fairness report dictionaries
    
    Returns:
        Plotly figure
    """
    if not fairness_reports:
        fig = go.Figure()
        fig.add_annotation(
            text="No fairness data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    batch_ids = [r["batch_id"] for r in fairness_reports]
    critical = [r["summary"]["critical_violations"] for r in fairness_reports]
    warning = [r["summary"]["warning_violations"] for r in fairness_reports]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Critical",
        x=batch_ids,
        y=critical,
        marker_color=COLORS["danger"]
    ))
    
    fig.add_trace(go.Bar(
        name="Warning",
        x=batch_ids,
        y=warning,
        marker_color=COLORS["warning"]
    ))
    
    fig.update_layout(
        title="Fairness Violations by Batch",
        xaxis_title="Batch",
        yaxis_title="Number of Violations",
        barmode="stack",
        **CHART_CONFIG
    )
    
    return fig


if __name__ == "__main__":
    print("Dashboard Chart Components Loaded")
    print("Available components: metric_card, performance_trend, fairness_trend, drift_heatmap, etc.")