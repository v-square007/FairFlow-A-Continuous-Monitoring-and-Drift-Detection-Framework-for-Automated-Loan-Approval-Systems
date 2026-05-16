import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# 1. APP SETUP & INITIALIZATION
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.SLATE],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "FairFlow - Loan Approval Monitoring"

# ---------------------------------------------------------------------------
# 2. DESIGN SYSTEM & PALETTE
# ---------------------------------------------------------------------------
THEME_BG = "#0f111a"
CARD_BG = "#161b26"
TEXT_MAIN = "#f8f9fa"
TEXT_MUTED = "#8b9bb4"

ACCENT_PURPLE = "#b042ff"
ACCENT_CYAN = "#00f0ff"
ACCENT_GREEN = "#00e676"
ACCENT_PINK = "#ff2a85"
ACCENT_ORANGE = "#ff9500"

# ---------------------------------------------------------------------------
# 3. DATA LOADING FUNCTIONS
# ---------------------------------------------------------------------------
def load_monitoring_data():
    """Load all monitoring data from artifacts"""
    base_path = "artifacts/monitoring"
    
    # Load monitoring summary
    with open(f"{base_path}/monitoring_summary.json", 'r') as f:
        summary = json.load(f)
    
    # Load batch processing history
    with open(f"{base_path}/batch_processing_history.json", 'r') as f:
        batch_history = json.load(f)
    
    # Load fairness monitoring history
    with open(f"{base_path}/fairness_monitoring_history.json", 'r') as f:
        fairness_history = json.load(f)
    
    # Load all alerts
    with open(f"{base_path}/alerts/all_alerts.json", 'r') as f:
        alerts = json.load(f)
    
    # Load drift reports
    drift_reports = []
    for i in range(6):
        try:
            with open(f"{base_path}/drift_reports/drift_batch_0{i}.json", 'r') as f:
                drift_reports.append(json.load(f))
        except:
            pass
    
    return summary, batch_history, fairness_history, alerts, drift_reports

def prepare_batch_dataframe(batch_history):
    """Convert batch history to DataFrame"""
    batches = batch_history['batches']
    data = []
    
    for batch in batches:
        data.append({
            'batch_id': batch['batch_id'],
            'timestamp': batch['timestamp'],
            'batch_size': batch['batch_size'],
            'approval_rate': batch['predictions']['approval_rate'],
            'auc_roc': batch['performance']['auc_roc'],
            'accuracy': batch['performance']['accuracy'],
            'approved': batch['predictions']['approved'],
            'rejected': batch['predictions']['rejected'],
            'mean_probability': batch['predictions']['mean_probability']
        })
    
    return pd.DataFrame(data)

def prepare_fairness_dataframe(fairness_history):
    """Convert fairness history to DataFrame"""
    history = fairness_history['history']
    data = []
    
    for entry in history:
        batch_id = entry['batch_id']
        timestamp = entry['timestamp']
        
        # Sex metrics
        sex_metrics = entry['fairness_metrics']['Sex']
        
        # Age group metrics
        age_metrics = entry['fairness_metrics']['Age_group']
        
        data.append({
            'batch_id': batch_id,
            'timestamp': timestamp,
            'sex_male_approval': sex_metrics['approval_rates']['male'],
            'sex_female_approval': sex_metrics['approval_rates']['female'],
            'sex_dp_diff': sex_metrics['demographic_parity_difference'],
            'sex_di_ratio': sex_metrics['disparate_impact_ratio'],
            'sex_eo_diff': sex_metrics['equal_opportunity_difference'],
            'age_dp_diff': age_metrics['demographic_parity_difference'],
            'age_di_ratio': age_metrics['disparate_impact_ratio'],
            'age_eo_diff': age_metrics['equal_opportunity_difference']
        })
    
    return pd.DataFrame(data)

def prepare_alerts_dataframe(alerts_data):
    """Convert alerts to DataFrame"""
    alerts_list = alerts_data['alerts']
    data = []
    
    for alert in alerts_list:
        data.append({
            'alert_id': alert['alert_id'],
            'timestamp': alert['timestamp'],
            'type': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'batch_id': alert['batch_id'],
            'attribute': alert['details']['attribute'],
            'value': alert['details']['value'],
            'threshold': alert['details']['threshold']
        })
    
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# 4. LOAD DATA
# ---------------------------------------------------------------------------
summary, batch_history, fairness_history, alerts_data, drift_reports = load_monitoring_data()
batch_df = prepare_batch_dataframe(batch_history)
fairness_df = prepare_fairness_dataframe(fairness_history)
alerts_df = prepare_alerts_dataframe(alerts_data)

# ---------------------------------------------------------------------------
# 5. VISUALIZATION GENERATORS
# ---------------------------------------------------------------------------
def create_performance_time_series():
    """Create performance metrics over time"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=batch_df['batch_id'], 
        y=batch_df['auc_roc'],
        name='AUC-ROC',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_CYAN),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=batch_df['batch_id'], 
        y=batch_df['accuracy'],
        name='Accuracy',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_PURPLE),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=batch_df['batch_id'], 
        y=batch_df['approval_rate'],
        name='Approval Rate',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_GREEN),
        marker=dict(size=8)
    ))
    
    # Add baseline reference
    baseline_auc = summary['baseline_performance']['auc_roc']
    fig.add_hline(y=baseline_auc, line_dash="dot", line_color=ACCENT_ORANGE,
                  annotation_text=f"Baseline AUC: {baseline_auc:.4f}")
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False, title="Score")
    )
    
    return fig

def create_fairness_metrics_chart():
    """Create fairness metrics over time"""
    fig = go.Figure()
    
    # Demographic Parity Difference for Sex
    fig.add_trace(go.Scatter(
        x=fairness_df['batch_id'],
        y=fairness_df['sex_dp_diff'],
        name='Sex DP Diff',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_CYAN),
        marker=dict(size=8)
    ))
    
    # Demographic Parity Difference for Age
    fig.add_trace(go.Scatter(
        x=fairness_df['batch_id'],
        y=fairness_df['age_dp_diff'],
        name='Age DP Diff',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_PINK),
        marker=dict(size=8)
    ))
    
    # Add violation threshold
    fig.add_hline(y=0.1, line_dash="dot", line_color="red",
                  annotation_text="Violation Threshold (0.1)")
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False, title="DP Difference")
    )
    
    return fig

def create_disparate_impact_chart():
    """Create disparate impact ratio chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fairness_df['batch_id'],
        y=fairness_df['sex_di_ratio'],
        name='Sex DI Ratio',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_GREEN),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=fairness_df['batch_id'],
        y=fairness_df['age_di_ratio'],
        name='Age DI Ratio',
        mode='lines+markers',
        line=dict(width=3, color=ACCENT_ORANGE),
        marker=dict(size=8)
    ))
    
    # Add threshold line
    fig.add_hline(y=0.8, line_dash="dot", line_color="red",
                  annotation_text="Minimum Threshold (0.8)")
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False, title="DI Ratio")
    )
    
    return fig

def create_approval_rate_breakdown():
    """Create approval rate breakdown by demographics"""
    latest = fairness_history['history'][-1]
    
    # Sex breakdown
    sex_data = latest['fairness_metrics']['Sex']
    sex_groups = list(sex_data['approval_rates'].keys())
    sex_rates = list(sex_data['approval_rates'].values())
    
    # Age breakdown
    age_data = latest['fairness_metrics']['Age_group']
    age_groups = list(age_data['approval_rates'].keys())
    age_rates = list(age_data['approval_rates'].values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sex_groups,
        y=sex_rates,
        name='Sex',
        marker=dict(color=ACCENT_CYAN),
        text=[f"{r:.2%}" for r in sex_rates],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=age_groups,
        y=age_rates,
        name='Age Group',
        marker=dict(color=ACCENT_PURPLE),
        text=[f"{r:.2%}" for r in age_rates],
        textposition='auto'
    ))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MUTED),
        xaxis=dict(showgrid=False, linecolor="#222938"),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False, title="Approval Rate"),
        barmode='group'
    )
    
    return fig

def create_alert_timeline():
    """Create alert timeline visualization"""
    color_map = {
        'critical': ACCENT_PINK,
        'warning': ACCENT_ORANGE,
        'info': ACCENT_CYAN
    }
    
    fig = px.scatter(
        alerts_df,
        x='batch_id',
        y='type',
        color='severity',
        color_discrete_map=color_map,
        hover_data=['message', 'attribute', 'value'],
        title=""
    )
    
    fig.update_traces(marker=dict(size=14))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False),
        yaxis=dict(showgrid=False, title="")
    )
    
    return fig

def create_drift_status_donut():
    """Create drift detection status donut"""
    total_features = len(drift_reports[0]['features']) if drift_reports else 0
    features_with_drift = sum(1 for dr in drift_reports for f in dr['features'].values() if f['drift_detected'])
    features_no_drift = (total_features * len(drift_reports)) - features_with_drift
    
    fig = go.Figure(data=[go.Pie(
        labels=['No Drift Detected', 'Drift Detected'],
        values=[features_no_drift, features_with_drift],
        hole=.7,
        marker=dict(colors=[ACCENT_GREEN, ACCENT_PINK]),
        textinfo='percent',
        textfont=dict(color=TEXT_MAIN, size=12)
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1, 
                   font=dict(color=TEXT_MUTED)),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ---------------------------------------------------------------------------
# 6. REUSABLE UI COMPONENTS
# ---------------------------------------------------------------------------
def create_metric_card(title, value, subtitle, delta_positive=True, delta_str=""):
    delta_color = ACCENT_GREEN if delta_positive else ACCENT_PINK
    return html.Div(
        style={
            "backgroundColor": CARD_BG, "borderRadius": "16px", "padding": "24px",
            "border": "1px solid #222938", "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.2)"
        },
        children=[
            html.Div(title, style={"color": TEXT_MUTED, "fontSize": "14px", "fontWeight": "600", 
                                   "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.Div(value, style={"color": TEXT_MAIN, "fontSize": "36px", "fontWeight": "700", 
                                   "margin": "8px 0"}),
            html.Div([
                html.Span(f"{'▲' if delta_positive else '▼'} {delta_str} " if delta_str else "", 
                         style={"color": delta_color, "fontWeight": "700", "marginRight": "6px"}),
                html.Span(subtitle, style={"color": TEXT_MUTED, "fontSize": "12px"})
            ], style={"display": "flex", "alignItems": "center"})
        ]
    )

def create_graph_card(title, graph_object):
    return html.Div(
        style={
            "backgroundColor": CARD_BG, "borderRadius": "16px", "padding": "24px",
            "border": "1px solid #222938", "height": "100%", "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.2)"
        },
        children=[
            html.H5(title, style={"color": TEXT_MAIN, "fontWeight": "600", "marginBottom": "20px", 
                                 "fontSize": "16px", "letterSpacing": "0.5px"}),
            dcc.Graph(figure=graph_object, config={'displayModeBar': False}, style={"height": "280px"})
        ]
    )

# ---------------------------------------------------------------------------
# 7. APP LAYOUT
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": THEME_BG, "minHeight": "100vh", "padding": "0", 
           "fontFamily": "'Inter', 'Segoe UI', sans-serif"},
    children=[
        # Navigation Header
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Div(style={"width": "12px", "height": "12px", 
                                                        "backgroundColor": ACCENT_CYAN, "borderRadius": "50%"})),
                                dbc.Col(dbc.NavbarBrand("FAIRFLOW // LOAN MONITORING PLATFORM", 
                                                       className="ms-2", 
                                                       style={"color": TEXT_MAIN, "fontWeight": "800", 
                                                             "fontSize": "16px", "letterSpacing": "2px"})),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    html.Div(
                        children=[
                            html.Span(f"MONITORING: {summary['monitoring_summary']['total_batches_processed']} BATCHES", 
                                     style={"color": ACCENT_GREEN, "fontSize": "12px", "fontWeight": "700", 
                                           "letterSpacing": "1px", "backgroundColor": "rgba(0, 230, 118, 0.1)", 
                                           "padding": "6px 14px", "borderRadius": "20px"})
                        ]
                    )
                ],
                fluid=True,
            ),
            color="#0b0d14",
            dark=True,
            style={"borderBottom": "1px solid #1c2130", "padding": "15px 30px"}
        ),
        
        # Main Content
        dbc.Container(
            fluid=True,
            style={"padding": "32px 40px"},
            children=[
                # KPI Cards
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(create_metric_card(
                            "AUC-ROC Score", 
                            f"{summary['baseline_performance']['auc_roc']:.4f}",
                            "baseline performance",
                            delta_positive=True,
                            delta_str=""
                        ), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card(
                            "Model Accuracy", 
                            f"{summary['baseline_performance']['accuracy']:.2%}",
                            "on test dataset",
                            delta_positive=True,
                            delta_str=""
                        ), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card(
                            "Total Applications", 
                            f"{summary['batch_processing_summary']['total_applications']:,}",
                            f"approved: {summary['batch_processing_summary']['total_approved']:,}",
                            delta_positive=True,
                            delta_str=f"{summary['batch_processing_summary']['overall_approval_rate']:.1%}"
                        ), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card(
                            "Critical Alerts", 
                            f"{alerts_data['by_severity']['critical']}",
                            "fairness violations detected",
                            delta_positive=False,
                            delta_str=f"{summary['monitoring_summary']['batches_with_fairness_violations']}/6"
                        ), xs=12, sm=6, lg=3),
                    ]
                ),
                
                # Performance Charts
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(create_graph_card(
                            "Model Performance Metrics Over Batches", 
                            create_performance_time_series()
                        ), lg=8, md=12),
                        dbc.Col(create_graph_card(
                            "Drift Detection Status", 
                            create_drift_status_donut()
                        ), lg=4, md=12),
                    ]
                ),
                
                # Fairness Metrics
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(create_graph_card(
                            "Demographic Parity Difference (Fairness Metric)", 
                            create_fairness_metrics_chart()
                        ), lg=6, md=12),
                        dbc.Col(create_graph_card(
                            "Disparate Impact Ratio (80% Rule Compliance)", 
                            create_disparate_impact_chart()
                        ), lg=6, md=12),
                    ]
                ),
                
                # Approval Breakdown and Alerts
                dbc.Row(
                    className="g-4",
                    children=[
                        dbc.Col(create_graph_card(
                            "Latest Batch: Approval Rates by Demographics", 
                            create_approval_rate_breakdown()
                        ), lg=5, md=12),
                        dbc.Col(create_graph_card(
                            "Alert Timeline (All Fairness Violations)", 
                            create_alert_timeline()
                        ), lg=7, md=12),
                    ]
                ),
                
                # Alert Summary Table
                dbc.Row(
                    className="g-4 mt-4",
                    children=[
                        dbc.Col(
                            html.Div(
                                style={
                                    "backgroundColor": CARD_BG, "borderRadius": "16px", "padding": "24px",
                                    "border": "1px solid #222938", "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.2)"
                                },
                                children=[
                                    html.H5("Recent Alert Summary", 
                                           style={"color": TEXT_MAIN, "fontWeight": "600", 
                                                 "marginBottom": "20px", "fontSize": "16px"}),
                                    html.Div(
                                        className="table-responsive",
                                        children=dbc.Table(
                                            [
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("ALERT ID", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("BATCH", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("TYPE", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("ATTRIBUTE", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("VALUE", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("THRESHOLD", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("STATUS", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"})
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td(alert['alert_id'], style={"color": TEXT_MAIN, "fontSize": "13px"}),
                                                        html.Td(alert['batch_id'], style={"color": TEXT_MAIN, "fontSize": "13px"}),
                                                        html.Td(alert['type'].replace('_', ' ').title(), style={"color": TEXT_MAIN, "fontSize": "13px"}),
                                                        html.Td(alert['attribute'], style={"color": TEXT_MAIN, "fontSize": "13px"}),
                                                        html.Td(f"{alert['value']:.4f}", style={"color": ACCENT_PINK, "fontSize": "13px", "fontWeight": "600"}),
                                                        html.Td(f"{alert['threshold']:.2f}", style={"color": TEXT_MUTED, "fontSize": "13px"}),
                                                        html.Td(
                                                            html.Span("● ACTIVE", style={"color": ACCENT_PINK, "fontSize": "12px", "fontWeight": "700"}),
                                                            style={"fontSize": "13px"}
                                                        )
                                                    ], style={"borderBottom": "1px solid #1c2130"})
                                                    for alert in alerts_df.head(10).to_dict('records')
                                                ])
                                            ],
                                            borderless=True,
                                            hover=True,
                                            style={"verticalAlign": "middle"}
                                        )
                                    )
                                ]
                            ),
                            lg=12
                        )
                    ]
                )
            ]
        )
    ]
)

# ---------------------------------------------------------------------------
# 8. RUN SERVER
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)