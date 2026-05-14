"""
FairFlow Monitoring Dashboard
==============================
Main Dash application for real-time model monitoring.

Run with: python dashboard_app.py
Access at: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime

# Import custom modules
from dashboard_config import COLORS, DASHBOARD_CONFIG, LAYOUT, DASHBOARD_PAGES
from dashboard_data_loader import MonitoringDataLoader
from dashboard_components import (
    create_metric_card_figure, create_performance_trend_chart,
    create_fairness_trend_chart, create_drift_heatmap,
    create_confusion_matrix_chart, create_approval_rate_gauge,
    create_alert_timeline, create_batch_comparison_chart,
    create_drift_percentage_chart, create_fairness_violation_chart
)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=DASHBOARD_CONFIG["title"]
)

# Initialize data loader
data_loader = MonitoringDataLoader()

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_navbar():
    """Create navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H3("⚖️ FairFlow", className="text-white mb-0",
                               style={"fontWeight": "700"}),
                        html.Small("Credit Risk Monitoring Dashboard",
                                 className="text-white-50")
                    ]),
                    width="auto"
                ),
            ], align="center"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Small("Last Updated: ", className="text-white-50 me-2"),
                        html.Span(id="last-update-time", className="text-white")
                    ]),
                    width="auto"
                ),
                dbc.Col(
                    dbc.Button(
                        "🔄 Refresh",
                        id="refresh-button",
                        color="light",
                        size="sm",
                        outline=True,
                        className="ms-3"
                    ),
                    width="auto"
                )
            ], align="center")
        ], fluid=True),
        color=COLORS["sidebar_bg"],
        dark=True,
        className="mb-4",
        style={"padding": "1rem 0"}
    )


def create_sidebar():
    """Create sidebar with page navigation."""
    nav_items = []
    for page in DASHBOARD_PAGES:
        nav_items.append(
            dbc.NavLink(
                [html.Span(page["icon"], className="me-2"), page["label"]],
                href=f"#{page['id']}",
                id=f"link-{page['id']}",
                className="sidebar-link",
                active="exact"
            )
        )
    
    return html.Div([
        html.Div([
            html.H5("Navigation", className="text-white mb-3"),
            dbc.Nav(nav_items, vertical=True, pills=True)
        ], className="sidebar-content")
    ], className="sidebar")


def create_metric_card(title, value, change=None, icon="📊"):
    """Create metric card component."""
    change_element = html.Div()
    if change is not None:
        color = COLORS["success"] if change >= 0 else COLORS["danger"]
        arrow = "▲" if change >= 0 else "▼"
        change_element = html.Div([
            html.Span(f"{arrow} {abs(change):.2%}",
                     style={"color": color, "fontSize": "14px"})
        ], className="mt-2")
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "24px", "marginRight": "12px"}),
                html.Span(title, className="text-muted", style={"fontSize": "14px"})
            ], className="mb-2"),
            html.H3(value, className="mb-0", style={"color": COLORS["primary"]}),
            change_element
        ])
    ], className="metric-card shadow-sm")


# ============================================================================
# PAGE LAYOUTS
# ============================================================================

def create_overview_page():
    """Create overview page layout."""
    # Load data
    summary = data_loader.get_dashboard_summary()
    alerts_by_severity = data_loader.get_alerts_by_severity()
    
    return html.Div([
        # Header
        html.H2("📊 Dashboard Overview", className="mb-4"),
        
        # Key Metrics Row
        dbc.Row([
            dbc.Col(create_metric_card(
                "Total Batches",
                f"{summary['total_batches']}",
                icon="📦"
            ), md=3),
            dbc.Col(create_metric_card(
                "Applications Processed",
                f"{summary['total_applications']:,}",
                icon="📝"
            ), md=3),
            dbc.Col(create_metric_card(
                "Current AUC-ROC",
                f"{summary['current_auc']:.4f}" if summary['current_auc'] else "N/A",
                change=summary['auc_change'],
                icon="🎯"
            ), md=3),
            dbc.Col(create_metric_card(
                "Total Alerts",
                f"{summary['total_alerts']}",
                icon="🚨"
            ), md=3),
        ], className="mb-4"),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Trend"),
                    dbc.CardBody([
                        dcc.Graph(id="overview-performance-chart")
                    ])
                ], className="shadow-sm")
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Alert Summary"),
                    dbc.CardBody([
                        dcc.Graph(id="overview-alert-chart")
                    ])
                ], className="shadow-sm")
            ], md=4),
        ], className="mb-4"),
        
        # Bottom Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Alerts"),
                    dbc.CardBody(id="recent-alerts-list")
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Status"),
                    dbc.CardBody([
                        html.Div([
                            html.P([
                                html.Strong("Monitoring Status: "),
                                html.Span("🟢 Active", style={"color": COLORS["success"]})
                            ]),
                            html.P([
                                html.Strong("Batches with Drift: "),
                                html.Span(f"{summary['batches_with_drift']}/{summary['total_batches']}")
                            ]),
                            html.P([
                                html.Strong("Fairness Violations: "),
                                html.Span(f"{summary['batches_with_fairness_violations']}/{summary['total_batches']}")
                            ]),
                            html.P([
                                html.Strong("Critical Alerts: "),
                                html.Span(f"{summary['critical_alerts']}", 
                                        style={"color": COLORS["danger"] if summary['critical_alerts'] > 0 else COLORS["success"]})
                            ])
                        ])
                    ])
                ], className="shadow-sm")
            ], md=6),
        ])
    ])


def create_performance_page():
    """Create performance monitoring page."""
    return html.Div([
        html.H2("🎯 Performance Monitoring", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics Over Time"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-trend-chart")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Confusion Matrix (Latest Batch)"),
                    dbc.CardBody([
                        dcc.Graph(id="confusion-matrix-chart")
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Approval Rate"),
                    dbc.CardBody([
                        dcc.Graph(id="approval-rate-gauge")
                    ])
                ], className="shadow-sm")
            ], md=6),
        ])
    ])


def create_drift_page():
    """Create drift detection page."""
    return html.Div([
        html.H2("🔍 Drift Detection", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Drift Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(id="drift-heatmap")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Drift Percentage by Batch"),
                    dbc.CardBody([
                        dcc.Graph(id="drift-percentage-chart")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ])
    ])


def create_fairness_page():
    """Create fairness monitoring page."""
    return html.Div([
        html.H2("⚖️  Fairness Monitoring", className="mb-4"),
        
        # Attribute selector
        dbc.Row([
            dbc.Col([
                html.Label("Select Sensitive Attribute:"),
                dcc.Dropdown(
                    id="fairness-attribute-dropdown",
                    options=[
                        {"label": "Sex", "value": "Sex"},
                        {"label": "Age Group", "value": "Age_group"}
                    ],
                    value="Sex",
                    clearable=False
                )
            ], md=4),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Div(id="fairness-chart-title")),
                    dbc.CardBody([
                        dcc.Graph(id="fairness-trend-chart")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fairness Violations by Batch"),
                    dbc.CardBody([
                        dcc.Graph(id="fairness-violations-chart")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ])
    ])


def create_alerts_page():
    """Create alerts page."""
    return html.Div([
        html.H2("🚨 Alerts & Notifications", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Alert Timeline"),
                    dbc.CardBody([
                        dcc.Graph(id="alert-timeline-chart")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Alert Details"),
                    dbc.CardBody(id="alert-details-table")
                ], className="shadow-sm")
            ], md=12),
        ])
    ])


def create_batches_page():
    """Create batch details page."""
    batch_logs = data_loader.load_batch_logs()
    batch_options = [{"label": log["batch_id"], "value": log["batch_id"]} 
                     for log in batch_logs]
    
    return html.Div([
        html.H2("📦 Batch Details", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Batch:"),
                dcc.Dropdown(
                    id="batch-selector",
                    options=batch_options,
                    value=batch_options[0]["value"] if batch_options else None,
                    clearable=False
                )
            ], md=4),
        ], className="mb-4"),
        
        html.Div(id="batch-details-content")
    ])


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Interval(id="auto-refresh", interval=DASHBOARD_CONFIG["auto_refresh_interval"]),
    
    create_navbar(),
    
    dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col([
                create_sidebar()
            ], md=2, className="pe-0"),
            
            # Main Content
            dbc.Col([
                html.Div(id="page-content", className="p-4")
            ], md=10)
        ])
    ], fluid=True)
], style={"backgroundColor": COLORS["background"]})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output("page-content", "children"),
    Input("url", "hash")
)
def display_page(hash_value):
    """Route to appropriate page based on URL hash."""
    if not hash_value or hash_value == "#overview":
        return create_overview_page()
    elif hash_value == "#performance":
        return create_performance_page()
    elif hash_value == "#drift":
        return create_drift_page()
    elif hash_value == "#fairness":
        return create_fairness_page()
    elif hash_value == "#alerts":
        return create_alerts_page()
    elif hash_value == "#batches":
        return create_batches_page()
    else:
        return create_overview_page()


@app.callback(
    Output("last-update-time", "children"),
    [Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_timestamp(n_intervals, n_clicks):
    """Update last refresh timestamp."""
    return datetime.now().strftime("%H:%M:%S")


# Overview page callbacks
@app.callback(
    [Output("overview-performance-chart", "figure"),
     Output("overview-alert-chart", "figure"),
     Output("recent-alerts-list", "children")],
    [Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_overview(n_intervals, n_clicks):
    """Update overview page charts."""
    # Performance chart
    perf_df = data_loader.get_performance_trend()
    baseline = data_loader.load_baseline_data()
    perf_fig = create_performance_trend_chart(perf_df, baseline["performance"])
    
    # Alert summary pie chart
    alerts_by_severity = data_loader.get_alerts_by_severity()
    import plotly.graph_objects as go
    alert_fig = go.Figure(data=[go.Pie(
        labels=list(alerts_by_severity.keys()),
        values=list(alerts_by_severity.values()),
        marker_colors=[COLORS["danger"], COLORS["warning"], COLORS["info"]]
    )])
    alert_fig.update_layout(title="Alerts by Severity", height=300)
    
    # Recent alerts list
    recent_alerts = data_loader.get_recent_alerts(5)
    if not recent_alerts:
        alerts_content = html.P("No alerts", className="text-muted")
    else:
        alerts_content = html.Div([
            html.Div([
                html.Span(
                    f"🔴" if a["severity"] == "critical" else "⚠️" if a["severity"] == "warning" else "ℹ️",
                    className="me-2"
                ),
                html.Span(a["message"], className="me-2"),
                html.Small(a.get("batch_id", ""), className="text-muted")
            ], className="mb-2")
            for a in recent_alerts
        ])
    
    return perf_fig, alert_fig, alerts_content


# Performance page callbacks
@app.callback(
    [Output("performance-trend-chart", "figure"),
     Output("confusion-matrix-chart", "figure"),
     Output("approval-rate-gauge", "figure")],
    [Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_performance_page(n_intervals, n_clicks):
    """Update performance page."""
    perf_df = data_loader.get_performance_trend()
    baseline = data_loader.load_baseline_data()
    
    perf_fig = create_performance_trend_chart(perf_df, baseline["performance"])
    
    # Latest batch confusion matrix
    batch_logs = data_loader.load_batch_logs()
    if batch_logs:
        latest_cm = batch_logs[-1]["confusion_matrix"]
        cm_fig = create_confusion_matrix_chart(latest_cm)
        approval_rate = batch_logs[-1]["predictions"]["approval_rate"]
    else:
        cm_fig = go.Figure()
        approval_rate = 0
    
    gauge_fig = create_approval_rate_gauge(approval_rate)
    
    return perf_fig, cm_fig, gauge_fig


# Drift page callbacks
@app.callback(
    [Output("drift-heatmap", "figure"),
     Output("drift-percentage-chart", "figure")],
    [Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_drift_page(n_intervals, n_clicks):
    """Update drift detection page."""
    batch_ids, features, drift_matrix = data_loader.get_feature_drift_heatmap_data()
    heatmap_fig = create_drift_heatmap(batch_ids, features, drift_matrix)
    
    drift_df = data_loader.get_drift_summary()
    drift_pct_fig = create_drift_percentage_chart(drift_df)
    
    return heatmap_fig, drift_pct_fig


# Fairness page callbacks
@app.callback(
    [Output("fairness-trend-chart", "figure"),
     Output("fairness-violations-chart", "figure"),
     Output("fairness-chart-title", "children")],
    [Input("fairness-attribute-dropdown", "value"),
     Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_fairness_page(attribute, n_intervals, n_clicks):
    """Update fairness monitoring page."""
    fairness_df = data_loader.get_fairness_trend(attribute)
    baseline = data_loader.load_baseline_data()
    
    fairness_fig = create_fairness_trend_chart(fairness_df)
    
    fairness_reports = data_loader.load_fairness_reports()
    violations_fig = create_fairness_violation_chart(fairness_reports)
    
    title = f"Fairness Metrics - {attribute}"
    
    return fairness_fig, violations_fig, title


# Alerts page callbacks
@app.callback(
    [Output("alert-timeline-chart", "figure"),
     Output("alert-details-table", "children")],
    [Input("auto-refresh", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_alerts_page(n_intervals, n_clicks):
    """Update alerts page."""
    alerts = data_loader.get_recent_alerts(20)
    timeline_fig = create_alert_timeline(alerts)
    
    # Create alert table
    if not alerts:
        table_content = html.P("No alerts to display", className="text-muted")
    else:
        table_content = dbc.Table.from_dataframe(
            pd.DataFrame([{
                "Severity": a["severity"],
                "Type": a["type"],
                "Message": a["message"],
                "Batch": a.get("batch_id", "N/A"),
                "Time": a["timestamp"][:19]
            } for a in alerts]),
            striped=True,
            bordered=True,
            hover=True,
            size="sm"
        )
    
    return timeline_fig, table_content


# Batch details callback
@app.callback(
    Output("batch-details-content", "children"),
    [Input("batch-selector", "value")]
)
def update_batch_details(batch_id):
    """Update batch details page."""
    if not batch_id:
        return html.P("Select a batch", className="text-muted")
    
    details = data_loader.get_batch_details(batch_id)
    
    if not details["batch_log"]:
        return html.P("Batch data not found", className="text-muted")
    
    log = details["batch_log"]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Batch Summary"),
                dbc.CardBody([
                    html.P([html.Strong("Batch ID: "), log["batch_id"]]),
                    html.P([html.Strong("Samples: "), str(log["batch_size"])]),
                    html.P([html.Strong("Approved: "), str(log["predictions"]["approved"])]),
                    html.P([html.Strong("Rejected: "), str(log["predictions"]["rejected"])]),
                    html.P([html.Strong("Approval Rate: "), f"{log['predictions']['approval_rate']:.2%}"])
                ])
            ], className="shadow-sm mb-3"),
            dbc.Card([
                dbc.CardHeader("Performance"),
                dbc.CardBody([
                    html.P([html.Strong("AUC-ROC: "), f"{log['performance']['auc_roc']:.4f}"]),
                    html.P([html.Strong("Accuracy: "), f"{log['performance']['accuracy']:.4f}"]),
                    html.P([html.Strong("Bad Recall: "), f"{log['performance']['bad_credit']['recall']:.4f}"]),
                    html.P([html.Strong("Good Recall: "), f"{log['performance']['good_credit']['recall']:.4f}"])
                ])
            ], className="shadow-sm")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Alerts"),
                dbc.CardBody([
                    html.Div([
                        html.P(f"🚨 {a['message']}", className="mb-2") 
                        for a in details["alerts"]
                    ]) if details["alerts"] else html.P("No alerts", className="text-muted")
                ])
            ], className="shadow-sm")
        ], md=6)
    ])


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    print(f"Starting {DASHBOARD_CONFIG['title']}...")
    print(f"Access dashboard at: http://localhost:{DASHBOARD_CONFIG['port']}")
    
    app.run_server(
        debug=DASHBOARD_CONFIG["debug"],
        host=DASHBOARD_CONFIG["host"],
        port=DASHBOARD_CONFIG["port"]
    )