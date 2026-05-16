import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. APP SETUP & INITIALIZATION
# ---------------------------------------------------------------------------
# Using the SLATE Bootstrap theme as a baseline for dark mode typography
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.SLATE],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Executive Performance Analytics"

# ---------------------------------------------------------------------------
# 2. DESIGN SYSTEM & PALETTE (Inspired by Dashboard 3)
# ---------------------------------------------------------------------------
THEME_BG = "#0f111a"        # Deep midnight background
CARD_BG = "#161b26"         # Soft dark card container surface
TEXT_MAIN = "#f8f9fa"       # Off-white crisp text
TEXT_MUTED = "#8b9bb4"      # Desaturated secondary labels

# Vibrant Infographic Accent Gradients
ACCENT_PURPLE = "#b042ff"   # Primary highlight
ACCENT_CYAN = "#00f0ff"     # Secondary highlight
ACCENT_GREEN = "#00e676"    # Success metrics
ACCENT_PINK = "#ff2a85"     # Attention/Alert metrics

# ---------------------------------------------------------------------------
# 3. MOCK DATA GENERATION (Representing unified telemetry)
# ---------------------------------------------------------------------------
# Time-series data
dates = pd.date_range(start="2026-01-01", end="2026-05-15", freq="D")
np.random.seed(42)
ts_data = pd.DataFrame({
    "Date": dates,
    "Volume_Created": np.random.poisson(lam=45, size=len(dates)) + np.sin(np.arange(len(dates))/10)*15,
    "Volume_Resolved": np.random.poisson(lam=42, size=len(dates)) + np.sin(np.arange(len(dates))/10)*12,
})

# Categorical breakdowns
categories = ['Sales Operations', 'Infrastructure Setup', 'Core Bug Fixes', 'Feature Requests']
cat_counts = [44, 25, 19, 12]

weekly_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekly_perf = [55, 32, 70, 65, 95, 60, 15]

# ---------------------------------------------------------------------------
# 4. INFOGRAPHIC PLOTLY GENERATORS
# ---------------------------------------------------------------------------
def create_main_time_series():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_data["Date"], y=ts_data["Volume_Created"],
        name="Volume Created", mode="lines",
        line=dict(width=3, color=ACCENT_CYAN),
        fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.05)'
    ))
    fig.add_trace(go.Scatter(
        x=ts_data["Date"], y=ts_data["Volume_Resolved"],
        name="Volume Resolved", mode="lines",
        line=dict(width=3, color=ACCENT_PURPLE, dash='dot'),
        fill='tozeroy', fillcolor='rgba(176, 66, 255, 0.03)'
    ))
    
    # Add an infographic annotation peak identifier (inspired by Image 3)
    max_idx = ts_data["Volume_Created"].idxmax()
    fig.add_annotation(
        x=ts_data["Date"].iloc[max_idx], y=ts_data["Volume_Created"].iloc[max_idx],
        text=f"Peak: {int(ts_data['Volume_Created'].max())}",
        showarrow=True, arrowhead=2, arrowcolor=ACCENT_CYAN,
        font=dict(color=TEXT_MAIN, size=11),
        bgcolor=ACCENT_PINK, bordercolor=ACCENT_PINK, borderwidth=1, borderpad=4
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False)
    )
    return fig

def create_donut_chart():
    fig = go.Figure(data=[go.Pie(
        labels=categories, values=cat_counts, hole=.7,
        marker=dict(colors=[ACCENT_CYAN, ACCENT_PURPLE, '#1f77b4', ACCENT_GREEN]),
        textinfo='percent', textfont=dict(color=TEXT_MAIN, size=12)
    )])
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1, font=dict(color=TEXT_MUTED)),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_weekly_bar():
    # Neon gradient simulation through discrete sequence colors
    fig = go.Figure(data=[go.Bar(
        x=weekly_days, y=weekly_perf,
        marker=dict(
            color=weekly_perf,
            colorscale=[[0.0, ACCENT_PURPLE], [1.0, ACCENT_CYAN]],
            line=dict(width=0)
        ),
        width=0.4
    )])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MUTED),
        xaxis=dict(showgrid=False, linecolor="#222938"),
        yaxis=dict(showgrid=True, gridcolor="#222938", zeroline=False)
    )
    return fig

# ---------------------------------------------------------------------------
# 5. REUSABLE UI COMPONENT FACTORIES (Clean UI Architecture)
# ---------------------------------------------------------------------------
def create_metric_card(title, value, subtitle, delta_positive=True, delta_str=""):
    delta_color = ACCENT_GREEN if delta_positive else ACCENT_PINK
    return html.Div(
        style={
            "backgroundColor": CARD_BG, "borderRadius": "16px", "padding": "24px",
            "border": "1px solid #222938", "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.2)"
        },
        children=[
            html.Div(title, style={"color": TEXT_MUTED, "fontSize": "14px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.Div(value, style={"color": TEXT_MAIN, "fontSize": "36px", "fontWeight": "700", "margin": "8px 0"}),
            html.Div([
                html.Span(f"{'▲' if delta_positive else '▼'} {delta_str} ", style={"color": delta_color, "fontWeight": "700", "marginRight": "6px"}),
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
            html.H5(title, style={"color": TEXT_MAIN, "fontWeight": "600", "marginBottom": "20px", "fontSize": "16px", "letterSpacing": "0.5px"}),
            dcc.Graph(figure=graph_object, config={'displayModeBar': False}, style={"height": "280px"})
        ]
    )

# ---------------------------------------------------------------------------
# 6. APP LAYOUT STRUCTURE
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": THEME_BG, "minHeight": "100vh", "padding": "0", "fontFamily": "'Inter', 'Segoe UI', sans-serif"},
    children=[
        # Top Global Navigation Header
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Div(style={"width": "12px", "height": "12px", "backgroundColor": ACCENT_CYAN, "borderRadius": "50%"})),
                                dbc.Col(dbc.NavbarBrand("NEXUS // ANALYTICS PLATFORM", className="ms-2", style={"color": TEXT_MAIN, "fontWeight": "800", "fontSize": "16px", "letterSpacing": "2px"})),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    html.Div(
                        children=[
                            html.Span("SYSTEM TELEMETRY STATUS: LIVE", style={"color": ACCENT_GREEN, "fontSize": "12px", "fontWeight": "700", "letterSpacing": "1px", "backgroundColor": "rgba(0, 230, 118, 0.1)", "padding": "6px 14px", "borderRadius": "20px"})
                        ]
                    )
                ],
                fluid=True,
            ),
            color="#0b0d14",
            dark=True,
            style={"borderBottom": "1px solid #1c2130", "padding": "15px 30px"}
        ),
        
        # Main Grid View Workspace
        dbc.Container(
            fluid=True,
            style={"padding": "32px 40px"},
            children=[
                
                # SECTION 1: Strategic High-Level KPI Summary Tiles
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(create_metric_card("Avg First Response", "30h 15m", "vs previous cycle", delta_positive=False, delta_str="20%"), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card("Avg Full Resolution", "22h 40m", "optimized queue pipelines", delta_positive=True, delta_str="14%"), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card("Net Conversion Rate", "114%", "target threshold milestone", delta_positive=True, delta_str="8.4%"), xs=12, sm=6, lg=3),
                        dbc.Col(create_metric_card("Total Financial Delta", "$4,250.15", "gross organic growth metrics", delta_positive=True, delta_str="1.25%"), xs=12, sm=6, lg=3),
                    ]
                ),
                
                # SECTION 2: Interactive Operational Chart Analysis Blocks
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(create_graph_card("Volume Distribution Anomalies (Created vs. Resolved)", create_main_time_series()), lg=8, md=12),
                        dbc.Col(create_graph_card("System Performance Allocation / Weekday", create_weekly_bar()), lg=4, md=12),
                    ]
                ),
                
                # SECTION 3: Deep-Dive Micro Metrics & Proportional Ratios
                dbc.Row(
                    className="g-4",
                    children=[
                        dbc.Col(create_graph_card("Operational Domain Classifications", create_donut_chart()), lg=4, md=12),
                        
                        # High-Density Information Grid Map Component (Inspired by Matrix View in Screenshot 1)
                        dbc.Col(
                            html.Div(
                                style={
                                    "backgroundColor": CARD_BG, "borderRadius": "16px", "padding": "24px",
                                    "border": "1px solid #222938", "height": "100%", "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.2)"
                                },
                                children=[
                                    html.H5("Performance Heat Matrix Indices", style={"color": TEXT_MAIN, "fontWeight": "600", "marginBottom": "20px", "fontSize": "16px"}),
                                    html.Div(
                                        className="table-responsive",
                                        children=dbc.Table(
                                            [
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("SEGMENT RUNTIME", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px", "letterSpacing": "1px"}),
                                                        html.Th("2024", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("2025", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("2026 (YTD)", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px"}),
                                                        html.Th("INDEX DELTA", style={"color": TEXT_MUTED, "borderBottom": "2px solid #222938", "fontSize": "11px", "textAlign": "right"})
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td("Enterprise Farms Data Engine", style={"color": TEXT_MAIN, "fontWeight": "500", "fontSize": "13px"}),
                                                        html.Td("687%", style={"color": ACCENT_GREEN}), html.Td("205%", style={"color": ACCENT_GREEN}), html.Td("101%", style={"color": ACCENT_CYAN}),
                                                        html.Td("▲ Critical Max", style={"color": ACCENT_GREEN, "fontWeight": "600", "fontSize": "12px", "textAlign": "right"})
                                                    ], style={"borderBottom": "1px solid #1c2130"}),
                                                    html.Tr([
                                                        html.Td("Logistical Transit Fleets", style={"color": TEXT_MAIN, "fontWeight": "500", "fontSize": "13px"}),
                                                        html.Td("129%", style={"color": ACCENT_GREEN}), html.Td("1263%", style={"color": ACCENT_PINK}), html.Td("219%", style={"color": ACCENT_GREEN}),
                                                        html.Td("▼ Unstable Flow", style={"color": ACCENT_PINK, "fontWeight": "600", "fontSize": "12px", "textAlign": "right"})
                                                    ], style={"borderBottom": "1px solid #1c2130"}),
                                                    html.Tr([
                                                        html.Td("Emergency Core Response", style={"color": TEXT_MAIN, "fontWeight": "500", "fontSize": "13px"}),
                                                        html.Td("50%", style={"color": TEXT_MUTED}), html.Td("63%", style={"color": TEXT_MUTED}), html.Td("56%", style={"color": TEXT_MUTED}),
                                                        html.Td("■ Nominal Variance", style={"color": ACCENT_CYAN, "fontWeight": "600", "fontSize": "12px", "textAlign": "right"})
                                                    ], style={"borderBottom": "none"}),
                                                ])
                                            ],
                                            borderless=True,
                                            hover=True,
                                            style={"verticalAlign": "middle"}
                                        )
                                    )
                                ]
                            ),
                            lg=8, md=12
                        )
                    ]
                )
            ]
        )
    ]
)

# ---------------------------------------------------------------------------
# 7. EXECUTION ENGINE RULE EXPLICITNESS
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)