"""
Dashboard Configuration
=======================
Professional styling and configuration for the monitoring dashboard.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
ARTIFACTS_DIR = Path("artifacts")
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_CONFIG = {
    "title": "FairFlow - Credit Risk Monitoring Dashboard",
    "port": 8050,
    "host": "0.0.0.0",
    "debug": True,
    "auto_refresh_interval": 30000,  # 30 seconds in milliseconds
}

# ============================================================================
# PROFESSIONAL COLOR SCHEME - Clean & Minimal
# ============================================================================
COLORS = {
    # Primary colors - Professional and subtle
    "primary": "#5B6EFF",      # Modern blue
    "secondary": "#8B5CF6",    # Subtle purple
    "success": "#10B981",      # Clean green
    "warning": "#F59E0B",      # Warm orange
    "danger": "#EF4444",       # Clear red
    "info": "#3B82F6",         # Info blue
    
    # Background colors - Clean white/gray scheme
    "background": "#F9FAFB",   # Very light gray
    "card_bg": "#FFFFFF",      # Pure white
    "sidebar_bg": "#FFFFFF",   # White sidebar
    "header_bg": "#FFFFFF",    # White header
    
    # Text colors - Better contrast
    "text_primary": "#111827",   # Near black
    "text_secondary": "#6B7280", # Medium gray
    "text_light": "#9CA3AF",     # Light gray
    "text_white": "#FFFFFF",     # Pure white
    
    # Border colors
    "border_light": "#E5E7EB",   # Very light border
    "border_medium": "#D1D5DB",  # Medium border
    
    # Chart colors - Professional palette
    "chart_colors": [
        "#5B6EFF",  # Blue
        "#8B5CF6",  # Purple  
        "#10B981",  # Green
        "#F59E0B",  # Orange
        "#EF4444",  # Red
        "#3B82F6"   # Light blue
    ],
    
    # Metric status colors
    "metric_good": "#10B981",
    "metric_warning": "#F59E0B",
    "metric_bad": "#EF4444",
    "metric_neutral": "#6B7280",
    
    # Gradient colors for modern look
    "gradient_start": "#5B6EFF",
    "gradient_end": "#8B5CF6",
}

# ============================================================================
# TYPOGRAPHY - Professional & Clean
# ============================================================================
FONTS = {
    "heading": "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
    "body": "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
    "monospace": "'SF Mono', Monaco, 'Cascadia Code', 'Courier New', monospace",
}

FONT_SIZES = {
    "xs": "12px",
    "sm": "14px",
    "base": "16px",
    "lg": "18px",
    "xl": "20px",
    "2xl": "24px",
    "3xl": "30px",
    "4xl": "36px",
}

# ============================================================================
# LAYOUT SETTINGS - Modern & Spacious
# ============================================================================
LAYOUT = {
    "sidebar_width": "240px",
    "card_padding": "24px",
    "card_border_radius": "12px",
    "card_shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
    "card_shadow_hover": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
    "spacing": {
        "xs": "8px",
        "sm": "12px",
        "md": "16px",
        "lg": "24px",
        "xl": "32px",
        "2xl": "48px",
    },
    "border_width": "1px",
}

# ============================================================================
# CHART SETTINGS - Clean & Modern
# ============================================================================
CHART_CONFIG = {
    "template": "plotly_white",
    "height": 380,
    "margin": {"l": 50, "r": 20, "t": 40, "b": 60},
    "font": {
        "family": FONTS["body"],
        "size": 13,
        "color": COLORS["text_primary"]
    },
    "title_font": {
        "family": FONTS["heading"],
        "size": 18,
        "color": COLORS["text_primary"]
    },
    "showlegend": True,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.15,
        "xanchor": "left",
        "x": 0,
        "bgcolor": "rgba(255,255,255,0)",
        "bordercolor": "rgba(255,255,255,0)"
    },
    "plot_bgcolor": "#FFFFFF",
    "paper_bgcolor": "#FFFFFF",
    "xaxis": {
        "showgrid": False,
        "showline": True,
        "linecolor": "#E5E7EB",
        "linewidth": 1
    },
    "yaxis": {
        "showgrid": True,
        "gridcolor": "#F3F4F6",
        "showline": False
    }
}

# ============================================================================
# DASHBOARD TABS/PAGES
# ============================================================================
DASHBOARD_PAGES = [
    {
        "id": "overview",
        "label": "📊 Overview",
        "icon": "📊"
    },
    {
        "id": "performance",
        "label": "🎯 Performance",
        "icon": "🎯"
    },
    {
        "id": "drift",
        "label": "🔍 Drift Detection",
        "icon": "🔍"
    },
    {
        "id": "fairness",
        "label": "⚖️  Fairness",
        "icon": "⚖️"
    },
    {
        "id": "alerts",
        "label": "🚨 Alerts",
        "icon": "🚨"
    },
    {
        "id": "batches",
        "label": "📦 Batch Details",
        "icon": "📦"
    }
]

# ============================================================================
# METRIC THRESHOLDS (for color coding)
# ============================================================================
METRIC_THRESHOLDS = {
    "auc_roc": {
        "excellent": 0.85,
        "good": 0.75,
        "warning": 0.65,
    },
    "accuracy": {
        "excellent": 0.85,
        "good": 0.75,
        "warning": 0.65,
    },
    "demographic_parity": {
        "excellent": 0.05,
        "good": 0.1,
        "warning": 0.15,
    },
    "disparate_impact": {
        "excellent": 0.9,
        "good": 0.8,
        "warning": 0.7,
    }
}

# ============================================================================
# CUSTOM CSS STYLES - Professional & Modern
# ============================================================================
CUSTOM_CSS = """
/* Global Styles - Clean & Modern */
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}

body {
    background-color: #F9FAFB;
    margin: 0;
    padding: 0;
    color: #111827;
}

/* Remove default Dash styling */
._dash-loading {
    display: none;
}

/* Navbar - Clean & Elevated */
.navbar {
    background: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    padding: 16px 0;
}

/* Sidebar - Minimal & Clean */
.sidebar {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
    height: 100vh;
    position: fixed;
    width: 240px;
    padding: 24px 0;
}

.sidebar-link {
    color: #6B7280;
    font-size: 14px;
    font-weight: 500;
    padding: 12px 24px;
    margin: 4px 12px;
    border-radius: 8px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
}

.sidebar-link:hover {
    background: #F3F4F6;
    color: #111827;
    text-decoration: none;
}

.sidebar-link.active {
    background: #EEF2FF;
    color: #5B6EFF;
    font-weight: 600;
}

/* Card Styles - Modern & Elevated */
.card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
    overflow: hidden;
}

.card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.card-header {
    background: #FFFFFF;
    border-bottom: 1px solid #F3F4F6;
    padding: 16px 24px;
    font-weight: 600;
    font-size: 16px;
    color: #111827;
}

.card-body {
    padding: 24px;
}

/* Metric Card - Professional & Clean */
.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    padding: 24px;
    transition: all 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    margin: 12px 0;
    color: #111827;
}

.metric-label {
    font-size: 14px;
    color: #6B7280;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
}

.metric-change {
    font-size: 13px;
    font-weight: 500;
    margin-top: 8px;
}

.metric-icon {
    font-size: 20px;
    margin-bottom: 8px;
}

/* Alert Badge - Clean & Clear */
.alert-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

.alert-critical {
    background: #FEE2E2;
    color: #991B1B;
}

.alert-warning {
    background: #FEF3C7;
    color: #92400E;
}

.alert-info {
    background: #DBEAFE;
    color: #1E40AF;
}

/* Table Styles - Clean & Readable */
.dash-table-container {
    font-size: 14px;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
    border-collapse: collapse;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
    background: #F9FAFB;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
    color: #374151;
    border-bottom: 1px solid #E5E7EB;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
    padding: 12px 16px;
    border-bottom: 1px solid #F3F4F6;
    color: #111827;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover {
    background: #F9FAFB;
}

/* Dropdown - Modern Style */
.Select-control {
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    padding: 4px;
}

.Select-control:hover {
    border-color: #5B6EFF;
}

/* Status Indicator - Subtle & Clear */
.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.status-good { background: #10B981; }
.status-warning { background: #F59E0B; }
.status-bad { background: #EF4444; }
.status-neutral { background: #6B7280; }

/* Page Title */
h2 {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 24px;
}

h3 {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
}

/* Button Styles */
.btn {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Loading State */
._dash-loading-callback {
    opacity: 0.5;
}

/* Chart Container */
.js-plotly-plot {
    border-radius: 8px;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #F3F4F6;
}

::-webkit-scrollbar-thumb {
    background: #D1D5DB;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
}
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_metric_color(metric_name, value):
    """Get color for metric based on its value."""
    if metric_name not in METRIC_THRESHOLDS:
        return COLORS["metric_neutral"]
    
    thresholds = METRIC_THRESHOLDS[metric_name]
    
    # For metrics where higher is better (AUC, Accuracy, Disparate Impact)
    if metric_name in ["auc_roc", "accuracy", "disparate_impact"]:
        if value >= thresholds["excellent"]:
            return COLORS["metric_good"]
        elif value >= thresholds["good"]:
            return COLORS["metric_warning"]
        else:
            return COLORS["metric_bad"]
    
    # For metrics where lower is better (Demographic Parity)
    else:
        if value <= thresholds["excellent"]:
            return COLORS["metric_good"]
        elif value <= thresholds["good"]:
            return COLORS["metric_warning"]
        else:
            return COLORS["metric_bad"]


def get_alert_badge_class(severity):
    """Get CSS class for alert severity."""
    severity_map = {
        "critical": "alert-critical",
        "warning": "alert-warning",
        "info": "alert-info"
    }
    return severity_map.get(severity, "alert-info")


if __name__ == "__main__":
    print("Dashboard Configuration Loaded")
    print(f"Title: {DASHBOARD_CONFIG['title']}")
    print(f"Port: {DASHBOARD_CONFIG['port']}")
    print(f"Pages: {len(DASHBOARD_PAGES)}")