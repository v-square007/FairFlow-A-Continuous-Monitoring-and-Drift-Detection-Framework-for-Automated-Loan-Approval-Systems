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
# PROFESSIONAL COLOR SCHEME
# ============================================================================
COLORS = {
    # Primary colors
    "primary": "#2E86AB",      # Professional blue
    "secondary": "#A23B72",    # Purple accent
    "success": "#06A77D",      # Green for good metrics
    "warning": "#F18F01",      # Orange for warnings
    "danger": "#C73E1D",       # Red for critical alerts
    "info": "#4A90E2",         # Light blue for info
    
    # Background colors
    "background": "#F8F9FA",   # Light gray background
    "card_bg": "#FFFFFF",      # White card background
    "sidebar_bg": "#2C3E50",   # Dark sidebar
    
    # Text colors
    "text_primary": "#2C3E50", # Dark gray text
    "text_secondary": "#7F8C8D", # Light gray text
    "text_light": "#FFFFFF",   # White text
    
    # Chart colors
    "chart_colors": [
        "#2E86AB", "#A23B72", "#F18F01", 
        "#06A77D", "#C73E1D", "#4A90E2"
    ],
    
    # Metric status colors
    "metric_good": "#06A77D",
    "metric_warning": "#F18F01",
    "metric_bad": "#C73E1D",
    "metric_neutral": "#7F8C8D",
}

# ============================================================================
# TYPOGRAPHY
# ============================================================================
FONTS = {
    "heading": "'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    "body": "'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    "monospace": "'Courier New', 'Courier', monospace",
}

# ============================================================================
# LAYOUT SETTINGS
# ============================================================================
LAYOUT = {
    "sidebar_width": "280px",
    "card_padding": "20px",
    "card_border_radius": "8px",
    "card_shadow": "0 2px 4px rgba(0,0,0,0.1)",
    "spacing": {
        "xs": "8px",
        "sm": "16px",
        "md": "24px",
        "lg": "32px",
        "xl": "48px",
    }
}

# ============================================================================
# CHART SETTINGS
# ============================================================================
CHART_CONFIG = {
    "template": "plotly_white",
    "height": 400,
    "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    "font": {
        "family": FONTS["body"],
        "size": 12,
        "color": COLORS["text_primary"]
    },
    "title_font": {
        "family": FONTS["heading"],
        "size": 16,
        "color": COLORS["text_primary"],
        "weight": 600
    },
    "showlegend": True,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.2,
        "xanchor": "center",
        "x": 0.5
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
# CUSTOM CSS STYLES
# ============================================================================
CUSTOM_CSS = """
/* Global Styles */
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    background-color: #F8F9FA;
    margin: 0;
    padding: 0;
}

/* Card Styles */
.dashboard-card {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Metric Card */
.metric-card {
    text-align: center;
    padding: 24px;
    border-left: 4px solid #2E86AB;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
}

.metric-label {
    font-size: 14px;
    color: #7F8C8D;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-change {
    font-size: 12px;
    margin-top: 8px;
}

/* Alert Badge */
.alert-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.alert-critical {
    background: #FFE5E5;
    color: #C73E1D;
}

.alert-warning {
    background: #FFF4E5;
    color: #F18F01;
}

.alert-info {
    background: #E5F3FF;
    color: #4A90E2;
}

/* Table Styles */
.data-table {
    width: 100%;
    border-collapse: collapse;
}

.data-table th {
    background: #F8F9FA;
    padding: 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #E1E8ED;
}

.data-table td {
    padding: 12px;
    border-bottom: 1px solid #F0F0F0;
}

/* Status Indicator */
.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.status-good { background: #06A77D; }
.status-warning { background: #F18F01; }
.status-bad { background: #C73E1D; }
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