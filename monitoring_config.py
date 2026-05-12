"""
Part 2 Configuration: Drift & Fairness Monitoring
=================================================
Defines thresholds and settings for:
- Drift detection (statistical tests)
- Fairness monitoring (violation thresholds)
- Alert generation (when to trigger alerts)
- Batch processing (simulation parameters)
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"
BATCH_LOGS_DIR = MONITORING_DIR / "batch_logs"
DRIFT_REPORTS_DIR = MONITORING_DIR / "drift_reports"
FAIRNESS_REPORTS_DIR = MONITORING_DIR / "fairness_reports"
ALERTS_DIR = MONITORING_DIR / "alerts"

# Create directories
for dir_path in [MONITORING_DIR, BATCH_LOGS_DIR, DRIFT_REPORTS_DIR, 
                 FAIRNESS_REPORTS_DIR, ALERTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
BASELINE_METADATA_PATH = BASELINE_DIR / "metadata.json"
BASELINE_FAIRNESS_PATH = BASELINE_DIR / "fairness_audit_report.json"

# ============================================================================
# DRIFT DETECTION THRESHOLDS
# ============================================================================

# Statistical test p-value thresholds (null hypothesis: no drift)
# If p-value < threshold, reject null hypothesis → drift detected
DRIFT_THRESHOLDS = {
    "ks_test": 0.05,           # Kolmogorov-Smirnov test (continuous features)
    "chi_square": 0.05,        # Chi-square test (categorical features)
    "kl_divergence": 0.1,      # KL divergence threshold (prediction drift)
    "prediction_shift": 0.05,  # Max acceptable shift in prediction distribution
}

# Feature-specific drift sensitivity
# Some features naturally vary more, so we can be more lenient
FEATURE_DRIFT_CONFIG = {
    "Credit amount": {"method": "ks_test", "threshold": 0.05},
    "Duration": {"method": "ks_test", "threshold": 0.05},
    "Debt_ratio": {"method": "ks_test", "threshold": 0.05},
    "Job": {"method": "chi_square", "threshold": 0.05},
    "Housing": {"method": "chi_square", "threshold": 0.05},
    "Saving accounts": {"method": "chi_square", "threshold": 0.05},
    "Checking account": {"method": "chi_square", "threshold": 0.05},
    "Purpose": {"method": "chi_square", "threshold": 0.05},
}

# ============================================================================
# FAIRNESS MONITORING THRESHOLDS
# ============================================================================

# Fairness violation thresholds (from Part 1 requirements)
FAIRNESS_THRESHOLDS = {
    "demographic_parity_max": 0.1,      # Max acceptable difference in approval rates
    "disparate_impact_min": 0.8,        # Min acceptable ratio (worst/best group)
    "equal_opportunity_max": 0.1,       # Max acceptable difference in TPR
}

# Sensitive attributes to monitor
SENSITIVE_ATTRIBUTES = ["Sex", "Age_group"]

# Alert severity levels
ALERT_SEVERITY = {
    "critical": {
        "demographic_parity": 0.15,     # > 15% difference is critical
        "disparate_impact": 0.7,        # < 70% ratio is critical
        "prediction_drift": 0.15,       # > 15% shift is critical
    },
    "warning": {
        "demographic_parity": 0.1,      # > 10% difference is warning
        "disparate_impact": 0.8,        # < 80% ratio is warning
        "prediction_drift": 0.1,        # > 10% shift is warning
    },
}

# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Batch simulation settings
BATCH_CONFIG = {
    "batch_size": 150,              # Number of samples per batch (same as test set)
    "num_batches": 6,               # Number of batches to process
    "monitoring_frequency": 1,      # Run monitoring every N batches (1 = every batch)
    "drift_detection_window": 3,    # Compare against last N batches for trend analysis
}

# ============================================================================
# PERFORMANCE DEGRADATION THRESHOLDS
# ============================================================================

# Acceptable performance drop from baseline
PERFORMANCE_THRESHOLDS = {
    "auc_roc_min_drop": 0.05,       # Alert if AUC drops > 5% from baseline
    "accuracy_min_drop": 0.05,      # Alert if accuracy drops > 5%
    "recall_bad_min_drop": 0.1,     # Alert if bad recall drops > 10%
}

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

ALERT_CONFIG = {
    "enable_alerts": True,
    "alert_types": [
        "drift_detected",
        "fairness_violation", 
        "performance_degradation",
    ],
    "notification_channels": ["log_file", "json_report"],  # Can add email, slack, etc.
    "aggregate_alerts": True,       # Combine multiple alerts into summary
}

# ============================================================================
# MONITORING REPORT CONFIGURATION
# ============================================================================

REPORT_CONFIG = {
    "generate_per_batch": True,     # Generate report for each batch
    "generate_summary": True,       # Generate overall summary at end
    "include_visualizations": False, # Set to True for Part 3 (dashboard)
    "save_detailed_logs": True,     # Save detailed batch-level logs
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_baseline_metrics():
    """Load baseline performance and fairness metrics."""
    import json
    
    with open(BASELINE_METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    with open(BASELINE_FAIRNESS_PATH, 'r') as f:
        fairness = json.load(f)
    
    return {
        "performance": metadata["performance_metrics"]["test"],
        "fairness": fairness["by_sex"],
        "fairness_age": fairness["by_age_group"],
        "threshold": metadata["decision_threshold"],
    }


def check_fairness_violation(metric_value, metric_name):
    """
    Check if a fairness metric violates thresholds.
    
    Returns: tuple (is_violation, severity)
        severity: None, "warning", or "critical"
    """
    if metric_name == "demographic_parity":
        if abs(metric_value) > ALERT_SEVERITY["critical"]["demographic_parity"]:
            return True, "critical"
        elif abs(metric_value) > ALERT_SEVERITY["warning"]["demographic_parity"]:
            return True, "warning"
    
    elif metric_name == "disparate_impact":
        if metric_value < ALERT_SEVERITY["critical"]["disparate_impact"]:
            return True, "critical"
        elif metric_value < ALERT_SEVERITY["warning"]["disparate_impact"]:
            return True, "warning"
    
    return False, None


def check_performance_degradation(current_metrics, baseline_metrics):
    """
    Check if performance has degraded significantly from baseline.
    
    Returns: list of alerts
    """
    alerts = []
    
    # AUC-ROC check
    auc_drop = baseline_metrics["auc_roc"] - current_metrics["auc_roc"]
    if auc_drop > PERFORMANCE_THRESHOLDS["auc_roc_min_drop"]:
        alerts.append({
            "type": "performance_degradation",
            "metric": "auc_roc",
            "baseline": baseline_metrics["auc_roc"],
            "current": current_metrics["auc_roc"],
            "drop": auc_drop,
            "severity": "critical" if auc_drop > 0.1 else "warning"
        })
    
    # Accuracy check
    acc_drop = baseline_metrics["accuracy"] - current_metrics["accuracy"]
    if acc_drop > PERFORMANCE_THRESHOLDS["accuracy_min_drop"]:
        alerts.append({
            "type": "performance_degradation",
            "metric": "accuracy",
            "baseline": baseline_metrics["accuracy"],
            "current": current_metrics["accuracy"],
            "drop": acc_drop,
            "severity": "critical" if acc_drop > 0.1 else "warning"
        })
    
    # Bad recall check
    recall_drop = baseline_metrics["bad_recall"] - current_metrics["bad_recall"]
    if recall_drop > PERFORMANCE_THRESHOLDS["recall_bad_min_drop"]:
        alerts.append({
            "type": "performance_degradation",
            "metric": "bad_recall",
            "baseline": baseline_metrics["bad_recall"],
            "current": current_metrics["bad_recall"],
            "drop": recall_drop,
            "severity": "critical" if recall_drop > 0.15 else "warning"
        })
    
    return alerts


# Print configuration on import (for debugging)
if __name__ == "__main__":
    print("Part 2 Configuration Loaded")
    print("=" * 70)
    print(f"Monitoring Directory: {MONITORING_DIR}")
    print(f"Batch Size: {BATCH_CONFIG['batch_size']}")
    print(f"Number of Batches: {BATCH_CONFIG['num_batches']}")
    print(f"Drift Thresholds: {DRIFT_THRESHOLDS}")
    print(f"Fairness Thresholds: {FAIRNESS_THRESHOLDS}")
    print("=" * 70)