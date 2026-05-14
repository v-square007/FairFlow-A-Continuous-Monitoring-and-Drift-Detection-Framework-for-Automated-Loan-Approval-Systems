"""
Dashboard Data Loader
=====================
Loads and processes monitoring data from Part 2 artifacts.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class MonitoringDataLoader:
    """
    Loads monitoring data from Part 2 artifacts for dashboard visualization.
    """
    
    def __init__(self, monitoring_dir: str = "artifacts/monitoring",
                 baseline_dir: str = "artifacts/baseline"):
        """
        Initialize data loader.
        
        Args:
            monitoring_dir: Path to monitoring artifacts directory
            baseline_dir: Path to baseline artifacts directory
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.baseline_dir = Path(baseline_dir)
        
        # Data caches
        self._baseline_data = None
        self._batch_logs = None
        self._drift_reports = None
        self._fairness_reports = None
        self._alerts = None
        self._summary = None
    
    
    def load_baseline_data(self) -> Dict:
        """Load baseline performance and fairness metrics."""
        if self._baseline_data is not None:
            return self._baseline_data
        
        # Load metadata
        with open(self.baseline_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load fairness audit
        with open(self.baseline_dir / "fairness_audit_report.json", 'r') as f:
            fairness = json.load(f)
        
        self._baseline_data = {
            "metadata": metadata,
            "fairness": fairness,
            "performance": metadata["performance_metrics"]["test"],
            "training_info": metadata["training_info"]
        }
        
        return self._baseline_data
    
    
    def load_batch_logs(self) -> List[Dict]:
        """Load all batch processing logs."""
        if self._batch_logs is not None:
            return self._batch_logs
        
        batch_logs_dir = self.monitoring_dir / "batch_logs"
        
        if not batch_logs_dir.exists():
            return []
        
        logs = []
        for log_file in sorted(batch_logs_dir.glob("batch_*.json")):
            with open(log_file, 'r') as f:
                logs.append(json.load(f))
        
        self._batch_logs = logs
        return logs
    
    
    def load_drift_reports(self) -> List[Dict]:
        """Load all drift detection reports."""
        if self._drift_reports is not None:
            return self._drift_reports
        
        drift_dir = self.monitoring_dir / "drift_reports"
        
        if not drift_dir.exists():
            return []
        
        reports = []
        for report_file in sorted(drift_dir.glob("drift_batch_*.json")):
            with open(report_file, 'r') as f:
                data = json.load(f)
                # Extract batch_id from filename
                batch_id = report_file.stem.replace("drift_", "")
                data["batch_id"] = batch_id
                reports.append(data)
        
        self._drift_reports = reports
        return reports
    
    
    def load_fairness_reports(self) -> List[Dict]:
        """Load all fairness monitoring reports."""
        if self._fairness_reports is not None:
            return self._fairness_reports
        
        fairness_dir = self.monitoring_dir / "fairness_reports"
        
        if not fairness_dir.exists():
            return []
        
        reports = []
        for report_file in sorted(fairness_dir.glob("fairness_batch_*.json")):
            with open(report_file, 'r') as f:
                reports.append(json.load(f))
        
        self._fairness_reports = reports
        return reports
    
    
    def load_alerts(self) -> Dict:
        """Load all alerts."""
        if self._alerts is not None:
            return self._alerts
        
        alerts_file = self.monitoring_dir / "alerts" / "all_alerts.json"
        
        if not alerts_file.exists():
            return {"alerts": [], "by_severity": {}, "by_type": {}}
        
        with open(alerts_file, 'r') as f:
            self._alerts = json.load(f)
        
        return self._alerts
    
    
    def load_summary(self) -> Dict:
        """Load monitoring summary report."""
        if self._summary is not None:
            return self._summary
        
        summary_file = self.monitoring_dir / "monitoring_summary.json"
        
        if not summary_file.exists():
            return {}
        
        with open(summary_file, 'r') as f:
            self._summary = json.load(f)
        
        return self._summary
    
    
    def get_performance_trend(self) -> pd.DataFrame:
        """
        Get performance metrics trend across batches.
        
        Returns:
            DataFrame with columns: batch_id, auc_roc, accuracy, bad_recall, etc.
        """
        logs = self.load_batch_logs()
        
        if not logs:
            return pd.DataFrame()
        
        data = []
        for log in logs:
            perf = log["performance"]
            data.append({
                "batch_id": log["batch_id"],
                "batch_num": int(log["batch_id"].split("_")[1]),
                "auc_roc": perf.get("auc_roc"),
                "accuracy": perf["accuracy"],
                "bad_precision": perf["bad_credit"]["precision"],
                "bad_recall": perf["bad_credit"]["recall"],
                "bad_f1": perf["bad_credit"]["f1_score"],
                "good_precision": perf["good_credit"]["precision"],
                "good_recall": perf["good_credit"]["recall"],
                "good_f1": perf["good_credit"]["f1_score"],
                "approval_rate": log["predictions"]["approval_rate"]
            })
        
        return pd.DataFrame(data).sort_values("batch_num")
    
    
    def get_drift_summary(self) -> pd.DataFrame:
        """
        Get drift detection summary across batches.
        
        Returns:
            DataFrame with drift statistics per batch
        """
        reports = self.load_drift_reports()
        
        if not reports:
            return pd.DataFrame()
        
        data = []
        for report in reports:
            summary = report.get("summary", {})
            pred_drift = report.get("prediction_drift", {})
            
            data.append({
                "batch_id": report["batch_id"],
                "batch_num": int(report["batch_id"].split("_")[1]),
                "total_features": summary.get("total_features", 0),
                "features_with_drift": summary.get("features_with_drift", 0),
                "drift_percentage": summary.get("drift_percentage", 0),
                "prediction_drift": pred_drift.get("drift_detected", False),
                "kl_divergence": pred_drift.get("kl_divergence", 0)
            })
        
        return pd.DataFrame(data).sort_values("batch_num")
    
    
    def get_fairness_trend(self, attribute: str = "Sex") -> pd.DataFrame:
        """
        Get fairness metrics trend for a sensitive attribute.
        
        Args:
            attribute: "Sex" or "Age_group"
        
        Returns:
            DataFrame with fairness metrics per batch
        """
        reports = self.load_fairness_reports()
        
        if not reports:
            return pd.DataFrame()
        
        data = []
        for report in reports:
            batch_id = report["batch_id"]
            batch_num = int(batch_id.split("_")[1])
            
            # Get metrics for the attribute
            attr_metrics = report["fairness_metrics"].get(attribute, {})
            
            if attr_metrics:
                data.append({
                    "batch_id": batch_id,
                    "batch_num": batch_num,
                    "demographic_parity": attr_metrics.get("demographic_parity_difference"),
                    "disparate_impact": attr_metrics.get("disparate_impact_ratio"),
                    "equal_opportunity": attr_metrics.get("equal_opportunity_difference"),
                    "violations": len([v for v in report["violations"] if v["attribute"] == attribute])
                })
        
        return pd.DataFrame(data).sort_values("batch_num")
    
    
    def get_alerts_by_severity(self) -> Dict[str, int]:
        """Get count of alerts by severity."""
        alerts = self.load_alerts()
        return alerts.get("by_severity", {"critical": 0, "warning": 0, "info": 0})
    
    
    def get_alerts_by_type(self) -> Dict[str, int]:
        """Get count of alerts by type."""
        alerts = self.load_alerts()
        return alerts.get("by_type", {})
    
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get N most recent alerts."""
        alerts = self.load_alerts()
        all_alerts = alerts.get("alerts", [])
        return sorted(all_alerts, key=lambda x: x["timestamp"], reverse=True)[:n]
    
    
    def get_batch_details(self, batch_id: str) -> Dict:
        """
        Get detailed information for a specific batch.
        
        Args:
            batch_id: Batch identifier (e.g., "batch_00")
        
        Returns:
            Dictionary with batch details
        """
        # Load batch log
        batch_log = None
        for log in self.load_batch_logs():
            if log["batch_id"] == batch_id:
                batch_log = log
                break
        
        # Load drift report
        drift_report = None
        for report in self.load_drift_reports():
            if report["batch_id"] == batch_id:
                drift_report = report
                break
        
        # Load fairness report
        fairness_report = None
        for report in self.load_fairness_reports():
            if report["batch_id"] == batch_id:
                fairness_report = report
                break
        
        # Load batch alerts
        alerts = self.load_alerts()
        batch_alerts = [a for a in alerts.get("alerts", []) if a.get("batch_id") == batch_id]
        
        return {
            "batch_log": batch_log,
            "drift_report": drift_report,
            "fairness_report": fairness_report,
            "alerts": batch_alerts
        }
    
    
    def get_feature_drift_heatmap_data(self) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Get data for feature drift heatmap.
        
        Returns:
            Tuple of (batch_ids, feature_names, drift_matrix)
            drift_matrix: 1 if drift detected, 0 otherwise
        """
        reports = self.load_drift_reports()
        
        if not reports:
            return [], [], np.array([])
        
        # Get all features
        first_report = reports[0]
        features = list(first_report.get("features", {}).keys())
        
        # Get all batch IDs
        batch_ids = [r["batch_id"] for r in reports]
        
        # Create drift matrix
        drift_matrix = np.zeros((len(batch_ids), len(features)))
        
        for i, report in enumerate(reports):
            for j, feature in enumerate(features):
                feature_data = report.get("features", {}).get(feature, {})
                if feature_data.get("drift_detected", False):
                    drift_matrix[i, j] = 1
        
        return batch_ids, features, drift_matrix
    
    
    def get_dashboard_summary(self) -> Dict:
        """
        Get high-level summary for dashboard overview.
        
        Returns:
            Dictionary with key metrics
        """
        baseline = self.load_baseline_data()
        summary = self.load_summary()
        performance_trend = self.get_performance_trend()
        alerts = self.load_alerts()
        
        # Calculate average metrics
        if not performance_trend.empty:
            current_auc = performance_trend["auc_roc"].mean()
            current_accuracy = performance_trend["accuracy"].mean()
            current_approval_rate = performance_trend["approval_rate"].mean()
        else:
            current_auc = None
            current_accuracy = None
            current_approval_rate = None
        
        # Compare to baseline
        baseline_auc = baseline["performance"]["auc_roc"]
        baseline_accuracy = baseline["performance"]["accuracy"]
        
        return {
            "total_batches": len(self.load_batch_logs()),
            "total_applications": summary.get("batch_processing_summary", {}).get("total_applications", 0),
            "baseline_auc": baseline_auc,
            "current_auc": current_auc,
            "auc_change": current_auc - baseline_auc if current_auc else None,
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current_accuracy,
            "accuracy_change": current_accuracy - baseline_accuracy if current_accuracy else None,
            "approval_rate": current_approval_rate,
            "total_alerts": alerts.get("total_alerts", 0),
            "critical_alerts": alerts.get("by_severity", {}).get("critical", 0),
            "warning_alerts": alerts.get("by_severity", {}).get("warning", 0),
            "batches_with_drift": summary.get("monitoring_summary", {}).get("batches_with_drift", 0),
            "batches_with_fairness_violations": summary.get("monitoring_summary", {}).get("batches_with_fairness_violations", 0)
        }
    
    
    def refresh_cache(self):
        """Clear all cached data to force reload."""
        self._baseline_data = None
        self._batch_logs = None
        self._drift_reports = None
        self._fairness_reports = None
        self._alerts = None
        self._summary = None


if __name__ == "__main__":
    # Test data loader
    loader = MonitoringDataLoader()
    
    print("Testing Data Loader...")
    print(f"Baseline loaded: {loader.load_baseline_data() is not None}")
    print(f"Batch logs: {len(loader.load_batch_logs())}")
    print(f"Drift reports: {len(loader.load_drift_reports())}")
    print(f"Fairness reports: {len(loader.load_fairness_reports())}")
    print(f"Total alerts: {loader.load_alerts().get('total_alerts', 0)}")