"""
Alert Generation System
=======================
Generates alerts for:
- Drift detection (feature and prediction drift)
- Fairness violations (demographic parity, disparate impact)
- Performance degradation (AUC, accuracy drops)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class AlertSystem:
    """
    Generates and manages alerts for monitoring violations.
    """
    
    def __init__(self, alerts_dir: Path):
        """
        Initialize alert system.
        
        Args:
            alerts_dir: Directory to save alerts
        """
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.active_alerts = []
        self.alert_history = []
    
    
    def create_alert(self, 
                    alert_type: str,
                    severity: str,
                    message: str,
                    details: Dict,
                    batch_id: str = None) -> Dict:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert (drift, fairness_violation, performance_degradation)
            severity: Alert severity (critical, warning, info)
            message: Human-readable alert message
            details: Additional alert details
            batch_id: Associated batch ID
        
        Returns:
            Alert dictionary
        """
        alert = {
            "alert_id": f"alert_{len(self.alert_history):04d}",
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "severity": severity,
            "message": message,
            "batch_id": batch_id,
            "details": details,
            "status": "active"
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        return alert
    
    
    def generate_drift_alerts(self, drift_results: Dict, batch_id: str) -> List[Dict]:
        """
        Generate alerts from drift detection results.
        
        Args:
            drift_results: Results from DriftDetector
            batch_id: Batch identifier
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check feature drift
        if "features" in drift_results:
            for feature, result in drift_results["features"].items():
                if result.get("drift_detected", False):
                    # Determine severity based on p-value
                    p_value = result.get("p_value", result.get("kl_divergence", 1))
                    severity = "critical" if p_value < 0.01 else "warning"
                    
                    alert = self.create_alert(
                        alert_type="drift_detected",
                        severity=severity,
                        message=f"Drift detected in feature '{feature}'",
                        details={
                            "feature": feature,
                            "method": result.get("method"),
                            "p_value": p_value,
                            "interpretation": result.get("interpretation")
                        },
                        batch_id=batch_id
                    )
                    alerts.append(alert)
        
        # Check prediction drift
        if drift_results.get("prediction_drift", {}).get("drift_detected", False):
            pred_drift = drift_results["prediction_drift"]
            
            severity = "critical" if pred_drift["kl_divergence"] > 0.15 else "warning"
            
            alert = self.create_alert(
                alert_type="prediction_drift",
                severity=severity,
                message=f"Prediction distribution has shifted",
                details={
                    "kl_divergence": pred_drift["kl_divergence"],
                    "mean_shift": pred_drift["mean_shift"],
                    "interpretation": pred_drift["interpretation"]
                },
                batch_id=batch_id
            )
            alerts.append(alert)
        
        return alerts
    
    
    def generate_fairness_alerts(self, violations: List[Dict], batch_id: str) -> List[Dict]:
        """
        Generate alerts from fairness violations.
        
        Args:
            violations: List of fairness violations
            batch_id: Batch identifier
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for violation in violations:
            alert = self.create_alert(
                alert_type=violation["type"],
                severity=violation["severity"],
                message=violation["message"],
                details={
                    "attribute": violation["attribute"],
                    "value": violation["value"],
                    "threshold": violation["threshold"]
                },
                batch_id=batch_id
            )
            alerts.append(alert)
        
        return alerts
    
    
    def generate_performance_alerts(self, 
                                   comparison: Dict, 
                                   batch_id: str) -> List[Dict]:
        """
        Generate alerts from performance degradation.
        
        Args:
            comparison: Performance comparison results
            batch_id: Batch identifier
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        if comparison.get("degradation_detected", False):
            for deg_alert in comparison.get("degradation_alerts", []):
                alert = self.create_alert(
                    alert_type="performance_degradation",
                    severity=deg_alert["severity"],
                    message=deg_alert["message"],
                    details={
                        "metric": deg_alert["metric"]
                    },
                    batch_id=batch_id
                )
                alerts.append(alert)
        
        return alerts
    
    
    def save_alerts(self, batch_id: str = None):
        """
        Save alerts to file.
        
        Args:
            batch_id: If provided, save only alerts for this batch
        """
        if batch_id:
            # Save batch-specific alerts
            batch_alerts = [a for a in self.active_alerts if a.get("batch_id") == batch_id]
            filepath = self.alerts_dir / f"alerts_{batch_id}.json"
            data = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "alert_count": len(batch_alerts),
                "alerts": batch_alerts
            }
        else:
            # Save all alerts
            filepath = self.alerts_dir / "all_alerts.json"
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_alerts": len(self.alert_history),
                "active_alerts": len(self.active_alerts),
                "by_severity": self._count_by_severity(),
                "by_type": self._count_by_type(),
                "alerts": self.alert_history
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts."""
        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(self.active_alerts),
            "by_severity": self._count_by_severity(),
            "by_type": self._count_by_type(),
            "recent_alerts": self.alert_history[-5:] if self.alert_history else []
        }
    
    
    def _count_by_severity(self) -> Dict:
        """Count alerts by severity level."""
        counts = {"critical": 0, "warning": 0, "info": 0}
        for alert in self.alert_history:
            severity = alert.get("severity", "info")
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    
    def _count_by_type(self) -> Dict:
        """Count alerts by type."""
        counts = {}
        for alert in self.alert_history:
            alert_type = alert.get("type", "unknown")
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts
    
    
    def clear_alerts(self):
        """Clear active alerts."""
        self.active_alerts = []
    
    
    def print_alert_summary(self):
        """Print human-readable alert summary."""
        summary = self.get_alert_summary()
        
        print("\n" + "="*70)
        print("ALERT SUMMARY")
        print("="*70)
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print()
        
        print("By Severity:")
        for severity, count in summary['by_severity'].items():
            icon = "🔴" if severity == "critical" else "⚠️" if severity == "warning" else "ℹ️"
            print(f"  {icon} {severity.capitalize()}: {count}")
        
        print()
        print("By Type:")
        for alert_type, count in summary['by_type'].items():
            print(f"  - {alert_type}: {count}")
        
        if summary['recent_alerts']:
            print()
            print("Recent Alerts:")
            for alert in summary['recent_alerts'][-3:]:
                print(f"  [{alert['severity'].upper()}] {alert['message']} (Batch: {alert.get('batch_id', 'N/A')})")
        
        print("="*70)


if __name__ == "__main__":
    print("Alert Generation System Module Loaded")