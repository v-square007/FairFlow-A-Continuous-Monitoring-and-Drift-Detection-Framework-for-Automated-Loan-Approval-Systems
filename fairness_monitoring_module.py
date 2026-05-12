"""
Fairness Monitoring Module
===========================
Tracks fairness metrics over time and detects violations:
- Demographic Parity: Equal approval rates across groups
- Disparate Impact: Ratio of approval rates (min/max)
- Equal Opportunity: Equal TPR for positive class across groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import json


class FairnessMonitor:
    """
    Monitors fairness metrics and detects violations.
    """
    
    def __init__(self, baseline_fairness: Dict, sensitive_attributes: List[str]):
        """
        Initialize fairness monitor with baseline metrics.
        
        Args:
            baseline_fairness: Baseline fairness metrics from Part 1
            sensitive_attributes: List of sensitive attributes to monitor
        """
        self.baseline_fairness = baseline_fairness
        self.sensitive_attributes = sensitive_attributes
        self.history = []  # Track fairness over time
    
    
    def compute_fairness_metrics(self, 
                                 data: pd.DataFrame,
                                 predictions: np.ndarray,
                                 true_labels: np.ndarray,
                                 sensitive_attr: str) -> Dict:
        """
        Compute fairness metrics for a sensitive attribute.
        
        Args:
            data: DataFrame with sensitive attributes
            predictions: Model predictions (0 or 1)
            true_labels: Ground truth labels
            sensitive_attr: Attribute to analyze (e.g., 'Sex', 'Age_group')
        
        Returns:
            Dictionary with fairness metrics
        """
        if sensitive_attr not in data.columns:
            return {
                "error": f"{sensitive_attr} not found in data"
            }
        
        results = {
            "attribute": sensitive_attr,
            "timestamp": datetime.now().isoformat(),
            "groups": {},
            "approval_rates": {},
            "true_positive_rates": {}
        }
        
        # Calculate metrics per group
        for group in data[sensitive_attr].unique():
            group_idx = data[sensitive_attr] == group
            group_preds = predictions[group_idx]
            group_labels = true_labels[group_idx]
            
            # Approval rate (what fraction are predicted positive)
            approval_rate = group_preds.mean()
            results["approval_rates"][str(group)] = float(approval_rate)
            
            # True Positive Rate (for equal opportunity)
            positive_idx = group_labels == 1
            if positive_idx.sum() > 0:
                tpr = group_preds[positive_idx].mean()
                results["true_positive_rates"][str(group)] = float(tpr)
            else:
                results["true_positive_rates"][str(group)] = None
            
            # Group size
            results["groups"][str(group)] = {
                "size": int(group_idx.sum()),
                "approval_rate": float(approval_rate),
                "tpr": float(tpr) if positive_idx.sum() > 0 else None
            }
        
        # Calculate aggregate fairness metrics
        approval_rates = list(results["approval_rates"].values())
        
        # Demographic Parity: Difference between max and min approval rates
        results["demographic_parity_difference"] = round(
            max(approval_rates) - min(approval_rates), 4
        )
        
        # Disparate Impact: Ratio of min to max approval rates
        results["disparate_impact_ratio"] = round(
            min(approval_rates) / max(approval_rates), 4
        ) if max(approval_rates) > 0 else 0.0
        
        # Equal Opportunity: Difference in TPR
        tpr_values = [v for v in results["true_positive_rates"].values() if v is not None]
        if len(tpr_values) > 1:
            results["equal_opportunity_difference"] = round(
                max(tpr_values) - min(tpr_values), 4
            )
        else:
            results["equal_opportunity_difference"] = None
        
        return results
    
    
    def check_fairness_violations(self, 
                                  fairness_metrics: Dict,
                                  thresholds: Dict) -> List[Dict]:
        """
        Check if fairness metrics violate thresholds.
        
        Args:
            fairness_metrics: Current fairness metrics
            thresholds: Violation thresholds
        
        Returns:
            List of violation alerts
        """
        violations = []
        
        # Check demographic parity
        dp = fairness_metrics.get("demographic_parity_difference")
        if dp is not None and dp > thresholds["demographic_parity_max"]:
            severity = "critical" if dp > 0.15 else "warning"
            violations.append({
                "type": "demographic_parity_violation",
                "attribute": fairness_metrics["attribute"],
                "value": dp,
                "threshold": thresholds["demographic_parity_max"],
                "severity": severity,
                "message": f"Demographic parity difference ({dp:.4f}) exceeds threshold ({thresholds['demographic_parity_max']})"
            })
        
        # Check disparate impact
        di = fairness_metrics.get("disparate_impact_ratio")
        if di is not None and di < thresholds["disparate_impact_min"]:
            severity = "critical" if di < 0.7 else "warning"
            violations.append({
                "type": "disparate_impact_violation",
                "attribute": fairness_metrics["attribute"],
                "value": di,
                "threshold": thresholds["disparate_impact_min"],
                "severity": severity,
                "message": f"Disparate impact ratio ({di:.4f}) below threshold ({thresholds['disparate_impact_min']})"
            })
        
        # Check equal opportunity
        eo = fairness_metrics.get("equal_opportunity_difference")
        if eo is not None and eo > thresholds["equal_opportunity_max"]:
            severity = "critical" if eo > 0.15 else "warning"
            violations.append({
                "type": "equal_opportunity_violation",
                "attribute": fairness_metrics["attribute"],
                "value": eo,
                "threshold": thresholds["equal_opportunity_max"],
                "severity": severity,
                "message": f"Equal opportunity difference ({eo:.4f}) exceeds threshold ({thresholds['equal_opportunity_max']})"
            })
        
        return violations
    
    
    def monitor_batch(self,
                     batch_data: pd.DataFrame,
                     predictions: np.ndarray,
                     true_labels: np.ndarray,
                     thresholds: Dict,
                     batch_id: str = None) -> Dict:
        """
        Monitor fairness for a single batch.
        
        Args:
            batch_data: Batch data with sensitive attributes
            predictions: Model predictions
            true_labels: Ground truth labels
            thresholds: Fairness violation thresholds
            batch_id: Batch identifier
        
        Returns:
            Monitoring report for this batch
        """
        report = {
            "batch_id": batch_id or f"batch_{len(self.history)}",
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(batch_data),
            "fairness_metrics": {},
            "violations": [],
            "summary": {
                "total_violations": 0,
                "critical_violations": 0,
                "warning_violations": 0
            }
        }
        
        # Monitor each sensitive attribute
        for attr in self.sensitive_attributes:
            if attr in batch_data.columns:
                # Compute fairness metrics
                metrics = self.compute_fairness_metrics(
                    batch_data, predictions, true_labels, attr
                )
                report["fairness_metrics"][attr] = metrics
                
                # Check for violations
                violations = self.check_fairness_violations(metrics, thresholds)
                report["violations"].extend(violations)
                
                # Count violations by severity
                for v in violations:
                    if v["severity"] == "critical":
                        report["summary"]["critical_violations"] += 1
                    elif v["severity"] == "warning":
                        report["summary"]["warning_violations"] += 1
        
        report["summary"]["total_violations"] = len(report["violations"])
        
        # Add to history
        self.history.append(report)
        
        return report
    
    
    def compare_to_baseline(self, current_metrics: Dict, 
                           sensitive_attr: str) -> Dict:
        """
        Compare current fairness metrics to baseline.
        
        Args:
            current_metrics: Current fairness metrics
            sensitive_attr: Attribute being compared
        
        Returns:
            Comparison results
        """
        if sensitive_attr not in self.baseline_fairness:
            return {
                "error": f"No baseline for {sensitive_attr}"
            }
        
        baseline = self.baseline_fairness[sensitive_attr]
        
        comparison = {
            "attribute": sensitive_attr,
            "demographic_parity": {
                "baseline": baseline.get("demographic_parity_difference"),
                "current": current_metrics.get("demographic_parity_difference"),
                "change": None
            },
            "disparate_impact": {
                "baseline": baseline.get("disparate_impact_ratio"),
                "current": current_metrics.get("disparate_impact_ratio"),
                "change": None
            },
            "equal_opportunity": {
                "baseline": baseline.get("equal_opportunity_difference"),
                "current": current_metrics.get("equal_opportunity_difference"),
                "change": None
            }
        }
        
        # Calculate changes
        for metric in ["demographic_parity", "disparate_impact", "equal_opportunity"]:
            b_val = comparison[metric]["baseline"]
            c_val = comparison[metric]["current"]
            
            if b_val is not None and c_val is not None:
                if metric == "disparate_impact":
                    # For disparate impact, positive change = improvement (closer to 1)
                    comparison[metric]["change"] = round(c_val - b_val, 4)
                    comparison[metric]["improved"] = c_val > b_val
                else:
                    # For others, negative change = improvement (closer to 0)
                    comparison[metric]["change"] = round(c_val - b_val, 4)
                    comparison[metric]["improved"] = c_val < b_val
        
        return comparison
    
    
    def get_summary_report(self) -> Dict:
        """
        Get summary report across all monitored batches.
        
        Returns:
            Summary statistics
        """
        if not self.history:
            return {
                "error": "No monitoring history available"
            }
        
        summary = {
            "total_batches": len(self.history),
            "total_violations": sum(b["summary"]["total_violations"] for b in self.history),
            "critical_violations": sum(b["summary"]["critical_violations"] for b in self.history),
            "warning_violations": sum(b["summary"]["warning_violations"] for b in self.history),
            "batches_with_violations": sum(1 for b in self.history if b["summary"]["total_violations"] > 0),
            "violation_rate": round(
                sum(1 for b in self.history if b["summary"]["total_violations"] > 0) / len(self.history),
                4
            ),
            "by_attribute": {}
        }
        
        # Aggregate by sensitive attribute
        for attr in self.sensitive_attributes:
            attr_violations = [
                v for batch in self.history 
                for v in batch["violations"] 
                if v["attribute"] == attr
            ]
            
            summary["by_attribute"][attr] = {
                "total_violations": len(attr_violations),
                "critical": sum(1 for v in attr_violations if v["severity"] == "critical"),
                "warning": sum(1 for v in attr_violations if v["severity"] == "warning"),
                "violation_types": {}
            }
            
            # Count by violation type
            for v_type in ["demographic_parity_violation", "disparate_impact_violation", 
                          "equal_opportunity_violation"]:
                count = sum(1 for v in attr_violations if v["type"] == v_type)
                if count > 0:
                    summary["by_attribute"][attr]["violation_types"][v_type] = count
        
        return summary
    
    
    def save_history(self, filepath: str):
        """Save monitoring history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "baseline": self.baseline_fairness,
                "sensitive_attributes": self.sensitive_attributes,
                "history": self.history,
                "summary": self.get_summary_report()
            }, f, indent=2)
        print(f"Monitoring history saved to {filepath}")


if __name__ == "__main__":
    print("Fairness Monitoring Module Loaded")
    print("Tracks: Demographic Parity, Disparate Impact, Equal Opportunity")