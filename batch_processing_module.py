"""
Batch Processing Pipeline
=========================
Simulates production credit approval workflow with:
- Batch data loading and preprocessing
- Model predictions on batches
- Performance evaluation per batch
- Logging and artifact generation
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)


class BatchProcessor:
    """
    Processes credit applications in batches and evaluates performance.
    """
    
    def __init__(self, model_path: str, threshold: float = 0.35):
        """
        Initialize batch processor with trained model.
        
        Args:
            model_path: Path to trained model pickle file
            threshold: Decision threshold for predictions
        """
        # Load model
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)
        
        if isinstance(loaded, dict):
            self.model = loaded['model']
            self.features = loaded['features']
            self.threshold = loaded.get('threshold', threshold)
        else:
            # Old format
            self.model = loaded
            self.features = None
            self.threshold = threshold
        
        self.batch_history = []
        print(f"BatchProcessor initialized with threshold: {self.threshold:.2f}")
        print(f"Features: {len(self.features) if self.features else 'auto-detect'}")
    
    
    def process_batch(self, 
                     batch_data: pd.DataFrame,
                     batch_labels: pd.Series,
                     batch_id: str,
                     log_dir: Path = None) -> Dict:
        """
        Process a single batch of credit applications.
        
        Args:
            batch_data: Batch features
            batch_labels: True labels for evaluation
            batch_id: Batch identifier
            log_dir: Directory to save logs
        
        Returns:
            Dictionary with batch results
        """
        # Auto-detect features if not loaded from model
        if self.features is None:
            self.features = [col for col in batch_data.columns 
                           if col not in ['Sex', 'Age_group', 'Sex_original', 'Age_original']]
        
        # Get predictions
        y_prob = self.model.predict_proba(batch_data[self.features])[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(batch_labels, y_pred, y_prob)
        
        # Create batch report
        batch_report = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(batch_data),
            "threshold": self.threshold,
            "predictions": {
                "total_applications": len(y_pred),
                "approved": int(y_pred.sum()),
                "rejected": int((y_pred == 0).sum()),
                "approval_rate": float(y_pred.mean()),
                "mean_probability": float(y_prob.mean()),
                "std_probability": float(y_prob.std())
            },
            "performance": metrics,
            "confusion_matrix": self._get_confusion_matrix(batch_labels, y_pred),
        }
        
        # Add to history
        self.batch_history.append(batch_report)
        
        # Save log if directory provided
        if log_dir:
            log_path = log_dir / f"{batch_id}.json"
            with open(log_path, 'w') as f:
                json.dump(batch_report, f, indent=2)
            print(f"   Batch log saved: {log_path}")
        
        return batch_report
    
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # AUC-ROC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None
        
        # Precision, recall, f1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            "auc_roc": round(auc, 4) if auc else None,
            "accuracy": round(report["accuracy"], 4),
            "bad_credit": {
                "precision": round(report["0"]["precision"], 4),
                "recall": round(report["0"]["recall"], 4),
                "f1_score": round(report["0"]["f1-score"], 4),
                "support": int(report["0"]["support"])
            },
            "good_credit": {
                "precision": round(report["1"]["precision"], 4),
                "recall": round(report["1"]["recall"], 4),
                "f1_score": round(report["1"]["f1-score"], 4),
                "support": int(report["1"]["support"])
            },
            "macro_avg": {
                "precision": round(report["macro avg"]["precision"], 4),
                "recall": round(report["macro avg"]["recall"], 4),
                "f1_score": round(report["macro avg"]["f1-score"], 4)
            },
            "weighted_avg": {
                "precision": round(report["weighted avg"]["precision"], 4),
                "recall": round(report["weighted avg"]["recall"], 4),
                "f1_score": round(report["weighted avg"]["f1-score"], 4)
            }
        }
        
        return metrics
    
    
    def _get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Get confusion matrix as dictionary."""
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            "true_negative": int(cm[0, 0]),   # Correctly rejected bad credit
            "false_positive": int(cm[0, 1]),  # Incorrectly approved bad credit
            "false_negative": int(cm[1, 0]),  # Incorrectly rejected good credit
            "true_positive": int(cm[1, 1])    # Correctly approved good credit
        }
    
    
    def get_batch_results(self, batch_id: str) -> Dict:
        """Retrieve results for a specific batch."""
        for batch in self.batch_history:
            if batch["batch_id"] == batch_id:
                return batch
        return {"error": f"Batch {batch_id} not found"}
    
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all processed batches."""
        if not self.batch_history:
            return {"error": "No batches processed yet"}
        
        # Aggregate metrics
        total_applications = sum(b["batch_size"] for b in self.batch_history)
        total_approved = sum(b["predictions"]["approved"] for b in self.batch_history)
        
        # Average performance metrics
        auc_scores = [b["performance"]["auc_roc"] for b in self.batch_history 
                     if b["performance"]["auc_roc"] is not None]
        accuracies = [b["performance"]["accuracy"] for b in self.batch_history]
        approval_rates = [b["predictions"]["approval_rate"] for b in self.batch_history]
        
        summary = {
            "total_batches": len(self.batch_history),
            "total_applications": total_applications,
            "total_approved": total_approved,
            "total_rejected": total_applications - total_approved,
            "overall_approval_rate": round(total_approved / total_applications, 4),
            "average_metrics": {
                "auc_roc": round(np.mean(auc_scores), 4) if auc_scores else None,
                "accuracy": round(np.mean(accuracies), 4),
                "approval_rate": round(np.mean(approval_rates), 4)
            },
            "std_metrics": {
                "auc_roc": round(np.std(auc_scores), 4) if auc_scores else None,
                "accuracy": round(np.std(accuracies), 4),
                "approval_rate": round(np.std(approval_rates), 4)
            },
            "min_metrics": {
                "auc_roc": round(min(auc_scores), 4) if auc_scores else None,
                "accuracy": round(min(accuracies), 4)
            },
            "max_metrics": {
                "auc_roc": round(max(auc_scores), 4) if auc_scores else None,
                "accuracy": round(max(accuracies), 4)
            }
        }
        
        return summary
    
    
    def compare_to_baseline(self, baseline_metrics: Dict) -> Dict:
        """
        Compare batch processing results to baseline.
        
        Args:
            baseline_metrics: Baseline performance from Part 1
        
        Returns:
            Comparison report
        """
        if not self.batch_history:
            return {"error": "No batches processed yet"}
        
        summary = self.get_summary_statistics()
        
        comparison = {
            "baseline": baseline_metrics,
            "current": summary["average_metrics"],
            "performance_change": {},
            "degradation_detected": False,
            "degradation_alerts": []
        }
        
        # Compare AUC-ROC
        if baseline_metrics.get("auc_roc") and summary["average_metrics"]["auc_roc"]:
            auc_change = summary["average_metrics"]["auc_roc"] - baseline_metrics["auc_roc"]
            comparison["performance_change"]["auc_roc"] = {
                "baseline": baseline_metrics["auc_roc"],
                "current": summary["average_metrics"]["auc_roc"],
                "change": round(auc_change, 4),
                "percent_change": round(100 * auc_change / baseline_metrics["auc_roc"], 2)
            }
            
            if auc_change < -0.05:  # 5% drop threshold
                comparison["degradation_detected"] = True
                comparison["degradation_alerts"].append({
                    "metric": "auc_roc",
                    "severity": "critical" if auc_change < -0.1 else "warning",
                    "message": f"AUC-ROC dropped by {abs(auc_change):.4f} from baseline"
                })
        
        # Compare Accuracy
        if baseline_metrics.get("accuracy"):
            acc_change = summary["average_metrics"]["accuracy"] - baseline_metrics["accuracy"]
            comparison["performance_change"]["accuracy"] = {
                "baseline": baseline_metrics["accuracy"],
                "current": summary["average_metrics"]["accuracy"],
                "change": round(acc_change, 4),
                "percent_change": round(100 * acc_change / baseline_metrics["accuracy"], 2)
            }
            
            if acc_change < -0.05:
                comparison["degradation_detected"] = True
                comparison["degradation_alerts"].append({
                    "metric": "accuracy",
                    "severity": "critical" if acc_change < -0.1 else "warning",
                    "message": f"Accuracy dropped by {abs(acc_change):.4f} from baseline"
                })
        
        return comparison
    
    
    def save_history(self, filepath: Path):
        """Save batch processing history to JSON."""
        history_data = {
            "threshold": self.threshold,
            "features": self.features,
            "batches": self.batch_history,
            "summary": self.get_summary_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Batch history saved to {filepath}")


def create_batches_from_data(data_path: str, 
                            labels_path: str,
                            batch_size: int = 150,
                            num_batches: int = 6) -> List[Tuple[pd.DataFrame, pd.Series, str]]:
    """
    Create batches from test data for processing.
    
    Args:
        data_path: Path to test data parquet
        labels_path: Path to test labels parquet
        batch_size: Size of each batch
        num_batches: Number of batches to create
    
    Returns:
        List of (batch_data, batch_labels, batch_id) tuples
    """
    X = pd.read_parquet(data_path)
    y = pd.read_parquet(labels_path).squeeze()
    
    batches = []
    
    for i in range(num_batches):
        start_idx = (i * batch_size) % len(X)
        end_idx = start_idx + batch_size
        
        # Handle wrap-around if needed
        if end_idx > len(X):
            batch_X = pd.concat([X.iloc[start_idx:], X.iloc[:end_idx - len(X)]])
            batch_y = pd.concat([y.iloc[start_idx:], y.iloc[:end_idx - len(X)]])
        else:
            batch_X = X.iloc[start_idx:end_idx]
            batch_y = y.iloc[start_idx:end_idx]
        
        # Reset index
        batch_X = batch_X.reset_index(drop=True)
        batch_y = batch_y.reset_index(drop=True)
        
        batch_id = f"batch_{i:02d}"
        batches.append((batch_X, batch_y, batch_id))
    
    return batches


if __name__ == "__main__":
    print("Batch Processing Pipeline Module Loaded")
    print("Ready to process credit applications in batches")