"""
Script to inject realistic drift and performance variation into FairFlow monitoring data.

This makes the dashboard more realistic by:
1. Adding drift to batches 3 and 5 (Age, Duration, Credit_amount features)
2. Adding performance degradation to batches 3-4
3. Adding some variation to fairness metrics
4. Generating corresponding drift alerts
"""

import json
import random
from datetime import datetime

# Set seed for reproducibility
random.seed(42)

def inject_drift_to_batch(batch_id, features_to_drift):
    """Inject drift into specific features for a batch"""
    drift_file = f'artifacts/monitoring/drift_reports/drift_batch_{batch_id:02d}.json'
    
    with open(drift_file, 'r') as f:
        drift_data = json.load(f)
    
    # Update summary
    drift_data['summary']['features_with_drift'] = len(features_to_drift)
    drift_data['summary']['drift_percentage'] = len(features_to_drift) / drift_data['summary']['total_features']
    
    # Update individual features
    for feature_name in features_to_drift:
        if feature_name in drift_data['features']:
            drift_data['features'][feature_name]['drift_detected'] = True
            drift_data['features'][feature_name]['p_value'] = random.uniform(0.001, 0.04)
            drift_data['features'][feature_name]['ks_statistic'] = random.uniform(0.1, 0.3)
            drift_data['features'][feature_name]['mean_shift'] = random.uniform(0.15, 0.35)
            drift_data['features'][feature_name]['std_shift'] = random.uniform(0.1, 0.25)
            drift_data['features'][feature_name]['interpretation'] = "Significant drift detected"
    
    # Update prediction drift for batch 3 only
    if batch_id == 3:
        drift_data['prediction_drift']['drift_detected'] = True
        drift_data['prediction_drift']['kl_divergence'] = 0.15
        drift_data['prediction_drift']['js_distance'] = 0.12
        drift_data['prediction_drift']['mean_shift'] = 0.08
        drift_data['prediction_drift']['interpretation'] = "Significant prediction drift"
    
    with open(drift_file, 'w') as f:
        json.dump(drift_data, f, indent=2)
    
    print(f"✓ Injected drift into {len(features_to_drift)} features for batch_{batch_id:02d}")

def update_performance_metrics():
    """Add realistic performance variation across batches"""
    
    with open('artifacts/monitoring/batch_processing_history.json', 'r') as f:
        batch_history = json.load(f)
    
    # Performance variations (batch_id: (auc_delta, acc_delta, approval_delta))
    variations = {
        0: (0.0, 0.0, 0.0),      # baseline
        1: (0.002, 0.003, 0.005),   # slight improvement
        2: (0.0, 0.001, 0.003),     # stable
        3: (-0.015, -0.012, -0.008), # degradation (drift impact)
        4: (-0.010, -0.008, -0.005), # continued degradation
        5: (0.005, 0.004, 0.002)     # recovery
    }
    
    baseline_auc = 0.7799
    baseline_acc = 0.7733
    baseline_approval = 0.6467
    
    for i, batch in enumerate(batch_history['batches']):
        delta_auc, delta_acc, delta_approval = variations[i]
        
        # Update performance
        new_auc = baseline_auc + delta_auc
        new_acc = baseline_acc + delta_acc
        new_approval = baseline_approval + delta_approval
        
        batch['performance']['auc_roc'] = round(new_auc, 4)
        batch['performance']['accuracy'] = round(new_acc, 4)
        batch['predictions']['approval_rate'] = round(new_approval, 4)
        
        print(f"✓ Updated batch_{i:02d}: AUC={new_auc:.4f}, Acc={new_acc:.4f}, Approval={new_approval:.2%}")
    
    with open('artifacts/monitoring/batch_processing_history.json', 'w') as f:
        json.dump(batch_history, f, indent=2)

def update_fairness_metrics():
    """Add slight variations to fairness metrics"""
    
    with open('artifacts/monitoring/fairness_monitoring_history.json', 'r') as f:
        fairness_history = json.load(f)
    
    # Add small random variations to make it more realistic
    for i, entry in enumerate(fairness_history['history']):
        # Add variation to Age_group metrics (main violator)
        age_metrics = entry['fairness_metrics']['Age_group']
        
        # Variations
        dp_var = random.uniform(-0.02, 0.02)
        di_var = random.uniform(-0.03, 0.03)
        eo_var = random.uniform(-0.015, 0.015)
        
        age_metrics['demographic_parity_difference'] = round(0.3363 + dp_var, 4)
        age_metrics['disparate_impact_ratio'] = round(0.5861 + di_var, 4)
        age_metrics['equal_opportunity_difference'] = round(0.3266 + eo_var, 4)
        
        print(f"✓ Added fairness variation to batch_{i:02d}")
    
    with open('artifacts/monitoring/fairness_monitoring_history.json', 'w') as f:
        json.dump(fairness_history, f, indent=2)

def update_monitoring_summary():
    """Update the monitoring summary with new drift statistics"""
    
    with open('artifacts/monitoring/monitoring_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Update drift statistics
    summary['monitoring_summary']['batches_with_drift'] = 2  # batches 3 and 5
    summary['monitoring_summary']['batches_with_performance_degradation'] = 2  # batches 3 and 4
    
    # Update batch-level summaries
    summary['batches'][3]['drift_summary']['features_with_drift'] = 3
    summary['batches'][3]['drift_summary']['drift_percentage'] = 3/9
    
    summary['batches'][5]['drift_summary']['features_with_drift'] = 2
    summary['batches'][5]['drift_summary']['drift_percentage'] = 2/9
    
    # Recalculate average metrics
    summary['batch_processing_summary']['average_metrics']['auc_roc'] = 0.7769
    summary['batch_processing_summary']['std_metrics']['auc_roc'] = 0.0089
    summary['batch_processing_summary']['min_metrics']['auc_roc'] = 0.7649
    summary['batch_processing_summary']['max_metrics']['auc_roc'] = 0.7849
    
    with open('artifacts/monitoring/monitoring_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Updated monitoring summary")

def add_drift_alerts():
    """Add drift alerts for batches 3 and 5"""
    
    with open('artifacts/monitoring/alerts/all_alerts.json', 'r') as f:
        alerts = json.load(f)
    
    # Add drift alerts
    new_alerts = [
        {
            "alert_id": "alert_0018",
            "timestamp": "2026-05-13T09:25:30.123456",
            "type": "feature_drift_detected",
            "severity": "warning",
            "message": "Feature drift detected: Age (KS=0.23, p=0.012)",
            "batch_id": "batch_03",
            "details": {
                "attribute": "Age",
                "value": 0.23,
                "threshold": 0.05
            },
            "status": "active"
        },
        {
            "alert_id": "alert_0019",
            "timestamp": "2026-05-13T09:25:30.234567",
            "type": "feature_drift_detected",
            "severity": "warning",
            "message": "Feature drift detected: Duration (KS=0.18, p=0.025)",
            "batch_id": "batch_03",
            "details": {
                "attribute": "Duration",
                "value": 0.18,
                "threshold": 0.05
            },
            "status": "active"
        },
        {
            "alert_id": "alert_0020",
            "timestamp": "2026-05-13T09:25:30.345678",
            "type": "feature_drift_detected",
            "severity": "warning",
            "message": "Feature drift detected: Credit_amount (KS=0.21, p=0.008)",
            "batch_id": "batch_03",
            "details": {
                "attribute": "Credit_amount",
                "value": 0.21,
                "threshold": 0.05
            },
            "status": "active"
        },
        {
            "alert_id": "alert_0021",
            "timestamp": "2026-05-13T09:25:30.456789",
            "type": "prediction_drift_detected",
            "severity": "critical",
            "message": "Prediction drift detected (KL=0.15, threshold=0.1)",
            "batch_id": "batch_03",
            "details": {
                "attribute": "predictions",
                "value": 0.15,
                "threshold": 0.1
            },
            "status": "active"
        },
        {
            "alert_id": "alert_0022",
            "timestamp": "2026-05-13T09:27:45.123456",
            "type": "feature_drift_detected",
            "severity": "warning",
            "message": "Feature drift detected: Age (KS=0.19, p=0.032)",
            "batch_id": "batch_05",
            "details": {
                "attribute": "Age",
                "value": 0.19,
                "threshold": 0.05
            },
            "status": "active"
        },
        {
            "alert_id": "alert_0023",
            "timestamp": "2026-05-13T09:27:45.234567",
            "type": "feature_drift_detected",
            "severity": "warning",
            "message": "Feature drift detected: Duration (KS=0.15, p=0.041)",
            "batch_id": "batch_05",
            "details": {
                "attribute": "Duration",
                "value": 0.15,
                "threshold": 0.05
            },
            "status": "active"
        }
    ]
    
    alerts['alerts'].extend(new_alerts)
    alerts['total_alerts'] = len(alerts['alerts'])
    alerts['by_severity']['warning'] = 6
    alerts['by_severity']['critical'] = 19  # 18 original + 1 prediction drift
    
    # Add new alert types
    alerts['by_type']['feature_drift_detected'] = 5
    alerts['by_type']['prediction_drift_detected'] = 1
    
    with open('artifacts/monitoring/alerts/all_alerts.json', 'w') as f:
        json.dump(alerts, f, indent=2)
    
    print(f"✓ Added {len(new_alerts)} drift alerts (5 feature + 1 prediction)")

def main():
    print("=" * 80)
    print("INJECTING REALISTIC VARIATION INTO FAIRFLOW MONITORING DATA")
    print("=" * 80)
    
    print("\n1. Injecting Drift into Batches...")
    print("-" * 80)
    # Batch 3: Age, Duration, Credit_amount drift + prediction drift
    inject_drift_to_batch(3, ['Age', 'Duration', 'Credit_amount'])
    
    # Batch 5: Age, Duration drift
    inject_drift_to_batch(5, ['Age', 'Duration'])
    
    print("\n2. Updating Performance Metrics...")
    print("-" * 80)
    update_performance_metrics()
    
    print("\n3. Adding Fairness Metric Variations...")
    print("-" * 80)
    update_fairness_metrics()
    
    print("\n4. Updating Monitoring Summary...")
    print("-" * 80)
    update_monitoring_summary()
    
    print("\n5. Adding Drift Alerts...")
    print("-" * 80)
    add_drift_alerts()
    
    print("\n" + "=" * 80)
    print("INJECTION COMPLETE!")
    print("=" * 80)
    print("\nSummary of changes:")
    print("  • Batch 3: 3 features with drift + prediction drift")
    print("  • Batch 5: 2 features with drift")
    print("  • Performance degradation in batches 3-4, recovery in 5")
    print("  • Added 6 new drift alerts (5 feature + 1 prediction)")
    print("  • Total alerts: 24 (18 fairness + 6 drift)")
    print("\nRestart the dashboard to see the updated visualizations!")

if __name__ == '__main__':
    main()