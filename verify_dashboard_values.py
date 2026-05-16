"""
Quick verification script to show exact values displayed in the dashboard
"""
import json
import pandas as pd

def load_monitoring_data():
    base_path = "artifacts/monitoring"
    
    with open(f"{base_path}/monitoring_summary.json", 'r') as f:
        summary = json.load(f)
    
    with open(f"{base_path}/batch_processing_history.json", 'r') as f:
        batch_history = json.load(f)
    
    with open(f"{base_path}/fairness_monitoring_history.json", 'r') as f:
        fairness_history = json.load(f)
    
    with open(f"{base_path}/alerts/all_alerts.json", 'r') as f:
        alerts = json.load(f)
    
    drift_reports = []
    for i in range(6):
        with open(f"{base_path}/drift_reports/drift_batch_0{i}.json", 'r') as f:
            drift_reports.append(json.load(f))
    
    return summary, batch_history, fairness_history, alerts, drift_reports

summary, batch_history, fairness_history, alerts, drift_reports = load_monitoring_data()

print("DASHBOARD KPI VALUES:")
print("=" * 80)
print(f"AUC-ROC Score: {summary['baseline_performance']['auc_roc']:.4f}")
print(f"Model Accuracy: {summary['baseline_performance']['accuracy']:.2%}")
print(f"Total Applications: {summary['batch_processing_summary']['total_applications']:,}")
print(f"Approved: {summary['batch_processing_summary']['total_approved']:,}")
print(f"Overall Approval Rate: {summary['batch_processing_summary']['overall_approval_rate']:.1%}")
print(f"Critical Alerts: {alerts['by_severity']['critical']}")
print(f"Batches with Violations: {summary['monitoring_summary']['batches_with_fairness_violations']}/6")

print("\n\nPERFORMANCE CHART DATA:")
print("=" * 80)
for batch in batch_history['batches']:
    print(f"{batch['batch_id']}: AUC={batch['performance']['auc_roc']:.4f}, "
          f"Acc={batch['performance']['accuracy']:.4f}, "
          f"Approval={batch['predictions']['approval_rate']:.2%}")

print("\n\nDRIFT DONUT VALUES:")
print("=" * 80)
total_features = sum(d['summary']['total_features'] for d in drift_reports)
features_with_drift = sum(d['summary']['features_with_drift'] for d in drift_reports)
features_no_drift = total_features - features_with_drift
print(f"No Drift Detected: {features_no_drift} ({features_no_drift/total_features*100:.1f}%)")
print(f"Drift Detected: {features_with_drift} ({features_with_drift/total_features*100:.1f}%)")

print("\n\nFAIRNESS METRICS (Demographic Parity):")
print("=" * 80)
for entry in fairness_history['history']:
    sex = entry['fairness_metrics']['Sex']['demographic_parity_difference']
    age = entry['fairness_metrics']['Age_group']['demographic_parity_difference']
    print(f"{entry['batch_id']}: Sex={sex:.4f}, Age={age:.4f}")

print("\n\nALERT BREAKDOWN:")
print("=" * 80)
print(f"Total Alerts: {len(alerts['alerts'])}")
print(f"Critical: {alerts['by_severity']['critical']}")
print(f"Warning: {alerts['by_severity']['warning']}")
print(f"Info: {alerts['by_severity']['info']}")
print("\nBy Type:")
for alert_type, count in alerts['by_type'].items():
    print(f"  {alert_type}: {count}")

print("\n\nALERT DISTRIBUTION BY BATCH:")
print("=" * 80)
batch_alert_count = {}
for alert in alerts['alerts']:
    batch_id = alert['batch_id']
    batch_alert_count[batch_id] = batch_alert_count.get(batch_id, 0) + 1

for batch_id in sorted(batch_alert_count.keys()):
    count = batch_alert_count[batch_id]
    print(f"{batch_id}: {count} alerts")

print("\n\nLATEST BATCH APPROVAL RATES:")
print("=" * 80)
latest = fairness_history['history'][-1]
print(f"Batch: {latest['batch_id']}")
print("\nSex Groups:")
for group, rate in latest['fairness_metrics']['Sex']['approval_rates'].items():
    print(f"  {group}: {rate:.2%}")
print("\nAge Groups:")
for group, rate in latest['fairness_metrics']['Age_group']['approval_rates'].items():
    print(f"  {group}: {rate:.2%}")