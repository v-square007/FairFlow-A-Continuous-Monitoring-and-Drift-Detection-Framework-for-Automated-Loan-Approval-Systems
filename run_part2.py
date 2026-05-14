#!/usr/bin/env python3
"""
Part 2: Drift & Fairness Monitoring - Main Execution Script
============================================================
Orchestrates the complete monitoring pipeline:
1. Load baseline data and model
2. Process batches simulating production
3. Detect drift (features and predictions)
4. Monitor fairness violations
5. Track performance degradation
6. Generate alerts
7. Save monitoring artifacts

Usage:
    python run_part2.py
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import custom modules
from monitoring_config import (
    BASELINE_DIR, MONITORING_DIR, BATCH_LOGS_DIR, DRIFT_REPORTS_DIR,
    FAIRNESS_REPORTS_DIR, ALERTS_DIR, MODEL_PATH, BATCH_CONFIG,
    DRIFT_THRESHOLDS, FAIRNESS_THRESHOLDS, FEATURE_DRIFT_CONFIG,
    get_baseline_metrics
)
from drift_detection_module import DriftDetector
from fairness_monitoring_module import FairnessMonitor
from batch_processing_module import BatchProcessor, create_batches_from_data
from alert_system_module import AlertSystem

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    
def main():
    print("="*80)
    print("PART 2: DRIFT & FAIRNESS MONITORING SYSTEM")
    print("="*80)
    print()
    
    # ========================================================================
    # STEP 1: LOAD BASELINE DATA
    # ========================================================================
    print("[1/8] Loading baseline data and model...")
    
    try:
        baseline_metrics = get_baseline_metrics()
        print(f"   ✓ Baseline AUC-ROC: {baseline_metrics['performance']['auc_roc']}")
        print(f"   ✓ Baseline threshold: {baseline_metrics['threshold']}")
        
        # Load baseline data for drift detection
        X_test_baseline = pd.read_parquet("artifacts/splits/X_test.parquet")
        y_test_baseline = pd.read_parquet("artifacts/splits/y_test.parquet").squeeze()
        
        print(f"   ✓ Baseline samples: {len(X_test_baseline)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load baseline data: {e}")
        print("   Make sure Part 1 has been completed successfully.")
        return 1
    
    print()
    
    # ========================================================================
    # STEP 2: INITIALIZE MONITORING COMPONENTS
    # ========================================================================
    print("[2/8] Initializing monitoring components...")
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        model_path=str(MODEL_PATH),
        threshold=baseline_metrics['threshold']
    )
    print("   ✓ Batch processor initialized")
    
    # Get baseline predictions for drift detection
    baseline_predictions = batch_processor.model.predict_proba(
        X_test_baseline[batch_processor.features]
    )[:, 1]
    
    # Initialize drift detector
    drift_detector = DriftDetector(
        baseline_data=X_test_baseline,
        baseline_predictions=baseline_predictions
    )
    print("   ✓ Drift detector initialized")
    
    # Initialize fairness monitor
    fairness_monitor = FairnessMonitor(
        baseline_fairness={
            "Sex": baseline_metrics['fairness'],
            "Age_group": baseline_metrics['fairness_age']
        },
        sensitive_attributes=["Sex", "Age_group"]
    )
    print("   ✓ Fairness monitor initialized")
    
    # Initialize alert system
    alert_system = AlertSystem(alerts_dir=ALERTS_DIR)
    print("   ✓ Alert system initialized")
    
    print()
    
    # ========================================================================
    # STEP 3: CREATE BATCHES
    # ========================================================================
    print("[3/8] Creating batches from test data...")
    
    batches = create_batches_from_data(
        data_path="artifacts/splits/X_test.parquet",
        labels_path="artifacts/splits/y_test.parquet",
        batch_size=BATCH_CONFIG["batch_size"],
        num_batches=BATCH_CONFIG["num_batches"]
    )
    
    print(f"   ✓ Created {len(batches)} batches")
    print(f"   ✓ Batch size: {BATCH_CONFIG['batch_size']} samples")
    print()
    
    # ========================================================================
    # STEP 4: PROCESS BATCHES WITH MONITORING
    # ========================================================================
    print("[4/8] Processing batches and monitoring...")
    print()
    
    monitoring_results = {
        "batches": [],
        "drift_detected_count": 0,
        "fairness_violations_count": 0,
        "performance_alerts_count": 0
    }
    
    for idx, (batch_X, batch_y, batch_id) in enumerate(batches):
        print(f"   Processing {batch_id} ({idx+1}/{len(batches)})...")
        
        # ----------------------------------------------------------------
        # 4.1: Process Batch (Predictions + Performance)
        # ----------------------------------------------------------------
        batch_report = batch_processor.process_batch(
            batch_data=batch_X,
            batch_labels=batch_y,
            batch_id=batch_id,
            log_dir=BATCH_LOGS_DIR
        )
        
        print(f"      ✓ Predictions: {batch_report['predictions']['approved']} approved, "
              f"{batch_report['predictions']['rejected']} rejected")
        print(f"      ✓ Performance: AUC={batch_report['performance']['auc_roc']:.4f}, "
              f"Acc={batch_report['performance']['accuracy']:.4f}")
        
        # Get predictions for monitoring
        batch_predictions = batch_processor.model.predict_proba(
            batch_X[batch_processor.features]
        )[:, 1]
        batch_pred_labels = (batch_predictions >= batch_processor.threshold).astype(int)
        
        # ----------------------------------------------------------------
        # 4.2: Drift Detection
        # ----------------------------------------------------------------
        drift_results = drift_detector.detect_all_features_drift(
            current_data=batch_X,
            feature_config=FEATURE_DRIFT_CONFIG,
            threshold=DRIFT_THRESHOLDS["ks_test"]
        )
        
        # Prediction drift
        pred_drift = drift_detector.detect_prediction_drift(
            current_predictions=batch_predictions,
            threshold=DRIFT_THRESHOLDS["kl_divergence"]
        )
        drift_results["prediction_drift"] = pred_drift
        
        drift_count = drift_results["summary"]["features_with_drift"]
        pred_drift_detected = pred_drift.get("drift_detected", False)
        
        if drift_count > 0 or pred_drift_detected:
            print(f"      ⚠️  Drift: {drift_count} features, "
                  f"Prediction drift: {pred_drift_detected}")
            monitoring_results["drift_detected_count"] += 1
        else:
            print(f"      ✓ No drift detected")
        
        # Save drift report
        drift_report_path = DRIFT_REPORTS_DIR / f"drift_{batch_id}.json"
        with open(drift_report_path, 'w') as f:
            json.dump(convert_numpy_types(drift_results), f, indent=2)
        
        # ----------------------------------------------------------------
        # 4.3: Fairness Monitoring
        # ----------------------------------------------------------------
        fairness_report = fairness_monitor.monitor_batch(
            batch_data=batch_X,
            predictions=batch_pred_labels,
            true_labels=batch_y.values,
            thresholds=FAIRNESS_THRESHOLDS,
            batch_id=batch_id
        )
        
        violation_count = fairness_report["summary"]["total_violations"]
        critical_count = fairness_report["summary"]["critical_violations"]
        
        if violation_count > 0:
            print(f"      ⚠️  Fairness: {violation_count} violations "
                  f"({critical_count} critical)")
            monitoring_results["fairness_violations_count"] += 1
        else:
            print(f"      ✓ No fairness violations")
        
        # Save fairness report
        fairness_report_path = FAIRNESS_REPORTS_DIR / f"fairness_{batch_id}.json"
        with open(fairness_report_path, 'w') as f:
            json.dump(fairness_report, f, indent=2)
        
        # ----------------------------------------------------------------
        # 4.4: Performance Comparison
        # ----------------------------------------------------------------
        perf_comparison = batch_processor.compare_to_baseline(
            baseline_metrics=baseline_metrics['performance']
        )
        
        if perf_comparison.get("degradation_detected", False):
            print(f"      ⚠️  Performance degradation detected")
            monitoring_results["performance_alerts_count"] += 1
        
        # ----------------------------------------------------------------
        # 4.5: Generate Alerts
        # ----------------------------------------------------------------
        # Drift alerts
        drift_alerts = alert_system.generate_drift_alerts(drift_results, batch_id)
        
        # Fairness alerts
        fairness_alerts = alert_system.generate_fairness_alerts(
            fairness_report["violations"], batch_id
        )
        
        # Performance alerts
        perf_alerts = alert_system.generate_performance_alerts(
            perf_comparison, batch_id
        )
        
        total_alerts = len(drift_alerts) + len(fairness_alerts) + len(perf_alerts)
        if total_alerts > 0:
            print(f"      📢 Generated {total_alerts} alerts")
            alert_system.save_alerts(batch_id=batch_id)
        
        # Store batch results
        monitoring_results["batches"].append({
            "batch_id": batch_id,
            "performance": batch_report["performance"],
            "drift_summary": drift_results["summary"],
            "fairness_summary": fairness_report["summary"],
            "alerts_generated": total_alerts
        })
        
        print()
    
    # ========================================================================
    # STEP 5: SAVE MONITORING HISTORIES
    # ========================================================================
    print("[5/8] Saving monitoring histories...")
    
    # Save batch processing history
    batch_processor.save_history(MONITORING_DIR / "batch_processing_history.json")
    print("   ✓ Batch processing history saved")
    
    # Save fairness monitoring history
    fairness_monitor.save_history(MONITORING_DIR / "fairness_monitoring_history.json")
    print("   ✓ Fairness monitoring history saved")
    
    # Save all alerts
    alert_system.save_alerts()
    print("   ✓ Alerts saved")
    
    print()
    
    # ========================================================================
    # STEP 6: GENERATE SUMMARY REPORT
    # ========================================================================
    print("[6/8] Generating summary report...")
    
    summary_report = {
        "generated_at": datetime.now().isoformat(),
        "monitoring_config": {
            "num_batches": BATCH_CONFIG["num_batches"],
            "batch_size": BATCH_CONFIG["batch_size"],
            "drift_threshold": DRIFT_THRESHOLDS["ks_test"],
            "fairness_thresholds": FAIRNESS_THRESHOLDS
        },
        "baseline_performance": baseline_metrics['performance'],
        "monitoring_summary": {
            "total_batches_processed": len(batches),
            "batches_with_drift": monitoring_results["drift_detected_count"],
            "batches_with_fairness_violations": monitoring_results["fairness_violations_count"],
            "batches_with_performance_degradation": monitoring_results["performance_alerts_count"]
        },
        "batch_processing_summary": batch_processor.get_summary_statistics(),
        "fairness_summary": fairness_monitor.get_summary_report(),
        "alert_summary": alert_system.get_alert_summary(),
        "batches": monitoring_results["batches"]
    }
    
    summary_path = MONITORING_DIR / "monitoring_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"   ✓ Summary report saved: {summary_path}")
    print()
    
    # ========================================================================
    # STEP 7: PRINT RESULTS
    # ========================================================================
    print("[7/8] Monitoring Results:")
    print("-" * 80)
    
    print(f"\n📊 Batch Processing:")
    print(f"   Total Batches: {len(batches)}")
    print(f"   Total Applications: {summary_report['batch_processing_summary']['total_applications']}")
    print(f"   Overall Approval Rate: {summary_report['batch_processing_summary']['overall_approval_rate']:.2%}")
    print(f"   Average AUC-ROC: {summary_report['batch_processing_summary']['average_metrics']['auc_roc']:.4f}")
    print(f"   Average Accuracy: {summary_report['batch_processing_summary']['average_metrics']['accuracy']:.4f}")
    
    print(f"\n🔍 Drift Detection:")
    print(f"   Batches with Drift: {monitoring_results['drift_detected_count']}/{len(batches)}")
    
    print(f"\n⚖️  Fairness Monitoring:")
    print(f"   Batches with Violations: {monitoring_results['fairness_violations_count']}/{len(batches)}")
    print(f"   Total Violations: {summary_report['fairness_summary']['total_violations']}")
    print(f"   Critical Violations: {summary_report['fairness_summary']['critical_violations']}")
    
    # Print alert summary
    alert_system.print_alert_summary()
    
    # ========================================================================
    # STEP 8: VALIDATION
    # ========================================================================
    print("\n[8/8] Part 2 Deliverables Validation:")
    print("-" * 80)
    
    deliverables = [
        ("Batch processing logs", BATCH_LOGS_DIR, len(list(BATCH_LOGS_DIR.glob("*.json"))) > 0),
        ("Drift detection reports", DRIFT_REPORTS_DIR, len(list(DRIFT_REPORTS_DIR.glob("*.json"))) > 0),
        ("Fairness monitoring reports", FAIRNESS_REPORTS_DIR, len(list(FAIRNESS_REPORTS_DIR.glob("*.json"))) > 0),
        ("Alert files", ALERTS_DIR, len(list(ALERTS_DIR.glob("*.json"))) > 0),
        ("Monitoring summary", summary_path, summary_path.exists()),
        ("Batch history", MONITORING_DIR / "batch_processing_history.json", (MONITORING_DIR / "batch_processing_history.json").exists()),
        ("Fairness history", MONITORING_DIR / "fairness_monitoring_history.json", (MONITORING_DIR / "fairness_monitoring_history.json").exists()),
    ]
    
    all_pass = True
    for name, path, exists in deliverables:
        status = "✅ PASS" if exists else "❌ FAIL"
        print(f"   {status} {name}")
        if not exists:
            all_pass = False
    
    print()
    print("="*80)
    
    if all_pass:
        print("✅ SUCCESS: Part 2 monitoring complete!")
        print()
        print("All monitoring artifacts generated:")
        print(f"  → {BATCH_LOGS_DIR}")
        print(f"  → {DRIFT_REPORTS_DIR}")
        print(f"  → {FAIRNESS_REPORTS_DIR}")
        print(f"  → {ALERTS_DIR}")
        print(f"  → {summary_path}")
        print()
        print("Next step: Part 3 - Real-Time Dashboard")
    else:
        print("⚠️  WARNING: Some deliverables are missing")
        print("Please check the error messages above")
    
    print("="*80)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)