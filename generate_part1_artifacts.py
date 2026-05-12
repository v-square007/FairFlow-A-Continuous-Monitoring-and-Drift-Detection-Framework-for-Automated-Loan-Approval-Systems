#!/usr/bin/env python3
"""
PART 1: Generate Baseline Artifacts & Fairness Audit
=====================================================
This script generates all required artifacts for Part 1 deliverables:
1. Model metadata
2. Initial fairness audit report
3. Feature names
4. Baseline performance metrics
5. Evaluation results (confusion matrix, ROC curve data, feature importance)

Requirements from README Section "Part 1 - Model Training & Baseline Setup":
- Fairness audit with demographic parity, disparate impact, equal opportunity
- Model metadata with hyperparameters and performance
- Evaluation reports (CSV/JSON)
- All artifacts saved in structured directory

Usage:
    python generate_part1_artifacts.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Configuration
SPLITS_DIR = "artifacts/splits/"
MODEL_PATH = "artifacts/model.pkl"
BASELINE_DIR = "artifacts/baseline/"
EVALUATION_DIR = "artifacts/baseline/evaluation/"

# Create directories
Path(BASELINE_DIR).mkdir(parents=True, exist_ok=True)
Path(EVALUATION_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("PART 1: GENERATING BASELINE ARTIFACTS & FAIRNESS AUDIT")
print("="*80)
print()

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("[1/7] Loading model and validation data...")

with open(MODEL_PATH, "rb") as f:
    loaded = pickle.load(f)

# Handle both old and new model formats
if isinstance(loaded, dict):
    model = loaded['model']
    features = loaded['features']
    threshold = loaded.get('threshold', 0.35)
    best_params = loaded.get('best_params', {})
    cv_score = loaded.get('cv_score', None)
    val_score = loaded.get('val_score', None)
    fairness_weighting = loaded.get('fairness_weighting', False)
else:
    model = loaded
    # Fallback to all numeric columns if old format
    X_sample = pd.read_parquet(f"{SPLITS_DIR}X_val.parquet", nrows=1)
    features = [col for col in X_sample.columns 
                if col not in ['Sex', 'Age_group', 'Sex_original', 'Age_original']]
    threshold = 0.35
    best_params = {}
    cv_score = None
    val_score = None
    fairness_weighting = False

# Load splits
X_val = pd.read_parquet(f"{SPLITS_DIR}X_val.parquet")
y_val = pd.read_parquet(f"{SPLITS_DIR}y_val.parquet").squeeze()
X_test = pd.read_parquet(f"{SPLITS_DIR}X_test.parquet")
y_test = pd.read_parquet(f"{SPLITS_DIR}y_test.parquet").squeeze()

print(f"   ✓ Model loaded")
print(f"   ✓ Features: {len(features)}")
print(f"   ✓ Threshold: {threshold:.2f}")
print(f"   ✓ Validation samples: {len(X_val)}")
print(f"   ✓ Test samples: {len(X_test)}")
print()

# ============================================================================
# 2. GENERATE PREDICTIONS
# ============================================================================
print("[2/7] Generating predictions on validation and test sets...")

# Validation predictions
y_val_prob = model.predict_proba(X_val[features])[:, 1]
y_val_pred = (y_val_prob >= threshold).astype(int)

# Test predictions (for final evaluation only)
y_test_prob = model.predict_proba(X_test[features])[:, 1]
y_test_pred = (y_test_prob >= threshold).astype(int)

print(f"   ✓ Validation predictions generated")
print(f"   ✓ Test predictions generated")
print()

# ============================================================================
# 3. SAVE FEATURE NAMES (Part 1 Deliverable)
# ============================================================================
print("[3/7] Saving feature names...")

with open("artifacts/feature_names.json", "w") as f:
    json.dump(features, f, indent=2)

print(f"   ✓ Saved: artifacts/feature_names.json")
print()

# ============================================================================
# 4. GENERATE FAIRNESS AUDIT REPORT (Part 1 Deliverable #7)
# ============================================================================
print("[4/7] Conducting initial fairness audit...")

# Create audit structure
audit = {
    "generated_at": datetime.now().isoformat(),
    "model_version": "baseline_v1",
    "split": "validation",
    "threshold": threshold,
    "by_sex": {},
    "by_age_group": {}
}

pred_series = pd.Series(y_val_pred, index=X_val.index)

# Fairness metrics by Sex
if "Sex" in X_val.columns:
    sex_rates = {}
    sex_tpr = {}
    
    for grp, idx in X_val.groupby("Sex").groups.items():
        # Approval rate (demographic parity)
        approval_rate = float(pred_series.loc[idx].mean())
        sex_rates[str(grp)] = approval_rate
        
        # True Positive Rate (equal opportunity)
        mask_positive = (y_val.loc[idx] == 1)
        if mask_positive.sum() > 0:
            tpr = float(pred_series.loc[idx][mask_positive].mean())
            sex_tpr[str(grp)] = tpr
    
    rates = list(sex_rates.values())
    tpr_vals = [v for v in sex_tpr.values() if v is not None]
    
    audit["by_sex"] = {
        "approval_rates": sex_rates,
        "demographic_parity_difference": round(max(rates) - min(rates), 4),
        "disparate_impact_ratio": round(min(rates) / max(rates), 4),
        "demographic_parity_violation": (max(rates) - min(rates)) > 0.1,
        "disparate_impact_violation": (min(rates) / max(rates)) < 0.8,
        "true_positive_rates": sex_tpr,
        "equal_opportunity_difference": round(max(tpr_vals) - min(tpr_vals), 4) if tpr_vals else None,
        "equal_opportunity_violation": (max(tpr_vals) - min(tpr_vals)) > 0.1 if tpr_vals else None
    }

# Fairness metrics by Age Group
if "Age_group" in X_val.columns:
    age_rates = {}
    age_tpr = {}
    
    for grp, idx in X_val.groupby("Age_group").groups.items():
        approval_rate = float(pred_series.loc[idx].mean())
        age_rates[str(grp)] = approval_rate
        
        mask_positive = (y_val.loc[idx] == 1)
        if mask_positive.sum() > 0:
            tpr = float(pred_series.loc[idx][mask_positive].mean())
            age_tpr[str(grp)] = tpr
    
    rates = list(age_rates.values())
    tpr_vals = [v for v in age_tpr.values() if v is not None]
    
    audit["by_age_group"] = {
        "approval_rates": age_rates,
        "demographic_parity_difference": round(max(rates) - min(rates), 4),
        "disparate_impact_ratio": round(min(rates) / max(rates), 4),
        "demographic_parity_violation": (max(rates) - min(rates)) > 0.1,
        "disparate_impact_violation": (min(rates) / max(rates)) < 0.8,
        "true_positive_rates": age_tpr if age_tpr else None,
        "equal_opportunity_difference": round(max(tpr_vals) - min(tpr_vals), 4) if tpr_vals else None,
        "equal_opportunity_violation": (max(tpr_vals) - min(tpr_vals)) > 0.1 if tpr_vals else None
    }

# Save fairness audit
with open(f"{BASELINE_DIR}fairness_audit_report.json", "w") as f:
    json.dump(audit, f, indent=2)

print(f"   ✓ Saved: {BASELINE_DIR}fairness_audit_report.json")
print(f"   ✓ Demographic Parity (Sex): {audit['by_sex']['demographic_parity_difference']:.4f}")
print(f"   ✓ Disparate Impact (Sex): {audit['by_sex']['disparate_impact_ratio']:.4f}")
print()

# ============================================================================
# 5. GENERATE MODEL METADATA (Part 1 Deliverable #5.B)
# ============================================================================
print("[5/7] Generating model metadata...")

# Classification report for validation set
report_dict = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)

# Test set metrics (for final reporting)
test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_prob)

metadata = {
    "model_version": "baseline_v1",
    "model_file": "artifacts/model.pkl",
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "decision_threshold": threshold,
    "fairness_weighting_enabled": fairness_weighting,
    
    "hyperparameters": {
    "n_estimators": int(model.n_estimators),
    "max_depth": int(model.max_depth),
    "learning_rate": float(model.learning_rate),
    "scale_pos_weight": float(model.scale_pos_weight),
    "subsample": float(model.subsample) if hasattr(model, 'subsample') and model.subsample is not None else None,
    "colsample_bytree": float(model.colsample_bytree) if hasattr(model, 'colsample_bytree') and model.colsample_bytree is not None else None,
    },
    
    "best_params_from_cv": best_params if best_params else "Not tuned (using defaults)",
    
    "training_info": {
        "cv_auc_score": round(cv_score, 4) if cv_score else "Not available",
        "validation_auc_score": round(val_score, 4) if val_score else round(roc_auc_score(y_val, y_val_prob), 4),
        "hyperparameter_tuning_method": "5-fold GridSearchCV" if best_params else "None",
        "training_samples": 700,
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
    },
    
    "performance_metrics": {
        "validation": {
            "auc_roc": round(roc_auc_score(y_val, y_val_prob), 4),
            "accuracy": round(report_dict["accuracy"], 4),
            "bad_precision": round(report_dict["0"]["precision"], 4),
            "bad_recall": round(report_dict["0"]["recall"], 4),
            "bad_f1": round(report_dict["0"]["f1-score"], 4),
            "good_precision": round(report_dict["1"]["precision"], 4),
            "good_recall": round(report_dict["1"]["recall"], 4),
            "good_f1": round(report_dict["1"]["f1-score"], 4),
        },
        "test": {
            "auc_roc": round(test_auc, 4),
            "accuracy": round(test_report["accuracy"], 4),
            "bad_precision": round(test_report["0"]["precision"], 4),
            "bad_recall": round(test_report["0"]["recall"], 4),
            "bad_f1": round(test_report["0"]["f1-score"], 4),
            "good_precision": round(test_report["1"]["precision"], 4),
            "good_recall": round(test_report["1"]["recall"], 4),
            "good_f1": round(test_report["1"]["f1-score"], 4),
        }
    },
    
    "fairness_metrics": {
        "sex": {
            "demographic_parity": audit["by_sex"]["demographic_parity_difference"],
            "disparate_impact": audit["by_sex"]["disparate_impact_ratio"],
            "equal_opportunity": audit["by_sex"]["equal_opportunity_difference"],
        },
        "age_group": {
            "demographic_parity": audit["by_age_group"]["demographic_parity_difference"],
            "disparate_impact": audit["by_age_group"]["disparate_impact_ratio"],
        }
    },
    
    "features": {
        "count": len(features),
        "names": features,
        "sensitive_attributes": ["Sex", "Age_group"]
    }
}

# Save metadata
with open(f"{BASELINE_DIR}metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Saved: {BASELINE_DIR}metadata.json")
print()

# ============================================================================
# 6. GENERATE EVALUATION REPORTS (Part 1 Deliverable #5.C)
# ============================================================================
print("[6/7] Generating evaluation reports...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual_Bad', 'Actual_Good'],
    columns=['Predicted_Bad', 'Predicted_Good']
)
cm_df.to_csv(f"{EVALUATION_DIR}confusion_matrix.csv")
print(f"   ✓ Saved: {EVALUATION_DIR}confusion_matrix.csv")

# ROC Curve Data
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_prob)
roc_df = pd.DataFrame({
    'false_positive_rate': fpr,
    'true_positive_rate': tpr,
    'threshold': roc_thresholds
})
roc_df.to_csv(f"{EVALUATION_DIR}roc_curve_data.csv", index=False)
print(f"   ✓ Saved: {EVALUATION_DIR}roc_curve_data.csv")

# Precision-Recall Curve Data
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
pr_df = pd.DataFrame({
    'precision': precision[:-1],  # Last element is padding
    'recall': recall[:-1],
    'threshold': pr_thresholds
})
pr_df.to_csv(f"{EVALUATION_DIR}precision_recall_curve.csv", index=False)
print(f"   ✓ Saved: {EVALUATION_DIR}precision_recall_curve.csv")

# Feature Importance
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(f"{EVALUATION_DIR}feature_importance.csv", index=False)
    print(f"   ✓ Saved: {EVALUATION_DIR}feature_importance.csv")

print()

# ============================================================================
# 7. PRINT SUMMARY & SUCCESS CRITERIA CHECK
# ============================================================================
print("[7/7] Checking Part 1 success criteria...")
print()

print("="*80)
print("PART 1 SUCCESS CRITERIA VALIDATION")
print("="*80)

# Check success criteria from README
test_auc = metadata["performance_metrics"]["test"]["auc_roc"]
demo_parity = audit["by_sex"]["demographic_parity_difference"]
disp_impact = audit["by_sex"]["disparate_impact_ratio"]

criteria = [
    ("AUC-ROC > 0.75 on test set", test_auc > 0.75, f"{test_auc:.4f}"),
    ("Demographic parity < 0.1", demo_parity < 0.1, f"{demo_parity:.4f}"),
    ("Disparate impact > 0.8", disp_impact > 0.8, f"{disp_impact:.4f}"),
    ("Model saved and loadable", Path(MODEL_PATH).exists(), "Yes"),
    ("Feature names saved", Path("artifacts/feature_names.json").exists(), "Yes"),
    ("Metadata documented", Path(f"{BASELINE_DIR}metadata.json").exists(), "Yes"),
    ("Fairness audit completed", Path(f"{BASELINE_DIR}fairness_audit_report.json").exists(), "Yes"),
]

all_pass = True
for criterion, passed, value in criteria:
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"{criterion:<40} {status:>10}  ({value})")
    if not passed:
        all_pass = False

print("="*80)
print()

if all_pass:
    print("✅ SUCCESS: All Part 1 deliverables complete and criteria met!")
    print()
    print("Part 1 artifacts ready for:")
    print("  → Part 2: Drift & Fairness Monitoring")
    print("  → Part 3: Real-Time Dashboard")
else:
    print("⚠️  WARNING: Some success criteria not met.")
    print()
    print("Review the failures above. Common issues:")
    print("  - AUC-ROC < 0.75: Retrain with better hyperparameters")
    print("  - Fairness violations: Increase sample weights for underrepresented groups")

print()
print("="*80)
print("PART 1 ARTIFACTS GENERATED SUCCESSFULLY")
print("="*80)
print()
print("Generated artifacts:")
print(f"  ✓ {MODEL_PATH}")
print(f"  ✓ artifacts/feature_names.json")
print(f"  ✓ {BASELINE_DIR}metadata.json")
print(f"  ✓ {BASELINE_DIR}fairness_audit_report.json")
print(f"  ✓ {EVALUATION_DIR}confusion_matrix.csv")
print(f"  ✓ {EVALUATION_DIR}roc_curve_data.csv")
print(f"  ✓ {EVALUATION_DIR}precision_recall_curve.csv")
print(f"  ✓ {EVALUATION_DIR}feature_importance.csv")
print()