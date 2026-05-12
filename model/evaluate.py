import pandas as pd, pickle
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing.config import SPLITS_DIR, MODEL_PATH, SENSITIVE_ATTRS
from .model_config import MODEL_FEATURES, DECISION_THRESHOLD

def evaluate_model(split="val"):
    """
    Evaluate model on validation or test set.
    Now uses threshold and features from the saved model data.
    """
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)
    
    # Handle new format (Part 1 compliant)
    if isinstance(loaded, dict):
        model = loaded['model']
        features = loaded['features']
        threshold = loaded.get('threshold', DECISION_THRESHOLD)
    else:
        # Handle old format (backward compatibility)
        model = loaded
        features = [c for c in MODEL_FEATURES if c in X.columns]
        threshold = DECISION_THRESHOLD

    X = pd.read_parquet(f"{SPLITS_DIR}X_{split}.parquet")
    y = pd.read_parquet(f"{SPLITS_DIR}y_{split}.parquet").squeeze()

    y_prob = model.predict_proba(X[features])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*70}")
    print(f"Evaluation on {split.upper()} set (threshold={threshold:.2f})")
    print(f"{'='*70}")
    print(classification_report(y, y_pred, target_names=["bad", "good"], digits=3))
    auc = roc_auc_score(y, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")
    print(f"Approval Rate: {y_pred.mean():.2%}")
    print(f"{'='*70}\n")
    
    return y_pred, y_prob

if __name__ == "__main__":
    evaluate_model()