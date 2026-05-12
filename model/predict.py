import pickle
import pandas as pd
from .model_config import MODEL_FEATURES, DECISION_THRESHOLD
from preprocessing.config import MODEL_PATH

_model_cache = None  

def load_model():
    """
    Load model from pickle file.
    Now handles both old format (just model) and new format (dict with model, features, threshold).
    """
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            loaded = pickle.load(f)
            
            # Handle new format (Part 1 compliant)
            if isinstance(loaded, dict):
                _model_cache = loaded
            else:
                # Handle old format (backward compatibility)
                _model_cache = {
                    'model': loaded,
                    'features': MODEL_FEATURES,
                    'threshold': DECISION_THRESHOLD
                }
    return _model_cache

def predict_batch(X_batch: pd.DataFrame):
    """
    Predict on a batch of data using the trained model.
    
    Args:
        X_batch: DataFrame with features (including sensitive attributes if present)
    
    Returns:
        y_pred: Binary predictions (0=bad, 1=good)
        y_prob: Probability of being good credit risk
    """
    model_data = load_model()
    model = model_data['model']
    features = model_data['features']
    threshold = model_data.get('threshold', DECISION_THRESHOLD)
    
    # Use only the features the model was trained on
    y_prob = model.predict_proba(X_batch[features])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob