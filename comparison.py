import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing.config import (MODEL_FEATURES, NUMERICAL_FEATURES,
                                  CATEGORICAL_FEATURES, MODEL_PATH,
                                  SPLITS_DIR)

OUTPUT_CSV = Path("artifacts/model_comparison.csv")
THRESHOLD = 0.35


def load_artifact_model():
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict):
        model = loaded["model"]
        features = loaded.get("features", MODEL_FEATURES)
        threshold = loaded.get("threshold", THRESHOLD)
    else:
        model = loaded
        features = MODEL_FEATURES
        threshold = THRESHOLD

    return model, list(features), threshold


def load_splits():
    X_train = pd.read_parquet(f"{SPLITS_DIR}X_train.parquet")
    y_train = pd.read_parquet(f"{SPLITS_DIR}y_train.parquet").squeeze()
    X_test = pd.read_parquet(f"{SPLITS_DIR}X_test.parquet")
    y_test = pd.read_parquet(f"{SPLITS_DIR}y_test.parquet").squeeze()
    return X_train, y_train, X_test, y_test


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.cluster_to_class_ = {}
        self.default_class_ = 0

    def fit(self, X, y=None):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = self.kmeans_.fit_predict(X_arr)

        if y is not None:
            y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            for cluster in range(self.n_clusters):
                cluster_y = y_arr[labels == cluster]
                if cluster_y.size > 0:
                    self.cluster_to_class_[cluster] = int(np.bincount(cluster_y).argmax())
                else:
                    self.cluster_to_class_[cluster] = 0
            self.default_class_ = int(np.bincount(y_arr).argmax())

        return self

    def predict(self, X):
        if self.kmeans_ is None:
            raise ValueError("KMeansClassifier must be fitted before calling predict.")

        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        cluster_ids = self.kmeans_.predict(X_arr)
        return np.array([self.cluster_to_class_.get(cluster, self.default_class_) for cluster in cluster_ids], dtype=int)


def build_pipeline(clf, numeric_features):
    scaler = ColumnTransformer(
        transformers=[
            ("scale_numeric", StandardScaler(), numeric_features),
        ],
        remainder="passthrough",
    )
    return Pipeline([("scaler", scaler), ("clf", clf)])


def eval_model(model, X, y, threshold=0.5):
    y_prob = None

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
        try:
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        except Exception:
            y_prob = None

    y_pred = model.predict(X)

    if threshold is not None and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4) if y_prob is not None else None,
    }
    return metrics


def get_baseline_classifiers():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "K-Means": KMeansClassifier(n_clusters=2, random_state=42),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, random_state=42),
    }


def build_comparison_table():
    X_train, y_train, X_test, y_test = load_splits()
    artifact_model, artifact_features, threshold = load_artifact_model()

    # Use only the matching feature subset for all models.
    feature_columns = [c for c in artifact_features if c in X_train.columns]
    if not feature_columns:
        raise ValueError("No matching features found between the artifact model and split data.")

    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    numeric_columns = [col for col in NUMERICAL_FEATURES if col in feature_columns]

    rows = []

    # Evaluate the saved project model first.
    if hasattr(artifact_model, "predict_proba") or hasattr(artifact_model, "decision_function"):
        try:
            metrics = eval_model(artifact_model, X_test, y_test, threshold=threshold)
            rows.append({"Method": "Fairflow", **metrics})
        except Exception as exc:
            raise RuntimeError(f"Failed to evaluate saved model: {exc}")
    else:
        raise RuntimeError("Saved model does not expose predict_proba or decision_function.")

    # Train standard comparators on the same train/test split.
    for name, clf in get_baseline_classifiers().items():
        pipeline = build_pipeline(clf, numeric_columns)
        pipeline.fit(X_train, y_train)
        metrics = eval_model(pipeline, X_test, y_test, threshold=0.5)
        rows.append({"Method": name, **metrics})

    comparison_df = pd.DataFrame(rows)
    return comparison_df


def main():
    comparison_df = build_comparison_table()
    print("\nModel comparison on test set:\n")
    print(comparison_df.to_string(index=False))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved comparison table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
