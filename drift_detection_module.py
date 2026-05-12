"""
Drift Detection Module
======================
Implements statistical tests to detect data drift:
- KS Test: For continuous features
- Chi-Square Test: For categorical features
- KL Divergence: For prediction distributions
- Prediction Shift Detection: For model output drift
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """
    Detects distribution drift between baseline and current data.
    """
    
    def __init__(self, baseline_data: pd.DataFrame, baseline_predictions: np.ndarray = None):
        """
        Initialize drift detector with baseline data.
        
        Args:
            baseline_data: Reference dataset (from training/validation)
            baseline_predictions: Reference predictions (optional)
        """
        self.baseline_data = baseline_data
        self.baseline_predictions = baseline_predictions
        
        # Store baseline statistics for continuous features
        self.baseline_stats = {}
        for col in baseline_data.select_dtypes(include=[np.number]).columns:
            self.baseline_stats[col] = {
                'mean': baseline_data[col].mean(),
                'std': baseline_data[col].std(),
                'min': baseline_data[col].min(),
                'max': baseline_data[col].max(),
                'quantiles': baseline_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Store baseline distributions for categorical features
        self.baseline_categorical = {}
        for col in baseline_data.select_dtypes(exclude=[np.number]).columns:
            self.baseline_categorical[col] = baseline_data[col].value_counts(normalize=True).to_dict()
    
    
    def detect_feature_drift(self, current_data: pd.DataFrame, 
                           feature: str, 
                           method: str = "auto",
                           threshold: float = 0.05) -> Dict:
        """
        Detect drift in a single feature.
        
        Args:
            current_data: New data to compare against baseline
            feature: Feature name to check
            method: "ks_test", "chi_square", or "auto"
            threshold: P-value threshold for significance
        
        Returns:
            Dictionary with drift detection results
        """
        if feature not in current_data.columns:
            return {
                "feature": feature,
                "drift_detected": False,
                "error": f"Feature {feature} not found in current data"
            }
        
        if feature not in self.baseline_data.columns:
            return {
                "feature": feature,
                "drift_detected": False,
                "error": f"Feature {feature} not found in baseline data"
            }
        
        # Auto-detect method based on feature type
        if method == "auto":
            if pd.api.types.is_numeric_dtype(current_data[feature]):
                method = "ks_test"
            else:
                method = "chi_square"
        
        # Apply appropriate test
        if method == "ks_test":
            return self._ks_test_drift(feature, current_data[feature], threshold)
        elif method == "chi_square":
            return self._chi_square_drift(feature, current_data[feature], threshold)
        else:
            return {
                "feature": feature,
                "drift_detected": False,
                "error": f"Unknown method: {method}"
            }
    
    
    def _ks_test_drift(self, feature: str, current_values: pd.Series, 
                      threshold: float) -> Dict:
        """
        Kolmogorov-Smirnov test for continuous feature drift.
        Tests if current distribution differs from baseline.
        """
        baseline_values = self.baseline_data[feature].dropna()
        current_values = current_values.dropna()
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(baseline_values, current_values)
        
        # Drift detected if p-value < threshold (reject null hypothesis of same distribution)
        drift_detected = p_value < threshold
        
        # Calculate additional metrics
        mean_shift = current_values.mean() - baseline_values.mean()
        std_shift = current_values.std() - baseline_values.std()
        
        return {
            "feature": feature,
            "method": "ks_test",
            "drift_detected": drift_detected,
            "p_value": round(p_value, 4),
            "ks_statistic": round(statistic, 4),
            "threshold": threshold,
            "baseline_mean": round(baseline_values.mean(), 4),
            "current_mean": round(current_values.mean(), 4),
            "mean_shift": round(mean_shift, 4),
            "baseline_std": round(baseline_values.std(), 4),
            "current_std": round(current_values.std(), 4),
            "std_shift": round(std_shift, 4),
            "interpretation": "Distribution differs significantly" if drift_detected else "No significant drift"
        }
    
    
    def _chi_square_drift(self, feature: str, current_values: pd.Series,
                         threshold: float) -> Dict:
        """
        Chi-square test for categorical feature drift.
        Tests if category proportions differ from baseline.
        """
        baseline_dist = self.baseline_data[feature].value_counts()
        current_dist = current_values.value_counts()
        
        # Align categories (handle new/missing categories)
        all_categories = set(baseline_dist.index) | set(current_dist.index)
        baseline_counts = [baseline_dist.get(cat, 0) for cat in all_categories]
        current_counts = [current_dist.get(cat, 0) for cat in all_categories]
        
        # Perform chi-square test
        if sum(current_counts) == 0:
            return {
                "feature": feature,
                "method": "chi_square",
                "drift_detected": False,
                "error": "No data in current sample"
            }
        
        # Expected counts based on baseline proportions
        total_current = sum(current_counts)
        expected_counts = [count / sum(baseline_counts) * total_current 
                          for count in baseline_counts]
        
        # Chi-square test
        statistic, p_value = stats.chisquare(current_counts, expected_counts)
        
        drift_detected = p_value < threshold
        
        # Find categories with largest shifts
        baseline_props = {cat: baseline_dist.get(cat, 0) / len(self.baseline_data) 
                         for cat in all_categories}
        current_props = {cat: current_dist.get(cat, 0) / len(current_values) 
                        for cat in all_categories}
        
        prop_shifts = {cat: current_props[cat] - baseline_props[cat] 
                      for cat in all_categories}
        
        # Top shifted categories
        top_shifts = sorted(prop_shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        return {
            "feature": feature,
            "method": "chi_square",
            "drift_detected": drift_detected,
            "p_value": round(p_value, 4),
            "chi2_statistic": round(statistic, 4),
            "threshold": threshold,
            "num_categories": len(all_categories),
            "top_shifted_categories": [
                {"category": cat, "shift": round(shift, 4)} 
                for cat, shift in top_shifts
            ],
            "interpretation": "Category distribution differs significantly" if drift_detected else "No significant drift"
        }
    
    
    def detect_prediction_drift(self, current_predictions: np.ndarray,
                               threshold: float = 0.1) -> Dict:
        """
        Detect drift in model predictions using KL divergence.
        
        Args:
            current_predictions: Current batch predictions (probabilities)
            threshold: KL divergence threshold
        
        Returns:
            Dictionary with prediction drift results
        """
        if self.baseline_predictions is None:
            return {
                "drift_detected": False,
                "error": "No baseline predictions available"
            }
        
        # Create histogram bins
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        
        baseline_hist, _ = np.histogram(self.baseline_predictions, bins=bins, density=True)
        current_hist, _ = np.histogram(current_predictions, bins=bins, density=True)
        
        # Normalize to probability distributions
        baseline_dist = baseline_hist / baseline_hist.sum()
        current_dist = current_hist / current_hist.sum()
        
        # Calculate KL divergence (add small epsilon to avoid log(0))
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon
        
        kl_divergence = stats.entropy(current_dist, baseline_dist)
        
        # Also calculate Jensen-Shannon distance (symmetric)
        js_distance = jensenshannon(baseline_dist, current_dist)
        
        # Mean shift in predictions
        mean_shift = current_predictions.mean() - self.baseline_predictions.mean()
        
        drift_detected = kl_divergence > threshold
        
        return {
            "method": "kl_divergence",
            "drift_detected": drift_detected,
            "kl_divergence": round(kl_divergence, 4),
            "js_distance": round(js_distance, 4),
            "threshold": threshold,
            "baseline_mean_prob": round(self.baseline_predictions.mean(), 4),
            "current_mean_prob": round(current_predictions.mean(), 4),
            "mean_shift": round(mean_shift, 4),
            "interpretation": "Prediction distribution shifted significantly" if drift_detected else "No significant prediction drift"
        }
    
    
    def detect_all_features_drift(self, current_data: pd.DataFrame,
                                  feature_config: Dict = None,
                                  threshold: float = 0.05) -> Dict:
        """
        Detect drift across all features.
        
        Args:
            current_data: Current batch data
            feature_config: Dictionary mapping features to test methods and thresholds
            threshold: Default threshold if not specified per-feature
        
        Returns:
            Dictionary with results for all features
        """
        results = {
            "summary": {
                "total_features": 0,
                "features_with_drift": 0,
                "drift_percentage": 0.0
            },
            "features": {}
        }
        
        # Get features to check (exclude sensitive attributes if present)
        features_to_check = [col for col in current_data.columns 
                           if col not in ['Sex', 'Age_group', 'Sex_original', 'Age_original']]
        
        results["summary"]["total_features"] = len(features_to_check)
        
        for feature in features_to_check:
            # Get feature-specific config if available
            if feature_config and feature in feature_config:
                method = feature_config[feature].get("method", "auto")
                feat_threshold = feature_config[feature].get("threshold", threshold)
            else:
                method = "auto"
                feat_threshold = threshold
            
            # Detect drift for this feature
            feature_result = self.detect_feature_drift(
                current_data, feature, method, feat_threshold
            )
            
            results["features"][feature] = feature_result
            
            if feature_result.get("drift_detected", False):
                results["summary"]["features_with_drift"] += 1
        
        # Calculate drift percentage
        if results["summary"]["total_features"] > 0:
            results["summary"]["drift_percentage"] = round(
                100 * results["summary"]["features_with_drift"] / results["summary"]["total_features"],
                2
            )
        
        return results


def quick_drift_check(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                     threshold: float = 0.05) -> pd.DataFrame:
    """
    Quick drift check across all features, returns summary DataFrame.
    
    Args:
        baseline_df: Reference data
        current_df: Current data to check
        threshold: P-value threshold
    
    Returns:
        DataFrame with drift summary for each feature
    """
    detector = DriftDetector(baseline_df)
    results = detector.detect_all_features_drift(current_df, threshold=threshold)
    
    # Convert to DataFrame for easy viewing
    rows = []
    for feature, result in results["features"].items():
        if "error" not in result:
            rows.append({
                "Feature": feature,
                "Method": result["method"],
                "Drift_Detected": result["drift_detected"],
                "P_Value": result.get("p_value", result.get("kl_divergence", 0)),
                "Statistic": result.get("ks_statistic", result.get("chi2_statistic", 0))
            })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage
    print("Drift Detection Module Loaded")
    print("Available methods: ks_test, chi_square, kl_divergence")