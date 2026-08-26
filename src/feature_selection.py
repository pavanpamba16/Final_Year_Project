import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

def get_mutual_info_scores(X, y):
    """Computes Mutual Information score for each feature."""
    scores = mutual_info_classif(X, y, random_state=123)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

def get_anova_f_scores(X, y):
    """Computes ANOVA F-statistic for each feature."""
    f_vals, p_vals = f_classif(X, y)
    # Replace NaNs or Infs with 0 if any constant features exist
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(f_vals, index=X.columns).sort_values(ascending=False)

def get_rf_importance_scores(X, y):
    """Computes feature importances using a Random Forest model."""
    rf = RandomForestClassifier(random_state=123, n_estimators=100)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

def get_permutation_importance_scores(model, X, y):
    """Computes Permutation Feature Importance on a validation/test set."""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=123, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

def get_shap_importance_scores(model, X):
    """Computes SHAP-based importance (mean absolute SHAP value) for features."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle cases where shap returns list of arrays (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
            
        mean_shap = np.abs(shap_values).mean(axis=0)
        return pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
    except Exception as e:
        print(f"Error calculating SHAP importance: {e}")
        # Fallback to zeros if SHAP fails
        return pd.Series(0.0, index=X.columns).sort_values(ascending=False)

def run_feature_selection_report(X, y):
    """Generates a summary DataFrame ranking features across multiple criteria."""
    mi = get_mutual_info_scores(X, y)
    anova = get_anova_f_scores(X, y)
    rf = get_rf_importance_scores(X, y)
    
    report = pd.DataFrame({
        'Mutual Info': mi,
        'ANOVA F-Score': anova,
        'RF Importance': rf
    })
    
    # Calculate average rank across all methods
    report['MI Rank'] = report['Mutual Info'].rank(ascending=False)
    report['ANOVA Rank'] = report['ANOVA F-Score'].rank(ascending=False)
    report['RF Rank'] = report['RF Importance'].rank(ascending=False)
    report['Avg Rank'] = report[['MI Rank', 'ANOVA Rank', 'RF Rank']].mean(axis=1)
    
    return report.sort_values(by='Avg Rank')
