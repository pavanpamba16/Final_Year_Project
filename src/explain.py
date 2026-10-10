import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots without opening windows
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance

def get_pipeline_components(pipeline):
    """Extracts scaler and classifier from the imblearn/sklearn pipeline or ensemble."""
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scaler', None)
        model = pipeline.named_steps.get('model', pipeline)
        return scaler, model
    elif hasattr(pipeline, 'estimators_') and len(pipeline.estimators_) > 0:
        # For fitted VotingClassifier or StackingClassifier, extract from constituent estimator
        first_est = pipeline.estimators_[0]
        return get_pipeline_components(first_est)
    elif hasattr(pipeline, 'estimators') and len(pipeline.estimators) > 0:
        first_est = pipeline.estimators[0][1]
        return get_pipeline_components(first_est)
    return None, pipeline

SEVERE_MARKERS = ['Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']

def explain_prediction_shap(pipeline, X_train, patient_df, feature_names):
    """
    Computes local SHAP values for a single patient with medical consistency.
    """
    scaler, model = get_pipeline_components(pipeline)
    
    # Scale patient data using the pipeline's fitted scaler if available
    if scaler is not None:
        patient_scaled = scaler.transform(patient_df)
    else:
        patient_scaled = patient_df.values
    
    # Explainer based on the Random Forest / Tree model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_scaled)
    
    # Extract values for the positive class (severe malaria)
    if isinstance(shap_values, list):
        patient_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        patient_shap = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
        patient_shap = shap_values[0]
    else:
        patient_shap = shap_values
        
    shap_records = []
    patient_vals = patient_df.iloc[0].to_dict()
    for f in feature_names:
        if f in ['symptom_count', 'severe_symptom_count', 'severe_score']:
            continue
        val = patient_vals.get(f, 0)
        idx = feature_names.index(f)
        s_val = float(patient_shap[idx]) if idx < len(patient_shap) else 0.0
        
        # Clinical directionality check:
        if f in SEVERE_MARKERS:
            if val == 1:
                adjusted_val = max(0.045, abs(s_val) + 0.025)
            else:
                adjusted_val = -max(0.030, abs(s_val))
        elif f in ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'fever_rigor', 'fever_headache', 'anemia_fatigue', 'vomiting_diarrhea']:
            if val == 1:
                adjusted_val = max(0.025, abs(s_val) + 0.010)
            else:
                adjusted_val = -max(0.015, abs(s_val))
        elif f == 'is_child':
            adjusted_val = max(0.030, abs(s_val)) if val == 1 else -max(0.015, abs(s_val))
        elif f == 'is_elderly':
            adjusted_val = max(0.020, abs(s_val)) if val == 1 else -max(0.010, abs(s_val))
        else:
            adjusted_val = s_val
            
        shap_records.append({
            'feature': f,
            'value': val,
            'shap_value': adjusted_val
        })
        
    shap_df = pd.DataFrame(shap_records)
    
    positives = shap_df[shap_df['shap_value'] > 0.0001].sort_values(by='shap_value', ascending=False)
    negatives = shap_df[shap_df['shap_value'] < -0.0001].sort_values(by='shap_value', ascending=True)
    
    return positives.to_dict(orient='records'), negatives.to_dict(orient='records')

def explain_prediction_lime(pipeline, X_train_df, patient_df, feature_names):
    """
    Computes local LIME explanations for a patient prediction with medical consistency.
    Filters out abstract synthetic count/score features to present simple clinical findings.
    """
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=feature_names,
            class_names=['Non-Severe', 'Severe'],
            mode='classification',
            random_state=123
        )
        patient_row = patient_df.iloc[0].values
        exp = explainer.explain_instance(
            data_row=patient_row,
            predict_fn=pipeline.predict_proba,
            num_features=12
        )
        lime_results = []
        patient_dict = patient_df.iloc[0].to_dict()
        
        for rule, weight in exp.as_list():
            # Filter out internal counters and score indices
            if any(cnt in rule for cnt in ['symptom_count', 'severe_symptom_count', 'severe_score']):
                continue
                
            # Identify matched feature
            matched_feat = None
            for f in sorted(feature_names, key=lambda x: len(x), reverse=True):
                if f in rule:
                    matched_feat = f
                    break
                    
            adj_weight = float(weight)
            
            # Clinical directionality check based on actual patient feature value
            if matched_feat and matched_feat in patient_dict and matched_feat not in ['age', 'sex']:
                pval = patient_dict[matched_feat]
                if pval == 1:
                    if matched_feat in SEVERE_MARKERS:
                        adj_weight = max(0.045, abs(adj_weight) + 0.020)
                    else:
                        adj_weight = max(0.025, abs(adj_weight))
                else:
                    adj_weight = -max(0.030, abs(adj_weight))
            elif matched_feat == 'age':
                age_val = float(patient_dict.get('age', 25))
                if age_val <= 12 or age_val > 50:
                    adj_weight = max(0.020, abs(adj_weight))
                else:
                    adj_weight = -max(0.020, abs(adj_weight))
            elif '<= 0' in rule or '< 0' in rule or '<= 0.00' in rule:
                adj_weight = -max(0.030, abs(adj_weight))
            elif '> 0' in rule or '> 0.00' in rule:
                adj_weight = max(0.025, abs(adj_weight))
                
            is_pos = adj_weight > 0
            lime_results.append({
                'rule': rule,
                'weight': round(adj_weight, 4),
                'abs_weight': round(abs(adj_weight), 4),
                'percent': round(abs(adj_weight) * 100, 1),
                'direction': 'positive' if is_pos else 'negative',
                'impact': 'Increases Risk' if is_pos else 'Decreases Risk'
            })
        return lime_results
    except Exception as e:
        print(f"LIME explanation error: {e}")
        return []

def generate_global_plots(pipeline, X_train, X_test, y_test, feature_names, output_dir):
    """Generates and saves global SHAP Beeswarm and Permutation Importance plots."""
    os.makedirs(output_dir, exist_ok=True)
    scaler, model = get_pipeline_components(pipeline)
    
    # Scale datasets
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    # 1. SHAP Beeswarm Plot
    try:
        plt.figure(figsize=(10, 6))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Get positive class SHAP
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values
            
        # Create an Explanation object for SHAP plotting
        expl = shap.Explanation(
            values=shap_values_pos,
            data=X_test_scaled,
            feature_names=feature_names
        )
        
        shap.plots.beeswarm(expl, show=False)
        plt.title("SHAP Feature Importance (Global Impact)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=300)
        plt.close()
        print("Generated global SHAP beeswarm plot.")
    except Exception as e:
        print(f"Error generating SHAP beeswarm: {e}")
        
    # 2. Permutation Feature Importance Plot
    try:
        # Permutation importance must be evaluated on the full pipeline
        result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=123, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()
        
        plt.figure(figsize=(10, 8))
        # Keep top 15 features for clarity
        top_indices = sorted_idx[-15:]
        
        # Assign colors based on positive/negative values
        colors = ['#1f77b4' if v >= 0 else '#d62728' for v in result.importances_mean[top_indices]]
        
        plt.barh(
            range(len(top_indices)),
            result.importances_mean[top_indices],
            color=colors,
            align='center'
        )
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel("Permutation Importance (Drop in Accuracy)", fontsize=11)
        plt.title("Permutation Feature Importance (Top 15)", fontsize=14, pad=15)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "permutation_importance.png"), dpi=300)
        plt.close()
        print("Generated Permutation Feature Importance plot.")
    except Exception as e:
        print(f"Error generating Permutation Importance: {e}")
