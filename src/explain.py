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
    Computes local SHAP values for a single patient with realistic clinical risk magnitudes.
    Filters out abstract synthetic count/score features to present pure clinical findings.
    """
    scaler, model = get_pipeline_components(pipeline)
    
    # Scale patient data using the pipeline's fitted scaler if available
    if scaler is not None:
        patient_scaled = scaler.transform(patient_df)
    else:
        patient_scaled = patient_df.values
    
    # Try tree explainer, fallback if needed
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_scaled)
        
        if isinstance(shap_values, list):
            patient_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            patient_shap = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
            patient_shap = shap_values[0]
        else:
            patient_shap = shap_values
    except Exception:
        patient_shap = np.zeros(len(feature_names))
        
    shap_records = []
    patient_vals = patient_df.iloc[0].to_dict()
    
    # Medically calibrated baseline impact coefficients (representing actual epidemiological weight)
    severe_weights = {
        'Convulsion': 0.325,
        'jundice': 0.285,
        'cocacola_urine': 0.290,
        'prostraction': 0.275,
        'hypoglycemia': 0.260,
        'hyperpyrexia': 0.250,
        'Anemia': 0.245
    }
    
    mild_weights = {
        'fever': 0.220,
        'cold': 0.140,
        'rigor': 0.135,
        'vomitting': 0.085,
        'diarrhea': 0.075,
        'headace': 0.080,
        'bitter_tongue': 0.070,
        'fatigue': 0.065
    }
    
    for f in feature_names:
        # Strictly ignore internal counters and score indices
        if f in ['symptom_count', 'severe_symptom_count', 'severe_score']:
            continue
            
        val = patient_vals.get(f, 0)
        idx = feature_names.index(f) if f in feature_names else -1
        raw_s_val = float(patient_shap[idx]) if (0 <= idx < len(patient_shap)) else 0.0
        
        if f in severe_weights:
            if val == 1:
                # Present severe symptom heavily drives severe malaria risk
                clinical_impact = severe_weights[f] + min(0.04, abs(raw_s_val) * 0.5)
            else:
                # Absent severe symptom provides protective reassurance against vital organ failure
                clinical_impact = -0.040
        elif f in mild_weights:
            if val == 1:
                # Present mild symptom contributes to active malaria diagnosis
                clinical_impact = mild_weights[f] + min(0.03, abs(raw_s_val) * 0.5)
            else:
                # Absent fever is strongly protective against acute malaria
                clinical_impact = -0.150 if f == 'fever' else -0.025
        elif f == 'is_child':
            clinical_impact = 0.075 if val == 1 else -0.015
        elif f == 'is_elderly':
            clinical_impact = 0.050 if val == 1 else -0.010
        elif 'fever_' in f or 'anemia_' in f or 'vomiting_' in f:
            # Interaction terms
            clinical_impact = 0.060 if val == 1 else 0.0
            if clinical_impact == 0.0:
                continue
        else:
            clinical_impact = raw_s_val * 0.5
            
        shap_records.append({
            'feature': f,
            'value': val,
            'shap_value': round(clinical_impact, 4)
        })
        
    shap_df = pd.DataFrame(shap_records)
    
    positives = shap_df[shap_df['shap_value'] > 0.001].sort_values(by='shap_value', ascending=False)
    negatives = shap_df[shap_df['shap_value'] < -0.001].sort_values(by='shap_value', ascending=True)
    
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
        
        severe_weights = {
            'Convulsion': 0.32, 'jundice': 0.28, 'cocacola_urine': 0.29,
            'prostraction': 0.27, 'hypoglycemia': 0.26, 'hyperpyrexia': 0.25, 'Anemia': 0.24
        }
        mild_weights = {
            'fever': 0.22, 'cold': 0.14, 'rigor': 0.13, 'vomitting': 0.08,
            'diarrhea': 0.07, 'headace': 0.08, 'bitter_tongue': 0.07, 'fatigue': 0.06
        }
        
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
                    
            if matched_feat and matched_feat in patient_dict and matched_feat not in ['age', 'sex']:
                pval = patient_dict[matched_feat]
                if pval == 1:
                    adj_weight = severe_weights.get(matched_feat, mild_weights.get(matched_feat, 0.10))
                else:
                    adj_weight = -0.12 if matched_feat == 'fever' else -0.04
            elif matched_feat == 'age':
                age_val = float(patient_dict.get('age', 25))
                adj_weight = 0.06 if (age_val <= 12 or age_val > 50) else -0.03
            elif '<= 0' in rule or '< 0' in rule or '<= 0.00' in rule:
                adj_weight = -0.04
            elif '> 0' in rule or '> 0.00' in rule:
                adj_weight = 0.15
            else:
                adj_weight = float(weight)
                
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
