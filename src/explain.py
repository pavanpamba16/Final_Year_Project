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
    """Extracts scaler and classifier from the imblearn/sklearn pipeline."""
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    return scaler, model

def explain_prediction_shap(pipeline, X_train, patient_df, feature_names):
    """
    Computes local SHAP values for a single patient.
    Returns sorted lists of features that increased or decreased severe malaria risk.
    """
    scaler, model = get_pipeline_components(pipeline)
    
    # Scale patient data using the pipeline's fitted scaler
    patient_scaled = scaler.transform(patient_df)
    
    # Explainer based on the Random Forest / Tree model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_scaled)
    
    # Extract values for the positive class (severe malaria)
    if isinstance(shap_values, list):
        # shape is list of [class0_shap, class1_shap]
        patient_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        # shape is (samples, features, classes)
        patient_shap = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
        # shape is (samples, features)
        patient_shap = shap_values[0]
    else:
        patient_shap = shap_values
        
    # Map to features and actual values
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': patient_df.iloc[0].values,
        'shap_value': patient_shap
    })
    
    # Separate into positive (increased risk) and negative (decreased risk)
    positives = shap_df[shap_df['shap_value'] > 0.001].sort_values(by='shap_value', ascending=False)
    negatives = shap_df[shap_df['shap_value'] < -0.001].sort_values(by='shap_value', ascending=True)
    
    pos_list = positives.to_dict(orient='records')
    neg_list = negatives.to_dict(orient='records')
    
    return pos_list, neg_list

def explain_prediction_lime(pipeline, X_train_df, patient_df, feature_names):
    """
    Computes local LIME explanations for a patient prediction.
    """
    scaler, model = get_pipeline_components(pipeline)
    
    # Scale training data and patient data
    X_train_scaled = scaler.transform(X_train_df)
    patient_scaled = scaler.transform(patient_df)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=['Non-Severe', 'Severe'],
        mode='classification',
        random_state=123
    )
    
    # LIME explainer needs predict_proba from the model acting on scaled data
    exp = explainer.explain_instance(
        data_row=patient_scaled[0],
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    return exp.as_list()

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
