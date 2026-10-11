import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, preds)
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    
    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'specificity': round(float(spec), 4),
        'f1': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'brier_score': round(float(brier), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def train_clinical_models(data_path=None, output_dir=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'MultiCohort_Malaria_Clinical_Data.csv')
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'models', 'clinical')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Clinical Training] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    target_col = 'severe_maleria'
    features = [c for c in df.columns if c not in [target_col, 'cohort_source']]
    
    X = df[features]
    y = df[target_col]
    
    print(f"[Clinical Training] Total instances: {len(df)}, Features: {len(features)}")
    print(f"[Clinical Training] Target class balance: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # 70% Train, 15% Validation, 15% Untouched Test (Stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"[Clinical Training] Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    candidate_models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random_Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=150, depth=4, verbose=0, random_state=42, auto_class_weights='Balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42)
    }
    
    # Voting ensemble combining best tree models
    rf_best = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
    cb_best = CatBoostClassifier(iterations=150, depth=4, verbose=0, random_state=42, auto_class_weights='Balanced')
    candidate_models['Voting_Ensemble_RF_CB'] = VotingClassifier(
        estimators=[('rf', rf_best), ('cb', cb_best)], voting='soft'
    )
    
    val_results = {}
    fitted_models = {}
    
    print("\n" + "=" * 75)
    print("VALIDATION SET MODEL COMPARISON")
    print("=" * 75)
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, val_probs)
        val_results[name] = metrics
        print(f"{name:25s} | Acc: {metrics['accuracy']:.3f} | Recall: {metrics['recall']:.3f} | Prec: {metrics['precision']:.3f} | F1: {metrics['f1']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
        
    # Select best model based on validation F1-score & ROC-AUC
    best_name = max(val_results, key=lambda k: (val_results[k]['f1'] + val_results[k]['roc_auc']))
    best_model = fitted_models[best_name]
    print("\n" + "=" * 75)
    print(f"SELECTED PRODUCTION MODEL (BASED ON VALIDATION): {best_name}")
    print("=" * 75)
    
    # Evaluate selected model on UNTOUCHED held-out test set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs)
    
    print(f"Held-Out Test Accuracy:     {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Held-Out Test Sensitivity:  {test_metrics['recall'] * 100:.2f}%")
    print(f"Held-Out Test Specificity:  {test_metrics['specificity'] * 100:.2f}%")
    print(f"Held-Out Test Precision:    {test_metrics['precision'] * 100:.2f}%")
    print(f"Held-Out Test F1-Score:     {test_metrics['f1']:.4f}")
    print(f"Held-Out Test ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"Held-Out Test PR-AUC:       {test_metrics['pr_auc']:.4f}")
    print(f"Held-Out Test Brier Score:  {test_metrics['brier_score']:.4f}")
    print(f"Confusion Matrix [TN, FP, FN, TP]: {test_metrics['tn']}, {test_metrics['fp']}, {test_metrics['fn']}, {test_metrics['tp']}")
    
    # Save package
    model_save_path = os.path.join(output_dir, 'clinical_model.pkl')
    package = {
        'model_name': best_name,
        'model': best_model,
        'features': features,
        'target_col': target_col,
        'target_labels': {
            0: 'Non-Severe Presentation (Low/Moderate Severity)',
            1: 'Severe Malaria Complications (High Acute Risk)'
        },
        'decision_threshold': 0.50,
        'validation_metrics': val_results[best_name],
        'test_metrics': test_metrics,
        'all_validation_results': val_results,
        'created_at': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(package, model_save_path)
    print(f"\nSaved production clinical model package to: {model_save_path}")
    return package

if __name__ == '__main__':
    train_clinical_models()
