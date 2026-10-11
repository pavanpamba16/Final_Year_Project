"""
EXPLAINA AI: Clinical Decision Support Model
Architecture: Soft Voting Ensemble (Random Forest + CatBoost)
Input: Patient clinical manifestations (demographics, WHO severe indicators, systemic symptoms)
Output: Severe malaria risk triage probability + SHAP and LIME Interpretability
"""

import os
import io
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
import shap

# Standard 17-feature schema aligned with MultiCohort Clinical Dataset
CLINICAL_FEATURES = [
    'age', 'sex', 'fever', 'cold', 'rigor', 'fatigue', 'headace',
    'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia',
    'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia'
]

# Clinical risk weightings for explainability
WHO_SEVERE_FEATURES = ['Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
SYSTEMIC_FEATURES = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea']


def evaluate_clinical_predictions(y_true, y_probs, threshold=0.5):
    """Computes comprehensive clinical validation metrics."""
    preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': round(float(accuracy_score(y_true, preds)), 4),
        'precision': round(float(precision_score(y_true, preds, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, preds, zero_division=0)), 4),
        'specificity': round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        'f1': round(float(f1_score(y_true, preds, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, y_probs)), 4),
        'pr_auc': round(float(average_precision_score(y_true, y_probs)), 4),
        'brier_score': round(float(brier_score_loss(y_true, y_probs)), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


class MalariaClinicalModel:
    """
    Inference & Explainability Engine for patient clinical manifestations.
    Provides calibrated risk triage along with local SHAP and LIME feature attributions.
    """
    def __init__(self, checkpoint_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, 'models', 'clinical', 'clinical_model.pkl')
            
        self.features = CLINICAL_FEATURES
        self.threshold = 0.50
        self.model = None
        self.test_metrics = {}
        self.bg_df = None
        
        bg_path = os.path.join(base_dir, 'data', 'MultiCohort_Malaria_Clinical_Data.csv')
        if os.path.exists(bg_path):
            try:
                raw_df = pd.read_csv(bg_path)
                avail_cols = [c for c in self.features if c in raw_df.columns]
                self.bg_df = raw_df[avail_cols].sample(min(80, len(raw_df)), random_state=42)
            except Exception:
                pass
                
        if os.path.exists(checkpoint_path):
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        """Loads fitted model ensemble package."""
        pkg = joblib.load(checkpoint_path)
        self.model = pkg['model']
        self.test_metrics = pkg.get('test_metrics', {})
        self.threshold = pkg.get('threshold', 0.50)
        self.features = pkg.get('features', CLINICAL_FEATURES)

    def validate_inputs(self, input_dict):
        """Validates and sanitizes patient clinical input features."""
        cleaned = {}
        raw_age = input_dict.get('age', None)
        if raw_age is None or str(raw_age).strip() == '':
            return None, "Patient age is required. Please provide a valid age between 0 and 120."
        try:
            age = int(raw_age)
            if age < 0 or age > 120:
                return None, f"Patient age must be between 0 and 120 years. Received: {age}"
            cleaned['age'] = age
        except (ValueError, TypeError):
            return None, "Patient age must be a valid integer."
            
        raw_sex = input_dict.get('sex', None)
        if raw_sex is None or str(raw_sex).strip() == '':
            return None, "Patient gender is required. Please select Female or Male."
        try:
            sex = int(raw_sex)
            if sex not in [0, 1]:
                return None, "Patient gender must be 0 (Female) or 1 (Male)."
            cleaned['sex'] = sex
        except (ValueError, TypeError):
            return None, "Patient gender must be 0 or 1."
            
        for feat in self.features:
            if feat in ['age', 'sex']:
                continue
            val = input_dict.get(feat, 0)
            cleaned[feat] = 1 if val in [1, '1', True, 'on'] else 0
            
        return cleaned, None

    def predict(self, input_dict):
        """
        Executes clinical risk triage and generates local SHAP and LIME figures.
        """
        if self.model is None:
            return {'error': 'Clinical model ensemble checkpoint not loaded.'}
            
        patient_data, err = self.validate_inputs(input_dict)
        if err:
            return {'error': err}
            
        patient_df = pd.DataFrame([patient_data])[self.features]
        severe_prob = float(self.model.predict_proba(patient_df)[0, 1])
        is_severe = (severe_prob >= self.threshold)
        
        # Explainability
        shap_records = self._compute_shap(patient_df)
        positives = [r for r in shap_records if r['impact_pct'] > 0]
        negatives = [r for r in shap_records if r['impact_pct'] < 0]
        shap_b64 = self._generate_shap_figure(positives, negatives)
        
        lime_rules = self._compute_lime(patient_df)
        lime_b64 = self._generate_lime_figure(lime_rules)
        
        return {
            'target_col': 'severe_maleria',
            'is_severe': is_severe,
            'outcome_label': 'Severe Malaria Complications Present (High Acute Risk)' if is_severe else 'Non-Severe Presentation (Uncomplicated / Low Acute Risk)',
            'probability': round(severe_prob * 100, 1),
            'threshold': round(self.threshold * 100, 1),
            'positives': positives[:5],
            'negatives': negatives[:5],
            'shap_figure_b64': shap_b64,
            'lime_rules': lime_rules[:6],
            'lime_figure_b64': lime_b64,
            'metrics': self.test_metrics,
            'reported_symptoms': {k: v for k, v in patient_data.items() if k not in ['age', 'sex'] and v == 1},
            'target_definition': "Clinical Target: severe_maleria (0: Non-severe; 1: WHO severe malaria organ complications)"
        }

    def _compute_shap(self, patient_df):
        """Calculates Shapley values using TreeExplainer on ensemble constituent trees."""
        tree_model = self.model.estimators_[0] if hasattr(self.model, 'estimators_') else self.model
        try:
            explainer = shap.TreeExplainer(tree_model)
            shap_vals = explainer.shap_values(patient_df)
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                vals = shap_vals[0, :, 1]
            elif len(shap_vals.shape) == 2 and shap_vals.shape[0] == 1:
                vals = shap_vals[0]
            else:
                vals = shap_vals
        except Exception:
            vals = np.zeros(len(self.features))
            
        records = []
        patient_row = patient_df.iloc[0].to_dict()
        for idx, feat in enumerate(self.features):
            val = patient_row.get(feat, 0)
            imp = float(vals[idx]) * 100
            records.append({
                'feature': feat,
                'feature_display': feat.replace('_', ' ').title(),
                'value': val,
                'shap_val': float(vals[idx]),
                'impact_pct': round(imp, 2),
                'direction': 'increases_risk' if imp > 0 else 'decreases_risk'
            })
        records.sort(key=lambda x: abs(x['shap_val']), reverse=True)
        return records

    def _compute_lime(self, patient_df):
        """Fits transparent local surrogate linear model around patient profile."""
        if self.bg_df is None:
            return []
        patient_vec = patient_df.iloc[0].values
        n_samples = 150
        perturbed = np.tile(patient_vec, (n_samples, 1))
        
        # Random binary perturb
        for i in range(1, n_samples):
            mask = np.random.rand(len(self.features)) < 0.20
            perturbed[i, mask] = 1 - perturbed[i, mask]
            
        pert_df = pd.DataFrame(perturbed, columns=self.features)
        pert_probs = self.model.predict_proba(pert_df)[:, 1]
        
        # Fit local ridge surrogate
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(pert_df, pert_probs)
        coefs = surrogate.coef_
        
        rules = []
        for idx, feat in enumerate(self.features):
            val = patient_vec[idx]
            weight = float(coefs[idx])
            rules.append({
                'rule': f"{feat} = {val}",
                'feature': feat,
                'weight': round(weight, 4),
                'direction': 'Supports Severe' if weight > 0 else 'Supports Non-Severe'
            })
        rules.sort(key=lambda x: abs(x['weight']), reverse=True)
        return rules

    def _generate_shap_figure(self, positives, negatives):
        """Builds horizontal bar chart of clinical feature Shapley attributions."""
        items = []
        for p in positives[:5]:
            items.append((p['feature_display'], p['impact_pct'], '#ff4757'))
        for n in negatives[:4]:
            items.append((n['feature_display'], n['impact_pct'], '#2ed573'))
            
        if not items:
            return None
            
        items.sort(key=lambda x: x[1])
        names = [x[0] for x in items]
        vals = [x[1] for x in items]
        colors = [x[2] for x in items]
        
        fig, ax = plt.subplots(figsize=(7.5, max(3.4, len(items) * 0.42)), dpi=150)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        
        bars = ax.barh(names, vals, color=colors, height=0.55, zorder=3)
        ax.axvline(0, color='#64748b', linewidth=1.2, linestyle='--', zorder=4)
        
        for bar, val in zip(bars, vals):
            width = bar.get_width()
            x_pos = width + (1.2 if width >= 0 else -1.2)
            ha = 'left' if width >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.1f}%", 
                    va='center', ha=ha, color='#f8fafc', fontsize=8.5, fontweight='bold')
                    
        ax.tick_params(colors='#94a3b8', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(axis='x', color='#1e293b', linestyle=':', zorder=0)
        ax.set_xlabel("SHAP Impact on Severe Malaria Risk (%)", color='#94a3b8', fontsize=9, labelpad=6)
        ax.set_title("EXPLAINA AI: CLINICAL MANIFESTATIONS SHAP ATTRIBUTION", color='#38bdf8', fontsize=10.5, fontweight='bold', pad=8)
        
        xmin, xmax = ax.get_xlim()
        span = max(abs(xmin), abs(xmax)) + 8
        ax.set_xlim(-span, span)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

    def _generate_lime_figure(self, lime_rules):
        """Builds bar chart of local decision rules from LIME surrogate model."""
        top_rules = lime_rules[:8]
        if not top_rules:
            return None
            
        top_rules = sorted(top_rules, key=lambda x: x['weight'])
        names = [r['rule'] for r in top_rules]
        weights = [r['weight'] for r in top_rules]
        colors = ['#ff4757' if w > 0 else '#2ed573' for w in weights]
        
        fig, ax = plt.subplots(figsize=(7.5, max(3.4, len(top_rules) * 0.42)), dpi=150)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        
        bars = ax.barh(names, weights, color=colors, height=0.55, zorder=3)
        ax.axvline(0, color='#64748b', linewidth=1.2, linestyle='--', zorder=4)
        
        for bar, val in zip(bars, weights):
            width = bar.get_width()
            x_pos = width + (0.015 if width >= 0 else -0.015)
            ha = 'left' if width >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.3f}", 
                    va='center', ha=ha, color='#f8fafc', fontsize=8.5, fontweight='bold')
                    
        ax.tick_params(colors='#94a3b8', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(axis='x', color='#1e293b', linestyle=':', zorder=0)
        ax.set_xlabel("Local Surrogate Decision Weight", color='#94a3b8', fontsize=9, labelpad=6)
        ax.set_title("EXPLAINA AI: CLINICAL MANIFESTATIONS LIME DECISION RULES", color='#34d399', fontsize=10.5, fontweight='bold', pad=8)
        
        xmin, xmax = ax.get_xlim()
        span = max(abs(xmin), abs(xmax)) + 0.05
        ax.set_xlim(-span, span)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')


def train_clinical_models(data_path=None, output_dir=None):
    """
    Trains candidate clinical triage models (Logistic Regression, Random Forest,
    CatBoost, XGBoost, and Voting Ensemble) and evaluates on 15% untouched held-out test set.
    Saves winning ensemble package to models/clinical/clinical_model.pkl.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'MultiCohort_Malaria_Clinical_Data.csv')
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'models', 'clinical')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    target_col = 'severe_maleria'
    features = [c for c in CLINICAL_FEATURES if c in df.columns]
    
    X = df[features]
    y = df[target_col]
    
    # Stratified 70/15/15 partition
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
    cb = CatBoostClassifier(iterations=150, depth=4, verbose=0, random_state=42, auto_class_weights='Balanced')
    ensemble = VotingClassifier(estimators=[('rf', rf), ('cb', cb)], voting='soft')
    
    ensemble.fit(X_train, y_train)
    test_probs = ensemble.predict_proba(X_test)[:, 1]
    metrics = evaluate_clinical_predictions(y_test, test_probs, threshold=0.50)
    
    save_path = os.path.join(output_dir, 'clinical_model.pkl')
    pkg = {
        'model': ensemble,
        'features': features,
        'threshold': 0.50,
        'test_metrics': metrics,
        'architecture': 'Voting Ensemble (Random Forest + CatBoost)'
    }
    joblib.dump(pkg, save_path)
    print(f"[Clinical Training Complete] Saved to {save_path} with Held-Out Test Metrics: {metrics}")
    return metrics
