import os
import sys
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

# Add src to python path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import MalariaFeatureEngineer
from explain import explain_prediction_shap, explain_prediction_lime

app = Flask(__name__)

# Load models and training data for explainers
MODEL_PATH = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\improved\production_model.pkl"
BASELINE_PATH = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\baseline\baseline_rf.pkl"
DATA_PATH = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\data\Malaria-Data.csv"

# Global references
model_package = None
baseline_model = None
X_train_engineered = None
raw_symptoms_list = [
    'fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 
    'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 
    'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia'
]

def load_resources():
    global model_package, baseline_model, X_train_engineered
    
    # Load Upgraded Production Model
    if os.path.exists(MODEL_PATH):
        try:
            model_package = joblib.load(MODEL_PATH)
            print(f"Loaded production model v2.0 from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading production model: {e}")
            
    # Load Baseline Model
    if os.path.exists(BASELINE_PATH):
        try:
            baseline_model = joblib.load(BASELINE_PATH)
            print(f"Loaded baseline model from {BASELINE_PATH}")
        except Exception as e:
            print(f"Error loading baseline model: {e}")
            
    # Load and engineer training data for explainers
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            X = df.drop(columns=['severe_maleria'])
            fe = MalariaFeatureEngineer()
            X_train_engineered = fe.fit_transform(X)
            print(f"Loaded and engineered background data for explainers. Shape: {X_train_engineered.shape}")
        except Exception as e:
            print(f"Error loading training data: {e}")

# Initial load of resources
load_resources()

def process_form_input(form_data):
    """Parses form parameters into a single-row DataFrame matching raw features."""
    # Read age and sex
    try:
        age = int(form_data.get('age', 30))
    except ValueError:
        age = 30
        
    try:
        sex = int(form_data.get('sex', 0))
    except ValueError:
        sex = 0
        
    # Read symptoms (1 if checked, 0 otherwise)
    patient_dict = {
        'age': age,
        'sex': sex
    }
    for symptom in raw_symptoms_list:
        patient_dict[symptom] = int(form_data.get(symptom, 0) == '1' or form_data.get(symptom, 0) == 1)
        
    return pd.DataFrame([patient_dict])

@app.route('/')
def index():
    # Render symptom entry page
    metrics = model_package['metrics'] if model_package else None
    return render_template('index.html', symptoms=raw_symptoms_list, model_metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if model_package is None:
        return "Production model not loaded. Please train models first.", 500
        
    # Parse inputs from form
    patient_raw_df = process_form_input(request.form)
    
    # Run Feature Engineering
    fe = model_package['feature_engineer']
    patient_eng_df = fe.transform(patient_raw_df)
    
    # Get model and threshold
    pipeline = model_package['model']
    threshold = model_package['threshold']
    feature_names = model_package['features']
    
    # Run Prediction
    # pipeline predict_proba takes care of scaling internally
    prob = float(pipeline.predict_proba(patient_eng_df)[0, 1])
    risk = "HIGH" if prob >= threshold else "LOW"
    
    # Run SHAP Explainability
    pos_contribs, neg_contribs = explain_prediction_shap(
        pipeline=pipeline,
        X_train=X_train_engineered,
        patient_df=patient_eng_df,
        feature_names=feature_names
    )
    
    # Run LIME Explainability
    raw_lime = explain_prediction_lime(
        pipeline=pipeline,
        X_train_df=X_train_engineered,
        patient_df=patient_eng_df,
        feature_names=feature_names
    )
    
    # Re-map feature names to more user-friendly representations
    friendly_names = {
        'age': 'Age', 'sex': 'Gender (Male)', 'fever': 'Fever', 'cold': 'Cold/Chills',
        'rigor': 'Rigors (Shivering)', 'fatigue': 'Fatigue/Weakness', 'headace': 'Headache',
        'bitter_tongue': 'Bitter Tongue', 'vomitting': 'Vomiting', 'diarrhea': 'Diarrhea',
        'Convulsion': 'Convulsions', 'Anemia': 'Anemia', 'jundice': 'Jaundice',
        'cocacola_urine': 'Coca-Cola Urine (Hemoglobinuria)', 'hypoglycemia': 'Hypoglycemia',
        'prostraction': 'Prostration (Extreme Exhaustion)', 'hyperpyrexia': 'Hyperpyrexia (Extremely High Fever)',
        'is_child': 'Is Child (<= 12 years)', 'is_elderly': 'Is Elderly (> 50 years)',
        'symptom_count': 'Total Symptom Count', 'severe_symptom_count': 'Severe Symptom Count',
        'fever_rigor': 'Fever + Rigor Interaction', 'fever_headache': 'Fever + Headache Interaction',
        'anemia_fatigue': 'Anemia + Fatigue Interaction', 'vomiting_diarrhea': 'Vomiting + Diarrhea Interaction',
        'severe_score': 'Symptom Severity Index'
    }
    
    # Convert lists to human-friendly versions
    for item in pos_contribs:
        item['friendly_name'] = friendly_names.get(item['feature'], item['feature'])
    for item in neg_contribs:
        item['friendly_name'] = friendly_names.get(item['feature'], item['feature'])
        
    # Clean LIME rules with friendly names
    lime_explanations = []
    for l_item in raw_lime:
        clean_rule = l_item['rule']
        for raw_k, friendly_k in friendly_names.items():
            if raw_k in clean_rule:
                clean_rule = clean_rule.replace(raw_k, friendly_k)
        l_item['clean_rule'] = clean_rule
        lime_explanations.append(l_item)
        
    # Render results template
    result_data = {
        'age': patient_raw_df.iloc[0]['age'],
        'sex': 'Male' if patient_raw_df.iloc[0]['sex'] == 1 else 'Female',
        'raw_symptoms': {s: patient_raw_df.iloc[0][s] for s in raw_symptoms_list if patient_raw_df.iloc[0][s] == 1},
        'probability': round(prob * 100, 1),
        'risk': risk,
        'threshold': round(threshold * 100, 1),
        'positives': pos_contribs[:5], # Top 5 contributors
        'negatives': neg_contribs[:5], # Top 5 mitigators
        'lime_explanations': lime_explanations[:6], # Top 6 LIME rules
        'model_name': model_package['version'],
        'model_metrics': model_package['metrics'],
        'date': model_package['date']
    }
    
    return render_template('result.html', data=result_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model_package is None:
        return jsonify({'error': 'Production model not loaded'}), 500
        
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'Invalid JSON body'}), 400
        
    patient_raw_df = process_form_input(json_data)
    fe = model_package['feature_engineer']
    patient_eng_df = fe.transform(patient_raw_df)
    
    pipeline = model_package['model']
    threshold = model_package['threshold']
    
    prob = float(pipeline.predict_proba(patient_eng_df)[0, 1])
    risk = "HIGH" if prob >= threshold else "LOW"
    
    return jsonify({
        'risk': risk,
        'probability': prob,
        'threshold': threshold
    })

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
