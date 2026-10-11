import os
import sys
import time
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from explain_ai_engine import ExplainAIMultimodalOrchestrator, CLINICAL_FEATURES

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLE_SMEARS_FOLDER = os.path.join(BASE_DIR, 'static', 'sample_smears')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_SMEARS_FOLDER, exist_ok=True)

# Initialize the EXPLAINA AI Multi-Modal Engine
orchestrator = ExplainAIMultimodalOrchestrator()

def extract_image_input(request):
    """
    Extracts image input from request (either uploaded file or sample smear choice).
    Returns (image_path_or_file, error_message)
    """
    # 1. Check custom uploaded file
    if 'smear_image' in request.files:
        uploaded_file = request.files['smear_image']
        if uploaded_file and uploaded_file.filename != '':
            # Validate extension
            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            if ext not in ['.png', '.jpg', '.jpeg']:
                return None, f"Unsupported file extension '{ext}'. Please upload a PNG, JPG, or JPEG image."
                
            safe_filename = f"upload_{int(time.time())}_{os.path.basename(uploaded_file.filename)}"
            save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            uploaded_file.save(save_path)
            return save_path, None
            
    # 2. Check sample smear selection
    sample_choice = request.form.get('sample_smear', '').strip()
    if sample_choice:
        sample_path = os.path.join(SAMPLE_SMEARS_FOLDER, sample_choice)
        if os.path.exists(sample_path):
            return sample_path, None
            
    return None, "No blood smear image provided. Please upload an image or select a reference sample."

def extract_clinical_input(form_data):
    """
    Extracts and parses clinical input features from form data.
    """
    cleaned = {}
    cleaned['age'] = form_data.get('age', None)
    cleaned['sex'] = form_data.get('sex', None)
    
    active_symptoms = 0
    for feat in CLINICAL_FEATURES:
        if feat in ['age', 'sex']:
            continue
        val = 1 if form_data.get(feat, 0) in ['1', 1, 'on', True] else 0
        cleaned[feat] = val
        if val == 1:
            active_symptoms += 1
            
    cleaned['_active_symptoms_count'] = active_symptoms
    return cleaned

@app.route('/')
def index():
    return render_template(
        'index.html',
        image_metrics=orchestrator.image_engine.metrics,
        clinical_metrics=orchestrator.clinical_engine.test_metrics,
        multimodal_metrics=orchestrator.multimodal_metrics
    )

@app.route('/predict', methods=['POST'])
def predict_dispatch():
    """Dispatches request to appropriate mode based on selected mode input."""
    mode = request.form.get('mode', 'c').lower().strip()
    if mode == 'a':
        return predict_mode_a()
    elif mode == 'b':
        return predict_mode_b()
    else:
        return predict_mode_c()

@app.route('/predict/mode-a', methods=['POST'])
def predict_mode_a():
    """Mode A: Image Only Pipeline."""
    img_path, err = extract_image_input(request)
    if err:
        return render_template('result.html', data={
            'mode_code': 'a',
            'mode_display': 'Mode A: Image Only',
            'error': err,
            'image_result': {'error': err}
        })
        
    res = orchestrator.evaluate_mode_a(img_path)
    res['mode_code'] = 'a'
    res['mode_display'] = 'Mode A: Image Only'
    return render_template('result.html', data=res)

@app.route('/predict/mode-b', methods=['POST'])
def predict_mode_b():
    """Mode B: Clinical Only Pipeline."""
    clin_dict = extract_clinical_input(request.form)
    
    if clin_dict.get('_active_symptoms_count', 0) == 0:
        return render_template('result.html', data={
            'mode_code': 'b',
            'mode_display': 'Mode B: Clinical Only',
            'error_title': 'Clinical Symptoms Required for Mode B',
            'error': 'Please select at least one active clinical symptom or sign observed during patient triage before submitting Mode B (Clinical Only).'
        })
        
    res = orchestrator.evaluate_mode_b(clin_dict)
    res['mode_code'] = 'b'
    res['mode_display'] = 'Mode B: Clinical Only'
    if 'error' in res:
        res['error_title'] = 'Clinical Validation Error'
    return render_template('result.html', data=res)

@app.route('/predict/mode-c', methods=['POST'])
def predict_mode_c():
    """Mode C: Image + Clinical Dual-Stream Pipeline with Conflict Detection."""
    img_path, img_err = extract_image_input(request)
    clin_dict = extract_clinical_input(request.form)
    
    if img_err:
        return render_template('result.html', data={
            'mode_code': 'c',
            'mode_display': 'Mode C: Dual-Stream (Image Missing)',
            'error_title': 'Blood Smear Image Missing for Mode C',
            'error': 'Mode C requires both a microscopic blood cell image and clinical manifestations. Please select one of the reference smears (Parasitized or Uninfected) or upload a single PNG/JPG image file. If you wish to evaluate clinical symptoms without an image, please select Mode B (Clinical Only).'
        })
        
    if clin_dict.get('_active_symptoms_count', 0) == 0:
        return render_template('result.html', data={
            'mode_code': 'c',
            'mode_display': 'Mode C: Dual-Stream (Symptoms Missing)',
            'error_title': 'Clinical Symptoms Required for Mode C',
            'error': 'Mode C requires at least one active clinical symptom or manifestation to be selected (such as Fever, Headache, Jaundice, Convulsions, Anemia, etc.) along with patient age, gender, and a blood smear image.'
        })
        
    res = orchestrator.evaluate_mode_c(img_path, clin_dict)
    res['mode_code'] = 'c'
    res['mode_display'] = 'Mode C: Dual-Stream (Image + Clinical)'
    if 'error' in res:
        res['error_title'] = 'Assessment Input Validation Error'
    return render_template('result.html', data=res)

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint exposing held-out test evaluation metrics across Image, Clinical, and Multimodal models."""
    return jsonify({
        'status': 'success',
        'image_model': {
            'architecture': 'MobileNetV2 Transfer Learning',
            'partition': 'Patient-Disjoint (Zero Patient Leakage)',
            'metrics_test': orchestrator.image_engine.metrics
        },
        'clinical_model': {
            'architecture': 'Voting Ensemble (Random Forest + CatBoost)',
            'target': 'severe_maleria',
            'metrics_test': orchestrator.clinical_engine.test_metrics
        },
        'multimodal_model': {
            'architecture': 'Dual-Stream Synergistic Fusion',
            'fusion_strategy': 'Dual-Stream Independent Inference with Polarity Conflict Synthesis',
            'metrics_test': orchestrator.multimodal_metrics
        }
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic inference across Mode A, Mode B, or Mode C."""
    json_data = request.get_json(silent=True) or {}
    mode = json_data.get('mode', 'c').lower()
    
    if mode == 'a':
        img_path = json_data.get('image_path')
        if not img_path:
            return jsonify({'error': 'image_path required for Mode A'}), 400
        res = orchestrator.evaluate_mode_a(img_path)
        return jsonify(res)
        
    elif mode == 'b':
        clin_data = json_data.get('clinical_data', {})
        res = orchestrator.evaluate_mode_b(clin_data)
        return jsonify(res)
        
    else: # mode == 'c'
        img_path = json_data.get('image_path')
        clin_data = json_data.get('clinical_data', {})
        if not img_path:
            return jsonify({'error': 'image_path required for Mode C'}), 400
        res = orchestrator.evaluate_mode_c(img_path, clin_data)
        return jsonify(res)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
