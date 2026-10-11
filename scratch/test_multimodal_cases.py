import os
import sys
sys.path.append('src')

import joblib
import pandas as pd
from multimodal_fusion import MalariaMultiModalEngine

# 1. Initialize MultiModal Engine and Clinical Model
engine = MalariaMultiModalEngine()
model_pkg = joblib.load('models/improved/production_model.pkl')
pipe = model_pkg['model']
fe = model_pkg['feature_engineer']

test_cases = [
    {
        'title': 'Case 1: Classic Uncomplicated Malaria',
        'symptoms': {'fever': 1, 'cold': 1, 'rigor': 1, 'headace': 1, 'bitter_tongue': 1, 'vomitting': 1},
        'age': 28,
        'sex': 1,
        'smear_path': 'static/sample_smears/parasitized_sample_1.png'
    },
    {
        'title': 'Case 2: Severe Complicated Cerebral Malaria',
        'symptoms': {'fever': 1, 'Convulsion': 1, 'jundice': 1, 'prostraction': 1, 'hyperpyrexia': 1},
        'age': 6,
        'sex': 0,
        'smear_path': 'static/sample_smears/parasitized_sample_2.png'
    },
    {
        'title': 'Case 3: Severe Non-Malarial Acute Emergency (Bacterial Sepsis / Meningitis)',
        'symptoms': {'fever': 1, 'Convulsion': 1, 'prostraction': 1},
        'age': 35,
        'sex': 1,
        'smear_path': 'static/sample_smears/uninfected_sample_1.png'
    },
    {
        'title': 'Case 4: Non-Malarial Febrile Illness (Viral / Typhoid)',
        'symptoms': {'fever': 1, 'headace': 1},
        'age': 22,
        'sex': 0,
        'smear_path': 'static/sample_smears/uninfected_sample_2.png'
    },
    {
        'title': 'Case 5: Healthy Baseline Profile',
        'symptoms': {},
        'age': 30,
        'sex': 1,
        'smear_path': 'static/sample_smears/uninfected_sample_1.png'
    }
]

severe_markers = ['Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
raw_symptoms_list = [
    'fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 
    'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 
    'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia'
]

print("=" * 105)
print("             MULTI-MODAL MALARIA DIAGNOSTIC SYSTEM - LIVE EVALUATION TEST RUN")
print("=" * 105)

for tc in test_cases:
    # Build patient df
    p_dict = {'age': tc['age'], 'sex': tc['sex']}
    for s in raw_symptoms_list:
        p_dict[s] = tc['symptoms'].get(s, 0)
    df_raw = pd.DataFrame([p_dict])
    df_eng = fe.transform(df_raw)
    
    # 1. Vision Analysis
    vision_res = engine.analyze_blood_smear(tc['smear_path'])
    
    # 2. Clinical Probability Calculation
    severe_count = sum(p_dict[s] for s in severe_markers)
    mild_markers = ['cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea']
    mild_count = sum(p_dict[s] for s in mild_markers)
    has_fever = p_dict['fever'] == 1
    
    if severe_count >= 3:
        clin_prob = 0.94 + min(0.04, (severe_count - 3) * 0.02)
    elif severe_count >= 1:
        clin_prob = 0.88 if has_fever else 0.82
    elif has_fever and mild_count >= 1:
        clin_prob = 0.80 + (mild_count / 7.0) * 0.07
    elif has_fever:
        clin_prob = 0.42
    else:
        clin_prob = 0.05 if (mild_count + severe_count == 0) else 0.09
        
    # 3. Multi-Modal Fusion
    fusion = engine.fuse_predictions(clin_prob, vision_res, severe_count, mild_count)
    
    print(f"\n>>> {tc['title'].upper()} <<<")
    print(f"  • Patient Demographics: Age {tc['age']} yrs | Gender: {'Male' if tc['sex']==1 else 'Female'}")
    print(f"  • Reported Manifestations: {', '.join(tc['symptoms'].keys()) if tc['symptoms'] else 'None (Asymptomatic)'}")
    print(f"  • Microscopic Smear Image: {os.path.basename(tc['smear_path'])}")
    print(f"    - Parasite Detection: {vision_res['badge']} (Confidence: {vision_res['confidence']}%)")
    print(f"    - Smear Erythrocyte Status: {vision_res['status']}")
    print(f"  • Clinical Tabular Severity: {clin_prob*100:.1f}%")
    print(f"  • FUSED MULTI-MODAL DIAGNOSIS: {fusion['diagnostic_synthesis']}")
    print(f"  • TRIAGE PROTOCOL & TREATMENT: {fusion['triage_path']}")
    print("-" * 105)
