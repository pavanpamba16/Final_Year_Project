import requests
import re

base_url = 'http://127.0.0.1:5000/predict'

test_cases = [
    ('All zeros (healthy baseline)', {}),
    ('Headache only (no fever)', {'headace': 1}),
    ('Headache + Fatigue (2 symptoms without fever)', {'headace': 1, 'fatigue': 1}),
    ('Fever only (1 symptom)', {'fever': 1}),
    ('Fever + Cold (2 symptoms)', {'fever': 1, 'cold': 1}),
    ('Fever + Headache (2 symptoms)', {'fever': 1, 'headace': 1}),
    ('Fever + Cold + Rigor (3 symptoms: Triad)', {'fever': 1, 'cold': 1, 'rigor': 1}),
    ('Classic Malaria (Fever+Cold+Rigor+Headache+BitterTongue+Vomiting)', {'fever': 1, 'cold': 1, 'rigor': 1, 'headace': 1, 'bitter_tongue': 1, 'vomitting': 1}),
    ('Convulsions only', {'Convulsion': 1}),
    ('Anemia only', {'Anemia': 1}),
    ('Jaundice only', {'jundice': 1}),
    ('Coca-cola urine only', {'cocacola_urine': 1}),
    ('Hypoglycemia only', {'hypoglycemia': 1}),
    ('Prostration only', {'prostraction': 1}),
    ('Hyperpyrexia only', {'hyperpyrexia': 1}),
    ('Fever + Convulsions', {'fever': 1, 'Convulsion': 1}),
    ('Fever + Convulsions + Jaundice', {'fever': 1, 'Convulsion': 1, 'jundice': 1}),
    ('Critical: 5 severe markers', {'Convulsion': 1, 'jundice': 1, 'Anemia': 1, 'cocacola_urine': 1, 'hypoglycemia': 1})
]

print(f"{'Test Case':65s} | {'Score':>7} | {'Badge':40s} | {'Triage'}")
print("-" * 140)

for name, s_dict in test_cases:
    payload = {'age': 25, 'sex': 0, 'sample_smear': ''}
    for col in ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']:
        payload[col] = s_dict.get(col, 0)
    
    r = requests.post(base_url, data=payload)
    text = r.text
    
    prob_m = re.search(r'<span class="probability-display">([0-9.]+)%</span>', text)
    prob_str = prob_m.group(1) if prob_m else 'N/A'
    
    badge_m = re.search(r'<div class="risk-badge[^>]*>\s*([^\n\r<]+)\s*</div>', text)
    badge_str = badge_m.group(1).strip() if badge_m else 'N/A'
    
    triage_m = re.search(r'<span class="diagnosis-status-pill">([^<]+)</span>', text)
    triage_str = triage_m.group(1).strip() if triage_m else 'N/A'
    
    print(f"{name:65s} | {prob_str:>6}% | {badge_str:40s} | {triage_str}")
