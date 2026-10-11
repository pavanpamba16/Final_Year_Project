import requests
import re

url = 'http://127.0.0.1:5000/predict'

test_cases = [
    {
        'desc': 'No symptoms (Healthy baseline)',
        'payload': {'age': 25, 'sex': 1}
    },
    {
        'desc': 'Headache only (No fever)',
        'payload': {'age': 25, 'sex': 1, 'headace': 1}
    },
    {
        'desc': '2 mild symptoms (Fever + Cold)',
        'payload': {'age': 25, 'sex': 1, 'fever': 1, 'cold': 1}
    },
    {
        'desc': 'Classic Active Malaria (Fever + Cold + Rigors)',
        'payload': {'age': 25, 'sex': 1, 'fever': 1, 'cold': 1, 'rigor': 1}
    },
    {
        'desc': 'Severe Malaria (Fever + Convulsions + Jaundice)',
        'payload': {'age': 25, 'sex': 1, 'fever': 1, 'Convulsion': 1, 'jundice': 1}
    },
    {
        'desc': 'Multi-modal: Parasite Confirmed Smear + Fever & Rigors',
        'payload': {'age': 25, 'sex': 1, 'fever': 1, 'rigor': 1, 'sample_smear': 'parasitized_sample_1.png'}
    },
    {
        'desc': 'Multi-modal: Negative Smear + Fever & Headache',
        'payload': {'age': 25, 'sex': 1, 'fever': 1, 'headace': 1, 'sample_smear': 'uninfected_sample_1.png'}
    }
]

print("=" * 110)
print(f"{'Patient Presentation':52s} | {'Score':>6} | {'PRIMARY MALARIA VERDICT'}")
print("=" * 110)

for tc in test_cases:
    r = requests.post(url, data=tc['payload'])
    text = r.text
    
    score_m = re.search(r'<span class="probability-display">([0-9.]+)%</span>', text)
    score_str = score_m.group(1) if score_m else 'N/A'
    
    verdict_m = re.search(r'<h3 class="verdict-main-heading">\s*([^\n\r<]+)\s*</h3>', text)
    verdict_str = verdict_m.group(1).strip() if verdict_m else 'N/A'
    
    print(f"{tc['desc']:52s} | {score_str:>5}% | {verdict_str}")

print("=" * 110)
