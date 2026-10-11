import requests
from bs4 import BeautifulSoup

test_cases = [
    ('Uninfected Smear + Fever/Headache', {'age': '28', 'fever': '1', 'headace': '1', 'sample_smear': 'uninfected_sample_1.png'}),
    ('Uninfected Smear + Convulsions', {'age': '6', 'fever': '1', 'Convulsion': '1', 'sample_smear': 'uninfected_sample_1.png'}),
    ('Parasitized Smear + Fever/Headache', {'age': '28', 'fever': '1', 'headace': '1', 'sample_smear': 'parasitized_sample_1.png'}),
    ('Parasitized Smear + Convulsions', {'age': '6', 'fever': '1', 'Convulsion': '1', 'sample_smear': 'parasitized_sample_1.png'}),
    ('Symptoms Only + Fever/Headache', {'age': '28', 'fever': '1', 'headace': '1', 'sample_smear': ''}),
    ('Symptoms Only + Convulsions', {'age': '6', 'fever': '1', 'Convulsion': '1', 'sample_smear': ''}),
]

for name, d in test_cases:
    r = requests.post('http://127.0.0.1:5000/predict', data=d)
    soup = BeautifulSoup(r.text, 'html.parser')
    diag = soup.select_one('.diagnosis-header h4')
    diag_text = diag.get_text().strip() if diag else 'N/A'
    
    badge = soup.select_one('.diagnosis-status-pill')
    badge_text = badge.get_text().strip() if badge else 'N/A'
    
    score_tag = soup.select_one('.telemetry-grid .metric-item:nth-of-type(1) .metric-value')
    score_text = score_tag.get_text().strip() if score_tag else 'N/A'
    
    thresh_tag = soup.select_one('.telemetry-grid .metric-item:nth-of-type(4) .metric-value')
    thresh_text = thresh_tag.get_text().strip() if thresh_tag else 'N/A'
    
    rec = soup.select_one('.diagnosis-recommendation')
    rec_text = rec.get_text().strip() if rec else 'N/A'

    print(f"=== {name} ===")
    print(f"  Score: {score_text} | Threshold: {thresh_text}")
    print(f"  Triage Pill: {badge_text}")
    print(f"  Diagnosis: {diag_text}")
    print(f"  Rec: {rec_text[:70]}...")
    print()
