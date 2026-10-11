import urllib.request
import urllib.parse
import re
import pandas as pd

df = pd.read_csv('data/MultiCohort_Malaria_Clinical_Data.csv')
sample_indices = [15, 80, 200, 700]

for idx in sample_indices:
    row = df.iloc[idx]
    actual_label = row['severe_maleria']
    
    # Pack into POST form data
    fields = {'age': int(row['age']), 'sex': int(row['sex'])}
    for s in ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']:
        if int(row[s]) == 1:
            fields[s] = 1
            
    encoded_data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request('http://127.0.0.1:5000/predict', data=encoded_data, method='POST')
    html = urllib.request.urlopen(req).read().decode('utf-8')
    
    prob_m = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html)
    verdict_m = re.search(r'<span class="diagnosis-status-pill">([^<]+)</span>', html)
    
    print(f"Dataset Row #{idx}: Actual Target={actual_label} | Predicted Score={prob_m.group(1)}% | Status={verdict_m.group(1)}")
