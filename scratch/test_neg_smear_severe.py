import urllib.request
import urllib.parse
import re

raw_cols = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
form_fields = {
    'age': 30,
    'sex': 0,
    'sample_smear': 'uninfected_sample_1.png'
}
for s in raw_cols:
    form_fields[s] = 1

data = urllib.parse.urlencode(form_fields).encode()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, method='POST')
html = urllib.request.urlopen(req).read().decode('utf-8')

prob_match = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html)
verdict_match = re.search(r'<span class="diagnosis-status-pill">([^<]+)</span>', html)

print('With Negative Smear + All Severe Symptoms: Score =', prob_match.group(1) if prob_match else 'None')
print('Triage Status:', verdict_match.group(1) if verdict_match else 'None')
