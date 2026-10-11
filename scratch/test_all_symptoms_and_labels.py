import urllib.request
import urllib.parse
import re

# Form with all symptoms checked and default sample_smear="" (Symptoms Only)
raw_cols = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
form_fields = {
    'age': 30,
    'sex': 0,
    'sample_smear': ''
}
for s in raw_cols:
    form_fields[s] = 1

data = urllib.parse.urlencode(form_fields).encode()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, method='POST')
html = urllib.request.urlopen(req).read().decode('utf-8')

prob_match = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html)
verdict_match = re.search(r'<span class="diagnosis-status-pill">([^<]+)</span>', html)

print("Predicted Probability with All Symptoms:", prob_match.group(1) if prob_match else "Not found")
print("Status Pill:", verdict_match.group(1) if verdict_match else "Not found")
print("Contains 'Absence of Gender':", 'Absence of Gender' in html)
print("Contains 'Absence of Is Child':", 'Absence of Is Child' in html)
print("Contains 'Female Patient':", 'Female Patient' in html)
print("Contains 'Adult Patient':", 'Adult Patient' in html)
