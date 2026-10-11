import urllib.request
import urllib.parse
import re

# Test 1: Random input with fever + cold + rigor
data1 = urllib.parse.urlencode({'age': 28, 'sex': 1, 'fever': 1, 'cold': 1, 'rigor': 1}).encode()
req1 = urllib.request.Request('http://127.0.0.1:5000/predict', data=data1, method='POST')
html1 = urllib.request.urlopen(req1).read().decode('utf-8')

# Test 2: Random input with multiple severe symptoms
data2 = urllib.parse.urlencode({'age': 40, 'sex': 0, 'fever': 1, 'Convulsion': 1, 'Anemia': 1, 'jundice': 1, 'cocacola_urine': 1}).encode()
req2 = urllib.request.Request('http://127.0.0.1:5000/predict', data=data2, method='POST')
html2 = urllib.request.urlopen(req2).read().decode('utf-8')

# Test 3: Random input with convulsion only
data3 = urllib.parse.urlencode({'age': 10, 'sex': 1, 'Convulsion': 1}).encode()
req3 = urllib.request.Request('http://127.0.0.1:5000/predict', data=data3, method='POST')
html3 = urllib.request.urlopen(req3).read().decode('utf-8')

prob1 = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html1)
prob2 = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html2)
prob3 = re.search(r'([0-9.]+)%\s*</span>\s*<span class="probability-out-of"', html3)

print("Test 1 (Fever+Cold+Rigor) Prob:", prob1.group(1) if prob1 else "Not found")
print("Test 2 (Multi-Severe) Prob:", prob2.group(1) if prob2 else "Not found")
print("Test 3 (Convulsion Only) Prob:", prob3.group(1) if prob3 else "Not found")

print("Contains Combined Accuracy:", "Combined Accuracy" in html1)
print("Contains Combined ROC-AUC:", "Combined ROC-AUC" in html1)
print("Contains Threshold:", "Threshold:" in html1)
print("Contains Decision Threshold:", "Decision Threshold" in html1)
print("Still contains Cutoff:", "Cutoff:" in html1 or "Model Cutoff" in html1)
