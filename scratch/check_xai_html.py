import urllib.request
import urllib.parse
import re

params = {
    'age': 25,
    'sex': 1,
    'fever': 1,
    'Convulsion': 1,
    'cold': 1
}
data = urllib.parse.urlencode(params).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data)
resp = urllib.request.urlopen(req).read().decode('utf-8')

print("=== SHAP EXTRACTED FROM HTML ===")
matches = re.findall(r'<div class="xai-meta">\s*<span class="xai-name">(.*?)</span>\s*<span class="xai-val[^"]*">(.*?)</span>', resp)
for name, val in matches:
    print(f"  {name:45} : {val}")
