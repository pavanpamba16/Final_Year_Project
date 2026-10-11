import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from app import app

def test_full_pipeline():
    client = app.test_client()

    print("=" * 60)
    print("TEST 1: MULTI-MODAL PREDICTION (PARASITIZED SAMPLE SMEAR + CLINICAL)")
    print("=" * 60)
    
    t0 = time.time()
    response = client.post('/predict', data={
        'age': '28',
        'sex': '0',
        'fever': '1',
        'cold': '1',
        'rigor': '1',
        'Convulsion': '1',
        'cocacola_urine': '1',
        'sample_smear': 'parasitized_sample_1.png'
    }, content_type='multipart/form-data')
    t1 = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Elapsed Time: {t1 - t0:.2f} seconds")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    html = response.data.decode('utf-8')
    assert "Unified SHAP Single Figure" in html
    assert "Unified LIME Single Figure" in html
    assert "data:image/png;base64," in html
    print("TEST 1 PASSED: Unified Multi-Modal figures rendered successfully!")

    print("\n" + "=" * 60)
    print("TEST 2: MULTI-MODAL PREDICTION (UNINFECTED SAMPLE SMEAR + CLINICAL)")
    print("=" * 60)
    
    t0 = time.time()
    response = client.post('/predict', data={
        'age': '35',
        'sex': '1',
        'fever': '1',
        'headace': '1',
        'sample_smear': 'uninfected_sample_1.png'
    }, content_type='multipart/form-data')
    t1 = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Elapsed Time: {t1 - t0:.2f} seconds")
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Unified SHAP Single Figure" in html
    assert "Unified LIME Single Figure" in html
    print("TEST 2 PASSED: Negative smear unified figures rendered successfully!")

    print("\n" + "=" * 60)
    print("TEST 3: CLINICAL SYMPTOMS ONLY (NO SMEAR IMAGE)")
    print("=" * 60)
    
    t0 = time.time()
    response = client.post('/predict', data={
        'age': '45',
        'sex': '0',
        'fever': '1',
        'cold': '1',
        'rigor': '1',
        'sample_smear': ''
    }, content_type='multipart/form-data')
    t1 = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Elapsed Time: {t1 - t0:.2f} seconds")
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "data:image/png;base64," in html
    print("TEST 3 PASSED: Pure clinical fallback figures rendered successfully!")

if __name__ == '__main__':
    test_full_pipeline()
