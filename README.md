# Advanced Explainable Malaria Prediction System (v2.0)

An upgraded, clinically robust, and explainable Machine Learning system for classifying **severe malaria risk** based on patient demographics and clinical manifestations.

---

## 1. Project Overview & Problem Statement
Malaria remains a leading cause of global morbidity and mortality. Early stratification of patients at risk of developing severe malaria is critical for timely clinical intervention and resource allocation. 

### Critical Audit & Baseline Limitations
In **Version 1.0** of this project, a Random Forest model was reported to achieve an accuracy of **81.95%** and a ROC-AUC of **86.96%**. 
* **The Fatal Flaw (Data Leakage)**: Our technical audit revealed that oversampling (SMOTE/Random Oversampling) was applied to the *entire dataset* **before** performing the train/test split. Because samples from the minority class were duplicated before splitting, identical records were present in both the training and test sets. This caused the model to memorize samples, leading to an artificially inflated accuracy.
* **The Reality**: When evaluated correctly using a standard train/test split *without* data leakage, the baseline Random Forest model's accuracy drops to **64.7%** and its **Recall (Sensitivity) collapses to only 18.9%**. A sensitivity of 18.9% is clinically dangerous, as it misses over 80% of severe cases.

---

## 2. Advanced Upgrades (Version 2.0)
To establish a scientifically sound and clinically viable tool, the following upgrades were implemented:
1. **Leakage-Free Resampling Pipeline**: All data preprocessing (imputation, scaling, SMOTE/SMOTEENN) is encapsulated inside an `imblearn` pipeline, ensuring resampling is strictly fitted *only* on the training folds during cross-validation.
2. **Medical Feature Engineering**: Derived features based on clinical guidelines:
   - `symptom_count`: Total clinical manifestations.
   - `severe_symptom_count`: Count of WHO-defined severe malaria indicators (Convulsions, Anemia, Jaundice, Coca-Cola Urine, Hypoglycemia, Prostration, Hyperpyrexia).
   - `severe_score`: Weighted index where severe symptoms count double.
   - *Interactions*: Fever + Rigor (classic presentation), Fever + Headache, Anemia + Fatigue, Vomiting + Diarrhea.
   - *Pediatric/Geriatric Risk Groups*: Categorized age indicators.
3. **Optimized Decision Threshold**: We shifted the decision threshold from `0.50` to `0.347` based on validation data to maximize **Recall (Sensitivity) to 89.7%** (only 3 false negatives on the test set), accepting a trade-off in specificity to prioritize patient safety.
4. **Explainable AI Integration**: Patient-level explanations are computed on-the-fly using **SHAP (Shapley Additive exPlanations)**, highlighting the top contributing factors that increase or decrease severe risk.
5. **Interactive UI**: A modern, glassmorphic clinical dashboard built with Flask and Vanilla CSS.

---

## 3. Technology Stack
* **Language**: Python 3.x
* **Core ML**: Scikit-Learn, Imbalanced-Learn, XGBoost, CatBoost
* **Explainability**: SHAP, LIME
* **Backend**: Flask
* **Frontend**: HTML5, Vanilla CSS3 (modern dark UI, glassmorphism, responsive layout)

---

## 4. Project Structure
```text
Final-Project/
├── data/
│   └── Malaria-Data.csv               <- Original dataset
├── models/
│   ├── baseline/                      <- Saved baseline model (Simple RF, non-leaked)
│   └── improved/                      <- Upgraded production model package
├── notebooks/                         <- Original research notebook and copies
├── src/
│   ├── preprocessing.py               <- Pipeline operations
│   ├── feature_engineering.py         <- Medical feature indicators
│   ├── feature_selection.py           <- Correlation and importance selectors
│   ├── train.py                       <- Pipeline cross-validation & training
│   ├── evaluate.py                    <- Metrics computations and plotting
│   └── explain.py                     <- SHAP and LIME local explainers
├── templates/
│   ├── index.html                     <- Main symptom entry dashboard
│   └── result.html                    <- Interactive AI risk report
├── static/
│   └── css/
│       └── style.css                  <- Premium custom CSS style
├── app.py                             <- Flask backend app
├── experiments.csv                    <- Experiment tracking log
├── requirements.txt                   <- Project dependencies
└── README.md                          <- This document
```

---

## 5. Model Evaluation & Comparison

Tested on a strictly untouched 25% test set:

| Model | Preprocessing | Balancing | Accuracy | Recall | Specificity | ROC-AUC | F1-Score |
| ----- | ------------- | --------- | -------- | ------ | ----------- | ------- | -------- |
| **Validated Random Forest (v2.1)** | Feat. Eng. + Scaling | Balanced | **84.2%** | **84.5%** | **83.9%** | **91.7%** | **85.1%** |
| **CatBoost Classifier** | Feat. Eng. + Scaling | Balanced | 83.5% | 84.5% | 82.3% | 91.9% | 84.5% |
| **Baseline RF (Unbalanced)** | Scaling Only | None | 64.7% | 18.9% | 100.0% | 58.0% | 28.0% |

> [!NOTE]
> The Validated Random Forest model achieves high, balanced clinical accuracy (84.2%), high specificity (83.9%), high recall (84.5%), and strong discrimination (91.7% ROC-AUC) using the standard calibrated decision threshold (0.50).

---

## 6. Installation & Quick Start

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
Launch the Flask server:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 7. Explainable AI & Patient Interpretation (SHAP & LIME)
When a prediction is run:
* **SHAP Local Explanation**: Computes Shapley additive values for the patient and highlights top positive drivers (e.g. Convulsions, Jaundice, Coca-Cola Urine) and mitigating factors.
* **LIME Local Explanation**: Computes local rule-based feature impacts through tabular perturbation, outputting clear interpretable rules and weights in plain clinical terminology.

---

---

## 8. Multi-Modal Vision Fusion (Implemented)
To achieve gold-standard laboratory confirmation alongside clinical triage, the system includes an end-to-end **Multi-Modal Vision Fusion Architecture**:

### 1. Dataset & Microscopic Imaging
* **Dataset**: Authentic NIH Giemsa-Stained Thin Blood Smear Microscopic Erythrocyte Dataset (`data/cell_images/Dataset/`).
* **Classes**:
  * `Parasitized` ($P. falciparum$ ring-stage and schizont infected erythrocytes)
  * `Uninfected` (Healthy normal red blood cells)
* **Dataset Partitioning**: 416 training images, 134 strictly unseen test images.

### 2. Deep Vision CNN Architecture (`src/train_vision.py`)
* **3-Block Convolutional Backbone**:
  * Block 1: `Conv2d(3, 32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(32, 32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
  * Block 2: `Conv2d(32, 64)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(64, 64)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
  * Block 3: `Conv2d(64, 128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(128, 128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
  * Global Aggregation: `AdaptiveAvgPool2d((1, 1))`
  * Classifier Head: `Dropout(0.4)` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(64, 2)`
* **Training Hyperparameters**:
  * Optimizer: Adam ($\text{lr} = 0.001$, weight decay = $1\times 10^{-4}$)
  * Loss: CrossEntropyLoss
  * Image Resolution: $64 \times 64$ RGB with normalized tensor transform.

### 3. Microscopic Test Evaluation Metrics
Tested on 134 unseen microscopic blood smear images:
* **Accuracy**: **97.76%**
* **Sensitivity / Recall**: **96.70%** (Detects 88 out of 91 infected blood smears)
* **Specificity**: **100.00%** (Zero false positives on healthy blood smears)
* **Precision**: **100.00%**
* **ROC-AUC**: **99.87%**

### 4. Mathematical Clinical Triangulation (`src/multimodal_fusion.py`)
Fuses image parasitemia evidence ($P_{\text{parasite}}$) with clinical manifestation severity ($P_{\text{clinical}}$):
1. **Microscopy (+) + WHO Severe Complications**: Confirmed Severe Malaria ($P_{\text{fused}} \in [78\%, 98.5\%]$) $\rightarrow$ *Emergency Parenteral IV Artesunate*.
2. **Microscopy (+) + Mild Symptoms**: Confirmed Uncomplicated Malaria ($P_{\text{fused}} \in [25\%, 38\%]$) $\rightarrow$ *Outpatient Oral ACT (e.g. Coartem)*.
3. **Microscopy (-) + Severe Symptoms**: Severe Non-Malarial Syndrome ($P_{\text{fused}} = 25\%$) $\rightarrow$ *Urgent differential for Bacterial Sepsis / Meningitis*.
4. **Microscopy (-) + Mild Symptoms**: Non-Malarial Febrile Illness ($P_{\text{fused}} = 10\%$) $\rightarrow$ *Outpatient differential for Viral Syndrome / Typhoid*.

---

## 9. Future Work & Academic Publication Scope (Final Year Project)
For academic paper publication (e.g., IEEE JBHI, Elsevier AIIM, Nature Sci Rep) and final year project defense, the following research extensions remain active:

1. **Edge IoMT Offline Mobile Deployment**:
   - Applying INT8 post-training quantization to export models to ONNX and TFLite formats for offline mobile diagnostic execution on low-cost Android smartphones in remote rural clinics without internet connectivity.
2. **Point-of-Care (PoC) Continuous Biomarkers**:
   - Expanding input parameters to continuous quantitative biomarkers (Parasite density per $\mu\text{L}$, hemoglobin levels, blood lactate, and automated lateral-flow RDT optical strip densitometry).
3. **Federated Learning & Cross-Cohort Multi-Hospital Validation**:
   - Implementing privacy-preserving Federated Learning (FedAvg with Differential Privacy) across multi-center international hospital registries to validate generalizability across diverse endemic zones.
4. **Electronic Health Record (EHR) & HL7 FHIR Interoperability**:
   - Exposing certified HL7 FHIR APIs to seamlessly integrate automated severe malaria alerts directly into hospital triage systems.

> For the comprehensive thesis abstract, mathematical formulation, defense viva Q&A cheatsheet, and journal submission guide, see [ACADEMIC_PUBLICATION_GUIDE.md](ACADEMIC_PUBLICATION_GUIDE.md).


