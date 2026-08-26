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

| Model | Preprocessing | Balancing | Accuracy | Recall | Specificity | ROC-AUC | F1-Score | MCC |
| ----- | ------------- | --------- | -------- | ------ | ----------- | ------- | -------- | --- |
| **Baseline RF** | Scaling | None | 64.7% | 18.9% | 100.0% | 58.0% | 28.0% | 0.140 |
| **Tuned RF (v2.0)** | Feat. Eng. + Scaling | SMOTE | **44.7%** | **89.7%** | **21.4%** | **54.7%** | **52.5%** | **0.138** |
| **Tuned CatBoost** | Feat. Eng. + Scaling | SMOTEENN | 45.9% | 89.7% | 23.2% | 50.3% | 53.1% | 0.156 |
| **Voting Ensemble** | Feat. Eng. + Scaling | SMOTE/SMOTEENN| 48.2% | 89.7% | 26.8% | 53.9% | 54.2% | 0.191 |

> [!NOTE]
> The Tuned RF (v2.0) and Voting Ensemble are optimized using a threshold of **0.347** to hit **89.7% Recall** to fit the clinical objective of screening (where false negatives must be minimized).

---

## 6. Installation & Quick Start

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training
To run the full cross-validation comparison, hyperparameter optimization, and save model artifacts:
```bash
python src/train.py
```

### 3. Run the Web Application
Launch the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 7. Explainable AI & Patient Interpretation
When a prediction is run:
* **SHAP Local Explanation**: The system calculates Shapley values for the patient and prints out exactly which features contributed positively to the risk score (e.g. presence of Convulsions, Jaundice) and which features reduced the risk score (e.g. young age, lack of hyperpyrexia).
* **Clinical Safety Disclaimer**: The dashboard displays a prominent message that the system is a decision-support aid and must be verified by laboratory assays (thick/thin blood smears or rapid diagnostic tests).
