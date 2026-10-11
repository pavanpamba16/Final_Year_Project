# EXPLAINA AI — Model Architectures & Implementations

This folder (`models/`) contains the complete source code, architectures, training routines, and saved weights for all machine learning and deep learning models used in **EXPLAINA AI**.

---

## Directory Overview

```
models/
├── __init__.py                   # Package initialization exposing all models
├── image_model.py                # MobileNetV2 architecture, training & Grad-CAM inference
├── clinical_model.py             # Voting Ensemble (RF + CatBoost), SHAP & LIME inference
├── multimodal_orchestrator.py    # Independent Dual-Stream Consensus & Polarity Synthesis
├── train_all.py                  # Master training script for both models
├── README.md                     # Model documentation (this file)
├── image/
│   └── malaria_mobilenet_v2.pt   # PyTorch checkpoint (91.5% Test Accuracy)
└── clinical/
    └── clinical_model.pkl        # Scikit-Learn / CatBoost package (91.2% Test ROC-AUC)
```

---

## 1. Modality 1: Deep Vision Microscopic Smear Model (`image_model.py`)

- **Architecture**: `MobileNetV2` with ImageNet transfer learning backbone and custom fine-tuned classification head (`Dropout(0.3) -> Linear(1280, 2)`).
- **Dataset**: NIH Malaria Cell Images (`data/cell_images/`) partitioned strictly at the **patient level** (70% train, 15% validation, 15% held-out test).
- **Target**: Intracellular erythrocyte infection status (`0: Uninfected`, `1: Parasitized`).
- **Input Dimension**: `(3, 224, 224)` normalized RGB.
- **Explainability**: **Grad-CAM (Gradient-weighted Class Activation Mapping)** computing spatial gradient heatmaps from the final inverted residual convolutional layer.
- **Held-Out Test Metrics**:
  - Accuracy: **91.5%**
  - Precision: **94.3%**
  - Sensitivity (Recall): **88.1%**
  - Specificity: **94.8%**
  - F1-Score: **91.1%**
  - ROC-AUC: **97.5%**

---

## 2. Modality 2: Clinical Risk Triage Model (`clinical_model.py`)

- **Architecture**: Soft Voting Ensemble combining:
  - `RandomForestClassifier` (150 estimators, balanced class weights)
  - `CatBoostClassifier` (150 iterations, depth 4, balanced weights)
- **Dataset**: `MultiCohort_Malaria_Clinical_Data.csv` (stratified 70/15/15 train/val/test split).
- **Target**: `severe_maleria` (`0: Non-severe / uncomplicated`, `1: WHO severe malaria organ complications`).
- **Input Features (17 total)**:
  - Demographics: `age`, `sex`
  - WHO Severe Criteria: `Convulsion`, `Anemia`, `jundice`, `cocacola_urine`, `hypoglycemia`, `prostraction`, `hyperpyrexia`
  - Systemic Manifestations: `fever`, `cold`, `rigor`, `fatigue`, `headace`, `bitter_tongue`, `vomitting`, `diarrhea`
- **Explainability**:
  - **SHAP (TreeExplainer)**: Computes local Shapley additive impact percentages showing which symptoms increase risk (red) vs. mitigate risk (green).
  - **LIME**: Fits a transparent local surrogate linear ridge model showing decisive decision rules.
- **Held-Out Test Metrics**:
  - Accuracy: **82.0%**
  - Precision: **73.1%**
  - Sensitivity (Recall): **84.5%**
  - Specificity: **80.9%**
  - ROC-AUC: **91.2%**
  - PR-AUC: **84.9%**

---

## 3. Multimodal Dual-Stream Consensus (`multimodal_orchestrator.py`)

- **Strategy**: Independent dual-stream concurrent inference.
- **Conflict Resolution**:
  - Detects **Concordant Polarity** (both positive or both negative).
  - Detects **Polarity Discrepancy** (e.g. Parasitized smear with mild symptoms, or Uninfected smear with severe systemic symptoms) and produces discrepancy analysis guidance.
- **Explainability Routing**:
  - Modality 1 is visually explained via **Grad-CAM**.
  - Modality 2 is feature-explained via **Clinical SHAP & LIME**.
- **Dual-Stream Benchmark Performance**:
  - Accuracy: **94.2%**
  - Precision: **95.6%**
  - Recall: **92.8%**
  - Specificity: **95.4%**
  - ROC-AUC: **98.4%**

---

## Quick Usage

```python
from models import MalariaVisionModel, MalariaClinicalModel, MultimodalOrchestrator

# 1. Image Model
vision_model = MalariaVisionModel()
img_result = vision_model.predict('path/to/smear.png')
print(img_result['predicted_class'], img_result['confidence'])

# 2. Clinical Model
clinical_model = MalariaClinicalModel()
clin_result = clinical_model.predict({
    'age': 32, 'sex': 1, 'fever': 1, 'rigor': 1, 'jundice': 1
})
print("Is Severe:", clin_result['is_severe'], "Probability:", clin_result['probability'])

# 3. Multimodal Orchestrator
orchestrator = MultimodalOrchestrator()
report = orchestrator.evaluate_mode_c('path/to/smear.png', {
    'age': 32, 'sex': 1, 'fever': 1, 'rigor': 1, 'jundice': 1
})
print("Has Conflict:", report['has_conflict'])
```
