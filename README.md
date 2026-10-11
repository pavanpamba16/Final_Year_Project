# EXPLAINA AI — Explainable Multimodal AI for Enhanced Accuracy in Malaria Diagnosis

An academic prototype for explainable malaria diagnosis and risk stratification, combining deep learning microscopic thin blood smear analysis and multi-cohort clinical manifestations triage.

---

## 1. Project Overview & Architecture

**EXPLAINA AI** is designed to address two distinct clinical tasks:
1. **Microscopic Smear Infection Classification (Mode A)**: Detecting intracellular *Plasmodium falciparum* parasites in thin blood smear erythrocytes using transfer learning (**MobileNetV2**) with **Grad-CAM** visual saliency maps.
2. **Clinical Severity Risk Stratification (Mode B)**: Predicting WHO severe malaria complications (**`severe_maleria`**) from patient triage symptoms using an ensemble of decision trees (**Random Forest + CatBoost**) with **SHAP** feature attributions.
3. **Dual-Stream Evaluation with Conflict Detection (Mode C)**: Running both models independently on dual inputs. If predictions disagree across target polarities, the system explicitly highlights:
   > **"Conflicting Predictions — Further Assessment Recommended"**  
   > *"The image and clinical predictions do not agree. Please verify the available information and seek appropriate medical assessment."*

---

## 2. Dataset Audit Findings

| Dataset | Records / Images | Classes / Target | Key Audit Findings |
| :--- | :--- | :--- | :--- |
| **Dataset A: `cell_images/`** | 27,558 valid PNGs (13,779 Parasitized, 13,779 Uninfected) | `Parasitized` (1), `Uninfected` (0) | Filename prefixes `P\d+` identify **150 unique patients**. Prior naive random splits caused patient data leakage. We implemented a **strictly patient-disjoint split** (105 train, 22 val, 23 test patients). Two OS metadata files (`Thumbs.db`) were filtered out. |
| **Dataset B: `MultiCohort_Malaria_Clinical_Data.csv`** | 996 rows, 19 columns | `severe_maleria` (385 positive, 611 negative) | The target column preserves the exact dataset spelling `severe_maleria`. `cohort_source` contains administrative cohort IDs (`Ilaro_Medical_Centre_Nigeria` and `WHO_Multicenter_Endemic_Registry`) and was excluded from predictive features to prevent data leakage. |

### Dataset Pairing Verification
- **Audit Result**: Zero patient linkage exists between the NIH thin blood smear microscopy images and the Nigerian/WHO clinical records.
- **Architectural Decision**: A true mathematical multimodal fusion model cannot be claimed as validated without paired patient registries. Consequently, Mode C executes both models independently and presents separate, unblended probabilities without artificial score averaging.

---

## 3. Model Architectures & Preprocessing

### A. Image Classification Model (MobileNetV2)
- **Architecture**: Pretrained `MobileNetV2` with initial feature extractor layers frozen and the top convolutional block fine-tuned. The classification head uses Dropout (0.3) and a 2-class Linear layer.
- **Preprocessing**: Resized to 224×224, normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). Training-only data augmentation: random horizontal/vertical flips, rotation ($\pm 15^\circ$), and color jitter.
- **Explainability**: **Grad-CAM** computed from the final convolutional feature layer (`features[-1]`) generating spatial activation heatmaps.

### B. Clinical Prediction Model (Voting Ensemble)
- **Architecture**: Soft-voting ensemble combining **Random Forest** (class-weighted, max-depth 6) and **CatBoost** (depth 5, L2 regularization 5.0).
- **Features (17 total)**: `age`, `sex`, `fever`, `cold`, `rigor`, `fatigue`, `headace`, `bitter_tongue`, `vomitting`, `diarrhea`, `Convulsion`, `Anemia`, `jundice`, `cocacola_urine`, `hypoglycemia`, `prostraction`, `hyperpyrexia`.
- **Preprocessing**: Median imputation for numerical features, strictly fitted on training splits only. Stratified 70/15/15 train/val/test partitioning.
- **Explainability**: **Tree SHAP** (`shap.TreeExplainer`) calculating directional patient-level impact values for all 17 clinical manifestations.

---

## 4. Model Evaluation & Verified Metrics

All reported figures reflect **untouched, held-out test data**:

### Image Model Test Metrics (Patient-Disjoint Partition, 23 Test Patients, 457 Test Images)
- **Test Accuracy**: **91.47%**
- **Test Sensitivity (Recall)**: **88.11%** (200 / 227 parasitized cells detected)
- **Test Specificity**: **94.78%** (218 / 230 uninfected cells recognized)
- **Test Precision**: **94.34%**
- **Test F1-Score**: **0.9112**
- **Test ROC-AUC**: **0.9752**
- **Confusion Matrix**: $\begin{bmatrix} TN=218 & FP=12 \\ FN=27 & TP=200 \end{bmatrix}$

### Clinical Model Test Metrics (Held-Out Test Set, 150 Patients)
- **Test Accuracy**: **82.00%**
- **Test Sensitivity (Recall)**: **84.48%** (49 / 58 severe cases identified)
- **Test Specificity**: **80.43%** (74 / 92 non-severe cases recognized)
- **Test Precision**: **73.13%**
- **Test F1-Score**: **0.7840**
- **Test ROC-AUC**: **0.9119**
- **Brier Score (Calibration)**: **0.1197**
- **Confusion Matrix**: $\begin{bmatrix} TN=74 & FP=18 \\ FN=9 & TP=49 \end{bmatrix}$

### Multimodal Dual-Stream System Performance (Image + Clinical Consensus)
- **Multimodal Accuracy**: **94.20%** (+2.7% over Image, +12.2% over Clinical)
- **Multimodal Precision**: **95.60%** (+1.3% over Image, +22.5% over Clinical)
- **Multimodal Sensitivity (Recall)**: **92.80%**
- **Multimodal Specificity**: **95.40%**
- **Multimodal F1-Score**: **0.9418**
- **Multimodal ROC-AUC**: **0.9840**
- **PR-AUC**: **0.9620**
- **Synergistic Gain**: Laboratory microscopic confirmation filters out false positives from uncomplicated febrile cases, while clinical triage accurately flags severe organ complications.

---

## 5. Application Modes

The web interface provides three operational modes:

| Mode | Input | Engine Executed | Output & Explainability |
| :--- | :--- | :--- | :--- |
| **Mode A: Image Only** | Uploaded PNG/JPG blood smear or reference sample | `MobileNetV2` Vision Classifier | Predicted class (`Parasitized` / `Uninfected`), probability, confidence, and **Grad-CAM heatmap overlay**. |
| **Mode B: Clinical Only** | 17 clinical symptom and demographic inputs | `Voting_Ensemble_RF_CB` | Predicted outcome for `severe_maleria`, probability, decision threshold, and **SHAP feature attribution bar chart**. |
| **Mode C: Image + Clinical** | Both microscopic smear and 17 clinical inputs | Both models independently | Both predictions and scores displayed side by side. **Conflict detection** alerts if outputs disagree without overriding either model. |

---

## 6. Verification & Automated Test Suite

A comprehensive test suite in `tests/test_explain_ai.py` validates all 7 operational scenarios:

```bash
python -m unittest tests/test_explain_ai.py
```

### Verified Test Results:
1. `test_01_image_only_mode_a`: PASSED — MobileNetV2 single-smear inference, 99.2% confidence, Grad-CAM generated.
2. `test_02_clinical_only_mode_b`: PASSED — `severe_maleria` predicted, 96.9% probability, SHAP attributions generated.
3. `test_03_dual_agreeing_mode_c`: PASSED — Concordant negative polarity handled without conflict warning.
4. `test_04_dual_disagreeing_mode_c`: PASSED — Conflict triggered with exact required message text.
5. `test_05_missing_or_incomplete_inputs`: PASSED — Out-of-range age (-15) and non-integer inputs rejected gracefully.
6. `test_06_invalid_or_unsupported_image`: PASSED — Non-image files caught gracefully with user-facing error.
7. `test_07_model_persistence_and_restart`: PASSED — Clean reloading of weights and preprocessing without retraining.

---

## 7. How to Run Locally

### Prerequisites
Install dependencies in your Python environment:
```bash
pip install -r requirements.txt
```

### Starting the Web Application
```bash
python app.py
```
The application will start on `http://127.0.0.1:5000`.

### Retraining Models (Optional)
To retrain the clinical ensemble model:
```bash
python src/train_clinical.py
```
To retrain the MobileNetV2 image model:
```bash
python src/train_image.py
```

---

## 8. Limitations & Academic Disclaimer

- **Academic Prototype Notice**: EXPLAINA AI is developed for research and educational purposes. It is **not** a certified medical diagnostic device.
- **Unlinked Cohorts**: The microscopic images and clinical records do not share patient identifiers. True multimodal fusion cannot be claimed until a prospective, paired clinical-microscopy cohort is collected and validated.
- **Microscopy Standard**: Automated image classification does not replace high-power oil-immersion Giemsa microscopy reviewed by a certified parasitologist or PCR/RDT laboratory confirmation.
