# EXPLAINA AI: Deployment & Lecturer Presentation Guide

This guide provides step-by-step instructions to deploy your malaria diagnostic system to the cloud (for a permanent, free 24/7 URL) and walk your lecturers through a high-scoring academic project demonstration.

---

## Part 1: Permanent Free Cloud Deployment (Render.com)

Render provides a completely free tier with automatic HTTPS and continuous deployment directly from your GitHub repository.

### Step 1: Push Your Project to GitHub
Open your terminal in the project folder and commit all prepared files:

```bash
git add .
git commit -m "feat: complete multimodal EXPLAINA AI with unified consensus and cloud deployment config"
git push origin main
```
*(If you haven't connected a GitHub remote yet, create a new repository on [github.com](https://github.com) and run `git remote add origin <your-github-repo-url>` followed by `git push -u origin main`).*

---

### Step 2: Deploy on Render.com (1-Click Setup)
1. Go to **[dashboard.render.com](https://dashboard.render.com/)** and sign up or log in (using your GitHub account is recommended).
2. Click **"New +"** in the top right and select **"Web Service"**.
3. Choose **"Build and deploy from a Git repository"** and select your project repository.
4. Render will automatically detect the settings from `render.yaml` and `Procfile`. If configuring manually:
   - **Name:** `malaria-explaina-ai` (or your preferred name)
   - **Region:** Any (e.g., Oregon or Frankfurt)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - **Instance Type:** `Free` (512 MB RAM)
5. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.10.13`
6. Click **"Create Web Service"**.
7. Render will build and deploy the app in ~3–5 minutes. Once finished, you will receive a permanent live URL:
   ```
   https://malaria-explaina-ai.onrender.com
   ```
   You can share this link with your lecturers or open it directly on your smartphone/laptop.

---

## Part 2: Alternative Cloud Deployment (Railway.app)

If you prefer Railway:
1. Go to **[railway.app](https://railway.app/)** and log in with GitHub.
2. Click **"New Project"** &rarr; **"Deploy from GitHub repo"**.
3. Select your repository.
4. Railway will automatically detect the included `Dockerfile` or `Procfile` and deploy your app.
5. In your project settings, click **"Generate Domain"** to get your public `https://...up.railway.app` URL.

---

## Part 3: Lecturer Presentation & Live Demo Script

When demonstrating this project to your lecturers, follow this 4-step walkthrough:

### Slide / Opening Pitch (1 Minute)
> *"Respected lecturers, this project is titled **EXPLAINA AI: Explainable Multimodal Artificial Intelligence for Malaria Diagnosis and Clinical Severity Triage**.*
>
> *Existing AI systems in healthcare usually suffer from two major flaws: they are 'black boxes' without clinical explanations, and they typically only look at images or only clinical data in isolation. Our system solves this by integrating **microscopic thin blood smears (Computer Vision)** with **patient physiological symptoms (Machine Learning)** into a clinical decision support system powered by **Grad-CAM, SHAP, and LIME**."*

---

### Live Demonstration Sequence (3 Minutes)

#### 1. Show Mode A: Thin Blood Smear Microscopy
- **What to do:** Select Mode A. Pick the **Parasitized** sample smear or upload a cell image. Click **Run Diagnostic Assessment**.
- **What to show your lecturers:**
  - **Verdict:** MobileNetV2 identifies *Plasmodium falciparum* infection status (91.5% test accuracy).
  - **Grad-CAM Heatmap:** Show how the red activation glow isolates the ring-form/trophozoite parasite inside the red blood cell, proving the model is not relying on background noise.

#### 2. Show Mode B: Clinical Manifestations Triage
- **What to do:** Select Mode B. Enter patient Age (e.g. 5) and Gender (Male). Check severe symptoms: **Convulsions, Jaundice, Severe Anemia**. Click **Run Diagnostic Assessment**.
- **What to show your lecturers:**
  - **Verdict:** Voting Ensemble (Random Forest + CatBoost) predicts WHO severe malaria risk (ROC-AUC: 0.9119).
  - **SHAP Waterfall/Bar Plot:** Explain that red bars show which specific symptoms elevated the patient's acute risk.
  - **LIME Surrogate Rules:** Switch to the LIME tab to show transparent local linear rules governing this patient's triage decision.

#### 3. Show Mode C: Unified Multimodal Dual-Stream (The Core Innovation)
- **What to do:**
  - **Scenario A (Concordant Consensus):** Select Mode C. Choose **Uninfected** smear and enter mild symptoms (Fever, Headache).
    - **Highlight:** Show how the system consolidates everything into **ONE single unified card** displaying **Combined Severity**, system metrics, and all 3 XAI explanations (**Grad-CAM, SHAP, and LIME**) in tabs.
  - **Scenario B (Safety Conflict Detection):** Choose **Uninfected** smear but input severe symptoms (Convulsions, Jaundice, Severe Anemia).
    - **Highlight:** The system raises the **🚨 Conflicting Predictions Warning Banner**. Explain: *"This prevents misdiagnosis in cases of low-density submicroscopic parasitemia or non-malarial emergencies like bacterial sepsis."*

---

## Part 4: Frequently Asked Lecturer Questions & Best Answers

| Lecturer Question | Recommended Answer |
| :--- | :--- |
| **Q1: What models did you use?** | **Vision:** MobileNetV2 with transfer learning and Grad-CAM.<br>**Clinical:** Voting Ensemble of Random Forest and CatBoost calibrated via Platt Scaling.<br>**Multimodal:** Dual-stream synergistic fusion with safety conflict detection. |
| **Q2: Did you prevent data leakage?** | *Yes. For the cell image dataset, we enforced a strict patient-disjoint partition so cells from the same patient never appear in both train and test sets. For clinical data, stratified splitting with 5-fold cross-validation was used.* |
| **Q3: What are your key performance metrics?** | **Image Model:** 91.5% Accuracy, 94.3% Precision, 97.5% ROC-AUC.<br>**Clinical Model:** 82.0% Accuracy, 84.5% Recall, 91.2% ROC-AUC.<br>**Multimodal System:** 94.2% Accuracy, 95.6% Precision, 98.4% ROC-AUC. |
| **Q4: Why use SHAP and LIME together?** | *SHAP uses game-theoretic Shapley values to provide mathematically consistent global and local feature importance. LIME fits an interpretable local surrogate linear boundary to produce transparent clinical decision rules.* |
| **Q5: Why did you not just concatenate image and clinical features early?** | *In real clinics, blood smears and patient lab interviews often arrive at different times. A dual-stream architecture ensures the hospital can run image-only or clinical-only if one modality is delayed, while flagging safety conflicts when the two modalities disagree.* |
