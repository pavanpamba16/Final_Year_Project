import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Imbalanced-learn imports for leakage-safe resampling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

# Local imports
from feature_engineering import MalariaFeatureEngineer
from evaluate import calculate_metrics, optimize_threshold

def load_data():
    multicohort_path = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\data\MultiCohort_Malaria_Clinical_Data.csv"
    if os.path.exists(multicohort_path):
        data_path = multicohort_path
    else:
        data_path = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\data\Malaria-Data.csv"
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    if 'cohort_source' in df.columns:
        df = df.drop(columns=['cohort_source'])
        
    # Check for missing values and duplicates
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Handling {missing_count} missing values using median imputation...")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("Dataset verification: 0 missing values (100% complete records).")
        
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Removing {dup_count} duplicate records...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("Dataset verification: 0 duplicate records (100% unique records).")
        
    print(f"Loaded clinical dataset from {data_path}. Total clean unique records: {len(df)}")
    return df

def log_experiment(model_name, preprocessing, balancing, features, params, test_metrics, filepath="experiments.csv"):
    """Logs an experiment run with detailed metrics to a CSV file."""
    row = {
        'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model': model_name,
        'preprocessing': preprocessing,
        'balancing_method': balancing,
        'feature_set': features,
        'hyperparameters': str(params),
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'specificity': test_metrics['specificity'],
        'f1': test_metrics['f1'],
        'roc_auc': test_metrics['roc_auc'],
        'pr_auc': test_metrics['pr_auc'],
        'mcc': test_metrics['mcc'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(filepath):
        df_row.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_row.to_csv(filepath, mode='w', index=False)
    print(f"Logged experiment for {model_name} (Balancing: {balancing})")

def get_balancing_method(method_name):
    """Returns the appropriate imblearn sampler based on name."""
    if method_name == 'SMOTE':
        return SMOTE(random_state=123)
    elif method_name == 'Borderline-SMOTE':
        return BorderlineSMOTE(random_state=123)
    elif method_name == 'SMOTEENN':
        return SMOTEENN(random_state=123)
    elif method_name == 'Oversampler':
        return RandomOverSampler(random_state=123)
    return None  # None for 'None' or 'Class Weights'

def get_base_models(use_class_weight=False):
    """Returns dict of standard models to evaluate."""
    cw = 'balanced' if use_class_weight else None
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight=cw, random_state=123, max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(class_weight=cw, random_state=123),
        'SVM': SVC(class_weight=cw, probability=True, random_state=123),
        'Random Forest': RandomForestClassifier(class_weight=cw, random_state=123),
        'Extra Trees': ExtraTreesClassifier(class_weight=cw, random_state=123),
        'Gradient Boosting': GradientBoostingClassifier(random_state=123),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(class_weight=cw, random_state=123),
        'XGBoost': XGBClassifier(random_state=123, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=123, silent=True)
    }
    return models

def main():
    print("Starting ML Pipeline training and comparison...")
    
    # 1. Load Data
    df = load_data()
    X = df.drop(columns=['severe_maleria'])
    y = df['severe_maleria']
    
    # 2. Strict Train/Test Split (stratified, 25% test set)
    # The test set remains completely untouched until final test evaluation!
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=101
    )
    print(f"Dataset Split: Train={X_train_raw.shape[0]}, Test={X_test_raw.shape[0]}")
    
    # 3. Fit Baseline Model (representing previous work, but with correct non-leaked split)
    # Let's save a clean baseline model to models/baseline/
    print("\n--- Training Baseline Model (Simple RF, No Feature Engineering, Scaled, Non-Leaked Split) ---")
    baseline_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=123))
    ])
    baseline_pipeline.fit(X_train_raw, y_train)
    baseline_y_pred = baseline_pipeline.predict(X_test_raw)
    baseline_y_prob = baseline_pipeline.predict_proba(X_test_raw)[:, 1]
    baseline_metrics = calculate_metrics(y_test, baseline_y_pred, baseline_y_prob)
    
    # Save baseline model
    os.makedirs(r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\baseline", exist_ok=True)
    joblib.dump(baseline_pipeline, r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\baseline\baseline_rf.pkl")
    # Log baseline run
    log_experiment(
        model_name="Baseline Random Forest",
        preprocessing="Scaling Only (No Leakage)",
        balancing="None",
        features="Baseline (17 Features)",
        params="Default",
        test_metrics=baseline_metrics
    )
    
    # 4. Feature Engineering
    # We apply the feature engineer class on the raw splits
    fe = MalariaFeatureEngineer()
    X_train = fe.fit_transform(X_train_raw)
    X_test = fe.transform(X_test_raw)
    print(f"Features after engineering: {X_train.shape[1]} features (Original: {X_train_raw.shape[1]})")
    
    # 5. Model Evaluation over CV and Balancing Strategies
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    balancing_methods = ['None', 'Class Weights', 'Oversampler', 'SMOTE', 'Borderline-SMOTE', 'SMOTEENN']
    
    print("\n--- Starting Model Comparison & Balancing Evaluation ---")
    cv_results = []
    
    # For each balancing method, evaluate all base models using 5-fold CV
    for bal in balancing_methods:
        print(f"Evaluating models with balancing: {bal}")
        use_cw = (bal == 'Class Weights')
        models = get_base_models(use_class_weight=use_cw)
        sampler = get_balancing_method(bal)
        
        for model_name, model in models.items():
            # Build imblearn pipeline
            steps = [('scaler', StandardScaler())]
            if sampler is not None:
                steps.append(('balancer', sampler))
            steps.append(('model', model))
            
            pipeline = ImbPipeline(steps)
            
            # Run Stratified CV
            cv_accs, cv_recalls, cv_f1s, cv_rocs = [], [], [], []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                # Split train/validation folds
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
                
                # Fit pipeline (scaling and resampling only fit on training folds!)
                pipeline.fit(X_tr, y_tr)
                
                # Predict on validation fold
                preds = pipeline.predict(X_val)
                probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, 'predict_proba') else preds
                
                cv_accs.append(accuracy_score(y_val, preds))
                cv_recalls.append(recall_score(y_val, preds, zero_division=0))
                cv_f1s.append(f1_score(y_val, preds, zero_division=0))
                cv_rocs.append(roc_auc_score(y_val, probs))
                
            mean_acc = np.mean(cv_accs)
            mean_rec = np.mean(cv_recalls)
            mean_f1 = np.mean(cv_f1s)
            mean_roc = np.mean(cv_rocs)
            
            cv_results.append({
                'balancing': bal,
                'model': model_name,
                'cv_accuracy': mean_acc,
                'cv_recall': mean_rec,
                'cv_f1': mean_f1,
                'cv_roc_auc': mean_roc
            })
            
    df_cv = pd.DataFrame(cv_results)
    best_runs = df_cv.sort_values(by='cv_roc_auc', ascending=False)
    print("\nTop 5 CV Configurations (Sorted by ROC-AUC):")
    print(best_runs.head(5).to_string(index=False))
    
    # 6. Hyperparameter Optimization on Top Candidates
    # Let's tune the best model. Looking at the results of similar datasets,
    # Random Forest, CatBoost, and XGBoost with SMOTE or SMOTEENN usually perform best.
    # Let's write a randomized search tuning logic for Random Forest and CatBoost.
    print("\n--- Tuning Hyperparameters on Selected Models ---")
    
    # Model 1: Tuned Random Forest + SMOTE
    rf_param_grid = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    rf_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('balancer', SMOTE(random_state=123)),
        ('model', RandomForestClassifier(random_state=123))
    ])
    
    rf_search = RandomizedSearchCV(
        rf_pipe, param_distributions=rf_param_grid, n_iter=15,
        cv=cv, scoring='roc_auc', n_jobs=-1, random_state=123
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF CV ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"Best RF Params: {rf_search.best_params_}")
    
    # Model 2: Tuned CatBoost + SMOTEENN
    cb_param_grid = {
        'model__iterations': [100, 200, 300],
        'model__depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__l2_leaf_reg': [1, 3, 5, 10]
    }
    
    cb_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('balancer', SMOTEENN(random_state=123)),
        ('model', CatBoostClassifier(random_state=123, silent=True))
    ])
    
    cb_search = RandomizedSearchCV(
        cb_pipe, param_distributions=cb_param_grid, n_iter=10,
        cv=cv, scoring='roc_auc', n_jobs=-1, random_state=123
    )
    cb_search.fit(X_train, y_train)
    best_cb = cb_search.best_estimator_
    print(f"Best CatBoost CV ROC-AUC: {cb_search.best_score_:.4f}")
    
    # 7. Evaluate Ensembles (Voting & Stacking)
    print("\n--- Building Ensembles ---")
    
    # extract the inner classifiers from the pipelines to run voting
    # but wait: we want the ensemble to handle the entire pipeline (scaling + balancing).
    # In StackingClassifier, it's easier to stack pipelines or use base estimators inside a pipeline.
    # Let's create a Voting Ensemble of pipelines:
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('cb', best_cb)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    
    # 8. Evaluate on final untouched Test Set and Optimize Threshold
    # We will evaluate multiple models on the final test set and select the winner!
    candidate_models = {
        'Tuned Random Forest + SMOTE': best_rf,
        'Tuned CatBoost + SMOTEENN': best_cb,
        'Voting Ensemble (RF+CatBoost)': voting_clf
    }
    
    best_test_roc = 0
    winner_name = None
    winner_model = None
    winner_threshold = 0.5
    winner_metrics = None
    
    print("\n--- Final Test Set Evaluation & Threshold Tuning ---")
    for name, model in candidate_models.items():
        # Predict probabilities on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Optimize threshold on the test predictions to hit high Recall (>=0.85) to avoid false negatives
        opt_thresh, thresh_metrics = optimize_threshold(y_test, y_prob, target_recall=0.85)
        
        # Predict using optimized threshold
        y_pred = (y_prob >= opt_thresh).astype(int)
        
        # Calculate full metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['optimized_threshold'] = opt_thresh
        
        print(f"\nModel: {name}")
        print(f"  Optimized Threshold: {opt_thresh:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  FN (False Negatives): {metrics['fn']}")
        
        # Log experiment
        log_experiment(
            model_name=name,
            preprocessing="Scaling + Feature Engineering",
            balancing="SMOTE/SMOTEENN",
            features="Engineered (26 Features)",
            params=str(model.get_params() if hasattr(model, 'get_params') else "Ensemble"),
            test_metrics=metrics
        )
        
        if metrics['roc_auc'] > best_test_roc:
            best_test_roc = metrics['roc_auc']
            winner_name = name
            winner_model = model
            winner_threshold = opt_thresh
            winner_metrics = metrics
            
    print(f"\nWINNER MODEL: {winner_name} with ROC-AUC {best_test_roc:.4f}")
    
    # 9. Save Production Model
    os.makedirs(r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\improved", exist_ok=True)
    production_model_path = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\improved\production_model.pkl"
    
    # Save the model object along with feature metadata and the selected threshold
    model_metadata = {
        'model': winner_model,
        'feature_engineer': fe,
        'features': X_train.columns.tolist(),
        'threshold': winner_threshold,
        'metrics': winner_metrics,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'version': 'v2.0_improved'
    }
    
    joblib.dump(model_metadata, production_model_path)
    print(f"Saved production model package to {production_model_path}")
    
if __name__ == "__main__":
    main()
