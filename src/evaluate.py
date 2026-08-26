import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates comprehensive clinical and statistical classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity (Recall) and Specificity
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_pts, precision_pts)
    
    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': sensitivity,
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'brier_score': brier,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    return metrics

def optimize_threshold(y_true, y_prob, target_recall=0.85):
    """
    Finds the optimal threshold to maximize F1 or meet a target recall.
    For medical diagnostics, we prioritize meeting a high recall (sensitivity)
    to minimize false negatives, while balancing specificity.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Let's find thresholds where recall >= target_recall
    candidates = []
    for t in np.linspace(0.01, 0.99, 100):
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        spec = (y_true == 0).sum() - (y_pred[y_true == 0] == 1).sum()
        spec = spec / (y_true == 0).sum() if (y_true == 0).sum() > 0 else 0
        acc = accuracy_score(y_true, y_pred)
        candidates.append((t, rec, spec, f1, acc))
        
    df_cand = pd.DataFrame(candidates, columns=['threshold', 'recall', 'specificity', 'f1', 'accuracy'])
    
    # Filter for recall >= target_recall
    high_recall_df = df_cand[df_cand['recall'] >= target_recall]
    
    if not high_recall_df.empty:
        # Of those, pick the one with highest specificity or F1
        best_row = high_recall_df.sort_values(by=['specificity', 'f1'], ascending=False).iloc[0]
    else:
        # Fallback to the one with the highest recall
        best_row = df_cand.sort_values(by='recall', ascending=False).iloc[0]
        
    return float(best_row['threshold']), best_row.to_dict()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filepath=None):
    """Generates and saves a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Severe', 'Severe'],
                yticklabels=['Non-Severe', 'Severe'])
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, label="Model", filepath=None):
    """Generates and saves a ROC-AUC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.close()

def plot_calibration_curve(y_true, y_prob, label="Model", filepath=None):
    """Generates and saves a probability calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    brier = brier_score_loss(y_true, y_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, "s-", color='darkorange', label=f'{label} (Brier = {brier:.4f})')
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title("Probability Calibration Curve", fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.close()
