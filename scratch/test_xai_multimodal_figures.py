import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from src.train_vision import MalariaVisionCNN

def mark_boundaries_numpy(img_rgb, mask, color=(0.2, 0.9, 0.4)):
    res = img_rgb.copy().astype(np.float32)
    if res.max() > 1.0:
        res = res / 255.0
    h, w = mask.shape
    edges = np.zeros((h, w), dtype=bool)
    edges[:-1, :] |= (mask[:-1, :] != mask[1:, :])
    edges[1:, :] |= (mask[:-1, :] != mask[1:, :])
    edges[:, :-1] |= (mask[:, :-1] != mask[:, 1:])
    edges[:, 1:] |= (mask[:, :-1] != mask[:, 1:])
    for c in range(3):
        res[edges, c] = color[c]
    return res

def test_generate_figures():
    os.makedirs('scratch', exist_ok=True)
    
    # Load vision model
    pkg = torch.load('models/vision/malaria_vision_cnn.pt', map_location='cpu')
    model = MalariaVisionCNN(num_classes=2)
    model.load_state_dict(pkg['state_dict'])
    model.eval()
    
    # Load sample image
    img_path = 'static/sample_smears/parasitized_sample_1.png'
    raw_img = Image.open(img_path).convert('RGB')
    resized_img = raw_img.resize((64, 64))
    img_np = np.array(resized_img)
    
    # Synthetic / mock clinical data for test
    positives = [
        {'friendly_name': 'Convulsions (Seizures)', 'shap_value': 0.325},
        {'friendly_name': 'Coca-Cola Urine', 'shap_value': 0.290},
        {'friendly_name': 'High Fever (Temp >= 41°C)', 'shap_value': 0.250},
        {'friendly_name': 'Severe Anemia', 'shap_value': 0.245}
    ]
    negatives = [
        {'friendly_name': 'Absence of Jaundice', 'shap_value': -0.120},
        {'friendly_name': 'Adult Patient (Age 28)', 'shap_value': -0.055}
    ]
    lime_clinical = [
        {'clean_rule': 'Convulsions (Seizures): Present', 'weight': 0.285},
        {'clean_rule': 'Coca-Cola Urine: Present', 'weight': 0.210},
        {'clean_rule': 'Fever >= 37.5°C: Present', 'weight': 0.165},
        {'clean_rule': 'Absence of Jaundice', 'weight': -0.095},
        {'clean_rule': 'Adult Patient (Age 28)', 'weight': -0.040}
    ]

    # --- 1. TEST MULTI-MODAL LIME FIGURE ---
    def predict_fn(images):
        imgs = images.astype(np.float32)
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        imgs_norm = (imgs - 0.5) / 0.5
        tensors = torch.tensor(imgs_norm, dtype=torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    explainer = lime_image.LimeImageExplainer(random_state=123)
    explanation = explainer.explain_instance(
        img_np, 
        predict_fn, 
        top_labels=2, 
        hide_color=0, 
        num_samples=120,
        random_seed=123
    )
    
    # Class 1 is Parasitized
    top_label = 1 if explanation.top_labels[0] == 1 else explanation.top_labels[1] if len(explanation.top_labels) > 1 else 0
    temp, mask = explanation.get_image_and_mask(
        top_label, 
        positive_only=False, 
        num_features=6, 
        hide_rest=False
    )
    
    # Create LIME combined figure (Image LIME + Tabular LIME)
    fig, (ax_img, ax_tab) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor('#0f172a')
    ax_img.set_facecolor('#0f172a')
    ax_tab.set_facecolor('#0f172a')
    
    # Image LIME overlay
    img_bound = mark_boundaries_numpy(temp, mask, color=(0.2, 0.9, 0.4))
    # Positive superpixel mask overlay
    overlay = img_bound.copy()
    overlay[mask == 1] = overlay[mask == 1] * 0.7 + np.array([0.9, 0.2, 0.3]) * 0.3 # red tint for risk regions
    
    ax_img.imshow(overlay)
    ax_img.set_title("LIME Image: Smear Perturbation\n(Red: Parasite Feature, Green: Boundary)", 
                     color='#f8fafc', fontsize=10.5, fontweight='bold', pad=8)
    ax_img.axis('off')
    
    # Clinical Tabular LIME
    items = []
    for item in lime_clinical[:7]:
        name = item.get('clean_rule', item.get('rule', ''))
        w = item['weight']
        c = '#ff4757' if w > 0 else '#2ed573'
        items.append((name, w, c))
    items.sort(key=lambda x: x[1])
    
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    cols = [x[2] for x in items]
    
    bars = ax_tab.barh(names, vals, color=cols, height=0.55, zorder=3)
    ax_tab.axvline(0, color='#64748b', linewidth=1.2, linestyle='--', zorder=4)
    for bar, val in zip(bars, vals):
        width = bar.get_width()
        x_pos = width + (0.015 if width >= 0 else -0.015)
        ha = 'left' if width >= 0 else 'right'
        ax_tab.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.3f}", 
                    va='center', ha=ha, color='#f8fafc', fontsize=8.5, fontweight='bold')
    ax_tab.tick_params(colors='#94a3b8', labelsize=8.5)
    for s in ax_tab.spines.values():
        s.set_color('#334155')
    ax_tab.grid(axis='x', color='#1e293b', linestyle=':', zorder=0)
    ax_tab.set_xlabel("LIME Feature Weight", color='#94a3b8', fontsize=9, labelpad=6)
    ax_tab.set_title("LIME Clinical: Symptom Perturbation Rules", color='#f8fafc', fontsize=10.5, fontweight='bold', pad=8)
    
    xmin, xmax = ax_tab.get_xlim()
    span = max(abs(xmin), abs(xmax)) + 0.06
    ax_tab.set_xlim(-span, span)
    
    fig.suptitle("UNIFIED LIME MULTI-MODAL EXPLANATION (IMAGE + CLINICAL)", 
                 color='#38bdf8', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('scratch/unified_lime_figure.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print("Saved scratch/unified_lime_figure.png")

    # --- 2. TEST MULTI-MODAL SHAP FIGURE ---
    # Path-Shapley (Integrated Gradients)
    img_tensor = torch.tensor((img_np.astype(np.float32) / 255.0 - 0.5) / 0.5, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    baseline = torch.zeros_like(img_tensor)
    steps = 25
    alphas = torch.linspace(0.0, 1.0, steps).view(-1, 1, 1, 1)
    interpolated = baseline + alphas * (img_tensor - baseline)
    interpolated.requires_grad_(True)
    logits = model(interpolated)
    probs = torch.softmax(logits, dim=1)[:, 1]
    grads = torch.autograd.grad(probs.sum(), interpolated)[0]
    avg_grads = grads.mean(dim=0, keepdim=True)
    ig = ((img_tensor - baseline) * avg_grads).detach().squeeze(0).numpy() # (3, 64, 64)
    # Aggregate across color channels: sum or L2 norm or signed mean
    shap_img_attribution = ig.mean(axis=0) # (64, 64)
    
    # Create SHAP combined figure (Image SHAP Heatmap + Tabular SHAP Bar Chart)
    fig, (ax_img, ax_tab) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor('#0f172a')
    ax_img.set_facecolor('#0f172a')
    ax_tab.set_facecolor('#0f172a')
    
    # Plot original image with SHAP attribution overlay
    ax_img.imshow(img_np)
    # Normalize attribution for colormap
    abs_max = max(np.percentile(np.abs(shap_img_attribution), 99), 1e-6)
    im = ax_img.imshow(shap_img_attribution, cmap='coolwarm', alpha=0.65, vmin=-abs_max, vmax=abs_max)
    cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=7.5)
    cbar.outline.set_edgecolor('#334155')
    cbar.set_label('Shapley Attribution', color='#94a3b8', fontsize=8)
    
    ax_img.set_title("SHAP Image: Microscopic Smear Heatmap\n(Red: Parasite Attribution, Blue: Healthy)", 
                     color='#f8fafc', fontsize=10.5, fontweight='bold', pad=8)
    ax_img.axis('off')
    
    # Clinical Tabular SHAP
    items = []
    for p in positives[:5]:
        items.append((p.get('friendly_name', p.get('feature', '')), p['shap_value'] * 100, '#ff4757'))
    for n in negatives[:5]:
        items.append((n.get('friendly_name', n.get('feature', '')), n['shap_value'] * 100, '#2ed573'))
    items.sort(key=lambda x: x[1])
    
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    cols = [x[2] for x in items]
    
    bars = ax_tab.barh(names, vals, color=cols, height=0.55, zorder=3)
    ax_tab.axvline(0, color='#64748b', linewidth=1.2, linestyle='--', zorder=4)
    for bar, val in zip(bars, vals):
        width = bar.get_width()
        x_pos = width + (1.2 if width >= 0 else -1.2)
        ha = 'left' if width >= 0 else 'right'
        ax_tab.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.1f}%", 
                    va='center', ha=ha, color='#f8fafc', fontsize=8.5, fontweight='bold')
    ax_tab.tick_params(colors='#94a3b8', labelsize=8.5)
    for s in ax_tab.spines.values():
        s.set_color('#334155')
    ax_tab.grid(axis='x', color='#1e293b', linestyle=':', zorder=0)
    ax_tab.set_xlabel("SHAP Impact on Risk (%)", color='#94a3b8', fontsize=9, labelpad=6)
    ax_tab.set_title("SHAP Clinical: Feature Risk Attribution", color='#f8fafc', fontsize=10.5, fontweight='bold', pad=8)
    
    xmin, xmax = ax_tab.get_xlim()
    span = max(abs(xmin), abs(xmax)) + 8
    ax_tab.set_xlim(-span, span)
    
    fig.suptitle("UNIFIED SHAP MULTI-MODAL EXPLANATION (IMAGE + CLINICAL)", 
                 color='#38bdf8', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('scratch/unified_shap_figure.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print("Saved scratch/unified_shap_figure.png")

if __name__ == '__main__':
    test_generate_figures()
