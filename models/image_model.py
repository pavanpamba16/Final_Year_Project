"""
EXPLAINA AI: Deep Vision Image Classification Model
Architecture: MobileNetV2 Fine-Tuned Transfer Learning for Microscopic Blood Smears
Input: Thin blood smear erythrocyte micrographs (224x224 RGB)
Output: Cellular infection classification (Parasitized vs. Uninfected) + Grad-CAM Heatmap
"""

import os
import io
import re
import time
import random
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Deterministic seeding for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class PatientCellDataset(Dataset):
    """
    Custom PyTorch Dataset for thin blood smear erythrocyte micrographs with patient tracking.
    Enforces strict patient-level separation to eliminate data leakage.
    """
    def __init__(self, file_tuples, transform=None):
        self.file_tuples = file_tuples  # list of (file_path, label, patient_id)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_tuples)
        
    def __getitem__(self, idx):
        path, label, pid = self.file_tuples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)


def build_patient_level_splits(cell_images_dir, samples_per_patient_limit=10):
    """
    Partitions the NIH malaria cell images dataset into strictly disjoint patient-level splits:
    70% Train, 15% Validation, 15% Held-out Test.
    Guarantees no patient's erythrocyte images appear across multiple splits.
    """
    classes = {'Uninfected': 0, 'Parasitized': 1}
    patient_files = {}  # pid -> {'Parasitized': [], 'Uninfected': []}
    
    for cls_name in ['Parasitized', 'Uninfected']:
        cls_dir = os.path.join(cell_images_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        for f in os.listdir(cls_dir):
            if not f.lower().endswith('.png'):
                continue
            f_path = os.path.join(cls_dir, f)
            m = re.search(r'P(\d+)', f)
            pid = 'P' + m.group(1) if m else 'Unknown'
            
            if pid not in patient_files:
                patient_files[pid] = {'Parasitized': [], 'Uninfected': []}
            patient_files[pid][cls_name].append(f_path)
            
    valid_pids = sorted([
        p for p in patient_files 
        if p != 'Unknown' and len(patient_files[p]['Parasitized']) > 0 and len(patient_files[p]['Uninfected']) > 0
    ], key=lambda x: int(x[1:]))
    
    rng = random.Random(42)
    rng.shuffle(valid_pids)
    
    n_total = len(valid_pids)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_pids = set(valid_pids[:n_train])
    val_pids = set(valid_pids[n_train:n_train + n_val])
    test_pids = set(valid_pids[n_train + n_val:])
    
    def collect_split(pids):
        split_items = []
        for pid in pids:
            for cls_name, label in classes.items():
                items = patient_files[pid][cls_name]
                rng.shuffle(items)
                selected = items[:samples_per_patient_limit]
                for pth in selected:
                    split_items.append((pth, label, pid))
        return split_items
        
    return (
        collect_split(train_pids),
        collect_split(val_pids),
        collect_split(test_pids),
        train_pids, val_pids, test_pids
    )


def build_mobilenet_v2_classifier(num_classes=2):
    """
    Constructs the MobileNetV2 architecture with pre-trained ImageNet feature extraction backbone
    and customized classification head.
    Features:
    - Inverted residual blocks with linear bottlenecks.
    - Fine-tuning enabled on top feature layers.
    - Dropout (p=0.3) for regularization.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze lower feature extraction layers
    for param in model.features[:-4].parameters():
        param.requires_grad = False
    # Unfreeze top inverted residual blocks for domain-specific fine tuning
    for param in model.features[-4:].parameters():
        param.requires_grad = True
        
    # Replace standard 1000-class classifier with 2-class malaria head
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


class MalariaVisionModel:
    """
    High-level inference wrapper for the trained MobileNetV2 vision classifier.
    Produces predicted class, confidence scores, and visual Grad-CAM interpretability heatmaps.
    """
    def __init__(self, checkpoint_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, 'models', 'image', 'malaria_mobilenet_v2.pt')
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['Uninfected', 'Parasitized']
        self.img_size = 224
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.metrics = {}
        if os.path.exists(checkpoint_path):
            self.load_model(checkpoint_path)
            
    def load_model(self, checkpoint_path):
        """Loads trained model weights and evaluation metrics."""
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.last_channel, 2)
        )
        pkg = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(pkg, dict) and 'state_dict' in pkg:
            self.model.load_state_dict(pkg['state_dict'])
            self.metrics = pkg.get('metrics_test', pkg.get('test_metrics', {}))
        elif isinstance(pkg, dict) and 'model_state_dict' in pkg:
            self.model.load_state_dict(pkg['model_state_dict'])
            self.metrics = pkg.get('test_metrics', {})
        else:
            self.model.load_state_dict(pkg)
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path_or_pil):
        """
        Executes deep vision inference on a thin blood smear erythrocyte image.
        Returns:
            predicted_class: 'Parasitized' or 'Uninfected'
            confidence: model probability percentage (0-100)
            grad_cam_b64: base64-encoded Grad-CAM activation figure
            metrics: held-out test evaluation metrics
        """
        if self.model is None:
            return {'error': 'MobileNetV2 image model checkpoint not loaded.'}
            
        try:
            if isinstance(image_path_or_pil, str):
                pil_img = Image.open(image_path_or_pil).convert('RGB')
            else:
                pil_img = image_path_or_pil.convert('RGB')
        except Exception as e:
            return {'error': f"Invalid or unreadable image file: {str(e)}"}
            
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Attach Grad-CAM hooks to the final convolutional feature layer
        target_layer = self.model.features[-1]
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
            
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
            
        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)
        
        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        
        uninfected_prob = float(probs[0])
        parasite_prob = float(probs[1])
        pred_idx = int(np.argmax(probs))
        pred_label = self.classes[pred_idx]
        is_parasitized = (pred_idx == 1)
        confidence = round(float(probs[pred_idx]) * 100, 1)
        
        # Backward pass for Grad-CAM gradient attribution
        self.model.zero_grad()
        score = logits[0, pred_idx]
        score.backward()
        
        h_fwd.remove()
        h_bwd.remove()
        
        # Compute Grad-CAM map
        grad_cam_b64 = None
        if len(activations) > 0 and len(gradients) > 0:
            act = activations[0].detach()
            grad = gradients[0].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1).squeeze(0)
            cam = np.maximum(cam.cpu().numpy(), 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            grad_cam_b64 = self._generate_gradcam_figure(pil_img, cam, pred_label, confidence)
            
        return {
            'predicted_class': pred_label,
            'is_parasitized': is_parasitized,
            'parasite_probability': round(parasite_prob * 100, 1),
            'uninfected_probability': round(uninfected_prob * 100, 1),
            'confidence': confidence,
            'grad_cam_b64': grad_cam_b64,
            'metrics': self.metrics,
            'target_definition': "Erythrocyte Infection Status (Intracellular Plasmodium falciparum ring/trophozoite vs Healthy RBC)"
        }
        
    def _generate_gradcam_figure(self, pil_img, cam, pred_label, confidence):
        """Generates publication-grade side-by-side visualization of Smear and Grad-CAM Activation Heatmap."""
        img_np = np.array(pil_img.resize((self.img_size, self.img_size)))
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        cam_resized = np.array(cam_img) / 255.0
        
        fig, (ax_orig, ax_cam) = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=150)
        fig.patch.set_facecolor('#0f172a')
        ax_orig.set_facecolor('#0f172a')
        ax_cam.set_facecolor('#0f172a')
        
        # Original Smear
        ax_orig.imshow(img_np)
        ax_orig.set_title("Input Microscopic Smear", color='#f8fafc', fontsize=9.5, fontweight='bold', pad=6)
        ax_orig.axis('off')
        
        # Grad-CAM Overlay
        ax_cam.imshow(img_np)
        im = ax_cam.imshow(cam_resized, cmap='jet', alpha=0.55)
        cbar = plt.colorbar(im, ax=ax_cam, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='#94a3b8', labelsize=7)
        cbar.outline.set_edgecolor('#334155')
        cbar.set_label('Grad-CAM Activation', color='#94a3b8', fontsize=7.5)
        
        badge_color = '#ff4757' if pred_label == 'Parasitized' else '#2ed573'
        ax_cam.set_title(f"Grad-CAM Attribution: {pred_label} ({confidence}%)", color=badge_color, fontsize=9.5, fontweight='bold', pad=6)
        ax_cam.axis('off')
        
        fig.suptitle("EXPLAINA AI: MICROSCOPIC BLOOD SMEAR ATTRIBUTION (GRAD-CAM)", color='#38bdf8', fontsize=10.5, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')


def train_image_model(epochs=4, batch_size=32, lr=1e-3, data_dir=None, save_path=None):
    """Full training pipeline with patient-level split partitioning and evaluation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, 'data', 'cell_images')
    if save_path is None:
        save_path = os.path.join(base_dir, 'models', 'image', 'malaria_mobilenet_v2.pt')
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Image Model Training] Running on compute device: {device}")
    
    train_tuples, val_tuples, test_tuples, tr_pids, val_pids, te_pids = build_patient_level_splits(
        data_dir, samples_per_patient_limit=10
    )
    
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_loader = DataLoader(PatientCellDataset(train_tuples, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatientCellDataset(val_tuples, eval_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PatientCellDataset(test_tuples, eval_transform), batch_size=batch_size, shuffle=False)
    
    model = build_mobilenet_v2_classifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_tuples):.4f} - Val Acc: {val_acc*100:.2f}% - Val F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_val_f1 or best_state is None:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            
    # Load best checkpoint and evaluate on untouched test set
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_targets.extend(labels.numpy())
            
    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': round(float(accuracy_score(test_targets, test_preds)), 4),
        'precision': round(float(precision_score(test_targets, test_preds, zero_division=0)), 4),
        'recall': round(float(recall_score(test_targets, test_preds, zero_division=0)), 4),
        'specificity': round(float(tn / (tn + fp)), 4),
        'f1': round(float(f1_score(test_targets, test_preds, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(test_targets, test_probs)), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    
    torch.save({
        'model_state_dict': best_state,
        'test_metrics': metrics,
        'architecture': 'MobileNetV2 Transfer Learning',
        'input_size': (224, 224, 3)
    }, save_path)
    print(f"Model saved to {save_path} with Test Metrics: {metrics}")
    return metrics
