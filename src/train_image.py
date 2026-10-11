import os
import re
import time
import random
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PatientCellDataset(Dataset):
    """Custom PyTorch dataset for thin blood smear erythrocyte images with patient tracking."""
    def __init__(self, file_tuples, transform=None):
        self.file_tuples = file_tuples # list of (file_path, label, patient_id)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_tuples)
        
    def __getitem__(self, idx):
        path, label, pid = self.file_tuples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
        except Exception:
            # Fallback for corrupted file
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

def build_patient_level_splits(cell_images_dir, samples_per_patient_limit=12):
    """
    Partitions the NIH dataset into strictly disjoint patient-level splits.
    Guarantees no patient appears in multiple splits.
    """
    classes = {'Uninfected': 0, 'Parasitized': 1}
    patient_files = {} # pid -> {'Parasitized': [], 'Uninfected': []}
    
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
            
    # Remove unknown or single-class patients for balanced grouping
    valid_pids = sorted([
        p for p in patient_files 
        if p != 'Unknown' and len(patient_files[p]['Parasitized']) > 0 and len(patient_files[p]['Uninfected']) > 0
    ], key=lambda x: int(x[1:]))
    
    rng = random.Random(42)
    rng.shuffle(valid_pids)
    
    n_total = len(valid_pids)
    n_train = int(n_total * 0.70) # 105 patients
    n_val = int(n_total * 0.15)   # 22 patients
    
    train_pids = set(valid_pids[:n_train])
    val_pids = set(valid_pids[n_train:n_train + n_val])
    test_pids = set(valid_pids[n_train + n_val:])
    
    print(f"[Patient Partition] Total Unique Patients: {n_total}")
    print(f"[Patient Partition] Train Patients: {len(train_pids)}, Val Patients: {len(val_pids)}, Test Patients: {len(test_pids)}")
    
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
        
    train_tuples = collect_split(train_pids)
    val_tuples = collect_split(val_pids)
    test_tuples = collect_split(test_pids)
    
    print(f"[Dataset Samples] Train: {len(train_tuples)}, Val: {len(val_tuples)}, Test: {len(test_tuples)}")
    return train_tuples, val_tuples, test_tuples, list(train_pids), list(val_pids), list(test_pids)

def build_mobilenet_v2_classifier(num_classes=2):
    """Initializes MobileNetV2 with pre-trained ImageNet features and custom classification head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze initial feature extractor blocks for efficient transfer learning
    for param in model.features[:-4].parameters():
        param.requires_grad = False
        
    # Unfreeze top feature layers for fine-tuning to thin smear microscopy
    for param in model.features[-4:].parameters():
        param.requires_grad = True
        
    # Replace classifier head with dropout + 2-class linear output
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_image_model(epochs=4, batch_size=32, lr=1e-3):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cell_images_dir = os.path.join(base_dir, 'data', 'cell_images')
    output_dir = os.path.join(base_dir, 'models', 'image')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Image Training] Using compute device: {device}")
    
    train_tuples, val_tuples, test_tuples, tr_pids, val_pids, te_pids = build_patient_level_splits(
        cell_images_dir, samples_per_patient_limit=10
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
    
    print("\n" + "=" * 65)
    print("TRAINING MOBILENET-V2 ON PATIENT-DISJOINT BLOOD SMEARS")
    print("=" * 65)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_targets, val_probs)
        t_elapsed = time.time() - t0
        
        print(f"Epoch {epoch}/{epochs} [{t_elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            
    # Load best weights
    model.load_state_dict(best_state)
    
    # Evaluate on untouched Held-Out Test Patients
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            
    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_auc = roc_auc_score(test_targets, test_probs)
    
    print("\n" + "=" * 65)
    print("HELD-OUT PATIENT TEST EVALUATION (PATIENTS NEVER SEEN IN TRAIN)")
    print("=" * 65)
    print(f"Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"Test Sensitivity: {test_rec * 100:.2f}%")
    print(f"Test Specificity: {test_spec * 100:.2f}%")
    print(f"Test Precision:   {test_prec * 100:.2f}%")
    print(f"Test F1-Score:    {test_f1:.4f}")
    print(f"Test ROC-AUC:     {test_auc:.4f}")
    print(f"Confusion Matrix [TN, FP, FN, TP]: {tn}, {fp}, {fn}, {tp}")
    
    save_path = os.path.join(output_dir, 'malaria_mobilenet_v2.pt')
    package = {
        'model_name': 'MobileNetV2_TransferLearning',
        'state_dict': model.state_dict(),
        'img_size': img_size,
        'mean': mean,
        'std': std,
        'class_names': ['Uninfected', 'Parasitized'],
        'metrics_test': {
            'accuracy': round(float(test_acc), 4),
            'precision': round(float(test_prec), 4),
            'recall': round(float(test_rec), 4),
            'specificity': round(float(test_spec), 4),
            'f1': round(float(test_f1), 4),
            'roc_auc': round(float(test_auc), 4),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'patient_splits': {
            'train_patients_count': len(tr_pids),
            'val_patients_count': len(val_pids),
            'test_patients_count': len(te_pids)
        },
        'trained_at': datetime.now().isoformat()
    }
    torch.save(package, save_path)
    print(f"\nSaved image model checkpoint to: {save_path}")
    return package

if __name__ == '__main__':
    train_image_model(epochs=3, batch_size=32, lr=1e-3)
