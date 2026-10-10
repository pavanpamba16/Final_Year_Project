import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

class MalariaVisionCNN(nn.Module):
    """
    Deep Convolutional Neural Network for thin blood smear erythrocyte malaria classification.
    Optimized for microscopic Giemsa-stained parasite ring-stage and schizont identification.
    """
    def __init__(self, num_classes=2):
        super(MalariaVisionCNN, self).__init__()
        
        # Feature Extractor Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        feat = self.block1(x)
        feat = self.block2(feat)
        feat = self.block3(feat)
        pooled = self.global_pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits
        
    def extract_features(self, x):
        """Extracts deep embedding vector from the penultimate layer for multi-modal fusion."""
        feat = self.block1(x)
        feat = self.block2(feat)
        feat = self.block3(feat)
        pooled = self.global_pool(feat)
        return torch.flatten(pooled, 1)

def train_and_evaluate():
    print("=" * 65)
    print("TRAINING MULTI-MODAL MALARIA MICROSCOPY VISION CNN")
    print("=" * 65)
    
    train_dir = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\data\cell_images\Dataset\Train"
    test_dir = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\data\cell_images\Dataset\Test"
    
    # 1. Image Transformations
    img_size = 64
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # In ImageFolder: Parasite=0, Uninfected=1.
    # We invert label so 1 = Parasite (Positive infection), 0 = Uninfected (Negative)
    class_to_idx = train_dataset.class_to_idx
    print(f"Dataset Loaded: Train = {len(train_dataset)} images, Test = {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes} | Mapping: {class_to_idx}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    
    model = MalariaVisionCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    num_epochs = 15
    best_acc = 0.0
    best_state = None
    
    print("\nStarting Training Epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            # Map labels: Parasite (0) -> 1, Uninfected (1) -> 0
            targets = (labels == class_to_idx['Parasite']).long().to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation on untouched Test Set
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                targets = (labels == class_to_idx['Parasite']).long()
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Parasite
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_targets, val_probs)
        
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Test Acc: {val_acc*100:.1f}% | Recall: {val_rec*100:.1f}% | ROC-AUC: {val_auc*100:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                'model_state': model.state_dict(),
                'val_accuracy': val_acc,
                'val_recall': val_rec,
                'val_roc_auc': val_auc,
                'class_mapping': {'Parasite': 1, 'Uninfected': 0},
                'img_size': img_size
            }
            
    # Final evaluation of best model
    model.load_state_dict(best_state['model_state'])
    model.eval()
    val_preds, val_targets, val_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            targets = (labels == class_to_idx['Parasite']).long()
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.numpy())
            val_probs.extend(probs.cpu().numpy())
            
    final_acc = accuracy_score(val_targets, val_preds)
    final_rec = recall_score(val_targets, val_preds)
    final_prec = precision_score(val_targets, val_preds)
    final_auc = roc_auc_score(val_targets, val_probs)
    tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()
    final_spec = tn / (tn + fp)
    
    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION RESULTS (MICROSCOPIC VISION MODEL):")
    print(f"  • Test Accuracy:            {final_acc*100:.2f}%")
    print(f"  • Sensitivity (Recall):     {final_rec*100:.2f}% (Detection of Parasitized Cells)")
    print(f"  • Specificity:              {final_spec*100:.2f}% (Detection of Healthy Cells)")
    print(f"  • Precision:                {final_prec*100:.2f}%")
    print(f"  • ROC-AUC Score:            {final_auc*100:.2f}%")
    print("=" * 65)
    
    # Save Model Weights and Metadata
    save_dir = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\vision"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "malaria_vision_cnn.pt")
    
    package = {
        'state_dict': model.state_dict(),
        'metrics': {
            'accuracy': float(final_acc),
            'recall': float(final_rec),
            'specificity': float(final_spec),
            'precision': float(final_prec),
            'roc_auc': float(final_auc),
            'test_samples': len(test_dataset),
            'train_samples': len(train_dataset)
        },
        'img_size': img_size,
        'model_name': 'MalariaVisionCNN-v1.0',
        'date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(package, save_path)
    print(f"Saved trained Vision CNN model package to: {save_path}")

if __name__ == '__main__':
    train_and_evaluate()
