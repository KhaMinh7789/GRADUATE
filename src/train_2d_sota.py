import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

from dataset_offline import CovidCoughDatasetOffline
from model_2d_sota import AST_DNDF_Model
from torch.cuda.amp import GradScaler
from torch.utils.data import WeightedRandomSampler

BATCH_SIZE = 16 
EPOCHS = 15     
N_SPLITS = 5    

def mixup_data(x, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class BinaryFocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting (Lin et al., 2017).
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t = alpha if y=1 (COVID-19), else (1 - alpha) if y=0 (Healthy)
    
    FIXED (v2): Applied alpha per-class, not uniformly.
    This correctly addresses class imbalance:
      - Positive class (COVID-19): alpha_t = 0.8 (upweight)
      - Negative class (Healthy):  alpha_t = 0.2 (downweight)
    
    Consistent with: train_1d_sota.py BinaryFocalLoss implementation.
    Hyperparameters: alpha=0.8, gamma=2.0 (unified throughout thesis)
    """
    def __init__(self, alpha=0.8, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha  # alpha=0.8: positive class weight
        self.gamma = gamma  # gamma=2.0: standard focusing parameter (Lin et al., 2017)

    def forward(self, inputs, targets):
        # BCE loss (reduction=none for per-sample weighting)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # Probability of correct prediction
        pt = torch.exp(-bce_loss)
        # FIXED: Per-class alpha weighting
        # alpha_t = alpha if target=1 (COVID), else (1-alpha) if target=0 (Healthy)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        # Focal loss: alpha_t * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    return acc, sens, spec, auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 SOTA MODE: Audio Spectrogram Transformer (AST) + DNDF")
    print(f"📌 Chế độ: {N_SPLITS}-Fold CV | MixUp | Focal Loss | Cosine LR")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    feature_dir = os.path.join(base_dir, 'data', 'features_2d_ast')

    print("\n🔍 Đang khởi tạo và kiểm tra dữ liệu 2D...")
    full_dataset = CovidCoughDatasetOffline(metadata_path, feature_dir)
    labels = full_dataset.data['label'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*20} VÒNG LẶP FOLD {fold + 1}/{N_SPLITS} {'='*20}")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[int(l)] for l in train_labels])
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights))

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

        model = AST_DNDF_Model(num_classes=1).to(device)
        criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        
        # Tách Learning Rate: AST học chậm, DNDF học nhanh
        optimizer = optim.AdamW([
            {'params': model.ast.parameters(), 'lr': 1e-5},
            {'params': model.dndf.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        # Bộ giảm tốc độ học Cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler()
        best_auc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for inputs, targets in loop:
                inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
                mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.4, device=device)
                
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(mixed_inputs)
                
                outputs = outputs.float()
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Cập nhật Scheduler
            scheduler.step()

            print(f"\n⏳ Đang đánh giá trên tập Validation...")
            acc, sens, spec, auc = evaluate(model, val_loader, criterion, device)
            
            print(f"🟢 Epoch {epoch+1:02d}/{EPOCHS} | Train Loss (TB): {train_loss/len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"   Metrics -> ACC: {acc:.4f} | SENS: {sens:.4f} | SPEC: {spec:.4f} | AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(base_dir, f'best_ast_fold_{fold+1}.pth'))
                print(f"   🏆 Đã lưu mô hình tốt nhất (AUC: {best_auc:.4f})")
        
        print(f"🏆 Fold {fold + 1} Hoàn tất! Best AUC: {best_auc:.4f}")
        fold_results.append(best_auc)

    print(f"\n🎉 HOÀN TẤT {N_SPLITS}-FOLD CV! AUC Trung bình: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

if __name__ == '__main__':
    main()