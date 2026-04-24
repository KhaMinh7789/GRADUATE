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
from model_1d_sota import Wav2Vec2_DNDF_Model
from torch.cuda.amp import GradScaler
from torch.utils.data import WeightedRandomSampler

BATCH_SIZE = 8
EPOCHS = 15
N_SPLITS = 5

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [] ,[]
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
    print(f"\n🚀 SOTA MODE: Wav2Vec 2.0 (1D Raw Audio) + DNDF")
    print(f"📌 Chế độ: {N_SPLITS}-Fold CV | NO MixUp | Focal Loss | Cosine LR")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    feature_dir = os.path.join(base_dir, 'data', 'features_1d_raw')

    print("\n🔍 Đang khởi tạo và kiểm tra dữ liệu 1D...")
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

        model = Wav2Vec2_DNDF_Model(num_classes=1).to(device)
        criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        
        # Tách Learning Rate: Wav2Vec học chậm, DNDF học nhanh
        optimizer = optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
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
                optimizer.zero_grad()

                # KHÔNG DÙNG MIXUP CHO 1D
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                
                outputs = outputs.float()
                loss = criterion(outputs, targets)
                
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
                torch.save(model.state_dict(), os.path.join(base_dir, f'best_wav2vec_fold_{fold+1}.pth'))
                print(f"   🏆 Đã lưu mô hình tốt nhất (AUC: {best_auc:.4f})")
        
        print(f"🏆 Fold {fold + 1} Hoàn tất! Best AUC: {best_auc:.4f}")
        fold_results.append(best_auc)

    print(f"\n🎉 HOÀN TẤT {N_SPLITS}-FOLD CV! AUC Trung bình: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

if __name__ == '__main__':
    main()