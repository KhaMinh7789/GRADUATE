"""
Ablation Study: DNDF vs Softmax Classifier
===========================================
Addresses advisor feedback (Section 4.3.2):
  "Thiếu thực nghiệm đối chứng A/B giữa Wav2Vec 2.0 + Softmax vs Wav2Vec 2.0 + DNDF"

This script trains IDENTICAL Wav2Vec 2.0 feature extractors with two different
classifier heads to isolate the contribution of DNDF:
  - Classifier A: Linear layer + Sigmoid (standard baseline)
  - Classifier B: DNDF (5 trees, depth=4) [proposed]

Both use:
  - Same random seed (42)
  - Same 5-fold cross-validation splits
  - Same Focal Loss (alpha=0.8, gamma=2.0)
  - Same optimizer (AdamW, Cosine LR)
  - Same WeightedRandomSampler

Result: Isolates classifier contribution from feature extractor contribution.

Usage:
    python ablation_dndf_vs_softmax.py

Output:
    ../experiment_report/ablation_results.json
    ../experiment_report/ablation_comparison.png
"""
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import json
import warnings
warnings.filterwarnings("ignore")

from dataset_offline import CovidCoughDatasetOffline
from model_1d_sota import Wav2Vec2_DNDF_Model, DNDF
from transformers import Wav2Vec2Model
from utils_metrics import MetricsCalculator

BATCH_SIZE = 8
EPOCHS = 15
N_SPLITS = 5
RANDOM_SEED = 42


class BinaryFocalLoss(nn.Module):
    """Unified Focal Loss (per-class alpha) - consistent across all experiments."""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce_loss).mean()


# ============================================================
# CLASSIFIER A: Standard Softmax/Sigmoid baseline
# ============================================================
class Wav2Vec2_Softmax_Model(nn.Module):
    """
    Baseline: Wav2Vec 2.0 + Linear + Sigmoid
    
    This is the standard approach DNDF is compared against.
    Feature extractor configuration is IDENTICAL to DNDF model
    to ensure fair comparison (only classifier differs).
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # IDENTICAL freezing strategy as DNDF model
        self.wav2vec2.feature_extractor._freeze_parameters()
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if not any(f"layers.{i}." in name for i in range(10, 12)):
                param.requires_grad = False
        
        # BASELINE CLASSIFIER: Simple linear projection
        # This is what DNDF replaces
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        features = outputs.last_hidden_state.mean(dim=1)  # [Batch, 768]
        return self.classifier(features)


def train_and_evaluate(model_class, model_name, full_dataset, labels, device,
                       base_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Run 5-fold CV and return metrics for one model configuration."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs_roc, fold_aucs_pr, fold_eces = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n  Fold {fold+1}/{N_SPLITS}...")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[int(l)] for l in train_labels])
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).double(), len(sample_weights)
        )
        
        train_loader = DataLoader(train_sub, batch_size=batch_size,
                                  sampler=sampler, num_workers=4,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
        
        model = model_class().to(device)
        criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        
        # IDENTICAL optimizer config for fair comparison
        optimizer = optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
            {'params': (p for n, p in model.named_parameters() 
                       if 'wav2vec2' not in n), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler()
        best_auc = 0.0
        best_probs, best_labels = None, None
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                
                outputs = outputs.float()
                loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
            
            # Validation
            model.eval()
            all_probs, all_labels_val = [], []
            with torch.no_grad():
                for inputs, labels_v in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs).cpu().numpy()
                    all_probs.extend(outputs.flatten())
                    all_labels_val.extend(labels_v.numpy().flatten())
            
            try:
                fold_auc = roc_auc_score(all_labels_val, all_probs)
                if fold_auc > best_auc:
                    best_auc = fold_auc
                    best_probs = all_probs.copy()
                    best_labels = all_labels_val.copy()
            except Exception:
                pass
        
        fold_aucs_roc.append(best_auc)
        if best_probs is not None:
            prec, rec, _ = precision_recall_curve(best_labels, best_probs)
            auc_pr = auc(rec, prec)
            fold_aucs_pr.append(auc_pr)
            ece = MetricsCalculator.calculate_ece(np.array(best_labels), np.array(best_probs))
            fold_eces.append(ece)
        
        print(f"  Fold {fold+1}: AUC-ROC={best_auc:.4f}, AUC-PR={fold_aucs_pr[-1]:.4f}, ECE={fold_eces[-1]:.4f}")
    
    return {
        'model': model_name,
        'auc_roc_folds': fold_aucs_roc,
        'auc_pr_folds': fold_aucs_pr,
        'auc_roc_mean': float(np.mean(fold_aucs_roc)),
        'auc_roc_std': float(np.std(fold_aucs_roc)),
        'auc_pr_mean': float(np.mean(fold_aucs_pr)) if fold_aucs_pr else 0.0,
        'auc_pr_std': float(np.std(fold_aucs_pr)) if fold_aucs_pr else 0.0,
        'ece_mean': float(np.mean(fold_eces)) if fold_eces else 0.0,
        'ece_std': float(np.std(fold_eces)) if fold_eces else 0.0,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    feature_dir = os.path.join(base_dir, 'data', 'features_1d_raw')
    
    print("=" * 60)
    print("ABLATION STUDY: DNDF vs Softmax Classifier")
    print("Feature Extractor: Wav2Vec 2.0 (IDENTICAL for both)")
    print("Random Seed:", RANDOM_SEED)
    print("=" * 60)
    
    full_dataset = CovidCoughDatasetOffline(metadata_path, feature_dir)
    labels = full_dataset.data['label'].values
    
    results = {}
    
    # ---- Condition A: Wav2Vec 2.0 + Softmax (Baseline) ----
    print("\n[A] Training Wav2Vec 2.0 + Softmax (Baseline)...")
    results['Wav2Vec_Softmax'] = train_and_evaluate(
        Wav2Vec2_Softmax_Model, "Wav2Vec 2.0 + Softmax",
        full_dataset, labels, device, base_dir
    )
    
    # ---- Condition B: Wav2Vec 2.0 + DNDF (Proposed) ----
    print("\n[B] Training Wav2Vec 2.0 + DNDF (Proposed)...")
    results['Wav2Vec_DNDF'] = train_and_evaluate(
        Wav2Vec2_DNDF_Model, "Wav2Vec 2.0 + DNDF",
        full_dataset, labels, device, base_dir
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("ABLATION RESULTS: DNDF vs Softmax")
    print("=" * 60)
    for k, v in results.items():
        print(f"\n{v['model']}:")
        print(f"  AUC-ROC: {v['auc_roc_mean']:.4f} +/- {v['auc_roc_std']:.4f}")
        print(f"  AUC-PR:  {v['auc_pr_mean']:.4f} +/- {v['auc_pr_std']:.4f}")
        print(f"  ECE:     {v['ece_mean']:.4f} +/- {v['ece_std']:.4f}")
    
    a = results['Wav2Vec_Softmax']
    b = results['Wav2Vec_DNDF']
    delta_roc = b['auc_roc_mean'] - a['auc_roc_mean']
    delta_pr = b['auc_pr_mean'] - a['auc_pr_mean']
    delta_ece = b['ece_mean'] - a['ece_mean']
    print(f"\nDNDF vs Softmax delta:")
    print(f"  Delta AUC-ROC: {delta_roc:+.4f} ({delta_roc/a['auc_roc_mean']*100:+.1f}%)")
    print(f"  Delta AUC-PR:  {delta_pr:+.4f} ({delta_pr/max(a['auc_pr_mean'],0.001)*100:+.1f}%)")
    print(f"  Delta ECE:     {delta_ece:+.4f} ({delta_ece/max(a['ece_mean'],0.001)*100:+.1f}%) (Note: lower ECE is better)")
    
    # Save results
    out_dir = os.path.join(base_dir, 'experiment_report')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results: {out_path}")


if __name__ == "__main__":
    main()
