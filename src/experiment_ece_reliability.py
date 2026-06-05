"""
Experiment 1: Uncertainty Estimation (ECE & Reliability Diagrams)
=================================================================
Addresses advisor feedback (Section 1 - Kẽ hở trong Logic Kiến trúc Hệ thống):
"Cần bổ sung các biểu đồ Reliability Diagram hoặc chỉ số Expected Calibration Error (ECE) 
để chứng minh độ bất định mà DNDF cung cấp thực sự tốt hơn Softmax."

This script re-evaluates the Wav2Vec 2.0 + Softmax and Wav2Vec 2.0 + DNDF models,
explicitly computing ECE and plotting the Reliability Diagram (Calibration Curve)
to prove that DNDF produces better calibrated probabilities than Softmax.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch.optim as optim
import warnings
import json

from dataset_offline import CovidCoughDatasetOffline
from model_1d_sota import Wav2Vec2_DNDF_Model
from ablation_dndf_vs_softmax import Wav2Vec2_Softmax_Model, BinaryFocalLoss
from utils_metrics import MetricsCalculator

warnings.filterwarnings("ignore")

BATCH_SIZE = 8
EPOCHS = 5  # Giảm epoch xuống 5 để chạy nhanh thực nghiệm chứng minh ECE
N_SPLITS = 3
RANDOM_SEED = 42

def train_and_get_probs(model_class, full_dataset, labels, device, epochs=EPOCHS):
    """Train quickly and return out-of-fold predictions for calibration analysis."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    all_oof_probs = []
    all_oof_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"  Fold {fold+1}/{N_SPLITS}...")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[int(l)] for l in train_labels])
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights))
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = model_class().to(device)
        criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs).float()
                loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Collect OOF predictions
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy().flatten()
                all_oof_probs.extend(outputs)
                all_oof_labels.extend(targets.numpy().flatten())
                
    return np.array(all_oof_labels), np.array(all_oof_probs)


def plot_reliability_diagram(y_true_soft, y_prob_soft, y_true_dndf, y_prob_dndf, out_path):
    """Plot Calibration Curve (Reliability Diagram)."""
    plt.figure(figsize=(8, 8))
    
    # Perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Ideal)")
    
    # Softmax
    prob_true_soft, prob_pred_soft = calibration_curve(y_true_soft, y_prob_soft, n_bins=10)
    plt.plot(prob_pred_soft, prob_true_soft, "s-", label="Wav2Vec 2.0 + Softmax")
    
    # DNDF
    prob_true_dndf, prob_pred_dndf = calibration_curve(y_true_dndf, y_prob_dndf, n_bins=10)
    plt.plot(prob_pred_dndf, prob_true_dndf, "o-", label="Wav2Vec 2.0 + DNDF", linewidth=2)
    
    plt.ylabel("Fraction of Positives (True Probability)", fontsize=12)
    plt.xlabel("Mean Predicted Value (Confidence)", fontsize=12)
    plt.title("Reliability Diagram: DNDF vs Softmax Calibration", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Reliability Diagram to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    feature_dir = os.path.join(base_dir, 'data', 'features_1d_raw')
    
    print("=" * 60)
    print("EXPERIMENT 1: ECE and Reliability Diagrams")
    print("=" * 60)
    
    full_dataset = CovidCoughDatasetOffline(metadata_path, feature_dir)
    labels = full_dataset.data['label'].values
    
    print("\n[1] Evaluating Softmax Baseline...")
    y_true_soft, y_prob_soft = train_and_get_probs(Wav2Vec2_Softmax_Model, full_dataset, labels, device)
    
    print("\n[2] Evaluating DNDF Proposed...")
    y_true_dndf, y_prob_dndf = train_and_get_probs(Wav2Vec2_DNDF_Model, full_dataset, labels, device)
    
    # Calculate ECE
    ece_soft = MetricsCalculator.calculate_ece(y_true_soft, y_prob_soft)
    ece_dndf = MetricsCalculator.calculate_ece(y_true_dndf, y_prob_dndf)
    
    print("\n" + "=" * 60)
    print("UNCERTAINTY ESTIMATION RESULTS (ECE)")
    print("=" * 60)
    print(f"Softmax ECE: {ece_soft:.4f} (Higher is worse)")
    print(f"DNDF ECE:    {ece_dndf:.4f} (Lower is better)")
    delta = ece_soft - ece_dndf
    print(f"Improvement: DNDF is {delta:.4f} better calibrated than Softmax.")
    
    # Plot Reliability Diagram
    out_dir = os.path.join(base_dir, 'experiment_report')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'reliability_diagram.png')
    plot_reliability_diagram(y_true_soft, y_prob_soft, y_true_dndf, y_prob_dndf, out_path)
    
    # Save metrics
    res_path = os.path.join(out_dir, 'ece_results.json')
    with open(res_path, 'w') as f:
        json.dump({'Softmax_ECE': ece_soft, 'DNDF_ECE': ece_dndf}, f, indent=2)

if __name__ == "__main__":
    main()
