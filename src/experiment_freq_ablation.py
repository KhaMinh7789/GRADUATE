"""
Experiment 3: Objective XAI Validation (Frequency Ablation)
===========================================================
Addresses advisor feedback (Section 2 - Tính khách quan của các "Tính năng ẩn" trong XAI):
"Việc tự ánh xạ Feature 370 vào dải tần 2.5–4.5 kHz mang tính hậu kiểm (post-hoc) và có thể mang tính chủ quan. 
Cần thực hiện Ablation on Frequency Bins (triệt tiêu từng dải tần số trên dữ liệu đầu vào) 
để chứng minh một cách thực nghiệm rằng khi mất dải tần 2-6 kHz, mô hình thực sự thất bại."

This script validates the LIME XAI explanation objectively. It applies a digital band-stop filter 
using Fast Fourier Transform (FFT) to completely remove the 2000Hz - 6000Hz frequency band from the 
raw audio waveform before it enters the Wav2Vec 2.0 feature extractor. 

If the model relies on the "explosive phase" of the cough (which XAI maps to 2.5-4.5kHz), 
removing this band will cause a catastrophic drop in AUC-PR.
"""
import os
import torch
import numpy as np
import json
import warnings
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim

from dataset_offline import CovidCoughDatasetOffline
from model_1d_sota import Wav2Vec2_DNDF_Model
from ablation_dndf_vs_softmax import BinaryFocalLoss
from utils_metrics import MetricsCalculator

warnings.filterwarnings("ignore")

BATCH_SIZE = 8
EPOCHS = 5
N_SPLITS = 3
RANDOM_SEED = 42

def remove_frequency_band(audio_tensor, sr=16000, low_freq=2000, high_freq=6000):
    """
    Applies a strict band-stop filter by zeroing out FFT bins.
    audio_tensor: shape (Batch, Length) or (Length,)
    """
    N = audio_tensor.shape[-1]
    
    # Forward FFT
    fft = torch.fft.rfft(audio_tensor)
    
    # Calculate frequencies for each bin
    freqs = torch.fft.rfftfreq(N, d=1/sr).to(audio_tensor.device)
    
    # Create mask for the band we want to destroy (e.g. 2000Hz to 6000Hz)
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    
    # Zero out those frequencies
    fft[..., mask] = 0
    
    # Inverse FFT back to time domain
    return torch.fft.irfft(fft, n=N)


def train_and_evaluate_ablation(model_class, full_dataset, labels, device, ablate_freq=False):
    """Run CV with optional frequency ablation on the inputs."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs_pr = []
    
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
        
        for epoch in range(EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                
                # ---> APPLY FREQUENCY ABLATION IF REQUESTED <---
                if ablate_freq:
                    inputs = remove_frequency_band(inputs, sr=16000, low_freq=2000, high_freq=6000)
                
                targets = targets.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs).float()
                loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                
                # ---> APPLY FREQUENCY ABLATION TO VALIDATION SET TOO <---
                if ablate_freq:
                    inputs = remove_frequency_band(inputs, sr=16000, low_freq=2000, high_freq=6000)
                    
                outputs = model(inputs).cpu().numpy().flatten()
                all_probs.extend(outputs)
                all_labels.extend(targets.numpy().flatten())
                
        metrics = MetricsCalculator.calculate_all_metrics(all_labels, np.round(all_probs), all_probs)
        fold_aucs_pr.append(metrics['auc_pr'])
        
    return np.mean(fold_aucs_pr)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    feature_dir = os.path.join(base_dir, 'data', 'features_1d_raw')
    
    print("=" * 60)
    print("EXPERIMENT 3: Objective XAI Validation (Frequency Ablation)")
    print("=" * 60)
    
    full_dataset = CovidCoughDatasetOffline(metadata_path, feature_dir)
    labels = full_dataset.data['label'].values
    
    # Note: For time-saving in this demonstration, we only run the Ablated model.
    # The Baseline DNDF AUC-PR is typically ~0.526 from the Ablation Study (Bảng 3).
    # You can re-run the Baseline here if you want an exact 1-to-1 matching seed comparison.
    
    print("\nTraining and Evaluating Wav2Vec 2.0 + DNDF with 2-6 kHz Band ABLATED...")
    ablated_auc_pr = train_and_evaluate_ablation(Wav2Vec2_DNDF_Model, full_dataset, labels, device, ablate_freq=True)
    
    print("\n" + "=" * 60)
    print("FREQUENCY ABLATION RESULTS")
    print("=" * 60)
    print(f"Baseline AUC-PR (from prev study): ~0.5260")
    print(f"Ablated (No 2-6 kHz) AUC-PR:      {ablated_auc_pr:.4f}")
    
    # Save result
    out_dir = os.path.join(base_dir, 'experiment_report')
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, 'frequency_ablation_results.json')
    with open(res_path, 'w') as f:
        json.dump({
            'Baseline_AUC_PR_approx': 0.5260,
            'Ablated_2_6kHz_AUC_PR': float(ablated_auc_pr)
        }, f, indent=2)

if __name__ == "__main__":
    main()
