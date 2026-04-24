import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# Gọi các module 1D
from Covid_Cough_Research.src.old_stage.dataset_1d import CovidCoughDataset1D
from Covid_Cough_Research.src.old_stage.model_1d import HybridModel1D

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

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
    except ValueError:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, sens, spec, auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Bắt đầu thực nghiệm trên: {device.type.upper()}")
    print("📌 Chế độ: Nhánh 1D Signal (1D-CNN + DNDF)")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    audio_dir = os.path.join(base_dir, 'data', 'audios')

    print("\n⏳ Đang load dữ liệu 1D từ ổ cứng...")
    full_dataset = CovidCoughDataset1D(metadata_path, audio_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print("⚖️ Đang tính toán phân bố nhãn để cân bằng dữ liệu...")
    train_labels = [full_dataset.data.iloc[i]['label'] for i in train_dataset.indices]
    class_counts = pd.Series(train_labels).value_counts().to_dict()
    num_healthy = class_counts.get(0, 1)
    num_covid = class_counts.get(1, 1)
    
    # Cân bằng dữ liệu (giống hệt nhánh 2D)
    weights = [1.0 / num_healthy if label == 0 else 1.0 / num_covid for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = HybridModel1D(num_classes=1).to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN 1D ({EPOCHS} VÒNG)...\n")
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, acc, sens, spec, auc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        print(f"🟢 Epoch {epoch+1:02d}/{EPOCHS} | Thời gian: {epoch_time:.2f}s")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Metrics -> ACC: {acc:.4f} | SENS: {sens:.4f} | SPEC: {spec:.4f} | AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            model_save_path = os.path.join(base_dir, 'best_1d_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"   🏆 Đã lưu mô hình tốt nhất (AUC tăng lên {best_auc:.4f})")
        print("-" * 60)

    print("🎉 HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN HƯỚNG 1D SIGNAL!")

if __name__ == '__main__':
    main()