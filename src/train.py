import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

# Import file của bạn (lưu ý: đảm bảo tên file model.py và dataset.py viết đúng)
from model import CoughCNN1D, CoughCNN2D
from dataset import RealCoughDataset

# ==========================================
# CẤU HÌNH THỰC NGHIỆM
# ==========================================
MODE = '2D'           # Thử 2D trước vì nó thường cho kết quả cao hơn
BATCH_SIZE = 16       
EPOCHS = 10           # Tăng lên 10 vòng vì đây là data thật
LEARNING_RATE = 1e-4

# Thiết lập đường dẫn tuyệt đối để không bao giờ bị lỗi FileNotFoundError
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'metadata.csv')
AUDIO_DIR = os.path.join(BASE_DIR, 'data', 'audios')

# ==========================================
# HÀM ĐÁNH GIÁ (EVALUATION METRICS)
# ==========================================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
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

# ==========================================
# VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Bắt đầu thực nghiệm trên: {device}")
    print(f"📌 Chế độ đang chạy: Hướng {MODE} CNN với dữ liệu Virufy (Segmented)")

    print("⏳ Đang load dữ liệu thật...")
    full_dataset = RealCoughDataset(csv_file=CSV_PATH, audio_dir=AUDIO_DIR, mode=MODE)
    
    # Chia 80% Train, 20% Validate
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CoughCNN1D() if MODE == '1D' else CoughCNN2D()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"🔥 Bắt đầu Training {EPOCHS} Epochs...\n")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validate
        val_loss, acc, sens, spec, auc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Metrics -> ACC: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | AUC: {auc:.4f}\n")

    print("✅ Hoàn thành huấn luyện trên dữ liệu thực tế!")

if __name__ == '__main__':
    main()