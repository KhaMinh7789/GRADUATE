import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
import warnings

# Tắt các cảnh báo không cần thiết của librosa/torchaudio
warnings.filterwarnings("ignore")

# Import các module của chúng ta
from Covid_Cough_Research.src.old_stage.dataset import CovidCoughDataset
from Covid_Cough_Research.src.old_stage.model import HybridModel2D

# ==========================================
# CẤU HÌNH THỰC NGHIỆM (HYPERPARAMETERS)
# ==========================================
BATCH_SIZE = 32      # Đưa 32 file vào card đồ họa cùng lúc
EPOCHS = 20          # Lặp qua toàn bộ 13,000 data 20 lần
LEARNING_RATE = 1e-4 # Tốc độ học (nhỏ thì học chậm nhưng chắc)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        # alpha=0.8: Ép mô hình dành 80% sự ưu tiên cho việc tìm ra COVID-19 (nhãn 1)
        self.alpha = alpha 
        # gamma=2.0: Trừng phạt mạnh những ca dự đoán sai với độ tự tin cao
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Xác suất dự đoán đúng
        
        # Gán trọng số alpha cho ca COVID và (1-alpha) cho ca Khỏe mạnh
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

def evaluate(model, loader, criterion, device):
    """Hàm đánh giá mô hình trên tập Validation"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Dự đoán (Ngưỡng 0.5)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Tính toán các chỉ số Y tế
    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, zero_division=0) # Độ nhạy (Sensitivity)
    
    # Tính Độ đặc hiệu (Specificity) từ Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Tính AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return running_loss / len(loader.dataset), acc, sens, spec, auc

def main():
    # 1. Cấu hình phần cứng (Ưu tiên dùng GPU RTX 3500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Bắt đầu thực nghiệm trên: {device.type.upper()}")
    print("📌 Chế độ: Nhánh 2D Spectrogram (Hybrid CNN + DNDF)")

    # 2. Chuẩn bị Dữ liệu
    print("\n⏳ Đang load dữ liệu từ ổ cứng (có thể mất vài chục giây)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    audio_dir = os.path.join(base_dir, 'data', 'audios')

    full_dataset = CovidCoughDataset(metadata_path, audio_dir)
    
    # Chia 80% Train, 20% Validate
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader giúp bốc data theo từng Batch để ném lên GPU
    # (ĐÃ XÓA TRAIN_LOADER CŨ Ở ĐÂY VÀ THAY BẰNG CƠ CHẾ DƯỚI ĐÂY)

    # --- BẮT ĐẦU ĐOẠN XỬ LÝ MẤT CÂN BẰNG ---
    print("\n⚖️ Đang tính toán phân bố nhãn để cân bằng dữ liệu...")
    train_labels = [full_dataset.data.iloc[i]['label'] for i in train_dataset.indices]
    import pandas as pd
    class_counts = pd.Series(train_labels).value_counts().to_dict()
    num_healthy = class_counts.get(0, 1)
    num_covid = class_counts.get(1, 1)
    
    print(f"   Thống kê tập Train: {num_healthy} Khỏe mạnh (0) | {num_covid} COVID-19 (1)")
    
    # Tính trọng số: Nhãn nào ít thì trọng số khi bốc sẽ cao lên
    weights = [1.0 / num_healthy if label == 0 else 1.0 / num_covid for label in train_labels]
    
    # Cấu hình Sampler để ép mô hình học tỷ lệ 50-50
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Cập nhật train_loader (Dùng sampler thì bắt buộc bỏ shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # --- KẾT THÚC ĐOẠN XỬ LÝ MẤT CÂN BẰNG ---

    print(f"✅ Đã chia tập dữ liệu: {train_size} mẫu Train | {val_size} mẫu Validate.")

    # 3. Khởi tạo Mô hình và Thuật toán tối ưu
    model = HybridModel2D(num_classes=1).to(device)
    
    # Do DNDF đã dùng Sigmoid ở lá cây, ta dùng BCELoss thay vì BCEWithLogitsLoss
    criterion = nn.BCELoss() 
    # KÍCH HOẠT FOCAL LOSS ĐỂ TRỊ BỆNH BỎ SÓT COVID-19
    # criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🔥 BẮT ĐẦU HUẤN LUYỆN ({EPOCHS} VÒNG)...\n")
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()    # Xóa bộ nhớ gradient
            outputs = model(inputs)  # Suy luận dự đoán
            loss = criterion(outputs, labels) # Tính sai số
            loss.backward()          # Lan truyền ngược
            optimizer.step()         # Cập nhật trọng số

            train_loss += loss.item() * inputs.size(0)
            
            # In tiến độ chạy của từng Batch (cho đỡ sốt ruột)
            if (batch_idx + 1) % 50 == 0:
                print(f"   [Epoch {epoch+1}] Đã train được {batch_idx+1}/{len(train_loader)} batches...")

        train_loss /= len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        val_loss, acc, sens, spec, auc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        # In báo cáo của Epoch
        print(f"🟢 Epoch {epoch+1:02d}/{EPOCHS} | Thời gian: {epoch_time:.2f}s")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Metrics -> ACC: {acc:.4f} | SENS: {sens:.4f} | SPEC: {spec:.4f} | AUC: {auc:.4f}")
        
        # Lưu lại mô hình tốt nhất
        if auc > best_auc:
            best_auc = auc
            model_save_path = os.path.join(base_dir, 'best_2d_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"   🏆 Đã lưu mô hình tốt nhất (AUC tăng lên {best_auc:.4f})")
        print("-" * 60)

    print("🎉 HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN HƯỚNG 2D SPECTROGRAM!")

if __name__ == '__main__':
    main()