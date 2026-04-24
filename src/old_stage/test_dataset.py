import os
import torch
from Covid_Cough_Research.src.old_stage.dataset import CovidCoughDataset
from torch.utils.data import DataLoader

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    audio_dir = os.path.join(base_dir, 'data', 'audios')

    print("⏳ Đang khởi tạo Dataset...")
    dataset = CovidCoughDataset(metadata_path, audio_dir)
    print(f"✅ Đã tải thành công Dataset với {len(dataset)} mẫu.")

    # Tạo DataLoader bốc ngẫu nhiên 4 mẫu (batch_size=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Lấy thử 1 mẻ (batch) dữ liệu đầu tiên
    features, labels = next(iter(dataloader))
    
    print("\n--- TEST KẾT QUẢ ---")
    print(f"Kích thước 1 mẻ đặc trưng 2D (Batch_size, Channel, Mel_bins, Time_steps): {features.shape}")
    print(f"Nhãn của 4 mẫu này: {labels}")

if __name__ == "__main__":
    main()