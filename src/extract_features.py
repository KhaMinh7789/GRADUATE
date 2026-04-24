import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Covid_Cough_Research.src.old_stage.dataset import CovidCoughDataset # Import class dataset cũ của bạn

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    audio_dir = os.path.join(base_dir, 'data', 'audios')
    
    # Tạo thư mục mới để chứa ảnh phổ dạng Tensor
    save_dir = os.path.join(base_dir, 'data', 'features_2d_ast')
    os.makedirs(save_dir, exist_ok=True)

    print("⏳ Đang khởi tạo bộ đọc âm thanh gốc...")
    dataset = CovidCoughDataset(metadata_path, audio_dir)
    
    print(f"🚀 BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CHO {len(dataset)} MẪU ÂM THANH")
    print(f"Thư mục lưu trữ: {save_dir}")
    
    for i in tqdm(range(len(dataset))):
        # Lấy tên file từ metadata (Giả sử cột tên file là 'file_path'. Nếu file CSV của bạn dùng tên cột khác như 'filename', hãy đổi chữ 'file_path' tương ứng)
        file_name = dataset.data.iloc[i]['file_path'] 
        
        try:
            # 1. CPU đọc file .wav, chuyển thành Mel-Spectrogram (shape: [1, n_mels, time])
            spectrogram, label = dataset[i]
            
            # 2. Ép chuẩn kích thước 128x1024 cho AST 
            spectrogram = spectrogram.unsqueeze(0) # Thành [1, 1, n_mels, time] để nội suy
            spectrogram_resized = F.interpolate(spectrogram, size=(128, 1024), mode='bilinear', align_corners=False)
            spectrogram_resized = spectrogram_resized.squeeze(0) # Trả về [1, 128, 1024]
            
            # 3. Lưu thành file .pt xuống ổ SSD
            save_path = os.path.join(save_dir, f"{file_name}.pt")
            torch.save(spectrogram_resized, save_path)
            
        except Exception as e:
            print(f"\n❌ Lỗi ở file {file_name}: {e}")

    print("\n🎉 HOÀN TẤT! Dữ liệu đã sẵn sàng để GPU huấn luyện siêu tốc.")

if __name__ == '__main__':
    main()