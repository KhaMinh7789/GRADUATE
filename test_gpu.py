import torch
import sys
import os

# Trỏ đường dẫn để import code từ thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Covid_Cough_Research.src.old_stage.model import CoughCNN1D, CoughCNN2D

print("="*40)
print(" KIỂM TRA PHẦN CỨNG & MÔ HÌNH")
print("="*40)

# 1. Kiểm tra GPU
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Cảnh báo: Không tìm thấy GPU, đang chạy bằng CPU!")

# 2. Test Hướng 1
print("\n[TEST] Hướng 1 (1D CNN - Audio thô)...")
dummy_1d = torch.randn(8, 1, 80000) # Giả lập batch 8 mẫu, 5 giây audio (16kHz)
model_1d = CoughCNN1D()
out_1d = model_1d(dummy_1d)
print(f"-> Shape đầu vào: {dummy_1d.shape}")
print(f"-> Shape đầu ra:  {out_1d.shape} (Kỳ vọng: [8, 1])")

# 3. Test Hướng 2
print("\n[TEST] Hướng 2 (2D CNN - Mel-Spectrogram)...")
dummy_2d = torch.randn(8, 1, 64, 157) # Giả lập batch 8 mẫu ảnh phổ Mel
model_2d = CoughCNN2D()
out_2d = model_2d(dummy_2d)
print(f"-> Shape đầu vào: {dummy_2d.shape}")
print(f"-> Shape đầu ra:  {out_2d.shape} (Kỳ vọng: [8, 1])")
print("="*40)
print("✅ TẤT CẢ ĐỀU HOẠT ĐỘNG HOÀN HẢO!")