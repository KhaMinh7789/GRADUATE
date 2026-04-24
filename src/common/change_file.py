import os
from pathlib import Path

# Chỉ định folder cần thay đổi
folder_path = "D:\\GROWTH\\GRADUTION_MASTER\\Covid_Cough_Research\\data\\coughvid_20211012"  # Thay đổi đường dẫn tại đây

# Chuyển đổi thành Path object
folder = Path(folder_path)

# Tìm tất cả file .wav và đổi tên thành .webm
for wav_file in folder.glob("*.wav"):
    new_name = wav_file.with_suffix(".webm")
    wav_file.rename(new_name)
    print(f"Đổi tên: {wav_file.name} → {new_name.name}")

print("Hoàn thành!")