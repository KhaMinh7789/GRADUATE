import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class CovidCoughDatasetOffline(Dataset):
    def __init__(self, metadata_path, feature_dir):
        self.feature_dir = feature_dir
        df = pd.read_csv(metadata_path)
        
        # --- BƯỚC LỌC DỮ LIỆU THÔNG MINH ---
        # Tự động kiểm tra file Tensor (.pt) có tồn tại không. Nếu có mới đưa vào huấn luyện.
        valid_rows = []
        for idx, row in df.iterrows():
            file_name = row['file_path'] 
            pt_path = os.path.join(feature_dir, f"{file_name}.pt")
            
            if os.path.exists(pt_path):
                valid_rows.append(row)
                
        # Cập nhật lại dataframe chỉ chứa dữ liệu sạch
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"✔️ Đã tải {len(self.data)} mẫu HỢP LỆ từ thư mục {os.path.basename(feature_dir)}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['file_path']
        label = row['label']
        
        # Đọc trực tiếp Tensor đã tiền xử lý
        feature_path = os.path.join(self.feature_dir, f"{file_name}.pt")
        feature = torch.load(feature_path)
        
        return feature, label