import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class CovidCoughDatasetOffline(Dataset):
    def __init__(self, metadata_path, feature_dir, filter_expert=False, use_metadata=False):
        self.feature_dir = feature_dir
        self.use_metadata = use_metadata
        df = pd.read_csv(metadata_path)
        
        # --- BƯỚC LỌC DỮ LIỆU THÔNG MINH ---
        if filter_expert and 'expert_evaluation_score' in df.columns:
            # Lọc chỉ lấy các mẫu có nhãn đáng tin cậy cao
            df = df[df['expert_evaluation_score'] >= 0.8]
            print(f"✔️ Đã áp dụng filter_expert: Loại bỏ các mẫu nhiễu nhãn (Label Noise).")
            
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
        
        if self.use_metadata:
            # Rút trích một số thông tin metadata để đưa vào mạng (ví dụ: tuổi, giới tính, tiền sử bệnh)
            # Default = 0 nếu không có
            age = float(row.get('age', 0) or 0) / 100.0  # Normalize to 0-1
            gender = 1.0 if str(row.get('gender', '')).lower() in ['m', 'male'] else 0.0
            smoke = 1.0 if str(row.get('smoker', '')).lower() in ['true', 'yes', '1'] else 0.0
            asthma = 1.0 if str(row.get('asthma', '')).lower() in ['true', 'yes', '1'] else 0.0
            
            meta_tensor = torch.tensor([age, gender, smoke, asthma], dtype=torch.float32)
            return feature, label, meta_tensor
        
        return feature, label