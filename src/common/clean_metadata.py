import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'metadata.csv')
    
    print("⏳ Đang quét và lọc trùng lặp trong metadata.csv...")
    
    df = pd.read_csv(csv_path)
    old_len = len(df)
    
    # Xóa các dòng bị trùng lặp đường dẫn file (file_path)
    df_cleaned = df.drop_duplicates(subset=['file_path'], keep='first')
    new_len = len(df_cleaned)
    
    df_cleaned.to_csv(csv_path, index=False)
    
    print(f"🗑️ Đã xóa {old_len - new_len} dòng bị trùng lặp do chạy script nhiều lần.")
    print(f"✅ TỔNG SỐ DATA THỰC TẾ ĐỂ HUẤN LUYỆN LÀ: {new_len} files.")

if __name__ == '__main__':
    main()