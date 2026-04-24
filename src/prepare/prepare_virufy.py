import os
import shutil
import pandas as pd

def main():
    print("⏳ Đang dọn dẹp và chuẩn bị dữ liệu Virufy...")
    
    # Đường dẫn thư mục (Dựa theo ảnh bạn chụp)
    # Lưu ý: Chạy script từ thư mục gốc Covid_Cough_Research
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(base_dir, 'data', 'virufy-data-main', 'clinical', 'segmented')
    
    dest_audio_dir = os.path.join(base_dir, 'data', 'audios')
    csv_out_path = os.path.join(base_dir, 'data', 'metadata.csv')

    # 1. Tạo thư mục data/audios nếu chưa có
    os.makedirs(dest_audio_dir, exist_ok=True)

    data_records = []

    # Hàm xử lý từng thư mục (neg/pos)
    def process_folder(folder_name, label):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Cảnh báo: Không tìm thấy thư mục {folder_path}")
            return
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3') or file_name.endswith('.wav'):
                # Copy file âm thanh ra thư mục audios chung
                src_file = os.path.join(folder_path, file_name)
                dst_file = os.path.join(dest_audio_dir, file_name)
                shutil.copy2(src_file, dst_file)
                
                # Lưu thông tin vào danh sách để làm file CSV
                data_records.append({
                    'file_path': file_name,
                    'label': label
                })

    # 2. Gom dữ liệu Âm tính (Negative -> Nhãn 0)
    print("-> Đang gom file Âm tính (Negative)...")
    process_folder('neg', 0)

    # 3. Gom dữ liệu Dương tính (Positive -> Nhãn 1)
    print("-> Đang gom file Dương tính (Positive)...")
    process_folder('pos', 1)

    # 4. Tạo file metadata.csv
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv(csv_out_path, index=False)
        print(f"\n✅ HOÀN TẤT! Đã copy {len(df)} file âm thanh vào thư mục data/audios/")
        print(f"✅ Đã tạo thành công file file nhãn tại: {csv_out_path}")
    else:
        print("\n❌ Lỗi: Không tìm thấy file âm thanh nào. Hãy kiểm tra lại đường dẫn giải nén!")

if __name__ == '__main__':
    main()