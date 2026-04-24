import os
import shutil
import pandas as pd

def main():
    print("⏳ Đang xử lý và gộp dữ liệu Coswara...")
    
    # Thiết lập đường dẫn dựa theo cấu trúc hiện tại của bạn
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    coswara_dir = os.path.join(base_dir, 'data', 'Coswara-Data-master') # Tên thư mục chuẩn theo ảnh
    coswara_csv = os.path.join(coswara_dir, 'combined_data.csv')
    
    dest_audio_dir = os.path.join(base_dir, 'data', 'audios')
    main_csv_path = os.path.join(base_dir, 'data', 'metadata.csv')

    if not os.path.exists(coswara_csv):
        print(f"❌ LỖI: Không tìm thấy file {coswara_csv}.")
        return

    # 1. Đọc file nhãn của Coswara
    print("-> Đang phân tích file nhãn combined_data.csv...")
    df_coswara = pd.read_csv(coswara_csv)
    
    # Lọc nhãn: Khỏe mạnh (healthy) -> 0, Bệnh (các loại positive) -> 1
    label_map = {}
    for index, row in df_coswara.iterrows():
        uid = str(row['id'])
        status = str(row['covid_status']).lower()
        if status == 'healthy':
            label_map[uid] = 0
        elif 'positive' in status: # Bao phủ cả positive_mild, positive_moderate...
            label_map[uid] = 1

    # 2. Quét và copy file âm thanh ho sâu
    print("-> Đang trích xuất file cough-heavy.wav...")
    new_records = []
    count_found = 0

    # Dùng os.walk để quét toàn bộ thư mục con
    for root, dirs, files in os.walk(coswara_dir):
        if 'cough-heavy.wav' in files:
            # Lấy tên thư mục chứa file làm ID bệnh nhân
            patient_id = os.path.basename(root)
            
            if patient_id in label_map:
                label = label_map[patient_id]
                
                # Đổi tên file để không bị trùng (Vd: coswara_12345_heavy.wav)
                new_filename = f"coswara_{patient_id}_heavy.wav"
                src_file = os.path.join(root, 'cough-heavy.wav')
                dst_file = os.path.join(dest_audio_dir, new_filename)
                
                # Copy sang thư mục audios chung
                shutil.copy2(src_file, dst_file)
                
                # Lưu thông tin để đưa vào metadata
                new_records.append({
                    'file_path': new_filename,
                    'label': label
                })
                count_found += 1

    # 3. Ghi nối (Append) vào sổ sách metadata.csv chung
    if new_records:
        df_new = pd.DataFrame(new_records)
        
        # Nối tiếp vào file metadata.csv đã có sẵn (của Virufy)
        if os.path.exists(main_csv_path):
            df_main = pd.read_csv(main_csv_path)
            df_combined = pd.concat([df_main, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        # Lưu lại file
        df_combined.to_csv(main_csv_path, index=False)
        print(f"\n✅ HOÀN TẤT! Đã copy thêm {count_found} file âm thanh từ Coswara.")
        print(f"📊 Tổng số data hiện tại trong thư mục audios (Virufy + Coswara): {len(df_combined)} files.")
    else:
        print("\n❌ Không tìm thấy file âm thanh nào. LƯU Ý: Bạn đã chạy file extract_data.py bên trong thư mục Coswara-Data-master để bung các cục .tar.gz ra chưa?")

if __name__ == '__main__':
    main()