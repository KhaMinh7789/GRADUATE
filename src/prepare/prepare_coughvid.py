import os
import shutil
import pandas as pd

def main():
    print("⏳ Đang xử lý 'Trùm cuối' COUGHVID...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # LƯU Ý: Đổi tên 'coughvid_20211012' cho khớp với thư mục giải nén của bạn
    coughvid_dir = os.path.join(base_dir, 'data', 'coughvid_20211012') 
    coughvid_csv = os.path.join(coughvid_dir, 'metadata_compiled.csv')
    
    dest_audio_dir = os.path.join(base_dir, 'data', 'audios')
    main_csv_path = os.path.join(base_dir, 'data', 'metadata.csv')

    if not os.path.exists(coughvid_csv):
        print(f"❌ LỖI: Không tìm thấy file {coughvid_csv}.")
        print("Hãy đảm bảo bạn đã giải nén bộ COUGHVID và đặt đúng tên thư mục.")
        return

    # 1. Đọc và lọc dữ liệu cực gắt
    print("-> Đang quét file nhãn metadata_compiled.csv và lọc nhiễu...")
    df = pd.read_csv(coughvid_csv)
    
    # Lọc 1: Bỏ các file rác (xác suất là tiếng ho < 0.8)
    df = df[df['cough_detected'] >= 0.8]
    
    # Lọc 2: Chỉ lấy người chắc chắn bị COVID hoặc chắc chắn Khỏe mạnh
    valid_statuses = ['COVID-19', 'healthy']
    df = df[df['status'].isin(valid_statuses)]

    # 2. Tìm, đổi tên và copy file
    print(f"-> Đã lọc được {len(df)} mẫu đạt chuẩn. Đang tiến hành copy...")
    new_records = []
    count_found = 0

    for index, row in df.iterrows():
        uuid = str(row['uuid'])
        status = str(row['status'])
        
        # Gán nhãn: COVID-19 -> 1, healthy -> 0
        label = 1 if status == 'COVID-19' else 0
        
        # COUGHVID có thể lưu file dưới dạng .webm hoặc .ogg
        src_webm = os.path.join(coughvid_dir, f"{uuid}.webm")
        src_ogg = os.path.join(coughvid_dir, f"{uuid}.ogg")
        
        src_file = None
        ext = None
        
        if os.path.exists(src_webm):
            src_file = src_webm
            ext = '.webm'
        elif os.path.exists(src_ogg):
            src_file = src_ogg
            ext = '.ogg'
            
        if src_file:
            # Đổi tên file để tránh trùng lặp
            new_filename = f"coughvid_{uuid}{ext}"
            dst_file = os.path.join(dest_audio_dir, new_filename)
            
            # Copy file sang thư mục audios chung
            shutil.copy2(src_file, dst_file)
            
            # Ghi chép vào sổ sách
            new_records.append({
                'file_path': new_filename,
                'label': label
            })
            count_found += 1
            
            # In tiến độ cho đỡ chán
            if count_found % 500 == 0:
                print(f"   Đã copy {count_found} files...")

    # 3. Ghi nối (Append) vào metadata.csv chung
    if new_records:
        df_new = pd.DataFrame(new_records)
        if os.path.exists(main_csv_path):
            df_main = pd.read_csv(main_csv_path)
            df_combined = pd.concat([df_main, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined.to_csv(main_csv_path, index=False)
        print(f"\n✅ HOÀN TẤT! Đã vớt được {count_found} file âm thanh siêu xịn từ COUGHVID.")
        print(f"📊 TỔNG CỘNG LƯỢNG DATA SAU PHASE 1 (Virufy + Coswara + COUGHVID): {len(df_combined)} files.")
    else:
        print("\n❌ Không copy được file nào. Bạn kiểm tra lại định dạng file xem nhé!")

if __name__ == '__main__':
    main()