import os
import torch
import torchaudio
from tqdm import tqdm
import pandas as pd
import subprocess
import io

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    audio_dir = os.path.join(base_dir, 'data', 'audios')
    
    save_dir = os.path.join(base_dir, 'data', 'features_1d_raw')
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    target_sr = 16000
    max_length = target_sr * 5 # 5 giây = 80,000 điểm

    print(f"🚀 BẮT ĐẦU TRÍCH XUẤT 1D (CHẾ ĐỘ FFMPEG TRỰC TIẾP) CHO {len(df)} MẪU")
    
    for i in tqdm(range(len(df))):
        file_name = df.iloc[i]['file_path']
        file_path = os.path.join(audio_dir, file_name)
        
        try:
            # 1. Vũ khí tối thượng: Gọi trực tiếp hệ thống FFmpeg
            # Tự động ép tất cả về Mono (1 kênh) và Sample Rate 16000Hz
            command = [
                'ffmpeg', '-i', file_path,
                '-ac', '1', '-ar', str(target_sr),
                '-f', 'wav', '-hide_banner', '-loglevel', 'error', '-'
            ]
            
            # Khởi chạy luồng ẩn để đọc dữ liệu
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Lỗi giải mã: {err.decode('utf-8', errors='ignore')}")
            
            # 2. PyTorch đón ngay dữ liệu chuẩn WAV từ bộ nhớ RAM
            waveform, sr = torchaudio.load(io.BytesIO(out), format="wav")

            # 3. Cắt gọt hoặc đệm (Padding) để cố định độ dài 80,000 điểm
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            else:
                pad_amount = max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            # 4. Chuẩn hóa (Z-score normalization) - Bắt buộc cho Wav2Vec 2.0
            waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)

            # 5. Lưu xuống ổ cứng
            save_path = os.path.join(save_dir, f"{file_name}.pt")
            torch.save(waveform.squeeze(0), save_path)
            
        except Exception as e:
            # Nếu có lỗi, in ra rõ ràng để bắt bệnh
            print(f"\n❌ Lỗi ở file {file_name}: {e}")

if __name__ == '__main__':
    main()