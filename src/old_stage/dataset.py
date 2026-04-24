import os
import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class CovidCoughDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_sr=16000, max_duration=5.0):
        """
        Khởi tạo Dataset.
        :param metadata_path: Đường dẫn tới file metadata.csv (chứa nhãn).
        :param audio_dir: Đường dẫn tới thư mục chứa toàn bộ file âm thanh.
        :param target_sr: Tần số lấy mẫu chuẩn (16000 Hz là chuẩn cho giọng nói/ho).
        :param max_duration: Độ dài tối đa của file âm thanh (giây).
        """
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.max_length = int(target_sr * max_duration) # Tính tổng số frame
        
        # Đọc sổ sách metadata
        self.data = pd.read_csv(metadata_path)
        
        # Bộ chuyển đổi âm thanh sang ảnh phổ Mel-spectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64 # Trích xuất 64 dải tần số đặc trưng
        )
        
        # Chuyển đổi biên độ (Amplitude) sang Decibel (dB) để AI dễ học hơn
        self.amplitude_to_db = T.AmplitudeToDB()

        # --- THÊM MỚI: Kỹ thuật SpecAugment ---
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Lấy thông tin từ CSV
        file_name = self.data.iloc[idx]['file_path']
        label = int(self.data.iloc[idx]['label'])
        file_path = os.path.join(self.audio_dir, file_name)

        # 2. Đọc file âm thanh
        try:
            waveform, sr = torchaudio.load(file_path)

            # 🔴 CHẶN LỖI: Kiểm tra xem file có bị rỗng không (0 frames)
            if waveform.numel() == 0 or waveform.shape[1] == 0:
                raise ValueError("File âm thanh rỗng (0 giây)!")
        except Exception as e:
            # Xử lý lỗi nếu file hỏng, không tồn tại hoặc bị rỗng
            # Trả về một mảng toàn số 0 (khoảng lặng) thay thế
            waveform = torch.zeros(1, self.max_length)
            sr = self.target_sr

        # Nếu âm thanh có 2 kênh (Stereo), gộp lại thành 1 kênh (Mono)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resampling (Chuẩn hóa tần số lấy mẫu về 16kHz)
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 4. Padding / Truncating (Cắt gọt cho bằng độ dài chuẩn)
        current_length = waveform.shape[1]
        if current_length > self.max_length:
            # Nếu dài quá -> Cắt bớt phần đuôi
            waveform = waveform[:, :self.max_length]
        elif current_length < self.max_length:
            # Nếu ngắn quá -> Chèn thêm khoảng lặng (số 0) vào đuôi
            padding = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # 5. Rút trích đặc trưng 2D: Chuyển Sóng âm -> Mel-Spectrogram -> Decibel
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # --- THÊM MỚI: Áp dụng SpecAugment ngẫu nhiên (50% xác suất) ---
        import random
        if random.random() < 0.5:
            mel_spec_db = self.freq_masking(mel_spec_db)
            mel_spec_db = self.time_masking(mel_spec_db)

        # Trả về: Ảnh phổ 2D (Tensor) và Nhãn (Tensor)
        return mel_spec_db, torch.tensor(label, dtype=torch.long)