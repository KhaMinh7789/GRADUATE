import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class RealCoughDataset(Dataset):
    def __init__(self, csv_file, audio_dir, target_sr=16000, max_sec=5, mode='2D'):
        """
        csv_file: Đường dẫn tới file metadata.csv (chứa cột 'file_path' và 'label')
        audio_dir: Thư mục chứa các file âm thanh
        mode: '1D' (tín hiệu sóng thô) hoặc '2D' (Ảnh phổ Mel)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.max_length = target_sr * max_sec # 16000 * 5 = 80000 mẫu
        self.mode = mode

        # Bộ chuyển đổi sang Mel-spectrogram cho mô hình 2D
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. Lấy tên file và nhãn từ file CSV
        audio_name = str(self.data_frame.iloc[idx]['file_path'])
        file_path = os.path.join(self.audio_dir, audio_name)
        label = float(self.data_frame.iloc[idx]['label'])

        # 2. Load âm thanh
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Nếu file lỗi, trả về một tensor toàn số 0 để không bị gián đoạn training
            waveform = torch.zeros(1, self.max_length)
            sr = self.target_sr

        # 3. Resample nếu Sample Rate không chuẩn 16kHz
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Chuyển về Mono (1 kênh) nếu file là Stereo (2 kênh)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Padding / Truncating (Cắt hoặc chèn thêm để mọi file dài chuẩn 5 giây)
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # 5. Trả về định dạng theo Hướng thực nghiệm
        if self.mode == '1D':
            return waveform, torch.tensor(label, dtype=torch.float32)
        elif self.mode == '2D':
            mel_spec = self.mel_spectrogram(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            return mel_spec_db, torch.tensor(label, dtype=torch.float32)