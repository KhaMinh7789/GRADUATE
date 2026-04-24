import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class CoughDataset1D(Dataset):
    def __init__(self, csv_file, audio_dir, target_sr=16000, max_length_s=5):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        # Tính tổng số mẫu (Ví dụ: 16000 * 5 = 80000 mẫu)
        self.max_length = target_sr * max_length_s 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_path']
        label = self.df.iloc[idx]['label']
        file_path = os.path.join(self.audio_dir, file_name)

        # Load file âm thanh
        waveform, sr = torchaudio.load(file_path)

        # 1. Chuyển Stereo thành Mono (nếu có)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Đồng bộ Sample Rate (Resample)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 3. Cắt gọt (Truncate) hoặc độn thêm (Pad) để cùng kích thước
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Output chuẩn cho 1D model: [1, 80000]
        return waveform, torch.tensor(label, dtype=torch.float32)