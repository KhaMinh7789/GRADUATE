import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# HƯỚNG 1: MÔ HÌNH 1D CNN (Tín hiệu thô)
# ==========================================
class CoughCNN1D(nn.Module):
    def __init__(self):
        super(CoughCNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4, 4),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# HƯỚNG 2: MÔ HÌNH 2D CNN (Ảnh Mel-Spectrogram)
# ==========================================
class CoughCNN2D(nn.Module):
    def __init__(self):
        super(CoughCNN2D, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        # Sửa layer đầu cho ảnh 1 kênh (grayscale) thay vì 3 kênh (RGB)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Sửa layer cuối ra 1 giá trị output
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)