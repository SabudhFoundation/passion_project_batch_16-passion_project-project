import torch
import torch.nn as nn


class FrequencyChannel(nn.Module):
    """
    Frequency branch of the dual-channel detector.

    Converts the input RGB tensor to a log-magnitude FFT spectrum,
    then processes it through a lightweight CNN to produce a
    feature_dim-dimensional embedding.

    Input : (B, 3, H, W) — pixels normalised to [-1, 1]
    Output: (B, feature_dim)
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        print(f"[FrequencyChannel] FFT-CNN branch initialised | out_dim={feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1] — convert to [0, 1]
        x = (x + 1.0) / 2.0
        # Grayscale via luminance weights
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)
        # 2D FFT → shift zero-freq to centre → log-magnitude
        fft = torch.fft.fftshift(torch.fft.fft2(gray))
        mag = torch.log1p(torch.abs(fft)).unsqueeze(1)  # (B, 1, H, W)
        return self.projection(self.cnn(mag).flatten(1))
