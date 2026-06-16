import torch
import torch.nn as nn
from .spatial_channel   import SpatialChannel
from .frequency_channel import FrequencyChannel


class DualChannelDetector(nn.Module):
    """
    Dual-Channel Deepfake Detector
    --------------------------------
    Spatial  : InceptionV3 or ResNet50 → 512-d
    Frequency: FFT CNN                 → 512-d
    Fusion   : Concat → Dense(256) → Sigmoid
    """
    def __init__(self, backbone='inception', feature_dim=512):
        super().__init__()
        self.spatial    = SpatialChannel(backbone, feature_dim)
        self.frequency  = FrequencyChannel(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(
            torch.cat([self.spatial(x), self.frequency(x)], dim=1)
        ).squeeze(1)

    def unfreeze_spatial_top(self, n=3):
        self.spatial.unfreeze_top(n)
