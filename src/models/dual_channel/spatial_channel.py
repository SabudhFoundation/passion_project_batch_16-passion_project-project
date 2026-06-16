import torch.nn as nn
from torchvision import models


class SpatialChannel(nn.Module):
    def __init__(self, backbone='inception', feature_dim=512, freeze=True):
        super().__init__()
        if backbone == 'inception':
            base = models.inception_v3(pretrained=True)
            base.aux_logits = False
            base.AuxLogits  = None
            in_features     = base.fc.in_features
            base.fc         = nn.Identity()
            self.backbone   = base
            self.name       = 'InceptionV3'

        elif backbone == 'resnet':
            base          = models.resnet50(pretrained=True)
            in_features   = base.fc.in_features
            base.fc       = nn.Identity()
            self.backbone = base
            self.name     = 'ResNet50'
        else:
            raise ValueError("backbone must be 'inception' or 'resnet'")

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        print(f"[SpatialChannel] {self.name} loaded | Frozen: {freeze}")

    def unfreeze_top(self, n=3):
        for layer in list(self.backbone.children())[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"[SpatialChannel] Unfroze top-{n} | Trainable: {trainable:,}")

    def forward(self, x):
        return self.projection(self.backbone(x))
