"""
cnn.py — AcousticEncoder: knowledge-distilled audio feature extractor.

MobileNet V1-style depthwise-separable CNN trained offline via knowledge
distillation from YAMNet. Converts log-mel spectrograms to compact 16D
embeddings for on-device anomaly detection.

Architecture
------------
Input:  (batch, 1, 64, 61) — log-mel spectrogram of a 0.975 s audio frame
Output: (batch, 16)         — embedding vector

Parameters:  ~547K
Flash (INT8): ~562 KB  (weights + INT32 biases + per-channel quant scales + metadata)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise + pointwise convolution block with BatchNorm and ReLU.

    Reduces parameters ~8-9x vs a standard conv with the same channel counts.
    BN layers are folded into conv weights during TFLite export, so they add
    no runtime cost on device.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pw   = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_dw(self.dw(x)))
        x = F.relu(self.bn_pw(self.pw(x)))
        return x


class AcousticEncoder(nn.Module):
    """
    Knowledge-distilled audio encoder for on-device anomaly detection.

    Trained offline on FSD50K using YAMNet → PCA(16D) as teacher.
    Deployed frozen on Arduino Nano 33 BLE as a TFLite Micro INT8 model.

    Activation shapes through the network (single sample, INT8 bytes):
        Input       (1, 1, 64, 61)   →   3,904 B
        Stem        (1, 32, 32, 31)  →  31,744 B
        B1 DW  ★    (1, 32, 16, 16)  →   8,192 B   peak: 39,936 B ← SRAM bottleneck
        B1 PW       (1, 64, 16, 16)  →  16,384 B
        B2 DW       (1, 64,  8,  8)  →   4,096 B
        B2–B7 ...   ↘ spatial shrinks monotonically
        Pool/Head   (16,)            →      16 B
    """

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32,  64,  stride=2),
            DepthwiseSeparableBlock(64,  128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1),
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 256, stride=1),
            DepthwiseSeparableBlock(256, 512, stride=2),
            DepthwiseSeparableBlock(512, 512, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
