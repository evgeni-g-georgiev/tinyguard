"""
Tiny encoder architectures for f_c.

These must satisfy Arduino Nano 33 BLE deployment constraints:
  - ~4-8K trainable parameters (encoder only)
  - INT8 quantisable (standard ops only: Conv, BN, ReLU, Linear, Pool)
  - Forward pass fits in ~30KB SRAM tensor arena

Two architectures:
  - TinyConvEncoder: Conv2D on full mel-spectrogram (treats it as an image)
  - TinyTCNEncoder: Conv1D along time axis (treats mel bins as channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConvEncoder(nn.Module):
    """Small Conv2D encoder that treats mel-spectrograms as single-channel images.

    Architecture:
        Conv2d(1→8, 3x3, stride=2) → BN → ReLU    (64x64 → 32x32)
        Conv2d(8→16, 3x3, stride=2) → BN → ReLU   (32x32 → 16x16)
        Conv2d(16→32, 3x3, stride=2) → BN → ReLU  (16x16 → 8x8)
        AdaptiveAvgPool2d(1)                        (8x8 → 1x1)
        Linear(32 → embedding_dim)
        L2 normalise

    ~6.5K parameters. Receptive field covers full spectrogram.
    """

    def __init__(self, config: dict):
        super().__init__()
        conv_cfg = config["model"]["conv"]
        channels = conv_cfg["channels"]  # [1, 8, 16, 32]
        ks = conv_cfg["kernel_size"]
        stride = conv_cfg["stride"]
        emb_dim = config["model"]["embedding_dim"]

        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], ks, stride=stride, padding=ks // 2),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(channels[-1], emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel-spectrogram to L2-normalised embedding.

        Args:
            x: (batch, 1, n_mels, window_frames)
        Returns:
            (batch, embedding_dim) L2-normalised embeddings
        """
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.embed(h)
        return F.normalize(h, p=2, dim=1)


class CausalConv1d(nn.Module):
    """Left-padded causal convolution for temporal processing."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Single TCN block: causal dilated conv → BN → ReLU + residual."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x))) + x


class TinyTCNEncoder(nn.Module):
    """Small temporal convolutional encoder.

    Treats mel-spectrogram as (batch, mel_bins, time_frames) and processes
    along the time axis with dilated causal convolutions.

    Architecture:
        Conv1d(64→16, k=1)  pointwise reduction
        TCNBlock(16, k=3, d=1)   receptive field: 3
        TCNBlock(16, k=3, d=2)   receptive field: 7
        TCNBlock(16, k=3, d=4)   receptive field: 15
        AdaptiveAvgPool1d(1)
        Linear(16 → embedding_dim)
        L2 normalise

    ~3.8K parameters. Receptive field: 15 frames ≈ 60ms.
    """

    def __init__(self, config: dict):
        super().__init__()
        tcn_cfg = config["model"]["tcn"]
        n_mels = config["audio"]["n_mels"]
        hidden = tcn_cfg["hidden_channels"]
        n_layers = tcn_cfg["num_layers"]
        ks = tcn_cfg["kernel_size"]
        emb_dim = config["model"]["embedding_dim"]

        # Pointwise reduction from mel bins to hidden channels
        self.reduce = nn.Conv1d(n_mels, hidden, kernel_size=1)

        # Dilated TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, ks, dilation=2**i)
            for i in range(n_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Linear(hidden, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel-spectrogram to L2-normalised embedding.

        Args:
            x: (batch, 1, n_mels, window_frames)
        Returns:
            (batch, embedding_dim) L2-normalised embeddings
        """
        # Remove channel dim: (B, 1, F, T) → (B, F, T)
        h = x.squeeze(1)
        h = self.reduce(h)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).flatten(1)
        h = self.embed(h)
        return F.normalize(h, p=2, dim=1)


def build_encoder(config: dict) -> nn.Module:
    """Factory function to create encoder from config."""
    encoder_type = config["model"]["encoder"]
    if encoder_type == "conv":
        return TinyConvEncoder(config)
    elif encoder_type == "tcn":
        return TinyTCNEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
