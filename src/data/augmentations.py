"""
Spectrogram augmentations for contrastive learning.

These define what the encoder learns to be INVARIANT to (noise, gain, masking)
and therefore what it learns to be SENSITIVE to (harmonic structure, spectral
envelope, temporal patterns — the features that matter for anomaly detection).
"""

import torch
import torch.nn as nn


class SpectrogramAugmentation(nn.Module):
    """Applies random augmentations to a log-mel spectrogram.

    Each call produces a different random augmentation, so calling twice
    on the same input produces two different views for contrastive learning.

    Input shape: (batch, 1, n_mels, time_frames)
    Output shape: same
    """

    def __init__(self, config: dict):
        super().__init__()
        aug = config["augmentations"]
        self.time_mask_max = aug["time_mask_max_frames"]
        self.freq_mask_max = aug["freq_mask_max_bins"]
        self.noise_snr_low, self.noise_snr_high = aug["noise_snr_db"]
        self.gain_low, self.gain_high = aug["gain_db"]
        self.roll_max = aug["time_roll_max_frames"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a batch of spectrograms."""
        x = self._time_roll(x)
        x = self._gain(x)
        x = self._additive_noise(x)
        x = self._time_mask(x)
        x = self._freq_mask(x)
        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random contiguous block of time frames."""
        if self.time_mask_max == 0:
            return x
        B, C, F, T = x.shape
        mask_len = torch.randint(0, self.time_mask_max + 1, (B,))
        mask_start = torch.randint(0, T, (B,))
        for i in range(B):
            end = min(mask_start[i] + mask_len[i], T)
            x[i, :, :, mask_start[i]:end] = 0.0
        return x

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random contiguous block of frequency bins."""
        if self.freq_mask_max == 0:
            return x
        B, C, F, T = x.shape
        mask_len = torch.randint(0, self.freq_mask_max + 1, (B,))
        mask_start = torch.randint(0, F, (B,))
        for i in range(B):
            end = min(mask_start[i] + mask_len[i], F)
            x[i, :, mask_start[i]:end, :] = 0.0
        return x

    def _additive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at a random SNR."""
        snr_db = torch.empty(x.shape[0], 1, 1, 1, device=x.device)
        snr_db.uniform_(self.noise_snr_low, self.noise_snr_high)
        signal_power = x.pow(2).mean(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-10)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise

    def _gain(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random gain in dB."""
        gain_db = torch.empty(x.shape[0], 1, 1, 1, device=x.device)
        gain_db.uniform_(self.gain_low, self.gain_high)
        return x * (10 ** (gain_db / 20))

    def _time_roll(self, x: torch.Tensor) -> torch.Tensor:
        """Circular shift along time axis."""
        if self.roll_max == 0:
            return x
        shifts = torch.randint(-self.roll_max, self.roll_max + 1, (1,)).item()
        return torch.roll(x, shifts=shifts, dims=-1)
