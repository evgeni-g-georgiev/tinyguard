"""
f_s: on-device trainable separator.

Architecture: Linear(16→8, no bias) + ReLU
    - No bias is critical (Deep SVDD requirement): with bias, the network
      can trivially collapse to mapping everything to the centroid c by
      setting W=0, b=c. Without bias, it must use the input structure.
    - ReLU is trivially implementable on a microcontroller (max(0, x)).

Training: Deep SVDD (Ruff et al., ICML 2018)
    - Loss = (1/N) Σ ||f_s(x_i) - c||² + (λ/2)||W||²
    - c is fixed after initialisation (mean of first forward pass)
    - Weight decay prevents the model from collapsing

On-device gradient (for Arduino implementation):
    z = W @ x           (8,) = (8,16) @ (16,)
    a = ReLU(z)          (8,)
    loss = ||a - c||²
    dL/da = 2(a - c)     (8,)
    dL/dz = dL/da * (z > 0)   (element-wise, ReLU gradient)
    dL/dW = outer(dL/dz, x)   (8,16) — one outer product
    W -= lr * (dL/dW + λ * W)

Total: 128 params (W) + 8 stored (c) = 136 values.
"""

import torch
import torch.nn as nn


class FsSeparator(nn.Module):
    """On-device trainable separator: projects 16D → 8D for anomaly scoring."""

    def __init__(self, input_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.relu = nn.ReLU()

        # Centroid in projected space — set after first forward pass
        self.register_buffer("centroid", torch.zeros(output_dim))
        self.centroid_initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings to the separator space.

        Args:
            x: (batch, input_dim) f_c embeddings (PCA-projected)
        Returns:
            (batch, output_dim) projected embeddings
        """
        return self.relu(self.projection(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores as squared distance to centroid.

        Args:
            x: (batch, input_dim) f_c embeddings
        Returns:
            (batch,) anomaly scores
        """
        projected = self.forward(x)
        return ((projected - self.centroid) ** 2).sum(dim=1)

    def init_centroid(self, x: torch.Tensor):
        """Set centroid to mean of projected normal embeddings.

        Must be called once before training, using a batch of normal data.
        After this, centroid is fixed (not updated during training).
        """
        with torch.no_grad():
            projected = self.forward(x)
            self.centroid.copy_(projected.mean(dim=0))
            self.centroid_initialized = True

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
