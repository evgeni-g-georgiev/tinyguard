"""
Contrastive learning framework for f_c encoder training.

Uses NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss from SimCLR.
For each batch of N spectrograms, two augmented views are created (2N total).
Positive pairs are the two views of the same spectrogram; all other 2(N-1)
pairs in the batch serve as negatives.

The projection head maps encoder output to a higher-dimensional space where
the contrastive loss is applied. At evaluation/deployment, only the encoder
is used — the projection head is discarded. This follows the SimCLR finding
that the projection head acts as a buffer protecting the encoder's
representation from losing information not useful for the contrastive task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.augmentations import SpectrogramAugmentation
from src.fc.encoders import build_encoder, count_parameters


class ProjectionHead(nn.Module):
    """MLP projection head: Linear → ReLU → Linear.

    Maps encoder output to the space where contrastive loss is computed.
    Discarded after training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class ContrastiveModel(nn.Module):
    """Wraps encoder + augmentation + projection head for contrastive training.

    Forward pass:
        1. Apply two random augmentations to input batch → view_a, view_b
        2. Encode both views with the encoder → emb_a, emb_b
        3. Project both with the projection head → proj_a, proj_b
        4. Compute NT-Xent loss on projections

    At evaluation, call encoder directly (or use self.encode()).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = build_encoder(config)
        self.augment = SpectrogramAugmentation(config)

        emb_dim = config["model"]["embedding_dim"]
        proj_cfg = config["model"]["projection"]
        self.projector = ProjectionHead(
            input_dim=emb_dim,
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
        )

        self.temperature = config["contrastive"]["temperature"]

        enc_params = count_parameters(self.encoder)
        proj_params = count_parameters(self.projector)
        print(f"Encoder parameters: {enc_params:,}")
        print(f"Projection head parameters: {proj_params:,} (discarded after training)")
        print(f"Total trainable: {enc_params + proj_params:,}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without augmentation or projection (for evaluation).

        Args:
            x: (batch, 1, n_mels, window_frames)
        Returns:
            (batch, embedding_dim) L2-normalised embeddings
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict:
        """Full contrastive forward pass.

        Args:
            x: (batch, 1, n_mels, window_frames)
        Returns:
            dict with keys: loss, emb_a, emb_b
        """
        # Create two augmented views
        view_a = self.augment(x.clone())
        view_b = self.augment(x.clone())

        # Encode both views
        emb_a = self.encoder(view_a)
        emb_b = self.encoder(view_b)

        # Project to contrastive space
        proj_a = self.projector(emb_a)
        proj_b = self.projector(emb_b)

        # Compute NT-Xent loss
        loss = self._nt_xent_loss(proj_a, proj_b)

        return {"loss": loss, "emb_a": emb_a.detach(), "emb_b": emb_b.detach()}

    def _nt_xent_loss(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        """Normalized Temperature-scaled Cross-Entropy Loss.

        For a batch of N samples with two views each (2N total), each sample's
        positive pair is its other view. The remaining 2(N-1) samples are negatives.

        Args:
            za: (N, D) projections from view a
            zb: (N, D) projections from view b
        Returns:
            scalar loss
        """
        N = za.shape[0]
        z = torch.cat([za, zb], dim=0)  # (2N, D)

        # Cosine similarity matrix (2N x 2N)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=sim.device),
            torch.arange(0, N, device=sim.device),
        ])

        # NT-Xent: -log(exp(sim_pos/τ) / Σ exp(sim_neg/τ))
        pos_sim = sim[torch.arange(2 * N, device=sim.device), pos_idx]
        loss = -pos_sim + torch.logsumexp(sim, dim=1)

        return loss.mean()
