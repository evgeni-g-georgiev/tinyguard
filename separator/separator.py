"""
separator.py — FsSeparator: on-device anomaly detector via Deep SVDD.

Architecture
------------
Linear(input_dim→32, bias=True) + ReLU + Linear(32→8, bias=False) + ReLU

The output layer has no bias — required by Deep SVDD to prevent the trivial
collapse where the network ignores input and maps everything to the centroid
by setting W2=0, b=c.

Parameters: 16×32 + 32 + 32×8 = 800  (3.2 KB at float32)

Training
--------
Deep SVDD (Ruff et al., ICML 2018):
  - Centroid c = mean output of a forward pass through all training data,
    computed once and then frozen.
  - Loss per batch = mean(||f_s(x) - c||²)
  - Weight decay via optimiser prevents collapse to a degenerate solution.
  - Early stopping on a held-out validation split.

Scoring
-------
Per clip: max over all frame-level squared L2 distances from centroid.
Max-frame scoring catches partial anomalies that mean-pooling would dilute.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FsSeparator(nn.Module):

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(x))))


def train_fs(
    embeddings: np.ndarray,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[FsSeparator, np.ndarray]:
    """
    Train Deep SVDD on normal-clip embeddings.

    Trains for a fixed number of epochs on all available data — no val split or
    early stopping. This reflects the on-device reality: embeddings are buffered
    during the 10-minute collection window (~75KB), then training runs for a
    fixed epoch count after collection (~24–48 ms/epoch on Cortex-M33 @ 160 MHz).

    Args:
        embeddings:   (N, input_dim) float32 — all frames from all normal clips stacked.
        lr:           SGD learning rate.
        weight_decay: L2 regularisation — prevents hypersphere collapse.
        epochs:       fixed number of training epochs.
        batch_size:   mini-batch size.
        seed:         RNG seed for reproducibility.

    Returns:
        model:    trained FsSeparator (on CPU, eval mode).
        centroid: (output_dim,) float32 numpy array — fixed hypersphere centre.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X            = torch.tensor(embeddings, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    input_dim = embeddings.shape[1]
    model     = FsSeparator(input_dim=input_dim).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine LR decay over fixed epochs — large updates early, fine-tuning at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # Initialise centroid: mean output of a single forward pass through all
    # training data. Fixed for the rest of training — the network learns to
    # pull normal embeddings toward this fixed point rather than chasing a
    # moving target, which is the standard Deep SVDD setup.
    model.eval()
    with torch.no_grad():
        projs    = torch.cat([model(batch[0].to(device)) for batch in train_loader])
        centroid = projs.mean(dim=0)   # (output_dim,)

    # Fixed-epoch training loop
    model.train()
    for _ in range(epochs):
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            # SVDD loss: mean squared L2 distance of each sample's projection
            # from the fixed centroid. Minimising this compacts normal data
            # around c; anomalies at test time land far from c.
            loss = ((model(batch) - centroid) ** 2).sum(dim=1).mean()
            loss.backward()
            optimiser.step()
        scheduler.step()

    model.eval()
    return model.cpu(), centroid.cpu().numpy()


def score_clips(
    clip_embeddings: list[np.ndarray],
    model: FsSeparator,
    centroid: np.ndarray,
) -> list[float]:
    """
    Score a list of clips via max-frame squared L2 distance from centroid.

    Args:
        clip_embeddings: list of (n_frames, input_dim) arrays, one per clip.
        model:           trained FsSeparator (eval mode, on CPU).
        centroid:        (output_dim,) numpy array.

    Returns:
        List of float scores, one per clip.
    """
    c = torch.tensor(centroid, dtype=torch.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for embs in clip_embeddings:
            proj        = model(torch.tensor(embs, dtype=torch.float32))  # (n_frames, 8)
            frame_dists = ((proj - c) ** 2).sum(dim=1)                    # (n_frames,)
            scores.append(float(frame_dists.max()))
    return scores
