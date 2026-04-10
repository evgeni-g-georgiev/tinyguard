"""Wraps the Evgeni's FsSeparator + train_fs as an on-device SVDD separator.

All SVDD state (model weights, centroid) lives inside this class.
The team's train_fs() and FsSeparator are imported and called directly.
"""

import numpy as np 
import torch 

from simulation.registry import register_separator
from simulation.inference_models.base_on_device_separator import BaseOnDeviceSeparator
from separator.separator import FsSeparator, train_fs


@register_separator("svdd")
class SVDDSeparator(BaseOnDeviceSeparator):

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 32,
        output_dim: int = 8,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        threshold_percentile: float = 95.0,
        holdout_fraction: float = 0.2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold_percentile = threshold_percentile
        self.holdout_fraction = holdout_fraction

        self.train_kwargs = dict(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        self.model: FsSeparator | None = None
        self.centroid: np.ndarray | None = None
        self.threshold: float | None = None 

    def calibrate(self, warmup_embeddings: list[np.ndarray]) -> None:
        """Train SVDD on warmup embeddings.

        Args:
            warmup_embeddings: List of (n_frames, input_dim) arrays, one per clip.
        """
        n_total = len(warmup_embeddings)
        n_holdout = max(1, int(n_total * self.holdout_fraction))

        train_clips = warmup_embeddings[:-n_holdout]
        holdout_clips = warmup_embeddings[-n_holdout:]

        # Train SVDD on the training portion only 
        all_frames = np.concatenate(train_clips, axis=0)
        assert all_frames.ndim == 2 and all_frames.shape[1] == self.input_dim, (
            f"Expected(N, {self.input_dim}), got {all_frames.shape}"
        )
        self.model, self.centroid = train_fs(
            all_frames, **self.train_kwargs
        )

        # Use the held-out unsesen normals to capture the threshold [Add later some std]
        holdout_scores = list(map(self.score, holdout_clips))
        self.threshold = float(
            np.percentile(holdout_scores, self.threshold_percentile)
        )


    def score(self, clip_embedding: np.ndarray) -> float: 
        """Max-frame squared L2 distance from centroid.

        Args:
            clip_embedding: (n_frames, input_dim) array for a single clip.

        Returns:
            Anomaly score — higher means more anomalous.
        """
        centroid_tensor = torch.tensor(self.centroid, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            projected = self.model(torch.tensor(clip_embedding, dtype=torch.float32))
            frame_distances = ((projected - centroid_tensor) ** 2).sum(dim=1)
            return float(frame_distances.max())
    
    def get_shareable_state(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "centroid": self.centroid.copy() if self.centroid is not None else None,
            "threshold": self.threshold,
        }

    def merge_state(self, neighbour_states: list[dict]) -> None:
        pass  # No-op until Merge learing is implemented