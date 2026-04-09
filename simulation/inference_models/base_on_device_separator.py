
"""Base class for on-device separators.

A separator is the trainable component that runs on each device. It owns all
of its internal state (model weights, centroids, GMM parameters, etc.) and
exposes a uniform interface for calibration, scoring, and federated sharing.

Input is always list[np.ndarray] for calibrate (one array per clip) and
np.ndarray for score (single clip). Each implementation asserts the expected
shape on entry.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseOnDeviceSeparator(ABC):

    @abstractmethod
    def calibrate(self, warmup_embeddings: list[np.ndarray]) -> None:
        """Fit on warmup data from the preprocessor + embedder chain.

        Args:
            warmup_embeddings: One array per warmup clip. Shape per clip
                depends on the pipeline:
                - SVDD: (n_frames, latent_dim)
                - GMM:  (n_mels,) after TWFR
        """
    
    @abstractmethod 
    def score(self, clip_embedding: np.ndarray) -> float: 
        """Score a single clip. Higher balues indicate greater anomaly. 
        
        Args: 
            clip_embedding: Single clip after preporcessing + frozen_embedding. 
            
        Returns: 
            Anomaly score as float. 
        """
    
    @abstractmethod
    def get_shareable_state(self) -> dict:
        """Return state to share with neighbours during federation."""


    @abstractmethod
    def merge_state(self, neighbour_states: list[dict]) -> None:
        """Incorporate received state from neighbours."""
