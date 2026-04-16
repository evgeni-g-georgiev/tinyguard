
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

    def _compute_threshold(self, scores: list[float]) -> float:
        """Compute detection threshold from holdout scores using the configured mode."""
        match self.threshold_mode:
            case "percentile": return float(np.percentile(scores, self.threshold_percentile))
            case "max_margin": return float(np.max(scores)) + self.threshold_margin
            case "n_sigma":    return float(np.mean(scores) + self.n_sigma * np.std(scores))
            case _:
                raise ValueError(
                    f"Unknown threshold_mode '{self.threshold_mode}'. "
                    f"Expected: percentile, max_margin, n_sigma"
                )
        
    # ── State Layer  ────────────────────────────────────────────────────────
    def state(self, score: float, ** kwargs) -> int: 
        """Returns the current state (0 = normal, 1 = anomoaly) for this clip. 
        
        Default: stateless threshold compare.
        Return 0 if no threshold has been calibrated yet.
        """
        threshold = getattr(self, "threshold", None)
        if threshold is None: 
            return 0 
        return int(score > threshold)
    

    def reset_state(self) -> None: 
        """Reser the temporal state (latching mode engineer reset). 
        
        Default: no-op, since the base state() 0 is stateless.
        Stateful overrides should clear their accumataroes here
        """
        pass 


    @abstractmethod
    def get_shareable_state(self) -> dict:
        """Return state to share with neighbours during federation."""


    @abstractmethod
    def merge_state(self, neighbour_states: list[dict]) -> None:
        """Incorporate received state from neighbours."""

    @abstractmethod
    def description(self) -> str:                                                   
        """Return a short human-readable label for plot titles.
                                                                
        Examples:                                                                   
            "SVDD (1312 params)"
            "GMM (2 components, diag cov)"                                          
            "Identity"                    
        """ 

    @abstractmethod
    def project(self, clip_embedding: np.ndarray) -> np.ndarray:                    
        """Return the post-separator latent vector for visualisation.
                                                                                    
        This is the intermediate representation the separator scores in,
        NOT the final scalar score. Used by the reporting module to                 
        generate latent space plots.                               
                                                                                    
        Args:   
            clip_embedding: Output of the preprocessor + embedder chain.            
                                                                        
        Returns:                                                                    
            Post-separator latent vector. Shape depends on the separator:
                SVDD:     (n_frames, output_dim) — post-network projection          
                GMM:      (n_mels,) — TWFR feature vector                           
                Identity: input unchanged                                           
        """ 