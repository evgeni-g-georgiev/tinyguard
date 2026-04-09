"""Base class for frozen embedding models.
                                                                                    
A frozen embedder converts preprocessed audio features into a fixed-dimensional
latent vector. Weights are loaded from a checkpoint and never updated during          
simulation.                                                                           
"""
from abc import ABC, abstractmethod 
import numpy as np

class BaseFrozenEmbedder(ABC):

    @abstractmethod
    def embed(self, preprocessed_audio: np.ndarray) -> np.ndarray:
        """Compress preprocessed audio into latent embeddings.

        Args:
            preprocessed_audio: Output of the preprocessing stage.
                Shape depends on the pipeline configuration:
                - AcousticEncoder: (n_frames, 1, n_mels, width)
                - Identity: any shape (passed through unchanged)

        Returns:
            Embedding array, typically (n_frames, latent_dim).
        """