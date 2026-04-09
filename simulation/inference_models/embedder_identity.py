"""Identity embedder — pass-through for pipelines with no learned encoder.

Used by the GMM pipeline where the TWFR preprocessor output is already the
final representation used by the separator.
"""

import numpy as np

from simulation.registry import register_embedder
from simulation.inference_models.base_frozen_embedder import BaseFrozenEmbedder


@register_embedder("identity")
class IdentityEmbedder(BaseFrozenEmbedder):

    def embed(self, preprocessed_audio: np.ndarray) -> np.ndarray:
        return preprocessed_audio
    
    