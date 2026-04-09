""" Wraps Evgeni's AcousticEncoder as a frozen embedder for the simulation. 

Load a checkpoint from distillation/train.py, freezes all weights,
and validates that the preprocessing contract matches the simulation config

"""

import numpy as np 
import torch 

from simulation.registry import register_embedder
from simulation.inference_models.base_frozen_embedder import BaseFrozenEmbedder
from distillation.cnn import AcousticEncoder 


@register_embedder("acoustic_encoder")
class AcousticEmbedder(BaseFrozenEmbedder):

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.embedding_dim = checkpoint["embedding_dim"]
        self.preporcessing_contract = checkpoint.get("preporcessing", {})
        self.device = device 

        self.encoder = AcousticEncoder(embedding_dim=self.embedding_dim) 
        self.encoder.load_state_dict()
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False 

        
    def validate_preprocessing(self, audio_config: dict) -> None: 
        """Assert that simulation audio config matches the checkpoint contract.

        Args:
            audio_config: The 'audio' section from default.yaml.

        Raises:
            ValueError: If any preprocessing parameter mismatches.
        """
        if not self.preprocessing_contract:
            return  # legacy checkpoint without contract — skip validation - will remove 

        for key in ("n_mels", "n_fft", "hop_length", "sample_rate"):
            expected = self.preprocessing_contract.get(key)
            actual = audio_config.get(key)
            if expected is not None and actual is not None and expected != actual:
                raise ValueError(
                    f"Checkpoint trained with {key}={expected}, "
                    f"but simulation config says {key}={actual}. "
                    f"Either retrain the embedder or update the config."
                )
            
    def embed(self, spectrograms: np.ndarray) -> np.ndarray:
        """Run frozen AcousticEncoder on log-mel spectrograms.

        Args:
            spectrograms: (n_frames, 1, n_mels, width) float32 numpy array.

        Returns:
            (n_frames, embedding_dim) float32 numpy array.
        """
        with torch.no_grad():
            tensor = torch.tensor(
                spectrograms,
                dtype=torch.float32,
                device=self.device
            )
            return self.encoder(tensor).cpu().numpy()