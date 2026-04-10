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
                
    def __init__(
        self,
        checkpoint: str,
        latent_dim: int,
        sample_rate: int,                                                             
        frame_seconds: float,
        n_mels: int,                                                                  
        n_fft: int,
        hop_length: int,
        center: bool = True,
        device: str = "cpu",                                                          
    ):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)        
                
        # Cross-check: YAML latent_dim must match the trained model's dim             
        ckpt_dim = ckpt["embedding_dim"]
        if ckpt_dim != latent_dim:                                                    
            raise ValueError(                                                         
                f"Config latent_dim={latent_dim} but checkpoint was trained "
                f"with embedding_dim={ckpt_dim}. Either update default.yaml "         
                f"or use a checkpoint trained with latent_dim={latent_dim}."          
            )                                                                         
                                                                                    
        self.embedding_dim = latent_dim
        self.device = device
                                                                                    
        # Audio params kept for preprocessing contract validation
        self.audio_config = {                                                         
            "sample_rate": sample_rate,                                               
            "frame_seconds": frame_seconds,
            "n_mels": n_mels,                                                         
            "n_fft": n_fft,
            "hop_length": hop_length,
            "center": center,                                                         
        }
        self.preprocessing_contract = ckpt.get("preprocessing", {})                   
                                                                                    
        self.encoder = AcousticEncoder(embedding_dim=latent_dim)
        self.encoder.load_state_dict(ckpt["model_state_dict"])                        
        self.encoder.eval()                                                           
        for p in self.encoder.parameters():
            p.requires_grad = False                                                   
                                                                                    
    def validate_preprocessing(self, audio_config: dict) -> None:
        """Assert that simulation audio config matches the checkpoint contract.       
                                                                                    
        Skipped silently if the checkpoint has no preprocessing block (legacy).       
        """                                                                           
        if not self.preprocessing_contract:                                           
            return                                                                    

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
            tensor = torch.tensor(spectrograms, dtype=torch.float32,                  
device=self.device)
            return self.encoder(tensor).cpu().numpy() 