"""Wraps the team's GMMDetector as an on-device GMM separator.
                                                                                    
The GMMDetector (gmm/detector.py) implements TWFR-GMM from Guan et al. 2023.        
It already exposes fit/score and computes its own threshold from held-out           
normal data. This wrapper adapts its interface to BaseOnDeviceSeparator             
so it plugs into the simulation's lockstep loop.                                    
                                                                                    
Pipeline data flow when using this separator:                                       
    .wav file                                                                       
    -> TWFRPreprocessor: full-clip log-mel (n_mels from config.GMM_N_MELS)        
        -> IdentityEmbedder: pass-through                                           
        -> GMMSeparator.calibrate / .score: list[(n_mels, T)] log-mels            
                                                                                    
The shape of the log-mel arrays is determined by the team's preprocessor            
(gmm/features.py reads GMM_N_MELS from config.py). This wrapper makes no            
assumptions about the dimension — it just passes the arrays through to              
the team's GMMDetector unchanged.                                                   
"""                                                                                 
                                                                                    
import numpy as np                                                                  

from simulation.registry import register_separator                                  
from simulation.inference_models.base_on_device_separator import BaseOnDeviceSeparator                                                               
from gmm.detector import GMMDetector
                                                                                    
                
@register_separator("gmm")
class GMMSeparator(BaseOnDeviceSeparator):

    def __init__(                                                                   
        self,
        n_components: int = 2,                                                      
        covariance_type: str = "diag",
        threshold_pct: int = 95,                                                    
        seed: int = 42,
        holdout_fraction: float = 0.2,                                              
    ):                                                                              
        self.n_components = n_components
        self.covariance_type = covariance_type                                      
        self.threshold_pct = threshold_pct
        self.seed = seed                                                            
        self.holdout_fraction = holdout_fraction
                                                                                    
        self.detector: GMMDetector | None = None
        self.threshold: float | None = None
                                                                                    
    def calibrate(self, warmup_log_mels: list[np.ndarray]) -> None:
        """Fit GMM on warmup log-mel spectrograms.                                  
                                                                                    
        Splits warmup into train + holdout (same pattern as SVDDSeparator):         
            - train clips fit the GMM and run the r-search                          
            - holdout clips set the detection threshold from unseen normals         
                                                                                    
        Args:                                                                       
            warmup_log_mels: List of (n_mels, n_frames) arrays, one per clip.       
                Shape is determined by the active preprocessor.                     
        """                                                                         
        n_total = len(warmup_log_mels)                                              
        n_holdout = max(1, int(n_total * self.holdout_fraction))                    
                                                                                    
        train_clips = warmup_log_mels[:-n_holdout]
        holdout_clips = warmup_log_mels[-n_holdout:]                                
                
        # Construct and fit the team's detector. fit() runs the r-search,           
        # fits the GMM, and computes its own threshold from val_log_mels.
        self.detector = GMMDetector(                                                
            n_components=self.n_components,
            covariance_type=self.covariance_type,                                   
            threshold_pct=self.threshold_pct,
            seed=self.seed,                                                         
        )
        self.detector.fit(train_clips, val_log_mels=holdout_clips)                  
                
        # Surface the detector's threshold under our own attribute name             
        # so Node.process_clip can compute predicted_label uniformly.
        self.threshold = self.detector.threshold_                                   
                
    def score(self, clip_log_mel: np.ndarray) -> float:                             
        """Score a single clip's log-mel spectrogram via NLL under the GMM.
                                                                                    
        Args:   
            clip_log_mel: (n_mels, T) log-mel spectrogram.                          
                                                                                    
        Returns:
            Negative log-likelihood. Higher = more anomalous.                       
        """     
        return self.detector.score(clip_log_mel)                                    

    def get_shareable_state(self) -> dict:                                          
        """Federation state — GMM mixture params + threshold."""
        if self.detector is None:                                                   
            return {}
        return {                                                                    
            "gmm": self.detector.gmm_,
            "r": self.detector.r_,                                                  
            "threshold": self.detector.threshold_,
        }                                                                           
                
    def merge_state(self, neighbour_states: list[dict]) -> None:                    
        pass  # No-op until federation is implemented for GMM