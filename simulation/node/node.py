"""Node — a single simulated on-device anomaly detector.
                                                                                    
Each Node represents one physical machine (e.g. fan_id_00). It holds                  
references to the shared frozen pipeline (preprocessor + embedder) and                
owns its own separator instance which contains all trainable state.                   
                                                                                    
The Node is deliberately thin — it orchestrates the flow from raw audio               
to anomaly score but contains no model logic itself.                                  
"""                                                                                   

import numpy as np 

from simulation.inference_models.base_frozen_embedder import BaseFrozenEmbedder       
from simulation.inference_models.base_on_device_separator import BaseOnDeviceSeparator
from simulation.node.base_topology import BaseTopology                                
from simulation.node.base_merge import BaseMerge    


class Node:

    def __init__(
        self,
        node_id: str,                                                                 
        machine_type: str,
        machine_id: str,                                                              
        preprocessor,                                                               
        frozen_embedder: BaseFrozenEmbedder,
        separator: BaseOnDeviceSeparator,                                             
        topology: BaseTopology,
        merge: BaseMerge,     
        manual_reset: bool = False,                                                        
    ):                                                                              
        self.node_id = node_id
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.preprocessor = preprocessor                                              
        self.frozen_embedder = frozen_embedder
        self.separator = separator                                                    
        self.topology = topology                                                    
        self.merge = merge
        self.manual_reset = manual_reset

        self.scores: list[float] = []                                                 
        self.labels: list[int] = []
        self.predictions: list[int | None] = []
        self.state_predictions: list[int |None] = []

        # Internal state for manual-reset mode (circuit-breaker for anomoly dection)
        # _alarm_held is cleared by reset_state() at the engineer-reset moment. 
        # _prev_label is read/written by the lockstep loop to detect block 
        # transitions (anomaly block -> normal region). 
        self._alarm_held: bool = False 
        self._prev_label: int | None = None 
    
    def _wav_to_embedding(self, wav_path: str) -> np.ndarray:                       
        """Run the frozen pipeline: wav → preprocessor → embedder → embedding."""
        preprocessed_audio = self.preprocessor.process(wav_path)                      
        embedding = self.frozen_embedder.embed(preprocessed_audio)
        return embedding   
    

    def warmup(self, clip_paths: list[str]) -> None:
        """Calibrate the separator on warmup clips.

        Runs each clip through the frozen pipeline, then passes all
        resulting embeddings to the separator for calibration (e.g.
        SVDD centroid computation, GMM fitting).

        Args:
            clip_paths: Paths to warmup .wav files.
        """
        warmup_embeddings = [
            self._wav_to_embedding(path) for path in clip_paths
        ]
        self.separator.calibrate(warmup_embeddings)

    def process_clip(self, wav_path: str, label: int) -> tuple[float, int | None]:
        """Process a single test clip: embed, score, predict, record.                               
                                                                                                    
        Returns:                                                                                    
            (score, predicted_label) where predicted_label is 1 if the                              
            score exceeds the separator's threshold, 0 if below, or None                            
            if the separator does not expose a threshold.                                           
        """                                                                                         
        embedding = self._wav_to_embedding(wav_path)                                                
        score = self.separator.score(embedding)     
                                                                                                    
        threshold = getattr(self.separator, "threshold", None)
        predicted_label = int(score > threshold) if threshold is not None else None    

        # ── State prediction (held open i manual-reset mode) ────────────────────────────────────────────────────────             
        state_pred = self._compute_state(score, threshold) 
                                                                                    
        self.scores.append(score)                                                                   
        self.labels.append(label)                                                                   
        self.predictions.append(predicted_label)
        self.state_predictions.append(state_pred)
                                                                                                    
        return score, predicted_label  
    

    def _compute_state(self,
                       score: float,
                       threshold: float | None,
    ) -> int | None:
        """Compute the state prediction for this clip.

        Delegates the raw "is this clip an anomaly state?" decision to
        separator.state(score). In manual_reset mode, once the separator
        has ever fired in the current window the Node holds the alarm
        high until reset_state() is called — i.e. the node behaves like                          
        a circuit breaker that needs a manual reset to clear.
                                                                                                
        In the default (non-manual-reset) mode, the state simply mirrors
        whatever separator.state(score) returns on each clip.                                    
        """                                                                                      
        if threshold is None:                                                                    
            return None                                                                          
        raw_state = self.separator.state(score)
        if not self.manual_reset:
            return raw_state                                                                     
        if raw_state == 1:
            self._alarm_held = True                                                              
        return 1 if self._alarm_held else 0
    
    def reset_state(self) -> None:                                               
        """Clear the held alarm (manual-reset engineer action).                                  
                                                                                                
        Called by the lockstep loop on the first normal clip after an
        anomaly block when manual_reset mode is on. Clears the Node's                            
        own alarm-held flag and delegates to separator.reset_state() so                          
        any stateful separator override (e.g. future EWMA) can clear                             
        its accumulators at the same moment.                                                     
        """                                                                                      
        self._alarm_held = False                                                                 
        self.separator.reset_state()                                                             
                  


    def get_neighbours(self) -> list[str]:
        """Return IDs of nodes this node can communicate with."""
        return self.topology.neighbours(self.node_id)