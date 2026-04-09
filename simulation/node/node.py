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
    ):                                                                              
        self.node_id = node_id
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.preprocessor = preprocessor                                              
        self.frozen_embedder = frozen_embedder
        self.separator = separator                                                    
        self.topology = topology                                                    
        self.merge = merge

        self.scores: list[float] = []                                                 
        self.labels: list[int] = []

    
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

    def process_clip(self, wav_path: str, label: int) -> float:
        """Process a single test clip: embed, score, and record.

        Args:
            wav_path: Path to the .wav file.
            label: Ground truth label (0=normal, 1=abnormal).

        Returns:
            Anomaly score (higher = more anomalous).
        """
        embedding = self._wav_to_embedding(wav_path)
        score = self.separator.score(embedding)
        self.scores.append(score)
        self.labels.append(label)
        return score

    def get_neighbours(self) -> list[str]:
        """Return IDs of nodes this node can communicate with."""
        return self.topology.neighbours(self.node_id)