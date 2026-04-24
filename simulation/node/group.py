                                                                                                                            
"""Group — holds the N nodes of one machine and computes their fused score.

A Group exists only when len(channels) > 1.  For a single-channel run we skip
construction entirely and every "per-group" artefact simply mirrors the single
node.

Paper mapping (Kanjo & Aslanov 2026 §3, eq. 3/5):
    θ_i  = Node.detector.gmm_.{mu_, sigma2_, pi_}   — per-node GMM state
    ϕ_j  = Node.mu_val, Node.sigma_val              — confidence signals
    M_i  = Group.score(per_node_nlls)               — score-space weighted z-sum

Fusion math (generalised to N peers):
    z_i(x)      = (NLL_i(x) - μ_val_i) / σ_val_i
    w           = softmax(-σ_val / temperature)      ∈ R^N,  sum(w) = 1
    fused_z(x)  = Σ_i w_i · z_i(x)

Lower σ_val_i means the node's val NLLs are more concentrated → higher
confidence → larger weight in the fused score. Mirrors the 2-node fusion in
deployment/node_learning.h, generalised here to N nodes.
"""                                                                                                                             
                                                                                                                                
from dataclasses import dataclass, field                                                                                        

import numpy as np                                                                                                              
                
from simulation.node.node import Node

                                                                                                                                
_SIGMA_FLOOR = 1e-8
                                                                                                                                
                
def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector."""
    e = np.exp(x - x.max())                                                                                                     
    return e / e.sum()
                                                                                                                                
                
@dataclass                                                                                                                      
class Group:    
    # Identity + config
    machine_type:  str
    machine_id:    str                                                                                                          
    nodes:         list[Node]
    temperature:   float = 100.0                                                                                                
    threshold_pct: float = 0.95
    cusum_h_sigma: float = 5.0                                                                                                  
    cusum_h_floor: float = 1.0                                                                                                  
                                                                                                                                
    # Populated by finalise_fusion() — called once after every node calibrates                                                  
    w:            np.ndarray | None = None   # fusion weights, sum to 1                                                         
    threshold_:   float | None      = None   # k in fused z-score space                                                         
    cusum_h_:     float | None      = None   # h in fused z-score space                                                         
    fused_val_z_: np.ndarray | None = None   # diagnostic                                                                       
                                                                                                                                
    # Stateful CUSUM accumulator for fused scores                                                                               
    _cusum_S: float = 0.0                                                                                                       

    # Cached node statistics — set by finalise_fusion(), used by score()
    _mu_vals:    np.ndarray | None = None
    _sigma_vals: np.ndarray | None = None
                                                                                                                             
    # Per-timestep log
    fused_scores: list[float] = field(default_factory=list)                                                                     
    labels:       list[int]   = field(default_factory=list)                                                                     
    cusum_S:      list[float] = field(default_factory=list)
    alarms:       list[bool]  = field(default_factory=list)                                                                     
                
    # ── Identity helpers ──────────────────────────────────────────────────                                                    
    @property   
    def group_id(self) -> str:                                                                                                  
        return f"{self.machine_type}_{self.machine_id}"
                                                                                                                                
    @property
    def k(self) -> float: return self.threshold_                                                                                
    @property   
    def h(self) -> float: return self.cusum_h_
                                                                                                                                
    # ── Calibration-time fusion setup ─────────────────────────────────────                                                    
    def finalise_fusion(self) -> None:                                                                                          
        """Compute fusion weights and fused CUSUM params from peer statistics.                                                  
                                                                                                                                
        Must be called AFTER every node in self.nodes has run calibrate().
        """                                                                                                                     
        mu_vals = np.array([n.mu_val for n in self.nodes], dtype=np.float64)
        sigma_vals = np.array([n.sigma_val for n in self.nodes], dtype=np.float64)
        self.w  = _softmax(-sigma_vals / self.temperature).astype(np.float64)
                                                                                                                                
        # Z-normalise each node's val NLLs, weight-sum over nodes.                                                              
        # Take min length in case of rare clip-count mismatch.                                                                  
        m = min(len(n.val_nlls) for n in self.nodes)                                                                            
        z_stack = ((np.stack([n.val_nlls[:m] for n in self.nodes]) - mu_vals[:, None])
               / np.maximum(sigma_vals[:, None], _SIGMA_FLOOR))                                                    # (N, m)
        self.fused_val_z_ = (self.w[:, None] * z_stack).sum(axis=0)   # (m,)                                                    
                                                                                                                                
        # Same floor-percentile + σ-multiplier rule as GMMDetector._calibrate.                                                  
        sorted_z = np.sort(self.fused_val_z_)                                                                                   
        pct_idx  = min(int(m * self.threshold_pct), m - 1)                                                                      
        self.threshold_ = float(sorted_z[pct_idx])                                                                              
        self.cusum_h_   = float(max(                                                                                            
            self.cusum_h_sigma * float(self.fused_val_z_.std()),                                                                
            self.cusum_h_floor,                                                                                                 
        ))
                                                                                                                                
    # ── Per-timestep fused scoring ────────────────────────────────────────                                                    
    def score(self, per_node_nlls: list[float]) -> float:
        """Fuse this timestep's per-node NLLs into one z-score."""
        nlls       = np.array(per_node_nlls, dtype=np.float64)
        mu_vals    = np.array([n.mu_val    for n in self.nodes], dtype=np.float64)
        sigma_vals = np.array([n.sigma_val for n in self.nodes], dtype=np.float64)
        z = (nlls - mu_vals) / np.maximum(sigma_vals, _SIGMA_FLOOR)
        return float(np.dot(self.w, z))                                                                                   
                                                                                                                                
    def cusum_update(self, fused_z: float, label: int) -> bool:                                                                 
        """Advance the fused CUSUM, append traces. Returns alarm bool."""
        self._cusum_S = max(0.0, self._cusum_S + fused_z - self.threshold_)                                                     
        fired = self._cusum_S >= self.cusum_h_                                                                                                                        
        self.fused_scores.append(fused_z)                                                                                       
        self.labels.append(label)
        self.cusum_S.append(self._cusum_S)                                                                                      
        self.alarms.append(fired)

        if fired:                                                                                                               
            self._cusum_S = 0.0

        return fired
                                                                                                                                
    def cusum_reset(self) -> None:
        self._cusum_S = 0.0