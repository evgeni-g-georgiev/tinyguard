"""Pure metric computation - dataclasses and functions, no IO.

This module is the foundation that formatters and reportin build on. 
Every function here is pure: give inputs, return outputs, no side effects.
"""

from dataclasses import dataclass 

import numpy as np
from sklearn.metrics import roc_auc_score

from simulation.node.node import Node                                                                 

# ── Dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NodeMetrics:
    auc:       float
    precision: float
    recall:    float
    f1:        float
    tp: int; tn: int; fp: int; fn: int

                                                              
@dataclass(frozen=True)                                                                
class BlockStateMetrics:                                                                         
    """Block-level metrics computed from a node's state_predictions.

    Fires-at-all-inside-block semantics:
    - block_tp: anomaly block where state was 1 at any point inside                            
    - block_fn: anomaly block where state stayed 0 throughout (missed)                         
    - block_fp: normal region where state was 1 at any point                                   
    - block_tn: normal region where state stayed 0 throughout                                  
                                                                                                
    mean_lag:    average clips-from-block-start to first fire, over detected blocks.             
    mean_unflag: average clips-from-block-end to first un-fire, over detected blocks.            
                None in manual_reset mode (the engineer resets, not the tracker).               
    """                                                                                          
    block_tp: int                                                                                
    block_fp: int                                                                                
    block_fn: int
    block_tn: int
    block_precision: float                                                                       
    block_recall:    float
    block_f1:        float                                                                       
    mean_lag:    float | None
    mean_unflag: float | None
    missed_blocks:  int
    total_blocks:   int 


# ── Pure helpers ────────────────────────────────────────────────────────
             
def contiguous_runs(values: list[int], target: int) -> list[tuple[int, int]]:                         
    """Return [(start, end_inclusive), ...] for each contiguous run of `target`.
                                                                                                    
    Example:    
        contiguous_runs([0, 1, 1, 0, 1, 0], target=1)                                                 
        → [(1, 2), (4, 4)]                                                                            
    """
    runs: list[tuple[int, int]] = []                                                                  
    start: int | None = None                                                                          
    for i, v in enumerate(values):
        if v == target and start is None:                                                             
            start = i
        elif v != target and start is not None:                                                       
            runs.append((start, i - 1))
            start = None                                                                              
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs                                                                                       

                                                                                                    
def find_anomaly_bands(labels: list[int]) -> list[tuple[int, int]]:
    """Convenience alias: contiguous runs of label==1."""
    return contiguous_runs(labels, target=1)                                                          
                                                                                                    
                                                                                                    
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:                                    
    """Returns (precision, recall, f1)."""                                                            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0    
    return precision, recall, f1
                                                                                                    
                                                                                                    
def _anomaly_block_outcomes(
    anomaly_blocks: list[tuple[int, int]],                                                            
    state: list[int],
    manual_reset: bool,
) -> tuple[int, int, list[int], list[int]]:
    """Returns (block_tp, block_fn, lags, unflags) over all anomaly blocks."""                        
    block_tp, block_fn, lags, unflags = 0, 0, [], []                                                  
    for start, end in anomaly_blocks:                                                                 
        first_fire = next((i for i in range(start, end + 1) if state[i] == 1), None)                  
        if first_fire is None:                                                                        
            block_fn += 1                                                                             
            continue                                                                                  
        block_tp += 1                                                                                 
        lags.append(first_fire - start)
        if not manual_reset and end + 1 < len(state):
            unflag = next((j - (end + 1) for j in range(end + 1, len(state)) if state[j] == 0), None) 
            if unflag is not None:                                                                    
                unflags.append(unflag)                                                                
    return block_tp, block_fn, lags, unflags                                                          
                
                                                                                                    
def _normal_region_outcomes(
    normal_regions: list[tuple[int, int]],
    state: list[int],
) -> tuple[int, int]:
    """Returns (block_fp, block_tn) over all normal regions."""                                       
    fired = [any(state[i] == 1 for i in range(start, end + 1)) for start, end in normal_regions]
    return sum(fired), sum(not f for f in fired)    

# ── Per-node metric computation ────────────────────────────────────────────────────────

def node_metrics(node: Node) -> NodeMetrics | None: 
    """Return clip-level metrics for a node, or None if single-class."""
    if len(set(node.labels)) < 2:                                                                     
        return None
                                                                                                    
    auc = roc_auc_score(node.labels, node.scores)
    tp = sum(l == 1 and p == 1 for l, p in zip(node.labels, node.predictions))                        
    tn = sum(l == 0 and p == 0 for l, p in zip(node.labels, node.predictions))                        
    fp = sum(l == 0 and p == 1 for l, p in zip(node.labels, node.predictions))
    fn = sum(l == 1 and p == 0 for l, p in zip(node.labels, node.predictions))                        
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0                                              
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0    
                                                                                                    
    return NodeMetrics(auc=auc, precision=precision, recall=recall, f1=f1,                            
                        tp=tp, tn=tn, fp=fp, fn=fn)  


def block_state_metrics(node: Node, manual_reset: bool) -> BlockStateMetrics | None:
    """Compute block-level metrics from node.state_predictions + node.labels."""
    state, labels = node.state_predictions, node.labels                                               
    if not state or all(s is None for s in state):                                                    
        return None                                                                                   
                                                                                                    
    anomaly_blocks = contiguous_runs(labels, target=1)                                                
    if not anomaly_blocks:
        return None                                                                                   
    normal_regions = contiguous_runs(labels, target=0)
                                                                                                    
    block_tp, block_fn, lags, unflags = _anomaly_block_outcomes(anomaly_blocks, state, manual_reset)
    block_fp, block_tn                = _normal_region_outcomes(normal_regions, state)                
    precision, recall, f1             = _prf(block_tp, block_fp, block_fn)                            

    return BlockStateMetrics(                                                                         
        block_tp=block_tp, block_fp=block_fp,
        block_fn=block_fn, block_tn=block_tn,                                                         
        block_precision=precision, block_recall=recall, block_f1=f1,
        mean_lag=float(np.mean(lags)) if lags else None,                                              
        mean_unflag=float(np.mean(unflags)) if unflags else None,                                     
        missed_blocks=block_fn,
        total_blocks=block_tp + block_fn,                                                             
    ) 