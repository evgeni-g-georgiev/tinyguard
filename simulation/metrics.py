"""Pure metric computation for nodes and groups.                                      
                                                                                    
Clip-level: treat per-clip alarms as the prediction signal → TP/TN/FP/FN/P/R/F1
            plus AUC from the continuous score.                                       
Block-level: treat per-clip alarms as a 0/1 state trace → block confusion plus        
            mean detection lag and mean "unflag" lag.                                
                                                                                    
CUSUM is always in auto-reset mode (alarm → accumulator reset), so                    
mean_unflag is always populated when any block was detected.                             
                                                                                    
All functions are pure: take (labels, scores, alarms), return a dataclass.            
"""                                                                                   

from dataclasses import dataclass                                                     
                
import numpy as np
from sklearn.metrics import roc_auc_score
                                                                                    

# ── Dataclasses ─────────────────────────────────────────────────────────            
                
@dataclass(frozen=True)
class ClipMetrics:
    auc:       float | None                                                           
    precision: float
    recall:    float                                                                  
    f1:        float
    tp: int; tn: int; fp: int; fn: int                                                

                                                                                    
@dataclass(frozen=True)
class BlockMetrics:
    """Block-level metrics.  Fires-at-all-inside-block semantics.                     
                                                                                    
    block_tp: anomaly block where alarm fired at any point inside                   
    block_fn: anomaly block with no alarms (missed)                                 
    block_fp: normal region where alarm fired at any point                          
    block_tn: normal region with no alarms (clean)                                  
    """                                                                               
    block_tp: int                                                                     
    block_fp: int
    block_fn: int
    block_tn: int
    block_precision: float                                                            
    block_recall:    float
    block_f1:        float                                                            
    mean_lag:    float | None   # clips from anomaly-band start → first alarm
    mean_unflag: float | None   # clips from anomaly-band end   → first non-alarm     
    missed_blocks: int                                                                
    total_blocks:  int                                                                
                                                                                    
                                                                                    
# ── Pure helpers ─────────────────────────────────────────────────────────
                                                                                    
def contiguous_runs(values, target) -> list[tuple[int, int]]:                         
    """[(start, end_inclusive), ...] for each contiguous run equal to `target`."""
    runs, start = [], None                                                            
    for i, v in enumerate(values):                                                    
        if v == target and start is None:                                             
            start = i                                                                 
        elif v != target and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:                                                             
        runs.append((start, len(values) - 1))
    return runs                                                                       
                
                                                                                    
def find_anomaly_bands(labels) -> list[tuple[int, int]]:
    return contiguous_runs(labels, target=1)                                          
                                                                                    
                                                                                    
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:                    
    p = tp / (tp + fp) if (tp + fp) else 0.0                                          
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0                                       
    return p, r, f
                                                                                    
                                                                                    
# ── Per-signal metric computation ────────────────────────────────────────
                                                                                    
def clip_metrics(labels, scores, alarms) -> ClipMetrics | None:                       
    """AUC + clip-level confusion from alarms.  None for single-class labels."""
    labels = list(labels)                                                             
    alarms = [bool(a) for a in alarms]
    if len(set(labels)) < 2:                                                          
        return None
                                                                                    
    auc = float(roc_auc_score(labels, scores))                                        
    tp = sum(l == 1 and a     for l, a in zip(labels, alarms))
    tn = sum(l == 0 and not a for l, a in zip(labels, alarms))                        
    fp = sum(l == 0 and a     for l, a in zip(labels, alarms))
    fn = sum(l == 1 and not a for l, a in zip(labels, alarms))                        
    p, r, f = _prf(tp, fp, fn)
    return ClipMetrics(auc=auc, precision=p, recall=r, f1=f,                          
                        tp=tp, tn=tn, fp=fp, fn=fn)
                                                                                    
                
def block_metrics(labels, alarms) -> BlockMetrics | None:                             
    """Block-level metrics treating the alarm trace as the state signal."""
    alarms = [1 if a else 0 for a in alarms]                                          
    anomaly_blocks = contiguous_runs(labels, target=1)
    if not anomaly_blocks:                                                            
        return None
    normal_regions = contiguous_runs(labels, target=0)                                
                                                                                    
    block_tp, block_fn, lags, unflags = 0, 0, [], []                                  
    for start, end in anomaly_blocks:                                                 
        first_fire = next(                                                            
            (i for i in range(start, end + 1) if alarms[i] == 1), None                
        )
        if first_fire is None:                                                        
            block_fn += 1
            continue                                                                  
        block_tp += 1
        lags.append(first_fire - start)                                               
        # Unflag: clips from band end until next non-alarm.
        # For CUSUM (auto-reset) this is usually 0 or 1.                              
        if end + 1 < len(alarms):                                                     
            unflag = next(                                                            
                (j - (end + 1)                                                        
                for j in range(end + 1, len(alarms)) if alarms[j] == 0),             
                None,
            )                                                                         
            if unflag is not None:
                unflags.append(unflag)
                                                                                    
    fired = [any(alarms[i] == 1 for i in range(s, e + 1))                             
            for s, e in normal_regions]                                              
    block_fp = sum(fired)                                                             
    block_tn = sum(not f for f in fired)
                                                                                    
    p, r, f = _prf(block_tp, block_fp, block_fn)
    return BlockMetrics(                                                              
        block_tp=block_tp, block_fp=block_fp,                                         
        block_fn=block_fn, block_tn=block_tn,
        block_precision=p, block_recall=r, block_f1=f,                                
        mean_lag   =float(np.mean(lags))    if lags    else None,                     
        mean_unflag=float(np.mean(unflags)) if unflags else None,                     
        missed_blocks=block_fn,                                                       
        total_blocks =block_tp + block_fn,                                            
    )                                                                                 

                                                                                    
# ── Convenience wrappers keyed by Node / Group ───────────────────────────
                                                                                    
def node_clip_metrics(node):   return clip_metrics(node.labels, node.scores,          
node.alarms)
def node_block_metrics(node):  return block_metrics(node.labels,                      
node.state)                                                                          
def group_clip_metrics(g):     return clip_metrics(g.labels,    g.fused_scores,
g.alarms)                                                                             
def group_block_metrics(g):    return block_metrics(g.labels,
g.state) 