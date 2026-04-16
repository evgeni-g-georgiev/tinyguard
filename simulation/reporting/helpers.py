"""Shared helpers used across the reporting subpackage.
                                                                                                    
Pure computation functions (node_stats, compute_bracket_data) and one
matplotlib drawing primitive (draw_state_brackets). These are the                                     
building blocks that results.py and timeline_plots.py compose.
"""                                                                                                   
                
import numpy as np                                                                                    
from matplotlib.transforms import blended_transform_factory
from sklearn.metrics import roc_auc_score
                                                                                                    
from simulation.node.node import Node
from simulation.metrics import find_anomaly_bands                                                     
                

def node_stats(node: Node) -> dict:
    """Compute per-node stats: AUC, detection rate, false alarm rate, confusion."""
    has_two_classes = len(set(node.labels)) >= 2                                                      
    has_predictions = all(p is not None for p in node.predictions)
                                                                                                    
    auc = roc_auc_score(node.labels, node.scores) if has_two_classes else None                        
                                                                                                    
    tp = tn = fp = fn = 0                                                                             
    detection_rate = false_alarm_rate = None

    if has_predictions:
        for label, pred in zip(node.labels, node.predictions):
            if label == 1 and pred == 1:                                                              
                tp += 1
            elif label == 0 and pred == 0:                                                            
                tn += 1                                                                               
            elif label == 0 and pred == 1:
                fp += 1                                                                               
            elif label == 1 and pred == 0:
                fn += 1
                                                                                                    
        if (tp + fn) > 0:
            detection_rate = tp / (tp + fn)                                                           
        if (fp + tn) > 0:
            false_alarm_rate = fp / (fp + tn)
                                                                                                    
    return {
        "auc":              auc,                                                                      
        "detection_rate":   detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }                                                                                                 

                                                                                                    
def compute_bracket_data(
        node: Node,
        manual_reset: bool
        ) -> dict | None:
    """Extract per-block lag and unflag arrays from a node's state predictions."""
    state = node.state_predictions
    if not state or all(s is None for s in state):
        return None

    bands = find_anomaly_bands(node.labels)
    if not bands:
        return None

    lags    = [next((i - start for i in range(start, end + 1) if state[i] == 1), None)
               for start, end in bands]
    unflags = [next((j - (end + 1) for j in range(end + 1, len(state)) if state[j] == 0), None)
               if not manual_reset and lag is not None and end + 1 < len(state) else None
               for (start, end), lag in zip(bands, lags)]

    return {"bands": bands, "lags": lags, "unflags": unflags}


def draw_state_brackets(
    ax,
    bracket_data: dict,                                                                               
    manual_reset: bool,
) -> None:                                                                                            
    """Draw horizontal bars above anomaly bands showing state detection timing.
                                                                                                    
    Uses a blended transform (x = data coordinates, y = axes fraction) so
    bars sit at a fixed height regardless of score range.                                             
                                                                                                    
    Bar types:                                                                                        
        GREEN  at y=0.97 : detected block, width = lag+1 clips.                                       
        RED    at y=0.97 : missed block, spans full block width.                                      
        BLUE   at y=0.93 : recovery lag (non-manual-reset mode only).                                 
    """                                                                                               
    trans = blended_transform_factory(ax.transData, ax.transAxes)
                                                                                                    
    y_detect = 0.97                                                                                   
    y_unflag = 0.93
                                                                                                    
    green_labeled = False
    red_labeled = False
    blue_labeled = False

    bands = bracket_data["bands"]                                                                     
    lags = bracket_data["lags"]
    unflags = bracket_data["unflags"]                                                                 
                
    for (start, end), lag, unflag in zip(bands, lags, unflags):                                       
        if lag is None:
            label = None if red_labeled else "Missed block"                                           
            red_labeled = True                                                                        
            ax.plot(
                [start - 0.5, end + 0.5], [y_detect, y_detect],                                       
                color="crimson", linewidth=3,                                                         
                solid_capstyle="butt", transform=trans, label=label,                                  
            )                                                                                         
        else:                                                                                         
            label = None if green_labeled else "Detection lag"
            green_labeled = True                                                                      
            ax.plot(
                [start - 0.5, start + lag + 0.5], [y_detect, y_detect],                               
                color="forestgreen", linewidth=3,                                                     
                solid_capstyle="butt", transform=trans, label=label,
            )                                                                                         
                
            if not manual_reset and unflag is not None:                                               
                label = None if blue_labeled else "Recovery lag"
                blue_labeled = True                                                                   
                ax.plot(
                    [end + 0.5, end + 1 + unflag + 0.5], [y_unflag, y_unflag],
                    color="royalblue", linewidth=3,                                                   
                    solid_capstyle="butt", transform=trans, label=label,
                )                  