"""Terminal output formatting — turns metrics into printed rows.
                                                                                                    
Pure string-returning functions plus print_results() which orchestrates
the full results block. Imports from metrics, nothing else.                                           
"""  

from collections import defaultdict 

import numpy as np 

from simulation.node.node import Node 
from simulation.lockstep import TimestepResult 
from simulation.metrics import (
    NodeMetrics,
    BlockStateMetrics, 
    node_metrics, 
    block_state_metrics,
)


# ── Per-timestep live display ────────────────────────────────────────────────
#                                                                                                     
# Symbols for the per-node status grid:                                                               
#   ·  true negative   (normal predicted normal)
#   ✓  true positive   (anomaly predicted anomaly — caught it)                                        
#   ✗  false negative  (anomaly predicted normal  — missed)
#   !  false positive  (normal predicted anomaly  — false alarm)                                      
#   ?  no prediction   (separator has no threshold)
                                                                                                    
def status_symbol(label: int, prediction: int | None) -> str:                                         
    if prediction is None:                                                                            
        return "?"                                                                                    
    if label == 1 and prediction == 1:
        return "✓"                                                                                    
    if label == 1 and prediction == 0:
        return "✗"                                                                                    
    if label == 0 and prediction == 1:
        return "!"
    return "·"
                                                                                                    

def format_step(result: TimestepResult, machine_types: list[str]) -> str:                             
    """Build a one-line summary of all 16 nodes for one timestep."""
    by_type: dict[str, list[str]] = {mt: [] for mt in machine_types}                                  
    n_anom = 0                                                                                        
    n_correct = 0                                                                                     
    n_with_pred = 0                                                                                   
                                                                                                    
    for r in result.node_results:
        by_type[r.machine_type].append(status_symbol(r.label, r.predicted_label))                     
        if r.label == 1:                                                                              
            n_anom += 1
        if r.predicted_label is not None:                                                             
            n_with_pred += 1                                                                          
            if r.label == r.predicted_label:
                n_correct += 1                                                                        
                
    groups = [
        f"{mt[:3]}:{''.join(by_type[mt])}" for mt in machine_types
    ]                                                                                                 
    accuracy = f"{n_correct:2d}/{n_with_pred:2d}" if n_with_pred else "n/a"
                                                                                                    
    return (    
        f"  t={result.timestep:3d}  |  "                                                              
        f"{'  '.join(groups)}  |  "                                                                   
        f"anom {n_anom:2d}/16  acc {accuracy}"                                                        
    ) 


# ── Per-node result rows ────────────────────────────────────────────────────────

def format_node_row(node_id: str, m: NodeMetrics) -> str:                                             
    return (f"  {node_id}: AUC={m.auc:.4f}  "
            f"|  TP={m.tp:3d} TN={m.tn:3d} FP={m.fp:3d} FN={m.fn:3d}  "                               
            f"|  P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")                                 
                                                                                                    
                                                                                                    
def format_mean_row(label: str, metrics: list[NodeMetrics]) -> str:
    return (f"  {label:12s} mean  "                                                                   
            f"AUC={np.mean([m.auc       for m in metrics]):.4f}  "                                    
            f"P={np.mean([m.precision   for m in metrics]):.3f}  "
            f"R={np.mean([m.recall      for m in metrics]):.3f}  "                                    
            f"F1={np.mean([m.f1         for m in metrics]):.3f}")


                                                                                                    
# ── Per-node state rows ─────────────────────────────────────────────────────                        
                                                                                                    
def format_state_row(
    node_id: str,
    m: BlockStateMetrics,
    manual_reset: bool,
) -> str:
    mode_tag = "mr" if manual_reset else "auto"
    indent = " " * (4 + len(node_id))                                                                 

    lag_str = (                                                                                       
        f"Lag={m.mean_lag:4.1f}"
        if m.mean_lag is not None else "Lag= N/A"                                                     
    )
    unflag_str = (                                                                                    
        f"Unflag={m.mean_unflag:4.1f}"
        if (not manual_reset and m.mean_unflag is not None)
        else "Unflag=   —"                                                                            
    )
                                                                                                    
    return (    
        f"{indent}state ({mode_tag})  "
        f"|  bTP={m.block_tp:3d} bTN={m.block_tn:3d} bFP={m.block_fp:3d} bFN={m.block_fn:3d}  "
        f"|  bP={m.block_precision:.3f} bR={m.block_recall:.3f} bF1={m.block_f1:.3f}  "               
        f"|  {lag_str}  {unflag_str}  Miss={m.missed_blocks:2d}/{m.total_blocks:2d}"                  
    )                                                                                                 
                                                                                                    
                                                                                                    
def format_state_mean_row(
    label: str,
    metrics: list[BlockStateMetrics],
    manual_reset: bool,
) -> str:
    lags = [m.mean_lag for m in metrics if m.mean_lag is not None]
    unflags = [m.mean_unflag for m in metrics if m.mean_unflag is not None]                           

    lag_str = (                                                                                       
        f"Lag={np.mean(lags):4.1f}" if lags else "Lag= N/A"
    )                                                                                                 
    unflag_str = (
        f"Unflag={np.mean(unflags):4.1f}"                                                             
        if (not manual_reset and unflags)                                                             
        else "Unflag=   —"
    )                                                                                                 
    total_missed = sum(m.missed_blocks for m in metrics)
    total_blocks = sum(m.total_blocks for m in metrics)

    return (                                                                                          
        f"  {label:12s} state mean  "
        f"|  bP={np.mean([m.block_precision for m in metrics]):.3f}  "                                
        f"bR={np.mean([m.block_recall for m in metrics]):.3f}  "
        f"bF1={np.mean([m.block_f1 for m in metrics]):.3f}  "                                         
        f"|  {lag_str}  {unflag_str}  Miss={total_missed:3d}/{total_blocks:3d}"
    )  

      
                                                                                                    
# ── Full results block ──────────────────────────────────────────────────────

def print_results(                                                                                    
    nodes_by_type: dict[str, list[Node]],
    state_enabled: bool = False,                                                                      
    manual_reset: bool = False,
) -> None:
    """Compute and print clip-level and (optionally) block-state metrics."""
    print("\n" + "=" * 60)                                                                            
    print("Results")
    print("=" * 60)                                                                                   
                                                                                                    
    type_metrics: dict[str, list[NodeMetrics]] = defaultdict(list)
    type_state_metrics: dict[str, list[BlockStateMetrics]] = defaultdict(list)                        
                
    for machine_type, nodes in nodes_by_type.items():                                                 
        for node in nodes:
            m = node_metrics(node)                                                                    
            if m is None:
                print(f"  {node.node_id}: AUC = N/A (single class)")
                continue
            type_metrics[machine_type].append(m)
            print(format_node_row(node.node_id, m))                                                   

            if state_enabled:                                                                         
                sm = block_state_metrics(node, manual_reset)
                if sm is not None:                                                                    
                    type_state_metrics[machine_type].append(sm)
                    print(format_state_row(node.node_id, sm, manual_reset))                           
                
    print("-" * 60)                                                                                   
    for machine_type, metrics in type_metrics.items():
        print(format_mean_row(machine_type, metrics))                                                 
        if state_enabled and type_state_metrics.get(machine_type):
            print(format_state_mean_row(
                machine_type, type_state_metrics[machine_type], manual_reset,                         
            ))
                                                                                                    
    all_metrics = [m for metrics in type_metrics.values() for m in metrics]
    if all_metrics:
        print("\n" + format_mean_row("Overall", all_metrics))
                                                                                                    
    if state_enabled:
        all_state_metrics = [                                                                         
            m for metrics in type_state_metrics.values() for m in metrics
        ]
        if all_state_metrics:
            print(format_state_mean_row("Overall", all_state_metrics, manual_reset))
                                                                                                    
    print("=" * 60)