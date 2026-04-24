"""Terminal output — per-timestep live line + final result tables.
                                                                                    
Three result sections:
1. Per-node (independent)        — clip + block metrics + per-type means            
2. Per-group (node-learning)     — same shape, only when n_nodes > 1                
3. NL-vs-independent comparison  — fused AUC vs mean-of-nodes AUC                   
                                                                                    
result_lines(...) returns list[str] used by both print_results (terminal)             
and reporting.results.save_results (summary.txt).                                     
"""             
                                                                                    
from collections import defaultdict
                                                                                    
import numpy as np

from simulation.lockstep   import TimestepResult                                      
from simulation.metrics    import (
    ClipMetrics, BlockMetrics,                                                        
    node_clip_metrics, node_block_metrics,
    group_clip_metrics, group_block_metrics,                                          
)
from simulation.node.node  import Node                                                
from simulation.node.group import Group                                               

                                                                                    
_WIDTH = 88     
_SEP   = "─" * _WIDTH
_HEAD  = "=" * _WIDTH                                                                 

                                                                                    
# ── Per-timestep live line (unchanged glyph idea) ────────────────────────
_GLYPH = {(0, False): "·", (1, True): "✓", (1, False): "✗", (0, True): "!"}           
                                                                                    
                                                                                    
def format_step(step: TimestepResult, machine_types: list[str]) -> str:               
    by_type: dict[str, list[str]] = {mt: [] for mt in machine_types}                  
    n_anom = n_fired = 0                                                              
    for r in step.node_results:                                                       
        by_type[r.machine_type].append(_GLYPH[(r.label, r.alarm)])                    
        if r.label == 1: n_anom  += 1                                                 
        if r.alarm:      n_fired += 1                                                 
                                                                                    
    chunks = [f"{mt[:3]}:{''.join(by_type[mt])}" for mt in machine_types]             
    line   = f"  t={step.timestep:3d}  |  {'  '.join(chunks)}"
    if step.group_results:                                                            
        g_glyphs = [_GLYPH[(g.label, g.alarm)] for g in step.group_results]
        line += f"  ||  fused:{''.join(g_glyphs)}"                                    
    n_total = len(step.node_results)                                                  
    line += f"  |  anom {n_anom}/{n_total}  fired {n_fired}/{n_total}"                
    return line                                                                       
                                                                                    
                
# ── Row formatters ───────────────────────────────────────────────────────           
                
def _fmt_clip_row(label: str, m: ClipMetrics) -> str:                                 
    auc = f"{m.auc:.4f}" if m.auc is not None else " N/A  "
    return (f"  {label:<24}  AUC={auc}  "                                             
            f"|  TP={m.tp:3d} TN={m.tn:3d} FP={m.fp:3d} FN={m.fn:3d}  "
            f"|  P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")                 
                
                                                                                    
def _fmt_block_row(indent_width: int, m: BlockMetrics) -> str:
    lag    = f"{m.mean_lag:4.1f}"    if m.mean_lag    is not None else " N/A"         
    unflag = f"{m.mean_unflag:4.1f}" if m.mean_unflag is not None else " N/A"         
    pad    = " " * indent_width
    return (f"{pad}block              "                                               
            f"|  bTP={m.block_tp:3d} bTN={m.block_tn:3d} "                            
            f"bFP={m.block_fp:3d} bFN={m.block_fn:3d}  "                              
            f"|  bP={m.block_precision:.3f} bR={m.block_recall:.3f} "                 
            f"bF1={m.block_f1:.3f}  "
            f"|  Lag={lag}  Unflag={unflag}  "                                        
            f"Miss={m.missed_blocks:2d}/{m.total_blocks:2d}")
                                                                                    
                                                                                    
def _fmt_clip_mean(label: str, metrics: list[ClipMetrics]) -> str:                    
    aucs = [m.auc for m in metrics if m.auc is not None]                              
    auc = f"{np.mean(aucs):.4f}" if aucs else " N/A  "
    return (f"  {label:<24}  mean   AUC={auc}  "                                      
            f"P={np.mean([m.precision for m in metrics]):.3f}  "
            f"R={np.mean([m.recall    for m in metrics]):.3f}  "                      
            f"F1={np.mean([m.f1       for m in metrics]):.3f}")
                                                                                    
                
def _fmt_block_mean(label: str, metrics: list[BlockMetrics]) -> str:                  
    lags    = [m.mean_lag    for m in metrics if m.mean_lag    is not None]
    unflags = [m.mean_unflag for m in metrics if m.mean_unflag is not None]           
    lag    = f"{np.mean(lags):4.1f}"    if lags    else " N/A"
    unflag = f"{np.mean(unflags):4.1f}" if unflags else " N/A"                        
    miss   = sum(m.missed_blocks for m in metrics)                                    
    total  = sum(m.total_blocks  for m in metrics)                                    
    return (f"  {label:<24}  block  "                                                 
            f"|  bP={np.mean([m.block_precision for m in metrics]):.3f}  "
            f"bR={np.mean([m.block_recall for m in metrics]):.3f}  "                  
            f"bF1={np.mean([m.block_f1 for m in metrics]):.3f}  "
            f"|  Lag={lag}  Unflag={unflag}  Miss={miss:3d}/{total:3d}")              
                                                                                    
                                                                                    
# ── Section builders (return list[str]) ──────────────────────────────────           
                
def _lines_per_node(nodes_by_type: dict[str, list[Node]]) -> list[str]:               
    out = ["", "Per-node (independent)", _SEP]
    type_clip:  dict[str, list[ClipMetrics]]  = defaultdict(list)                     
    type_block: dict[str, list[BlockMetrics]] = defaultdict(list)                     
                                                                                    
    for mtype, nodes in nodes_by_type.items():                                        
        for n in nodes:
            cm = node_clip_metrics(n)                                                 
            bm = node_block_metrics(n)
            if cm is None:
                out.append(f"  {n.node_id}: AUC = N/A (single class)")                
                continue                                                              
            type_clip[mtype].append(cm)                                               
            out.append(_fmt_clip_row(n.node_id, cm))                                  
            if bm is not None:                                                        
                type_block[mtype].append(bm)
                out.append(_fmt_block_row(4 + len(n.node_id), bm))                    
                                                                                    
    out.append(_SEP)
    for mtype in nodes_by_type:                                                       
        if mtype in type_clip:                                                        
            out.append(_fmt_clip_mean(mtype, type_clip[mtype]))
            if mtype in type_block:                                                   
                out.append(_fmt_block_mean(mtype, type_block[mtype]))
                                                                                    
    all_clip  = [m for ms in type_clip.values()  for m in ms]
    all_block = [m for ms in type_block.values() for m in ms]                         
    if all_clip:                                                                      
        out.append(_SEP)
        out.append(_fmt_clip_mean("Overall", all_clip))                               
    if all_block:                                                                     
        out.append(_fmt_block_mean("Overall", all_block))
    return out                                                                        
                
                                                                                    
def _lines_per_group(groups_by_type: dict[str, list[Group]]) -> list[str]:
    n_groups = sum(len(gs) for gs in groups_by_type.values())                         
    if n_groups == 0:                                                                 
        return []
                                                                                    
    out = ["", "Per-group (node-learning fused)", _SEP]                               
    type_clip:  dict[str, list[ClipMetrics]]  = defaultdict(list)
    type_block: dict[str, list[BlockMetrics]] = defaultdict(list)                     
                                                                                    
    for mtype, groups in groups_by_type.items():
        for g in groups:                                                              
            cm = group_clip_metrics(g)
            bm = group_block_metrics(g)
            w_str = np.round(g.w, 2).tolist()                                         
            label = f"{g.group_id} (n={len(g.nodes)} w={w_str})"                      
            if cm is None:                                                            
                out.append(f"  {label}: AUC = N/A (single class)")                    
                continue
            type_clip[mtype].append(cm)                                               
            out.append(_fmt_clip_row(label, cm))
            if bm is not None:                                                        
                type_block[mtype].append(bm)
                out.append(_fmt_block_row(4 + len(label), bm))                        
                
    out.append(_SEP)                                                                  
    for mtype in groups_by_type:
        if mtype in type_clip:                                                        
            out.append(_fmt_clip_mean(f"{mtype} fused", type_clip[mtype]))
            if mtype in type_block:                                                   
                out.append(_fmt_block_mean(f"{mtype} fused", type_block[mtype]))
                                                                                    
    all_clip  = [m for ms in type_clip.values()  for m in ms]                         
    all_block = [m for ms in type_block.values() for m in ms]
    if all_clip:                                                                      
        out.append(_SEP)
        out.append(_fmt_clip_mean("Overall fused", all_clip))
    if all_block:                                                                     
        out.append(_fmt_block_mean("Overall fused", all_block))
    return out                                                                        
                
                                                                                    
def _lines_nl_vs_independent(
    nodes_by_type:  dict[str, list[Node]],                                            
    groups_by_type: dict[str, list[Group]],
) -> list[str]:                                                                       
    n_groups = sum(len(gs) for gs in groups_by_type.values())
    if n_groups == 0:                                                                 
        return []

    out = [""]                                                                        
    out.append("Node-learning vs independent  (mean AUC of peer nodes vs fused AUC)")
    hdr = (f"  {'Machine':<16}  {'n':>3}  {'mean AUC':>9}  "                          
            f"{'fused AUC':>10}  {'Δ':>7}  {'fused alarms':>13}")                      
    sep = "─" * len(hdr)                                                              
    out.append(sep); out.append(hdr); out.append(sep)                                 
                                                                                    
    mean_aucs:  list[float] = []                                                      
    fused_aucs: list[float] = []
    for mtype, groups in groups_by_type.items():
        for g in groups:
            per_node_aucs = []
            for n in g.nodes:
                cm = node_clip_metrics(n)
                if cm is not None and cm.auc is not None:
                    per_node_aucs.append(cm.auc)                                      
            gm = group_clip_metrics(g)
            mean_auc  = float(np.mean(per_node_aucs)) if per_node_aucs else float("nan")                                                                          
            fused_auc = gm.auc if (gm and gm.auc is not None) else float("nan")
            delta     = fused_auc - mean_auc                                          
            out.append(f"  {g.group_id:<16}  {len(g.nodes):>3d}  "
                        f"{mean_auc:>9.3f}  {fused_auc:>10.3f}  "                      
                        f"{delta:>+7.3f}  {sum(g.alarms):>13d}")                       
            if not np.isnan(mean_auc):  mean_aucs.append(mean_auc)                    
            if not np.isnan(fused_auc): fused_aucs.append(fused_auc)                  
                                                                                    
    out.append(sep)                                                                   
    if mean_aucs and fused_aucs:                                                      
        d = float(np.mean(fused_aucs) - np.mean(mean_aucs))
        out.append(f"  {'Overall':<16}        "                                       
                    f"{np.mean(mean_aucs):>9.3f}  {np.mean(fused_aucs):>10.3f}  "
                    f"{d:>+7.3f}")                                                    
    out.append(sep)                                                                   
    return out                                                                        
                                                                                    
                
# ── Public API ───────────────────────────────────────────────────────────

def result_lines(                                                                     
    nodes_by_type:  dict[str, list[Node]],
    groups_by_type: dict[str, list[Group]],                                           
) -> list[str]: 
    """All three sections, ready to print() or write() line by line."""               
    out = ["", _HEAD, "Results", _HEAD]                                               
    out += _lines_per_node(nodes_by_type)                                             
    out += _lines_per_group(groups_by_type)                                           
    out += _lines_nl_vs_independent(nodes_by_type, groups_by_type)                    
    out.append(_HEAD)                                                                 
    out.append("")
    return out                                                                        
                
                                                                                    
def print_results(
    nodes_by_type:  dict[str, list[Node]],                                            
    groups_by_type: dict[str, list[Group]],
) -> None:
    for line in result_lines(nodes_by_type, groups_by_type):
        print(line)