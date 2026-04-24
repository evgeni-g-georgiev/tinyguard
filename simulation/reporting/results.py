"""Results serialisation — JSON, config copy, summary.txt.
                                                                                                                                
Full-trace schema: per_node carries score + cusum_S + label + alarm traces;
per_group carries fused_score + cusum_S + label + alarm traces.  Everything                                                     
the reporting layer needs to replot without rerunning.                                                                          
"""                                                                                                                             
                                                                                                                                
import json                                                                                                                     
import shutil   
from datetime import datetime
from pathlib import Path
                                                                                                                                
import numpy as np
from sklearn.metrics import roc_auc_score                                                                                       
                
from simulation.node.node  import Node
from simulation.node.group import Group
                                                                                                                                

# ── Run dir ──────────────────────────────────────────────────────────────                                                     
                
def make_run_dir(base_dir: Path = Path("simulation/outputs/runs")) -> Path:                                                     
    """Create a fresh timestamped run directory and return its path."""
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                                                                     
    run_dir  = base_dir / ts                                                                                                    
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()                                                                                                 
    return run_dir                                                                                                              

                                                                                                                                
# ── JSON save ────────────────────────────────────────────────────────────
                                                                                                                                
def _safe_auc(labels, scores):
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:                                                                                                           
        return None
                                                                                                                                
                
def _node_entry(n: Node) -> dict:
    return {
        "machine_type": n.machine_type,
        "machine_id":   n.machine_id,                                                                                           
        "channel":      n.channel,
        "r":            float(n.r),                                                                                             
        "k":            float(n.k),                                                                                             
        "h":            float(n.h),
        "mu_val":       float(n.mu_val),                                                                                        
        "sigma_val":    float(n.sigma_val),
        "auc":          _safe_auc(n.labels, n.scores),                                                                          
        "n_alarms":     int(sum(n.alarms)),
        "scores":       [float(s) for s in n.scores],                                                                           
        "cusum_S":      [float(s) for s in n.cusum_S],
        "labels":       [int(l)   for l in n.labels],                                                                           
        "alarms":       [bool(a)  for a in n.alarms],
    }                                                                                                                           
                
                                                                                                                                
def _group_entry(g: Group) -> dict:
    mean_node_auc = float(np.nanmean([
        _safe_auc(n.labels, n.scores) or np.nan for n in g.nodes                                                                
    ]))
    return {                                                                                                                    
        "machine_type":  g.machine_type,                                                                                        
        "machine_id":    g.machine_id,
        "n_nodes":       len(g.nodes),                                                                                          
        "node_ids":      [n.node_id for n in g.nodes],                                                                          
        "w":             [float(x) for x in g.w],
        "temperature":   g.temperature,                                                                                         
        "k":             float(g.k),
        "h":             float(g.h),                                                                                            
        "auc":           _safe_auc(g.labels, g.fused_scores),
        "mean_node_auc": mean_node_auc,                                                                                         
        "n_alarms":      int(sum(g.alarms)),
        "fused_scores":  [float(s) for s in g.fused_scores],                                                                    
        "cusum_S":       [float(s) for s in g.cusum_S],
        "labels":        [int(l)   for l in g.labels],                                                                          
        "alarms":        [bool(a)  for a in g.alarms],                                                                          
    }
                                                                                                                                
                
def save_results(
    *,
    nodes_by_type:   dict[str, list[Node]],
    groups_by_type:  dict[str, list[Group]],                                                                                    
    config:          dict,
    config_path:     Path,                                                                                                      
    runtime_seconds: float,                                                                                                     
    run_dir:         Path,
) -> None:                                                                                                                      
    """Write config.yaml, results.json, summary.txt into run_dir."""
    shutil.copy(config_path, run_dir / "config.yaml")                                                                           

    per_node  = {n.node_id: _node_entry(n)                                                                                      
                for ns in nodes_by_type.values()  for n in ns}
    per_group = {g.group_id: _group_entry(g)                                                                                    
                for gs in groups_by_type.values() for g in gs}                                                                 

    # Aggregate AUC summary.                                                                                                    
    node_aucs  = [e["auc"]          for e in per_node.values()  if e["auc"] is not None]
    group_aucs = [e["auc"]          for e in per_group.values() if e["auc"] is not None]                                        
    mean_of_nodes_per_group = [e["mean_node_auc"] for e in per_group.values()                                                   
                                if not np.isnan(e["mean_node_auc"])]                                                            
                                                                                                                                
    summary = {                                                                                                                 
        "mean_node_auc":   float(np.mean(node_aucs))  if node_aucs  else None,
        "mean_fused_auc":  float(np.mean(group_aucs)) if group_aucs else None,                                                  
        "nl_gain_vs_mean": (
            float(np.mean(group_aucs) - np.mean(mean_of_nodes_per_group))                                                       
            if group_aucs and mean_of_nodes_per_group else None                                                                 
        ),                                                                                                                      
    }                                                                                                                           
                
    out = {                                                                                                                     
        "runtime_seconds": runtime_seconds,
        "config":          config,                                                                                              
        "per_node":        per_node,
        "per_group":       per_group,
        "summary":         summary,                                                                                             
    }
                                                                                                                                
    with open(run_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    _write_summary_txt(                                                               
        run_dir, config, config_path, runtime_seconds, summary,
        nodes_by_type, groups_by_type,                                  
    )                                                

def _write_summary_txt(                                                               
    run_dir, config, config_path, runtime_seconds, summary,
    nodes_by_type, groups_by_type,                                                    
) -> None:
    from simulation.formatters import result_lines                                    
                                                                                    
    mins, secs = int(runtime_seconds // 60), int(runtime_seconds % 60)                
                                                                                    
    header = [                                                                        
        f"Run: {run_dir.name}",
        f"Config: {config_path}",
        f"n_nodes={config['n_nodes']}  T={config['temperature']}  "                   
        f"n_mels={config['gmm']['n_mels']}  snr={config['snr']}",                     
        f"Shuffle: {config['simulation']['shuffle_mode']}  "                          
        f"Warmup: {config['simulation']['warmup_count']}  "                           
        f"Seed: {config['simulation']['seed']}",                                      
        f"Runtime: {mins}m {secs}s",                                                  
        "",     
        f"Top-level AUC summary:",                                                    
        f"  mean per-node AUC:  {summary['mean_node_auc']}",                          
        f"  mean fused AUC:     {summary['mean_fused_auc']}",                         
        f"  NL gain (fused − mean of nodes): {summary['nl_gain_vs_mean']}",           
    ]                                                                                 
                
    lines = header + result_lines(nodes_by_type, groups_by_type)                      
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")