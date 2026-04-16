"""Results serialisation — JSON, config copy, summary.txt."""
                                                                                                    
import json
import shutil                                                                                         
from datetime import datetime
from pathlib import Path

import numpy as np

from simulation.node.node import Node
from simulation.reporting.helpers import node_stats
                                                                                                    

def make_run_dir(base_dir: Path = Path("simulation/outputs/runs")) -> Path:                           
    """Create a fresh timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                                          
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)                                                       
    (run_dir / "plots").mkdir()
    return run_dir                                                                                    
                
                                                                                                    
def save_results(
    nodes_by_type: dict[str, list[Node]],                                                             
    config: dict,
    config_path: Path,
    runtime_seconds: float,
    run_dir: Path,
) -> None:                                                                                            
    """Write config.yaml, results.json, and summary.txt to the run directory."""
                                                                                                    
    shutil.copy(config_path, run_dir / "config.yaml")
                                                                                                    
    node_results: dict[str, dict] = {}                                                                
    type_aucs: dict[str, list[float]] = {}
                                                                                                    
    for machine_type, nodes in nodes_by_type.items():
        type_aucs[machine_type] = []
        for node in nodes:                                                                            
            stats = node_stats(node)
            node_results[node.node_id] = {                                                            
                "machine_type":     node.machine_type,
                "machine_id":       node.machine_id,                                                  
                "scores":           [float(s) for s in node.scores],
                "labels":           list(node.labels),                                                
                "predictions":      [int(p) if p is not None else None for p in node.predictions],
                "threshold":        getattr(node.separator, "threshold", None),                       
                "auc":              stats["auc"],
                "detection_rate":   stats["detection_rate"],                                          
                "false_alarm_rate": stats["false_alarm_rate"],
                "confusion": {                                                                        
                    "tp": stats["tp"], "tn": stats["tn"],
                    "fp": stats["fp"], "fn": stats["fn"],                                             
                },
            }                                                                                         
            if stats["auc"] is not None:
                type_aucs[machine_type].append(stats["auc"])                                          

    summary = {                                                                                       
        f"{mt}_mean_auc": float(np.mean(aucs)) if aucs else None
        for mt, aucs in type_aucs.items()                                                             
    }
    all_aucs = [a for aucs in type_aucs.values() for a in aucs]                                       
    summary["overall_mean_auc"] = float(np.mean(all_aucs)) if all_aucs else None
                                                                                                    
    results = { 
        "config":  config,                                                                            
        "nodes":   node_results,
        "summary": summary,
    }                                                                                                 

    with open(run_dir / "results.json", "w") as f:                                                    
        json.dump(results, f, indent=2)

    pipeline_str = (                                                                                  
        f"{config['preprocessor']} + "
        f"{config['frozen_embedder']} + "                                                             
        f"{config['separator']}"
    )                                                                                                 
    sim_block = config.get("simulation", {})
                                                                                                    
    runtime_minutes = int(runtime_seconds // 60)                                                      
    runtime_secs    = int(runtime_seconds % 60)
                                                                                                    
    lines: list[str] = []                                                                             
    lines.append(f"Run: {run_dir.name}")
    lines.append(f"Config: {config_path}")                                                            
    lines.append(f"Pipeline: {pipeline_str}")                                                         
    lines.append(
        f"Shuffle: {sim_block.get('shuffle_mode', '?')}  "                                            
        f"Warmup: {sim_block.get('warmup_count', '?')}  "                                             
        f"Seed: {sim_block.get('seed', '?')}"
    )                                                                                                 
    lines.append(f"Runtime: {runtime_minutes}m {runtime_secs}s")
    lines.append("-" * 56)                                                                            
    lines.append("")                                                                                  
    lines.append("Results")
    lines.append("=" * 56)                                                                            
                
    for machine_type, nodes in nodes_by_type.items():
        for node in nodes:
            r = node_results[node.node_id]                                                            
            if r["auc"] is None:
                lines.append(f"{node.node_id}: AUC = N/A (single class)")                             
                continue                                                                              
            c = r["confusion"]
            lines.append(                                                                             
                f"{node.node_id}: AUC = {r['auc']:.4f}  "
                f"|  TP={c['tp']:3d} TN={c['tn']:3d} "
                f"FP={c['fp']:3d} FN={c['fn']:3d}"                                                    
            )
                                                                                                    
    lines.append("-" * 56)
    for machine_type in nodes_by_type:
        mean = summary.get(f"{machine_type}_mean_auc")                                                
        if mean is not None:                                                                          
            lines.append(f"{machine_type} mean AUC: {mean:.4f}")                                      
    lines.append("")                                                                                  
    if summary["overall_mean_auc"] is not None:                                                       
        lines.append(f"Overall mean AUC: {summary['overall_mean_auc']:.4f}")
    lines.append("=" * 56)                                                                            
                                                                                                    
    with open(run_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")  