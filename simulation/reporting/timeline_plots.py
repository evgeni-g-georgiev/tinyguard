"""Score-over-time plots — per-node full-size, grid, and state grid."""
                                                                                                    
from pathlib import Path                                                                              
                                                                                                    
import matplotlib.pyplot as plt                                                                       
import numpy as np

from simulation.node.node import Node
from simulation.metrics import find_anomaly_bands
from simulation.reporting.helpers import (
    node_stats,                                                                                       
    compute_bracket_data,
    draw_state_brackets,                                                                              
)               

                                                                                                    
def _draw_node_axes(ax, node: Node, config: dict, compact: bool = False) -> None:
    """Render one node's scatter + threshold + bands onto an existing Axes."""                        
    timesteps = np.arange(len(node.scores))                                                           
    scores = np.array(node.scores)
    labels = np.array(node.labels)                                                                    
                
    normal_mask = labels == 0                                                                         
    anomaly_mask = labels == 1

    shuffle_mode = config.get("simulation", {}).get("shuffle_mode", "random")                         
    show_bands = shuffle_mode in ("block_random", "block_fixed")
                                                                                                    
    bands: list[tuple[int, int]] = []
    if show_bands:                                                                                    
        bands = find_anomaly_bands(node.labels)                                                       
        for i, (start, end) in enumerate(bands):
            ax.axvspan(                                                                               
                start - 0.5, end + 0.5,                                                               
                alpha=0.18, color="lightcoral",
                label="Anomaly injection" if i == 0 else None,                                        
            )                                                                                         

    ax.scatter(                                                                                       
        timesteps[normal_mask], scores[normal_mask],
        s=18, c="steelblue", alpha=0.7, label="Normal clips",                                         
    )                                                                                                 
    ax.scatter(                                                                                       
        timesteps[anomaly_mask], scores[anomaly_mask],                                                
        s=18, c="darkorange", alpha=0.7, label="Anomaly clips",
    )

    threshold = getattr(node.separator, "threshold", None)                                            
    if threshold is not None:
        ax.axhline(                                                                                   
            threshold, color="firebrick", linestyle="--", linewidth=1.2,
            label=f"Threshold ({threshold:.3f})",                                                     
        )
                                                                                                    
    stats = node_stats(node)
    auc_str = f"{stats['auc']:.4f}" if stats["auc"] is not None else "N/A"
                                                                                                    
    if compact:
        ax.set_title(                                                                                 
            f"{node.machine_type} {node.machine_id}: AUC={auc_str}",
            fontsize=10,                                                                              
        )
    else:                                                                                             
        manual_reset = config.get("simulation", {}).get("manual_reset", False)
        bracket_data = compute_bracket_data(node, manual_reset) if bands else None                    

        if bracket_data is not None:                                                                  
            draw_state_brackets(ax, bracket_data, manual_reset)
                                                                                                    
        det_str = (
            f"{stats['detection_rate'] * 100:.1f}%"
            if stats["detection_rate"] is not None else "N/A"                                         
        )
        fa_str = (                                                                                    
            f"{stats['false_alarm_rate'] * 100:.1f}%"
            if stats["false_alarm_rate"] is not None else "N/A"                                       
        )
                                                                                                    
        title_suffix = ""                                                                             
        if bracket_data is not None:
            lags = bracket_data["lags"]                                                               
            caught = [l for l in lags if l is not None]
            missed = len(lags) - len(caught)                                                          
            lag_part = (
                f"Lag={np.mean(caught):.1f} clips"                                                    
                if caught else "Lag=N/A"                                                              
            )
            title_suffix = f"  |  {lag_part}  ({missed}/{len(lags)} missed)"                          
                
        ax.set_title(                                                                                 
            f"Node: {node.machine_type} {node.machine_id}\n"
            f"Detection={det_str}  |  False alarm={fa_str}  |  "                                      
            f"AUC={auc_str}{title_suffix}",                                                           
            fontsize=11,
        )                                                                                             
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Anomaly score")                                                                
        ax.legend(loc="upper left", fontsize=8)
                                                                                                    
    ax.grid(alpha=0.3)
                                                                                                    
                
def _render_state_grid(
    nodes_by_type: dict[str, list[Node]],
    config: dict,
    plots_dir: Path,
) -> None:                                                                                            
    """Render a 4x4 grid — same as grid.png but with state brackets overlaid."""
    manual_reset = config.get("simulation", {}).get("manual_reset", False)                            
    machine_types = list(nodes_by_type.keys())
    n_cols = len(machine_types)                                                                       
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())
                                                                                                    
    fig, axes = plt.subplots(                                                                         
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),                                                             
        sharex="col", sharey="row",
    )

    sample_node = next(iter(nodes_by_type.values()))[0]                                               
    sep_desc = sample_node.separator.description()
    mode_tag = "manual-reset" if manual_reset else "auto"                                             
    fig.suptitle(                                                                                     
        f"{sep_desc}  |  block state ({mode_tag})",
        fontsize=14, fontweight="bold", y=0.995,                                                      
    )           
                                                                                                    
    for col, machine_type in enumerate(machine_types):
        nodes = nodes_by_type[machine_type]
        for row, node in enumerate(nodes):                                                            
            ax = axes[row, col] if n_rows > 1 else axes[col]
                                                                                                    
            _draw_node_axes(ax, node, config, compact=True)                                           

            bracket_data = compute_bracket_data(node, manual_reset)                                   
            if bracket_data is not None:
                draw_state_brackets(ax, bracket_data, manual_reset)
                                                                                                    
                lags = bracket_data["lags"]
                caught = [l for l in lags if l is not None]                                           
                missed = len(lags) - len(caught)
                lag_str = f"Lag={np.mean(caught):.1f}" if caught else "Lag=N/A"
                ax.set_title(                                                                         
                    f"{node.machine_type} {node.machine_id}  "
                    f"{lag_str}  Miss={missed}/{len(lags)}",                                          
                    fontsize=9,
                )                                                                                     
                
    fig.supxlabel("Timestep", fontsize=11)                                                            
    fig.supylabel("Anomaly score", fontsize=11)
    fig.tight_layout()                                                                                
    fig.savefig(plots_dir / "grid_state.png", dpi=110)
    plt.close(fig)                                                                                    

                                                                                                    
def save_plots( 
    nodes_by_type: dict[str, list[Node]],
    config: dict,
    run_dir: Path,                                                                                    
) -> None:
    """Write plots/grid.png and plots/<node_id>.png x 16."""                                          
    plots_dir = run_dir / "plots"                                                                     

    sample_node = next(iter(nodes_by_type.values()))[0]                                               
    sep_desc = sample_node.separator.description()
    embedder_name = config["frozen_embedder"]                                                         
    fig_title = f"{sep_desc}  |  {embedder_name} embeddings"
                                                                                                    
    for nodes in nodes_by_type.values():                                                              
        for node in nodes:                                                                            
            fig, ax = plt.subplots(figsize=(11, 4.5))                                                 
            fig.suptitle(fig_title, fontsize=12, fontweight="bold")
            _draw_node_axes(ax, node, config, compact=False)                                          
            fig.tight_layout()
            fig.savefig(plots_dir / f"{node.node_id}.png", dpi=150)                                   
            plt.close(fig)                                                                            

    machine_types = list(nodes_by_type.keys())                                                        
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())
                                                                                                    
    fig, axes = plt.subplots(
        n_rows, n_cols,                                                                               
        figsize=(5 * n_cols, 3 * n_rows),
        sharex="col", sharey="row",
    )                                                                                                 
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.995)
                                                                                                    
    for col, machine_type in enumerate(machine_types):                                                
        nodes = nodes_by_type[machine_type]
        for row, node in enumerate(nodes):                                                            
            ax = axes[row, col] if n_rows > 1 else axes[col]
            _draw_node_axes(ax, node, config, compact=True)
                                                                                                    
    fig.supxlabel("Timestep", fontsize=11)
    fig.supylabel("Anomaly score", fontsize=11)                                                       
    fig.tight_layout()
    fig.savefig(plots_dir / "grid.png", dpi=100)
    plt.close(fig)                                                                                    

    state_enabled = config.get("simulation", {}).get("state_enabled", False)                          
    has_state = state_enabled and any(
        len(node.state_predictions) > 0                                                               
        for nodes in nodes_by_type.values()
        for node in nodes                                                                             
    )
    if has_state:                                                                                     
        print("  Rendering state grid...")
        _render_state_grid(nodes_by_type, config, plots_dir)