"""Orchestrator — reads YAML, builds nodes + groups, runs lockstep.                                                             
                                                                                                                                
No registries, no builders module.  Construction is a flat pass over config.                                                    
                                                                                                                                
Usage:          
    python -m simulation.run_simulation                                                                                         
    python -m simulation.run_simulation --config simulation/configs/default.yaml                                                
"""
                                                                                                                                
import argparse 
import time
from pathlib import Path
                                                                                                                                
import yaml
                                                                                                                                
from simulation                         import lockstep
from simulation.data.simulation_loader  import load_all_timelines
from simulation.formatters              import format_step, print_results                                                       
from simulation.node.node               import Node
from simulation.node.group              import Group                                                                            
from simulation.reporting               import (                                                                                
    make_run_dir, save_results, save_plots, save_latent_plots,
)                                                                                                                               
                                                                                                                                
                                                                                                                                
VALID_SNRS = ("6dB", "0dB", "-6dB")                                                                                             
                                                                                                                                
                                                                                                                                
# ── Config resolution ────────────────────────────────────────────────────
                                                                                                                                
def _resolve_snr(config: dict) -> str:
    """Validate snr and expand {snr} placeholders in data paths."""
    snr = config.get("snr")                                                                                                     
    if snr not in VALID_SNRS:
        raise ValueError(f"snr must be one of {VALID_SNRS}, got {snr!r}")                                                       
    data = config["data"]
    data["mimii_root"] = data["mimii_root"].format(snr=snr)                                                                     
    data["splits_dir"] = data["splits_dir"].format(snr=snr)
    return snr                                                                                                                  
                
                                                                                                                                
# ── Build nodes + groups ────────────────────────────────────────────────
# Replaces simulation/builders.py entirely.
                                                                                                                                
def build_nodes_and_groups(
    config: dict,                                                                                                               
) -> tuple[dict[str, list[Node]], dict[str, list[Group]]]:
    """Flat construction of all nodes and (if n_nodes > 1) their groups."""                                                     
    data    = config["data"]                                                                                                    
    gmm_cfg = config["gmm"]  
    sim_cfg = config["simulation"]                                                                                                   
    n_nodes = config["n_nodes"]                                                                                                 
    temp    = config["temperature"]      
    manual_reset = sim_cfg.get("manual_reset", False )                                                                                       
                                                                                                                                
    if not 1 <= n_nodes <= 8:                                                                                                   
        raise ValueError(f"n_nodes must be in 1..8, got {n_nodes}")                                                             
                                                                                                                                
    nodes_by_type:  dict[str, list[Node]]  = {mt: [] for mt in data["machine_types"]}                                           
    groups_by_type: dict[str, list[Group]] = {mt: [] for mt in data["machine_types"]}
                                                                                                                                
    for mtype in data["machine_types"]:
        for mid in data["machine_ids"]:
            machine_nodes = [                                                                                                   
                Node(
                    node_id       = f"{mtype}_{mid}_ch{ch}",                                                                    
                    machine_type  = mtype,                                                                                      
                    machine_id    = mid,
                    channel       = ch,                                                                                         
                    n_mels        = gmm_cfg["n_mels"],
                    n_components  = gmm_cfg["n_components"],                                                                    
                    threshold_pct = gmm_cfg["threshold_pct"],                                                                   
                    cusum_h_sigma = gmm_cfg["cusum_h_sigma"],                                                                   
                    cusum_h_floor = gmm_cfg["cusum_h_floor"],                                                                   
                    seed          = gmm_cfg["seed"], 
                    r_candidates  = gmm_cfg["r_candidates"],               
                    manual_reset  = manual_reset,                                                                                   
                )                                                                                                               
                for ch in range(n_nodes)                                                                                        
            ]   
            nodes_by_type[mtype].extend(machine_nodes)                                                                          

            if n_nodes > 1:                                                                                                     
                groups_by_type[mtype].append(Group(
                    machine_type  = mtype,
                    machine_id    = mid,
                    nodes         = machine_nodes,
                    temperature   = temp,
                    threshold_pct = gmm_cfg["threshold_pct"],                                                                   
                    cusum_h_sigma = gmm_cfg["cusum_h_sigma"],
                    cusum_h_floor = gmm_cfg["cusum_h_floor"],  
                    manual_reset  = manual_reset,                                                                 
                ))
                                                                                                                                
    return nodes_by_type, groups_by_type

                                                                                                                                
                                                                                                                            
# ── Orchestration ────────────────────────────────────────────────────────                                                  
                                                                                                                            
def run_with_config(
    config:         dict,                                                                                                    
    config_path:    Path | None = None,                                                                                      
    save_artefacts: bool        = True,
    verbose_steps:  bool        = True,                                                                                      
) -> dict:                                                                                                                   
    """Run one simulation from an in-memory config dict.
                                                                                                                            
    Parameters  
    ----------
    config : dict
        Resolved-or-not config; this function calls _resolve_snr() so paths                                                  
        with {snr} placeholders get expanded in place.  Idempotent.                                                          
    config_path : Path | None                                                                                                
        Original config path, used only by save_results for provenance.                                                      
    save_artefacts : bool                                                                                                    
        If False, skips run_dir, results.json, plots, latent plots.
    verbose_steps : bool                                                                                                     
        If False, suppresses the per-step format_step print and print_results.                                               
                                                                                                                            
    Returns                                                                                                                  
    -------                                                                                                                  
    dict with keys: nodes_by_type, groups_by_type, runtime_seconds, run_dir.
    """                                                                                                                      
    sim  = config["simulation"]
    data = config["data"]                                                                                                    
    snr  = _resolve_snr(config)
                                                                                                                            
    if verbose_steps:
        print(f"SNR:        {snr}")                                                                                          
        print(f"n_nodes:    {config['n_nodes']}  (channels {list(range(config['n_nodes']))})")                               
        print(f"GMM:        n_mels={config['gmm']['n_mels']}"                                                                
            f"  n_components={config['gmm']['n_components']}")                                                             
        print(f"Fusion T:   {config['temperature']}")                                                                        
        print(f"Shuffle:    {sim['shuffle_mode']}")                                                                          
        print(f"Warmup:     {sim['warmup_count']}")
        print()                                                                                                              
                
    run_dir = make_run_dir() if save_artefacts else None                                                                     
    if run_dir is not None:
        print(f"Run directory: {run_dir}\n")                                                                                 
                                                                                                                            
    start = time.time()
                                                                                                                            
    # Auto-split if splits dir is missing.                                                                                   
    splits_dir = Path(data["splits_dir"])
    if not splits_dir.exists():                                                                                              
        print(f"Splits dir missing: {splits_dir}")
        print(f"Running split_data for snr={snr}...")                                                                        
        from simulation.data.split_data import split_data                                                                    
        try:                                                                                                                 
            split_data(                                                                                                      
                mimii_root=Path(data["mimii_root"]),                                                                         
                splits_dir=splits_dir,                                                                                       
                machine_types=data["machine_types"],
            )                                                                                                                
        except FileNotFoundError as e:
            raise FileNotFoundError(                                                                                         
                f"Cannot auto-split: {e}\n"
                f"Raw MIMII data for snr={snr} is missing. "                                                                 
                f"Run: python data/download_mimii.py --snr {snr}"                                                            
            ) from e                                                                                                         
                                                                                                                            
    timelines_by_type = load_all_timelines(                                                                                  
        splits_dir     = splits_dir,
        machine_types  = data["machine_types"],                                                                              
        machine_ids    = data["machine_ids"],
        warmup_count   = sim["warmup_count"],                                                                                
        shuffle_mode   = sim["shuffle_mode"],                                                                                
        seed           = sim["seed"],
        block_size     = sim.get("block_size", 5),                                                                           
        block_interval = sim.get("block_interval", 20),
    )                                                                                                                        
                
    nodes_by_type, groups_by_type = build_nodes_and_groups(config)                                                           
                
    if verbose_steps:                                                                                                        
        print("Legend:  · TN   ✓ TP   ✗ FN (missed)   ! FP (false alarm)\n")
                                                                                                                            
    for step in lockstep.run(                                                                                                
        nodes_by_type, groups_by_type, timelines_by_type,                                                                    
        n_fit_clips=config["gmm"]["n_fit_clips"],                                                                            
        n_val_clips=config["gmm"]["n_val_clips"],                                                                            
    ):
        if verbose_steps:                                                                                                    
            print(format_step(step, data["machine_types"]))
                                                                                                                            
    if verbose_steps:                                                                                                        
        print_results(nodes_by_type, groups_by_type)                                                                         
                                                                                                                            
    runtime = time.time() - start

    if save_artefacts:                                                                                                       
        print(f"\nSaving run artefacts to {run_dir}")
        save_results(                                                                                                        
            nodes_by_type   = nodes_by_type,                                                                                 
            groups_by_type  = groups_by_type,
            config          = config,                                                                                        
            config_path     = config_path,                                                                                   
            runtime_seconds = runtime,
            run_dir         = run_dir,                                                                                       
        )       
        save_plots(
            nodes_by_type   = nodes_by_type,                                                                                 
            groups_by_type  = groups_by_type,
            config          = config,                                                                                        
            run_dir         = run_dir,
        )                                                                                                                    
        if config.get("latent_plot", {}).get("enabled", False):
            save_latent_plots(                                                                                               
                nodes_by_type     = nodes_by_type,
                timelines_by_type = timelines_by_type,                                                                       
                config            = config,
                run_dir           = run_dir,                                                                                 
            )
        print(f"Done.  Runtime: {runtime:.1f}s")                                                                             
                                                                                                                            
    return {
        "nodes_by_type":   nodes_by_type,                                                                                    
        "groups_by_type":  groups_by_type,
        "runtime_seconds": runtime,
        "run_dir":         run_dir,
    }                                                                                                                        

                                                                                                                            
# ── Main ─────────────────────────────────────────────────────────────────

def main(config_path: str = "simulation/configs/default.yaml") -> None:                                                      
    with open(config_path) as f:
        config = yaml.safe_load(f)                                                                                           
    print(f"Config:     {config_path}")                                                                                      
    run_with_config(
        config,                                                                                                              
        config_path    = Path(config_path),
        save_artefacts = True,                                                                                               
        verbose_steps  = True,
    )
                                                                                                                                
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run lockstep simulation")                                                          
    p.add_argument(
        "--config",                                                                                                             
        default="simulation/configs/default.yaml",
        help="Path to simulation config YAML",                                                                                  
    )           
    args = p.parse_args()
    main(config_path=args.config)         