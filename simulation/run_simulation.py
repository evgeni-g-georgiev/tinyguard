                                                                                                    
"""Orchestrator — wire up components from config and run the simulation.
                                                                                                    
This is the only file that reads default.yaml. It constructs all components,
passes them into Nodes, and runs the lockstep loop. Everything downstream                             
receives fully configured objects via dependency injection.                                           
                                                                                                    
Usage:                                                                                                
    python -m simulation.run_simulation                                                               
    python -m simulation.run_simulation --config simulation/configs/experiment.yaml
"""

import argparse
from pathlib import Path

import yaml                                                                                           
import time
                                                                                                    
from simulation.data.simulation_loader import load_all_timelines
from simulation import lockstep
from simulation.builders import build_shared_components, build_nodes                                  
from simulation.formatters import format_step, print_results
from simulation.reporting import make_run_dir, save_results, save_plots, save_latent_plots            
                                                                                                    
                                                                                                    
VALID_SNRS = ("6dB", "0dB", "-6dB")                                                                   
                                                                                                    
                
def _resolve_snr(config: dict) -> str:
    """Validate the top-level snr key and resolve {snr} placeholders."""
    snr = config.get("snr")                                                                           
    if snr not in VALID_SNRS:
        raise ValueError(                                                                             
            f"Top-level config key 'snr' must be one of {VALID_SNRS}, got {snr!r}"
        )                                                                                             
    data = config["data"]
    data["mimii_root"] = data["mimii_root"].format(snr=snr)                                           
    data["splits_dir"] = data["splits_dir"].format(snr=snr)                                           
    return snr
                                                                                                    
                                                                                                    
def main(config_path: str = "simulation/configs/default.yaml"):
    """Load config, build components, run simulation, print results."""                               
                
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sim = config["simulation"]                                                                        
    data = config["data"]
                                                                                                    
    state_enabled = sim.get("state_enabled", False)
    manual_reset = sim.get("manual_reset", False)
    snr = _resolve_snr(config)
                                                                                                    
    print(f"Config: {config_path}")
    print(f"SNR:             {snr}")                                                                  
    print(f"Preprocessor:    {config['preprocessor']}")
    print(f"Frozen embedder: {config['frozen_embedder']}")                                            
    print(f"Separator:       {config['separator']}")
    print(f"Topology:        {config['topology']}")                                                   
    print(f"Merge:           {config['merge']}")
    print(f"Shuffle:         {sim['shuffle_mode']}")                                                  
    print(f"Warmup count:    {sim['warmup_count']}")
    print()                                                                                           
                                                                                                    
    run_dir = make_run_dir()
    print(f"Run directory: {run_dir}")                                                                
    print()     

    start_time = time.time()

    # Auto-run split_data if splits dir for this SNR doesn't exist                                    
    splits_dir = Path(data["splits_dir"])
    if not splits_dir.exists():                                                                       
        print(f"\nSplits dir missing: {splits_dir}")                                                  
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
                                                                                                    
    # Load shuffled timelines from the split directories
    timelines_by_type = load_all_timelines(                                                           
        splits_dir=splits_dir,
        machine_types=data["machine_types"],
        machine_ids=data["machine_ids"],
        warmup_count=sim["warmup_count"],                                                             
        shuffle_mode=sim["shuffle_mode"],
        seed=sim["seed"],                                                                             
        block_size=sim.get("block_size", 5),                                                          
        block_interval=sim.get("block_interval", 20),
    )                                                                                                 
                                                                                                    
    # Build components and nodes
    preprocessor, frozen_embedder, topology, merge = build_shared_components(config)                  
    nodes_by_type = build_nodes(
        config, preprocessor, frozen_embedder, topology, merge, timelines_by_type,
    )                                                                                                 

    # Run lockstep                                                                                    
    fed = config.get("federation", {})

    print()
    print("Legend:  · TN   ✓ TP   ✗ FN (missed)   ! FP (false alarm)")
    print()                                                                                           

    for result in lockstep.run(                                                                       
        nodes_by_type,
        timelines_by_type,                                                                            
        federation_enabled=fed.get("enabled", False),
        federation_interval=fed.get("interval", 10),
    ):                                                                                                
        print(format_step(result, data["machine_types"]))
                                                                                                    
    print_results(
        nodes_by_type,
        state_enabled=state_enabled,
        manual_reset=manual_reset,                                                                    
    )
                                                                                                    
    runtime_seconds = time.time() - start_time

    print(f"\nSaving run artefacts to {run_dir}")                                                     
    save_results(
        nodes_by_type=nodes_by_type,                                                                  
        config=config,
        config_path=Path(config_path),                                                                
        runtime_seconds=runtime_seconds,
        run_dir=run_dir,                                                                              
    )           
    save_plots(                                                                                       
        nodes_by_type=nodes_by_type,
        config=config,                                                                                
        run_dir=run_dir,
    )
    save_latent_plots(
        nodes_by_type=nodes_by_type,
        timelines_by_type=timelines_by_type,                                                          
        config=config,
        run_dir=run_dir,                                                                              
    )           

    print(f"Done. Runtime: {runtime_seconds:.1f}s")

                                                                                                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lockstep simulation")                           
    parser.add_argument(
        "--config", default="simulation/configs/default.yaml",
        help="Path to simulation config YAML",
    )                                                                                                 
    args = parser.parse_args()
    main(config_path=args.config)