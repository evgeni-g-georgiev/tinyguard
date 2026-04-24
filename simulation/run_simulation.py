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

# Maps display SNR ("-6dB") to on-disk mimii dir suffix ("neg6db"). The splits
# dir keeps the display form (e.g. simulation/data/splits/-6dB/).
_MIMII_SNR_DIR = {"-6dB": "neg6db", "0dB": "0db", "6dB": "6db"}


# ── Config resolution ────────────────────────────────────────────────────

def _resolve_snr(config: dict) -> str:
    """Validate snr and expand {snr} placeholders in data paths."""
    snr = config.get("snr")
    if snr not in VALID_SNRS:
        raise ValueError(f"snr must be one of {VALID_SNRS}, got {snr!r}")
    data = config["data"]
    data["mimii_root"] = data["mimii_root"].format(snr=_MIMII_SNR_DIR[snr])
    data["splits_dir"] = data["splits_dir"].format(snr=snr)
    return snr


def _resolve_channels(config: dict) -> list[int]:
    """Read explicit `channels` list, or fall back to legacy `n_nodes`."""
    channels = config.get("channels")
    if channels is None:
        n_nodes = config.get("n_nodes")
        if n_nodes is None:
            raise ValueError("config must provide either 'channels' or 'n_nodes'")
        channels = list(range(n_nodes))

    if not isinstance(channels, list) or not channels:
        raise ValueError(f"channels must be a non-empty list, got {channels!r}")
    if len(channels) > 8:
        raise ValueError(f"channels may have at most 8 entries, got {len(channels)}")
    if any((not isinstance(c, int)) or c < 0 or c > 7 for c in channels):
        raise ValueError(f"every channel must be an int in 0..7, got {channels!r}")
    if len(set(channels)) != len(channels):
        raise ValueError(f"channels must be unique, got {channels!r}")
    return channels                                                                                                                  
                
                                                                                                                                
# ── Build nodes + groups ────────────────────────────────────────────────
# Replaces simulation/builders.py entirely.
                                                                                                                                
def build_nodes_and_groups(
    config: dict,
) -> tuple[dict[str, list[Node]], dict[str, list[Group]]]:
    """Flat construction of all nodes and (if len(channels) > 1) their groups."""
    data     = config["data"]
    gmm_cfg  = config["gmm"]
    channels = _resolve_channels(config)
    temp     = config["temperature"]

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
                )
                for ch in channels
            ]
            nodes_by_type[mtype].extend(machine_nodes)

            if len(channels) > 1:
                groups_by_type[mtype].append(Group(
                    machine_type  = mtype,
                    machine_id    = mid,
                    nodes         = machine_nodes,
                    temperature   = temp,
                    threshold_pct = gmm_cfg["threshold_pct"],
                    cusum_h_sigma = gmm_cfg["cusum_h_sigma"],
                    cusum_h_floor = gmm_cfg["cusum_h_floor"],
                ))

    return nodes_by_type, groups_by_type

                                                                                                                                
# ── Main ─────────────────────────────────────────────────────────────────
                                                                                                                                
def main(config_path: str = "simulation/configs/default.yaml") -> None:                                                         
    with open(config_path) as f:
        config = yaml.safe_load(f)                                                                                              
                
    sim      = config["simulation"]
    data     = config["data"]
    snr      = _resolve_snr(config)
    channels = _resolve_channels(config)
    config["channels"] = channels   # normalise so downstream sees a canonical list

    print(f"Config:     {config_path}")
    print(f"SNR:        {snr}")
    print(f"Channels:   {channels}  (n={len(channels)})")
    print(f"GMM:        n_mels={config['gmm']['n_mels']}"
        f"  n_components={config['gmm']['n_components']}")
    print(f"Fusion T:   {config['temperature']}")
    print(f"Shuffle:    {sim['shuffle_mode']}")
    print(f"Warmup:     {sim['warmup_count']}")
    print()                                                                                                                     
                
    run_dir = make_run_dir()                                                                                                    
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
                                                                                                                                
    # Load shuffled timelines, one per (mtype, mid).                                                                            
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
                                                                                                                                
    # Build nodes + groups.
    nodes_by_type, groups_by_type = build_nodes_and_groups(config)

    # Run lockstep.
    print("Legend:  · TN   ✓ TP   ✗ FN (missed)   ! FP (false alarm)\n")
    for step in lockstep.run(nodes_by_type, groups_by_type, timelines_by_type):                                                 
        print(format_step(step, data["machine_types"]))                                                                         
                                                                                                                                
    # Print summary tables.                                                                                                     
    print_results(nodes_by_type, groups_by_type)
                                                                                                                                
    runtime = time.time() - start                                                                                               

    # Save artefacts.                                                                                                           
    print(f"\nSaving run artefacts to {run_dir}")
    save_results(                                                                                                               
        nodes_by_type   = nodes_by_type,
        groups_by_type  = groups_by_type,                                                                                       
        config          = config,                                                                                               
        config_path     = Path(config_path),
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

                                                                                                                                
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run lockstep simulation")                                                          
    p.add_argument(
        "--config",                                                                                                             
        default="simulation/configs/default.yaml",
        help="Path to simulation config YAML",                                                                                  
    )           
    args = p.parse_args()
    main(config_path=args.config)         