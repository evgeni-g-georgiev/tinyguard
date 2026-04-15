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

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score  
import time      
from collections import defaultdict
from dataclasses import dataclass                                       

from simulation.registry import (                                                     
    create_embedder,
    create_separator,                                                                 
    create_merge,
    create_topology,                                                                  
)               
from simulation.node.node import Node
from simulation.data.simulation_loader import load_all_timelines
from simulation import lockstep  
from simulation.lockstep import TimestepResult                                                    
from simulation.reporting import make_run_dir, save_results, save_plots, save_latent_plots

from simulation.memory_accountant import MemoryAccountant   


VALID_SNRS = ("6dB", "0dB", "-6dB")    


@dataclass(frozen=True)
class NodeMetrics:
    auc:       float
    precision: float
    recall:    float
    f1:        float
    tp: int; tn: int; fp: int; fn: int

                                                                                    
# ── Component construction ───────────────────────────────────────────────────       
                                                                                    
def _create_from_config(create_fn, config: dict, selector_key: str):
    """Look up a selector and instantiate its named block.
                                                                                    
    Args:
        create_fn: Registry create function (e.g. create_separator).                  
        config: Full config dict.
        selector_key: Top-level key naming the selector
                    (e.g. "frozen_embedder", "topology").
                                                                                    
    Returns:
        Instantiated component with the selector's params applied.                    
    """         
    name = config[selector_key]
    kwargs = config.get(name) or {}                                                   
    return create_fn(name, **kwargs)
                                                                                    
                                                                                    
def _build_preprocessor(config: dict):
    """Construct a preprocessor based on the active selector.

    Returns an object with a .process(wav_path) -> np.ndarray method.

    Audio params for log_mel come from the active embedder's config block,
    because the spectrogram shape must match what the embedder was trained
    with. The twfr and identity preprocessors don't need that link — twfr
    reads its constants from config.py (matching the team's gmm/features.py),
    and identity is for pre-extracted data.
    """
    name = config["preprocessor"]

    if name == "log_mel":
        from preprocessing.loader import load_audio, split_into_chunks
        from preprocessing.separator_input import load_clip_log_mels

        embedder_name = config["frozen_embedder"]
        embedder_block = config.get(embedder_name) or {}

        if "sample_rate" not in embedder_block or "frame_seconds" not in embedder_block:
            raise ValueError(
                f"log_mel preprocessor needs sample_rate and frame_seconds in "
                f"the active embedder's config block ('{embedder_name}'), "
                f"but they're missing. The log_mel preprocessor only makes "
                f"sense paired with an embedder that declares these params."
            )

        class LogMelPreprocessor:
            def __init__(self, audio_config: dict):
                self.sample_rate = audio_config["sample_rate"]
                self.frame_seconds = audio_config["frame_seconds"]

            def process(self, wav_path: str) -> np.ndarray:
                return load_clip_log_mels(wav_path)

        return LogMelPreprocessor(embedder_block)

    elif name == "twfr":
        from gmm.features import load_log_mel

        class TWFRPreprocessor:
            def process(self, wav_path: str) -> np.ndarray:
                return load_log_mel(wav_path)

        return TWFRPreprocessor()

    elif name == "identity":
        class IdentityPreprocessor:
            def process(self, wav_path: str) -> np.ndarray:
                raise NotImplementedError(
                    "Identity preprocessor requires pre-extracted data"
                )

        return IdentityPreprocessor()

    else:
        raise ValueError(f"Unknown preprocessor: '{name}'")
                                                                                    

def _build_shared_components(config: dict):                                           
    """Construct components shared across all nodes.
                                                                                    
    Returns:
        (preprocessor, frozen_embedder, topology, merge)                              
    """         
    preprocessor = _build_preprocessor(config)
                                                                                    
    frozen_embedder = _create_from_config(
        create_embedder, config, "frozen_embedder",                                   
    )           
    embedder_block = config.get(config["frozen_embedder"]) or {}     
                                                                                    
    topology = _create_from_config(create_topology, config, "topology")               
    merge = _create_from_config(create_merge, config, "merge")
                                                                                    
    return preprocessor, frozen_embedder, topology, merge

                                                                                    
def _build_nodes(
    config: dict,                                                                     
    preprocessor,
    frozen_embedder,
    topology,
    merge,
    timelines_by_type: dict,
) -> dict[str, list[Node]]:
    """Construct all 16 nodes.                                                        

    Shared components are passed in. Each node gets a fresh separator                 
    instance that owns its own state. The separator's input_dim is
    auto-wired from the active embedder's output dimension.                           
    """                                                                               
    separator_name = config["separator"]                                              
    separator_kwargs = (config.get(separator_name) or {}).copy()                      
                                                                                    
    # Auto-wire input_dim from the embedder's latent dimension
    if hasattr(frozen_embedder, "embedding_dim"):                                     
        separator_kwargs["input_dim"] = frozen_embedder.embedding_dim                 

    nodes_by_type: dict[str, list[Node]] = {}                                         
                
    for machine_type, timelines in timelines_by_type.items():                         
        nodes = []
        for timeline in timelines:                                                    
            separator = create_separator(separator_name, **separator_kwargs)
            nodes.append(Node(
                node_id=timeline.node_id,                                             
                machine_type=timeline.machine_type,
                machine_id=timeline.machine_id,                                       
                preprocessor=preprocessor,
                frozen_embedder=frozen_embedder,                                      
                separator=separator,
                topology=topology,                                                    
                merge=merge,
            ))
        nodes_by_type[machine_type] = nodes
                                                                                    
    return nodes_by_type



# ── Pre-timestep formatting ────────────────────────────────────────────────────────  
#                                                                              
# Symbols for the per-node status grid:                                                             
#   ·  true negative   (normal predicted normal)
#   ✓  true positive   (anomaly predicted anomaly — caught it)                                      
#   ✗  false negative  (anomaly predicted normal  — missed)   
#   !  false positive  (normal predicted anomaly  — false alarm)                                    
#   ?  no prediction   (separator has no threshold)        

def _status_symbol(label: int, prediction: int | None) -> str:                                      
    if prediction is None:                                                                          
        return "?"                                                                                  
    if label == 1 and prediction == 1:                                                              
        return "✓"                                                                                  
    if label == 1 and prediction == 0:                                                              
        return "✗"                                                                                  
    if label == 0 and prediction == 1:
        return "!"                                                                                  
    return "·"    
                                                                                                    
                                            
def _format_step(result: TimestepResult, machine_types: list[str]) -> str:
    """Build a one-line summary of all 16 nodes for one timestep."""      
    by_type: dict[str, list[str]] = {mt: [] for mt in machine_types}                                
    n_anom = 0                                                                                      
    n_correct = 0                                                                                   
    n_with_pred = 0                                                                                 
                                                                                                    
    for r in result.node_results:                                                                   
        by_type[r.machine_type].append(_status_symbol(r.label, r.predicted_label))                  
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

                
# ── Results ──────────────────────────────────────────────────────────────────

def _node_metrics(node: Node) -> NodeMetrics | None:
    """Return metrics for a node, or None if single-class."""
    if len(set(node.labels)) < 2:
        return None

    auc = roc_auc_score(node.labels, node.scores)
    tp = sum(l == 1 and p == 1 for l, p in zip(node.labels, node.predictions))
    tn = sum(l == 0 and p == 0 for l, p in zip(node.labels, node.predictions))
    fp = sum(l == 0 and p == 1 for l, p in zip(node.labels, node.predictions))
    fn = sum(l == 1 and p == 0 for l, p in zip(node.labels, node.predictions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return NodeMetrics(auc=auc, precision=precision, recall=recall, f1=f1,
                       tp=tp, tn=tn, fp=fp, fn=fn)


def _format_node_row(node_id: str, m: NodeMetrics) -> str:
    return (f"  {node_id}: AUC={m.auc:.4f}  "
            f"|  TP={m.tp:3d} TN={m.tn:3d} FP={m.fp:3d} FN={m.fn:3d}  "
            f"|  P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")


def _format_mean_row(label: str, metrics: list[NodeMetrics]) -> str:
    return (f"  {label:12s} mean  "
            f"AUC={np.mean([m.auc       for m in metrics]):.4f}  "
            f"P={np.mean([m.precision   for m in metrics]):.3f}  "
            f"R={np.mean([m.recall      for m in metrics]):.3f}  "
            f"F1={np.mean([m.f1         for m in metrics]):.3f}")


def _print_results(nodes_by_type: dict[str, list[Node]]) -> None:
    """Compute and print AUC per node and per machine type."""
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    type_metrics: dict[str, list[NodeMetrics]] = defaultdict(list)

    for machine_type, nodes in nodes_by_type.items():
        for node in nodes:
            m = _node_metrics(node)
            if m is None:
                print(f"  {node.node_id}: AUC = N/A (single class)")
                continue
            type_metrics[machine_type].append(m)
            print(_format_node_row(node.node_id, m))

    print("-" * 60)
    for machine_type, metrics in type_metrics.items():
        print(_format_mean_row(machine_type, metrics))

    all_metrics = [m for metrics in type_metrics.values() for m in metrics]
    if all_metrics:
        print("\n" + _format_mean_row("Overall", all_metrics))

    print("=" * 60)


# ── SNR resolution ────────────────────────────────────────────────────────

def _resolve_snr(config: dict) -> str:                                          
    """Validate the top-level snr key and resolve {snr} placeholders.
                                                                                                    
    Mutates config["data"]["mimii_root"] and config["data"]["splits_dir"]
    in-place, replacing any {snr} token with the chosen SNR. Returns the                              
    resolved SNR string.                                                                              
    """                                                                                               
    snr = config.get("snr")                                                                           
    if snr not in VALID_SNRS:                                                                         
        raise ValueError(
            f"Top-level config key 'snr' must be one of {VALID_SNRS}, got {snr!r}"
        )                                                                                             

    data = config["data"]                                                                             
    data["mimii_root"] = data["mimii_root"].format(snr=snr)
    data["splits_dir"] = data["splits_dir"].format(snr=snr)                                           
    return snr                                                                                        

                                                                                    
# ── Main ─────────────────────────────────────────────────────────────────────

def main(config_path: str = "simulation/configs/default.yaml"):                       
    """Load config, build components, run simulation, print results."""

    # Memory allocation 
    accountant = MemoryAccountant()                                                                              
    accountant.snapshot("system_init")  
                                                                                    
    with open(config_path) as f:
        config = yaml.safe_load(f)                                                    
                                                                                    
    sim = config["simulation"]
    data = config["data"]  

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

    # ── Auto-run split_data if splits dir for this SNR doesnt exits ──────────────────────────]
    splits_dir = Path(data["splits_dir"])
    if not splits_dir.exists():
        print(f"\nSplits dir missing: {splits_dir}")
        print(f"Running spli_data for snr={snr}...")
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
    preprocessor, frozen_embedder, topology, merge = _build_shared_components(config)
    accountant.snapshot("embedder_loaded", frozen_embedder=frozen_embedder)                                      
                                                                                                                   
    nodes_by_type = _build_nodes(                                                                                
        config, preprocessor, frozen_embedder, topology, merge, timelines_by_type,                               
    )                                                                                                            
    accountant.snapshot(
        "nodes_built_pre_warmup",
        nodes_by_type=nodes_by_type,                                                                             
        frozen_embedder=frozen_embedder,
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
        n_anomalies = sum(1 for r in result.node_results if r.label == 1)
        print(_format_step(result, data["machine_types"]))                                       
                                                                                    
    _print_results(nodes_by_type)  

    accountant.snapshot(
        "post_evaluation",                                                                                       
        nodes_by_type=nodes_by_type,
        frozen_embedder=frozen_embedder,
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

    accountant.save(run_dir) 
    
                     
    print(f"Done. Runtime: {runtime_seconds:.1f}s")                                                   
                
                                                                                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lockstep simulation")
    parser.add_argument(
        "--config", default="simulation/configs/default.yaml",
        help="Path to simulation config YAML",                                        
    )
    args = parser.parse_args()                                                        
    main(config_path=args.config)    