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

# from simulation.memory_accountant import MemoryAccountant   


VALID_SNRS = ("6dB", "0dB", "-6dB")    


@dataclass(frozen=True)
class NodeMetrics:
    auc:       float
    precision: float
    recall:    float
    f1:        float
    tp: int; tn: int; fp: int; fn: int

                                                              
@dataclass(frozen=True)                                                                
class BlockStateMetrics:                                                                         
    """Block-level metrics computed from a node's state_predictions.

    Fires-at-all-inside-block semantics:
    - block_tp: anomaly block where state was 1 at any point inside                            
    - block_fn: anomaly block where state stayed 0 throughout (missed)                         
    - block_fp: normal region where state was 1 at any point                                   
    - block_tn: normal region where state stayed 0 throughout                                  
                                                                                                
    mean_lag:    average clips-from-block-start to first fire, over detected blocks.             
    mean_unflag: average clips-from-block-end to first un-fire, over detected blocks.            
                None in manual_reset mode (the engineer resets, not the tracker).               
    """                                                                                          
    block_tp: int                                                                                
    block_fp: int                                                                                
    block_fn: int
    block_tn: int
    block_precision: float                                                                       
    block_recall:    float
    block_f1:        float                                                                       
    mean_lag:    float | None
    mean_unflag: float | None
    missed_blocks:  int
    total_blocks:   int 

                                                                                    
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

    # ── Manual-reset mode is a run-wide flag ────────────────────────────────────────────────────────             
    # Every node sees the same value, read once here and passed into 
    # each Node constructor. Default False so existing config without
    # this key keep working. See Node._compute_state for the semantics.
    manual_reset = config.get("simulation", {}).get("manual_reset", False)

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
                manual_reset=manual_reset
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
def _contiguous_runs(values: list[int], target: int) -> list[tuple[int, int]]:
    """Return [(start, end_inclusive), ...] for each contiguous run of 'target'.
    
    Example: 
        _contiguous_runs([0, 1, 1, 0, 1, 0], target=1)
        -> [(1, 2), (4, 4)]
    """
    runs: list[tuple[int, int]] = [] 
    start: int | None = None 
    for i, v in enumerate(values):
        if v == target and start is None: 
            start = i 
        elif v != target and start is not None: 
            runs.append((start, i - 1))
            start = None 
    if start is not None: 
        runs.append((start, len(values) - 1))
    return runs 


def _anomaly_block_outcomes(
    anomaly_blocks: list[tuple[int, int]],
    state: list[int],
    manual_reset: bool,
) -> tuple[int, int, list[int], list[int]]:
    """Returns (block_tp, block_fn, lags, unflags) over all anomaly blocks."""
    block_tp, block_fn, lags, unflags = 0, 0, [], []
    for start, end in anomaly_blocks:
        first_fire = next((i for i in range(start, end + 1) if state[i] == 1), None)
        if first_fire is None:
            block_fn += 1
            continue
        block_tp += 1
        lags.append(first_fire - start)
        if not manual_reset and end + 1 < len(state):
            unflag = next((j - (end + 1) for j in range(end + 1, len(state)) if state[j] == 0), None)
            if unflag is not None:
                unflags.append(unflag)
    return block_tp, block_fn, lags, unflags


def _normal_region_outcomes(
    normal_regions: list[tuple[int, int]],
    state: list[int],
) -> tuple[int, int]:
    """Returns (block_fp, block_tn) over all normal regions."""
    fired = [any(state[i] == 1 for i in range(start, end + 1)) for start, end in normal_regions]
    return sum(fired), sum(not f for f in fired)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Returns (precision, recall, f1)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _block_state_metrics(node: Node, manual_reset: bool) -> BlockStateMetrics | None:
    """Compute block-level metrics from node.state_predictions + node.labels."""
    state, labels = node.state_predictions, node.labels
    if not state or all(s is None for s in state):
        return None

    anomaly_blocks = _contiguous_runs(labels, target=1)
    if not anomaly_blocks:
        return None
    normal_regions = _contiguous_runs(labels, target=0)

    block_tp, block_fn, lags, unflags = _anomaly_block_outcomes(anomaly_blocks, state, manual_reset)
    block_fp, block_tn                = _normal_region_outcomes(normal_regions, state)
    precision, recall, f1             = _prf(block_tp, block_fp, block_fn)

    return BlockStateMetrics(
        block_tp=block_tp, block_fp=block_fp,
        block_fn=block_fn, block_tn=block_tn,
        block_precision=precision, block_recall=recall, block_f1=f1,
        mean_lag=float(np.mean(lags))    if lags    else None,
        mean_unflag=float(np.mean(unflags)) if unflags else None,
        missed_blocks=block_fn,
        total_blocks=block_tp + block_fn,
    )




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


def _format_state_row(                                                           
    node_id: str,
    m: BlockStateMetrics,
    manual_reset: bool,
) -> str:
    """Format the second per-node line showing block-level state metrics.

    Indented under the clip row so 'state' aligns with the AUC column above.
    """
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


def _format_state_mean_row(                                                      
    label: str,
    metrics: list[BlockStateMetrics],
    manual_reset: bool,
) -> str:
    """Format the per-type / overall state mean row."""
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



def _print_results(nodes_by_type: dict[str, list[Node]], state_enabled: bool = False, manual_reset: bool = False) -> None:
    """Compute and print AUC per node and per machine type."""
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    type_metrics: dict[str, list[NodeMetrics]] = defaultdict(list)
    type_state_metrics: dict[str, list[BlockStateMetrics]] = defaultdict(list)


    for machine_type, nodes in nodes_by_type.items():
        for node in nodes:
            m = _node_metrics(node)
            if m is None:
                print(f"  {node.node_id}: AUC = N/A (single class)")
                continue
            type_metrics[machine_type].append(m)
            print(_format_node_row(node.node_id, m))

            # ── Optional second row for state metrics  ────────────────────────────────────────────────────────
            if state_enabled:
                sm = _block_state_metrics(node, manual_reset)
                if sm is not None: 
                    type_state_metrics[machine_type].append(sm)
                    print(_format_state_row(node.node_id, sm, manual_reset))

    print("-" * 60)
    for machine_type, metrics in type_metrics.items():
        print(_format_mean_row(machine_type, metrics))
        # ── Optional per-type state mean row ──                                  
        if state_enabled and type_state_metrics.get(machine_type):
            print(_format_state_mean_row(
                machine_type, type_state_metrics[machine_type], manual_reset,
            ))

    all_metrics = [m for metrics in type_metrics.values() for m in metrics]

    if all_metrics:
        print("\n" + _format_mean_row("Overall", all_metrics))

    # ── Optional overall state mean row ──                                       
    if state_enabled:
        all_state_metrics = [
            m for metrics in type_state_metrics.values() for m in metrics
        ]
        if all_state_metrics:
            print(_format_state_mean_row("Overall", all_state_metrics, manual_reset))

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
    # accountant = MemoryAccountant()                                                                              
    # accountant.snapshot("system_init")  

                                                                                    
    with open(config_path) as f:
        config = yaml.safe_load(f)                                                    
                                                                                    
    sim = config["simulation"]
    data = config["data"]  

    state_enabled = config["simulation"].get("state_enabled", False)             
    manual_reset = config["simulation"].get("manual_reset", False)   

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
    # accountant.snapshot("embedder_loaded", frozen_embedder=frozen_embedder)                                      
                                                                                                                   
    nodes_by_type = _build_nodes(                                                                                
        config, preprocessor, frozen_embedder, topology, merge, timelines_by_type,                               
    )                                                                                                            
    # accountant.snapshot(
    #     "nodes_built_pre_warmup",
    #     nodes_by_type=nodes_by_type,                                                                             
    #     frozen_embedder=frozen_embedder,
    # )                                                                              
                
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
                                                                                    
    _print_results(
        nodes_by_type,
        state_enabled=state_enabled,
        manual_reset=manual_reset,
                )  

    # accountant.snapshot(
    #     "post_evaluation",                                                                                       
    #     nodes_by_type=nodes_by_type,
    #     frozen_embedder=frozen_embedder,
    # )   

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

    # accountant.save(run_dir) 
    
                     
    print(f"Done. Runtime: {runtime_seconds:.1f}s")                                                   
                
                                                                                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lockstep simulation")
    parser.add_argument(
        "--config", default="simulation/configs/default.yaml",
        help="Path to simulation config YAML",                                        
    )
    args = parser.parse_args()                                                        
    main(config_path=args.config)    