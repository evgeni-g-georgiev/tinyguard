"""Lockstep simulation loop — runs all nodes through time simultaneously.

At each timestep every node across every machine type processes one clip              
before the simulation advances. This enforces the temporal constraint
that would exist on real deployed devices: no node sees the future.                   
                
Three phases per timestep:
    1. Process  — each node: preprocess → embed → score (inside Node)
    2. Federate — share and merge state between neighbours (if enabled)               
    3. Learn    — update separator on new data (if enabled, future)                   
                                                                                    
The loop also handles calibration (warmup) before evaluation begins.                  
"""                                                                                   
                                                                                    
from dataclasses import dataclass, field
from typing import Iterator

from simulation.node.node import Node
from simulation.data.simulation_loader import NodeTimeline

                                                                                    
# ── Result types ─────────────────────────────────────────────────────────────
                                                                                    
@dataclass      
class NodeStepResult:
    """One node's output for a single timestep."""
    node_id: str
    machine_type: str
    score: float
    label: int
    predicted_label: int | None 


@dataclass
class TimestepResult:
    """All nodes' outputs for a single timestep."""
    timestep: int                                                                     
    node_results: list[NodeStepResult] = field(default_factory=list)
                                                                                    
                
# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate(                                                                        
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict[str, list[NodeTimeline]],                                 
) -> None:      
    """Fit every node's separator on its warmup clips.
                                                                                    
    Each node runs its warmup clips through the frozen pipeline
    (preprocessor → embedder) and passes the resulting embeddings                     
    to its separator for calibration.
    """                                                                               
    for machine_type, nodes in nodes_by_type.items():
        timelines = timelines_by_type[machine_type]                                   
                
        for node, timeline in zip(nodes, timelines):
            assert node.node_id == timeline.node_id, (
                f"Node/timeline mismatch: {node.node_id} vs {timeline.node_id}"       
            )
            print(f"  Calibrating {node.node_id} "                                    
                f"({len(timeline.warmup_paths)} clips)")
            node.warmup(timeline.warmup_paths)       
            threshold = getattr(node.separator, "threshold", None)
            if threshold is not None:
                print(f"    thrshold = {threshold:.4f}")                                 

                                                                                    
# ── Federation ───────────────────────────────────────────────────────────────

def _federate(nodes_by_type: dict[str, list[Node]]) -> None:                          
    """Share and merge separator state between neighbours within each type.
                                                                                    
    For each node, gathers shareable state from its neighbours (as
    defined by topology), then merges that state into its separator                   
    (as defined by the merge operator).                                               
    """                                                                               
    for machine_type, nodes in nodes_by_type.items():                                 
        # Build a lookup so we can find nodes by ID                                   
        node_lookup = {node.node_id: node for node in nodes}                          

        for node in nodes:                                                            
            neighbour_ids = node.get_neighbours()
            if not neighbour_ids:                                                     
                continue
                                                                                    
            neighbour_states = [                                                      
                node_lookup[nid].separator.get_shareable_state()
                for nid in neighbour_ids                                              
            ]   

            local_state = node.separator.get_shareable_state()                        
            merged = node.merge.merge(local_state, neighbour_states)
                                                                                    
            node.separator.merge_state([merged])
                                                                                    
                                                                                    
# ── Evaluation ───────────────────────────────────────────────────────────────
                                                                                    
def evaluate(   
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict[str, list[NodeTimeline]],
    federation_enabled: bool = False,
    federation_interval: int = 10,                                                    
) -> Iterator[TimestepResult]:
    """Lockstep evaluation — yield one TimestepResult per timestep.                   
                
    At each timestep t:                                                               
        1. Every node processes its t-th clip (preprocess → embed → score)
        2. If federation is enabled and t falls on the merge interval,                
            nodes share and merge state within their machine type                      
        3. Yield the collected results                                                
                                                                                    
    Future: step 2.5 — online separator update (train_step) can be                    
    added here when we implement online SVDD or contrastive learning.
    """                                                                               
    first_type = next(iter(timelines_by_type))
    n_timesteps = len(timelines_by_type[first_type][0].test_paths)                    
                                                                                    
    for t in range(n_timesteps):
        step = TimestepResult(timestep=t)                                             
                
        # Step 1: every node processes one clip                                       
        for machine_type, nodes in nodes_by_type.items():
            timelines = timelines_by_type[machine_type]                               
                
            for node, timeline in zip(nodes, timelines):     
                curr_label = timeline.test_labels[t]

                # ── Manual-reset at bloack boundaries ────────────────────────────────────────────────────────
                if node.manual_reset and node._prev_label == 1 and curr_label == 0:
                    node.reset_state()

                score, predicted = node.process_clip(
                    wav_path=timeline.test_paths[t],
                    label=timeline.test_labels[t],                                                  
                )       
                
                node._prev_label = curr_label

                step.node_results.append(                                                           
                    NodeStepResult(      
                        node_id=node.node_id,
                        machine_type=node.machine_type,
                        score=score,    
                        label=timeline.test_labels[t],
                        predicted_label=predicted,                                                  
                    )
                )   

        # Step 2: federation (if enabled and on interval)                             
        if federation_enabled and (t + 1) % federation_interval == 0:
            _federate(nodes_by_type)                                                  
                
        # Future Step 3: online separator update    - we  need to think about this more.                                   
        # for machine_type, nodes in nodes_by_type.items():
        #     for node in nodes:                                                      
        #         node.separator.train_step(latest_embedding)
                                                                                    
        yield step
                                                                                    
                
# ── Entry point ──────────────────────────────────────────────────────────────

def run(                                                                              
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict[str, list[NodeTimeline]],                                 
    federation_enabled: bool = False,
    federation_interval: int = 10,
) -> Iterator[TimestepResult]:
    """Run the full simulation: calibrate then evaluate.                              

    Calibration is blocking. Evaluation yields results one timestep                   
    at a time.  
    """
    print("Phase 1: Calibration")
    calibrate(nodes_by_type, timelines_by_type)
                                                                                    
    print("Phase 2: Evaluation")
    yield from evaluate(                                                              
        nodes_by_type,
        timelines_by_type,
        federation_enabled=federation_enabled,
        federation_interval=federation_interval,
    )
    