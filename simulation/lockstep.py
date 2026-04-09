"""Lockstep simulation loop — runs all nodes through time simultaneously.             

At each timestep every node across every machine type processes one clip              
before the simulation advances. This enforces the temporal constraint
that would exist on real deployed devices: no node sees the future.                   
                                                                                    
Two phases:
    1. Calibration — each node fits its separator on warmup clips.                    
    2. Evaluation  — one clip per node per timestep, scored and recorded.
                                                                                    
The evaluation phase is a generator: it yields a TimestepResult after
each timestep, giving the caller live access to scores and labels.                    
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
    score: float
    label: int                                                                        
                

@dataclass
class TimestepResult:
    """All nodes' outputs for a single timestep."""
    timestep: int
    node_results: list[NodeStepResult] = field(default_factory=list)       


# ── Phases ───────────────────────────────────────────────────────────────────

def calibrate(                                                                        
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict[str, list[NodeTimeline]],                                 
) -> None:      
    """Phase 1: fit every node's separator on its warmup clips.
                                                                                    
    Each node runs its warmup clips through the frozen pipeline and
    passes the resulting embeddings to its separator for calibration.                 
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

                                                                                    
def evaluate(
    nodes_by_type: dict[str, list[Node]],                                             
    timelines_by_type: dict[str, list[NodeTimeline]],
) -> Iterator[TimestepResult]:
    """Phase 2: lockstep evaluation — yield one TimestepResult per timestep.
                                                                                    
    At each timestep t, every node processes its t-th test clip before
    the simulation advances to t+1.                                                   
    """         
    first_type = next(iter(timelines_by_type))
    n_timesteps = len(timelines_by_type[first_type][0].test_paths)                    

    for t in range(n_timesteps):                                                      
        step = TimestepResult(timestep=t)

        for machine_type, nodes in nodes_by_type.items():                             
            timelines = timelines_by_type[machine_type]
                                                                                    
            for node, timeline in zip(nodes, timelines):
                score = node.process_clip(
                    wav_path=timeline.test_paths[t],
                    label=timeline.test_labels[t],                                    
                )
                step.node_results.append(                                             
                    NodeStepResult(
                        node_id=node.node_id,
                        score=score,
                        label=timeline.test_labels[t],
                    )                                                                 
                )
                                                                                    
        yield step

# ── Main ──────────────────────────────────────────────────────────────

def run(                                                                              
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict[str, list[NodeTimeline]],                                 
) -> Iterator[TimestepResult]:
    """Run the full simulation: calibrate then evaluate.

    Calibration is blocking. Evaluation yields results one timestep                   
    at a time.
    """                                                                               
    print("Phase 1: Calibration")
    calibrate(nodes_by_type, timelines_by_type)
                                                                                    
    print("Phase 2: Evaluation")
    yield from evaluate(nodes_by_type, timelines_by_type)    