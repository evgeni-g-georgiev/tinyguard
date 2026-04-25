"""Lockstep loop — every node advances one timestep per iteration.
                                                                                                                                
Three phases per run:
1. Calibration       — every node runs r-search + GMM fit independently.                                                      
2. Fusion setup      — for each group (n_nodes > 1), compute weights + CUSUM.                                                 
3. Evaluation        — for each timestep, every node scores one clip, then                                                    
                        every group fuses the per-node scores.                                                                 
                                                                                                                                
No federation / periodic merge. Fusion is calibration-time only.                                                                
"""                                                                                                                             

from dataclasses import dataclass, field                                                                                        
from typing import Iterator

                                                                                           
from simulation.node.node  import Node                                                                                          
from simulation.node.group import Group
from simulation.data.simulation_loader import NodeTimeline
                                                                                                                                
                                                                                                                                
# ── Result types ─────────────────────────────────────────────────────────                                                     
                                                                                                                                
@dataclass      
class NodeStepResult:
    node_id:      str
    machine_type: str
    score:        float
    label:        int
    alarm:        bool
    cusum_S:      float                                                                                                         

                                                                                                                                
@dataclass      
class GroupStepResult:
    group_id:     str
    machine_type: str
    fused_z:      float
    label:        int
    alarm:        bool
    cusum_S:      float                                                                                                         

                                                                                                                                
@dataclass      
class TimestepResult:
    timestep:      int
    node_results:  list[NodeStepResult]  = field(default_factory=list)
    group_results: list[GroupStepResult] = field(default_factory=list)                                                          

                                                                                                                                
# ── Calibration ──────────────────────────────────────────────────────────
                                                                                                                                
def calibrate(  
    nodes_by_type:     dict[str, list[Node]],
    groups_by_type:    dict[str, list[Group]],                                                                                  
    timelines_by_type: dict[str, list[NodeTimeline]],
    n_fit_clips: int,
    n_val_clips: int,
) -> None:                                                                                                                      
    """Phase 1: every node calibrates.  Phase 2: every group sets weights."""
    # Lookup: (mtype, mid) → timeline (all nodes for one machine share it).                                                     
    timeline_lookup = _timeline_lookup(timelines_by_type)                                                                       
                                                                                                                                
    # Phase 1 — per-node r-search + GMM fit.                                                                                    
    for mtype, nodes in nodes_by_type.items():                                                                                  
        for node in nodes:                                                                                                      
            tl = timeline_lookup[(mtype, node.machine_id)]            
            fit_paths = tl.warmup_paths[:n_fit_clips]                 
            val_paths = tl.warmup_paths[n_fit_clips:n_fit_clips + n_val_clips]                                                                           
            print(f"  Calibrating {node.node_id}  (mic {node.channel})")                                                        
            node.calibrate(fit_paths, val_paths)                                                                                
            print(f"    r={node.r:.2f}  k={node.k:.3f}  h={node.h:.3f}"                                                         
                f"  μ_val={node.mu_val:.3f}")                                                                                 
                                                                                                                                
    # Phase 2 — per-group fusion weights.                                                                                       
    for mtype, groups in groups_by_type.items():                                                                                
        for g in groups:                                                                                                        
            g.finalise_fusion()
            print(f"  Group {g.group_id}  w={g.w.round(3).tolist()}"                                                            
                f"  k={g.k:.3f}  h={g.h:.3f}")                                                                                

                                                                                                                                
# ── Evaluation ───────────────────────────────────────────────────────────
                                                                                                                                
def evaluate(   
    nodes_by_type:     dict[str, list[Node]],
    groups_by_type:    dict[str, list[Group]],
    timelines_by_type: dict[str, list[NodeTimeline]],                                                                           
) -> Iterator[TimestepResult]:
    """Lockstep — yield one TimestepResult per timestep."""    


    # (mtype, mid) → [Node, ...] for band-boundary state resets       
    nodes_by_machine: dict[tuple[str, str], list[Node]] = {}          
    for mtype, nodes in nodes_by_type.items():                        
        for n in nodes:
            nodes_by_machine.setdefault((mtype, n.machine_id),[]).append(n)                                                         

    # (mtype, mid) → Group for same                                   
    group_by_machine = {
        (g.machine_type, g.machine_id): g                             
        for groups in groups_by_type.values() for g in groups
    }                                                                 



    timeline_lookup = _timeline_lookup(timelines_by_type)                                                                       
                                                                                                                                
    first_type  = next(iter(timelines_by_type))                                                                                 
    n_timesteps = len(timelines_by_type[first_type][0].test_paths)                                                              
                                                                                                                                
    for t in range(n_timesteps):
        step = TimestepResult(timestep=t)  

        # Band boundary: anomaly → normal transition triggers state_reset.
        if t > 0:                                                     
            for (mtype, mid), tl in timeline_lookup.items():
                if tl.test_labels[t - 1] == 1 and tl.test_labels[t] == 0:                                                                   
                    for node in nodes_by_machine[(mtype, mid)]:
                        node.state_reset()                            
                    if (mtype, mid) in group_by_machine:              
                        group_by_machine[(mtype, mid)].state_reset()                                                                                     
                
        # Phase A — every node scores its timestep.                                                                             
        for mtype, nodes in nodes_by_type.items():
            for node in nodes:                                                                                                  
                tl    = timeline_lookup[(mtype, node.machine_id)]
                label = tl.test_labels[t]                                                                                       
                nll, alarm = node.score(tl.test_paths[t], label=label)
                step.node_results.append(NodeStepResult(                                                                        
                    node_id      = node.node_id,                                                                                
                    machine_type = mtype,
                    score        = nll,                                                                                         
                    label        = label,
                    alarm        = alarm,
                    cusum_S      = node.cusum_S[-1],                                                                            
                ))
                                                                                                                                
        # Phase B — every group fuses its nodes' per-node NLLs.
        for mtype, groups in groups_by_type.items():
            for g in groups:                                                                                                    
                per_node_nlls = [n.scores[-1] for n in g.nodes]
                label         = g.nodes[0].labels[-1]                                                                           
                fused_z       = g.score(per_node_nlls)                                                                          
                fired         = g.cusum_update(fused_z, label=label)
                step.group_results.append(GroupStepResult(                                                                      
                    group_id     = g.group_id,                                                                                  
                    machine_type = mtype,
                    fused_z      = fused_z,                                                                                     
                    label        = label,
                    alarm        = fired,
                    cusum_S      = g.cusum_S[-1],                                                                               
                ))
                                                                                                                                
        yield step


# ── Entry point ──────────────────────────────────────────────────────────

def run(                                                                                                                        
    nodes_by_type:     dict[str, list[Node]],
    groups_by_type:    dict[str, list[Group]],                                                                                  
    timelines_by_type: dict[str, list[NodeTimeline]],
    n_fit_clips: int, 
    n_val_clips: int,
) -> Iterator[TimestepResult]:                                                                                                  
    """Run the full simulation: calibrate then evaluate."""
    print("Phase 1: Calibration + fusion setup")                                                                                
    calibrate(nodes_by_type, groups_by_type, timelines_by_type, n_fit_clips, n_val_clips)                                                                 
    print("Phase 2: Evaluation")                                                                                                
    yield from evaluate(nodes_by_type, groups_by_type, timelines_by_type)                                                       
                                                                                                                                
                                                                                                                                
# ── Helpers ──────────────────────────────────────────────────────────────                                                     
                                                                                                                                
def _timeline_lookup(
    timelines_by_type: dict[str, list[NodeTimeline]],
) -> dict[tuple[str, str], NodeTimeline]:
    """(mtype, mid) → NodeTimeline.  Flattens the by-type dict."""                                                              
    return {
        (t.machine_type, t.machine_id): t                                                                                       
        for ts in timelines_by_type.values()
        for t  in ts                                                                                                            
    }