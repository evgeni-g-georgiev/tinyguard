"""Split MIMII dataset into warmup / test_normal / test_abnormal per machine.

Reads from the already-extracted MIMII directory structure produced by
``data/download_mimii.py``:
    data/mimii_{snr}/{machine_type}/{machine_id}/{normal,abnormal}/*.wav

Produces (under ``simulation/data/splits/{snr}/``):
    {machine_type}/{machine_id}/warmup/*.wav
    {machine_type}/{machine_id}/test_normal/*.wav
    {machine_type}/{machine_id}/test_abnormal/*.wav
    surplus_abnormal/{machine_type}/{machine_id}/*.wav

Key properties:
    - Every .wav ends up in exactly one bucket (partition property).
    - test_normal count = test_abnormal count = min_abnormal across all nodes.
    - warmup count = min leftover normals across all nodes (simulation loader
      truncates to the configured warmup_count hyperparameter at runtime).
    - Equal timeline lengths guaranteed across all nodes.

The pipeline separates planning from execution so the split logic can be
tested without touching the filesystem. Files are placed via symlinks rather
than copies so the splits are cheap to (re)create.

Usage (typically called automatically from simulation/run_simulation.py;
this CLI is an escape hatch):
    python -m simulation.data.split_data --snr 6dB
    # which resolves to:
    #   --mimii-root  data/mimii_6db
    #   --splits-dir  simulation/data/splits/6dB
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import argparse
import os

# ── Data Classes ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NodeSources:
    """All source files discovered for a single machine node.                         
                                                                                    
    Frozen (immutable) so it cannot be modified after construction.                   
    """                                                                               
    machine_type: str                                                                 
    machine_id: str
    normal_files: tuple[Path, ...]
    abnormal_files: tuple[Path, ...]                                                  

                                                                                    
@dataclass(frozen=True)
class NodeSplitPlan:
    """Describes where every file for one node will be copied.
                                                                                    
    Each field holds the source file paths assigned to that bucket.                   
    Every source file appears in exactly one field (partition property).              
    """                                                                               
    machine_type: str
    machine_id: str                                                                   
    warmup: tuple[Path, ...]
    test_normal: tuple[Path, ...]                                                     
    test_abnormal: tuple[Path, ...]
    surplus_abnormal: tuple[Path, ...]    

# ── Planning Functions ────────────────────────────────────────────────────────

def compute_min_abnormal(manifests: list[NodeSources]) -> int:
    """Return the smallest abnormal file count across all nodes.
                                                                                    
    This determines the simulation test size — every node gets the same               
    number of abnormal files so the simulation timeline is uniform.                   
                                                                                    
    Raises:     
        ValueError: If any node has zero abnormal files, or if manifests              
            is empty.
    """
    if not manifests:                                                                 
        raise ValueError("No manifests provided.")
                                                                                    
    counts = [len(m.abnormal_files) for m in manifests]                               

    if min(counts) == 0:                                                              
        empty = [
            f"{m.machine_type}/{m.machine_id}"
            for m in manifests if len(m.abnormal_files) == 0                          
        ]
        raise ValueError(f"Nodes with zero abnormal files: {empty}")                  
                                                                                    
    return min(counts)
                                                                                    
                
def compute_max_warmup(
    manifests: list[NodeSources],
    min_abnormal: int,
) -> int:
    """Return the largest warmup count that all nodes can provide.                    

    After reserving min_abnormal normal files for test_normal, the                    
    remaining normals are available for warmup. The max warmup is
    the minimum of these remainders across all nodes.                                 
                                                                                    
    Raises:                                                                           
        ValueError: If any node cannot provide even one warmup file.                  
    """         
    remainders = [
        len(m.normal_files) - min_abnormal for m in manifests
    ]                                                                                 

    if min(remainders) <= 0:                                                          
        short = [
            f"{m.machine_type}/{m.machine_id} "
            f"(normals={len(m.normal_files)}, need>{min_abnormal})"                   
            for m, r in zip(manifests, remainders) if r <= 0
        ]                                                                             
        raise ValueError(
            f"Nodes without enough normal files for warmup: {short}"                  
        )                                                                             

    return min(remainders)                                                            
                

def plan_node_split(
    sources: NodeSources,
    min_abnormal: int,
    max_warmup: int,
) -> NodeSplitPlan:                                                                   
    """Assign every file in a node to exactly one bucket.
                                                                                    
    Allocation order for NORMAL files (from the sorted list):                         
        1. Last max_warmup files           → warmup                                   
        2. Next-to-last min_abnormal files → test_normal                              
                
    Allocation order for ABNORMAL files (from the sorted list):                       
        1. First min_abnormal files → test_abnormal
        2. Remaining files          → surplus_abnormal                                

    Args:                                                                             
        sources: All files for this node.
        min_abnormal: Uniform count for test_normal and test_abnormal.
        max_warmup: Number of normal files reserved for warmup.                       
                                                                                    
    Returns:                                                                          
        A NodeSplitPlan with every file assigned to exactly one bucket.               
    """         
    normal = sources.normal_files
    abnormal = sources.abnormal_files                                                 

    # ── Abnormal Allocation ────────────────────────────────────────────────────────                                                           
    test_abnormal = abnormal[:min_abnormal]
    surplus_abnormal = abnormal[min_abnormal:]                                        

    # ── Normal Allocation ────────────────────────────────────────────────────────                                       
    warmup = normal[-max_warmup:]
    test_normal = normal[-(max_warmup + min_abnormal):-max_warmup]                    

    return NodeSplitPlan(                                                             
        machine_type=sources.machine_type,
        machine_id=sources.machine_id,                                                
        warmup=warmup,
        test_normal=test_normal,
        test_abnormal=test_abnormal,
        surplus_abnormal=surplus_abnormal,
    )       

                                                                                        
def plan_all_splits(
    manifests: list[NodeSources],
) -> tuple[list[NodeSplitPlan], int, int]:
    """Plan the full split for all nodes.                                             

    Returns:                                                                          
        (plans, min_abnormal, max_warmup) so the caller can log
        the computed counts.                                                          
    """
    min_abnormal = compute_min_abnormal(manifests)                                    
    max_warmup = compute_max_warmup(manifests, min_abnormal)                          

    plans = [                                                                         
        plan_node_split(m, min_abnormal, max_warmup)
        for m in manifests                                                            
    ]
    return plans, min_abnormal, max_warmup  

# ── Filesystem Interaction ────────────────────────────────────────────────────────
def discover_sources(
    mimii_root: Path,
    machine_types: list[str],
) -> list[NodeSources]:
    """Walk the MIMII directory and build a manifest per node.

    Expects the structure produced by data/download_mimii.py:
        data/mimii_{snr}/{machine_type}/{machine_id}/{normal,abnormal}/*.wav

    Raises:
        FileNotFoundError: If mimii_root or a machine_type dir doesn't exist.
    """                                                                              
    if not mimii_root.exists():
        raise FileNotFoundError(f"MIMII root not found: {mimii_root}")                

    manifests = []                                                                    
                
    for machine_type in machine_types:
        machine_dir = mimii_root / machine_type
                                                                                    
        if not machine_dir.exists():
            raise FileNotFoundError(                                                  
                f"Expected machine type directory not found: {machine_dir}"
            )                                                                         

        for machine_id_dir in sorted(machine_dir.iterdir()):                          
            if not machine_id_dir.is_dir():
                continue                                                              

            normal_files = tuple(sorted(                                              
                (machine_id_dir / "normal").glob("*.wav")
            ))                                                                        
            abnormal_files = tuple(sorted(
                (machine_id_dir / "abnormal").glob("*.wav")                           
            ))  

            manifests.append(NodeSources(
                machine_type=machine_type,
                machine_id=machine_id_dir.name,                                       
                normal_files=normal_files,
                abnormal_files=abnormal_files,                                        
            ))  

    return manifests

def _copy_files(files: tuple[Path, ...], dest_dir: Path) -> None:
    """Symlink a tuple of files into dest_dir, creating it if needed.

    Splits use symlinks rather than copies so re-splitting is cheap and the
    split tree stays small even at MIMII scale.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        os.symlink(f.resolve(), dest_dir / f.name)
                                                                                    
                                                                                    
def execute_split(plan: NodeSplitPlan, splits_dir: Path) -> None:
    """Copy files according to one node's split plan."""                              
    mt, mid = plan.machine_type, plan.machine_id                                      

    _copy_files(plan.warmup,                                                          
                splits_dir / mt / mid / "warmup")
    _copy_files(plan.test_normal,                                                     
                splits_dir / mt / mid / "test_normal")
    _copy_files(plan.test_abnormal,                                                   
                splits_dir / mt / mid / "test_abnormal")
    _copy_files(plan.surplus_abnormal,                                                
                splits_dir / "surplus_abnormal" / mt / mid)
                                                                                    
                                                                                    
def clean_previous_splits(splits_dir: Path) -> None:
    """Remove all previous split output."""                                           
    if splits_dir.exists():
        print(f"  Removing previous splits at {splits_dir}")
        shutil.rmtree(splits_dir)        

# ── Splitting the Data  ────────────────────────────────────────────────────────
def split_data( 
    mimii_root: Path,
    splits_dir: Path,
    machine_types: list[str],
) -> None:
    """Execute the full data splitting pipeline.
                                                                                    
    Steps:
        1. Clean previous splits                                                      
        2. Discover all node manifests
        3. Plan the split (pure logic, no IO)
        4. Execute the plan (copy files)                                              
    """
    print("Step 1/4: Cleaning previous splits")                                       
    clean_previous_splits(splits_dir)                                                 

    print("Step 2/4: Discovering file manifests")                                     
    manifests = discover_sources(mimii_root, machine_types)
    print(f"  Found {len(manifests)} nodes")                                          

    for m in manifests:                                                               
        print(f"    {m.machine_type}/{m.machine_id}: "
            f"{len(m.normal_files)} normal, {len(m.abnormal_files)} abnormal")      

    print("Step 3/4: Planning splits")                                                
    plans, min_abnormal, max_warmup = plan_all_splits(manifests)
    print(f"  test_normal per node:    {min_abnormal}")                               
    print(f"  test_abnormal per node:  {min_abnormal}")                               
    print(f"  max warmup per node:     {max_warmup}")                                 
                                                                                    
    print("Step 4/4: Executing splits (copying files)")                               
    for plan in plans:
        print(f"  Copying {plan.machine_type}/{plan.machine_id}")                     
        execute_split(plan, splits_dir)                                               

    print("Done.")    

                                                                                        
# Display SNR ("-6dB") to on-disk MIMII suffix ("neg6db"). Mirrors
# run_simulation._MIMII_SNR_DIR.
_MIMII_SNR_DIR = {"-6dB": "neg6db", "0dB": "0db", "6dB": "6db"}


def main():
    parser = argparse.ArgumentParser(
        description="Split MIMII into warmup/test_normal/test_abnormal"
    )
    parser.add_argument(
        "--snr", default="6dB",
        choices=["6dB", "0dB", "-6dB"],
        help="Which MIMII SNR variant to split (default: 6dB).",
    )
    parser.add_argument(
        "--mimii-root", type=Path, default=None,
        help="Path to extracted MIMII data for the chosen SNR. "
             "Defaults to data/mimii_<dir> where <dir> is "
             "neg6db / 0db / 6db.",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("simulation/data/splits"),
        help="Output directory for splits. The chosen SNR is appended as a "
             "subdir, giving simulation/data/splits/<snr>/.",
    )
    parser.add_argument(
        "--machine-types", nargs="+",
        default=["fan", "pump", "slider", "valve"],
        help="Machine types to include",
    )
    args = parser.parse_args()

    mimii_root = (
        args.mimii_root
        if args.mimii_root is not None
        else Path("data") / f"mimii_{_MIMII_SNR_DIR[args.snr]}"
    )
    splits_dir = args.splits_dir / args.snr

    print(f"SNR: {args.snr}")                                                     
    print(f"  mimii_root: {mimii_root}")                                        
    print(f"  splits_dir: {splits_dir}")  
                                                                                    
    split_data( 
        mimii_root=mimii_root,
        splits_dir=splits_dir,
        machine_types=args.machine_types,
    )                                                                                 

                                                                                    
if __name__ == "__main__":
    main()