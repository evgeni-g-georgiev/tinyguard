"""sweep_n.py — Sweep over n_nodes; record metrics; plot.
                                                                                                                            
Runs the lockstep simulation for each (n, seed) cell with artefact-writing
disabled, and appends one row per (machine, view, metric) to a tidy CSV.                                                     
A 2x3 panel PNG is plotted at the end.                                                                                       
                                                                                                                            
Usage                                                                                                                        
-----                                                                                                                        
    python -m simulation.sweep_n
    python -m simulation.sweep_n --n-grid 1,2,4,8 --seeds 5                                                                  
    python -m simulation.sweep_n --base simulation/configs/default.yaml \\
        --out-dir simulation/outputs/sweeps/2026-04-25_quick                                                                 
                                                                                                                            
Output (under --out-dir, default simulation/outputs/sweeps/<timestamp>/):                                                    
    sweep_results.csv   — long-format: (n, seed, machine_type, machine_id,                                                   
                                        view, metric, value, runtime_seconds)                                                
    sweep_plot.png      — 2x3 panel: 5 metrics + 1 spare; lines = machine_type;                                              
                                    bold dashed black = aggregate; band = ±1 SE.                                            
    sweep_config.yaml   — base config + grid + total runtime, for reproducibility.                                           
"""                                                                                                                          
                                                                                                                            
import argparse                                                                                                              
import copy     
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt                                                                                              
import numpy as np
import pandas as pd                                                                                                          
import yaml     

from simulation.metrics        import clip_metrics, block_metrics
from simulation.run_simulation import run_with_config
                                                                                                                            
                                                                                                                            
METRIC_NAMES = ("auc", "clip_f1", "block_f1", "mean_lag", "fa_rate")                                                         
CSV_COLS     = ["n", "seed", "machine_type", "machine_id",                                                                   
                "view", "metric", "value", "runtime_seconds"]                                                                

                                                                                                                            
# ── Metric extraction ────────────────────────────────────────────────────
                                                                                                                            
def _aggregate(cms, bms) -> dict:
    """Mean of each metric across one or more (clip, block) detector pairs."""
    def _mean(xs):                                                                                                           
        xs = [x for x in xs if x is not None and not np.isnan(x)]                                                            
        return float(np.mean(xs)) if xs else float("nan")                                                                    
                                                                                                                            
    aucs      = [c.auc      for c in cms if c is not None]                                                                   
    clip_f1s  = [c.f1       for c in cms if c is not None]                                                                   
    fas       = [c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0                                                              
                for c in cms if c is not None]                                                                              
    block_f1s = [b.block_f1 for b in bms if b is not None]
    lags      = [b.mean_lag for b in bms                                                                                     
                if b is not None and b.mean_lag is not None]
    return {                                                                                                                 
        "auc":      _mean(aucs),                                                                                             
        "clip_f1":  _mean(clip_f1s),
        "block_f1": _mean(block_f1s),                                                                                        
        "mean_lag": _mean(lags),
        "fa_rate":  _mean(fas),                                                                                              
    }
                                                                                                                            
                                                                                                                            
def extract_rows(result: dict, n: int, seed: int) -> list[dict]:
    """One row per (machine, view, metric) for one (n, seed) run."""                                                         
    runtime        = result["runtime_seconds"]
    nodes_by_type  = result["nodes_by_type"]                                                                                 
    groups_by_type = result["groups_by_type"]
                                                                                                                            
    rows: list[dict] = []                                                                                                    

    # ── Node view: per-machine, averaged across that machine's channels.                                                    
    nodes_by_machine: dict[tuple[str, str], list] = {}
    for ns in nodes_by_type.values():                                                                                        
        for nd in ns:
            nodes_by_machine.setdefault(                                                                                     
                (nd.machine_type, nd.machine_id), []
            ).append(nd)                                                                                                     
                
    for (mtype, mid), nds in nodes_by_machine.items():                                                                       
        cms = [clip_metrics(nd.labels, nd.scores, nd.alarms) for nd in nds]
        bms = [block_metrics(nd.labels, nd.alarms)            for nd in nds]                                                 
        m   = _aggregate(cms, bms)
        for metric in METRIC_NAMES:                                                                                          
            rows.append({
                "n": n, "seed": seed,                                                                                        
                "machine_type": mtype, "machine_id": mid,                                                                    
                "view": "node", "metric": metric, "value": m[metric],
                "runtime_seconds": runtime,                                                                                  
            })  
                                                                                                                            
    # ── Group view: only when n >= 2 (no groups exist for n=1).                                                             
    for gs in groups_by_type.values():                                                                                       
        for g in gs:                                                                                                         
            cm = clip_metrics(g.labels, g.fused_scores, g.alarms)                                                            
            bm = block_metrics(g.labels, g.alarms)
            m  = _aggregate([cm], [bm])                                                                                      
            for metric in METRIC_NAMES:
                rows.append({                                                                                                
                    "n": n, "seed": seed,
                    "machine_type": g.machine_type, "machine_id": g.machine_id,
                    "view": "group", "metric": metric, "value": m[metric],                                                   
                    "runtime_seconds": runtime,
                })                                                                                                           
                
    # ── Primary view: group when present, else node.  Same metrics, retagged.                                               
    primary_src = "group" if any(r["view"] == "group" for r in rows) else "node"
    rows.extend([dict(r, view="primary") for r in rows if r["view"] == primary_src])                                         
                                                                                                                            
    return rows                                                                                                              
                                                                                                                            
                
# ── CSV helpers ──────────────────────────────────────────────────────────

def load_completed(csv_path: Path) -> set[tuple[int, int]]:                                                                  
    """Set of (n, seed) cells already present in the CSV.  Empty if no CSV."""
    if not csv_path.exists():                                                                                                
        return set()
    df = pd.read_csv(csv_path, usecols=["n", "seed"])                                                                        
    return {(int(n), int(s))                                                                                                 
            for n, s in df.drop_duplicates().itertuples(index=False)}
                                                                                                                            
                
def append_rows(csv_path: Path, rows: list[dict]) -> None:                                                                   
    df = pd.DataFrame(rows, columns=CSV_COLS)
    write_header = not csv_path.exists()                                                                                     
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
                                                                                                                            
                
# ── Plot ─────────────────────────────────────────────────────────────────                                                  
                
PANELS = [                                                                                                                   
    ("auc",      "AUC"),
    ("clip_f1",  "Clip-level F1"),                                                                                           
    ("block_f1", "Block-level F1"),                                                                                          
    ("mean_lag", "Mean detection lag (clips)"),
    ("fa_rate",  "False-alarm rate"),                                                                                        
]               
                                                                                                                            
                                                                                                                            
def plot_sweep(csv_path: Path, png_path: Path) -> None:
    df = pd.read_csv(csv_path)                                                                                               
    df = df[df["view"] == "primary"]
    if df.empty:
        print(f"plot_sweep: no primary rows in {csv_path}, skipping plot.")
        return                                                                                                               

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)                                                             
    axes = list(axes.flat)
    mtypes = sorted(df["machine_type"].unique())                                                                             
    cmap   = plt.get_cmap("tab10")
    colour = {mt: cmap(i) for i, mt in enumerate(mtypes)}                                                                    
                
    for ax, (metric, title) in zip(axes, PANELS):                                                                            
        sub = df[df["metric"] == metric]
        if sub.empty:                                                                                                        
            ax.set_title(f"{title}\n(no data)")
            continue                                                                                                         

        for mt in mtypes:                                                                                                    
            sub_mt  = sub[sub["machine_type"] == mt]
            grouped = sub_mt.groupby("n")["value"]                                                                           
            ns      = sorted(grouped.groups.keys())                                                                          
            mean    = grouped.mean().reindex(ns)                                                                             
            se      = grouped.sem().reindex(ns).fillna(0.0)                                                                  
            ax.plot(ns, mean.values, color=colour[mt], lw=1.6,                                                               
                    marker="o", label=mt, alpha=0.85)                                                                        
            ax.fill_between(ns, mean - se, mean + se,                                                                        
                            color=colour[mt], alpha=0.18)
                                                                                                                            
        agg    = sub.groupby("n")["value"]                                                                                   
        ns_a   = sorted(agg.groups.keys())                                                                                   
        mean_a = agg.mean().reindex(ns_a)                                                                                    
        se_a   = agg.sem().reindex(ns_a).fillna(0.0)
        ax.plot(ns_a, mean_a.values, color="black", lw=2.5, ls="--",                                                         
                marker="s", label="aggregate")                                                                               
        ax.fill_between(ns_a, mean_a - se_a, mean_a + se_a,                                                                  
                        color="black", alpha=0.10)                                                                           
                
        ax.set_title(title)                                                                                                  
        ax.set_xlabel("n_nodes")
        ax.set_ylabel(metric)                                                                                                
        ax.grid(alpha=0.3)
        if metric == "auc":                                                                                                  
            ax.legend(fontsize=8, loc="lower right")                                                                         

    for ax in axes[len(PANELS):]:                                                                                            
        ax.axis("off")
                                                                                                                            
    fig.suptitle(
        f"Metric vs n_nodes  |  "
        f"{df['n'].nunique()} n-values × "                                                                                   
        f"{df['seed'].nunique()} seeds × "                                                                                   
        f"{df['machine_id'].nunique()} ids per type",                                                                        
        fontsize=13, fontweight="bold",                                                                                      
    )           
    fig.tight_layout()                                                                                                       
    fig.savefig(png_path, dpi=140, bbox_inches="tight")                                                                      
    plt.close(fig)
                                                                                                                            
                                                                                                                            
# ── Main ─────────────────────────────────────────────────────────────────
                                                                                                                            
def _parse_n_grid(s: str) -> list[int]:                                                                                      
    return [int(x.strip()) for x in s.split(",") if x.strip()]
                                                                                                                            
                
def main() -> None:
    p = argparse.ArgumentParser(description="Sweep over n_nodes")
    p.add_argument("--base",       default="simulation/configs/default.yaml",                                                
                    help="Base config YAML.")                                                                                 
    p.add_argument("--n-grid",     default="1,2,3,4,5,6,7,8",                                                                
                    help="Comma-separated n_nodes values.")                                                                   
    p.add_argument("--seeds",      type=int, default=3,
                    help="Number of seeds per n.")                                                                            
    p.add_argument("--seed-start", type=int, default=42,
                    help="First seed value.  Subsequent seeds are seed-start+1, +2, ...")                                     
    p.add_argument("--out-dir",    default=None,                                                                             
                    help="Output directory.  Default: simulation/outputs/sweeps/<timestamp>/")                                
    p.add_argument("--no-plot",    action="store_true",                                                                      
                    help="Skip plotting at the end; just write the CSV.")                                                     
    args = p.parse_args()                                                                                                    
                                                                                                                            
    n_grid = _parse_n_grid(args.n_grid)                                                                                      
    seeds  = list(range(args.seed_start, args.seed_start + args.seeds))
                                                                                                                            
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                                                                 
    out_dir   = Path(args.out_dir or f"simulation/outputs/sweeps/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)                                                                               
    csv_path = out_dir / "sweep_results.csv"
    png_path = out_dir / "sweep_plot.png"                                                                                    
                
    with open(args.base) as f:                                                                                               
        base_cfg = yaml.safe_load(f)
                                                                                                                            
    prov = {
        "base_config_path": str(Path(args.base).resolve()),                                                                  
        "n_grid":           n_grid,                                                                                          
        "seeds":            seeds,
        "created":          timestamp,                                                                                       
        "base_config":      base_cfg,                                                                                        
    }
    with open(out_dir / "sweep_config.yaml", "w") as f:                                                                      
        yaml.dump(prov, f, sort_keys=False)                                                                                  

    completed = load_completed(csv_path)                                                                                     
    total     = len(n_grid) * len(seeds)
    todo      = total - len(completed)                                                                                       
    print(f"Sweep dir: {out_dir}")
    print(f"n_grid:    {n_grid}")                                                                                            
    print(f"seeds:     {seeds}")
    print(f"Already complete: {len(completed)}/{total}.  To run: {todo}.\n")                                                 
                                                                                                                            
    sweep_start = time.time()
                                                                                                                            
    for n in n_grid:
        for seed in seeds:
            cell = (n, seed)
            if cell in completed:                                                                                            
                continue
                                                                                                                            
            cfg = copy.deepcopy(base_cfg)
            # Override the base config's channels list — this is the key
            # _resolve_channels() actually reads.  Setting cfg["n_nodes"]
            # alone has no effect because `channels` takes precedence.
            cfg["channels"]           = list(range(n))
            cfg["simulation"]["seed"] = seed
            cfg["gmm"]["seed"]        = seed                                                                                 

            cell_start = time.time()                                                                                         
            print(f"[n={n} seed={seed}] running...", flush=True)
            try:                                                                                                             
                result = run_with_config(
                    cfg,                                                                                                     
                    config_path    = None,
                    save_artefacts = False,                                                                                  
                    verbose_steps  = False,
                )                                                                                                            
            except Exception as e:
                print(f"[n={n} seed={seed}] FAILED: {e!r}", flush=True)                                                      
                continue
                                                                                                                            
            rows = extract_rows(result, n, seed)                                                                             
            append_rows(csv_path, rows)
            completed.add(cell)                                                                                              
                
            df_cell = pd.DataFrame(rows)                                                                                     
            agg = (df_cell[df_cell["view"] == "primary"]
                    .groupby("metric")["value"].mean().to_dict())                                                             
            print(                                                                                                           
                f"[n={n} seed={seed}] done in {time.time()-cell_start:.1f}s  "
                f"AUC={agg.get('auc', float('nan')):.3f}  "                                                                  
                f"clipF1={agg.get('clip_f1', float('nan')):.3f}  "                                                           
                f"blockF1={agg.get('block_f1', float('nan')):.3f}  "                                                         
                f"lag={agg.get('mean_lag', float('nan')):.1f}  "                                                             
                f"FA={agg.get('fa_rate', float('nan')):.3f}",
                flush=True,                                                                                                  
            )   
                                                                                                                            
    sweep_runtime = time.time() - sweep_start                                                                                
    print(f"\nSweep done.  Total runtime: {sweep_runtime:.1f}s")
    print(f"CSV: {csv_path}")                                                                                                
                
    prov["sweep_runtime_s"] = sweep_runtime                                                                                  
    with open(out_dir / "sweep_config.yaml", "w") as f:
        yaml.dump(prov, f, sort_keys=False)                                                                                  
                                                                                                                            
    if not args.no_plot:
        plot_sweep(csv_path, png_path)                                                                                       
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()