"""Run reporting — save config, results, summary, and plots after a simulation.
                                                                                    
This is the only module that writes to disk under simulation/outputs/runs/          
and the only module that imports matplotlib. It's invoked at the end of             
run_simulation.main() once the lockstep loop has finished.                          
                                                                                    
Output layout per run:                                                              
    simulation/outputs/runs/<timestamp>/                                            
        config.yaml          (verbatim copy of the YAML used)                       
        results.json         (per-node arrays + AUC + summary)                      
        summary.txt          (human-readable, with metadata header)                 
        plots/                                                                      
            grid.png         (4 cols × 4 rows, all 16 nodes)
            <node_id>.png    (one per node, full size)                              
            latent/
                grid_per_frame.png
                grid_per_clip.png
                <node_id>.png   (3-panel: per_frame + per_clip + histogram)         
"""
                                                                                    
import json     
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score                                                              
from sklearn.manifold import TSNE                                                   
                                                                     
                
from simulation.node.node import Node
                                                                                    
                                                                                    
# ── Run directory ────────────────────────────────────────────────────────────     
                                                                                    
def make_run_dir(base_dir: Path = Path("simulation/outputs/runs")) -> Path:         
    """Create a fresh timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                        
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()                                                     
    return run_dir
                                                                                    
                                                                                    
# ── Per-node statistics ──────────────────────────────────────────────────────
                                                                                    
def _node_stats(node: Node) -> dict:
    """Compute per-node stats: AUC, detection rate, false alarm rate, confusion."""
    has_two_classes = len(set(node.labels)) >= 2
    has_predictions = all(p is not None for p in node.predictions)                  
                                                                                    
    auc = roc_auc_score(node.labels, node.scores) if has_two_classes else None      
                                                                                    
    tp = tn = fp = fn = 0                                                           
    detection_rate = false_alarm_rate = None
                                                                                    
    if has_predictions:
        for label, pred in zip(node.labels, node.predictions):                      
            if label == 1 and pred == 1:                                            
                tp += 1
            elif label == 0 and pred == 0:                                          
                tn += 1
            elif label == 0 and pred == 1:
                fp += 1                                                             
            elif label == 1 and pred == 0:
                fn += 1                                                             
                
        if (tp + fn) > 0:                                                           
            detection_rate = tp / (tp + fn)
        if (fp + tn) > 0:                                                           
            false_alarm_rate = fp / (fp + tn)

    return {                                                                        
        "auc":              auc,
        "detection_rate":   detection_rate,                                         
        "false_alarm_rate": false_alarm_rate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }                                                                               

                                                                                    
# ── Anomaly band detection ───────────────────────────────────────────────────
                                                                                    
def _find_anomaly_bands(labels: list[int]) -> list[tuple[int, int]]:                
    """Group consecutive label==1 indices into (start, end_inclusive) tuples."""
    bands: list[tuple[int, int]] = []                                               
    start: int | None = None
    for i, label in enumerate(labels):                                              
        if label == 1 and start is None:                                            
            start = i
        elif label == 0 and start is not None:                                      
            bands.append((start, i - 1))
            start = None                                                            
    if start is not None:
        bands.append((start, len(labels) - 1))                                      
    return bands

                                                                                    
# ── Saving config + results + summary ────────────────────────────────────────
                                                                                    
def save_results(
    nodes_by_type: dict[str, list[Node]],
    config: dict,
    config_path: Path,
    runtime_seconds: float,                                                         
    run_dir: Path,
) -> None:                                                                          
    """Write config.yaml, results.json, and summary.txt to the run directory."""
                                                                                    
    shutil.copy(config_path, run_dir / "config.yaml")
                                                                                    
    node_results: dict[str, dict] = {}                                              
    type_aucs: dict[str, list[float]] = {}
                                                                                    
    for machine_type, nodes in nodes_by_type.items():                               
        type_aucs[machine_type] = []
        for node in nodes:                                                          
            stats = _node_stats(node)
            node_results[node.node_id] = {
                "machine_type":     node.machine_type,                              
                "machine_id":       node.machine_id,
                "scores":           [float(s) for s in node.scores],                
                "labels":           list(node.labels),
                "predictions":      [int(p) if p is not None else None for p in node.predictions],
                "threshold":        getattr(node.separator, "threshold", None),     
                "auc":              stats["auc"],
                "detection_rate":   stats["detection_rate"],
                "false_alarm_rate": stats["false_alarm_rate"],                      
                "confusion": {
                    "tp": stats["tp"], "tn": stats["tn"],                           
                    "fp": stats["fp"], "fn": stats["fn"],                           
                },
            }                                                                       
            if stats["auc"] is not None:
                type_aucs[machine_type].append(stats["auc"])

    summary = {
        f"{mt}_mean_auc": float(np.mean(aucs)) if aucs else None
        for mt, aucs in type_aucs.items()                                           
    }                                                                               
    all_aucs = [a for aucs in type_aucs.values() for a in aucs]                     
    summary["overall_mean_auc"] = float(np.mean(all_aucs)) if all_aucs else None    
                                                                                    
    results = {
        "config":  config,                                                          
        "nodes":   node_results,
        "summary": summary,                                                         
    }
                                                                                    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pipeline_str = (                                                                
        f"{config['preprocessor']} + "
        f"{config['frozen_embedder']} + "                                           
        f"{config['separator']}"
    )                                                                               
    sim_block = config.get("simulation", {})
                                                                                    
    runtime_minutes = int(runtime_seconds // 60)                                    
    runtime_secs    = int(runtime_seconds % 60)
                                                                                    
    lines: list[str] = []
    lines.append(f"Run: {run_dir.name}")                                            
    lines.append(f"Config: {config_path}")                                          
    lines.append(f"Pipeline: {pipeline_str}")
    lines.append(                                                                   
        f"Shuffle: {sim_block.get('shuffle_mode', '?')}  "                          
        f"Warmup: {sim_block.get('warmup_count', '?')}  "
        f"Seed: {sim_block.get('seed', '?')}"                                       
    )                                                                               
    lines.append(f"Runtime: {runtime_minutes}m {runtime_secs}s")
    lines.append("-" * 56)                                                          
    lines.append("")                                                                
    lines.append("Results")
    lines.append("=" * 56)                                                          
                
    for machine_type, nodes in nodes_by_type.items():
        for node in nodes:
            r = node_results[node.node_id]
            if r["auc"] is None:
                lines.append(f"{node.node_id}: AUC = N/A (single class)")           
                continue                                                            
            c = r["confusion"]                                                      
            lines.append(                                                           
                f"{node.node_id}: AUC = {r['auc']:.4f}  "
                f"|  TP={c['tp']:3d} TN={c['tn']:3d} "                              
                f"FP={c['fp']:3d} FN={c['fn']:3d}"                                  
            )                                                                       
                                                                                    
    lines.append("-" * 56)                                                          
    for machine_type in nodes_by_type:
        mean = summary.get(f"{machine_type}_mean_auc")                              
        if mean is not None:                                                        
            lines.append(f"{machine_type} mean AUC: {mean:.4f}")
    lines.append("")                                                                
    if summary["overall_mean_auc"] is not None:
        lines.append(f"Overall mean AUC: {summary['overall_mean_auc']:.4f}")        
    lines.append("=" * 56)
                                                                                    
    with open(run_dir / "summary.txt", "w") as f:                                   
        f.write("\n".join(lines) + "\n")
                                                                                    
                                                                                    
# ── Score-over-time plotting (existing) ──────────────────────────────────────
                                                                                    
def _draw_node_axes(ax, node: Node, config: dict, compact: bool = False) -> None:   
    """Render one node's scatter + threshold + bands onto an existing Axes."""
    timesteps = np.arange(len(node.scores))                                         
    scores = np.array(node.scores)
    labels = np.array(node.labels)                                                  
                
    normal_mask = labels == 0                                                       
    anomaly_mask = labels == 1
                                                                                    
    shuffle_mode = config.get("simulation", {}).get("shuffle_mode", "random")
    show_bands = shuffle_mode in ("block_random", "block_fixed")                    
                                                                                    
    if show_bands:
        bands = _find_anomaly_bands(node.labels)                                    
        for i, (start, end) in enumerate(bands):
            ax.axvspan(
                start - 0.5, end + 0.5,
                alpha=0.18, color="lightcoral",
                label="Anomaly injection" if i == 0 else None,                      
            )
                                                                                    
    ax.scatter(                                                                     
        timesteps[normal_mask], scores[normal_mask],
        s=18, c="steelblue", alpha=0.7, label="Normal clips",                       
    )                                                                               
    ax.scatter(
        timesteps[anomaly_mask], scores[anomaly_mask],                              
        s=18, c="darkorange", alpha=0.7, label="Anomaly clips",
    )

    threshold = getattr(node.separator, "threshold", None)                          
    if threshold is not None:
        ax.axhline(                                                                 
            threshold, color="firebrick", linestyle="--", linewidth=1.2,
            label=f"Threshold ({threshold:.3f})",                                   
        )
                                                                                    
    stats = _node_stats(node)                                                       
    auc_str = f"{stats['auc']:.4f}" if stats["auc"] is not None else "N/A"
                                                                                    
    if compact: 
        ax.set_title(                                                               
            f"{node.machine_type} {node.machine_id}: AUC={auc_str}",
            fontsize=10,                                                            
        )
    else:                                                                           
        det_str = (
            f"{stats['detection_rate'] * 100:.1f}%"                                 
            if stats["detection_rate"] is not None else "N/A"
        )                                                                           
        fa_str = (
            f"{stats['false_alarm_rate'] * 100:.1f}%"
            if stats["false_alarm_rate"] is not None else "N/A"
        )
        ax.set_title(
            f"Node: {node.machine_type} {node.machine_id}\n"
            f"Detection={det_str}  |  False alarm={fa_str}  |  AUC={auc_str}",
            fontsize=11,
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Anomaly score")
        ax.legend(loc="upper left", fontsize=8)

    ax.grid(alpha=0.3)


def save_plots(
    nodes_by_type: dict[str, list[Node]],
    config: dict,
    run_dir: Path,
) -> None:
    """Write plots/grid.png and plots/<node_id>.png × 16."""
    plots_dir = run_dir / "plots"                                                   

    sample_node = next(iter(nodes_by_type.values()))[0]                             
    sep_desc = sample_node.separator.description()
    embedder_name = config["frozen_embedder"]                                       
    fig_title = f"{sep_desc}  |  {embedder_name} embeddings"
                                                                                    
    for nodes in nodes_by_type.values():                                            
        for node in nodes:                                                          
            fig, ax = plt.subplots(figsize=(11, 4.5))                               
            fig.suptitle(fig_title, fontsize=12, fontweight="bold")                 
            _draw_node_axes(ax, node, config, compact=False)
            fig.tight_layout()                                                      
            fig.savefig(plots_dir / f"{node.node_id}.png", dpi=150)
            plt.close(fig)                                                          
                
    machine_types = list(nodes_by_type.keys())                                      
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),                                           
        sharex="col", sharey="row",
    )                                                                               
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.995)
                                                                                    
    for col, machine_type in enumerate(machine_types):
        nodes = nodes_by_type[machine_type]                                         
        for row, node in enumerate(nodes):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            _draw_node_axes(ax, node, config, compact=True)
                                                                                    
    fig.supxlabel("Timestep", fontsize=11)
    fig.supylabel("Anomaly score", fontsize=11)                                     
    fig.tight_layout()                                                              
    fig.savefig(plots_dir / "grid.png", dpi=100)
    plt.close(fig)                                                                  
                                                    

# ── Latent space plot helpers ────────────────────────────────────────────────     
                                                                                    
def _capture_node_projections(node: Node, timeline) -> dict:
    """Re-run the frozen pipeline and capture per-clip projections for plotting.    
                                                                                    
    For each warmup and test clip, runs preprocessor → embedder →                   
separator.project()                                                                 
    and stacks the resulting frame projections. The separator does NOT carry        
    these during the simulation — we recompute them here to keep memory             
    accounting honest.                                                              
                                                                                    
    Returns:                                                                        
        Dict with keys:
            warmup_projections : (n_warmup_frames, latent_dim) array                
            warmup_clip_indices: list of which warmup clip each frame came from     
            warmup_labels      : list of 0s, length n_warmup_frames                 
            test_projections   : (n_test_frames, latent_dim) array                  
            test_clip_indices  : list of which test clip each frame came from       
            test_labels        : list of 0/1, length n_test_frames                  
    """                                                                             
    def _project_clip(wav_path: str) -> np.ndarray:
        preprocessed = node.preprocessor.process(wav_path)                          
        embedded = node.frozen_embedder.embed(preprocessed)                         
        projected = node.separator.project(embedded)                                
        if projected.ndim == 1:                                                     
            projected = projected[np.newaxis, :]                                    
        return projected
                                                                                    
    # Warmup clips                                                                  
    warmup_arrays = []
    warmup_clip_indices: list[int] = []                                             
    for clip_idx, wav_path in enumerate(timeline.warmup_paths):
        clip_proj = _project_clip(wav_path)                                         
        warmup_arrays.append(clip_proj)
        warmup_clip_indices.extend([clip_idx] * len(clip_proj))                     
    warmup_projections = np.concatenate(warmup_arrays, axis=0)                      
    warmup_labels = [0] * len(warmup_projections)
                                                                                    
    # Test clips
    test_arrays = []                                                                
    test_labels: list[int] = []
    test_clip_indices: list[int] = []
    for clip_idx, (wav_path, label) in enumerate(                                   
        zip(timeline.test_paths, timeline.test_labels)
    ):                                                                              
        clip_proj = _project_clip(wav_path)
        test_arrays.append(clip_proj)                                               
        test_labels.extend([label] * len(clip_proj))
        test_clip_indices.extend([clip_idx] * len(clip_proj))                       
    test_projections = np.concatenate(test_arrays, axis=0)                          

    return {                                                                        
        "warmup_projections":  warmup_projections,
        "warmup_clip_indices": warmup_clip_indices,                                 
        "warmup_labels":       warmup_labels,
        "test_projections":    test_projections,                                    
        "test_clip_indices":   test_clip_indices,                                   
        "test_labels":         test_labels,
    }                                                                               
                

def _run_tsne(
    points: np.ndarray,
    perplexity: int = 30,                                                           
    random_state: int = 42,
) -> np.ndarray:                                                                    
    """Project an (N, D) array down to (N, 2) via t-SNE.
                                                                                    
    Asserts that there are enough points for the requested perplexity —             
    sklearn's heuristic is that perplexity should be less than n_samples / 3.       
    """                                                                             
    n_samples = points.shape[0]
    if n_samples < perplexity * 3 + 1:
        raise ValueError(
            f"t-SNE needs at least {perplexity * 3 + 1} points for "
            f"perplexity={perplexity}, got {n_samples}. Lower the "
            f"perplexity in default.yaml or use more clips."                        
        )                                                                           
                                                                                    
    tsne = TSNE(                                                                    
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,                                                  
        init="pca",
    )                                                                               
    return tsne.fit_transform(points)

                                                                                    
def _node_distance_data(node: Node) -> dict:
    """Return per-clip score data for the histogram panel.                          
                                                                                    
    Uses the already-recorded node.scores rather than recomputing — these           
    are exactly what the separator decides on (max-frame distance for SVDD,         
    NLL for GMM).                                                                   
    """                                                                             
    scores = np.array(node.scores)                                                  
    labels = np.array(node.labels)                                                  
    threshold = getattr(node.separator, "threshold", None)                          

    return {                                                                        
        "normal_scores":  scores[labels == 0],
        "anomaly_scores": scores[labels == 1],
        "threshold":      threshold,
    }

                                                                                    
def _stack_clip_max(
    projections: np.ndarray,                                                        
    labels: list[int],
    clip_indices: list[int],
    centroid: np.ndarray | None,                                                    
) -> tuple[np.ndarray, list[int]]:
    """For each clip, return the single frame with the highest distance from        
centroid.                                                                           
                                                                                    
    Args:                                                                           
        projections: (n_frames_total, latent_dim) — all frames stacked.
        labels: per-frame labels (length n_frames_total).                           
        clip_indices: which clip each frame belongs to.
        centroid: SVDD centroid in latent space. If None, uses L2 norm of           
            each frame as a proxy distance.                                         
                                                                                    
    Returns:                                                                        
        (per_clip_max_points, per_clip_labels)                                      
    """                                                                             
    clip_to_frames: dict[int, list[int]] = {}
    for frame_idx, clip_idx in enumerate(clip_indices):                             
        clip_to_frames.setdefault(clip_idx, []).append(frame_idx)                   

    max_points = []                                                                 
    max_labels = []
    for clip_idx in sorted(clip_to_frames):                                         
        frames = clip_to_frames[clip_idx]
        clip_proj = projections[frames]                                             
                                                                                    
        if centroid is not None:
            distances = ((clip_proj - centroid) ** 2).sum(axis=1)                   
        else:                                                                       
            distances = (clip_proj ** 2).sum(axis=1)
                                                                                    
        max_frame_local = int(np.argmax(distances))                                 
        max_points.append(clip_proj[max_frame_local])
        max_labels.append(labels[frames[max_frame_local]])                          
                                                                                    
    return np.array(max_points), max_labels
                                                                                    
                
def _render_node_latent_figure(
    node: Node,
    captured: dict,                                                                 
    plot_config: dict,
    latent_dir: Path,                                                               
) -> None:      
    """Render the per-node 3-panel latent space figure.
                                                                                    
    Panels: per-frame t-SNE | per-clip-max t-SNE | distance histogram.              
    Any panel can be disabled via plot_config flags.                                
    """                                                                             
    show_per_frame = plot_config.get("per_frame", True)
    show_per_clip = plot_config.get("per_clip", True)                               
    show_histogram = plot_config.get("histogram", True)                             

    panels = [name for name, on in (                                                
        ("per_frame", show_per_frame),
        ("per_clip",  show_per_clip),                                               
        ("histogram", show_histogram),
    ) if on]

    if not panels:
        return

    perplexity = plot_config.get("perplexity", 30)
    random_state = plot_config.get("random_state", 42)
    centroid = getattr(node.separator, "centroid", None)                            

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))          
    if len(panels) == 1:
        axes = [axes]

    sep_desc = node.separator.description()
    fig.suptitle(
        f"{sep_desc}  |  {node.machine_type} {node.machine_id}",                    
        fontsize=12, fontweight="bold",
    )                                                                               
                
    panel_axes = dict(zip(panels, axes))                                            
                
    # Per-frame t-SNE
    if show_per_frame:
        ax = panel_axes["per_frame"]                                                
        all_points = np.concatenate([
            captured["warmup_projections"],                                         
            captured["test_projections"],
        ], axis=0)                                                                  
        all_labels = np.array(
            captured["warmup_labels"] + captured["test_labels"]
        )                                                                           
        n_warmup = len(captured["warmup_projections"])
                                                                                    
        try:                                                                        
            proj_2d = _run_tsne(all_points, perplexity, random_state)
        except ValueError as exc:                                                   
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",                   
                    fontsize=8, transform=ax.transAxes, wrap=True)
        else:                                                                       
            warmup_pts = proj_2d[:n_warmup]
            test_pts = proj_2d[n_warmup:]                                           
            test_lbl = all_labels[n_warmup:]
                                                                                    
            ax.scatter(                                                             
                warmup_pts[:, 0], warmup_pts[:, 1],
                s=10, c="steelblue", marker="o", alpha=0.4,                         
                label="Warmup (normal)",                                            
            )
            ax.scatter(                                                             
                test_pts[test_lbl == 0, 0], test_pts[test_lbl == 0, 1],
                s=14, c="steelblue", marker=".", alpha=0.7,                         
                label="Test normal",                                                
            )                                                                       
            ax.scatter(                                                             
                test_pts[test_lbl == 1, 0], test_pts[test_lbl == 1, 1],
                s=14, c="darkorange", marker=".", alpha=0.7,                        
                label="Test anomaly",
            )                                                                       
            ax.legend(loc="best", fontsize=7)
                                                                                    
        ax.set_title("per-frame latent (t-SNE)", fontsize=10)
        ax.set_xticks([])                                                           
        ax.set_yticks([])

    # Per-clip max-distance t-SNE                                                   
    if show_per_clip:
        ax = panel_axes["per_clip"]                                                 
                
        warmup_pts, warmup_lbls = _stack_clip_max(                                  
            captured["warmup_projections"],
            captured["warmup_labels"],                                              
            captured["warmup_clip_indices"],
            centroid,                                                               
        )
        test_pts, test_lbls = _stack_clip_max(                                      
            captured["test_projections"],
            captured["test_labels"],
            captured["test_clip_indices"],
            centroid,
        )

        all_clip_pts = np.concatenate([warmup_pts, test_pts], axis=0)

        try:
            proj_2d = _run_tsne(all_clip_pts, perplexity, random_state)
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                    fontsize=8, transform=ax.transAxes, wrap=True)                  
        else:
            n_warmup = len(warmup_pts)                                              
            warmup_2d = proj_2d[:n_warmup]
            test_2d = proj_2d[n_warmup:]
            test_lbl = np.array(test_lbls)                                          

            ax.scatter(                                                             
                warmup_2d[:, 0], warmup_2d[:, 1],
                s=20, c="steelblue", marker="o", alpha=0.5,
                label="Warmup",                                                     
            )
            ax.scatter(                                                             
                test_2d[test_lbl == 0, 0], test_2d[test_lbl == 0, 1],
                s=20, c="steelblue", marker=".", alpha=0.7,                         
                label="Test normal",
            )                                                                       
            ax.scatter(
                test_2d[test_lbl == 1, 0], test_2d[test_lbl == 1, 1],               
                s=20, c="darkorange", marker=".", alpha=0.7,
                label="Test anomaly",                                               
            )
            ax.legend(loc="best", fontsize=7)                                       
                
        ax.set_title("per-clip max-frame (t-SNE)", fontsize=10)                     
        ax.set_xticks([])
        ax.set_yticks([])                                                           
                
    # Distance histogram                                                            
    if show_histogram:
        ax = panel_axes["histogram"]                                                
        dist = _node_distance_data(node)

        ax.hist(
            dist["normal_scores"], bins=30, alpha=0.6,
            color="steelblue", label="Normal",
        )                                                                           
        ax.hist(
            dist["anomaly_scores"], bins=30, alpha=0.6,                             
            color="darkorange", label="Anomaly",
        )                                                                           
        if dist["threshold"] is not None:
            ax.axvline(                                                             
                dist["threshold"], color="firebrick",
                linestyle="--", linewidth=1.5,
                label=f"Threshold ({dist['threshold']:.3f})",                       
            )
        ax.set_title("score distribution", fontsize=10)                             
        ax.set_xlabel("anomaly score")                                              
        ax.set_ylabel("clip count")
        ax.legend(loc="best", fontsize=7)                                           
        ax.grid(alpha=0.3)                                                          

    fig.tight_layout()                                                              
    fig.savefig(latent_dir / f"{node.node_id}.png", dpi=130)
    plt.close(fig)                                                                  



def _render_score_grid(
    nodes_by_type: dict[str, list[Node]],
    plot_config: dict,
    latent_dir: Path,
) -> None:
    """Render a 4-row × 4-col grid of score-distribution histograms."""
    machine_types = list(nodes_by_type.keys())
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
    )

    sample_node = next(iter(nodes_by_type.values()))[0]
    sep_desc = sample_node.separator.description()
    fig.suptitle(
        f"{sep_desc}  |  score distribution",
        fontsize=14, fontweight="bold",
    )

    for col, machine_type in enumerate(machine_types):
        nodes = nodes_by_type[machine_type]
        for row, node in enumerate(nodes):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            dist = _node_distance_data(node)

            ax.hist(
                dist["normal_scores"], bins=30, alpha=0.6,
                color="steelblue", label="Normal",
            )
            ax.hist(
                dist["anomaly_scores"], bins=30, alpha=0.6,
                color="darkorange", label="Anomaly",
            )
            if dist["threshold"] is not None:
                ax.axvline(
                    dist["threshold"], color="firebrick",
                    linestyle="--", linewidth=1.2,
                    label=f"thr={dist['threshold']:.2f}",
                )
            ax.set_title(f"{node.machine_type} {node.machine_id}", fontsize=10)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=6)

    fig.tight_layout()
    fig.savefig(latent_dir / "grid_scores.png", dpi=110)
    plt.close(fig)

                                                                                    
def _render_latent_grid(
    nodes_by_type: dict[str, list[Node]],
    captured_by_node: dict[str, dict],                                              
    variant: str,            # "per_frame" or "per_clip"
    plot_config: dict,                                                              
    latent_dir: Path,
) -> None:                                                                          
    """Render a 4-row × 4-col grid of t-SNE plots, one panel per node."""
    perplexity = plot_config.get("perplexity", 30)                                  
    random_state = plot_config.get("random_state", 42)                              
                                                                                    
    machine_types = list(nodes_by_type.keys())                                      
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),                                         
    )
                                                                                    
    sample_node = next(iter(nodes_by_type.values()))[0]
    sep_desc = sample_node.separator.description()                                  
    title_variant = (
        "per-frame latent (t-SNE)" if variant == "per_frame"                        
        else "per-clip max-frame (t-SNE)"                                           
    )                                                                               
    fig.suptitle(f"{sep_desc}  |  {title_variant}", fontsize=14, fontweight="bold") 
                                                                                    
    for col, machine_type in enumerate(machine_types):
        nodes = nodes_by_type[machine_type]                                         
        for row, node in enumerate(nodes):
            ax = axes[row, col] if n_rows > 1 else axes[col]                        
            captured = captured_by_node[node.node_id]
            centroid = getattr(node.separator, "centroid", None)                    
                
            if variant == "per_frame":
                points = np.concatenate([
                    captured["warmup_projections"],
                    captured["test_projections"],                                   
                ], axis=0)
                labels = np.array(                                                  
                    captured["warmup_labels"] + captured["test_labels"]
                )                                                                   
            else:  # per_clip
                warmup_pts, warmup_lbls = _stack_clip_max(                          
                    captured["warmup_projections"],
                    captured["warmup_labels"],
                    captured["warmup_clip_indices"],
                    centroid,
                )
                test_pts, test_lbls = _stack_clip_max(
                    captured["test_projections"],
                    captured["test_labels"],                                        
                    captured["test_clip_indices"],
                    centroid,                                                       
                )
                points = np.concatenate([warmup_pts, test_pts], axis=0)
                labels = np.array(list(warmup_lbls) + list(test_lbls))              

            try:                                                                    
                proj_2d = _run_tsne(points, perplexity, random_state)
            except ValueError as exc:                                               
                ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)                         
                ax.set_title(f"{node.machine_type} {node.machine_id}", fontsize=10)
                ax.set_xticks([])                                                   
                ax.set_yticks([])
                continue                                                            
                                                                                    
            ax.scatter(
                proj_2d[labels == 0, 0], proj_2d[labels == 0, 1],                   
                s=8, c="steelblue", alpha=0.6,
            )                                                                       
            ax.scatter(
                proj_2d[labels == 1, 0], proj_2d[labels == 1, 1],                   
                s=8, c="darkorange", alpha=0.6,
            )
            ax.set_title(f"{node.machine_type} {node.machine_id}", fontsize=10)
            ax.set_xticks([])                                                       
            ax.set_yticks([])
                                                                                    
    fig.tight_layout()
    fig.savefig(latent_dir / f"grid_{variant}.png", dpi=110)
    plt.close(fig)                                                                  

                                                                                    
def save_latent_plots(
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict,                                                        
    config: dict,
    run_dir: Path,                                                                  
) -> None:                                                                          
    """Render latent space plots for every node + grid summaries.
                                                                                    
    Three phases:
        1. Capture: re-run preprocessor + embedder + separator.project()            
            for every clip on every node. Slow.                                      
        2. Per-node render: 3-panel figure per node (per-frame t-SNE,               
            per-clip max t-SNE, distance histogram).                                 
        3. Grid render: 4×4 grid of t-SNEs, one for each variant.                   
                                                                                    
    Skipped entirely if config['latent_plot']['enabled'] is False.                  
    """                                                                             
    plot_config = config.get("latent_plot", {}) or {}                               
    if not plot_config.get("enabled", False):
        return

    latent_dir = run_dir / "plots" / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)
                                                                                    
    print("\nCapturing latent projections...")
    captured_by_node: dict[str, dict] = {}                                          
    for machine_type, nodes in nodes_by_type.items():
        timelines = timelines_by_type[machine_type]                                 
        for node, timeline in zip(nodes, timelines):
            print(f"  {node.node_id}")                                              
            captured_by_node[node.node_id] = _capture_node_projections(node,
timeline)                                                                           
                
    print("\nRendering per-node latent figures...")                                 
    for machine_type, nodes in nodes_by_type.items():
        for node in nodes:
            _render_node_latent_figure(                                             
                node, captured_by_node[node.node_id], plot_config, latent_dir,
            )                                                                       
                
    print("\nRendering latent grid figures...")                                     
    if plot_config.get("per_frame", True):
        _render_latent_grid(                                                        
            nodes_by_type, captured_by_node, "per_frame", plot_config, latent_dir,
        )
    if plot_config.get("per_clip", True):
        _render_latent_grid(
            nodes_by_type, captured_by_node, "per_clip", plot_config, latent_dir,
        )     
    
    print("\nRendering score distribution grid...")                             
    _render_score_grid(nodes_by_type, plot_config, latent_dir)