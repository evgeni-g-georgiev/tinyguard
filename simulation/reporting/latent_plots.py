            
"""Latent space visualisation — t-SNE projections, score histograms, grids."""
                                                                                                    
from pathlib import Path
                                                                                                    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
                                                                                                    
from simulation.node.node import Node
                                                                                                    
                                                                                                    
# ── Private helpers (only used within this module) ───────────────────────────
                                                                                                    
def _node_distance_data(node: Node) -> dict:
    """Return per-clip score data for the histogram panel."""
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
    """For each clip, return the single frame with the highest distance from centroid."""             
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
                
                                                                                                    
def _capture_node_projections(node: Node, timeline) -> dict:
    """Re-run the frozen pipeline and capture per-clip projections for plotting."""                   
    def _project_clip(wav_path: str) -> np.ndarray:                                                   
        preprocessed = node.preprocessor.process(wav_path)                                            
        embedded = node.frozen_embedder.embed(preprocessed)                                           
        projected = node.separator.project(embedded)                                                  
        if projected.ndim == 1:                                                                       
            projected = projected[np.newaxis, :]
        return projected                                                                              

    warmup_arrays = []                                                                                
    warmup_clip_indices: list[int] = []
    for clip_idx, wav_path in enumerate(timeline.warmup_paths):                                       
        clip_proj = _project_clip(wav_path)
        warmup_arrays.append(clip_proj)                                                               
        warmup_clip_indices.extend([clip_idx] * len(clip_proj))
    warmup_projections = np.concatenate(warmup_arrays, axis=0)                                        
    warmup_labels = [0] * len(warmup_projections)
                                                                                                    
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
    """Project an (N, D) array down to (N, 2) via t-SNE."""                                           
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

                                                                                                    
# ── Per-node latent figure ───────────────────────────────────────────────────
                                                                                                    
def _render_node_latent_figure(
    node: Node,
    captured: dict,
    plot_config: dict,
    latent_dir: Path,                                                                                 
) -> None:
    """Render the per-node 3-panel latent space figure."""                                            
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
                                                                                                    
            ax.scatter(warmup_pts[:, 0], warmup_pts[:, 1],
                        s=10, c="steelblue", marker="o", alpha=0.4, label="Warmup (normal)")           
            ax.scatter(test_pts[test_lbl == 0, 0], test_pts[test_lbl == 0, 1],                        
                        s=14, c="steelblue", marker=".", alpha=0.7, label="Test normal")               
            ax.scatter(test_pts[test_lbl == 1, 0], test_pts[test_lbl == 1, 1],                        
                        s=14, c="darkorange", marker=".", alpha=0.7, label="Test anomaly")             
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
                                                                                                    
            ax.scatter(warmup_2d[:, 0], warmup_2d[:, 1],                                              
                        s=20, c="steelblue", marker="o", alpha=0.5, label="Warmup")
            ax.scatter(test_2d[test_lbl == 0, 0], test_2d[test_lbl == 0, 1],                          
                        s=20, c="steelblue", marker=".", alpha=0.7, label="Test normal")               
            ax.scatter(test_2d[test_lbl == 1, 0], test_2d[test_lbl == 1, 1],
                        s=20, c="darkorange", marker=".", alpha=0.7, label="Test anomaly")             
            ax.legend(loc="best", fontsize=7)
                                                                                                    
        ax.set_title("per-clip max-frame (t-SNE)", fontsize=10)
        ax.set_xticks([])                                                                             
        ax.set_yticks([])
                                                                                                    
    # Distance histogram
    if show_histogram:                                                                                
        ax = panel_axes["histogram"]
        dist = _node_distance_data(node)
                                                                                                    
        ax.hist(dist["normal_scores"], bins=30, alpha=0.6,
                color="steelblue", label="Normal")                                                    
        ax.hist(dist["anomaly_scores"], bins=30, alpha=0.6,
                color="darkorange", label="Anomaly")                                                  
        if dist["threshold"] is not None:
            ax.axvline(dist["threshold"], color="firebrick",                                          
                        linestyle="--", linewidth=1.5,                                                 
                        label=f"Threshold ({dist['threshold']:.3f})")
        ax.set_title("score distribution", fontsize=10)                                               
        ax.set_xlabel("anomaly score")                                                                
        ax.set_ylabel("clip count")
        ax.legend(loc="best", fontsize=7)                                                             
        ax.grid(alpha=0.3)                                                                            

    fig.tight_layout()                                                                                
    fig.savefig(latent_dir / f"{node.node_id}.png", dpi=130)
    plt.close(fig)                                                                                    

                                                                                                    
# ── Grid plots ───────────────────────────────────────────────────────────────

def _render_score_grid(                                                                               
    nodes_by_type: dict[str, list[Node]],
    plot_config: dict,                                                                                
    latent_dir: Path,
) -> None:
    """Render a 4x4 grid of score-distribution histograms."""
    machine_types = list(nodes_by_type.keys())                                                        
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())                                      
                                                                                                    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))                      
                                                                                                    
    sample_node = next(iter(nodes_by_type.values()))[0]                                               
    sep_desc = sample_node.separator.description()
    fig.suptitle(f"{sep_desc}  |  score distribution", fontsize=14, fontweight="bold")
                                                                                                    
    for col, machine_type in enumerate(machine_types):
        nodes = nodes_by_type[machine_type]                                                           
        for row, node in enumerate(nodes):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            dist = _node_distance_data(node)                                                          

            ax.hist(dist["normal_scores"], bins=30, alpha=0.6,                                        
                    color="steelblue", label="Normal")
            ax.hist(dist["anomaly_scores"], bins=30, alpha=0.6,                                       
                    color="darkorange", label="Anomaly")
            if dist["threshold"] is not None:                                                         
                ax.axvline(dist["threshold"], color="firebrick",
                            linestyle="--", linewidth=1.2,                                             
                            label=f"thr={dist['threshold']:.2f}")
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
    variant: str,                                                                                     
    plot_config: dict,
    latent_dir: Path,                                                                                 
) -> None:      
    """Render a 4x4 grid of t-SNE plots, one panel per node."""
    perplexity = plot_config.get("perplexity", 30)                                                    
    random_state = plot_config.get("random_state", 42)
                                                                                                    
    machine_types = list(nodes_by_type.keys())                                                        
    n_cols = len(machine_types)
    n_rows = max(len(nodes) for nodes in nodes_by_type.values())                                      
                
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))                      

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
            else:
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
                
            ax.scatter(proj_2d[labels == 0, 0], proj_2d[labels == 0, 1],
                        s=8, c="steelblue", alpha=0.6)
            ax.scatter(proj_2d[labels == 1, 0], proj_2d[labels == 1, 1],                              
                        s=8, c="darkorange", alpha=0.6)                                                
            ax.set_title(f"{node.machine_type} {node.machine_id}", fontsize=10)                       
            ax.set_xticks([])                                                                         
            ax.set_yticks([])                                                                         

    fig.tight_layout()                                                                                
    fig.savefig(latent_dir / f"grid_{variant}.png", dpi=110)
    plt.close(fig)

                                                                                                    
# ── Entry point ──────────────────────────────────────────────────────────────
                                                                                                    
def save_latent_plots(
    nodes_by_type: dict[str, list[Node]],
    timelines_by_type: dict,
    config: dict,                                                                                     
    run_dir: Path,
) -> None:                                                                                            
    """Render latent space plots for every node + grid summaries.
                                                                                                    
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
            captured_by_node[node.node_id] = _capture_node_projections(node, timeline)
                                                                                                    
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