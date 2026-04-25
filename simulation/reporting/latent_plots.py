"""Latent space visualisation for the GMM pipeline.

Per-node figure has two panels (t-SNE on the per-clip TWFR feature, and an
NLL-score histogram). The per-clip GMM feature is one vector per clip, so
there is no per-frame vs per-clip distinction to render.

Filter: only nodes whose channel index is in config.latent_plot.node_subset
are rendered. Empty list means every channel.

Output under <run_dir>/plots/latent/, one folder per channel:
    ch<N>/<machine_type>_<machine_id>.png   per-node 2-panel figure
    ch<N>/grid_tsne.png                     grid of t-SNE panels (one cell per machine)
    ch<N>/grid_scores.png                   grid of score histograms (one cell per machine)
"""                                                                                   
                
from pathlib import Path                                                              
                
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
                                                                                    
from gmm.features import extract_feature_r, load_log_mel
                                                                                    
from simulation.node.node import Node


_COL_NORM   = "steelblue"
_COL_ANOM   = "darkorange"
_COL_WARMUP = "#6a8caf"                                                               
_COL_K      = "firebrick"
_COL_H      = "#7b241c"                                                               
                
                                                                                    
# ── Feature extraction ───────────────────────────────────────────────────
                                                                                    
def _extract_features(node: Node, wav_paths) -> np.ndarray:
    """TWFR feature per clip for this node's channel + r."""
    feats = [                                                                         
        extract_feature_r(
            load_log_mel(p, n_mels=node.n_mels, channel=node.channel),                
            node.r,                                                                   
        )
        for p in wav_paths                                                            
    ]           
    return np.stack(feats)

                                                                                    
# ── t-SNE panel ──────────────────────────────────────────────────────────
                                                                                    
def _run_tsne(points: np.ndarray, perplexity: int, random_state: int) -> np.ndarray | None:
    if points.shape[0] < perplexity * 3 + 1:
        return None                                                                   
    return TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, init="pca").fit_transform(points)          
                                                                                    
                                                                                    
def _render_tsne_on_ax(ax, warmup_feats, test_feats, test_labels, cfg) -> None:       
    perp  = cfg.get("perplexity", 30)                                                 
    state = cfg.get("random_state", 42)

    all_pts = np.concatenate([warmup_feats, test_feats], axis=0)                      
    proj    = _run_tsne(all_pts, perp, state)
    if proj is None:                                                                  
        ax.text(0.5, 0.5,
                f"t-SNE skipped:\nneed ≥{perp*3+1} samples, got {len(all_pts)}.\n"
                f"Lower latent_plot.perplexity in default.yaml.",                     
                ha="center", va="center", transform=ax.transAxes, fontsize=8)         
        ax.set_xticks([]); ax.set_yticks([])                                          
        return                                                                        
                
    n_warmup  = len(warmup_feats)                                                     
    warmup_2d = proj[:n_warmup]
    test_2d   = proj[n_warmup:]                                                       
    test_lbl  = np.asarray(test_labels)
                                                                                    
    ax.scatter(warmup_2d[:, 0], warmup_2d[:, 1],
                s=14, c=_COL_WARMUP, marker="o", alpha=0.4, label="Warmup (normal)")   
    ax.scatter(test_2d[test_lbl == 0, 0], test_2d[test_lbl == 0, 1],                  
                s=16, c=_COL_NORM, alpha=0.7, label="Test normal")                     
    ax.scatter(test_2d[test_lbl == 1, 0], test_2d[test_lbl == 1, 1],                  
                s=16, c=_COL_ANOM, alpha=0.7, label="Test anomaly")                    
    ax.set_xticks([]); ax.set_yticks([])
                                                                                    
                
def _render_histogram_on_ax(ax, node: Node) -> None:                                  
    scores = np.asarray(node.scores)
    labels = np.asarray(node.labels)
                                                                                    
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, color=_COL_NORM, label="Normal")
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, color=_COL_ANOM, label="Anomaly")
    ax.axvline(node.k, color=_COL_K, ls="--", lw=1.2, label=f"k ({node.k:.2f})")      
    ax.axvline(node.h, color=_COL_H, ls=":",  lw=1.2, label=f"h ({node.h:.2f})")      
    ax.set_xlabel("NLL score"); ax.set_ylabel("clip count")                           
    ax.grid(alpha=0.3)                                                                
                                                                                    
                
# ── Per-node 2-panel figure ──────────────────────────────────────────────           
                
def _render_node_figure(
    node: Node, warmup_feats, test_feats, test_labels, cfg, out_path: Path,
) -> None:                                                                            
    fig, (ax_tsne, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(                                                                     
        f"TWFR-GMM  |  {node.node_id}  r={node.r:.2f}  n_mels={node.n_mels}",         
        fontsize=12, fontweight="bold",
    )                                                                                 
                
    ax_tsne.set_title("Per-clip TWFR (t-SNE)", fontsize=10)                           
    _render_tsne_on_ax(ax_tsne, warmup_feats, test_feats, test_labels, cfg)
    ax_tsne.legend(loc="best", fontsize=7, framealpha=0.9)                            
                                                                                    
    ax_hist.set_title("Score distribution", fontsize=10)                              
    _render_histogram_on_ax(ax_hist, node)                                            
    ax_hist.legend(loc="best", fontsize=7, framealpha=0.9)
                                                                                    
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")                               
    plt.close(fig)                                                                    

                                                                                    
# ── 4x4 grids ────────────────────────────────────────────────────────────

def _render_tsne_grid(rendered, cfg, out_path: Path, title: str) -> None:             
    """rendered: list of (node, warmup_feats, test_feats, test_labels)."""
    n = len(rendered)                                                                 
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols,                                              
                            figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)                      
                
    for ax, (node, wf, tf, tl) in zip(axes.flat, rendered):                           
        _render_tsne_on_ax(ax, wf, tf, tl, cfg)
        ax.set_title(                                                                 
            f"{node.machine_type} {node.machine_id} ch{node.channel}  r={node.r:.2f}",
            fontsize=9,                                                               
        )       
    for ax in axes.flat[n:]:                                                          
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", color=_COL_WARMUP, alpha=0.4, lw=0,
                    markersize=6, label="Warmup"),                                     
        plt.Line2D([], [], marker=".", color=_COL_NORM,  alpha=0.7, lw=0,             
                    markersize=7, label="Test normal"),                                
        plt.Line2D([], [], marker=".", color=_COL_ANOM,  alpha=0.7, lw=0,             
                    markersize=7, label="Test anomaly"),                               
    ]           
    fig.legend(handles=handles, loc="lower center", ncol=3,                           
                fontsize=9, frameon=False)                                             
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")                               
    plt.close(fig)                                                                    

                                                                                    
def _render_score_grid(nodes, out_path: Path, title: str) -> None:
    n = len(nodes)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols,                                              
                            figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)                      
                
    for ax, node in zip(axes.flat, nodes):                                            
        _render_histogram_on_ax(ax, node)
        ax.set_title(                                                                 
            f"{node.machine_type} {node.machine_id} ch{node.channel}",
            fontsize=9,                                                               
        )                                                                             
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:                                                          
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_COL_NORM, alpha=0.6, label="Normal"),
        plt.Rectangle((0, 0), 1, 1, color=_COL_ANOM, alpha=0.6, label="Anomaly"),     
        plt.Line2D([], [], color=_COL_K, ls="--", lw=1.2, label="k"),                 
        plt.Line2D([], [], color=_COL_H, ls=":",  lw=1.2, label="h"),                 
    ]                                                                                 
    fig.legend(handles=handles, loc="lower center", ncol=4,                           
                fontsize=9, frameon=False)                                             
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")                               
    plt.close(fig)                                                                    

                                                                                    
# ── Public entry ─────────────────────────────────────────────────────────

def save_latent_plots(                                                                
    *,
    nodes_by_type,                                                                    
    timelines_by_type,
    config,
    run_dir: Path,
) -> None:
    """Render per-node 2-panel figures + per-channel 4x4 grids."""
    cfg = config.get("latent_plot", {})                                               
    if not cfg.get("enabled", False):                                                 
        return                                                                        
                                                                                    
    subset = cfg.get("node_subset")   # list[int]; empty ⇒ all channels
    wanted = (lambda _ch: True) if not subset else (lambda ch: ch in subset)
                                                                                    
    latent_root = run_dir / "plots" / "latent"
    latent_root.mkdir(parents=True, exist_ok=True)                                    
                                                                                    
    timeline_lookup = {
        (t.machine_type, t.machine_id): t                                             
        for ts in timelines_by_type.values() for t in ts
    }                                                                                 

    # Accumulator so we can render the grids once per selected channel.               
    # Key = channel index, value = list of (node, warmup_feats, test_feats, test_labels).                                                                         
    by_channel: dict[int, list] = {}
                                                                                    
    print("\nRendering latent plots...")
    for mtype, nodes in nodes_by_type.items():                                        
        for node in nodes:
            if not wanted(node.channel):                                              
                continue
            print(f"  {node.node_id}")                                                
            tl = timeline_lookup[(mtype, node.machine_id)]

            warmup_feats = _extract_features(node, tl.warmup_paths)                   
            test_feats   = _extract_features(node, tl.test_paths)
            test_labels  = list(tl.test_labels)                                       
                                                                                    
            ch_dir = latent_root / f"ch{node.channel}"
            ch_dir.mkdir(exist_ok=True)                                                           
            _render_node_figure(
                node, warmup_feats, test_feats, test_labels, cfg,                                 
                ch_dir / f"{node.machine_type}_{node.machine_id}.png",
            )                                                                                     
   
            by_channel.setdefault(node.channel, []).append(                           
                (node, warmup_feats, test_feats, test_labels),
            )                                                                         
                
    # Per-channel 4x4 grids — stored inside plots/latent/ch<N>/.
    for ch, rendered in by_channel.items():                                               
        nodes_only = [r[0] for r in rendered]
        ch_dir = latent_root / f"ch{ch}"                                                  
        ch_dir.mkdir(exist_ok=True)
        _render_tsne_grid(                                                                
            rendered, cfg,                                                                
            ch_dir / "grid_tsne.png",
            f"TWFR-GMM  |  per-clip t-SNE  |  channel {ch}",                              
        )                                                                                 
        _render_score_grid(
            nodes_only,                                                                   
            ch_dir / "grid_scores.png",
            f"TWFR-GMM  |  score distribution  |  channel {ch}",
        )  