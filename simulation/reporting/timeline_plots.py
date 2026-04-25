"""Timeline plots — per-node, per-group (fused), grid, and compare_single.
                                                                                    
Built on the 47d8f94 styling (anomaly injection shading, missed-block brackets,
auto-generated title with detection/FA/AUC/lag) with GMM-era additions:               
- S_t rolling CUSUM accumulator overlay (toggle)                                    
- k (clip-level threshold) and h (alarm height) horizontal lines (toggle)           
- per-group fused-z-score plot with per-node z-score overlays when n_nodes > 1      
- compare_single/ folder: the selected channel's baseline plot for                  
    NL-vs-independent comparison                                                      
"""                                                                                   
                                                                                    
from pathlib import Path                                                              

import matplotlib.pyplot as plt                                                       
from matplotlib.transforms import blended_transform_factory
import numpy as np                                                                    
from sklearn.metrics import roc_auc_score
                                                                                    
from simulation.node.node  import Node                                                
from simulation.node.group import Group
                                                                                    
                                                                                    
# ── Colour palette (pinned to match 47d8f94) ─────────────────────────────
_COL_NORM  = "steelblue"                                                              
_COL_ANOM  = "darkorange"
_COL_K     = "firebrick"                                                              
_COL_H     = "#7b241c"
_COL_BAND  = "lightcoral"                                                             
_COL_S_T   = "#555555"
_COL_DET   = "forestgreen"                                                            
_COL_MISS  = "crimson"
_COL_UNFLAG = "royalblue" 
                                                                                    
                                                                                    
# ── Anomaly-band + bracket helpers ───────────────────────────────────────
                                                                                    
def _anomaly_bands(labels) -> list[tuple[int, int]]:                                  
    """Contiguous runs of label==1, returned as (start, end_inclusive)."""
    runs: list[tuple[int, int]] = []                                                  
    start = None                                                                      
    for i, v in enumerate(labels):                                                    
        if v == 1 and start is None:                                                  
            start = i                                                                 
        elif v != 1 and start is not None:
            runs.append((start, i - 1))                                               
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs                                                                       

                                                                                    
def _bracket_data(labels, state, manual_reset: bool) -> dict | None:
    """Per-band detection outcome driven by the state trace.                                  

    Returns a dict with parallel lists aligned to each anomaly band::                         
                                                                                            
        bands   : [(start, end_inclusive), ...]
        lags    : None if state never went 1 in the band, else                                
                first_alarm_index − band_start  (green bracket width)
        unflags : None in manual_reset mode OR for a missed band OR if the                    
                timeline ends before state returns to 0.  Otherwise the                     
                number of clips after band_end until state[i] == 0                          
                (blue bracket width).                                                       
                                                                                            
    `state` is the post-processed 0/1 trace (Node.state / Group.state), which                 
    in auto mode (`manual_reset=False`) equals the per-clip CUSUM alarm, and                  
    in manual-reset mode stays latched at 1 until `state_reset()` fires at                    
    the next normal-band boundary.                                                            
    """                                                                                       
    state = [int(s) for s in state]                                                           
    bands = _anomaly_bands(labels)                                                            
    if not bands:
        return None
                                                                                            
    lags:    list[int | None] = []
    unflags: list[int | None] = []                                                            
    for (start, end) in bands:
        fired = next((i for i in range(start, end + 1) if state[i] == 1), None)
        lags.append(None if fired is None else fired - start)                                 

        if fired is None or manual_reset or end + 1 >= len(state):                            
            unflags.append(None)
        else:                                                                                 
            first_normal = next(
                (j - (end + 1)
                for j in range(end + 1, len(state)) if state[j] == 0),
                None,                                                                         
            )
            unflags.append(first_normal)                                                      
                
    return {"bands": bands, "lags": lags, "unflags": unflags}                                        
                
                                                                                    
def _draw_brackets(ax, bracket_data: dict) -> None:                                           
    """Draw outcome brackets above anomaly bands.
                                                                                            
    Green  (y=0.97): detection lag — from band start to first alarm.
    Red    (y=0.97): missed block  — no state=1 anywhere in the band.                       
    Blue   (y=0.93): return-to-normal lag — from band end to first                          
                        state=0 clip (only in auto mode, detected bands).                      
    """                                                                                       
    trans = blended_transform_factory(ax.transData, ax.transAxes)                             
    y_detect, y_unflag = 0.97, 0.93                                                           
                
    labeled_det = labeled_miss = labeled_unflag = False                                       
    bands   = bracket_data["bands"]
    lags    = bracket_data["lags"]                                                            
    unflags = bracket_data.get("unflags", [None] * len(bands))
                                                                                            
    for (start, end), lag, unflag in zip(bands, lags, unflags):                               
        if lag is None:                                                                       
            label = None if labeled_miss else "Missed block"                                  
            labeled_miss = True
            ax.plot(
                [start - 0.5, end + 0.5], [y_detect, y_detect],
                color=_COL_MISS, linewidth=3, solid_capstyle="butt",                          
                transform=trans, label=label,
            )                                                                                 
            continue                                                                          

        # Green — detection lag                                                               
        label = None if labeled_det else "Detection lag"
        labeled_det = True                                                                    
        ax.plot(
            [start - 0.5, start + lag + 0.5], [y_detect, y_detect],                           
            color=_COL_DET, linewidth=3, solid_capstyle="butt",
            transform=trans, label=label,                                                     
        )
                                                                                            
        # Blue — return-to-normal lag (auto mode only, caught blocks only)                    
        if unflag is not None:
            label = None if labeled_unflag else "Return-to-normal lag"                        
            labeled_unflag = True
            ax.plot(                                                                          
                [end + 0.5, end + 1 + unflag + 0.5], [y_unflag, y_unflag],
                color=_COL_UNFLAG, linewidth=3, solid_capstyle="butt",                        
                transform=trans, label=label,                                                 
            )                                   
                                                                                    
                                                                                    
# ── Stat helpers ─────────────────────────────────────────────────────────
                                                                                    
def _auc(labels, scores) -> float | None:
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:                                                                 
        return None
                                                                                    
                
def _rate_stats(labels, alarms) -> dict:
    """Per-clip detection and false-alarm rates from CUSUM alarm bools."""
    tp = fn = fp = tn = 0                                                             
    for l, a in zip(labels, alarms):
        if   l == 1 and a:      tp += 1                                               
        elif l == 1 and not a:  fn += 1
        elif l == 0 and a:      fp += 1                                               
        else:                   tn += 1
    return {                                                                          
        "det": tp / (tp + fn) if (tp + fn) else None,
        "fa":  fp / (fp + tn) if (fp + tn) else None,                                 
    }
                                                                                    
                                                                                    
def _format_full_title(
    header: str, labels, scores, alarms, bracket_data,
) -> str:                                                                                     
    """Two-line title.  Detection / false-alarm / AUC driven by alarms
    (clip-level); Lag / Unflag / Miss driven by the state-based brackets."""                  
    rates = _rate_stats(labels, alarms)                                                       
    auc   = _auc(labels, scores)                                                              
                                                                                            
    det_str = f"{rates['det']*100:.1f}%" if rates["det"] is not None else "N/A"               
    fa_str  = f"{rates['fa'] *100:.1f}%" if rates["fa"]  is not None else "N/A"
    auc_str = f"{auc:.4f}"               if auc is not None else "N/A"                        
                                                                                            
    lag_suffix = ""                                                                           
    if bracket_data is not None and bracket_data["bands"]:                                    
        lags    = bracket_data["lags"]
        unflags = bracket_data.get("unflags", [])                                             

        caught   = [l for l in lags if l is not None]                                         
        unflag_n = [u for u in unflags if u is not None]
        missed   = len(lags) - len(caught)                                                    
                                                                                            
        lag_part = f"Lag={np.mean(caught):.1f}" if caught else "Lag=N/A"
        if unflag_n:                                                                          
            lag_part += f"  Unflag={np.mean(unflag_n):.1f}"
        lag_suffix = f"  |  {lag_part}  ({missed}/{len(lags)} missed)"                        
                                                                                            
    return (f"{header}\n"                                                                     
            f"Detection={det_str}  |  False alarm={fa_str}  |  "                              
            f"AUC={auc_str}{lag_suffix}") 

                                                                                    
def _compact_title(head: str, labels, scores, bracket_data) -> str:
    auc = _auc(labels, scores)                                                        
    auc_str = f"{auc:.3f}" if auc is not None else "N/A"
    miss = ""                                                                         
    if bracket_data is not None and bracket_data["bands"]:
        m = sum(1 for l in bracket_data["lags"] if l is None)                         
        miss = f"  Miss={m}/{len(bracket_data['lags'])}"
    return f"{head}  AUC={auc_str}{miss}"                                             
                                                                                    
                                                                                    
# ── Per-node axes ────────────────────────────────────────────────────────           
                
def _draw_node_axes(ax, node: Node, config: dict, compact: bool = False) -> None:             
    """One node's scatter + bands + k/h lines + S_t overlay + brackets + title.
                                                                                            
    Brackets are driven by node.state (the latched/passthrough trace), so
    detection / missed / return-to-normal outcomes honour manual_reset.                       
    Clip-level AUC / detection-rate / FA-rate still come from node.alarms.                    
    """                                                                                       
    t       = np.arange(len(node.scores))                                                     
    scores  = np.asarray(node.scores)                                                         
    labels  = np.asarray(node.labels)                                                         
    alarms  = node.alarms                                                                     
                                                                                            
    plot_cfg = config.get("plot", {})                                                         

    # Anomaly-injection shading + bracket outcomes.                                           
    show_bands = (
        config.get("simulation", {}).get("shuffle_mode", "random")                            
        in ("block_random", "block_fixed")                                                    
    )                                                                                         
    bracket_data = (                                                                          
        _bracket_data(list(labels), node.state, node.manual_reset)                            
        if show_bands else None                                                               
    )
    if bracket_data is not None:                                                              
        for i, (start, end) in enumerate(bracket_data["bands"]):
            ax.axvspan(                                                                       
                start - 0.5, end + 0.5, alpha=0.18, color=_COL_BAND,
                label="Anomaly injection" if i == 0 else None,                                
            )                                                                                 

    # Normal / anomaly clip scatter.                                                          
    nm, am = labels == 0, labels == 1
    ax.scatter(t[nm], scores[nm], s=18, c=_COL_NORM, alpha=0.7,                               
                label="Normal clips", zorder=3)                                                
    ax.scatter(t[am], scores[am], s=18, c=_COL_ANOM, alpha=0.7,                               
                label="Anomaly clips", zorder=3)                                               
                
    # k and h horizontal lines.                                                               
    if plot_cfg.get("show_k_and_h_lines", True):
        ax.axhline(node.k, color=_COL_K, ls="--", lw=1.2,                                     
                    label=f"k ({node.k:.3f})")                                                 
        ax.axhline(node.h, color=_COL_H, ls=":",  lw=1.2,                                     
                    label=f"h ({node.h:.3f})")                                                 
                                                                                            
    # CUSUM accumulator overlay.                                                              
    if plot_cfg.get("show_cusum_accumulator", True):
        ax.plot(t, node.cusum_S, color=_COL_S_T, lw=0.9, alpha=0.6,                           
                label="S_t", zorder=2)                                                        
                                                                                            
    # Brackets sit on top of everything.                                                      
    if bracket_data is not None:
        _draw_brackets(ax, bracket_data)                                                      
                
    header = (f"Node: {node.machine_type} {node.machine_id} "                                 
            f"ch{node.channel}  r={node.r:.2f}")
    if compact:                                                                               
        ax.set_title(                                                                         
            _compact_title(                                                                   
                f"{node.machine_type} {node.machine_id} ch{node.channel}",                    
                node.labels, node.scores, bracket_data,                                       
            ),
            fontsize=9,                                                                       
        )                                                                                     
    else:
        ax.set_title(                                                                         
            _format_full_title(
                header, node.labels, node.scores, alarms, bracket_data,
            ),                                                                                
            fontsize=11,
        )                                                                                     
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Anomaly score (NLL)")
        ax.legend(                                                                            
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=5, fontsize=7, framealpha=0.9, frameon=True,                                 
            handlelength=1.5, columnspacing=1.2,                                              
        )                                                                                     
                                                                                            
    ax.grid(alpha=0.3)                                                                     
                
                                                                                    
# ── Per-group axes (fused z-score) ───────────────────────────────────────

def _draw_group_axes(ax, g: Group, config: dict, compact: bool = False) -> None:
    """Fused-z-score timeline with per-node overlays + brackets + title.                      
                                                                                            
    Brackets are driven by g.state (latched/passthrough), consistent with                     
    per-node plots.  Per-node z-score overlays are collapsed to a single                      
    legend entry and drawn in neutral grey.                                                   
    """                                                                                       
    t       = np.arange(len(g.fused_scores))                                                  
    fused   = np.asarray(g.fused_scores)                                                      
    labels  = np.asarray(g.labels)                                                            
    alarms  = g.alarms
                                                                                            
    plot_cfg = config.get("plot", {})

    # Anomaly bands + bracket outcomes.                                                       
    show_bands = (
        config.get("simulation", {}).get("shuffle_mode", "random")                            
        in ("block_random", "block_fixed")
    )                                                                                         
    bracket_data = (
        _bracket_data(list(labels), g.state, g.manual_reset)                                  
        if show_bands else None
    )
    if bracket_data is not None:
        for i, (start, end) in enumerate(bracket_data["bands"]):                              
            ax.axvspan(
                start - 0.5, end + 0.5, alpha=0.18, color=_COL_BAND,                          
                label="Anomaly injection" if i == 0 else None,
            )

    # Per-node z-score overlays (dimmed, single legend entry).                                
    if plot_cfg.get("show_per_node", True):
        n_count   = len(g.nodes)                                                              
        overlay_c = "#888888"                                                                 
        for i, n in enumerate(g.nodes):                                                       
            z = (np.asarray(n.scores) - n.mu_val) / max(n.sigma_val, 1e-8)                    
            label = (f"per-node z-score (n={n_count})" if i == 0 else None)                   
            ax.plot(np.arange(len(z)), z, alpha=0.35, lw=0.9,
                    color=overlay_c, label=label, zorder=1)                                   
                
    # Fused scatter (primary).                                                                
    if plot_cfg.get("show_fused", True):
        nm, am = labels == 0, labels == 1                                                     
        ax.scatter(t[nm], fused[nm], s=20, c=_COL_NORM, alpha=0.85,
                    label="Normal (fused)", zorder=3)                                          
        ax.scatter(t[am], fused[am], s=20, c=_COL_ANOM, alpha=0.85,
                    label="Anomaly (fused)", zorder=3)                                         
                
    # k / h in fused z-score space.                                                           
    if plot_cfg.get("show_k_and_h_lines", True):
        ax.axhline(g.k, color=_COL_K, ls="--", lw=1.2, label=f"k ({g.k:.3f})")                
        ax.axhline(g.h, color=_COL_H, ls=":",  lw=1.2, label=f"h ({g.h:.3f})")                
                                                                                            
    # S_t overlay.                                                                            
    if plot_cfg.get("show_cusum_accumulator", True):                                          
        ax.plot(t, g.cusum_S, color=_COL_S_T, lw=0.9, alpha=0.6,
                label="S_t", zorder=2)                                                        

    # Brackets on top.                                                                        
    if bracket_data is not None:
        _draw_brackets(ax, bracket_data)                                                      

    head_full = (                                                                             
        f"Group: {g.machine_type} {g.machine_id}  n={len(g.nodes)}  "
        f"w={np.round(g.w, 2).tolist()}"                                                      
    )                                                                                         
    head_compact = f"{g.machine_type} {g.machine_id}  n={len(g.nodes)}"                       
                                                                                            
    if compact: 
        ax.set_title(                                                                         
            _compact_title(head_compact, list(labels), list(fused),
                            bracket_data),                                                     
            fontsize=9,
        )                                                                                     
    else:       
        ax.set_title(
            _format_full_title(head_full, list(labels), list(fused),
                                alarms, bracket_data),                                         
            fontsize=11,
        )                                                                                     
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Fused z-score")
        ax.legend(                                                                            
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=5, fontsize=7, framealpha=0.9, frameon=True,                                 
            handlelength=1.5, columnspacing=1.2,                                              
        )
                                                                                            
    ax.grid(alpha=0.3)                                                             

                                                                                    
# ── Grid ────────────────────────────────────────────────────────────────

def _render_grid(views, config: dict, out_path: Path, title: str) -> None:            
    """4-col grid; rows computed from view count.  Does not share y-axes."""
    n_views = len(views)                                                              
    cols = 4                                                                          
    rows = (n_views + cols - 1) // cols                                               
    fig, axes = plt.subplots(rows, cols,                                              
                            figsize=(5 * cols, 3 * rows),
                            sharex="col", squeeze=False)                            
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
                                                                                    
    for ax, (_, view) in zip(axes.flat, views):                                       
        if isinstance(view, Group):                                                   
            _draw_group_axes(ax, view, config, compact=True)                          
        else:   
            _draw_node_axes(ax, view, config, compact=True)
    for ax in axes.flat[n_views:]:                                                    
        ax.axis("off")                                                                
                                                                                    
    fig.supxlabel("Timestep", fontsize=11)                                            
    fig.supylabel("Score", fontsize=11)
    fig.tight_layout()                                                                
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)                                                                    
                                                                                    

# ── Public entry ─────────────────────────────────────────────────────────           
                
def save_plots(                                                                       
    *,
    nodes_by_type:  dict[str, list[Node]],                                            
    groups_by_type: dict[str, list[Group]],                                           
    config:         dict,
    run_dir:        Path,                                                             
) -> None:      
    """Write per-node, per-group, grid.png, and compare_single/ plots."""             
    plots_dir = run_dir / "plots"                                                     
    plots_dir.mkdir(exist_ok=True)
                                                                                    
    n_nodes    = config["n_nodes"]                                                    
    has_groups = n_nodes > 1                                                          
    gmm        = config["gmm"]                                                        
    sep_desc   = (f"TWFR-GMM ({gmm['n_components']} components, diag cov, "           
                f"n_mels={gmm['n_mels']})")                                         
                                                                                    
    # Per-node full-size plots — grouped into plots/ch<N>/ folders.
    for nodes in nodes_by_type.values():                                                  
        for n in nodes:
            ch_dir = plots_dir / f"ch{n.channel}"                                         
            ch_dir.mkdir(exist_ok=True)
                                                                                            
            fig, ax = plt.subplots(figsize=(11, 4.5))                                     
            fig.suptitle(sep_desc, fontsize=12, fontweight="bold")
            _draw_node_axes(ax, n, config, compact=False)                                 
            fig.tight_layout()                                                            
            fig.savefig(
                ch_dir / f"{n.machine_type}_{n.machine_id}.png",                          
                dpi=150, bbox_inches="tight",
            )                                                                             
            plt.close(fig)
                                                                                    
    # Per-group (fused) full-size plots — grouped into plots/fused/.
    if has_groups:                                                                        
        fused_dir = plots_dir / "fused"
        fused_dir.mkdir(exist_ok=True)                                                    
        for groups in groups_by_type.values():                                            
            for g in groups:
                fig, ax = plt.subplots(figsize=(11, 4.5))                                 
                fig.suptitle(f"{sep_desc}  |  node learning fusion",
                            fontsize=12, fontweight="bold")                              
                _draw_group_axes(ax, g, config, compact=False)
                fig.tight_layout()                                                        
                fig.savefig(                                                              
                    fused_dir / f"{g.group_id}.png",
                    dpi=150, bbox_inches="tight",                                         
                )   
                plt.close(fig)                                                      

    # Primary grid — fused if n_nodes > 1 else per-node                               
    primary_views: list[tuple[str, object]] = []
    if has_groups:                                                                    
        for groups in groups_by_type.values():
            for g in groups:                                                          
                primary_views.append((g.group_id, g))
        title = f"{sep_desc}  |  fused (n={n_nodes}) across all machines"             
    else:                                                                             
        for nodes in nodes_by_type.values():
            for n in nodes:                                                           
                primary_views.append((n.node_id, n))
        title = f"{sep_desc}  |  single-node across all machines"                     
    _render_grid(primary_views, config, plots_dir / "grid.png", title)
                                                                                    
    # Compare-single (baseline) views when n_nodes > 1                                
    if has_groups:                                                                    
        idx = config.get("plot", {}).get("compare_node_idx", 0)                       
                                                                                    
        # Grid view
        compare_views: list[tuple[str, object]] = []                                  
        for groups in groups_by_type.values():
            for g in groups:                                                          
                if idx < len(g.nodes):
                    compare_views.append((g.group_id, g.nodes[idx]))                  
        _render_grid(
            compare_views, config,                                                    
            plots_dir / "grid_compare_single.png",
            f"{sep_desc}  |  independent single-node (channel {idx}) — baseline",     
        )                                                                             

        # Per-machine full-size views                                                 
        cmp_dir = plots_dir / "compare_single"
        cmp_dir.mkdir(exist_ok=True)                                                  
        for groups in groups_by_type.values():                                        
            for g in groups:
                if idx < len(g.nodes):                                                
                    chosen = g.nodes[idx]
                    fig, ax = plt.subplots(figsize=(11, 4.5))                         
                    fig.suptitle(f"{sep_desc}  |  independent ch{chosen.channel}",
                                fontsize=12, fontweight="bold")                      
                    _draw_node_axes(ax, chosen, config, compact=False)
                    fig.tight_layout()                                                
                    fig.savefig(cmp_dir / f"{g.group_id}_ch{chosen.channel}.png", dpi=150, bbox_inches="tight")                                                                              
                    plt.close(fig)