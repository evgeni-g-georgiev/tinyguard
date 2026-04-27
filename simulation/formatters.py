"""Terminal output — per-timestep live line + final result tables.

Three result sections:
1. Per-node (independent)        — clip + block metrics + per-type means
2. Per-group (node-learning)     — same shape, only when len(channels) > 1
3. NL-vs-independent comparison  — fused AUC vs mean-of-nodes AUC

result_lines(...) returns list[str] used by both print_results (terminal)
and reporting.results.save_results (summary.txt).
"""             
                                                                                    
from collections import defaultdict
                                                                                    
import numpy as np

from simulation.lockstep   import TimestepResult                                      
from simulation.metrics    import (
    ClipMetrics, BlockMetrics,
    clip_metrics, block_metrics,
    node_clip_metrics, node_block_metrics,
    group_clip_metrics, group_block_metrics,
)
from simulation.node.node  import Node                                                
from simulation.node.group import Group                                               

                                                                                    
_WIDTH = 88     
_SEP   = "─" * _WIDTH
_HEAD  = "=" * _WIDTH                                                                 

                                                                                    
# ── Per-timestep live line (unchanged glyph idea) ────────────────────────
_GLYPH = {(0, False): "·", (1, True): "✓", (1, False): "✗", (0, True): "!"}           
                                                                                    
                                                                                    
def format_step(step: TimestepResult, machine_types: list[str]) -> str:               
    by_type: dict[str, list[str]] = {mt: [] for mt in machine_types}                  
    n_anom = n_fired = 0                                                              
    for r in step.node_results:                                                       
        by_type[r.machine_type].append(_GLYPH[(r.label, r.alarm)])                    
        if r.label == 1: n_anom  += 1                                                 
        if r.alarm:      n_fired += 1                                                 
                                                                                    
    chunks = [f"{mt[:3]}:{''.join(by_type[mt])}" for mt in machine_types]             
    line   = f"  t={step.timestep:3d}  |  {'  '.join(chunks)}"
    if step.group_results:                                                            
        g_glyphs = [_GLYPH[(g.label, g.alarm)] for g in step.group_results]
        line += f"  ||  fused:{''.join(g_glyphs)}"                                    
    n_total = len(step.node_results)                                                  
    line += f"  |  anom {n_anom}/{n_total}  fired {n_fired}/{n_total}"                
    return line                                                                       
                                                                                    
                
# ── Row formatters ───────────────────────────────────────────────────────           
                
def _fmt_clip_row(label: str, m: ClipMetrics) -> str:                                 
    auc = f"{m.auc:.4f}" if m.auc is not None else " N/A  "
    return (f"  {label:<24}  AUC={auc}  "                                             
            f"|  TP={m.tp:3d} TN={m.tn:3d} FP={m.fp:3d} FN={m.fn:3d}  "
            f"|  P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")                 
                
                                                                                    
def _fmt_block_row(indent_width: int, m: BlockMetrics) -> str:
    lag    = f"{m.mean_lag:4.1f}"    if m.mean_lag    is not None else " N/A"         
    unflag = f"{m.mean_unflag:4.1f}" if m.mean_unflag is not None else " N/A"         
    pad    = " " * indent_width
    return (f"{pad}block              "                                               
            f"|  bTP={m.block_tp:3d} bTN={m.block_tn:3d} "                            
            f"bFP={m.block_fp:3d} bFN={m.block_fn:3d}  "                              
            f"|  bP={m.block_precision:.3f} bR={m.block_recall:.3f} "                 
            f"bF1={m.block_f1:.3f}  "
            f"|  Lag={lag}  Unflag={unflag}  "                                        
            f"Miss={m.missed_blocks:2d}/{m.total_blocks:2d}")
                                                                                    
                                                                                    
def _fmt_clip_mean(label: str, metrics: list[ClipMetrics]) -> str:                    
    aucs = [m.auc for m in metrics if m.auc is not None]                              
    auc = f"{np.mean(aucs):.4f}" if aucs else " N/A  "
    return (f"  {label:<24}  mean   AUC={auc}  "                                      
            f"P={np.mean([m.precision for m in metrics]):.3f}  "
            f"R={np.mean([m.recall    for m in metrics]):.3f}  "                      
            f"F1={np.mean([m.f1       for m in metrics]):.3f}")
                                                                                    
                
def _fmt_block_mean(label: str, metrics: list[BlockMetrics]) -> str:                  
    lags    = [m.mean_lag    for m in metrics if m.mean_lag    is not None]
    unflags = [m.mean_unflag for m in metrics if m.mean_unflag is not None]           
    lag    = f"{np.mean(lags):4.1f}"    if lags    else " N/A"
    unflag = f"{np.mean(unflags):4.1f}" if unflags else " N/A"                        
    miss   = sum(m.missed_blocks for m in metrics)                                    
    total  = sum(m.total_blocks  for m in metrics)                                    
    return (f"  {label:<24}  block  "                                                 
            f"|  bP={np.mean([m.block_precision for m in metrics]):.3f}  "
            f"bR={np.mean([m.block_recall for m in metrics]):.3f}  "                  
            f"bF1={np.mean([m.block_f1 for m in metrics]):.3f}  "
            f"|  Lag={lag}  Unflag={unflag}  Miss={miss:3d}/{total:3d}")              
                                                                                    
                                                                                    
# ── Section builders (return list[str]) ──────────────────────────────────           
                
def _lines_per_node(nodes_by_type: dict[str, list[Node]]) -> list[str]:               
    out = ["", "Per-node (independent)", _SEP]
    type_clip:  dict[str, list[ClipMetrics]]  = defaultdict(list)                     
    type_block: dict[str, list[BlockMetrics]] = defaultdict(list)                     
                                                                                    
    for mtype, nodes in nodes_by_type.items():                                        
        for n in nodes:
            cm = node_clip_metrics(n)                                                 
            bm = node_block_metrics(n)
            if cm is None:
                out.append(f"  {n.node_id}: AUC = N/A (single class)")                
                continue                                                              
            type_clip[mtype].append(cm)                                               
            out.append(_fmt_clip_row(n.node_id, cm))                                  
            if bm is not None:                                                        
                type_block[mtype].append(bm)
                out.append(_fmt_block_row(4 + len(n.node_id), bm))                    
                                                                                    
    out.append(_SEP)
    for mtype in nodes_by_type:                                                       
        if mtype in type_clip:                                                        
            out.append(_fmt_clip_mean(mtype, type_clip[mtype]))
            if mtype in type_block:                                                   
                out.append(_fmt_block_mean(mtype, type_block[mtype]))
                                                                                    
    all_clip  = [m for ms in type_clip.values()  for m in ms]
    all_block = [m for ms in type_block.values() for m in ms]                         
    if all_clip:                                                                      
        out.append(_SEP)
        out.append(_fmt_clip_mean("Overall", all_clip))                               
    if all_block:                                                                     
        out.append(_fmt_block_mean("Overall", all_block))
    return out                                                                        
                
                                                                                    
def _lines_per_group(groups_by_type: dict[str, list[Group]]) -> list[str]:
    n_groups = sum(len(gs) for gs in groups_by_type.values())                         
    if n_groups == 0:                                                                 
        return []
                                                                                    
    out = ["", "Per-group (node-learning fused)", _SEP]                               
    type_clip:  dict[str, list[ClipMetrics]]  = defaultdict(list)
    type_block: dict[str, list[BlockMetrics]] = defaultdict(list)                     
                                                                                    
    for mtype, groups in groups_by_type.items():
        for g in groups:                                                              
            cm = group_clip_metrics(g)
            bm = group_block_metrics(g)
            w_str = np.round(g.w, 2).tolist()                                         
            label = f"{g.group_id} (n={len(g.nodes)} w={w_str})"                      
            if cm is None:                                                            
                out.append(f"  {label}: AUC = N/A (single class)")                    
                continue
            type_clip[mtype].append(cm)                                               
            out.append(_fmt_clip_row(label, cm))
            if bm is not None:                                                        
                type_block[mtype].append(bm)
                out.append(_fmt_block_row(4 + len(label), bm))                        
                
    out.append(_SEP)                                                                  
    for mtype in groups_by_type:
        if mtype in type_clip:                                                        
            out.append(_fmt_clip_mean(f"{mtype} fused", type_clip[mtype]))
            if mtype in type_block:                                                   
                out.append(_fmt_block_mean(f"{mtype} fused", type_block[mtype]))
                                                                                    
    all_clip  = [m for ms in type_clip.values()  for m in ms]                         
    all_block = [m for ms in type_block.values() for m in ms]
    if all_clip:                                                                      
        out.append(_SEP)
        out.append(_fmt_clip_mean("Overall fused", all_clip))
    if all_block:                                                                     
        out.append(_fmt_block_mean("Overall fused", all_block))
    return out                                                                        
                
                                                                                    
def _lines_nl_vs_independent(
    nodes_by_type:  dict[str, list[Node]],                                            
    groups_by_type: dict[str, list[Group]],
) -> list[str]:                                                                       
    n_groups = sum(len(gs) for gs in groups_by_type.values())
    if n_groups == 0:                                                                 
        return []

    out = [""]                                                                        
    out.append("Node-learning vs independent  (mean AUC of peer nodes vs fused AUC)")
    hdr = (f"  {'Machine':<16}  {'n':>3}  {'mean AUC':>9}  "                          
            f"{'fused AUC':>10}  {'Δ':>7}  {'fused alarms':>13}")                      
    sep = "─" * len(hdr)                                                              
    out.append(sep); out.append(hdr); out.append(sep)                                 
                                                                                    
    mean_aucs:  list[float] = []                                                      
    fused_aucs: list[float] = []
    for mtype, groups in groups_by_type.items():
        for g in groups:
            per_node_aucs = []
            for n in g.nodes:
                cm = node_clip_metrics(n)
                if cm is not None and cm.auc is not None:
                    per_node_aucs.append(cm.auc)                                      
            gm = group_clip_metrics(g)
            mean_auc  = float(np.mean(per_node_aucs)) if per_node_aucs else float("nan")                                                                          
            fused_auc = gm.auc if (gm and gm.auc is not None) else float("nan")
            delta     = fused_auc - mean_auc                                          
            out.append(f"  {g.group_id:<16}  {len(g.nodes):>3d}  "
                        f"{mean_auc:>9.3f}  {fused_auc:>10.3f}  "                      
                        f"{delta:>+7.3f}  {sum(g.alarms):>13d}")                       
            if not np.isnan(mean_auc):  mean_aucs.append(mean_auc)                    
            if not np.isnan(fused_auc): fused_aucs.append(fused_auc)                  
                                                                                    
    out.append(sep)                                                                   
    if mean_aucs and fused_aucs:                                                      
        d = float(np.mean(fused_aucs) - np.mean(mean_aucs))
        out.append(f"  {'Overall':<16}        "                                       
                    f"{np.mean(mean_aucs):>9.3f}  {np.mean(fused_aucs):>10.3f}  "
                    f"{d:>+7.3f}")                                                    
    out.append(sep)                                                                   
    return out                                                                        
                                                                                    
                
# ── Public API ───────────────────────────────────────────────────────────

def result_lines(                                                                     
    nodes_by_type:  dict[str, list[Node]],
    groups_by_type: dict[str, list[Group]],                                           
) -> list[str]: 
    """All three sections, ready to print() or write() line by line."""               
    out = ["", _HEAD, "Results", _HEAD]                                               
    out += _lines_per_node(nodes_by_type)                                             
    out += _lines_per_group(groups_by_type)                                           
    out += _lines_nl_vs_independent(nodes_by_type, groups_by_type)                    
    out.append(_HEAD)                                                                 
    out.append("")
    return out                                                                        
                
                                                                                    
def print_results(
    nodes_by_type:  dict[str, list[Node]],
    groups_by_type: dict[str, list[Group]],
) -> None:
    for line in result_lines(nodes_by_type, groups_by_type):
        print(line)


# ── Baseline summary table (terminal-only) ───────────────────────────────
#
# Compact paper-style summary printed at the very end of a run so the
# headline numbers are easy to copy into a report. Reads from the same
# per-node clip / block metrics the main results section uses, but
# aggregates per machine type and emits a fixed-width table.

_DISPLAY_NAMES = {
    "fan":    "Fan",
    "valve":  "Valve",
    "pump":   "Pump",
    "slider": "Slide rail",
}

_BL_LABEL_W   = 12   # "Slide rail" fits in 10
_BL_NUM_W     = 7    # " 0.075 ", "  —   "
_BL_DELAY_W   = 8    # "  0.0  "
_BL_AUC_W     = 7    # "0.9962"
_BL_PREC_W    = 9    # "  0.992  "


def _bl_cell(val, places: int = 3, width: int = _BL_NUM_W) -> str:
    """Centre a numeric metric in `width` chars; render `None` as an em dash."""
    if val is None:
        return "—".center(width)
    return f"{val:.{places}f}".center(width)


def _bl_combine(
    clips:  list[ClipMetrics  | None],
    blocks: list[BlockMetrics | None],
) -> dict:
    """Mean of each table column across an iterable of (clip, block) pairs.

    Shared by the single-node and fused-group aggregators so both views fill
    the same row schema.
    """
    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    valid_clip  = [m for m in clips  if m is not None]
    valid_block = [m for m in blocks if m is not None]

    fa_rates = []
    for m in valid_block:
        denom = m.block_fp + m.block_tn
        if denom > 0:
            fa_rates.append(m.block_fp / denom)

    return {
        "det_rate":  _mean_or_none([m.block_recall for m in valid_block]),
        "fa_rate":   _mean_or_none(fa_rates),
        "det_delay": _mean_or_none([m.mean_lag     for m in valid_block]),
        "auc":       _mean_or_none([m.auc          for m in valid_clip]),
        "recall":    _mean_or_none([m.recall       for m in valid_clip]),
        "precision": _mean_or_none([m.precision    for m in valid_clip]),
        "f1":        _mean_or_none([m.f1           for m in valid_clip]),
    }


def _bl_aggregate_node(nodes: list[Node], channel: int) -> dict | None:
    """Per-type aggregate from one channel's per-node metrics."""
    ch_nodes = [n for n in nodes if n.channel == channel]
    if not ch_nodes:
        return None
    clips  = [node_clip_metrics(n)  for n in ch_nodes]
    blocks = [node_block_metrics(n) for n in ch_nodes]
    return _bl_combine(clips, blocks)


def _bl_aggregate_group(groups: list[Group]) -> dict | None:
    """Per-type aggregate from per-machine fused group metrics.

    Uses g.alarms for clip metrics (per-clip CUSUM fires) and g.state for
    block metrics (operator-facing flag trace). Mirrors the per-node
    convention; under manual_reset=False both signals coincide.
    """
    if not groups:
        return None
    clips  = [clip_metrics(g.labels, g.fused_scores, g.alarms) for g in groups]
    blocks = [block_metrics(g.labels, g.state)                 for g in groups]
    return _bl_combine(clips, blocks)


def _bl_row(label: str, r: dict) -> str:
    """One body row of the baseline table."""
    return (
        f"  {label:<{_BL_LABEL_W}} │ "
        f"{_bl_cell(r['det_rate'],  places=3)} │ "
        f"{_bl_cell(r['fa_rate'],   places=3)} │ "
        f"{_bl_cell(r['det_delay'], places=1, width=_BL_DELAY_W)} │ "
        f"{_bl_cell(r['auc'],       places=4, width=_BL_AUC_W)} │ "
        f"{_bl_cell(r['recall'],    places=3)} │ "
        f"{_bl_cell(r['precision'], places=3, width=_BL_PREC_W)} │ "
        f"{_bl_cell(r['f1'],        places=3)}"
    )


def _baseline_table_lines(
    nodes_by_type:  dict[str, list[Node]],
    groups_by_type: dict[str, list[Group]],
    config:         dict,
) -> list[str]:
    """Render the baseline table as a list of lines.

    With one configured channel the rows show per-node metrics for that
    channel. With more than one channel the rows show per-machine fused
    group metrics, which is the headline number node learning produces.
    """
    channels = config.get("channels") or [0]
    snr      = config.get("snr", "?")
    machine_types = config.get("data", {}).get(
        "machine_types", list(nodes_by_type)
    )

    snr_label = f"{snr.replace('dB', ' dB')} SNR"

    use_fused = len(channels) > 1 and any(
        groups_by_type.get(mt) for mt in machine_types
    )
    view_label = (
        f"fused, n={len(channels)}" if use_fused
        else f"channel {channels[0]}"
    )

    rows: list[tuple[str, dict]] = []
    for mt in machine_types:
        if use_fused:
            agg = _bl_aggregate_group(groups_by_type.get(mt, []))
        else:
            if mt not in nodes_by_type:
                continue
            agg = _bl_aggregate_node(nodes_by_type[mt], channels[0])
        if agg is not None:
            rows.append((_DISPLAY_NAMES.get(mt, mt.capitalize()), agg))

    if not rows:
        return [
            "",
            f"  (no data for view '{view_label}'; baseline table skipped)",
            "",
        ]

    # Average across machine types.
    def _avg(key):
        vals = [r[key] for _, r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    avg_row = {
        k: _avg(k)
        for k in ("det_rate", "fa_rate", "det_delay",
                  "auc", "recall", "precision", "f1")
    }

    # Header construction. Two-tier: spanning groups on top, columns below.
    # Compute total inner width consumed by each metric group so the upper
    # banner can be centred over its sub-columns.
    sep = " │ "
    epi_inner = _BL_NUM_W + len(sep) + _BL_NUM_W + len(sep) + _BL_DELAY_W
    clip_inner = (_BL_AUC_W + len(sep) + _BL_NUM_W + len(sep)
                  + _BL_PREC_W + len(sep) + _BL_NUM_W)

    label_pad = " " * (2 + _BL_LABEL_W + 1)   # "  Setting       "
    epi_banner  = "Episode-level metrics".center(epi_inner)
    clip_banner = "Clip-level metrics".center(clip_inner)
    hdr_top = f"{label_pad}│ {epi_banner} │ {clip_banner}"

    hdr_bot = (
        f"  {'Setting':<{_BL_LABEL_W}} │ "
        f"{'Det.rate'.center(_BL_NUM_W)} │ "
        f"{'FA rate'.center(_BL_NUM_W)} │ "
        f"{'Det.delay'.center(_BL_DELAY_W)} │ "
        f"{'AUC'.center(_BL_AUC_W)} │ "
        f"{'Recall'.center(_BL_NUM_W)} │ "
        f"{'Precision'.center(_BL_PREC_W)} │ "
        f"{'F1'.center(_BL_NUM_W)}"
    )

    rule = "─" * len(hdr_bot)

    out: list[str] = []
    out.append("")
    out.append(rule)
    out.append(
        f"  Baseline evaluation results by machine type ({view_label})"
    )
    out.append(rule)
    out.append(hdr_top)
    out.append(hdr_bot)
    out.append(rule)
    out.append(f"  {snr_label}")
    for label, r in rows:
        out.append(_bl_row(label, r))
    out.append(rule)
    out.append(_bl_row("Average", avg_row))
    out.append(rule)
    out.append("")
    return out


def print_baseline_table(
    nodes_by_type:  dict[str, list[Node]],
    groups_by_type: dict[str, list[Group]],
    config:         dict,
) -> None:
    """Terminal-only paper-style summary table; not written to summary.txt."""
    for line in _baseline_table_lines(nodes_by_type, groups_by_type, config):
        print(line)