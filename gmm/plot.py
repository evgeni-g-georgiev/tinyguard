"""
plot.py — Timeline plots for TWFR-GMM anomaly detection results.

Produces per-machine scatter plots with an identical visual style to
inference/run.py::plot_machine().  The same colour palette, axis labels,
annotation arrows, and grid settings are used so results from the two
pipelines can be compared side-by-side.

Works for all three detector variants:
  Node A       — y-axis label "Anomaly score (NLL)", subtitle "r=1.0 mean pooling"
  Node B       — y-axis label "Anomaly score (NLL)", subtitle "r=0.5 GWRP"
  NodeLearning — y-axis label "Anomaly score (fused z-score)", subtitle "r=1.0/0.5 fusion"

Dynamic labels are injected by train.py via result["r_desc"] and
result["score_label"] keys.  This module reads those keys and falls back to
sensible defaults if absent, so it never needs to know which variant produced
the result.

Rolling-mean overlay: if events contain a "rolling_mean" field, a grey line is
overlaid on the scatter.  This mirrors the Serial diagnostic output in
tinyml_gmm.ino and aids visual inspection without affecting detection logic.

This module has no repo-level imports — all required data is passed as arguments,
making it trivially usable in isolation.

Dependencies: matplotlib, numpy, pathlib, sklearn.metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


# ── Color constants (pinned to match inference/run.py) ────────────────────────
_COL_NORMAL    = "#5b9bd5"   # blue — normal clip scatter points
_COL_ANOMALY   = "#f4a261"   # orange — anomaly clip scatter points
_COL_SHADING   = "#fde8e8"   # light red — anomaly injection window background
_COL_THRESHOLD = "#c0392b"   # dark red — threshold line, annotations


# ── Public API ────────────────────────────────────────────────────────────────

def plot_machine(
    result: dict,
    mtype: str,
    mid: str,
    out_dir: Path,
) -> str:
    """Generate and save a per-machine timeline anomaly score plot.

    Replicates the visual style of ``inference/run.py::plot_machine()``
    exactly. The x-axis shows wall-clock time in seconds; each dot represents
    one 10 s clip. Blue dots are normal clips, orange are anomaly clips. The
    dashed red line is the detection threshold. Anomaly injection windows are
    shaded in light red, and detection events are annotated with arrows.

    Time encoding: ``result["events"]`` stores ``t`` in minutes (matching the
    convention in inference/run.py). The local helper ``ts(e) = e["t"] * 60``
    converts back to seconds for plotting.

    Parameters
    ----------
    result : dict
        Output of :func:`evaluate.evaluate_machine`. Must contain:
        ``events``, ``segments``, ``threshold``, ``round_results``,
        ``n_rounds``. May also contain ``r`` and ``n_components`` for the
        figure subtitle.
    mtype : str
        Machine type, e.g. ``'fan'``.
    mid : str
        Machine ID, e.g. ``'id_00'``.
    out_dir : Path
        Directory where the PNG will be written. Must already exist.

    Returns
    -------
    out_path : str
        Absolute path of the saved PNG file.
    """
    events        = result["events"]
    segments      = result["segments"]
    threshold     = result["threshold"]
    round_results = result["round_results"]
    n_rounds      = result["n_rounds"]
    n_components  = result.get("n_components", "?")
    # CUSUM parameters — read from the first round_result (same for all rounds).
    first_rr  = round_results[0] if round_results else {}
    cusum_h   = first_rr.get("cusum_h", float("nan"))
    cusum_k   = first_rr.get("cusum_k", float("nan"))

    # Converts stored minutes back to seconds for x-axis (matches inference/run.py)
    ts = lambda e: e["t"] * 60

    normal_evts  = [e for e in events if e["phase"] == "normal"]
    anomaly_evts = [e for e in events if e["phase"] == "anomaly"]

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n_detected      = sum(1 for rr in round_results if rr["detected"])
    detection_pct   = 100 * n_detected / n_rounds if n_rounds > 0 else 0.0
    n_fps           = sum(rr["n_false_pos"] for rr in round_results)
    n_norm_mon      = sum(rr["n_normal_clips"] for rr in round_results)
    false_alarm_pct = 100 * n_fps / n_norm_mon if n_norm_mon > 0 else 0.0

    try:
        auc_labels = [0] * len(normal_evts) + [1] * len(anomaly_evts)
        auc_scores = [e["score"] for e in normal_evts + anomaly_evts]
        auc = roc_auc_score(auc_labels, auc_scores)
    except Exception:
        # Guard against degenerate cases (single class present)
        auc = float("nan")

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade anomaly injection windows
    anom_patched = False
    for seg in segments:
        if seg["phase"] == "anomaly":
            ax.axvspan(
                seg["t_start"] * 60, seg["t_end"] * 60,
                color=_COL_SHADING, alpha=0.8, zorder=0,
                label="Anomaly injection" if not anom_patched else None,
            )
            anom_patched = True

    # Scatter plots
    if normal_evts:
        ax.scatter(
            [ts(e) for e in normal_evts], [e["score"] for e in normal_evts],
            color=_COL_NORMAL, s=20, alpha=0.7, zorder=2, label="Normal clips",
        )
    if anomaly_evts:
        ax.scatter(
            [ts(e) for e in anomaly_evts], [e["score"] for e in anomaly_evts],
            color=_COL_ANOMALY, s=20, alpha=0.8, zorder=2, label="Anomaly clips",
        )

    # Rolling mean overlay (diagnostic only — mirrors Serial output in C++).
    # Only drawn when events contain the "rolling_mean" field.
    if events and "rolling_mean" in events[0]:
        rm_t      = [ts(e) for e in events]
        rm_scores = [e["rolling_mean"] for e in events]
        ax.plot(rm_t, rm_scores, color="#555555", linewidth=0.9, alpha=0.5,
                zorder=3, label=f"Rolling mean (w={5})")

    # Detection threshold line
    ax.axhline(
        threshold, color=_COL_THRESHOLD, linestyle="--", linewidth=1.3,
        label=f"Threshold ({threshold:.3f})", zorder=3,
    )

    # ── Per-round detection annotations ───────────────────────────────────────
    for rr in round_results:
        anom_evts_r = [e for e in events if e["phase"] == "anomaly" and e["round"] == rr["round"]]
        anom_seg    = next(s for s in segments if s["phase"] == "anomaly" and s["round"] == rr["round"])

        if rr["detected"] and rr["detection_idx"] is not None and rr["detection_idx"] < len(anom_evts_r):
            idx          = rr["detection_idx"]
            detect_t_s   = ts(anom_evts_r[idx])
            detect_score = anom_evts_r[idx]["score"]
            delay        = rr["detection_delay_secs"]
            ax.annotate(
                f"Detected!\n({delay}s delay)",
                xy=(detect_t_s, detect_score),
                xytext=(30, 25), textcoords="offset points",
                color=_COL_THRESHOLD, fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_COL_THRESHOLD, lw=1.2),
                zorder=5,
            )
        else:
            mid_t_s = (anom_seg["t_start"] + anom_seg["t_end"]) / 2 * 60
            ax.text(
                mid_t_s, threshold * 0.6, "Not detected",
                ha="center", va="center", fontsize=8,
                color=_COL_THRESHOLD, style="italic", alpha=0.85,
            )

    # ── Axes formatting ───────────────────────────────────────────────────────
    score_label = result.get("score_label", "Anomaly score  (NLL)")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel(score_label, fontsize=10)
    ax.set_xlim(left=0)
    ax.margins(y=0.2)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    node_label = f"{mtype.capitalize()} {mid.replace('_', ' ')}"
    ax.set_title(
        f"Node: {node_label}\n"
        f"Detection={detection_pct:.0f}%  |  "
        f"False alarm={false_alarm_pct:.1f}%  |  "
        f"AUC={auc:.3f}",
        fontsize=9, color="#333333",
    )
    r_desc = result.get("r_desc", f"r={result.get('r', '?')} mean pooling")
    fig.suptitle(
        "TinyML Deployment Simulation — Anomalous Sound Detection\n"
        f"Arduino Nano 33 BLE  |  "
        f"GMM ({n_components} component{'s' if n_components != 1 else ''}, "
        f"{r_desc})  |  "
        f"CUSUM k={cusum_k:.3f}  h={cusum_h:.3f}",
        fontsize=11, fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    out_path = out_dir / f"{mtype}_{mid}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)
