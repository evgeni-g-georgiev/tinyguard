"""Per-machine timeline scatter plot of anomaly scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from gmm.config import ROLLING_WINDOW


_COL_NORMAL    = "#5b9bd5"
_COL_ANOMALY   = "#f4a261"
_COL_SHADING   = "#fde8e8"
_COL_THRESHOLD = "#c0392b"


def plot_machine(result: dict, mtype: str, mid: str, out_dir: Path) -> str:
    """Write ``{out_dir}/{mtype}_{mid}.png`` and return its path.

    ``result`` is the dict returned by ``evaluate_machine``. ``out_dir`` must
    exist. Optional keys ``r_desc`` and ``score_label`` customise labels per
    detector variant.
    """
    events        = result["events"]
    segments      = result["segments"]
    threshold     = result["threshold"]
    round_results = result["round_results"]
    n_rounds      = result["n_rounds"]
    n_components  = result.get("n_components", "?")
    first_rr  = round_results[0] if round_results else {}
    cusum_h   = first_rr.get("cusum_h", float("nan"))
    cusum_k   = first_rr.get("cusum_k", float("nan"))

    # Stored times are minutes; x-axis is seconds.
    ts = lambda e: e["t"] * 60

    normal_evts  = [e for e in events if e["phase"] == "normal"]
    anomaly_evts = [e for e in events if e["phase"] == "anomaly"]

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
        auc = float("nan")

    fig, ax = plt.subplots(figsize=(14, 5))

    anom_patched = False
    for seg in segments:
        if seg["phase"] == "anomaly":
            ax.axvspan(
                seg["t_start"] * 60, seg["t_end"] * 60,
                color=_COL_SHADING, alpha=0.8, zorder=0,
                label="Anomaly injection" if not anom_patched else None,
            )
            anom_patched = True

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

    if events and "rolling_mean" in events[0]:
        rm_t      = [ts(e) for e in events]
        rm_scores = [e["rolling_mean"] for e in events]
        ax.plot(rm_t, rm_scores, color="#555555", linewidth=0.9, alpha=0.5,
                zorder=3, label=f"Rolling mean (w={ROLLING_WINDOW})")

    ax.axhline(
        threshold, color=_COL_THRESHOLD, linestyle="--", linewidth=1.3,
        label=f"Threshold ({threshold:.3f})", zorder=3,
    )

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

    plt.tight_layout()
    out_path = out_dir / f"{mtype}_{mid}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)
