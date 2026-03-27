"""
Simulate real-world deployment of TinyML anomalous sound detection nodes.

Models what happens when you power on 2-3 Arduino Nano 33 BLE chips on a factory
floor, each near a different machine:

  Phase 1 — Training (2 min):  Node hears normal machine sounds, trains SVDD on-the-fly.
  Phase 2 — Monitoring (normal): Node listens to more normal sounds, confirming low false alarm rate.
  Phase 3 — Monitoring (anomaly): Machine develops a fault — node must detect the change.

Each 10-second audio clip is processed one-at-a-time (streaming), simulating
real-time operation.  The threshold is set from training-phase scores only (no
future knowledge).

Output:
  - Timeline plots showing anomaly scores over time with threshold + annotations
  - Per-node detection summary (time-to-detection, false alarm rate, AUC)
  - Console log mimicking what a deployed node would print over serial

Usage:
    python scripts/simulate_deployment.py
    python scripts/simulate_deployment.py --nodes fan/id_00 pump/id_02 slider/id_04
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fs.model import FsSeparator
from sklearn.metrics import roc_auc_score

EMBEDDINGS_DIR = "outputs/embeddings"
BASELINE_DIR = "outputs/fc_baseline"
OUTPUT_DIR = "outputs/deployment_simulation"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]

# ── Deployment parameters (match Arduino constraints) ──
CLIP_DURATION_S = 10.0        # each MIMII clip ≈ 10 seconds
TRAINING_DURATION_S = 120.0   # 2 minutes of normal audio for training
MONITOR_NORMAL_S = 300.0      # 5 minutes of normal monitoring
MONITOR_ANOMALY_S = 300.0     # 5 minutes of anomaly monitoring
THRESHOLD_PERCENTILE = 95     # anomaly threshold from training scores


def load_machine_embeddings():
    """Load cached 1024D embeddings, PCA-project to 16D, per machine."""
    pca_components = np.load(os.path.join(BASELINE_DIR, "pca_components.npy"))
    pca_mean = np.load(os.path.join(BASELINE_DIR, "pca_mean.npy"))

    machines = defaultdict(lambda: {"normal": [], "abnormal": []})
    for mtype in MACHINE_TYPES:
        mtype_dir = Path(EMBEDDINGS_DIR) / mtype
        if not mtype_dir.exists():
            continue
        for mid_dir in sorted(mtype_dir.iterdir()):
            if not mid_dir.is_dir():
                continue
            machine_key = f"{mtype}/{mid_dir.name}"
            for label in ("normal", "abnormal"):
                label_dir = mid_dir / label
                if not label_dir.exists():
                    continue
                for npy_path in sorted(label_dir.glob("*.npy")):
                    frames = np.load(npy_path)
                    if frames.shape[0] == 0:
                        continue
                    clip_1024 = frames.mean(axis=0)
                    clip_16 = (clip_1024 - pca_mean) @ pca_components.T
                    machines[machine_key][label].append(clip_16.astype(np.float32))

    return dict(machines)


class DeployedNode:
    """Simulates a single deployed TinyML node.

    Memory footprint on Arduino Nano 33 BLE:
      - f_s weights: 128 floats (512 bytes)
      - centroid: 8 floats (32 bytes)
      - threshold: 1 float (4 bytes)
      - PCA transform: 16×1024 + 1024 = 17,408 floats (68 KB)
        (shared across nodes if multi-chip, or preloaded in flash)
      Total model: ~69 KB (well within 256 KB SRAM / 1 MB flash)
    """

    def __init__(self, node_id: str, machine_key: str, seed: int = 42):
        self.node_id = node_id
        self.machine_key = machine_key
        self.seed = seed

        torch.manual_seed(seed)
        self.model = FsSeparator(input_dim=16, output_dim=8)
        self.threshold = float("inf")

        # Training state
        self._training_clips = []
        self._training_scores = []
        self._is_trained = False

        # Monitoring log
        self.log = []  # list of {time, score, label, alert}

    def train_on_clips(self, normal_clips: list, epochs: int = 100,
                       lr: float = 0.01, weight_decay: float = 1e-4,
                       batch_size: int = 64):
        """Train SVDD on normal clips (simulates 2-min on-device learning)."""
        self._training_clips = normal_clips
        train_data = np.stack(normal_clips).astype(np.float32)
        train_t = torch.from_numpy(train_data)

        # Init centroid
        self.model.init_centroid(train_t)

        # SGD training (Arduino-feasible)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.model.train()
        n = len(train_t)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                batch = train_t[perm[start:start + batch_size]]
                proj = self.model(batch)
                loss = ((proj - self.model.centroid) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Set threshold from training scores
        self.model.eval()
        with torch.no_grad():
            scores = self.model.anomaly_score(train_t).numpy()
        self._training_scores = scores.tolist()
        self.threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
        self._is_trained = True

    def score_clip(self, clip: np.ndarray) -> float:
        """Score a single clip (real-time inference)."""
        self.model.eval()
        with torch.no_grad():
            score = self.model.anomaly_score(
                torch.from_numpy(clip.reshape(1, -1).astype(np.float32))
            ).item()
        return score

    def monitor_clip(self, clip: np.ndarray, time_s: float,
                     true_label: str) -> dict:
        """Process one clip during monitoring: score, compare to threshold, log."""
        score = self.score_clip(clip)
        alert = score > self.threshold
        entry = {
            "time_s": time_s,
            "score": score,
            "label": true_label,
            "alert": alert,
        }
        self.log.append(entry)
        return entry

    def get_summary(self) -> dict:
        """Compute deployment summary statistics."""
        if not self.log:
            return {}

        normal_entries = [e for e in self.log if e["label"] == "normal"]
        anomaly_entries = [e for e in self.log if e["label"] == "abnormal"]

        # False alarm rate (during normal monitoring)
        false_alarms = sum(1 for e in normal_entries if e["alert"])
        far = false_alarms / max(len(normal_entries), 1)

        # Detection rate (during anomaly monitoring)
        true_detections = sum(1 for e in anomaly_entries if e["alert"])
        detection_rate = true_detections / max(len(anomaly_entries), 1)

        # Time to first detection
        first_detection_time = None
        anomaly_start_time = None
        if anomaly_entries:
            anomaly_start_time = anomaly_entries[0]["time_s"]
            for e in anomaly_entries:
                if e["alert"]:
                    first_detection_time = e["time_s"]
                    break

        time_to_detect = None
        if first_detection_time is not None and anomaly_start_time is not None:
            time_to_detect = first_detection_time - anomaly_start_time

        # AUC over all monitoring clips
        all_scores = [e["score"] for e in self.log]
        all_labels = [1 if e["label"] == "abnormal" else 0 for e in self.log]
        auc = roc_auc_score(all_labels, all_scores) if sum(all_labels) > 0 else None

        return {
            "machine": self.machine_key,
            "n_training_clips": len(self._training_clips),
            "training_duration_s": len(self._training_clips) * CLIP_DURATION_S,
            "threshold": self.threshold,
            "n_monitor_normal": len(normal_entries),
            "n_monitor_anomaly": len(anomaly_entries),
            "false_alarm_rate": far,
            "detection_rate": detection_rate,
            "time_to_first_detection_s": time_to_detect,
            "auc": auc,
            "model_params": self.model.param_count,
        }


def run_deployment(machine_key: str, clips_normal: list, clips_abnormal: list,
                   node_id: str = "Node", seed: int = 42, verbose: bool = True):
    """Run full deployment simulation for one node."""
    n_train = int(TRAINING_DURATION_S / CLIP_DURATION_S)
    n_monitor_normal = int(MONITOR_NORMAL_S / CLIP_DURATION_S)
    n_monitor_anomaly = int(MONITOR_ANOMALY_S / CLIP_DURATION_S)

    # Ensure we have enough clips
    n_train = min(n_train, len(clips_normal) // 2)  # reserve half for monitoring
    remaining_normal = len(clips_normal) - n_train
    n_monitor_normal = min(n_monitor_normal, remaining_normal)
    n_monitor_anomaly = min(n_monitor_anomaly, len(clips_abnormal))

    if n_train < 3:
        if verbose:
            print(f"  [{node_id}] Not enough normal clips for training")
        return None

    train_clips = clips_normal[:n_train]
    monitor_normal_clips = clips_normal[n_train:n_train + n_monitor_normal]
    monitor_anomaly_clips = clips_abnormal[:n_monitor_anomaly]

    node = DeployedNode(node_id, machine_key, seed=seed)

    # ── Phase 1: Training ──
    if verbose:
        print(f"\n  [{node_id}] === PHASE 1: TRAINING ({machine_key}) ===")
        print(f"  [{node_id}] Listening to {n_train} normal clips "
              f"({n_train * CLIP_DURATION_S:.0f}s of audio)...")

    node.train_on_clips(train_clips)

    if verbose:
        print(f"  [{node_id}] SVDD trained: 128 params, centroid set")
        print(f"  [{node_id}] Threshold = {node.threshold:.4f} "
              f"({THRESHOLD_PERCENTILE}th percentile of training scores)")
        train_mean = np.mean(node._training_scores)
        train_max = np.max(node._training_scores)
        print(f"  [{node_id}] Training scores: mean={train_mean:.4f}, max={train_max:.4f}")

    # ── Phase 2: Monitor normal ──
    if verbose:
        print(f"\n  [{node_id}] === PHASE 2: MONITORING NORMAL ({n_monitor_normal} clips) ===")

    time_s = n_train * CLIP_DURATION_S  # time continues from training
    alerts_normal = 0
    for clip in monitor_normal_clips:
        entry = node.monitor_clip(clip, time_s, "normal")
        if entry["alert"]:
            alerts_normal += 1
            if verbose:
                print(f"  [{node_id}] t={time_s:6.0f}s  score={entry['score']:.4f}  "
                      f"⚠ FALSE ALARM (normal clip scored above threshold)")
        time_s += CLIP_DURATION_S

    if verbose:
        print(f"  [{node_id}] Normal monitoring complete: "
              f"{alerts_normal}/{n_monitor_normal} false alarms "
              f"({100*alerts_normal/max(n_monitor_normal,1):.1f}%)")

    # ── Phase 3: Monitor anomaly ──
    if verbose:
        anomaly_start = time_s
        print(f"\n  [{node_id}] === PHASE 3: ANOMALY INJECTION at t={anomaly_start:.0f}s ===")

    first_detection = None
    detections = 0
    for clip in monitor_anomaly_clips:
        entry = node.monitor_clip(clip, time_s, "abnormal")
        if entry["alert"]:
            detections += 1
            if first_detection is None:
                first_detection = time_s
                if verbose:
                    delay = time_s - anomaly_start
                    print(f"  [{node_id}] t={time_s:6.0f}s  score={entry['score']:.4f}  "
                          f"*** ANOMALY DETECTED *** (delay={delay:.0f}s)")
        time_s += CLIP_DURATION_S

    if verbose:
        print(f"  [{node_id}] Anomaly monitoring complete: "
              f"{detections}/{n_monitor_anomaly} detections "
              f"({100*detections/max(n_monitor_anomaly,1):.1f}%)")
        summary = node.get_summary()
        if summary.get("time_to_first_detection_s") is not None:
            print(f"  [{node_id}] Time to first detection: "
                  f"{summary['time_to_first_detection_s']:.0f}s")
        if summary.get("auc") is not None:
            print(f"  [{node_id}] Monitoring AUC: {summary['auc']:.4f}")

    return node


def plot_node_timeline(node: DeployedNode, ax: plt.Axes, training_end_s: float):
    """Plot anomaly score timeline for one node."""
    times = [e["time_s"] for e in node.log]
    scores = [e["score"] for e in node.log]
    labels = [e["label"] for e in node.log]

    # Split into normal and anomaly segments
    normal_times = [t for t, l in zip(times, labels) if l == "normal"]
    normal_scores = [s for s, l in zip(scores, labels) if l == "normal"]
    anomaly_times = [t for t, l in zip(times, labels) if l == "abnormal"]
    anomaly_scores = [s for s, l in zip(scores, labels) if l == "abnormal"]

    # Shade regions
    if times:
        t_min, t_max = min(times), max(times)
        # Training region (before monitoring)
        ax.axvspan(0, training_end_s, alpha=0.08, color="steelblue",
                   label="Training phase")

        # Anomaly region
        if anomaly_times:
            ax.axvspan(min(anomaly_times), max(anomaly_times) + CLIP_DURATION_S,
                       alpha=0.10, color="coral", label="Anomaly injection")

    # Plot scores
    ax.scatter(normal_times, normal_scores, c="steelblue", s=20, alpha=0.7,
               zorder=3, label="Normal clips")
    ax.scatter(anomaly_times, anomaly_scores, c="coral", s=20, alpha=0.7,
               zorder=3, label="Anomaly clips")

    # Threshold line
    ax.axhline(y=node.threshold, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Threshold ({node.threshold:.3f})")

    # Mark first detection
    summary = node.get_summary()
    if anomaly_times and summary.get("time_to_first_detection_s") is not None:
        first_anom_time = min(anomaly_times)
        detect_time = first_anom_time + summary["time_to_first_detection_s"]
        detect_score = None
        for e in node.log:
            if e["time_s"] == detect_time and e["alert"]:
                detect_score = e["score"]
                break
        if detect_score is not None:
            ax.annotate(
                f"Detected!\n({summary['time_to_first_detection_s']:.0f}s delay)",
                xy=(detect_time, detect_score),
                xytext=(detect_time + 30, detect_score * 1.15),
                fontsize=8, fontweight="bold", color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
            )

    # Formatting
    mtype, mid = node.machine_key.split("/")
    det_rate = summary.get("detection_rate", 0)
    far = summary.get("false_alarm_rate", 0)
    auc = summary.get("auc", 0)
    ax.set_title(
        f"{node.node_id}: {mtype} {mid}\n"
        f"Detection={100*det_rate:.0f}%  |  False alarm={100*far:.1f}%  |  AUC={auc:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Anomaly score")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate real-world deployment of TinyML anomaly detection nodes"
    )
    parser.add_argument(
        "--nodes", nargs="+", default=None,
        help="Machine keys for nodes (e.g., fan/id_00 pump/id_02). "
             "Default: pick 3 diverse machines automatically."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  TinyML DEPLOYMENT SIMULATION")
    print("  Anomalous Sound Detection on Arduino Nano 33 BLE")
    print("=" * 70)

    print("\nLoading YAMNet embeddings (PCA-projected to 16D)...")
    machines = load_machine_embeddings()
    valid = {k: v for k, v in machines.items()
             if len(v["normal"]) > 20 and len(v["abnormal"]) > 10}
    print(f"Available machines: {len(valid)}")

    # Select nodes
    if args.nodes:
        selected = [k for k in args.nodes if k in valid]
    else:
        # Pick 3 diverse machines: one fan, one pump, one slider/valve
        candidates = sorted(valid.keys())
        selected = []
        for target_type in ["fan", "pump", "valve"]:
            for k in candidates:
                if k.startswith(target_type) and k not in selected:
                    selected.append(k)
                    break
        if len(selected) < 2:
            selected = candidates[:3]

    print(f"Selected nodes: {selected}\n")

    # ── Deployment constraints summary ──
    print("-" * 70)
    print("DEPLOYMENT CONSTRAINTS (Arduino Nano 33 BLE)")
    print("-" * 70)
    print(f"  f_s model:           128 params × 4 bytes = 512 bytes")
    print(f"  Centroid:            8 floats × 4 bytes   = 32 bytes")
    print(f"  Threshold:           1 float × 4 bytes    = 4 bytes")
    print(f"  PCA (in flash):      16×1024 + 1024       = 68 KB")
    print(f"  YAMNet (TFLite):     ~180 KB (quantised)")
    print(f"  Total model memory:  ~249 KB  (of 256 KB SRAM + 1 MB flash)")
    print(f"  Training time:       {TRAINING_DURATION_S:.0f}s ({TRAINING_DURATION_S/60:.0f} min)")
    print(f"  Training clips:      {int(TRAINING_DURATION_S / CLIP_DURATION_S)} "
          f"× {CLIP_DURATION_S:.0f}s clips")
    print(f"  Anomaly threshold:   {THRESHOLD_PERCENTILE}th percentile of training scores")
    print(f"  Inference:           1 clip ({CLIP_DURATION_S:.0f}s) → 1 score → threshold check")
    print()

    # ── Run deployment for each node ──
    nodes = []
    summaries = []

    for i, machine_key in enumerate(selected):
        data = valid[machine_key]
        normal_clips = data["normal"]
        abnormal_clips = data["abnormal"]

        print("=" * 70)
        print(f"  NODE {i+1}: {machine_key}")
        print(f"  Available: {len(normal_clips)} normal, {len(abnormal_clips)} abnormal clips")
        print("=" * 70)

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(args.seed + i)
        normal_order = rng.permutation(len(normal_clips))
        anomaly_order = rng.permutation(len(abnormal_clips))
        shuffled_normal = [normal_clips[j] for j in normal_order]
        shuffled_anomaly = [abnormal_clips[j] for j in anomaly_order]

        node = run_deployment(
            machine_key, shuffled_normal, shuffled_anomaly,
            node_id=f"Node {i+1}", seed=args.seed + i, verbose=True,
        )

        if node is not None:
            nodes.append(node)
            summaries.append(node.get_summary())

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  DEPLOYMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Node':<10s}  {'Machine':<18s}  {'Train':>6s}  {'Det%':>6s}  "
          f"{'FAR%':>6s}  {'T_det':>6s}  {'AUC':>6s}")
    print("-" * 66)

    for i, s in enumerate(summaries):
        t_det = f"{s['time_to_first_detection_s']:.0f}s" if s.get("time_to_first_detection_s") is not None else "N/A"
        auc_str = f"{s['auc']:.3f}" if s.get("auc") is not None else "N/A"
        print(f"Node {i+1:<4d}  {s['machine']:<18s}  "
              f"{s['n_training_clips']:>4d}cl  "
              f"{100*s['detection_rate']:>5.1f}  "
              f"{100*s['false_alarm_rate']:>5.1f}  "
              f"{t_det:>6s}  {auc_str:>6s}")

    mean_det = np.mean([s["detection_rate"] for s in summaries])
    mean_far = np.mean([s["false_alarm_rate"] for s in summaries])
    mean_auc = np.mean([s["auc"] for s in summaries if s.get("auc")])
    print("-" * 66)
    print(f"{'MEAN':<10s}  {'':<18s}  {'':>6s}  "
          f"{100*mean_det:>5.1f}  {100*mean_far:>5.1f}  {'':>6s}  {mean_auc:>6.3f}")

    # ── Timeline plots ──
    n_nodes = len(nodes)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(14, 4.5 * n_nodes), squeeze=False)

    for i, node in enumerate(nodes):
        n_train = len(node._training_clips)
        training_end_s = n_train * CLIP_DURATION_S
        plot_node_timeline(node, axes[i, 0], training_end_s)

    fig.suptitle(
        "TinyML Deployment Simulation — Anomalous Sound Detection\n"
        "Arduino Nano 33 BLE  |  SVDD (128 params)  |  YAMNet + PCA embeddings",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deployment_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nTimeline plot saved to {OUTPUT_DIR}/deployment_timeline.png")

    # ── Score distribution plot ──
    fig, axes = plt.subplots(1, n_nodes, figsize=(5 * n_nodes, 4), squeeze=False)
    for i, node in enumerate(nodes):
        ax = axes[0, i]
        normal_scores = [e["score"] for e in node.log if e["label"] == "normal"]
        anomaly_scores = [e["score"] for e in node.log if e["label"] == "abnormal"]

        ax.hist(normal_scores, bins=25, alpha=0.6, color="steelblue",
                label="Normal", density=True)
        ax.hist(anomaly_scores, bins=25, alpha=0.6, color="coral",
                label="Anomaly", density=True)
        ax.axvline(node.threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"Threshold")

        mtype, mid = node.machine_key.split("/")
        s = node.get_summary()
        ax.set_title(f"{mtype} {mid}\nAUC={s['auc']:.3f}", fontsize=10)
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle("Score Distributions During Monitoring", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deployment_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to {OUTPUT_DIR}/deployment_distributions.png")

    # ── Save YAML summary ──
    yaml_summary = {
        "deployment_config": {
            "training_duration_s": TRAINING_DURATION_S,
            "monitor_normal_duration_s": MONITOR_NORMAL_S,
            "monitor_anomaly_duration_s": MONITOR_ANOMALY_S,
            "clip_duration_s": CLIP_DURATION_S,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "model": "FsSeparator (Deep SVDD)",
            "model_params": 128,
            "model_memory_bytes": 548,
            "embedding": "YAMNet 1024D → PCA 16D",
        },
        "nodes": {
            f"node_{i+1}": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in s.items()
            }
            for i, s in enumerate(summaries)
        },
        "aggregate": {
            "mean_detection_rate": float(mean_det),
            "mean_false_alarm_rate": float(mean_far),
            "mean_auc": float(mean_auc),
            "n_nodes": n_nodes,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.yaml"), "w") as f:
        yaml.dump(yaml_summary, f, default_flow_style=False)
    print(f"Summary saved to {OUTPUT_DIR}/summary.yaml")

    print(f"\nDone. All outputs in {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
