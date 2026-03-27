"""
Cold start experiment: how quickly does a chip become useful?

Simulates deployment with limited normal audio. For each machine,
trains f_c centroid and f_s on only the first N normal clips
(N = 5, 10, 20, 40, 80, 160, all), then evaluates AUC using
ALL clips (normal + abnormal).

This answers: "If I deploy a chip and it hears N normal sounds,
how good is its anomaly detection?"

Usage:
    python scripts/cold_start.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fs.model import FsSeparator
from sklearn.metrics import roc_auc_score

EMBEDDINGS_DIR = "outputs/embeddings"
BASELINE_DIR = "outputs/fc_baseline"
OUTPUT_DIR = "outputs/cold_start"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
N_VALUES = [5, 10, 20, 40, 80, 160]


def load_machine_embeddings():
    """Load cached embeddings and PCA-project to 16D, per machine."""
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
                    machines[machine_key][label].append(clip_16)

    for key in machines:
        for label in ("normal", "abnormal"):
            if machines[key][label]:
                machines[key][label] = np.stack(machines[key][label]).astype(np.float32)
            else:
                machines[key][label] = np.zeros((0, 16), dtype=np.float32)

    return dict(machines)


def fc_auc_at_n(normal_all, abnormal, n):
    """f_c-only AUC using centroid from first n normal clips."""
    centroid = normal_all[:n].mean(axis=0)
    normal_dists = np.linalg.norm(normal_all - centroid, axis=1)
    abnormal_dists = np.linalg.norm(abnormal - centroid, axis=1)
    scores = np.concatenate([normal_dists, abnormal_dists])
    labels = np.concatenate([np.zeros(len(normal_all)), np.ones(len(abnormal))])
    return roc_auc_score(labels, scores)


def fs_auc_at_n(normal_all, abnormal, n, seed=42):
    """f_c+f_s AUC using f_s trained on first n normal clips."""
    torch.manual_seed(seed)
    train_embs = normal_all[:n]

    model = FsSeparator(16, 8)
    train_tensor = torch.from_numpy(train_embs)
    model.init_centroid(train_tensor)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(100):
        perm = torch.randperm(len(train_tensor))
        for start in range(0, len(train_tensor), 64):
            batch = train_tensor[perm[start:start + 64]]
            projected = model(batch)
            loss = ((projected - model.centroid) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        normal_scores = model.anomaly_score(torch.from_numpy(normal_all)).numpy()
        abnormal_scores = model.anomaly_score(torch.from_numpy(abnormal)).numpy()

    scores = np.concatenate([normal_scores, abnormal_scores])
    labels = np.concatenate([np.zeros(len(normal_all)), np.ones(len(abnormal))])
    return roc_auc_score(labels, scores)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings...")
    machines = load_machine_embeddings()

    # Collect results
    fc_results = defaultdict(dict)   # machine → {n: auc}
    fs_results = defaultdict(dict)

    for machine_key in sorted(machines.keys()):
        data = machines[machine_key]
        normal = data["normal"]
        abnormal = data["abnormal"]
        if len(normal) == 0 or len(abnormal) == 0:
            continue

        # Shuffle normal clips (simulate random arrival order)
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(normal))
        normal_shuffled = normal[perm]

        n_values = [n for n in N_VALUES if n < len(normal_shuffled)] + [len(normal_shuffled)]

        for n in n_values:
            fc_results[machine_key][n] = fc_auc_at_n(normal_shuffled, abnormal, n)
            fs_results[machine_key][n] = fs_auc_at_n(normal_shuffled, abnormal, n)

        # Print progress
        full_n = len(normal_shuffled)
        print(f"{machine_key:<20s}  N_normal={full_n:>4d}  "
              f"f_c@10={fc_results[machine_key].get(10, 0):.3f}  "
              f"f_s@10={fs_results[machine_key].get(10, 0):.3f}  "
              f"f_c@all={fc_results[machine_key][full_n]:.3f}  "
              f"f_s@all={fs_results[machine_key][full_n]:.3f}")

    # Compute mean AUC across machines at each N
    all_n_values = sorted(set(n for m in fc_results.values() for n in m.keys()))
    mean_fc_by_n = {}
    mean_fs_by_n = {}

    for n in all_n_values:
        fc_vals = [fc_results[m][n] for m in fc_results if n in fc_results[m]]
        fs_vals = [fs_results[m][n] for m in fs_results if n in fs_results[m]]
        if fc_vals:
            mean_fc_by_n[n] = np.mean(fc_vals)
            mean_fs_by_n[n] = np.mean(fs_vals)

    print(f"\n{'N clips':>8s}  {'mean f_c AUC':>12s}  {'mean f_s AUC':>12s}  {'Δ':>7s}")
    print("-" * 45)
    for n in sorted(mean_fc_by_n.keys()):
        delta = mean_fs_by_n[n] - mean_fc_by_n[n]
        print(f"{n:>8d}  {mean_fc_by_n[n]:>12.4f}  {mean_fs_by_n[n]:>12.4f}  {delta:>+7.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: mean AUC vs N
    ax = axes[0]
    ns = sorted(mean_fc_by_n.keys())
    ax.plot(ns, [mean_fc_by_n[n] for n in ns], "o-", color="steelblue",
            label="f_c only (centroid)", markersize=6)
    ax.plot(ns, [mean_fs_by_n[n] for n in ns], "s-", color="coral",
            label="f_c + f_s (SVDD)", markersize=6)
    ax.set_xlabel("Number of normal clips seen")
    ax.set_ylabel("Mean AUC (across all machines)")
    ax.set_title("Cold Start: AUC vs Training Data")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    # Right: per-machine curves (f_s only, to avoid clutter)
    ax = axes[1]
    for machine_key in sorted(fs_results.keys()):
        ns_m = sorted(fs_results[machine_key].keys())
        vals = [fs_results[machine_key][n] for n in ns_m]
        ax.plot(ns_m, vals, ".-", alpha=0.5, markersize=4, label=machine_key)
    ax.set_xlabel("Number of normal clips seen")
    ax.set_ylabel("AUC")
    ax.set_title("Cold Start: per-machine f_c+f_s AUC")
    ax.set_xscale("log")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cold_start.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {os.path.abspath(OUTPUT_DIR)}/cold_start.png")


if __name__ == "__main__":
    main()
