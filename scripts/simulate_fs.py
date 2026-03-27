"""
Simulate on-device f_s training and evaluate anomaly detection.

For each machine:
  1. Load its 16D PCA embeddings (from cached YAMNet + PCA)
  2. Train f_s (16→ReLU→8 + centroid) on normal clips only
  3. Score all clips by distance to centroid in projected space
  4. Compute AUC and compare to f_c-only baseline

This simulates what each chip would do after deployment: hear normal
audio, learn a compact representation, then detect anomalies.

Usage:
    python scripts/simulate_fs.py
    python scripts/simulate_fs.py --epochs 50 --lr 0.01
"""

import argparse
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
from src.evaluation.metrics import print_evaluation

EMBEDDINGS_DIR = "outputs/embeddings"
BASELINE_DIR = "outputs/fc_baseline"
OUTPUT_DIR = "outputs/fs_simulation"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]


def load_machine_embeddings() -> dict:
    """Load cached 1024D embeddings, apply saved PCA, return per-machine data.

    Returns:
        dict mapping "machine_type/machine_id" → {
            "normal": (N, 16) array,
            "abnormal": (M, 16) array,
        }
    """
    # Load PCA transform
    pca_components = np.load(os.path.join(BASELINE_DIR, "pca_components.npy"))  # (16, 1024)
    pca_mean = np.load(os.path.join(BASELINE_DIR, "pca_mean.npy"))  # (1024,)

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
                    frames = np.load(npy_path)  # (n_frames, 1024)
                    if frames.shape[0] == 0:
                        continue
                    # Mean-pool frames → clip embedding, then PCA project
                    clip_1024 = frames.mean(axis=0)
                    clip_16 = (clip_1024 - pca_mean) @ pca_components.T
                    machines[machine_key][label].append(clip_16)

    # Stack into arrays
    for key in machines:
        for label in ("normal", "abnormal"):
            if machines[key][label]:
                machines[key][label] = np.stack(machines[key][label]).astype(np.float32)
            else:
                machines[key][label] = np.zeros((0, 16), dtype=np.float32)

    return dict(machines)


def train_fs_for_machine(
    normal_embs: np.ndarray,
    input_dim: int = 16,
    output_dim: int = 8,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seed: int = 42,
) -> FsSeparator:
    """Train f_s on normal embeddings for one machine.

    Simulates on-device learning with SGD (no momentum, no Adam —
    only what's feasible on a microcontroller).
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")  # simulating microcontroller

    model = FsSeparator(input_dim, output_dim).to(device)
    normal_tensor = torch.from_numpy(normal_embs).to(device)

    # Initialise centroid from first forward pass
    model.init_centroid(normal_tensor)

    # Plain SGD — matches what Arduino can do
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    n = len(normal_tensor)
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch = normal_tensor[perm[start:start + batch_size]]
            projected = model(batch)

            # Deep SVDD loss: mean squared distance to centroid
            loss = ((projected - model.centroid) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    return model


def evaluate_machine(
    model: FsSeparator,
    normal_embs: np.ndarray,
    abnormal_embs: np.ndarray,
) -> dict:
    """Evaluate f_s on one machine: compute AUC from anomaly scores."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    with torch.no_grad():
        normal_scores = model.anomaly_score(
            torch.from_numpy(normal_embs)
        ).numpy()
        abnormal_scores = model.anomaly_score(
            torch.from_numpy(abnormal_embs)
        ).numpy()

    all_scores = np.concatenate([normal_scores, abnormal_scores])
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(abnormal_scores)),
    ])

    auc = roc_auc_score(all_labels, all_scores)

    return {
        "auc": auc,
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_std": float(normal_scores.std()),
        "abnormal_score_mean": float(abnormal_scores.mean()),
        "abnormal_score_std": float(abnormal_scores.std()),
        "separation_ratio": float(abnormal_scores.mean() / (normal_scores.mean() + 1e-8)),
    }


def baseline_centroid_auc(normal_embs: np.ndarray, abnormal_embs: np.ndarray) -> float:
    """f_c-only baseline: centroid distance in raw 16D PCA space."""
    from sklearn.metrics import roc_auc_score

    centroid = normal_embs.mean(axis=0)
    normal_dists = np.linalg.norm(normal_embs - centroid, axis=1)
    abnormal_dists = np.linalg.norm(abnormal_embs - centroid, axis=1)

    all_dists = np.concatenate([normal_dists, abnormal_dists])
    all_labels = np.concatenate([np.zeros(len(normal_dists)), np.ones(len(abnormal_dists))])

    return roc_auc_score(all_labels, all_dists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dim", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading PCA-projected embeddings per machine...")
    machines = load_machine_embeddings()
    print(f"Found {len(machines)} machines\n")

    # Train and evaluate f_s for each machine
    results = {}
    baseline_aucs = {}

    print(f"{'Machine':<20s}  {'f_c AUC':>8s}  {'f_c+f_s AUC':>11s}  {'Δ':>7s}  {'Sep ratio':>9s}")
    print("-" * 65)

    for machine_key in sorted(machines.keys()):
        data = machines[machine_key]
        normal = data["normal"]
        abnormal = data["abnormal"]

        if len(normal) == 0 or len(abnormal) == 0:
            continue

        # f_c-only baseline
        fc_auc = baseline_centroid_auc(normal, abnormal)
        baseline_aucs[machine_key] = fc_auc

        # Train f_s
        model = train_fs_for_machine(
            normal,
            output_dim=args.output_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Evaluate
        result = evaluate_machine(model, normal, abnormal)
        results[machine_key] = result

        delta = result["auc"] - fc_auc
        print(f"{machine_key:<20s}  {fc_auc:>8.4f}  {result['auc']:>11.4f}  {delta:>+7.4f}  {result['separation_ratio']:>9.2f}")

    # Summary
    mean_fc = np.mean(list(baseline_aucs.values()))
    mean_fs = np.mean([r["auc"] for r in results.values()])
    delta = mean_fs - mean_fc

    print("-" * 65)
    print(f"{'MEAN':<20s}  {mean_fc:>8.4f}  {mean_fs:>11.4f}  {delta:>+7.4f}")
    print(f"\nf_s params: {FsSeparator(16, args.output_dim).param_count}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    machine_names = sorted(results.keys())
    x = np.arange(len(machine_names))
    fc_vals = [baseline_aucs[m] for m in machine_names]
    fs_vals = [results[m]["auc"] for m in machine_names]

    # Bar chart
    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, fc_vals, width, label="f_c only (centroid)", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, fs_vals, width, label="f_c + f_s (SVDD)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("/", "\n") for m in machine_names], fontsize=7)
    ax.set_ylabel("AUC")
    ax.set_title("Per-machine AUC: f_c only vs f_c + f_s")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    ax.grid(axis="y", alpha=0.3)

    # Score distributions for a sample machine (pick the one with best improvement)
    deltas = {m: results[m]["auc"] - baseline_aucs[m] for m in machine_names}
    best_machine = max(deltas, key=deltas.get)

    ax = axes[1]
    data = machines[best_machine]
    model = train_fs_for_machine(
        data["normal"], output_dim=args.output_dim, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        seed=args.seed,
    )
    with torch.no_grad():
        normal_scores = model.anomaly_score(torch.from_numpy(data["normal"])).numpy()
        abnormal_scores = model.anomaly_score(torch.from_numpy(data["abnormal"])).numpy()

    ax.hist(normal_scores, bins=40, alpha=0.6, label="normal", color="steelblue", density=True)
    ax.hist(abnormal_scores, bins=40, alpha=0.6, label="abnormal", color="coral", density=True)
    ax.set_xlabel("Anomaly score (distance to centroid)")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution: {best_machine}\n"
                 f"(f_c AUC={baseline_aucs[best_machine]:.3f} → "
                 f"f_c+f_s AUC={results[best_machine]['auc']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fs_results.png"), dpi=150)
    plt.close()

    # Save results
    summary = {
        "config": {
            "output_dim": args.output_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "param_count": FsSeparator(16, args.output_dim).param_count,
        },
        "mean_auc_fc_only": float(mean_fc),
        "mean_auc_fc_fs": float(mean_fs),
        "improvement": float(delta),
        "per_machine": {
            m: {"fc_auc": float(baseline_aucs[m]), "fs_auc": float(results[m]["auc"])}
            for m in machine_names
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.yaml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
