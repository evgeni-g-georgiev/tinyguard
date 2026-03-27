"""Evaluate YAMNet + PCA as f_c baseline.

Steps:
  1. Load cached 1024D YAMNet embeddings from outputs/embeddings/
  2. Fit PCA (1024 → 16D) on ALL normal clip embeddings (no labels used)
  3. Project all embeddings to 16D
  4. Evaluate: AUC, separation ratio, overlap %, cosine sim, silhouette
  5. Save PCA transform and plots

This measures how well a general-purpose pretrained audio model separates
normal from abnormal machine sounds — WITHOUT any MIMII-specific training.
The gap between this baseline and f_c+f_s performance IS the contribution
of on-device learning.

Usage:
    python scripts/evaluate_yamnet.py
    python scripts/evaluate_yamnet.py --pca_dim 32   # try different PCA dims
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
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.metrics import (
    per_machine_auc,
    separation_ratio,
    compute_silhouette,
    overlap_percentage,
    cosine_similarity_stats,
    print_evaluation,
)

EMBEDDINGS_DIR = "outputs/embeddings"
OUTPUT_DIR = "outputs/fc_baseline"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]


def load_cached_embeddings() -> dict:
    """Load all cached YAMNet embeddings and aggregate to clip-level.

    Returns dict matching the format expected by metrics.py:
        embeddings: (N_clips, 1024) array — mean-pooled across frames
        machine_types: list of str
        machine_ids: list of str
        labels: list of str ("normal" / "abnormal")
        clip_indices: list of int
    """
    embeddings = []
    machine_types = []
    machine_ids = []
    labels = []
    clip_indices = []
    clip_idx = 0

    for mtype in MACHINE_TYPES:
        mtype_dir = Path(EMBEDDINGS_DIR) / mtype
        if not mtype_dir.exists():
            continue
        for mid_dir in sorted(mtype_dir.iterdir()):
            if not mid_dir.is_dir():
                continue
            for label in ("normal", "abnormal"):
                label_dir = mid_dir / label
                if not label_dir.exists():
                    continue
                for npy_path in sorted(label_dir.glob("*.npy")):
                    frames = np.load(npy_path)  # (n_frames, 1024)
                    if frames.shape[0] == 0:
                        continue
                    # Mean-pool across frames → single clip embedding
                    clip_emb = frames.mean(axis=0)
                    embeddings.append(clip_emb)
                    machine_types.append(mtype)
                    machine_ids.append(mid_dir.name)
                    labels.append(label)
                    clip_indices.append(clip_idx)
                    clip_idx += 1

    return {
        "embeddings": np.array(embeddings),
        "machine_types": machine_types,
        "machine_ids": machine_ids,
        "labels": labels,
        "clip_indices": clip_indices,
    }


def fit_pca(clip_data: dict, n_components: int) -> tuple:
    """Fit PCA on normal embeddings only (unsupervised — no labels used).

    Returns:
        pca: fitted PCA object
        projected: dict with same structure but embeddings are (N, n_components)
    """
    # Fit on normal clips only
    normal_mask = [i for i, l in enumerate(clip_data["labels"]) if l == "normal"]
    normal_embs = clip_data["embeddings"][normal_mask]

    print(f"Fitting PCA({clip_data['embeddings'].shape[1]} → {n_components}) "
          f"on {len(normal_embs)} normal clips...")

    pca = PCA(n_components=n_components)
    pca.fit(normal_embs)

    # Report variance retention
    var_explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_explained)
    print(f"  Variance retained: {cumulative[-1]*100:.1f}%")
    print(f"  Top 3 PCs: {var_explained[0]*100:.1f}%, "
          f"{var_explained[1]*100:.1f}%, {var_explained[2]*100:.1f}%")

    # Project ALL embeddings (normal + abnormal)
    projected_embs = pca.transform(clip_data["embeddings"])

    projected = {
        "embeddings": projected_embs,
        "machine_types": clip_data["machine_types"],
        "machine_ids": clip_data["machine_ids"],
        "labels": clip_data["labels"],
        "clip_indices": clip_data["clip_indices"],
    }

    return pca, projected


def plot_pca_variance(pca, output_dir: str):
    """Plot PCA explained variance curve."""
    var = pca.explained_variance_ratio_
    cumulative = np.cumsum(var)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(1, len(var) + 1), var, alpha=0.6, label="Individual")
    ax.plot(range(1, len(var) + 1), cumulative, "r-o", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Variance (fitted on normal clips)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_variance.png"), dpi=150)
    plt.close()


def plot_tsne(clip_data: dict, output_dir: str):
    """Plot t-SNE coloured by machine type and by normal/abnormal."""
    embs = clip_data["embeddings"]

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embs)

    # By machine type
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    machine_keys = [
        f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        for i in range(len(embs))
    ]
    unique_machines = sorted(set(machine_keys))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_machines)))
    machine_to_color = {m: colors[i] for i, m in enumerate(unique_machines)}

    ax = axes[0]
    for machine in unique_machines:
        mask = [i for i, m in enumerate(machine_keys) if m == machine]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[machine_to_color[machine]],
                   s=8, alpha=0.5, label=machine)
    ax.set_title("t-SNE by machine")
    ax.legend(fontsize=6, markerscale=3, loc="best", ncol=2)

    # By label
    ax = axes[1]
    normal_mask = [i for i, l in enumerate(clip_data["labels"]) if l == "normal"]
    abnormal_mask = [i for i, l in enumerate(clip_data["labels"]) if l == "abnormal"]
    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
               c="steelblue", s=8, alpha=0.4, label="normal")
    ax.scatter(coords[abnormal_mask, 0], coords[abnormal_mask, 1],
               c="red", s=8, alpha=0.6, label="abnormal")
    ax.set_title("t-SNE by label")
    ax.legend(markerscale=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_dim", type=int, default=16, help="PCA output dimension")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load cached embeddings
    print("Loading cached YAMNet embeddings...")
    clip_data = load_cached_embeddings()
    n_clips = len(clip_data["labels"])
    n_normal = sum(1 for l in clip_data["labels"] if l == "normal")
    n_abnormal = n_clips - n_normal
    print(f"  {n_clips} clips ({n_normal} normal, {n_abnormal} abnormal)")
    print(f"  Embedding dim: {clip_data['embeddings'].shape[1]}")

    # Also evaluate raw 1024D (before PCA) for comparison
    print("\n--- Raw 1024D YAMNet embeddings (no PCA) ---")
    raw_results = {
        "clip_data": clip_data,
        "auc": per_machine_auc(clip_data),
        "separation_ratio": separation_ratio(clip_data),
        "silhouette": compute_silhouette(clip_data),
        "overlap_percentage": overlap_percentage(clip_data),
        "cosine_similarity_stats": cosine_similarity_stats(clip_data),
    }
    print_evaluation(raw_results)

    # Step 2: PCA projection
    print(f"\n--- PCA projected to {args.pca_dim}D ---")
    pca, projected_data = fit_pca(clip_data, args.pca_dim)

    # Step 3: Evaluate
    pca_results = {
        "clip_data": projected_data,
        "auc": per_machine_auc(projected_data),
        "separation_ratio": separation_ratio(projected_data),
        "silhouette": compute_silhouette(projected_data),
        "overlap_percentage": overlap_percentage(projected_data),
        "cosine_similarity_stats": cosine_similarity_stats(projected_data),
    }
    print_evaluation(pca_results)

    # Step 4: Save PCA and plots
    print("\nGenerating plots...")
    plot_pca_variance(pca, OUTPUT_DIR)
    plot_tsne(projected_data, OUTPUT_DIR)

    # Save PCA transform for deployment
    np.save(os.path.join(OUTPUT_DIR, "pca_components.npy"), pca.components_)
    np.save(os.path.join(OUTPUT_DIR, "pca_mean.npy"), pca.mean_)

    # Save summary
    summary = {
        "pca_dim": args.pca_dim,
        "variance_retained": float(np.sum(pca.explained_variance_ratio_)),
        "n_clips": n_clips,
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "raw_1024d": {
            "mean_auc": float(raw_results["auc"].get("mean", 0)),
            "silhouette": float(raw_results["silhouette"]),
        },
        "pca_projected": {
            "mean_auc": float(pca_results["auc"].get("mean", 0)),
            "silhouette": float(pca_results["silhouette"]),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.yaml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to {os.path.abspath(OUTPUT_DIR)}/")
    print("Files: pca_variance.png, tsne.png, pca_components.npy, pca_mean.npy, summary.yaml")


if __name__ == "__main__":
    main()
