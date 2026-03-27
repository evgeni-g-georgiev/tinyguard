"""
Visualisation tools for assessing f_c embedding quality.

Provides multiple views into the embedding space:
  1. t-SNE: 2D projection colored by machine and by label
  2. Distance distributions: per-machine histograms of distance to centroid
  3. PCA variance explained: how much structure the 16D space captures
  4. Training curves: loss and learning rate over epochs
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne(clip_data: dict, output_dir: str, perplexity: int = 30, seed: int = 42):
    """Generate t-SNE plots colored by machine identity and by label.

    Produces two PNG files:
      - tsne_by_machine.png: each machine type/ID gets a unique color
      - tsne_by_label.png: normal vs abnormal

    Args:
        clip_data: dict from aggregate_to_clips() with embeddings and metadata
        output_dir: directory to save plots
        perplexity: t-SNE perplexity parameter
        seed: random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    embeddings = clip_data["embeddings"]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                random_state=seed, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    # --- Plot by machine ---
    machine_labels = [
        f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        for i in range(len(clip_data["labels"]))
    ]
    unique_machines = sorted(set(machine_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_machines), 1)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, machine in enumerate(unique_machines):
        mask = [m == machine for m in machine_labels]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=machine, s=15, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("t-SNE of f_c embeddings — by machine")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tsne_by_machine.png"), dpi=150)
    plt.close(fig)

    # --- Plot by label ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color, marker in [("normal", "steelblue", "o"), ("abnormal", "red", "x")]:
        mask = [clip_data["labels"][i] == label for i in range(len(clip_data["labels"]))]
        if any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, label=label, s=15, alpha=0.7, marker=marker)
    ax.legend()
    ax.set_title("t-SNE of f_c embeddings — normal vs abnormal")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tsne_by_label.png"), dpi=150)
    plt.close(fig)

    print(f"t-SNE plots saved to {output_dir}/")


def plot_distance_distributions(clip_data: dict, output_dir: str):
    """Plot per-machine histograms of distance to centroid.

    For each machine, shows the distribution of distances for normal vs
    abnormal clips. Good embeddings show clear separation between the
    two distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    from collections import defaultdict

    machines = defaultdict(lambda: {"normal": [], "abnormal": []})

    for i in range(len(clip_data["labels"])):
        key = f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        machines[key][clip_data["labels"][i]].append(clip_data["embeddings"][i])

    # Filter to machines that have both normal and abnormal
    machines_with_both = {k: v for k, v in machines.items()
                         if v["normal"] and v["abnormal"]}

    if not machines_with_both:
        print("No machines with both normal and abnormal data for distance plots.")
        return

    n_machines = len(machines_with_both)
    cols = min(4, n_machines)
    rows = (n_machines + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, (machine_key, data) in enumerate(sorted(machines_with_both.items())):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        normal_embs = np.array(data["normal"])
        abnormal_embs = np.array(data["abnormal"])
        centroid = normal_embs.mean(axis=0)

        normal_dists = np.linalg.norm(normal_embs - centroid, axis=1)
        abnormal_dists = np.linalg.norm(abnormal_embs - centroid, axis=1)

        bins = np.linspace(
            0, max(normal_dists.max(), abnormal_dists.max()) * 1.1, 30
        )
        ax.hist(normal_dists, bins=bins, alpha=0.6, label="normal", color="steelblue")
        ax.hist(abnormal_dists, bins=bins, alpha=0.6, label="abnormal", color="red")
        ax.set_title(machine_key, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Distance to centroid", fontsize=8)

    # Hide unused axes
    for idx in range(n_machines, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Distance to centroid — normal vs abnormal", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "distance_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"Distance distribution plot saved to {output_dir}/")


def plot_pca_variance(clip_data: dict, output_dir: str):
    """Plot cumulative PCA variance explained.

    Shows how much of the total variance in the embedding space is captured
    by each principal component. Helps assess whether the 16D space is being
    used efficiently or if dimensions are wasted.
    """
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA().fit(clip_data["embeddings"])
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_,
           alpha=0.6, label="Individual")
    ax.plot(range(1, len(cumvar) + 1), cumvar, "ro-", markersize=4,
            label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title("PCA variance explained in 16D embedding space")
    ax.legend()
    ax.set_xticks(range(1, len(cumvar) + 1))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_variance.png"), dpi=150)
    plt.close(fig)
    print(f"PCA variance plot saved to {output_dir}/")


def plot_training_curves(history: dict, output_dir: str):
    """Plot training and validation loss curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Contrastive loss (NT-Xent)")
    ax1.set_title("Training curves")
    ax1.legend()

    ax2.plot(epochs, history["lr"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Learning rate schedule")
    ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_dir}/")
