"""
Evaluation metrics for f_c embedding quality.

These simulate the real deployment pipeline:
  1. Compute embeddings for all windows using the frozen encoder
  2. Aggregate window embeddings to clip-level (mean pooling)
  3. For each machine: compute centroid from normal clips, score test clips
     by distance to centroid, compute AUC

Key metrics:
  - Per-machine AUC: the primary metric (simulates on-device anomaly detection)
  - Separation ratio: mean anomaly distance / mean normal distance to centroid
  - Silhouette score: overall embedding cluster quality
  - Overlap percentage: fraction of anomalies within the 95th percentile of
    normal distances (lower is better; target < 60% mean across machines)
  - Cosine similarity stats: mean/std pairwise cosine similarity within normal
    clusters (detects representation collapse; target mean < 0.9, std > 0.05)
"""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_embeddings(encoder, loader: DataLoader, device: torch.device) -> dict:
    """Compute embeddings and collect metadata for all evaluation samples.

    Returns dict with keys:
        embeddings: (N, D) numpy array
        machine_types: list of str
        machine_ids: list of str
        labels: list of str ("normal" / "abnormal")
        clip_indices: list of int
    """
    encoder.eval()
    all_embs = []
    all_meta = defaultdict(list)

    for batch_tensor, batch_meta in loader:
        batch_tensor = batch_tensor.to(device)
        emb = encoder(batch_tensor).cpu().numpy()
        all_embs.append(emb)

        for key in ["machine_type", "machine_id", "label"]:
            all_meta[key].extend(batch_meta[key])
        all_meta["clip_index"].extend(batch_meta["clip_index"].tolist()
                                      if torch.is_tensor(batch_meta["clip_index"])
                                      else batch_meta["clip_index"])

    return {
        "embeddings":    np.concatenate(all_embs, axis=0),
        "machine_types": all_meta["machine_type"],
        "machine_ids":   all_meta["machine_id"],
        "labels":        all_meta["label"],
        "clip_indices":  all_meta["clip_index"],
    }


def aggregate_to_clips(data: dict) -> dict:
    """Average window-level embeddings to clip-level.

    Returns same structure but with one embedding per clip.
    """
    clip_groups = defaultdict(lambda: {"embs": [], "meta": None})

    for i, clip_idx in enumerate(data["clip_indices"]):
        key = (data["machine_types"][i], data["machine_ids"][i], clip_idx)
        clip_groups[key]["embs"].append(data["embeddings"][i])
        if clip_groups[key]["meta"] is None:
            clip_groups[key]["meta"] = {
                "machine_type": data["machine_types"][i],
                "machine_id":   data["machine_ids"][i],
                "label":        data["labels"][i],
                "clip_index":   clip_idx,
            }

    embeddings = []
    machine_types, machine_ids, labels, clip_indices = [], [], [], []

    for key, group in clip_groups.items():
        embeddings.append(np.mean(group["embs"], axis=0))
        meta = group["meta"]
        machine_types.append(meta["machine_type"])
        machine_ids.append(meta["machine_id"])
        labels.append(meta["label"])
        clip_indices.append(meta["clip_index"])

    return {
        "embeddings":    np.array(embeddings),
        "machine_types": machine_types,
        "machine_ids":   machine_ids,
        "labels":        labels,
        "clip_indices":  clip_indices,
    }


def per_machine_auc(clip_data: dict) -> dict:
    """Compute centroid-distance AUC for each machine.

    For each machine:
      1. Compute centroid from its normal clip embeddings
      2. Score all its clips by Euclidean distance to centroid
      3. Compute AUC (anomaly = positive class)

    This exactly simulates the deployment pipeline.

    Returns:
        dict mapping "machine_type/machine_id" → AUC, plus "mean" key.
    """
    machines = defaultdict(lambda: {"normal_embs": [], "all_embs": [], "all_labels": []})

    for i in range(len(clip_data["labels"])):
        key = f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        emb   = clip_data["embeddings"][i]
        label = clip_data["labels"][i]

        machines[key]["all_embs"].append(emb)
        machines[key]["all_labels"].append(1 if label == "abnormal" else 0)
        if label == "normal":
            machines[key]["normal_embs"].append(emb)

    results = {}
    for machine_key, data in sorted(machines.items()):
        normal_embs = np.array(data["normal_embs"])
        all_embs    = np.array(data["all_embs"])
        all_labels  = np.array(data["all_labels"])

        if len(normal_embs) == 0 or len(np.unique(all_labels)) < 2:
            continue

        centroid  = normal_embs.mean(axis=0)
        distances = np.linalg.norm(all_embs - centroid, axis=1)
        auc       = roc_auc_score(all_labels, distances)
        results[machine_key] = auc

    if results:
        results["mean"] = np.mean(list(results.values()))

    return results


def separation_ratio(clip_data: dict) -> dict:
    """Per-machine ratio of mean anomaly distance to mean normal distance.

    Values > 1 indicate anomalies are farther from centroid than normals.
    Target: > 1.2 for usable separation.
    """
    machines = defaultdict(lambda: {"normal_embs": [], "abnormal_embs": []})

    for i in range(len(clip_data["labels"])):
        key = f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        emb = clip_data["embeddings"][i]
        if clip_data["labels"][i] == "normal":
            machines[key]["normal_embs"].append(emb)
        else:
            machines[key]["abnormal_embs"].append(emb)

    results = {}
    for machine_key, data in sorted(machines.items()):
        if not data["normal_embs"] or not data["abnormal_embs"]:
            continue

        normal_embs   = np.array(data["normal_embs"])
        abnormal_embs = np.array(data["abnormal_embs"])
        centroid      = normal_embs.mean(axis=0)

        normal_dists   = np.linalg.norm(normal_embs   - centroid, axis=1)
        abnormal_dists = np.linalg.norm(abnormal_embs - centroid, axis=1)

        mean_normal   = normal_dists.mean()
        mean_abnormal = abnormal_dists.mean()

        if mean_normal > 0:
            results[machine_key] = float(mean_abnormal / mean_normal)

    if results:
        results["mean"] = np.mean([v for k, v in results.items() if k != "mean"])

    return results


def compute_silhouette(clip_data: dict) -> float:
    """Compute silhouette score using machine identity as cluster labels.

    Measures whether embeddings from the same machine are close together
    and embeddings from different machines are far apart.
    """
    labels = [
        f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        for i in range(len(clip_data["labels"]))
    ]

    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return 0.0

    label_to_int = {lbl: i for i, lbl in enumerate(sorted(unique_labels))}
    int_labels   = [label_to_int[lbl] for lbl in labels]

    return float(silhouette_score(clip_data["embeddings"], int_labels))


def overlap_percentage(clip_data: dict) -> dict:
    """Fraction of anomalous clips whose centroid distance falls within the
    95th percentile of normal clip distances (per machine).

    A value of 0% means all anomalies are clearly separated. The project
    target is < 60% averaged across machines.

    Args:
        clip_data: dict from aggregate_to_clips() with embeddings and metadata.

    Returns:
        dict mapping "machine_type/machine_id" → overlap fraction [0, 1],
        plus a "mean" key.
    """
    machines = defaultdict(lambda: {"normal_embs": [], "abnormal_embs": []})

    for i in range(len(clip_data["labels"])):
        key = f"{clip_data['machine_types'][i]}/{clip_data['machine_ids'][i]}"
        emb = clip_data["embeddings"][i]
        if clip_data["labels"][i] == "normal":
            machines[key]["normal_embs"].append(emb)
        else:
            machines[key]["abnormal_embs"].append(emb)

    results = {}
    for machine_key, data in sorted(machines.items()):
        if not data["normal_embs"] or not data["abnormal_embs"]:
            continue

        normal_embs   = np.array(data["normal_embs"])
        abnormal_embs = np.array(data["abnormal_embs"])
        centroid      = normal_embs.mean(axis=0)

        normal_dists   = np.linalg.norm(normal_embs   - centroid, axis=1)
        abnormal_dists = np.linalg.norm(abnormal_embs - centroid, axis=1)

        # 95th percentile of normal distances as threshold
        threshold = float(np.percentile(normal_dists, 95))

        # Fraction of anomalies whose distance is BELOW the threshold
        # (i.e., they overlap with the normal cluster)
        overlap = float(np.mean(abnormal_dists <= threshold))
        results[machine_key] = overlap

    if results:
        results["mean"] = float(np.mean([v for k, v in results.items() if k != "mean"]))

    return results


def cosine_similarity_stats(clip_data: dict) -> dict:
    """Mean and std of pairwise cosine similarities within normal clusters.

    Used to detect the OpenL3 failure mode: if mean cosine similarity is
    > 0.95 with very low std, the encoder has collapsed to a near-constant
    output and will fail at anomaly detection.

    Target: mean < 0.9 and std > 0.05 (meaningful embedding spread).

    Args:
        clip_data: dict from aggregate_to_clips() with embeddings and metadata.

    Returns:
        dict with "mean_cosine_sim" and "std_cosine_sim" computed over all
        pairwise cosine similarities between normal clip embeddings.
    """
    normal_embs = [
        clip_data["embeddings"][i]
        for i in range(len(clip_data["labels"]))
        if clip_data["labels"][i] == "normal"
    ]

    if len(normal_embs) < 2:
        return {"mean_cosine_sim": float("nan"), "std_cosine_sim": float("nan")}

    emb_matrix = np.array(normal_embs)  # (M, D)

    # L2 normalise each embedding row
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).clip(min=1e-8)
    emb_norm = emb_matrix / norms

    # Full pairwise cosine similarity matrix: (M, M)
    cos_sim = emb_norm @ emb_norm.T

    # Extract upper triangle (excluding diagonal) to get unique pairs
    triu_idx   = np.triu_indices(len(normal_embs), k=1)
    pair_sims  = cos_sim[triu_idx]

    return {
        "mean_cosine_sim": float(pair_sims.mean()),
        "std_cosine_sim":  float(pair_sims.std()),
    }


def evaluate_encoder(encoder, loader: DataLoader, device: torch.device) -> dict:
    """Run full evaluation pipeline.

    Returns dict with all metrics and intermediate data for visualisation.
    """
    print("Computing embeddings...")
    window_data = compute_embeddings(encoder, loader, device)
    print(f"  {len(window_data['embeddings'])} window embeddings computed")

    print("Aggregating to clip level...")
    clip_data = aggregate_to_clips(window_data)
    print(f"  {len(clip_data['embeddings'])} clip embeddings")

    print("Computing metrics...")
    auc_results       = per_machine_auc(clip_data)
    sep_results       = separation_ratio(clip_data)
    silhouette        = compute_silhouette(clip_data)
    overlap_results   = overlap_percentage(clip_data)
    cos_sim_stats     = cosine_similarity_stats(clip_data)

    return {
        "window_data":           window_data,
        "clip_data":             clip_data,
        "auc":                   auc_results,
        "separation_ratio":      sep_results,
        "silhouette":            silhouette,
        "overlap_percentage":    overlap_results,
        "cosine_similarity_stats": cos_sim_stats,
    }


def print_evaluation(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (simulated deployment)")
    print("=" * 60)

    print("\nPer-machine AUC (centroid distance scoring):")
    for key, val in sorted(results["auc"].items()):
        marker = " <<<" if key == "mean" else ""
        print(f"  {key:30s}: {val:.4f}{marker}")

    print("\nSeparation ratio (anomaly_dist / normal_dist, target > 1.2):")
    for key, val in sorted(results["separation_ratio"].items()):
        marker = " <<<" if key == "mean" else ""
        print(f"  {key:30s}: {val:.4f}{marker}")

    print(f"\nSilhouette score (cluster quality): {results['silhouette']:.4f}")

    if results.get("overlap_percentage"):
        print("\nOverlap percentage (anomalies within 95th pct of normal, target < 60%):")
        for key, val in sorted(results["overlap_percentage"].items()):
            marker = " <<<" if key == "mean" else ""
            print(f"  {key:30s}: {val * 100:.1f}%{marker}")

    if results.get("cosine_similarity_stats"):
        cs = results["cosine_similarity_stats"]
        print(
            f"\nCosine similarity (normal embeddings, target mean < 0.9, std > 0.05):"
            f"\n  mean = {cs['mean_cosine_sim']:.4f}"
            f"\n  std  = {cs['std_cosine_sim']:.4f}"
        )

    print("=" * 60)
