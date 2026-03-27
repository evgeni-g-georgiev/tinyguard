"""
Simulate node learning: Fisher-weighted scoring calibrated by peer encounters.

For each pair of MIMII machines (nodes A and B):
  1. Each node trains standard SVDD (weights frozen after this)
  2. Each node computes per-dimension variance on projected normal clips
  3. Nodes exchange prototypes + variance vectors via simulated BLE
  4. Gating: friend -> pool variances, foe -> cross-project for Fisher weights
  5. Score with Fisher-weighted Mahalanobis (no model retraining)

Reports four conditions:
  - f_c only (centroid in 16D PCA space)
  - f_c + f_s SVDD isotropic (isolated, no peers) — the proven baseline
  - f_c + f_s SVDD Mahalanobis (own variance, no peers)
  - f_c + f_s SVDD + Fisher node learning (peer-calibrated weights)

Usage:
    python scripts/simulate_node_learning.py
"""

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fs.node_learning import FisherNode, gate_peer, PrototypeMemory
from src.fs.model import FsSeparator
from sklearn.metrics import roc_auc_score

EMBEDDINGS_DIR = "outputs/embeddings"
BASELINE_DIR = "outputs/fc_baseline"
OUTPUT_DIR = "outputs/node_learning"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]


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


def fc_only_auc(normal, abnormal):
    """f_c-only baseline: centroid distance in raw 16D PCA space."""
    centroid = normal.mean(axis=0)
    scores = np.concatenate([
        np.linalg.norm(normal - centroid, axis=1),
        np.linalg.norm(abnormal - centroid, axis=1),
    ])
    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(abnormal))])
    return roc_auc_score(labels, scores)


def svdd_isotropic_auc(normal, abnormal, seed=42):
    """f_c+f_s SVDD baseline: isotropic scoring, no peers."""
    torch.manual_seed(seed)
    model = FsSeparator(16, 8)
    train_t = torch.from_numpy(normal)
    model.init_centroid(train_t)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(100):
        perm = torch.randperm(len(train_t))
        for start in range(0, len(train_t), 64):
            batch = train_t[perm[start:start + 64]]
            proj = model(batch)
            loss = ((proj - model.centroid) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        n_scores = model.anomaly_score(torch.from_numpy(normal)).numpy()
        a_scores = model.anomaly_score(torch.from_numpy(abnormal)).numpy()

    scores = np.concatenate([n_scores, a_scores])
    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(abnormal))])
    return roc_auc_score(labels, scores)


def mahalanobis_auc(normal, abnormal, seed=42):
    """SVDD + Mahalanobis scoring (own variance, no peers)."""
    node = FisherNode("solo", "solo", seed=seed)
    node.ingest_normal_clips(normal)
    node.train_svdd()
    node.compute_own_variance()
    # Score with Mahalanobis (no peer encounters)
    all_embs = np.concatenate([normal, abnormal])
    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(abnormal))])
    scores = node.score(all_embs)
    return roc_auc_score(labels, scores)


def calibrate_r_gate(machine_keys, valid_machines, tau_spawn, seed):
    """Calibrate R_gate from prototype distances using Youden's J."""
    proto_sets = {}
    for key in machine_keys:
        node = FisherNode("cal", key, tau_spawn=tau_spawn, seed=seed)
        node.ingest_normal_clips(valid_machines[key]["normal"])
        proto_sets[key] = node.memory.P_normal

    intra_dists, inter_dists = [], []
    for (ka, kb) in combinations(machine_keys, 2):
        pa, pb = proto_sets[ka], proto_sets[kb]
        if len(pa) == 0 or len(pb) == 0:
            continue
        d_min = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min()
        type_a, type_b = ka.split("/")[0], kb.split("/")[0]
        (intra_dists if type_a == type_b else inter_dists).append(d_min)

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)

    print(f"  Intra-type (same):  min={intra_dists.min():.3f}  "
          f"median={np.median(intra_dists):.3f}  max={intra_dists.max():.3f}  (n={len(intra_dists)})")
    print(f"  Inter-type (diff):  min={inter_dists.min():.3f}  "
          f"median={np.median(inter_dists):.3f}  max={inter_dists.max():.3f}  (n={len(inter_dists)})")

    sorted_dists = np.sort(np.unique(np.concatenate([intra_dists, inter_dists])))
    best_thresh, best_j = sorted_dists[0], -1
    for t in sorted_dists:
        tpr = np.sum(intra_dists <= t) / len(intra_dists)
        fpr = np.sum(inter_dists <= t) / len(inter_dists)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresh = t

    print(f"  Auto-calibrated R_gate = {best_thresh:.3f}  (Youden's J = {best_j:.3f})")
    return float(best_thresh), intra_dists, inter_dists


def simulate_pair(key_a, data_a, key_b, data_b, R_gate, tau_spawn, beta, seed):
    """Simulate 2-node deployment with Fisher-weighted scoring."""
    node_a = FisherNode("A", key_a, tau_spawn=tau_spawn, seed=seed)
    node_b = FisherNode("B", key_b, tau_spawn=tau_spawn, seed=seed + 1)

    # Phase 1: ingest + train SVDD + compute variance (all local, no peers)
    for node, data in [(node_a, data_a), (node_b, data_b)]:
        node.ingest_normal_clips(data["normal"])
        node.train_svdd()
        node.compute_own_variance()

    # Phase 2: BLE encounter — exchange prototypes + variances
    decision_a = node_a.receive_peer(node_b, R_gate=R_gate)
    decision_b = node_b.receive_peer(node_a, R_gate=R_gate)

    # Update weights with specified beta
    node_a._update_weights(beta=beta)
    node_b._update_weights(beta=beta)

    # Phase 3: evaluate
    def eval_node(node, data):
        all_embs = np.concatenate([data["normal"], data["abnormal"]])
        labels = np.concatenate([np.zeros(len(data["normal"])),
                                  np.ones(len(data["abnormal"]))])
        auc_fisher = roc_auc_score(labels, node.score(all_embs))
        return auc_fisher

    return {
        "auc_a": eval_node(node_a, data_a),
        "auc_b": eval_node(node_b, data_b),
        "gate_a": decision_a,
        "gate_b": decision_b,
        "weights_a": node_a.dim_weights.copy(),
        "weights_b": node_b.dim_weights.copy(),
    }


def simulate_triple(keys, all_data, R_gate, tau_spawn, beta, seed):
    """Simulate 3-node deployment."""
    nodes = []
    for i, key in enumerate(keys):
        node = FisherNode(f"N{i}", key, tau_spawn=tau_spawn, seed=seed + i)
        node.ingest_normal_clips(all_data[key]["normal"])
        node.train_svdd()
        node.compute_own_variance()
        nodes.append(node)

    # All pairs exchange
    decisions = {}
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                d = nodes[i].receive_peer(nodes[j], R_gate=R_gate)
                decisions[(i, j)] = d

    # Update weights
    for node in nodes:
        node._update_weights(beta=beta)

    # Evaluate
    aucs = []
    for i, node in enumerate(nodes):
        data = all_data[keys[i]]
        all_embs = np.concatenate([data["normal"], data["abnormal"]])
        labels = np.concatenate([np.zeros(len(data["normal"])),
                                  np.ones(len(data["abnormal"]))])
        aucs.append(roc_auc_score(labels, node.score(all_embs)))

    return aucs, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Max pairs to evaluate (0 = all)")
    parser.add_argument("--n-triples", type=int, default=20,
                        help="Number of random 3-node combos to sample")
    parser.add_argument("--R-gate", type=float, default=0,
                        help="Friend/foe gating threshold (0 = auto-calibrate)")
    parser.add_argument("--tau-spawn", type=float, default=2.0,
                        help="Prototype spawning distance threshold")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Foe Fisher weight strength (0 = pure Mahalanobis)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings...")
    machines = load_machine_embeddings()
    valid_machines = {k: v for k, v in machines.items()
                      if len(v["normal"]) > 0 and len(v["abnormal"]) > 0}
    machine_keys = sorted(valid_machines.keys())
    print(f"Found {len(machine_keys)} machines with both normal and abnormal clips\n")

    # ── Calibrate R_gate ──
    print("Calibrating R_gate from prototype distances...")
    if args.R_gate <= 0:
        args.R_gate, intra_dists, inter_dists = calibrate_r_gate(
            machine_keys, valid_machines, args.tau_spawn, args.seed)
    else:
        intra_dists = inter_dists = np.array([0.0])
        print(f"  Using provided R_gate = {args.R_gate:.3f}")
    print()

    # ── Baselines (computed once per machine) ──
    print("Computing baselines (4 conditions)...")
    fc_aucs = {}
    svdd_aucs = {}
    maha_aucs = {}
    for key in machine_keys:
        data = valid_machines[key]
        fc_aucs[key] = fc_only_auc(data["normal"], data["abnormal"])
        svdd_aucs[key] = svdd_isotropic_auc(data["normal"], data["abnormal"], seed=args.seed)
        maha_aucs[key] = mahalanobis_auc(data["normal"], data["abnormal"], seed=args.seed)

    mean_fc = np.mean(list(fc_aucs.values()))
    mean_svdd = np.mean(list(svdd_aucs.values()))
    mean_maha = np.mean(list(maha_aucs.values()))

    print(f"  f_c only mean AUC:            {mean_fc:.4f}")
    print(f"  f_c+f_s SVDD (isotropic):     {mean_svdd:.4f}")
    print(f"  f_c+f_s SVDD (Mahalanobis):   {mean_maha:.4f}")

    print(f"\n  Per-machine breakdown:")
    print(f"  {'Machine':<20s}  {'f_c':>7s}  {'SVDD':>7s}  {'Maha':>7s}  {'M-S':>6s}")
    print(f"  {'-'*52}")
    for key in machine_keys:
        delta = maha_aucs[key] - svdd_aucs[key]
        print(f"  {key:<20s}  {fc_aucs[key]:>7.4f}  {svdd_aucs[key]:>7.4f}  "
              f"{maha_aucs[key]:>7.4f}  {delta:>+6.4f}")
    print(f"  {'-'*52}")
    print(f"  {'MEAN':<20s}  {mean_fc:>7.4f}  {mean_svdd:>7.4f}  "
          f"{mean_maha:>7.4f}  {mean_maha - mean_svdd:>+6.4f}")
    print()

    # ── 2-node simulation ──
    print("=" * 70)
    print("2-NODE SIMULATION (Fisher-weighted scoring)")
    print("=" * 70)

    all_pairs = list(combinations(machine_keys, 2))
    if args.max_pairs > 0:
        all_pairs = all_pairs[:args.max_pairs]
    print(f"Evaluating {len(all_pairs)} pairs (beta={args.beta})...\n")

    pair_results = []
    nl_aucs_by_machine = defaultdict(list)
    same_type_results = []
    diff_type_results = []

    for i, (key_a, key_b) in enumerate(all_pairs):
        result = simulate_pair(
            key_a, valid_machines[key_a],
            key_b, valid_machines[key_b],
            R_gate=args.R_gate, tau_spawn=args.tau_spawn,
            beta=args.beta, seed=args.seed,
        )
        result["machine_a"] = key_a
        result["machine_b"] = key_b
        pair_results.append(result)

        nl_aucs_by_machine[key_a].append(result["auc_a"])
        nl_aucs_by_machine[key_b].append(result["auc_b"])

        type_a = key_a.split("/")[0]
        type_b = key_b.split("/")[0]
        (same_type_results if type_a == type_b else diff_type_results).append(result)

        if (i + 1) % 20 == 0 or i == len(all_pairs) - 1:
            print(f"  [{i+1}/{len(all_pairs)}] pairs done")

    # Summary table
    print(f"\n{'Pair type':<15s}  {'Count':>5s}  {'Mean AUC (A)':>12s}  {'Mean AUC (B)':>12s}  {'Gating':>12s}")
    print("-" * 65)
    if same_type_results:
        mean_a = np.mean([r["auc_a"] for r in same_type_results])
        mean_b = np.mean([r["auc_b"] for r in same_type_results])
        n_friend = sum(1 for r in same_type_results if r["gate_a"] == "friend")
        print(f"{'Same type':<15s}  {len(same_type_results):>5d}  {mean_a:>12.4f}  {mean_b:>12.4f}  "
              f"{n_friend:>4d} friend")
    if diff_type_results:
        mean_a = np.mean([r["auc_a"] for r in diff_type_results])
        mean_b = np.mean([r["auc_b"] for r in diff_type_results])
        n_foe = sum(1 for r in diff_type_results if r["gate_a"] == "foe")
        print(f"{'Different type':<15s}  {len(diff_type_results):>5d}  {mean_a:>12.4f}  {mean_b:>12.4f}  "
              f"{n_foe:>4d} foe")

    # Per-machine: all 4 conditions
    print(f"\n{'Machine':<20s}  {'f_c':>7s}  {'SVDD':>7s}  {'Maha':>7s}  {'Fisher':>7s}  {'F-S':>6s}")
    print("-" * 62)
    nl_mean_aucs = {}
    for key in machine_keys:
        nl_mean = np.mean(nl_aucs_by_machine[key]) if nl_aucs_by_machine[key] else 0
        nl_mean_aucs[key] = nl_mean
        delta = nl_mean - svdd_aucs[key]
        print(f"{key:<20s}  {fc_aucs[key]:>7.4f}  {svdd_aucs[key]:>7.4f}  "
              f"{maha_aucs[key]:>7.4f}  {nl_mean:>7.4f}  {delta:>+6.4f}")

    mean_nl = np.mean(list(nl_mean_aucs.values()))
    print("-" * 62)
    print(f"{'MEAN':<20s}  {mean_fc:>7.4f}  {mean_svdd:>7.4f}  "
          f"{mean_maha:>7.4f}  {mean_nl:>7.4f}  {mean_nl - mean_svdd:>+6.4f}")

    # ── 3-node simulation ──
    print(f"\n{'=' * 70}")
    print("3-NODE SIMULATION (Fisher-weighted scoring)")
    print(f"{'=' * 70}")

    rng = np.random.RandomState(args.seed)
    all_triples = list(combinations(machine_keys, 3))
    n_triples = min(args.n_triples, len(all_triples))
    triple_indices = rng.choice(len(all_triples), size=n_triples, replace=False)
    selected_triples = [all_triples[i] for i in triple_indices]

    print(f"Evaluating {n_triples} random 3-node combos...\n")

    triple_aucs = []
    for i, keys in enumerate(selected_triples):
        aucs, decisions = simulate_triple(
            keys, valid_machines,
            R_gate=args.R_gate, tau_spawn=args.tau_spawn,
            beta=args.beta, seed=args.seed,
        )
        triple_aucs.append(np.mean(aucs))
        n_foe = sum(1 for d in decisions.values() if d == "foe")
        if (i + 1) % 5 == 0 or i == n_triples - 1:
            print(f"  [{i+1}/{n_triples}] {keys} -> mean AUC {np.mean(aucs):.4f} "
                  f"({n_foe}/6 foe links)")

    mean_triple_nl = np.mean(triple_aucs)
    print(f"\n3-node mean AUC: {mean_triple_nl:.4f}  "
          f"(vs f_c {mean_fc:.4f}, SVDD {mean_svdd:.4f}, Maha {mean_maha:.4f})")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: per-machine bar chart (4 conditions)
    ax = axes[0]
    x = np.arange(len(machine_keys))
    w = 0.2
    ax.bar(x - 1.5*w, [fc_aucs[k] for k in machine_keys], w,
           label="f_c only", color="steelblue", alpha=0.8)
    ax.bar(x - 0.5*w, [svdd_aucs[k] for k in machine_keys], w,
           label="SVDD", color="coral", alpha=0.8)
    ax.bar(x + 0.5*w, [maha_aucs[k] for k in machine_keys], w,
           label="Mahalanobis", color="goldenrod", alpha=0.8)
    ax.bar(x + 1.5*w, [nl_mean_aucs[k] for k in machine_keys], w,
           label="Fisher+NL", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("/", "\n") for k in machine_keys], fontsize=5)
    ax.set_ylabel("AUC")
    ax.set_title("Per-machine AUC: 4 conditions")
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Fisher vs SVDD delta by pair type
    ax = axes[1]
    if same_type_results:
        same_deltas = []
        for r in same_type_results:
            same_deltas.append(r["auc_a"] - svdd_aucs[r["machine_a"]])
            same_deltas.append(r["auc_b"] - svdd_aucs[r["machine_b"]])
        ax.hist(same_deltas, bins=20, alpha=0.6,
                label=f"Same type (n={len(same_type_results)})", color="steelblue")
    if diff_type_results:
        diff_deltas = []
        for r in diff_type_results:
            diff_deltas.append(r["auc_a"] - svdd_aucs[r["machine_a"]])
            diff_deltas.append(r["auc_b"] - svdd_aucs[r["machine_b"]])
        ax.hist(diff_deltas, bins=20, alpha=0.6,
                label=f"Diff type (n={len(diff_type_results)})", color="coral")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("AUC change vs isotropic SVDD")
    ax.set_ylabel("Count")
    ax.set_title("Fisher NL delta distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: improvement per machine
    ax = axes[2]
    improvements = [nl_mean_aucs[k] - svdd_aucs[k] for k in machine_keys]
    colors = ["seagreen" if d >= 0 else "coral" for d in improvements]
    ax.barh(range(len(machine_keys)), improvements, color=colors, alpha=0.8)
    ax.set_yticks(range(len(machine_keys)))
    ax.set_yticklabels(machine_keys, fontsize=7)
    ax.set_xlabel("AUC change vs isotropic SVDD")
    ax.set_title("Fisher NL vs SVDD (per machine)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "node_learning_results.png"), dpi=150)
    plt.close()

    # ── Save results ──
    summary = {
        "config": {
            "R_gate": float(args.R_gate),
            "tau_spawn": args.tau_spawn,
            "beta": args.beta,
            "approach": "Fisher-weighted Mahalanobis (no model retraining)",
            "extra_memory": "8 floats = 32 bytes for dimension weights",
            "n_pairs_evaluated": len(all_pairs),
            "n_triples_evaluated": n_triples,
        },
        "baselines": {
            "mean_fc_auc": float(mean_fc),
            "mean_svdd_auc": float(mean_svdd),
            "mean_mahalanobis_auc": float(mean_maha),
        },
        "node_learning_2node": {
            "mean_fisher_auc": float(mean_nl),
            "improvement_over_svdd": float(mean_nl - mean_svdd),
            "improvement_over_mahalanobis": float(mean_nl - mean_maha),
        },
        "node_learning_3node": {
            "mean_fisher_auc": float(mean_triple_nl),
            "improvement_over_svdd": float(mean_triple_nl - mean_svdd),
        },
        "per_machine": {
            k: {
                "fc_auc": float(fc_aucs[k]),
                "svdd_auc": float(svdd_aucs[k]),
                "mahalanobis_auc": float(maha_aucs[k]),
                "fisher_nl_auc": float(nl_mean_aucs[k]),
            }
            for k in machine_keys
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.yaml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
