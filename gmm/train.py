#!/usr/bin/env python3
"""
train.py — Train and compare TWFR-GMM detectors across all 16 MIMII machines.

Runs three detector variants per machine and prints a side-by-side comparison:

  node_a       — Single node, r=NODE_R_A (=1.0, deployment-faithful mean pooling).
                 This is the hardware baseline: the only mode runnable on a single
                 Arduino Nano 33 BLE without exceeding its 256 KB SRAM budget.

  node_b       — Single node, r=NODE_R_B (=0.5, energy-weighted GWRP).
                 Requires a second physical device (full spectrogram buffer needed).

  node_learning — Two-node NodeLearning system: both nodes train independently,
                  then exchange confidence signals (μ_val, σ_val) and fuse their
                  z-normalised anomaly scores with fit-quality weights.
                  Implements the "collaborative learning" regime of the Node Learning
                  paradigm (Kanjo & Aslanov, 2026, §2, Figure 2).

Node Learning workflow (per machine)
-------------------------------------
  1. Load N_FIT_CLIPS normal clips.  Extract features at r=NODE_R_A for Node A
     and r=NODE_R_B for Node B independently.
  2. Each node fits its own 2-component diagonal GMM locally (eq. 1-2).
  3. Each node calibrates threshold and CUSUM from N_VAL_CLIPS held-out clips,
     storing (μ_val, σ_val) as confidence signals.
  4. NodeLearning reads these signals, computes fit-quality weights
     w_i = softmax(−μ_val_i), and calibrates a fused CUSUM on the weighted
     z-score distribution (eq. 3, 5).
  5. Evaluate all three variants on the same test set.
  6. Save artefacts, plots, and YAML summaries.

Usage
-----
  python gmm/train.py                           # -6 dB dataset (default)
  python gmm/train.py --dataset 0db             # 0 dB dataset
  python gmm/train.py --dataset 6db             # +6 dB dataset
  python gmm/train.py --verbose                 # print per-machine calibration
  python gmm/train.py --dataset 0db --verbose

Inputs
------
  preprocessing/outputs/mimii_splits/splits_{dataset}.json
  data/mimii_{dataset}/

Outputs  (all overwrite on re-run — no stale files accumulate)
-------
  gmm/outputs/{dataset}/node_a/
      {mtype}_{mid}.pkl    — fitted GMMDetector artefacts  (×16)
      {mtype}_{mid}.png    — timeline plots                (×16)
      results.yaml         — per-machine metrics
  gmm/outputs/{dataset}/node_b/
      (same structure)
  gmm/outputs/{dataset}/node_learning/
      {mtype}_{mid}.png    — timeline plots (fused z-score)(×16)
      results.yaml
  gmm/outputs/{dataset}/comparison.yaml  — full 3-way comparison
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GMM_6DB_DIR,
    GMM_0DB_DIR,
    GMM_NEG6DB_DIR,
    MACHINE_IDS,
    MACHINE_TYPES,
    MIMII_6DB_ROOT,
    MIMII_0DB_ROOT,
    MIMII_NEG6DB_ROOT,
    MIMII_6DB_SPLITS,
    MIMII_0DB_SPLITS,
    MIMII_NEG6DB_SPLITS,
)
from gmm.config import (
    N_FIT_CLIPS, N_MELS, N_TRAIN_CLIPS, N_VAL_CLIPS,
    NODE_R_A, NODE_R_B, R_CANDIDATES,
)
from gmm.detector import GMMDetector
from gmm.evaluate import evaluate_machine
from gmm.features import load_log_mel
from gmm.node_learning import NodeLearning
from gmm.plot import plot_machine


# ── Dataset registry ──────────────────────────────────────────────────────────
# Maps the --dataset argument to (MIMII_ROOT, SPLITS_PATH, OUTPUT_DIR).
# Adding a new dataset variant requires only a new entry here and corresponding
# constants in the root config.py.

_DATASET_CONFIG = {
    "neg6db": (MIMII_NEG6DB_ROOT, MIMII_NEG6DB_SPLITS, GMM_NEG6DB_DIR),
    "0db":    (MIMII_0DB_ROOT,    MIMII_0DB_SPLITS,    GMM_0DB_DIR),
    "6db":    (MIMII_6DB_ROOT,    MIMII_6DB_SPLITS,    GMM_6DB_DIR),
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare TWFR-GMM detectors for all 16 MIMII machines.\n"
            "Runs three variants: single-node r=NODE_R_A, single-node r=NODE_R_B,\n"
            "and two-node NodeLearning (collaborative inference)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_CONFIG),
        default="neg6db",
        help="Which MIMII SNR dataset to use.",
    )
    parser.add_argument(
        "--splits", type=str, default=None,
        help="Override path to splits JSON (default: inferred from --dataset).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output root directory (default: inferred from --dataset).",
    )
    parser.add_argument(
        "--n-mels", type=int, default=N_MELS, metavar="N",
        help=(
            f"Number of mel frequency bins (default: {N_MELS}). "
            "When not the default, outputs go to <dataset>_<N>mel/ automatically."
        ),
    )
    parser.add_argument(
        "--r-search", action="store_true",
        help=(
            f"Enable per-node r-search over R_CANDIDATES={R_CANDIDATES}. "
            "Each node independently selects the r that minimises its mean "
            "validation NLL, mirroring the r-search in deployment/tinyml_gmm.ino. "
            "Without this flag, Node A uses r=NODE_R_A and Node B uses r=NODE_R_B. "
            "Outputs go to <dataset>_rsearch/ automatically."
        ),
    )
    parser.add_argument(
        "--mic-a", type=int, default=0, metavar="N",
        help=(
            "Microphone channel index for Node A (0-7, default: 0). "
            "MIMII recordings contain 8 channels from a circular mic array. "
            "Using distinct channels per node simulates physically separated devices."
        ),
    )
    parser.add_argument(
        "--mic-b", type=int, default=1, metavar="N",
        help="Microphone channel index for Node B (0-7, default: 1).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-machine calibration parameters and node weights.",
    )
    return parser


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_clips(paths: list[str], desc: str, n_mels: int = N_MELS, channel: int | None = None) -> list:
    """Load log-mel spectrograms, skipping files that fail to load."""
    log_mels: list = []
    for path in tqdm(paths, desc=desc, leave=False, unit="clip"):
        try:
            log_mels.append(load_log_mel(path, n_mels=n_mels, channel=channel))
        except RuntimeError as exc:
            print(f"    Warning: {exc}", file=sys.stderr)
    return log_mels


def _fit_node(
    fit_log_mels: list,
    val_log_mels: list,
    r_candidates: list[float],
    n_mels: int,
) -> GMMDetector:
    """Fit a GMMDetector, selecting the best r from r_candidates.

    Single candidate → straightforward fixed-r fit (standard run).
    Multiple candidates → fit one detector per r, return the one with the
    lowest mean validation NLL (best fit quality).  This mirrors the
    r-search loop in deployment/tinyml_gmm.ino exactly: for each r,
    fit the GMM, score the val clips, keep the best.

    Parameters
    ----------
    fit_log_mels : list of np.ndarray, shape (n_mels, T)
    val_log_mels : list of np.ndarray, shape (n_mels, T)
    r_candidates : list of float   — GWRP decay values to evaluate
    n_mels : int                   — must match the loaded spectrograms

    Returns
    -------
    GMMDetector — fitted and calibrated, with the best r selected.
    """
    best: GMMDetector | None = None
    for r in r_candidates:
        det = GMMDetector(r=r, n_mels=n_mels)
        det.fit(fit_log_mels, val_log_mels)
        if best is None or det.mu_val_ < best.mu_val_:
            best = det
    return best


def _round_summary(round_results: list[dict]) -> str:
    """Format per-round metrics as a single console line."""
    parts = []
    for rr in round_results:
        delay = f"{rr['detection_delay_secs']}s" if rr["detected"] else "miss"
        parts.append(f"R{rr['round']}:FA={rr['n_false_pos']},delay={delay}")
    return "  ".join(parts)


def _result_row(result: dict | None) -> dict:
    """Extract aggregate scalar metrics from an evaluate_machine result dict."""
    if result is None:
        return {"detected": 0, "total": 0, "n_fp": 0, "n_norm": 0, "delays": [], "auc": None}
    rrs = result["round_results"]
    events = result["events"]
    scores = [e["score"] for e in events]
    labels = [1 if e["phase"] == "anomaly" else 0 for e in events]
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else None
    return {
        "detected": sum(1 for rr in rrs if rr["detected"]),
        "total":    len(rrs),
        "n_fp":     sum(rr["n_false_pos"] for rr in rrs),
        "n_norm":   sum(rr["n_normal_clips"] for rr in rrs),
        "delays":   [rr["detection_delay_secs"] for rr in rrs if rr["detected"]],
        "auc":      auc,
    }


def _augment_result(result: dict, r_desc: str, score_label: str) -> dict:
    """Inject plot metadata keys into an evaluate_machine result dict."""
    result["r_desc"]      = r_desc
    result["score_label"] = score_label
    return result


def _collect_yaml_entry(result: dict, det) -> dict:
    """Build the per-machine YAML entry for one variant."""
    return {
        "threshold": round(float(det.threshold_), 5),
        "cusum_k":   round(float(det.cusum_k_), 5),
        "cusum_h":   round(float(det.cusum_h_), 5),
        "n_rounds":  result["n_rounds"],
        "rounds":    result["round_results"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = _build_parser().parse_args()
    mimii_root, splits_default, out_default = _DATASET_CONFIG[args.dataset]

    splits_path = Path(args.splits)  if args.splits  else splits_default
    out_root    = Path(args.out_dir) if args.out_dir else out_default

    # Auto-suffix output dir so different configurations never overwrite each
    # other.  Explicit --out-dir always takes precedence over both suffixes.
    if not args.out_dir:
        if args.n_mels != N_MELS:
            out_root = out_root.parent / f"{out_root.name}_{args.n_mels}mel"
        if args.r_search:
            out_root = out_root.parent / f"{out_root.name}_rsearch"

    # Per-node r candidate lists — single-element list = fixed-r run.
    r_cands_a = R_CANDIDATES if args.r_search else [NODE_R_A]
    r_cands_b = R_CANDIDATES if args.r_search else [NODE_R_B]

    if not splits_path.exists():
        print(f"ERROR: Splits manifest not found at {splits_path}")
        print(
            f"Run:  python preprocessing/split_mimii.py "
            f"--data-root {mimii_root} --out {splits_path}"
        )
        sys.exit(1)

    # Output subdirectories — created fresh, overwrite on re-run.
    dir_node_a        = out_root / "node_a"
    dir_node_b        = out_root / "node_b"
    dir_node_learning = out_root / "node_learning"
    for d in (dir_node_a, dir_node_b, dir_node_learning):
        d.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)

    if args.r_search:
        node_a_desc = f"mic{args.mic_a}  r-search over {r_cands_a}"
        node_b_desc = f"mic{args.mic_b}  r-search over {r_cands_b}"
    else:
        node_a_desc = f"mic{args.mic_a}  r={NODE_R_A}"
        node_b_desc = f"mic{args.mic_b}  r={NODE_R_B}"

    print(
        f"TWFR-GMM Node Learning comparison  [{args.dataset}]\n"
        f"  Node A: {node_a_desc}\n"
        f"  Node B: {node_b_desc}\n"
        f"  n_mels={args.n_mels}  fit clips={N_FIT_CLIPS}  val clips={N_VAL_CLIPS}\n"
        f"  outputs → {out_root}\n"
    )

    results_node_a:        dict = {}
    results_node_b:        dict = {}
    results_node_learning: dict = {}
    comparison:            dict = {}

    # ── Per-machine loop ──────────────────────────────────────────────────────
    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            if key not in splits:
                print(f"  [{key}] not in splits manifest — skipping.")
                continue

            print(f"  [{key}]")

            # Load training clips.
            # The manifest provides N_TRAIN_CLIPS=60 normal clips.
            # The last N_VAL_CLIPS=10 are held out for calibration;
            # the GMM is fit on the first N_FIT_CLIPS=50.
            all_paths = [str(mimii_root / p) for p in splits[key]["train_normal"]]
            if len(all_paths) < N_TRAIN_CLIPS:
                print(
                    f"    WARNING: expected {N_TRAIN_CLIPS} train clips, "
                    f"got {len(all_paths)}."
                )

            fit_paths = all_paths[:N_FIT_CLIPS]
            val_paths = all_paths[N_FIT_CLIPS:N_FIT_CLIPS + N_VAL_CLIPS]

            fit_log_mels_a = _load_clips(fit_paths, "    fit A", n_mels=args.n_mels, channel=args.mic_a)
            val_log_mels_a = _load_clips(val_paths, "    val A", n_mels=args.n_mels, channel=args.mic_a)
            fit_log_mels_b = _load_clips(fit_paths, "    fit B", n_mels=args.n_mels, channel=args.mic_b)
            val_log_mels_b = _load_clips(val_paths, "    val B", n_mels=args.n_mels, channel=args.mic_b)

            if not fit_log_mels_a or not fit_log_mels_b:
                print(f"    No valid fit clips — skipping.\n")
                continue
            if not val_log_mels_a or not val_log_mels_b:
                print(f"    No valid val clips — skipping.\n")
                continue

            # ── Step 1: Fit Node A ────────────────────────────────────────────
            det_a = _fit_node(fit_log_mels_a, val_log_mels_a, r_cands_a, args.n_mels)
            det_a.channel_ = args.mic_a
            det_a.save(dir_node_a / f"{mtype}_{mid}.pkl")

            # ── Step 2: Fit Node B ────────────────────────────────────────────
            det_b = _fit_node(fit_log_mels_b, val_log_mels_b, r_cands_b, args.n_mels)
            det_b.channel_ = args.mic_b
            det_b.save(dir_node_b / f"{mtype}_{mid}.pkl")

            # ── Step 3: Build NodeLearning (pairwise collaborative inference) ─
            # Nodes exchange confidence signals (μ_val, σ_val) and calibrate
            # a fused CUSUM.  Paper eq. 3 (merge operator) and eq. 5
            # (context-weighted interaction).
            node_learning = NodeLearning(det_a, det_b)
            node_learning.channel_a_ = args.mic_a
            node_learning.channel_b_ = args.mic_b

            if args.verbose:
                print(
                    f"    Node A (r={det_a.r_}): "
                    f"thr={det_a.threshold_:.4f}  "
                    f"cusum_h={det_a.cusum_h_:.4f}  "
                    f"μ_val={det_a.mu_val_:.4f}"
                )
                print(
                    f"    Node B (r={det_b.r_}): "
                    f"thr={det_b.threshold_:.4f}  "
                    f"cusum_h={det_b.cusum_h_:.4f}  "
                    f"μ_val={det_b.mu_val_:.4f}"
                )
                print(
                    f"    NodeLearning: "
                    f"w_A={node_learning.w_a_:.4f}  w_B={node_learning.w_b_:.4f}  "
                    f"thr={node_learning.threshold_:.4f}  "
                    f"cusum_h={node_learning.cusum_h_:.4f}"
                )

            # ── Step 4: Evaluate all three variants on the same test set ──────
            res_a = evaluate_machine(mtype, mid, det_a,         splits, mimii_root)
            res_b = evaluate_machine(mtype, mid, det_b,         splits, mimii_root)
            res_f = evaluate_machine(mtype, mid, node_learning, splits, mimii_root)

            if res_a is None or res_b is None or res_f is None:
                print(f"    Evaluation skipped.\n")
                continue

            # ── Step 5: Inject plot metadata and save timeline PNGs ──────────
            _augment_result(res_a, f"mic{args.mic_a}  r={det_a.r_}", "Anomaly score  (NLL)")
            _augment_result(res_b, f"mic{args.mic_b}  r={det_b.r_}", "Anomaly score  (NLL)")
            _augment_result(res_f, f"mic{args.mic_a}/mic{args.mic_b}  r={det_a.r_}/{det_b.r_}  node learning",
                            "Anomaly score  (fused z-score)")

            plot_machine(res_a, mtype, mid, dir_node_a)
            plot_machine(res_b, mtype, mid, dir_node_b)
            plot_machine(res_f, mtype, mid, dir_node_learning)

            # ── Step 6: Collect results for YAML export ───────────────────────
            results_node_a[key]        = _collect_yaml_entry(res_a, det_a)
            results_node_b[key]        = _collect_yaml_entry(res_b, det_b)
            results_node_learning[key] = _collect_yaml_entry(res_f, node_learning)

            row_a = _result_row(res_a)
            row_b = _result_row(res_b)
            row_f = _result_row(res_f)

            comparison[key] = {
                "node_a": {
                    "r":      det_a.r_,
                    "det":    row_a["detected"],
                    "total":  row_a["total"],
                    "n_fp":   row_a["n_fp"],
                    "n_norm": row_a["n_norm"],
                    "delays": row_a["delays"],
                    "auc":    row_a["auc"],
                },
                "node_b": {
                    "r":      det_b.r_,
                    "det":    row_b["detected"],
                    "total":  row_b["total"],
                    "n_fp":   row_b["n_fp"],
                    "n_norm": row_b["n_norm"],
                    "delays": row_b["delays"],
                    "auc":    row_b["auc"],
                },
                "node_learning": {
                    "r":      f"{det_a.r_}/{det_b.r_}",
                    "w_a":    round(node_learning.w_a_, 4),
                    "w_b":    round(node_learning.w_b_, 4),
                    "det":    row_f["detected"],
                    "total":  row_f["total"],
                    "n_fp":   row_f["n_fp"],
                    "n_norm": row_f["n_norm"],
                    "delays": row_f["delays"],
                    "auc":    row_f["auc"],
                },
            }

            print(
                f"    Node A (r={det_a.r_}):   "
                f"{_round_summary(res_a['round_results'])}"
            )
            print(
                f"    Node B (r={det_b.r_}):   "
                f"{_round_summary(res_b['round_results'])}"
            )
            print(
                f"    NodeLearning:      "
                f"{_round_summary(res_f['round_results'])}"
                f"  [w_A={node_learning.w_a_:.3f} w_B={node_learning.w_b_:.3f}]"
            )
            print()

    if not comparison:
        print(
            "No results — check that MIMII data is present and the splits "
            "manifest exists for this dataset."
        )
        return

    # ── Aggregate summary table ───────────────────────────────────────────────
    def _agg(rows: list[dict]) -> dict:
        det    = sum(r["det"]   for r in rows)
        total  = sum(r["total"] for r in rows)
        fp     = sum(r["n_fp"]  for r in rows)
        norm   = sum(r["n_norm"] for r in rows)
        delays = [d for r in rows for d in r["delays"]]
        aucs   = [r["auc"] for r in rows if r["auc"] is not None]
        return {
            "det_rate":   f"{det}/{total} ({100*det/total:.0f}%)" if total else "n/a",
            "fa_pct":     f"{100*fp/norm:.1f}%"                   if norm  else "n/a",
            "mean_delay": f"{np.mean(delays):.0f}s"               if delays else "n/a",
            "mean_auc":   f"{np.mean(aucs):.4f}"                  if aucs  else "n/a",
        }

    rows_a = [v["node_a"]        for v in comparison.values()]
    rows_b = [v["node_b"]        for v in comparison.values()]
    rows_f = [v["node_learning"] for v in comparison.values()]

    agg_a = _agg(rows_a)
    agg_b = _agg(rows_b)
    agg_f = _agg(rows_f)

    def _r_label(rows: list[dict]) -> str:
        """Return the r value if all machines selected the same r, else 'search'."""
        unique = {str(row["r"]) for row in rows}
        return next(iter(unique)) if len(unique) == 1 else "search"

    lbl_a = _r_label(rows_a)
    lbl_b = _r_label(rows_b)
    lbl_f = _r_label(rows_f)

    sep = "─" * 75
    print(sep)
    print(f"  {'Variant':<26}  {'Detection':<14}  {'FA%':<8}  {'Mean delay':<12}  AUC")
    print(sep)
    print(
        f"  {f'Node A (r={lbl_a})':<26}  "
        f"{agg_a['det_rate']:<14}  {agg_a['fa_pct']:<8}  {agg_a['mean_delay']:<12}  {agg_a['mean_auc']}"
    )
    print(
        f"  {f'Node B (r={lbl_b})':<26}  "
        f"{agg_b['det_rate']:<14}  {agg_b['fa_pct']:<8}  {agg_b['mean_delay']:<12}  {agg_b['mean_auc']}"
    )
    print(
        f"  {f'NodeLearning ({lbl_f})':<26}  "
        f"{agg_f['det_rate']:<14}  {agg_f['fa_pct']:<8}  {agg_f['mean_delay']:<12}  {agg_f['mean_auc']}"
    )
    print(sep)
    print(f"\nPlots   → {out_root}/{{node_a,node_b,node_learning}}/*.png")
    print(f"Results → {out_root}/comparison.yaml\n")

    # ── Write YAMLs (overwrite on re-run) ─────────────────────────────────────
    for path, data in [
        (dir_node_a        / "results.yaml", results_node_a),
        (dir_node_b        / "results.yaml", results_node_b),
        (dir_node_learning / "results.yaml", results_node_learning),
        (out_root          / "comparison.yaml", comparison),
    ]:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
