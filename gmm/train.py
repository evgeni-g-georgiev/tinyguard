#!/usr/bin/env python3
"""
train.py — Train and evaluate TWFR-GMM anomaly detectors for all 16 MIMII machines.

Simulates the on-device deployment cycle using the same 60-clip training window
and 3-round monitoring evaluation as the SVDD pipeline (separator/train.py +
inference/run.py), allowing direct comparison of results.

For each machine:
  1. Load 60 normal training clips, compute full-clip log-mel spectrograms.
  2. Fit a GMMDetector (self-supervised r search + GMM fit + threshold).
  3. Save the fitted artefact as {mtype}_{mid}.pkl.
  4. Evaluate on test normal and abnormal clips (3 rounds × 5 min each).
  5. Generate a timeline plot and accumulate metrics.

Results are written to results.yaml in the output directory. The YAML schema
is identical to inference/outputs/inference/results.yaml, with one additional
field per machine (``r`` — the selected GWRP decay parameter).

Usage
-----
    python gmm/train.py                          # r search, 2 GMM components
    python gmm/train.py --r 1.0                  # fix r (mean pooling)
    python gmm/train.py --r 0.0                  # fix r (max pooling)
    python gmm/train.py --n-components 1         # single Gaussian
    python gmm/train.py --verbose                # print per-r log-likelihoods
    python gmm/train.py --out-dir /tmp/gmm_test  # custom output directory

Inputs
------
  preprocessing/outputs/mimii_splits/splits.json   fixed train/test manifest
  data/mimii/                                       MIMII WAV files

Outputs
-------
  gmm/outputs/gmm/{mtype}_{mid}.pkl    fitted detector artefacts (×16)
  gmm/outputs/gmm/{mtype}_{mid}.png   per-machine timeline plots   (×16)
  gmm/outputs/gmm/results.yaml         aggregate metrics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GMM_DIR,
    MACHINE_IDS,
    MACHINE_TYPES,
    MIMII_ROOT,
    MIMII_SPLITS,
)
from gmm.detector import GMMDetector
from gmm.evaluate import evaluate_machine
from gmm.features import load_log_mel
from gmm.plot import plot_machine


# ── Argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TWFR-GMM anomaly detectors for all 16 MIMII machines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits", type=str, default=str(MIMII_SPLITS),
        help="Path to the splits.json manifest.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(GMM_DIR),
        help="Output directory for .pkl artefacts, .png plots, and results.yaml.",
    )
    parser.add_argument(
        "--n-components", type=int, default=2,
        help="Number of Gaussian mixture components.",
    )
    parser.add_argument(
        "--r", type=float, default=None,
        help=(
            "Fix the GWRP decay parameter r ∈ [0, 1]. "
            "If omitted, self-supervised r search is performed per machine "
            "(recommended — no anomaly labels required)."
        ),
    )
    parser.add_argument(
        "--threshold-pct", type=int, default=95,
        help="Percentile of training NLL scores used as the detection threshold.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for GMM initialisation.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-r mean log-likelihoods during the self-supervised r search.",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    splits_path = Path(args.splits)
    if not splits_path.exists():
        print(f"ERROR: Splits manifest not found at {splits_path}")
        print("Run:  python preprocessing/split_mimii.py")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)

    r_desc = f"r={args.r:.2f}" if args.r is not None else "r=auto (self-supervised search)"
    print(
        f"TWFR-GMM  |  {args.n_components} component(s)  |  {r_desc}  |  "
        f"threshold={args.threshold_pct}th pct\n"
    )

    all_results: dict = {}

    # ── Per-machine loop ──────────────────────────────────────────────────────
    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            if key not in splits:
                print(f"  [{key}] not in splits manifest — skipping.")
                continue

            print(f"  [{key}]")

            # 1. Load training clips and compute log-mel spectrograms.
            #    The last VAL_CLIPS are held out for threshold calibration only;
            #    the GMM is fit on the remaining clips. This prevents the GMM
            #    from scoring its own training data for threshold setting, which
            #    causes artificially negative NLLs and a threshold that no test
            #    clip can reach (100% FP).
            VAL_CLIPS = 10
            all_paths = [str(MIMII_ROOT / p) for p in splits[key]["train_normal"]]
            fit_paths = all_paths[:-VAL_CLIPS]
            val_paths = all_paths[-VAL_CLIPS:]

            fit_log_mels: list = []
            for path in tqdm(fit_paths, desc="    load fit", leave=False, unit="clip"):
                try:
                    fit_log_mels.append(load_log_mel(path))
                except RuntimeError as exc:
                    print(f"    Warning: {exc}", file=sys.stderr)

            val_log_mels: list = []
            for path in tqdm(val_paths, desc="    load val", leave=False, unit="clip"):
                try:
                    val_log_mels.append(load_log_mel(path))
                except RuntimeError as exc:
                    print(f"    Warning: {exc}", file=sys.stderr)

            if not fit_log_mels:
                print(f"    No valid training clips — skipping.\n")
                continue

            # 2. Fit detector: r search (if args.r is None) + GMM + threshold.
            #    val_log_mels is passed so threshold is calibrated on held-out data.
            detector = GMMDetector(
                n_components=args.n_components,
                threshold_pct=args.threshold_pct,
                seed=args.seed,
            )
            detector.fit(
                fit_log_mels,
                r=args.r,
                verbose=args.verbose,
                val_log_mels=val_log_mels if val_log_mels else None,
            )

            # 3. Save artefact
            artefact_path = out_dir / f"{mtype}_{mid}.pkl"
            detector.save(artefact_path)

            # 4. Evaluate: alternating normal/anomaly monitoring rounds
            result = evaluate_machine(mtype, mid, detector, splits)
            if result is None:
                print(f"    Evaluation skipped (key missing from splits).\n")
                continue

            # 5. Plot timeline
            plot_path = plot_machine(result, mtype, mid, out_dir)

            # 6. Collect results for YAML export
            all_results[key] = {
                "threshold": round(float(detector.threshold_), 5),
                "r":         round(float(detector.r_), 4),
                "n_rounds":  result["n_rounds"],
                "rounds":    result["round_results"],
            }

            # Console summary for this machine
            print(
                f"    r={detector.r_:.2f}  "
                f"threshold={detector.threshold_:.4f}  "
                f"→ {artefact_path.name}"
            )
            for rr in result["round_results"]:
                delay = (
                    f"{rr['detection_delay_secs']}s"
                    if rr["detected"] else "not detected"
                )
                thr = rr.get("threshold_round", float("nan"))
                print(
                    f"    Round {rr['round']}: "
                    f"FA={rr['n_false_pos']}  "
                    f"thr_round={thr:.1f}  "
                    f"Delay={delay}"
                )
            print(f"    Plot → {plot_path}\n")

    if not all_results:
        print("No results — check that MIMII data is present and splits manifest exists.")
        return

    # ── Overall summary (mirrors inference/run.py format) ─────────────────────
    all_rounds   = [rr for res in all_results.values() for rr in res["rounds"]]
    total_fp     = sum(rr["n_false_pos"]    for rr in all_rounds)
    total_norm   = sum(rr["n_normal_clips"] for rr in all_rounds)
    n_detected   = sum(1 for rr in all_rounds if rr["detected"])
    total_rounds = len(all_rounds)
    delays       = [rr["detection_delay_secs"] for rr in all_rounds if rr["detected"]]
    mean_delay   = float(np.mean(delays))   if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    max_delay    = int(max(delays))         if delays else None

    print(f"{'─' * 55}")
    print(f"  Machines evaluated  : {len(all_results)}")
    print(
        f"  Total normal clips  : {total_norm}  |  "
        f"False alarm events: {total_fp}"
    )
    print(
        f"  Detection rate      : {n_detected}/{total_rounds} rounds "
        f"({100 * n_detected / total_rounds:.0f}%)"
    )
    print(
        f"  Mean detection delay: {mean_delay:.0f}s  "
        f"(median {median_delay:.0f}s, max {max_delay}s)"
    )
    print(f"{'─' * 55}")
    print(f"\nPlots   → {out_dir}/*.png")
    print(f"Results → {out_dir}/results.yaml")
    print("\nTo compare with SVDD pipeline:")
    print("  python inference/run.py")

    # ── Write results YAML ────────────────────────────────────────────────────
    results_path = out_dir / "results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
