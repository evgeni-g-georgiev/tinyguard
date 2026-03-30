#!/usr/bin/env python3
"""
test_teacher.py — YAMNet + PCA(16D) + Deep SVDD teacher ceiling on MIMII.

Pipeline:
    WAV → YAMNet (1024D) → PCA (16D) → f_s SVDD → anomaly score

Normal clips are split 80/20:
  - Train (80%): fit f_s SVDD
  - Held-out (20%): set threshold (95th percentile of held-out scores)
Evaluation uses held-out normal + all abnormal clips.

Prerequisites
-------------
    python scripts/prepare_teacher.py

Usage
-----
    python scripts/test_teacher.py
"""

import glob
import os
import sys
import warnings

# Ensure project root is on the path when running scripts directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from ai_edge_litert.interpreter import Interpreter

from src.models.separator import train_fs, score_clips

# ── config ─────────────────────────────────────────────────────────────────

YAMNET_PATH   = "models/yamnet/yamnet.tflite"
DATA_ROOT     = "data"
PCA_DIR       = "outputs/pca"
OUTPUT_DIR    = "outputs/teacher_baseline"

SAMPLE_RATE   = 16_000
FRAME_LEN     = 15_600
EMB_IDX       = 115
EMB_SCALE     = 0.022328350692987442
EMB_ZP        = -128
THRESHOLD_PCT     = 95
NORMAL_TRAIN_FRAC = 0.8   # fraction of normal clips used to fit SVDD

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]


# ── YAMNet helpers ──────────────────────────────────────────────────────────

def load_yamnet(path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = Interpreter(path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    return interp, interp.get_input_details()


def embed_clip(wav_path: str, interp, input_details) -> np.ndarray | None:
    """Return (n_frames, 1024) float32 YAMNet embeddings, or None."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    n_frames  = len(audio) // FRAME_LEN
    if n_frames == 0:
        return None
    out = np.empty((n_frames, 1024), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * FRAME_LEN : (i + 1) * FRAME_LEN].astype(np.float32)
        interp.set_tensor(input_details[0]["index"], frame)
        interp.invoke()
        raw    = interp.get_tensor(EMB_IDX).squeeze()
        out[i] = (raw.astype(np.float32) - EMB_ZP) * EMB_SCALE
    return out


def project(embeddings: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project (N, 1024) → (N, 16) using pre-fitted PCA."""
    return (embeddings - mean) @ components.T


# ── per-machine evaluation ──────────────────────────────────────────────────

def evaluate_machine(
    machine_type: str,
    machine_id: str,
    interp,
    input_details,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> dict | None:

    base       = os.path.join(DATA_ROOT, machine_type, machine_id)
    norm_paths = sorted(glob.glob(os.path.join(base, "normal",   "*.wav")))
    anom_paths = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))
    if not norm_paths or not anom_paths:
        return None

    # Split normal clips into train (SVDD fitting) and held-out (threshold + eval)
    n_train     = max(1, int(len(norm_paths) * NORMAL_TRAIN_FRAC))
    train_paths = norm_paths[:n_train]
    held_paths  = norm_paths[n_train:]

    # Extract and project train-normal embeddings; fit SVDD
    train_embs = []
    for path in tqdm(train_paths, desc="  norm-tr ", leave=False, unit="clip"):
        raw = embed_clip(path, interp, input_details)
        if raw is not None:
            train_embs.append(project(raw, pca_components, pca_mean))
    if not train_embs:
        return None

    model, centroid = train_fs(np.vstack(train_embs))

    # Threshold from training scores — mirrors deployment (chip sets threshold
    # from the listen window, not a separate held-out set)
    train_scores = score_clips(train_embs, model, centroid)
    threshold    = float(np.percentile(train_scores, THRESHOLD_PCT))

    # Score held-out normal clips — used for AUC/P/R evaluation only
    held_embs = []
    for path in tqdm(held_paths, desc="  norm-ho ", leave=False, unit="clip"):
        raw = embed_clip(path, interp, input_details)
        if raw is not None:
            held_embs.append(project(raw, pca_components, pca_mean))

    norm_scores = score_clips(held_embs, model, centroid)

    # Extract, project, and score abnormal clips
    anom_embs = []
    for path in tqdm(anom_paths, desc="  abnormal", leave=False, unit="clip"):
        raw = embed_clip(path, interp, input_details)
        if raw is not None:
            anom_embs.append(project(raw, pca_components, pca_mean))
    if not anom_embs:
        return None

    anom_scores = score_clips(anom_embs, model, centroid)

    labels    = np.array([0] * len(norm_scores) + [1] * len(anom_scores))
    scores    = np.array(norm_scores + anom_scores)
    auc       = float(roc_auc_score(labels, scores))
    preds     = (scores >= threshold).astype(int)
    precision = float(precision_score(labels, preds, zero_division=0))
    recall    = float(recall_score(labels, preds, zero_division=0))

    return {
        "auc":          round(auc,       3),
        "precision":    round(precision, 3),
        "recall":       round(recall,    3),
        "threshold":    round(threshold, 5),
        "n_norm_train": len(train_embs),
        "n_norm_held":  len(norm_scores),
        "n_abnormal":   len(anom_scores),
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    comp_path = os.path.join(PCA_DIR, "pca_components.npy")
    mean_path = os.path.join(PCA_DIR, "pca_mean.npy")
    if not os.path.exists(comp_path) or not os.path.exists(mean_path):
        print("ERROR: PCA matrices not found.")
        print("Run:  python scripts/prepare_teacher.py")
        sys.exit(1)

    pca_components = np.load(comp_path)   # (16, 1024)
    pca_mean       = np.load(mean_path)   # (1024,)
    print(f"PCA: {pca_components.shape[1]}D → {pca_components.shape[0]}D")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YAMNet …")
    interp, input_details = load_yamnet(YAMNET_PATH)

    results = {}
    rows    = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            print(f"\n[{key}]")
            res = evaluate_machine(
                mtype, mid, interp, input_details, pca_components, pca_mean
            )
            if res is None:
                print("  skipped — missing data")
                continue
            results[key] = res
            rows.append((key, res))
            print(
                f"  AUC={res['auc']:.3f}  "
                f"Precision={res['precision']:.3f}  "
                f"Recall={res['recall']:.3f}  "
                f"(norm_train={res['n_norm_train']}, norm_held={res['n_norm_held']}, "
                f"abnormal={res['n_abnormal']})"
            )

    if not rows:
        print("No results — check data paths.")
        return

    aucs  = [r["auc"]       for _, r in rows]
    precs = [r["precision"] for _, r in rows]
    recs  = [r["recall"]    for _, r in rows]
    summary = {
        "mean_auc":       round(float(np.mean(aucs)),  3),
        "mean_precision": round(float(np.mean(precs)), 3),
        "mean_recall":    round(float(np.mean(recs)),  3),
    }
    results["summary"] = summary

    col = 20
    print(f"\n{'─' * 62}")
    print(f"  {'Machine':<{col}} {'AUC':>6}  {'Precision':>9}  {'Recall':>6}")
    print(f"  {'─'*col}  {'─'*6}  {'─'*9}  {'─'*6}")
    for key, r in rows:
        print(f"  {key:<{col}} {r['auc']:>6.3f}  {r['precision']:>9.3f}  {r['recall']:>6.3f}")
    print(f"  {'─'*col}  {'─'*6}  {'─'*9}  {'─'*6}")
    print(
        f"  {'Mean':<{col}} {summary['mean_auc']:>6.3f}  "
        f"{summary['mean_precision']:>9.3f}  "
        f"{summary['mean_recall']:>6.3f}"
    )
    print(f"{'─' * 62}")
    print(f"\nThreshold: {THRESHOLD_PCT}th percentile of per-machine normal scores")
    print(f"Scoring:   max-frame squared L2 distance from SVDD centroid")
    print(f"Results →  {OUTPUT_DIR}/results.yaml")

    with open(os.path.join(OUTPUT_DIR, "results.yaml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
