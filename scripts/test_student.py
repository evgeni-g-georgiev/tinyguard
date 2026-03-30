#!/usr/bin/env python3
"""
test_student.py — Evaluate AcousticEncoder + Deep SVDD on MIMII (zero-shot).

Pipeline:
    WAV → log-mel (1, 64, 61) → AcousticEncoder → 16D → f_s SVDD → score

Normal clips are split 80/20:
  - Train (80%): fit f_s SVDD
  - Held-out (20%): set threshold (95th percentile of held-out scores)
Evaluation uses held-out normal + all abnormal clips.

Compare results against:
  - outputs/yamnet_baseline/results.yaml   (raw YAMNet + centroid)
  - outputs/teacher_baseline/results.yaml  (YAMNet + PCA + SVDD  ← target to match)

Prerequisites
-------------
    python scripts/train_student.py

Usage
-----
    python scripts/test_student.py [--checkpoint outputs/student/acoustic_encoder.pt]
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import torch
import yaml
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.cnn import AcousticEncoder
from src.models.separator import score_clips, train_fs

# ── config ─────────────────────────────────────────────────────────────────

DEFAULT_CKPT  = "outputs/student/acoustic_encoder.pt"
DATA_ROOT     = "data"
OUTPUT_DIR    = "outputs/student_baseline"

SAMPLE_RATE   = 16_000
FRAME_LEN     = 15_600
N_FFT         = 1024
HOP_LENGTH    = 256
N_MELS        = 64
LOG_OFFSET    = 1e-6
THRESHOLD_PCT     = 95
NORMAL_TRAIN_FRAC = 0.8   # fraction of normal clips used to fit SVDD

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]


# ── audio helpers ───────────────────────────────────────────────────────────

def log_mel(frame: np.ndarray) -> np.ndarray:
    """(15600,) float32 → (1, 64, 61) float32 log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=frame, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    return np.log(mel + LOG_OFFSET)[np.newaxis, :, :]


def embed_clip(wav_path: str, model: AcousticEncoder, device) -> np.ndarray | None:
    """
    Load a WAV, slice into 0.975 s frames, return (n_frames, 16) embeddings.
    Returns None if the clip is shorter than one frame.
    """
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    n_frames  = len(audio) // FRAME_LEN
    if n_frames == 0:
        return None

    mels = np.stack([
        log_mel(audio[i * FRAME_LEN : (i + 1) * FRAME_LEN].astype(np.float32))
        for i in range(n_frames)
    ])                                               # (n_frames, 1, 64, 61)

    with torch.no_grad():
        x    = torch.tensor(mels, dtype=torch.float32).to(device)
        embs = model(x).cpu().numpy()               # (n_frames, 16)
    return embs


# ── per-machine evaluation ──────────────────────────────────────────────────

def evaluate_machine(machine_type, machine_id, model, device):
    base       = os.path.join(DATA_ROOT, machine_type, machine_id)
    norm_paths = sorted(glob.glob(os.path.join(base, "normal",   "*.wav")))
    anom_paths = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))
    if not norm_paths or not anom_paths:
        return None

    # Split normal clips into train (SVDD fitting) and held-out (threshold + eval)
    n_train     = max(1, int(len(norm_paths) * NORMAL_TRAIN_FRAC))
    train_paths = norm_paths[:n_train]
    held_paths  = norm_paths[n_train:]

    # Extract embeddings for train-normal clips; fit SVDD
    train_embs = []
    for path in tqdm(train_paths, desc="  norm-tr ", leave=False, unit="clip"):
        embs = embed_clip(path, model, device)
        if embs is not None:
            train_embs.append(embs)
    if not train_embs:
        return None

    model_fs, centroid = train_fs(np.vstack(train_embs))

    # Threshold from training scores — mirrors deployment (chip sets threshold
    # from the listen window, not a separate held-out set)
    train_scores = score_clips(train_embs, model_fs, centroid)
    threshold    = float(np.percentile(train_scores, THRESHOLD_PCT))

    # Score held-out normal clips — used for AUC/P/R evaluation only
    held_embs = []
    for path in tqdm(held_paths, desc="  norm-ho ", leave=False, unit="clip"):
        embs = embed_clip(path, model, device)
        if embs is not None:
            held_embs.append(embs)

    norm_scores = score_clips(held_embs, model_fs, centroid)

    # Extract and score abnormal clips
    anom_embs = []
    for path in tqdm(anom_paths, desc="  abnormal", leave=False, unit="clip"):
        embs = embed_clip(path, model, device)
        if embs is not None:
            anom_embs.append(embs)
    if not anom_embs:
        return None

    anom_scores = score_clips(anom_embs, model_fs, centroid)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Run:  python scripts/train_student.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = AcousticEncoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Trained for {ckpt['epoch']} epochs  —  val MSE {ckpt['val_loss']:.5f}")
    print(f"  Parameters: {ckpt['param_count']:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    rows    = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            print(f"\n[{key}]")
            res = evaluate_machine(mtype, mid, model, device)
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

    # ── comparison table ───────────────────────────────────────────────────
    teacher_path = "outputs/teacher_baseline/results.yaml"
    teacher      = {}
    if os.path.exists(teacher_path):
        with open(teacher_path) as f:
            teacher = yaml.safe_load(f)

    col = 18
    W   = 78 if teacher else 62
    print(f"\n{'─' * W}")
    if teacher:
        print(f"  {'Machine':<{col}} {'Student':>7}  {'Teacher':>7}  {'Δ':>6}  "
              f"{'P':>6}  {'R':>6}")
        print(f"  {'─'*col}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
        for key, r in rows:
            t_auc = teacher.get(key, {}).get("auc", float("nan"))
            delta = r["auc"] - t_auc if not np.isnan(t_auc) else float("nan")
            d_str = f"{delta:+.3f}" if not np.isnan(delta) else "  n/a"
            print(f"  {key:<{col}} {r['auc']:>7.3f}  {t_auc:>7.3f}  {d_str:>6}  "
                  f"{r['precision']:>6.3f}  {r['recall']:>6.3f}")
        print(f"  {'─'*col}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
        t_mean = teacher.get("summary", {}).get("mean_auc", float("nan"))
        delta_mean = summary["mean_auc"] - t_mean if not np.isnan(t_mean) else float("nan")
        d_str = f"{delta_mean:+.3f}" if not np.isnan(delta_mean) else "  n/a"
        print(f"  {'Mean':<{col}} {summary['mean_auc']:>7.3f}  {t_mean:>7.3f}  "
              f"{d_str:>6}  {summary['mean_precision']:>6.3f}  "
              f"{summary['mean_recall']:>6.3f}")
    else:
        print(f"  {'Machine':<{col}} {'AUC':>6}  {'Precision':>9}  {'Recall':>6}")
        print(f"  {'─'*col}  {'─'*6}  {'─'*9}  {'─'*6}")
        for key, r in rows:
            print(f"  {key:<{col}} {r['auc']:>6.3f}  {r['precision']:>9.3f}  {r['recall']:>6.3f}")
        print(f"  {'─'*col}  {'─'*6}  {'─'*9}  {'─'*6}")
        print(f"  {'Mean':<{col}} {summary['mean_auc']:>6.3f}  "
              f"{summary['mean_precision']:>9.3f}  {summary['mean_recall']:>6.3f}")

    print(f"{'─' * W}")
    if teacher:
        print(f"\n  Δ = Student AUC − Teacher AUC  (positive = student matches or exceeds teacher)")
    print(f"\nResults → {OUTPUT_DIR}/results.yaml")

    with open(os.path.join(OUTPUT_DIR, "results.yaml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
