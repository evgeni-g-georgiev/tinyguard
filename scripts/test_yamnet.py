#!/usr/bin/env python3
"""
test_yamnet.py — Zero-shot YAMNet baseline on MIMII.

Uses the frozen YAMNet TFLite model (no fine-tuning) as a feature extractor.
For each of the 16 MIMII machines:
  1. Split normal clips 80/20 (train / held-out).
  2. Extract 1024D embeddings; compute centroid from train-normal frames only.
  3. Score each clip as the max per-frame squared L2 distance from centroid.
  4. Set threshold at the 95th percentile of held-out normal scores.
  5. Evaluate on held-out normal + all abnormal clips; report AUC, Precision, Recall.

Splitting normal clips avoids the leakage of setting the threshold on clips the
centroid was already fit to.
"""

import os
import glob
import numpy as np
import librosa
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from ai_edge_litert.interpreter import Interpreter

# ── config ─────────────────────────────────────────────────────────────────

YAMNET_PATH   = "models/yamnet/yamnet.tflite"
DATA_ROOT     = "data"
OUTPUT_DIR    = "outputs/yamnet_baseline"

SAMPLE_RATE   = 16_000
FRAME_LEN     = 15_600          # 0.975 s at 16 kHz — matches YAMNet's expected input
EMB_IDX       = 115             # tensor: layer28/reduce_mean, shape (1,1,1,1024), int8
EMB_SCALE     = 0.022328350692987442
EMB_ZP        = -128            # dequantise: (raw_int8 - EMB_ZP) * EMB_SCALE
THRESHOLD_PCT = 95              # percentile of normal scores used as detection threshold
NORMAL_TRAIN_FRAC = 0.8        # fraction of normal clips used to fit centroid

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]


# ── YAMNet inference ────────────────────────────────────────────────────────

def load_yamnet(path: str):
    # experimental_preserve_all_tensors=True is required to read intermediate
    # tensors (like the 1024D embedding at index 115) after invoke().
    interp = Interpreter(path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    return interp, interp.get_input_details()


def embed_clip(wav_path: str, interp, input_details) -> np.ndarray | None:
    """
    Load a WAV file and return per-frame 1024D embeddings, shape (n_frames, 1024).
    Returns None if the clip is shorter than one frame.
    """
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    n_frames  = len(audio) // FRAME_LEN
    if n_frames == 0:
        return None

    embeddings = np.empty((n_frames, 1024), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * FRAME_LEN : (i + 1) * FRAME_LEN].astype(np.float32)
        interp.set_tensor(input_details[0]["index"], frame)
        interp.invoke()
        raw = interp.get_tensor(EMB_IDX).squeeze()                  # int8 (1024,)
        embeddings[i] = (raw.astype(np.float32) - EMB_ZP) * EMB_SCALE
    return embeddings


# ── scoring ─────────────────────────────────────────────────────────────────

def score_clip(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    """Max-frame squared L2 distance from centroid — catches partial anomalies."""
    per_frame = np.sum((embeddings - centroid) ** 2, axis=1)   # (n_frames,)
    return float(np.max(per_frame))


# ── per-machine evaluation ──────────────────────────────────────────────────

def evaluate_machine(machine_type: str, machine_id: str, interp, input_details) -> dict | None:
    base     = os.path.join(DATA_ROOT, machine_type, machine_id)
    norm_paths = sorted(glob.glob(os.path.join(base, "normal",   "*.wav")))
    anom_paths = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))

    if not norm_paths or not anom_paths:
        return None

    # Split normal clips into train (centroid fitting) and held-out (threshold + eval)
    n_train = max(1, int(len(norm_paths) * NORMAL_TRAIN_FRAC))
    train_paths  = norm_paths[:n_train]
    held_paths   = norm_paths[n_train:]

    # Extract embeddings for train-normal clips; fit centroid
    train_embs = []
    for path in tqdm(train_paths, desc="  norm-tr ", leave=False, unit="clip"):
        embs = embed_clip(path, interp, input_details)
        if embs is not None:
            train_embs.append(embs)
    if not train_embs:
        return None

    centroid = np.mean(np.vstack(train_embs), axis=0)   # (1024,)

    # Score train-normal clips — derive threshold from these (mirrors deployment)
    train_scores = [score_clip(embs, centroid) for embs in train_embs]
    threshold    = float(np.percentile(train_scores, THRESHOLD_PCT))

    # Score held-out normal clips — used for AUC/P/R evaluation only
    held_embs = []
    for path in tqdm(held_paths, desc="  norm-ho ", leave=False, unit="clip"):
        embs = embed_clip(path, interp, input_details)
        if embs is not None:
            held_embs.append(embs)

    norm_scores = [score_clip(embs, centroid) for embs in held_embs]

    # Score abnormal clips
    anom_scores = []
    for path in tqdm(anom_paths, desc="  abnormal", leave=False, unit="clip"):
        embs = embed_clip(path, interp, input_details)
        if embs is not None:
            anom_scores.append(score_clip(embs, centroid))
    if not anom_scores:
        return None

    # Metrics
    labels = np.array([0] * len(norm_scores) + [1] * len(anom_scores))
    scores = np.array(norm_scores + anom_scores)
    auc    = float(roc_auc_score(labels, scores))
    preds  = (scores >= threshold).astype(int)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YAMNet …")
    interp, input_details = load_yamnet(YAMNET_PATH)
    print(f"  Input : waveform float32 [{FRAME_LEN}]")
    print(f"  Embed : tensor {EMB_IDX}, int8 (1024,), scale={EMB_SCALE:.5f}, zp={EMB_ZP}")

    results = {}
    rows    = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            print(f"\n[{key}]")
            res = evaluate_machine(mtype, mid, interp, input_details)
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
        print("No results — check data path.")
        return

    # Summary
    aucs   = [r["auc"]       for _, r in rows]
    precs  = [r["precision"] for _, r in rows]
    recs   = [r["recall"]    for _, r in rows]

    summary = {
        "mean_auc":       round(float(np.mean(aucs)),  3),
        "mean_precision": round(float(np.mean(precs)), 3),
        "mean_recall":    round(float(np.mean(recs)),  3),
    }
    results["summary"] = summary

    # Print table
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
    print(f"Scoring:   max-frame squared L2 distance from normal centroid")
    print(f"Results:   {OUTPUT_DIR}/results.yaml")

    out_path = os.path.join(OUTPUT_DIR, "results.yaml")
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
