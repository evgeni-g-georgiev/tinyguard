#!/usr/bin/env python3
"""
prepare_teacher.py — Extract YAMNet embeddings from FSD50K and fit PCA.

Run this once before test_teacher.py or distill_general.py.
Embeddings are cached to disk so YAMNet only runs over FSD50K once.

Outputs
-------
  outputs/fsd50k_cache/eval_embeddings.npy   (N_frames, 1024) float32  ~400 MB
  outputs/pca/pca_components.npy             (16, 1024)       float32
  outputs/pca/pca_mean.npy                   (1024,)          float32

Notes
-----
- PCA is fitted on FSD50K embeddings only — never on MIMII data.
  Fitting on MIMII would be leakage since MIMII is the evaluation set.
- If the cache already exists, embedding extraction is skipped.
- If PCA outputs already exist, PCA fitting is skipped.
"""

import glob
import os
import sys
import warnings

import numpy as np
import librosa
from sklearn.decomposition import PCA
from tqdm import tqdm

from ai_edge_litert.interpreter import Interpreter

# ── config ─────────────────────────────────────────────────────────────────

YAMNET_PATH  = "models/yamnet/yamnet.tflite"
FSD50K_AUDIO = "data/fsd50k/FSD50K.eval_audio"
CACHE_DIR    = "outputs/fsd50k_cache"
PCA_DIR      = "outputs/pca"

SAMPLE_RATE    = 16_000
FRAME_LEN      = 15_600          # 0.975 s — YAMNet's expected input length
EMB_IDX        = 115             # layer28/reduce_mean, (1,1,1,1024), int8
EMB_SCALE      = 0.022328350692987442
EMB_ZP         = -128

PCA_DIMS       = 32
PCA_MAX_FRAMES = 100_000         # subsample cap for PCA fitting


# ── YAMNet helpers ──────────────────────────────────────────────────────────

def load_yamnet(path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = Interpreter(path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    return interp, interp.get_input_details()


def embed_clip(wav_path: str, interp, input_details) -> np.ndarray | None:
    """Return (n_frames, 1024) float32 embeddings for one clip, or None."""
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


# ── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PCA_DIR,   exist_ok=True)

    emb_path  = os.path.join(CACHE_DIR, "eval_embeddings.npy")
    comp_path = os.path.join(PCA_DIR,   "pca_components.npy")
    mean_path = os.path.join(PCA_DIR,   "pca_mean.npy")

    # ── Step 1: extract YAMNet embeddings ──────────────────────────────────
    if os.path.exists(emb_path):
        print(f"Cached embeddings found — loading {emb_path} …")
        all_embeddings = np.load(emb_path)
        print(f"  {all_embeddings.shape[0]:,} frames, {all_embeddings.shape[1]}D")
    else:
        wav_files = sorted(glob.glob(
            os.path.join(FSD50K_AUDIO, "**", "*.wav"), recursive=True
        ))
        if not wav_files:
            print(f"ERROR: No WAV files found under {FSD50K_AUDIO}/")
            print("Run scripts/download_fsd50k.py first.")
            sys.exit(1)

        print(f"Found {len(wav_files):,} FSD50K clips.")
        print("Extracting YAMNet embeddings — takes ~15-20 min on a multi-core machine.")
        print("(Cached after first run — subsequent calls are instant.)\n")

        interp, input_details = load_yamnet(YAMNET_PATH)

        batches = []
        skipped = 0
        for path in tqdm(wav_files, unit="clip"):
            embs = embed_clip(path, interp, input_details)
            if embs is None:
                skipped += 1
            else:
                batches.append(embs)

        all_embeddings = np.vstack(batches)   # (N_frames, 1024)
        np.save(emb_path, all_embeddings)

        print(f"\nCached {all_embeddings.shape[0]:,} frames → {emb_path}")
        if skipped:
            print(f"  ({skipped} clips skipped — shorter than one frame)")

    # ── Step 2: fit PCA ────────────────────────────────────────────────────
    if os.path.exists(comp_path) and os.path.exists(mean_path):
        print(f"\nPCA matrices already exist — skipping fit.")
        print(f"  {comp_path}")
        print(f"  {mean_path}")
    else:
        n = all_embeddings.shape[0]
        if n > PCA_MAX_FRAMES:
            rng      = np.random.default_rng(42)
            idx      = rng.choice(n, size=PCA_MAX_FRAMES, replace=False)
            fit_data = all_embeddings[idx]
            print(f"\nFitting PCA({PCA_DIMS}D) on {PCA_MAX_FRAMES:,} / {n:,} frames …")
        else:
            fit_data = all_embeddings
            print(f"\nFitting PCA({PCA_DIMS}D) on {n:,} frames …")

        pca           = PCA(n_components=PCA_DIMS)
        pca.fit(fit_data)
        var_explained = pca.explained_variance_ratio_.sum()

        np.save(comp_path, pca.components_.astype(np.float32))   # (16, 1024)
        np.save(mean_path, pca.mean_.astype(np.float32))         # (1024,)

        print(f"  Variance explained: {var_explained:.1%}")
        print(f"  Saved → {comp_path}")
        print(f"  Saved → {mean_path}")

    print("\nDone. Ready to run test_teacher.py or distill_general.py.")


if __name__ == "__main__":
    main()
