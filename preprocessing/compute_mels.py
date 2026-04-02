#!/usr/bin/env python3
"""
compute_mels.py — Compute and cache log-mel spectrograms for FSD50K eval set.

Processes files in the same sorted order as extract_embeddings.py and skips the
same clips, so every mel frame at index i corresponds to teacher embedding i.
An assertion checks the frame counts match before saving.

Inputs
------
  data/fsd50k/FSD50K.eval_audio/             WAV clips
  outputs/fsd50k_cache/eval_embeddings.npy   teacher cache (must exist)

Output
------
  outputs/fsd50k_cache/eval_mels.npy   (N_frames, 1, 64, 61) float32  ~1.5 GB

Usage
-----
    python preprocessing/compute_mels.py

Prerequisites
-------------
    python preprocessing/extract_embeddings.py   (embeddings cache must exist)
"""

import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FSD50K_AUDIO, FSDCACHE_DIR,
    SAMPLE_RATE, FRAME_LEN, N_FFT, HOP_LENGTH, N_MELS, LOG_OFFSET,
)


# ── mel computation ─────────────────────────────────────────────────────────

def log_mel(frame: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram for a single 0.975 s audio frame.

    Args:
        frame: (15600,) float32 at 16 kHz
    Returns:
        (1, 64, 61) float32 — channel-first, ready for AcousticEncoder input
    """
    mel = librosa.feature.melspectrogram(
        y=frame, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    return np.log(mel + LOG_OFFSET)[np.newaxis, :, :]   # (1, 64, 61)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    FSDCACHE_DIR.mkdir(parents=True, exist_ok=True)

    mel_path = FSDCACHE_DIR / "eval_mels.npy"
    emb_path = FSDCACHE_DIR / "eval_embeddings.npy"

    if mel_path.exists():
        m = np.load(mel_path, mmap_mode="r")
        print(f"Mel cache already exists — {mel_path}")
        print(f"  shape: {m.shape}  ({m.nbytes / 1e9:.2f} GB)")
        return

    if not emb_path.exists():
        print("ERROR: Teacher embeddings cache not found.")
        print("Run:  python preprocessing/extract_embeddings.py  first.")
        sys.exit(1)

    n_emb_frames = np.load(emb_path, mmap_mode="r").shape[0]
    print(f"Teacher embeddings: {n_emb_frames:,} frames — mel cache must match exactly.")

    wav_files = sorted(glob.glob(str(FSD50K_AUDIO / "**" / "*.wav"), recursive=True))
    if not wav_files:
        print(f"ERROR: No WAV files found under {FSD50K_AUDIO}/")
        sys.exit(1)

    print(f"Found {len(wav_files):,} FSD50K clips.")
    print("Computing log-mel spectrograms …\n")

    batches = []
    for path in tqdm(wav_files, unit="clip"):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        n_frames  = len(audio) // FRAME_LEN
        if n_frames == 0:
            continue
        for i in range(n_frames):
            frame = audio[i * FRAME_LEN : (i + 1) * FRAME_LEN].astype(np.float32)
            batches.append(log_mel(frame))

    all_mels = np.stack(batches)   # (N_frames, 1, 64, 61)

    assert all_mels.shape[0] == n_emb_frames, (
        f"Frame count mismatch: {all_mels.shape[0]} mels vs "
        f"{n_emb_frames} embeddings. "
        f"Delete both caches and re-run extract_embeddings.py then compute_mels.py."
    )

    np.save(mel_path, all_mels)
    print(f"\nCached {all_mels.shape[0]:,} frames → {mel_path}")
    print(f"  shape: {all_mels.shape}  ({all_mels.nbytes / 1e9:.2f} GB)")
    print(f"  mel range: [{all_mels.min():.2f}, {all_mels.max():.2f}]")


if __name__ == "__main__":
    main()
