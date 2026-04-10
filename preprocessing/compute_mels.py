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
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FSD50K_AUDIO, FSDCACHE_DIR,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, LOG_OFFSET, chunk_seconds,
)
from preprocessing.loader import load_audio, split_into_chunks
from preprocessing.mel_spectrogram import make_log_mel_spectrogram



def _get_mel_output_paths():
    """Return output paths for the mel cache and teacher embedding cache."""
    mel_path = FSDCACHE_DIR / "eval_mels.npy"
    emb_path = FSDCACHE_DIR / "eval_embeddings.npy"
    return mel_path, emb_path


def _load_teacher_frame_count(emb_path):
    """Load the expected number of teacher embeddings from the cache.

    Args:
    - emb_path: Path to the cached teacher embedding file.

    Returns:
        The number of cached teacher embeddings.
    """
    if not emb_path.exists():
        print("ERROR: Teacher embeddings cache not found.")
        print("Run:  python preprocessing/extract_embeddings.py  first.")
        sys.exit(1)

    return np.load(emb_path, mmap_mode="r").shape[0]


def _list_fsd50k_audio_files():
    """List all FSD50K evaluation audio files in sorted order."""
    wav_files = sorted(glob.glob(str(FSD50K_AUDIO / "**" / "*.wav"), recursive=True))

    if not wav_files:
        print(f"ERROR: No WAV files found under {FSD50K_AUDIO}/")
        sys.exit(1)

    return wav_files



def _log_mel(chunk):
    """Compute a log-mel spectrogram for one audio chunk."""
    return make_log_mel_spectrogram(
        waveform=chunk,
        chunk_seconds=chunk_seconds,
        sampling_frequency=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH, 
    ) # (1, 64, 61)


def _compute_mels_from_files(wav_files):
    """Compute and stack mel spectrograms from multiple audio files.

    Args:
    - wav_files: Sorted list of input audio file paths.

    Returns:
    - A float32 array of shape (N_chunks, 1, 64, 61) containing one mel spectrogram per chunk.
    """
    batches = []

    for path in tqdm(wav_files, unit="clip"):
        audio, _ = load_audio(path, sampling_frequency=SAMPLE_RATE, mono=True)
        chunks = split_into_chunks(audio, sampling_frequency=SAMPLE_RATE, chunk_seconds=chunk_seconds)

        if len(chunks) == 0:
            continue

        for chunk in chunks:
            batches.append(_log_mel(chunk.astype(np.float32)))

    return np.stack(batches) # (N_frames, 1, 64, 61)



def compute_mels():
    """Compute log-mel spectrograms for FSD50K and cache them to disk.

    Returns:
    - A dictionary containing the saved mel cache path and the total number of computed chunks.
    """
    FSDCACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Define output paths for the mel cache and teacher embedding cache
    mel_path, emb_path = _get_mel_output_paths()

    # Step 2: Return early if the mel cache already exists
    if mel_path.exists():
        m = np.load(mel_path, mmap_mode="r")
        print(f"Mel cache already exists — {mel_path}")
        print(f"  shape: {m.shape}  ({m.nbytes / 1e9:.2f} GB)")
        return {
            "mels_path": mel_path,
            "n_chunks": m.shape[0],
        }

    # Step 3: Load the expected number of teacher embeddings for alignment
    n_emb_frames = _load_teacher_frame_count(emb_path)
    print(f"Teacher embeddings: {n_emb_frames:,} frames — mel cache must match exactly.")

    # Step 4: Collect input audio files and compute mel spectrograms
    wav_files = _list_fsd50k_audio_files()
    print(f"Found {len(wav_files):,} FSD50K clips.")
    print("Computing log-mel spectrograms …\n")

    all_mels = _compute_mels_from_files(wav_files)

    # Step 5: Check alignment with teacher embeddings and save the mel cache
    assert all_mels.shape[0] == n_emb_frames, (
        f"Frame count mismatch: {all_mels.shape[0]} mels vs "
        f"{n_emb_frames} embeddings. "
        f"Delete both caches and re-run extract_embeddings.py then compute_mels.py."
    )

    np.save(mel_path, all_mels)

    print(f"\nCached {all_mels.shape[0]:,} frames → {mel_path}")
    print(f"  shape: {all_mels.shape}  ({all_mels.nbytes / 1e9:.2f} GB)")
    print(f"  mel range: [{all_mels.min():.2f}, {all_mels.max():.2f}]")

    return {
        "mels_path": mel_path,
        "n_chunks": all_mels.shape[0],
    }


if __name__ == "__main__":
    compute_mels()
