"""
extract_embeddings.py — Distillation stage 1: build teacher targets from FSD50K.

This module extracts YAMNet embeddings from FSD50K audio and fits the PCA
projection used for AcousticEncoder distillation. It is primarily an internal
stage of the distillation pipeline and is typically invoked by
distillation/train.py, but it can also be run directly to rebuild the cached
teacher targets.

Inputs
------
  data/fsd50k/FSD50K.eval_audio/          WAV clips
  data/yamnet/yamnet.tflite               YAMNet TFLite model

Outputs
-------
  distillation/outputs/fsd50k_cache/eval_embeddings.npy
  distillation/outputs/pca/pca_components.npy
  distillation/outputs/pca/pca_mean.npy

Notes
-----
- PCA is fitted on FSD50K embeddings only.
- Cached outputs are reused if they already exist.
"""


import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FSD50K_AUDIO, FSDCACHE_DIR, PCA_DIR,
    SAMPLE_RATE, EMB_IDX, EMB_SCALE, EMB_ZP, PCA_DIMS, chunk_seconds,
)
from preprocessing.loader import load_audio, split_into_chunks
from preprocessing.yamnet_loading import load_yamnet



PCA_MAX_FRAMES = 100_000   # subsample cap for PCA fitting



def _get_extract_output_paths():
    """Return output paths for the embedding cache and PCA files."""
    emb_path = FSDCACHE_DIR / "eval_embeddings.npy"
    comp_path = PCA_DIR / "pca_components.npy"
    mean_path = PCA_DIR / "pca_mean.npy"
    return emb_path, comp_path, mean_path

def _list_fsd50k_audio_files():
    """List all FSD50K evaluation audio files in sorted order."""
    wav_files = sorted(glob.glob(str(FSD50K_AUDIO / "**" / "*.wav"), recursive=True))

    if not wav_files:
        print(f"ERROR: No WAV files found under {FSD50K_AUDIO}/")
        print("Run:  python data/download_fsd50k.py  first.")
        sys.exit(1)

    return wav_files


def _embed_clip(wav_path: str, interp, input_details) -> np.ndarray | None:
    """Extract YAMNet embeddings for all chunks in one audio clip.

    Args:
    - wav_path: Path to the input audio file.
    - interp: Allocated YAMNet TFLite interpreter.
    - input_details: Input tensor metadata returned by the interpreter.

    Returns:
    - A float32 array of shape (n_chunks, 1024) containing one
      embedding per chunk, or None if the clip is shorter than one chunk.
    """
    audio, _ = load_audio(wav_path, sampling_frequency=SAMPLE_RATE, mono=True)
    chunks = split_into_chunks(audio, sampling_frequency=SAMPLE_RATE, chunk_seconds=chunk_seconds)

    if len(chunks) == 0:
        return None

    out = np.empty((len(chunks), 1024), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        chunk = chunk.astype(np.float32)
        interp.set_tensor(input_details[0]["index"], chunk)
        interp.invoke()
        raw = interp.get_tensor(EMB_IDX).squeeze()
        # Dequantise INT8 output back to float32:
        # YAMNet's embedding tensor is quantised; EMB_ZP and EMB_SCALE are read
        # from the tensor's quantisation parameters in the .tflite flatbuffer.
        out[i] = (raw.astype(np.float32) - EMB_ZP) * EMB_SCALE

    return out


def _extract_embeddings_from_files(wav_files, interp, input_details):
    """Extract and stack YAMNet embeddings from multiple audio files.

    Args:
    - wav_files: Sorted list of input audio file paths.
    - interp: Allocated YAMNet TFLite interpreter.
    - input_details: Input tensor metadata returned by the interpreter.

    Returns:
        A tuple (all_embeddings, skipped), where 'all_embeddings' is a
        float32 array of shape '(N_chunks, 1024)' and 'skipped' is the
        number of files shorter than one chunk.
    """
    batches = []
    skipped = 0

    for path in tqdm(wav_files, unit="clip"):
        embs = _embed_clip(path, interp, input_details)
        if embs is None:
            skipped += 1
        else:
            batches.append(embs)

    all_embeddings = np.vstack(batches)   # (N_frames, 1024)

    return all_embeddings, skipped

def _ensure_pca_outputs(all_embeddings, comp_path, mean_path):
    """Fit PCA on cached embeddings and save the PCA outputs.

    Args:
    - all_embeddings: Float32 embedding array of shape ``(N_chunks, 1024)``.
    - comp_path: Output path for the PCA component matrix.
    - mean_path: Output path for the PCA mean vector.
    """
    if comp_path.exists() and mean_path.exists():
        print(f"\nPCA matrices already exist — skipping fit.")
        print(f"  {comp_path}")
        print(f"  {mean_path}")
        return

    n = all_embeddings.shape[0]
    if n > PCA_MAX_FRAMES:
        # PCA memory scales quadratically with n_samples; subsample to cap
        # memory and runtime without meaningfully affecting the fit.
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=PCA_MAX_FRAMES, replace=False)
        fit_data = all_embeddings[idx]
        print(f"\nFitting PCA({PCA_DIMS}D) on {PCA_MAX_FRAMES:,} / {n:,} frames …")
    else:
        fit_data = all_embeddings
        print(f"\nFitting PCA({PCA_DIMS}D) on {n:,} frames …")

    pca = PCA(n_components=PCA_DIMS)
    pca.fit(fit_data)
    var_explained = pca.explained_variance_ratio_.sum()

    np.save(comp_path, pca.components_.astype(np.float32))   # (32, 1024)
    np.save(mean_path, pca.mean_.astype(np.float32))         # (1024,)

    print(f"  Variance explained: {var_explained:.1%}")
    print(f"  Saved to {comp_path}")
    print(f"  Saved to {mean_path}")


def extract_embeddings():
    """Extract YAMNet embeddings from FSD50K and ensure PCA outputs exist.

    Returns:
        A dictionary containing the saved output paths and the total number
        of extracted chunks.
    """
    FSDCACHE_DIR.mkdir(parents=True, exist_ok=True)
    PCA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Define output paths for the embedding cache and PCA files
    emb_path, comp_path, mean_path = _get_extract_output_paths()

    # Step 2: Load cached embeddings if they already exist, otherwise extract them
    if emb_path.exists():
        print(f"Cached embeddings found — loading {emb_path} …")
        all_embeddings = np.load(emb_path)
        print(f"  {all_embeddings.shape[0]:,} frames, {all_embeddings.shape[1]}D")
    else:
        wav_files = _list_fsd50k_audio_files()

        print(f"Found {len(wav_files):,} FSD50K clips.")
        print("Extracting YAMNet embeddings — takes ~15-20 min on a multi-core machine.")
        print("(Cached after first run — subsequent calls are instant.)\n")

        interp, input_details = load_yamnet()
        all_embeddings, skipped = _extract_embeddings_from_files(wav_files, interp, input_details)

        np.save(emb_path, all_embeddings)

        print(f"\nCached {all_embeddings.shape[0]:,} frames -> {emb_path}")
        if skipped:
            print(f"  ({skipped} clips skipped — shorter than one chunk)")

    # Step 3: Ensure PCA outputs exist for projecting teacher embeddings to 32D
    _ensure_pca_outputs(all_embeddings, comp_path, mean_path)

    print("\nDone. Ready to run distillation/compute_mels.py.")

    return {
        "embeddings_path": emb_path,
        "pca_components_path": comp_path,
        "pca_mean_path": mean_path,
        "n_chunks": all_embeddings.shape[0],
    }


if __name__ == "__main__":
    extract_embeddings()
