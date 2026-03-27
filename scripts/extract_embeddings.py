"""Extract 1024D YAMNet embeddings from all MIMII clips and cache to disk.

YAMNet processes 0.975s frames (15600 samples at 16kHz).
Each 10s MIMII clip yields ~10 embedding frames of shape (1024,).

Output structure:
    outputs/embeddings/
        fan/id_00/normal/00000000.npy      — (N_frames, 1024) float32
        fan/id_00/abnormal/00000000.npy
        ...
        metadata.yaml                      — clip count, machine types, etc.

Usage:
    python scripts/extract_embeddings.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# YAMNet constants
YAMNET_SR = 16000
YAMNET_FRAME_LEN = 15600  # 0.975s at 16kHz — YAMNet's expected input length
EMBEDDING_TENSOR_INDEX = 115  # tower0/network/layer28/reduce_mean → (1,1,1,1024)

MODEL_PATH = "models/yamnet/yamnet.tflite"
DATA_ROOT = "data/"
OUTPUT_DIR = "outputs/embeddings"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]


def load_interpreter():
    """Load the YAMNet TFLite model with intermediate tensor preservation."""
    from ai_edge_litert.interpreter import Interpreter
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = Interpreter(
            model_path=MODEL_PATH,
            experimental_preserve_all_tensors=True,
        )
    interp.allocate_tensors()

    # Cache dequantization params for the embedding tensor
    for d in interp.get_tensor_details():
        if d["index"] == EMBEDDING_TENSOR_INDEX:
            qp = d["quantization_parameters"]
            interp._emb_scale = qp["scales"][0]
            interp._emb_zp = qp["zero_points"][0]
            break

    return interp


def extract_embeddings(interp, wav_path: str) -> np.ndarray:
    """Extract 1024D embeddings from a WAV file.

    Args:
        interp: TFLite interpreter with YAMNet loaded.
        wav_path: Path to 16kHz mono WAV file.

    Returns:
        Array of shape (n_frames, 1024), float32.
    """
    import soundfile as sf

    waveform, sr = sf.read(wav_path, dtype="float32")

    # Convert to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed (MIMII is already 16kHz, but be safe)
    if sr != YAMNET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=YAMNET_SR)

    # Slice into non-overlapping frames of YAMNET_FRAME_LEN
    n_samples = len(waveform)
    embeddings = []

    for start in range(0, n_samples - YAMNET_FRAME_LEN + 1, YAMNET_FRAME_LEN):
        frame = waveform[start:start + YAMNET_FRAME_LEN].astype(np.float32)

        # Feed into YAMNet
        input_details = interp.get_input_details()
        interp.resize_tensor_input(input_details[0]["index"], frame.shape)
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]["index"], frame)
        interp.invoke()

        # Grab the 1024D embedding from the reduce_mean layer and dequantize
        raw = interp.get_tensor(EMBEDDING_TENSOR_INDEX)  # (1, 1, 1, 1024) int8
        emb = (raw.reshape(1024).astype(np.float32) - interp._emb_zp) * interp._emb_scale
        embeddings.append(emb)

    if not embeddings:
        return np.zeros((0, 1024), dtype=np.float32)

    return np.stack(embeddings)


def discover_clips(data_root: str, machine_types: list) -> list:
    """Walk MIMII directory tree and collect clip paths with metadata."""
    clips = []
    root = Path(data_root)

    for mtype in machine_types:
        mtype_dir = root / mtype
        if not mtype_dir.exists():
            continue
        for mid_dir in sorted(mtype_dir.iterdir()):
            if not mid_dir.is_dir():
                continue
            for label in ("normal", "abnormal"):
                label_dir = mid_dir / label
                if not label_dir.exists():
                    continue
                for wav_path in sorted(label_dir.glob("*.wav")):
                    clips.append({
                        "path": str(wav_path),
                        "machine_type": mtype,
                        "machine_id": mid_dir.name,
                        "label": label,
                        "filename": wav_path.stem,
                    })
    return clips


def main():
    print("Loading YAMNet TFLite model...")
    interp = load_interpreter()

    # Verify embedding extraction works on a dummy input
    input_details = interp.get_input_details()
    dummy = np.zeros(YAMNET_FRAME_LEN, dtype=np.float32)
    interp.resize_tensor_input(input_details[0]["index"], dummy.shape)
    interp.allocate_tensors()
    interp.set_tensor(input_details[0]["index"], dummy)
    interp.invoke()
    raw = interp.get_tensor(EMBEDDING_TENSOR_INDEX)
    emb = (raw.reshape(1024).astype(np.float32) - interp._emb_zp) * interp._emb_scale
    print(f"Verified: embedding shape = {emb.shape}, range = [{emb.min():.3f}, {emb.max():.3f}]")

    # Discover all clips
    clips = discover_clips(DATA_ROOT, MACHINE_TYPES)
    print(f"Found {len(clips)} clips across {MACHINE_TYPES}")

    # Count by type
    from collections import Counter
    type_counts = Counter(c["machine_type"] for c in clips)
    for mt, count in sorted(type_counts.items()):
        print(f"  {mt}: {count} clips")

    # Extract and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_frames = 0
    errors = 0

    for clip in tqdm(clips, desc="Extracting embeddings"):
        out_dir = os.path.join(
            OUTPUT_DIR, clip["machine_type"], clip["machine_id"], clip["label"]
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{clip['filename']}.npy")

        # Skip if already extracted
        if os.path.exists(out_path):
            emb = np.load(out_path)
            total_frames += emb.shape[0]
            continue

        try:
            emb = extract_embeddings(interp, clip["path"])
            np.save(out_path, emb)
            total_frames += emb.shape[0]
        except Exception as e:
            print(f"\nError processing {clip['path']}: {e}")
            errors += 1

    # Save metadata
    metadata = {
        "model": "YAMNet (MediaPipe TFLite, float32)",
        "embedding_dim": 1024,
        "embedding_tensor_index": EMBEDDING_TENSOR_INDEX,
        "frame_length_samples": YAMNET_FRAME_LEN,
        "sample_rate": YAMNET_SR,
        "total_clips": len(clips),
        "total_frames": int(total_frames),
        "errors": errors,
        "machine_types": MACHINE_TYPES,
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"\nDone! {len(clips)} clips → {total_frames} frames of 1024D embeddings")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}/")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
