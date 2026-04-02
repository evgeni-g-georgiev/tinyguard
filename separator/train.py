#!/usr/bin/env python3
"""
train.py — Simulate the on-device 10-minute SVDD training window.

Realistic on-device flow being simulated:
  1. Collect phase (10 min): each audio frame is processed through the frozen
     AcousticEncoder in real time and the 32D embedding stored (~75KB total).
     Raw audio and mel spectrograms are discarded after each frame.
  2. Train phase: FsSeparator (Deep SVDD) is trained on all stored embeddings
     for a fixed number of epochs. At ~24–48 ms/epoch on Cortex-M33 @ 160 MHz
     (~2.4M MACs/epoch), 50 epochs takes ~1–2.5 seconds.
  3. Threshold is set at the 95th percentile of training scores, then the
     device switches to continuous monitoring.

For each of the 16 MIMII machines, loads the pre-determined training clips
from the splits manifest, extracts 32D embeddings via the frozen AcousticEncoder,
trains FsSeparator (Deep SVDD), and saves the trained artefact.

Each saved artefact contains everything inference/run.py needs to score clips:
  model state dict, centroid, threshold, and architecture parameters.

Inputs
------
  outputs/student/acoustic_encoder.pt      frozen AcousticEncoder checkpoint
  outputs/mimii_splits/splits.json         train/test clip manifest
  data/mimii/                              MIMII WAV files

Output
------
  outputs/separator/{machine_type}_{machine_id}.pt   one file per machine

Usage
-----
    python separator/train.py
    python separator/train.py --checkpoint outputs/student/acoustic_encoder.pt
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STUDENT_DIR, MIMII_SPLITS, SEPARATOR_DIR, MIMII_ROOT,
    MACHINE_TYPES, MACHINE_IDS,
    SAMPLE_RATE, FRAME_LEN, N_FFT, HOP_LENGTH, N_MELS, LOG_OFFSET,
    THRESHOLD_PCT, FS_EPOCHS,
)
from distillation.cnn import AcousticEncoder
from separator.separator import train_fs, score_clips


# ── audio helpers ─────────────────────────────────────────────────────────────

def log_mel(frame: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=frame, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    return np.log(mel + LOG_OFFSET)[np.newaxis, :, :]   # (1, 64, 61)


def embed_clip(wav_path: str, model: AcousticEncoder, device) -> np.ndarray | None:
    """Return (n_frames, embedding_dim) float32 embeddings for one clip, or None if too short."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    n_frames  = len(audio) // FRAME_LEN
    if n_frames == 0:
        return None
    # Stack all frames into a batch and run a single forward pass for efficiency
    mels = np.stack([
        log_mel(audio[i * FRAME_LEN:(i + 1) * FRAME_LEN].astype(np.float32))
        for i in range(n_frames)
    ])
    with torch.no_grad():
        embs = model(torch.tensor(mels, dtype=torch.float32).to(device)).cpu().numpy()
    return embs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(STUDENT_DIR / "acoustic_encoder.pt"))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Run:  python distillation/train.py")
        sys.exit(1)

    if not MIMII_SPLITS.exists():
        print(f"ERROR: Splits manifest not found at {MIMII_SPLITS}")
        print("Run:  python preprocessing/split_mimii.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AcousticEncoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded AcousticEncoder: epoch {ckpt['epoch']}, val MSE {ckpt['val_loss']:.5f}")

    with open(MIMII_SPLITS) as f:
        splits = json.load(f)

    SEPARATOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining SVDD for {len(splits)} machines …\n")
    summary_rows = []

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            if key not in splits:
                print(f"  [{key}] not in splits manifest — skipping.")
                continue

            train_paths = [str(MIMII_ROOT / p) for p in splits[key]["train_normal"]]
            print(f"  [{key}]  {len(train_paths)} training clips")

            # Extract embeddings
            train_embs = []
            for path in tqdm(train_paths, desc=f"    embed", leave=False, unit="clip"):
                embs = embed_clip(path, model, device)
                if embs is not None:
                    train_embs.append(embs)

            if not train_embs:
                print(f"    No valid embeddings — skipping.")
                continue

            # Flatten all per-clip frame embeddings into a single matrix for SVDD training
            stacked = np.vstack(train_embs)   # (N_frames, 32)

            # Train SVDD — returns the trained model and the fixed centroid
            model_fs, centroid = train_fs(stacked, epochs=FS_EPOCHS)

            # Threshold: 95th percentile of training clip scores. Clips scoring
            # above this at inference time are flagged as anomalous.
            train_scores = score_clips(train_embs, model_fs, centroid)
            threshold    = float(np.percentile(train_scores, THRESHOLD_PCT))

            n_fs_params = sum(p.numel() for p in model_fs.parameters())

            # Save artefact
            out_path = SEPARATOR_DIR / f"{mtype}_{mid}.pt"
            torch.save({
                "state_dict":  model_fs.state_dict(),
                "centroid":    centroid,
                "threshold":   threshold,
                "input_dim":   stacked.shape[1],
                "hidden_dim":  32,
                "output_dim":  8,
                "n_params":    n_fs_params,
                "n_train_clips": len(train_embs),
            }, out_path)

            summary_rows.append((key, threshold, n_fs_params, len(train_embs)))
            print(f"    threshold={threshold:.4f}  params={n_fs_params}  → {out_path.name}")

    print(f"\n{'─' * 55}")
    print(f"  {'Machine':<20}  {'Threshold':>10}  {'Params':>8}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*8}")
    for key, thr, n_p, _ in summary_rows:
        print(f"  {key:<20}  {thr:>10.4f}  {n_p:>8}")
    print(f"{'─' * 55}")
    print(f"\nSaved {len(summary_rows)} artefacts → {SEPARATOR_DIR}/")
    print("Next:  python inference/run.py")


if __name__ == "__main__":
    main()
