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

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STUDENT_DIR, MIMII_SPLITS, SEPARATOR_DIR, MIMII_ROOT,
    MACHINE_TYPES, MACHINE_IDS,
    THRESHOLD_PCT, FS_EPOCHS,
)
from distillation.cnn import AcousticEncoder
from preprocessing.separator_input import load_clip_log_mels
from separator.separator import train_fs, score_clips



def _validate_inputs(checkpoint_path):
    """Check that the frozen encoder checkpoint and split manifest exist."""
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Run:  python -m distillation.train")
        sys.exit(1)

    if not MIMII_SPLITS.exists():
        print(f"ERROR: Splits manifest not found at {MIMII_SPLITS}")
        print("Run:  python preprocessing/split_mimii.py")
        sys.exit(1)

    return ckpt_path


def _select_device():
    """Select the best available torch device and print it."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    return device


def _load_splits():
    """Load the MIMII train/test split manifest."""
    with open(MIMII_SPLITS) as f:
        return json.load(f)


def _load_frozen_encoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AcousticEncoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model, ckpt



def _embed_clip(wav_path, model, device):
    mels = load_clip_log_mels(wav_path)
    if mels is None:
        return None

    with torch.no_grad():
        embs = model(torch.tensor(mels, dtype=torch.float32).to(device)).cpu().numpy()

    return embs


def _train_machine_separator(train_paths, model, device):
    """Train one machine-specific SVDD separator from normal training clips."""
    train_embs = []

    for path in tqdm(train_paths, desc="    embed", leave=False, unit="clip"):
        embs = _embed_clip(path, model, device)
        if embs is not None:
            train_embs.append(embs)

    if not train_embs:
        return None

    # Flatten all per-clip frame embeddings into a single matrix for SVDD training
    stacked = np.vstack(train_embs) # (N_frames, 32)
    # Train SVDD — returns the trained model and the fixed centroid
    model_fs, centroid = train_fs(stacked, epochs=FS_EPOCHS)

    # Threshold: 95th percentile of training clip scores. Clips scoring
    # above this at inference time are flagged as anomalous.
    train_scores = score_clips(train_embs, model_fs, centroid)
    threshold = float(np.percentile(train_scores, THRESHOLD_PCT))
    n_fs_params = sum(p.numel() for p in model_fs.parameters())

    # return artefact
    return {
        "state_dict": model_fs.state_dict(),
        "centroid": centroid,
        "threshold": threshold,
        "input_dim": stacked.shape[1],
        "hidden_dim": 32,
        "output_dim": 8,
        "n_params": n_fs_params,
        "n_train_clips": len(train_embs),
    }


def _train_all_machine_separators(splits, model, device):
    """Train and save one separator artefact per machine."""
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

            artifact = _train_machine_separator(train_paths, model, device)

            if artifact is None:
                print(f"    No valid embeddings — skipping.")
                continue

            out_path = SEPARATOR_DIR / f"{mtype}_{mid}.pt"
            torch.save(artifact, out_path)

            summary_rows.append(
                (key, artifact["threshold"], artifact["n_params"], artifact["n_train_clips"])
            )
            print(
                f"    threshold={artifact['threshold']:.4f}  "
                f"params={artifact['n_params']}  → {out_path.name}"
            )

    return summary_rows


def _print_summary(summary_rows):
    """Print the separator training summary."""
    print(f"\n{'─' * 55}")
    print(f"  {'Machine':<20}  {'Threshold':>10}  {'Params':>8}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*8}")

    for key, thr, n_p, _ in summary_rows:
        print(f"  {key:<20}  {thr:>10.4f}  {n_p:>8}")

    print(f"{'─' * 55}")
    print(f"\nSaved {len(summary_rows)} artefacts → {SEPARATOR_DIR}/")
    print("Next:  python inference/run.py")



def train_separator(checkpoint_path):
    ckpt_path = _validate_inputs(checkpoint_path)
    device = _select_device()
    model, ckpt = _load_frozen_encoder(ckpt_path, device)

    print(f"Loaded AcousticEncoder: epoch {ckpt['epoch']}, val MSE {ckpt['val_loss']:.5f}")

    splits = _load_splits()
    summary_rows = _train_all_machine_separators(splits, model, device)
    _print_summary(summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(STUDENT_DIR / "acoustic_encoder.pt"))
    args = parser.parse_args()
    train_separator(args.checkpoint)

