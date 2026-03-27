"""
MIMII dataset loading pipeline.

Handles all machine types (fan, pump, valve, slider), computes log-mel
spectrograms from raw WAV files, and windows them into ~1-second segments.

Two dataset classes:
  - ContrastiveDataset: returns windows without labels (for f_c training)
  - EvaluationDataset: returns windows with full metadata (for evaluation)
"""

import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_clips(data_root: str, machine_types: list[str]) -> list[dict]:
    """Walk the MIMII directory structure and find all audio clips.

    Returns a list of dicts with keys:
        path, machine_type, machine_id, label ("normal" or "abnormal")
    """
    clips = []
    data_root = Path(data_root)

    for mtype in machine_types:
        mtype_dir = data_root / mtype
        if not mtype_dir.exists():
            continue

        for machine_id_dir in sorted(mtype_dir.iterdir()):
            if not machine_id_dir.is_dir():
                continue
            machine_id = machine_id_dir.name

            for label in ["normal", "abnormal"]:
                label_dir = machine_id_dir / label
                if not label_dir.exists():
                    continue

                for wav_path in sorted(label_dir.glob("*.wav")):
                    clips.append({
                        "path": str(wav_path),
                        "machine_type": mtype,
                        "machine_id": machine_id,
                        "label": label,
                    })

    return clips


def compute_log_mel(wav_path: str, config: dict) -> np.ndarray:
    """Load WAV file and compute log-mel spectrogram.

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames).
    """
    audio_cfg = config["audio"]
    y, _ = librosa.load(wav_path, sr=audio_cfg["sample_rate"], mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        power=audio_cfg["power"],
    )
    log_mel = np.log(mel + audio_cfg["log_offset"])
    return log_mel.astype(np.float32)


def window_spectrogram(spec: np.ndarray, window_frames: int) -> list[np.ndarray]:
    """Slice a spectrogram into non-overlapping windows.

    Args:
        spec: shape (n_mels, n_frames)
        window_frames: number of frames per window

    Returns:
        List of arrays, each shape (n_mels, window_frames)
    """
    n_frames = spec.shape[1]
    windows = []
    for start in range(0, n_frames - window_frames + 1, window_frames):
        windows.append(spec[:, start:start + window_frames])
    return windows


def precompute_dataset(clips: list[dict], config: dict, desc: str = "Processing") -> list[dict]:
    """Precompute mel-spectrograms and windows for all clips.

    Returns list of dicts with keys:
        window (np.ndarray), machine_type, machine_id, label, clip_index
    """
    from tqdm import tqdm
    window_frames = config["audio"]["window_frames"]
    samples = []

    for clip_idx, clip in enumerate(tqdm(clips, desc=desc)):
        try:
            spec = compute_log_mel(clip["path"], config)
        except Exception as e:
            print(f"Warning: failed to load {clip['path']}: {e}")
            continue

        windows = window_spectrogram(spec, window_frames)
        for w in windows:
            samples.append({
                "window": w,
                "machine_type": clip["machine_type"],
                "machine_id": clip["machine_id"],
                "label": clip["label"],
                "clip_index": clip_idx,
            })

    return samples


class ContrastiveDataset(Dataset):
    """Dataset for contrastive f_c training.

    Returns mel-spectrogram windows as tensors of shape (1, n_mels, window_frames).
    No labels are used — the contrastive model creates augmented pairs.
    """

    def __init__(self, samples: list[dict], mean: float = 0.0, std: float = 1.0):
        self.windows = [s["window"] for s in samples]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx].copy()
        w = (w - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(w).unsqueeze(0)  # (1, n_mels, window_frames)


class EvaluationDataset(Dataset):
    """Dataset for evaluating f_c embedding quality.

    Returns (window_tensor, metadata_dict) where metadata includes
    machine_type, machine_id, label, and clip_index for aggregation.
    """

    def __init__(self, samples: list[dict], mean: float = 0.0, std: float = 1.0):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        w = s["window"].copy()
        w = (w - self.mean) / (self.std + 1e-8)
        tensor = torch.from_numpy(w).unsqueeze(0)  # (1, n_mels, window_frames)

        metadata = {
            "machine_type": s["machine_type"],
            "machine_id": s["machine_id"],
            "label": s["label"],
            "clip_index": s["clip_index"],
        }
        return tensor, metadata


def create_datasets(config: dict, verbose: bool = True):
    """Build train and evaluation datasets from MIMII data.

    Training set: random 80% of normal windows (across all machines).
    Evaluation set: remaining 20% normal + all abnormal windows.

    Returns:
        train_dataset: ContrastiveDataset
        eval_dataset: EvaluationDataset
        stats: dict with normalisation mean/std
    """
    data_cfg = config["data"]
    clips = discover_clips(data_cfg["root"], data_cfg["machine_types"])

    if verbose:
        n_normal = sum(1 for c in clips if c["label"] == "normal")
        n_abnormal = sum(1 for c in clips if c["label"] == "abnormal")
        machine_types = set(c["machine_type"] for c in clips)
        print(f"Found {len(clips)} clips ({n_normal} normal, {n_abnormal} abnormal)")
        print(f"Machine types: {sorted(machine_types)}")

    # Split normal clips into train/eval
    normal_clips = [c for c in clips if c["label"] == "normal"]
    abnormal_clips = [c for c in clips if c["label"] == "abnormal"]

    # Shuffle normal clips deterministically, then split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(normal_clips))
    split_idx = int(len(normal_clips) * data_cfg["train_split"])

    train_clips = [normal_clips[i] for i in indices[:split_idx]]
    eval_normal_clips = [normal_clips[i] for i in indices[split_idx:]]
    eval_clips = eval_normal_clips + abnormal_clips

    if verbose:
        print(f"Train clips: {len(train_clips)} (normal only)")
        print(f"Eval clips: {len(eval_clips)} ({len(eval_normal_clips)} normal + {len(abnormal_clips)} abnormal)")
        print("Precomputing mel-spectrograms...")

    # Precompute windows
    train_samples = precompute_dataset(train_clips, config, desc="Train clips")
    eval_samples = precompute_dataset(eval_clips, config, desc="Eval clips")

    if verbose:
        print(f"Train windows: {len(train_samples)}")
        print(f"Eval windows: {len(eval_samples)}")

    # Compute normalisation stats from training data only
    all_train_windows = np.stack([s["window"] for s in train_samples])
    mean = float(all_train_windows.mean())
    std = float(all_train_windows.std())

    if verbose:
        print(f"Normalisation: mean={mean:.4f}, std={std:.4f}")

    train_dataset = ContrastiveDataset(train_samples, mean=mean, std=std)
    eval_dataset = EvaluationDataset(eval_samples, mean=mean, std=std)
    stats = {"mean": mean, "std": std}

    return train_dataset, eval_dataset, stats


def create_dataloaders(config: dict, verbose: bool = True):
    """Create DataLoaders for training and evaluation.

    Returns:
        train_loader, eval_loader, stats
    """
    train_dataset, eval_dataset, stats = create_datasets(config, verbose=verbose)

    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, eval_loader, stats
