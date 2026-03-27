"""
MIMII dataset loading pipeline for f_c SupCon training.

Three dataset roles:
    LabelledWindowDataset  — mel-spec windows with machine_id integer labels.
                             Used for both training and validation (SupCon loss).
    EvaluationDataset      — mel-spec windows with full metadata (machine_type,
                             machine_id, label, clip_index). Used for AUC evaluation
                             on held-out machine types.

Data splits (configured via train/eval_machine_types in fc.yaml):
    train_dataset  — 80% of normal clips from train_machine_types
    val_dataset    — 20% of normal clips from train_machine_types
    eval_dataset   — all clips (normal + abnormal) from eval_machine_types

Normalisation statistics are computed from train_dataset only.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Clip discovery and audio processing
# ---------------------------------------------------------------------------

def discover_clips(data_root: str, machine_types: list) -> list:
    """Walk the MIMII directory tree and collect metadata for every audio clip.

    Expected structure:
        {data_root}/{machine_type}/{machine_id}/{normal|abnormal}/*.wav

    Args:
        data_root:     Root directory of the MIMII dataset.
        machine_types: List of machine type names to include.

    Returns:
        List of dicts with keys: path, machine_type, machine_id, label.
    """
    clips = []
    root = Path(data_root)

    for mtype in machine_types:
        mtype_dir = root / mtype
        if not mtype_dir.exists():
            continue

        for machine_id_dir in sorted(mtype_dir.iterdir()):
            if not machine_id_dir.is_dir():
                continue
            machine_id = machine_id_dir.name

            for label in ("normal", "abnormal"):
                label_dir = machine_id_dir / label
                if not label_dir.exists():
                    continue
                for wav_path in sorted(label_dir.glob("*.wav")):
                    clips.append({
                        "path":         str(wav_path),
                        "machine_type": mtype,
                        "machine_id":   machine_id,
                        "label":        label,
                    })

    return clips


def compute_log_mel(wav_path: str, config: dict) -> np.ndarray:
    """Load a WAV file and compute a log-mel spectrogram.

    Args:
        wav_path: Path to the WAV file.
        config:   Full config dict (uses config["audio"] sub-dict).

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames), float32.
    """
    cfg = config["audio"]
    y, _ = librosa.load(wav_path, sr=cfg["sample_rate"], mono=True)
    mel  = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=cfg["power"],
    )
    return np.log(mel + cfg["log_offset"]).astype(np.float32)


def window_spectrogram(spec: np.ndarray, window_frames: int) -> list:
    """Slice a spectrogram into non-overlapping fixed-length windows.

    Args:
        spec:          Shape (n_mels, n_frames).
        window_frames: Number of time frames per window.

    Returns:
        List of arrays each of shape (n_mels, window_frames).
        Incomplete trailing windows are discarded.
    """
    n_frames = spec.shape[1]
    windows  = []
    for start in range(0, n_frames - window_frames + 1, window_frames):
        windows.append(spec[:, start : start + window_frames])
    return windows


def precompute_windows(clips: list, config: dict, desc: str = "Processing") -> list:
    """Precompute log-mel spectrograms and slice into windows for a clip list.

    Args:
        clips:  List of clip dicts (path, machine_type, machine_id, label).
        config: Full config dict.
        desc:   tqdm progress bar label.

    Returns:
        List of window dicts with keys:
            window (np.ndarray), machine_type, machine_id, label, clip_index.
    """
    window_frames = config["audio"]["window_frames"]
    samples = []

    for clip_idx, clip in enumerate(tqdm(clips, desc=desc, leave=False)):
        try:
            spec = compute_log_mel(clip["path"], config)
        except Exception as exc:
            print(f"Warning: failed to load {clip['path']}: {exc}")
            continue

        for window in window_spectrogram(spec, window_frames):
            samples.append({
                "window":       window,
                "machine_type": clip["machine_type"],
                "machine_id":   clip["machine_id"],
                "label":        clip["label"],
                "clip_index":   clip_idx,
            })

    return samples


# ---------------------------------------------------------------------------
# Machine label encoding
# ---------------------------------------------------------------------------

def build_label_map(clips: list) -> dict:
    """Build a deterministic integer label map from machine_type/machine_id pairs.

    Args:
        clips: List of clip dicts (must include machine_type and machine_id).

    Returns:
        Dict mapping "machine_type/machine_id" → integer (0-indexed, sorted).
    """
    unique = sorted({f"{c['machine_type']}/{c['machine_id']}" for c in clips})
    return {name: idx for idx, name in enumerate(unique)}


def assign_window_labels(samples: list, label_map: dict) -> list:
    """Assign integer machine_id labels to precomputed window samples.

    Args:
        samples:   List of window dicts from precompute_windows().
        label_map: Mapping from "machine_type/machine_id" → int.

    Returns:
        List of integer labels, one per sample.
    """
    labels = []
    for s in samples:
        key = f"{s['machine_type']}/{s['machine_id']}"
        labels.append(label_map.get(key, -1))
    return labels


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class LabelledWindowDataset(Dataset):
    """Dataset for SupCon f_c training and validation.

    Returns (tensor, label) pairs where:
        tensor — (1, n_mels, window_frames) normalised log-mel window
        label  — integer machine_id label (index into label_map)

    Used for both train and val loaders (augmentation is applied in the
    trainer's forward pass, not here).
    """

    def __init__(self, samples: list, labels: list, mean: float = 0.0, std: float = 1.0):
        """
        Args:
            samples: List of window dicts from precompute_windows().
            labels:  Integer label for each sample (same length as samples).
            mean:    Training-set mean for normalisation.
            std:     Training-set std for normalisation.
        """
        assert len(samples) == len(labels), "samples and labels must have equal length"
        self.windows = [s["window"] for s in samples]
        self.labels  = labels
        self.mean    = mean
        self.std     = std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx].copy()
        w = (w - self.mean) / (self.std + 1e-8)
        tensor = torch.from_numpy(w).unsqueeze(0)   # (1, n_mels, window_frames)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


class EvaluationDataset(Dataset):
    """Dataset for evaluating f_c embedding quality on held-out machine types.

    Returns (tensor, metadata_dict) pairs where metadata includes machine_type,
    machine_id, label (normal/abnormal), and clip_index.
    Used in evaluate_fc.py to compute AUC, separation ratio, etc.
    """

    def __init__(self, samples: list, mean: float = 0.0, std: float = 1.0):
        """
        Args:
            samples: List of window dicts from precompute_windows().
            mean:    Training-set mean for normalisation.
            std:     Training-set std for normalisation.
        """
        self.samples = samples
        self.mean    = mean
        self.std     = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        w = s["window"].copy()
        w = (w - self.mean) / (self.std + 1e-8)
        tensor = torch.from_numpy(w).unsqueeze(0)   # (1, n_mels, window_frames)
        metadata = {
            "machine_type": s["machine_type"],
            "machine_id":   s["machine_id"],
            "label":        s["label"],
            "clip_index":   s["clip_index"],
        }
        return tensor, metadata


# ---------------------------------------------------------------------------
# Dataset and DataLoader construction
# ---------------------------------------------------------------------------

def create_datasets(config: dict, verbose: bool = True):
    """Build train, validation, and evaluation datasets from MIMII data.

    Splits:
        train_dataset — 80% of normal clips from train_machine_types
        val_dataset   — 20% of normal clips from train_machine_types
        eval_dataset  — all clips (normal + abnormal) from eval_machine_types

    Normalisation statistics are derived from train_dataset only.

    Args:
        config:  Full config dict loaded from fc.yaml.
        verbose: Print dataset statistics.

    Returns:
        Tuple of (train_dataset, val_dataset, eval_dataset, stats, label_map) where:
            train_dataset — LabelledWindowDataset
            val_dataset   — LabelledWindowDataset
            eval_dataset  — EvaluationDataset (or None if eval_machine_types is empty)
            stats         — dict {"mean": float, "std": float}
            label_map     — dict mapping "machine_type/machine_id" → int
    """
    data_cfg = config["data"]
    train_types = data_cfg["train_machine_types"]
    eval_types  = data_cfg.get("eval_machine_types", [])

    # Discover all clips for each split
    train_eval_clips = discover_clips(data_cfg["root"], train_types)
    eval_clips_all   = discover_clips(data_cfg["root"], eval_types) if eval_types else []

    if verbose:
        n_normal   = sum(1 for c in train_eval_clips if c["label"] == "normal")
        n_abnormal = sum(1 for c in train_eval_clips if c["label"] == "abnormal")
        print(f"Train/val machines ({train_types}):")
        print(f"  {len(train_eval_clips)} clips  ({n_normal} normal, {n_abnormal} abnormal)")
        if eval_clips_all:
            ne = sum(1 for c in eval_clips_all if c["label"] == "normal")
            na = sum(1 for c in eval_clips_all if c["label"] == "abnormal")
            print(f"Eval machines ({eval_types}):")
            print(f"  {len(eval_clips_all)} clips  ({ne} normal, {na} abnormal)")

    # Split normal clips from train_types into 80% train / 20% val
    normal_clips   = [c for c in train_eval_clips if c["label"] == "normal"]
    rng            = np.random.RandomState(42)
    indices        = rng.permutation(len(normal_clips))
    split_idx      = int(len(normal_clips) * data_cfg["train_split"])
    train_clips    = [normal_clips[i] for i in indices[:split_idx]]
    val_clips      = [normal_clips[i] for i in indices[split_idx:]]

    if verbose:
        print(f"Train clips: {len(train_clips)}  Val clips: {len(val_clips)}")
        print("Precomputing mel-spectrograms for train split...")

    # Precompute windows
    train_samples = precompute_windows(train_clips, config, desc="Train")
    val_samples   = precompute_windows(val_clips,   config, desc="Val  ")

    if verbose:
        print(f"Train windows: {len(train_samples)}  Val windows: {len(val_samples)}")

    # Build label map from training clips only
    # (val uses the same map so label integers are consistent)
    label_map        = build_label_map(train_clips)
    train_labels     = assign_window_labels(train_samples, label_map)
    val_labels       = assign_window_labels(val_samples,   label_map)

    if verbose:
        print(f"Machine classes: {len(label_map)}  {sorted(label_map.keys())}")

    # Normalisation from training windows only
    all_train = np.stack([s["window"] for s in train_samples])
    mean = float(all_train.mean())
    std  = float(all_train.std())
    if verbose:
        print(f"Normalisation stats: mean={mean:.4f}  std={std:.4f}")

    train_dataset = LabelledWindowDataset(train_samples, train_labels, mean, std)
    val_dataset   = LabelledWindowDataset(val_samples,   val_labels,   mean, std)

    # Evaluation dataset (held-out machine types)
    if eval_clips_all:
        if verbose:
            print("Precomputing mel-spectrograms for eval split...")
        eval_samples  = precompute_windows(eval_clips_all, config, desc="Eval ")
        eval_dataset  = EvaluationDataset(eval_samples, mean, std)
    else:
        eval_dataset = None

    stats = {"mean": mean, "std": std}
    return train_dataset, val_dataset, eval_dataset, stats, label_map


def create_dataloaders(config: dict, verbose: bool = True):
    """Create DataLoaders for SupCon training, validation, and evaluation.

    Args:
        config:  Full config dict loaded from fc.yaml.
        verbose: Print dataset statistics.

    Returns:
        Tuple of (train_loader, val_loader, eval_loader, stats, label_map).
        eval_loader may be None if eval_machine_types is empty.
    """
    train_dataset, val_dataset, eval_dataset, stats, label_map = create_datasets(
        config, verbose=verbose
    )
    train_cfg = config["training"]
    workers   = train_cfg["num_workers"]
    bs        = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,   # SupCon needs full batches for meaningful negatives
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

    return train_loader, val_loader, eval_loader, stats, label_map


def create_eval_loader(config: dict, stats: dict, verbose: bool = True):
    """Create a DataLoader for the eval split only — no train/val precomputation.

    Used by evaluate_fc.py so it doesn't waste time recomputing train/val
    spectrograms that are not needed for evaluation.

    Args:
        config:  Full config dict loaded from fc.yaml.
        stats:   Normalisation stats dict {"mean": float, "std": float}.
                 Pass the values saved to norm_stats.yaml during training.
        verbose: Print clip counts.

    Returns:
        eval_loader — DataLoader over EvaluationDataset, or None if
        eval_machine_types is empty.
    """
    data_cfg   = config["data"]
    eval_types = data_cfg.get("eval_machine_types", [])

    if not eval_types:
        return None

    eval_clips = discover_clips(data_cfg["root"], eval_types)

    if verbose:
        ne = sum(1 for c in eval_clips if c["label"] == "normal")
        na = sum(1 for c in eval_clips if c["label"] == "abnormal")
        print(f"Eval machines ({eval_types}):")
        print(f"  {len(eval_clips)} clips  ({ne} normal, {na} abnormal)")
        print("Precomputing mel-spectrograms for eval split...")

    eval_samples = precompute_windows(eval_clips, config, desc="Eval ")

    if verbose:
        print(f"  {len(eval_samples)} windows")

    eval_dataset = EvaluationDataset(eval_samples, stats["mean"], stats["std"])

    train_cfg = config["training"]
    return DataLoader(
        eval_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
