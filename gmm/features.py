"""TWFR feature extraction via Global Weighted Ranking Pooling."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GMM_N_MELS
from preprocessing.gmm_input import load_full_clip_log_mel


def load_log_mel(wav_path: str, n_mels: int = GMM_N_MELS, channel: int | None = None) -> np.ndarray:
    """Return the (n_mels, T) log-mel spectrogram for one WAV clip.

    ``channel`` picks a single mic channel (0-7 for MIMII); ``None`` mixes to mono.
    """
    return load_full_clip_log_mel(wav_path, n_mels=n_mels, channel=channel)


def gwrp_weights(T: int, r: float) -> np.ndarray:
    """GWRP weights ``P(r)[i] = r**i / sum_j r**j``, shape (T,), summing to 1.

    ``r=0`` → [1, 0, …] (max pooling); ``r=1`` → uniform 1/T (mean pooling).
    """
    if r >= 1.0:
        return np.ones(T) / T
    if r <= 0.0:
        w = np.zeros(T)
        w[0] = 1.0
        return w
    w = r ** np.arange(T)
    return w / w.sum()


def extract_feature_r(log_mel: np.ndarray, r: float) -> np.ndarray:
    """TWFR feature vector (n_mels,) for the given ``r`` in [0, 1].

    For each mel bin the T frame values are sorted descending and combined with
    ``gwrp_weights(T, r)``. The r=0 and r=1 endpoints take O(T) fast paths.
    """
    if r >= 1.0:
        return log_mel.mean(axis=1).astype(np.float32)
    if r <= 0.0:
        return log_mel.max(axis=1).astype(np.float32)
    _, T = log_mel.shape
    weights    = gwrp_weights(T, r)
    sorted_mel = np.sort(log_mel, axis=1)[:, ::-1]
    return (sorted_mel @ weights).astype(np.float32)


def extract_feature(log_mel: np.ndarray) -> np.ndarray:
    """Mean-pooled TWFR feature (r = 1.0)."""
    return extract_feature_r(log_mel, 1.0)
