"""
features.py — TWFR (Time-Weighted Frequency Domain Representation) feature extraction.

Implements the GWRP-based audio representation from:
  Guan et al., "Time-Weighted Frequency Domain Audio Representation with GMM
  Estimator for Anomalous Sound Detection", arXiv:2305.03328 (2023).

Key difference from the CNN pipeline in this repo: TWFR operates on the full
10 s clip mel spectrogram in a single pass (≈625 time frames at hop=256),
not on 0.975 s sub-windows. The time axis is therefore the entire recording,
allowing the weighting to trade off stationary vs. transient content.

All audio constants (SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, LOG_OFFSET) are
imported from config.py — the single source of truth shared across the repo.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GMM_HOP_LENGTH, GMM_N_MELS, LOG_OFFSET, N_FFT, SAMPLE_RATE

# Alias with shorter names for readability within this module.
# GMM_N_MELS=128 and GMM_HOP_LENGTH=512 match the paper (Guan et al. 2023),
# and differ deliberately from the SVDD pipeline (N_MELS=64, HOP_LENGTH=256).
_N_MELS     = GMM_N_MELS       # 128 Mel-filter banks
_HOP_LENGTH = GMM_HOP_LENGTH   # 50 % overlap of N_FFT=1024


# ── Public API ────────────────────────────────────────────────────────────────

def load_log_mel(wav_path: str) -> np.ndarray:
    """Load a WAV file and compute its log-mel spectrogram over the full clip.

    Parameters
    ----------
    wav_path : str
        Absolute path to a mono or stereo WAV file. Stereo is down-mixed.

    Returns
    -------
    log_mel : np.ndarray, shape (GMM_N_MELS, T), dtype float32
        Log-mel spectrogram where T ≈ n_samples / GMM_HOP_LENGTH. For a 10 s
        clip at 16 kHz with GMM_HOP_LENGTH=512, T ≈ 312.

    Raises
    ------
    RuntimeError
        If the file cannot be loaded or produces zero audio samples.
    """
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load '{wav_path}': {exc}") from exc

    if audio.size == 0:
        raise RuntimeError(f"Zero samples after loading '{wav_path}'.")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=_HOP_LENGTH,
        n_mels=_N_MELS,
        power=2.0,
    )
    return np.log(mel + LOG_OFFSET).astype(np.float32)


def gwrp_weights(N: int, r: float) -> np.ndarray:
    """Compute the Global Weighted Ranking Pooling (GWRP) weight vector.

    Weights are defined as::

        P(r)[i] = r^i / z(r),   z(r) = sum(r^j for j in 0..N-1)

    where index 0 receives the highest weight and corresponds to the
    largest value after descending sort (energy-weighted attention).

    Special cases (handled without numerical issues):
    * r = 1.0  →  uniform weights  1/N  (mean pooling)
    * r = 0.0  →  [1, 0, 0, ...]      (max pooling)

    Parameters
    ----------
    N : int
        Number of time frames (length of the weight vector).
    r : float
        Decay parameter in [0, 1].

    Returns
    -------
    weights : np.ndarray, shape (N,), dtype float64
        Non-negative weights summing to 1.0.
    """
    if r == 1.0:
        return np.ones(N) / N
    if r == 0.0:
        w = np.zeros(N)
        w[0] = 1.0
        return w
    w = r ** np.arange(N)
    return w / w.sum()


def twfr_feature(log_mel: np.ndarray, r: float) -> np.ndarray:
    """Compute the Time-Weighted Frequency Domain Representation (TWFR).

    For each mel-frequency bin (row of ``log_mel``), the T time-frame values
    are sorted in descending order and then combined via a weighted dot product
    with the GWRP weight vector P(r). This produces one scalar per frequency
    bin, yielding an M-dimensional feature vector for the clip.

    The operation generalises mean (r=1) and max (r=0) pooling:
    * r → 1 emphasises the stationary component (average energy).
    * r → 0 emphasises the peak transient (maximum energy frame).
    Intermediate r values blend both characteristics adaptively.

    Parameters
    ----------
    log_mel : np.ndarray, shape (M, T)
        Log-mel spectrogram with M frequency bins and T time frames.
    r : float
        GWRP decay parameter in [0, 1].

    Returns
    -------
    feature : np.ndarray, shape (M,), dtype float32
        TWFR feature vector. Dimensionality equals N_MELS (64 by default).

    Notes
    -----
    Invariants (verified in tests):
      ``twfr_feature(lm, r=1.0)`` equals ``lm.mean(axis=1)``  (mean pooling)
      ``twfr_feature(lm, r=0.0)`` equals ``lm.max(axis=1)``   (max pooling)
    """
    _, T = log_mel.shape
    # Sort each frequency bin's values in descending order of energy.
    sorted_mel = np.sort(log_mel, axis=1)[:, ::-1]   # shape (M, T)
    weights = gwrp_weights(T, r)                       # shape (T,)
    return (sorted_mel @ weights).astype(np.float32)  # shape (M,)
