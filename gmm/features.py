"""
features.py — Feature extraction for the deployment-faithful GMM simulation.

Two extraction paths:

  extract_feature(log_mel)        r=1.0 only — direct equivalent of
                                  spectrogram_get_feature() in deployment/
                                  spectrogram.h.  Used by the single-node
                                  baseline and the hardware simulation.

  extract_feature_r(log_mel, r)   General GWRP for any r ∈ [0, 1].  Used
                                  by node learning experiments where each
                                  node operates at a different r value.
                                  At r=1.0 this is identical to the above.

Both return an (N_MELS,) float32 vector.

Hardware-induced node heterogeneity
------------------------------------
The choice of r creates the functional heterogeneity that motivates the
Node Learning approach (Kanjo & Aslanov, 2026, §3):

  r = 1.0  Mean pooling.  Computed as a running sum — no buffer needed.
           The *only* r value feasible on the Arduino Nano 33 BLE (256 KB SRAM)
           without storing the full (128 × 312) spectrogram (≈156 KB).

  r = 0.5  Energy-weighted GWRP.  Requires sorting each mel bin across all
           time frames, which demands the full spectrogram buffer.  Feasible
           only on a *second* co-located node.

Two nodes with r=1.0 and r=0.5 form a functionally complementary pair
(paper §2): they perceive different temporal structure of the same audio and
therefore carry different information about normality.  NodeLearning fuses
their scores to exploit this diversity.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GMM_N_MELS
from preprocessing.gmm_input import load_full_clip_log_mel


def load_log_mel(wav_path: str, n_mels: int = GMM_N_MELS, channel: int | None = None) -> np.ndarray:
    """Load a WAV file and return its full-clip log-mel spectrogram.

    Parameters
    ----------
    wav_path : str
    n_mels : int
        Number of mel frequency bins.  Defaults to GMM_N_MELS (128).
    channel : int or None
        Microphone channel index (0-7 for MIMII).  None mixes all channels
        to mono (legacy behaviour).

    Returns
    -------
    log_mel : np.ndarray, shape (n_mels, T)
    """
    return load_full_clip_log_mel(wav_path, n_mels=n_mels, channel=channel)


def gwrp_weights(T: int, r: float) -> np.ndarray:
    """Compute Global Weighted Ranking Pooling weights, shape (T,).

    Weights are defined as:
        P(r)[i] = r^i / Z(r),   Z(r) = sum(r^j for j in 0..T-1)

    Index 0 receives the highest weight and corresponds to the largest
    value after descending sort (energy-weighted attention).

    Special cases handled without numerical issues:
      r = 1.0  →  uniform weights  1/T  (mean pooling)
      r = 0.0  →  [1, 0, 0, ...]       (max pooling)

    Parameters
    ----------
    T : int
        Number of time frames.
    r : float
        Decay parameter in [0, 1].

    Returns
    -------
    weights : np.ndarray, shape (T,), dtype float64
        Non-negative weights summing to 1.0.
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
    """Compute the TWFR feature for any r value.

    For each mel-frequency bin, the T time-frame values are sorted in
    descending order and combined via the GWRP weight vector P(r).

    At r=1.0 this is mathematically identical to extract_feature() and to
    spectrogram_get_feature() in deployment/spectrogram.h: sort is
    order-invariant under uniform weights so no sort is needed.

    Parameters
    ----------
    log_mel : np.ndarray, shape (N_MELS, T)
    r : float
        GWRP decay parameter in [0, 1].

    Returns
    -------
    feature : np.ndarray, shape (N_MELS,), dtype float32
    """
    if r >= 1.0:
        return log_mel.mean(axis=1).astype(np.float32)
    if r <= 0.0:
        return log_mel.max(axis=1).astype(np.float32)
    _, T = log_mel.shape
    weights    = gwrp_weights(T, r)                          # (T,)
    sorted_mel = np.sort(log_mel, axis=1)[:, ::-1]          # (N_MELS, T) descending
    return (sorted_mel @ weights).astype(np.float32)         # (N_MELS,)


def extract_feature(log_mel: np.ndarray) -> np.ndarray:
    """Compute the TWFR feature at r=1.0: mean over the time axis.

    Equivalent to spectrogram_get_feature() in deployment/spectrogram.h.
    Kept as the named entry point for the deployment-faithful single-node
    baseline so call sites read clearly.

    Parameters
    ----------
    log_mel : np.ndarray, shape (N_MELS, T)

    Returns
    -------
    feature : np.ndarray, shape (N_MELS,), dtype float32
    """
    return extract_feature_r(log_mel, 1.0)
