"""Per-machine monitoring simulation.

Works with any detector exposing the ``GMMDetector`` interface
(``score``, ``cusum_update``, ``cusum_reset``, ``cusum_false_alarms``,
``threshold_``, ``cusum_k_``, ``cusum_h_``, ``r_``, ``n_components``).
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ROOT as _DEFAULT_MIMII_ROOT
from gmm.config import CLIP_SECS, N_MELS, ROLLING_WINDOW
from gmm.features import load_log_mel


def _score_paths(wav_paths: list[str], detector, desc: str = "") -> list[float]:
    """Score each WAV file once. WAVs that fail to load are skipped with a warning.

    The detector's optional ``channel_a_``/``channel_b_`` attributes trigger a
    two-channel ``score_pair`` path; ``channel_`` selects a single mic channel;
    otherwise audio is mono-mixed.
    """
    n_mels    = getattr(detector, "n_mels_",   N_MELS)
    channel_a = getattr(detector, "channel_a_", None)
    channel_b = getattr(detector, "channel_b_", None)
    channel   = getattr(detector, "channel_",   None)

    scores: list[float] = []
    for path in tqdm(wav_paths, desc=desc, leave=False, unit="clip"):
        try:
            if channel_a is not None:
                log_mel_a = load_log_mel(path, n_mels=n_mels, channel=channel_a)
                log_mel_b = load_log_mel(path, n_mels=n_mels, channel=channel_b)
                scores.append(detector.score_pair(log_mel_a, log_mel_b))
            else:
                scores.append(detector.score(load_log_mel(path, n_mels=n_mels, channel=channel)))
        except RuntimeError as exc:
            print(f"  Warning: {exc}", file=sys.stderr)
    return scores


def _rolling_mean(scores: list[float], i: int) -> float:
    """Trailing mean of length ``ROLLING_WINDOW`` ending at index ``i``."""
    return float(np.mean(scores[max(0, i - ROLLING_WINDOW + 1):i + 1]))


def evaluate_machine(
    mtype: str,
    mid: str,
    detector,
    splits: dict,
    mimii_root=None,
) -> dict | None:
    """Run the (normal, anomaly) monitoring rounds for one machine.

    For each round: score the normal window and count CUSUM alarms as false
    positives, then score the anomaly window and record the index of the first
    alarm as the detection event.

    Returns a dict of events, segments, threshold, round_results, n_rounds, r,
    n_components, or ``None`` if the machine is absent from ``splits``.
    """
    if mimii_root is None:
        mimii_root = _DEFAULT_MIMII_ROOT

    key = f"{mtype}/{mid}"
    if key not in splits:
        return None

    test_normal_paths   = [str(mimii_root / p) for p in splits[key]["test_normal"]]
    test_abnormal_paths = [str(mimii_root / p) for p in splits[key]["test_abnormal"]]
    n_rounds            = splits[key]["n_rounds"]
    monitor_clips       = len(test_normal_paths) // n_rounds

    # Time is seconds internally; events/segments store minutes for plotting.
    events:   list[dict] = []
    segments: list[dict] = []
    t = 0.0

    def _add_segment(phase: str, round_idx: int, wav_paths: list[str]) -> list[float]:
        nonlocal t
        seg_start = t
        scores = _score_paths(wav_paths, detector, desc=f"    {phase[:4]} r{round_idx}")
        for i, score in enumerate(scores):
            events.append({
                "t":            t / 60,
                "score":        score,
                "rolling_mean": _rolling_mean(scores, i),
                "phase":        phase,
                "round":        round_idx,
            })
            t += CLIP_SECS
        segments.append({
            "phase":   phase,
            "round":   round_idx,
            "t_start": seg_start / 60,
            "t_end":   t / 60,
        })
        return scores

    round_results: list[dict] = []

    for r in range(n_rounds):
        norm_paths = test_normal_paths[r * monitor_clips:(r + 1) * monitor_clips]
        anom_paths = test_abnormal_paths[r * monitor_clips:(r + 1) * monitor_clips]

        norm_scores = _add_segment("normal",  r + 1, norm_paths)
        n_false_pos = detector.cusum_false_alarms(norm_scores)

        detector.cusum_reset()
        anom_scores   = _add_segment("anomaly", r + 1, anom_paths)
        detection_idx = None
        detector.cusum_reset()
        for idx, s in enumerate(anom_scores):
            if detector.cusum_update(s):
                detection_idx = idx
                break

        round_results.append({
            "round":                r + 1,
            "n_false_pos":          n_false_pos,
            "n_normal_clips":       len(norm_scores),
            "n_anom_clips":         len(anom_scores),
            "detected":             detection_idx is not None,
            "detection_delay_secs": (int(detection_idx * CLIP_SECS)
                                     if detection_idx is not None else None),
            "detection_idx":        detection_idx,
            "cusum_h":              detector.cusum_h_,
            "cusum_k":              detector.cusum_k_,
        })

    return {
        "events":        events,
        "segments":      segments,
        "threshold":     detector.threshold_,
        "round_results": round_results,
        "n_rounds":      n_rounds,
        "r":             detector.r_,
        "n_components":  detector.n_components,
    }
