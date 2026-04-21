"""
evaluate.py — Per-machine monitoring simulation for any detector variant.

Mirrors the COLLECT → TRAIN → MONITOR state machine in tinyml_gmm.ino.
Accepts any detector that implements the GMMDetector duck-type interface:
score(), cusum_update(), cusum_reset(), cusum_false_alarms(), threshold_,
cusum_k_, cusum_h_, r_, n_components.  This means GMMDetector (single-node
baseline) and NodeLearning (two-node collaborative system) are both supported
without any changes to this module.

Detection mechanism
-------------------
  Normal window:   cusum_false_alarms(scores) — counts independent CUSUM alarm
                   events with accumulator reset after each (mirrors the C++).
  Anomaly window:  cusum_update(score) per clip — first alarm = detection event.

CUSUM accumulator resets
------------------------
  * Explicit reset before each monitoring window — matches tinyml_gmm.ino.
  * Automatic reset inside cusum_update() on alarm — first alarm ends the
    anomaly search for that round.
  * Rolling mean (ROLLING_WINDOW=5 clips) stored in event dicts for
    diagnostics/plotting only.  Does NOT affect detection, matching the C++
    where rolling_mean() output goes to Serial only.
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ROOT as _DEFAULT_MIMII_ROOT
from gmm.config import CLIP_SECS, N_MELS, ROLLING_WINDOW
from gmm.features import load_log_mel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_paths(
    wav_paths: list[str],
    detector,
    desc: str = "",
) -> list[float]:
    """Score a list of WAV files, returning one score per clip.

    Supports three modes depending on attributes present on the detector:
      - NodeLearning with mic heterogeneity (channel_a_ and channel_b_ set):
        loads each clip twice (once per channel) and calls score_pair().
      - Single-node detector with channel_ set:
        loads the specified mic channel and calls score().
      - Legacy / mono-mix (no channel attributes):
        loads mono-mixed audio and calls score().

    Files that raise RuntimeError during loading are skipped with a warning.
    The mel resolution is read from detector.n_mels_ (falls back to N_MELS).
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
    """Mean of scores[max(0, i-ROLLING_WINDOW+1) : i+1].

    Display / diagnostics only — does not affect detection.
    Matches rolling_mean() in tinyml_gmm.ino.
    """
    return float(np.mean(scores[max(0, i - ROLLING_WINDOW + 1):i + 1]))


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_machine(
    mtype: str,
    mid: str,
    detector,
    splits: dict,
    mimii_root=None,
) -> dict | None:
    """Run the monitoring simulation for one machine.

    Mirrors the per-round MONITOR loop in tinyml_gmm.ino:

    For each round:
      1. Normal window  — score clips, count CUSUM alarms as false positives.
         The accumulator is reset before the window (fresh start each round).
      2. Anomaly window — score clips, record the first CUSUM alarm index as
         the detection event.  Accumulator reset before this window too.

    Parameters
    ----------
    mtype : str
        Machine type, e.g. 'fan'.
    mid : str
        Machine ID, e.g. 'id_00'.
    detector : GMMDetector or NodeLearning
        Fitted detector (fit() must have been called).  Any object implementing
        the duck-type interface: score(), cusum_update(), cusum_reset(),
        cusum_false_alarms(), threshold_, cusum_k_, cusum_h_, r_, n_components.
    splits : dict
        Splits manifest loaded from the appropriate splits_{dataset}.json.
    mimii_root : Path, optional
        Root directory of the MIMII dataset.  Defaults to MIMII_ROOT from
        the root config (backwards-compatible alias for the -6dB dataset).

    Returns
    -------
    result : dict or None
        Keys: events, segments, threshold, round_results, n_rounds,
              r, n_components.
        Returns None if the machine key is absent from splits.
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

    # Timeline state — t in seconds internally, stored as t/60 (minutes) in
    # events/segments to match the convention in inference/run.py.
    events:   list[dict] = []
    segments: list[dict] = []
    t = 0.0

    def _add_segment(phase: str, round_idx: int, wav_paths: list[str]) -> list[float]:
        """Score all clips in one window, advancing the timeline."""
        nonlocal t
        seg_start = t
        scores = _score_paths(wav_paths, detector,
                               desc=f"    {phase[:4]} r{round_idx}")
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

        # Normal window: count false positive alarm events.
        # cusum_false_alarms() resets the accumulator internally and resets
        # after each alarm, counting independent excursion events.
        norm_scores = _add_segment("normal",  r + 1, norm_paths)
        n_false_pos = detector.cusum_false_alarms(norm_scores)

        # Anomaly window: find first detection alarm.
        # Reset accumulator before the window for a fresh start.
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
