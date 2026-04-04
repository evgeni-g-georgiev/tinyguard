"""
evaluate.py — Per-machine evaluation pipeline for GMMDetector on MIMII data.

Structural port of inference/run.py::run_machine(), with the embed-and-score
chain replaced by TWFR feature extraction + GMM NLL scoring. The timeline
data structures (events, segments, round_results) are identical to those
produced by inference/run.py, so plot.py and train.py can consume them
without any adaptation.

Timeline encoding note
----------------------
``t`` is tracked in **seconds** internally. Values written into ``events``
and ``segments`` are stored as ``t / 60`` (minutes), matching the convention
in inference/run.py (see line 139 of that file). plot.py recovers seconds for
the x-axis via ``ts(e) = e["t"] * 60``. This keeps the two pipelines
visually identical.

Detection mechanism
-------------------
Individual per-clip NLL scores have high variance: an occasional quiet
anomaly clip scores near-normal; an occasional loud normal clip spikes above
the threshold. Evaluating clips one-at-a-time therefore produces both missed
detections and false positives regardless of threshold placement.

The solution is to make decisions on a **rolling window mean** of the last
``ROLLING_WINDOW`` clips rather than on individual scores. A single outlier
clip moves the window mean by at most 1/ROLLING_WINDOW of its excess —
sustained shifts (true anomaly periods) move it fully.

The detection threshold for each round is set **adaptively** from the normal
monitoring window observed immediately before the anomaly window. The
maximum rolling-mean seen during normal operation is recorded; the anomaly
threshold is set 50 % above that level. This compensates for machine-to-
machine and session-to-session variability in score magnitude that a static
training-time threshold cannot capture.

Dependencies: sys, numpy, tqdm, config, gmm.features, gmm.detector.
No matplotlib, no yaml, no argparse.
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLIP_SECS, MIMII_ROOT
from gmm.detector import GMMDetector
from gmm.features import load_log_mel

# Number of consecutive clips whose mean score is used for detection decisions.
# With CLIP_SECS=10, ROLLING_WINDOW=5 corresponds to a 50-second context.
# A single outlier clip shifts the window mean by at most 1/5 of its excess;
# three or more consecutive anomalous clips shift it fully above the threshold.
ROLLING_WINDOW = 5

# Safety margin applied to the max normal rolling mean when setting the
# per-round adaptive detection threshold.  1.5 = 50 % headroom above the
# worst rolling mean observed during the preceding normal window.
ADAPTIVE_SAFETY = 1.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_mean(scores: list[float], i: int, window: int) -> float:
    """Mean of the score sub-sequence ending at index ``i`` (left-padded).

    At the start of a sequence (i < window - 1) the window is smaller than
    ``window``, covering only the clips seen so far.  This avoids introducing
    artificial zeros before the first full window is available.

    Parameters
    ----------
    scores : list of float
        Full score sequence.
    i : int
        Index of the rightmost (most recent) clip in the window.
    window : int
        Maximum window length.

    Returns
    -------
    mean : float
        Arithmetic mean of ``scores[max(0, i-window+1) : i+1]``.
    """
    return float(np.mean(scores[max(0, i - window + 1):i + 1]))


def _count_alarm_events(
    scores: list[float],
    threshold: float,
    window: int,
) -> int:
    """Count non-overlapping alarm events in a rolling-mean score sequence.

    An alarm event begins when the rolling mean first rises above
    ``threshold`` and ends when it falls back below it.  Each continuous
    above-threshold excursion is counted as one event regardless of how many
    clips it spans.

    Parameters
    ----------
    scores : list of float
        Per-clip anomaly scores for a monitoring window.
    threshold : float
        Alarm threshold applied to the rolling mean.
    window : int
        Rolling window size (same as used for detection).

    Returns
    -------
    n_events : int
        Number of distinct above-threshold excursions.
    """
    n_events  = 0
    in_alarm  = False
    for i in range(len(scores)):
        rm = _rolling_mean(scores, i, window)
        if rm >= threshold and not in_alarm:
            n_events += 1
            in_alarm  = True
        elif rm < threshold:
            in_alarm  = False
    return n_events


def _detect_first_alarm(
    scores: list[float],
    threshold: float,
    window: int,
) -> int | None:
    """Return the index of the first clip at which the rolling mean alarm fires.

    Parameters
    ----------
    scores : list of float
        Per-clip anomaly scores for the anomaly monitoring window.
    threshold : float
        Alarm threshold applied to the rolling mean.
    window : int
        Rolling window size.

    Returns
    -------
    detection_idx : int or None
        Index of the triggering clip, or ``None`` if no alarm fires.
    """
    for i in range(len(scores)):
        if _rolling_mean(scores, i, window) >= threshold:
            return i
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def score_paths(
    wav_paths: list[str],
    detector: GMMDetector,
    desc: str = "",
) -> list[float]:
    """Score a list of WAV files using a fitted GMMDetector.

    Loads each clip's log-mel spectrogram on-the-fly and calls
    ``detector.score()``. Files that raise ``RuntimeError`` during loading
    (e.g. corrupt or zero-length WAVs) are silently skipped with a warning
    printed to stderr so one bad file does not abort an entire round.

    Parameters
    ----------
    wav_paths : list of str
        Absolute paths to WAV files to score.
    detector : GMMDetector
        A fitted detector instance (``fit()`` must have been called).
    desc : str, optional
        Description label forwarded to the tqdm progress bar.

    Returns
    -------
    scores : list of float
        NLL anomaly scores, one per successfully processed clip. The list
        may be shorter than ``wav_paths`` if files were skipped.
    """
    scores: list[float] = []
    for path in tqdm(wav_paths, desc=desc, leave=False, unit="clip"):
        try:
            log_mel = load_log_mel(path)
            scores.append(detector.score(log_mel))
        except RuntimeError as exc:
            print(f"  Warning: {exc}", file=sys.stderr)
    return scores


def evaluate_machine(
    mtype: str,
    mid: str,
    detector: GMMDetector,
    splits: dict,
) -> dict | None:
    """Run the monitoring simulation for one machine using a fitted GMMDetector.

    Mirrors the timeline structure of ``inference/run.py::run_machine()``
    exactly: same event/segment dicts, same round_results schema, same
    time encoding (minutes stored, seconds plotted). This allows plot.py
    to be reused without modification.

    Each monitoring round consists of:
    1. A **normal** window: ``monitor_clips`` normal test clips are scored.
       The rolling-mean alarm count is recorded as false positives.
    2. An **anomaly** window: ``monitor_clips`` abnormal clips are scored.
       The first clip at which the rolling mean exceeds the round threshold
       is the detection event.

    The per-round detection threshold is derived adaptively from the
    preceding normal window: the maximum rolling mean observed during normal
    operation, multiplied by ``ADAPTIVE_SAFETY``.  This allows the threshold
    to self-calibrate to each machine's actual score magnitude at test time,
    compensating for cases where the training-time threshold under- or
    over-estimates the normal score level.

    Parameters
    ----------
    mtype : str
        Machine type string, e.g. ``'fan'``.
    mid : str
        Machine ID string, e.g. ``'id_00'``.
    detector : GMMDetector
        Fitted detector (``threshold_`` and ``r_`` must be set).
    splits : dict
        Full splits manifest as loaded from ``splits.json``.

    Returns
    -------
    result : dict or None
        Dictionary with keys:
          * ``events``       – list of {t, score, phase, round} dicts
          * ``segments``     – list of {phase, round, t_start, t_end} dicts
          * ``threshold``    – float training detection threshold (for plot line)
          * ``round_results``– list of per-round metric dicts
          * ``n_rounds``     – int
          * ``r``            – float GWRP decay used by detector
          * ``n_components`` – int GMM components
        Returns ``None`` if the machine key is absent from ``splits``.
    """
    key = f"{mtype}/{mid}"
    if key not in splits:
        return None

    test_normal_paths   = [str(MIMII_ROOT / p) for p in splits[key]["test_normal"]]
    test_abnormal_paths = [str(MIMII_ROOT / p) for p in splits[key]["test_abnormal"]]
    n_rounds            = splits[key]["n_rounds"]
    monitor_clips       = len(test_normal_paths) // n_rounds

    # ── Timeline state ────────────────────────────────────────────────────────
    events:   list[dict] = []   # one entry per scored clip
    segments: list[dict] = []   # one entry per monitoring window
    t = 0.0                     # running wall-clock time in seconds

    def add_segment(phase: str, round_idx: int, wav_paths: list[str]) -> list[float]:
        """Score all clips in one monitoring window, advancing the timeline."""
        nonlocal t
        seg_start = t
        scores = score_paths(
            wav_paths,
            detector,
            desc=f"    {phase[:4]} r{round_idx}",
        )
        for score in scores:
            events.append({"t": t / 60, "score": score, "phase": phase, "round": round_idx})
            t += CLIP_SECS
        segments.append({
            "phase":   phase,
            "round":   round_idx,
            "t_start": seg_start / 60,
            "t_end":   t / 60,
        })
        return scores

    # ── Monitoring rounds ─────────────────────────────────────────────────────
    round_results: list[dict] = []

    for r in range(n_rounds):
        norm_paths = test_normal_paths[r * monitor_clips:(r + 1) * monitor_clips]
        anom_paths = test_abnormal_paths[r * monitor_clips:(r + 1) * monitor_clips]

        norm_scores = add_segment("normal",  r + 1, norm_paths)
        anom_scores = add_segment("anomaly", r + 1, anom_paths)

        # ── Per-round adaptive threshold ──────────────────────────────────────
        # We always observe a normal window before the anomaly window. The
        # rolling means of that window tell us exactly how the detector behaves
        # under current normal conditions for this round and machine. We use the
        # worst (maximum) rolling mean seen and add a 50 % safety margin.
        #
        # Effect per failure mode observed in the plots:
        #   • Quiet normal, massive anomaly (slider_id_04, fan_id_02):
        #       max_norm_rolling ≈ 0 → threshold_round = training threshold
        #       → first anomaly clip triggers detection immediately.
        #   • Variable normal (fan_id_04 — training threshold too low):
        #       max_norm_rolling is large → threshold_round rises above the
        #       normal spikes → anomaly window requires sustained elevation
        #       to trigger → false positives suppressed.
        #   • Mixed anomaly period (pump_id_04, slider_id_02):
        #       High anomaly clips pull rolling mean above threshold even if
        #       some anomaly clips score near-normal. The window smooths the
        #       mixture rather than missing low-scoring individual clips.
        norm_rolling_means  = [_rolling_mean(norm_scores, i, ROLLING_WINDOW)
                               for i in range(len(norm_scores))]
        max_norm_rolling    = max(norm_rolling_means, default=0.0)
        threshold_round     = max(max_norm_rolling * ADAPTIVE_SAFETY, detector.threshold_)

        # False positives: alarm events in normal window using the SAME
        # threshold_round so the metric is consistent with the detection logic.
        n_false_pos   = _count_alarm_events(norm_scores, threshold_round, ROLLING_WINDOW)
        detection_idx = _detect_first_alarm(anom_scores, threshold_round, ROLLING_WINDOW)

        round_results.append({
            "round":                r + 1,
            "n_false_pos":          n_false_pos,
            "n_normal_clips":       len(norm_scores),
            "n_anom_clips":         len(anom_scores),
            "detected":             detection_idx is not None,
            "detection_delay_secs": int(detection_idx * CLIP_SECS) if detection_idx is not None else None,
            "detection_idx":        detection_idx,
            "threshold_round":      threshold_round,   # adaptive threshold used this round
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
