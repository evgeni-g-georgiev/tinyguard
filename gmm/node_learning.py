"""
node_learning.py — Two-node collaborative anomaly detection.

NodeLearning implements the "collaborative learning" operating regime of the
Node Learning paradigm (Kanjo & Aslanov, 2026, §2, Figure 2).

Background
----------
The fundamental constraint of deploying a GMM-based anomaly detector on an
Arduino Nano 33 BLE is memory: storing the full (N_MELS×N_FRAMES) = (128×312)
spectrogram for the GWRP sort requires 156 KB of SRAM, leaving the 256 KB
device without enough space for anything else.  Only r=1.0 (mean pooling,
computable as a running accumulator) is feasible on a single node.

The Node Learning approach circumvents this by deploying *two* co-located
physical nodes, each with a fixed r value:

  Node A — mean pooling or GWRP, r selected by r-search.
  Node B — complementary GWRP, r selected by diversity-constrained r-search
            (|r_B − r_A| ≥ diversity_margin) to enforce functional complementarity.

After each node trains its own GMM locally (no data exchange, no shared model),
they share lightweight confidence signals — the mean and standard deviation of
their held-out validation NLL distributions.  These signals allow the system to
produce a single fused anomaly score that is better calibrated than either node
alone.

Node Learning realisation
-------------------------
This class concretely realises the following concepts from the paper:

  Pairwise interaction (§3, eq. 3):
      θ_i^{t+1} ← M_i(θ_i^{t+1}, {θ_j^t}_{j∈N_i(t)}, c_i, c_j)
    The merge operator M_i here is score-space fusion: Node A integrates
    Node B's learned statistics (μ_val, σ_val) at inference time without
    modifying its own GMM parameters.  This is "collaborative inference"
    rather than parameter averaging — consistent with the paper's description
    of M_i as abstracting "feature sharing, partial model updates, distillation,
    or other transfer mechanisms."

  Context-weighted interaction (§3, eq. 5):
      θ_i^{t+1} ← M_i(θ_i^{t+1}, {ϕ_j^t}_{j∈N_i(t)}, c_i, c_j)
    Fit-quality weights w_i = softmax(−μ_val_i / T) serve as the context signal
    ϕ_j: a node that fits normal data better (lower mean val NLL) is trusted
    more at inference time.  Temperature T (default 100) softens the weights
    toward equal blending — at T=1 the softmax collapses to hard 0/1 selection
    on most machines; at T=100 weights are near-equal (~0.49–0.55) with a small
    bias toward the better-fitting node that is decisive on borderline detections.

  Functional complementarity (§2, Hossain et al.):
    Node A and Node B perceive different temporal structures of the same audio
    due to their different r values.  Their GMMs therefore carry complementary
    information about normality, and the fusion exploits this diversity.

  Heterogeneous nodes (§3):
    Hardware-induced heterogeneity (r=1.0 forced by memory, r=0.5 on a second
    device) matches the paper's heterogeneous-node regime where "nodes differ
    in … sensing modality, compute capability."

Fusion mechanism
----------------
1. Z-score normalisation per node:
       z_i(x) = (NLL_i(x) − μ_val_i) / σ_val_i
   This expresses each NLL as "standard deviations above this node's normal-class
   distribution," making scores from the two GMMs directly comparable regardless
   of their different absolute NLL scales.

2. Fit-quality weighted sum:
       fused_z(x) = w_A × z_A(x) + w_B × z_B(x)
       w_i = softmax(−μ_val_i)   (numerically stable)
   Lower mean val NLL → better GMM fit → higher weight.

3. CUSUM calibrated on fused val z-scores:
   The same floor-index 95th percentile and σ-multiplier as GMMDetector are
   applied to the distribution of fused_val_z, so the detection criterion is
   consistently defined across all three variants (Node A, Node B, NodeLearning).

Interface
---------
NodeLearning exposes the same public API as GMMDetector (duck-typing):

    score(log_mel)             → float   (fused z-score)
    cusum_update(score)        → bool    (CUSUM alarm)
    cusum_reset()                        (reset accumulator)
    cusum_false_alarms(scores) → int
    threshold_                 float
    cusum_k_                   float
    cusum_h_                   float
    r_                         str       e.g. "1.0/0.5"
    n_components               int       (from detector_a)

This means evaluate.py, plot.py, and train.py work identically for all three
detector variants without any modifications.
"""

import numpy as np

from gmm.config import CUSUM_H_FLOOR, CUSUM_H_SIGMA, THRESHOLD_PCT
from gmm.detector import GMMDetector

_SIGMA_FLOOR = 1e-8


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtract max before exp to avoid overflow."""
    e = np.exp(x - x.max())
    return e / e.sum()


class NodeLearning:
    """Two-node collaborative anomaly detector (Node Learning paradigm).

    Combines two independently fitted GMMDetectors — each representing one
    physical edge node with a distinct r value — via z-score normalisation
    and fit-quality weighted score fusion.

    See module docstring for the full theoretical background and mapping to the
    Kanjo & Aslanov (2026) Node Learning paper.

    Parameters
    ----------
    detector_a : GMMDetector
        Node A — already fitted.  Typically r=NODE_R_A=1.0 (mean pooling,
        deployment-faithful hardware baseline).
    detector_b : GMMDetector
        Node B — already fitted.  Typically r=NODE_R_B=0.5 (energy-weighted
        GWRP, capturing high-energy transients).
    threshold_pct : float
        Percentile for CUSUM threshold calibration on fused val z-scores.
        Default matches GMMDetector (THRESHOLD_PCT = 0.95).
    cusum_h_sigma : float
        CUSUM alarm height multiplier.  Default matches GMMDetector.
    cusum_h_floor : float
        Minimum cusum_h.  Default matches GMMDetector.
    temperature : float
        Softmax temperature for fit-quality weights: softmax(-μ_val / T).
        T=1 → hard selection (collapses on most machines). T=100 (default) →
        near-equal weights with a small bias toward the better-fitting node.

    Attributes (set after __init__)
    --------------------------------
    w_a_, w_b_       : float    — fit-quality softmax weights (sum to 1.0)
    fused_val_z_     : ndarray  — fused val z-scores (diagnostic)
    threshold_       : float    — CUSUM threshold in z-score space
    cusum_k_         : float    — = threshold_
    cusum_h_         : float    — CUSUM alarm height
    r_               : str      — descriptor, e.g. "1.0/0.5"
    n_components     : int      — from detector_a
    """

    def __init__(
        self,
        detector_a:    GMMDetector,
        detector_b:    GMMDetector,
        threshold_pct: float = THRESHOLD_PCT,
        cusum_h_sigma: float = CUSUM_H_SIGMA,
        cusum_h_floor: float = CUSUM_H_FLOOR,
        temperature:   float = 100.0,
    ) -> None:
        if detector_a.gmm_ is None or detector_b.gmm_ is None:
            raise RuntimeError("Both detectors must be fitted before NodeLearning.")
        if detector_a.val_nlls_ is None or detector_b.val_nlls_ is None:
            raise RuntimeError(
                "Both detectors must have val_nlls_ populated. "
                "Ensure fit() was called with val_log_mels."
            )

        self._det_a = detector_a
        self._det_b = detector_b
        self.threshold_pct = threshold_pct
        self.cusum_h_sigma = cusum_h_sigma
        self.cusum_h_floor = cusum_h_floor

        # Human-readable descriptor for plot subtitles and YAML output.
        self.r_           = f"{detector_a.r_}/{detector_b.r_}"
        self.n_components = detector_a.n_components
        self.n_mels_      = detector_a.n_mels_

        # ── Fit-quality weights (paper §3, eq. 5 context-weighted interaction) ─
        # weights = softmax(-μ_val / T): lower mean val NLL → higher weight.
        # Temperature T softens the distribution; default T=100 gives near-equal
        # weights with a small bias toward the better-fitting node.
        mu_vals   = np.array([detector_a.mu_val_, detector_b.mu_val_], dtype=np.float64)
        weights   = _softmax(-mu_vals / temperature)
        self.w_a_ = float(weights[0])
        self.w_b_ = float(weights[1])

        # ── Fused val z-scores (used for CUSUM calibration) ───────────────────
        # Z-normalise each node's val NLLs independently to make them comparable
        # regardless of the different absolute scales of the two GMMs.
        # Take min length in case of rare clip-count mismatch.
        n   = min(len(detector_a.val_nlls_), len(detector_b.val_nlls_))
        z_a = (detector_a.val_nlls_[:n] - detector_a.mu_val_) / detector_a.sigma_val_
        z_b = (detector_b.val_nlls_[:n] - detector_b.mu_val_) / detector_b.sigma_val_
        self.fused_val_z_ = self.w_a_ * z_a + self.w_b_ * z_b

        # ── CUSUM calibration on fused val z-scores ───────────────────────────
        # Mirrors GMMDetector._calibrate(): floor-index percentile + σ-multiplier.
        sorted_z  = np.sort(self.fused_val_z_)
        n_z       = len(sorted_z)
        pct_idx   = min(int(n_z * threshold_pct), n_z - 1)
        self.threshold_ = float(sorted_z[pct_idx])
        self.cusum_k_   = self.threshold_
        self.cusum_h_   = float(
            max(cusum_h_sigma * float(self.fused_val_z_.std()), cusum_h_floor)
        )

        # Stateful CUSUM accumulator — same semantics as GMMDetector._cusum_S.
        self._cusum_S: float = 0.0

        # Microphone channel indices — set by train.py after construction so
        # evaluate._score_paths knows which channel to load per node.
        # None means mono mix (legacy / backwards-compatible behaviour).
        self.channel_a_: int | None = None
        self.channel_b_: int | None = None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, log_mel: np.ndarray) -> float:
        """Compute the fused anomaly z-score when both nodes hear the same audio.

        Used in the r-heterogeneity-only baseline where a single (mono-mixed)
        log-mel is shared between nodes.  For mic-heterogeneity experiments use
        score_pair() instead.
        """
        return self.score_pair(log_mel, log_mel)

    def score_pair(self, log_mel_a: np.ndarray, log_mel_b: np.ndarray) -> float:
        """Compute the fused anomaly z-score from two independent node inputs.

        Each node scores its own audio independently, then z-normalised NLLs
        are combined with fit-quality weights (paper §3, eq. 3 merge operator).

        Parameters
        ----------
        log_mel_a : np.ndarray, shape (N_MELS, T)  — Node A's audio (e.g. mic 0)
        log_mel_b : np.ndarray, shape (N_MELS, T)  — Node B's audio (e.g. mic 1)

        Returns
        -------
        fused_z : float
            Weighted sum of z-normalised NLLs.  Higher = more anomalous.
        """
        nll_a = self._det_a.score(log_mel_a)
        nll_b = self._det_b.score(log_mel_b)
        z_a   = (nll_a - self._det_a.mu_val_) / self._det_a.sigma_val_
        z_b   = (nll_b - self._det_b.mu_val_) / self._det_b.sigma_val_
        return float(self.w_a_ * z_a + self.w_b_ * z_b)

    # ── Stateful CUSUM (mirrors GMMDetector exactly) ──────────────────────────

    def cusum_reset(self) -> None:
        """Reset the CUSUM accumulator to zero."""
        self._cusum_S = 0.0

    def cusum_update(self, score: float) -> bool:
        """Feed one clip's fused z-score into the CUSUM.

        Mirrors cusum_update() in deployment/detector.h:
            S = max(0, S + score − cusum_k)
            Alarm (and reset) when S ≥ cusum_h.

        Parameters
        ----------
        score : float — fused z-score from self.score().

        Returns
        -------
        True if an alarm fires (S reached cusum_h), else False.
        """
        self._cusum_S = max(0.0, self._cusum_S + score - self.cusum_k_)
        if self._cusum_S >= self.cusum_h_:
            self._cusum_S = 0.0
            return True
        return False

    def cusum_false_alarms(self, scores: list[float]) -> int:
        """Count CUSUM alarms in a sequence of fused z-scores.

        Resets the accumulator before counting, and resets again after each
        alarm — each alarm is an independent excursion event, matching the
        cusum_false_alarms() behaviour in GMMDetector.
        """
        self.cusum_reset()
        n_alarms = 0
        for s in scores:
            if self.cusum_update(s):
                n_alarms += 1
        return n_alarms
