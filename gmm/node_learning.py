"""Two-node score fusion for TWFR-GMM anomaly detection.

Each node contributes its own fitted ``GMMDetector``. At inference, per-node
NLLs are z-normalised using each node's val stats and combined with
fit-quality softmax weights. A CUSUM is calibrated on the fused val z-scores
so the fused detector exposes the same API as a single-node ``GMMDetector``.
"""

import numpy as np

from gmm.config import CUSUM_H_FLOOR, CUSUM_H_SIGMA, THRESHOLD_PCT
from gmm.detector import GMMDetector

_SIGMA_FLOOR = 1e-8


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class NodeLearning:
    """Fused two-node anomaly detector.

    Fusion::

        z_i(x)  = (NLL_i(x) - mu_val_i) / sigma_val_i
        w_i     = softmax(-sigma_val_i / T)
        fused_z = w_a * z_a + w_b * z_b

    Parameters
    ----------
    detector_a, detector_b : GMMDetector
        Fitted nodes with populated ``val_nlls_``, ``mu_val_``, ``sigma_val_``.
    threshold_pct : float
        Percentile used to calibrate the CUSUM threshold on fused val z-scores.
    cusum_h_sigma, cusum_h_floor : float
        CUSUM alarm-height multiplier and minimum.
    temperature : float
        Softmax temperature. Small T → hard selection; large T → equal weights.
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
            raise RuntimeError("Both detectors must have val_nlls_ populated (call fit()).")

        self._det_a = detector_a
        self._det_b = detector_b
        self.threshold_pct = threshold_pct
        self.cusum_h_sigma = cusum_h_sigma
        self.cusum_h_floor = cusum_h_floor
        self.temperature   = temperature

        self.r_           = f"{detector_a.r_}/{detector_b.r_}"
        self.n_components = detector_a.n_components
        self.n_mels_      = detector_a.n_mels_

        sigmas  = np.array([detector_a.sigma_val_, detector_b.sigma_val_], dtype=np.float64)
        weights = _softmax(-sigmas / temperature)
        self.w_a_ = float(weights[0])
        self.w_b_ = float(weights[1])

        n   = min(len(detector_a.val_nlls_), len(detector_b.val_nlls_))
        z_a = (detector_a.val_nlls_[:n] - detector_a.mu_val_) / detector_a.sigma_val_
        z_b = (detector_b.val_nlls_[:n] - detector_b.mu_val_) / detector_b.sigma_val_
        self.fused_val_z_ = self.w_a_ * z_a + self.w_b_ * z_b

        sorted_z = np.sort(self.fused_val_z_)
        n_z      = len(sorted_z)
        pct_idx  = min(int(n_z * threshold_pct), n_z - 1)
        self.threshold_ = float(sorted_z[pct_idx])
        self.cusum_k_   = self.threshold_
        self.cusum_h_   = float(
            max(cusum_h_sigma * float(self.fused_val_z_.std()), cusum_h_floor)
        )

        self._cusum_S: float = 0.0

        # Populated by train.py so evaluate._score_paths loads the right mic
        # per node. None means mono mix.
        self.channel_a_: int | None = None
        self.channel_b_: int | None = None

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, log_mel: np.ndarray) -> float:
        """Fused z-score when both nodes hear the same audio."""
        return self.score_pair(log_mel, log_mel)

    def score_pair(self, log_mel_a: np.ndarray, log_mel_b: np.ndarray) -> float:
        """Fused z-score from two independent log-mel inputs."""
        nll_a = self._det_a.score(log_mel_a)
        nll_b = self._det_b.score(log_mel_b)
        z_a   = (nll_a - self._det_a.mu_val_) / self._det_a.sigma_val_
        z_b   = (nll_b - self._det_b.mu_val_) / self._det_b.sigma_val_
        return float(self.w_a_ * z_a + self.w_b_ * z_b)

    # ── Online CUSUM ─────────────────────────────────────────────────────────

    def cusum_reset(self) -> None:
        self._cusum_S = 0.0

    def cusum_update(self, score: float) -> bool:
        self._cusum_S = max(0.0, self._cusum_S + score - self.cusum_k_)
        if self._cusum_S >= self.cusum_h_:
            self._cusum_S = 0.0
            return True
        return False

    def cusum_false_alarms(self, scores: list[float]) -> int:
        self.cusum_reset()
        n_alarms = 0
        for s in scores:
            if self.cusum_update(s):
                n_alarms += 1
        return n_alarms
