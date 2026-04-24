"""Single-node TWFR-GMM detector with CUSUM calibration."""

import pickle
from pathlib import Path

import numpy as np

from gmm.config import (
    CUSUM_H_FLOOR,
    CUSUM_H_SIGMA,
    N_COMPONENTS,
    N_MELS,
    R,
    SEED,
    THRESHOLD_PCT,
)
from gmm.features import extract_feature_r
from gmm.gmm import DiagGMM

_SIGMA_FLOOR = 1e-8   # guard against division by zero in z-score fusion


class GMMDetector:
    """TWFR-GMM anomaly detector for one node.

    Fits a ``DiagGMM`` on TWFR features extracted at the given ``r``, then
    calibrates a threshold and a Page-Hinkley CUSUM from a held-out val split.

    Example::

        det = GMMDetector(r=1.0)
        det.fit(fit_log_mels, val_log_mels)
        alarm = det.cusum_update(det.score(log_mel))

    Attributes set by ``fit``:
        gmm_        DiagGMM
        threshold_  95th-percentile NLL on val clips
        cusum_k_    CUSUM reference level (= threshold_)
        cusum_h_    CUSUM alarm height
        val_nlls_   per-val-clip NLLs (used by simulation/node/group.py fusion)
        mu_val_     mean(val_nlls_)  (confidence signal)
        sigma_val_  std(val_nlls_)   (confidence signal, floored)
    """

    def __init__(
        self,
        r:              float = R,
        n_components:   int   = N_COMPONENTS,
        seed:           int   = SEED,
        cusum_h_sigma:  float = CUSUM_H_SIGMA,
        cusum_h_floor:  float = CUSUM_H_FLOOR,
        threshold_pct:  float = THRESHOLD_PCT,
        n_mels:         int   = N_MELS,
    ) -> None:
        self.r_            = r
        self.n_components  = n_components
        self.seed          = seed
        self.cusum_h_sigma = cusum_h_sigma
        self.cusum_h_floor = cusum_h_floor
        self.threshold_pct = threshold_pct
        self.n_mels_       = n_mels

        self.gmm_        : DiagGMM | None    = None
        self.threshold_  : float | None      = None
        self.cusum_k_    : float | None      = None
        self.cusum_h_    : float | None      = None
        self.train_nlls_ : np.ndarray | None = None
        self.val_nlls_   : np.ndarray | None = None
        self.mu_val_     : float | None      = None
        self.sigma_val_  : float | None      = None

        self._cusum_S: float = 0.0

    # ── Fit & calibrate ──────────────────────────────────────────────────────

    def fit(
        self,
        fit_log_mels: list[np.ndarray],
        val_log_mels: list[np.ndarray],
    ) -> "GMMDetector":
        """Fit the GMM on ``fit_log_mels`` and calibrate from ``val_log_mels``."""
        X_fit = np.stack([extract_feature_r(lm, self.r_) for lm in fit_log_mels])
        X_val = np.stack([extract_feature_r(lm, self.r_) for lm in val_log_mels])

        self.gmm_ = DiagGMM(n_components=self.n_components, seed=self.seed)
        self.gmm_.fit(X_fit)

        self.train_nlls_ = self.gmm_.score_samples(X_fit)
        self._calibrate(X_val)
        return self

    def _calibrate(self, X_val: np.ndarray) -> None:
        val_nlls    = self.gmm_.score_samples(X_val)
        sorted_nlls = np.sort(val_nlls)
        n           = len(sorted_nlls)
        pct_idx     = min(int(n * self.threshold_pct), n - 1)

        self.threshold_ = float(sorted_nlls[pct_idx])
        self.cusum_k_   = self.threshold_
        self.cusum_h_   = float(
            max(self.cusum_h_sigma * float(val_nlls.std()), self.cusum_h_floor)
        )
        self._cusum_S = 0.0

        self.val_nlls_  = val_nlls
        self.mu_val_    = float(val_nlls.mean())
        self.sigma_val_ = float(max(float(val_nlls.std()), _SIGMA_FLOOR))

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, log_mel: np.ndarray) -> float:
        """Return the anomaly score (NLL) for one clip. Higher is more anomalous."""
        if self.gmm_ is None:
            raise RuntimeError("GMMDetector.fit() must be called before score().")
        feat = extract_feature_r(log_mel, self.r_)
        return float(self.gmm_.score_samples(feat.reshape(1, -1))[0])

    # ── Online CUSUM ─────────────────────────────────────────────────────────

    def cusum_reset(self) -> None:
        self._cusum_S = 0.0

    def cusum_update(self, score: float) -> bool:
        """Feed one score into the CUSUM; return True when the alarm fires."""
        self._cusum_S = max(0.0, self._cusum_S + score - self.cusum_k_)
        if self._cusum_S >= self.cusum_h_:
            self._cusum_S = 0.0
            return True
        return False

    def cusum_false_alarms(self, scores: list[float]) -> int:
        """Count independent CUSUM alarm events across ``scores``."""
        self.cusum_reset()
        n_alarms = 0
        for s in scores:
            if self.cusum_update(s):
                n_alarms += 1
        return n_alarms

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted detector to ``path``."""
        if self.gmm_ is None:
            raise RuntimeError("GMMDetector.fit() must be called before save().")
        artefact = {
            "r":             self.r_,
            "mu":            self.gmm_.mu_,
            "sigma2":        self.gmm_.sigma2_,
            "pi":            self.gmm_.pi_,
            "threshold":     self.threshold_,
            "cusum_k":       self.cusum_k_,
            "cusum_h":       self.cusum_h_,
            "train_nlls":    self.train_nlls_,
            "val_nlls":      self.val_nlls_,
            "mu_val":        self.mu_val_,
            "sigma_val":     self.sigma_val_,
            "n_components":  self.n_components,
            "threshold_pct": self.threshold_pct,
            "cusum_h_sigma": self.cusum_h_sigma,
            "cusum_h_floor": self.cusum_h_floor,
            "seed":          self.seed,
            "n_mels":        self.n_mels_,
            "n_train_clips": len(self.train_nlls_),
            "feature_dim":   int(self.gmm_.mu_.shape[1]),
        }
        with open(path, "wb") as f:
            pickle.dump(artefact, f)

    @classmethod
    def load(cls, path: str | Path) -> "GMMDetector":
        """Load a pickled detector from ``path``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}")
        with open(path, "rb") as f:
            art = pickle.load(f)

        det = cls(
            r             = art.get("r", R),
            n_components  = art["n_components"],
            seed          = art["seed"],
            cusum_h_sigma = art.get("cusum_h_sigma", CUSUM_H_SIGMA),
            cusum_h_floor = art.get("cusum_h_floor", CUSUM_H_FLOOR),
            threshold_pct = art.get("threshold_pct", THRESHOLD_PCT),
            n_mels        = art.get("n_mels", N_MELS),
        )

        gmm         = DiagGMM(n_components=art["n_components"], seed=art["seed"])
        gmm.mu_     = art["mu"]
        gmm.sigma2_ = art["sigma2"]
        gmm.pi_     = art["pi"]
        gmm._update_lognorm()

        det.gmm_        = gmm
        det.threshold_  = art["threshold"]
        det.cusum_k_    = art.get("cusum_k", art["threshold"])
        det.cusum_h_    = art.get("cusum_h")
        det.train_nlls_ = art["train_nlls"]
        det.val_nlls_   = art.get("val_nlls")
        det.mu_val_     = art.get("mu_val")
        det.sigma_val_  = art.get("sigma_val")
        return det
