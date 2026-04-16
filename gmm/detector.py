"""
detector.py — GMMDetector: single autonomous node detector.

Mirrors the combined behaviour of deployment/gmm.h + deployment/detector.h.

Calibration (mirrors calibrate() in detector.h):
  threshold_  = val_nlls[ floor(N_VAL_CLIPS * THRESHOLD_PCT) ]
  cusum_k_    = threshold_
  cusum_h_    = max(CUSUM_H_SIGMA * std(val_nlls), CUSUM_H_FLOOR)

Detection (mirrors cusum_update() in detector.h):
  S = max(0, S + score - cusum_k)
  alarm fires when S >= cusum_h, then S resets to 0.

The r parameter controls which GWRP feature is extracted.  Defaults to
R=1.0 (mean pooling, the deployment-faithful hardware baseline).  Set
r=0.5 (or any other value in [0,1]) to run the same pipeline with a
different temporal pooling strategy — used by the node learning framework.

Node Learning role
------------------
Each GMMDetector instance corresponds to one autonomous physical node in the
Node Learning framework (Kanjo & Aslanov, 2026, §3, eq. 1-2):

  θ_i  = {mu_, sigma2_, pi_}   — the node's learnable GMM state
  D_i  = fit_log_mels           — the node's local training data
  c_i  = r                      — the node's sensing context (GWRP decay)

  fit()   implements  U_i(θ_i, D_i, c_i)  — local adaptation, no communication.
  score() implements  ℓ(θ_i; x)          — per-clip NLL (anomaly score).

After fitting, three confidence signals are stored for use by NodeLearning:
  val_nlls_  : NLL scores on the N_VAL_CLIPS held-out clips
  mu_val_    : mean of val_nlls_   (fit quality signal)
  sigma_val_ : std  of val_nlls_   (floored at 1e-8 to avoid division by zero)

These signals are the "learned knowledge" exchanged between nodes in the Node
Learning merge operator (paper eq. 3).  They are lightweight scalars — not raw
audio, not GMM parameters.
"""

import pickle
from pathlib import Path

import numpy as np

from gmm.config import (
    CUSUM_H_FLOOR,
    CUSUM_H_SIGMA,
    N_COMPONENTS,
    R,
    SEED,
    THRESHOLD_PCT,
)
from gmm.features import extract_feature_r
from gmm.gmm import DiagGMM

_SIGMA_FLOOR = 1e-8   # guards NodeLearning z-score division


class GMMDetector:
    """Deployment-faithful TWFR-GMM anomaly detector.

    Wraps DiagGMM with threshold calibration and a stateful online CUSUM
    that mirrors cusum_update() / calibrate() in deployment/detector.h.

    Workflow::

        detector = GMMDetector(r=1.0)
        detector.fit(fit_log_mels, val_log_mels)  # train GMM + calibrate
        score    = detector.score(log_mel)         # per-clip NLL
        alarm    = detector.cusum_update(score)    # online, stateful
        detector.cusum_reset()                     # call between rounds
        detector.save("fan_id_00.pkl")

    Parameters
    ----------
    r : float
        GWRP decay parameter used for feature extraction.  Default R=1.0
        (mean pooling, matches deployment hardware).  Use other values only
        in node-learning experiments — on-device only r=1.0 is feasible
        without the full spectrogram buffer.
    n_components : int
        Number of GMM components.
    seed : int
        Random seed for DiagGMM initialisation.
    cusum_h_sigma : float
        CUSUM alarm height multiplier: cusum_h = cusum_h_sigma × std(val_nlls).
    cusum_h_floor : float
        Minimum cusum_h — guards against degenerate val NLL distributions.
    threshold_pct : float
        Detection threshold percentile as a fraction (e.g. 0.95 = 95th pct).
        Uses floor indexing to match the C++ implementation.

    Attributes (set after fit())
    ----------------------------
    gmm_        : DiagGMM
    r_          : float    — r value used (= constructor arg r)
    threshold_  : float    — detection threshold (in NLL space)
    cusum_k_    : float    — CUSUM reference level (= threshold_)
    cusum_h_    : float    — CUSUM alarm height
    train_nlls_ : ndarray  — NLL scores on fit clips (diagnostics)
    val_nlls_   : ndarray  — NLL scores on val clips  (used by NodeLearning)
    mu_val_     : float    — mean of val_nlls_         (used by NodeLearning)
    sigma_val_  : float    — std  of val_nlls_         (used by NodeLearning)
    """

    def __init__(
        self,
        r:              float = R,
        n_components:   int   = N_COMPONENTS,
        seed:           int   = SEED,
        cusum_h_sigma:  float = CUSUM_H_SIGMA,
        cusum_h_floor:  float = CUSUM_H_FLOOR,
        threshold_pct:  float = THRESHOLD_PCT,
    ) -> None:
        self.r_            = r
        self.n_components  = n_components
        self.seed          = seed
        self.cusum_h_sigma = cusum_h_sigma
        self.cusum_h_floor = cusum_h_floor
        self.threshold_pct = threshold_pct

        # Populated by fit()
        self.gmm_        : DiagGMM | None    = None
        self.threshold_  : float | None      = None
        self.cusum_k_    : float | None      = None
        self.cusum_h_    : float | None      = None
        self.train_nlls_ : np.ndarray | None = None
        self.val_nlls_   : np.ndarray | None = None   # for NodeLearning
        self.mu_val_     : float | None      = None   # for NodeLearning
        self.sigma_val_  : float | None      = None   # for NodeLearning

        # Stateful CUSUM accumulator — mirrors det_cusum_S in detector.h.
        self._cusum_S: float = 0.0

    # ── Fit & calibrate ───────────────────────────────────────────────────────

    def fit(
        self,
        fit_log_mels: list[np.ndarray],
        val_log_mels: list[np.ndarray],
    ) -> "GMMDetector":
        """Fit the GMM and calibrate thresholds.

        Parameters
        ----------
        fit_log_mels : list of np.ndarray, each shape (N_MELS, T)
            N_FIT_CLIPS log-mel spectrograms used to train the GMM.
        val_log_mels : list of np.ndarray, each shape (N_MELS, T)
            N_VAL_CLIPS held-out log-mel spectrograms used only for
            threshold and CUSUM calibration.

        Returns
        -------
        self : GMMDetector
        """
        X_fit = np.stack([extract_feature_r(lm, self.r_) for lm in fit_log_mels])
        X_val = np.stack([extract_feature_r(lm, self.r_) for lm in val_log_mels])

        self.gmm_ = DiagGMM(n_components=self.n_components, seed=self.seed)
        self.gmm_.fit(X_fit)

        self.train_nlls_ = self.gmm_.score_samples(X_fit)
        self._calibrate(X_val)
        return self

    def _calibrate(self, X_val: np.ndarray) -> None:
        """Set threshold and CUSUM parameters from held-out val features.

        Also stores val_nlls_, mu_val_, sigma_val_ for use by NodeLearning.

        Mirrors calibrate() in deployment/detector.h:
          threshold = sorted_val_nlls[ floor(N * THRESHOLD_PCT) ]
          cusum_h   = max(CUSUM_H_SIGMA * std(val_nlls), CUSUM_H_FLOOR)
        """
        val_nlls    = self.gmm_.score_samples(X_val)
        sorted_nlls = np.sort(val_nlls)
        n           = len(sorted_nlls)
        pct_idx     = min(int(n * self.threshold_pct), n - 1)

        self.threshold_ = float(sorted_nlls[pct_idx])
        self.cusum_k_   = self.threshold_
        self.cusum_h_   = float(
            max(self.cusum_h_sigma * float(val_nlls.std()), self.cusum_h_floor)
        )
        self._cusum_S   = 0.0

        # Store val distribution statistics for NodeLearning z-score normalisation.
        self.val_nlls_  = val_nlls
        self.mu_val_    = float(val_nlls.mean())
        self.sigma_val_ = float(max(float(val_nlls.std()), _SIGMA_FLOOR))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, log_mel: np.ndarray) -> float:
        """Compute the anomaly score (NLL) for a single clip.

        Parameters
        ----------
        log_mel : np.ndarray, shape (N_MELS, T)

        Returns
        -------
        nll : float  — higher values indicate greater anomaly.
        """
        if self.gmm_ is None:
            raise RuntimeError("GMMDetector.fit() must be called before score().")
        feat = extract_feature_r(log_mel, self.r_)
        return float(self.gmm_.score_samples(feat.reshape(1, -1))[0])

    # ── Stateful CUSUM ────────────────────────────────────────────────────────

    def cusum_reset(self) -> None:
        """Reset the CUSUM accumulator to zero."""
        self._cusum_S = 0.0

    def cusum_update(self, score: float) -> bool:
        """Feed one clip's score into the CUSUM accumulator.

        Mirrors cusum_update() in deployment/detector.h:
            S = max(0, S + score - cusum_k)
            if S >= cusum_h: reset S, return True
        """
        self._cusum_S = max(0.0, self._cusum_S + score - self.cusum_k_)
        if self._cusum_S >= self.cusum_h_:
            self._cusum_S = 0.0
            return True
        return False

    def cusum_false_alarms(self, scores: list[float]) -> int:
        """Count CUSUM alarms in a sequence of scores."""
        self.cusum_reset()
        n_alarms = 0
        for s in scores:
            if self.cusum_update(s):
                n_alarms += 1
        return n_alarms

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the fitted detector to a .pkl artefact file."""
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
            "n_train_clips": len(self.train_nlls_),
            "feature_dim":   int(self.gmm_.mu_.shape[1]),
        }
        with open(path, "wb") as f:
            pickle.dump(artefact, f)

    @classmethod
    def load(cls, path: str | Path) -> "GMMDetector":
        """Load a fitted detector from a .pkl artefact file."""
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
        )

        gmm           = DiagGMM(n_components=art["n_components"], seed=art["seed"])
        gmm.mu_       = art["mu"]
        gmm.sigma2_   = art["sigma2"]
        gmm.pi_       = art["pi"]
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
