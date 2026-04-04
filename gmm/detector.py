"""
detector.py — GMMDetector: TWFR-GMM anomaly detector.

Implements the GMM anomaly estimator from:
  Guan et al., "Time-Weighted Frequency Domain Audio Representation with GMM
  Estimator for Anomalous Sound Detection", arXiv:2305.03328 (2023).

The detector wraps a scikit-learn GaussianMixture and adds:
  * Self-supervised r search: selects the GWRP decay parameter r by fitting
    a GMM for each candidate and choosing the r with the highest mean
    log-likelihood on training data — no anomaly labels required, and the
    approach is directly realisable on-device.
  * Threshold calibration: sets the detection threshold at the configured
    percentile of training NLL scores (default: 95th), matching the SVDD
    pipeline in separator/separator.py.
  * Artefact persistence: save/load via pickle so evaluation can be decoupled
    from training.

Dependencies: numpy, scikit-learn, pickle, gmm.features (no torch, no librosa).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent.parent))
from gmm.features import twfr_feature


# ── GMMDetector ───────────────────────────────────────────────────────────────

class GMMDetector:
    """TWFR-GMM anomaly detector (Guan et al., arXiv:2305.03328).

    Workflow::

        detector = GMMDetector(n_components=2)
        detector.fit(train_log_mels)        # r search + GMM fit + threshold
        score = detector.score(test_log_mel) # NLL — higher means more anomalous
        detector.save("fan_id_00.pkl")

        # Later, for evaluation only:
        detector = GMMDetector.load("fan_id_00.pkl")
        score = detector.score(test_log_mel)

    Parameters
    ----------
    n_components : int, optional
        Number of Gaussian mixture components. Default 2 (per the paper).
    covariance_type : str, optional
        GMM covariance structure passed to sklearn. Default ``'diag'``.
        The paper uses 128-dimensional features with ~50–60 training clips;
        full covariance would require estimating 8,256 parameters per
        component from 50 samples — massively underdetermined. Diagonal
        covariance (128 parameters/component) is well-conditioned and
        matches the practical constraints of the paper's setup.
    threshold_pct : int, optional
        Percentile of training NLL scores used as the detection threshold.
        Default 95, matching THRESHOLD_PCT in config.py.
    seed : int, optional
        Random state for GMM initialisation. Default 42.

    Attributes
    ----------
    r_ : float or None
        Selected GWRP decay parameter. None until :meth:`fit` is called.
    gmm_ : GaussianMixture or None
        Fitted sklearn GMM. None until :meth:`fit` is called.
    threshold_ : float or None
        Detection threshold (``threshold_pct``-th percentile of training NLL).
        None until :meth:`fit` is called.
    train_nlls_ : np.ndarray or None
        Per-clip NLL scores on training data, shape (N_train,).
        None until :meth:`fit` is called. Useful for diagnostics.
    """

    # Candidate r values for self-supervised search.
    # Covers the full range from pure max-pooling (0) to pure mean-pooling (1)
    # with finer resolution near 1, where most machine types land (Guan et al.).
    R_GRID: list[float] = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "diag",
        threshold_pct: int = 95,
        seed: int = 42,
        cusum_h_sigma: float = 5.0,
    ) -> None:
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.threshold_pct   = threshold_pct
        self.seed            = seed
        # cusum_h_ is set to cusum_h_sigma × std(val_nlls) during fit().
        # Interpretation: alarm fires after accumulating cusum_h_sigma standard
        # deviations worth of excess score. A single outlier normal clip scores at
        # most ~1–2σ above the reference, so cusum_h_sigma=5 requires roughly 3–5
        # consecutive anomalous clips before triggering — matching the temporal
        # persistence visible in the plots.
        self.cusum_h_sigma   = cusum_h_sigma

        # Set after fit()
        self.r_          : float | None      = None
        self.gmm_        : GaussianMixture | None = None
        self.threshold_  : float | None      = None
        self.train_nlls_ : np.ndarray | None = None

        # CUSUM detection parameters — set by fit() alongside threshold_.
        # cusum_k_ : reference value (= threshold_); clips above this contribute
        #            positively to the CUSUM accumulator.
        # cusum_h_ : alarm threshold; derived from the standard deviation of
        #            held-out validation NLLs × cusum_h_sigma.
        self.cusum_k_    : float | None      = None
        self.cusum_h_    : float | None      = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_features(
        self, log_mels: list[np.ndarray], r: float
    ) -> np.ndarray:
        """Batch-compute TWFR features for a list of log-mel spectrograms.

        Parameters
        ----------
        log_mels : list of np.ndarray, each shape (M, T)
            Full-clip log-mel spectrograms.
        r : float
            GWRP decay parameter.

        Returns
        -------
        X : np.ndarray, shape (N, M), dtype float32
            Feature matrix, one row per clip.
        """
        return np.stack([twfr_feature(lm, r) for lm in log_mels])

    def _fit_gmm(self, X: np.ndarray) -> tuple[GaussianMixture, float]:
        """Fit a GMM on X and return (fitted_gmm, mean_log_likelihood).

        Mean log-likelihood is used for r selection: a higher value means the
        GMM describes the training distribution more tightly, which corresponds
        to a better-suited r for this machine's acoustic character.

        Parameters
        ----------
        X : np.ndarray, shape (N, M)
            Feature matrix to fit on.

        Returns
        -------
        gmm : GaussianMixture
            Fitted sklearn GaussianMixture.
        mean_ll : float
            Mean log-likelihood of X under the fitted GMM (gmm.score(X)).
        """
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
        )
        gmm.fit(X)
        # gmm.score() returns mean log P(x) per sample — scalar
        return gmm, float(gmm.score(X))

    def _max_component_nll(self, X: np.ndarray) -> np.ndarray:
        """Per-sample anomaly scores using Eq. 3 from Guan et al. (2023).

        The paper defines the anomaly score as:
            A(X̄) = -max_{k∈[1,K]} log N(R(X̄)|μk, Σk)

        This takes the **negative of the best-fitting component's log-
        likelihood**, rather than the mixture log-likelihood returned by
        sklearn's ``score_samples()``. The key difference:

        * ``score_samples()``  →  log Σ_k π_k N(x|μk, Σk)   (mixture, with weights)
        * this method         →  -max_k log N(x|μk, Σk)      (max component, no weights)

        For K=1 (single Gaussian) the two are equivalent up to a constant.
        For K=2, this rewards proximity to the *nearest* cluster, which is the
        correct semantics when the normal distribution is bimodal: a point near
        either cluster should score low (not anomalous).

        Uses sklearn's internal ``_estimate_log_prob(X)`` which returns
        log N(x|μk, Σk) for each component without mixing weights, matching
        the paper formula exactly.

        Parameters
        ----------
        X : np.ndarray, shape (N, M)
            Feature matrix, one row per clip.

        Returns
        -------
        nlls : np.ndarray, shape (N,)
            Anomaly score per sample. Higher = more anomalous.
        """
        # _estimate_log_prob returns log N(x|mu_k, sigma_k), shape (N, K)
        log_probs = self.gmm_._estimate_log_prob(X)   # (N, K)
        return -np.max(log_probs, axis=1)             # (N,)

    # ── Public API ────────────────────────────────────────────────────────────

    def search_r(
        self,
        log_mels: list[np.ndarray],
        verbose: bool = False,
    ) -> float:
        """Self-supervised search for the optimal GWRP decay parameter r.

        For each candidate r in R_GRID, extracts TWFR features from the
        supplied log-mel spectrograms, fits a GMM, and records the mean
        log-likelihood. The r yielding the highest mean log-likelihood is
        selected — no anomaly labels are required.

        Intuition: when r is well-matched to the machine's temporal structure
        (stationary vs. transient), the GMM achieves a tighter, higher-
        likelihood fit to the normal training distribution.

        Parameters
        ----------
        log_mels : list of np.ndarray
            Log-mel spectrograms of normal training clips.
        verbose : bool, optional
            If True, prints per-r mean log-likelihood to stdout.

        Returns
        -------
        best_r : float
            The r value with the highest mean log-likelihood.
        """
        best_r, best_ll = self.R_GRID[0], -np.inf

        for r in self.R_GRID:
            X = self._extract_features(log_mels, r)
            _, mean_ll = self._fit_gmm(X)
            if verbose:
                print(f"      r={r:.2f}  mean_ll={mean_ll:.4f}")
            if mean_ll > best_ll:
                best_ll = mean_ll
                best_r  = r

        return best_r

    def fit(
        self,
        log_mels: list[np.ndarray],
        r: float | None = None,
        verbose: bool = False,
        val_log_mels: list[np.ndarray] | None = None,
    ) -> "GMMDetector":
        """Fit the detector on normal training clips.

        If ``r`` is None, :meth:`search_r` is called first to select it
        automatically. After fitting, ``r_``, ``gmm_``, ``threshold_``, and
        ``train_nlls_`` are populated.

        Parameters
        ----------
        log_mels : list of np.ndarray
            Log-mel spectrograms used to fit the GMM (full 10 s clips).
        r : float or None, optional
            If provided, skips the r search and uses this value directly.
        verbose : bool, optional
            If True, prints per-r log-likelihoods during the r search.
        val_log_mels : list of np.ndarray or None, optional
            Held-out normal clips used solely to calibrate the detection
            threshold. When provided, these clips are *not* used to fit the
            GMM, so their NLL scores reflect genuine generalisation to unseen
            data. This avoids the train-set threshold contamination problem
            where the GMM achieves artificially high log-likelihood on its
            own training clips, pushing the threshold to a regime that no
            test clip ever reaches.

            Recommended split: 50 clips to fit, 10 clips to calibrate.
            If None, the threshold is computed on the training clips (legacy
            behaviour — inflated FP rate expected).

        Returns
        -------
        self : GMMDetector
            Fitted detector (supports method chaining).
        """
        if r is None:
            r = self.search_r(log_mels, verbose=verbose)

        self.r_ = r

        X             = self._extract_features(log_mels, r)   # (N, M)
        self.gmm_, _  = self._fit_gmm(X)

        # Anomaly score per Eq. 3 of Guan et al. (2023): -max_k log N(x|μk, Σk).
        # Compute threshold on held-out validation clips when provided to avoid
        # train-set contamination: the GMM scores its own training clips
        # optimistically, placing the threshold in a regime test data never
        # reaches and causing 100% FP.
        self.train_nlls_ = self._max_component_nll(X)
        if val_log_mels is not None:
            X_val           = self._extract_features(val_log_mels, r)
            val_nlls        = self._max_component_nll(X_val)
            self.threshold_ = float(np.percentile(val_nlls, self.threshold_pct))
            # CUSUM parameters derived from val NLL distribution:
            #   k = threshold (clips scoring above this accumulate suspicion)
            #   h = cusum_h_sigma × std(val_nlls)
            # The std of val_nlls represents natural score variability on unseen
            # normal clips. A single outlier normal clip contributes at most ~1–2σ
            # to the accumulator before it decays; sustained anomaly scoring at
            # (threshold + many σ) reaches h quickly.
            self.cusum_k_ = self.threshold_
            self.cusum_h_ = float(np.std(val_nlls) * self.cusum_h_sigma)
        else:
            self.threshold_ = float(np.percentile(self.train_nlls_, self.threshold_pct))
            self.cusum_k_   = self.threshold_
            self.cusum_h_   = float(np.std(self.train_nlls_) * self.cusum_h_sigma)

        return self

    def score(self, log_mel: np.ndarray) -> float:
        """Compute the anomaly score for a single clip.

        Parameters
        ----------
        log_mel : np.ndarray, shape (M, T)
            Full-clip log-mel spectrogram.

        Returns
        -------
        nll : float
            Negative log-likelihood under the fitted GMM. Higher values
            indicate greater deviation from the learned normal distribution.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.gmm_ is None:
            raise RuntimeError(
                "GMMDetector.fit() must be called before score()."
            )
        feat = twfr_feature(log_mel, self.r_)    # (M,)
        return float(self._max_component_nll(feat.reshape(1, -1))[0])

    def cusum_detect(
        self,
        scores: list[float],
        h: float | None = None,
    ) -> int | None:
        """Run one-sided CUSUM over a sequence of anomaly scores.

        Implements the Page-Hinkley CUSUM for detecting an upward shift in
        the mean score (normal → anomalous):

            S_0 = 0
            S_t = max(0, S_{t-1} + score_t − cusum_k_)
            alarm if S_t ≥ h

        The ``max(0, ...)`` operation is the key property: when scores return
        to normal (score_t < cusum_k_), the accumulator is pulled toward zero
        and resets naturally. No explicit reset after an anomalous period is
        needed. An isolated high-scoring clip (a common FP source with
        per-clip thresholding) contributes once then decays; three or more
        consecutive anomalous clips drive S_t past h.

        Parameters
        ----------
        scores : list of float
            Anomaly scores for a sequence of clips, in temporal order.
        h : float or None, optional
            Alarm threshold. If None, uses ``self.cusum_h_`` (the training-
            calibrated value). Pass an adaptive per-round h derived from the
            preceding normal monitoring window to improve robustness against
            machine-to-machine threshold variability.

        Returns
        -------
        detection_idx : int or None
            Index of the first clip at which the CUSUM alarm fires, or
            ``None`` if no alarm fires within the sequence.
        """
        if self.cusum_k_ is None or self.cusum_h_ is None:
            raise RuntimeError(
                "GMMDetector.fit() must be called before cusum_detect()."
            )
        h_eff = h if h is not None else self.cusum_h_
        S = 0.0
        for i, s in enumerate(scores):
            S = max(0.0, S + s - self.cusum_k_)
            if S >= h_eff:
                return i
        return None

    def cusum_false_alarms(self, scores: list[float]) -> int:
        """Count CUSUM alarms in a normal monitoring window.

        Runs CUSUM on ``scores`` and counts how many times the accumulator
        exceeds ``cusum_h_``. After each alarm the accumulator resets to 0,
        allowing further alarms within the same window.

        Parameters
        ----------
        scores : list of float
            Anomaly scores for a sequence of clips scored during a normal
            monitoring window.

        Returns
        -------
        n_alarms : int
            Number of CUSUM alarm events (0 means no false positives).
        """
        if self.cusum_k_ is None or self.cusum_h_ is None:
            raise RuntimeError(
                "GMMDetector.fit() must be called before cusum_false_alarms()."
            )
        S, n_alarms = 0.0, 0
        for s in scores:
            S = max(0.0, S + s - self.cusum_k_)
            if S >= self.cusum_h_:
                n_alarms += 1
                S = 0.0   # reset accumulator after each alarm
        return n_alarms

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the fitted detector to a .pkl artefact file.

        The artefact dict mirrors the vocabulary used by separator/train.py
        (.pt files) where possible, making the two pipelines directly
        comparable.

        Parameters
        ----------
        path : str or Path
            Destination file path (conventionally ends in ``.pkl``).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.gmm_ is None:
            raise RuntimeError(
                "GMMDetector.fit() must be called before save()."
            )
        artefact = {
            "r":               self.r_,
            "gmm":             self.gmm_,           # sklearn object (pickle-safe)
            "threshold":       self.threshold_,
            "cusum_k":         self.cusum_k_,
            "cusum_h":         self.cusum_h_,
            "train_nlls":      self.train_nlls_,    # (N_train,) — for diagnostics
            "n_components":    self.n_components,
            "covariance_type": self.covariance_type,
            "threshold_pct":   self.threshold_pct,
            "cusum_h_sigma":   self.cusum_h_sigma,
            "seed":            self.seed,
            "n_train_clips":   len(self.train_nlls_),
            "feature_dim":     int(self.gmm_.means_.shape[1]),  # = GMM_N_MELS
        }
        with open(path, "wb") as f:
            pickle.dump(artefact, f)

    @classmethod
    def load(cls, path: str | Path) -> "GMMDetector":
        """Load a fitted detector from a .pkl artefact file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pkl`` file previously written by :meth:`save`.

        Returns
        -------
        detector : GMMDetector
            Fully initialised detector ready for :meth:`score`.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}")

        with open(path, "rb") as f:
            art = pickle.load(f)

        det = cls(
            n_components=art["n_components"],
            covariance_type=art["covariance_type"],
            threshold_pct=art["threshold_pct"],
            cusum_h_sigma=art.get("cusum_h_sigma", 5.0),
            seed=art["seed"],
        )
        det.r_          = art["r"]
        det.gmm_        = art["gmm"]
        det.threshold_  = art["threshold"]
        det.cusum_k_    = art.get("cusum_k", art["threshold"])
        det.cusum_h_    = art.get("cusum_h")
        det.train_nlls_ = art["train_nlls"]
        return det
