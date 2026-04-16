"""
gmm.py — Hand-rolled diagonal-covariance GMM in pure numpy.

Mirrors deployment/gmm.h function by function.  Every algorithmic choice
matches the C++ exactly so that training results are directly comparable:

  Initialisation  gmm_init()      Two random training samples as initial
                                  means; global per-dimension variance as
                                  initial sigma².  (sklearn uses k-means++.)

  E-step          e_step()        Log-space responsibilities with
                                  log-sum-exp normalisation.

  M-step          m_step()        Weighted mean / variance update with a
                                  collapse guard: if N_k < MIN_NK_FRAC × N,
                                  the component is reinitialised from a
                                  random training sample.

  Log-normaliser  update_lognorm  Precomputed after each M-step and cached
                                  in _lognorm to avoid redundant log/sum
                                  calls during scoring.

  Scoring         score_clip()    -max_k log N(x | μ_k, Σ_k) with NO
                                  mixing weights — Guan et al. (2023) Eq. 3.
                                  Differs from sklearn score_samples() which
                                  includes pi_k in the mixture LL.

Node Learning role
------------------
In the Node Learning framework (Kanjo & Aslanov, 2026, §3, eq. 1-2), DiagGMM
is the learnable state θ_i of a single autonomous node.  fit() implements the
local adaptation rule U_i(θ_i, D_i, c_i) — it runs entirely on local data
without any external communication, as required by the single-node learning
definition.  score_samples() is the inference function that produces the per-clip
NLL values later used either for standalone detection or as inputs to the
NodeLearning merge operator.
"""

import numpy as np

from gmm.config import (
    EM_TOL,
    MAX_EM_ITER,
    MIN_NK_FRAC,
    N_COMPONENTS,
    SEED,
    VARIANCE_FLOOR,
)

# log(2π) — matches LOG2PI = 1.8378770664f in deployment/gmm.h
_LOG2PI: float = float(np.log(2.0 * np.pi))


class DiagGMM:
    """Diagonal-covariance Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of mixture components.  Default: gmm/config.N_COMPONENTS.
    max_iter : int
        Maximum EM iterations.  Default: gmm/config.MAX_EM_ITER.
    tol : float
        Convergence tolerance on the change in mean log-likelihood between
        successive iterations.  Default: gmm/config.EM_TOL.
    variance_floor : float
        Minimum per-dimension variance — prevents degenerate components.
        Default: gmm/config.VARIANCE_FLOOR.
    min_nk_frac : float
        Collapse threshold: a component whose effective count falls below
        min_nk_frac × N is reinitialised.  Default: gmm/config.MIN_NK_FRAC.
    seed : int
        Random seed for initialisation.  Default: gmm/config.SEED.

    Attributes (set after fit())
    ----------------------------
    mu_      : np.ndarray, shape (K, D)  — component means
    sigma2_  : np.ndarray, shape (K, D)  — diagonal variances
    pi_      : np.ndarray, shape (K,)    — mixing weights
    _lognorm : np.ndarray, shape (K,)    — cached log-normaliser
    """

    def __init__(
        self,
        n_components:   int   = N_COMPONENTS,
        max_iter:       int   = MAX_EM_ITER,
        tol:            float = EM_TOL,
        variance_floor: float = VARIANCE_FLOOR,
        min_nk_frac:    float = MIN_NK_FRAC,
        seed:           int   = SEED,
    ) -> None:
        self.n_components   = n_components
        self.max_iter       = max_iter
        self.tol            = tol
        self.variance_floor = variance_floor
        self.min_nk_frac    = min_nk_frac
        self.seed           = seed

        self.mu_     : np.ndarray | None = None
        self.sigma2_ : np.ndarray | None = None
        self.pi_     : np.ndarray | None = None
        self._lognorm: np.ndarray | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_lognorm(self) -> None:
        """Cache lognorm[k] = -0.5 * (D*log(2π) + Σ_d log(σ²_k[d])).

        Mirrors update_lognorm() in deployment/gmm.h.  Called after
        initialisation and after every M-step.
        """
        D = self.mu_.shape[1]
        self._lognorm = -0.5 * (D * _LOG2PI + np.sum(np.log(self.sigma2_), axis=1))

    def _log_component_prob(self, X: np.ndarray) -> np.ndarray:
        """Per-component log N(x | μ_k, Σ_k) for every sample, shape (N, K).

        Excludes mixing weights — equivalent to gm_lognorm[k] + quad(x, k)
        in deployment/gmm.h.  Used in both the E-step (where log pi is added
        separately) and scoring (where pi is deliberately omitted per Guan
        et al. Eq. 3).

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        log_probs : np.ndarray, shape (N, K)
        """
        # diff[n, k, d] = x[n, d] - mu[k, d]
        diff = X[:, None, :] - self.mu_[None, :, :]          # (N, K, D)
        # quad[n, k]   = -0.5 * Σ_d diff² / σ²
        quad = -0.5 * np.sum(diff ** 2 / self.sigma2_[None, :, :], axis=2)  # (N, K)
        return self._lognorm[None, :] + quad                  # (N, K)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Compute responsibilities r[n, k], shape (N, K).

        Mirrors e_step() in deployment/gmm.h: log-responsibilities are
        computed, then normalised via log-sum-exp for numerical stability.
        """
        lp  = np.log(self.pi_)[None, :] + self._log_component_prob(X)  # (N, K)
        mx  = lp.max(axis=1, keepdims=True)
        lse = mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True))
        return np.exp(lp - lse)                               # (N, K)

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Update mu_, sigma2_, pi_ from responsibilities.

        Mirrors m_step() in deployment/gmm.h, including the collapse guard:
        if N_k < min_nk_frac × N, the component is reinitialised from a
        random training sample rather than collapsing to a degenerate spike.

        Parameters
        ----------
        X    : np.ndarray, shape (N, D)
        resp : np.ndarray, shape (N, K)
        """
        N, _ = X.shape
        Nk   = resp.sum(axis=0)   # (K,)

        for k in range(self.n_components):
            if Nk[k] < self.min_nk_frac * N:
                # Collapse guard — mirrors the equivalent block in m_step().
                idx              = int(self._rng.integers(N))
                self.mu_[k]      = X[idx].copy()
                self.sigma2_[k]  = np.ones(X.shape[1], dtype=np.float32)
                self.pi_[k]      = 1.0 / self.n_components
                continue

            r_k             = resp[:, k:k + 1]              # (N, 1)
            self.mu_[k]     = (r_k * X).sum(axis=0) / Nk[k]
            diff            = X - self.mu_[k]               # (N, D)
            self.sigma2_[k] = np.maximum(
                (r_k * diff ** 2).sum(axis=0) / Nk[k],
                self.variance_floor,
            )
            self.pi_[k]     = float(Nk[k] / N)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "DiagGMM":
        """Fit the GMM on feature matrix X via EM.

        Mirrors gmm_init() + fit_gmm() in deployment/gmm.h.

        Initialisation:
          * Means   — two distinct random training samples (gmm_init rand()%N).
          * Sigma²  — global per-dimension variance across all samples.
          * Pi      — uniform 1/K.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix; one row per training clip.

        Returns
        -------
        self : DiagGMM
        """
        N, D     = X.shape
        self._rng = np.random.default_rng(self.seed)

        # Initialise means from two distinct random samples (mirrors gmm_init).
        idx          = self._rng.choice(N, size=self.n_components, replace=False)
        self.mu_     = X[idx].astype(np.float32).copy()

        # Initial sigma²: global per-dimension variance (mirrors gmm_init).
        global_var   = X.var(axis=0).astype(np.float32)
        self.sigma2_ = np.tile(
            np.maximum(global_var, self.variance_floor),
            (self.n_components, 1),
        )

        self.pi_     = np.full(self.n_components, 1.0 / self.n_components,
                               dtype=np.float32)
        self._update_lognorm()

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            resp    = self._e_step(X)
            self._m_step(X, resp)
            self._update_lognorm()

            ll = self.mean_log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """-max_k log N(x | μ_k, Σ_k) per sample, shape (N,).

        Mirrors score_clip() in deployment/gmm.h exactly.  Mixing weights
        are deliberately excluded — this implements Guan et al. (2023) Eq. 3
        and differs from sklearn's score_samples() which returns the full
        mixture log-likelihood including pi_k.

        Higher values indicate greater deviation from normal.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        nlls : np.ndarray, shape (N,)
        """
        log_probs = self._log_component_prob(X)   # (N, K) — no pi
        return -log_probs.max(axis=1).astype(np.float32)

    def mean_log_likelihood(self, X: np.ndarray) -> float:
        """Full mixture mean log-likelihood (used for convergence check).

        Includes mixing weights — matches the LL computation in fit_gmm()
        in deployment/gmm.h.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        mean_ll : float
        """
        lp  = np.log(self.pi_)[None, :] + self._log_component_prob(X)  # (N, K)
        mx  = lp.max(axis=1)
        ll  = mx + np.log(np.exp(lp - mx[:, None]).sum(axis=1))
        return float(ll.mean())
