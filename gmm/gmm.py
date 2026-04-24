"""Hand-rolled diagonal-covariance Gaussian Mixture Model in pure numpy."""

import numpy as np

from gmm.config import (
    EM_TOL,
    MAX_EM_ITER,
    MIN_NK_FRAC,
    N_COMPONENTS,
    SEED,
    VARIANCE_FLOOR,
)

_LOG2PI: float = float(np.log(2.0 * np.pi))


class DiagGMM:
    """Diagonal-covariance GMM fit by EM.

    Scoring returns ``-max_k log N(x | mu_k, Sigma_k)`` without mixing weights,
    following Guan et al. (2023) Eq. 3.

    Attributes set by ``fit``:
        mu_      (K, D)  component means
        sigma2_  (K, D)  diagonal variances
        pi_      (K,)    mixing weights
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

    def _update_lognorm(self) -> None:
        D = self.mu_.shape[1]
        self._lognorm = -0.5 * (D * _LOG2PI + np.sum(np.log(self.sigma2_), axis=1))

    def _log_component_prob(self, X: np.ndarray) -> np.ndarray:
        """Per-component ``log N(x | mu_k, Sigma_k)``, shape (N, K). No ``pi_k``."""
        diff = X[:, None, :] - self.mu_[None, :, :]
        quad = -0.5 * np.sum(diff ** 2 / self.sigma2_[None, :, :], axis=2)
        return self._lognorm[None, :] + quad

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        lp  = np.log(self.pi_)[None, :] + self._log_component_prob(X)
        mx  = lp.max(axis=1, keepdims=True)
        lse = mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True))
        return np.exp(lp - lse)

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        N, _ = X.shape
        Nk   = resp.sum(axis=0)

        for k in range(self.n_components):
            if Nk[k] < self.min_nk_frac * N:
                # Collapsed component: reinitialise from a random sample.
                idx              = int(self._rng.integers(N))
                self.mu_[k]      = X[idx].copy()
                self.sigma2_[k]  = np.ones(X.shape[1], dtype=np.float32)
                self.pi_[k]      = 1.0 / self.n_components
                continue

            r_k             = resp[:, k:k + 1]
            self.mu_[k]     = (r_k * X).sum(axis=0) / Nk[k]
            diff            = X - self.mu_[k]
            self.sigma2_[k] = np.maximum(
                (r_k * diff ** 2).sum(axis=0) / Nk[k],
                self.variance_floor,
            )
            self.pi_[k]     = float(Nk[k] / N)

    def fit(self, X: np.ndarray) -> "DiagGMM":
        """Fit on feature matrix ``X`` (N, D) and return self."""
        N, D      = X.shape
        self._rng = np.random.default_rng(self.seed)

        idx          = self._rng.choice(N, size=self.n_components, replace=False)
        self.mu_     = X[idx].astype(np.float32).copy()

        global_var   = X.var(axis=0).astype(np.float32)
        self.sigma2_ = np.tile(
            np.maximum(global_var, self.variance_floor),
            (self.n_components, 1),
        )

        self.pi_ = np.full(self.n_components, 1.0 / self.n_components, dtype=np.float32)
        self._update_lognorm()

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            self._update_lognorm()

            ll = self.mean_log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Negative max-component log-likelihood per sample (higher = more anomalous)."""
        log_probs = self._log_component_prob(X)
        return -log_probs.max(axis=1).astype(np.float32)

    def mean_log_likelihood(self, X: np.ndarray) -> float:
        """Full mixture mean log-likelihood (used for EM convergence)."""
        lp = np.log(self.pi_)[None, :] + self._log_component_prob(X)
        mx = lp.max(axis=1)
        ll = mx + np.log(np.exp(lp - mx[:, None]).sum(axis=1))
        return float(ll.mean())
