"""Node — one physical detector (one mic channel, one GMMDetector).

A Node is a thin data container:
  - identity: (machine_type, machine_id, channel, node_id)
  - state:    a fitted gmm.detector.GMMDetector (owned)
  - IO:       calibrate(fit_paths, val_paths) and score(wav_path, label)

No preprocessor wrapper, no embedder, no separator abstraction.  load_log_mel
and GMMDetector are called directly.  Everything the node "does" is a straight
passthrough to gmm/.
"""

from dataclasses import dataclass, field

import numpy as np

from gmm.config   import R_CANDIDATES
from gmm.detector import GMMDetector
from gmm.features import load_log_mel


@dataclass
class Node:
    # Identity
    node_id:       str
    machine_type:  str
    machine_id:    str
    channel:       int        # 0..7 on MIMII

    # GMM / detector hyperparameters (passthrough to GMMDetector)
    n_mels:        int
    n_components:  int   = 2
    cusum_h_sigma: float = 20.0
    cusum_h_floor: float = 1.0
    seed:          int   = 42

    # Detection state post-processing
    manual_reset:  bool  = False

    # Populated by calibrate()
    detector: GMMDetector | None = None
    _state_held: bool = field(default=False, repr=False)

    # Per-timestep log (used by reporting / plotting)
    scores:  list[float] = field(default_factory=list)
    labels:  list[int]   = field(default_factory=list)
    cusum_S: list[float] = field(default_factory=list)
    alarms:  list[bool]  = field(default_factory=list)
    state:   list[int]   = field(default_factory=list)

    # ── Calibration ───────────────────────────────────────────────────────
    def calibrate(
        self,
        fit_paths:  list[str],
        val_paths:  list[str],
        claimed_rs: set[float] | None = None,
    ) -> None:
        """Per-node r-search with optional greedy diversity constraint.

        Fits a detector at every r in R_CANDIDATES. If ``claimed_rs`` is given,
        picks the best detector whose r is NOT already claimed by another node
        in the same machine group. Falls back to the globally best detector if
        every candidate is claimed (relevant only when N > len(R_CANDIDATES)).
        """
        fit_mels = [load_log_mel(p, n_mels=self.n_mels, channel=self.channel)
                    for p in fit_paths]
        val_mels = [load_log_mel(p, n_mels=self.n_mels, channel=self.channel)
                    for p in val_paths]

        best:           GMMDetector | None = None
        best_unclaimed: GMMDetector | None = None
        for r in R_CANDIDATES:
            det = GMMDetector(
                r             = r,
                n_components  = self.n_components,
                cusum_h_sigma = self.cusum_h_sigma,
                cusum_h_floor = self.cusum_h_floor,
                n_mels        = self.n_mels,
                seed          = self.seed,
            )
            det.fit(fit_mels, val_mels)
            if best is None or det.mu_val_ < best.mu_val_:
                best = det
            if claimed_rs is None or r not in claimed_rs:
                if best_unclaimed is None or det.mu_val_ < best_unclaimed.mu_val_:
                    best_unclaimed = det
        self.detector = best_unclaimed if best_unclaimed is not None else best

    # ── Scoring ───────────────────────────────────────────────────────────
    def score(self, wav_path: str, label: int) -> tuple[float, bool]:
        """Load clip, compute NLL, advance CUSUM, append traces."""
        log_mel = load_log_mel(wav_path, n_mels=self.n_mels, channel=self.channel)
        nll   = self.detector.score(log_mel)

        d     = self.detector
        new_S = max(0.0, d._cusum_S + nll - d.cusum_k_)
        alarm = new_S >= d.cusum_h_
        d._cusum_S = 0.0 if alarm else new_S

        if self.manual_reset:
            self._state_held |= alarm
        current_state = 1 if (self._state_held if self.manual_reset else alarm) else 0

        self.scores.append(nll)
        self.labels.append(label)
        self.cusum_S.append(new_S)
        self.alarms.append(alarm)
        self.state.append(current_state)
        return nll, alarm

    def cusum_reset(self) -> None:
        self.detector.cusum_reset()

    def state_reset(self) -> None:
        """Clear the latched state (simulates engineer acknowledging)."""
        self._state_held = False

    # ── Convenience passthroughs (used by Group + reporting) ──────────────
    @property
    def r(self) -> float:             return self.detector.r_
    @property
    def k(self) -> float:             return self.detector.cusum_k_
    @property
    def h(self) -> float:             return self.detector.cusum_h_
    @property
    def mu_val(self) -> float:        return self.detector.mu_val_
    @property
    def sigma_val(self) -> float:     return self.detector.sigma_val_
    @property
    def val_nlls(self) -> np.ndarray: return self.detector.val_nlls_
