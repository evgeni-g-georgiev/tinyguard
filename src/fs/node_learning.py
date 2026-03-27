"""
Node learning: Fisher-weighted scoring calibrated by peer encounters.

Philosophy:
  - SVDD trains f_s weights and centroid (the proven 0.83 baseline)
  - f_s weights NEVER change after SVDD training
  - Node learning only changes how we SCORE — not how we project
  - Scoring upgrades from isotropic to Fisher-weighted Mahalanobis

Scoring progression:
  1. No peers:   score = Σ_d (f_s(x)_d - c_d)² / σ²_d       (Mahalanobis)
  2. + Friend:   score = Σ_d (f_s(x)_d - c_d)² / σ²_pooled_d (better variance)
  3. + Foe:      score = Σ_d w_d * (f_s(x)_d - c_d)²          (Fisher-weighted)
     where w_d = (1 + β·foe_dist_d) / (σ²_d + ε)

Fisher's discriminant ratio per dimension:
  w_d ∝ between-class variance / within-class variance
  - between-class = how far foe prototypes project from my centroid on dim d
  - within-class  = how much my normal clips spread on dim d

On-device cost: 8 extra floats (dimension weights) = 32 bytes.
BLE exchange: prototypes (512 bytes) + variance vector (32 bytes) = 544 bytes.
"""

import torch
import numpy as np

from src.fs.model import FsSeparator


# ── Prototype memory (streaming k-means in f_c space) ──

class PrototypeMemory:
    """Prototype storage in f_c space (16D) with streaming k-means."""

    def __init__(self, max_normal: int = 8, max_foe: int = 8, dim: int = 16,
                 tau_spawn: float = 2.0, beta: float = 0.9):
        self.max_normal = max_normal
        self.max_foe = max_foe
        self.dim = dim
        self.tau_spawn = tau_spawn
        self.beta = beta

        self.P_normal = np.zeros((0, dim), dtype=np.float32)
        self.P_foe = np.zeros((0, dim), dtype=np.float32)

    def update_normal(self, embedding: np.ndarray):
        """Streaming k-means: EMA update if close, spawn if far and slots free."""
        embedding = embedding.reshape(1, -1)
        if len(self.P_normal) == 0:
            self.P_normal = embedding.copy()
            return
        dists = np.linalg.norm(self.P_normal - embedding, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < self.tau_spawn:
            self.P_normal[idx] = self.beta * self.P_normal[idx] + (1 - self.beta) * embedding[0]
        elif len(self.P_normal) < self.max_normal:
            self.P_normal = np.vstack([self.P_normal, embedding])
        else:
            self.P_normal[idx] = self.beta * self.P_normal[idx] + (1 - self.beta) * embedding[0]

    def add_foe(self, prototypes: np.ndarray):
        """Store foe prototypes. FIFO if full."""
        if len(prototypes) == 0:
            return
        self.P_foe = (np.vstack([self.P_foe, prototypes])
                      if len(self.P_foe) > 0 else prototypes.copy())
        if len(self.P_foe) > self.max_foe:
            self.P_foe = self.P_foe[-self.max_foe:]

    def merge_friend(self, peer_prototypes: np.ndarray, alpha: float = 0.9):
        """EMA merge friend prototypes into P_normal."""
        for proto in peer_prototypes:
            if len(self.P_normal) == 0:
                self.P_normal = proto.reshape(1, -1).copy()
                continue
            dists = np.linalg.norm(self.P_normal - proto.reshape(1, -1), axis=1)
            idx = np.argmin(dists)
            self.P_normal[idx] = alpha * self.P_normal[idx] + (1 - alpha) * proto

    @property
    def has_foe(self) -> bool:
        return len(self.P_foe) > 0


# ── Gating ──

def gate_peer(own_P_normal: np.ndarray, peer_P_normal: np.ndarray,
              R_gate: float = 3.0) -> str:
    """Friend/foe gating: min prototype distance < R_gate -> friend, else foe."""
    if len(own_P_normal) == 0 or len(peer_P_normal) == 0:
        return "foe"
    dists = np.linalg.norm(
        own_P_normal[:, None, :] - peer_P_normal[None, :, :], axis=2
    )
    return "friend" if dists.min() < R_gate else "foe"


# ── Fisher-weighted node ──

class FisherNode:
    """A TinyML node: SVDD base + Fisher-weighted scoring from peer encounters.

    Key invariant: f_s weights are NEVER modified after SVDD training.
    Node learning only changes the per-dimension scoring weights.

    Lifecycle:
      1. ingest_normal_clips()  -> streaming k-means for prototypes
      2. train_svdd()           -> standard SVDD (weights fixed after this)
      3. compute_own_variance() -> per-dim variance of projected normal clips
      4. receive_peer()         -> gating -> friend or foe actions
      5. score()                -> Fisher-weighted Mahalanobis distance
    """

    def __init__(self, node_id: str, machine_key: str,
                 input_dim: int = 16, output_dim: int = 8,
                 max_prototypes: int = 8, tau_spawn: float = 2.0,
                 seed: int = 42):
        self.node_id = node_id
        self.machine_key = machine_key
        self.seed = seed
        self.output_dim = output_dim

        torch.manual_seed(seed)
        self.model = FsSeparator(input_dim, output_dim)
        self.memory = PrototypeMemory(
            max_normal=max_prototypes, max_foe=max_prototypes,
            dim=input_dim, tau_spawn=tau_spawn,
        )
        self._normal_data = None

        # Scoring weights — initialised after SVDD training
        self.own_variance = np.ones(output_dim, dtype=np.float32)
        self.dim_weights = np.ones(output_dim, dtype=np.float32)
        self._friend_variances = []  # collected from friends for pooling
        self._foe_dists = []         # collected from foes for Fisher weighting

    def ingest_normal_clips(self, normal_embeddings: np.ndarray):
        """Build prototypes and store raw data for SVDD training."""
        self._normal_data = normal_embeddings.copy()
        for emb in normal_embeddings:
            self.memory.update_normal(emb)

    def train_svdd(self, epochs: int = 100, lr: float = 0.01,
                   weight_decay: float = 1e-4, batch_size: int = 64):
        """Standard SVDD training. After this, weights are FROZEN."""
        if self._normal_data is None:
            return
        train_t = torch.from_numpy(self._normal_data)
        self.model.init_centroid(train_t)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(train_t))
            for start in range(0, len(train_t), batch_size):
                batch = train_t[perm[start:start + batch_size]]
                proj = self.model(batch)
                loss = ((proj - self.model.centroid) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def compute_own_variance(self):
        """Compute per-dimension variance of projected normal clips.

        This is the 'within-class variance' for Mahalanobis scoring.
        On-device: computed once after SVDD training, stored as 8 floats.
        """
        if self._normal_data is None:
            return
        self.model.eval()
        with torch.no_grad():
            proj = self.model(torch.from_numpy(self._normal_data)).numpy()
        centroid = self.model.centroid.numpy()
        # Per-dimension variance around centroid
        self.own_variance = np.var(proj - centroid, axis=0).astype(np.float32)
        # Initialise weights as Mahalanobis (no peers yet)
        self._update_weights()

    def receive_peer(self, peer_node, R_gate: float = 3.0):
        """BLE encounter: gate, then act on friend/foe classification."""
        decision = gate_peer(self.memory.P_normal, peer_node.memory.P_normal, R_gate)

        if decision == "friend":
            # Merge prototypes
            self.memory.merge_friend(peer_node.memory.P_normal)
            # Collect friend's variance for pooling (32 bytes over BLE)
            self._friend_variances.append(peer_node.own_variance.copy())
        else:
            # Store foe prototypes
            self.memory.add_foe(peer_node.memory.P_normal)
            # Cross-project foe prototypes through MY f_s
            self.model.eval()
            with torch.no_grad():
                proj_foe = self.model(
                    torch.from_numpy(peer_node.memory.P_normal)
                ).numpy()
            centroid = self.model.centroid.numpy()
            # Per-dimension mean squared distance of foe from MY centroid
            foe_dist = np.mean((proj_foe - centroid) ** 2, axis=0)
            self._foe_dists.append(foe_dist)

        # Recompute weights with new peer information
        self._update_weights()
        return decision

    def _update_weights(self, beta: float = 1.0, epsilon: float = 1e-6):
        """Recompute Fisher-weighted scoring weights.

        w_d = (1 + beta * foe_dist_d) / (sigma^2_d + epsilon)

        - No peers:  w_d = 1 / sigma^2_d                    (Mahalanobis)
        - + Friend:  sigma^2_d pooled across friends         (better estimate)
        - + Foe:     numerator boosted by foe separation     (Fisher ratio)
        """
        # Within-class variance: pool own + friend variances
        if self._friend_variances:
            all_vars = [self.own_variance] + self._friend_variances
            sigma_sq = np.mean(all_vars, axis=0)
        else:
            sigma_sq = self.own_variance.copy()

        # Between-class signal: average foe distances (if any)
        if self._foe_dists:
            foe_dist = np.mean(self._foe_dists, axis=0)
            self.dim_weights = (1.0 + beta * foe_dist) / (sigma_sq + epsilon)
        else:
            # Pure Mahalanobis
            self.dim_weights = 1.0 / (sigma_sq + epsilon)

        # Normalise so weights sum to output_dim (keeps score scale stable)
        self.dim_weights = (self.dim_weights / self.dim_weights.sum()
                            * self.output_dim).astype(np.float32)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Fisher-weighted anomaly score: Σ_d w_d * (f_s(x)_d - c_d)².

        Falls back to Mahalanobis (w_d = 1/σ²_d) if no foe peers,
        and to isotropic SVDD if compute_own_variance() wasn't called.
        """
        self.model.eval()
        with torch.no_grad():
            proj = self.model(torch.from_numpy(embeddings)).numpy()
        centroid = self.model.centroid.numpy()
        sq_diff = (proj - centroid) ** 2  # (N, output_dim)
        return (sq_diff * self.dim_weights).sum(axis=1)

    def score_isotropic(self, embeddings: np.ndarray) -> np.ndarray:
        """Plain SVDD score (no weighting) — for baseline comparison."""
        self.model.eval()
        with torch.no_grad():
            return self.model.anomaly_score(
                torch.from_numpy(embeddings)
            ).numpy()
