# gmm/: TWFR-GMM Detection Primitives

Library of single-node detection primitives shared by `simulation/` (Python
N-node evaluation framework) and `deployment/` (C++ 2-node Arduino port).
This package is not an entry point; orchestration lives in `simulation/`.

```
config.py       all tunable constants (deployment mirror + experiment knobs)
features.py     load_log_mel, gwrp_weights, extract_feature_r
gmm.py          DiagGMM (hand-rolled numpy, mirrors deployment/gmm.h)
detector.py     GMMDetector: fit + calibrate + score + save/load
```

## Pipeline in one paragraph

Each normal clip is loaded as a log-mel spectrogram and compressed to one
`(n_mels,)` feature vector by GWRP pooling across the time axis. For each
node, the r value is chosen by an r-search over `R_CANDIDATES`; multi-node
fusion at `simulation/node/group.py` adds a greedy "no-sharing" rule so
peers prefer different r values. Each node fits a 2-component diagonal GMM
via EM on 50 fit clips and calibrates the threshold to the max NLL on 10
held-out val clips. The new clip's anomaly score is the negative
log-likelihood under its best component (Guan et al. Eq. 3). A
Page-Hinkley CUSUM drives the alarm.

For multi-node fusion, each node z-normalises its own NLL with its
stored `mu_val` and `sigma_val`, and the scores are fused with
`w_i = softmax(-sigma_val_i / T)` (lower val NLL std = more consistent
node = higher weight). A fused CUSUM is calibrated on the fused val
z-scores using the same max-of-val rule. The math lives in
`simulation/node/group.py` for the Python side and
`deployment/node_learning.h` for the on-device side.

## Running

This package is imported from `simulation/`; there is no standalone CLI.
To run the full pipeline:

```bash
python -m simulation.run_simulation
```

See [simulation/README.md](../simulation/README.md) and
[simulation/configs/README.md](../simulation/configs/README.md) for the
configuration surface (channel selection, SNR, fusion temperature,
shuffle mode, etc.).

## Key design choices

**Why diagonal covariance?** 50 training clips against `n_mels`
dimensions is too few to estimate a full covariance matrix. Diagonal
covariance (`n_mels` parameters per component) fits comfortably.

**Why 50 fit / 10 val?** The GMM attains artificially high likelihood on
its own training clips. Calibrating the threshold on those clips would
place it in a regime test clips never reach. The 10 held-out clips give
a calibration distribution that matches test data.

**Why fuse on sigma rather than mu?** In practice `sigma_val` is a more
reliable fit-quality signal than `mu_val`: a lower val NLL std means the
GMM generalises consistently across the held-out normal clips. The C++
port (`deployment/node_learning.h`) uses sigma as well.

**Why max-component NLL (Guan et al. Eq. 3)?** For a bimodal normal
distribution, a clip close to either cluster should score low. The
max-component formulation implements this correctly; the standard
mixture log-likelihood would down-weight one cluster by its `pi_k`.

**Why the greedy diversity rule?** Earlier versions used a hard
`|r_B - r_A| >= 0.25` margin; that doesn't scale to N > 2 nodes against
a 4-element `R_CANDIDATES` grid. The greedy rule (each node picks the
best r not already claimed by a peer on the same machine) generalises
cleanly and degrades to "share an r" only when N > len(R_CANDIDATES).
