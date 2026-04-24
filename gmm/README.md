# gmm/ — TWFR-GMM Anomaly Detection Pipeline

Python implementation of the two-node TWFR-GMM anomaly detector. The
entry point is [train.py](train.py); everything else is a self-contained
module with a single responsibility.

```
config.py         all tunable constants (deployment mirror + experiment knobs)
features.py       load_log_mel, gwrp_weights, extract_feature_r
gmm.py            DiagGMM (hand-rolled numpy, mirrors deployment/gmm.h)
detector.py       GMMDetector: fit + calibrate + score + save/load
node_learning.py  NodeLearning: two-node score fusion
evaluate.py       per-machine monitoring simulation
plot.py           per-machine timeline scatter plot
train.py          CLI entry point; orchestrates all 16 machines
```

## Pipeline in one paragraph

Each normal clip is loaded as a log-mel spectrogram and compressed to one
`(n_mels,)` feature vector by GWRP pooling across the time axis. For each
node, the r value is chosen by an r-search over `R_CANDIDATES`; Node B
additionally constrains its candidates by `|r - r_A| >= diversity_margin`.
Each node fits a 2-component diagonal GMM via EM on 50 fit clips and
calibrates a 95th-percentile threshold on 10 held-out val clips. The new
clip's anomaly score is the negative log-likelihood under its best
component (Guan et al. Eq. 3). A Page-Hinkley CUSUM drives the alarm.

For the two-node system, each node z-normalises its own NLL with its
stored `mu_val` and `sigma_val`, and the scores are fused with
`w_i = softmax(-sigma_val_i / T)` (lower val NLL std = more consistent
node = higher weight). A fused CUSUM is calibrated on the fused val
z-scores using the same 95th-percentile rule.

## Running

```bash
python gmm/train.py --dataset neg6db         # default
python gmm/train.py --dataset 0db --verbose
python gmm/train.py --dataset 6db --diversity-margin 0.5 --temperature 1
```

Full flag list:

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `neg6db` | `neg6db`, `0db`, or `6db` |
| `--splits` | inferred | Override path to splits JSON |
| `--out-dir` | inferred | Override output root |
| `--n-mels` | 64 | Mel bin count (output dir is suffixed when non-default) |
| `--diversity-margin` | 0.25 | Minimum \|r_B - r_A\| required |
| `--temperature` | 100 | Softmax temperature for fusion weights |
| `--mic-a`, `--mic-b` | 0, 1 | Microphone channel per node |
| `--verbose` | off | Print per-machine calibration parameters |

## Outputs

For each dataset, `gmm/outputs/{dataset}/` receives:

```
node_a/{mtype}_{mid}.pkl        fitted GMMDetector artefact  (x16)
node_a/{mtype}_{mid}.png        timeline plot                (x16)
node_a/results.yaml             per-machine metrics
node_b/                         (same structure)
node_learning/*.png, results.yaml
comparison.yaml                 3-way side-by-side summary
```

Each timeline plot shows per-clip score over wall-clock time with anomaly
injection windows shaded and detection events annotated.

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
uses sigma as well.

**Why max-component NLL (Guan et al. Eq. 3)?** For a bimodal normal
distribution, a clip close to either cluster should score low. The
max-component formulation implements this correctly; the standard
mixture log-likelihood would down-weight one cluster by its `pi_k`.
