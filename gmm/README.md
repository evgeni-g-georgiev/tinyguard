# TWFR-GMM Anomalous Sound Detection

An alternative anomaly detection pipeline for the MIMII dataset, based on the
paper:

> Guan et al., *"Time-Weighted Frequency Domain Audio Representation with GMM
> Estimator for Anomalous Sound Detection"*, arXiv:2305.03328 (2023).

This module replaces the CNN encoder + Deep SVDD pipeline (`separator/`,
`inference/`) with a fully signal-processing-based approach: no neural network
is trained or required. The same 60-clip training budget and 3-round monitoring
evaluation structure are used, so results are directly comparable.

---

## Overview

The existing pipeline encodes each audio sub-window using a frozen CNN
(distilled from YAMNet) and learns a compact representation with Deep SVDD.
Its bottleneck is the CNN encoder: trained on general-purpose audio, it
compresses spectral information relevant to broad sound categories, discarding
the fine-grained machine-state variation needed to separate normal from
anomalous industrial sounds.

TWFR-GMM bypasses the encoder entirely. It computes a compact feature vector
directly from the raw log-mel spectrogram using a deterministic, parameterless
pooling operation, then fits a Gaussian Mixture Model on normal training clips.
Anomaly scores are negative log-likelihoods under the fitted GMM. There are no
neural network weights to train or store.

---

## Methodology

### Step 1 — Audio representation: log-mel spectrogram

Each 10-second clip is loaded at 16 kHz and converted to a log-mel spectrogram
with the following parameters (matching the paper):

| Parameter | Value |
|---|---|
| FFT window | 1024 samples |
| Hop length | 512 samples (50% overlap) |
| Mel filter banks | 128 |
| Frequency range | librosa defaults (0 Hz – 8 kHz) |

The result is a matrix of shape **(128 mel bins × ~311 time frames)** for each
10-second clip. The full clip is processed in one pass — unlike the SVDD
pipeline, which operates on 0.975-second sub-windows.

### Step 2 — TWFR feature extraction

The Time-Weighted Frequency Domain Representation (TWFR) compresses the
(128 × 311) spectrogram into a single **128-dimensional feature vector**, one
scalar per mel frequency bin.

For each frequency bin independently:
1. The 311 time-frame values are **sorted in descending order** (highest energy
   first), discarding temporal order.
2. The sorted sequence is combined via a **weighted dot product** with the
   Global Weighted Ranking Pooling (GWRP) weight vector **P(r)**:

   ```
   P(r)[i] = r^i / sum(r^j for j in 0..T-1)
   ```

   where `r ∈ [0, 1]` is a decay parameter and index 0 corresponds to the
   highest-energy frame.

The weight vector P(r) interpolates between two extremes:
- **r = 0** → only the maximum energy frame contributes (max pooling)
- **r = 1** → all frames contribute equally (mean pooling)
- **Intermediate r** → more weight on high-energy frames, less on low-energy
  frames, blending stationary and transient information

This generalisation matters because different machine types have different
temporal characteristics: fans tend to be stationary (r → 1 suits them),
while valves have transient click events (lower r suits them better).

The TWFR operation is implemented in `gmm/features.py::twfr_feature()`.

### Step 3 — Self-supervised r selection

The decay parameter `r` is selected **per machine, without anomaly labels**,
by the following procedure:

For each candidate value in the grid `[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]`:
1. Extract TWFR features from all training clips using that r.
2. Fit a GMM on the resulting feature matrix.
3. Record the mean log-likelihood of the training features under the GMM.

The r that produces the **highest mean log-likelihood** is selected. Intuitively,
when r matches a machine's temporal structure, the resulting feature vectors
cluster more tightly, and the GMM achieves a higher-likelihood fit. No anomaly
clips are needed for this search.

This procedure is implemented in `gmm/detector.py::GMMDetector.search_r()`.

### Step 4 — GMM fitting

With the optimal r fixed, TWFR features are extracted from the **50 training
clips** (see training split below) and a Gaussian Mixture Model is fit using
the EM algorithm via scikit-learn's `GaussianMixture`.

Default configuration:
- **2 components** (captures bimodal normal distributions, e.g. a machine with
  two distinct operating modes)
- **Diagonal covariance** (each component has an independent variance per
  dimension; full covariance would require estimating 128×129/2 = 8,256
  parameters per component from only 50 samples — massively underdetermined)

### Step 5 — Anomaly scoring

Given a new clip, the anomaly score is defined following Equation 3 of the
paper:

```
A(X) = -max_{k in [1, K]} log N(R(X) | μ_k, Σ_k)
```

That is, the score is the **negative log-likelihood under the best-fitting
GMM component**. This differs from the standard mixture log-likelihood
(`log Σ_k π_k N(x|μ_k, Σ_k)`): it rewards proximity to the *nearest* cluster
rather than the weighted average of all clusters. For a bimodal normal
distribution, a clip near either cluster should score low (not anomalous),
which the max-component formulation handles correctly.

Higher scores indicate greater deviation from the learned normal distribution.

### Step 6 — Threshold calibration (training split)

A critical implementation detail: the threshold must **not** be computed on the
same clips used to fit the GMM. Because the GMM is optimised to maximise
likelihood on its training data, those clips achieve artificially high
log-likelihood (very negative NLL), placing the threshold in a regime that test
clips never reach, causing a 100% false positive rate.

The 60 available training clips are therefore split:
- **50 clips** — used to fit the GMM and perform the r search
- **10 clips** — held out and used only to calibrate the detection threshold

The threshold is the **95th percentile of the NLL scores on the 10 held-out
clips**. These clips were not seen during fitting, so their scores reflect
genuine generalisation to unseen normal audio.

### Step 7 — Adaptive rolling-window detection

Individual per-clip NLL scores have inherent variance: occasionally a normal
clip scores high (an isolated spike), and occasionally an anomalous clip scores
near-normal (a moment within the recording where the fault is not acoustically
manifest). Evaluating clips one at a time against a fixed threshold therefore
produces both false positives and missed detections even when the distributions
are well-separated.

Detection decisions are instead made on a **rolling window mean** of the last
`ROLLING_WINDOW = 5` consecutive clip scores:

```
window_mean[i] = mean(scores[max(0, i-4) : i+1])
```

A single outlier clip shifts the window mean by at most 1/5 of its excess. A
sustained anomaly period drives the window mean consistently above the
threshold.

**Adaptive per-round threshold**

The detection threshold is re-calibrated for each monitoring round from the
**normal window that precedes it**. The deployment sequence is always: observe
normal operation first, then switch to potential anomaly monitoring. The
rolling means computed on those 30 normal clips are recorded; the detection
threshold for the subsequent anomaly window is set as:

```
threshold_round = max(rolling_means_on_normal_window) × 1.5
```

This compensates for machine-to-machine and session-to-session variation in
score magnitude that a single training-time threshold cannot capture. If the
normal window is very quiet (all clips score well below the training threshold),
the training threshold serves as a floor. If the normal window shows elevated
variability, the threshold rises accordingly, preventing false alarms driven
by that variability pattern.

**False alarm counting**

False alarms are counted as distinct above-threshold *excursion events* in the
normal monitoring window (using the same `threshold_round`): a contiguous block
of clips whose rolling mean stays above the threshold counts as one event,
regardless of how many clips it spans.

---

## Module structure

```
gmm/
├── features.py     Signal processing only: load_log_mel(), gwrp_weights(), twfr_feature()
├── detector.py     GMMDetector class: r search, GMM fit, scoring, save/load
├── evaluate.py     Per-machine evaluation: rolling-window detection, adaptive threshold
├── plot.py         Timeline scatter plots (identical visual style to inference/run.py)
├── train.py        CLI entry point: orchestrates all 16 machines end-to-end
└── outputs/
    └── gmm/        Created at runtime: .pkl artefacts, .png plots, results.yaml
```

Each module has a single responsibility:
- `features.py` — no ML, no sklearn, no plotting
- `detector.py` — no WAV loading, no argparse, no plotting
- `evaluate.py` — no matplotlib, no yaml
- `plot.py` — no repo-level imports (all data passed as arguments)
- `train.py` — orchestration only, no inline math or plotting code

---

## Prerequisites

The GMM pipeline shares the same data and preprocessing as the rest of the
repo. The following must exist before running:

1. **MIMII WAV files** at `data/mimii/` (fan, pump, slider, valve)
2. **Splits manifest** at `preprocessing/outputs/mimii_splits/splits.json`

If the splits manifest does not exist, generate it first:

```bash
python preprocessing/split_mimii.py
```

Python dependencies (all present in the shared environment):
`librosa`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`

---

## Running

All commands are run from the repository root.

### Full run — self-supervised r search (recommended)

```bash
python gmm/train.py
```

Trains and evaluates all 16 machines (4 types × 4 IDs). For each machine, the
r search tries 9 candidate values and selects the one that maximises GMM
log-likelihood on the training data. Use `--verbose` to print the per-r
log-likelihoods:

```bash
python gmm/train.py --verbose
```

### Fix r (ablation)

Skip the r search and use a fixed pooling strategy for all machines:

```bash
python gmm/train.py --r 1.0   # pure mean pooling
python gmm/train.py --r 0.0   # pure max pooling
```

### Other options

```bash
python gmm/train.py --n-components 1    # single Gaussian (no mixture)
python gmm/train.py --threshold-pct 90  # more aggressive threshold
python gmm/train.py --seed 0            # different GMM initialisation
python gmm/train.py --out-dir /tmp/test # custom output directory
```

All arguments with defaults:

| Argument | Default | Description |
|---|---|---|
| `--splits` | `preprocessing/outputs/mimii_splits/splits.json` | Path to splits manifest |
| `--out-dir` | `gmm/outputs/gmm` | Output directory |
| `--n-components` | `2` | Number of GMM components |
| `--r` | *(not set)* | Fix r; omit for self-supervised search |
| `--threshold-pct` | `95` | Percentile of val NLLs for threshold |
| `--seed` | `42` | Random seed for GMM EM initialisation |
| `--verbose` | *(flag)* | Print per-r log-likelihoods |

---

## Outputs

After a full run, the following files are written to `gmm/outputs/gmm/`:

| File | Description |
|---|---|
| `{mtype}_{mid}.pkl` | Fitted `GMMDetector` artefact (16 files) |
| `{mtype}_{mid}.png` | Per-machine timeline plot (16 files) |
| `results.yaml` | Aggregate metrics for all machines |

### Timeline plots

Each plot shows the anomaly score (NLL) for every scored clip against wall-clock
time. Normal clips are shown in blue, anomaly clips in orange. The shaded red
regions mark the anomaly injection windows. Detection events are annotated with
arrows showing the detection delay. The dashed red line is the training-time
detection threshold (the adaptive per-round threshold is not plotted, as it
varies per round).

### results.yaml schema

```yaml
fan/id_00:
  threshold: 170.3821    # 95th pct of held-out val NLLs
  r: 0.99                # selected GWRP decay parameter
  n_rounds: 3
  rounds:
    - round: 1
      n_false_pos: 0     # CUSUM false alarm events in normal window
      n_normal_clips: 30
      n_anom_clips: 30
      detected: true
      detection_delay_secs: 0
      detection_idx: 0
      threshold_round: 255.5  # adaptive threshold used for this round
    - round: 2
      ...
```

The schema is identical to `inference/outputs/inference/results.yaml` (produced
by the SVDD pipeline) with two additional fields per round: `threshold_round`
(adaptive detection threshold) and the top-level `r` field per machine.

---

## Comparing with the SVDD pipeline

After running both pipelines, their `results.yaml` files can be compared
directly — the round-level schema is shared. The SVDD pipeline results are at
`inference/outputs/inference/results.yaml`.

```bash
# Run SVDD pipeline (if not already done)
python separator/train.py
python inference/run.py

# Then compare results.yaml files side by side
```

---

## Key design decisions and their rationale

**Why diagonal covariance?**
With 50 training clips and 128-dimensional features, fitting a full covariance
matrix would require estimating 8,256 parameters per component — far more than
the number of training samples. Diagonal covariance (128 parameters per
component) is well-conditioned and appropriate for this regime.

**Why 50 fit / 10 val split?**
The GMM achieves artificially high log-likelihood on its own training clips. If
the threshold is computed from those clips, it is set in a regime that unseen
normal clips never reach, resulting in 100% false positives. The 10 held-out
clips are unseen by the GMM, so their NLL distribution matches that of test
data. This is the standard train/validation separation applied to threshold
calibration.

**Why rolling window detection instead of per-clip thresholding?**
Individual clip scores vary even within a stable period: a normal clip can
occasionally score high, and an anomalous clip can occasionally score near-
normal (if the fault does not manifest acoustically in that particular
10-second recording). The rolling mean over 5 consecutive clips smooths this
variance: a single outlier moves the mean by 1/5 of its excess, while a
sustained anomaly period drives the mean consistently above the threshold.

**Why adaptive per-round threshold?**
The training threshold is estimated from only 10 held-out clips — a small
sample that may not fully capture the range of normal score variation across
different test sessions and machines. The preceding normal monitoring window
(30 clips) is always observed before anomaly detection begins, and provides a
larger, test-time calibrated sample. Setting the threshold at 1.5× the maximum
rolling mean seen during that window gives a principled, machine-specific
safety margin.

**Why the max-component NLL score (Eq. 3) rather than mixture NLL?**
With K=2 components, the standard mixture log-likelihood
`log Σ_k π_k N(x|μ_k, Σ_k)` weights contributions by the mixing proportions.
The paper's formulation `−max_k log N(x|μ_k, Σ_k)` asks instead: how well
does the *nearest* cluster explain this clip? A clip near either normal cluster
should score low. The max-component formulation implements this correctly.
