# TinyML: N-Node TWFR-GMM Anomaly Detection

Anomaly detection for industrial machine sounds. The same TWFR-GMM detector
runs in two places:

- **Python (`simulation/`)**: full evaluation framework on the MIMII
  dataset, configurable for 1 to 8 microphone channels per machine, with
  block-level metrics, AUC, detection-lag plots, and JSON traces.
- **C++ (`deployment/`)**: on-device port for two co-located Arduino
  Nano 33 BLE Sense Rev 2 boards. Each board records audio from its own
  microphone, trains a Gaussian Mixture Model on-chip, and exchanges
  lightweight confidence signals over Bluetooth LE. A fused score across
  the pair drives a CUSUM alarm. No neural networks, no training data
  leaves the device, no cloud.

The approach follows the TWFR-GMM paper (Guan et al., *arXiv:2305.03328*,
2023) and the collaborative-inference regime of the Node Learning paradigm
(Kanjo & Aslanov, 2026).

Evaluated on the [MIMII dataset](https://zenodo.org/record/3384388) at
three SNR levels (-6 dB, 0 dB, +6 dB) across 16 machines.

## How the pipeline works

Each 10-second audio clip is turned into a log-mel spectrogram. For every
mel bin, frames are sorted by energy and combined with a Global Weighted
Ranking Pooling (GWRP) weight vector parameterised by `r`:

- `r = 0` is max pooling, which emphasises transients.
- `r = 1` is mean pooling, which emphasises steady-state energy.
- Intermediate `r` blends the two.

Each clip is compressed to one `(n_mels,)` feature vector. A 2-component
diagonal Gaussian Mixture Model is fit on 50 normal clips and its
threshold is calibrated on 10 held-out normal clips. A new clip is scored
by the negative log-likelihood under its best-fitting component
(Eq. 3 of Guan et al.). A Page-Hinkley CUSUM over the stream of scores
drives the alarm.

When N nodes (mic channels) are run together, each picks its own `r` via
a greedy diversity rule: every node prefers an `r` that no peer on the
same machine has already claimed. They trade their validation statistics,
z-normalise their own NLL scores, and fuse them with fit-quality weights
`w_i = softmax(-sigma_val_i / T)`. Lower val NLL standard deviation means
a more consistent node and a higher weight. The fused z-score feeds the
shared CUSUM.

## Repo layout

```
simulation/     Python N-node simulation framework (this is the main entry point)
deployment/     Arduino C++ implementation of the 2-node case
gmm/            Shared detection primitives (config, GMM, features, detector)
preprocessing/  Audio loading and log-mel spectrogram builder
data/           MIMII download and extraction script
config.py       Repo-level path constants
```

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

Requirements on the host: Python 3.10+, wget and unzip on PATH.

### 2. Download MIMII data

Each SNR variant is ~30 GB on disk after extraction.

```bash
python data/download_mimii.py --snr -6dB   # default
python data/download_mimii.py --snr 0dB
python data/download_mimii.py --snr 6dB
```

Files land at `data/mimii_neg6db/`, `data/mimii_0db/`, `data/mimii_6db/`.

### 3. Run the simulation

```bash
python -m simulation.run_simulation
```

The simulation auto-splits the data on first run (no separate split
command needed). Each run produces a timestamped directory under
`simulation/outputs/runs/<timestamp>/` with `results.json`, `summary.txt`,
and per-channel timeline plots. See [simulation/README.md](simulation/README.md)
for details.

To pick specific microphone channels, edit `simulation/configs/default.yaml`:

```yaml
channels: [1, 4, 6]   # 1..8 entries, 0..7 values
snr: "-6dB"           # 6dB | 0dB | -6dB
```

### 4. Flash the Arduinos

See [deployment/README.md](deployment/README.md) for step-by-step
instructions. In short: set `NODE_ID` to `NODE_A` or `NODE_B` in
`deployment/config.h` before flashing each board, upload the sketch with
the Arduino IDE, power both boards, and watch the Serial monitor.

## Key configuration

All shared algorithm constants live in [gmm/config.py](gmm/config.py),
which the C++ [deployment/config.h](deployment/config.h) mirrors. Defaults:

- 16 kHz audio, 1024-sample FFT, 512-sample hop, 64 mel bins
- 60 training clips per machine (50 fit + 10 calibration)
- 2-component diagonal GMM
- CUSUM reference level k set to the max of the val NLLs
- r-search grid `{0.5, 0.7, 0.9, 1.0}` with greedy diversity across nodes
- Fusion temperature `T = 100`
