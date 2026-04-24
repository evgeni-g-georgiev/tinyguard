# TinyML: Two-Node TWFR-GMM Anomaly Detection

Anomaly detection for industrial machine sounds, designed to run on two
co-located Arduino Nano 33 BLE Sense Rev 2 boards. Each board records
audio from its own microphone, trains a Gaussian Mixture Model on-chip,
and exchanges lightweight confidence signals over Bluetooth LE. A fused
score across the pair drives a CUSUM alarm. No neural networks, no
training data leaves the device, no cloud.

The approach follows the TWFR-GMM paper (Guan et al., *arXiv:2305.03328*,
2023) and the collaborative-inference regime of the Node Learning
paradigm (Kanjo & Aslanov, 2026).

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

The two nodes each pick a different `r` (with a diversity margin
enforced between them) and listen on different microphone channels. They
trade their validation statistics, z-normalise their own NLL scores, and
fuse them with fit-quality weights
`w_i = softmax(-sigma_val_i / T)`. Lower val NLL standard deviation
means a more consistent node and a higher weight. The fused z-score
feeds the shared CUSUM.

## Repo layout

```
gmm/            Python GMM pipeline (features, detector, node learning, train)
deployment/     Arduino C++ implementation that mirrors the Python pipeline
preprocessing/  Audio loading, log-mel builder, MIMII split manifest generator
data/           MIMII download and extraction script
simulation/     (not covered by this README)
config.py       Repo-level path and constant definitions
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

### 3. Create split manifests

```bash
python preprocessing/split_mimii.py                                     # -6 dB
python preprocessing/split_mimii.py --data-root data/mimii_0db \
    --out preprocessing/outputs/mimii_splits/splits_0db.json            # 0 dB
python preprocessing/split_mimii.py --data-root data/mimii_6db \
    --out preprocessing/outputs/mimii_splits/splits_6db.json            # +6 dB
```

### 4. Train and evaluate

```bash
python gmm/train.py --dataset neg6db   # default
python gmm/train.py --dataset 0db
python gmm/train.py --dataset 6db
```

For each of the 16 machines this runs three detectors side by side:
Node A alone, Node B alone, and the two-node NodeLearning fusion.
Outputs go to `gmm/outputs/{dataset}/`:

```
node_a/*.pkl, *.png, results.yaml
node_b/*.pkl, *.png, results.yaml
node_learning/*.png,  results.yaml
comparison.yaml
```

Useful flags:

```
--diversity-margin DELTA   minimum |r_B - r_A| (default 0.25)
--temperature T            softmax temperature for fusion weights (default 100)
--mic-a N, --mic-b N       microphone channels for each node (default 0 and 1)
--n-mels N                 mel bin count (default 64; output dir is suffixed)
--verbose                  print per-machine calibration parameters
```

### 5. Flash the Arduinos

See [deployment/README.md](deployment/README.md) for step-by-step
instructions. In short: set `NODE_ID` to `NODE_A` or `NODE_B` in
`deployment/config.h` before flashing each board, upload the sketch with
the Arduino IDE, power both boards, and watch the Serial monitor.

## Key configuration

All shared constants live in [gmm/config.py](gmm/config.py), which the
C++ [deployment/config.h](deployment/config.h) mirrors. Defaults:

- 16 kHz audio, 1024-sample FFT, 512-sample hop, 64 mel bins
- 60 training clips per machine (50 fit + 10 calibration)
- 2-component diagonal GMM
- CUSUM threshold at the 95th percentile of val NLLs
- r-search grid `{0.0, 0.25, 0.5, 0.75, 1.0}`
- Fusion temperature `T = 100`
