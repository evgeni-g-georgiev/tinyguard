# TinyML — TWFR-GMM Anomaly Detection

A lightweight anomaly detection system for industrial machines, designed to run entirely on an Arduino Nano 33 BLE Sense Rev 2. The pipeline uses **Time-frequency weighted reconstruction with a Gaussian Mixture Model (TWFR-GMM)** — pure signal processing, no neural networks, no external dependencies on device.

Evaluated on the [MIMII dataset](https://zenodo.org/record/3384388) (fans, pumps, sliders, valves at three SNR levels).

---

## Repo structure

```
gmm/                 Python GMM pipeline (features → detector → evaluate → train)
deployment/          Arduino C++ implementation of the same pipeline
simulation/          Multi-node lockstep evaluation framework
preprocessing/       Shared audio loading, mel-spectrogram, MIMII split manifests
data/                MIMII download and extraction scripts
config.py            Shared path and constant definitions
```

---

## Quick start

### 1. Download MIMII data

```bash
python data/setup_data.py
# or download a specific SNR variant:
python data/download_mimii.py --snr 6dB
```

### 2. Generate train/test split manifests

```bash
python preprocessing/split_mimii.py
```

### 3. Train GMM detectors

```bash
# All 16 machines, default SNR (-6 dB), with r-search:
python gmm/train.py --r-search

# Specify SNR:
python gmm/train.py --dataset 6db --r-search
```

Trained detectors and results are written to `gmm/outputs/{snr}/`.

### 4. Run multi-node simulation

```bash
python simulation/run_simulation.py
```

Results and plots are written to `simulation/outputs/`.

---

## Arduino deployment

See [deployment/README.md](deployment/README.md) for step-by-step instructions to flash and run the GMM pipeline on an Arduino Nano 33 BLE Sense Rev 2.

Key files:
- [deployment/tinyml_gmm.ino](deployment/tinyml_gmm.ino) — top-level sketch
- [deployment/config.h](deployment/config.h) — all tunable parameters
- [deployment/export_mel_filterbank.py](deployment/export_mel_filterbank.py) — regenerate the pre-computed mel filterbank header
