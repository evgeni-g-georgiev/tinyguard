# tinyGUARD

tinyGUARD (Tiny Gaussian Unsupervised Anomaly Recognition Device) is an industrial machine anomaly detection system that runs
end-to-end on a microcontroller with as little as 256 KB of SRAM and 1 MB
of flash. It listens to a machine through a microphone, learns what normal
operation sounds like during a 10-minute warm-up, and then flags abnormal
sound as it happens. Training, calibration and inference all happen on the
device.

This repository contains the full system: a Python simulation framework that
evaluates tinyGUARD across the MIMII dataset of industrial machine sounds,
and a C++ port that runs on a pair of Arduino Nano 33 BLE Sense Rev 2
boards exchanging confidence signals over Bluetooth LE.

## Deployment Demo
The image below redirects to a live video demonstration of tinyGUARD with two nodes around one fan. This includes: warm-up, BLE sync, normal and anomalous periods, and successful defaulting to single-node operation when one node is disconnected.

[![tinyGUARD (tiny Gaussian Unsupervised Anomaly Recognition Device) - Live Demonstration](https://img.youtube.com/vi/D8IELthSQNI/maxresdefault.jpg)](https://www.youtube.com/watch?v=D8IELthSQNI)

## How it works

tinyGUARD turns each 10-second audio clip into a log-mel spectrogram, ranks
the energies in each mel bin from loudest to quietest, and pools them into
one feature vector. The pooling weights are parameterised by a single value
`r`: at `r = 0` only the loudest frame contributes (max pooling), at `r = 1`
all frames contribute equally (mean pooling), and intermediate values blend
the two. Each machine ends up with the `r` that best separates its normal
sounds, since some machines are dominated by steady drones and others by
short transients.

A two-component diagonal Gaussian mixture model is fit on 50 normal feature
vectors collected during warm-up. Ten more held-out clips calibrate an
anomaly threshold. At inference, the negative log-likelihood under the
best-fitting component is the anomaly score for the new clip. A one-sided
cumulative sum (CUSUM) accumulates evidence over consecutive clips and
fires the alarm only when scores have stayed elevated long enough to rule
out a brief background noise. This trades a small detection delay for a low
false-alarm rate, which matters more in a factory setting where every false
alarm interrupts production.

When more than one microphone listens to the same machine, the nodes
collaborate. Each picks an `r` no peer has claimed, calibrates its own
detector, then exchanges fit-quality statistics. The nodes z-normalise their
own scores and fuse them with weights that favour better-fitting nodes. The
fused stream feeds a shared CUSUM. On hardware, two Arduino boards run this
exchange over BLE.

The methodology and evaluation results are written up in the project report.

## Repo Layout

```
simulation/     Python evaluation framework on MIMII (1 to 8 nodes)
deployment/     Arduino C++ port for two boards
gmm/            Detector primitives shared by both
preprocessing/  WAV loading and log-mel spectrogram
data/           MIMII downloader
config.py       Path constants
```

## Getting started

The Python code requires Python 3.8, 3.9, or 3.10. Create and activate an
environment with one of the following before installing dependencies.

Using venv. If `python3.10` is not on PATH, install it first. On macOS:

```bash
brew install python@3.10
```

Then:

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Using conda:

```bash
conda create -n tinyguard python=3.10 -y
conda activate tinyguard
```

The downloader needs `wget` and `unzip` on PATH. On macOS:

```bash
brew install wget
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

To download MIMII dataset:

```bash
python data/download_mimii.py --snr 6dB
```

Run the simulation. The first run builds the warmup/test splits
on its own.

```bash
python -m simulation.run_simulation
```

Each run writes a timestamped folder under `simulation/outputs/runs/` with
metrics, full traces, and timeline plots. To change the set of
microphone channels, edit `simulation/configs/default.yaml`:

```yaml
channels: [0, 4]    # one entry per node, mic indices 0..7
```

See [simulation/README.md](simulation/README.md) for the full simulation
workflow and [deployment/README.md](deployment/README.md) for flashing the
Arduino boards.

## Configuration

The algorithm constants used by both implementations live in
[gmm/config.py](gmm/config.py). The C++ side mirrors them in
[deployment/config.h](deployment/config.h). Defaults: 16 kHz audio, 1024
FFT, 512 hop, 64 mel bins, 50 fit clips plus 10 calibration clips, two GMM
components, `r` chosen from `{0.5, 0.7, 0.9, 1.0}`.

## References

The probabilistic detector is inspired by the TWFR-GMM submission to
DCASE 2023 (Guan et al.). The collaborative inference setup builds on the
node learning paradigm of Kanjo & Aslanov (2026). See project report for full citations.
