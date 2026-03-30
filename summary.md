# TinyML Anomalous Sound Detection — Full Pipeline Summary

## Product Vision

A single chip that a customer deploys on **any** machine — factory pump, conveyor belt, industrial fan, etc. The chip:

1. **Listens** for 10 minutes to learn what "normal" sounds like
2. **Monitors** continuously, alerting when the sound pattern changes (anomaly)
3. **Zero-shot**: no pre-training on the specific machine type. The customer plugs it in and it just works.

Target hardware: **Arduino Nano 33 BLE** (2 MB flash, ~756 KB SRAM).

---

## Architecture Overview

The system has three components, each with a distinct role:

```
                    OFFLINE (PC/GPU)                          ON-DEVICE (Arduino)
              ┌──────────────────────────┐           ┌─────────────────────────────────┐
              │                          │           │                                 │
  FSD50K      │  YAMNet ──► PCA ──► 16D │  distill  │  Audio ──► Mel ──► MediumCNN    │
  (general    │  (teacher)   ↓          │  ───────► │              (f_c, frozen)       │
   audio)     │          MediumCNN      │           │                   │              │
              │          (student)      │           │                   ▼ 16D          │
              │                          │           │              f_s (SVDD)          │
              │                          │           │           (trained on-device)    │
              │                          │           │                   │              │
              │                          │           │            anomaly score         │
              │                          │           │          (threshold → alert)     │
              └──────────────────────────┘           └─────────────────────────────────┘
```

### f_c — Feature Extractor (frozen, in flash)

**MediumCNN**: a MobileNet v1-style CNN that converts raw audio into compact 16D embeddings.

- **Input**: log-mel spectrogram of one 0.975s audio frame — shape `(1, 64, 61)`
- **Output**: 16D embedding vector
- **Architecture**: stem conv + 7 depthwise-separable blocks + global avg pool + linear head
- **Size**: ~547K params (~547 KB int8, well within 2 MB flash)
- **Training**: knowledge distillation from YAMNet (Google's audio classifier) on FSD50K
- **Key property**: learns a *general* audio representation — never sees any target machine during training

### f_s — Separator (trained on-device, in SRAM)

**FsSeparator**: a tiny 2-layer MLP trained on-device using Deep SVDD.

- **Input**: 16D embedding from f_c
- **Architecture**: Linear(16→32, bias) + ReLU + Linear(32→8, no bias) + ReLU
- **Output**: 8D projected vector
- **Size**: 800 params (3.2 KB float32)
- **Training method**: Deep SVDD (Ruff et al., ICML 2018)
  - Loss = `(1/N) Σ ||f_s(x_i) - c||² + (λ/2)||W||²`
  - Centroid `c` is fixed after initialisation (mean of first forward pass)
  - Weight decay prevents collapse
- **No bias on output layer**: Deep SVDD requirement — with bias, the network can trivially map everything to centroid by setting W2=0, b=c
- **Anomaly score**: `||f_s(embedding) - c||²` — squared distance from centroid
- **Threshold**: set from 95th percentile of training clip scores
- **On-device backprop**: computed analytically (no autograd library needed)

### PCA — Dimensionality Reduction (absorbed into distillation)

- YAMNet outputs 1024D embeddings; these are projected to 16D via PCA
- PCA is fitted on FSD50K teacher embeddings (**not** on MIMII evaluation data — this would be leakage)
- The student (MediumCNN) learns to directly output 16D, so **PCA is NOT needed on device**
- PCA components are only used offline during distillation target preparation

---

## Memory Budget (Arduino Nano 33 BLE)

All model size estimates use `M = P × (B/8)` bytes, where B = bits per parameter.
INT8 quantisation gives 4× density over FP32. CNN weights are approximately normally
distributed after training, so INT8 post-training quantisation preserves accuracy well.

### Flash (2 MB = 2,048 KB)
| Component | Size |
|---|---|
| MediumCNN (int8) | ~547 KB |
| Application code | ~50 KB (estimate) |
| **Total** | **~597 KB** (fits in 2 MB) |

### SRAM (~756 KB)
| Component | Size |
|---|---|
| MediumCNN inference arena | ~60-120 KB (TFLite Micro reuses buffers between layers) |
| Audio buffer (one 0.975s frame, float32) | ~61 KB |
| f_s weights + centroid + threshold | ~3.2 KB |
| f_s training gradient buffers | ~4.5 KB |
| Embedding buffers (16D + 8D) | ~96 bytes |
| Training clip buffer (12 clips × 16D) | ~768 bytes |
| **Total** | **~130-190 KB** (fits in 756 KB) |

The dominant SRAM cost is the TFLite Micro arena. The arena only needs to hold the two
largest adjacent activation tensors simultaneously (TFLite reuses memory across layers).
Largest activation in MediumCNN is after Block 1: `(64, 16, 16)` = 16 KB at INT8.

---

## Offline Pipeline — Step by Step

These steps are run once on a GPU machine. The output is a frozen MediumCNN model that ships to every device.

### Prerequisites

```
data/                   # MIMII dataset (for evaluation only)
  fan/id_00/normal/     # WAV files, 10s each, 16kHz
  fan/id_00/abnormal/
  fan/id_02/...
  pump/id_00/...
  slider/id_00/...
  valve/id_00/...
```

MIMII has 4 machine types × 4 machine IDs = 16 machines total. Each machine has ~200 normal clips and ~50-200 abnormal clips.

```
models/yamnet/yamnet.tflite   # already present in repo
```

### Step 1: Download FSD50K and Train MediumCNN

```bash
python scripts/distill_general.py
# Or with eval-only (faster, less data):
python scripts/distill_general.py --eval_only
```

This single script handles the full offline pipeline:

1. **Download FSD50K from Zenodo**
   - eval set: ~10K clips (~6 GB, 2 zip parts: `.z01` + `.zip`)
   - dev set: ~40K clips (~14 GB, 5 zip parts: `.z01`–`.z04` + `.zip`)
   - Split zip extraction uses `7z`

2. **Extract YAMNet teacher embeddings** for every FSD50K clip
   - Load audio, resample to 16kHz
   - Slice into 0.975s frames (15,600 samples, non-overlapping)
   - Per frame: compute log-mel spectrogram `(1, 64, 61)` → run YAMNet TFLite → extract 1024D embedding (tensor index 115) → dequantise
   - Cache to `outputs/fsd50k_cache/`: `mels_eval.npy`, `teachers_eval.npy`, `mels_dev.npy`, `teachers_dev.npy`

   Audio parameters:
   - Sample rate: 16,000 Hz
   - Frame length: 15,600 samples (0.975s)
   - FFT size: 1024, hop: 256, mel bins: 64, log offset: 1e-6

3. **Fit PCA on FSD50K embeddings**
   - Subsample to 100K frames if larger
   - Project: `teachers_16d = (teachers_1024 - mean) @ components.T`
   - Save `outputs/distill_general/pca_components.npy` (16×1024) and `pca_mean.npy` (1024,)

4. **Train MediumCNN** (knowledge distillation)
   - Loss: MSE between student 16D output and PCA-projected teacher 16D
   - Optimiser: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, 50 epochs, batch 256
   - Best model saved by validation loss (90/10 split)
   - Output: `outputs/distill_general/medium_cnn.pt`

5. **Evaluate on MIMII** (zero-shot, all 16 machines)
   - Extract per-frame 16D embeddings via MediumCNN and YAMNet→PCA (for comparison)
   - Per machine: train f_s (Deep SVDD) on normal frames, score all clips with max-frame scoring, compute AUC
   - Output: `outputs/distill_general/results.yaml`

MediumCNN architecture:
```
Stem:    Conv2d(1→32, 3×3, stride=2) + BN + ReLU     → (32, 32, 31)
Block 1: DepthwiseSeparable(32→64, stride=2)          → (64, 16, 16)
Block 2: DepthwiseSeparable(64→128, stride=2)         → (128, 8, 8)
Block 3: DepthwiseSeparable(128→128, stride=1)        → (128, 8, 8)
Block 4: DepthwiseSeparable(128→256, stride=2)        → (256, 4, 4)
Block 5: DepthwiseSeparable(256→256, stride=1)        → (256, 4, 4)
Block 6: DepthwiseSeparable(256→512, stride=2)        → (512, 2, 2)
Block 7: DepthwiseSeparable(512→512, stride=1)        → (512, 2, 2)
Pool:    AdaptiveAvgPool2d(1)                          → (512,)
Head:    Linear(512→16)                                → (16,)
```

Each DepthwiseSeparableConv = depthwise Conv2d(in, in, 3×3, groups=in) + BN + ReLU + pointwise Conv2d(in, out, 1×1) + BN + ReLU.

### Step 2: Deployment Simulation

```bash
python scripts/simulate_deployment.py
```

Runs a full deployment simulation on **all 16 MIMII machines** with timeline plots.

For each machine:
1. **Training phase (10 min)**: Take 60 normal clips, extract per-frame embeddings through MediumCNN, train f_s (Deep SVDD, SGD lr=0.01, weight_decay=1e-4, early stopping patience=20)
2. **Normal monitoring (5 min)**: 30 normal clips scored with max-frame scoring
3. **Anomaly monitoring (5 min)**: 30 abnormal clips scored with max-frame scoring
4. Single clip above threshold = alert (no consecutive requirement)

Outputs:
- `outputs/deployment/deployment_timeline.png` — 2-column grid of per-machine timelines
- `outputs/deployment/results.yaml` — per-machine AUC, recall, precision, FAR, first detection delay

### Step 3: (Optional) Memory Audit

```bash
python scripts/memory_audit.py
```

Audits flash and SRAM requirements for the deployed model: parameter counts, INT8 sizes, activation peak, arena estimate.

---

## On-Device Inference (What Runs on Arduino)

At deployment time, the Arduino runs this loop every ~1 second:

```
1. Capture 0.975s of audio (15,600 samples at 16kHz)
2. Compute log-mel spectrogram → (1, 64, 61)
3. Run MediumCNN forward pass → 16D embedding
4. Run f_s forward pass → 8D projected vector
5. Compute anomaly score = ||projected - centroid||²
6. If score > threshold → ALERT
```

### On-Device f_s Training (First 10 Minutes)

During the initial learning phase, f_s is trained using analytical gradients (no autograd):

```
Forward pass:
  z1 = W1 @ x + b1          (32,) = (32,16) @ (16,) + (32,)
  a1 = ReLU(z1)              (32,)
  z2 = W2 @ a1               (8,)  = (8,32) @ (32,)
  a2 = ReLU(z2)              (8,)
  loss = ||a2 - c||²

Backward pass:
  dL/da2 = 2(a2 - c)                (8,)
  dL/dz2 = dL/da2 * (z2 > 0)       (8,)    element-wise ReLU gradient
  dL/dW2 = outer(dL/dz2, a1)       (8,32)   one outer product
  dL/da1 = W2.T @ dL/dz2           (32,)    matrix-vector multiply
  dL/dz1 = dL/da1 * (z1 > 0)       (32,)    element-wise ReLU gradient
  dL/dW1 = outer(dL/dz1, x)        (32,16)  one outer product
  dL/db1 = dL/dz1                   (32,)

Parameter updates (SGD with weight decay):
  W1 -= lr * (dL/dW1 + λ * W1)
  b1 -= lr * (dL/db1 + λ * b1)
  W2 -= lr * (dL/dW2 + λ * W2)
```

Centroid `c` is set once as the mean of projected embeddings from the first batch, then fixed.

---

## Key Design Decisions

### Why knowledge distillation instead of training from scratch?
YAMNet was trained on AudioSet (2M+ clips, 527 classes) and captures rich audio semantics. We can't train something this powerful on-device. Instead, we compress its knowledge into a model small enough for a microcontroller.

### Why FSD50K and not MIMII for training?
MIMII is our evaluation data — it simulates the customer's machine. Training on MIMII would be leakage. FSD50K is a large, diverse dataset of general audio that teaches the model to understand sound structure broadly. This is what enables zero-shot deployment.

### Why 16D embeddings?
PCA on YAMNet's 1024D embeddings shows that ~16 components capture the important variance for anomaly detection. 16D is also small enough that f_s (800 params) can learn to separate normal/anomalous in the projected space on-device.

### Why max-frame scoring instead of mean-pooling?
A 10-second clip contains ~10 frames of 0.975s. An anomaly might only appear in 1-2 frames. Mean-pooling dilutes the signal. Max-frame scoring catches even brief anomalies.

### Why Deep SVDD instead of simpler methods?
Deep SVDD is a learned projection that can capture non-linear boundaries in embedding space. A simple threshold on raw distance would miss complex anomaly patterns. The 2-layer architecture (800 params) is small enough to train on-device with analytical gradients.

### Why no bias on f_s output layer?
Deep SVDD requirement. With bias, the network can trivially collapse: set W2=0 and b=c, mapping everything to the centroid regardless of input. Without bias, the network must actually use the input structure.

### Why INT8 for MediumCNN?
INT8 gives 4× parameter density over FP32. Pre-trained CNN weights are approximately normally distributed, so INT8 post-training quantisation preserves accuracy well — models are more sensitive to dynamic range than precision. 547K params × 1 byte = 547 KB, leaving over 1.4 MB of flash headroom.

---

## File Map

### Source code
```
src/models/cnn.py              — MediumCNN architecture (the student model)
src/models/separator.py        — FsSeparator architecture (on-device trainable)
src/models/__init__.py         — package init
src/__init__.py                — package init
```

### Scripts
```
scripts/distill_general.py          — MAIN PIPELINE: download FSD50K → extract teachers → fit PCA → train MediumCNN → evaluate on MIMII
scripts/simulate_deployment.py      — deployment simulation on all 16 MIMII machines with timeline plots
scripts/memory_audit.py             — audits flash/SRAM requirements for deployment
```

### Output directories
```
outputs/distill_general/       — MediumCNN checkpoint, PCA matrices, MIMII evaluation results
outputs/deployment/            — deployment simulation results and plots
outputs/fsd50k_cache/          — cached FSD50K mel spectrograms and YAMNet teacher embeddings
```

### Models
```
models/yamnet/yamnet.tflite    — pretrained YAMNet (already present)
```

---

## Current Status

### What exists
- `data/` — MIMII dataset, all 16 machines ✓
- `models/yamnet/yamnet.tflite` — YAMNet ✓
- `requirements.txt` — dependencies ✓

### What needs to be built
1. `src/models/cnn.py` — MediumCNN architecture
2. `src/models/separator.py` — FsSeparator architecture
3. `scripts/distill_general.py` — full offline pipeline
4. `scripts/simulate_deployment.py` — deployment simulation
5. `scripts/memory_audit.py` — memory audit

### Build order
```bash
# 1. Implement model files (src/)
# 2. Run the full pipeline
python scripts/distill_general.py          # ~3-6 hours: downloads FSD50K, trains MediumCNN, evaluates on MIMII
# Or faster with eval set only:
python scripts/distill_general.py --eval_only

# 3. Run deployment simulation
python scripts/simulate_deployment.py      # ~30-60 minutes

# 4. Optional: check memory budget
python scripts/memory_audit.py
```

---

## Leakage Analysis

The pipeline has **zero MIMII leakage**:

| Step | Data used | MIMII involved? |
|---|---|---|
| YAMNet | Pre-trained on AudioSet | No |
| FSD50K download | Zenodo | No |
| YAMNet embedding extraction | FSD50K clips | No |
| PCA fitting | FSD50K teacher embeddings | No |
| MediumCNN training | FSD50K mels + PCA-projected teachers | No |
| f_s training | Normal clips from ONE machine (simulates deployment) | Yes, but only normal clips from the specific machine being tested — this is the "10 minutes of listening" the customer does |
| Evaluation | Normal + abnormal clips | Yes, this is the test |

The only time MIMII data is used is during f_s training (simulating the customer's 10-minute learning phase) and evaluation. MediumCNN never sees any MIMII data during its training.
