# TinyML Anomalous Sound Detection

Zero-shot anomalous sound detection for factory machinery, running entirely on an **Arduino Nano 33 BLE** microcontroller (2 MB flash, 756 KB SRAM).

A node is placed next to a machine, listens for **10 minutes of normal operation**, trains a tiny anomaly detector on-device, and then continuously monitors for faults — with no internet connection, no cloud, and no prior knowledge of the specific machine type.

---

## How It Works

The system answers one question: **"Does this machine sound normal right now?"**

Each node operates independently. There is no training dataset of anomalies, no labels, and no pre-programming for specific machine types. A node placed next to a fan learns what _that fan_ sounds like. The same hardware placed next to a pump learns what _that pump_ sounds like.

This is possible because anomaly detection is a **one-class problem**: the model only needs to learn the statistical distribution of normal sounds, then flag anything that deviates significantly.

**Three on-device phases:**

1. **Training (10 minutes):** The node records 60 audio clips (10 s each), extracts 32-dimensional embeddings via the on-device CNN, and trains a small two-layer network (1,312 parameters) to map these embeddings toward a fixed hypersphere centroid. The anomaly threshold is set at the 95th percentile of training scores.

2. **Normal monitoring:** The node continuously scores new 10-second clips by their squared distance from the centroid. Scores below the threshold are classified as normal.

3. **Anomaly monitoring:** When a fault develops, embeddings drift away from the centroid. Scores exceed the threshold and an alert is raised.

---

## System Architecture

The pipeline has two components: a frozen **feature extractor** (f\_c) — a CNN distilled from YAMNet — and a trainable **anomaly detector** (f\_s) — a two-layer Deep SVDD network trained on-device.

```
                         FROZEN (pre-loaded in flash)             TRAINED ON-DEVICE
                    ──────────────────────────────────────    ────────────────────────────
                    │                                    │    │                          │
  Audio (16 kHz) ──► AcousticEncoder (f_c)              ├───► FsSeparator (f_s)         │
  10 s clip          MobileNet V1-style DSC CNN          │    Deep SVDD 32→32→8          │
                     Input: (1, 64, 61) log-mel          │    score = max ||f_s(x)−c||²  ├──► Alert?
                     Output: (1, 32) embedding           │    threshold @ 95th pct       │
                     ~554K params, ~562 KB INT8          │    1,312 params, ~5 KB        │
                    ──────────────────────────────────────    ────────────────────────────
```

**Why a distilled CNN and not YAMNet directly?**

YAMNet is a 4 MB float32 model — far too large for the Arduino's 2 MB flash. Instead, we train a compact MobileNet V1-style CNN (AcousticEncoder) to reproduce YAMNet's embeddings using knowledge distillation on FSD50K. The CNN is then quantised to INT8, giving a ~562 KB model that fits comfortably in flash.

---

## Offline Training Pipeline

The offline pipeline runs once on a server/GPU to produce the frozen AcousticEncoder weights that are pre-loaded onto every Arduino. MIMII data is **never used** in this pipeline — it is reserved entirely for evaluation.

```
FSD50K audio ──► [1] YAMNet (4 MB, float32)  ──► 1024D embeddings (N_frames × 1024)
                 [2] PCA (32 components)       ──► 32D targets     (N_frames × 32)
                 [3] AcousticEncoder training  ──► acoustic_encoder.pt
                     MSE(student, PCA targets)
```

### Step 1 — Teacher embeddings (`preprocessing/extract_embeddings.py`)

Runs YAMNet over every clip in the FSD50K eval set to extract 1024D embeddings. Each 10 s clip is sliced into 0.975 s frames (15,600 samples at 16 kHz); YAMNet processes each frame independently. Then fits PCA(32 components) on the resulting embeddings.

**Output:** `preprocessing/outputs/fsd50k_cache/eval_embeddings.npy`, `preprocessing/outputs/pca/pca_components.npy`, `preprocessing/outputs/pca/pca_mean.npy`

### Step 2 — Mel spectrogram cache (`preprocessing/compute_mels.py`)

Computes log-mel spectrograms for every FSD50K frame in the same sorted order as Step 1, so frame index `i` in the mel cache corresponds exactly to frame index `i` in the embedding cache.

Each frame → `(1, 64, 61)` log-mel spectrogram (64 mel bins, 61 time steps):

```python
mel = librosa.feature.melspectrogram(frame, sr=16000, n_fft=1024, hop_length=256, n_mels=64)
log_mel = log(mel + 1e-6)    # (64, 61) → add channel dim → (1, 64, 61)
```

**Output:** `preprocessing/outputs/fsd50k_cache/eval_mels.npy` (N × 1 × 64 × 61, ~1.5 GB)

### Step 3 — Student distillation (`distillation/train.py`)

Trains AcousticEncoder to reproduce the PCA-projected YAMNet embeddings via MSE:

```
Loss = MSE(AcousticEncoder(mel), PCA_project(YAMNet_embed))
```

- Optimiser: AdamW, lr=1e-3, weight decay=1e-4
- Schedule: CosineAnnealingLR over 50 epochs
- Batch size: 256, 90/10 train/val split

**Output:** `distillation/outputs/student/acoustic_encoder.pt`

---

## On-Device Pipeline

### Feature Extraction: AcousticEncoder (f\_c)

A MobileNet V1-style depthwise-separable CNN that converts log-mel spectrograms to 32-dimensional embeddings.

```
Input: (1, 1, 64, 61) log-mel spectrogram  ← 0.975 s audio frame

Stem:   Conv2d(1→32, 3×3, stride=2) + BN + ReLU        → (1, 32, 32, 31)
B1:     DW(32, stride=2) + BN + ReLU                    → (1, 32, 16, 16)  ← SRAM peak
        PW(32→64)        + BN + ReLU                    → (1, 64, 16, 16)
B2:     DW(64, stride=2) + BN + ReLU                    → (1, 64,  8,  8)
        PW(64→128)       + BN + ReLU                    → (1,128,  8,  8)
B3:     DW(128, stride=1) + PW(128→128)                 → (1,128,  8,  8)
B4:     DW(128, stride=2) + PW(128→256)                 → (1,256,  4,  4)
B5:     DW(256, stride=1) + PW(256→256)                 → (1,256,  4,  4)
B6:     DW(256, stride=2) + PW(256→512)                 → (1,512,  2,  2)
B7:     DW(512, stride=1) + PW(512→512)                 → (1,512,  2,  2)
Pool:   AdaptiveAvgPool2d(1)                             → (1,512,  1,  1)
Head:   Linear(512→32)                                   → (1, 32)

Parameters: ~554K
Flash (INT8 TFLite): ~562 KB
```

Each 10 s clip is sliced into ~10 frames (0.975 s each). The encoder processes each frame independently, yielding a `(10, 32)` embedding matrix per clip. Depthwise-separable convolutions reduce parameters ~8–9× versus standard convolutions. BatchNorm is folded into conv weights at TFLite export, adding zero runtime cost.

### Anomaly Detection: FsSeparator (f\_s)

A two-layer network trained on-device using Deep SVDD (Ruff et al., ICML 2018). Maps 32D embeddings into an 8D space where normal sounds cluster around a fixed centroid.

```
f_s(x) = ReLU( W₂ · ReLU( W₁ · x + b₁ ) )

W₁ ∈ ℝ^{32×32}, b₁ ∈ ℝ^{32}   (fc1: bias=True)
W₂ ∈ ℝ^{8×32}                  (fc2: bias=False ← required by Deep SVDD)

Parameters: 32×32 + 32 + 32×8 = 1,312
```

The final layer has no bias to prevent trivial collapse: with bias, the network could satisfy the loss by setting W₂=0, b₂=c, mapping all inputs to the centroid regardless of content.

**Training (on-device, 10-minute window):**

1. Forward-pass all 60 training clips through f\_c and stack the per-frame embeddings.
2. Initialise centroid `c = mean(f_s(x_i))`. **Fix c — never update it.**
3. Minimise `L = (1/N) Σᵢ ||f_s(xᵢ) − c||²` with SGD (lr=0.01, weight decay=1e-4, no momentum).
4. Set threshold `τ = percentile(scores, 95)` on all training clips.

**Scoring:**

```
score(clip) = max_frame ||f_s(frame_embedding) − c||²
```

Max-frame scoring means a single anomalous frame (e.g. a brief mechanical knock) flags the whole clip, rather than being diluted by mean-pooling.

---

## Repository Structure

Each folder has its own `README.md` with detailed input/output documentation.

```
tinyml/
├── config.py                    # All paths and constants (single source of truth)
│
├── data/                        # Data acquisition
│   ├── download_fsd50k.py       # Download FSD50K eval set (~6.2 GB)
│   ├── download_mimii.py        # Download MIMII 6 dB dataset (~30 GB)
│   ├── fsd50k/                  # FSD50K eval audio
│   ├── mimii/                   # MIMII audio: {machine}/{id}/{normal,abnormal}/*.wav
│   └── yamnet/                  # YAMNet TFLite model (offline use only)
│
├── preprocessing/               # Feature extraction + train/test split
│   ├── extract_embeddings.py    # YAMNet → FSD50K embeddings + PCA(32D)
│   ├── compute_mels.py          # FSD50K WAVs → log-mel spectrogram cache
│   └── split_mimii.py           # Fixed MIMII train/test manifest
│
├── distillation/                # AcousticEncoder (f_c) training
│   ├── cnn.py                   # Model definition: MobileNet V1-style DSC CNN
│   └── train.py                 # Knowledge distillation via MSE on FSD50K
│
├── separator/                   # On-device SVDD training simulation
│   ├── separator.py             # Model definition: FsSeparator + train_fs/score_clips
│   └── train.py                 # 10-min training window → artefacts per machine
│
├── inference/                   # On-device monitoring simulation
│   └── run.py                   # Normal + anomaly rounds → plots + results.yaml
│
├── scripts/                     # Standalone utilities
│   └── memory_audit.py          # Flash + SRAM budget audit
│
├── preprocessing/outputs/       # fsd50k_cache/, pca/, mimii_splits/
├── distillation/outputs/        # student/ (checkpoint + curve), export/
├── separator/outputs/           # {mtype}_{mid}.pt × 16
└── inference/outputs/           # results.yaml + per-machine plots
```

---

## Running the Code

**Prerequisites:** Python 3.10+, PyTorch, NumPy, scikit-learn, librosa, matplotlib, tqdm, PyYAML, ai-edge-litert.

```bash
# ── Data acquisition ──────────────────────────────────────────────────────────
python data/download_fsd50k.py               # → data/fsd50k/  (~6.2 GB)
python data/download_mimii.py                # → data/mimii/   (~30 GB)

# ── Preprocessing (run once — skips automatically if outputs exist) ───────────
python preprocessing/extract_embeddings.py   # ~15–20 min
python preprocessing/compute_mels.py         # ~30 min, ~1.5 GB
python preprocessing/split_mimii.py          # fast

# ── Distillation (run once) ───────────────────────────────────────────────────
python distillation/train.py                 # ~10 min on GPU
python distillation/train.py --epochs 100 --batch 256 --lr 1e-3

# ── On-device simulation ──────────────────────────────────────────────────────
python separator/train.py                    # SVDD training, one artefact per machine
python inference/run.py                      # monitoring simulation + plots

# ── Utilities ─────────────────────────────────────────────────────────────────
python scripts/memory_audit.py
python scripts/memory_audit.py --tflite distillation/outputs/export/acoustic_encoder_int8.tflite
```

All results (YAML summaries + PNG plots) are written to each stage's `outputs/` subfolder. Re-running a stage overwrites its previous outputs.
