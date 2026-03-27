# TinyML Project Summary

## Ultimate Goal
Deploy tiny ML models on **Arduino Nano 33 BLE** chips (2–4 MB memory budget) attached to industrial machines. Each chip detects anomalies locally and shares knowledge with other chips over BLE — demonstrating the power of **decentralised node learning** (arXiv:2602.16814).

The key contribution is showing that on-device learning (f_s) combined with BLE-based prototype exchange achieves competitive anomaly detection — even though the frozen encoder (f_c) has **never seen any MIMII labels or machine IDs**. All task-specific learning happens on-device.

---

## The Three-Component Pipeline

### f_c — frozen pretrained acoustic encoder *(Phase 1, current focus)*
- **YAMNet** (MobileNet v1, pretrained on AudioSet 521 classes, ~3.2M params, ~900 KB INT8 TFLite)
- Frozen at deployment — never trained on MIMII or any anomaly detection data
- Extracts 1024D general audio embeddings per ~1s frame
- **PCA projection** (1024→16D) computed from industrial/mechanical audio (no MIMII labels used) to compress to a deployment-friendly dimensionality
- Philosophically: f_c is a **general acoustic feature extractor**, not a machine-specific model. A customer deploys chips without any f_c retraining.

**Why YAMNet over custom SupCon encoder:**
- The previous approach (SupCon with machine IDs) was "secretly learning everything beforehand" — f_c learned to cluster machines by ID, leaving nothing for f_s/node learning to contribute.
- YAMNet provides rich, general audio features without any exposure to MIMII labels. This makes the node learning contribution genuine and honest.
- With a 2–4 MB budget, YAMNet fits directly — no knowledge distillation needed.
- YAMNet's AudioSet training covers mechanical/environmental sounds, making it suitable for industrial audio despite being general-purpose.

**Alternatives considered:**
- PANNs CNN6 (~100K params, ~100 KB): much lighter but weaker features. Good fallback if memory is tight.
- EfficientAT mn01 (~120K params): distilled from Transformer ensemble, competitive quality. Another fallback option.
- Custom SupCon encoder (previous approach, 6.5K params): rejected — too task-specific, undermines node learning contribution.

### f_s — online-trainable separator *(Phase 2, next)*
- Tiny MLP (16→16→8), ~400 params
- Runs on-device after deployment, learns from live normal audio only
- Learns a normal prototype (centroid) per machine; anomaly score = distance to centroid
- No anomaly labels needed — pure unsupervised on-device learning
- This is where task-specific learning happens

### BLE node learning *(Phase 3)*
- Multiple chips exchange prototypes over BLE
- Friend/Foe gating: accept prototypes from similar machines, reject distant ones
- Soft-merge received prototypes with local ones (weighted by similarity)
- Collectively converge faster than any single chip alone
- Demonstrates the core thesis: decentralised collaboration under tight memory constraints

---

## Dataset: MIMII
4 industrial machine types: **fan, pump, slider, valve** — each with 4 machine IDs (id_00, id_02, id_04, id_06), normal + abnormal WAV files at 16kHz. Located at `data/`.

**Important framing:** MIMII is treated as the **deployment target** — the machines we deploy chips on. f_c has never seen MIMII data during its training (it was pretrained on AudioSet). All MIMII-specific learning is done by f_s on-device.

---

## Implementation Plan

### Step 1: Set up YAMNet as f_c backbone
- Load pretrained YAMNet (TensorFlow Hub or torch port)
- Write script to extract 1024D embeddings from all MIMII clips
- Cache embeddings to disk for fast iteration

### Step 2: Compute PCA projection (1024→16D)
- Fit PCA on embeddings from **all** MIMII normal clips (or a general industrial audio corpus)
- No labels used — purely unsupervised dimensionality reduction
- Save PCA transform for deployment
- Verify variance retention across dimensions (avoid the PC1-dominance problem seen with the old encoder)

### Step 3: Evaluate f_c baseline
- Run existing evaluation metrics (AUC, separation ratio, overlap %, cosine sim) on YAMNet+PCA embeddings
- Expected: modest AUC (~0.55–0.65) since f_c is general-purpose and not optimised for anomaly detection
- This is intentional — the gap between f_c baseline and f_c+f_s performance IS the contribution

### Step 4: Build f_s simulation in Python
- Train 16→16→8 MLP on frozen f_c embeddings
- Hinge/contrastive loss with on-device-realistic constraints
- Show that f_s significantly improves AUC over f_c-only baseline
- VICReg-style variance regularisation to prevent dimensional collapse in f_s's learned space

### Step 5: Simulate BLE node learning
- Multi-node simulation: each node sees one machine, exchanges prototypes
- Show collaboration improves convergence speed and/or final AUC
- Friend/Foe gating evaluation

---

## Key Files
| File | Role |
|---|---|
| `configs/fc.yaml` | Pipeline configuration (audio params, data paths, model settings) |
| `src/data/dataset.py` | MIMII clip discovery, mel-spec computation, windowing, dataset classes |
| `src/evaluation/metrics.py` | AUC, separation ratio, overlap %, cosine sim, silhouette |
| `src/evaluation/visualisation.py` | t-SNE, distance distribution plots |

---

## Previous Results (archived — old SupCon approach)

The custom TinyConvEncoder (6,528 params) trained with SupCon on fan/pump/slider achieved mean AUC 0.5748 on valve — essentially random. This approach was abandoned because:
1. Poor generalisation to unseen machine types
2. PCA collapse (PC1 = 50% variance, 3 PCs = 80%)
3. Philosophically wrong: training f_c on machine IDs means f_c is doing the anomaly detection work, not f_s/node learning
