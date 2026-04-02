# Deployment Memory Analysis

**Hardware target: Arduino Uno Q — MCU only**
STM32, Arm Cortex-M33 @ 160 MHz, **2 MB flash**, **786 KB SRAM**

The board has two processors. The MPU (4 GB RAM) is used exclusively for Bluetooth (Python + bleak). All TinyML constraints apply to the MCU only.

---

## Deployment Phases

The on-device lifecycle has three distinct phases with different memory profiles:

| Phase | Duration | What runs |
|---|---|---|
| **Collect** | 10 min | f_c inference on every audio frame; embeddings buffer grows |
| **Train** | ~5 sec | f_s forward + backward + SGD update, all embeddings in SRAM |
| **Monitor** | continuous | f_c inference + f_s inference per frame |

---

## Memory Formulas

```
Weights:          P × B/8  bytes
Gradients:        P × B/8  bytes         (training only)
Optimizer states: Adam → 2 × P × B/8    SGD (no momentum) → 0
Activations:      layer-by-layer trace   (stored during forward for backward pass)
```

---

## f_c — AcousticEncoder (inference only, TFLite INT8)

f_c is frozen at distillation time and deployed as a TFLite INT8 model. It never trains on-device.

### Flash (model storage)

| Item | Calculation | Bytes |
|---|---|---|
| INT8 weights | ~547,000 params × 1 B | ~534 KB |
| INT32 biases (Linear head only) | 32 × 4 B | 128 B |
| Per-channel quant scales (all conv/linear output channels) | (32+32+64+128+128+256+256+512+512+32) × 4 B | ~8 KB |
| TFLite flatbuffer overhead (op codes, tensor descriptors, metadata) | estimated | ~5–15 KB |
| **Total flash estimate** | | **~547–557 KB** |

**Confidence: ESTIMATE (±15%)** — INT8 weight count is exact from the architecture; flatbuffer overhead requires an actual TFLite export to measure.

### SRAM during inference (tensor arena)

TFLite Micro's greedy memory planner reuses freed activation buffers. For a sequential network
the peak SRAM is the maximum of `(input bytes + output bytes)` over all layer transitions.
Activations are INT8 = 1 byte/value.

| Layer | Output shape | Output (B) | Input (B) | Peak (in + out) |
|---|---|---|---|---|
| Input | (1, 1, 64, 61) | 3,904 | — | — |
| Stem conv stride=2 | (1, 32, 32, 31) | 31,744 | 3,904 | 35,648 |
| B1 depthwise stride=2 | (1, 32, 16, 16) | 8,192 | 31,744 | **39,936 ← peak** |
| B1 pointwise | (1, 64, 16, 16) | 16,384 | 8,192 | 24,576 |
| B2 depthwise stride=2 | (1, 64, 8, 8) | 4,096 | 16,384 | 20,480 |
| B2 pointwise | (1, 128, 8, 8) | 8,192 | 4,096 | 12,288 |
| B3 depthwise stride=1 | (1, 128, 8, 8) | 8,192 | 8,192 | 16,384 |
| B3 pointwise | (1, 128, 8, 8) | 8,192 | 8,192 | 16,384 |
| B4 depthwise stride=2 | (1, 128, 4, 4) | 2,048 | 8,192 | 10,240 |
| B4 pointwise | (1, 256, 4, 4) | 4,096 | 2,048 | 6,144 |
| B5 depthwise stride=1 | (1, 256, 4, 4) | 4,096 | 4,096 | 8,192 |
| B5 pointwise | (1, 256, 4, 4) | 4,096 | 4,096 | 8,192 |
| B6 depthwise stride=2 | (1, 256, 2, 2) | 1,024 | 4,096 | 5,120 |
| B6 pointwise | (1, 512, 2, 2) | 2,048 | 1,024 | 3,072 |
| B7 depthwise stride=1 | (1, 512, 2, 2) | 2,048 | 2,048 | 4,096 |
| B7 pointwise | (1, 512, 2, 2) | 2,048 | 2,048 | 4,096 |
| AvgPool + head | (1, 32) | 128 | 2,048 | 2,176 |

**TFLM tensor arena lower bound: 39,936 B ≈ 40 KB**

The actual arena will be slightly larger due to:
- 16-byte alignment padding on every tensor
- Per-op CMSIS-NN scratch buffers (largest is B1 depthwise: 32 channels × 3×3 × 2 B = 576 B)

These are small relative to the 40 KB peak, so the arena is expected to land around **40–50 KB**.

**Confidence: TIGHT (±20%)** — lower bound is exact from the architecture; actual figure requires `RecordingMicroInterpreter` profiling on TFLM.

---

## f_s — FsSeparator (training + inference, float32)

Architecture: `Linear(32→32, bias=True) → ReLU → Linear(32→8, bias=False) → ReLU`

Trained on-device from scratch at float32. Never quantized (weights must remain updatable for re-training across deployments).

### Exact parameter count

| Layer | Params |
|---|---|
| fc1 weights | 32 × 32 = 1,024 |
| fc1 bias | 32 |
| fc2 weights | 32 × 8 = 256 |
| fc2 bias | none (bias=False, required by Deep SVDD) |
| **Total** | **1,312** |

### Training memory (float32, batch_size = 32)

| Component | Formula | Bytes | Confidence |
|---|---|---|---|
| Weights | 1,312 × 4 B | 5,248 | EXACT |
| Gradients | 1,312 × 4 B | 5,248 | EXACT |
| Optimizer states | SGD, momentum=0 → 0 | 0 | EXACT |
| Activations | see derivation below | 13,312 | EXACT |
| **Training total** | | **23,808 B ≈ 23 KB** | |

**Activation memory derivation**

For backprop, PyTorch must retain the tensors needed to compute each parameter's gradient:

| Tensor | Why stored | Size (B=32, float32) |
|---|---|---|
| x₀ = input to fc1 | needed for ∂W₁/∂L = (∂L/∂z₁)ᵀ · x₀ | 32 × 32 × 4 = 4,096 |
| z₁ = pre-ReLU output of fc1 | needed for ReLU₁ gradient (sign mask) | 32 × 32 × 4 = 4,096 |
| x₁ = relu(z₁) = input to fc2 | needed for ∂W₂/∂L | 32 × 32 × 4 = 4,096 |
| z₂ = pre-ReLU output of fc2 | needed for ReLU₂ gradient | 32 × 8 × 4 = 1,024 |
| **Total** | | **13,312 B** |

### Inference memory (single sample, float32)

| Component | Bytes | Confidence |
|---|---|---|
| Weights | 5,248 | EXACT |
| Peak activation (input + output of fc1: 128 + 128 B) | 256 | EXACT |
| Centroid (8 floats) | 32 | EXACT |
| **Inference total** | **5,536 B ≈ 5.4 KB** | |

---

## Supporting SRAM Components

| Component | Calculation | Bytes | Confidence |
|---|---|---|---|
| Audio input buffer | 15,600 samples × 2 B (int16) | 31,200 | EXACT |
| Log-mel spectrogram | 1 × 64 × 61 × 4 B (float32) | 15,616 | EXACT |
| Embeddings buffer (full) | 600 frames × 32D × 4 B | 76,800 | EXACT |
| Centroid | 8 × 4 B | 32 | EXACT |
| Threshold | 1 × 4 B | 4 | EXACT |
| FFT scratch | 1,024 samples × 8 B (complex float32) | 8,192 | TIGHT |
| Stack + heap | empirical (typical Cortex-M4F) | ~20,000 | UNKNOWN |

---

## Per-Phase Peak SRAM

| Component | Collect | Train | Monitor |
|---|---|---|---|
| Embeddings buffer | 76,800 | 76,800 | — |
| f_c tensor arena | ~40,000–50,000 | — (freed) | ~40,000–50,000 |
| Audio input buffer | 31,200 | — | 31,200 |
| Log-mel buffer | 15,616 | — | 15,616 |
| FFT scratch | 8,192 | — | 8,192 |
| f_s weights | — | 5,248 | 5,248 |
| f_s gradients | — | 5,248 | — |
| f_s activations | — | 13,312 | 256 |
| Centroid + threshold | — | 36 | 36 |
| Stack + heap | ~20,000 | ~20,000 | ~20,000 |
| **Total** | **~192 KB** | **~120 KB** | **~121 KB** |
| **Headroom vs 786 KB** | **594 KB** | **666 KB** | **665 KB** |

All three phases are comfortably within budget. The bottleneck is the **Collect phase** at ~192 KB, driven by the full embeddings buffer (75 KB) coexisting with the f_c tensor arena (~40–50 KB) and the audio DSP buffers (~55 KB).

---

## f_s Training Time Estimate

The 50-epoch training run after the 10-minute collection window. Derived from first principles — not benchmarked.

**FLOP count per epoch (batch_size=32, 600 frames → 19 batches):**

| Pass | Operation | MACs per batch | × 19 batches |
|---|---|---|---|
| Forward | fc1 (32 samples × 32in × 32out) | 32,768 | 622,592 |
| Forward | fc2 (32 × 32in × 8out) | 8,192 | 155,648 |
| Forward | SVDD loss | ~512 | ~9,728 |
| Backward | ∂W₁ (32 × 32 × 32) | 32,768 | 622,592 |
| Backward | ∂W₂ (32 × 32 × 8) | 8,192 | 155,648 |
| Backward | ReLU masks + propagation | ~512 | ~9,728 |
| Update | SGD step (1,312 params × 2 ops) | 2,624 | 49,856 |
| **Total** | | | **~2.4M MACs/epoch** |

**Throughput on Cortex-M33 @ 160 MHz:**

The M33 FPU can issue one FMLA per cycle in ideal conditions. Practical throughput for small dense matrix ops is lower due to memory latency, loop overhead, and scalar fallback for small matrices.

| Effective throughput | Time per epoch | 50 epochs |
|---|---|---|
| 50 MFLOPS (conservative) | ~48 ms | ~2.4 s |
| 100 MFLOPS (optimistic) | ~24 ms | ~1.2 s |

**Estimate: 50 epochs takes ~1–2.5 seconds on the MCU.** This justifies the fixed-epoch approach: training completes in a few seconds immediately after the 10-minute collection window. The previous figure of "~5 seconds" in the code comments was based on an incorrect assumption of a Cortex-M4F at 64 MHz — the actual M33 at 160 MHz is ~2.5× faster. The qualitative conclusion is unchanged.

**Confidence: ESTIMATE (±50%)** — derivation is sound but actual throughput depends on compiler optimisation, CMSIS-DSP library usage, and cache behaviour. Only MCU benchmarking gives the exact figure.

---

## Flash Summary

| Component | Bytes | Confidence |
|---|---|---|
| f_c TFLite INT8 model | ~547–557 KB | ESTIMATE (±15%) |
| f_s weights (stored after training) | 5,248 B | EXACT |
| TFLM runtime library (op kernels for this model) | ~100–200 KB | UNKNOWN |
| Application firmware (C++ orchestration code) | unknown | UNKNOWN |
| **Measured components total** | **~552–562 KB** | |
| **Budget** | **2,048 KB** | |

The f_c model alone consumes ~27% of flash. The TFLM runtime and application code are the main unknowns on the flash side; they require compiling for Cortex-M4F and inspecting the `.map` file.

---

## Confidence Summary

| Level | Definition | Components |
|---|---|---|
| **EXACT** | Derivable from code constants, zero assumptions | f_s weights, gradients, optimizer states, activations; all supporting buffers |
| **TIGHT (±20%)** | Architecture lower bound + well-documented TFLM behaviour | f_c tensor arena |
| **ESTIMATE (±15%)** | INT8 param count is exact; serialisation overhead estimated | f_c flash size |
| **UNKNOWN** | Requires export, profiling, or compilation | f_c actual tensor arena, TFLM runtime flash, application code flash, stack/heap |

### To increase confidence on the UNKNOWN items

1. **f_c actual tensor arena** — export the model to TFLite INT8 and run `RecordingMicroInterpreter`; it reports the exact bytes used by the memory planner.
2. **f_c flash** — inspect the `.tflite` file size directly after export.
3. **TFLM runtime + app code flash** — compile the full firmware for `ARDUINO_NANO33BLE` target and check the `.map` file for `.text` + `.rodata` sizes.
4. **Stack/heap** — instrument the firmware with `uxTaskGetStackHighWaterMark` (FreeRTOS) or equivalent to measure actual peak stack depth at runtime.
